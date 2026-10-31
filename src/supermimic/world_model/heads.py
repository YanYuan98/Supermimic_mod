"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the 71-d simulator state
``s = [qpos(36) | qvel(35)]`` (root position expressed relative to the env origin,
see ``StateIO``), the previous clean action, the sim context ``xi`` and the
reference frame, so the same code serves the simulator-side tests and the
imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.rewards import TrackingRewardCfg, reanchor_reference, tracking_reward_terms
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import (
    heading_yaw,
    matrix_from_quat,
    projected_gravity,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

OBS_DIM = 93
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))


def split_state(s: torch.Tensor):
    """-> root_pos (…,3), root_quat (…,4), joint_pos (…,29), root_lin_vel_w (…,3), root_ang_vel_b (…,3), joint_vel (…,29)."""
    return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free 93-d proprioceptive observation (paper Table A.1) from state + context.

    xi[..., 1:30] is the encoder bias: observed joint positions are ``q + bias - q0``.
    """
    _, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1:30]
    return torch.cat([q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1)


ANCHOR_OBS_DIM = 9


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor, mode: str = "pos_ori") -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame (= mjlab's motion_anchor_pos_b /
    motion_anchor_ori_b): [pos (3), rot6d (6)] for ``pos_ori``, rot6d only for ``ori``."""
    q12 = quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat)
    rot6d = matrix_from_quat(q12)[..., :2].reshape(*q12.shape[:-1], 6)
    if mode == "ori":
        return rot6d
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    return torch.cat([pos, rot6d], dim=-1)


# Critic-only privileged tracking errors (F5): root error 4 + anchor pose 9 + body positions 14*3 + joints 29.
CRITIC_ERR_DIM = 4 + 9 + 14 * 3 + 29


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, N, 93) oldest→newest  ->  (N, 930) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)
    joint_pos: torch.Tensor | None = None  # (N, 29) (critic error features only)


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: G1Kinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., :36], s[..., 36:])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7:36],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    @staticmethod
    def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame, ground_offset: torch.Tensor | None = None) -> torch.Tensor:
        """[height error, tilt error] (N, 2) — the quantities the paper thresholds (0.25 m, 0.8). ``ground_offset``
        (N) = ground under the robot pelvis - ground under the reference pelvis: the height error then compares the
        heights above the local ground (``height_termination = "ground"``) instead of absolute heights."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        if ground_offset is not None:
            height = height + ground_offset
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, height_thr: float = 0.25, tilt_thr: float = 0.8) -> torch.Tensor:
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref)  # pelvis == free-joint root
        return (f[..., 0].abs() > height_thr) | (f[..., 1].abs() > tilt_thr)

    def critic_error_features(self, s: torch.Tensor, ref: ReferenceFrame, bodies=None) -> torch.Tensor:
        """Privileged tracking errors of state ``s`` w.r.t. reference ``ref`` for the critic (N, 84):
        [root error (4), reference anchor pose in the robot anchor frame (9), re-anchored reference
        body positions minus robot body positions in the robot anchor heading frame (42),
        reference minus robot joint positions (29)]. Differentiable in ``s`` (through FK)."""
        b = bodies if bodies is not None else self.robot_bodies(s)
        ai = self.anchor_index
        ra_pos, ra_quat = b["pos"][..., ai, :], b["quat"][..., ai, :]
        ma_pos, ma_quat = ref.body_pos_w[..., ai, :], ref.body_quat_w[..., ai, :]
        rel_pos, _ = reanchor_reference(ref.body_pos_w, ref.body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)
        nb = rel_pos.shape[-2]
        hq = yaw_quat(ra_quat)[..., None, :].expand(*ra_quat.shape[:-1], nb, 4)
        e_body = quat_apply_inverse(hq, rel_pos - b["pos"]).reshape(*ra_pos.shape[:-1], nb * 3)
        e_anchor = anchor_relative_obs(ra_pos, ra_quat, ma_pos, ma_quat, "pos_ori")
        e_joint = ref.joint_pos - s[..., 7:36]
        return torch.cat([self.root_error_features(s, ref), e_anchor, e_body, e_joint], dim=-1)

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
