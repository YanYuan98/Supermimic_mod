"""Compiled pieces of an imagined rollout (plan 3.6: launch-overhead control).

The rollout is split the way supermimic_vibed does it, which minimises kernel launches:

* ``ImagStep`` — the only per-step work: goal anchor term (FK of the pelvis -> torso chain only),
  policy forward, world-model step, next noise-free observation + sensor noise, history shift.
  Policy, world model and observation are fused into one compiled graph per step.
* ``ImagHeads`` — everything that does not feed back into the rollout, evaluated once on all
  (H+1)·n imagined states: full-body FK, reward, termination logits and the critic's privileged
  tracking errors. Gradients reach every state through the stacked tensors exactly as if they
  were computed step by step (the policy's actions never depend on rewards).

Terrain (docs/VISION_TERRAIN_PLAN.md §2.6, ``terrain`` given): every step looks the policy scan up at the
episode's shifted points, pushes it into the scan history and corrupts it (latency, offsets, point noise,
dropout: the sim's noise model, only on the policy's copy), while the world model gets the noise-free
condition of the current predicted state. ``ImagHeads`` additionally returns the critic's noise-free
terrain features and the ground height for all (H+1)·n states. Without ``terrain`` both run main's graphs.

Depth (``depth`` given, a ``DepthSensor``): instead of the scan, every step renders the frame of the state ``lag``
steps old (same sensor model as the simulator), encodes it and pushes [e, g_cam] into the frame history on the
envs' sample steps; the policy reads the history slots. The step also returns the aggregator's hidden h_t (input of
the auxiliary scan head) and which of the current scan cells the slot cameras saw (its loss mask).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from supermimic.goal_spec import anchor_mode
from supermimic.terrain.heightmap import noisy_scan, push_scan
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_relative_obs, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

if TYPE_CHECKING:
    from supermimic.terrain.depth import DepthParams, DepthSensor
    from supermimic.terrain.inputs import TerrainInputs

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
# Goal-side anchor terms (mjlab tracking defaults): position 0.25 m, orientation 0.05.
ANCHOR_NOISE_HALF = {"pos_ori": torch.tensor([0.25] * 3 + [0.05] * 6), "ori": torch.tensor([0.05] * 6)}


def _compile(fn, mode: str | None):
    if not mode:
        return fn
    return torch.compile(fn, mode=None if mode == "default" else mode, dynamic=False)


class ImagStep:
    """(s_t, noisy obs history, reference goal part of g_t, reference anchor) -> (s_{t+1}, a_t, history', g_t)."""

    def __init__(
        self,
        wm: WorldModel,
        policy,
        default_joint_pos: torch.Tensor,
        anchor_obs: str | bool = "none",
        obs_noise: bool = True,
        compile_mode: str | None = None,
        *,
        terrain: TerrainInputs | None = None,
        depth: DepthSensor | None = None,
    ):
        self.wm, self.policy, self.q0 = wm, policy, default_joint_pos
        self.mode = anchor_mode(anchor_obs)
        self.obs_noise = obs_noise
        dev = default_joint_pos.device
        self.noise_half = OBS_NOISE_HALF.to(dev)
        self.anchor_noise_half = ANCHOR_NOISE_HALF[self.mode].to(dev) if self.mode != "none" else None
        heads = wm.heads
        self.torso = int(heads.tracked_idx[heads.anchor_index])  # FK body index of the anchor (torso_link)
        self.terrain = terrain
        if terrain is not None:
            from dataclasses import replace

            # Policy-side scan corruption follows the observation-noise switch (like the sim's obs_noise).
            self.scan_noise = terrain.scan_noise if obs_noise else replace(terrain.scan_noise, enabled=False)
        self.depth = depth
        if depth is not None:
            assert terrain is not None, "depth needs the terrain inputs (world model / critic)"
            self.depth_noise = obs_noise and depth.cfg.noise
        step = self._step if terrain is None else (self._step_depth if depth is not None else self._step_terrain)
        self._fn = _compile(step, compile_mode)
        if depth is not None:
            self._init_frame_fn = _compile(self._init_frame, compile_mode)
        self._goal_fn = _compile(self.goal, compile_mode)

    def noisy(self, o: torch.Tensor) -> torch.Tensor:
        return o + (torch.rand_like(o) * 2 - 1) * self.noise_half if self.obs_noise else o

    def goal(self, s: torch.Tensor, goal_ref: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
        """Full goal g_t: reference features + (noisy) reference anchor pose in the robot anchor frame of s_t."""
        if self.mode == "none":
            return goal_ref
        pos, quat = self.wm.heads.fk.chain_pose(s[..., :36], self.torso)
        rel = anchor_relative_obs(pos, quat, ref_anchor_pos, ref_anchor_quat, self.mode)
        if self.obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * self.anchor_noise_half
        return torch.cat([goal_ref, rel], dim=-1)

    def _step(self, s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi):
        g = self.goal(s, goal_ref, ref_anchor_pos, ref_anchor_quat)
        a = self.policy(flatten_history_term_major(hist), g)
        s_next = self.wm.step(s, a, xi)
        obs = self.noisy(analytic_observation(s_next, a, xi, self.q0))
        return s_next, a, torch.cat([hist[1:], obs[None]], dim=0), g

    def __call__(self, s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi):
        return self._fn(s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi)

    def _step_terrain(self, s, hist, scan_hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, noise, grid):
        tr = self.terrain
        scan_hist = push_scan(scan_hist, tr.policy_scan_query(s, origin, noise, grid))  # lag 0 = current
        scan_obs = noisy_scan(scan_hist, noise, self.scan_noise)  # noise only on the policy's copy
        g = self.goal(s, goal_ref, ref_anchor_pos, ref_anchor_quat)
        a = self.policy(flatten_history_term_major(hist), g, scan=scan_obs)
        s_next = self.wm.step_at(s, a, xi, origin, grid)  # noise-free condition of s; re-queried next step
        obs = self.noisy(analytic_observation(s_next, a, xi, self.q0))
        return s_next, a, torch.cat([hist[1:], obs[None]], dim=0), scan_hist, g

    def step_terrain(self, s, hist, scan_hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, noise, grid):
        """-> (s_{t+1}, a_t, history', scan history', g_t); ``noise``: per-start ``ScanNoiseParams``."""
        return self._fn(s, hist, scan_hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, noise, grid)


    def _step_depth(self, s, hist, q_hist, e_hist, cam_hist, sample, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, dparams, grid):
        from torch.utils.checkpoint import checkpoint

        ds, pol = self.depth, self.policy
        L = q_hist.shape[0]
        q_hist = torch.cat([q_hist[1:], s[None, :, :36]], dim=0)  # newest = s_t
        if sample:  # python bool: one compiled graph for sample steps, one for the others
            lag, drop = ds.frame_draws(dparams, self.depth_noise)
            qcap = torch.gather(q_hist, 0, (L - 1 - lag).view(1, -1, 1).expand(1, q_hist.shape[1], 36))[0]
            obs = ds.observe(qcap, origin, dparams, grid, self.depth_noise)
            # Activation checkpointing: only the image is kept for the backward pass (the convs are recomputed).
            e = checkpoint(pol.depth.encoder, obs.img, use_reentrant=False)
            feat = torch.cat([e, obs.g_cam], dim=-1)
            cam = torch.cat([obs.cam_pos, obs.cam_R.flatten(-2)], dim=-1)
            feat = torch.where(drop[:, None], e_hist[-1], feat)
            cam = torch.where(drop[:, None], cam_hist[-1], cam)
            e_hist = torch.cat([e_hist[1:], feat[None]], dim=0)
            cam_hist = torch.cat([cam_hist[1:], cam[None]], dim=0)
        idx = ds.slot_idx
        slots = e_hist[idx]
        if ds.cfg.pose_input != "none":  # ablation P*: exact pose of each slot camera relative to the current torso
            from supermimic.terrain.depth import relative_pose

            with torch.no_grad():
                tp, tq = self.wm.heads.fk.chain_pose(s[:, :36].detach(), self.depth.torso)
                cs = cam_hist[idx]
                rel = relative_pose(cs[..., 0:3], cs[..., 3:12].unflatten(-1, (3, 3)), tp + origin, tq)
            slots = torch.cat([slots, rel], dim=-1)
        z_terr, h = pol.depth.latent(slots.transpose(0, 1))
        # Auxiliary-loss mask: scan cells of the current state inside a slot camera's frustum.
        with torch.no_grad():
            tm = self.terrain.tmap
            root_w = s[:, 0:3].detach() + origin
            xy = tm.scan_points(root_w, s[:, 3:7].detach())
            pts = torch.cat([xy, tm.height(xy, grid)[..., None]], dim=-1)
            cs = cam_hist[idx]
            vis = ds.visible(pts, cs[..., 0:3], cs[..., 3:12].unflatten(-1, (3, 3)))
        g = self.goal(s, goal_ref, ref_anchor_pos, ref_anchor_quat)
        a = pol.decoder(pol.encoders["motion"](g), flatten_history_term_major(hist), z_terr)
        s_next = self.wm.step_at(s, a, xi, origin, grid)
        o = self.noisy(analytic_observation(s_next, a, xi, self.q0))
        return s_next, a, torch.cat([hist[1:], o[None]], dim=0), q_hist, e_hist, cam_hist, g, h, vis

    def _init_frame(self, qpos, origin, dparams, grid):
        obs = self.depth.observe(qpos, origin, dparams, grid, self.depth_noise)
        return torch.cat([self.policy.depth.encoder(obs.img), obs.g_cam], dim=-1), torch.cat([obs.cam_pos, obs.cam_R.flatten(-2)], dim=-1)

    def init_frame(self, qpos, origin, dparams: DepthParams, grid):
        """(frame feature [e, g_cam], true camera pose (pos, R)) of one history frame per start (no gradient)."""
        return self._init_frame_fn(qpos, origin, dparams, grid)

    def step_depth(self, s, hist, q_hist, e_hist, cam_hist, sample: bool, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, dparams: DepthParams, grid):
        """-> (s_{t+1}, a_t, history', qpos history', frame history', camera history', g_t, h_t, aux mask). ``sample``:
        whether step t is a frame-sample step (the imagination uses one phase for the whole batch, so this is a
        Python bool and non-sample steps skip rendering and encoding)."""
        return self._fn(s, hist, q_hist, e_hist, cam_hist, sample, goal_ref, ref_anchor_pos, ref_anchor_quat, xi, origin, dparams, grid)


class ImagHeads:
    """Batched reward / termination / critic features over a whole rollout."""

    def __init__(self, wm: WorldModel, critic_features: bool = True, compile_mode: str | None = None, *, terrain: TerrainInputs | None = None):
        self.wm = wm
        self.critic_features = critic_features
        self.terrain = terrain
        self._fn = _compile(self._heads, compile_mode)
        if terrain is not None:
            self._terrain_fn = _compile(self._critic_terrain, compile_mode)

    def _critic_terrain(self, states, origin, grid):
        """Noise-free critic terrain features (H+1, n, 215) and ground height (H+1, n) of every state."""
        return self.terrain.critic_features(states, origin[None], grid)

    def critic_terrain(self, states, origin, grid):
        return self._terrain_fn(states, origin, grid)

    def _heads(self, states, actions, prev_actions, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat, ref_joint_pos, term_offset=None):
        """states (H+1, n, 71); actions / prev_actions (H, n, 29); reference tensors for frames 0..H; ``term_offset``:
        None, (H, n) ground-relative termination offset of s_1..s_H (``height_termination = "ground"``) or (H+1, n) ground
        under s_0..s_H (``reference_lift = "robot"``, unlifted references)."""
        H1, n = states.shape[:2]
        H = H1 - 1
        heads = self.wm.heads
        flat = states.reshape(H1 * n, -1)
        bodies = heads.robot_bodies(flat)
        # reference_lift = "robot": ``term_offset`` (H+1, n) is the ground under each s_t; frame t is lifted by the ground
        # under s_t for the critic features and under s_{t+1} for the reward / termination of reaching s_{t+1}.
        robot_lift = term_offset is not None and term_offset.shape[0] == H1
        tgt_body, tgt_root = ref_body_pos[: H * n], ref_root_pos[: H * n]
        if robot_lift:
            z_cur, z_next = term_offset.reshape(-1), term_offset[1:].reshape(-1)
            ref_body_pos = ref_body_pos + torch.nn.functional.pad(z_cur[:, None, None], (2, 0))
            ref_root_pos = ref_root_pos + torch.nn.functional.pad(z_cur[:, None], (2, 0))
            tgt_body = tgt_body + torch.nn.functional.pad(z_next[:, None, None], (2, 0))
            tgt_root = tgt_root + torch.nn.functional.pad(z_next[:, None], (2, 0))
            term_offset = None
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat, ref_joint_pos)
        # Reward r_{t+1} of reaching s_{t+1} under target frame t (frames 0..H-1, states 1..H).
        tgt = ReferenceFrame(tgt_body, ref_body_quat[: H * n], ref_body_lin[: H * n], ref_body_ang[: H * n], tgt_root, ref_root_quat[: H * n], ref_joint_pos[: H * n])
        nxt = {k: v[n:] for k, v in bodies.items()}
        terms = heads.reward(flat[n:], actions.reshape(H * n, -1), prev_actions.reshape(H * n, -1), tgt, nxt)
        term_vec = torch.stack([terms[k].reshape(H, n) for k in REWARD_TERM_NAMES], dim=0)  # (8, H, n)
        logits = self.wm.termination_logits(flat[n:], tgt, None if term_offset is None else term_offset.reshape(H * n)).reshape(H, n)
        # Critic features of s_t against target frame t (t = 0..H).
        if self.critic_features:
            crit = heads.critic_error_features(flat, ref, bodies)
        else:
            crit = heads.root_error_features(flat, ref)
        return terms["total"].reshape(H, n), term_vec, logits, crit.reshape(H1, n, -1)

    def __call__(self, states, actions, prev_actions, ref: ReferenceFrame, term_offset=None):
        return self._fn(states, actions, prev_actions, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w, ref.joint_pos, term_offset)
