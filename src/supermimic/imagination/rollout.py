"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagHeads, ImagStep
from supermimic.world_model.heads import analytic_observation
from supermimic.world_model.wm_trainer import reference_at


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal_dim)
    crit_feat: torch.Tensor  # (H+1, n, 84) privileged tracking errors (or (…, 4) root errors only)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics, tensors)
    crit_terr: torch.Tensor | None = None  # (H+1, n, 215) noise-free critic terrain features (terrain only)
    ground_z: torch.Tensor | None = None  # (H+1, n) blurred ground under the pelvis (terrain only)


def imagine(
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    step_fn: ImagStep,
    heads_fn: ImagHeads,
    future_frames: int = 10,
    goal_features: str = "v1",
) -> ImagRollout:
    """Roll the policy through the world model for ``horizon`` steps from replay-buffer starts.

    Observation noise and the goal's anchor term follow ``step_fn``'s settings. With ``step_fn.terrain``
    the references are lifted, the policy sees the noisy delayed scan (history seeded from the logged
    ``hist_states``, noise parameters drawn per start) and the critic features include the terrain.
    """
    device = start.state.device
    n = start.state.shape[0]
    H = horizon
    ai = step_fn.wm.heads.anchor_index
    terrain = step_fn.terrain

    # Every target frame of the rollout is known up front (frame + t, clamped at the clip end):
    # gather the references and the reference part of the goals for all steps in one call each.
    last = lib.last_frame_of(start.frame)
    frames = torch.minimum(start.frame[None, :] + torch.arange(H + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    flat_frames = frames.reshape(-1)
    if terrain is None:
        ref = reference_at(lib, flat_frames)
    else:
        grid = terrain.grid
        ref = terrain.reference(lib, flat_frames, start.origin[None].expand(H + 1, n, 3).reshape(-1, 3), grid)
    goal_ref = lib.goal_window(flat_frames, future_frames, goal_features).reshape(H + 1, n, -1)
    anchor_pos = ref.body_pos_w[:, ai].reshape(H + 1, n, 3)
    anchor_quat = ref.body_quat_w[:, ai].reshape(H + 1, n, 4)

    # Initial observation history from the logged states / actions (noise re-sampled per frame).
    q0 = lib.default_joint_pos
    hist = torch.stack([step_fn.noisy(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])])

    s = start.state
    states, actions, goals = [s], [], []
    if terrain is None:
        for t in range(H):
            s, a, hist, g = step_fn(s, hist, goal_ref[t], anchor_pos[t], anchor_quat[t], start.xi)
            states.append(s)
            actions.append(a)
            goals.append(g)
    else:
        from supermimic.terrain import sample_scan_noise

        # Per-start scan noise; the scan history before s_t is rebuilt from the logged states s_{t-L}..s_{t-1}
        # (backfilled at episode start, like the sim's reset fill); step 0 pushes the scan of s_t itself.
        noise = sample_scan_noise(n, step_fn.scan_noise, device)
        L = step_fn.scan_noise.history_len
        assert start.hist_states.shape[0] > L, "observation history must be longer than the scan history"
        scan_hist = terrain.policy_scan_query(start.hist_states[-L - 1 : -1], start.origin, noise, grid)  # (L, n, 187)
        for t in range(H):
            s, a, hist, scan_hist, g = step_fn.step_terrain(s, hist, scan_hist, goal_ref[t], anchor_pos[t], anchor_quat[t], start.xi, start.origin, noise, grid)
            states.append(s)
            actions.append(a)
            goals.append(g)
    goals.append(step_fn.goal(s, goal_ref[H], anchor_pos[H], anchor_quat[H]))  # g_H for the critic

    states = torch.stack(states)
    actions = torch.stack(actions)
    prev_actions = torch.cat([start.prev_action[None], actions[:-1]], dim=0)
    rewards, term_vec, logits, crit = heads_fn(states, actions, prev_actions, ref)
    crit_terr = ground_z = None
    if terrain is not None:
        crit_terr, ground_z = heads_fn.critic_terrain(states, start.origin, grid)
    d = torch.sigmoid(logits)  # (H, n) termination probability of s_1..s_H
    survival = torch.cat([torch.ones(1, n, device=device), torch.cumprod(1.0 - d, dim=0)], dim=0)
    return ImagRollout(
        states=states,
        actions=actions,
        rewards=rewards,
        dhat=torch.cat([torch.zeros(1, n, device=device), d], dim=0),
        survival=survival,
        goals=torch.stack(goals),
        crit_feat=crit,
        frames=frames,
        reward_terms={k: v for k, v in zip(REWARD_TERM_NAMES, term_vec.detach().mean(dim=(1, 2)).unbind(0))},
        crit_terr=crit_terr,
        ground_z=ground_z,
    )
