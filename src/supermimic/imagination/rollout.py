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

    Observation noise and the goal's anchor term follow ``step_fn``'s settings.
    """
    device = start.state.device
    n = start.state.shape[0]
    H = horizon
    ai = step_fn.wm.heads.anchor_index

    # Every target frame of the rollout is known up front (frame + t, clamped at the clip end):
    # gather the references and the reference part of the goals for all steps in one call each.
    last = lib.last_frame_of(start.frame)
    frames = torch.minimum(start.frame[None, :] + torch.arange(H + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    flat_frames = frames.reshape(-1)
    ref = reference_at(lib, flat_frames)
    goal_ref = lib.goal_window(flat_frames, future_frames, goal_features).reshape(H + 1, n, -1)
    anchor_pos = ref.body_pos_w[:, ai].reshape(H + 1, n, 3)
    anchor_quat = ref.body_quat_w[:, ai].reshape(H + 1, n, 4)

    # Initial observation history from the logged states / actions (noise re-sampled per frame).
    q0 = lib.default_joint_pos
    hist = torch.stack([step_fn.noisy(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])])

    s = start.state
    states, actions, goals = [s], [], []
    for t in range(H):
        s, a, hist, g = step_fn(s, hist, goal_ref[t], anchor_pos[t], anchor_quat[t], start.xi)
        states.append(s)
        actions.append(a)
        goals.append(g)
    goals.append(step_fn.goal(s, goal_ref[H], anchor_pos[H], anchor_quat[H]))  # g_H for the critic

    states = torch.stack(states)
    actions = torch.stack(actions)
    prev_actions = torch.cat([start.prev_action[None], actions[:-1]], dim=0)
    rewards, term_vec, logits, crit = heads_fn(states, actions, prev_actions, ref)
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
    )
