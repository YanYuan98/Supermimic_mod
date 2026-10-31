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
    aux_h: torch.Tensor | None = None  # (H, n, agg_hidden) depth aggregator hidden of s_0..s_{H-1} (depth only)
    aux_vis: torch.Tensor | None = None  # (H, n, 187) scan cells seen by the slot cameras (depth only)


def imagine(
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    step_fn: ImagStep,
    heads_fn: ImagHeads,
    future_frames: int = 10,
    goal_features: str = "v1",
    obs_history: int | None = None,
) -> ImagRollout:
    """Roll the policy through the world model for ``horizon`` steps from replay-buffer starts.

    Observation noise and the goal's anchor term follow ``step_fn``'s settings. With ``step_fn.terrain``
    the references are lifted, the policy sees the noisy delayed scan (history seeded from the logged
    ``hist_states``, noise parameters drawn per start) and the critic features include the terrain. With
    ``step_fn.depth`` the policy sees rendered depth frames instead (``_init_depth_history``).
    ``obs_history``: proprioceptive history length (the last rows of ``hist_states``; default all rows).
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
    n_hist = start.hist_states.shape[0]
    k0 = 0 if obs_history is None else n_hist - obs_history
    hist = torch.stack([step_fn.noisy(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(k0, n_hist)])

    s = start.state
    states, actions, goals = [s], [], []
    aux_h = aux_vis = None
    if step_fn.depth is not None:
        # One frame-sample phase for the whole batch (drawn per update): sample steps are known on the host, so
        # the other steps skip rendering / encoding (the simulator keeps a per-episode phase).
        k = step_fn.depth.cfg.sample_every
        phase = int(torch.randint(0, k, ()).item())
        dparams, q_hist, e_hist, cam_hist = _init_depth_history(step_fn, start, terrain.grid, phase=phase)
        if torch.is_grad_enabled():
            e_hist.requires_grad_(True)  # same autograd signature on every step: fewer compiled variants of the step
        hs, vs = [], []
        for t in range(H):
            s, a, hist, q_hist, e_hist, cam_hist, g, h, vis = step_fn.step_depth(
                s, hist, q_hist, e_hist, cam_hist, (t + phase) % k == 0, goal_ref[t], anchor_pos[t], anchor_quat[t], start.xi, start.origin, dparams, terrain.grid
            )
            states.append(s)
            actions.append(a)
            goals.append(g)
            hs.append(h)
            vs.append(vis)
        aux_h, aux_vis = torch.stack(hs), torch.stack(vs)
    elif terrain is None:
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
    crit_terr = ground_z = term_offset = None
    if terrain is not None:
        crit_terr, ground_z = heads_fn.critic_terrain(states, start.origin, grid)
        if getattr(step_fn.wm.cfg, "height_termination", "reference") == "ground":
            # s_{t+1} is judged against target frame t: ground under s_1..s_H minus ground under reference roots 0..H-1.
            ref_xy = ref.root_pos_w[: H * n].reshape(H, n, 3)
            term_offset = terrain.ground_offset(states[1:, :, 0:3], ref_xy, start.origin[None], grid)
    rewards, term_vec, logits, crit = heads_fn(states, actions, prev_actions, ref, term_offset)
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
        aux_h=aux_h,
        aux_vis=aux_vis,
    )


@torch.no_grad()
def _init_depth_history(step_fn: ImagStep, start: StartBatch, grid, dparams=None, phase: int | None = None):
    """Per-start camera parameters and the histories before step 0, rebuilt from the logged states: qpos (L, n, 36),
    frame features [e, g_cam] (F, n, slot_dim) and true camera poses (F, n, 12) of the sample steps t < 0
    ((t + phase) % sample_every == 0), each frame rendered from the state ``lag`` (+ jitter) steps earlier. Episode
    starts are backfilled in ``hist_states`` (the simulator fills its history with the reset frame likewise).
    The frame features carry no gradient (the encoder is trained on the frames of the rollout)."""
    from dataclasses import replace

    from supermimic.terrain.depth import sample_depth_params

    ds, pol = step_fn.depth, step_fn.policy
    cfg = ds.cfg
    noise = step_fn.depth_noise
    hs = start.hist_states
    Hh, n = hs.shape[0], hs.shape[1]
    assert Hh >= cfg.start_hist, f"depth needs {cfg.start_hist} logged states per start, got {Hh}"
    dev = hs.device
    if dparams is None:
        dparams = sample_depth_params(n, cfg if noise else replace(cfg, noise=False), dev)
    if phase is not None:
        dparams = dparams._replace(phase=torch.full_like(dparams.phase, phase))
    k, F, L = cfg.sample_every, cfg.hist_frames, cfg.state_hist
    t_last = -1 - ((dparams.phase - 1) % k)  # newest sample step before 0
    t_f = t_last[None, :] - k * torch.arange(F - 1, -1, -1, device=dev)[:, None]  # (F, n) oldest first
    lag = torch.stack([ds.frame_draws(dparams, noise)[0] for _ in range(F)])  # (F, n)
    row = (Hh - 1 + t_f - lag).clamp_min(0)
    qcap = hs[row, torch.arange(n, device=dev)[None, :].expand(F, n)][..., :36]
    init_frame = getattr(step_fn, "init_frame", None)
    if init_frame is None:  # eager fallback (tests drive this with a stand-in step function)

        def init_frame(q, o, prm, g):
            obs = ds.observe(q, o, prm, g, noise)
            return torch.cat([pol.depth.encoder(obs.img), obs.g_cam], dim=-1), torch.cat([obs.cam_pos, obs.cam_R.flatten(-2)], dim=-1)

    # ``rows`` frame rows (x all starts) per call, the last chunk padded: few launches, one compiled shape. The camera
    # parameters are per start; the lag already chose the state.
    rows = 8
    Fp = -(-F // rows) * rows
    qpad = torch.cat([qcap, qcap[-1:].expand(Fp - F, n, 36)]) if Fp > F else qcap
    rep = lambda x: x[None].expand(rows, *x.shape).reshape(rows * n, *x.shape[1:])  # noqa: E731
    prm, org = type(dparams)(*(rep(x) for x in dparams)), rep(start.origin)
    feats, cams = zip(*(init_frame(qpad[i : i + rows].reshape(rows * n, 36), org, prm, grid) for i in range(0, Fp, rows)))
    e_hist = torch.cat(feats).view(Fp, n, -1)[:F]
    cam_hist = torch.cat(cams).view(Fp, n, -1)[:F]
    q_hist = hs[Hh - 1 - L : Hh - 1, :, :36].clone()
    return dparams, q_hist, e_hist, cam_hist
