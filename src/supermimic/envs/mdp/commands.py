"""Multi-clip motion command with future-frame goals and return-based reset sampling.

Exposes the same attribute API as mjlab's ``MotionCommand`` (``anchor_pos_w``,
``robot_body_pos_w``, ``body_pos_relative_w`` ...) so mjlab's tracking reward
functions work unchanged, plus:

* ``goal``        (N, K*68) future-frame goal features (plan D5)
* ``root_*``      reference pelvis quantities for the paper's root terminations
* ``clip_end``    (N,) True once the clip's last frame has been reached (time-out)
* return-based reset sampling over 1 s bins of the whole library (plan D14)

Terrain (``cfg.terrain`` = plane | mixed, docs/VISION_TERRAIN_PLAN.md §2.2-2.3): the command owns the
``TerrainMap`` (rasterised in ``__init__``: the observation manager evaluates the ``terrain`` observation
at construction) and the policy scan history + per-episode scan noise. On ``mixed`` every reset also
samples the tile (column ~ type proportions, row ~ U{0..L}) and a spawn that keeps the reference path in
the tile, writes the episode origin o_ep = (p_spawn.xy - lib_root_xy(start), 0) into
``env.scene.env_origins`` (the single owner of the reference-to-world translation; ``StateIO`` and the
recorder read it live), lifts the spawn so no sole point penetrates the terrain, and lifts the
reference by the blurred ground under its root (``body_pos_w`` / ``root_pos_w`` / ``anchor_pos_w``, the
same ``reference_lift`` the world model and the imagination use). ``none`` is main's code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.terrain.heightmap import ScanNoiseCfg, TerrainMapCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from supermimic.terrain.heightmap import TerrainMap


def build_terrain_map(mj_model, gen_cfg: TerrainGeneratorCfg | None, device, cfg: TerrainMapCfg) -> TerrainMap:
    """``TerrainMap.build`` that rasterises once per node under torchrun: local rank 0 builds and writes the
    disk cache, everybody waits at a barrier, the other ranks then load the cached map."""
    import os

    import torch.distributed as dist

    from supermimic.terrain.heightmap import TerrainMap

    if gen_cfg is None or not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return TerrainMap.build(mj_model, gen_cfg, device, cfg)
    if cfg.cache_dir is None:
        raise ValueError("multi-GPU terrain needs the raster cache (TerrainMapCfg.cache_dir)")
    first = int(os.environ.get("LOCAL_RANK", dist.get_rank())) == 0
    tmap = TerrainMap.build(mj_model, gen_cfg, device, cfg) if first else None
    dist.barrier()
    if tmap is None:
        tmap = TerrainMap.build(mj_model, gen_cfg, device, cfg)
        assert tmap.info.get("cache_hit"), "rank > 0 should load the terrain map rank 0 just cached"
    return tmap


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_root_body_index = self.robot.body_names.index(cfg.root_body_name)

        default_joint_pos = self.robot.data.default_joint_pos[0].detach()
        self.library = MotionLibrary(
            cfg.npz_files, cfg.body_names, default_joint_pos, device=self.device, bin_frames=cfg.bin_frames,
            compact=cfg.compact_library, store_dtype=torch.float16 if cfg.library_fp16 else torch.float32,
        )
        print(f"[INFO] {self.library.summary()}")
        lib = self.library

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._init_terrain(env)
        self._refresh_ref()
        self.body_pos_relative_w = torch.zeros(n, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(n, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Return-based reset sampling state.
        self.bin_return = torch.full((lib.num_bins,), cfg.g_max, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.max_ep_steps = min(int(math.ceil(env.cfg.episode_length_s / env.step_dt)) + 2, cfg.max_logged_steps)
        # Evaluation hook: when >= 0, the next reset of that env starts exactly at this global frame.
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=self.device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=self.device, dtype=torch.float64)
        self.bootstrap_fn = None  # optional: env_ids -> value estimate for truncated episodes

        for k in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
                  "error_joint_pos", "error_joint_vel", "sampling_entropy", "episode_return_to_go"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ terrain

    def _init_terrain(self, env: ManagerBasedRlEnv) -> None:
        cfg = self.cfg
        self.tmap = None  # TerrainMap (plane | mixed)
        self.layout = None  # TerrainLayout (mixed only)
        self._dz = None  # (N, 3) reference lift [0, 0, lift(frame)] (mixed only)
        if cfg.terrain == "none":
            return
        if cfg.terrain not in ("plane", "mixed"):
            raise ValueError(f"unknown terrain {cfg.terrain!r}")
        # o_ep must stay fixed within an episode: no mid-episode command resampling.
        assert min(cfg.resampling_time_range) >= env.cfg.episode_length_s, cfg.resampling_time_range
        from supermimic.data.motion_convert import load_g1_model
        from supermimic.kinematics.g1_fk import G1Kinematics
        from supermimic.terrain import default_window_frames, path_bbox, sample_scan_noise
        from supermimic.terrain.heightmap import SCAN_DIM

        n = self.num_envs
        self.fk = G1Kinematics(load_g1_model(), device=self.device)
        gen_cfg = env.scene.terrain.cfg.terrain_generator if cfg.terrain == "mixed" else None
        self.tmap = build_terrain_map(env.sim.mj_model, gen_cfg, self.device, cfg.terrain_map)
        # Policy scan: history of the scans at the episode's shifted query points + per-episode noise.
        self.scan_hist = torch.zeros(cfg.scan_noise.history_len, n, SCAN_DIM, device=self.device)
        self.scan_noise = sample_scan_noise(n, cfg.scan_noise, self.device)
        self._scan_reset = torch.ones(n, dtype=torch.bool, device=self.device)
        if cfg.terrain == "plane":
            return
        self.layout = self.tmap.layout
        self.layout.check_origins(env.scene.terrain.terrain_origins)
        self.terrain_level = int(cfg.terrain_level)
        self.path_bbox = path_bbox(self.library, default_window_frames(env.cfg.episode_length_s, cfg.spawn_horizon))
        self.ep_row = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ep_col = torch.zeros(n, dtype=torch.long, device=self.device)
        # Evaluation hook: (row, col) >= 0 pins the next reset's tile and centres the reference path in it.
        self.forced_tile = torch.full((n, 2), -1, dtype=torch.long, device=self.device)
        # Finished-episode statistics per column: [normalised return sum, episodes, failures]; one copy for
        # the curriculum (reset at each level check), one for logging (reset at each log).
        self.terrain_stats_curriculum = torch.zeros(3, self.layout.num_cols, device=self.device)
        self.terrain_stats_log = torch.zeros(3, self.layout.num_cols, device=self.device)
        self.spawn_misfit = torch.zeros((), dtype=torch.long, device=self.device)  # paths that did not fit their tile
        self._dz = torch.zeros(n, 3, device=self.device)
        env.scene.env_origins[:, 2] = 0.0  # o_ep.z = 0 (mjlab initialises the origins at the tile spawn z)

    @property
    def policy_scan(self) -> torch.Tensor:
        """(N, 187) noisy, delayed policy height scan (the ``terrain`` observation)."""
        from supermimic.terrain import noisy_scan

        return noisy_scan(self.scan_hist, self.scan_noise, self.cfg.scan_noise)

    def _update_scan(self, env_ids: torch.Tensor | None) -> None:
        """Push the scan of the current state (qpos is fresh: called after ``sim.forward()``) into the
        policy scan history; envs flagged at reset get their whole history filled with it (lag 0 = current)."""
        from supermimic.terrain import push_scan

        q = self._env.sim.data.qpos[:]
        ids = slice(None) if env_ids is None else env_ids
        root = q[ids, 0:7]
        scan = self.tmap.scan(root[:, 0:3], root[:, 3:7], xy_shift=self.scan_noise.offset_xy[ids])
        hist = self.scan_hist[:, ids]
        new = torch.where(self._scan_reset[ids][None, :, None], scan[None].expand_as(hist), push_scan(hist, scan))
        if env_ids is None:
            self.scan_hist.copy_(new)
        else:
            self.scan_hist[:, env_ids] = new
        self._scan_reset[ids] = False
        if self.layout is not None and env_ids is None:
            self.tmap.count_out_of_bounds(root[:, 0:2])  # env-steps with the pelvis off the rasterised map

    def _reset_scan(self, env_ids: torch.Tensor) -> None:
        from supermimic.terrain import resample_scan_noise_

        resample_scan_noise_(self.scan_noise, env_ids, self.cfg.scan_noise)
        self._scan_reset[env_ids] = True

    def _place(self, env_ids: torch.Tensor, frames: torch.Tensor, row: torch.Tensor, col: torch.Tensor, centred: torch.Tensor) -> torch.Tensor:
        """Choose the tile + spawn of new episodes and write o_ep into ``env.scene.env_origins``; (k, 3)."""
        from supermimic.terrain import sample_spawn

        sp = sample_spawn(self.library, frames, self.path_bbox, self.layout, self.terrain_level, self.cfg.spawn_margin, row=row, col=col, centred=centred)
        self._env.scene.env_origins[env_ids] = sp.origin
        self.ep_row[env_ids] = sp.row
        self.ep_col[env_ids] = sp.col
        self.spawn_misfit += (~sp.fits).sum()
        return sp.origin

    def _lifted_spawn(self, env_ids: torch.Tensor, root_pos: torch.Tensor, root_ori: torch.Tensor, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Root position of a (noised) reference spawn lifted clear of the terrain (plan §2.2) and the clipped
        joints; ``root_pos`` is world-frame and already includes the reference lift of the start frame."""
        from supermimic.terrain import spawn_lift

        soft = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft[:, :, 0], soft[:, :, 1])
        origin = self._env.scene.env_origins[env_ids]
        qpos = torch.cat([root_pos - origin, root_ori, joint_pos], dim=-1)
        root_pos = root_pos.clone()
        root_pos[:, 2] += spawn_lift(qpos, origin, self.fk, self.tmap)
        return root_pos, joint_pos

    def _track_terrain_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes of ``env_ids`` (normalised return over the survived steps, failure =
        terminated) into the per-column statistics of their tile (curriculum + logging). Host-sync free.

        Episodes that reached the clip end in fewer than ``terrain_stats_min_steps`` steps without failing are left
        out: the return sampler starts ~40% of all episodes in the bin with the lowest return-to-go, which is
        always some clip's last bin (no bootstrap past a clip end). Those few-step post-reset episodes lowered the
        mean return by an amount that depends on which clip's last bin won (Phase D, same frozen policy: 0.67 vs
        0.76 per-episode means; 0.81 / 0.82 without them) and diluted the failure rate (logging-window means
        0.01-0.08 vs 0.11)."""
        lens = self.ep_len[env_ids]
        valid = torch.arange(self.max_ep_steps, device=self.device)[None, :] < lens[:, None]
        ret = (self.ep_rewards[env_ids] * valid).sum(-1) / (self.cfg.max_step_reward * lens.clamp_min(1))
        tm = getattr(self._env, "termination_manager", None)
        failed = tm.terminated[env_ids].float() if tm is not None else torch.zeros_like(ret)
        short_clip_end = self.clip_end[env_ids] & (failed == 0) & (lens < self.cfg.terrain_stats_min_steps)
        done = ((lens > 0) & ~short_clip_end).float()
        add = torch.stack([ret * done, done, failed * done])  # (3, k)
        col = self.ep_col[env_ids]
        for stats in (self.terrain_stats_curriculum, self.terrain_stats_log):
            stats.index_add_(1, col, add)

    # ------------------------------------------------------------------ reference (mjlab MotionCommand API)

    def _refresh_ref(self) -> None:
        """Cache the reference state / tracked-body states of the current target frames (+ terrain lift)."""
        self._ref = self.library.ref(self.frame)
        self._bodies = self.library.bodies(self.frame)
        if self._dz is not None:
            from supermimic.terrain import reference_lift

            self._dz[:, 2] = reference_lift(self.library, self.frame, self._env.scene.env_origins, self.tmap)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames, self.cfg.goal_features)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        p = self._bodies.pos_w + self._env.scene.env_origins[:, None, :]
        return p if self._dz is None else p + self._dz[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies.lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies.ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        p = self._bodies.pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins
        return p if self._dz is None else p + self._dz

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w[:, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        p = self._ref.root_pos_w + self._env.scene.env_origins
        return p if self._dz is None else p + self._dz

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref.root_quat_w

    # ------------------------------------------------------------------ robot

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_root_body_index]

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # ------------------------------------------------------------------ reward bookkeeping for the sampler

    def _log_step_rewards(self, env_mask: torch.Tensor | None = None) -> None:
        """Append this step's simulator reward (and the bin it was earned in) to each env's episode log."""
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_mask is None else env_mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward[ids]
        self.ep_bins[ids, pos] = self.library.bin_of_frame(self.frame[ids])
        self.ep_len[ids] = torch.minimum(self.ep_len[ids] + 1, torch.full_like(pos, self.max_ep_steps))

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes' discounted returns-to-go into the per-bin estimates."""
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max().item())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        r = r * tmask
        gp = self._gamma_pow[:T]
        # Return-to-go R_t = sum_{s>=t} gamma^{s-t} r_s via a reverse cumulative sum.
        disc = r * gp[None, :]
        rtg = torch.flip(torch.cumsum(torch.flip(disc, [1]), 1), [1]) / gp[None, :]
        # Truncated (timed-out) episodes: bootstrap with the critic if available, otherwise
        # drop the steps too close to the truncation point (plan D14).
        terminated = self._env.termination_manager.terminated[env_ids] if hasattr(self._env, "termination_manager") else torch.ones_like(lens, dtype=torch.bool)
        truncated = ~terminated & ~self.clip_end[env_ids]
        if truncated.any():
            if self.bootstrap_fn is not None:
                v = self.bootstrap_fn(env_ids).double()  # (n,)
                remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0).double()
                rtg = rtg + truncated[:, None] * (cfg.gamma ** remaining) * v[:, None]
            else:
                remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
                tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        mean = sums[touched] / cnts[touched]
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[touched] + cfg.ema_alpha * mean
        self.metrics["episode_return_to_go"][env_ids] = rtg[:, 0].float()
        self.ep_len[env_ids] = 0

    # ------------------------------------------------------------------ sampling

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "start":
            clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
            return lib.clip_start[clips]
        if cfg.sampling_mode == "uniform":
            # Uniform over frames (long clips get proportionally more starts), leaving >= 2 steps.
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
        assert cfg.sampling_mode == "return"
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        f = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        self.metrics["sampling_entropy"][:] = ent
        return f

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # Episode bookkeeping for the envs being reset (terminal reward is already in reward_buf).
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        if self.ep_len.sum() > 0 or hasattr(self._env, "reward_buf"):
            self._log_step_rewards(mask)
            if self.layout is not None:
                self._track_terrain_episodes(env_ids)
            self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0

        sampled = self._sample_frames(len(env_ids))
        forced = self.forced_frame[env_ids]
        self.frame[env_ids] = torch.where(forced >= 0, forced, sampled)
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        if self.layout is not None:
            tile = self.forced_tile[env_ids]
            self._place(env_ids, self.frame[env_ids], tile[:, 0], tile[:, 1], centred=tile[:, 0] >= 0)
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        if self.layout is not None:
            root_pos[:, 2] += self.tmap.ground(root_pos[:, 0:2])  # reference lift of the start frame
        root_ori = r.root_quat_w.clone()
        root_lin_vel = r.root_lin_vel_w.clone()
        root_ang_vel = r.root_ang_vel_w.clone()
        joint_pos = r.joint_pos.clone()
        joint_vel = r.joint_vel

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + rs[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rs[:, :3]
        root_ang_vel = root_ang_vel + rs[:, 3:]
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, device=self.device)
        if self.layout is not None:
            root_pos, joint_pos = self._lifted_spawn(env_ids, root_pos, root_ori, joint_pos)

        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
        if self.tmap is not None:
            self._reset_scan(env_ids)

    def update_relative_body_poses(self) -> None:
        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        delta_pos_w = robot_anchor_pos
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Clip end must be judged against the clip of the *current* frame (frame+1 may already be
        # the first frame of the next clip in the concatenated library).
        last = self.library.last_frame_of(self.frame)
        if env_ids is None:
            # Per-step path: log the reward earned against the current target frame for the envs
            # that were NOT just reset (their terminal reward was logged in _resample_command).
            reset_buf = getattr(self._env, "reset_buf", None)
            self._log_step_rewards(None if reset_buf is None else ~reset_buf)
            self.frame += 1
        else:
            self.frame[env_ids] += 1
        over = self.frame > last
        if over.any():
            self.clip_end |= over
            self.frame = torch.where(over, last, self.frame)
        self._refresh_ref()
        self.update_relative_body_poses()
        if self.tmap is not None:
            self._update_scan(env_ids)

    def reset_to_frame(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Deterministic reset to exact reference frames (no noise); used by evaluation. On ``mixed`` the
        reference path is centred in the ``forced_tile`` (or the env's current tile)."""
        self.ep_len[env_ids] = 0
        self.frame[env_ids] = frame
        self.clip_end[env_ids] = False
        r = self.library.ref(frame)
        joint_pos = r.joint_pos
        if self.layout is not None:
            tile = self.forced_tile[env_ids]
            row = torch.where(tile[:, 0] >= 0, tile[:, 0], self.ep_row[env_ids])
            col = torch.where(tile[:, 1] >= 0, tile[:, 1], self.ep_col[env_ids])
            self._place(env_ids, frame, row, col, centred=torch.ones_like(row, dtype=torch.bool))
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        if self.layout is not None:
            root_pos[:, 2] += self.tmap.ground(root_pos[:, 0:2])
            root_pos, joint_pos = self._lifted_spawn(env_ids, root_pos, r.root_quat_w, joint_pos)
        self._write_reference_state_to_sim(env_ids, root_pos, r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w, joint_pos, r.joint_vel)
        if self.tmap is not None:
            self._reset_scan(env_ids)

    # ------------------------------------------------------------------ debug visualization (viewer only)

    _ghost_model = None

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the reference pose as a translucent ghost robot (same approach as mjlab's MotionCommand)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            import copy

            m = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(m.ngeom):
                if m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0:
                    m.geom_rgba[gi, 3] = 0  # hide collision geoms
                else:
                    m.geom_rgba[gi] = (0.2, 0.6, 1.0, 0.45)
            self._ghost_model = m
        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        root_pos = self.root_pos_w.cpu().numpy()
        root_quat = self.root_quat_w.cpu().numpy()
        joint_pos = self.joint_pos.cpu().numpy()
        import numpy as np

        for e in env_indices:
            qpos = np.zeros(self._ghost_model.nq)
            qpos[free_q[0:3]] = root_pos[e]
            qpos[free_q[3:7]] = root_quat[e]
            qpos[joint_q] = joint_pos[e]
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{e}")


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
    npz_files: list[str]
    anchor_body_name: str
    root_body_name: str = "pelvis"
    body_names: tuple[str, ...]
    entity_name: str = "robot"
    future_frames: int = 10
    goal_features: str = "v1"  # "v1" (68-d per frame) | "traj" (77-d per frame)
    bin_frames: int = 50
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    # Return-based sampling (plan D14).
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024
    compact_library: bool = False
    library_fp16: bool = True
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # Terrain (VISION_TERRAIN_PLAN): "none" = main's code path | "plane" | "mixed" (scene terrain generator).
    terrain: str = "none"
    terrain_level: int = 2  # initial curriculum level L (spawn rows ~ U{0..L})
    terrain_map: TerrainMapCfg = field(default_factory=TerrainMapCfg)
    scan_noise: ScanNoiseCfg = field(default_factory=ScanNoiseCfg)  # policy scan corruption (disabled for evaluation)
    spawn_margin: float = 0.5  # m between the reference path bbox and the tile edge
    spawn_horizon: int = 32  # the spawn bbox covers one episode + this many frames (imagination horizon)
    max_step_reward: float = 0.1  # normalisation of the terrain-episode returns (curriculum)
    terrain_stats_min_steps: int = 100  # shorter non-failed clip-end episodes are not counted (see _track_terrain_episodes)

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)
