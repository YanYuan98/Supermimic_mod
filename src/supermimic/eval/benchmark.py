"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends.

Normalised return (paper §4): the undiscounted reward sum over the episode divided by the
largest value it can take, i.e. max step reward * the number of steps of the *complete* clip,
so a policy that terminates early scores low. ``norm_return_survived`` keeps the v1 definition
(divided by the steps actually taken) for comparison with the old experiment log only.

Terrain (docs/VISION_TERRAIN_PLAN.md M0/M3): ``terrain="plane"`` runs the same protocol in the terrain-capable
environment on the flat plane (regression of terrain models); ``terrain="mixed"`` builds the terrain grid with a
fixed seed and plays every clip once in each requested (column, row) tile, the whole clip centred in the tile
(``MultiClipMotionCommand.forced_tile``); failures use the terrain-lifted reference (the command's), and the
results are reported per column x row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.goal_spec import anchor_mode
from supermimic.policy.nets import Policy, PolicyCfg

EVAL_TERRAIN_SEED = 1000  # fixed benchmark map, distinct from the training default (seed 0)
EVAL_TERRAIN_ROWS = (0, 3, 6, 9)
# Outer flat border of the evaluation map. Whole clips (~250 s) drift several metres from the reference (the
# policy has no global position feedback), so robots on the outermost tiles (row 0 / 9, the first / last
# column) walked off the training map's 4 m border, i.e. off the world (Phase C). 12 m keeps them on flat ground.
EVAL_TERRAIN_BORDER = 12.0


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    norm_return_survived: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    terrain: str | None = None  # terrain column name (mixed benchmark only)
    row: int = -1  # terrain row / difficulty (mixed benchmark only)
    left_tile_step: int = -1  # mixed: first step with the pelvis outside its tile (drift), -1 if it stayed on it


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = dict(ck["config"]["policy"])
    pc.setdefault("input_norm", "fixed")  # configs saved before these options existed (v1 / early v2)
    pc.setdefault("activation", "silu")
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    cfg = dict(ck["config"])
    cfg["_meta"] = {k: ck.get(k) for k in ("iteration", "iteration_unit", "env_steps", "num_updates", "train_seconds", "terrain_level")}
    return policy, cfg


def terrain_env_kwargs_for_eval(cfg: dict, terrain: str, seed: int | None = EVAL_TERRAIN_SEED, border: float | None = EVAL_TERRAIN_BORDER, scan_noise_scale: float = 0.0) -> dict:
    """Terrain fields of ``SuperMimicEnvCfgArgs`` for evaluating a checkpoint (its stored ``TerrainCfg``, or
    the defaults for flat checkpoints) on ``terrain`` = none | plane | mixed, with an explicit map seed and the
    evaluation border (``None``: the checkpoint's). The tiles do not depend on the border.

    ``scan_noise_scale`` > 0 (plan M5-A robustness): the policy scan gets the checkpoint's training scan noise with
    every magnitude (point / offsets / dropout / max lag) times this factor, although the benchmark otherwise runs
    noise-free; 0 = noise-free scan (the standard protocol)."""
    if terrain == "none":
        return {}
    from supermimic.config import TerrainCfg, terrain_env_kwargs

    t = dict(cfg.get("terrain") or asdict(TerrainCfg()))
    t["kind"] = terrain
    if seed is not None:
        t["seed"] = seed
    if border is not None:
        t["border"] = border
    if scan_noise_scale <= 0.0:
        return terrain_env_kwargs(t)
    s = scan_noise_scale
    n = dict(t["scan_noise"])
    t["scan_noise"] = dict(n, enabled=True, point=n["point"] * s, offset_z=n["offset_z"] * s, offset_xy=n["offset_xy"] * s, dropout=min(n["dropout"] * s, 1.0), max_lag=int(round(n["max_lag"] * s)))
    return {**terrain_env_kwargs(t), "scan_noise_always": True}


def act(policy: Policy, obs) -> torch.Tensor:
    """Deterministic action from an env observation dict (the scan only for terrain-capable policies)."""
    if policy.terrain is None:
        return policy(obs["policy"], obs["goal"])
    return policy(obs["policy"], obs["goal"], scan=obs["terrain"])


@torch.no_grad()
def run_benchmark(
    policy: Policy,
    npz_files: list[str],
    device: str = "cuda:0",
    num_envs: int | None = None,
    future_frames: int = 10,
    verbose: bool = True,
    anchor_obs: str | bool = "none",
    goal_features: str = "v1",
    self_collision: bool = True,
    *,
    terrain_kwargs: dict | None = None,
    terrain_cols: list[str] | None = None,
    terrain_rows: tuple[int, ...] = EVAL_TERRAIN_ROWS,
) -> list[ClipResult]:
    """``terrain_kwargs`` (``terrain_env_kwargs_for_eval``): {} = main's flat env; "plane" = one job per clip;
    "mixed" = one job per (clip, column, row) with ``terrain_cols`` (default all) x ``terrain_rows``."""
    terrain_kwargs = terrain_kwargs or {}
    mixed = terrain_kwargs.get("terrain") == "mixed"
    if policy.terrain is not None and not terrain_kwargs:
        raise ValueError("terrain-capable policy: evaluate with terrain 'plane' or 'mixed'")
    n_files = len(npz_files)
    if mixed:
        from supermimic.terrain.presets import mixed_sub_terrains

        names = list(mixed_sub_terrains(terrain_kwargs.get("terrain_size", 12.0)))
        cols = [names.index(c) for c in (terrain_cols or names)]
        nrows = terrain_kwargs.get("terrain_rows", 10)
        assert all(0 <= r < nrows for r in terrain_rows), (terrain_rows, nrows)
        jobs = [(c, col, row) for col in cols for row in terrain_rows for c in range(n_files)]
    else:
        jobs = [(c, -1, -1) for c in range(n_files)]
    n_jobs = len(jobs)
    num_envs = min(num_envs or (n_jobs if mixed else 1024), n_jobs)
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_mode(anchor_obs),
        goal_features=goal_features,
        self_collision=self_collision,
        **terrain_kwargs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        max_r = 0.1  # paper reward upper bound per step (dt * 5)
        job_clip = torch.tensor([j[0] for j in jobs], device=device)
        job_tile = torch.tensor([[j[2], j[1]] for j in jobs], device=device)  # (row, col)

        def assign(ids: torch.Tensor, job: torch.Tensor) -> None:
            j = torch.clamp(job, max=n_jobs - 1)  # finished envs replay the last job (ignored)
            cmd.forced_frame[ids] = lib.clip_start[job_clip[j]]
            if mixed:
                cmd.forced_tile[ids] = job_tile[j]

        # Job queue: env e plays jobs e, e+N, e+2N, ...
        all_ids = torch.arange(num_envs, device=device)
        current = all_ids.clone()
        nxt = current + num_envs
        assign(all_ids, current)
        obs, _ = env.reset()
        assign(all_ids, nxt)
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_jobs
        col_names = cmd.layout.col_names if mixed else None
        if mixed:
            tile_centre = cmd.layout.tile_centres(device)  # (rows, cols, 2)
            tile_half = torch.tensor(cmd.layout.tile_size, device=device) * 0.5
            left_tile = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        while remaining > 0:
            obs, rew, term, tout, _ = env.step(act(policy, obs))
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            if mixed:  # pre-reset pelvis of this step's state is gone for done envs: their last check was a step ago
                off = ((cmd.robot_root_pos_w[:, 0:2] - tile_centre[cmd.ep_row, cmd.ep_col]).abs() > tile_half).any(-1)
                left_tile = torch.where((left_tile < 0) & off & ~(term | tout), steps, left_tile)
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ji = int(current[e])
                    if ji < n_jobs and ji not in results:
                        ci, col, row = jobs[ji]
                        st = int(steps[e])
                        # A complete clip of L frames takes L env steps here (the reset already
                        # advances the target to frame 1; the clip-end time-out fires one step after
                        # the last frame is reached, which is scored against the last frame again).
                        full = lib.clips[ci].length
                        if not bool(term[e]) and st != full:
                            print(f"[WARN] {lib.clips[ci].name}: completed in {st} steps, expected {full}")
                        results[ji] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=full,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(full, st, 1)),
                            norm_return_survived=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            terrain=col_names[col] if mixed else None,
                            row=row,
                            left_tile_step=int(left_tile[e]) if mixed else -1,
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_jobs):
                            print(f"  evaluated {len(results)}/{n_jobs}")
                # advance the queue for these envs (the auto-reset already used forced_frame / forced_tile)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_jobs
                active[ids[finished]] = False
                assign(ids, nxt[ids])
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
                if mixed:
                    left_tile[ids] = -1
        if mixed:
            print(f"  terrain: spawn misfits {int(cmd.spawn_misfit)}, off-map pelvis env-steps {int(cmd.tmap.oob_count)}")
        return [results[i] for i in range(n_jobs)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return: total reward over total attainable reward of the whole benchmark.
    tot_frames = sum(r.frames for r in results)
    w_ret = sum(r.norm_return * r.frames for r in results) / max(tot_frames, 1)
    old_mean = sum(r.norm_return_survived for r in results) / max(n, 1)
    tot_steps = sum(r.steps for r in results)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret, "mean_norm_return_survived_v1": old_mean}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def summarize_terrain(results: list[ClipResult]) -> dict:
    """{column: {row: {clips, failed, failed_on_tile, left_tile, mean_norm_return}}} of a mixed benchmark ({}
    otherwise). ``failed_on_tile``: failures before the pelvis first left its tile; ``left_tile``: clips that drifted
    off their tile at some point (their later steps are on neighbouring tiles / the border)."""
    out: dict = {}
    for r in results:
        if r.terrain is None:
            continue
        cell = out.setdefault(r.terrain, {}).setdefault(str(r.row), {"clips": 0, "failed": 0, "failed_on_tile": 0, "left_tile": 0, "ret_sum": 0.0})
        cell["clips"] += 1
        cell["failed"] += int(r.failed)
        cell["failed_on_tile"] += int(r.failed and r.left_tile_step < 0)
        cell["left_tile"] += int(r.left_tile_step >= 0)
        cell["ret_sum"] += r.norm_return
    for rows in out.values():
        for cell in rows.values():
            cell["mean_norm_return"] = cell.pop("ret_sum") / max(cell["clips"], 1)
    return out


def format_terrain_table(table: dict, key: str = "failed") -> str:
    """Column x row table of 'return / <key>' cells (key: failed | failed_on_tile | left_tile)."""
    rows = sorted({int(r) for cells in table.values() for r in cells})
    lines = [f"{'terrain':12s} " + " ".join(f"{'row ' + str(r):>14s}" for r in rows) + f" {'all':>14s}"]
    for name, cells in table.items():
        parts = []
        for r in rows:
            c = cells.get(str(r))
            parts.append(f"{c['mean_norm_return']:.3f} / {c.get(key, 0):>3d}".rjust(14) if c else " " * 14)
        n = sum(c["clips"] for c in cells.values())
        ret = sum(c["mean_norm_return"] * c["clips"] for c in cells.values()) / max(n, 1)
        parts.append(f"{ret:.3f} / {sum(c.get(key, 0) for c in cells.values()):>3d}".rjust(14))
        lines.append(f"{name:12s} " + " ".join(parts))
    return "\n".join(lines)


def save_results(results: list[ClipResult], path: str | Path, meta: dict | None = None) -> None:
    out = {"checkpoint": meta or {}, "summary": summarize(results)}
    table = summarize_terrain(results)
    if table:
        out["terrain"] = table
    out["clips"] = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(out, indent=1))
