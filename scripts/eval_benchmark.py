"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1

Terrain (docs/VISION_TERRAIN_PLAN.md): ``--terrain plane`` evaluates in the terrain-capable environment on the
flat plane (default for terrain-capable checkpoints; regression vs. the flat benchmark); ``--terrain mixed``
plays every clip once per (column, row) tile of a fixed-seed terrain grid (default: all 9 columns x rows
0/3/6/9) and reports return / failures per column x row:

  uv run python scripts/eval_benchmark.py --ckpt logs/wmt-m3/ckpt_latest.pt --terrain mixed

``--scan-noise-scale 2`` (plan M5-A): the same benchmark with the policy scan corrupted by the training scan noise,
every magnitude doubled (the standard protocol is noise-free).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import tyro

from supermimic.goal_spec import anchor_mode
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    EVAL_TERRAIN_ROWS,
    EVAL_TERRAIN_SEED,
    format_terrain_table,
    load_policy_from_checkpoint,
    run_benchmark,
    save_results,
    summarize,
    summarize_terrain,
    terrain_env_kwargs_for_eval,
)


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    manifest: Path | None = None,
    num_envs: int | None = None,
    device: str = "cuda:0",
    out: Path | None = None,
    terrain: Literal["none", "plane", "mixed"] | None = None,
    terrain_seed: int = EVAL_TERRAIN_SEED,
    terrain_cols: list[str] | None = None,
    terrain_rows: tuple[int, ...] = EVAL_TERRAIN_ROWS,
    scan_noise_scale: float = 0.0,
) -> None:
    """``terrain``: default "plane" for terrain-capable checkpoints, "none" otherwise. ``num_envs``: default
    1024 (flat / plane) or one env per (clip, tile) job (mixed). ``scan_noise_scale`` > 0: policy scan noise =
    the checkpoint's training scan noise with every magnitude times this factor (0: noise-free scan)."""
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    if terrain is None:
        terrain = "plane" if policy.terrain is not None else "none"
    if scan_noise_scale > 0 and terrain == "none":
        raise ValueError("--scan-noise-scale needs a terrain-capable evaluation (--terrain plane | mixed)")
    tkw = terrain_env_kwargs_for_eval(cfg, terrain, terrain_seed, scan_noise_scale=scan_noise_scale)
    if scan_noise_scale > 0:
        print(f"[eval] policy scan noise x{scan_noise_scale:g}: {tkw['scan_noise']}")
    results = run_benchmark(
        policy, files, device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], anchor_obs=anchor_mode(cfg["env"].get("anchor_obs", False)),
        goal_features=cfg["env"].get("goal_features", "v1"), self_collision=cfg["env"].get("self_collision", True),
        terrain_kwargs=tkw, terrain_cols=terrain_cols, terrain_rows=terrain_rows,
    )  # fmt: skip
    s = summarize(results)
    meta = cfg.get("_meta", {})
    meta["eval_terrain"] = {"kind": terrain, **({"seed": tkw.get("terrain_seed"), "rows": list(terrain_rows)} if terrain == "mixed" else {}), "scan_noise_scale": scan_noise_scale}
    hours = (meta.get("train_seconds") or 0.0) / 3600
    print(f"\ncheckpoint: iteration {meta.get('iteration')} ({meta.get('iteration_unit') or 'env steps (v1)'}), env steps {meta.get('env_steps')}, train time {hours:.2f} h, eval terrain {terrain}")
    print(f"{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f} | v1 (survived-steps) {s['mean_norm_return_survived_v1']:.4f}")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    table = summarize_terrain(results)
    if table:
        print(f"\nper terrain column x row (seed {tkw['terrain_seed']}): mean norm return / failed clips (of {len(files)})")
        print(format_terrain_table(table))
        print("\nfailed before the pelvis first left its tile (whole clips drift; later steps are on other tiles):")
        print(format_terrain_table(table, "failed_on_tile"))
        print("\nclips that left their tile at some point:")
        print(format_terrain_table(table, "left_tile"))
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        where = f" [{r.terrain} row {r.row}]" if r.terrain else ""
        print(f"  {r.name:32s}{where} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    suffix = ("" if terrain == "none" else f"_{terrain}") + (f"_scan{scan_noise_scale:g}x" if scan_noise_scale > 0 else "")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}{suffix}.json"
    save_results(results, out, meta)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
