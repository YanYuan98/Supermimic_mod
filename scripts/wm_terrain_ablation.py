"""World-model terrain-condition ablation on one shared replay stream (docs/VISION_TERRAIN_PLAN.md M2).

A ``mixed`` env with the curriculum level pinned (spawn rows ~ U{0..L}) is driven by a frozen policy (no policy /
critic updates, input normalisers untouched); every iteration collects 1/UTD env steps like the training loop,
samples ``batches_per_update`` window batches once and updates N world models (``--modes`` = none | foot | scan |
both) on exactly those batches with ``WorldModelTrainer.update_on`` (the training code path: same normaliser
updates, lifted reference roots, ``window_loss``). Each model is warm-started from ``--init-from`` like the
trainer's ``--init-from`` (new terrain input columns zero-initialised) and may live on its own GPU (the batches
are copied; the env runs on the first device). A held-out fraction of the envs is never trained on; their windows
are the evaluation set (same windows for every model):

* open-loop ``nse_1step`` / ``nse_kstep`` (mean normalised squared state error after 1 / k steps) overall, per
  terrain column and row, on the flat column, on all non-flat columns ("terrain") and on windows where a sole
  point of the logged states comes within ``edge_radius`` (xy) of a step edge (neighbour height difference >
  ``edge_threshold``; distance map precomputed once from the height map, then a lookup);
* termination accuracy / recall of the termination head on the same windows.

At the end every model is saved (``wm_<mode>.pt``) together with an ``--init-from`` checkpoint ``init_<mode>.pt`` (the
flat checkpoint's policy / critic + the terrain-trained world model) for the terrain policy training.

The final table applies the M2 gate: terrain nse_kstep <= 1.5 x flat, and the near-edge error vs ``none``.

  CUDA_VISIBLE_DEVICES=3,4,5,6 uv run python scripts/wm_terrain_ablation.py --run-name wmt-c-m2 \\
      --devices cuda:0 cuda:1 cuda:2 cuda:3 --max-hours 1.25
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.config import TerrainCfg, terrain_env_kwargs
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import WindowBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.eval.benchmark import act, load_policy_from_checkpoint
from supermimic.goal_spec import anchor_mode
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.terrain import HeightGrid, TerrainInputs, TerrainMap
from supermimic.training.schedule import cosine_lr
from supermimic.training.warm_start import NEW_PREFIXES, WM_IN, load_expanded
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

STAT_KEYS = ("n", "nse_1", "nse_k", "term_acc", "term_pos", "term_recall")


def _tuple(v):
    return tuple(v) if isinstance(v, list) else v


def wm_cfg_from(d: dict, terrain: str) -> WorldModelCfg:
    d = {k: _tuple(v) for k, v in d.items()}
    d["state_spec"] = StateSpecCfg(**{k: _tuple(v) for k, v in d["state_spec"].items()})
    d["terrain"] = terrain
    return WorldModelCfg(**d)


def to_device(w: WindowBatch, device: torch.device) -> WindowBatch:
    """The fields ``WorldModelTrainer`` reads, on ``device`` (a_clean / reward are not used)."""
    if w.states.device == device:
        return w
    f = lambda x: x.to(device, non_blocking=True)  # noqa: E731
    return WindowBatch(f(w.states), f(w.a_exec), None, f(w.xi), None, f(w.done), f(w.frame), f(w.origin))


class Model:
    """One world model + trainer on its device, with its own copies of the map / FK / library."""

    def __init__(self, mode: str, device: torch.device, cmd, robot, wm_dict: dict, init_sd: dict | None, tcfg: WMTrainerCfg, tmap0: TerrainMap):
        self.mode, self.device = mode, device
        lib0 = cmd.library
        if device == lib0.device:
            lib, tmap = lib0, tmap0
        else:
            c = cmd.cfg
            lib = MotionLibrary(c.npz_files, c.body_names, robot.data.default_joint_pos[0].to(device), device=device, bin_frames=c.bin_frames, compact=c.compact_library, store_dtype=torch.float16 if c.library_fp16 else torch.float32)
            g = tmap0.grid
            tmap = TerrainMap(HeightGrid(g.h.to(device), g.g.to(device)), tmap0.spec, tmap0.feet, device, tmap0.layout, tmap0.info)
        self.lib, self.tmap = lib, tmap
        self.fk = G1Kinematics(load_g1_model(), device=device)
        heads = AnalyticHeads(self.fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].to(device), robot.data.default_joint_pos[0].to(device))
        self.terrain = TerrainInputs(tmap, self.fk, mode)
        self.wm = WorldModel(wm_cfg_from(wm_dict, mode), heads, device, terrain=self.terrain)
        self.fresh = load_expanded(self.wm, init_sd, WM_IN, init_sd[WM_IN].shape[1], NEW_PREFIXES["wm"]) if init_sd is not None else None
        self.trainer = WorldModelTrainer(self.wm, lib, tcfg, terrain=self.terrain)

    @torch.no_grad()
    def window_eval(self, w: WindowBatch, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        """(err (k, n), term logits (k, n)) of the windows (on this model's device)."""
        errs, logits = [], []
        n = w.states.shape[1]
        for i in range(0, n, chunk):
            s = slice(i, i + chunk)
            wi = WindowBatch(w.states[:, s], w.a_exec[:, s], None, w.xi[s], None, w.done[:, s], w.frame[:, s], w.origin[s])
            rp, rq = self.trainer.reference_roots(wi)
            e, lg = self.wm.window_errors(wi.states, wi.a_exec, wi.xi, rp, rq, origin=wi.origin, grid=self.terrain.grid)
            errs.append(e)
            logits.append(lg)
        return torch.cat(errs, 1), torch.cat(logits, 1)


def group_stats(err: torch.Tensor, logits: torch.Tensor, done: torch.Tensor, sel: torch.Tensor) -> dict:
    """Mean metrics over the windows in ``sel`` (n,) bool."""
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    e = err[:, sel]
    pred = (logits[:, sel] > 0).float()
    d = done[:, sel]
    pos = d.sum()
    return {
        "n": n,
        "nse_1": float(e[0].mean()),
        "nse_k": float(e[-1].mean()),
        "term_acc": float((pred == d).float().mean()),
        "term_pos": int(pos),
        "term_recall": float((pred * d).sum() / pos) if pos > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate(models: list[Model], w: WindowBatch, cmd, edge_threshold: float, edge_radius: float) -> dict:
    """Metrics of every model on the same held-out windows, grouped by tile / near-edge flag."""
    lay = cmd.layout
    tmap, fk = cmd.tmap, cmd.fk
    k1, n = w.states.shape[:2]
    root_xy = w.states[0, :, 0:2] + w.origin[:, 0:2]
    hx, hy = lay.grid_half_extent
    row = ((root_xy[:, 0] + hx) / lay.tile_size[0]).floor().long()
    col = ((root_xy[:, 1] + hy) / lay.tile_size[1]).floor().long()
    on_grid = (row >= 0) & (row < lay.num_rows) & (col >= 0) & (col < lay.num_cols)
    sole = tmap.foot_points(w.states, w.origin[None], fk).sole  # (k+1, n, 2, 5, 3)
    edist = tmap.edge_dist(sole[..., 0:2], edge_threshold).flatten(2).amin(-1).amin(0)  # (n,)
    near = edist < edge_radius
    flat_col = lay.col_index("flat")
    groups = {
        "all": torch.ones(n, dtype=torch.bool, device=w.states.device),
        "flat": on_grid & (col == flat_col),
        "terrain": on_grid & (col != flat_col),
        "edge": on_grid & near,
        "terrain_no_edge": on_grid & (col != flat_col) & ~near,
    }
    for c, name in enumerate(lay.col_names):
        groups[f"col/{name}"] = on_grid & (col == c)
        groups[f"edge/{name}"] = on_grid & (col == c) & near
    for r in range(lay.num_rows):
        groups[f"row/{r}"] = on_grid & (row == r) & (col != flat_col)
    out = {"_counts": {"windows": n, "off_grid": int((~on_grid).sum()), "near_edge": int(near.sum())}}
    for m in models:
        wd = to_device(w, m.device)
        err, logits = m.window_eval(wd)
        err, logits = err.to(w.states.device), logits.to(w.states.device)
        out[m.mode] = {g: group_stats(err, logits, w.done, sel) for g, sel in groups.items()}
    return out


def format_table(res: dict, modes: list[str], gate_ratio: float) -> str:
    cols = ["all", "flat", "terrain", "edge", "terrain_no_edge"]
    lines = [f"{'model':6s} " + " ".join(f"{c + ' nse1/nsek':>24s}" for c in cols) + f" {'terr/flat':>9s} {'edge/none':>9s} {'term acc':>8s} {'recall':>6s}"]
    for m in modes:
        r = res[m]
        cells = [f"{r[c].get('nse_1', float('nan')):.4f}/{r[c].get('nse_k', float('nan')):.4f}".rjust(24) for c in cols]
        ratio = r["terrain"]["nse_k"] / r["flat"]["nse_k"]
        edge_rel = r["edge"]["nse_k"] / res["none"]["edge"]["nse_k"] if "none" in res else float("nan")
        lines.append(f"{m:6s} " + " ".join(cells) + f" {ratio:9.2f} {edge_rel:9.2f} {r['all']['term_acc']:8.4f} {r['all']['term_recall']:6.3f}" + ("" if ratio <= gate_ratio else "  (gate fail)"))
    return "\n".join(lines)


def format_columns(res: dict, modes: list[str], names: tuple[str, ...]) -> str:
    lines = [f"{'nse_k by column':16s} " + " ".join(f"{m:>16s}" for m in modes) + "   (near-edge nse_k in brackets)"]
    for c in names:
        parts = []
        for m in modes:
            a, e = res[m][f"col/{c}"], res[m][f"edge/{c}"]
            parts.append((f"{a.get('nse_k', float('nan')):.4f}" + (f" [{e['nse_k']:.3f}]" if e.get("n", 0) >= 20 else "")).rjust(16))
        lines.append(f"{c:16s} " + " ".join(parts) + f"   n={res[modes[0]][f'col/{c}']['n']}/{res[modes[0]][f'edge/{c}']['n']}")
    return "\n".join(lines)


@torch.no_grad()
def main(
    run_name: str = "wmt-c-m2",
    log_root: Path = Path("logs"),
    policy_ckpt: Path = Path("logs/wmt-m0-warmstart/ckpt_latest.pt"),
    init_from: Path | None = Path("logs/g1-selfcol-2gpu/ckpt_latest.pt"),
    modes: tuple[str, ...] = ("none", "foot", "scan", "both"),
    devices: tuple[str, ...] = ("cuda:0",),
    level: int = 9,
    terrain_seed: int = 0,
    num_envs: int = 4096,
    holdout: float = 0.1,
    window: int = 8,
    eval_window: int | None = None,
    max_iterations: int = 1_000_000,
    max_hours: float = 1.25,
    eval_every: int = 500,
    eval_windows: int = 16384,
    final_eval_windows: int = 131072,
    edge_threshold: float = 0.03,
    edge_radius: float = 0.10,
    gate_ratio: float = 1.5,
    edge_weight: float = 1.0,
    log_interval: int = 100,
    seed: int = 0,
) -> None:
    """``init_from``: flat checkpoint whose world model warm-starts every model (None: fresh models). ``devices``:
    the env runs on devices[0], model i on devices[i % len(devices)]. ``edge_weight`` > 1 (plan §7 mitigation):
    training windows whose logged states put a sole point within ``edge_radius`` of a step edge are sampled with
    that relative weight. ``eval_window`` (default ``window``): length of the evaluation windows (nse_k is reported
    at this k; e.g. train with window 12, evaluate nse_8 to compare with k = 8 runs)."""
    t_start = time.time()
    ek = eval_window or window
    log_dir = Path(log_root) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    dev0 = torch.device(devices[0])
    torch.manual_seed(seed)
    policy, pcfg = load_policy_from_checkpoint(policy_ckpt, str(dev0))  # frozen (eval mode, never updated)
    ecfg = pcfg["env"]
    t = dict(pcfg.get("terrain") or asdict(TerrainCfg()))
    t.update(kind="mixed", seed=terrain_seed)
    t["curriculum"] = dict(t["curriculum"], start_level=level)
    data = pcfg.get("data") or {}
    files = [str(f) for f in discover_npz(Path(data.get("npz_dir", "data/npz/lafan1")), names=data.get("clips"), limit=data.get("limit"))]
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, num_envs=num_envs, future_frames=ecfg["future_frames"], episode_length_s=ecfg["episode_length_s"],
            action_noise_std=ecfg["action_noise_std"], obs_noise=ecfg["obs_noise"], reset_noise=ecfg["reset_noise"],
            domain_randomization=ecfg["domain_randomization"], sampling_mode=ecfg["sampling_mode"], replay_capacity=ecfg["replay_capacity"],
            anchor_obs=anchor_mode(ecfg["anchor_obs"]), goal_features=ecfg["goal_features"], self_collision=ecfg["self_collision"], seed=seed,
            **terrain_env_kwargs(t),
        ),
        device=str(dev0),
    )  # fmt: skip
    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    buffer = env.recorder_manager.get_term("replay").buffer
    init = torch.load(init_from, map_location="cpu", weights_only=False) if init_from is not None else None
    wm_dict = (init or {}).get("config", pcfg)["world_model"]
    tcfg = WMTrainerCfg(**{**asdict(WMTrainerCfg()), "window": window})
    models = []
    for i, mode in enumerate(modes):
        d = torch.device(devices[i % len(devices)])
        sd = None if init is None else {k: v.to(d) for k, v in init["wm"].items()}
        models.append(Model(mode, d, cmd, robot, wm_dict, sd, tcfg, cmd.tmap))
        print(f"[M2] model {mode} on {d}" + (f" (warm start; fresh {models[-1].fresh})" if init is not None else " (fresh)"), flush=True)
    # Held-out envs: their windows are never trained on (evaluation set).
    gen = torch.Generator(device="cpu").manual_seed(seed)
    held = torch.zeros(num_envs, dtype=torch.bool)
    held[torch.randperm(num_envs, generator=gen)[: int(round(holdout * num_envs))]] = True
    held = held.to(dev0)
    cmd.tmap.edge_distance_field(edge_threshold)  # once (host EDT), then lookups
    edge_row = torch.zeros(buffer.E, buffer.T, dtype=torch.bool, device=dev0) if edge_weight != 1.0 else None
    env_ar = torch.arange(buffer.E, device=dev0)

    def mark_edge_rows(n_new: int) -> None:
        """Near-edge flag of the rows written by the last ``n_new`` env steps (+ terminal rows)."""
        for j in range(n_new + 2):
            t_ = (buffer.ptr - 1 - j) % buffer.T
            sole = cmd.tmap.foot_points(buffer.s[env_ar, t_], buffer.origin[env_ar, t_], cmd.fk).sole
            edge_row[env_ar, t_] = cmd.tmap.edge_dist(sole[..., 0:2], edge_threshold).flatten(1).amin(-1) < edge_radius
    (log_dir / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in locals().items() if k in ("run_name", "policy_ckpt", "init_from", "modes", "devices", "level", "terrain_seed", "num_envs", "holdout", "window", "eval_window", "max_hours", "eval_every", "eval_windows", "final_eval_windows", "edge_threshold", "edge_radius", "edge_weight", "seed")}, indent=1, default=str))
    jsonl = open(log_dir / "metrics.jsonl", "a")

    obs, _ = env.reset()
    utd = 0.4
    warmup = 40
    for _ in range(warmup):
        obs, *_ = env.step(act(policy, obs))
    if edge_row is not None:
        mark_edge_rows(warmup)
    acc, it = 0.0, 0
    agg: dict[str, dict[str, torch.Tensor]] = {m.mode: {} for m in models}
    agg_n = 0
    t_sim = t_wm = 0.0
    last = time.time()
    while it < max_iterations:
        t0 = time.perf_counter()
        acc += 1.0 / utd
        n_steps = int(acc)
        acc -= n_steps
        for _ in range(n_steps):
            obs, *_ = env.step(act(policy, obs))
        t1 = time.perf_counter()
        t_sim += t1 - t0
        lr = cosine_lr(it, tcfg.lr)
        mask = buffer.window_mask(window) & ~held[:, None]
        if edge_row is not None:
            mark_edge_rows(n_steps)
            near = edge_row.clone()
            for j in range(1, window + 1):
                near |= torch.roll(edge_row, -j, dims=1)
            mask = mask.float() * torch.where(near, edge_weight, 1.0)
        batches = [buffer.sample_windows(tcfg.batch_size, window, mask=mask) for _ in range(tcfg.batches_per_update)]
        for m in models:  # no host sync inside: models on other GPUs run concurrently
            m.trainer.set_lr(lr)
            with torch.enable_grad():
                met = m.trainer.update_on([to_device(w, m.device) for w in batches], sync=False)
            a = agg[m.mode]
            for k, v in met.items():
                a[k] = a.get(k, 0.0) + v
        agg_n += 1
        it += 1
        if it % log_interval == 0 or it == 1:
            for m in models:
                torch.cuda.synchronize(m.device)
            t_wm += time.perf_counter() - t1
            hours = (time.time() - t_start) / 3600
            msg = " | ".join(f"{mode} nse1 {float(a['nse_1step']) / agg_n:.4f} nse{window} {float(a['nse_kstep']) / agg_n:.4f}" for mode, a in agg.items())
            print(f"[M2] it {it} | {hours:.2f} h | {log_interval / (time.time() - last):.2f} it/s | sim {t_sim / max(it, 1):.3f} s/it | {msg}", flush=True)
            jsonl.write(json.dumps({"it": it, "hours": hours, "train": {mode: {k: float(v) / agg_n for k, v in a.items()} for mode, a in agg.items()}}) + "\n")
            jsonl.flush()
            agg = {m.mode: {} for m in models}
            agg_n = 0
            last = time.time()
        else:
            t_wm += time.perf_counter() - t1
        stop = (time.time() - t_start) / 3600 > max_hours
        if it % eval_every == 0 or stop:
            emask = buffer.window_mask(ek) & held[:, None]
            res = evaluate(models, buffer.sample_windows(eval_windows, ek, mask=emask), cmd, edge_threshold, edge_radius)
            jsonl.write(json.dumps({"it": it, "hours": (time.time() - t_start) / 3600, "eval": res}) + "\n")
            jsonl.flush()
            print(f"[M2] eval it {it} ({res['_counts']}):\n" + format_table(res, list(modes), gate_ratio), flush=True)
        if stop:
            break

    # Final evaluation on a large held-out sample, the gate, and the trained models.
    emask = buffer.window_mask(ek) & held[:, None]
    parts = [evaluate(models, buffer.sample_windows(16384, ek, mask=emask), cmd, edge_threshold, edge_radius) for _ in range(max(1, final_eval_windows // 16384))]
    res = merge_evals(parts, list(modes))
    gate = {}
    for m in modes:
        r = res[m]
        gate[m] = {
            "terrain_over_flat_nse_k": r["terrain"]["nse_k"] / r["flat"]["nse_k"],
            "edge_over_flat_nse_k": r["edge"]["nse_k"] / r["flat"]["nse_k"],
            "edge_nse_k_vs_none": r["edge"]["nse_k"] / res["none"]["edge"]["nse_k"] if "none" in res else None,
            "pass_terrain": r["terrain"]["nse_k"] <= gate_ratio * r["flat"]["nse_k"],
            "worst_column_over_flat": max((r[f"col/{c}"]["nse_k"] / r["flat"]["nse_k"], c) for c in cmd.layout.col_names if c != "flat"),
        }
    out = {"iterations": it, "hours": (time.time() - t_start) / 3600, "sim_s_per_it": t_sim / max(it, 1), "wm_s_per_it": t_wm / max(it, 1), "eval": res, "gate": gate}
    (log_dir / "results.json").write_text(json.dumps(out, indent=1))
    print(f"\n[M2] final ({it} iterations, {out['hours']:.2f} h, {res['_counts']}):\n" + format_table(res, list(modes), gate_ratio))
    print("\n" + format_columns(res, list(modes), cmd.layout.col_names))
    print("\ngate:", json.dumps(gate, indent=1))
    for m in models:
        sd = {k: v.cpu() for k, v in m.wm.state_dict().items()}
        torch.save({"wm": sd, "mode": m.mode, "iterations": it, "config": {"world_model": asdict(m.wm.cfg)}}, log_dir / f"wm_{m.mode}.pt")
        if init is not None and m.mode != "none":
            # --init-from checkpoint for terrain training: the flat policy / critic + this terrain-trained world model
            # (warm_start loads it as is, terr_norm included), so the world-model warm-up can be short.
            torch.save({**init, "wm": sd, "m2": {"mode": m.mode, "iterations": it, "run": run_name}}, log_dir / f"init_{m.mode}.pt")
    env.close()


def merge_evals(parts: list[dict], modes: list[str]) -> dict:
    """Window-count weighted merge of several ``evaluate`` results."""
    out: dict = {"_counts": {k: sum(p["_counts"][k] for p in parts) for k in parts[0]["_counts"]}}
    for m in modes:
        out[m] = {}
        for g in parts[0][m]:
            cells = [p[m][g] for p in parts if p[m][g].get("n", 0) > 0]
            n = sum(c["n"] for c in cells)
            if n == 0:
                out[m][g] = {"n": 0}
                continue
            cell = {"n": n}
            for k in ("nse_1", "nse_k", "term_acc"):
                cell[k] = sum(c[k] * c["n"] for c in cells) / n
            pos = sum(c["term_pos"] for c in cells)
            cell["term_pos"] = pos
            cell["term_recall"] = sum(c["term_recall"] * c["term_pos"] for c in cells if c["term_pos"] > 0) / pos if pos > 0 else float("nan")
            out[m][g] = cell
    return out


if __name__ == "__main__":
    tyro.cli(main)
