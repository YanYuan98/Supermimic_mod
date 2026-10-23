"""Tabulate the benchmark results of runs (from logs/<run>/benchmark_ckpt_*.json written by eval_benchmark.py).

  uv run python scripts/summarize_runs.py v4-s0 v4-s1 v4-s2 [--target 0.82]
"""

from __future__ import annotations

import json
from pathlib import Path

import tyro


def main(runs: tyro.conf.Positional[list[str]], log_root: Path = Path("logs"), target: float = 0.82) -> None:
    for run in runs:
        rows = []
        for f in sorted((log_root / run).glob("benchmark_ckpt_*.json")):
            d = json.loads(f.read_text())
            meta, s = d.get("checkpoint", {}), d["summary"]
            hours = (meta.get("train_seconds") or 0.0) / 3600
            rows.append((meta.get("iteration") or 0, hours, s["mean_norm_return"], s["failed"], s["clips"], f.stem.replace("benchmark_", "")))
        rows.sort()
        print(f"== {run}")
        first = next((r for r in rows if r[2] >= target), None)
        for it, h, ret, failed, n, name in rows:
            mark = "  <- first >= target" if first is not None and (it, name) == (first[0], first[5]) else ""
            print(f"  {name:18s} it {it:6d}  {h:5.2f} h  return {ret:.4f}  failed {failed}/{n}{mark}")
        if first is None:
            print(f"  (target {target} not reached)")


if __name__ == "__main__":
    tyro.cli(main)
