#!/bin/bash
# Evaluate new checkpoints of local runs as they appear (Appendix F protocol, all clips of <npz_dir>).
#   CUDA_VISIBLE_DEVICES=7 scripts/eval_watch.sh data/npz/lafan1 <run> [<run> ...]
npz=$1; shift
cd "$(dirname "$0")/.." || exit 1
while true; do
  for run in "$@"; do
    for ck in $(ls logs/$run/ckpt_0*.pt 2>/dev/null | sort); do
      out=logs/$run/benchmark_$(basename $ck .pt).json
      [ -f "$out" ] && continue
      env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt $ck --npz-dir $npz > logs/$run/eval_$(basename $ck .pt).log 2>&1
      echo "$(date +%H:%M) $(basename $ck): $(grep -E 'clips \|' logs/$run/eval_$(basename $ck .pt).log)" >> logs/$run/eval_summary.txt
    done
  done
  sleep 120
done
