#!/bin/bash
# Evaluate new checkpoints of one run as they appear (Appendix F protocol, 40 LAFAN1 clips); exits after
# the final (24000) checkpoint is scored.   CUDA_VISIBLE_DEVICES=<g> scripts/eval_watch_v5.sh <run>
run=$1; final=${2:-0024000}
cd "$(dirname "$0")/.." || exit 1
source scripts/env_cuda12_driver.sh
export PYTHONPATH=$PWD/src
while true; do
  for ck in $(ls logs/$run/ckpt_0*.pt 2>/dev/null | sort); do
    out=logs/$run/benchmark_$(basename $ck .pt).json
    [ -f "$out" ] && continue
    sleep 20  # let torch.save finish
    .venv/bin/python scripts/eval_benchmark.py --ckpt $ck --npz-dir data/npz/lafan1 > logs/$run/eval_$(basename $ck .pt).log 2>&1
    echo "$(date +%H:%M) $(basename $ck): $(grep -E 'clips \|' logs/$run/eval_$(basename $ck .pt).log)" >> logs/$run/eval_summary.txt
  done
  [ -f logs/$run/benchmark_ckpt_$final.json ] && exit 0
  sleep 120
done
