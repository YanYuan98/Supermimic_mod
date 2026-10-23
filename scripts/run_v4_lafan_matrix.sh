#!/bin/bash
# v4 LAFAN1 experiment matrix (docs/EXPERIMENTS.md, v4): default (v3) configuration x 3 seeds + 3 ablations, 3 h cap each.
#   scripts/run_v4_lafan_matrix.sh "1 2 3 4 5 6"     # GPUs for the six runs (in the order below)
# Set UV_NO_SYNC=1 when the venv carries machine-specific torch/warp builds (e.g. driver < 580).
gpus=(${1:-1 2 3 4 5 6})
hours=${HOURS:-3}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
common="--data.npz-dir data/npz/lafan1 --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours $hours"
runs=(
  "v4-s0|--seed 0"
  "v4-s1|--seed 1"
  "v4-s2|--seed 2"
  "v4-selfcol|--env.self-collision"
  "v4-wmfixed|--world-model.normalization fixed --world-model.no-layer-norm --world-model.termination learned"
  "v4-polfixed|--policy.input-norm fixed --policy.activation silu --policy-trainer.start-margin 0"
)
for i in "${!runs[@]}"; do
  [ $i -ge ${#gpus[@]} ] && break
  name=${runs[$i]%%|*}; extra=${runs[$i]#*|}
  CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup env -u PYTHONPATH uv run sm-train --run-name $name $common $extra > logs/$name.out 2>&1 &
  echo "started $name on GPU ${gpus[$i]} (pid $!)"
done
