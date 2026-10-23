#!/bin/bash
# v2 LAFAN1 experiment matrix (docs/IMPROVEMENT_PLAN.md §3): one run per GPU, 3 h cap each.
#   scripts/run_v2_lafan_matrix.sh "1 2 3 4 5 6"     # GPUs for the six runs (in the order below)
# Set UV_NO_SYNC=1 when the venv carries machine-specific torch/warp builds (e.g. driver < 580).
gpus=(${1:-1 2 3 4 5 6})
hours=${HOURS:-3}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
# v2 settings spelled out (the defaults moved to the v3 configuration afterwards).
v2="--env.goal-features v1 --env.self-collision --world-model.normalization fixed --world-model.no-layer-norm --world-model.termination learned --policy.input-norm fixed --policy.activation silu --policy-trainer.start-margin 0"
common="--data.npz-dir data/npz/lafan1 --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours $hours $v2"
runs=(
  "v2-ori|"
  "v2-ori-traj|--env.goal-features traj"
  "v2-none|--env.anchor-obs none"
  "v2-posori|--env.anchor-obs pos_ori"
  "v2-ori-oldcritic|--critic.no-error-features"
  "v2-ori-bf16|--amp bf16"
)
for i in "${!runs[@]}"; do
  [ $i -ge ${#gpus[@]} ] && break
  name=${runs[$i]%%|*}; extra=${runs[$i]#*|}
  CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup env -u PYTHONPATH uv run sm-train --run-name $name $common $extra > logs/$name.out 2>&1 &
  echo "started $name on GPU ${gpus[$i]} (pid $!)"
done
