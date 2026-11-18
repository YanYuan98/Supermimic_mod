#!/bin/bash
# v5 ablation screen (one seed, 24000 iterations, bf16): supermimic_vibed choices vs the v4 defaults.
#   scripts/run_v5_ablation.sh <round> "<gpus>"       round 1: base A1 B1 C1 D1 E1 E2; round 2: F1 A1B1 C1D1
#   SMOKE=1 scripts/run_v5_ablation.sh all "1 2 ..."  short smoke runs (1024 envs, 60 iterations)
#   SEED=1 scripts/run_v5_ablation.sh "base A1 E2 C1" "4 5 6 7"   extra seeds (runs named v5-<name>-s<seed>)
# Runs the worktree's code with the shared venv (PYTHONPATH -> src; no `uv run`, which would re-point the
# venv's editable install). Each training run gets an eval watcher on the same GPU.
round=${1:-1}
gpus=(${2:-1 2 3 4 5 6 7})
cd "$(dirname "$0")/.." || exit 1
source scripts/env_cuda12_driver.sh
export PYTHONPATH=$PWD/src
mkdir -p logs
A1="--world-model.loss-weighting delta_std"
B1="--world-model.term-loss-weight 0"
C1="--critic.input-norm running --critic.activation elu"
D1="--critic.features vibed"
declare -A V=(
  [base]=""
  [A1]="$A1"
  [B1]="$B1"
  [C1]="$C1"
  [D1]="$D1"
  [E1]="--env.sampler-init zero"
  [E2]="--env.sampler-init zero --env.sampler-credit start_bin --env.sampler-ema-alpha 0.2 --env.sampler-memory-alpha 0.001"
  [F1]="--env.goal-features vibed"
  [A1B1]="$A1 $B1"
  [C1D1]="$C1 $D1"
)
case $round in
  1) names=(base A1 B1 C1 D1 E1 E2) ;;
  2) names=(F1 A1B1 C1D1) ;;
  all) names=(base A1 B1 C1 D1 E1 E2 F1 A1B1 C1D1) ;;
  *) names=($round) ;;
esac
seed=${SEED:-0}
common="--data.npz-dir data/npz/lafan1 --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.total-iterations 24000 --seed $seed --amp bf16"
prefix=v5
suffix=""
[ "$seed" != 0 ] && suffix="-s$seed"
if [ -n "$SMOKE" ]; then
  common="--data.npz-dir data/npz/lafan1 --env.num-envs 1024 --policy-trainer.batch-size 1024 --schedule.total-iterations 60 --schedule.wm-warmup-iterations 20 --schedule.log-interval 10 --schedule.checkpoint-interval 60 --seed 0 --amp bf16"
  prefix=smoke5
fi
for i in "${!names[@]}"; do
  g=${gpus[$((i % ${#gpus[@]}))]}
  n=${names[$i]}; run=$prefix-$n$suffix
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python -m supermimic.scripts.train --run-name $run $common ${V[$n]} > logs/$run.out 2>&1 &
  echo "started $run on GPU $g (pid $!): ${V[$n]}"
  if [ -z "$SMOKE" ]; then
    CUDA_VISIBLE_DEVICES=$g nohup scripts/eval_watch_v5.sh $run > logs/eval_watch_$run.out 2>&1 &
  fi
done
