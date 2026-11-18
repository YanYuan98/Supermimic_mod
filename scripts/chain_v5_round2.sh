#!/bin/bash
# Start v5 round 2 on GPUs 1-3 once every round-1 training process has exited.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "run-name v5-(base|A1|B1|C1|D1|E1|E2) " >/dev/null; do sleep 60; done
echo "$(date '+%F %T') round 1 trainings finished; starting round 2"
scripts/run_v5_ablation.sh 2 "1 2 3"
