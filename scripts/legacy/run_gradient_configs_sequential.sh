#!/bin/bash
# Runs the 4 memory-hungry (analytic-gradient) configs strictly one at a time
# to avoid GPU OOM, while PPO/SAC (no analytic gradient) keep running in
# parallel in their own separate processes.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs
source ../venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

echo "=== [sequential] gppo_low starting $(date) ==="
python3 experiments/run_real_experiment.py --episodes 500 --only gppo_low --out results/results_500_gppo_low.json >> logs/log_500_gppo_low.log 2>&1
echo "=== [sequential] gppo_low done $(date) ==="

echo "=== [sequential] gppo_high starting $(date) ==="
python3 experiments/run_real_experiment.py --episodes 500 --only gppo_high --out results/results_500_gppo_high.json >> logs/log_500_gppo_high.log 2>&1
echo "=== [sequential] gppo_high done $(date) ==="

echo "=== [sequential] gsac_low starting $(date) ==="
python3 experiments/run_sac_experiment.py --episodes 500 --only gsac_low --out results/results_500_gsac_low.json >> logs/log_500_gsac_low.log 2>&1
echo "=== [sequential] gsac_low done $(date) ==="

echo "=== [sequential] gsac_high starting $(date) ==="
python3 experiments/run_sac_experiment.py --episodes 500 --only gsac_high --out results/results_500_gsac_high.json >> logs/log_500_gsac_high.log 2>&1
echo "=== [sequential] gsac_high done $(date) ==="

echo "=== [sequential] ALL GRADIENT CONFIGS DONE $(date) ==="
