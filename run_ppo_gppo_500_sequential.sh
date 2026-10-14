#!/bin/bash
# Strictly sequential (no parallelism — a consumer GPU here cannot handle
# multiple concurrent JAX processes without severe contention/stalls, as
# discovered empirically). PPO + GPPO only, 500 episodes each, both Re
# regimes, matching the Nature paper's Fig 3b panel exactly.
set -e
cd /home/haktan/Projects/HydroGymDeney/hydrogym_real
source ../venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

echo "=== [seq] ppo_low starting $(date) ==="
python3 -u run_real_experiment.py --episodes 500 --only ppo_low --out results_500_ppo_low.json > log_500_ppo_low.log 2>&1
echo "=== [seq] ppo_low done $(date) ==="

echo "=== [seq] ppo_high starting $(date) ==="
python3 -u run_real_experiment.py --episodes 500 --only ppo_high --out results_500_ppo_high.json > log_500_ppo_high.log 2>&1
echo "=== [seq] ppo_high done $(date) ==="

echo "=== [seq] gppo_low starting $(date) ==="
python3 -u run_real_experiment.py --episodes 500 --only gppo_low --out results_500_gppo_low.json > log_500_gppo_low.log 2>&1
echo "=== [seq] gppo_low done $(date) ==="

echo "=== [seq] gppo_high starting $(date) ==="
python3 -u run_real_experiment.py --episodes 500 --only gppo_high --out results_500_gppo_high.json > log_500_gppo_high.log 2>&1
echo "=== [seq] gppo_high done $(date) ==="

echo "=== [seq] PPO+GPPO ALL DONE $(date) ==="
