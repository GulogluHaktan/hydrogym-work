#!/bin/bash
# Strictly sequential (this consumer GPU cannot handle multiple concurrent
# JAX processes without severe contention/OOM, as discovered empirically
# with the PPO/GPPO 500-episode run). SAC + GSAC, 500 episodes each, both
# Re regimes.
set -e
cd /home/haktan/Projects/HydroGymDeney/hydrogym_real
source ../venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

echo "=== [seq] sac_low starting $(date) ==="
python3 -u run_sac_experiment.py --episodes 500 --only sac_low --out results_500_sac_low.json > log_500_sac_low.log 2>&1
echo "=== [seq] sac_low done $(date) ==="

echo "=== [seq] sac_high starting $(date) ==="
python3 -u run_sac_experiment.py --episodes 500 --only sac_high --out results_500_sac_high.json > log_500_sac_high.log 2>&1
echo "=== [seq] sac_high done $(date) ==="

echo "=== [seq] gsac_low starting $(date) ==="
python3 -u run_sac_experiment.py --episodes 500 --only gsac_low --out results_500_gsac_low.json > log_500_gsac_low.log 2>&1
echo "=== [seq] gsac_low done $(date) ==="

echo "=== [seq] gsac_high starting $(date) ==="
python3 -u run_sac_experiment.py --episodes 500 --only gsac_high --out results_500_gsac_high.json > log_500_gsac_high.log 2>&1
echo "=== [seq] gsac_high done $(date) ==="

echo "=== [seq] SAC+GSAC ALL DONE $(date) ==="
