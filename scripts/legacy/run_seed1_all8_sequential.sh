#!/bin/bash
# Second-seed (seed=1) replication of the full 500-episode, 8-config sweep
# (PPO/GPPO/SAC/GSAC x Re=45/300). Strictly sequential — this consumer GPU
# cannot handle multiple concurrent JAX processes without severe contention/
# OOM (discovered empirically in the seed=0 round, see
# run_ppo_gppo_500_sequential.sh / run_sac_gsac_500_sequential.sh comments).
# Outputs are suffixed _seed1 so seed=0 results are never overwritten.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs
source ../venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

run() {
  local script=$1 only=$2 out=$3 log=$4
  echo "=== [seed1] ${only} starting $(date) ==="
  python3 -u "$script" --episodes 500 --seed 1 --only "$only" --out "$out" > "$log" 2>&1
  echo "=== [seed1] ${only} done $(date) ==="
}

run run_real_experiment.py ppo_low   results/results_500_seed1_ppo_low.json   logs/log_500_seed1_ppo_low.log
run run_real_experiment.py ppo_high  results/results_500_seed1_ppo_high.json  logs/log_500_seed1_ppo_high.log
run run_real_experiment.py gppo_low  results/results_500_seed1_gppo_low.json  logs/log_500_seed1_gppo_low.log
run run_real_experiment.py gppo_high results/results_500_seed1_gppo_high.json logs/log_500_seed1_gppo_high.log
run run_sac_experiment.py  sac_low   results/results_500_seed1_sac_low.json   logs/log_500_seed1_sac_low.log
run run_sac_experiment.py  sac_high  results/results_500_seed1_sac_high.json  logs/log_500_seed1_sac_high.log
run run_sac_experiment.py  gsac_low  results/results_500_seed1_gsac_low.json  logs/log_500_seed1_gsac_low.log
run run_sac_experiment.py  gsac_high results/results_500_seed1_gsac_high.json logs/log_500_seed1_gsac_high.log

echo "=== [seed1] ALL 8 CONFIGS DONE $(date) ==="
