#!/bin/bash
# Second-seed (seed=1) replication of the full 500-episode, 8-config sweep
# (PPO/GPPO/SAC/GSAC x Re=45/300). Strictly sequential — this consumer GPU
# cannot handle multiple concurrent JAX processes without severe contention/
# OOM (discovered empirically in the seed=0 round, see
# run_ppo_gppo_500_sequential.sh / run_sac_gsac_500_sequential.sh comments).
# Outputs are suffixed _seed2 so seed=0 results are never overwritten.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

run() {
  local script=$1 only=$2 out=$3 log=$4
  echo "=== [seed2] ${only} starting $(date) ==="
  python3 -u "$script" --episodes 500 --seed 2 --only "$only" --out "$out" > "$log" 2>&1
  echo "=== [seed2] ${only} done $(date) ==="
}

run run_real_experiment.py ppo_low   results_500_seed2_ppo_low.json   log_500_seed2_ppo_low.log
run run_real_experiment.py ppo_high  results_500_seed2_ppo_high.json  log_500_seed2_ppo_high.log
run run_real_experiment.py gppo_low  results_500_seed2_gppo_low.json  log_500_seed2_gppo_low.log
run run_real_experiment.py gppo_high results_500_seed2_gppo_high.json log_500_seed2_gppo_high.log
run run_sac_experiment.py  sac_low   results_500_seed2_sac_low.json   log_500_seed2_sac_low.log
run run_sac_experiment.py  sac_high  results_500_seed2_sac_high.json  log_500_seed2_sac_high.log
run run_sac_experiment.py  gsac_low  results_500_seed2_gsac_low.json  log_500_seed2_gsac_low.log
run run_sac_experiment.py  gsac_high results_500_seed2_gsac_high.json log_500_seed2_gsac_high.log

echo "=== [seed2] ALL 8 CONFIGS DONE $(date) ==="
