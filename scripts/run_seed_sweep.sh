#!/bin/bash
# Portable, parametrized replacement for run_seed{1,2}_all8_sequential.sh.
# Runs the full 500-episode, 8-config sweep (PPO/GPPO/SAC/GSAC x Re=45/300)
# for ONE seed, strictly one config at a time (this class of consumer GPU
# cannot run concurrent JAX processes without contention/OOM).
#
#   bash scripts/run_seed_sweep.sh <SEED>
#   nohup bash scripts/run_seed_sweep.sh 2 > logs/log_seed2_driver.log 2>&1 &
#
# Outputs: results/results_500_seed<SEED>_<config>.json  +  logs/log_500_seed<SEED>_<config>.log
# Wall time reference (seed=1, RTX 5060 laptop 8 GB): ~8h34m total.
#   ppo_low/high ~27m each, gppo_low/high ~57m each, sac_low/high ~29m each,
#   gsac_low ~3h34m (analytic-gradient outlier), gsac_high ~1h14m.
set -e

SEED="${1:?usage: run_seed_sweep.sh <SEED>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
cd "$HERE"
mkdir -p results logs

# activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$HERE/../venv/bin/activate"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

run() {
  local script=$1 only=$2
  local out="results/results_500_seed${SEED}_${only}.json"
  local log="logs/log_500_seed${SEED}_${only}.log"
  if [ -f "$out" ]; then
    echo "=== [seed$SEED] $only already done ($out) — skipping $(date)"
    return
  fi
  echo "=== [seed$SEED] $only starting $(date) ==="
  python3 -u "$script" --episodes 500 --seed "$SEED" --only "$only" --out "$out" > "$log" 2>&1
  echo "=== [seed$SEED] $only done $(date) ==="
}

run experiments/run_real_experiment.py ppo_low
run experiments/run_real_experiment.py ppo_high
run experiments/run_real_experiment.py gppo_low
run experiments/run_real_experiment.py gppo_high
run experiments/run_sac_experiment.py  sac_low
run experiments/run_sac_experiment.py  sac_high
run experiments/run_sac_experiment.py  gsac_low
run experiments/run_sac_experiment.py  gsac_high

echo "=== [seed$SEED] ALL 8 CONFIGS DONE $(date) ==="
