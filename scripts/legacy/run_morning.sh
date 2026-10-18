#!/bin/bash
# Morning run: (1) redo the one lost seed=3 config (gsac_high — its dump was lost to a
# killed process), then (2) the full seed=4 sweep. run_seed_sweep.sh skips any config
# whose results/results_500_seed{N}_{cfg}.json already exists, so step 1 only runs gsac_high.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs
set -e

echo "=== $(date) : seed=3 tamamlama (sadece gsac_high) ==="
bash scripts/run_seed_sweep.sh 3 > logs/log_seed3_redo_driver.log 2>&1

echo "=== $(date) : seed=4 tam sweep ==="
bash scripts/run_seed_sweep.sh 4 > logs/log_seed4_driver.log 2>&1

echo "=== $(date) : BITTI — n=5 hazir ==="
