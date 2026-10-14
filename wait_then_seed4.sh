#!/bin/bash
# Waits for the seed=3 sweep to finish, then launches the seed=4 sweep (5th seed, n=5).
# Launch detached:  nohup bash wait_then_seed4.sh > log_seed4_chain.log 2>&1 &
cd "$(dirname "${BASH_SOURCE[0]}")"
until grep -q "ALL 8 CONFIGS DONE" log_seed3_driver.log 2>/dev/null; do
  sleep 60
done
echo "=== [chain] seed3 confirmed done, launching seed4 $(date) ==="
exec bash run_seed_sweep.sh 4 > log_seed4_driver.log 2>&1
