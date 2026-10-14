#!/bin/bash
# Waits for the seed=2 sweep to finish, then launches the seed=3 sweep.
# Launch detached:  nohup bash wait_then_seed3.sh > log_seed3_chain.log 2>&1 &
cd "$(dirname "${BASH_SOURCE[0]}")"
until grep -q "ALL 8 CONFIGS DONE" log_seed2_driver.log 2>/dev/null; do
  sleep 60
done
echo "=== [chain] seed2 confirmed done, launching seed3 $(date) ==="
exec bash run_seed_sweep.sh 3 > log_seed3_driver.log 2>&1
