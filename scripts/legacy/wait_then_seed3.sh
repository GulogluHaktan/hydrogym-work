#!/bin/bash
# Waits for the seed=2 sweep to finish, then launches the seed=3 sweep.
# Launch detached:  nohup bash wait_then_seed3.sh > logs/log_seed3_chain.log 2>&1 &
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs
until grep -q "ALL 8 CONFIGS DONE" logs/log_seed2_driver.log 2>/dev/null; do
  sleep 60
done
echo "=== [chain] seed2 confirmed done, launching seed3 $(date) ==="
exec bash scripts/run_seed_sweep.sh 3 > logs/log_seed3_driver.log 2>&1
