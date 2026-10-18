#!/bin/bash
# Waits for the seed=1 driver to finish, then launches the seed=2 sweep.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs
until grep -q "ALL 8 CONFIGS DONE" logs/log_seed1_driver.log 2>/dev/null; do
  sleep 30
done
echo "=== [chain] seed1 confirmed done, launching seed2 $(date) ==="
./run_seed2_all8_sequential.sh
