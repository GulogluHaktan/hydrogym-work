#!/bin/bash
# Prints ONE status line for the seed=2 sweep: current config, episode progress,
# configs done, estimated time remaining, projected finish time.
# Per-config reference durations (seconds) are from the seed=1 driver log.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p results logs

SEED=2
# ordered remaining configs after ppo_low, with seed=1 reference seconds
NAMES=(ppo_low ppo_high gppo_low gppo_high sac_low sac_high gsac_low gsac_high)
EST=(1640 1637 3362 3425 1748 1747 12821 4448)

done_count=0
remain=0
current=""
cur_est=0
for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"; e="${EST[$i]}"
  if [ -f "results/results_500_seed${SEED}_${n}.json" ]; then
    done_count=$((done_count+1))
  else
    if [ -z "$current" ]; then current="$n"; cur_est="$e"; fi
    remain=$((remain+e))
  fi
done

if [ -z "$current" ]; then
  echo "[seed$SEED] ALL 8 CONFIGS DONE — $(date '+%H:%M')"
  exit 0
fi

log="log_500_seed${SEED}_${current}.log"
prog="?"; elapsed=0
if [ -f "$log" ]; then
  line=$(grep -E "ep [0-9]+/[0-9]+" "$log" | tail -1)
  prog=$(echo "$line" | grep -oE "ep [0-9]+/[0-9]+" | head -1)
  elapsed=$(echo "$line" | grep -oE "t=[0-9.]+s" | grep -oE "[0-9.]+" | cut -d. -f1)
  [ -z "$elapsed" ] && elapsed=0
fi

# refine: if we have episode progress, scale this config's estimate by it
ep_now=$(echo "$prog" | grep -oE "[0-9]+/" | tr -d /)
if [ -n "$ep_now" ] && [ "$ep_now" -gt 5 ] 2>/dev/null && [ "$elapsed" -gt 0 ]; then
  cur_total=$(( elapsed * 500 / ep_now ))
  cur_left=$(( cur_total - elapsed ))
  [ "$cur_left" -lt 0 ] && cur_left=0
  remain=$(( remain - cur_est + cur_left ))
fi

h=$(( remain / 3600 )); m=$(( (remain % 3600) / 60 ))
eta=$(date -d "+${remain} seconds" '+%H:%M' 2>/dev/null || date '+%H:%M')
echo "[seed$SEED] $(date '+%H:%M') | çalışan: $current $prog | biten: $((done_count))/8 | kalan ~${h}s ${m}dk | tahmini bitiş ~${eta}"
