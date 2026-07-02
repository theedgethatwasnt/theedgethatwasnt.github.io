#!/bin/bash
# Phase 0b-amplitude — shrink the sine signal toward the spread floor (clean, noise=0).
# Finds the crossover where signal magnitude drops below the 1.7p round-trip spread.
set -u
cd "$(dirname "$0")"
OUT=phase0b_amp_curve.txt
echo "=== Phase 0b amplitude curve (clean) $(date -u +%H:%M:%S) UTC ===" > "$OUT"
printf "%-8s | %-12s | %-12s | %-10s | %s\n" \
  "amp_p" "best_OOS_A5" "best_raw_pnl" "best_hold" "best_cfg" >> "$OUT"
for amp in 50 20 10 5 3 2 1; do
  tag="SINE_a${amp}"
  python3 make_sine_dataset.py --amp-pips "$amp" --tag "$tag" > "p0ba_gen_${tag}.log" 2>&1
  python3 train_cma_exit_v2.py --pair "$tag" --meta3-name "meta3_${tag}.parquet" \
      --t-max-alloc 400 --seed 42 --gens 100 --mode sweep > "p0ba_${tag}.log" 2>&1
  best=$(grep -A8 "LEADERBOARD" "p0ba_${tag}.log" | grep -E "sweep:" | head -1)
  cfg=$(echo "$best" | awk '{print $1}')
  oos=$(echo "$best" | grep -oE "OOS=[+-]?[0-9.]+" | cut -d= -f2)
  raw=$(echo "$best" | grep -oE "pnl=[+-]?[0-9.]+" | cut -d= -f2)
  hold=$(echo "$best" | grep -oE "hold=[0-9]+b")
  printf "%-8s | %-12s | %-12s | %-10s | %s\n" "$amp" "$oos" "$raw" "$hold" "$cfg" | tee -a "$OUT"
  rm -f "features_${tag}.parquet" "meta3_${tag}.parquet"
done
echo "=== done $(date -u +%H:%M:%S) UTC ===" >> "$OUT"
