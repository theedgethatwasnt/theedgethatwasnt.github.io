#!/bin/bash
# Phase 0b — sine + calibrated noise → edge-vs-SNR curve.
# Sweep per-bar noise σ; for each, regenerate sine, train the bank, record best OOS.
# Signal: 50p amplitude over 180b (~0.28p/bar velocity). Cumulative noise over the
# 180-bar ride ≈ noise·√180 ≈ noise·13.4 — crosses 50p signal near noise≈3.7p/bar.
set -u
cd "$(dirname "$0")"
OUT=phase0b_curve.txt
echo "=== Phase 0b edge-vs-noise curve  $(date -u +%H:%M:%S) UTC ===" > "$OUT"
printf "%-8s | %-10s | %-12s | %-12s | %-10s | %s\n" \
  "noise_p" "naive_bar" "best_OOS_A5" "best_raw_pnl" "best_hold" "best_cfg" >> "$OUT"

for noise in 0 0.5 1 2 4 8 16; do
  tag="SINE_n${noise}"
  python3 make_sine_dataset.py --noise-pips "$noise" --tag "$tag" \
      > "p0b_gen_${tag}.log" 2>&1
  python3 train_cma_exit_v2.py --pair "$tag" --meta3-name "meta3_${tag}.parquet" \
      --t-max-alloc 400 --seed 42 --gens 100 --mode sweep \
      > "p0b_${tag}.log" 2>&1
  # parse: naive bar + best leaderboard row
  naive=$(grep "first-negative-tick" "p0b_${tag}.log" | grep -oE "amddp5=[+-]?[0-9.]+" | head -1 | cut -d= -f2)
  best=$(grep -A8 "LEADERBOARD" "p0b_${tag}.log" | grep -E "sweep:" | head -1)
  cfg=$(echo "$best" | awk '{print $1}')
  oos=$(echo "$best" | grep -oE "OOS=[+-]?[0-9.]+" | cut -d= -f2)
  raw=$(echo "$best" | grep -oE "pnl=[+-]?[0-9.]+" | cut -d= -f2)
  hold=$(echo "$best" | grep -oE "hold=[0-9]+b" )
  printf "%-8s | %-10s | %-12s | %-12s | %-10s | %s\n" \
    "$noise" "$naive" "$oos" "$raw" "$hold" "$cfg" | tee -a "$OUT"
  # cleanup big parquet to save disk
  rm -f "features_${tag}.parquet" "meta3_${tag}.parquet"
done
echo "=== done $(date -u +%H:%M:%S) UTC ===" >> "$OUT"
