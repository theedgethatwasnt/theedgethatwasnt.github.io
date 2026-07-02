#!/bin/bash
# Simplified ESCMA v2 — STRICTLY SEQUENTIAL (one python at a time). Never parallel
# (parallel CMA froze the box before). Each seed loads the batch once and runs the
# whole activation bank (sweep ×6) + evolved. Spread charged up front (realistic).
set -u
cd "$(dirname "$0")"
RESULTS=v2_results.txt
echo "=== ESCMA v2 sequential run started $(date -u +%H:%M:%S) UTC ===" > "$RESULTS"

for seed in 42 7 123; do
  log="train_v2_s${seed}.log"
  avail=$(free -m | awk '/^Mem:/{print $7}')
  echo ">>> $(date -u +%H:%M:%S)  seed=$seed  (avail=${avail}MB)" | tee -a "$RESULTS"
  if [ "$avail" -lt 6500 ]; then
    echo "    SKIP — only ${avail}MB available (need ~5.7GB peak), refusing to risk a freeze" | tee -a "$RESULTS"
    continue
  fi
  python3 train_cma_exit_v2.py --seed "$seed" --gens 100 --mode both \
      --meta3-name meta3_USD_JPY_wav_z2_k2.parquet --t-max-alloc 1440 \
      > "$log" 2>&1
  rc=$?
  echo "--- seed $seed (rc=$rc) — leaderboard ---" >> "$RESULTS"
  sed -n '/LEADERBOARD/,/^====/p' "$log" >> "$RESULTS"
  grep "baseline] OOS first-negative" "$log" >> "$RESULTS"
  echo "" >> "$RESULTS"
done

echo "=== ESCMA v2 sequential run DONE $(date -u +%H:%M:%S) UTC ===" >> "$RESULTS"
