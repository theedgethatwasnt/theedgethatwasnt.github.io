#!/bin/bash
# Sequential fade-mode training across gate/wavelet variants.
# Runs ONE at a time (avoids the resource contention that killed the parallel agents).
# Each appends its OOS AMDDP5 line to variant_sweep_results.txt.
set -u
cd "$(dirname "$0")"
RESULTS=variant_sweep_results.txt
echo "=== variant sweep started $(date -u +%H:%M:%S) UTC ===" > "$RESULTS"

# Priority order: strongest wavelet filter first, then weaker, then gate-tighten, then baseline.
# (label : meta3 file)
declare -a JOBS=(
  "fade_wav_z3k3:meta3_USD_JPY_wav_z3_k3.parquet"
  "fade_wav_z3k2:meta3_USD_JPY_wav_z3_k2.parquet"
  "fade_wav_z2k3:meta3_USD_JPY_wav_z2_k3.parquet"
  "fade_wav_z2k2:meta3_USD_JPY_wav_z2_k2.parquet"
  "fade_t3:meta3_USD_JPY_t3.parquet"
  "fade_base40:meta3_USD_JPY.parquet"
)

for job in "${JOBS[@]}"; do
  label="${job%%:*}"
  meta="${job##*:}"
  log="train_${label}.log"
  echo ">>> $(date -u +%H:%M:%S) training $label on $meta" | tee -a "$RESULTS"
  python3 train_cma_exit.py --mode fade --gens 40 --pair USD_JPY \
      --meta3-name "$meta" --label "$label" > "$log" 2>&1
  rc=$?
  # Pull the OOS block (AMDDP5 sum, raw pnl, WR, first-neg baseline) into the results file
  echo "--- $label (rc=$rc) ---" >> "$RESULTS"
  grep -E "AMDDP5   sum|Raw PnL  sum|WR =|First negative tick|n=[0-9]+  IS" "$log" | head -8 >> "$RESULTS"
  echo "" >> "$RESULTS"
done

echo "=== variant sweep DONE $(date -u +%H:%M:%S) UTC ===" >> "$RESULTS"
