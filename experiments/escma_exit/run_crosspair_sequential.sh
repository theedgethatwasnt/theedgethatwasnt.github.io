#!/bin/bash
# Cross-pair sighted-shock-fade ESCMA confirmation — STRICTLY SEQUENTIAL.
# ONE heavy python process at a time. The box froze once when 3 CMA trainings
# (each ~1GB parquet + ~2GB dense arrays) ran in parallel on 15GB RAM. NEVER
# parallelize. Each step memory-guards: skip/abort if <3000MB available.
set -u
cd "$(dirname "$0")"
RESULTS=crosspair_results.txt
MIN_MB=3000

ts() { date -u +%H:%M:%S; }
avail_mb() { free -m | awk '/^Mem:/{print $7}'; }

guard() {  # $1 = step label
  local a; a=$(avail_mb)
  echo ">>> $(ts) $1  (avail=${a}MB)" | tee -a "$RESULTS"
  if [ "$a" -lt "$MIN_MB" ]; then
    echo "    SKIP/WAIT — only ${a}MB available (<${MIN_MB}); refusing to risk a freeze." | tee -a "$RESULTS"
    return 1
  fi
  return 0
}

run_oos() {  # $1 = logfile, $2 = label  — extract OOS report block
  local log="$1" label="$2"
  echo "--- $label ---" >> "$RESULTS"
  awk '/OOS REPORT/{p=1} p' "$log" | \
    grep -E "AMDDP5   sum|Raw PnL  sum|^  WR =|Fixed \+20p|First negative|Hold to time|Exit causes" \
    >> "$RESULTS"
  echo "" >> "$RESULTS"
}

echo "=== cross-pair sequential run started $(ts) UTC ===" > "$RESULTS"

# ──────────────────────────────────────────────────────────────────────────
# PART A — USD_JPY 100-gen confirmation (current N_POS=3 network), 3 seeds
# ──────────────────────────────────────────────────────────────────────────
echo "" >> "$RESULTS"
echo "########## PART A — USD_JPY 100g (N_POS=3) ##########" >> "$RESULTS"
for seed in 42 7 123; do
  label="confirm_z2k2_s${seed}"
  log="train_${label}.log"
  guard "PART A train $label" || continue
  python3 train_cma_exit.py --gens 100 --pair USD_JPY \
      --meta3-name meta3_USD_JPY_wav_z2_k2.parquet --mode fade --t-max-alloc 1440 \
      --seed "$seed" --label "$label" > "$log" 2>&1
  echo "    rc=$? $(ts)" | tee -a "$RESULTS"
  run_oos "$log" "$label"
done

# ──────────────────────────────────────────────────────────────────────────
# PART B — cross-pair full pipeline, ONE pair at a time, ONE step at a time
# ──────────────────────────────────────────────────────────────────────────
for PAIR in EUR_JPY GBP_JPY AUD_JPY; do
  echo "" >> "$RESULTS"
  echo "########## PART B — $PAIR ##########" >> "$RESULTS"

  # (a) entry chopper → meta_<PAIR>.parquet + samples
  guard "PART B $PAIR chopper" || continue
  python3 entry_chopper.py --pair "$PAIR" > "chopper_${PAIR}.log" 2>&1
  echo "    chopper rc=$? $(ts)" | tee -a "$RESULTS"
  grep -E "events|sigma|wrote|meta_|sample" "chopper_${PAIR}.log" | tail -6 >> "$RESULTS"

  # (b1) precompute features → features_<PAIR>.parquet (writes the parquet)
  guard "PART B $PAIR precompute(write)" || continue
  python3 precompute_features.py --pair "$PAIR" > "precompute_${PAIR}.log" 2>&1
  echo "    precompute rc=$? $(ts)" | tee -a "$RESULTS"
  grep -E "features_|rows|wrote|\[write\]|MB" "precompute_${PAIR}.log" | tail -6 >> "$RESULTS"
  # (b2) R7 look-ahead verify (separate call — --verify exits without writing)
  guard "PART B $PAIR precompute(verify R7)" || continue
  python3 precompute_features.py --pair "$PAIR" --verify > "precompute_verify_${PAIR}.log" 2>&1
  echo "    R7 verify rc=$? $(ts)" | tee -a "$RESULTS"
  grep -E "R7|PASS|FAIL|verify|look-ahead|lookahead|match|identical" "precompute_verify_${PAIR}.log" | tail -8 >> "$RESULTS"

  # (c) rebuild meta3 (three-index form)
  guard "PART B $PAIR rebuild_meta" || continue
  python3 rebuild_meta.py --pair "$PAIR" > "rebuild_${PAIR}.log" 2>&1
  echo "    rebuild_meta rc=$? $(ts)" | tee -a "$RESULTS"
  grep -E "events|split counts|t_event range" "rebuild_${PAIR}.log" | tail -4 >> "$RESULTS"

  # (d) wavelet gate z2_k2 (causal; must PASS)
  guard "PART B $PAIR wavelet_gate" || continue
  python3 wavelet_gate.py --pair "$PAIR" --z-thrs 2 --k-bands 2 > "wavelet_${PAIR}.log" 2>&1
  echo "    wavelet_gate rc=$? $(ts)" | tee -a "$RESULTS"
  grep -E "Causality check:|kept|z=2 k=2|sigma-gate baseline" "wavelet_${PAIR}.log" | tail -6 >> "$RESULTS"

  # (e) train sighted fade × 3 seeds, 100 gens
  for seed in 42 7 123; do
    label="crosspair_${PAIR}_s${seed}"
    log="train_${label}.log"
    guard "PART B $PAIR train $label" || continue
    python3 train_cma_exit.py --gens 100 --pair "$PAIR" \
        --meta3-name "meta3_${PAIR}_wav_z2_k2.parquet" --mode fade --t-max-alloc 1440 \
        --seed "$seed" --label "$label" > "$log" 2>&1
    echo "    train rc=$? $(ts)" | tee -a "$RESULTS"
    run_oos "$log" "$label"
  done

  # free the ~1GB features parquet for this pair (gitignored, regenerable) so
  # disk + cache stay bounded before the next pair's build.
  rm -f "features_${PAIR}.parquet"
  echo "    cleaned features_${PAIR}.parquet $(ts)" | tee -a "$RESULTS"
done

echo "" >> "$RESULTS"
echo "=== cross-pair sequential run DONE $(ts) UTC ===" >> "$RESULTS"
