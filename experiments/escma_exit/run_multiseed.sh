#!/bin/bash
# 100-gen, multi-seed fade training on the two best wavelet filters.
# Sequential (one at a time) to avoid the contention that killed the parallel agents.
# Two filters: z2_k2 (best total / biggest margin over naive) and z3_k3 (best per-event).
set -u
cd "$(dirname "$0")"
RESULTS=multiseed_results.txt
echo "=== multiseed 100-gen sweep started $(date -u +%H:%M:%S) UTC ===" > "$RESULTS"

SEEDS=(42 7 123)
declare -a FILTERS=(
  "z2k2:meta3_USD_JPY_wav_z2_k2.parquet"
  "z3k3:meta3_USD_JPY_wav_z3_k3.parquet"
)

for f in "${FILTERS[@]}"; do
  fname="${f%%:*}"
  meta="${f##*:}"
  for seed in "${SEEDS[@]}"; do
    label="fade_${fname}_s${seed}_g100"
    log="train_${label}.log"
    echo ">>> $(date -u +%H:%M:%S) $fname seed=$seed (100 gens) on $meta" | tee -a "$RESULTS"
    python3 train_cma_exit.py --mode fade --gens 100 --pair USD_JPY \
        --meta3-name "$meta" --seed "$seed" --label "$label" > "$log" 2>&1
    rc=$?
    echo "--- $fname seed=$seed (rc=$rc) ---" >> "$RESULTS"
    grep -E "AMDDP1   sum|AMDDP5   sum|AMDDP10  sum|Raw PnL  sum|WR =|First negative tick|n=[0-9]+  IS" "$log" | head -10 >> "$RESULTS"
    echo "" >> "$RESULTS"
  done
done

echo "=== multiseed sweep DONE $(date -u +%H:%M:%S) UTC ===" >> "$RESULTS"

# Compact per-filter summary (mean OOS AMDDP5 across seeds)
python3 - <<'PY' >> "$RESULTS" 2>&1
import re, glob
print("\n=== SUMMARY: OOS AMDDP5 sum by filter × seed ===")
for fname in ("z2k2","z3k3"):
    vals=[]
    for seed in (42,7,123):
        try:
            txt=open(f"train_fade_{fname}_s{seed}_g100.log").read()
            # last OOS AMDDP5 sum in the file = the OOS report (IS report also prints one earlier)
            ms=re.findall(r"AMDDP5\s+sum\s*=\s*([+-]?[\d.]+)", txt)
            if ms: vals.append((seed,float(ms[-1])))
        except FileNotFoundError: pass
    if vals:
        mean=sum(v for _,v in vals)/len(vals)
        detail="  ".join(f"s{s}={v:+.1f}" for s,v in vals)
        print(f"  {fname}: mean={mean:+.1f}   ({detail})")
PY
