#!/bin/bash
# Position-aware exit-alpha test — STRICTLY SEQUENTIAL (one python at a time).
# The prior parallel launch froze the box (3×~3GB > 15GB RAM). NEVER parallelize.
set -u
cd "$(dirname "$0")"
RESULTS=posaware_results.txt
echo "=== position-aware sequential run started $(date -u +%H:%M:%S) UTC ===" > "$RESULTS"

# Job list: label : meta3 : mode
# A) random-entry exit-alpha (direction already random in meta → mode moot, use continuation)
# B) shock-fade z2_k2 WITH position sight (does sight rescue the +68/one-negative-seed result?)
declare -a JOBS=(
  "posaware_rand_s42:meta3_USD_JPY_random.parquet:continuation:42"
  "posaware_rand_s7:meta3_USD_JPY_random.parquet:continuation:7"
  "posaware_rand_s123:meta3_USD_JPY_random.parquet:continuation:123"
  "posaware_z2k2_s42:meta3_USD_JPY_wav_z2_k2.parquet:fade:42"
  "posaware_z2k2_s7:meta3_USD_JPY_wav_z2_k2.parquet:fade:7"
  "posaware_z2k2_s123:meta3_USD_JPY_wav_z2_k2.parquet:fade:123"
)

for job in "${JOBS[@]}"; do
  IFS=':' read -r label meta mode seed <<< "$job"
  log="train_${label}.log"
  # memory guard: skip if <2.5GB available (avoid a freeze)
  avail=$(free -m | awk '/^Mem:/{print $7}')
  echo ">>> $(date -u +%H:%M:%S) $label  seed=$seed  mode=$mode  (avail=${avail}MB)" | tee -a "$RESULTS"
  if [ "$avail" -lt 2500 ]; then
    echo "    SKIP — only ${avail}MB available, refusing to risk a freeze" | tee -a "$RESULTS"
    continue
  fi
  python3 train_cma_exit.py --gens 60 --pair USD_JPY \
      --meta3-name "$meta" --mode "$mode" --seed "$seed" --label "$label" > "$log" 2>&1
  rc=$?
  echo "--- $label (rc=$rc) ---" >> "$RESULTS"
  grep -E "AMDDP5   sum|Raw PnL  sum|WR =|First negative tick|n=[0-9]+  IS" "$log" | head -8 >> "$RESULTS"
  echo "" >> "$RESULTS"
done

echo "=== position-aware sequential run DONE $(date -u +%H:%M:%S) UTC ===" >> "$RESULTS"

# Summary: mean OOS AMDDP5 across seeds per experiment + the spread floor for random
python3 - <<'PY' >> "$RESULTS" 2>&1
import re, glob
import pandas as pd
print("\n=== SUMMARY ===")
# spread floor for the random set = -(mean entry spread) * n_oos
try:
    m = pd.read_parquet("meta3_USD_JPY_random.parquet")
    n_oos = int((m['split']=='OOS').sum())
    print(f"random-entry OOS trades: {n_oos}  (spread floor ≈ -1.7p × n = ~{-1.7*n_oos:.0f}p raw)")
except Exception as e:
    print("floor calc skipped:", e)
for exp in ("rand","z2k2"):
    vals=[]
    for seed in (42,7,123):
        try:
            txt=open(f"train_posaware_{exp}_s{seed}.log").read()
            ms=re.findall(r"AMDDP5\s+sum\s*=\s*([+-]?[\d.]+)", txt)
            rs=re.findall(r"Raw PnL\s+sum\s*=\s*([+-]?[\d.]+)", txt)
            if ms: vals.append((seed,float(ms[-1]), float(rs[-1]) if rs else None))
        except FileNotFoundError: pass
    if vals:
        mean=sum(v[1] for v in vals)/len(vals)
        det="  ".join(f"s{s}=A5:{a:+.0f}/raw:{r:+.0f}" for s,a,r in vals)
        print(f"  {exp}: mean_AMDDP5={mean:+.1f}   ({det})")
PY
