#!/bin/bash
# Collect multi-seed CMA-NN results from Hetzner and pick best seed per pair
set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
CMA_DIR="$PROJECT_DIR/research/experiments/cma_5in"
RESULTS_DIR="$CMA_DIR/results"
SERVER_FILE="$CMA_DIR/multiseed_servers.txt"

if [ ! -f "$SERVER_FILE" ]; then
    echo "ERROR: No server file at $SERVER_FILE"
    exit 1
fi

echo "Collecting results from all servers..."
while read IDX IP; do
    echo "  Server $IDX ($IP)..."
    # Check completion
    ssh root@$IP 'grep -c "ALL DONE" /root/fx-core/research/experiments/cma_5in/results/*.log 2>/dev/null || echo "STILL RUNNING"' || true
    # Collect pkl + json files
    scp -q root@$IP:/root/fx-core/research/experiments/cma_5in/results/ms_*.pkl "$RESULTS_DIR/" 2>/dev/null || true
    scp -q root@$IP:/root/fx-core/research/experiments/cma_5in/results/ms_*.json "$RESULTS_DIR/" 2>/dev/null || true
done < "$SERVER_FILE"

echo ""
echo "Picking best seed per pair per TF..."

python3 - <<'PYEOF'
import json, glob
from pathlib import Path
from collections import defaultdict

RESULTS = Path("research/experiments/cma_5in/results")
best = defaultdict(lambda: {"oos_pps": -9999, "seed": None, "file": None})

# Parse all ms_*_result.json (not yet available, so parse pkl metadata)
for pkl in sorted(RESULTS.glob("ms_*_best.pkl")):
    import pickle
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    oos_pps = d.get("oos", {}).get("pips_per_day", 0)
    # Parse pair and seed from filename: ms_H1_v3_plus_macd_hist_CHF_JPY_s42_best.pkl
    parts = pkl.stem.replace("_best", "").split("_")
    # Find seed (sNN) and pair (X_Y)
    seed_idx = [i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()]
    if not seed_idx: continue
    si = seed_idx[-1]
    seed = int(parts[si][1:])
    pair = parts[si-2] + "_" + parts[si-1]
    tf = parts[1]  # H1 or M5
    key = f"{tf}_{pair}"

    if oos_pps > best[key]["oos_pps"]:
        best[key] = {"oos_pps": oos_pps, "seed": seed, "file": str(pkl)}

print(f"\n{'TF':>3} {'Pair':>8} {'Best Seed':>9} {'OOS p/d':>8}")
print("-" * 35)
for key in sorted(best.keys()):
    tf, pair = key.split("_", 1)
    b = best[key]
    if b["seed"] is not None:
        print(f"{tf:>3} {pair:>8} s{b['seed']:<8} {b['oos_pps']:>+8.1f}")
PYEOF

echo ""
echo "All results in $RESULTS_DIR/ms_*"
echo ""
echo "To delete servers:"
while read IDX IP; do
    echo "  hcloud server delete cma-ms-$IDX"
done < "$SERVER_FILE"
