#!/bin/bash
# Master orchestrator: wait for any running CMA, rebuild parquets, screen all
# ported-but-not-validated candidates, send telegram summary.
set -uo pipefail  # allow errors, we want to keep going

cd /path/to/projects/fx-core
MSG() {
    python3 -c "import sys; sys.path.insert(0,'.'); from lib.notify import _send; _send('''$1''')"
}

# 1. Wait for any active CMA to finish (up to 30 min)
echo "[master] waiting for in-flight CMA runs..."
for i in {1..60}; do
    n=$(ps -ef | grep test_slot4_swap_cma | grep -v grep | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "[master] no CMA running"
        break
    fi
    echo "[master] $n CMA procs still running, check #$i/60"
    sleep 30
done

# 2. Rebuild parquets with full feature set
echo "[master] rebuilding causal parquets with ALL ported features..."
MSG "🔄 Rebuilding causal parquets with 32+ ported indicators..."
rm -f data/m5_ohlc/EUR_JPY_M5_kalman10_causal.parquet data/m5_ohlc/USD_JPY_M5_kalman10_causal.parquet
python3 research/experiments/cma_5in/build_causal_parquets.py --pairs EUR_JPY USD_JPY --smoother kalman10 --workers 2 2>&1 | tail -5

# 3. Screen every ported-but-not-validated candidate
echo "[master] screening all ported-but-not-validated..."
MSG "🚀 Starting tier-1 screen on all non-validated ported candidates"

PORTED=$(python3 -c "
import json
d = json.load(open('research/experiments/cma_5in/indicator_loop/candidates.json'))
out = [c['name'] for c in d['candidates']
       if c.get('port_status') == 'ported' and not c.get('validated')]
print(' '.join(out))
")
echo "[master] queue: $PORTED"

for cand in $PORTED; do
    echo "[master] === screening $cand ==="
    python3 research/experiments/cma_5in/indicator_loop/loop.py \
        --candidate "$cand" --pairs EUR_JPY USD_JPY --seeds 42 137 \
        --gens 100 --parallel 4 --inner 2 2>&1 | tail -30
    echo "[master] $cand done"
done

# 4. Final summary
echo "[master] all done, producing summary..."
python3 <<'EOF'
import json, glob
rows = []
for f in glob.glob('research/experiments/cma_5in/indicator_loop/results/slot4_*.json'):
    d = json.load(open(f))
    rows.append(d)
# Group by candidate
by_cand = {}
for r in rows:
    by_cand.setdefault(r['candidate'], []).append(r['oos']['pd'])
table = sorted([(c, sum(v)/len(v), min(v), max(v), len(v))
                for c, v in by_cand.items()], key=lambda x: -x[1])
lines = ["📊 Tier-1 sweep complete — leaderboard (avg OOS p/d):"]
for c, avg, lo, hi, n in table:
    star = " 🏆" if avg > -2.0 else ""
    lines.append(f"{c:20s} {avg:+.2f}  [{lo:+.2f} … {hi:+.2f}] (n={n}){star}")
msg = "\n".join(lines)
print(msg)
import sys
sys.path.insert(0,'.')
from lib.notify import _send
_send(msg)
EOF

echo "[master] complete"
