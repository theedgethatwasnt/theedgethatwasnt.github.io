#!/bin/bash
# Collect tier-2 results from all servers, aggregate leaderboard, delete servers.
# Usage: collect_slot4_full.sh [--keep]   # --keep leaves servers running

set -uo pipefail

cd "$(dirname "$0")/../../../.."
LOOP=research/experiments/cma_5in/indicator_loop
mkdir -p $LOOP/results_tier2

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

if [ ! -f $LOOP/tier2_servers.txt ]; then
    echo "No tier2_servers.txt — nothing to collect."
    exit 1
fi

mapfile -t SERVERS < $LOOP/tier2_servers.txt
echo "=== Collecting from ${#SERVERS[@]} servers ==="

for s in "${SERVERS[@]}"; do
    NAME=$(echo "$s" | cut -d: -f1)
    IP=$(echo "$s" | cut -d: -f2)
    echo "  $NAME ($IP)..."
    # Pull results (JSON + PKL + log). Not interrupting if server is still running.
    rsync -az --include="slot4_*.json" --include="slot4_*.pkl" --include="slot4_*.log" --exclude="*" \
        -e "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no" \
        "root@$IP:/root/fx-core/research/experiments/cma_5in/indicator_loop/results/" \
        "$LOOP/results_tier2/" 2>/dev/null || echo "    (timeout or no results)"
done

echo
echo "=== Tier-2 results in $LOOP/results_tier2 ==="
echo "Total JSON files: $(ls $LOOP/results_tier2/*.json 2>/dev/null | wc -l)"

# Aggregate leaderboard
python3 <<'EOF_AGG'
import json, glob
from collections import defaultdict
rows = defaultdict(list)
for f in glob.glob('research/experiments/cma_5in/indicator_loop/results_tier2/slot4_*.json'):
    d = json.load(open(f))
    rows[d['candidate']].append(d['oos']['pd'])
if not rows:
    print("No results yet.")
    raise SystemExit
table = sorted([(c, sum(v)/len(v), min(v), max(v), len(v))
                for c, v in rows.items()], key=lambda x: -x[1])
print(f"\n{'rank':>4}  {'candidate':20s} {'avg':>8} {'lo':>8} {'hi':>8} {'n':>3}")
for i, (c, avg, lo, hi, n) in enumerate(table, 1):
    star = " 🏆" if avg > -2.0 else ""
    complete = " (COMPLETE)" if n >= 48 else f" ({n}/48)"
    print(f"{i:>4}  {c:20s} {avg:+8.2f} {lo:+8.2f} {hi:+8.2f} {n:>3}{star}{complete}")

# Telegram
import sys
sys.path.insert(0, '.')
from lib.notify import _send
positive = [c for c, avg, *_ in table if avg > 0]
top5 = table[:5]
msg = f"📊 Tier-2 interim ({sum(n for _,_,_,_,n in table)} runs across {len(table)} cands)\n\n"
msg += "TOP 5:\n"
for c, avg, lo, hi, n in top5:
    msg += f"  {c}  {avg:+.2f}  [{lo:+.2f}…{hi:+.2f}] n={n}\n"
if positive:
    msg += f"\n🟢 POSITIVE OOS: {', '.join(positive)}"
_send(msg)
EOF_AGG

if [ "$KEEP" = "0" ]; then
    # Check if all jobs done before deleting
    ALL_DONE=1
    for s in "${SERVERS[@]}"; do
        IP=$(echo "$s" | cut -d: -f2)
        status=$(ssh -o ConnectTimeout=10 "root@$IP" "ps -ef | grep test_slot4_swap_cma | grep -v grep | wc -l" 2>/dev/null || echo "?")
        if [ "$status" != "0" ]; then
            ALL_DONE=0
            echo "  $IP: still running ($status procs)"
        else
            echo "  $IP: finished"
        fi
    done
    if [ "$ALL_DONE" = "1" ]; then
        echo "=== All jobs complete. Deleting servers ==="
        for s in "${SERVERS[@]}"; do
            NAME=$(echo "$s" | cut -d: -f1)
            hcloud server delete "$NAME" 2>/dev/null || true
            echo "  deleted $NAME"
        done
        rm -f $LOOP/tier2_servers.txt
    else
        echo "Some servers still running. Re-run this script later (or use --keep to force collect)."
    fi
fi
