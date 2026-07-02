#!/bin/bash
# Collect subset experiment results from all 4 servers

SERVERS=(<SERVER_IP_10> <SERVER_IP_11> <SERVER_IP_12> <SERVER_IP_13>)
MODES=(S1 S2 S3 S4)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/results/subsets"
mkdir -p "$OUT"

echo "Collecting results..."
for idx in 0 1 2 3; do
    ip="${SERVERS[$idx]}"
    mode="${MODES[$idx]}"
    echo "  $mode from $ip..."
    # Logs
    scp -o StrictHostKeyChecking=no -r \
        "root@$ip:/root/neat/results/subsets/" \
        "$OUT/${mode}_remote/" 2>/dev/null || true
    # Best pkl files
    ssh -o StrictHostKeyChecking=no root@$ip \
        "find /root/neat/results/swingdim -name '*${mode}*best*.pkl' 2>/dev/null" | \
    while read f; do
        scp -o StrictHostKeyChecking=no "root@$ip:$f" "$OUT/" 2>/dev/null || true
    done
done

echo ""
echo "Summary of OOS results:"
for f in "$OUT"/**/*results.json "$OUT"/**/*summary.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json, sys
d = json.load(open('$f'))
if isinstance(d, list): d = d[0]
pair = d.get('pair','?')
mode = d.get('mode','?')
seed = d.get('seed','?')
ppd  = d.get('oos',{}).get('pips_per_day','?')
nt   = d.get('oos',{}).get('n_trades','?')
fit  = d.get('fitness','?')
print(f'  {mode:4} {pair:8} s{seed}: {ppd:>7} p/day  {nt:>5}T  fit={fit:.2f}')
" 2>/dev/null
done
