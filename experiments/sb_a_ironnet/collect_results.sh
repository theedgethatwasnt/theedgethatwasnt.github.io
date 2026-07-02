#!/usr/bin/env bash
# Collect SB_A IronNet results from Hetzner servers and optionally delete them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
SERVERS_FILE="$SCRIPT_DIR/hetzner_servers.txt"

if [ ! -f "$SERVERS_FILE" ]; then
    echo "ERROR: $SERVERS_FILE not found. Run deploy_hetzner.sh first."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Collecting results from Hetzner servers ==="
while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    IP=$(echo "$line" | awk '{print $2}')
    echo "  $NAME ($IP)..."
    rsync -az --progress root@"$IP":/root/sba_exp/results/ "$RESULTS_DIR/$NAME/" 2>/dev/null || \
        echo "  WARNING: rsync failed for $NAME"
done < "$SERVERS_FILE"

echo ""
echo "=== Summary ==="
echo "JSON results:"
find "$RESULTS_DIR" -name "*.json" | sort | while read f; do
    python3 -c "
import json, sys
d = json.load(open('$f' if True else sys.argv[1]))
print(f\"  {d.get('pair','?'):12s} seed={d.get('seed','?')} OOS_Sharpe={d.get('oos_sharpe', float('nan')):.3f} ppd={d.get('pips_per_day', float('nan')):.2f}\")
" 2>/dev/null || echo "  (parse error: $f)"
done

echo ""
echo "Log tail (best lines):"
find "$RESULTS_DIR" -name "*.log" | sort | xargs grep -h "BEST\|OOS\|gates" 2>/dev/null | tail -30

if [ "${1:-}" == "--delete" ]; then
    echo ""
    echo "=== Deleting servers ==="
    while IFS= read -r line; do
        NAME=$(echo "$line" | awk '{print $1}')
        echo "  Deleting $NAME..."
        hcloud server delete "$NAME" 2>/dev/null || echo "  (already gone)"
    done < "$SERVERS_FILE"
    rm -f "$SERVERS_FILE"
    echo "Servers deleted."
fi
