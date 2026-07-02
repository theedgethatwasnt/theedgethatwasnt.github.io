#!/bin/bash
# Collect slot-4 results from Hetzner servers + delete them.
# Usage: collect_slot4.sh <candidate> <server1_name:ip> <server2_name:ip> ...
set -euo pipefail

CAND="${1:?candidate required}"
shift
SERVERS=("$@")
LOCAL_RESULTS="research/experiments/cma_5in/indicator_loop/results"
mkdir -p "$LOCAL_RESULTS"

echo "=== Pulling $CAND results from ${#SERVERS[@]} servers ==="
for s in "${SERVERS[@]}"; do
    NAME=$(echo "$s" | cut -d: -f1)
    IP=$(echo "$s" | cut -d: -f2)
    echo "  $NAME ($IP)..."
    rsync -az --include="slot4_${CAND}_*.json" --include="slot4_${CAND}_*.pkl" --include="slot4_${CAND}_*.log" --exclude="*" \
        "root@$IP:/root/fx-core/research/experiments/cma_5in/indicator_loop/results/" \
        "$LOCAL_RESULTS/" || echo "    (no results yet on $NAME)"
done

echo "=== Deleting servers ==="
for s in "${SERVERS[@]}"; do
    NAME=$(echo "$s" | cut -d: -f1)
    hcloud server delete "$NAME" --yes || true
done

echo
echo "Local results in $LOCAL_RESULTS:"
ls -1 "$LOCAL_RESULTS" | grep "slot4_${CAND}" | wc -l
echo
echo "Aggregate via: python3 research/experiments/cma_5in/indicator_loop/loop.py --candidate $CAND --aggregate-only"
