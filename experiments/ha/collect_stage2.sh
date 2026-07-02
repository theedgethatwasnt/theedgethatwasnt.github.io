#!/bin/bash
# Collect Stage 2 results from Hetzner servers
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$SCRIPT_DIR/results/stage2"
mkdir -p "$RESULTS"

if [ ! -f "$SCRIPT_DIR/stage2_servers.txt" ]; then
    echo "No stage2_servers.txt found — did you run deploy_stage2.sh?"
    exit 1
fi

read -r -a SERVERS < "$SCRIPT_DIR/stage2_servers.txt"

VARIANTS=("S2-long" "S2-long" "S2-both" "S2-both")
SEEDS=(42 137 42 137)

echo "Collecting results from ${#SERVERS[@]} servers..."
for i in 0 1 2 3; do
    IP="${SERVERS[$i]}"
    VAR="${VARIANTS[$i]}"
    SEED="${SEEDS[$i]}"
    echo "  Server $((i+1)) ($IP): $VAR seed $SEED"
    scp -o StrictHostKeyChecking=no "root@$IP:/root/neat/results/stage2/${VAR}_s${SEED}_best.pkl" "$RESULTS/" 2>/dev/null || echo "    (genome not found)"
    scp -o StrictHostKeyChecking=no "root@$IP:/root/neat/results/stage2/${VAR}_s${SEED}_result.json" "$RESULTS/" 2>/dev/null || echo "    (result not found)"
    scp -o StrictHostKeyChecking=no "root@$IP:/root/neat/results/stage2/${VAR}_s${SEED}.log" "$RESULTS/" 2>/dev/null || echo "    (log not found)"
done

echo ""
echo "Results collected in $RESULTS/"
ls -la "$RESULTS/"

echo ""
echo "To cleanup servers:"
echo "  for i in 1 2 3 4; do hcloud server delete ha-stage2-\$i --yes; done"
