#!/bin/bash
# Collect H1 per-pair training results from Hetzner servers
set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
RESULTS_DIR="$PROJECT_DIR/research/experiments/asi_mc/results/ironnet_h1"
SERVER_FILE="$PROJECT_DIR/research/experiments/asi_mc/h1_servers.txt"

if [ ! -f "$SERVER_FILE" ]; then
    echo "ERROR: $SERVER_FILE not found. Run deploy_h1_perpair.sh first."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Collecting results..."
while read IDX IP; do
    echo "  Server $IDX ($IP):"
    scp -q root@$IP:/root/fx-core/research/experiments/asi_mc/results/ironnet_h1/*best*.pkl "$RESULTS_DIR/" 2>/dev/null || echo "    No pkl files"
    scp -q root@$IP:/root/fx-core/research/experiments/asi_mc/results/ironnet_h1/*result*.json "$RESULTS_DIR/" 2>/dev/null || echo "    No result files"
    scp -q root@$IP:/root/fx-core/research/experiments/asi_mc/results/ironnet_h1/*.log "$RESULTS_DIR/" 2>/dev/null || echo "    No logs"
    # Also grab checkpoints
    scp -q root@$IP:/root/fx-core/research/experiments/asi_mc/results/ironnet_h1/*_ckpt "$RESULTS_DIR/" 2>/dev/null || true
    echo "    Done"
done < "$SERVER_FILE"

echo ""
echo "Results collected to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.pkl 2>/dev/null | wc -l | xargs -I{} echo "Genomes: {}"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null | wc -l | xargs -I{} echo "Result JSONs: {}"

echo ""
echo "Delete servers? Run:"
echo "  for i in 1 2 3; do echo y | hcloud server delete ironnet-h1-\$i; done"
