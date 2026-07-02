#!/bin/bash
# Collect CMA-NN 12-pair grid results from Hetzner
set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
CMA_DIR="$PROJECT_DIR/research/experiments/cma_5in"
RESULTS_DIR="$CMA_DIR/results"
SERVER_FILE="$CMA_DIR/grid12_server.txt"

if [ ! -f "$SERVER_FILE" ]; then
    echo "ERROR: No server file at $SERVER_FILE"
    exit 1
fi

IP=$(cat "$SERVER_FILE")
echo "Collecting from $IP..."

# Check if training is done
ssh root@$IP 'grep -q "ALL DONE" /root/grid_run.log 2>/dev/null && echo "DONE" || echo "STILL RUNNING"'

# Collect result files
echo "Downloading pickles + JSONs..."
scp -q root@$IP:/root/fx-core/research/experiments/cma_5in/results/grid12_*.pkl "$RESULTS_DIR/" 2>/dev/null || true
scp -q root@$IP:/root/fx-core/research/experiments/cma_5in/results/grid12_*.json "$RESULTS_DIR/" 2>/dev/null || true

# Collect the full log
scp -q root@$IP:/root/grid_run.log "$RESULTS_DIR/grid12_run.log" 2>/dev/null || true

echo ""
echo "Results collected to $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/grid12_* 2>/dev/null | head -30

echo ""
echo "To delete server: hcloud server delete cma-grid-1 --yes"
