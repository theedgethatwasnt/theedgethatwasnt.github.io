#!/bin/bash
# Re-run PINN on causal (post-RCA) ss_h1 features.
# Only the 4 ss_h1-consuming modes need rerunning; baseline+hyper were clean.
#
# 4 modes × 3 pairs × 3 seeds = 36 runs on 1 × cx53 (~30-45 min).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SERVER=pinn-rerun
echo "=== PINN rerun (causal ss_h1) — 1 × cx53 ==="

hcloud server create \
    --name "$SERVER" \
    --type cx53 \
    --image ubuntu-24.04 \
    --location hel1 \
    --ssh-key "user@host" \
    2>&1 | grep -E "^(Server|IPv4)" || true

echo "Waiting 30s for boot..."
sleep 30

IP=$(hcloud server ip $SERVER)
echo "Server: $IP"
ssh-keygen -f ~/.ssh/known_hosts -R "$IP" 2>/dev/null || true

ssh -o StrictHostKeyChecking=accept-new root@$IP "
    apt-get update -qq && \
    apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
    python3 -m venv /root/venv && \
    /root/venv/bin/pip install -q numpy pandas numba cma pyarrow && \
    mkdir -p /root/pinn/data/unified_indicators /root/pinn/data/pinn_features /root/pinn/research/experiments/cma_5in
"

scp -q "$SCRIPT_DIR/train_pinn_cma.py" root@$IP:/root/pinn/research/experiments/cma_5in/

echo "Uploading CAUSAL ss_h1 (post-4-17 fix)..."
for PAIR in EUR_JPY CAD_JPY EUR_GBP; do
    scp -q "$PROJECT_ROOT/data/unified_indicators/${PAIR}_unified.parquet" \
        root@$IP:/root/pinn/data/unified_indicators/
    scp -q "$PROJECT_ROOT/data/pinn_features/${PAIR}_ss_h1.npy" \
        root@$IP:/root/pinn/data/pinn_features/
done

# Runner — only the 4 ss_h1-consuming modes
ssh root@$IP "cat > /root/run.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd /root/pinn
export CMA5IN_DATA_DIR=/root/pinn/data/unified_indicators
source /root/venv/bin/activate
export PYTHONUNBUFFERED=1

PAIRS=(EUR_JPY CAD_JPY EUR_GBP)
SEEDS=(42 123 777)
MODES=(inputs inputs_hyper inputs_fitness full)
RESULTS=/root/pinn/research/experiments/cma_5in/results_pinn_causal
mkdir -p \$RESULTS

echo \"PINN rerun started: \$(date)\"
COUNT=0
for MODE in \${MODES[@]}; do
    for PAIR in \${PAIRS[@]}; do
        for SEED in \${SEEDS[@]}; do
            COUNT=\$((COUNT + 1))
            TAG=\"pinn_\${MODE}_\${PAIR}_s\${SEED}\"
            if [ -f \"\$RESULTS/\${TAG}_summary.json\" ]; then
                echo \"[\$COUNT/36] SKIP \$TAG\"
                continue
            fi
            echo \"[\$COUNT/36] \$TAG ...\"
            # Redirect results to the new dir by setting RESULTS_DIR via env?
            # Simpler: run in default dir, then move.
            python3 research/experiments/cma_5in/train_pinn_cma.py \\
                --pair \$PAIR --mode \$MODE --seed \$SEED \\
                --gens 200 --popsize 24 --workers 14 2>&1 | tail -8
            DEFAULT_RESULT=/root/pinn/research/experiments/cma_5in/results_pinn
            if [ -f \"\$DEFAULT_RESULT/\${TAG}_best.pkl\" ]; then
                mv \"\$DEFAULT_RESULT/\${TAG}_best.pkl\" \$RESULTS/
                mv \"\$DEFAULT_RESULT/\${TAG}_summary.json\" \$RESULTS/
            fi
        done
    done
done
echo \"PINN rerun done: \$(date)\"
SCRIPT
chmod +x /root/run.sh"

ssh root@$IP "nohup /root/run.sh > /root/pinn_rerun.log 2>&1 &"
echo "Launched on $IP"
echo ""
echo "=== Monitor ==="
echo "  ssh root@$IP 'tail -f /root/pinn_rerun.log'"
echo ""
echo "=== Collect + cleanup (when done) ==="
echo "  scp root@$IP:/root/pinn/research/experiments/cma_5in/results_pinn_causal/*.pkl $SCRIPT_DIR/results_pinn_causal/"
echo "  scp root@$IP:/root/pinn/research/experiments/cma_5in/results_pinn_causal/*.json $SCRIPT_DIR/results_pinn_causal/"
echo "  hcloud server delete $SERVER"

echo "$IP" > "$SCRIPT_DIR/pinn_rerun_ip.txt"
