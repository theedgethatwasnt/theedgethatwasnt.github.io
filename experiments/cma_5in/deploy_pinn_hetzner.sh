#!/bin/bash
# Deploy PINN-CMA experiment to 2 Hetzner cx53 servers
# Server 1: baseline, inputs, hyper (27 runs)
# Server 2: inputs_hyper, inputs_fitness, full (27 runs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== PINN-CMA Hetzner Deployment ==="
echo "Creating 2 × cx53 servers in hel1..."

# Create servers
for i in 1 2; do
    hcloud server create \
        --name "pinn-exp-$i" \
        --type cx53 \
        --image ubuntu-24.04 \
        --location hel1 \
        --ssh-key "user@host" \
        2>&1 | grep -E "^(Server|IPv4)" || true
done

echo ""
echo "Waiting 30s for servers to boot..."
sleep 30

# Get IPs
IP1=$(hcloud server ip pinn-exp-1)
IP2=$(hcloud server ip pinn-exp-2)
echo "Server 1: $IP1"
echo "Server 2: $IP2"

# Setup function
setup_server() {
    local IP=$1
    local NAME=$2
    echo ""
    echo "=== Setting up $NAME ($IP) ==="

    # Install deps
    ssh -o StrictHostKeyChecking=no root@$IP "
        apt-get update -qq && \
        apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
        python3 -m venv /root/venv && \
        /root/venv/bin/pip install -q numpy pandas numba cma pyarrow && \
        mkdir -p /root/pinn/data/unified_indicators /root/pinn/data/pinn_features /root/pinn/research/experiments/cma_5in
    "

    # Upload scripts
    scp -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/train_pinn_cma.py" \
        "$SCRIPT_DIR/analyze_pinn_results.py" \
        root@$IP:/root/pinn/research/experiments/cma_5in/

    # Upload data (only 3 pairs needed)
    for PAIR in EUR_JPY CAD_JPY EUR_GBP; do
        echo "  Uploading $PAIR data..."
        scp -q "$PROJECT_ROOT/data/unified_indicators/${PAIR}_unified.parquet" \
            root@$IP:/root/pinn/data/unified_indicators/
        scp -q "$PROJECT_ROOT/data/pinn_features/${PAIR}_ss_h1.npy" \
            root@$IP:/root/pinn/data/pinn_features/
    done

    echo "  $NAME setup complete."
}

setup_server $IP1 "pinn-exp-1"
setup_server $IP2 "pinn-exp-2"

# Create per-server runner scripts
echo ""
echo "=== Creating runner scripts ==="

# Server 1: baseline, inputs, hyper
ssh root@$IP1 "cat > /root/run.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd /root/pinn
export CMA5IN_DATA_DIR=/root/pinn/data/unified_indicators
source /root/venv/bin/activate

PAIRS=(EUR_JPY CAD_JPY EUR_GBP)
SEEDS=(42 123 777)
MODES=(baseline inputs hyper)
RESULTS=/root/pinn/research/experiments/cma_5in/results_pinn
mkdir -p \$RESULTS

echo \"Server 1 started: \$(date)\"
COUNT=0
for MODE in \${MODES[@]}; do
    for PAIR in \${PAIRS[@]}; do
        for SEED in \${SEEDS[@]}; do
            COUNT=\$((COUNT + 1))
            TAG=\"pinn_\${MODE}_\${PAIR}_s\${SEED}\"
            if [ -f \"\$RESULTS/\${TAG}_summary.json\" ]; then
                echo \"[\$COUNT/27] SKIP \$TAG\"
                continue
            fi
            echo \"[\$COUNT/27] \$TAG ...\"
            python3 research/experiments/cma_5in/train_pinn_cma.py \
                --pair \$PAIR --mode \$MODE --seed \$SEED \
                --gens 200 --popsize 24 --workers 14 2>&1 | tail -8
            echo \"\"
        done
    done
done
echo \"Server 1 done: \$(date)\"
SCRIPT
chmod +x /root/run.sh"

# Server 2: inputs_hyper, inputs_fitness, full
ssh root@$IP2 "cat > /root/run.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd /root/pinn
export CMA5IN_DATA_DIR=/root/pinn/data/unified_indicators
source /root/venv/bin/activate

PAIRS=(EUR_JPY CAD_JPY EUR_GBP)
SEEDS=(42 123 777)
MODES=(inputs_hyper inputs_fitness full)
RESULTS=/root/pinn/research/experiments/cma_5in/results_pinn
mkdir -p \$RESULTS

echo \"Server 2 started: \$(date)\"
COUNT=0
for MODE in \${MODES[@]}; do
    for PAIR in \${PAIRS[@]}; do
        for SEED in \${SEEDS[@]}; do
            COUNT=\$((COUNT + 1))
            TAG=\"pinn_\${MODE}_\${PAIR}_s\${SEED}\"
            if [ -f \"\$RESULTS/\${TAG}_summary.json\" ]; then
                echo \"[\$COUNT/27] SKIP \$TAG\"
                continue
            fi
            echo \"[\$COUNT/27] \$TAG ...\"
            python3 research/experiments/cma_5in/train_pinn_cma.py \
                --pair \$PAIR --mode \$MODE --seed \$SEED \
                --gens 200 --popsize 24 --workers 14 2>&1 | tail -8
            echo \"\"
        done
    done
done
echo \"Server 2 done: \$(date)\"
SCRIPT
chmod +x /root/run.sh"

# Launch both
echo ""
echo "=== Launching experiments ==="
ssh root@$IP1 "nohup /root/run.sh > /root/pinn_exp.log 2>&1 &"
echo "Server 1 ($IP1): launched (27 runs: baseline, inputs, hyper)"

ssh root@$IP2 "nohup /root/run.sh > /root/pinn_exp.log 2>&1 &"
echo "Server 2 ($IP2): launched (27 runs: inputs_hyper, inputs_fitness, full)"

echo ""
echo "=== Monitor ==="
echo "  ssh root@$IP1 'tail -f /root/pinn_exp.log'"
echo "  ssh root@$IP2 'tail -f /root/pinn_exp.log'"
echo ""
echo "=== Collect results when done ==="
echo "  scp root@$IP1:/root/pinn/research/experiments/cma_5in/results_pinn/*.json results_pinn/"
echo "  scp root@$IP1:/root/pinn/research/experiments/cma_5in/results_pinn/*.pkl results_pinn/"
echo "  scp root@$IP2:/root/pinn/research/experiments/cma_5in/results_pinn/*.json results_pinn/"
echo "  scp root@$IP2:/root/pinn/research/experiments/cma_5in/results_pinn/*.pkl results_pinn/"
echo ""
echo "=== Cleanup when done ==="
echo "  hcloud server delete pinn-exp-1 --yes && hcloud server delete pinn-exp-2 --yes"
