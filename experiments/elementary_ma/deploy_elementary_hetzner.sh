#!/bin/bash
# Deploy elementary-MA campaign to 3 Hetzner cx53 servers.
# Work split: 5 arms × 12 pairs × 3 seeds = 180 runs
#   Server 1: E-atan + E-pz-norm  (72 runs)
#   Server 2: E-pz-rank + E-mom   (72 runs)
#   Server 3: E-all               (36 runs — bigger network, more time/run)
# All three write to their own live_results.jsonl — monitor pulls from all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Elementary-MA Hetzner Deployment ==="
echo "Creating 3 × cx53 servers in hel1..."

for i in 1 2 3; do
    hcloud server create \
        --name "elem-exp-$i" \
        --type cx53 \
        --image ubuntu-24.04 \
        --location hel1 \
        --ssh-key "user@host" \
        2>&1 | grep -E "^(Server|IPv4)" || true
done

echo "Waiting 30s for servers to boot..."
sleep 30

IP1=$(hcloud server ip elem-exp-1)
IP2=$(hcloud server ip elem-exp-2)
IP3=$(hcloud server ip elem-exp-3)
echo "Server 1: $IP1"
echo "Server 2: $IP2"
echo "Server 3: $IP3"

# Clean stale host keys
for IP in $IP1 $IP2 $IP3; do
    ssh-keygen -f ~/.ssh/known_hosts -R "$IP" 2>/dev/null || true
done

PAIRS="EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP"
SEEDS="42 123 777"

# ── Setup function ────────────────────────────────────────────────────────
setup_server() {
    local IP=$1
    local NAME=$2
    echo ""
    echo "=== Setting up $NAME ($IP) ==="

    ssh -o StrictHostKeyChecking=accept-new root@$IP "
        apt-get update -qq && \
        apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
        python3 -m venv /root/venv && \
        /root/venv/bin/pip install -q numpy pandas numba cma pyarrow && \
        mkdir -p /root/elementary_ma/data/unified_indicators \
                 /root/elementary_ma/data/elementary_ma_features \
                 /root/elementary_ma/research/experiments/elementary_ma/results \
                 /root/elementary_ma/research/experiments/elementary_ma/results/progress
    "

    # Upload code
    scp -q "$SCRIPT_DIR/features.py" \
           "$SCRIPT_DIR/train_elementary_cma.py" \
           "$SCRIPT_DIR/monitor_progress.py" \
        root@$IP:/root/elementary_ma/research/experiments/elementary_ma/

    # Upload unified parquets + precomputed features (all 12 pairs)
    echo "  Uploading unified parquets + features..."
    for PAIR in $PAIRS; do
        scp -q "$PROJECT_ROOT/data/unified_indicators/${PAIR}_unified.parquet" \
            root@$IP:/root/elementary_ma/data/unified_indicators/
        scp -q "$PROJECT_ROOT/data/elementary_ma_features/${PAIR}_features.npz" \
            root@$IP:/root/elementary_ma/data/elementary_ma_features/
    done
    echo "  $NAME setup complete."
}

setup_server $IP1 "elem-exp-1"
setup_server $IP2 "elem-exp-2"
setup_server $IP3 "elem-exp-3"

# ── Runner scripts — each writes to its own live_results.jsonl ────────────

make_runner() {
    local IP=$1
    local ARMS_CSV=$2
    local LABEL=$3
    ssh root@$IP "cat > /root/run.sh" <<SCRIPT
#!/bin/bash
set -eu
cd /root/elementary_ma
export CMA5IN_DATA_DIR=/root/elementary_ma/data/unified_indicators
export ELEMENTARY_FEATURES_DIR=/root/elementary_ma/data/elementary_ma_features
source /root/venv/bin/activate
export PYTHONUNBUFFERED=1

PAIRS="$PAIRS"
SEEDS="$SEEDS"
ARMS="$ARMS_CSV"
RESULTS=/root/elementary_ma/research/experiments/elementary_ma/results
mkdir -p \$RESULTS/progress

echo "$LABEL started: \$(date)"
COUNT=0
for ARM in \$ARMS; do
    for PAIR in \$PAIRS; do
        for SEED in \$SEEDS; do
            COUNT=\$((COUNT + 1))
            TAG="\${ARM}_\${PAIR}_s\${SEED}"
            if [ -f "\$RESULTS/\${TAG}_best.pkl" ]; then
                echo "[$LABEL \$COUNT] SKIP \$TAG"
                continue
            fi
            echo "[$LABEL \$COUNT] \$TAG ..."
            python3 research/experiments/elementary_ma/train_elementary_cma.py \\
                --pair \$PAIR --arm \$ARM --seed \$SEED \\
                --gens 200 --popsize 24 --workers 14 2>&1 | tail -6
        done
    done
done
echo "$LABEL done: \$(date)"
SCRIPT
    ssh root@$IP "chmod +x /root/run.sh"
}

make_runner $IP1 "E-atan E-pz-norm" "S1"
make_runner $IP2 "E-pz-rank E-mom" "S2"
make_runner $IP3 "E-all" "S3"

# ── Launch ────────────────────────────────────────────────────────────────
echo ""
echo "=== Launching all three ==="
for IP in $IP1 $IP2 $IP3; do
    ssh root@$IP "nohup /root/run.sh > /root/elem_exp.log 2>&1 &"
done
echo ""
echo "=== Monitor live ==="
echo "  # Aggregated leaderboard:"
echo "  python3 $SCRIPT_DIR/monitor_progress.py --servers $IP1 $IP2 $IP3 --watch 60"
echo ""
echo "  # Single-server log tail:"
echo "  ssh root@$IP1 'tail -f /root/elem_exp.log'"
echo ""
echo "=== Collect when done ==="
echo "  mkdir -p $SCRIPT_DIR/results/collected"
echo "  for IP in $IP1 $IP2 $IP3; do"
echo "    scp root@\$IP:/root/elementary_ma/research/experiments/elementary_ma/results/*.pkl \\"
echo "        $SCRIPT_DIR/results/collected/ 2>/dev/null"
echo "    scp root@\$IP:/root/elementary_ma/research/experiments/elementary_ma/results/live_results.jsonl \\"
echo "        $SCRIPT_DIR/results/live_results_\$(echo \$IP | tr . _).jsonl 2>/dev/null"
echo "  done"
echo ""
echo "=== Cleanup ==="
echo "  hcloud server delete elem-exp-1 && hcloud server delete elem-exp-2 && hcloud server delete elem-exp-3"

# Save IPs for later
echo "$IP1 $IP2 $IP3" > "$SCRIPT_DIR/elem_servers.txt"
echo "  (server IPs saved to $SCRIPT_DIR/elem_servers.txt)"
