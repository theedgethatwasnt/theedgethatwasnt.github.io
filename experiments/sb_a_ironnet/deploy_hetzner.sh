#!/usr/bin/env bash
# Deploy SB_A IronNet CMA training to 4 Hetzner servers.
# Layout: 3 pairs × 4 seeds per server = 12 runs each, 48 total.
# Servers: ccx23 (8 vCPU, 16GB) ~$0.08/hr

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RANGE_DATA_DIR="$PROJECT_ROOT/data/range_bar_causal"
RESULTS_DIR="$SCRIPT_DIR/results"
SERVERS_FILE="$SCRIPT_DIR/hetzner_servers.txt"

mkdir -p "$RESULTS_DIR"

SERVER_TYPE="cx43"   # 8 shared vCPU, 16GB — no dedicated-core limit

# 4 servers × 3 pairs each
declare -a SERVER_PAIRS=(
    "EUR_JPY GBP_JPY USD_JPY"
    "EUR_USD GBP_USD AUD_USD"
    "NZD_USD EUR_GBP AUD_JPY"
    "CAD_JPY CHF_JPY NZD_JPY"
)

echo "=== Creating 4 Hetzner ccx23 servers ==="
> "$SERVERS_FILE"

for i in 1 2 3 4; do
    NAME="sba-exp-$i"
    echo "Creating $NAME..."
    IP=$(hcloud server create \
        --name "$NAME" \
        --type "$SERVER_TYPE" \
        --image ubuntu-24.04 \
        --location hel1 \
        --ssh-key "user@host" \
        --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['server']['public_net']['ipv4']['ip'])")
    echo "$NAME $IP ${SERVER_PAIRS[$((i-1))]}" >> "$SERVERS_FILE"
    echo "  $NAME -> $IP"
done

echo ""
echo "Waiting 30s for servers to boot..."
sleep 30

# Setup + upload to each server
while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    IP=$(echo "$line" | awk '{print $2}')
    PAIRS="${line#* * }"
    # Re-parse pairs
    PAIR1=$(echo "$line" | awk '{print $3}')
    PAIR2=$(echo "$line" | awk '{print $4}')
    PAIR3=$(echo "$line" | awk '{print $5}')

    echo "=== Setting up $NAME ($IP): $PAIR1 $PAIR2 $PAIR3 ==="

    # Install dependencies
    ssh -o StrictHostKeyChecking=no root@"$IP" "
        apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git 2>/dev/null
        python3 -m venv /root/venv
        /root/venv/bin/pip install -q --upgrade pip
        /root/venv/bin/pip install -q numpy pandas cma numba pyarrow
        mkdir -p /root/sba_exp/data /root/sba_exp/results /root/sba_exp/lib
    " &

done < "$SERVERS_FILE"
wait

echo ""
echo "=== Uploading code and data ==="
while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    IP=$(echo "$line" | awk '{print $2}')
    PAIR1=$(echo "$line" | awk '{print $3}')
    PAIR2=$(echo "$line" | awk '{print $4}')
    PAIR3=$(echo "$line" | awk '{print $5}')

    echo "Uploading to $NAME ($IP)..."

    # Upload training script and lib dependency
    scp -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/train_sb_a_cma.py" \
        root@"$IP":/root/sba_exp/

    scp -o StrictHostKeyChecking=no \
        "$PROJECT_ROOT/lib/incremental_topsbots.py" \
        root@"$IP":/root/sba_exp/lib/

    # Upload range bar data for the 3 assigned pairs
    for PAIR in $PAIR1 $PAIR2 $PAIR3; do
        PFILE="$RANGE_DATA_DIR/${PAIR}_range10_causal.parquet"
        if [ -f "$PFILE" ]; then
            scp -o StrictHostKeyChecking=no "$PFILE" root@"$IP":/root/sba_exp/data/
        else
            echo "  WARNING: $PFILE not found!"
        fi
    done

done < "$SERVERS_FILE"

echo ""
echo "=== Launching training runs ==="
while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    IP=$(echo "$line" | awk '{print $2}')
    PAIR1=$(echo "$line" | awk '{print $3}')
    PAIR2=$(echo "$line" | awk '{print $4}')
    PAIR3=$(echo "$line" | awk '{print $5}')

    echo "Launching on $NAME: $PAIR1 $PAIR2 $PAIR3 (seeds 0-3)"

    ssh -o StrictHostKeyChecking=no root@"$IP" "
        cd /root/sba_exp
        mkdir -p results

        for PAIR in $PAIR1 $PAIR2 $PAIR3; do
            for SEED in 0 1 2 3; do
                LOG=\"results/\${PAIR}_seed\${SEED}.log\"
                PYTHONPATH=/root/sba_exp RANGE_DATA_DIR=/root/sba_exp/data \
                nohup /root/venv/bin/python3 train_sb_a_cma.py \\
                    --pair \$PAIR --seed \$SEED --workers 8 \\
                    > \$LOG 2>&1 &
                echo \"  Started \$PAIR seed=\$SEED (PID=\$!)\"
            done
        done
        echo 'All launched on $NAME'
    " &

done < "$SERVERS_FILE"
wait

echo ""
echo "=== All 48 runs launched ==="
echo "Servers:"
cat "$SERVERS_FILE"
echo ""
echo "Monitor: ssh root@<IP> 'tail -f /root/sba_exp/results/*.log | grep -E \"Gen|BEST|ERROR\"'"
echo "Collect: ./collect_results.sh"
