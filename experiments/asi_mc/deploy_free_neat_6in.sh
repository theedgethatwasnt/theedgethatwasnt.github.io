#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Free NEAT v2 — 6-input variant (V5 inputs + free topology)
#
# Answers: does free topology find a better architecture for V5's
# 6-input set than IronNet's fixed 6→4→3 + skip?
#
# Inputs: MC_D, MC_dD, ER_norm, range_pos(30), bb_width(20), UPnL
# Config: neat_config_free_6in.ini (num_inputs=6)
# All other settings identical to Free NEAT v2 (4-input):
#   - 13 activations, seeded (1/activation)
#   - 4 islands, migrate every 10 gens
#   - Zigzag pretrain 50 gens
#   - 500 gens, stall=150
#
# 2 servers × 1 run each:
#   Server 1: EUR_GBP, seed 42
#   Server 2: CAD_JPY, seed 137
#
# Runtime: ~5-6h | Server: cx53 (~$0.10/hr) | Cost: ~$1.00
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

# Verify parquets exist
N_PARQUETS=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$N_PARQUETS" -lt 2 ]; then
    echo "ERROR: Need EUR_GBP and CAD_JPY parquets in $DATA_DIR"
    exit 1
fi
echo "OK: $N_PARQUETS parquets found"

RUNS=(
    "EUR_GBP:42"
    "CAD_JPY:137"
)
GENS=500
SINE_GENS=30
PRETRAIN_GENS=50
POP=150
ISLANDS=4
MIGRATE_EVERY=10
MAX_HOLD=200
STALL_LIMIT=150
N_INPUTS=6
RANGE_WINDOW=30
BB_WINDOW=20

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Free NEAT v2 6-input: 13 acts + seeded + 4 islands + zigzag"
echo "Inputs: MC_D, MC_dD, ER_norm, range_pos(30), bb_width(20), UPnL"
echo "2 servers | 500 gens | stall=150 | ~5h"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Creating 2 servers..."

SERVERS=()
for i in 1 2; do
    NAME="free-neat-6in-$i"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        echo "  hel1 full, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    echo "  $NAME: $IP"
    SERVERS+=("$IP")
    ssh-keygen -f '/path/to/.ssh/known_hosts' -R "$IP" 2>/dev/null || true
done

echo ""
echo "Servers: ${SERVERS[*]}"
echo "${SERVERS[*]}" > "$SCRIPT_DIR/free_neat_6in_servers.txt"

# Wait for SSH
echo "Waiting for SSH..."
for IP in "${SERVERS[@]}"; do
    (
        for attempt in $(seq 1 30); do
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@$IP 'echo ok' 2>/dev/null && break
            sleep 2
        done
        echo "  $IP: ready"
    ) &
done
wait
echo "All servers reachable"

# Setup
echo ""
echo "Setting up servers..."
for IP in "${SERVERS[@]}"; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q "neat-python==0.92" numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/free_neat_6in /root/neat/lib' 2>/dev/null

        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$DATA_DIR/EUR_GBP_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$DATA_DIR/CAD_JPY_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true

        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$SCRIPT_DIR/train_free_neat.py" root@$IP:/root/neat/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$SCRIPT_DIR/neat_config_free_6in.ini" root@$IP:/root/neat/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
        ssh -o StrictHostKeyChecking=no root@$IP 'touch /root/neat/lib/__init__.py'
        echo "  [$IP] Ready"
    ) &
done
wait
echo "All servers ready!"

# Launch
echo ""
echo "Launching training runs..."

TG_ENV=""
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

for i in "${!RUNS[@]}"; do
    IFS=: read -r PAIR SEED <<< "${RUNS[$i]}"
    IP="${SERVERS[$i]}"
    LOG="results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED}.log"
    echo "  Server $((i+1)) ($IP): ${PAIR} seed ${SEED}"
    ssh -o StrictHostKeyChecking=no root@$IP "
        source /root/venv/bin/activate && cd /root/neat
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_free_neat.py \
            --pair $PAIR --seed $SEED \
            --gens $GENS --sine-gens $SINE_GENS --pretrain-gens $PRETRAIN_GENS \
            --pop $POP --islands $ISLANDS \
            --migrate-every $MIGRATE_EVERY \
            --max-hold $MAX_HOLD --stall-limit $STALL_LIMIT \
            --n-inputs $N_INPUTS \
            --range-window $RANGE_WINDOW --bb-window $BB_WINDOW \
        > $LOG 2>&1 &
        disown; echo launched
    " 2>/dev/null
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Free NEAT v2 6-input runs launched!"
echo ""
echo "Monitor:"
for i in "${!RUNS[@]}"; do
    IFS=: read -r PAIR SEED <<< "${RUNS[$i]}"
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED}.log'"
done
echo ""
echo "Collect (after ~5h):"
echo "  mkdir -p $SCRIPT_DIR/results/free_neat_6in"
echo "  for i in 0 1; do"
echo "    SERVERS=(${SERVERS[*]})"
echo "    RUNS=(EUR_GBP CAD_JPY)"
echo "    scp root@\${SERVERS[\$i]}:/root/neat/results/free_neat_6in/*_best.pkl $SCRIPT_DIR/results/free_neat_6in/"
echo "    scp root@\${SERVERS[\$i]}:/root/neat/results/free_neat_6in/*_result.json $SCRIPT_DIR/results/free_neat_6in/"
echo "  done"
echo ""
echo "Cleanup:"
echo "  hcloud server delete free-neat-6in-1 && hcloud server delete free-neat-6in-2"
echo "═══════════════════════════════════════════════════════"
