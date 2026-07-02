#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# IronNet V5 — Deploy to 2 Hetzner servers
#
# V5: MC(D) + MC(dD) + ER_norm + range_pos(30) + bb_width(20) + UPnL
#     Fixed topology: 6→4→3 + skip (54 connections)
#
# 2 servers × 4 islands = 8 populations in parallel:
#   Server 1: EUR_GBP, seed 42
#   Server 2: CAD_JPY, seed 137
#
# Training: Sine(30g) → Zigzag(50g) → WF P&L(200g)
# Compare OOS vs IronNet V3 baselines.
#
# Server: cx53 (16 vCPU, 32GB), ~$0.10/hr
# Total cost: 2 × $0.10 × 3h = ~$0.60
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

# Verify parquets exist
N_PARQUETS=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$N_PARQUETS" -lt 2 ]; then
    echo "ERROR: Need at least EUR_GBP and CAD_JPY parquets in $DATA_DIR"
    exit 1
fi
echo "OK: $N_PARQUETS parquets found"

RUNS=(
    "EUR_GBP:42"
    "CAD_JPY:137"
)
GENS=200
PRETRAIN_GENS=50
SINE_GENS=30
POP=150
ISLANDS=4
MAX_HOLD=200

# ── Phase 1: Create 2 servers ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "IronNet V5: 6-input fixed topology (54 conn)"
echo "2 servers × 4 islands | Sine→Zigzag→WF"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Creating 2 servers..."

SERVERS=()
for i in 1 2; do
    NAME="ironnet-v5-$i"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        echo "  hel1 failed, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    echo "  $NAME: $IP"
    SERVERS+=("$IP")
done

echo ""
echo "Servers: ${SERVERS[*]}"
echo "${SERVERS[*]}" > "$SCRIPT_DIR/v5_servers.txt"

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

# ── Phase 2: Setup all servers in parallel ──
echo ""
echo "Setting up servers..."

for IP in "${SERVERS[@]}"; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/ironnet_v5 /root/neat/lib' 2>/dev/null

        # Copy training data (just the 2 pairs we need)
        rsync -az "$DATA_DIR/EUR_GBP_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true
        rsync -az "$DATA_DIR/CAD_JPY_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true

        rsync -az "$SCRIPT_DIR/train_ironnet_v5.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_6in_3out.ini" root@$IP:/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
        ssh root@$IP 'touch /root/neat/lib/__init__.py'
        echo "  [$IP] Ready"
    ) &
done
wait
echo "All servers ready!"

# ── Phase 3: Launch training ──
echo ""
echo "Launching training runs..."

TG_ENV=""
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

for i in "${!RUNS[@]}"; do
    IFS=: read -r PAIR SEED <<< "${RUNS[$i]}"
    IP="${SERVERS[$i]}"
    LOG="results/ironnet_v5/iron_v5_${PAIR}_s${SEED}.log"
    echo "  Server $((i+1)) ($IP): ${PAIR} seed ${SEED}"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_ironnet_v5.py \
            --pair $PAIR \
            --seed $SEED \
            --sine-gens $SINE_GENS \
            --pretrain-gens $PRETRAIN_GENS \
            --gens $GENS \
            --islands $ISLANDS \
            --pop $POP \
            --max-hold $MAX_HOLD \
        > $LOG 2>&1 &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "2 IronNet V5 training runs launched!"
echo ""
echo "Monitor:"
for i in "${!RUNS[@]}"; do
    IFS=: read -r PAIR SEED <<< "${RUNS[$i]}"
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/ironnet_v5/iron_v5_${PAIR}_s${SEED}.log'"
done
echo ""
echo "Collect results (after ~3h):"
echo "  mkdir -p $SCRIPT_DIR/results/ironnet_v5"
echo "  for IP in ${SERVERS[*]}; do"
echo "    scp 'root@\$IP:/root/neat/results/ironnet_v5/*_best.pkl' $SCRIPT_DIR/results/ironnet_v5/"
echo "    scp 'root@\$IP:/root/neat/results/ironnet_v5/*_result.json' $SCRIPT_DIR/results/ironnet_v5/ 2>/dev/null || true"
echo "  done"
echo ""
echo "Cleanup (AFTER collecting):"
echo "  for i in 1 2; do hcloud server delete ironnet-v5-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
