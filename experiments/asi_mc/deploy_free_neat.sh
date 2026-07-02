#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Free NEAT v2 — Deploy to 2 Hetzner servers
#
# Fixes vs v1:
#   - All 13 activations (added sech/dog/gabor/sinc/mex_hat/
#     morlet_re/morlet_im/sigmoid/haar — v1 had only 4)
#   - Seeded population: 1 genome/activation injected at gen 0
#     (activation_study technique — ~2x faster early convergence)
#   - 4 islands with migration every 10 gens (was 1 population)
#   - Zigzag pretrain 50 gens before WF evolution
#   - Stall limit 150 (was 60 — topology needs time to develop)
#   - 500 gens (was 300)
#
# 2 servers × 1 run each:
#   Server 1: EUR_GBP, seed 42
#   Server 2: CAD_JPY, seed 137
#
# Runtime: ~4-6h (longer due to islands + pretrain)
# Server: cx53 (16 vCPU, 32GB), ~$0.10/hr
# Total cost: 2 × $0.10 × 5h = ~$1.00
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

# ── Phase 1: Create 2 servers ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "Free NEAT v2: 13 activations + seeded + 4 islands + zigzag pretrain"
echo "2 servers | 500 gens | stall=150 | ~5h"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Creating 2 servers..."

SERVERS=()
for i in 1 2; do
    NAME="free-neat-$i"
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
echo "${SERVERS[*]}" > "$SCRIPT_DIR/free_neat_servers.txt"

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

# ── Phase 2: Setup servers ──
echo ""
echo "Setting up servers..."

for IP in "${SERVERS[@]}"; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/free_neat /root/neat/lib' 2>/dev/null

        rsync -az "$DATA_DIR/EUR_GBP_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true
        rsync -az "$DATA_DIR/CAD_JPY_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null || true

        rsync -az "$SCRIPT_DIR/train_free_neat.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_free_4in.ini" root@$IP:/root/neat/
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
    LOG="results/free_neat/free_neat_v2_${PAIR}_s${SEED}.log"
    echo "  Server $((i+1)) ($IP): ${PAIR} seed ${SEED}"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_free_neat.py \
            --pair $PAIR \
            --seed $SEED \
            --gens $GENS \
            --sine-gens $SINE_GENS \
            --pretrain-gens $PRETRAIN_GENS \
            --pop $POP \
            --islands $ISLANDS \
            --migrate-every $MIGRATE_EVERY \
            --max-hold $MAX_HOLD \
            --stall-limit $STALL_LIMIT \
        > $LOG 2>&1 &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "2 Free NEAT training runs launched!"
echo ""
echo "Monitor:"
for i in "${!RUNS[@]}"; do
    IFS=: read -r PAIR SEED <<< "${RUNS[$i]}"
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/free_neat/free_neat_${PAIR}_s${SEED}.log'"
done
echo ""
echo "Collect results (after ~3h or stall):"
echo "  mkdir -p $SCRIPT_DIR/results/free_neat"
echo "  for IP in ${SERVERS[*]}; do"
echo "    scp 'root@\$IP:/root/neat/results/free_neat/*_best.pkl' $SCRIPT_DIR/results/free_neat/"
echo "    scp 'root@\$IP:/root/neat/results/free_neat/*_result.json' $SCRIPT_DIR/results/free_neat/ 2>/dev/null || true"
echo "  done"
echo ""
echo "Cleanup (AFTER collecting):"
echo "  for i in 1 2; do hcloud server delete free-neat-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
