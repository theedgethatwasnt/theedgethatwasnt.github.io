#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Zigzag Per-Pair V3 — Deploy 2 pairs to 2 Hetzner servers
#
# Server 1: EUR_GBP (weakest pair, most room for improvement)
# Server 2: CAD_JPY (mid-tier pair, baseline comparison)
#
# Each server: 4 islands × 150 pop, 50 pretrain + 150 evolve gens
# Server: cx53 (16 vCPU, 32GB), ~$0.10/hr
# ETA: ~1.5-2h per server, total cost ~$0.40
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

PAIRS=("EUR_GBP" "CAD_JPY")
SEED=42
PRETRAIN_GENS=50
EVOLVE_GENS=150
POP=150
ISLANDS=4
MAX_HOLD=200

echo "═══════════════════════════════════════════════════════"
echo "Zigzag Per-Pair V3: EUR_GBP + CAD_JPY"
echo "50 pretrain (zigzag) + 150 evolve (P&L) | 4×150 islands"
echo "═══════════════════════════════════════════════════════"

# ── Phase 1: Create servers ──
echo ""
echo "Creating 2 servers..."
SERVERS=()
for i in 1 2; do
    NAME="zz-perpair-$i"
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

echo "Servers: ${SERVERS[*]}"
echo "${SERVERS[*]}" > "$SCRIPT_DIR/zz_perpair_servers.txt"

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

# ── Phase 2: Setup servers ──
echo ""
echo "Setting up servers..."
for IP in "${SERVERS[@]}"; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/zigzag_perpair /root/neat/lib' 2>/dev/null

        # Only send the 2 needed parquets (not all 12)
        for PAIR in "${PAIRS[@]}"; do
            rsync -az "$DATA_DIR/${PAIR}_asi_mc.parquet" root@$IP:/root/neat/data/
        done
        rsync -az "$SCRIPT_DIR/train_zigzag_perpair.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_4in_3out.ini" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_3out.ini" root@$IP:/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/pair_config.py" root@$IP:/root/neat/lib/ 2>/dev/null || true
        rsync -az "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/ 2>/dev/null || true
        ssh root@$IP 'touch /root/neat/lib/__init__.py'
        echo "  [$IP] Ready"
    ) &
done
wait
echo "All servers ready!"

# ── Phase 3: Launch training ──
echo ""
echo "Launching training..."

TG_ENV=""
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

for i in "${!PAIRS[@]}"; do
    PAIR="${PAIRS[$i]}"
    IP="${SERVERS[$i]}"
    LOG="results/zigzag_perpair/zz_v3_${PAIR}_s${SEED}.log"
    echo "  Server $((i+1)) ($IP): $PAIR seed $SEED"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_zigzag_perpair.py \
            --pair $PAIR \
            --seed $SEED \
            --pretrain-gens $PRETRAIN_GENS \
            --gens $EVOLVE_GENS \
            --islands $ISLANDS \
            --pop $POP \
            --max-hold $MAX_HOLD \
        > $LOG 2>&1 &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "2 training runs launched!"
echo ""
echo "Monitor:"
for i in "${!PAIRS[@]}"; do
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/zigzag_perpair/zz_v3_${PAIRS[$i]}_s${SEED}.log'"
done
echo ""
echo "Collect:"
echo "  mkdir -p $SCRIPT_DIR/results/zigzag_perpair"
echo "  for IP in ${SERVERS[*]}; do"
echo "    scp -r root@\$IP:/root/neat/results/zigzag_perpair/* $SCRIPT_DIR/results/zigzag_perpair/"
echo "  done"
echo ""
echo "Cleanup:"
echo "  for i in 1 2; do hcloud server delete zz-perpair-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
