#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# ASI-MC Variant C — Deploy to 5 Hetzner servers
#
# 6 runs (3 thresholds × 2 seeds), parallel across 5 servers:
#   Server 1: C02 seed 42       (threshold 0.02)
#   Server 2: C02 seed 137      (threshold 0.02)
#   Server 3: C05 seed 42       (threshold 0.05)
#   Server 4: C05 seed 137      (threshold 0.05)
#   Server 5: C10 seed 42 + C10 seed 137 (sequential, threshold 0.10)
#
# Pre-computed indicator parquets (~36MB upload per server).
# Server: cx53 (16 vCPU, 32GB), ~$0.10/hr
# Total cost: ~$0.50/hr × ~1hr = ~$0.50
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

# Verify all 12 indicator parquets exist
N_PARQUETS=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$N_PARQUETS" -lt 12 ]; then
    echo "ERROR: Only $N_PARQUETS/12 indicator parquets in $DATA_DIR"
    echo "Run export_indicators.py first!"
    exit 1
fi
echo "OK: $N_PARQUETS indicator parquets ready ($(du -sh "$DATA_DIR" | cut -f1))"

TRAIN_PAIRS="EUR_JPY,USD_JPY,GBP_JPY,AUD_JPY"
GENS=200
POP=150
ISLANDS=4
MAX_HOLD=200

# Run definitions: VARIANT SEED
RUNS=(
    "C02 42"
    "C02 137"
    "C05 42"
    "C05 137"
    "C10 42"
    "C10 137"
)

# ── Phase 1: Create 5 servers ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "ASI-MC Variant C: Quantized Inputs"
echo "3 thresholds × 2 seeds = 6 runs on 5 servers"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Creating 5 servers..."

SERVERS=()
for i in 1 2 3 4 5; do
    NAME="asi-mc-c-$i"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        echo "  hel1 failed for $NAME, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    echo "  $NAME: $IP"
    SERVERS+=("$IP")
done

echo ""
echo "Servers: ${SERVERS[*]}"
echo "${SERVERS[*]}" > "$SCRIPT_DIR/varC_servers.txt"

# Wait for all servers SSH
echo "Waiting for SSH on all servers..."
for IP in "${SERVERS[@]}"; do
    for attempt in $(seq 1 30); do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@$IP 'echo ok' 2>/dev/null && break
        sleep 2
    done &
done
wait
echo "  All servers reachable"

# ── Phase 2: Setup all servers in parallel ──
echo ""
echo "Setting up all 5 servers in parallel..."

for IP in "${SERVERS[@]}"; do
    (
        echo "  [$IP] Installing packages..."
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results /root/neat/lib' 2>/dev/null

        echo "  [$IP] Uploading data + scripts..."
        rsync -az "$DATA_DIR/"*.parquet root@$IP:/root/neat/data/
        rsync -az "$SCRIPT_DIR/train_from_indicators.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/train_pretrain_continue.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_3out.ini" root@$IP:/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/pair_config.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
        ssh root@$IP 'touch /root/neat/lib/__init__.py && ln -sf /root/neat/lib/asi_indicator.py /root/neat/asi_indicator.py'
        echo "  [$IP] Ready"
    ) &
done
wait
echo "All servers ready!"

# ── Phase 3: Launch training ──
echo ""
echo "Launching training runs..."

# Telegram env vars
TG_ENV=""
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

# Servers 1-4: one run each (C02×2, C05×2)
# Server 5: two runs sequential (C10×2)
ASSIGNMENTS=(
    "0:C02:42"       # Server 1
    "1:C02:137"      # Server 2
    "2:C05:42"       # Server 3
    "3:C05:137"      # Server 4
)

for ASSIGN in "${ASSIGNMENTS[@]}"; do
    IFS=: read -r IDX VAR SEED <<< "$ASSIGN"
    IP="${SERVERS[$IDX]}"
    LOG="results/v${VAR}_s${SEED}.log"
    echo "  Server $((IDX+1)) ($IP): $VAR seed $SEED"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_from_indicators.py \
            --variant $VAR --pairs $TRAIN_PAIRS --seed $SEED \
            --gens $GENS --islands $ISLANDS --pop $POP --max-hold $MAX_HOLD \
        > $LOG 2>&1 &"
done

# Server 5: two sequential runs
IP5="${SERVERS[4]}"
echo "  Server 5 ($IP5): C10 seed 42 + C10 seed 137 (sequential)"
ssh root@$IP5 "source /root/venv/bin/activate && cd /root/neat && \
    ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
    nohup bash -c '
        python3 train_from_indicators.py \
            --variant C10 --pairs $TRAIN_PAIRS --seed 42 \
            --gens $GENS --islands $ISLANDS --pop $POP --max-hold $MAX_HOLD \
            2>&1 | tee results/vC10_s42.log
        python3 train_from_indicators.py \
            --variant C10 --pairs $TRAIN_PAIRS --seed 137 \
            --gens $GENS --islands $ISLANDS --pop $POP --max-hold $MAX_HOLD \
            2>&1 | tee results/vC10_s137.log
    ' > results/server5_full.log 2>&1 &"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "All 6 training runs launched across 5 servers!"
echo ""
echo "Monitor:"
for i in 0 1 2 3; do
    IFS=: read -r _ VAR SEED <<< "${ASSIGNMENTS[$i]}"
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/v${VAR}_s${SEED}.log'"
done
echo "  ssh root@${SERVERS[4]} 'tail -f /root/neat/results/server5_full.log'"
echo ""
echo "Collect results when done:"
echo "  mkdir -p $SCRIPT_DIR/results"
echo "  for IP in ${SERVERS[*]}; do"
echo "    scp root@\$IP:/root/neat/results/vC*_best.pkl $SCRIPT_DIR/results/"
echo "    scp root@\$IP:/root/neat/results/vC*_result.json $SCRIPT_DIR/results/"
echo "  done"
echo ""
echo "Cleanup (AFTER collecting results):"
echo "  for i in 1 2 3 4 5; do hcloud server delete asi-mc-c-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
