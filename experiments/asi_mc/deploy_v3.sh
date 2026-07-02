#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# ASI-MC V3 — Deploy to 4 Hetzner servers
#
# V3: MC(D) + MC(dD) + ER_norm + UPnL (4 inputs, 3 outputs)
# ER_norm = Kaufman ER(60 M5 bars) arctan-normalized
# Seeded from v2 genome — saves ~50 gens of structure search
#
# 4 servers × 4 islands = 16 populations in parallel:
#   Server 1: seed 42,  exponent 0.4
#   Server 2: seed 137, exponent 0.5
#   Server 3: seed 23,  exponent 0.5
#   Server 4: seed 99,  exponent 0.7
#
# ETA reduction vs V2 training:
#   - 150 gens (vs 200) = -25% wall time
#   - Seeded from v2 = faster convergence (~50 gen head start)
#   - cx53 16vCPU = same server as before
#   - Total: ~2h vs ~3h for Variant A
#
# Server: cx53 (16 vCPU, 32GB), ~$0.10/hr
# Total cost: 4 × $0.10 × 2h = ~$0.80
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"
MODELS_DIR="$REPO_DIR/models"
SEED_GENOME="$MODELS_DIR/asi_mc_v2_best.pkl"

# Verify parquets exist and have er_norm column
N_PARQUETS=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$N_PARQUETS" -lt 12 ]; then
    echo "ERROR: Only $N_PARQUETS/12 indicator parquets in $DATA_DIR"
    echo "Run export_indicators.py first!"
    exit 1
fi

# Quick check for er_norm column in first parquet
FIRST_PARQUET=$(ls "$DATA_DIR"/*.parquet | head -1)
HAS_ER=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$FIRST_PARQUET', engine='pyarrow')
print('yes' if 'er_norm' in df.columns else 'no')
" 2>/dev/null)

if [ "$HAS_ER" != "yes" ]; then
    echo "ERROR: 'er_norm' column not found in parquets."
    echo "Run first: python3 export_d_indicators.py"
    exit 1
fi
echo "OK: $N_PARQUETS parquets with er_norm ($(du -sh "$DATA_DIR" | cut -f1))"

# Verify seed genome
if [ ! -f "$SEED_GENOME" ]; then
    echo "ERROR: Seed genome not found: $SEED_GENOME"
    exit 1
fi
echo "OK: Seed genome: $(basename $SEED_GENOME)"

ALL_PAIRS="EUR_JPY,USD_JPY,GBP_JPY,AUD_JPY,CAD_JPY,CHF_JPY,NZD_JPY,EUR_USD,GBP_USD,AUD_USD,NZD_USD,EUR_GBP"
GENS=150
POP=150
ISLANDS=4
MAX_HOLD=200

# Server config: SEED EXPONENT
RUNS=(
    "42:0.4"
    "137:0.5"
    "23:0.5"
    "99:0.7"
)

# ── Phase 1: Create 4 servers ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "ASI-MC V3: MC(D) + MC(dD) + ER_norm + UPnL"
echo "4 servers × 4 islands = 16 populations | 150 gens | seeded from v2"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Creating 4 servers..."

SERVERS=()
for i in 1 2 3 4; do
    NAME="asi-mc-v3-$i"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        echo "  hel1 failed for $NAME, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    echo "  $NAME: $IP"
    SERVERS+=("$IP")
done

echo ""
echo "Servers: ${SERVERS[*]}"
echo "${SERVERS[*]}" > "$SCRIPT_DIR/v3_servers.txt"

# Wait for SSH
echo "Waiting for SSH on all servers..."
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
echo "Setting up all 4 servers in parallel..."

for IP in "${SERVERS[@]}"; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results /root/neat/lib /root/neat/models' 2>/dev/null

        rsync -az --progress "$DATA_DIR/"*.parquet root@$IP:/root/neat/data/
        rsync -az "$SEED_GENOME" root@$IP:/root/neat/models/
        rsync -az "$SCRIPT_DIR/train_from_indicators.py" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/train_pretrain_continue.py" root@$IP:/root/neat/ 2>/dev/null || true
        rsync -az "$SCRIPT_DIR/neat_config_3out.ini" root@$IP:/root/neat/
        rsync -az "$SCRIPT_DIR/neat_config_4in_3out.ini" root@$IP:/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/pair_config.py" root@$IP:/root/neat/lib/
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
    IFS=: read -r SEED EXP <<< "${RUNS[$i]}"
    IP="${SERVERS[$i]}"
    LOG="results/v3_s${SEED}_exp${EXP}.log"
    echo "  Server $((i+1)) ($IP): seed $SEED exponent $EXP"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_from_indicators.py \
            --mode v3 \
            --pairs $ALL_PAIRS \
            --seed $SEED \
            --gens $GENS \
            --islands $ISLANDS \
            --pop $POP \
            --max-hold $MAX_HOLD \
            --genome /root/neat/models/asi_mc_v2_best.pkl \
        > $LOG 2>&1 &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "4 V3 training runs launched!"
echo ""
echo "Monitor:"
for i in "${!RUNS[@]}"; do
    IFS=: read -r SEED EXP <<< "${RUNS[$i]}"
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/v3_s${SEED}_exp${EXP}.log'"
done
echo ""
echo "Collect results:"
echo "  mkdir -p $SCRIPT_DIR/results/v3"
echo "  for IP in ${SERVERS[*]}; do"
echo "    scp root@\$IP:/root/neat/results/v3*best*.pkl $SCRIPT_DIR/results/v3/"
echo "    scp root@\$IP:/root/neat/results/v3*result.json $SCRIPT_DIR/results/v3/ 2>/dev/null || true"
echo "  done"
echo ""
echo "Cleanup (AFTER collecting results):"
echo "  for i in 1 2 3 4; do hcloud server delete asi-mc-v3-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
