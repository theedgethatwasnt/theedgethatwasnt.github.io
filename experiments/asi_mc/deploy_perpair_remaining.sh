#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# IronNet V3 Per-Pair — Deploy 10 remaining pairs to Hetzner
#
# EUR_GBP and CAD_JPY already trained (results/ironnet/).
# This trains the other 10 pairs, 1 server each.
#
# Architecture: 4 inputs → 4 hidden → 3 outputs + skip (V3, fixed topology)
# Inputs: MC(D), MC(dD), ER_norm, UPnL
# Training: Sine(30g) → Zigzag(50g) → WF P&L(200g)
#
# 10 servers × cx53 × ~3h = ~$3.00
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

REMAINING_PAIRS=(
    AUD_JPY
    AUD_USD
    CHF_JPY
    EUR_JPY
    EUR_USD
    GBP_JPY
    GBP_USD
    NZD_JPY
    NZD_USD
    USD_JPY
)
SEED=42
GENS=200
PRETRAIN_GENS=50
SINE_GENS=30
ISLANDS=4
POP=150
MAX_HOLD=200

# Verify parquets
echo "Checking parquets..."
MISSING=0
for PAIR in "${REMAINING_PAIRS[@]}"; do
    F="$DATA_DIR/${PAIR}_asi_mc.parquet"
    if [ ! -f "$F" ]; then
        echo "  MISSING: $F"
        MISSING=$((MISSING+1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: $MISSING parquet(s) missing. Attach neat-data volume and copy first."
    exit 1
fi
echo "OK: all 10 parquets found"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "IronNet V3 Per-Pair: 10 remaining pairs"
echo "10 servers × 4 islands × 200 gens | ~\$3.00 total"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Phase 1: Create all 10 servers in parallel ──
echo "Creating 10 servers..."
declare -A SERVER_IPS

for i in "${!REMAINING_PAIRS[@]}"; do
    PAIR="${REMAINING_PAIRS[$i]}"
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    (
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location hel1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
        if [ -z "$IP" ]; then
            echo "  hel1 quota hit for $PAIR, trying nbg1..."
            IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
                --location nbg1 --ssh-key "user@host" \
                -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
        fi
        echo "$PAIR $IP"
    ) &
done

# Collect IPs from parallel output
wait
echo ""

# Re-query all server IPs (reliable after creation)
echo "Collecting server IPs..."
SERVERS_FILE="$SCRIPT_DIR/perpair_servers.txt"
> "$SERVERS_FILE"
for PAIR in "${REMAINING_PAIRS[@]}"; do
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    IP=$(hcloud server describe "$NAME" -o json 2>/dev/null | \
         python3 -c "import sys,json; d=json.load(sys.stdin); print(d['public_net']['ipv4']['ip'])" 2>/dev/null)
    echo "$PAIR $IP"
    echo "$PAIR $IP" >> "$SERVERS_FILE"
done

# ── Phase 2: Wait for SSH on all servers ──
echo ""
echo "Waiting for SSH on all servers..."
while IFS=' ' read -r PAIR IP; do
    (
        for attempt in $(seq 1 40); do
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@$IP 'echo ok' 2>/dev/null && break
            sleep 3
        done
        echo "  $PAIR ($IP): ready"
    ) &
done < "$SERVERS_FILE"
wait
echo "All servers reachable"

# ── Phase 3: Setup all servers in parallel ──
echo ""
echo "Installing dependencies and uploading data..."

while IFS=' ' read -r PAIR IP; do
    (
        # Install
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/ironnet /root/neat/lib' 2>/dev/null

        # Upload just the parquet for this pair (saves transfer time)
        rsync -az "$DATA_DIR/${PAIR}_asi_mc.parquet" root@$IP:/root/neat/data/ 2>/dev/null

        # Upload scripts and libs
        rsync -az "$SCRIPT_DIR/train_ironnet_perpair.py" root@$IP:/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
        ssh root@$IP 'touch /root/neat/lib/__init__.py'
        echo "  [$PAIR] $IP: ready"
    ) &
done < "$SERVERS_FILE"
wait
echo "All servers ready!"

# ── Phase 4: Launch training on all servers ──
echo ""
echo "Launching training..."

TG_ENV=""
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

while IFS=' ' read -r PAIR IP; do
    LOG="results/ironnet/iron_v3_${PAIR}_s${SEED}.log"
    echo "  $PAIR → $IP"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup python3 train_ironnet_perpair.py \
            --pair $PAIR \
            --seed $SEED \
            --sine-gens $SINE_GENS \
            --pretrain-gens $PRETRAIN_GENS \
            --gens $GENS \
            --islands $ISLANDS \
            --pop $POP \
            --max-hold $MAX_HOLD \
        > $LOG 2>&1 &"
done < "$SERVERS_FILE"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "10 IronNet V3 per-pair runs launched!"
echo ""
echo "Monitor any pair:"
echo "  ssh root@<IP> 'tail -f /root/neat/results/ironnet/iron_v3_<PAIR>_s42.log'"
echo ""
echo "Server list: $SERVERS_FILE"
echo ""
echo "Collect results (after ~3h):"
cat << 'COLLECT'
  mkdir -p research/experiments/asi_mc/results/ironnet
  while IFS=' ' read -r PAIR IP; do
      scp "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_best.pkl" \
          research/experiments/asi_mc/results/ironnet/ 2>/dev/null || true
      scp "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_result.json" \
          research/experiments/asi_mc/results/ironnet/ 2>/dev/null || true
      echo "  $PAIR: collected"
  done < research/experiments/asi_mc/perpair_servers.txt
COLLECT
echo ""
echo "Cleanup (AFTER collecting):"
cat << 'CLEANUP'
  while IFS=' ' read -r PAIR IP; do
      NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
      hcloud server delete "$NAME" --yes
  done < research/experiments/asi_mc/perpair_servers.txt
CLEANUP
echo "═══════════════════════════════════════════════════════"
