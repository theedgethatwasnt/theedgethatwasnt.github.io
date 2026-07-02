#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CMA-NN Multi-Seed × 12 Pairs × {H1, M5} on Hetzner
#
# 4 seeds × 12 pairs × 2 TFs = 96 runs
# H1: ~20s/run × 48 = ~16 min per server (2 servers)
# M5: ~3min/run × 48 = ~2.4h per server (2 servers)
#
# 4 servers total, each handles 12 pairs at one seed+TF combo
# Server layout (interleaved for load balance):
#   S1: H1 seeds 42,137 (24 runs, ~8 min)
#   S2: H1 seeds 23,99  (24 runs, ~8 min)
#   S3: M5 seeds 42,137 (24 runs, ~1.2h)
#   S4: M5 seeds 23,99  (24 runs, ~1.2h)
#
# Cost: 4 × ccx23 × 1.5h × $0.07/hr ≈ $0.42
# ═══════════════════════════════════════════════════════════════

set -e

GENS=200
POPSIZE=24
EXTRAS="macd_hist"
MAX_HOLD_M5=200
MAX_HOLD_H1=17

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
CMA_DIR="$PROJECT_DIR/research/experiments/cma_5in"
RESULTS_DIR="$CMA_DIR/results"
SERVER_FILE="$CMA_DIR/multiseed_servers.txt"

ALL_PAIRS="EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP"

mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════"
echo "  CMA-NN Multi-Seed: 4 seeds × 12 pairs × {H1,M5}"
echo "  Gens: $GENS, Pop: $POPSIZE, Act: sin"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Create 4 servers ──
echo ""
echo "Creating 4 Hetzner servers..."
> "$SERVER_FILE"

for i in 1 2 3 4; do
    NAME="cma-ms-$i"
    if hcloud server describe "$NAME" > /dev/null 2>&1; then
        echo "  $NAME already exists"
    else
        hcloud server create --name "$NAME" --type ccx23 --image ubuntu-24.04 \
            --location hel1 --ssh-key "user@host" --without-ipv6 2>/dev/null
    fi
    IP=$(hcloud server ip "$NAME")
    echo "$i $IP" >> "$SERVER_FILE"
    echo "  $NAME → $IP"
done

echo ""
echo "Waiting 30s for servers to boot..."
sleep 30

# ── Step 2: Setup all servers ──
while read IDX IP; do
    echo "Setting up server $IDX ($IP)..."
    ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP'
set -e
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba pyarrow cma
mkdir -p /root/fx-core/lib \
         /root/fx-core/research/experiments/cma_5in/results \
         /root/fx-core/data/m5_ohlc \
         /root/fx-core/data/unified_indicators
SETUP

    scp -q "$PROJECT_DIR/lib/fast_eval.py" root@$IP:/root/fx-core/lib/
    scp -q "$PROJECT_DIR/lib/asi_indicator.py" root@$IP:/root/fx-core/lib/
    scp -q "$CMA_DIR/train_cma_v2.py" root@$IP:/root/fx-core/research/experiments/cma_5in/
    scp -q "$CMA_DIR/extra_indicators.py" root@$IP:/root/fx-core/research/experiments/cma_5in/
    rsync -az "$PROJECT_DIR/data/m5_ohlc/" root@$IP:/root/fx-core/data/m5_ohlc/
    rsync -az "$PROJECT_DIR/data/unified_indicators/" root@$IP:/root/fx-core/data/unified_indicators/
    echo "  Server $IDX ready"
done < "$SERVER_FILE"

# ── Step 3: Create runner scripts on each server ──

# Server assignments:
#   S1: H1 seeds 42+137    S2: H1 seeds 23+99
#   S3: M5 seeds 42+137    S4: M5 seeds 23+99

declare -A SERVER_TF
declare -A SERVER_SEEDS
SERVER_TF[1]="H1"; SERVER_SEEDS[1]="42 137"
SERVER_TF[2]="H1"; SERVER_SEEDS[2]="23 99"
SERVER_TF[3]="M5"; SERVER_SEEDS[3]="42 137"
SERVER_TF[4]="M5"; SERVER_SEEDS[4]="23 99"

while read IDX IP; do
    TF=${SERVER_TF[$IDX]}
    SEEDS=${SERVER_SEEDS[$IDX]}
    if [ "$TF" = "H1" ]; then
        MAX_HOLD=$MAX_HOLD_H1
    else
        MAX_HOLD=$MAX_HOLD_M5
    fi
    LABEL="ms_${TF}"

    echo "Launching server $IDX ($IP): TF=$TF seeds=$SEEDS"

    ssh root@$IP bash <<TRAIN &
set -e
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/cma_5in

for SEED in $SEEDS; do
    for PAIR in $ALL_PAIRS; do
        echo "\$(date) Training \$PAIR seed \$SEED TF $TF"
        PYTHONUNBUFFERED=1 python3 train_cma_v2.py \\
            --pair \$PAIR --seed \$SEED \\
            --gens $GENS --features v3_plus --extras $EXTRAS \\
            --fixed-activation sin --popsize $POPSIZE \\
            --workers 4 --max-hold $MAX_HOLD \\
            --tf $TF --label $LABEL \\
            2>&1 | tail -5
        echo "  done \$PAIR s\$SEED $TF"
    done
done
echo "\$(date) SERVER $IDX ALL DONE ($TF $SEEDS)"
TRAIN

done < "$SERVER_FILE"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All 4 servers launched!"
echo ""
echo "  Server layout:"
while read IDX IP; do
    echo "    S$IDX ($IP): ${SERVER_TF[$IDX]} seeds=${SERVER_SEEDS[$IDX]}"
done < "$SERVER_FILE"
echo ""
echo "  Monitor (pick a server):"
while read IDX IP; do
    echo "    ssh root@$IP 'tail -f /root/fx-core/research/experiments/cma_5in/results/*.log 2>/dev/null || tail -20 /proc/\$(pgrep -f train_cma)/fd/1 2>/dev/null'"
done < "$SERVER_FILE"
echo ""
echo "  Collect when done:"
echo "    bash collect_multiseed.sh"
echo "═══════════════════════════════════════════════════"
