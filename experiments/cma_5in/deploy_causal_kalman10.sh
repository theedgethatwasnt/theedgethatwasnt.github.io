#!/bin/bash
# Deploy CMA-NN 12-pair grid on CAUSAL features (kalman10 smoother).
#
# Uses:
#   - FXFeatureBuilder with smoother=kalman10 (causal, no lookahead, true MACD histogram)
#   - 7→8(sin)→3 architecture
#   - V3+macd_hist inputs
#   - 4 seeds (42, 137, 23, 99) for multi-seed robustness
#
# 4 Hetzner servers × 12 pairs × 4 seeds = 48 runs per server, ~3h.
# Cost: 4 × ccx23 × 3h × $0.07/hr ≈ $0.84
set -e

GENS=200
POPSIZE=24
SEEDS="42 137 23 99"
SMOOTHER=kalman10
EXTRAS=macd_hist

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
CMA_DIR="$PROJECT_DIR/research/experiments/cma_5in"
RESULTS_DIR="$CMA_DIR/results"
SERVER_FILE="$CMA_DIR/causal_kalman10_servers.txt"

ALL_PAIRS="EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════"
echo "  CMA-NN CAUSAL KALMAN10 × 12 Pairs × 4 Seeds"
echo "  Smoother: $SMOOTHER (Kalman q=1.0, replaces SMA5)"
echo "  Features: mc_d_a, mc_dd_a, er_norm, macd_hist (causal)"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Create 4 servers ──
> "$SERVER_FILE"
for i in 1 2 3 4; do
    NAME="cma-causal-$i"
    if hcloud server describe "$NAME" > /dev/null 2>&1; then
        IP=$(hcloud server ip "$NAME")
        echo "  $NAME exists → $IP"
    else
        hcloud server create --name "$NAME" --type ccx23 --image ubuntu-24.04 \
            --location hel1 --ssh-key "user@host" --without-ipv6 2>/dev/null || \
        hcloud server create --name "$NAME" --type cax31 --image ubuntu-24.04 \
            --location hel1 --ssh-key "user@host" --without-ipv6
        IP=$(hcloud server ip "$NAME")
        echo "  $NAME → $IP"
    fi
    echo "$i $IP" >> "$SERVER_FILE"
done

sleep 25
ssh-keygen -R "$(cat $SERVER_FILE | awk '{print $2}' | head -1)" 2>/dev/null || true

# ── Step 2: Setup servers ──
while read IDX IP; do
    echo "Setup $IDX ($IP)..."
    ssh-keygen -R $IP 2>/dev/null || true
    ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP' 2>&1 | tail -3
set -e
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba pyarrow cma
mkdir -p /root/fx-core/lib /root/fx-core/research/experiments/cma_5in/results /root/fx-core/data/m5_ohlc
SETUP
    scp -o StrictHostKeyChecking=no -q $PROJECT_DIR/lib/fast_eval.py $PROJECT_DIR/lib/asi_indicator.py $PROJECT_DIR/lib/incremental_features.py root@$IP:/root/fx-core/lib/
    scp -o StrictHostKeyChecking=no -q $CMA_DIR/train_cma_v2.py $CMA_DIR/extra_indicators.py root@$IP:/root/fx-core/research/experiments/cma_5in/
    # Send only the kalman10 causal parquets (smaller than full OHLC)
    rsync -az $PROJECT_DIR/data/m5_ohlc/*kalman10*causal.parquet root@$IP:/root/fx-core/data/m5_ohlc/
    echo "  Server $IDX ready"
done < "$SERVER_FILE"

# Assign seeds to servers (each server takes 1 seed × all 12 pairs)
declare -A SERVER_SEED
SERVER_SEED[1]=42
SERVER_SEED[2]=137
SERVER_SEED[3]=23
SERVER_SEED[4]=99

# ── Step 3: Launch training ──
while read IDX IP; do
    SEED=${SERVER_SEED[$IDX]}
    echo "Launch S$IDX ($IP): seed=$SEED, 12 pairs"
    ssh -o StrictHostKeyChecking=no root@$IP bash <<TRAIN &
set -e
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/cma_5in

for PAIR in $ALL_PAIRS; do
    echo "\$(date) Training \$PAIR seed $SEED kalman10"
    PYTHONUNBUFFERED=1 python3 train_cma_v2.py \\
        --pair \$PAIR --seed $SEED --gens $GENS \\
        --features v3_plus --extras $EXTRAS \\
        --fixed-activation sin --popsize $POPSIZE \\
        --workers 4 --label causal_k10 \\
        --causal-parquet $SMOOTHER 2>&1 | tail -4
done
echo "\$(date) S$IDX DONE"
TRAIN
done < "$SERVER_FILE"

echo ""
echo "═════════════════════════════════════════════"
echo "  All 4 servers launched!"
echo "  Monitor: ssh root@\$IP 'ls /root/fx-core/research/experiments/cma_5in/results/causal_k10*.pkl | wc -l'"
echo "  Collect:  bash collect_causal_kalman10.sh"
echo "═════════════════════════════════════════════"
