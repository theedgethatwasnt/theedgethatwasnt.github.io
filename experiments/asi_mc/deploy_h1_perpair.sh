#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# IronNet V3 H1 — Per-Pair Training on Hetzner (all 12 pairs)
#
# Architecture: 4→4→3 fixed (40 conn), activations {tanh, sin, cos, gauss}
# Inputs: MC_D, MC_dD, ER_norm, UPnL  (M5-computed, resampled to H1 via .last())
# Fitness: pips/day, worst WF chunk
#
# 3 servers × 4 pairs each, seed 42
# H1 data: ~13.6K bars per pair → fast training (~30min/pair)
# Expected runtime: ~2h per server
# Cost: 3 × ccx23 × 2h × $0.07/hr ≈ $0.42
# ═══════════════════════════════════════════════════════════════

set -e

GENS=200
PRETRAIN_GENS=50
SINE_GENS=30
POP=150
ISLANDS=4
STALL=60
SEED=42

# 3 servers, 4 pairs each
declare -a SERVER_PAIRS
SERVER_PAIRS[1]="EUR_JPY USD_JPY GBP_JPY AUD_JPY"
SERVER_PAIRS[2]="CAD_JPY CHF_JPY NZD_JPY EUR_USD"
SERVER_PAIRS[3]="GBP_USD AUD_USD NZD_USD EUR_GBP"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
RESULTS_DIR="$PROJECT_DIR/research/experiments/asi_mc/results/ironnet_h1"
SERVER_FILE="$PROJECT_DIR/research/experiments/asi_mc/h1_servers.txt"
DATA_DIR="$PROJECT_DIR/data/asi_mc_indicators"

mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════"
echo "  IronNet V3 H1 — 12 Pairs, 3 Servers"
echo "  Gens: sine=$SINE_GENS + zz=$PRETRAIN_GENS + WF=$GENS"
echo "  Pop: $ISLANDS islands × $POP"
echo "═══════════════════════════════════════════"

# ── Step 1: Create servers ──
echo ""
echo "Creating 3 cx53 servers..."
> "$SERVER_FILE"

for i in 1 2 3; do
    NAME="ironnet-h1-$i"
    echo "  Creating $NAME..."
    hcloud server create --name "$NAME" --type ccx23 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" --without-ipv6 2>/dev/null
    IP=$(hcloud server ip "$NAME")
    echo "$i $IP" >> "$SERVER_FILE"
    echo "  $NAME → $IP"
done

echo ""
echo "Waiting 30s for servers to boot..."
sleep 30

# ── Step 2: Setup each server ──
while read IDX IP; do
    echo ""
    echo "═══ Setting up server $IDX ($IP) ═══"

    ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP'
set -e
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba neat-python pyarrow
mkdir -p /root/fx-core/lib /root/fx-core/research/experiments/asi_mc/results/ironnet_h1 /root/fx-core/data/asi_mc_indicators
SETUP

    # Copy code
    scp -q "$PROJECT_DIR/lib/fast_eval.py" root@$IP:/root/fx-core/lib/
    scp -q "$PROJECT_DIR/lib/asi_indicator.py" root@$IP:/root/fx-core/lib/
    scp -q "$PROJECT_DIR/research/experiments/asi_mc/train_ironnet_perpair.py" root@$IP:/root/fx-core/research/experiments/asi_mc/
    scp -q "$PROJECT_DIR/research/experiments/asi_mc/neat_config_4in_3out.ini" root@$IP:/root/fx-core/research/experiments/asi_mc/

    # Copy parquet data (117MB, fast over network)
    rsync -az "$DATA_DIR/" root@$IP:/root/fx-core/data/asi_mc_indicators/

    echo "  Setup complete"
done < "$SERVER_FILE"

# ── Step 3: Launch training ──
echo ""
echo "Launching training on all servers..."

while read IDX IP; do
    PAIRS="${SERVER_PAIRS[$IDX]}"
    echo "  Server $IDX ($IP): $PAIRS"

    # Run training in background via nohup
    ssh root@$IP bash <<TRAIN &
set -e
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/asi_mc

for PAIR in $PAIRS; do
    echo "[\$(date)] Starting \$PAIR..."
    PYTHONUNBUFFERED=1 python3 train_ironnet_perpair.py \\
        --pair \$PAIR --seed $SEED --tf H1 \\
        --gens $GENS --pretrain-gens $PRETRAIN_GENS --sine-gens $SINE_GENS \\
        --pop $POP --islands $ISLANDS --stall-limit $STALL \\
        > results/ironnet_h1/\${PAIR}_s${SEED}.log 2>&1
    echo "[\$(date)] Done \$PAIR"
done
echo "[\$(date)] ALL DONE on server $IDX"
TRAIN

done < "$SERVER_FILE"

echo ""
echo "═══════════════════════════════════════════"
echo "  All training launched!"
echo ""
echo "  Monitor:"
cat "$SERVER_FILE" | while read IDX IP; do
    echo "    ssh root@$IP 'tail -f /root/fx-core/research/experiments/asi_mc/results/ironnet_h1/*.log'"
done
echo ""
echo "  Collect when done:"
echo "    bash collect_h1_perpair.sh"
echo "═══════════════════════════════════════════"
