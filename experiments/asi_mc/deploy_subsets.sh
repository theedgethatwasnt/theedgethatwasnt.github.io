#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Subset Experiment — Deploy to Hetzner
#
# 4 feature subsets, each 4 features + UPnL = 5 inputs
# Small NN: 5→5→10→3→3 (114 conn), fast convergence
# Fitness: pips/day, worst WF chunk (no MAE)
#
#   S1 (server 1): mc_d_a, mc_dd_a, er_norm, sb_a         ← momentum + breakout
#   S2 (server 2): sb_a, hh_asi, hl_asi, er_norm           ← swing HHHL structure
#   S3 (server 3): erp_p, erp_a, mc_d_a, er_norm           ← price/ASI range position
#   S4 (server 4): d_erp_p, d_erp_a, mc_dd_a, mc_d_a      ← velocity/acceleration
#
# Each server: EUR_GBP → CAD_JPY sequentially, seeds 42 + 137
# Quick params: 2 islands × 75 pop, sine=15g, zz=25g, WF=80g
# Expected runtime: ~1.5h per pair → ~3h total per server
# Cost: 4 × cx53 × 3h × $0.10/hr ≈ $1.20
#
# Prereqs:
#   - swing columns already in parquets (export_swing_indicators.py done)
#   - hcloud CLI configured with API key
# ═══════════════════════════════════════════════════════════════

set -e

GENS=80
PRETRAIN_GENS=25
SINE_GENS=15
POP=75
ISLANDS=2
MAX_HOLD=200
STALL=25
PAIRS="EUR_GBP CAD_JPY"
SEEDS="42 137"

# Server → mode mapping
declare -A SERVER_MODE
SERVER_MODE[1]="S1"
SERVER_MODE[2]="S2"
SERVER_MODE[3]="S3"
SERVER_MODE[4]="S4"

# Existing servers from swingdim_servers.txt
SERVERS=(<SERVER_IP_10> <SERVER_IP_11> <SERVER_IP_12> <SERVER_IP_13>)

TGTOKEN="${TELEGRAM_BOT_TOKEN:-}"
TGCHAT="${TELEGRAM_CHAT_ID:-}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Subset Experiment S1/S2/S3/S4 — 4 features each"
echo "2 islands × 75 pop | sine=${SINE_GENS}g zz=${PRETRAIN_GENS}g WF=${GENS}g"
echo "Pairs: $PAIRS | Seeds: $SEEDS"
echo "═══════════════════════════════════════════════════════"
echo ""

# Kill any running training on all servers first
echo ">>> Killing existing training processes..."
for ip in "${SERVERS[@]}"; do
    ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no root@$ip \
        'pkill -f train_swingdim.py 2>/dev/null; echo "cleared $HOSTNAME"' 2>/dev/null || true
done
sleep 3

# Sync updated train_swingdim.py to all servers
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo ">>> Syncing train_swingdim.py to all servers..."
for ip in "${SERVERS[@]}"; do
    scp -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/train_swingdim.py" \
        root@$ip:/root/neat/train_swingdim.py &
done
wait
echo "    Done."

# Launch on each server
for idx in 1 2 3 4; do
    ip="${SERVERS[$((idx-1))]}"
    mode="${SERVER_MODE[$idx]}"

    echo ""
    echo ">>> Server $idx ($ip) — mode=$mode"

    # Build sequential command: EUR_GBP s42, EUR_GBP s137, CAD_JPY s42, CAD_JPY s137
    CMD="cd /root/neat && source /root/venv/bin/activate && mkdir -p results/subsets"
    for pair in $PAIRS; do
        for seed in $SEEDS; do
            logfile="results/subsets/${mode}_${pair}_s${seed}.log"
            CMD+=" && echo '--- Starting ${mode} ${pair} s${seed} ---'"
            CMD+=" && ASI_MC_DATA_DIR=/root/neat/data"
            CMD+=" TELEGRAM_BOT_TOKEN=${TGTOKEN}"
            CMD+=" TELEGRAM_CHAT_ID=${TGCHAT}"
            CMD+=" PYTHONUNBUFFERED=1"
            CMD+=" python3 -u train_swingdim.py"
            CMD+=" --mode $mode"
            CMD+=" --pair $pair"
            CMD+=" --seed $seed"
            CMD+=" --islands $ISLANDS"
            CMD+=" --pop $POP"
            CMD+=" --sine-gens $SINE_GENS"
            CMD+=" --pretrain-gens $PRETRAIN_GENS"
            CMD+=" --gens $GENS"
            CMD+=" --max-hold $MAX_HOLD"
            CMD+=" --stall-limit $STALL"
            CMD+=" --wavelet"
            CMD+=" > $logfile 2>&1"
            CMD+=" && echo '--- Done ${mode} ${pair} s${seed} ---'"
        done
    done

    ssh -o StrictHostKeyChecking=no root@$ip \
        "nohup bash -c '$CMD' > /root/neat/results/subsets/${mode}_master.log 2>&1 &"
    echo "    Launched (PID will appear in master log)"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "All 4 servers launched."
echo ""
echo "Monitor:"
for idx in 1 2 3 4; do
    ip="${SERVERS[$((idx-1))]}"
    mode="${SERVER_MODE[$idx]}"
    echo "  S${idx} ($mode): ssh root@$ip 'tail -f /root/neat/results/subsets/${mode}_master.log'"
done
echo ""
echo "Collect results (~3h):"
echo "  bash collect_subsets.sh"
echo "═══════════════════════════════════════════════════════"
