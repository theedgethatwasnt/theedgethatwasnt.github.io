#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SwingDim Experiment — Deploy to Hetzner
#
# Tests swing structure indicators vs IronNet V3 baseline:
#   E1 (6-in): MC_D, MC_dD, ER_norm, SB_A, ERP_P, UPnL
#   E2 (7-in): MC_D, MC_dD, ER_norm, SB_A, HH_ASI, HL_ASI, UPnL
#   E3 (8-in): MC_D, MC_dD, ER_norm, ERP_A, HH_ASI, HL_ASI, ERP_P, UPnL
#
# Deployment plan (4 servers):
#   Server 1 — E1, EUR_GBP + CAD_JPY, seed 42   (fast, 2 pairs each)
#   Server 2 — E2, EUR_GBP + CAD_JPY, seed 42
#   Server 3 — E3, EUR_GBP + CAD_JPY, seed 42
#   Server 4 — V3 control, EUR_GBP + CAD_JPY, seed 137 (rerun for clean comparison)
#
# Why EUR_GBP + CAD_JPY?
#   V3 baselines: EUR_GBP=14.3p/day, CAD_JPY=41.6p/day
#   High-contrast pair: if swing dims help EUR_GBP, they add real signal
#   CAD_JPY is momentum-dominant — tests if swing context hurts it
#
# Step 1 (FIRST): run export_swing_indicators.py on Hetzner data node
#   (or attach neat-data volume and run locally)
# Step 2: this script — launch training
# Step 3: collect results, compare vs V3 baselines
#
# Prereqs:
#   - neat-data volume attached and mounted (or S5 parquets already in DATA_DIR)
#   - export_swing_indicators.py already run → swing columns exist in parquets
#   - hcloud CLI configured
#
# Cost: 4 × cx53 × ~4h × $0.10/hr = ~$1.60
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"

GENS=200
PRETRAIN_GENS=50
SINE_GENS=30
POP=150
ISLANDS=4
MAX_HOLD=200

# Pairs to test per server (one --pair per run; run sequentially per server)
PAIRS="EUR_GBP CAD_JPY"

# Runs: "MODE:SEED"
declare -A SERVER_MODE
SERVER_MODE[1]="E1:42"
SERVER_MODE[2]="E2:42"
SERVER_MODE[3]="E3:42"
SERVER_MODE[4]="V3:137"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "SwingDim Experiment: E1/E2/E3 vs V3 on EUR_GBP+CAD_JPY"
echo "4 servers × 2 pairs × 280 gens = ~4h each"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Verify parquets have swing columns ────────────────────────
echo "Checking parquet columns..."
python3 - <<'PYEOF'
import pandas as pd, sys
from pathlib import Path
DATA = Path("data/asi_mc_indicators")
REQUIRED = ["sb_a", "erp_a", "hh_asi", "hl_asi", "erp_p", "hh_price", "hl_price"]
missing_any = False
for pair in ["EUR_GBP", "CAD_JPY"]:
    p = DATA / f"{pair}_asi_mc.parquet"
    if not p.exists():
        print(f"  MISSING: {p}")
        missing_any = True
        continue
    df = pd.read_parquet(p, engine="pyarrow")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"  {pair}: MISSING columns {missing}")
        print(f"  Run first: bash research/experiments/asi_mc/deploy_export_swing.sh")
        missing_any = True
    else:
        print(f"  {pair}: OK ({len(df)} bars, {len(df.columns)} cols)")
if missing_any:
    sys.exit(1)
PYEOF
echo "Parquets OK"

# hcloud -o json sends "Waiting for..." lines to stdout before JSON — extract JSON line
_create_server() {
    local NAME=$1 LOC=$2
    hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location "$LOC" --ssh-key "user@host" > /dev/null 2>&1 || return 0
    hcloud server describe "$NAME" -o json 2>/dev/null | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['public_net']['ipv4']['ip'])" 2>/dev/null || true
}

# ── Phase 1: Create 4 servers ──────────────────────────────────
echo ""
echo "Creating 4 Hetzner servers..."
declare -A SERVER_IPS
for i in 1 2 3 4; do
    NAME="swingdim-$i"
    IP=$(_create_server "$NAME" "hel1")
    if [ -z "$IP" ]; then
        echo "  hel1 full, trying nbg1..."
        IP=$(_create_server "$NAME" "nbg1")
    fi
    echo "  swingdim-$i: $IP"
    SERVER_IPS[$i]="$IP"
done

ALL_IPS="${SERVER_IPS[1]} ${SERVER_IPS[2]} ${SERVER_IPS[3]} ${SERVER_IPS[4]}"
echo "$ALL_IPS" > "$SCRIPT_DIR/swingdim_servers.txt"

# ── Phase 2: Wait for SSH ──────────────────────────────────────
echo ""
echo "Waiting for SSH on all servers..."
for IP in $ALL_IPS; do
    (
        for attempt in $(seq 1 30); do
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@"$IP" 'echo ok' 2>/dev/null && break
            sleep 3
        done
        echo "  $IP: ready"
    ) &
done
wait
echo "All servers reachable"

# ── Phase 3: Setup servers in parallel ─────────────────────────
echo ""
echo "Setting up servers (pip install + file transfer)..."

for IP in $ALL_IPS; do
    (
        # Install dependencies
        ssh -o StrictHostKeyChecking=no root@"$IP" \
            'apt-get update -qq > /dev/null && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/swingdim /root/neat/lib /root/neat/research/experiments/asi_mc' 2>/dev/null

        # Transfer parquets (only the 2 pairs we need)
        for PAIR in EUR_GBP CAD_JPY; do
            rsync -az "$DATA_DIR/${PAIR}_asi_mc.parquet" root@"$IP":/root/neat/data/ 2>/dev/null || true
        done

        # Transfer code
        rsync -az "$SCRIPT_DIR/train_swingdim.py"             root@"$IP":/root/neat/
        rsync -az "$REPO_DIR/lib/fast_eval.py"                root@"$IP":/root/neat/lib/
        rsync -az "$REPO_DIR/lib/asi_indicator.py"            root@"$IP":/root/neat/lib/
        rsync -az "$REPO_DIR/lib/swing_indicators.py"         root@"$IP":/root/neat/lib/
        rsync -az "$REPO_DIR/lib/pair_config.py"              root@"$IP":/root/neat/lib/ 2>/dev/null || true

        # Create __init__.py
        ssh root@"$IP" 'touch /root/neat/lib/__init__.py'

        # Transfer any existing neat config files for reference
        for CFG in "$SCRIPT_DIR"/neat_config_*in_3out.ini; do
            [ -f "$CFG" ] && rsync -az "$CFG" root@"$IP":/root/neat/ || true
        done

        echo "  [$IP] Setup complete"
    ) &
done
wait
echo "All servers ready!"

# ── Phase 4: Launch training ────────────────────────────────────
echo ""
echo "Launching training runs..."

TG_ENV=""
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
    TG_ENV="TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
fi

for i in 1 2 3 4; do
    IFS=: read -r MODE SEED <<< "${SERVER_MODE[$i]}"
    IP="${SERVER_IPS[$i]}"
    LOG_PREFIX="results/swingdim/swingdim_${MODE}_s${SEED}"

    echo "  Server $i ($IP): mode=$MODE seed=$SEED pairs=[EUR_GBP, CAD_JPY]"

    # Run pairs sequentially on each server (EUR_GBP then CAD_JPY)
    ssh root@"$IP" "source /root/venv/bin/activate && cd /root/neat && \
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 $TG_ENV \
        nohup bash -c '
            python3 -u train_swingdim.py \
                --mode $MODE --pair EUR_GBP --seed $SEED \
                --sine-gens $SINE_GENS \
                --pretrain-gens $PRETRAIN_GENS \
                --gens $GENS \
                --islands $ISLANDS \
                --pop $POP \
                --max-hold $MAX_HOLD \
                > ${LOG_PREFIX}_EUR_GBP.log 2>&1
            python3 -u train_swingdim.py \
                --mode $MODE --pair CAD_JPY --seed $SEED \
                --sine-gens $SINE_GENS \
                --pretrain-gens $PRETRAIN_GENS \
                --gens $GENS \
                --islands $ISLANDS \
                --pop $POP \
                --max-hold $MAX_HOLD \
                > ${LOG_PREFIX}_CAD_JPY.log 2>&1
        ' &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "4 SwingDim experiments launched!"
echo ""
echo "Monitor training:"
for i in 1 2 3 4; do
    IFS=: read -r MODE SEED <<< "${SERVER_MODE[$i]}"
    IP="${SERVER_IPS[$i]}"
    echo "  $MODE s$SEED ($IP):"
    echo "    ssh root@$IP 'tail -f /root/neat/results/swingdim/swingdim_${MODE}_s${SEED}_EUR_GBP.log'"
done
echo ""
echo "Collect results (after ~4h per server):"
cat << 'COLLECT_EOF'
  mkdir -p research/experiments/asi_mc/results/swingdim
  for IP in $(cat research/experiments/asi_mc/swingdim_servers.txt); do
    scp "root@$IP:/root/neat/results/swingdim/*_best.pkl" \
        "research/experiments/asi_mc/results/swingdim/" 2>/dev/null || true
    scp "root@$IP:/root/neat/results/swingdim/*_results.json" \
        "research/experiments/asi_mc/results/swingdim/" 2>/dev/null || true
    scp "root@$IP:/root/neat/results/swingdim/*/summary.json" \
        "research/experiments/asi_mc/results/swingdim/" 2>/dev/null || true
  done
COLLECT_EOF
echo ""
echo "Compare results:"
cat << 'COMPARE_EOF'
  python3 - <<'PYEOF'
import json
from pathlib import Path
RES = Path("research/experiments/asi_mc/results/swingdim")
V3_BASELINE = {"EUR_GBP": 14.3, "CAD_JPY": 41.6}
print(f"{'Mode':<6} {'Pair':<10} {'p/day':>8} {'vs V3':>8} {'Trades':>7} {'Fitness':>10}")
print("-" * 55)
for f in sorted(RES.glob("*_results.json")):
    d = json.load(open(f))
    for r in d:
        pair = r["pair"]
        ppd = r["oos"]["pips_per_day"]
        delta = ppd - V3_BASELINE.get(pair, 0)
        sign = "+" if delta >= 0 else ""
        print(f"  {r['mode']:<4}  {pair:<10}  {ppd:>7.1f}  {sign}{delta:>7.1f}  "
              f"{r['oos']['n_trades']:>7}  {r['fitness']:>10.4f}")
PYEOF
COMPARE_EOF
echo ""
echo "Cleanup (AFTER collecting results):"
echo "  for i in 1 2 3 4; do hcloud server delete swingdim-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"

# Save server list for later reference
echo ""
echo "Server IPs saved to: $SCRIPT_DIR/swingdim_servers.txt"
hcloud server list | grep swingdim
