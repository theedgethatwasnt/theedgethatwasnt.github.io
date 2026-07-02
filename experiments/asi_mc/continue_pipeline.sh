#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Continue Pipeline: Wait for AUD_USD + EUR_USD → collect →
# delete → fire GBP_JPY + GBP_USD → wait → collect → delete →
# fire NZD_JPY + NZD_USD → wait → collect → delete → DONE
# Also waits for Free NEAT 6in → collect → delete
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG="$SCRIPT_DIR/continue_pipeline.log"
SEED=42; GENS=200; PRETRAIN_GENS=50; SINE_GENS=30; ISLANDS=4; POP=150; MAX_HOLD=200

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_pair() {
    local IP="$1" PAIR="$2"
    while true; do
        STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
            "tail -1 /root/neat/results/ironnet/iron_v3_${PAIR}_s42.log 2>/dev/null" 2>/dev/null || echo "")
        if echo "$STATUS" | grep -q "^Total:"; then
            log "  $PAIR DONE: $STATUS"; return 0
        fi
        if echo "$STATUS" | grep -q "Traceback\|Error\b"; then
            log "  $PAIR ERROR: $STATUS"; return 1
        fi
        log "  $PAIR: $STATUS"
        sleep 120
    done
}

setup_and_launch() {
    local PAIR="$1" IP="$2"
    log "  [$PAIR] Setting up $IP..."
    ssh -o StrictHostKeyChecking=no root@$IP \
        'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
         python3 -m venv /root/venv && source /root/venv/bin/activate && \
         pip install -q neat-python numba pandas pyarrow numpy requests && \
         mkdir -p /root/neat/data /root/neat/results/ironnet /root/neat/lib' 2>/dev/null
    scp -o StrictHostKeyChecking=no "$DATA_DIR/${PAIR}_asi_mc.parquet" root@$IP:/root/neat/data/
    rsync -az -e "ssh -o StrictHostKeyChecking=no" "$SCRIPT_DIR/train_ironnet_perpair.py" root@$IP:/root/neat/
    rsync -az -e "ssh -o StrictHostKeyChecking=no" "$SCRIPT_DIR/neat_config_4in_3out.ini" root@$IP:/root/neat/
    rsync -az -e "ssh -o StrictHostKeyChecking=no" "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
    rsync -az -e "ssh -o StrictHostKeyChecking=no" "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
    ssh -o StrictHostKeyChecking=no root@$IP 'touch /root/neat/lib/__init__.py'
    ssh -o StrictHostKeyChecking=no root@$IP "
        source /root/venv/bin/activate && cd /root/neat
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 \
        nohup python3 train_ironnet_perpair.py \
            --pair $PAIR --seed $SEED --sine-gens $SINE_GENS \
            --pretrain-gens $PRETRAIN_GENS --gens $GENS \
            --islands $ISLANDS --pop $POP --max-hold $MAX_HOLD \
        > results/ironnet/iron_v3_${PAIR}_s42.log 2>&1 &
        disown; echo launched
    " 2>/dev/null
    log "  [$PAIR] Launched on $IP"
}

collect_pair() {
    local PAIR="$1" IP="$2"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_best.pkl" \
        "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  $PAIR pkl: OK" || log "  $PAIR pkl: FAIL"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_result.json" \
        "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  $PAIR json: OK" || log "  $PAIR json: FAIL"
}

mkdir -p "$RESULTS_DIR/ironnet" "$RESULTS_DIR/free_neat_6in"

# ══════════════════════════════════════════════════════════════
# STAGE A: Wait for AUD_USD + EUR_USD
# ══════════════════════════════════════════════════════════════
log "═══ STAGE A: Waiting for AUD_USD + EUR_USD ═══"
AUD_IP="<SERVER_IP_14>"
EUR_IP="<SERVER_IP_4>"

wait_pair "$AUD_IP" "AUD_USD" &
wait_pair "$EUR_IP" "EUR_USD" &
wait

log "Collecting AUD_USD + EUR_USD..."
collect_pair "AUD_USD" "$AUD_IP"
collect_pair "EUR_USD" "$EUR_IP"

log "Deleting AUD_USD + EUR_USD servers..."
hcloud server delete ironnet-pp-aud-usd 2>/dev/null && log "  Deleted aud-usd" || log "  Already gone"
hcloud server delete ironnet-pp-eur-usd 2>/dev/null && log "  Deleted eur-usd" || log "  Already gone"

# ══════════════════════════════════════════════════════════════
# STAGE B: Fire GBP_JPY + GBP_USD
# ══════════════════════════════════════════════════════════════
log "═══ STAGE B: Fire GBP_JPY + GBP_USD ═══"
for PAIR in GBP_JPY GBP_USD; do
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        log "  hel1 full, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    ssh-keygen -f '/path/to/.ssh/known_hosts' -R "$IP" 2>/dev/null || true
    log "  $NAME: $IP"
    eval "IP_${PAIR//_/}=$IP"
done

sleep 30  # brief wait for SSH daemon

setup_and_launch "GBP_JPY" "$IP_GBPJPY" &
setup_and_launch "GBP_USD" "$IP_GBPUSD" &
wait

log "═══ STAGE B2: Wait for GBP_JPY + GBP_USD ═══"
wait_pair "$IP_GBPJPY" "GBP_JPY" &
wait_pair "$IP_GBPUSD" "GBP_USD" &
wait

log "Collecting GBP_JPY + GBP_USD..."
collect_pair "GBP_JPY" "$IP_GBPJPY"
collect_pair "GBP_USD" "$IP_GBPUSD"

log "Deleting GBP servers..."
hcloud server delete ironnet-pp-gbp-jpy 2>/dev/null || true
hcloud server delete ironnet-pp-gbp-usd 2>/dev/null || true

# ══════════════════════════════════════════════════════════════
# STAGE C: Fire NZD_JPY + NZD_USD
# ══════════════════════════════════════════════════════════════
log "═══ STAGE C: Fire NZD_JPY + NZD_USD ═══"
for PAIR in NZD_JPY NZD_USD; do
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        log "  hel1 full, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    ssh-keygen -f '/path/to/.ssh/known_hosts' -R "$IP" 2>/dev/null || true
    log "  $NAME: $IP"
    eval "IP_${PAIR//_/}=$IP"
done

sleep 30

setup_and_launch "NZD_JPY" "$IP_NZDJPY" &
setup_and_launch "NZD_USD" "$IP_NZDUSD" &
wait

log "═══ STAGE C2: Wait for NZD_JPY + NZD_USD ═══"
wait_pair "$IP_NZDJPY" "NZD_JPY" &
wait_pair "$IP_NZDUSD" "NZD_USD" &
wait

log "Collecting NZD_JPY + NZD_USD..."
collect_pair "NZD_JPY" "$IP_NZDJPY"
collect_pair "NZD_USD" "$IP_NZDUSD"

log "Deleting NZD servers..."
hcloud server delete ironnet-pp-nzd-jpy 2>/dev/null || true
hcloud server delete ironnet-pp-nzd-usd 2>/dev/null || true

# ══════════════════════════════════════════════════════════════
# STAGE D: Wait for Free NEAT 6in → collect → delete
# ══════════════════════════════════════════════════════════════
log "═══ STAGE D: Waiting for Free NEAT 6in ═══"
FN6_IPS=(<SERVER_IP_2> <SERVER_IP_3>)
FN6_RUNS=("EUR_GBP:0:42" "CAD_JPY:1:137")

wait_fn6() {
    local PAIR="$1" IP="$2" SEED="$3"
    while true; do
        STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
            "tail -1 /root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED}.log 2>/dev/null" 2>/dev/null || echo "")
        if echo "$STATUS" | grep -q "^Total:"; then
            log "  Free6in $PAIR DONE: $STATUS"; return 0
        fi
        log "  Free6in $PAIR: $STATUS"
        sleep 300
    done
}

wait_fn6 "EUR_GBP" "${FN6_IPS[0]}" "42" &
wait_fn6 "CAD_JPY" "${FN6_IPS[1]}" "137" &
wait

log "Collecting Free NEAT 6in..."
for ENTRY in "EUR_GBP:0:42" "CAD_JPY:1:137"; do
    IFS=: read -r PAIR IDX SEED <<< "$ENTRY"
    IP="${FN6_IPS[$IDX]}"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED}_best.pkl" \
        "$RESULTS_DIR/free_neat_6in/" 2>/dev/null && log "  $PAIR pkl: OK" || log "  $PAIR pkl: FAIL"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED}_result.json" \
        "$RESULTS_DIR/free_neat_6in/" 2>/dev/null && log "  $PAIR json: OK" || log "  $PAIR json: FAIL"
done

log "Deleting Free NEAT 6in servers..."
hcloud server delete free-neat-6in-1 2>/dev/null || true
hcloud server delete free-neat-6in-2 2>/dev/null || true

log "═══ ALL STAGES COMPLETE ═══"
hcloud server list
