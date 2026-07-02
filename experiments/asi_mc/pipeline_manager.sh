#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Experiment Pipeline Manager — auto collect→delete→fire
#
# Stage 1: Wait for USD_JPY → collect → delete → fire batch 2
# Stage 2: Wait for Free NEAT v2 (4in) → collect → delete → fire Free NEAT 6in
# Stage 3: Wait for batch 2 → collect → delete
# Stage 4: Wait for Free NEAT 6in → collect → delete → DONE
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG="$SCRIPT_DIR/pipeline.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_for_done() {
    local IP="$1" LOG_PATH="$2" LABEL="$3"
    log "Waiting for $LABEL ($IP)..."
    while true; do
        STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
            "tail -1 $LOG_PATH 2>/dev/null" 2>/dev/null || echo "")
        if echo "$STATUS" | grep -q "^Total:"; then
            log "$LABEL DONE: $STATUS"
            return 0
        fi
        if echo "$STATUS" | grep -q "Traceback\|Error\|error"; then
            log "$LABEL ERROR: $STATUS"
            return 1
        fi
        log "  $LABEL: $STATUS"
        sleep 120
    done
}

wait_for_all_done() {
    # Wait until all servers show "Total:" in their log
    local SERVERS_FILE="$1"
    local LOG_SUBDIR="$2"
    local LABEL_PREFIX="$3"
    log "Waiting for all $LABEL_PREFIX runs..."
    while true; do
        ALL_DONE=1
        while IFS=' ' read -r PAIR IP; do
            LOG_PATH="/root/neat/results/${LOG_SUBDIR}/iron_v3_${PAIR}_s42.log"
            STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
                "tail -1 $LOG_PATH 2>/dev/null" 2>/dev/null || echo "SSH_FAIL")
            if echo "$STATUS" | grep -q "^Total:"; then
                log "  $PAIR: DONE"
            else
                log "  $PAIR ($IP): $STATUS"
                ALL_DONE=0
            fi
        done < "$SERVERS_FILE"
        if [ "$ALL_DONE" -eq 1 ]; then
            log "All $LABEL_PREFIX done!"
            return 0
        fi
        sleep 120
    done
}

mkdir -p "$RESULTS_DIR/ironnet" "$RESULTS_DIR/free_neat" "$RESULTS_DIR/free_neat_6in"

# ══════════════════════════════════════════════════════════════
# STAGE 1: USD_JPY per-pair
# ══════════════════════════════════════════════════════════════
log "═══ STAGE 1: USD_JPY ═══"
USD_IP="<SERVER_IP_14>"
USD_LOG="/root/neat/results/ironnet/iron_v3_USD_JPY_s42.log"

wait_for_done "$USD_IP" "$USD_LOG" "USD_JPY" || true

log "Collecting USD_JPY..."
scp -o StrictHostKeyChecking=no \
    "root@${USD_IP}:/root/neat/results/ironnet/iron_v3_USD_JPY_s42_best.pkl" \
    "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  pkl: OK" || log "  pkl: FAIL"
scp -o StrictHostKeyChecking=no \
    "root@${USD_IP}:/root/neat/results/ironnet/iron_v3_USD_JPY_s42_result.json" \
    "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  json: OK" || log "  json: FAIL"

log "Deleting ironnet-pp-usd-jpy..."
hcloud server delete ironnet-pp-usd-jpy 2>/dev/null && log "  Deleted" || log "  Already gone"

# ══════════════════════════════════════════════════════════════
# STAGE 2: Fire batch 2 (6 remaining pairs)
# ══════════════════════════════════════════════════════════════
log "═══ STAGE 2: Fire batch 2 ═══"
BATCH2_PAIRS=(AUD_USD EUR_USD GBP_JPY GBP_USD NZD_JPY NZD_USD)
SEED=42; GENS=200; PRETRAIN_GENS=50; SINE_GENS=30; ISLANDS=4; POP=150; MAX_HOLD=200

BATCH2_SERVERS=()
for PAIR in "${BATCH2_PAIRS[@]}"; do
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        log "  hel1 full for $PAIR, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    ssh-keygen -f '/path/to/.ssh/known_hosts' -R "$IP" 2>/dev/null || true
    log "  $NAME: $IP"
    BATCH2_SERVERS+=("$PAIR $IP")
done

BATCH2_FILE="$SCRIPT_DIR/perpair_batch2_servers.txt"
printf '%s\n' "${BATCH2_SERVERS[@]}" > "$BATCH2_FILE"
log "Batch 2 servers written to $BATCH2_FILE"

# Wait for SSH
log "Waiting for batch 2 SSH..."
while IFS=' ' read -r PAIR IP; do
    (for a in $(seq 1 40); do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@$IP 'echo ok' 2>/dev/null && break
        sleep 3
    done; log "  $PAIR ($IP): SSH ready") &
done < "$BATCH2_FILE"
wait

# Setup batch 2 servers
log "Setting up batch 2..."
while IFS=' ' read -r PAIR IP; do
    (
        ssh -o StrictHostKeyChecking=no root@$IP \
            'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && \
             python3 -m venv /root/venv && source /root/venv/bin/activate && \
             pip install -q neat-python numba pandas pyarrow numpy requests && \
             mkdir -p /root/neat/data /root/neat/results/ironnet /root/neat/lib' 2>/dev/null
        scp -o StrictHostKeyChecking=no \
            "$DATA_DIR/${PAIR}_asi_mc.parquet" root@$IP:/root/neat/data/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$SCRIPT_DIR/train_ironnet_perpair.py" root@$IP:/root/neat/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$SCRIPT_DIR/neat_config_4in_3out.ini" root@$IP:/root/neat/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$REPO_DIR/lib/fast_eval.py" root@$IP:/root/neat/lib/
        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
            "$REPO_DIR/lib/asi_indicator.py" root@$IP:/root/neat/lib/
        ssh -o StrictHostKeyChecking=no root@$IP 'touch /root/neat/lib/__init__.py'
        log "  [$PAIR] $IP: ready"
    ) &
done < "$BATCH2_FILE"
wait

# Launch batch 2
log "Launching batch 2..."
while IFS=' ' read -r PAIR IP; do
    LOG_PATH="results/ironnet/iron_v3_${PAIR}_s${SEED}.log"
    ssh -o StrictHostKeyChecking=no root@$IP "
        source /root/venv/bin/activate && cd /root/neat
        ASI_MC_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 \
        nohup python3 train_ironnet_perpair.py \
            --pair $PAIR --seed $SEED --sine-gens $SINE_GENS \
            --pretrain-gens $PRETRAIN_GENS --gens $GENS \
            --islands $ISLANDS --pop $POP --max-hold $MAX_HOLD \
        > $LOG_PATH 2>&1 &
        disown
    " 2>/dev/null
    log "  Launched: $PAIR → $IP"
done < "$BATCH2_FILE"

# ══════════════════════════════════════════════════════════════
# STAGE 3: Wait for Free NEAT v2 (4in), then fire 6in
# ══════════════════════════════════════════════════════════════
log "═══ STAGE 3: Waiting for Free NEAT v2 (4in) ═══"
FN4_DONE=0
while [ $FN4_DONE -eq 0 ]; do
    FN4_DONE=1
    for ENTRY in "EUR_GBP:<SERVER_IP_2>:42" "CAD_JPY:<SERVER_IP_3>:137"; do
        IFS=: read -r PAIR IP SEED_FN <<< "$ENTRY"
        LOG_PATH="/root/neat/results/free_neat/free_neat_v2_${PAIR}_s${SEED_FN}.log"
        STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
            "tail -1 $LOG_PATH 2>/dev/null" 2>/dev/null || echo "")
        log "  Free NEAT v2 $PAIR: $STATUS"
        if ! echo "$STATUS" | grep -q "^Total:"; then
            FN4_DONE=0
        fi
    done
    [ $FN4_DONE -eq 0 ] && sleep 300
done
log "Free NEAT v2 (4in) complete!"

log "Collecting Free NEAT v2 (4in)..."
for ENTRY in "EUR_GBP:<SERVER_IP_2>:42" "CAD_JPY:<SERVER_IP_3>:137"; do
    IFS=: read -r PAIR IP SEED_FN <<< "$ENTRY"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat/free_neat_v2_${PAIR}_s${SEED_FN}_best.pkl" \
        "$RESULTS_DIR/free_neat/" 2>/dev/null && log "  $PAIR pkl: OK" || log "  $PAIR pkl: FAIL"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat/free_neat_v2_${PAIR}_s${SEED_FN}_result.json" \
        "$RESULTS_DIR/free_neat/" 2>/dev/null && log "  $PAIR json: OK" || log "  $PAIR json: FAIL"
done

log "Deleting Free NEAT v2 (4in) servers..."
hcloud server delete free-neat-1 2>/dev/null || true
hcloud server delete free-neat-2 2>/dev/null || true

log "Firing Free NEAT v2 6-input..."
bash "$SCRIPT_DIR/deploy_free_neat_6in.sh" 2>&1 | tee -a "$LOG"

# ══════════════════════════════════════════════════════════════
# STAGE 4: Wait for batch 2 to finish, collect, delete
# ══════════════════════════════════════════════════════════════
log "═══ STAGE 4: Waiting for batch 2 ═══"
wait_for_all_done "$BATCH2_FILE" "ironnet" "batch2"

log "Collecting batch 2..."
while IFS=' ' read -r PAIR IP; do
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_best.pkl" \
        "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  $PAIR pkl: OK" || log "  $PAIR pkl: FAIL"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_result.json" \
        "$RESULTS_DIR/ironnet/" 2>/dev/null && log "  $PAIR json: OK" || log "  $PAIR json: FAIL"
done < "$BATCH2_FILE"

log "Deleting batch 2 servers..."
while IFS=' ' read -r PAIR IP; do
    NAME="ironnet-pp-$(echo "$PAIR" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    hcloud server delete "$NAME" 2>/dev/null && log "  Deleted $NAME" || log "  $NAME already gone"
done < "$BATCH2_FILE"

# ══════════════════════════════════════════════════════════════
# STAGE 5: Wait for Free NEAT 6in to finish, collect, delete
# ══════════════════════════════════════════════════════════════
log "═══ STAGE 5: Waiting for Free NEAT v2 6-input ═══"
FN6_SERVERS_FILE="$SCRIPT_DIR/free_neat_6in_servers.txt"

# Poll until done (servers file created by deploy script)
FN6_DONE=0
while [ $FN6_DONE -eq 0 ]; do
    if [ ! -f "$FN6_SERVERS_FILE" ]; then
        log "  Waiting for 6in servers file..."
        sleep 60; continue
    fi
    FN6_DONE=1
    IPS=($(cat "$FN6_SERVERS_FILE"))
    for ENTRY in "EUR_GBP:0:42" "CAD_JPY:1:137"; do
        IFS=: read -r PAIR IDX SEED_FN <<< "$ENTRY"
        IP="${IPS[$IDX]}"
        LOG_PATH="/root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED_FN}.log"
        STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$IP \
            "tail -1 $LOG_PATH 2>/dev/null" 2>/dev/null || echo "")
        log "  Free NEAT 6in $PAIR: $STATUS"
        if ! echo "$STATUS" | grep -q "^Total:"; then
            FN6_DONE=0
        fi
    done
    [ $FN6_DONE -eq 0 ] && sleep 300
done
log "Free NEAT 6in complete!"

IPS=($(cat "$FN6_SERVERS_FILE"))
log "Collecting Free NEAT 6in..."
for ENTRY in "EUR_GBP:0:42" "CAD_JPY:1:137"; do
    IFS=: read -r PAIR IDX SEED_FN <<< "$ENTRY"
    IP="${IPS[$IDX]}"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED_FN}_best.pkl" \
        "$RESULTS_DIR/free_neat_6in/" 2>/dev/null && log "  $PAIR pkl: OK" || log "  $PAIR pkl: FAIL"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/free_neat_6in/free_neat_6in_${PAIR}_s${SEED_FN}_result.json" \
        "$RESULTS_DIR/free_neat_6in/" 2>/dev/null && log "  $PAIR json: OK" || log "  $PAIR json: FAIL"
done

log "Deleting Free NEAT 6in servers..."
hcloud server delete free-neat-6in-1 2>/dev/null || true
hcloud server delete free-neat-6in-2 2>/dev/null || true

log "═══ ALL STAGES COMPLETE ═══"
log "Results in: $RESULTS_DIR"
hcloud server list
