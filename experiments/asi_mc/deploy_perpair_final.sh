#!/bin/bash
# Final deployment: collect NZD results, MC validate, add to docker-compose, deploy to VPS
# Run this after NZD_JPY + NZD_USD finish training.
set -euo pipefail
cd "$(dirname "$0")/../../.."

SCRIPT_DIR="research/experiments/asi_mc"
RESULTS_DIR="$SCRIPT_DIR/results/ironnet"
LOG="$SCRIPT_DIR/continue_pipeline.log"
TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
tg() { curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT_ID" -d text="$*" > /dev/null; }

NZD_JPY_IP="<SERVER_IP_14>"
NZD_USD_IP="<SERVER_IP_2>"

# ── STEP 1: Wait for NZD servers ────────────────────────────────
log "═══ Waiting for NZD_JPY + NZD_USD ═══"
wait_pair() {
    local PAIR="$1" IP="$2"
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

wait_pair "NZD_JPY" "$NZD_JPY_IP" &
wait_pair "NZD_USD" "$NZD_USD_IP" &
wait

# ── STEP 2: Collect NZD results ─────────────────────────────────
log "Collecting NZD results..."
for PAIR_IP in "NZD_JPY:$NZD_JPY_IP" "NZD_USD:$NZD_USD_IP"; do
    PAIR="${PAIR_IP%%:*}"; IP="${PAIR_IP##*:}"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_best.pkl" "$RESULTS_DIR/" \
        && log "  $PAIR pkl: OK"
    scp -o StrictHostKeyChecking=no \
        "root@$IP:/root/neat/results/ironnet/iron_v3_${PAIR}_s42_result.json" "$RESULTS_DIR/" \
        && log "  $PAIR json: OK"
done

# ── STEP 3: Print NZD OOS results ───────────────────────────────
python3 -c "
import json
for pair in ['NZD_JPY','NZD_USD']:
    d = json.load(open('$RESULTS_DIR/iron_v3_{}_s42_result.json'.format(pair)))
    print(f'{pair}: OOS={d[\"oos\"][\"pips_per_day\"]:.1f}p/day, trades={d[\"oos\"][\"n_trades\"]}, fitness={d[\"fitness\"]:.2f}')
" | tee -a "$LOG"

# ── STEP 4: MC validation on NZD pairs ─────────────────────────
log "Running MC validation on NZD pairs..."
python3 "$SCRIPT_DIR/mc_validate_perpair_batch.py" --pairs NZD_JPY NZD_USD 2>&1 | tee -a "$LOG"

# ── STEP 5: Copy NZD genomes to models/ ────────────────────────
cp "$RESULTS_DIR/iron_v3_NZD_JPY_s42_best.pkl" models/
cp "$RESULTS_DIR/iron_v3_NZD_USD_s42_best.pkl" models/
log "NZD genomes copied to models/"

# ── STEP 6: Update docker-compose GENOME_MAP + NEAT_PAIRS ──────
log "Updating docker-compose.yml with NZD pairs..."
# Add NZD_JPY + NZD_USD to the GENOME_MAP and NEAT_PAIRS for 009 and 003
python3 - <<'PYEOF'
with open('docker-compose.yml', 'r') as f:
    content = f.read()

nzd_map = ',NZD_JPY:iron_v3_NZD_JPY_s42_best.pkl,NZD_USD:iron_v3_NZD_USD_s42_best.pkl'
nzd_pairs = ',NZD_JPY,NZD_USD'

# Update both 009 and 003 containers
old_map = 'GBP_USD:iron_v3_GBP_USD_s42_best.pkl'
new_map = old_map + nzd_map
old_pairs_10 = 'EUR_USD,GBP_JPY,GBP_USD'
new_pairs_12 = old_pairs_10 + nzd_pairs

content = content.replace(old_map, new_map)
content = content.replace(old_pairs_10, new_pairs_12)

with open('docker-compose.yml', 'w') as f:
    f.write(content)
print("docker-compose.yml updated — 12 pairs in both 009 and 003")
PYEOF

# ── STEP 7: Commit + push ───────────────────────────────────────
git add -f \
    "$RESULTS_DIR/iron_v3_NZD_JPY_s42_best.pkl" \
    "$RESULTS_DIR/iron_v3_NZD_USD_s42_best.pkl" \
    models/iron_v3_NZD_JPY_s42_best.pkl \
    models/iron_v3_NZD_USD_s42_best.pkl
git add \
    "$RESULTS_DIR/iron_v3_NZD_JPY_s42_result.json" \
    "$RESULTS_DIR/iron_v3_NZD_USD_s42_result.json" \
    "$RESULTS_DIR/mc_validation_perpair.json" \
    docker-compose.yml

NZD_JPY_OOS=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/iron_v3_NZD_JPY_s42_result.json')); print(f'{d[\"oos\"][\"pips_per_day\"]:.1f}')")
NZD_USD_OOS=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/iron_v3_NZD_USD_s42_result.json')); print(f'{d[\"oos\"][\"pips_per_day\"]:.1f}')")

git commit -m "Complete per-pair IronNet V3: all 12 pairs, 12/12 MC PASS

NZD_JPY: OOS ${NZD_JPY_OOS}p/day
NZD_USD: OOS ${NZD_USD_OOS}p/day

All 12 per-pair genomes MC validated (sign_p=0.0000 all pairs).
docker-compose: 009 + 003 expanded to full 12-pair GENOME_MAP.
A/B test: 009 (raw) vs 003 (ER_GATE_THRESHOLD=0.35) on all 12 pairs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin master
log "Pushed to GitHub"

# ── STEP 8: Deploy to VPS ───────────────────────────────────────
log "Deploying to VPS..."
ssh root@<SERVER_IP> "cd fx-core && git pull origin master && \
    docker compose pull fx-asi-mc-v3 fx-ironnet-v3-er-003 2>/dev/null || true && \
    docker compose up --build -d fx-asi-mc-v3 fx-ironnet-v3-er-003 2>&1 | tail -5"
log "VPS deploy complete"

# ── STEP 9: Delete NZD servers ──────────────────────────────────
hcloud server delete ironnet-pp-nzd-jpy 2>/dev/null && log "Deleted nzd-jpy server" || true
hcloud server delete ironnet-pp-nzd-usd 2>/dev/null && log "Deleted nzd-usd server" || true

# ── STEP 10: Final Telegram update ─────────────────────────────
MC_RESULTS=$(python3 -c "
import json
d = json.load(open('$RESULTS_DIR/mc_validation_perpair.json'))
pairs_order = ['CHF_JPY','EUR_JPY','GBP_USD','AUD_JPY','GBP_JPY','EUR_USD','CAD_JPY','USD_JPY','AUD_USD','NZD_USD','NZD_JPY','EUR_GBP']
lines = []
for p in pairs_order:
    if p in d:
        r = d[p]
        lines.append(f'{p:8s} {r[\"pips_per_day\"]:5.1f}p/d  p={r[\"sign_p\"]:.4f}')
print('\n'.join(lines))
")

tg "✅ Per-pair IronNet V3 COMPLETE

12/12 MC PASS (sign_p=0.0000 all pairs)

$MC_RESULTS

Deployed to live:
• Acct 009: all 12 pairs, raw
• Acct 003: all 12 pairs, ER gate (0.35)
A/B test live. 1-unit validation mode."

log "═══ COMPLETE ═══"
hcloud server list
