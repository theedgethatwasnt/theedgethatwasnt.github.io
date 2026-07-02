#!/bin/bash
# Experiment B: Extended architecture sweep on CHF_JPY
# Tests requested by user:
#   1. gauss first row (gauss→X→tanh)
#   2. gauss→sin→tanh
#   3. 4-layer 7→10sin→7→3 with skip (sin→gauss→{bank})
# Plus send results via Telegram
set -e
cd /path/to/projects/fx-core
S="research/experiments/cma_5in/train_cma_mixed.py"
RESULTS=""

run_and_collect() {
    local desc="$1"
    shift
    echo ""
    echo "$(date) ══ $desc ══"
    OUTPUT=$(python3 $S "$@" 2>&1)
    LAST=$(echo "$OUTPUT" | grep -E "OOS:" | tail -1)
    echo "$OUTPUT" | tail -8
    RESULTS="${RESULTS}${desc}: ${LAST}\n"
}

# ── Group 1: gauss as L1 (replacing sin) ──
run_and_collect "gauss→sin→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,sin,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→tanh→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,tanh,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→gauss→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,gauss,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→mhat→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,mexican_hat,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→sech→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,sech,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→morlet→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,morlet,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→dog→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,dog,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "gauss→cos→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations gauss,cos,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

# ── Group 2: 4-layer wide (10→7→3 hidden) with sin→gauss→{bank} ──
for BANK_ACT in tanh sin gauss sech dog morlet mexican_hat cos; do
    run_and_collect "4L 10sin→7gauss→3${BANK_ACT} +skip" --pair CHF_JPY --extras macd_hist \
        --layers 10,7,3 --activations sin,gauss,${BANK_ACT} --skip --seed 42 --gens 200 --workers 4 --label expB_wide
done

# ── Group 3: cos as L1 ──
run_and_collect "cos→gauss→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations cos,gauss,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

run_and_collect "cos→sin→tanh +skip" --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations cos,sin,tanh --skip --seed 42 --gens 200 --workers 4 --label expB

# ── Send Telegram summary ──
echo ""
echo "$(date) ALL DONE"
echo ""
echo "RESULTS SUMMARY:"
echo -e "$RESULTS"

# Send via Telegram
source .env 2>/dev/null || true
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    MSG="🔬 CMA-NN Exp B — Architecture Sweep Complete (CHF_JPY)

$(echo -e "$RESULTS")

H1 Multi-Seed (4 seeds × 12 pairs):
Best-seed avg: +41.5 p/d
Mean across all seeds: +40.2 p/d
Top: GBP_JPY +80.4 (s23), USD_JPY +67.3 (s42)

M5 multi-seed still running on Hetzner (ETA ~1h)"

    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="${MSG}" \
        -d parse_mode="Markdown" > /dev/null 2>&1
    echo "Telegram sent"
else
    echo "No Telegram credentials in .env"
fi
