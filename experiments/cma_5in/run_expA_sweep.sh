#!/bin/bash
# Experiment A: Mixed activation sweep on CHF_JPY
# Each run: ~4 min. Total: ~36 min for 9 variants.
set -e
cd /path/to/projects/fx-core
S="research/experiments/cma_5in/train_cma_mixed.py"

echo "$(date) Starting Experiment A sweep..."

# Middle-layer activation sweep (sin→X→tanh, 3,3,3, skip)
for MID in gauss sech dog morlet tanh sin; do
    echo ""
    echo "$(date) ══ sin→${MID}→tanh ══"
    python3 $S --pair CHF_JPY --extras macd_hist \
        --layers 3,3,3 --activations sin,${MID},tanh --skip \
        --seed 42 --gens 200 --workers 4 --label expA 2>&1 | tail -8
done

# Control: flat 8×sin (replicates train_cma_v2)
echo ""
echo "$(date) ══ Control: flat 8×sin ══"
python3 $S --pair CHF_JPY --extras macd_hist \
    --layers 8 --activations sin \
    --seed 42 --gens 200 --workers 4 --label expA_ctrl 2>&1 | tail -8

# Wide sandwich
echo ""
echo "$(date) ══ Wide: 5,5,5 sin→mhat→tanh + skip ══"
python3 $S --pair CHF_JPY --extras macd_hist \
    --layers 5,5,5 --activations sin,mexican_hat,tanh --skip \
    --seed 42 --gens 200 --workers 4 --label expA_wide 2>&1 | tail -8

# 4-layer deep
echo ""
echo "$(date) ══ Deep: 4-layer sin→mhat→tanh→tanh + skip ══"
python3 $S --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3,3 --activations sin,mexican_hat,tanh,tanh --skip \
    --seed 42 --gens 200 --workers 4 --label expA_4layer 2>&1 | tail -8

echo ""
echo "$(date) ALL DONE"

echo ""
echo "$(date) ══ sin→cos→tanh ══"
python3 $S --pair CHF_JPY --extras macd_hist \
    --layers 3,3,3 --activations sin,cos,tanh --skip \
    --seed 42 --gens 200 --workers 4 --label expA 2>&1 | tail -8
