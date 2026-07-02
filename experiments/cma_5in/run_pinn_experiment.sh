#!/bin/bash
# PINN-CMA Experiment Runner
# Runs all 54 combos: 6 arms × 3 pairs × 3 seeds
# Usage: bash run_pinn_experiment.sh [--workers N]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WORKERS=${1:-4}
GENS=200
POPSIZE=24

PAIRS=("EUR_JPY" "CAD_JPY" "EUR_GBP")
SEEDS=(42 123 777)
MODES=("baseline" "inputs" "hyper" "inputs_hyper" "inputs_fitness" "full")

RESULTS_DIR="$SCRIPT_DIR/results_pinn"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/experiment_log.txt"
echo "PINN-CMA Experiment started: $(date)" | tee "$LOG"
echo "Workers=$WORKERS Gens=$GENS Popsize=$POPSIZE" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

TOTAL=$((${#MODES[@]} * ${#PAIRS[@]} * ${#SEEDS[@]}))
COUNT=0
STARTED=$(date +%s)

for MODE in "${MODES[@]}"; do
    for PAIR in "${PAIRS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            TAG="pinn_${MODE}_${PAIR}_s${SEED}"

            # Skip if already done
            if [ -f "$RESULTS_DIR/${TAG}_summary.json" ]; then
                echo "[$COUNT/$TOTAL] SKIP $TAG (already done)" | tee -a "$LOG"
                continue
            fi

            echo "[$COUNT/$TOTAL] $TAG ..." | tee -a "$LOG"
            T0=$(date +%s)

            python3 train_pinn_cma.py \
                --pair "$PAIR" \
                --mode "$MODE" \
                --seed "$SEED" \
                --gens "$GENS" \
                --popsize "$POPSIZE" \
                --workers "$WORKERS" \
                2>&1 | tail -5 | tee -a "$LOG"

            T1=$(date +%s)
            ELAPSED=$((T1 - T0))
            echo "  Done in ${ELAPSED}s" | tee -a "$LOG"
            echo "" | tee -a "$LOG"
        done
    done
done

FINISHED=$(date +%s)
TOTAL_TIME=$(( (FINISHED - STARTED) / 60 ))
echo "==========================================" | tee -a "$LOG"
echo "Experiment complete: ${TOTAL_TIME} min total" | tee -a "$LOG"
echo "Results in: $RESULTS_DIR" | tee -a "$LOG"

# Run analysis
echo "" | tee -a "$LOG"
echo "Running analysis..." | tee -a "$LOG"
python3 analyze_pinn_results.py 2>&1 | tee -a "$LOG"
