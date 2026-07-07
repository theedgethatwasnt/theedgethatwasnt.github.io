#!/bin/bash
# run_full_battery.sh — runs on the Hetzner box only (per CLAUDE.md execution rules).
# Sequences: gate2 parity (BLOCKING) -> IS battery -> gates 3-5 -> is_summary.md.
set -e
cd /root/work/code/research/experiments/scratch_tail

echo "=== GATE 2: R7 parity vs live paper trail (BLOCKING) ==="
set +e
/root/venv/bin/python3 gate2_parity.py \
  --parity-data-dir /root/work/data/parity_m5_ba \
  --trades-db /root/work/trades_2026-07-06.duckdb \
  --out-dir results
GATE2_STATUS=$?
set -e
if [ $GATE2_STATUS -ne 0 ]; then
  echo "GATE 2 FAILED — stopping per PREREGISTRATION.md decision rule (blocking gate)."
  exit 1
fi

echo ""
echo "=== IS BATTERY: 6 pairs x 9 arm-runs ==="
/root/venv/bin/python3 run_is_battery.py --data-dir /root/work/data/m5_ba --out-dir results

echo ""
echo "=== GATES 3-5 ==="
/root/venv/bin/python3 compute_gates.py --out-dir results

echo ""
echo "=== SUMMARY ==="
/root/venv/bin/python3 make_summary.py --out-dir results
cat results/is_summary.md
