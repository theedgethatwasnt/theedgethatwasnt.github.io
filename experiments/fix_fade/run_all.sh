#!/bin/bash
# run_all.sh — London-Fix Fade full IS battery, run on Hetzner (root@<rnd-box>).
#   code:    /root/work/code_fix/  (this directory, rsynced)
#   data:    /root/work/data/m5_ba/
# Usage: cd /root/work/code_fix && ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=/root/venv/bin/python3
DATA_DIR=${DATA_DIR:-/root/work/data/m5_ba}
OUT_DIR=${OUT_DIR:-results}

mkdir -p "$OUT_DIR"

echo "=== 1/5: pytest (incl. gate 1 RW self-test, DST tests, harness tests) ==="
/root/venv/bin/python -m pytest -x -q

echo "=== 2/5: RW self-test artifact (gate 1 detail for the summary) ==="
"$PY" rw_selftest.py --out-dir "$OUT_DIR"

echo "=== 3/5: IS battery — 12 pairs x 3 arms ==="
"$PY" run_is_battery.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== 4/5: gate table + per-pair/portfolio summaries + month-end split ==="
"$PY" compute_gates.py --trades-csv "$OUT_DIR/is_battery_trades.csv" --out-dir "$OUT_DIR"

echo "=== 5/5: results/is_summary.md ==="
"$PY" make_summary.py --out-dir "$OUT_DIR"

echo "=== DONE ==="
cat "$OUT_DIR/is_summary.md"
