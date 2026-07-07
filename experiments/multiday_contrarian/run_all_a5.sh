#!/usr/bin/env bash
# run_all_a5.sh — Task A5 orchestration. Run on the Hetzner box from /root/multiday/code/:
#   /root/venv/bin/python3 -m pytest -x -q   (gate 1 + regression, should already be 37/37)
#   ./run_all_a5.sh /root/multiday/data/m5_ba results
set -euo pipefail
DATA_DIR="${1:-/root/multiday/data/m5_ba}"
OUT_DIR="${2:-results}"
PY="${PYTHON:-/root/venv/bin/python3}"

mkdir -p "$OUT_DIR"

echo "=== [1/6] pytest (regression + gate 1) ==="
"$PY" -m pytest -x -q

echo "=== [2/6] primary IS battery (12 pairs x 3 arms) ==="
"$PY" run_is_battery.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --verify-derivation

echo "=== [3/6] gate table (3-6) ==="
"$PY" compute_gates.py --trades-csv "$OUT_DIR/is_battery_trades.csv" --out-dir "$OUT_DIR"

echo "=== [4/6] secondary (a): CSI StrengthSpread H4/64 ==="
"$PY" secondary_strengthspread.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== [5/6] secondary (b): D1 RSI(2) ==="
"$PY" secondary_rsi2.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== [5.5/6] secondary (c): equal-risk portfolio ==="
"$PY" equal_risk_portfolio.py --out-dir "$OUT_DIR"

echo "=== [6/6] is_summary.md ==="
"$PY" make_summary.py --out-dir "$OUT_DIR"

echo "=== DONE ==="
ls -la "$OUT_DIR"
