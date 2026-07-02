#!/bin/bash
# ═══════��══════════════════════════════��════════════════════════
# CMA-NN 12-Pair Grid + MC Validation on Hetzner
#
# Trains V3+macd_hist (the +73 p/day CHF_JPY winner) on all 12 pairs.
# Then runs MC sign-shuffle validation (10K shuffles) on each pair.
# Also runs multi-seed robustness (seeds 42, 137, 23, 99).
#
# Architecture: n_in→8 hidden (sin)→3 output (linear, argmax)
# Inputs: mc_d_a, mc_dd_a, er_norm, macd_hist + upnl, mae, mfe = 7
# Optimizer: CMA-ES, popsize 24, 200 gens, fixed sin activation
#
# 1 server, all 12 pairs sequential (~3 min/pair × 12 = ~36 min training)
# + MC validation (~2 min/pair × 12 = ~24 min)
# + multi-seed (3 extra seeds × ~3 min = ~9 min for CHF_JPY robustness)
# Total: ~70 min
# Cost: 1 × ccx23 × 1.5h × $0.07/hr ≈ $0.11
# ══════════════════════════════════════════════════════════��════

set -e

GENS=200
POPSIZE=24
SEED=42
EXTRAS="macd_hist"
LABEL="grid12"
MC_SHUFFLES=10000

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
RESULTS_DIR="$PROJECT_DIR/research/experiments/cma_5in/results"
CMA_DIR="$PROJECT_DIR/research/experiments/cma_5in"
SERVER_FILE="$CMA_DIR/grid12_server.txt"

ALL_PAIRS="EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP"
MULTI_SEED_PAIR="CHF_JPY"
EXTRA_SEEDS="137 23 99"

mkdir -p "$RESULTS_DIR"

echo "══════════════���════════════════════════════"
echo "  CMA-NN 12-Pair Grid + MC + Multi-Seed"
echo "  Extras: $EXTRAS"
echo "  Gens: $GENS, Pop: $POPSIZE, Act: sin"
echo "═══════════════════════════════════════════"

# ── Step 1: Create server ──
echo ""
echo "Creating Hetzner server..."
NAME="cma-grid-1"

# Check if server already exists
if hcloud server describe "$NAME" > /dev/null 2>&1; then
    echo "  Server $NAME already exists"
    IP=$(hcloud server ip "$NAME")
else
    hcloud server create --name "$NAME" --type ccx23 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" --without-ipv6 2>/dev/null
    IP=$(hcloud server ip "$NAME")
fi
echo "$IP" > "$SERVER_FILE"
echo "  $NAME → $IP"

echo ""
echo "Waiting 30s for server to boot..."
sleep 30

# ── Step 2: Setup server ──
echo "Setting up server..."
ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP'
set -e
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba pyarrow cma
mkdir -p /root/fx-core/lib \
         /root/fx-core/research/experiments/cma_5in/results \
         /root/fx-core/data/m5_ohlc \
         /root/fx-core/data/unified_indicators
SETUP
echo "  Base setup done"

# ── Step 3: Copy code ──
echo "Copying code..."
scp -q "$PROJECT_DIR/lib/fast_eval.py" root@$IP:/root/fx-core/lib/
scp -q "$PROJECT_DIR/lib/asi_indicator.py" root@$IP:/root/fx-core/lib/
scp -q "$CMA_DIR/train_cma_v2.py" root@$IP:/root/fx-core/research/experiments/cma_5in/
scp -q "$CMA_DIR/extra_indicators.py" root@$IP:/root/fx-core/research/experiments/cma_5in/
echo "  Code copied"

# ── Step 4: Copy data ��─
echo "Copying M5 OHLC data (84MB)..."
rsync -az "$PROJECT_DIR/data/m5_ohlc/" root@$IP:/root/fx-core/data/m5_ohlc/
echo "Copying unified indicators (250MB)..."
rsync -az "$PROJECT_DIR/data/unified_indicators/" root@$IP:/root/fx-core/data/unified_indicators/
echo "  Data synced"

# ── Step 5: Copy the MC validation + runner script ──
echo "Creating runner script on server..."
ssh root@$IP bash -s <<'RUNNER_SCRIPT'
cat > /root/run_grid.sh <<'GRIDEOF'
#!/bin/bash
set -e
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/cma_5in

ALL_PAIRS="EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP"

echo "════════════════════════════════════════════════"
echo "  PHASE 1: 12-Pair Grid (seed 42)"
echo "═════��══════════════════���═══════════════════════"

for PAIR in $ALL_PAIRS; do
    echo ""
    echo "──── Training $PAIR (seed 42) ────"
    PYTHONUNBUFFERED=1 python3 train_cma_v2.py \
        --pair $PAIR \
        --seed 42 \
        --gens 200 \
        --features v3_plus \
        --extras macd_hist \
        --fixed-activation sin \
        --popsize 24 \
        --workers 4 \
        --label grid12 \
        2>&1 | tail -5
    echo "  ✓ $PAIR done"
done

echo ""
echo "═══��══════════════════════════���═════════════════"
echo "  PHASE 2: Multi-Seed Robustness (CHF_JPY)"
echo "═══════════��═════════════════════════════��══════"

for SEED in 137 23 99; do
    echo ""
    echo "──── CHF_JPY seed $SEED ────"
    PYTHONUNBUFFERED=1 python3 train_cma_v2.py \
        --pair CHF_JPY \
        --seed $SEED \
        --gens 200 \
        --features v3_plus \
        --extras macd_hist \
        --fixed-activation sin \
        --popsize 24 \
        --workers 4 \
        --label grid12 \
        2>&1 | tail -5
    echo "  ✓ CHF_JPY s$SEED done"
done

echo ""
echo "═══════════════════════���════════════════════════"
echo "  PHASE 3: MC Sign-Shuffle Validation"
echo "══════════════════════════════════════════��═════"

python3 - <<'MCEOF'
import json, pickle, sys, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/root/fx-core")
sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.cma_5in.train_cma_v2 import (
    _compute_er_norm_v3, compute_m5_slope, simulate_chunk, N_HID, N_OUT,
    decode_act, activate
)
from lib.asi_indicator import compute_asi_mc
import pandas as pd
from numba import njit

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}
PAIR_SPREAD = {
    "EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
    "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
    "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
    "NZD_USD": 1.5, "EUR_GBP": 1.4,
}
ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]

N_SHUFFLES = 10000
RESULTS_DIR = Path("/root/fx-core/research/experiments/cma_5in/results")

@njit(cache=True)
def collect_pnls_cma(market_features, mid_close, pip, spread_pips, max_hold,
                     weights, n_in, fixed_act_id):
    """Extract per-trade PnLs for MC validation."""
    n_market = market_features.shape[0]
    n = market_features.shape[1]
    n_pos_state = 3
    total_in = n_market + n_pos_state

    # Parse weights
    w1_end = total_in * 8
    b1_end = w1_end + 8
    w2_end = b1_end + 24
    b2_end = w2_end + 3

    W1 = weights[:w1_end].reshape(8, total_in)
    b1 = weights[w1_end:b1_end]
    W2 = weights[b1_end:w2_end].reshape(3, 8)
    b2 = weights[w2_end:b2_end]

    if fixed_act_id < 0:
        act_genes = weights[b2_end:b2_end + 8]
    else:
        act_genes = np.empty(0)

    pnls = np.zeros(n)
    nt = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0
    worst_ae = 0.0
    best_fe = 0.0

    for i in range(200, n - 1):
        mid = mid_close[i]
        # Position state
        if position != 0:
            raw_pnl = (mid - entry_price) / pip * position
            upnl = np.tanh(raw_pnl / 20.0)
            ae = spread_pips + max(0.0, -raw_pnl)
            if ae > worst_ae:
                worst_ae = ae
            fe = max(0.0, raw_pnl)
            if fe > best_fe:
                best_fe = fe
            mae_in = np.tanh(worst_ae / 20.0)
            mfe_in = np.tanh(best_fe / 20.0)
        else:
            upnl = 0.0
            mae_in = 0.0
            mfe_in = 0.0

        # Build input
        inp = np.empty(total_in)
        for k in range(n_market):
            inp[k] = market_features[k, i]
        inp[n_market] = upnl
        inp[n_market + 1] = mae_in
        inp[n_market + 2] = mfe_in

        # Forward pass
        h = np.empty(8)
        for j in range(8):
            z = b1[j]
            for k in range(total_in):
                z += W1[j, k] * inp[k]
            if fixed_act_id >= 0:
                aid = fixed_act_id
            else:
                g = act_genes[j] - np.floor(act_genes[j])
                aid = int(g * 9)
                if aid < 0: aid = 0
                if aid >= 9: aid = 8

            if aid == 0: h[j] = np.tanh(z)
            elif aid == 1: h[j] = np.sin(z)
            elif aid == 2: h[j] = np.cos(z)
            elif aid == 3: h[j] = np.exp(-z * z)
            elif aid == 4:
                zc = max(-50.0, min(50.0, z))
                h[j] = 1.0 / np.cosh(zc)
            elif aid == 5: h[j] = np.exp(-z*z/2.0) - 0.5*np.exp(-z*z/8.0)
            elif aid == 6: h[j] = np.exp(-2.0*z*z) * np.cos(2.0*np.pi*z)
            elif aid == 7:
                if z > 1e-7 or z < -1e-7:
                    h[j] = np.sin(np.pi*z) / (np.pi*z)
                else:
                    h[j] = 1.0
            else: h[j] = np.sin(z) * np.exp(-z*z/2.0)

        out = np.empty(3)
        for j in range(3):
            z = b2[j]
            for k in range(8):
                z += W2[j, k] * h[k]
            out[j] = z

        signal = 0  # 0=hold, 1=buy, -1=sell, 2=flatten
        if out[0] > out[1] and out[0] > out[2]:
            signal = 1
        elif out[1] > out[0] and out[1] > out[2]:
            signal = -1
        else:
            signal = 2

        # Max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid - entry_price) / pip * position - spread_pips
            pnls[nt] = pnl; nt += 1
            position = 0; entry_price = 0.0; worst_ae = 0.0; best_fe = 0.0
            continue

        if position == 0:
            if signal == 1:
                position = 1; entry_price = mid; entry_bar = i
                worst_ae = spread_pips; best_fe = 0.0
            elif signal == -1:
                position = -1; entry_price = mid; entry_bar = i
                worst_ae = spread_pips; best_fe = 0.0
        else:
            close_it = False; new_pos = 0
            if signal == 2:
                close_it = True
            elif position == 1 and signal == -1:
                close_it = True; new_pos = -1
            elif position == -1 and signal == 1:
                close_it = True; new_pos = 1
            if close_it:
                pnl = (mid - entry_price) / pip * position - spread_pips
                pnls[nt] = pnl; nt += 1
                if new_pos != 0:
                    position = new_pos; entry_price = mid; entry_bar = i
                    worst_ae = spread_pips; best_fe = 0.0
                else:
                    position = 0; entry_price = 0.0; worst_ae = 0.0; best_fe = 0.0

    # Close open position
    if position != 0:
        pnl = (mid_close[n-1] - entry_price) / pip * position - spread_pips
        pnls[nt] = pnl; nt += 1

    return pnls[:nt]


def mc_sign_test(pnls, n_shuffles=10000, seed=42):
    actual = pnls.sum()
    rng = np.random.RandomState(seed)
    better = 0
    for _ in range(n_shuffles):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(pnls))
        if (pnls * signs).sum() >= actual:
            better += 1
    return better / n_shuffles


def load_pair_data(pair):
    """Load M5 OHLC + compute V3 indicators + macd_hist."""
    m5_path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
    df = pd.read_parquet(m5_path)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    # V3 indicators
    mc_d_a, mc_dd_a = compute_asi_mc(o, h, l, c)
    er_norm = _compute_er_norm_v3(c, window=60)

    # macd_hist from unified
    uni_path = PROJECT_ROOT / "data" / "unified_indicators" / f"{pair}_unified.parquet"
    df_u = pd.read_parquet(uni_path, columns=["timestamp", "macd_hist"])
    df_m = df[["timestamp"]].merge(df_u, on="timestamp", how="left")
    macd_raw = df_m["macd_hist"].fillna(0.0).values.astype(np.float64)
    macd_hist = np.clip(macd_raw / 2.0, -1.0, 1.0)

    market = np.stack([mc_d_a, mc_dd_a, er_norm, macd_hist], axis=0)
    return market, c


results = {}
mc_out_path = RESULTS_DIR / "grid12_mc_validation.json"

print("JIT warmup...")
dummy_m = np.zeros((4, 300))
dummy_c = np.zeros(300)
dummy_w = np.zeros(91)
try:
    collect_pnls_cma(dummy_m, dummy_c, 0.0001, 1.5, 200, dummy_w, 7, 1)
except:
    pass
print("  done")

for pair in ALL_PAIRS:
    pkl_path = RESULTS_DIR / f"grid12_v3_plus_macd_hist_{pair}_s42_best.pkl"
    if not pkl_path.exists():
        print(f"  {pair}: SKIP (no genome)")
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    weights = data["weights"]
    oos_info = data.get("oos", {})

    print(f"\n  {pair}: loading data...")
    market, mid_close = load_pair_data(pair)
    n = market.shape[1]
    oos_start = int(n * 0.7)

    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    print(f"  {pair}: collecting OOS P&Ls (bars {oos_start}:{n})...")
    pnls = collect_pnls_cma(
        market[:, oos_start:], mid_close[oos_start:],
        pip, spread, 200, weights, 7, 1  # fixed_act_id=1 (sin)
    )

    if len(pnls) == 0:
        print(f"  {pair}: 0 trades, SKIP MC")
        results[pair] = {"n_trades": 0, "sign_p": 1.0, "verdict": "NO_TRADES"}
        continue

    print(f"  {pair}: {len(pnls)} trades, total={pnls.sum():.1f}p, MC shuffle ({N_SHUFFLES})...")
    t0 = time.time()
    sign_p = mc_sign_test(pnls, N_SHUFFLES, seed=42)
    elapsed = time.time() - t0

    verdict = "PASS" if sign_p < 0.05 else "FAIL"
    avg_pnl = pnls.mean()
    total_pnl = pnls.sum()
    oos_days = (n - oos_start) / (12 * 24)  # M5 bars per day
    pps = total_pnl / oos_days if oos_days > 0 else 0

    results[pair] = {
        "n_trades": int(len(pnls)),
        "total_pnl": round(float(total_pnl), 1),
        "pips_per_day": round(float(pps), 1),
        "avg_pnl": round(float(avg_pnl), 2),
        "sign_p": round(float(sign_p), 6),
        "verdict": verdict,
        "mc_seconds": round(elapsed, 1),
    }
    print(f"  {pair}: {len(pnls)}T, {pps:.1f}p/day, sign_p={sign_p:.6f} {verdict} ({elapsed:.1f}s)")

    # Save incrementally
    with open(mc_out_path, "w") as f:
        json.dump(results, f, indent=2)

# Summary
print("\n" + "=" * 65)
print("  MC VALIDATION SUMMARY")
print("=" * 65)
n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
n_total = len(results)
pairs_pass = [p for p, r in results.items() if r["verdict"] == "PASS"]
avg_pps = np.mean([r["pips_per_day"] for r in results.values() if r["n_trades"] > 0])
print(f"  Passed: {n_pass}/{n_total}")
print(f"  Avg OOS: {avg_pps:.1f} p/day")
print(f"  Results: {mc_out_path}")
MCEOF

echo ""
echo "═══════════════════════════════════════���════════"
echo "  ALL DONE"
echo "═══════��═════════════════════��══════════════════"
date
GRIDEOF
chmod +x /root/run_grid.sh
RUNNER_SCRIPT
echo "  Runner script ready"

# ── Step 6: Launch training ──
echo ""
echo "Launching training on server ($IP)..."
ssh root@$IP "nohup bash /root/run_grid.sh > /root/grid_run.log 2>&1 &"

echo ""
echo "══════��════════════════════════════════════"
echo "  Training launched!"
echo ""
echo "  Monitor:"
echo "    ssh root@$IP 'tail -f /root/grid_run.log'"
echo ""
echo "  Collect when done:"
echo "    bash collect_12pair_grid.sh"
echo ""
echo "  Delete server when done:"
echo "    hcloud server delete cma-grid-1 --yes"
echo "═══════════════════════════════════════════"
