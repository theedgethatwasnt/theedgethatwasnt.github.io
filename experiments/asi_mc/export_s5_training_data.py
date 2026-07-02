#!/usr/bin/env python3
"""
Export S5-cadence training data for IronNet subset experiments.
===============================================================
Produces per-pair parquets with S5 mid prices + ALL subset indicators
forward-filled from M5 cadence.

Identity guarantee (training == live):
  ALL indicators computed by the SAME batch functions that underlie the
  live curator's streaming classes:
    compute_asi_mc()           ← same as ASIMC.compute() internals
    compute_er_norm()          ← same as KaufmanER / ASIMC ER path
    compute_asi()              ← called by SwingStructure.compute()
    compute_all_swing_features() ← called by SwingStructure.compute()

Forward-fill model (matches live curator ZMQ publish pattern):
  Indicators update once per M5 close (every 12 S5 bars).
  Between M5 closes, the last computed value is held — exactly as the
  curator publishes the same indicator message for all 12 S5 ticks.
  S5 mid prices give the actual entry/exit price at each 5-second step.

Spread:
  NOT stored in the parquet — injected at evaluation time from PAIR_SPREAD
  dict in train_swingdim.py (real mean bid-ask spreads from S5 BA data).
  Entry price = s5_mid ± spread_pips * pip (charged up front, correct model).

Output: data/s5_training/{PAIR}_s5_training.parquet
  Columns (all float32 except m5_bar_idx int32):
    s5_mid      — S5 mid price (bid+ask)/2
    mc_d_a      — ASI momentum direction  [-1, +1]
    mc_dd_a     — ASI momentum acceleration [-1, +1]
    er_norm     — Kaufman ER arctan-normalised [0, ~0.85]
    sb_a        — ASI swing breakout state {-1,-0.5,0,+0.5,+1}
    hh_asi      — ASI higher-high binary {0, 1}
    hl_asi      — ASI higher-low  binary {0, 1}
    erp_p       — price Extended Range Position (continuous, unclamped)
    erp_a       — ASI   Extended Range Position (continuous, unclamped)
    d_erp_p     — 1h delta of erp_p (erp_p[t] - erp_p[t-12M5])
    d_erp_a     — 1h delta of erp_a
    m5_bar_idx  — which M5 bar this S5 belongs to (debug / alignment)

Usage (Hetzner, neat-data volume mounted):
  S5_DATA_DIR=/mnt/neat-data ASI_MC_DATA_DIR=/root/neat/data \\
      python3 export_s5_training_data.py

Usage (local):
  python3 research/experiments/asi_mc/export_s5_training_data.py
"""

import sys
import os
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lib.asi_indicator import compute_asi_mc, compute_asi
from lib.swing_indicators import compute_all_swing_features

# ── Paths ──────────────────────────────────────────────────────────────────────
# S5 raw data (bid/ask OHLC parquets from OANDA)
S5_DATA_DIR = Path(os.environ.get(
    "S5_DATA_DIR",
    os.environ.get("NEAT_DATA_DIR",
                   str(PROJECT_ROOT / "data" / "scalper_parquet"))))

# Output directory
OUT_DIR = Path(os.environ.get(
    "ASI_MC_DATA_DIR",
    str(PROJECT_ROOT / "data" / "s5_training")))

ALL_PAIRS = [
    "EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
]

S5_PER_M5 = 12          # 12 × 5s = 60s = 1 M5 bar
ERP_DELTA_WINDOW = 12   # 12 M5 bars = 1 hour (same as SwingStructure.ERP_DELTA_WINDOW)


# ── Indicators ─────────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_er_norm(closes, window=60):
    """Kaufman ER, arctan-normalised to [0, ~0.85].
    Identical algorithm to lib/indicators.py KaufmanER + ASIMC ER path.
    """
    n = len(closes)
    result = np.zeros(n, dtype=np.float64)
    half_pi = np.pi / 2.0
    for i in range(window, n):
        net_change = abs(closes[i] - closes[i - window])
        total_path = 0.0
        for j in range(i - window + 1, i + 1):
            total_path += abs(closes[j] - closes[j - 1])
        if total_path > 0.0:
            er = net_change / total_path
            result[i] = np.arctan(er / 0.3) / half_pi
    return result


def compute_erp_deltas(erp_arr, window=ERP_DELTA_WINDOW):
    """1-hour rate of change for an ERP series.
    erp_arr[t] - erp_arr[t - window], both using last valid (non-nan) values.
    Matches SwingStructure._delta() logic.
    """
    n = len(erp_arr)
    delta = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        cur  = erp_arr[i]
        prev = erp_arr[i - window]
        if not (np.isnan(cur) or np.isnan(prev)):
            delta[i] = cur - prev
    return delta


# ── S5 OHLC extraction ─────────────────────────────────────────────────────────

def extract_s5_ohlc(s5_df):
    """Extract mid OHLC arrays from an S5 parquet (bid/ask columns)."""
    cols = s5_df.columns.tolist()

    def _mid(b, a):
        return ((s5_df[b].values + s5_df[a].values) / 2.0).astype(np.float64)

    if "bid_o" in cols and "ask_o" in cols:
        o = _mid("bid_o", "ask_o")
        h = _mid("bid_h", "ask_h")
        l = _mid("bid_l", "ask_l")
        c = _mid("bid_c", "ask_c")
    elif "open" in cols:
        o = s5_df["open"].values.astype(np.float64)
        h = s5_df["high"].values.astype(np.float64)
        l = s5_df["low"].values.astype(np.float64)
        c = s5_df["close"].values.astype(np.float64)
    else:
        raise ValueError(f"Unrecognised S5 columns: {cols[:10]}")

    mid = (o + c) / 2.0   # S5 mid price (used for entry/exit in evaluator)
    return o, h, l, c, mid


def aggregate_to_m5(s5_o, s5_h, s5_l, s5_c, n_m5):
    """Aggregate S5 → M5 OHLC using same logic as live curator."""
    m5_o = np.empty(n_m5, dtype=np.float64)
    m5_h = np.empty(n_m5, dtype=np.float64)
    m5_l = np.empty(n_m5, dtype=np.float64)
    m5_c = np.empty(n_m5, dtype=np.float64)
    for i in range(n_m5):
        s = i * S5_PER_M5
        e = s + S5_PER_M5
        m5_o[i] = s5_o[s]
        m5_h[i] = np.max(s5_h[s:e])
        m5_l[i] = np.min(s5_l[s:e])
        m5_c[i] = s5_c[e - 1]
    return m5_o, m5_h, m5_l, m5_c


# ── Per-pair export ────────────────────────────────────────────────────────────

def export_pair(pair, s5_data_dir, out_dir):
    """Compute all S5-cadence training columns for one pair and save parquet."""

    # ── 1. Load S5 raw data ────────────────────────────────────────────────────
    candidates = [
        s5_data_dir / f"{pair.replace('_','')}_S5_BA.parquet",
        s5_data_dir / f"{pair}_S5_BA.parquet",
        s5_data_dir / f"{pair.replace('_','')}_S5.parquet",
    ]
    s5_path = next((p for p in candidates if p.exists()), None)
    if s5_path is None:
        print(f"  {pair}: SKIP — no S5 parquet found in {s5_data_dir}")
        return False

    t0 = time.time()
    s5_df = pd.read_parquet(s5_path, engine="pyarrow")
    n_s5_raw = len(s5_df)

    # Trim to exact multiple of S5_PER_M5
    n_m5 = n_s5_raw // S5_PER_M5
    n_s5 = n_m5 * S5_PER_M5
    s5_df = s5_df.iloc[:n_s5].reset_index(drop=True)

    print(f"  {pair}: {n_s5_raw:,} S5 → {n_m5:,} M5 bars")

    s5_o, s5_h, s5_l, s5_c, s5_mid = extract_s5_ohlc(s5_df)
    del s5_df; gc.collect()

    # ── 2. Aggregate S5 → M5 ─────────────────────────────────────────────────
    m5_o, m5_h, m5_l, m5_c = aggregate_to_m5(s5_o, s5_h, s5_l, s5_c, n_m5)

    # ── 3. Compute M5-cadence indicators ─────────────────────────────────────
    # ASI-MC  — identical to ASIMC.compute() internals
    mc_d_m5, mc_dd_m5 = compute_asi_mc(m5_o, m5_h, m5_l, m5_c, n_m5)

    # ER norm — identical to KaufmanER path in curator
    er_m5 = compute_er_norm(m5_c, window=60)

    # Swing structure — identical to SwingStructure.compute() internals
    asi_m5 = compute_asi(m5_o, m5_h, m5_l, m5_c, n_m5)
    sw = compute_all_swing_features(m5_o, m5_h, m5_l, m5_c, asi_m5)

    # Pull last-valid arrays for each swing feature
    # NaN-fill then forward-fill (mirrors SwingStructure._last() behaviour)
    def _ffill(arr):
        """Forward-fill NaN — matches SwingStructure._last() per-bar."""
        out = np.array(arr, dtype=np.float64)
        last = 0.0
        for i in range(len(out)):
            if not np.isnan(out[i]):
                last = out[i]
            else:
                out[i] = last
        return out

    sb_a_m5    = _ffill(sw["sb_a"])
    hh_asi_m5  = _ffill(sw["hh_asi"])
    hl_asi_m5  = _ffill(sw["hl_asi"])
    erp_p_m5   = _ffill(sw["erp_p"])
    erp_a_m5   = _ffill(sw["erp_a"])

    # ── 4. Forward-fill M5 → S5 cadence ──────────────────────────────────────
    # Each M5 value held for S5_PER_M5=12 S5 bars (same as curator ZMQ hold)
    # NOTE: erp_p/erp_a/d_erp_p/d_erp_a EXCLUDED — raw values reach ±62K
    # (stale pivot reference + unbounded ASI = unusable NN inputs).
    def _fwd(m5_arr):
        out = np.empty(n_s5, dtype=np.float32)
        for i in range(n_m5):
            out[i * S5_PER_M5:(i + 1) * S5_PER_M5] = m5_arr[i]
        return out

    m5_idx_s5 = np.repeat(np.arange(n_m5, dtype=np.int32), S5_PER_M5)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair}_s5_training.parquet"

    pd.DataFrame({
        "s5_mid":     s5_mid[:n_s5].astype(np.float32),
        "mc_d_a":     _fwd(mc_d_m5),
        "mc_dd_a":    _fwd(mc_dd_m5),
        "er_norm":    _fwd(er_m5),
        "sb_a":       _fwd(sb_a_m5),
        "hh_asi":     _fwd(hh_asi_m5),
        "hl_asi":     _fwd(hl_asi_m5),
        "m5_bar_idx": m5_idx_s5,
    }).to_parquet(out_path, engine="pyarrow", compression="zstd")

    warmup = 200
    elapsed = time.time() - t0
    print(f"    → saved {out_path.name}  ({n_s5:,} S5 rows, {elapsed:.1f}s)")
    print(f"    mc_d_a  [{mc_d_m5[warmup:].min():.3f}, {mc_d_m5[warmup:].max():.3f}]  "
          f"mc_dd_a [{mc_dd_m5[warmup:].min():.3f}, {mc_dd_m5[warmup:].max():.3f}]")
    print(f"    er_norm [{er_m5[warmup:].min():.3f}, {er_m5[warmup:].max():.3f}]  "
          f"sb_a    [{sb_a_m5[warmup:].min():.2f}, {sb_a_m5[warmup:].max():.2f}]")
    print(f"    hh_asi  frac={hh_asi_m5[warmup:].mean():.2f}  "
          f"hl_asi  frac={hl_asi_m5[warmup:].mean():.2f}")

    del m5_o, m5_h, m5_l, m5_c, asi_m5, sw
    gc.collect()
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export S5-cadence training parquets")
    parser.add_argument("--pairs", nargs="+", default=ALL_PAIRS,
                        help="Pairs to export (default: all 12)")
    parser.add_argument("--s5-data-dir", default=str(S5_DATA_DIR),
                        help="Directory containing raw S5 BA parquets")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Output directory for s5_training parquets")
    args = parser.parse_args()

    s5_dir  = Path(args.s5_data_dir)
    out_dir = Path(args.out_dir)

    print("=" * 65)
    print("  Export S5-cadence training data (curator-identical indicators)")
    print(f"  S5 source : {s5_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Pairs     : {args.pairs}")
    print("=" * 65)
    print()

    ok = 0
    t_all = time.time()
    for pair in args.pairs:
        print(f"[{pair}]")
        if export_pair(pair, s5_dir, out_dir):
            ok += 1
        print()

    total = time.time() - t_all
    print(f"Done: {ok}/{len(args.pairs)} pairs in {total:.0f}s ({total/60:.1f} min)")
    print(f"Output: {out_dir}/{{PAIR}}_s5_training.parquet")
    print()
    print("Verify identity with live curator:")
    print("  python3 export_s5_training_data.py --pairs EUR_GBP")
    print("  # Compare a few rows against curator ZMQ output on same dates")


if __name__ == "__main__":
    main()
