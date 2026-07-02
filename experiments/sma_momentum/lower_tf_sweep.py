"""
Lower-TF sweep for SMA momentum confluence signal.

Tests whether the H1+M30 SMA momentum signal scales down to M30+M15 and M15+M5.
For each TF combo × SMA period × lag triplet × TP level, runs 12-pair portfolio
OOS validation with IS 3-fold gating.

Usage:
    python lower_tf_sweep.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/path/to/projects/fx-core/data/m5_ba")
RESULTS_DIR = Path("/path/to/projects/fx-core/research/experiments/sma_momentum/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = RESULTS_DIR / "lower_tf_sweep.csv"

# ── Pairs & pip sizes ──────────────────────────────────────────────────────────
PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "EUR_GBP", "AUD_USD",
    "NZD_JPY", "CHF_JPY", "NZD_USD", "CAD_JPY",
]
JPY_PAIRS = {"GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY", "NZD_JPY", "CHF_JPY", "CAD_JPY"}

def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001

# ── Sweep parameters ───────────────────────────────────────────────────────────
TF_COMBOS = [
    ("1h",  "30min", "H1+M30"),
    ("30min", "15min", "M30+M15"),
    ("15min", "5min",  "M15+M5"),
]
SMA_PERIODS  = [8, 16, 22]
LAG_TRIPLETS = [(3, 5, 8), (5, 8, 10), (8, 10, 15), (3, 8, 15)]
TP_LEVELS    = [5, 10, 15, 20]

IS_FRAC = 0.70   # first 70% = IS, rest = OOS
N_FOLDS = 3      # IS 3-fold walk-forward

# ── Signal builder ─────────────────────────────────────────────────────────────
def build_signal(df: pd.DataFrame, sma_n: int, lags: tuple, tf1: str, tf2: str) -> pd.Series:
    """
    Confluence momentum signal. Returns +1 (long) / -1 (short) / 0 (flat).
    Requires df to have a tz-aware DatetimeIndex.
    """
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        sma = rs.rolling(sma_n, min_periods=sma_n).mean().shift(1)
        sma = sma.reindex(df.index, method="ffill")
        for k in lags:
            moms.append(sma - sma.shift(k))

    all_moms = pd.concat(moms, axis=1)
    n_ind = len(moms)   # 2 TFs × 3 lags = 6

    score = (all_moms > 0).sum(axis=1)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score >= n_ind] = np.int8(1)
    sig[score <= 0]     = np.int8(-1)
    return sig

# ── Trade simulator ────────────────────────────────────────────────────────────
def simulate_tp(df: pd.DataFrame, sig: pd.Series, pip: float,
                tp_pips: float, sp_gate: float) -> np.ndarray:
    """
    Pure-Python trade loop (matches spec exactly).
    Returns np.array of per-trade P&L in pips (net of spread).
    """
    bid  = df["bid_c"].values.astype(np.float64)
    ask  = df["ask_c"].values.astype(np.float64)
    mid  = df["close"].values.astype(np.float64)
    sp   = (ask - bid) / pip
    s    = sig.values

    pnls: list = []
    in_trade = False
    dir_     = 0
    ep       = 0.0

    for i in range(1, len(df)):
        if in_trade:
            if (mid[i] - ep) / pip * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - ep) / pip * dir_ - sp[i])
                in_trade = False
        else:
            nd = s[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep      = ask[i] if nd == 1 else bid[i]
                dir_    = nd
                in_trade = True

    return np.array(pnls, dtype=np.float64)

# ── IS 3-fold validation ───────────────────────────────────────────────────────
def is_3fold_pass(df_is: pd.DataFrame, sig_is: pd.Series, pip: float,
                  tp_pips: float, sp_gate: float) -> bool:
    """Returns True if all 3 IS folds have positive p/d."""
    n = len(df_is)
    fold_size = n // N_FOLDS
    if fold_size == 0:
        return False

    all_pos = True
    for f in range(N_FOLDS):
        start = f * fold_size
        end   = (f + 1) * fold_size if f < N_FOLDS - 1 else n
        df_f  = df_is.iloc[start:end]
        sig_f = sig_is.iloc[start:end]
        pnls  = simulate_tp(df_f, sig_f, pip, tp_pips, sp_gate)
        n_bars_day = (len(df_f) / ((df_f.index[-1] - df_f.index[0]).total_seconds() / 86400)) if len(df_f) > 1 else 1
        total_days = len(df_f) / n_bars_day if n_bars_day else 1
        pd_val = pnls.sum() / max(total_days, 1.0)
        if pd_val <= 0:
            all_pos = False
            break

    return all_pos

# ── Per-pair evaluation ────────────────────────────────────────────────────────
def eval_pair(df: pd.DataFrame, pair: str, sma_n: int, lags: tuple,
              tf1: str, tf2: str, tp_pips: float):
    """
    Returns dict with IS and OOS metrics for one pair, or None if insufficient data.
    """
    pip = pip_size(pair)

    # IS/OOS split on rows
    split = int(len(df) * IS_FRAC)
    df_is  = df.iloc[:split]
    df_oos = df.iloc[split:]

    if len(df_is) < 500 or len(df_oos) < 100:
        return None

    # IS spread gate (p90 of IS spread)
    sp_is = (df_is["ask_c"] - df_is["bid_c"]) / pip
    sp_gate = float(np.percentile(sp_is.values, 90))

    # Build signal on FULL df so resampling gets context; then slice
    sig_full = build_signal(df, sma_n, lags, tf1, tf2)
    sig_is   = sig_full.iloc[:split]
    sig_oos  = sig_full.iloc[split:]

    # IS 3-fold
    is3 = is_3fold_pass(df_is, sig_is, pip, tp_pips, sp_gate)

    # OOS
    pnls_oos  = simulate_tp(df_oos, sig_oos, pip, tp_pips, sp_gate)
    days_oos  = (df_oos.index[-1] - df_oos.index[0]).total_seconds() / 86400
    if days_oos <= 0:
        return None

    oos_pd = pnls_oos.sum() / days_oos
    oos_td = len(pnls_oos) / days_oos
    oos_pos = bool(oos_pd > 0)

    return {
        "is3":    is3,
        "oos_pd": oos_pd,
        "oos_td": oos_td,
        "oos_pos": oos_pos,
    }

# ── Load all pair data once ────────────────────────────────────────────────────
def load_data() -> dict:
    print("Loading parquets...", flush=True)
    data = {}
    for pair in PAIRS:
        path = DATA_DIR / f"{pair}_M5_BA.parquet"
        df = pd.read_parquet(path)
        # Set tz-aware DatetimeIndex
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[pair] = df
        print(f"  {pair}: {len(df):,} bars  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    print()
    return data

# ── Main sweep ─────────────────────────────────────────────────────────────────
def main():
    data = load_data()

    records = []
    total_configs = len(TF_COMBOS) * len(SMA_PERIODS) * len(LAG_TRIPLETS) * len(TP_LEVELS)
    done = 0
    t0 = time.time()

    baseline_key = ("H1+M30", 16, (8, 10, 15), 20)
    baseline_row = None

    for tf1, tf2, tf_label in TF_COMBOS:
        for sma_n in SMA_PERIODS:
            for lags in LAG_TRIPLETS:
                for tp in TP_LEVELS:
                    done += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / done) * (total_configs - done) if done > 1 else 0
                    print(f"[{done:3d}/{total_configs}]  {tf_label}  SMA={sma_n}  "
                          f"lags={lags}  TP={tp}p  "
                          f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)",
                          flush=True)

                    portfolio_pd = 0.0
                    portfolio_td = 0.0
                    n_is3 = 0
                    n_pos = 0

                    for pair in PAIRS:
                        res = eval_pair(data[pair], pair, sma_n, lags, tf1, tf2, tp)
                        if res is None:
                            continue
                        portfolio_pd += res["oos_pd"]
                        portfolio_td += res["oos_td"]
                        if res["is3"]:
                            n_is3 += 1
                        if res["oos_pos"]:
                            n_pos += 1

                    row = {
                        "tf":       tf_label,
                        "sma_n":    sma_n,
                        "lags":     str(lags),
                        "tp_pips":  tp,
                        "port_pd":  round(portfolio_pd, 2),
                        "port_td":  round(portfolio_td, 3),
                        "n_is3":    n_is3,
                        "n_pos":    n_pos,
                    }
                    records.append(row)

                    is_baseline = (
                        tf_label == baseline_key[0] and
                        sma_n    == baseline_key[1] and
                        lags     == baseline_key[2] and
                        tp       == baseline_key[3]
                    )
                    if is_baseline:
                        baseline_row = row

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved → {OUT_CSV}")

    # ── Print full table ───────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("FULL RESULTS  (sorted by port_pd desc)")
    print("=" * 85)
    hdr = f"{'TF':<10} {'SMA':>5} {'Lags':<14} {'TP':>4}  {'port_pd':>8}  {'port_td':>8}  {'n_is3':>6}  {'n_pos':>6}"
    print(hdr)
    print("-" * 85)
    df_sorted = df_res.sort_values("port_pd", ascending=False)
    for _, r in df_sorted.iterrows():
        flag = ""
        if (r["tf"] == baseline_key[0] and r["sma_n"] == baseline_key[1] and
                r["lags"] == str(baseline_key[2]) and r["tp_pips"] == baseline_key[3]):
            flag = "  ← BASELINE"
        print(f"{r['tf']:<10} {r['sma_n']:>5} {r['lags']:<14} {r['tp_pips']:>4}  "
              f"{r['port_pd']:>8.2f}  {r['port_td']:>8.3f}  {r['n_is3']:>6}  {r['n_pos']:>6}{flag}")

    # ── Print top-20 (IS3 ≥ 8) ────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("TOP 20 CONFIGS  (IS3 ≥ 8/12, sorted by port_pd desc)")
    print("=" * 85)
    df_top = df_res[df_res["n_is3"] >= 8].sort_values("port_pd", ascending=False).head(20)
    if df_top.empty:
        print("  (no configs with IS3 ≥ 8)")
    else:
        print(hdr)
        print("-" * 85)
        for _, r in df_top.iterrows():
            flag = ""
            if (r["tf"] == baseline_key[0] and r["sma_n"] == baseline_key[1] and
                    r["lags"] == str(baseline_key[2]) and r["tp_pips"] == baseline_key[3]):
                flag = "  ← BASELINE"
            print(f"{r['tf']:<10} {r['sma_n']:>5} {r['lags']:<14} {r['tp_pips']:>4}  "
                  f"{r['port_pd']:>8.2f}  {r['port_td']:>8.3f}  {r['n_is3']:>6}  {r['n_pos']:>6}{flag}")

    # ── Baseline highlight ────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("BASELINE  (H1+M30, SMA=16, lags=(8,10,15), TP=20p)")
    print("=" * 85)
    if baseline_row:
        for k, v in baseline_row.items():
            print(f"  {k:<12}: {v}")
    else:
        print("  (baseline config not found in results)")

    # ── TF comparison summary ─────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("TF COMPARISON SUMMARY  (best per TF combo by port_pd, IS3 ≥ 6)")
    print("=" * 85)
    print(hdr)
    print("-" * 85)
    for tf_label in [t[2] for t in TF_COMBOS]:
        sub = df_res[(df_res["tf"] == tf_label) & (df_res["n_is3"] >= 6)]
        if sub.empty:
            sub = df_res[df_res["tf"] == tf_label]
        best = sub.sort_values("port_pd", ascending=False).iloc[0]
        flag = ""
        if (best["tf"] == baseline_key[0] and best["sma_n"] == baseline_key[1] and
                best["lags"] == str(baseline_key[2]) and best["tp_pips"] == baseline_key[3]):
            flag = "  ← BASELINE"
        print(f"{best['tf']:<10} {best['sma_n']:>5} {best['lags']:<14} {best['tp_pips']:>4}  "
              f"{best['port_pd']:>8.2f}  {best['port_td']:>8.3f}  {best['n_is3']:>6}  {best['n_pos']:>6}{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
