#!/usr/bin/env python3
"""
Stage 0: Heiken Ashi Feasibility Scan
======================================
Pure statistics — does HA direction predict anything?
Kill the idea in 35 minutes if not.

Tests:
  1. After HA color flip (bearish→bullish or vice versa), what is the
     average forward return over the next N bars?
  2. During a streak of same-color bars, what is the average per-bar return?
  3. Are these returns large enough to overcome spread?

Pass criteria: conditional return spread > 0.5 pips on at least 2 timeframes.
"""

import sys
import os
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "scalper_parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PAIRS_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}

PAIRS_SPREAD_PIPS = {
    "EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
    "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
    "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
    "NZD_USD": 1.5, "EUR_GBP": 1.4,
}

# Forward look windows (in bars of the given timeframe)
FORWARD_WINDOWS = [5, 10, 20, 50]

# Timeframes to test
TIMEFRAMES = {
    "S5": None,        # raw — no resample
    "M5": "5min",
    "H1": "1h",
}


# ── HA computation (vectorized) ────────────────────────────────────────────

def compute_ha(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """Compute Heiken Ashi OHLC. Returns ha_o, ha_h, ha_l, ha_c arrays."""
    n = len(o)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n, dtype=np.float64)

    # First bar: ha_o = (o + c) / 2
    ha_o[0] = (o[0] + c[0]) / 2.0
    # Iterate for ha_o (depends on previous ha_o and ha_c)
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0

    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(l, np.minimum(ha_o, ha_c))

    return ha_o, ha_h, ha_l, ha_c


def ha_direction(ha_o: np.ndarray, ha_c: np.ndarray) -> np.ndarray:
    """Returns +1 (bullish) or -1 (bearish) per bar."""
    return np.where(ha_c >= ha_o, 1.0, -1.0)


# ── Resample S5 to higher TF ──────────────────────────────────────────────

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample bid OHLC to a higher timeframe."""
    ts = pd.to_datetime(df["timestamp"])
    tmp = pd.DataFrame({
        "o": df["bid_o"].values,
        "h": df["bid_h"].values,
        "l": df["bid_l"].values,
        "c": df["bid_c"].values,
        "mid_c": ((df["bid_c"].values + df["ask_c"].values) / 2.0),
    }, index=ts)

    resampled = tmp.resample(rule).agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "mid_c": "last",
    }).dropna()

    return resampled


# ── Analysis per pair×timeframe ────────────────────────────────────────────

def analyze(pair: str, tf_name: str, o: np.ndarray, h: np.ndarray,
            l: np.ndarray, c: np.ndarray, mid_c: np.ndarray, pip: float,
            spread_pips: float) -> dict:
    """
    Compute HA, find flips, measure forward returns in pips.

    Returns dict with results for this pair×tf.
    """
    ha_o, ha_h, ha_l, ha_c = compute_ha(o, h, l, c)
    d = ha_direction(ha_o, ha_c)

    # ── Flips: where direction changes ──
    flips = np.where(d[1:] != d[:-1])[0] + 1  # index of the NEW bar after flip
    bull_flips = flips[d[flips] == 1.0]   # flipped to bullish
    bear_flips = flips[d[flips] == -1.0]  # flipped to bearish

    n = len(mid_c)
    result = {
        "pair": pair,
        "tf": tf_name,
        "n_bars": n,
        "n_bull_flips": len(bull_flips),
        "n_bear_flips": len(bear_flips),
    }

    # Forward returns after flips (in pips)
    for w in FORWARD_WINDOWS:
        # Bullish flip: expect price to go UP → long return
        valid_bull = bull_flips[bull_flips + w < n]
        if len(valid_bull) > 0:
            fwd_bull = (mid_c[valid_bull + w] - mid_c[valid_bull]) / pip
        else:
            fwd_bull = np.array([0.0])

        # Bearish flip: expect price to go DOWN → short return (negate)
        valid_bear = bear_flips[bear_flips + w < n]
        if len(valid_bear) > 0:
            fwd_bear = (mid_c[valid_bear] - mid_c[valid_bear + w]) / pip
        else:
            fwd_bear = np.array([0.0])

        # Combined: "go with the flip" return
        all_returns = np.concatenate([fwd_bull, fwd_bear])

        result[f"fwd_{w}_mean_pips"] = float(np.mean(all_returns))
        result[f"fwd_{w}_median_pips"] = float(np.median(all_returns))
        result[f"fwd_{w}_std_pips"] = float(np.std(all_returns))
        result[f"fwd_{w}_n"] = len(all_returns)
        result[f"fwd_{w}_pct_positive"] = float(np.mean(all_returns > 0) * 100)
        # Net after spread
        result[f"fwd_{w}_net_pips"] = float(np.mean(all_returns) - spread_pips)

    # ── Streak analysis: per-bar return during same-color streaks ──
    # During bullish streak: per-bar mid return should be positive
    bull_mask = d == 1.0
    bear_mask = d == -1.0

    # Per-bar return (bar-to-bar mid_c change in pips)
    bar_ret = np.diff(mid_c) / pip  # length n-1

    bull_bar_ret = bar_ret[bull_mask[1:]]  # bars that are bullish
    bear_bar_ret = bar_ret[bear_mask[1:]]  # bars that are bearish

    result["streak_bull_mean_pip_per_bar"] = float(np.mean(bull_bar_ret)) if len(bull_bar_ret) > 0 else 0.0
    result["streak_bear_mean_pip_per_bar"] = float(np.mean(bear_bar_ret)) if len(bear_bar_ret) > 0 else 0.0
    # Separation: bull should be positive, bear should be negative
    result["streak_separation_pips"] = result["streak_bull_mean_pip_per_bar"] - result["streak_bear_mean_pip_per_bar"]

    # ── Streak length distribution ──
    streak_lengths = []
    current_len = 1
    for i in range(1, len(d)):
        if d[i] == d[i - 1]:
            current_len += 1
        else:
            streak_lengths.append(current_len)
            current_len = 1
    streak_lengths.append(current_len)
    streak_arr = np.array(streak_lengths, dtype=np.float64)
    result["streak_mean_bars"] = float(np.mean(streak_arr))
    result["streak_median_bars"] = float(np.median(streak_arr))
    result["streak_max_bars"] = int(np.max(streak_arr))

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pair, pip in PAIRS_PIP.items():
        spread_pips = PAIRS_SPREAD_PIPS[pair]
        # Parquet uses OANDA naming without underscore
        parquet_name = pair.replace("_", "") + "_S5_BA.parquet"
        parquet_path = DATA_DIR / parquet_name
        if not parquet_path.exists():
            print(f"  SKIP {pair}: {parquet_path} not found")
            continue

        print(f"Loading {pair}...", end=" ", flush=True)
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        print(f"{len(df):,} S5 bars")

        # Compute mid close for return calculations
        df["mid_c"] = (df["bid_c"] + df["ask_c"]) / 2.0

        for tf_name, rule in TIMEFRAMES.items():
            if rule is None:
                # S5 raw
                o = df["bid_o"].values.astype(np.float64)
                h = df["bid_h"].values.astype(np.float64)
                l = df["bid_l"].values.astype(np.float64)
                c = df["bid_c"].values.astype(np.float64)
                mid_c = df["mid_c"].values.astype(np.float64)
            else:
                resampled = resample_ohlc(df, rule)
                o = resampled["o"].values.astype(np.float64)
                h = resampled["h"].values.astype(np.float64)
                l = resampled["l"].values.astype(np.float64)
                c = resampled["c"].values.astype(np.float64)
                mid_c = resampled["mid_c"].values.astype(np.float64)

            res = analyze(pair, tf_name, o, h, l, c, mid_c, pip, spread_pips)
            all_results.append(res)
            print(f"  {tf_name}: flips={res['n_bull_flips']+res['n_bear_flips']:,}, "
                  f"fwd10_mean={res['fwd_10_mean_pips']:.3f}p, "
                  f"fwd10_net={res['fwd_10_net_pips']:.3f}p, "
                  f"streak_sep={res['streak_separation_pips']:.4f}p")

        del df
        gc.collect()

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "stage0_results.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults saved to {csv_path}")

    # ── Summary table ──
    print("\n" + "=" * 100)
    print("STAGE 0 SUMMARY: HA Feasibility Scan")
    print("=" * 100)

    # Aggregate by timeframe
    for tf in TIMEFRAMES:
        tf_rows = results_df[results_df["tf"] == tf]
        print(f"\n── {tf} ──")
        print(f"  {'Pair':<10} {'Flips':>8} {'Fwd5_net':>10} {'Fwd10_net':>10} "
              f"{'Fwd20_net':>10} {'Fwd50_net':>10} {'Strk_sep':>10} {'Fwd10_%+':>8}")
        print(f"  {'-'*8:<10} {'-'*6:>8} {'-'*8:>10} {'-'*8:>10} "
              f"{'-'*8:>10} {'-'*8:>10} {'-'*8:>10} {'-'*6:>8}")
        for _, row in tf_rows.iterrows():
            flips = row["n_bull_flips"] + row["n_bear_flips"]
            print(f"  {row['pair']:<10} {flips:>8,} {row['fwd_5_net_pips']:>10.3f} "
                  f"{row['fwd_10_net_pips']:>10.3f} {row['fwd_20_net_pips']:>10.3f} "
                  f"{row['fwd_50_net_pips']:>10.3f} {row['streak_separation_pips']:>10.4f} "
                  f"{row['fwd_10_pct_positive']:>7.1f}%")

        # Timeframe average
        avg_fwd10_net = tf_rows["fwd_10_net_pips"].mean()
        avg_fwd20_net = tf_rows["fwd_20_net_pips"].mean()
        avg_streak_sep = tf_rows["streak_separation_pips"].mean()
        n_positive_fwd10 = (tf_rows["fwd_10_net_pips"] > 0).sum()
        print(f"\n  AVG fwd10_net={avg_fwd10_net:.3f}p, fwd20_net={avg_fwd20_net:.3f}p, "
              f"streak_sep={avg_streak_sep:.4f}p, pairs_net_positive={n_positive_fwd10}/12")

    # ── PASS/FAIL decision ──
    print("\n" + "=" * 100)
    print("PASS CRITERIA: conditional return spread > 0.5 pips on >= 2 timeframes")
    print("=" * 100)

    passing_tfs = []
    for tf in TIMEFRAMES:
        tf_rows = results_df[results_df["tf"] == tf]
        # Best forward window's average net return across pairs
        best_net = max(
            tf_rows[f"fwd_{w}_net_pips"].mean()
            for w in FORWARD_WINDOWS
        )
        # Also check streak separation as a second metric
        avg_streak_sep = tf_rows["streak_separation_pips"].mean()

        passed = best_net > 0.5 or avg_streak_sep > 0.5
        status = "🟢 PASS" if passed else "🔴 FAIL"
        print(f"  {tf}: best_avg_net={best_net:.3f}p, streak_sep={avg_streak_sep:.4f}p → {status}")
        if passed:
            passing_tfs.append(tf)

    print()
    if len(passing_tfs) >= 2:
        print("🟢 STAGE 0 PASSED — proceed to Stage 1")
        print(f"   Passing timeframes: {passing_tfs}")
    elif len(passing_tfs) == 1:
        print("🟡 STAGE 0 MARGINAL — only 1 timeframe passed, proceed with caution")
        print(f"   Passing timeframe: {passing_tfs}")
    else:
        print("🔴 STAGE 0 FAILED — HA direction has no predictive value. STOP.")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
