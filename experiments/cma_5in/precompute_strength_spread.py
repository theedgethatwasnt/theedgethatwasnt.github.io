#!/usr/bin/env python3
"""Precompute H1 StrengthSpread aligned to M5 timestamps for PINN-CMA experiment.

CAUSALITY — DO NOT CHANGE WITHOUT RE-READING THIS DOCSTRING
===========================================================

OANDA H1 timestamp convention: a bar with timestamp T spans [T, T+1h]. Its
CLOSE price, high, low, and therefore its return (close[T]/close[T-1]) is not
known until time T+1h.

    H1 bar @ T=22:00 → open 1.20427, close 1.20438.
    M5 close at 22:00 = 1.20436 (beginning of H1 bar).
    M5 close at 23:00 = 1.20438 (end of H1 bar, = H1.close[T=22:00]).

So spread[T] uses data that only exists by T+1h.

If we did `merge_asof(m5, ss_h1, direction='backward')` naively, then at M5
timestamp 22:05 we would fetch ss_h1[T=22:00] — which contains 55 min of future
information. That is the 4-12-style lookahead bug.

Fix: shift the ss_h1 index forward by +1h so that the "effective timestamp" is
the moment the H1 bar finishes (when spread is first knowable). After the shift,
M5 @ 22:05 will match ss_h1 shifted from T=21:00 (effective ts=22:00) — which
uses close[21:00] = actual 22:00 price. Causal.

Regression test lives at the bottom of this file.

Output: data/pinn_features/{pair}_ss_h1.npy — float64 array aligned 1:1 with
the unified_indicators parquet rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
CSI_FACTORS = Path.home() / "projects" / "csi_factor_study" / "results" / "factors_H1.parquet"
M5_DIR = PROJECT / "data" / "unified_indicators"
OUT_DIR = PROJECT / "data" / "pinn_features"

PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading H1 factors from {CSI_FACTORS}")
    h1_all = pd.read_parquet(CSI_FACTORS)
    print(f"  H1 shape: {h1_all.shape}, range: {h1_all.index.min()} → {h1_all.index.max()}")

    for pair in PAIRS:
        # Extract H1 StrengthSpread for this pair
        ss_col = ("Spread", pair)
        if ss_col not in h1_all.columns:
            print(f"  SKIP {pair}: no Spread column in H1 factors")
            continue

        ss_h1 = h1_all[ss_col].dropna().reset_index()
        ss_h1.columns = ["timestamp", "ss_h1"]
        ss_h1 = ss_h1.sort_values("timestamp")

        # CAUSALITY FIX: shift the effective timestamp forward by +1h because
        # the H1 bar with timestamp T only closes (and therefore ss_h1[T] is
        # only knowable) at T+1h. See module docstring.
        ss_h1["timestamp"] = ss_h1["timestamp"] + pd.Timedelta(hours=1)

        # Load M5 timestamps
        m5_path = M5_DIR / f"{pair}_unified.parquet"
        if not m5_path.exists():
            print(f"  SKIP {pair}: no M5 unified parquet")
            continue
        m5 = pd.read_parquet(m5_path, columns=["timestamp"])
        m5 = m5.sort_values("timestamp")

        # merge_asof: at M5 ts t, return ss_h1 with shifted_ts <= t. After the
        # +1h shift this means "the most recent H1 bar whose CLOSE has
        # already occurred by t" — no future leakage.
        merged = pd.merge_asof(
            m5, ss_h1,
            on="timestamp",
            direction="backward",
        )

        ss_arr = merged["ss_h1"].values.astype(np.float64)

        # Clip to [-3, +3] and normalize to [-1, 1]
        ss_arr = np.clip(ss_arr, -3.0, 3.0) / 3.0

        # NaN fill: M5 bars before first H1 bar get NaN → fill with 0 (neutral)
        nan_count = np.isnan(ss_arr).sum()
        if nan_count > 0:
            ss_arr = np.nan_to_num(ss_arr, nan=0.0)

        out_path = OUT_DIR / f"{pair}_ss_h1.npy"
        np.save(out_path, ss_arr)

        print(f"  {pair}: {len(ss_arr):,} bars, NaN-filled={nan_count:,}, "
              f"range=[{ss_arr.min():+.3f}, {ss_arr.max():+.3f}], "
              f"std={ss_arr.std():.3f}")

    # ── Causality verification (regression test — replaces the buggy one) ──
    #
    # The correct invariant: at M5 timestamp t, the ss_h1 value assigned MUST
    # be computable from H1 closes at bars whose CLOSE time (= H1 open
    # timestamp + 1h) is <= t. If the feature is sourced from an H1 bar whose
    # close is still in the future relative to t, that's the 4-12 bug.
    print("\n── Causality check (v2 post-4-17 RCA) ──")
    pair = "EUR_JPY"
    ss_src = h1_all[("Spread", pair)].dropna().reset_index()
    ss_src.columns = ["timestamp", "ss_h1"]
    ss_src = ss_src.sort_values("timestamp").reset_index(drop=True)
    # Apply the same +1h shift as in the production path
    ss_src_shifted = ss_src.copy()
    ss_src_shifted["timestamp"] = ss_src_shifted["timestamp"] + pd.Timedelta(hours=1)

    m5 = pd.read_parquet(M5_DIR / f"{pair}_unified.parquet", columns=["timestamp"])
    m5 = m5.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        m5, ss_src_shifted,
        on="timestamp",
        direction="backward",
    )

    # The only way to detect a leak: the H1 bar whose ss_h1 value matches what
    # we served at M5 ts t must have had its CLOSE time (= H1 open + 1h) <= t.
    np.random.seed(42)
    sample_idx = np.random.choice(len(merged), min(500, len(merged)), replace=False)
    violations = 0
    for idx in sample_idx:
        m5_ts = merged.iloc[idx]["timestamp"]
        ss_val = merged.iloc[idx]["ss_h1"]
        if pd.isna(ss_val):
            continue
        # Find the original (unshifted) H1 bar(s) with this spread value
        match = ss_src[np.isclose(ss_src["ss_h1"], ss_val, atol=1e-12, equal_nan=False)]
        if len(match) == 0:
            continue
        # Every matching H1 bar must have (open_ts + 1h) <= m5_ts.
        close_times = match["timestamp"] + pd.Timedelta(hours=1)
        if (close_times > m5_ts).all():
            violations += 1  # every source bar closes AFTER our M5 ts → leak

    print(f"  {pair}: checked {len(sample_idx)} random M5 bars, violations={violations}")
    if violations == 0:
        print("  PASS: every served ss_h1 value originates from an H1 bar")
        print("        whose close time is <= the requesting M5 timestamp.")
    else:
        print(f"  FAIL: {violations} lookahead violations detected!")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
