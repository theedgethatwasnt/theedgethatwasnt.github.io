#!/usr/bin/env python3
"""
stage1_data.py — momentum_confirm Stage 1 D1 loader. HARD SEGMENT GUARD.

PREREGISTRATION.md: "Stage 1 (backward out-of-sample): OANDA D1 deep history ~=2005->2020-10 —
never seen by the factor sweep or any prior experiment here." "Segment discipline: Stage 1
loaders hard-assert timestamps <= 2020-10-31. The observation window (2020-11->2024-08) and the
sealed window (2024-08-30->2026-05-21) must not be loaded at all."

STAGE1_END is pushed down to pyarrow at parquet-read time (filters=), exactly like
fx_factors/is_data.py's IS_END pattern (R6 — same guard shape, new boundary), PLUS a second,
independent post-read assertion.

Source data: data/d1_deep/<PAIR>_D1.parquet — OANDA native D-granularity candles (mid OHLC
only, no historical bid/ask this far back; documented R9 divergence from the M5-BA-aggregated
D1 bars fx_factors uses in its own 2020-11+ window — both are legitimate OANDA D1 conventions,
NY-17:00-anchored by default, but not bar-for-bar identical in edge cases). Spread cost for
Stage 1 is NOT read from bid_c/ask_c (absent) — median_spread.build_ba_columns() injects the
fx_factors MEASURED MEDIAN per-pair round-trip spread as a CONSTANT synthetic bid_c/ask_c
offset around every bar's mid close, so rebalance_engine.py's existing (ask_c-bid_c)/pip cost
logic runs completely unmodified on the new data (R9, documented, not a code change).

Only the 7 REQUIRED_PAIRS_FOR_INDEX (currency_index.py) are loaded — the momentum rule under
test never trades EUR_GBP or the other 5 pairs outside that set.
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STAGE1_END = pd.Timestamp("2020-11-01T00:00:00", tz="UTC")  # hard cutoff: bars <= 2020-10-31

# fx_factors-measured median D1 round-trip spread (pips), computed from data/m5_ba (bars.m5_to_d1,
# the SAME aggregation fx_factors itself uses) over its full available history 2020-11-11 ->
# 2026-05-19 — "Per-pair spread as fx_factors (measured medians)" (PREREGISTRATION.md "Costs").
# Reproduce via: research/experiments/momentum_confirm/median_spread.csv (committed alongside).
MEDIAN_SPREAD_PIPS = {
    "EUR_USD": 1.9,
    "GBP_USD": 2.9,
    "AUD_USD": 1.8,
    "NZD_USD": 2.2,
    "USD_JPY": 2.6,
    "CAD_JPY": 3.7,
    "CHF_JPY": 5.3,
}


def pip_of(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


def load_pair_stage1_d1(pair, data_dir):
    """One pair's deep-D1 parquet, hard-filtered to the Stage-1 segment, with synthetic
    bid_c/ask_c columns injected at the fixed fx_factors-measured median spread (round-trip
    pips) for that pair — see module docstring."""
    path = os.path.join(data_dir, f"{pair}_D1.parquet")
    df = pd.read_parquet(path, filters=[("timestamp", "<", STAGE1_END)])
    if len(df) == 0:
        raise RuntimeError(f"{pair}: 0 Stage-1 rows loaded from {path} — check path/filter")
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].max() < STAGE1_END, (
        f"{pair}: SEGMENT LEAK — loaded max ts {df['timestamp'].max()} >= STAGE1_END {STAGE1_END}"
    )
    pip = pip_of(pair)
    half_spread_price = (MEDIAN_SPREAD_PIPS[pair] * pip) / 2.0
    df["bid_c"] = df["close"] - half_spread_price
    df["ask_c"] = df["close"] + half_spread_price
    return df


def load_all_pairs_stage1(data_dir, pairs):
    pair_d1 = {}
    for pair in pairs:
        df = load_pair_stage1_d1(pair, data_dir)
        pair_d1[pair] = df
        print(f"[{pair}] {len(df)} Stage-1 D1 bars, {df['timestamp'].min()} -> {df['timestamp'].max()} "
              f"(spread={MEDIAN_SPREAD_PIPS[pair]}p)", flush=True)
    return pair_d1


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent.parent / "data" / "d1_deep")
    from currency_index import REQUIRED_PAIRS_FOR_INDEX  # noqa: E402  (fx_factors, added to path by caller)
    pd_ = load_all_pairs_stage1(data_dir, REQUIRED_PAIRS_FOR_INDEX)
    for p, df in pd_.items():
        print(p, len(df), df["timestamp"].min(), "->", df["timestamp"].max())
