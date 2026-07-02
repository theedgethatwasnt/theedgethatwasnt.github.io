#!/usr/bin/env python3
"""
Export training parquets using IDENTICAL code path as live curator.

This is the ONLY correct way to generate training data.
Uses the same ASIMC class from lib/indicators.py that the curator uses live.
Feeds M5 bars sequentially, exactly as the curator receives them.

Any export script that uses a different code path WILL cause train/live mismatch.

Usage:
  python3 export_curator_identical.py
  python3 export_curator_identical.py --pair EUR_JPY
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.indicators import ASIMC

DATA_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"
S5_DIR = PROJECT_ROOT / "data" / "scalper_parquet"  # Raw S5 data
OUTPUT_DIR = PROJECT_ROOT / "data" / "curator_identical"

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}


def export_pair_curator_identical(pair: str):
    """
    Feed M5 bars to ASIMC class one at a time, exactly as curator does.
    Capture mc_d, mc_dd at each bar.
    """
    # Load existing parquet for M5 OHLC (already resampled from S5)
    asi_path = DATA_DIR / f"{pair}_asi_mc.parquet"
    if not asi_path.exists():
        print(f"  {pair}: no source parquet")
        return

    df = pd.read_parquet(asi_path, engine="pyarrow")

    # Use the EXACT same ASIMC class as the curator
    asimc = ASIMC()

    mc_d_values = []
    mc_dd_values = []

    # Feed bars sequentially — same as curator's append_m5 + compute
    for i in range(len(df)):
        # Get M5 OHLC — use mid_close as proxy for all 4 if OHLC not available
        close = float(df.iloc[i]["mid_close"])
        # Try to get OHLC, fall back to close for all
        o = float(df.iloc[i].get("mid_open", close))
        h = float(df.iloc[i].get("mid_high", close))
        l = float(df.iloc[i].get("mid_low", close))

        asimc.append_m5(o, h, l, close)
        mc_d, mc_dd = asimc.compute()
        mc_d_values.append(mc_d)
        mc_dd_values.append(mc_dd)

    # Create output dataframe
    out_df = pd.DataFrame({
        "timestamp": df.get("timestamp", pd.Series(range(len(df)))),
        "mid_close": df["mid_close"].values,
        "mc_d_curator": mc_d_values,
        "mc_dd_curator": mc_dd_values,
    })

    # Compare with existing parquet values
    if "mc_d_a" in df.columns:
        old_mc_d = df["mc_d_a"].values
        new_mc_d = np.array(mc_d_values)
        # Skip warmup bars (first 100)
        diff = np.abs(old_mc_d[100:] - new_mc_d[100:])
        max_diff = np.max(diff) if len(diff) > 0 else 0
        mean_diff = np.mean(diff) if len(diff) > 0 else 0
        print(f"  {pair}: {len(df):,} bars | MC(D) diff: max={max_diff:.6f} mean={mean_diff:.6f}")

        if max_diff > 0.001:
            print(f"    ⚠️  SIGNIFICANT MISMATCH! max_diff={max_diff:.6f}")
            # Find where the mismatch starts
            mismatch_idx = np.argmax(diff > 0.001)
            print(f"    First mismatch at bar {mismatch_idx + 100}: "
                  f"old={old_mc_d[mismatch_idx+100]:.6f} new={new_mc_d[mismatch_idx+100]:.6f}")
        else:
            print(f"    ✅ MATCH — curator code produces identical values")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_curator.parquet"
    out_df.to_parquet(out_path, engine="pyarrow")
    print(f"    Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="")
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else list(PAIR_PIP.keys())

    print("Exporting training data using CURATOR-IDENTICAL code path")
    print("=" * 60)
    for pair in pairs:
        export_pair_curator_identical(pair)


if __name__ == "__main__":
    main()
