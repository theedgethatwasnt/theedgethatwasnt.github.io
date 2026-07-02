#!/usr/bin/env python3
"""
Export ASI-MC indicators on 10-pip range bars for NEAT training.

Instead of M5 bars (288/day, noisy), uses range bars (~35/day, clean).
Same MC(D) + MC(dD) inputs but sampled at range bar completion.

Output: {pair}_range10_asi_mc.parquet with columns:
  mid_close, mc_d, mc_dd, bar_direction, bar_duration
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "projects" / "muzero_asi_mc"))

from range_bars import build_range_bars

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "asi_mc_indicators"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "range_bar_indicators"

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}

def export_pair(pair: str, range_pips: float = 10.0):
    path = DATA_DIR / f"{pair}_asi_mc.parquet"
    df = pd.read_parquet(path, engine="pyarrow")

    pip = PAIR_PIP[pair]
    closes = df["mid_close"].values.astype(np.float64)
    mc_d = df["mc_d_a"].values.astype(np.float32)
    mc_dd = df["mc_dd_a"].values.astype(np.float32)

    bars = build_range_bars(closes, pip, range_pips)

    records = []
    for b in bars:
        idx = min(b.end_idx, len(mc_d) - 1)
        records.append({
            "mid_close": b.close,
            "mc_d": float(mc_d[idx]),
            "mc_dd": float(mc_dd[idx]),
            "bar_direction": b.direction,
            "bar_duration": b.n_s5_bars,
            "range_pips": b.range_pips,
        })

    out_df = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_range{int(range_pips)}_asi_mc.parquet"
    out_df.to_parquet(out_path, engine="pyarrow")
    print(f"  {pair}: {len(bars):,} range bars → {out_path.name}")
    return len(bars)

if __name__ == "__main__":
    total = 0
    for pair in PAIR_PIP:
        total += export_pair(pair)
    print(f"\nTotal: {total:,} range bars across {len(PAIR_PIP)} pairs")
