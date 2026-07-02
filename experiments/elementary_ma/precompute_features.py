#!/usr/bin/env python3
"""Precompute 9 elementary-MA features for all 12 pairs. Saves per-pair npy.

Output: data/elementary_ma_features/{pair}_features.npz containing all 9 feature
arrays aligned 1:1 with the unified_indicators parquet rows.

Computed once locally and shipped to Hetzner to save ~30% per training run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.elementary_ma.features import compute_all_features

DATA_DIR = PROJECT_ROOT / "data" / "unified_indicators"
OUT_DIR = PROJECT_ROOT / "data" / "elementary_ma_features"

PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]

PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        t0 = time.time()
        p = DATA_DIR / f"{pair}_unified.parquet"
        if not p.exists():
            print(f"SKIP {pair}: no parquet at {p}")
            continue

        df = pd.read_parquet(p, columns=["timestamp", "mid_close"])
        close = df["mid_close"].values.astype(np.float64)
        # NOTE: unified parquets have no bid/ask — use mid as bid proxy.
        # Live deployment must replace with curator.bid.
        bid = close
        pip = PIP[pair]

        feats = compute_all_features(bid, close, pip)

        # Compact storage: save as compressed npz
        out_path = OUT_DIR / f"{pair}_features.npz"
        np.savez_compressed(
            out_path,
            **{k: v.astype(np.float32) for k, v in feats.items()},
        )
        dt = time.time() - t0
        sample_valid = (~np.isnan(feats["atan_d_bid_sma5"])).sum()
        print(f"  {pair}: {len(close):,} bars, {len(feats)} feats, "
              f"valid={sample_valid:,}, {dt:.1f}s → {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
