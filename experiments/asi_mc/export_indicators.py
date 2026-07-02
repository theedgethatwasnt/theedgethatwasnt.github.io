#!/usr/bin/env python3
"""
Export curator-identical ASI-MC indicators to parquet.
=====================================================
Reads raw S5 parquet, computes indicators using the EXACT same code
the curator uses, saves per-pair indicator arrays + mid prices.

Training reads these exported files — zero mismatch with live.

Produces two variants:
  A: M5-only ASI → SMA5 → MC (single-TF, virtual resample in MC)
  B: Multi-TF ASI (S5/S30/M1/M5/H1) → SMA5 each → EMA3-EMA5 consensus

Output per pair: parquet with columns:
  timestamp, mid_close, mc_d_a, mc_dd_a, mc_d_b, mc_dd_b
  at M5 cadence.
"""

import sys
import os
import math
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi, sma_jit, compute_mc_on_series, TF_BARS_S5, TF_WEIGHTS, N_TFS, compute_asi_mc, compute_asi_mc_multitf
from lib.pair_config import PAIRS, ALL_PAIR_NAMES

DATA_DIR = PROJECT_ROOT / "data" / "scalper_parquet"
OUT_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"


def compute_variant_a(o_m5, h_m5, l_m5, c_m5, n_m5):
    """Variant A: ASI on M5 → SMA5 → MC via virtual TF resample."""
    asi = compute_asi(o_m5, h_m5, l_m5, c_m5, n_m5)
    smooth = sma_jit(asi, 5, n_m5)
    mc_d, mc_dd = compute_mc_on_series(smooth, n_m5, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d, mc_dd


def compute_variant_b(s5_df, m5_times, n_m5):
    """Variant B: ASI per real TF → SMA5 each → EMA3-EMA5 consensus at M5 cadence."""
    TFS = {
        'S5':  None,
        'S30': '30s',
        'M1':  '1min',
        'M5':  '5min',
        'H1':  '1h',
    }
    TF_SEC = {'S5': 5, 'S30': 30, 'M1': 60, 'M5': 300, 'H1': 3600}
    N_LAGS = 5

    # Per-TF: compute ASI, SMA5, map to M5 cadence, compute EMA3-EMA5
    mc_d = np.zeros(n_m5, dtype=np.float64)
    mc_dd = np.zeros(n_m5, dtype=np.float64)
    tw = 0.0

    for tf_name, rule in TFS.items():
        w = math.log2(max(TF_SEC[tf_name] / 5, 1)) + 1

        if rule is None:
            tf_df = s5_df
        else:
            tf_df = s5_df.resample(rule).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()

        o = tf_df['o'].values.astype(np.float64)
        h = tf_df['h'].values.astype(np.float64)
        l = tf_df['l'].values.astype(np.float64)
        c = tf_df['c'].values.astype(np.float64)
        n = len(o)

        if n < 20:
            continue

        asi = compute_asi(o, h, l, c, n)
        smooth = sma_jit(asi, 5, n)

        # Map to M5 cadence: for each M5 bar, find latest TF bar
        tf_times = tf_df.index
        mapped = np.zeros(n_m5, dtype=np.float64)
        j = 0
        for i, m5t in enumerate(m5_times):
            while j < len(tf_times) - 1 and tf_times[j + 1] <= m5t:
                j += 1
            mapped[i] = smooth[min(j, n - 1)]

        # EMA3-EMA5 on mapped series
        alpha3 = 2.0 / 4.0
        alpha5 = 2.0 / 6.0
        e3 = mapped[0]
        e5 = mapped[0]
        d_vals = np.zeros(n_m5, dtype=np.float64)
        for i in range(n_m5):
            e3 = alpha3 * mapped[i] + (1 - alpha3) * e3
            e5 = alpha5 * mapped[i] + (1 - alpha5) * e5
            d_vals[i] = e3 - e5

        # MC(D)
        for i in range(N_LAGS + 1, n_m5):
            pos = neg = 0
            for lag in range(N_LAGS):
                change = d_vals[i - lag] - d_vals[i - lag - 1]
                if change > 0:
                    pos += 1
                elif change < 0:
                    neg += 1
            mc_d[i] += w * (pos - neg) / N_LAGS

        # MC(dD)
        for i in range(N_LAGS + 2, n_m5):
            pos = neg = 0
            for lag in range(N_LAGS):
                ji = i - lag
                if ji >= 3:
                    dd_now = d_vals[ji] - 2 * d_vals[ji - 1] + d_vals[ji - 2]
                    dd_prev = d_vals[ji - 1] - 2 * d_vals[ji - 2] + d_vals[ji - 3]
                    change = dd_now - dd_prev
                    if change > 0:
                        pos += 1
                    elif change < 0:
                        neg += 1
            mc_dd[i] += w * (pos - neg) / N_LAGS

        tw += w

    if tw > 0:
        mc_d /= tw
        mc_dd /= tw

    return mc_d, mc_dd


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print(f"Exporting ASI-MC indicators for {len(ALL_PAIR_NAMES)} pairs...")
    print(f"Output: {OUT_DIR}/")

    for pair in ALL_PAIR_NAMES:
        t0 = time.time()
        parquet = DATA_DIR / f"{pair.replace('_', '')}_S5_BA.parquet"
        if not parquet.exists():
            print(f"  {pair}: SKIP (no parquet)")
            continue

        print(f"  {pair}...", end=" ", flush=True)
        df = pd.read_parquet(parquet, engine="pyarrow")

        # Use mid prices (matches curator)
        ts = pd.to_datetime(df["timestamp"])
        half_spread = (df["ask_c"].values - df["bid_c"].values) / 2.0
        s5_df = pd.DataFrame({
            "o": df["bid_o"].values + half_spread,
            "h": df["bid_h"].values + half_spread,
            "l": df["bid_l"].values + half_spread,
            "c": df["bid_c"].values + half_spread,
        }, index=ts)

        # Resample to M5
        m5 = s5_df.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        mid_close = m5["c"].values.astype(np.float64)
        m5_times = m5.index
        n_m5 = len(m5)

        # Variant A: M5-only ASI → SMA5 → MC (virtual TF resample)
        mc_d_a, mc_dd_a = compute_asi_mc(
            m5["o"].values.astype(np.float64),
            m5["h"].values.astype(np.float64),
            m5["l"].values.astype(np.float64),
            m5["c"].values.astype(np.float64),
            n_m5,
        )

        # Variant B: Multi-TF ASI (S5/S30/M1/M5/H1) → SMA5 → MC (JIT)
        o_s5 = s5_df["o"].values.astype(np.float64)
        h_s5 = s5_df["h"].values.astype(np.float64)
        l_s5 = s5_df["l"].values.astype(np.float64)
        c_s5 = s5_df["c"].values.astype(np.float64)
        n_s5 = len(o_s5)
        # Trim to exact multiple of 60 (M5 alignment)
        trim = (n_s5 // 60) * 60
        mc_d_b, mc_dd_b = compute_asi_mc_multitf(
            o_s5[:trim], h_s5[:trim], l_s5[:trim], c_s5[:trim], trim
        )
        # mc_d_b is at M5 cadence, length = trim // 60
        # Align with m5 (may differ by a few bars due to pandas vs integer resample)
        n_b = len(mc_d_b)
        if n_b < n_m5:
            mc_d_b = np.concatenate([np.zeros(n_m5 - n_b), mc_d_b])
            mc_dd_b = np.concatenate([np.zeros(n_m5 - n_b), mc_dd_b])
        elif n_b > n_m5:
            mc_d_b = mc_d_b[-n_m5:]
            mc_dd_b = mc_dd_b[-n_m5:]

        # Variant C: Quantized sign of A, three thresholds
        for thresh_name, thresh in [("c02", 0.02), ("c05", 0.05), ("c10", 0.10)]:
            locals()[f"mc_d_{thresh_name}"] = np.where(np.abs(mc_d_a) > thresh, np.sign(mc_d_a), 0.0)
            locals()[f"mc_dd_{thresh_name}"] = np.where(np.abs(mc_dd_a) > thresh, np.sign(mc_dd_a), 0.0)

        # Save
        out_df = pd.DataFrame({
            "timestamp": m5_times,
            "mid_close": mid_close,
            "mc_d_a": mc_d_a,
            "mc_dd_a": mc_dd_a,
            "mc_d_b": mc_d_b,
            "mc_dd_b": mc_dd_b,
            "mc_d_c02": locals()["mc_d_c02"],
            "mc_dd_c02": locals()["mc_dd_c02"],
            "mc_d_c05": locals()["mc_d_c05"],
            "mc_dd_c05": locals()["mc_dd_c05"],
            "mc_d_c10": locals()["mc_d_c10"],
            "mc_dd_c10": locals()["mc_dd_c10"],
        })
        out_path = OUT_DIR / f"{pair}_asi_mc.parquet"
        out_df.to_parquet(out_path, index=False, engine="pyarrow")

        elapsed = time.time() - t0
        print(f"{n_m5:,} M5 bars, A=[{mc_d_a.std():.3f}], B=[{mc_d_b.std():.3f}], {elapsed:.1f}s")

        del df, s5_df, m5
        gc.collect()

    total = time.time() - t_total
    print(f"\nDone in {total:.0f}s")
    print(f"Files: {OUT_DIR}/")

    # Summary
    files = list(OUT_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files)} files, {total_size:.1f} MB total")


if __name__ == "__main__":
    main()
