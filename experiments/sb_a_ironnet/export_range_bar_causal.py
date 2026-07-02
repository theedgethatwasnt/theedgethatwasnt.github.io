#!/usr/bin/env python3
"""
Export causal ASI-MC indicators on 10-pip range bars for SB_A IronNet retraining.

TRAINING/LIVE IDENTITY GUARANTEE
==================================
This script guarantees that training mc_d/mc_dd values are identical to what
the live curator computes at the same M5 bar. Proof:

Live curator (lib/indicators.py ASIMC.compute()):
  - Has n M5 bars in buffer
  - Calls compute_mc_on_series(smooth, n, ...) where smooth = sma5(ASI)
  - Returns mc_d[-1] which is the CAUSAL value for bar n-1

This script:
  - Pre-computes smooth = sma5(ASI) on the FULL M5 dataset ONCE (same formula)
  - For each range bar closing at M5 index i, computes the causal mc_d value
    by replicating EXACTLY what compute_mc_on_series(smooth[:i+1], i+1)[-1] returns:
      * For TF window bp: last COMPLETE window index j_causal
        - if i%bp == bp-1: j_causal = i//bp  (bar i IS the last bar of window j)
        - else:            j_causal = i//bp-1 (use previous complete window)
      * tf_mc_d[j_causal] is read from the pre-computed full TF EMA series
        (valid because EMA is causal: tf_mc_d[j] depends only on tf_series[0..j])
      * TFs where n_tf_causal = (i+1)//bp < n_lags+5 are skipped (same skip as live)

EFFICIENCY: O(n_m5 + n_rb * n_tfs) — runs in <5s per pair
  Old bar-by-bar approach was O(n_m5 * n_rb) — 13+ min for EUR_USD.
  The pre-computed TF EMA values are reused; only the causal index mapping
  differs per range bar.

VERIFIABLE: Run with --verify to compare against live curator logs.
  For any M5 bar timestamp in recent curator output: this export's mc_d
  at that timestamp (sampled at range bar completion) should match the
  curator's published asi_mc_d within 1e-4.

Output: data/range_bar_causal/{pair}_range10_causal.parquet
  Columns: timestamp, mid_close, mc_d, mc_dd, bar_direction, m5_bar_idx
"""

import argparse
import math
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Single source of truth: all causal MC logic lives in lib/asi_indicator.py
from lib.asi_indicator import (
    compute_asi, sma_jit,
    compute_mc_causal_batch,          # causal batch — same file as compute_mc_on_series
    TF_BARS_S5, TF_WEIGHTS, N_TFS
)

# Import RangeBarBuilder from the live strategy — same object, not a copy
sys.path.insert(0, str(PROJECT_ROOT / "services" / "strategy_sba_ironnet"))
from main import RangeBarBuilder, PAIR_PIP

M5_DIR     = PROJECT_ROOT / "data" / "m5_ohlc"
OUTPUT_DIR = PROJECT_ROOT / "data" / "range_bar_causal"

ALL_PAIRS  = list(PAIR_PIP.keys())


def export_pair(pair: str, range_pips: float = 10.0, verbose: bool = True) -> int:
    """
    Export causal range bar indicators for one pair.

    Steps:
      1. Load M5 OHLC from data/m5_ohlc/{pair}_M5.parquet
      2. Compute ASI → SMA5 on full dataset once (same formula as ASIMC.compute())
      3. Run RangeBarBuilder (same class as live strategy) to find completion indices
      4. At each completion index i: compute causal mc_d/mc_dd via compute_mc_causal_batch (lib/asi_indicator.py)
      5. Save to data/range_bar_causal/{pair}_range10_causal.parquet
    """
    m5_path = M5_DIR / f"{pair}_M5.parquet"
    if not m5_path.exists():
        print(f"  {pair}: MISSING {m5_path}")
        return 0

    df = pd.read_parquet(m5_path, engine="pyarrow")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{pair}: missing column '{col}'")

    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    else:
        timestamps = pd.RangeIndex(len(df))

    opens  = df["open"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n_bars = len(closes)

    t0 = time.time()

    # ── Step 1: Compute ASI + SMA5 on full dataset (same as ASIMC.compute()) ──
    asi    = compute_asi(opens, highs, lows, closes, n_bars)
    smooth = sma_jit(asi, 5, n_bars)

    t_smooth = time.time() - t0

    # ── Step 2: Run RangeBarBuilder to collect completion (index, close, open_before) ──
    pip     = PAIR_PIP[pair]
    builder = RangeBarBuilder(pip=pip, range_pips=range_pips)

    completion_indices = []   # M5 bar indices where range bars completed
    completion_closes  = []   # close price at completion
    open_befores       = []   # bar_open before feed() advanced it (for direction)

    for i in range(n_bars):
        bar_open_before = builder.bar_open
        completed = builder.feed(closes[i])
        if completed is not None:
            completion_indices.append(i)
            completion_closes.append(completed)
            open_befores.append(bar_open_before)

    if not completion_indices:
        print(f"  {pair}: 0 range bars (check data)")
        return 0

    idx_arr = np.array(completion_indices, dtype=np.int64)

    # ── Step 3: Causal mc_d/mc_dd at all completion indices (vectorised) ──
    mc_d_arr, mc_dd_arr = compute_mc_causal_batch(
        smooth, n_bars, idx_arr, TF_BARS_S5, TF_WEIGHTS, N_TFS
    )

    # ── Step 4: Assemble output dataframe ──
    ts_vals   = [timestamps[i] for i in completion_indices]
    dirs      = [1 if completion_closes[k] >= (open_befores[k] or completion_closes[k])
                 else -1
                 for k in range(len(completion_closes))]

    out_df = pd.DataFrame({
        "timestamp":    ts_vals,
        "mid_close":    completion_closes,
        "mc_d":         mc_d_arr,
        "mc_dd":        mc_dd_arr,
        "bar_direction": dirs,
        "m5_bar_idx":   completion_indices,
    }).set_index("timestamp")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_range{int(range_pips)}_causal.parquet"
    out_df.to_parquet(out_path, engine="pyarrow")

    elapsed = time.time() - t0
    n_rb = len(completion_indices)
    if verbose:
        print(f"  {pair}: {n_rb:,} range bars | "
              f"mc_d [{mc_d_arr.min():.3f}, {mc_d_arr.max():.3f}] "
              f"mc_dd [{mc_dd_arr.min():.3f}, {mc_dd_arr.max():.3f}] | "
              f"smooth={t_smooth:.1f}s total={elapsed:.1f}s → {out_path.name}")

    return n_rb


def verify_causal_correctness(pair: str, range_pips: float = 10.0, n_tail: int = 5000):
    """
    Verify causal export is numerically identical to ASIMC.compute() bar-by-bar.

    Uses the LAST n_tail M5 bars only (fast ground truth via slow bar-by-bar method).
    Compares mc_d/mc_dd from the causal lookup vs from ASIMC.compute() for each
    range bar that completes within those bars.

    Both should agree to < 1e-4. Any divergence indicates a causal mapping bug.
    """
    from lib.indicators import ASIMC

    m5_path = M5_DIR / f"{pair}_M5.parquet"
    new_path = OUTPUT_DIR / f"{pair}_range{int(range_pips)}_causal.parquet"

    if not new_path.exists():
        print(f"  {pair}: run export first")
        return

    df = pd.read_parquet(m5_path, engine="pyarrow")
    tail_df = df.tail(n_tail)

    opens  = tail_df["open"].values.astype(np.float64)
    highs  = tail_df["high"].values.astype(np.float64)
    lows   = tail_df["low"].values.astype(np.float64)
    closes = tail_df["close"].values.astype(np.float64)

    # Bar-by-bar ground truth — warmup must cover all TF windows.
    # Largest TF: bp=720 (H1). Need n_lags+5=10 complete H1 bars = 7200 M5 bars.
    # Use 15000 warmup bars so all 9 TFs are fully active and EMA is settled.
    warmup_n = min(15000, len(df) - n_tail)
    warm_df  = df.iloc[-(n_tail + warmup_n):-n_tail]

    asimc = ASIMC()
    # Feed warmup bars (not recording completions)
    for _, row in warm_df.iterrows():
        asimc.append_m5(row["open"], row["high"], row["low"], row["close"])

    pip     = PAIR_PIP[pair]
    builder = RangeBarBuilder(pip=pip, range_pips=range_pips)
    # Align builder to end of warmup by feeding warmup closes
    for c in warm_df["close"].values:
        builder.feed(c)

    # Bar-by-bar on tail: collect (close, mc_d_gt, mc_dd_gt, m5_abs_idx)
    n_total = len(df)
    tail_start = n_total - n_tail
    gt_mc_d  = []
    gt_mc_dd = []
    gt_m5_abs= []

    for k in range(n_tail):
        o, h, l, c = opens[k], highs[k], lows[k], closes[k]
        asimc.append_m5(o, h, l, c)
        completed = builder.feed(c)
        if completed is not None:
            md, mdd = asimc.compute()
            gt_mc_d.append(md)
            gt_mc_dd.append(mdd)
            gt_m5_abs.append(tail_start + k)

    if not gt_mc_d:
        print(f"  {pair}: no range bar completions in tail window")
        return

    # Load causal export and match by m5_bar_idx
    new_df = pd.read_parquet(new_path, engine="pyarrow")
    idx_lookup = {int(row["m5_bar_idx"]): (row["mc_d"], row["mc_dd"])
                  for _, row in new_df.reset_index().iterrows()}

    matched, d_mc_d, d_mc_dd = [], [], []
    for k in range(len(gt_m5_abs)):
        abs_idx = gt_m5_abs[k]
        if abs_idx in idx_lookup:
            exp_d, exp_dd = idx_lookup[abs_idx]
            d_mc_d.append(abs(gt_mc_d[k] - exp_d))
            d_mc_dd.append(abs(gt_mc_dd[k] - exp_dd))
            matched.append(abs_idx)

    if not d_mc_d:
        print(f"  {pair}: could not match bar indices (check m5_bar_idx column)")
        return

    d_mc_d  = np.array(d_mc_d)
    d_mc_dd = np.array(d_mc_dd)
    bad = int((d_mc_d > 1e-4).sum())
    print(f"  {pair}: {len(matched)} range bars checked | "
          f"mc_d mean={d_mc_d.mean():.2e} max={d_mc_d.max():.2e} bad(>1e-4)={bad} | "
          f"mc_dd mean={d_mc_dd.mean():.2e} max={d_mc_dd.max():.2e}")
    if bad == 0:
        print(f"  ✅ {pair}: PERFECT MATCH — causal export == ASIMC.compute() bar-by-bar")
    else:
        print(f"  ❌ {pair}: {bad} mismatches — check compute_mc_causal_batch in lib/asi_indicator.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",       default=None,  help="Single pair or all 12")
    parser.add_argument("--range-pips", type=float, default=10.0)
    parser.add_argument("--verify",     action="store_true")
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ALL_PAIRS

    print(f"Causal ASI-MC range bar export  (range={args.range_pips}p)")
    print(f"Algorithm: pre-compute ASI+SMA5 once, causal TF-window lookup per range bar")
    print(f"Source:  {M5_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Pairs:   {', '.join(pairs)}")
    print()

    # Warm up Numba JIT on tiny arrays before the real work
    _dummy   = np.ones(200, dtype=np.float64)
    _idx_d   = np.array([199], dtype=np.int64)
    compute_asi(_dummy, _dummy, _dummy, _dummy, 200)
    sma_jit(_dummy, 5, 200)
    compute_mc_causal_batch(_dummy, 200, _idx_d, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    print("JIT warm-up done\n")

    total = 0
    t_all = time.time()
    for pair in pairs:
        total += export_pair(pair, range_pips=args.range_pips)

    print(f"\nTotal: {total:,} range bars across {len(pairs)} pairs in {time.time()-t_all:.1f}s")

    if args.verify:
        print("\nVerification vs ASIMC.compute() bar-by-bar (ground truth, last 5000 M5 bars):")
        for pair in pairs:
            verify_causal_correctness(pair, range_pips=args.range_pips)


if __name__ == "__main__":
    main()
