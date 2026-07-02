#!/usr/bin/env python3
"""random_entry_meta3 — RANDOM-entry test set for the exit-alpha experiment
============================================================================

Generates meta3_<PAIR>_random.parquet with the SAME schema as meta3:
    sample_id, t_pre, t_event, t_timeout, direction, split

The clean test: pick N random bar indices and random ±1 directions (NOT
outcome-derived — no lookahead), then ask whether the position-aware exit can
extract edge from a RANDOM entry. If a fully-sighted exit beats the -spread
floor on these, pure exit alpha exists.

Rules respected:
  • FIXED seed for reproducibility, but direction is random (50/50), independent
    of any future price — strictly no lookahead.
  • t_event >= 1500 (warm features) and <= n_bars - 17280 (room post-entry).
  • t_pre = t_event - 720, t_timeout = t_event + min(17280, available).
  • IS/OOS 70/30 chronological split by t_event.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

WARM = 1500
PRE = 720
POST = 17280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--data-dir", default=str(SCRIPT_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    feat_path = data_dir / f"features_{args.pair}.parquet"
    n_bars = pq.ParquetFile(feat_path).metadata.num_rows
    print(f"features rows = {n_bars}")

    lo = WARM
    hi = n_bars - POST  # ensures full POST room after entry
    if hi <= lo:
        raise SystemExit(f"not enough bars: lo={lo} hi={hi}")

    rng = np.random.default_rng(args.seed)
    # Random entry bars (unique), random direction — NO lookahead.
    t_event = rng.choice(np.arange(lo, hi, dtype=np.int64),
                         size=args.n, replace=False)
    t_event.sort()  # chronological for the IS/OOS split
    direction = rng.choice(np.array([-1, 1], dtype=np.int8), size=args.n)

    t_pre = t_event - PRE
    t_timeout = t_event + np.minimum(POST, n_bars - t_event).astype(np.int64)

    # IS/OOS 70/30 chronological by t_event
    split = np.empty(args.n, dtype=object)
    cut = int(args.n * 0.70)
    split[:cut] = "IS"
    split[cut:] = "OOS"

    df = pd.DataFrame({
        "sample_id": np.arange(args.n, dtype=np.int64),
        "t_pre": t_pre,
        "t_event": t_event,
        "t_timeout": t_timeout,
        "direction": direction,
        "split": split,
    })

    out = data_dir / f"meta3_{args.pair}_random.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  N={len(df)}  "
          f"IS={int((df.split=='IS').sum())} OOS={int((df.split=='OOS').sum())}")
    print(f"direction balance: {df.direction.value_counts().to_dict()}")
    print(f"t_event range: [{int(df.t_event.min())}, {int(df.t_event.max())}]")
    print(f"post len (all should be {POST}): "
          f"{(df.t_timeout - df.t_event).unique()[:5].tolist()}")


if __name__ == "__main__":
    main()
