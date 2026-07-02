"""
chop_momentum_entries.py — Phase 1 entry chopper (2026-06-12).

Momentum-continuation entries on REAL data (not shock-fade): enter in the DIRECTION
of aligned multi-lag momentum, betting it continues.

  LONG  at the rising edge of:  (c−c5 > 0) & (c−c12 > 0) & (c−c120 > 0)
  SHORT at the rising edge of:  (c−c5 < 0) & (c−c12 < 0) & (c−c120 < 0)

Lags 5/12/120 S5 bars = 25s / 1min / 10min (user's rule). "Rising edge" = first bar
the condition turns true (so we enter once per aligned-momentum episode, not every bar).
A min-gap and an optional max-N cap keep the trade set independent and memory-bounded.

Writes meta3_<PAIR>_mom.parquet (same schema as the other meta3 files). The exit ESCMA
then reads the same mn_* features and learns when to exit (incl. SL/TP). All causal:
c−c_lag uses only past closes; the exit reads only the current bar.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--lags", default="5,12,120")
    ap.add_argument("--min-gap", type=int, default=288, help="min S5 bars between entries")
    ap.add_argument("--hold-cap", type=int, default=1440, help="t_timeout = t_event + this")
    ap.add_argument("--max-n", type=int, default=6000, help="subsample evenly if more")
    ap.add_argument("--warmup", type=int, default=720)
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",")]
    feat_path = SCRIPT_DIR / f"features_{args.pair}.parquet"
    close = pq.read_table(feat_path, columns=["close"]).column("close").to_numpy().astype(np.float64)
    n = close.shape[0]

    # causal multi-lag momentum
    long_cond = np.ones(n, dtype=bool)
    short_cond = np.ones(n, dtype=bool)
    for L in lags:
        m = np.full(n, np.nan)
        m[L:] = close[L:] - close[:-L]
        long_cond &= (m > 0)
        short_cond &= (m < 0)
    long_cond = np.nan_to_num(long_cond).astype(bool)
    short_cond = np.nan_to_num(short_cond).astype(bool)

    # rising edges (condition turns true)
    le = long_cond.copy();  le[1:] &= ~long_cond[:-1];  le[0] = False
    se = short_cond.copy(); se[1:] &= ~short_cond[:-1]; se[0] = False

    edge_t = np.where(le | se)[0]
    edge_t = edge_t[(edge_t > args.warmup) & (edge_t < n - args.hold_cap - 10)]
    edge_dir = np.where(le[edge_t], 1, -1)

    # enforce min-gap
    events = []
    last = -10**9
    sid = 0
    for t, d in zip(edge_t.tolist(), edge_dir.tolist()):
        if t - last < args.min_gap:
            continue
        events.append((sid, t - 60, t, t + args.hold_cap, d)); sid += 1; last = t

    m = pd.DataFrame(events, columns=["sample_id", "t_pre", "t_event", "t_timeout", "direction"])
    # even subsample to max-n (preserve temporal order for the split)
    if len(m) > args.max_n:
        idx = np.linspace(0, len(m) - 1, args.max_n).astype(int)
        m = m.iloc[idx].reset_index(drop=True)
        m["sample_id"] = np.arange(len(m))

    cut = int(len(m) * 0.70)
    m["split"] = ["IS"] * cut + ["OOS"] * (len(m) - cut)
    out = SCRIPT_DIR / f"meta3_{args.pair}_mom.parquet"
    m.to_parquet(out)

    print(f"[mom-chop] {args.pair} lags={lags} min_gap={args.min_gap}")
    print(f"[mom-chop] raw edges={len(edge_t)}  after gap+cap={len(m)}  "
          f"(IS={cut} OOS={len(m)-cut})  long={(m.direction==1).sum()} short={(m.direction==-1).sum()}")
    print(f"[wrote] {out.name}")


if __name__ == "__main__":
    main()
