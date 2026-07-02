"""Build causal feature parquets for all 12 pairs using FXFeatureBuilder.

Args:
  --smoother: sma5 | kalman10 | ema3 | rma5
  --pairs: space-separated pair list (default: all 12)
"""
import argparse, sys, time, os
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

PAIRS = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","EUR_JPY","GBP_JPY",
         "AUD_JPY","CAD_JPY","CHF_JPY","NZD_JPY","NZD_USD","EUR_GBP"]


def build_one(args_tuple):
    pair, smoother = args_tuple
    import pandas as pd
    from lib.incremental_features import FXFeatureBuilder
    out = PROJECT / f"data/m5_ohlc/{pair}_M5_{smoother}_causal.parquet"
    if out.exists():
        return f"{pair}: skipped (exists)"
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    t0 = time.time()
    b = FXFeatureBuilder(pair, smoother=smoother)
    r = b.walk_history(df)
    r.to_parquet(out)
    return f"{pair} [{smoother}]: {len(df)} bars in {time.time()-t0:.0f}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoother", default="kalman10", choices=["sma5", "kalman10", "ema3", "rma5"])
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    pairs = args.pairs or PAIRS
    tasks = [(p, args.smoother) for p in pairs]
    print(f"Building {len(tasks)} parquets with smoother={args.smoother}")
    t0 = time.time()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(build_one, tasks):
            print(r, flush=True)
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
