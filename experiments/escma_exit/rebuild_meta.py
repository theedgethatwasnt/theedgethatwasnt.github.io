"""
rebuild_meta.py — Re-emit the chopper's event meta in explicit three-index form.

The committed entry_chopper writes meta_<PAIR>.parquet with `t_event_idx` +
`n_post_bars`. This post-processor (which does NOT touch the chopper's
event-detection logic or thresholds) re-expresses each event as three explicit
bar indices that map directly to row slices of features_<PAIR>.parquet:

    sample_id   — carried from meta
    t_pre       = t_event - PRE_BARS          (start of 1h context window)
    t_event     = t_event_idx                 (the event bar)
    t_timeout   = t_event + n_post_bars        (max-hold end; n_post_bars capped
                                                at 17280 by the chopper)
    direction   — carried from meta
    split       — carried from meta

Writes meta3_<PAIR>.parquet (new file; the old meta is left in place).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRE_BARS = 720  # chopper PRE_BARS_DEFAULT (= 1h context window)


def rebuild(pair: str, data_dir: Path | None = None,
            pre_bars: int = PRE_BARS) -> Path:
    if data_dir is None:
        data_dir = SCRIPT_DIR
    meta_path = data_dir / f"meta_{pair}.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta missing: {meta_path}")

    meta = pd.read_parquet(meta_path)
    t_event = meta["t_event_idx"].astype("int64")
    n_post = meta["n_post_bars"].astype("int64")

    meta3 = pd.DataFrame({
        "sample_id": meta["sample_id"].astype("int64"),
        "t_pre": (t_event - pre_bars).astype("int64"),
        "t_event": t_event,
        "t_timeout": (t_event + n_post).astype("int64"),
        "direction": meta["direction"].astype("int8"),
        "split": meta["split"].astype(str),
    })

    out_path = data_dir / f"meta3_{pair}.parquet"
    meta3.to_parquet(out_path, index=False, compression="zstd")
    print(f"[rebuild_meta] {len(meta3):,} events → {out_path} "
          f"({out_path.stat().st_size/1e3:.1f} KB)")
    print(f"  t_pre   range: [{meta3['t_pre'].min()}, {meta3['t_pre'].max()}]")
    print(f"  t_event range: [{meta3['t_event'].min()}, {meta3['t_event'].max()}]")
    print(f"  t_timeout range: [{meta3['t_timeout'].min()}, {meta3['t_timeout'].max()}]")
    print(f"  split counts: {meta3['split'].value_counts().to_dict()}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--pre-bars", type=int, default=PRE_BARS)
    args = ap.parse_args()
    rebuild(args.pair, data_dir=args.data_dir, pre_bars=args.pre_bars)


if __name__ == "__main__":
    main()
