"""
Fetch / top-up S5 bid-ask OHLC parquets for all 12 FX pairs.

Memory-safe: writes in 1M-row chunks to temp parquets, concatenates at end.
Schema: timestamp, open, high, low, close (mid), bid_c, ask_c, volume

Saves to data/s5_ba/{PAIR}_S5_BA.parquet

Usage:
    python3 research/experiments/vel_acc/fetch_s5_ba.py [--pairs EUR_USD,GBP_JPY]
"""
import gc
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import v20

ALL_PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "CHF_JPY", "CAD_JPY",
    "AUD_USD", "NZD_JPY", "NZD_USD", "EUR_GBP",
]
OUT_DIR          = ROOT / "data" / "s5_ba"
TMP_DIR          = ROOT / "data" / "s5_ba" / "_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_REQUEST = 5000
S5_SECONDS       = 5
FALLBACK_START   = datetime(2020, 11, 11, tzinfo=timezone.utc)
RATE_LIMIT_SLEEP = 0.15
CHUNK_ROWS       = 1_000_000   # write to disk every 1M rows

SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open",   pa.float32()),
    ("high",   pa.float32()),
    ("low",    pa.float32()),
    ("close",  pa.float32()),
    ("bid_c",  pa.float32()),
    ("ask_c",  pa.float32()),
    ("volume", pa.int32()),
])


def oanda_context():
    return v20.Context(
        hostname="api-fxtrade.oanda.com",
        port="443",
        token=os.environ.get("OANDA_API_KEY"),
    )


def fetch_pair(ctx, pair: str):
    out_path = OUT_DIR / f"{pair}_S5_BA.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["timestamp"])
        last_ts  = pd.to_datetime(existing["timestamp"]).max().to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        fetch_start = last_ts + timedelta(seconds=S5_SECONDS)
        n_existing  = len(pd.read_parquet(out_path))
        print(f"  {pair}: {n_existing:,} existing bars | fetching from {fetch_start.date()}")
    else:
        fetch_start = FALLBACK_START
        n_existing  = 0
        print(f"  {pair}: no existing file | full fetch from {fetch_start.date()}")

    end_dt    = datetime.now(timezone.utc) - timedelta(seconds=10)
    total_est = max(1, int((end_dt - fetch_start).total_seconds() / S5_SECONDS))
    n_req_est = max(1, total_est // BARS_PER_REQUEST + 1)
    print(f"    Est. ~{total_est:,} bars (~{n_req_est} requests)...")

    # Chunk writer: write temp parquet files every CHUNK_ROWS bars
    chunk_files  = []
    chunk_buf    = []   # list of (ts, o, h, l, c, bid_c, ask_c, vol) tuples
    total_fetched = 0
    fetch_from_dt = fetch_start
    t0 = time.time()

    def flush_chunk():
        if not chunk_buf:
            return
        idx   = len(chunk_files)
        tpath = TMP_DIR / f"{pair}_chunk{idx:04d}.parquet"
        df    = pd.DataFrame(chunk_buf, columns=["timestamp","open","high","low","close","bid_c","ask_c","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ["open","high","low","close","bid_c","ask_c"]:
            df[col] = df[col].astype("float32")
        df["volume"] = df["volume"].astype("int32")
        df.to_parquet(tpath, index=False, compression="snappy")
        chunk_files.append(tpath)
        chunk_buf.clear()
        gc.collect()

    while True:
        from_str = fetch_from_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            resp = ctx.instrument.candles(
                pair,
                granularity="S5",
                count=BARS_PER_REQUEST,
                fromTime=from_str,
                price="BA",
            )
        except Exception as e:
            print(f"\n    WARN: API error ({e}), retrying in 5s...")
            time.sleep(5)
            continue

        candles   = resp.body.get("candles", [])
        new_count = 0

        for c in candles:
            if not c.complete:
                continue
            bid, ask = c.bid, c.ask
            mid_o = (float(bid.o) + float(ask.o)) / 2
            mid_h = (float(bid.h) + float(ask.h)) / 2
            mid_l = (float(bid.l) + float(ask.l)) / 2
            mid_c = (float(bid.c) + float(ask.c)) / 2
            chunk_buf.append((
                c.time, mid_o, mid_h, mid_l, mid_c,
                float(bid.c), float(ask.c), int(c.volume),
            ))
            new_count += 1

        if new_count == 0:
            break

        total_fetched += new_count
        last_ts_str    = candles[-1].time if candles else from_str
        fetch_from_dt  = pd.Timestamp(last_ts_str, tz="UTC").to_pydatetime() + timedelta(seconds=S5_SECONDS)

        # Flush every CHUNK_ROWS rows
        if len(chunk_buf) >= CHUNK_ROWS:
            flush_chunk()

        elapsed = time.time() - t0
        pct     = min(100, int((fetch_from_dt - fetch_start).total_seconds() /
                               max(1, (end_dt - fetch_start).total_seconds()) * 100))
        rate    = total_fetched / max(1, elapsed)
        eta     = int((total_est - total_fetched) / max(1, rate))
        print(f"\r    {total_fetched:>10,} bars | {fetch_from_dt.date()} | {pct:3d}% | ETA {eta//60}m{eta%60:02d}s    ",
              end="", flush=True)

        if fetch_from_dt >= end_dt:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    flush_chunk()   # flush remaining
    print()

    if not chunk_files and n_existing == 0:
        print(f"  {pair}: nothing to save")
        return

    # Concatenate: existing + all chunks
    parts = []
    if out_path.exists():
        parts.append(pd.read_parquet(out_path))
    for cpath in chunk_files:
        parts.append(pd.read_parquet(cpath))
        cpath.unlink()   # delete temp file

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        for col in ["open","high","low","close","bid_c","ask_c"]:
            combined[col] = combined[col].astype("float32")
        combined["volume"] = combined["volume"].astype("int32")
        combined.to_parquet(out_path, index=False, compression="snappy")
        sz_mb = out_path.stat().st_size / 1e6
        print(f"  {pair}: {len(combined):,} total bars saved ({sz_mb:.0f} MB)  "
              f"({fetch_start.date()} → {combined['timestamp'].max().date()})")
        del combined, parts
        gc.collect()
    else:
        print(f"  {pair}: no new data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS),
                        help="Comma-separated pair list")
    args  = parser.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    ctx = oanda_context()
    print(f"Fetching S5 BA data for {len(pairs)} pairs → {OUT_DIR}")
    print(f"Fallback start: {FALLBACK_START.date()}")
    print(f"Chunk size: {CHUNK_ROWS:,} rows\n")

    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair}")
        t0 = time.time()
        try:
            fetch_pair(ctx, pair)
        except Exception as e:
            print(f"  ERROR: {e}")
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f}m\n")

    # Clean up empty tmp dir
    try:
        TMP_DIR.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
