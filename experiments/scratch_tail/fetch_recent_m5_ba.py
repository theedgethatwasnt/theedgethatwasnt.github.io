#!/usr/bin/env python3
"""
fetch_recent_m5_ba.py — LOCAL-only fetch of a short recent M5-BA window (2026-06-10 -> now,
5 days of pre-window warmup before the live paper trail's 2026-06-15 start) for the 6
scratch_tail pairs, used EXCLUSIVELY for gate-2 (R7) parity — never for arm evaluation (that
uses only the sealed IS window through is_data.py). Not "heavy" (a few days x 6 pairs), so this
runs locally per CLAUDE.md's "nothing heavy runs locally" rule, then the output parquet is
rsynced to the Hetzner box for the actual parity replay.

Usage: python3 fetch_recent_m5_ba.py [--out-dir DIR]
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))

import v20  # noqa: E402

from signal import PAIRS  # noqa: E402

BARS_PER_REQUEST = 5000
M5_SECONDS = 300


def fetch_m5_ba_pair(ctx, instrument, start_dt, end_dt):
    total_bars = int((end_dt - start_dt).total_seconds() / M5_SECONDS)
    print(f"  [{instrument}] fetching ~{total_bars:,} M5 BA bars ({start_dt} -> {end_dt})", flush=True)
    all_rows = []
    fetch_from = start_dt
    n_req = 0
    while fetch_from < end_dt:
        from_str = fetch_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            resp = ctx.instrument.candles(instrument=instrument, granularity="M5",
                                           count=BARS_PER_REQUEST, price="MBA", fromTime=from_str)
        except Exception as e:
            print(f"  [{instrument}] API error: {e} — retrying in 5s")
            time.sleep(5)
            continue
        if resp.status != 200:
            print(f"  [{instrument}] HTTP {resp.status} — retrying in 5s")
            time.sleep(5)
            continue
        candles = resp.body.get("candles", [])
        if not candles:
            break
        n_added = 0
        last_ts = None
        for c in candles:
            if not c.complete:
                continue
            bid = c.bid if (hasattr(c, "bid") and c.bid) else c.mid
            ask = c.ask if (hasattr(c, "ask") and c.ask) else c.mid
            all_rows.append({
                "timestamp": str(c.time),
                "open": float(c.mid.o), "high": float(c.mid.h), "low": float(c.mid.l), "close": float(c.mid.c),
                "bid_c": float(bid.c), "ask_c": float(ask.c), "volume": int(c.volume),
            })
            last_ts = str(c.time)
            n_added += 1
        n_req += 1
        if last_ts:
            fetch_from = pd.Timestamp(last_ts).to_pydatetime() + timedelta(seconds=300)
        else:
            break
        print(f"  [{instrument}] req#{n_req}: +{n_added} bars -> {len(all_rows):,} total", flush=True)
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_data"))
    ap.add_argument("--start", default="2026-06-10T00:00:00Z")
    args = ap.parse_args()

    api_key = os.environ.get("OANDA_API_KEY", "")
    if not api_key:
        raise ValueError("OANDA_API_KEY not set")
    ctx = v20.Context(hostname="api-fxtrade.oanda.com", port="443", token=api_key)

    os.makedirs(args.out_dir, exist_ok=True)
    start_dt = pd.Timestamp(args.start).to_pydatetime()
    end_dt = datetime.now(timezone.utc)

    for pair in PAIRS:
        out_path = Path(args.out_dir) / f"{pair}_M5_BA.parquet"
        df = fetch_m5_ba_pair(ctx, pair, start_dt, end_dt)
        if df.empty:
            print(f"  [{pair}] EMPTY — skipping")
            continue
        df.to_parquet(out_path, index=False)
        print(f"  [{pair}] saved {len(df):,} rows -> {out_path} "
              f"({df['timestamp'].min()} -> {df['timestamp'].max()})\n", flush=True)


if __name__ == "__main__":
    main()
