#!/usr/bin/env python3
"""
Download M5 OHLC candles from OANDA for all 12 pairs.
Saves to data/m5_ohlc/{PAIR}_M5.parquet with columns:
  timestamp, open, high, low, close, volume

Date range: 2021-01-01 to 2026-04-09 (~5 years, ~260K M5 bars per pair)
Split plan: Train 2021-2023 | Validate 2024 | Test/OOS 2025-2026
"""

import os
import sys
import time
import v20
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["OANDA_API_KEY"]
HOSTNAME = "api-fxtrade.oanda.com"

ALL_PAIRS = [
    "EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "CHF_JPY", "NZD_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
]

OUT_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = datetime(2021, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 4, 9, tzinfo=timezone.utc)


def fetch_candles(ctx, pair, start_dt, end_dt, granularity="M5"):
    """Fetch all candles between start and end in paginated chunks."""
    all_candles = []
    cursor = start_dt

    while cursor < end_dt:
        from_str = cursor.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        to_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        try:
            resp = ctx.instrument.candles(
                instrument=pair,
                granularity=granularity,
                fromTime=from_str,
                toTime=to_str,
                count=5000,
                price="M",  # mid prices
            )
        except Exception as e:
            print(f"    Error at {cursor}: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if resp.status != 200:
            print(f"    HTTP {resp.status} at {cursor}, retrying...")
            time.sleep(2)
            continue

        candles = resp.body.get("candles", [])
        if not candles:
            break

        for c in candles:
            if c.complete:
                all_candles.append({
                    "timestamp": pd.Timestamp(str(c.time)),
                    "open": float(c.mid.o),
                    "high": float(c.mid.h),
                    "low": float(c.mid.l),
                    "close": float(c.mid.c),
                    "volume": int(c.volume),
                })

        # Move cursor past last candle
        last_ts = pd.Timestamp(str(candles[-1].time))
        cursor = last_ts.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(seconds=1)

        print(f"    {pair}: {len(all_candles):,} candles, last={last_ts}")
        time.sleep(0.2)  # Rate limit

    return all_candles


def main():
    ctx = v20.Context(hostname=HOSTNAME, port="443", token=API_KEY)

    for pair in ALL_PAIRS:
        out_path = OUT_DIR / f"{pair}_M5.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            print(f"{pair}: already exists ({len(existing):,} bars), skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Downloading {pair} M5 OHLC")
        print(f"  {START.date()} → {END.date()}")
        print(f"{'='*50}")

        candles = fetch_candles(ctx, pair, START, END)

        if not candles:
            print(f"  WARNING: No candles for {pair}")
            continue

        df = pd.DataFrame(candles)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"  Saved: {out_path.name} — {len(df):,} bars")
        print(f"  Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
