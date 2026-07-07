"""
Fetch EUR/USD lower-timeframe bars from OANDA for shock-propagation analysis.
Saves mid OHLC + bid/ask close to parquet files.

Granularities fetched:
  S5  → data/s5_ohlc/EUR_USD_S5_BA.parquet     (~6 months)
  S30 → data/s30_ohlc/EUR_USD_S30_BA.parquet   (~6 months)
  M1  → data/m1_ohlc/EUR_USD_M1_BA.parquet     (~6 months)

Usage:
    python3 research/fetch_eurusd_lower_tf.py [--months 6]
"""
import os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import v20

INSTRUMENT = "EUR_USD"
BARS_PER_REQ = 5000
GRANULARITY_SECONDS = {"S5": 5, "S30": 30, "M1": 60}

OUT_DIRS = {
    "S5":  ROOT / "data" / "s5_ohlc",
    "S30": ROOT / "data" / "s30_ohlc",
    "M1":  ROOT / "data" / "m1_ohlc",
}
for d in OUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def fetch_bars(ctx, instrument, granularity, months):
    bar_sec = GRANULARITY_SECONDS[granularity]
    end_dt  = datetime.now(timezone.utc)
    # Only fetch trading hours (FX ~5d/7, ~72% uptime) — add extra buffer
    start_dt = end_dt - timedelta(days=int(months * 31 * 1.45))
    est_bars = int((end_dt - start_dt).total_seconds() / bar_sec)
    n_req = max(1, est_bars // BARS_PER_REQ)
    print(f"\n[{instrument} {granularity}] {start_dt.date()} → {end_dt.date()}"
          f"  est_bars≈{est_bars:,}  n_req≈{n_req}", flush=True)

    all_rows = []
    fetch_from = start_dt
    n_done = 0
    last_print = time.time()

    while fetch_from < end_dt:
        from_str = fetch_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        for attempt in range(5):
            try:
                resp = ctx.instrument.candles(
                    instrument=instrument,
                    granularity=granularity,
                    count=BARS_PER_REQ,
                    price="MBA",
                    fromTime=from_str,
                )
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  err: {e}  retrying in {wait}s")
                time.sleep(wait)
        else:
            print("  gave up after 5 attempts")
            break

        if resp.status != 200:
            print(f"  HTTP {resp.status} — breaking")
            break

        candles = resp.body.get("candles", [])
        if not candles:
            break

        last_ts = None
        for c in candles:
            if not c.complete:
                continue
            bid = c.bid if (hasattr(c, 'bid') and c.bid) else c.mid
            ask = c.ask if (hasattr(c, 'ask') and c.ask) else c.mid
            bo, bh, bl, bc = float(bid.o), float(bid.h), float(bid.l), float(bid.c)
            ao, ah, al, ac = float(ask.o), float(ask.h), float(ask.l), float(ask.c)
            all_rows.append({
                "timestamp": str(c.time),
                "open":   (bo + ao) / 2,
                "high":   (bh + ah) / 2,
                "low":    (bl + al) / 2,
                "close":  (bc + ac) / 2,
                "bid_c":  bc,
                "ask_c":  ac,
                "volume": int(c.volume) if hasattr(c, 'volume') and c.volume else 0,
            })
            last_ts = str(c.time)

        n_done += 1
        if time.time() - last_print > 10:
            pct = 100 * len(all_rows) / max(1, est_bars)
            print(f"  req={n_done}  bars={len(all_rows):,}  {pct:.0f}%  last={last_ts}", flush=True)
            last_print = time.time()

        if last_ts is None:
            break
        # Advance to next bar after last fetched
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        fetch_from = last_dt + timedelta(seconds=bar_sec)
        if fetch_from >= end_dt:
            break
        time.sleep(0.08)  # rate limit: ~12 req/s

    if not all_rows:
        print(f"  No bars fetched for {granularity}!")
        return

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df[["open","high","low","close","bid_c","ask_c"]] = (
        df[["open","high","low","close","bid_c","ask_c"]].astype("float32")
    )
    out_path = OUT_DIRS[granularity] / f"{instrument}_{granularity}_BA.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  ✅ Saved {len(df):,} bars → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=6, help="History depth in months")
    args = ap.parse_args()

    api_key = os.environ.get("OANDA_API_KEY", "")
    ctx = v20.Context(hostname="api-fxtrade.oanda.com", port="443", token=api_key)

    print(f"Fetching {INSTRUMENT} lower-TF bars  months={args.months}")
    for gran in ["M1", "S30", "S5"]:
        fetch_bars(ctx, INSTRUMENT, gran, args.months)
    print("\nDone.")


if __name__ == "__main__":
    main()
