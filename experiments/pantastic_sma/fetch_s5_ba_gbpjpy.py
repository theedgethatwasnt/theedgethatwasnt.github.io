"""
Fetch GBP_JPY S5 bid/ask OHLC from OANDA.
Saves to data/s5_ohlc/GBP_JPY_S5_BA.parquet

Usage:
    python3 fetch_s5_ba_gbpjpy.py [--days 30]
"""
import os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()
import v20

OUT_DIR = ROOT / "data" / "s5_ohlc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_REQUEST = 5000   # OANDA max
S5_SECONDS       = 5
PAIR             = "GBP_JPY"
PIP              = 0.01   # JPY pair


def fetch(ctx, days: int) -> pd.DataFrame:
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    est_bars = int((end_dt - start_dt).total_seconds() / S5_SECONDS)
    print(f"[{PAIR}] ~{est_bars:,} S5 bars  {start_dt.date()} → {end_dt.date()}", flush=True)

    rows, fetch_from, n_req = [], start_dt, 0

    while fetch_from < end_dt:
        from_str = fetch_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            resp = ctx.instrument.candles(
                instrument=PAIR,
                granularity="S5",
                count=BARS_PER_REQUEST,
                price="MBA",
                fromTime=from_str,
            )
        except Exception as e:
            print(f"  API error: {e} — retrying in 5s")
            time.sleep(5)
            continue

        if resp.status != 200:
            print(f"  HTTP {resp.status} — retrying in 5s")
            time.sleep(5)
            continue

        candles = resp.body.get("candles", [])
        if not candles:
            break

        last_ts, n_added = None, 0
        for c in candles:
            if not c.complete:
                continue
            bid = c.bid if (hasattr(c, 'bid') and c.bid) else c.mid
            ask = c.ask if (hasattr(c, 'ask') and c.ask) else c.mid
            rows.append({
                "timestamp": str(c.time),
                "bid_o": float(bid.o), "bid_h": float(bid.h),
                "bid_l": float(bid.l), "bid_c": float(bid.c),
                "ask_o": float(ask.o), "ask_h": float(ask.h),
                "ask_l": float(ask.l), "ask_c": float(ask.c),
                "volume": int(c.volume),
            })
            last_ts = str(c.time)
            n_added += 1

        n_req += 1
        if last_ts:
            fetch_from = pd.Timestamp(last_ts).to_pydatetime() + timedelta(seconds=S5_SECONDS)
        else:
            break

        pct = min(100, 100 * len(rows) / est_bars)
        print(f"  req#{n_req}: +{n_added} → {len(rows):,} total ({pct:.0f}%)", flush=True)
        time.sleep(0.2)

    print(f"Done: {len(rows):,} bars fetched")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()

    api_key = os.environ.get('OANDA_API_KEY', '')
    if not api_key:
        raise ValueError("OANDA_API_KEY not set")

    ctx = v20.Context(hostname='api-fxtrade.oanda.com', port='443', token=api_key)

    df = fetch(ctx, args.days)
    if df.empty:
        print("No data — check API key / pair name")
        return

    out = OUT_DIR / f"{PAIR}_S5_BA.parquet"
    df.to_parquet(out, index=False)

    sp = (df.ask_c - df.bid_c) / PIP
    print(f"\nSaved → {out}")
    print(f"Rows: {len(df):,}  ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
    print(f"Spread: min={sp.min():.2f}p  med={sp.median():.2f}p  "
          f"p90={sp.quantile(0.9):.2f}p  max={sp.max():.2f}p")


if __name__ == "__main__":
    main()
