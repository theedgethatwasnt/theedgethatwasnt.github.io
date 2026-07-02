"""
Fetch M5 bid/ask OHLC (BA) historical data from OANDA for all ZR pairs.
Saves to data/m5_ba/{pair}_M5_BA.parquet with columns:
  timestamp, open, high, low, close, bid_o, bid_h, bid_l, bid_c,
  ask_o, ask_h, ask_l, ask_c, volume

Usage:
    python3 fetch_m5_ba.py [--years 2]

Purpose: backtesting the live-spread ZR model using real historical spreads
instead of TOD/volume proxies.
"""
import os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from dotenv import load_dotenv
load_dotenv()

import v20

PAIRS = ["CHF_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "USD_JPY", "GBP_USD", "CAD_JPY"]
OUT_DIR = Path(__file__).parents[3] / "data" / "m5_ba"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_REQUEST = 5000  # OANDA max
M5_SECONDS = 300


def fetch_m5_ba_pair(ctx, instrument: str, years: float) -> pd.DataFrame:
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(365 * years))
    total_bars = int((end_dt - start_dt).total_seconds() / M5_SECONDS)
    print(f"  [{instrument}] Fetching ~{total_bars:,} M5 BA bars "
          f"({start_dt.date()} → {end_dt.date()})", flush=True)

    all_rows = []
    # Walk forward in chunks from start_dt
    fetch_from = start_dt
    n_req = 0

    while fetch_from < end_dt:
        from_str = fetch_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            resp = ctx.instrument.candles(
                instrument=instrument,
                granularity="M5",
                count=BARS_PER_REQUEST,
                price="MBA",
                fromTime=from_str,
            )
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
            bid = c.bid if (hasattr(c, 'bid') and c.bid) else c.mid
            ask = c.ask if (hasattr(c, 'ask') and c.ask) else c.mid
            all_rows.append({
                "timestamp": str(c.time),
                "open":  float(c.mid.o), "high": float(c.mid.h),
                "low":   float(c.mid.l), "close": float(c.mid.c),
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
            # Advance past last fetched bar (add M5 = 300s)
            fetch_from = pd.Timestamp(last_ts).to_pydatetime() + timedelta(seconds=300)
        else:
            break

        prog = min(100, 100 * len(all_rows) / total_bars)
        print(f"  [{instrument}] req#{n_req}: +{n_added} bars → {len(all_rows):,} total ({prog:.0f}%)",
              flush=True)
        time.sleep(0.2)  # rate limit

    print(f"  [{instrument}] Done: {len(all_rows):,} bars fetched")
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=float, default=2.0,
                        help='Years of history to fetch (default: 2)')
    parser.add_argument('--pairs', type=str, default='',
                        help='Comma-separated pairs to fetch (default: all ZR pairs)')
    args = parser.parse_args()

    pairs = ([p.strip() for p in args.pairs.split(',') if p.strip()]
             if args.pairs else PAIRS)

    api_key = os.environ.get('OANDA_API_KEY', '')
    if not api_key:
        raise ValueError("OANDA_API_KEY not set")

    ctx = v20.Context(
        hostname='api-fxtrade.oanda.com',
        port='443',
        token=api_key,
    )

    print(f"Fetching {args.years}y M5 BA data for: {pairs}")
    print(f"Output directory: {OUT_DIR}\n")

    for pair in pairs:
        out_path = OUT_DIR / f"{pair}_M5_BA.parquet"
        df = fetch_m5_ba_pair(ctx, pair, args.years)
        if df.empty:
            print(f"  [{pair}] EMPTY — skipping")
            continue
        df.to_parquet(out_path, index=False)
        spread_pips_map = {
            'CHF_JPY': 0.01, 'AUD_JPY': 0.01, 'EUR_JPY': 0.01,
            'NZD_JPY': 0.01, 'USD_JPY': 0.01, 'CAD_JPY': 0.01,
            'GBP_USD': 0.0001,
        }
        pip = spread_pips_map.get(pair, 0.01)
        sp = (df.ask_c - df.bid_c) / pip
        print(f"  [{pair}] Saved → {out_path}")
        print(f"    Spread stats: min={sp.min():.2f}p  med={sp.median():.2f}p  "
              f"p90={sp.quantile(0.9):.2f}p  max={sp.max():.2f}p  "
              f"pct>2.5p={100*(sp>2.5).mean():.1f}%\n")


if __name__ == "__main__":
    main()
