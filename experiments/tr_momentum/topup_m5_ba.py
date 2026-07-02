"""
Incrementally top-up M5 bid/ask OHLC parquet files for all 12 FX pairs.

Reads the last timestamp from each existing parquet and fetches only newer
bars from OANDA, then appends, deduplicates, sorts, and saves.

Parquet schema (preserved exactly):
  timestamp, open, high, low, close,
  bid_o, bid_h, bid_l, bid_c,
  ask_o, ask_h, ask_l, ask_c,
  volume

Usage:
    cd /path/to/projects/fx-core
    python3 research/experiments/tr_momentum/topup_m5_ba.py [--pairs GBP_JPY,USD_JPY]

If a parquet is missing entirely, falls back to a full 5.5-year fetch.
"""
import os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

# ------------------------------------------------------------------
# Path setup — project root is 3 levels up from this file
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import v20

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
ALL_PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "CHF_JPY", "CAD_JPY",
    "AUD_USD", "NZD_JPY", "NZD_USD", "EUR_GBP",
]

OUT_DIR = PROJECT_ROOT / "data" / "m5_ba"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_REQUEST = 5000   # OANDA max per request
M5_SECONDS       = 300    # 5 minutes in seconds
FALLBACK_YEARS   = 5.5    # full fetch when parquet is missing
RATE_LIMIT_SLEEP = 0.2    # seconds between OANDA requests

# Pip size per pair for spread display
PIP_MAP = {
    "GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01,
    "AUD_JPY": 0.01, "CHF_JPY": 0.01, "CAD_JPY": 0.01,
    "NZD_JPY": 0.01,
    "GBP_USD": 0.0001, "EUR_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}


# ------------------------------------------------------------------
# Core fetch: walk forward from fetch_from to now in 5000-bar chunks
# ------------------------------------------------------------------
def fetch_bars_from(ctx, instrument: str, from_dt: datetime) -> list[dict]:
    """
    Fetch all complete M5 bars from from_dt to now.
    Returns a list of row dicts (same schema as parquet).
    """
    end_dt   = datetime.now(timezone.utc)
    # Estimate for progress display only
    total_est = max(1, int((end_dt - from_dt).total_seconds() / M5_SECONDS))

    all_rows: list[dict] = []
    fetch_from = from_dt
    n_req = 0

    while fetch_from < end_dt:
        from_str = fetch_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        # --- API call with retry ---
        for attempt in range(1, 6):
            try:
                resp = ctx.instrument.candles(
                    instrument=instrument,
                    granularity="M5",
                    count=BARS_PER_REQUEST,
                    price="MBA",
                    fromTime=from_str,
                )
                break
            except Exception as e:
                wait = attempt * 5
                print(f"  [{instrument}] API error (attempt {attempt}/5): {e} "
                      f"— retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"  [{instrument}] Exhausted retries — stopping fetch.", flush=True)
            break

        if resp.status != 200:
            print(f"  [{instrument}] HTTP {resp.status} — retrying in 5s", flush=True)
            time.sleep(5)
            continue

        candles = resp.body.get("candles", [])
        if not candles:
            break  # no more data

        n_added = 0
        last_ts = None
        for c in candles:
            if not c.complete:
                # in-progress bar — skip; also signals end of available data
                continue
            bid = c.bid if (hasattr(c, "bid") and c.bid) else c.mid
            ask = c.ask if (hasattr(c, "ask") and c.ask) else c.mid
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
            # Advance pointer past the last complete bar
            fetch_from = (pd.Timestamp(last_ts).to_pydatetime().replace(tzinfo=timezone.utc)
                          + timedelta(seconds=M5_SECONDS))
        else:
            # No complete bars in this batch (probably at the live edge)
            break

        prog = min(100, 100 * len(all_rows) / total_est)
        print(f"  [{instrument}] req#{n_req}: +{n_added} bars "
              f"→ {len(all_rows):,} new total ({prog:.0f}%)", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)

    return all_rows


# ------------------------------------------------------------------
# Per-pair top-up logic
# ------------------------------------------------------------------
def topup_pair(ctx, pair: str) -> None:
    out_path = OUT_DIR / f"{pair}_M5_BA.parquet"

    # --- Load existing parquet (if present) ---
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        last_ts: pd.Timestamp = existing["timestamp"].max()
        last_dt = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
        # Fetch from the bar *after* the last stored bar
        fetch_from = last_dt + timedelta(seconds=M5_SECONDS)
        now = datetime.now(timezone.utc)
        gap_bars = max(0, int((now - fetch_from).total_seconds() / M5_SECONDS))
        print(f"[{pair}] Existing: {len(existing):,} bars, last={last_ts.date()}. "
              f"Gap: ~{gap_bars:,} bars to fetch.", flush=True)
    else:
        # Full fallback fetch
        fetch_from = datetime.now(timezone.utc) - timedelta(days=int(365 * FALLBACK_YEARS))
        existing = pd.DataFrame()
        print(f"[{pair}] No parquet found — full fetch from {fetch_from.date()}.", flush=True)

    # --- Fetch new bars ---
    now = datetime.now(timezone.utc)
    if fetch_from >= now:
        print(f"[{pair}] Already up to date. Nothing to fetch.\n", flush=True)
        return

    new_rows = fetch_bars_from(ctx, pair, fetch_from)

    if not new_rows:
        print(f"[{pair}] No new complete bars returned. Parquet unchanged.\n", flush=True)
        return

    new_df = pd.DataFrame(new_rows)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

    # --- Append, deduplicate, sort ---
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    before_dedup = len(combined)
    combined = (combined
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
    dupes_removed = before_dedup - len(combined)

    # Enforce column order matching the expected schema
    col_order = [
        "timestamp", "open", "high", "low", "close",
        "bid_o", "bid_h", "bid_l", "bid_c",
        "ask_o", "ask_h", "ask_l", "ask_c",
        "volume",
    ]
    combined = combined[col_order]

    # --- Save ---
    combined.to_parquet(out_path, index=False)

    # --- Summary ---
    date_min = combined["timestamp"].min().date()
    date_max = combined["timestamp"].max().date()
    new_bars  = len(new_df)
    total_bars = len(combined)

    pip = PIP_MAP.get(pair, 0.0001)
    sp  = (combined["ask_c"] - combined["bid_c"]) / pip
    sp_new = (new_df["ask_c"] - new_df["bid_c"]) / pip

    print(f"[{pair}] Done.")
    print(f"  Added : {new_bars:,} bars "
          f"({new_df['timestamp'].min().date()} → {new_df['timestamp'].max().date()})")
    print(f"  Total : {total_bars:,} bars ({date_min} → {date_max})"
          + (f"  [removed {dupes_removed} dupes]" if dupes_removed else ""))
    print(f"  Spread (new): med={sp_new.median():.2f}p  p90={sp_new.quantile(0.9):.2f}p  "
          f"max={sp_new.max():.2f}p")
    print(f"  Spread (all): med={sp.median():.2f}p  p90={sp.quantile(0.9):.2f}p\n",
          flush=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-up M5 BA parquets for all 12 TR momentum pairs."
    )
    parser.add_argument(
        "--pairs", type=str, default="",
        help="Comma-separated subset of pairs (default: all 12)",
    )
    args = parser.parse_args()

    pairs = (
        [p.strip() for p in args.pairs.split(",") if p.strip()]
        if args.pairs
        else ALL_PAIRS
    )

    # Validate pairs
    unknown = [p for p in pairs if p not in ALL_PAIRS]
    if unknown:
        print(f"WARNING: Unknown pairs (will still attempt): {unknown}", flush=True)

    api_key = os.environ.get("OANDA_API_KEY", "")
    if not api_key:
        raise ValueError("OANDA_API_KEY not set — check .env in project root")

    ctx = v20.Context(
        hostname="api-fxtrade.oanda.com",
        port="443",
        token=api_key,
    )

    print(f"Top-up M5 BA parquets for: {pairs}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Run time  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    for pair in pairs:
        try:
            topup_pair(ctx, pair)
        except Exception as e:
            print(f"[{pair}] FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print()  # blank line, continue with next pair

    print("All pairs processed.")


if __name__ == "__main__":
    main()
