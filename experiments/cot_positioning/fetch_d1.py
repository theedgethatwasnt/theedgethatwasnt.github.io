"""fetch_d1.py — COT Contrarian Positioning: OANDA D1 mid+bid/ask deep-history fetch.

Two-part fetch, extending the pattern of multiday_contrarian/fetch_cross_asset.py
(OANDA token lives only in the VPS curator container):

  1. REMOTE (inside fx-core-fx-data-curator-1): REMOTE_SNIPPET piped via
     `ssh <user>@<vps-host> 'docker exec -i fx-core-fx-data-curator-1 python3 -'`,
     prints ###BEGIN/###END-delimited CSV per pair to stdout.
  2. LOCAL: `reconstruct(raw_text)` parses that stream into
     data/d1_deep/<PAIR>_D1.parquet (mid OHLC + bid_c/ask_c close + tick volume).

price="MBA" (mid+bid+ask) so bid_c/ask_c close is real, not defaulted to mid — needed
both for R3a-compliant spread cost AND because the pre-registration wants "our measured
[D1 spread] medians" computed from real data, not an external assumption.

HARD CEILING: 2026-05-21 (matches the sibling experiments' data edge — PREREGISTRATION.md
"Prices"). Any bar timestamped after the ceiling is dropped at reconstruction time, not
merely filtered downstream — this fetcher enforces the ceiling itself so no file on disk
ever contains post-ceiling rows.

12 pairs — chosen to guarantee a DIRECT USD leg for every one of the 7 COT currencies
(EUR_USD, USD_JPY, GBP_USD, USD_CHF, AUD_USD, USD_CAD, NZD_USD = 7 legs the signal
actually trades), plus 5 major crosses for completeness/comparability with the sibling
experiments' pair universe (AUD_JPY, EUR_GBP, EUR_JPY, GBP_JPY, NZD_JPY). This differs
from is_data.PAIRS (multiday_contrarian's 12) which has CAD_JPY/CHF_JPY instead of
USD_CAD/USD_CHF — that set has no direct CAD or CHF vs USD leg, which this experiment
requires. Documented, not silently diverged.

Usage to refresh:
  ssh <user>@<vps-host> 'docker exec -i fx-core-fx-data-curator-1 python3 -' \\
      < <(python3 fetch_d1.py --emit-remote) > /tmp/d1_deep_raw.txt
  python3 fetch_d1.py --reconstruct /tmp/d1_deep_raw.txt
"""
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# NOTE (2026-07-07): deliberately NOT data/d1_deep/ — that path collided with a concurrent
# sibling experiment (research/experiments/momentum_confirm/fetch_d1_deep.py), which uses
# the SAME filenames but a mid-only schema (no bid_c/ask_c, column "timestamp" not "time")
# and silently clobbered 10/12 of this experiment's bid/ask files mid-session. This
# experiment's data has a materially different, incompatible schema (adds bid_c/ask_c for
# the real-spread cost model) so it gets its own directory to avoid repeat collisions.
OUT_DIR = REPO / "data" / "d1_deep_ba"

CEILING = "2026-05-21T00:00:00Z"

PAIRS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "AUD_JPY", "EUR_GBP", "EUR_JPY", "GBP_JPY", "NZD_JPY",
]

REMOTE_SNIPPET = r'''
import os, json
import urllib.request

TOKEN = os.environ["OANDA_API_KEY"]
HOST = "https://api-fxtrade.oanda.com"
INSTRUMENTS = ''' + repr(PAIRS) + r'''

def fetch(instr):
    rows = []
    to_time = None
    while True:
        url = f"{HOST}/v3/instruments/{instr}/candles?granularity=D&price=MBA&count=5000"
        if to_time:
            url += f"&to={to_time}&includeLast=false"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        candles = [c for c in data.get("candles", []) if c.get("complete")]
        if not candles:
            break
        batch = [(c["time"], c["mid"]["o"], c["mid"]["h"], c["mid"]["l"], c["mid"]["c"],
                  c["bid"]["c"], c["ask"]["c"], c["volume"]) for c in candles]
        rows = batch + rows
        if len(candles) < 4999:
            break
        to_time = candles[0]["time"]
        if len(rows) > 20000:
            break
    return rows

for instr in INSTRUMENTS:
    try:
        rows = fetch(instr)
    except Exception as e:
        print(f"###ERROR {instr} {e}", flush=True)
        continue
    print(f"###BEGIN {instr} {len(rows)}", flush=True)
    for t, o, h, l, c, b, a, v in rows:
        print(f"{t},{o},{h},{l},{c},{b},{a},{v}")
    print(f"###END {instr}", flush=True)
'''


def reconstruct(raw_text: str, out_dir: Path = OUT_DIR, ceiling: str = CEILING) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    ceiling_ts = pd.Timestamp(ceiling)
    blocks, cur = {}, None
    for line in raw_text.splitlines():
        if line.startswith("###BEGIN"):
            cur = line.split()[1]
            blocks[cur] = []
        elif line.startswith("###END"):
            cur = None
        elif line.startswith("###ERROR"):
            print("REMOTE ERROR:", line, file=sys.stderr)
        elif cur:
            blocks[cur].append(line)

    for instr, lines in blocks.items():
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            names=["time", "open", "high", "low", "close", "bid_c", "ask_c", "volume"],
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        n_before = len(df)
        df = df[df["time"] <= ceiling_ts]
        n_dropped = n_before - len(df)
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "bid_c", "ask_c"]:
            df[c] = df[c].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        path = out_dir / f"{instr}_D1.parquet"
        df.to_parquet(path, index=False)
        print(f"{instr}: {len(df)} bars  {df.time.min().date()} -> {df.time.max().date()}"
              f"  (dropped {n_dropped} post-ceiling)")


if __name__ == "__main__":
    if "--emit-remote" in sys.argv:
        print(REMOTE_SNIPPET)
    elif "--reconstruct" in sys.argv:
        raw = Path(sys.argv[sys.argv.index("--reconstruct") + 1]).read_text()
        reconstruct(raw)
    else:
        print(__doc__)
