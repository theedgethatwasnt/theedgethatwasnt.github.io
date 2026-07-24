"""fetch_d1_deep.py — momentum_confirm Stage 1: deep D1 FX price history (2005ish -> 2026-05-21
ceiling), for the 12 fx_factors pairs. Same two-part remote-fetch/local-reconstruct pattern as
research/experiments/multiday_contrarian/fetch_cross_asset.py (OANDA token lives only inside the
VPS curator container). Mid-price only ("price=M") — spread for the deep segment is NOT sourced
from historical bid/ask (unavailable this far back); Stage 1 injects the fx_factors MEASURED
MEDIAN per-pair round-trip spread (see momentum_confirm/median_spread.csv) as a constant
synthetic bid_c/ask_c offset around each D1 mid close, so the frozen rebalance_engine.py's
existing (ask_c-bid_c)/pip cost logic runs completely unmodified on the new data (R9: documented
data-source difference from the M5-BA-sourced IS/OOS window, not a code change).

Deep D1 is a SHARED resource (data/d1_deep/<PAIR>_D1.parquet, full history, no segment
filtering) — Stage 1's own hard <=2020-10-31 filter lives in stage1_data.py, not here.

Usage:
  ssh <user>@<vps-host> 'docker exec -i fx-core-fx-data-curator-1 python3 -' \
      < <(python3 fetch_d1_deep.py --emit-remote) > /tmp/d1_deep_raw.txt
  python3 fetch_d1_deep.py --reconstruct /tmp/d1_deep_raw.txt
"""
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "d1_deep"

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
    "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY",
]

REMOTE_SNIPPET = r'''
import os, json
import urllib.request

TOKEN = os.environ["OANDA_API_KEY"]
HOST = "https://api-fxtrade.oanda.com"
INSTRUMENTS = ''' + repr(PAIRS) + r'''
CEILING = "2026-05-21T00:00:00Z"

def fetch(instr):
    rows = []
    to_time = CEILING
    while True:
        url = f"{HOST}/v3/instruments/{instr}/candles?granularity=D&price=M&count=5000"
        if to_time:
            url += f"&to={to_time}&includeLast=false"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        candles = [c for c in data.get("candles", []) if c.get("complete")]
        if not candles:
            break
        batch = [(c["time"], c["mid"]["o"], c["mid"]["h"], c["mid"]["l"], c["mid"]["c"], c["volume"]) for c in candles]
        rows = batch + rows
        to_time = candles[0]["time"]
        if len(candles) < 10:
            break
        if len(rows) > 8000:
            break
    return rows

for instr in INSTRUMENTS:
    try:
        rows = fetch(instr)
    except Exception as e:
        print(f"###ERROR {instr} {e}", flush=True)
        continue
    print(f"###BEGIN {instr} {len(rows)}", flush=True)
    for t, o, h, l, c, v in rows:
        print(f"{t},{o},{h},{l},{c},{v}")
    print(f"###END {instr}", flush=True)
'''


def reconstruct(raw_text: str, out_dir: Path = OUT_DIR) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
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
        if not lines:
            print(f"{instr}: 0 rows, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(io.StringIO("\n".join(lines)),
                          names=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        path = out_dir / f"{instr}_D1.parquet"
        df.to_parquet(path, index=False)
        print(f"{instr}: {len(df)} bars  {df.timestamp.min()} -> {df.timestamp.max()}")


if __name__ == "__main__":
    if "--emit-remote" in sys.argv:
        print(REMOTE_SNIPPET)
    elif "--reconstruct" in sys.argv:
        raw = Path(sys.argv[sys.argv.index("--reconstruct") + 1]).read_text()
        reconstruct(raw)
    else:
        print(__doc__)
