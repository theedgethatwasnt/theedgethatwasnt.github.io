"""Workstream C — cross-asset D1 gate data (2026-07-07).

Two-part fetch, because the OANDA token lives only in the VPS curator container:

  1. REMOTE (inside fx-core-fx-data-curator-1): the `REMOTE_SNIPPET` below is piped
     via  `ssh aharon@87.99.154.24 'docker exec -i fx-core-fx-data-curator-1 python3 -'`
     and prints ###BEGIN/###END-delimited CSV per instrument to stdout.
  2. LOCAL: `reconstruct(raw_text)` parses that stream into
     data/cross_asset/<INSTRUMENT>_D1.parquet  (mid OHLC + tick volume, complete bars
     only, deduped + sorted).

Fetched 2026-07-07 coverage (paginated by `to=` + `includeLast=false`, 5000/page):
  SPX500_USD  6067 bars  2003-03-23 -> 2026-07-05
  NAS100_USD  6066 bars  2003-03-23 -> 2026-07-05
  XAU_USD     5607 bars  2006-03-19 -> 2026-07-05
  WTICO_USD   6363 bars  2003-01-22 -> 2026-07-05
Total 936 KB — committed (gitignore negation for data/cross_asset/).

Usage to refresh:
  ssh aharon@87.99.154.24 'docker exec -i fx-core-fx-data-curator-1 python3 -' \
      < <(python3 fetch_cross_asset.py --emit-remote) > /tmp/cross_asset_raw.txt
  python3 fetch_cross_asset.py --reconstruct /tmp/cross_asset_raw.txt
"""
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "cross_asset"

REMOTE_SNIPPET = r'''
import os, json
import urllib.request

TOKEN = os.environ["OANDA_API_KEY"]
HOST = "https://api-fxtrade.oanda.com"
INSTRUMENTS = ["SPX500_USD", "NAS100_USD", "XAU_USD", "WTICO_USD"]

def fetch(instr):
    rows = []
    to_time = None
    while True:
        url = f"{HOST}/v3/instruments/{instr}/candles?granularity=D&price=M&count=5000"
        if to_time:
            url += f"&to={to_time}&includeLast=false"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    # complete bars only (R1)
        candles = [c for c in data.get("candles", []) if c.get("complete")]
        if not candles:
            break
        batch = [(c["time"], c["mid"]["o"], c["mid"]["h"], c["mid"]["l"], c["mid"]["c"], c["volume"]) for c in candles]
        rows = batch + rows
        if len(candles) < 4999:
            break
        to_time = candles[0]["time"]
        if len(rows) > 30000:
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
        df = pd.read_csv(io.StringIO("\n".join(lines)),
                         names=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        path = out_dir / f"{instr}_D1.parquet"
        df.to_parquet(path, index=False)
        print(f"{instr}: {len(df)} bars  {df.time.min().date()} -> {df.time.max().date()}")


if __name__ == "__main__":
    if "--emit-remote" in sys.argv:
        print(REMOTE_SNIPPET)
    elif "--reconstruct" in sys.argv:
        raw = Path(sys.argv[sys.argv.index("--reconstruct") + 1]).read_text()
        reconstruct(raw)
    else:
        print(__doc__)
