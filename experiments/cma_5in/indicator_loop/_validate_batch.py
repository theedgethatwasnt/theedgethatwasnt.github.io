"""Validate a batch of ported candidates in one call.
Usage: python3 _validate_batch.py roc_10 range_pos_30 rsi_14 bb_width aroon_osc
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
from validate import validate_candidate, print_result, ValidationError
from loop import REF_FNS

LOOP_DIR = Path(__file__).parent
CAND = json.loads((LOOP_DIR / "candidates.json").read_text())


def main():
    names = sys.argv[1:]
    if not names:
        print("Usage: _validate_batch.py name1 name2 ...")
        return 1
    pair = "EUR_JPY"
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    print(f"Loaded {len(df)} bars of {pair}\n")

    meta_by_name = {c["name"]: c for c in CAND["candidates"]}

    all_passed = True
    for name in names:
        if name not in REF_FNS:
            print(f"\n[{name}] no reference_fn — skipping")
            continue
        rng = tuple(meta_by_name[name]["range"])
        try:
            r = validate_candidate(name, REF_FNS[name], name, df, rng, probe_bar=5000, n_bars_limit=20000)
            print_result(r)
        except ValidationError as e:
            print(f"\n[{name}] ❌ FAIL: {e}")
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
