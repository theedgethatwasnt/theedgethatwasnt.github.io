#!/usr/bin/env python3
"""
Phase 0 — causal P&F box-series + signed trend-age for GBP_JPY (NEAT input #1).

box=5 pips, reversal=1 box. Paints ALL boxes a bar traverses. Within-bar order =
R2 (bull bar: high then low; bear: low then high). Signed trend-age = pnf_dir ×
col_count (current column boxes, signed by direction) read at BAR CLOSE.

A fast numba kernel produces the series; it is validated BIT-FOR-BIT against the
canonical `lib/pnf_engine._update_pnf` on a subset (R6/R7 discipline) before the
full run. Output: per-box-CHANGE event rows (bar_idx, ts, signed_age, mid_close,
bid_close, ask_close) — the decision series the NEAT net trades on.
"""
import sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from lib import pnf_engine as pe  # noqa: E402

CACHE = Path(__file__).parent / "cache"; CACHE.mkdir(exist_ok=True)
# usage: phase0_pnf_series.py [PAIR] [REV] [BOX_PIPS]   (rev=3 default; 2/4/5 on back burner)
PAIR = sys.argv[1] if len(sys.argv) > 1 else "GBP_JPY"
REV = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BOX_PIPS = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
PIP = 0.01
BS = BOX_PIPS * PIP


@njit(cache=True)
def pnf_series_numba(o, h, l, c, bs, rev):
    """Replicates lib.pnf_engine._update_pnf integer mode, painting all boxes,
    R2 within-bar order. Returns per-bar (dir, col_count) after the bar."""
    n = len(c)
    dir_arr = np.zeros(n, dtype=np.int64)
    col_arr = np.zeros(n, dtype=np.int64)
    pnf_dir = 0; pnf_idx = 0; col_count = 0
    for i in range(n):
        # R2 within-bar order
        if c[i] >= o[i]:
            prices = (h[i], l[i])     # bull: high then low
        else:
            prices = (l[i], h[i])     # bear: low then high
        for price in prices:
            if pnf_dir == 0:
                pnf_idx = int(price / bs); pnf_dir = 1; col_count = 1
                continue
            new_idx = int(price / bs); delta = new_idx - pnf_idx
            if pnf_dir == 1:
                if delta >= 1:
                    pnf_idx = new_idx; col_count += delta
                elif delta <= -rev:
                    old_idx = pnf_idx; pnf_dir = -1; pnf_idx = new_idx
                    col_count = max(1, old_idx - new_idx)
            else:
                if delta <= -1:
                    pnf_idx = new_idx; col_count += (-delta)
                elif delta >= rev:
                    old_idx = pnf_idx; pnf_dir = 1; pnf_idx = new_idx
                    col_count = max(1, new_idx - old_idx)
        dir_arr[i] = pnf_dir; col_arr[i] = col_count
    return dir_arr, col_arr


def pnf_series_reference(o, h, l, c, bs, rev):
    """Canonical pure-Python path via pnf_engine._update_pnf (R6 source of truth)."""
    st = pe.PnFState()
    n = len(c); dir_arr = np.zeros(n, np.int64); col_arr = np.zeros(n, np.int64)
    for i in range(n):
        prices = (h[i], l[i]) if c[i] >= o[i] else (l[i], h[i])
        for price in prices:
            pe._update_pnf(st, price, bs, float(rev), price_based_rev=False)
        dir_arr[i] = st.pnf_dir; col_arr[i] = st.col_count
    return dir_arr, col_arr


def main():
    df = pd.read_parquet(PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    o = df["open"].values.astype(np.float64); h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64);  c = df["close"].values.astype(np.float64)
    n = len(c)
    print(f"{PAIR}: {n:,} S5 bars  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # ── R7 consistency gate: numba kernel == canonical engine on a 200k subset ──
    s = slice(0, 200_000)
    dN, cN = pnf_series_numba(o[s], h[s], l[s], c[s], BS, REV)
    dR, cR = pnf_series_reference(o[s], h[s], l[s], c[s], BS, REV)
    assert np.array_equal(dN, dR) and np.array_equal(cN, cR), \
        "R7 FAIL: numba kernel diverges from canonical pnf_engine"
    print(f"R7 gate PASS — numba kernel == lib.pnf_engine bit-for-bit on {s.stop:,} bars")

    # ── full series ─────────────────────────────────────────────────────────
    dir_arr, col_arr = pnf_series_numba(o, h, l, c, BS, REV)
    signed_age = dir_arr * col_arr
    # box-CHANGE events = bars where signed_age moved (box painted or reversal)
    chg = np.empty(n, dtype=bool); chg[0] = True
    chg[1:] = signed_age[1:] != signed_age[:-1]
    idx = np.where(chg)[0]
    ev = pd.DataFrame({
        "bar_idx": idx,
        "ts": df["timestamp"].values[idx],
        "signed_age": signed_age[idx].astype(np.int32),
        "mid": c[idx], "bid": df["bid_c"].values[idx], "ask": df["ask_c"].values[idx],
    })
    ev.to_parquet(CACHE / f"{PAIR}_pnf_b{int(BOX_PIPS)}_rev{REV}.parquet")

    # ── paint-all-boxes verification + stats ──────────────────────────────────
    dcol = np.diff(col_arr, prepend=col_arr[0])
    multibox = int((dcol > 1).sum())
    biggest = int(dcol.max())
    days = n / 17280
    print(f"\nbox-change events: {len(ev):,}  ({len(ev)/days:.1f}/day over {days:.0f} trading days)")
    print(f"paint-all-boxes: {multibox:,} bars painted >1 box in a single bar (max {biggest} boxes)")
    print(f"signed_age range: [{signed_age.min()}, {signed_age.max()}]  "
          f"(|age| p99={np.percentile(np.abs(signed_age),99):.0f})")
    print(f"reversals (dir flips): {int((np.diff(dir_arr)!=0).sum()):,}")
    print(f"cached → {CACHE/f'{PAIR}_pnf_b{int(BOX_PIPS)}_rev{REV}.parquet'}")


if __name__ == "__main__":
    main()
