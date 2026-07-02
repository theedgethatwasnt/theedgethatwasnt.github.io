#!/usr/bin/env python3
"""Confluence rarity across S30/M1/M5/M30/H1 for BOTH entry ideas.

For each timeframe, compute the per-bar signal STATE for two entry rules on SMA50 ±1σ:
  PULLBACK (idea #1):  slope>0 & low<=lower -> +1(long) ; slope<0 & high>=upper -> -1(short)
  BREAKOUT (idea #2):  slope just turned +  & close>upper -> +1 ; slope just turned - & close<lower -> -1
States are held per bar and projected CAUSALLY onto the finest (S30) grid (a coarser TF
contributes the last bar CLOSED at/before the S30 bar). A combo is 'in confluence' when ALL
its TFs share the same non-zero state. entries/yr/pair = distinct confluence EPISODES / year.
Reports singles, all 2-TF PAIRS (incl. S30/M5, M1/M30, M5/H1), and 3-TF, for BOTH rules.
MEMORY-SAFE: load each pair's S5 ONCE, del after resampling.
"""
import sys, gc
from itertools import combinations
import numpy as np, pandas as pd
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
TFS = [("S30", 30_000_000_000), ("M1", 60_000_000_000), ("M5", 300_000_000_000),
       ("M30", 1_800_000_000_000), ("H1", 3_600_000_000_000)]
NAMES = [t[0] for t in TFS]; NT = len(TFS)
SLOPE_LB = 3
YR_NS = 365.25 * 24 * 3600 * 1e9


def resample(df, ts_i64, tf_ns):
    bucket = ts_i64 // tf_ns
    g = df.groupby(bucket, sort=True)
    tsh = g['c'].last().index.to_numpy() * tf_ns
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(), tsh)


def states(h, l, c):
    """returns (pullback_state, breakout_state) int8 arrays."""
    n = len(c)
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    sd = pd.Series(c).rolling(50).std(ddof=0).to_numpy()
    valid = ~np.isnan(sma)
    lower = np.where(valid, sma - sd, -1e18); upper = np.where(valid, sma + sd, 1e18)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
    slope = np.where(valid, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    pull = np.where((slope > 0) & (l <= lower), 1,
                    np.where((slope < 0) & (h >= upper), -1, 0)).astype(np.int8)
    tneg = (slope < 0) & (sp >= 0) & valid; tpos = (slope > 0) & (sp <= 0) & valid
    brk = np.where(tpos & (c > upper), 1, np.where(tneg & (c < lower), -1, 0)).astype(np.int8)
    return pull, brk


def episodes(mask):
    if not mask.any():
        return 0
    return int((np.diff(mask.astype(np.int8)) == 1).sum()) + int(mask[0])


COMBOS = [cb for r in (1, 2, 3) for cb in combinations(range(NT), r)]
acc = {"pull": {cb: 0 for cb in COMBOS}, "brk": {cb: 0 for cb in COMBOS}}
years = 0.0
for p in PAIRS:
    d = load_pair_ba(p)
    ts_i64 = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c']})
    del d; gc.collect()
    projP = {}; projB = {}; s30_tsh = None
    for i, (nm, tf_ns) in enumerate(TFS):
        h, l, c, tsh = resample(df, ts_i64, tf_ns)
        pull, brk = states(h, l, c)
        closed = tsh + tf_ns
        if nm == "S30":
            s30_tsh = tsh; projP[i] = pull; projB[i] = brk
        else:
            idx = np.searchsorted(closed, s30_tsh, side="right") - 1
            ok = idx >= 0; ci = np.clip(idx, 0, len(pull) - 1)
            projP[i] = np.where(ok, pull[ci], 0).astype(np.int8)
            projB[i] = np.where(ok, brk[ci], 0).astype(np.int8)
        del h, l, c, tsh, pull, brk; gc.collect()
    yrs = (s30_tsh[-1] - s30_tsh[0]) / YR_NS; years += yrs
    for key, proj in (("pull", projP), ("brk", projB)):
        for cb in COMBOS:
            lm = np.ones(len(s30_tsh), bool); sm = np.ones(len(s30_tsh), bool)
            for i in cb:
                lm &= (proj[i] == 1); sm &= (proj[i] == -1)
            acc[key][cb] += episodes(lm) + episodes(sm)
    print(f"  {p} done ({yrs:.2f}yr)", flush=True)
    del df, ts_i64, projP, projB, s30_tsh; gc.collect()


def lbl(cb): return "+".join(NAMES[i] for i in cb)
for key, title in (("pull", "PULLBACK (idea #1: fade to band in trend)"),
                   ("brk", "BREAKOUT (idea #2: slope-flip + band break)")):
    A = acc[key]
    print(f"\n================  {title}  —  entries/yr/pair  ================")
    print("  singles:  " + "   ".join(f"{NAMES[i]}={A[(i,)]/max(years,1e-9):.0f}" for i in range(NT)))
    print("  -- 2-TF pairs --")
    for cb in [c for c in COMBOS if len(c) == 2]:
        print(f"    {lbl(cb):>10} {A[cb]/max(years,1e-9):>8.1f}/yr")
    print("  -- 3-TF --")
    for cb in [c for c in COMBOS if len(c) == 3]:
        print(f"    {lbl(cb):>14} {A[cb]/max(years,1e-9):>8.2f}/yr")
print(f"\ntotal ~{years:.1f} pair-years. Guide: >~50/yr/pair = tradeable & testable; "
      "<~10/yr = too rare to validate. (Breakout confluence ~0 expected: fresh slope-crosses "
      "rarely coincide across TFs -> use higher-TF as a trend FILTER, not simultaneous.)")
