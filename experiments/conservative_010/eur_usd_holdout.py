#!/usr/bin/env python3
"""Does EUR_USD make money OUT-OF-SAMPLE (in time)? EUR_USD_S5_BA has 8.26M rows
(2024-02..2026-05); the IS study only read the LAST 5M. The EARLIER ~3.26M rows are
untouched -> a genuine time-OOS for EUR_USD with the FROZEN H1 breakout config
(SMA50, K=1, TP50/SL100). Compares held-out (early) vs IS (recent 5M). Nothing tuned.
MEMORY-SAFE: single pair.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo, equity_drawdown

S5 = "/path/to/projects/fx-core/data/s5_ohlc/EUR_USD_S5_BA.parquet"
MAX_ROWS = 5_000_000; SLIP = 2.0; SLOPE_LB = 3; H1_NS = 3_600_000_000_000
K = 1.0; TP = 50.0; SL = 100.0; PIP = 0.0001


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if tneg[i] and c[i] < lower[i]: pos = -1; entry = bid[i]; continue
            if tpos[i] and c[i] > upper[i]: pos = 1; entry = ask[i]; continue
        if pos != 0:
            rsn = -1; exf = 0.0; fc = entry - pos*sl*pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0:
                tpl = entry + pos*tp*pip
                if pos == 1 and h[i] >= tpl: exf = tpl; rsn = 2
                elif pos == -1 and l[i] <= tpl: exf = tpl; rsn = 2
            if rsn >= 0:
                _pnl[nt] = (exf-entry)/pip if pos == 1 else (entry-exf)/pip
                nt += 1; pos = 0
    return _pnl[:nt]


def run(df, label):
    ts = df['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    b = ts // H1_NS
    g = pd.DataFrame({'h': df['high'].to_numpy(), 'l': df['low'].to_numpy(),
                      'c': df['close'].to_numpy(), 'bid': df['bid_c'].to_numpy(),
                      'ask': df['ask_c'].to_numpy()}).groupby(b, sort=True)
    h = g['h'].max().to_numpy(); l = g['l'].min().to_numpy(); c = g['c'].last().to_numpy()
    bid = g['bid'].last().to_numpy(); ask = g['ask'].last().to_numpy(); n = len(c)
    s = pd.Series(c); sma = s.rolling(50).mean().to_numpy(); sd = s.rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(sma) & ~np.isnan(sd)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]; slope = np.where(v, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    tneg = (slope < 0) & (sp >= 0) & v; tpos = (slope > 0) & (sp <= 0) & v
    lower = np.where(v, sma - K*sd, -1e18); upper = np.where(v, sma + K*sd, 1e18)
    pnl = _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, PIP, TP, SL, SLIP)
    N = len(pnl); eb = np.arange(N)
    wf = sum(1 for f in walk_forward(eb, pnl, N, 6) if f['net']>0) if N >= 30 else 0
    mc = monte_carlo(pnl, 300)['p_net'] if N >= 30 else 1.0
    dd = equity_drawdown(np.cumsum(pnl))['max_dd'] if N else 0
    d0 = str(df['timestamp'].iloc[0])[:10]; d1 = str(df['timestamp'].iloc[-1])[:10]
    print(f"  {label:>18} [{d0}..{d1}]: net={pnl.sum():>7.0f}p n={N:>4} "
          f"WR={100*np.mean(pnl>0) if N else 0:.0f}% WF {wf}/6 MC p_net={mc:.3f} eqDD={dd:.0f}p")
    return pnl.sum(), mc


print("EUR_USD — FROZEN H1 breakout (SMA50/K=1/TP50/SL100). time-OOS = untouched early rows.\n")
full = pq.read_table(S5, columns=['timestamp', 'high', 'low', 'close', 'bid_c', 'ask_c']).to_pandas()
full = full.sort_values('timestamp').reset_index(drop=True)
cut = len(full) - MAX_ROWS
held = full.iloc[:cut].reset_index(drop=True)     # EARLY, never read
recent = full.iloc[cut:].reset_index(drop=True)   # last 5M = the IS window
del full; gc.collect()
run(held, "HELD-OUT (early)")
run(recent, "IS (recent 5M)")
print("\nEUR_USD makes money OOS only if HELD-OUT is positive with MC p<0.05.")
