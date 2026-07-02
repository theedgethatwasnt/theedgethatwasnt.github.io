#!/usr/bin/env python3
"""Winning candidate (H1 breakout): does swapping the band center SMA50 -> EMA50 improve it?

Only change vs the SMA version: the band CENTER and the SLOPE use EMA50 (span=50, causal)
instead of SMA50. Band width stays K=1 * rolling std50 (same as SMA version) so we isolate the
SMA->EMA change. Entry unchanged: slope just turned neg + close < (center - sigma) -> short;
mirror long. Same bounded fade exit. H1, multi-pair, + drop-EUR_USD column. WF + MC gates.
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba
from validate import walk_forward, monte_carlo, equity_drawdown

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SLIP = 2.0; SLOPE_LB = 3; H1_NS = 3_600_000_000_000; K = 1.0
EXITS = [(50.0, 150.0), (50.0, 100.0), (30.0, 60.0)]
MATYPES = ["SMA", "EMA"]


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if tneg[i] and c[i] < lower[i]:
                pos = -1; entry = bid[i]; _eb[nt] = i; continue
            if tpos[i] and c[i] > upper[i]:
                pos = 1; entry = ask[i]; _eb[nt] = i; continue
        if pos != 0:
            rsn = -1; exf = 0.0
            fc = entry - pos * sl * pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0:
                tpl = entry + pos*tp*pip
                if pos == 1 and h[i] >= tpl: exf = tpl; rsn = 2
                elif pos == -1 and l[i] <= tpl: exf = tpl; rsn = 2
            if rsn >= 0:
                _pnl[nt] = (exf-entry)/pip if pos == 1 else (entry-exf)/pip
                nt += 1; pos = 0
    return _eb[:nt], _pnl[:nt]


def resample(df, ts, tf):
    b = ts // tf; g = df.groupby(b, sort=True)
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(),
            g['bid'].last().to_numpy(), g['ask'].last().to_numpy(),
            g['c'].last().index.to_numpy() * tf)


def signals(c, matype):
    n = len(c); s = pd.Series(c)
    center = (s.ewm(span=50, adjust=False).mean() if matype == "EMA"
              else s.rolling(50).mean()).to_numpy()
    sd = s.rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(center) & ~np.isnan(sd)
    slope = np.zeros(n); slope[SLOPE_LB:] = center[SLOPE_LB:] - center[:-SLOPE_LB]
    slope = np.where(v, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    lower = np.where(v, center - K*sd, -1e18); upper = np.where(v, center + K*sd, 1e18)
    tneg = (slope < 0) & (sp >= 0) & v; tpos = (slope > 0) & (sp <= 0) & v
    return lower, upper, tneg, tpos


def val(tl):
    net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    if N < 30: return N, float(net.sum()), 0, 1.0, 0.0
    eb = np.arange(N)
    wf = walk_forward(eb, net, N, n_folds=6); mc = monte_carlo(net, n=300)
    return N, float(net.sum()), sum(1 for f in wf if f['net']>0), mc['p_net'], \
        equity_drawdown(np.cumsum(net))['max_dd']


trades = {(m, e): {} for m in MATYPES for e in EXITS}
for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']}); del d; gc.collect()
    h, l, c, bid, ask, tsh = resample(df, ts, H1_NS)
    for m in MATYPES:
        lower, upper, tneg, tpos = signals(c, m)
        for e in EXITS:
            tp, sl = e
            eb, pnl = _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, tp, sl, SLIP)
            et = tsh[eb]
            trades[(m, e)][p] = [{'exit_time': int(et[i]), 'pnl_pips': float(pnl[i])}
                                 for i in range(len(pnl))]
    print(f"  {p} done", flush=True)
    del df, ts, h, l, c, bid, ask; gc.collect()


def agg(dct, pairs):
    tl = []
    for pr in pairs: tl += dct.get(pr, [])
    return sorted(tl, key=lambda t: t['exit_time'])


NOEU = [p for p in PAIRS if p != "EUR_USD"]
print("\n===== H1 BREAKOUT: SMA50 vs EMA50 band center (K=1) =====")
print(f"{'MA':>4} {'TP/SL':>8} | {'4p net':>7} {'n':>5} {'WF':>4} {'MCp':>6} {'eqDD':>7} | "
      f"{'exEU net':>8} {'WF':>4} {'MCp':>6} | per-pair EJ/EU/GU/UJ")
for m in MATYPES:
    for e in EXITS:
        N4, net4, wf4, p4, dd4 = val(agg(trades[(m, e)], PAIRS))
        N3, net3, wf3, p3, _ = val(agg(trades[(m, e)], NOEU))
        pp = {pr: sum(t['pnl_pips'] for t in trades[(m, e)].get(pr, [])) for pr in PAIRS}
        print(f"{m:>4} {int(e[0])}/{int(e[1])!s:>3} | {net4:>7.0f} {N4:>5} {wf4:>3}/6 {p4:>6.3f} "
              f"{dd4:>6.0f}p | {net3:>8.0f} {wf3:>3}/6 {p3:>6.3f} | "
              f"{pp['EUR_JPY']:.0f}/{pp['EUR_USD']:.0f}/{pp['GBP_USD']:.0f}/{pp['USD_JPY']:.0f}")
print("\nRef SMA baseline (from bb50_breakout): TP50/SL150 H1 = +2,756p, MC 0.010, EUR_USD=69%.")
print("Improve = EMA lifts net/MC AND/OR broadens past EUR_USD (higher exEU net, lower exEU p).")
