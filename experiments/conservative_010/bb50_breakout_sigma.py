#!/usr/bin/env python3
"""H1 BREAKOUT (idea #2, the surviving lead): sweep band width Kσ x exit, + concentration test.

Entry: slope just turned neg & close < (SMA50 - K*sigma) -> SHORT; mirror long above upper.
Sweeps K in {1,1.5,2,2.5} (band std-devs) x exit {TP/SL} grid. For each config reports the
4-pair result AND the 3-pair result with EUR_USD DROPPED (does the edge survive without the
pair that carried 69% of the net?). WF + MC on both. OOS shown for reference but is SPENT
(R8) — treat WF/MC as the gates for this exploration.
MEMORY-SAFE: one pair at a time (H1 only -> small/fast).
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba
from validate import split_is_oos, walk_forward, monte_carlo, equity_drawdown

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SLIP = 2.0; SLOPE_LB = 3; H1_NS = 3_600_000_000_000
KS = [1.0, 1.5, 2.0, 2.5]
EXITS = [(30.0, 60.0), (50.0, 100.0), (50.0, 150.0), (100.0, 150.0), (150.0, 150.0)]


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


def val(tl):
    net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    if N < 30: return N, float(net.sum()), 0, 1.0, 0.0
    eb = np.arange(N)
    wf = walk_forward(eb, net, N, n_folds=6); mc = monte_carlo(net, n=300)
    dd = equity_drawdown(np.cumsum(net))['max_dd']
    return N, float(net.sum()), sum(1 for f in wf if f['net']>0), mc['p_net'], dd


trades = {(k, e): {} for k in KS for e in EXITS}   # -> {pair: [pnls with exit_time]}
for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']}); del d; gc.collect()
    h, l, c, bid, ask, tsh = resample(df, ts, H1_NS); n = len(c)
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    sd = pd.Series(c).rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(sma)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
    slope = np.where(v, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    tneg = (slope < 0) & (sp >= 0) & v; tpos = (slope > 0) & (sp <= 0) & v
    for k in KS:
        lower = np.where(v, sma - k*sd, -1e18); upper = np.where(v, sma + k*sd, 1e18)
        for e in EXITS:
            tp, sl = e
            eb, pnl = _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, tp, sl, SLIP)
            et = tsh[eb]
            trades[(k, e)][p] = [{'exit_time': int(et[i]), 'pnl_pips': float(pnl[i])}
                                 for i in range(len(pnl))]
    print(f"  {p} done", flush=True)
    del df, ts, h, l, c, bid, ask, sma, sd, slope, sp, tneg, tpos; gc.collect()


def agg(dct, pairs):
    tl = []
    for pr in pairs:
        tl += dct.get(pr, [])
    return sorted(tl, key=lambda t: t['exit_time'])


NOEU = [p for p in PAIRS if p != "EUR_USD"]
print("\n===== H1 BREAKOUT — band Kσ x exit — 4-pair vs 3-pair(drop EUR_USD) =====")
print(f"{'K':>4} {'TP/SL':>8} | {'4p net':>7} {'n':>5} {'WF':>4} {'MCp':>6} | "
      f"{'3p net':>7} {'n':>5} {'WF':>4} {'MCp':>6}")
best = []
for k in KS:
    for e in EXITS:
        N4, net4, wf4, p4, dd4 = val(agg(trades[(k, e)], PAIRS))
        N3, net3, wf3, p3, dd3 = val(agg(trades[(k, e)], NOEU))
        flag = "  <<" if (net4 > 0 and p4 < 0.05 and net3 > 0 and p3 < 0.10) else ""
        print(f"{k:>4} {int(e[0])}/{int(e[1])!s:>3} | {net4:>7.0f} {N4:>5} {wf4:>3}/6 {p4:>6.3f} | "
              f"{net3:>7.0f} {N3:>5} {wf3:>3}/6 {p3:>6.3f}{flag}")
        if net4 > 0 and p4 < 0.05:
            best.append((k, e, net4, p4, net3, p3))
print("\n<< = positive+MCp<0.05 on 4 pairs AND still positive+p<0.10 without EUR_USD (survives concentration)")
if best:
    print("Configs passing 4-pair MC gate:")
    for k, e, n4, p4, n3, p3 in best:
        print(f"  K={k} TP{int(e[0])}/SL{int(e[1])}: 4p {n4:.0f}p (p={p4:.3f}), ex-EU {n3:.0f}p (p={p3:.3f})")
else:
    print("No config passed the 4-pair MC gate.")
print("Note: OOS is spent (R8); WF+MC are the gates here. ex-EU test = does edge survive w/o the 69% pair.")
