#!/usr/bin/env python3
"""Tradeable multi-TF form: M5 BB50 PULLBACK gated by the H1 trend.
Take the M5 pullback long ONLY when the H1 SMA50 slope is up; short ONLY when H1 slope is
down. (H1 projected causally onto the M5 grid = last H1 bar CLOSED at/before the M5 bar.)
Does the H1 filter sharpen the ungated M5 pullback (+1792p, MC p=0.20) into significance?

Two gate variants:
  TREND  : H1 slope sign aligns with entry direction (frequent).
  STRICT : H1 is ALSO in a full pullback signal, same direction (the ~64/yr confluence).
Same bounded fade exit (TP/SL grid, no PSAR/flip), worse-side fills + 2p slip, real spread.
Multi-pair + IS/OOS + WF + MC. MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba
from validate import split_is_oos, walk_forward, monte_carlo, equity_drawdown

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SLIP = 2.0; SLOPE_LB = 3
GRID = [(50.0, 150.0), (50.0, 100.0), (150.0, 150.0)]
HARNESS = (50.0, 150.0)
M5_NS = 300_000_000_000; H1_NS = 3_600_000_000_000


@nb.njit(cache=True)
def _kern(h, l, bid, ask, lower, upper, slope, gate_long, gate_short, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if slope[i] > 0.0 and l[i] <= lower[i] and gate_long[i]:
                pos = 1; entry = ask[i]; _eb[nt] = i; continue
            if slope[i] < 0.0 and h[i] >= upper[i] and gate_short[i]:
                pos = -1; entry = bid[i]; _eb[nt] = i; continue
        if pos != 0:
            rsn = -1; exf = 0.0
            fc = entry - pos * sl * pip
            if pos == 1 and l[i] <= fc:
                exf = fc - slip * pip; rsn = 0
            elif pos == -1 and h[i] >= fc:
                exf = fc + slip * pip; rsn = 0
            if rsn < 0:
                tpl = entry + pos * tp * pip
                if pos == 1 and h[i] >= tpl:
                    exf = tpl; rsn = 2
                elif pos == -1 and l[i] <= tpl:
                    exf = tpl; rsn = 2
            if rsn >= 0:
                _pnl[nt] = (exf - entry) / pip if pos == 1 else (entry - exf) / pip
                nt += 1; pos = 0
    return _eb[:nt], _pnl[:nt]


def resample(df, ts_i64, tf_ns):
    b = ts_i64 // tf_ns; g = df.groupby(b, sort=True)
    tsh = g['c'].last().index.to_numpy() * tf_ns
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(),
            g['bid'].last().to_numpy(), g['ask'].last().to_numpy(), tsh)


def feats(h, l, c):
    n = len(c)
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    sd = pd.Series(c).rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(sma)
    lower = np.where(v, sma - sd, -1e18); upper = np.where(v, sma + sd, 1e18)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
    slope = np.where(v, slope, 0.0)
    return lower, upper, slope, v


def validate(net):
    N = len(net)
    if N < 30: return N, float(net.sum()), None
    eb = np.arange(N); io = split_is_oos(eb, net, int(N*4/6))
    wf = walk_forward(eb, net, N, n_folds=6); mc = monte_carlo(net, n=300)
    dd = equity_drawdown(np.cumsum(net))['max_dd']
    return N, float(net.sum()), (io, sum(1 for f in wf if f['net']>0), mc, dd)


VARIANTS = ["ungated", "trend", "strict"]
res = {(vr, cfg): [] for vr in VARIANTS for cfg in GRID}
perpair = {(vr, cfg): {} for vr in VARIANTS for cfg in GRID}
for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts_i64 = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']}); del d; gc.collect()
    h5, l5, c5, bid5, ask5, tsh5 = resample(df, ts_i64, M5_NS)
    lo5, up5, sl5, v5 = feats(h5, l5, c5)
    h1h, h1l, h1c, _, _, tsh1 = resample(df, ts_i64, H1_NS)
    lo1, up1, sl1, v1 = feats(h1h, h1l, h1c)
    # project H1 slope-sign + H1 pullback-signal onto M5 grid (causal: last closed H1 bar)
    closed1 = tsh1 + H1_NS
    idx = np.searchsorted(closed1, tsh5, side="right") - 1
    ok = idx >= 0; ci = np.clip(idx, 0, len(sl1) - 1)
    h1_slope = np.where(ok, sl1[ci], 0.0)
    h1_pull_long = np.where(ok, ((sl1 > 0) & (h1l <= lo1))[ci], False)
    h1_pull_short = np.where(ok, ((sl1 < 0) & (h1h >= up1))[ci], False)
    gates = {
        "ungated": (np.ones(len(c5), bool), np.ones(len(c5), bool)),
        "trend":   (h1_slope > 0, h1_slope < 0),
        "strict":  (h1_pull_long, h1_pull_short),
    }
    for vr in VARIANTS:
        gl, gs = gates[vr]
        for cfg in GRID:
            tp, sl = cfg
            eb, pnl = _kern(h5, l5, bid5, ask5, lo5, up5, sl5, gl, gs, pip, tp, sl, SLIP)
            et = tsh5[eb]
            for i in range(len(pnl)):
                res[(vr, cfg)].append({'exit_time': int(et[i]), 'pnl_pips': float(pnl[i]), 'pair': p})
            perpair[(vr, cfg)][p] = float(pnl.sum())
    print(f"  {p} done", flush=True)
    del df, ts_i64, h5, l5, c5, bid5, ask5, lo5, up5, sl5, h1h, h1l, h1c, lo1, up1, sl1
    gc.collect()

print("\n===== M5 BB50 pullback: ungated vs H1-trend-filter vs H1-strict-confluence (TP50/SL150) =====")
print(f"{'variant':>9} {'net':>8} {'ntr':>6} {'IS':>7} {'OOS':>7} {'WR':>5} {'WF':>4} {'MC p':>7}  EJ/EU/GU/UJ")
for vr in VARIANTS:
    tl = sorted(res[(vr, HARNESS)], key=lambda t: t['exit_time'])
    net = np.array([t['pnl_pips'] for t in tl])
    N, ns, v = validate(net); pp = perpair[(vr, HARNESS)]
    if v:
        io, wfp, mc, dd = v
        print(f"{vr:>9} {ns:>8.0f} {N:>6d} {io['is_net']:>7.0f} {io['oos_net']:>7.0f} "
              f"{io['oos_wr']:>4.0f}% {wfp:>3}/6 {mc['p_net']:>7.3f}  "
              f"{pp['EUR_JPY']:.0f}/{pp['EUR_USD']:.0f}/{pp['GBP_USD']:.0f}/{pp['USD_JPY']:.0f}")
    else:
        print(f"{vr:>9} {ns:>8.0f} {N:>6d}  (too few)")
print("\nGoal: does the H1 filter push MC p_net below 0.05 AND broaden past EUR_USD? "
      "trend keeps trades; strict is the ~64/yr confluence (fewer trades).")
