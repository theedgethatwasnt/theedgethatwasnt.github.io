#!/usr/bin/env python3
"""Entry idea #2 (user, 2026-07-01): MOMENTUM breakdown (mirror of the pullback fade).
If SMA50 slope JUST turned negative AND price is below the LOWER band -> SHORT.
Mirror: slope JUST turned positive AND price above UPPER band -> LONG.
(Trade the trend as it flips + price confirms beyond the band. Continuation, not fade.)

- 'just turned negative' = slope crosses zero this bar: slope[i]<0 and slope[i-1]>=0.
- 'price below lower band' = close < lower band (state). Enter at that bar close (worse side).
- Same bounded fade EXIT (fixed TP + hard SL, no PSAR/flip), worse-side fills + 2p slip,
  real per-bar spread. Same TFs (M5/M30/H1), same TP/SL grid, TP50/SL150 highlighted.
- Multi-pair + IS/OOS + 6-fold WF + MC. Rarity + M5+M30+H1 confluence report.
MEMORY-SAFE: load each pair's S5 ONCE, one df, del S5, resample to all 3 TFs.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba
from validate import split_is_oos, walk_forward, monte_carlo, equity_drawdown

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
TFS = [("M5", 5), ("M30", 30), ("H1", 60)]
SLIP = 2.0; SLOPE_LB = 3
GRID = [(30.0, 60.0), (50.0, 100.0), (50.0, 150.0), (150.0, 150.0)]
HARNESS = (50.0, 150.0)
YR_NS = 365.25 * 24 * 3600 * 1e9


@nb.njit(cache=True)
def _kern_bo(h, l, c, bid, ask, lower, upper, turned_neg, turned_pos, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if turned_neg[i] and c[i] < lower[i]:            # SHORT breakdown
                pos = -1; entry = bid[i]; _eb[nt] = i; continue
            if turned_pos[i] and c[i] > upper[i]:            # LONG breakout
                pos = 1; entry = ask[i]; _eb[nt] = i; continue
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


def resample(df, ts_i64, tf_min):
    bucket = ts_i64 // (tf_min * 60_000_000_000)
    g = df.groupby(bucket, sort=True)
    tsh = g['c'].last().index.to_numpy() * (tf_min * 60_000_000_000)
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(),
            g['bid'].last().to_numpy(), g['ask'].last().to_numpy(), tsh)


def validate(tl):
    net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    if N < 30:
        return N, float(net.sum()), None
    eb = np.arange(N); io = split_is_oos(eb, net, int(N*4/6))
    wf = walk_forward(eb, net, N, n_folds=6); mc = monte_carlo(net, n=300)
    dd = equity_drawdown(np.cumsum(net))['max_dd']
    return N, float(net.sum()), (io, sum(1 for f in wf if f['net']>0), mc, dd)


trades = {(tf, c): [] for tf, _ in TFS for c in GRID}
perpair = {(tf, c): {} for tf, _ in TFS for c in GRID}
maxloss = {(tf, c): 0.0 for tf, _ in TFS for c in GRID}
sig_bars = {tf: 0 for tf, _ in TFS}; tot_bars = {tf: 0 for tf, _ in TFS}
conf_long = 0; conf_short = 0; per_pair_conf = {}

for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts_i64 = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']})
    del d; gc.collect()
    tf_sig = {}
    for tfname, tfmin in TFS:
        h, l, c, bid, ask, tsh = resample(df, ts_i64, tfmin)
        n = len(c)
        sma = pd.Series(c).rolling(50).mean().to_numpy()
        sd = pd.Series(c).rolling(50).std(ddof=0).to_numpy()
        valid = ~np.isnan(sma)
        lower = np.where(valid, sma - sd, -1e18); upper = np.where(valid, sma + sd, 1e18)
        slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
        slope = np.where(valid, slope, 0.0)
        sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]          # slope[i-1]
        turned_neg = (slope < 0) & (sp >= 0) & valid
        turned_pos = (slope > 0) & (sp <= 0) & valid
        for cfg in GRID:
            tp, sl = cfg
            eb, pnl = _kern_bo(h, l, c, bid, ask, lower, upper, turned_neg, turned_pos,
                               pip, tp, sl, SLIP)
            et = tsh[eb]
            for i in range(len(pnl)):
                trades[(tfname, cfg)].append({'exit_time': int(et[i]),
                                              'pnl_pips': float(pnl[i]), 'pair': p})
            perpair[(tfname, cfg)][p] = float(pnl.sum())
            if len(pnl): maxloss[(tfname, cfg)] = min(maxloss[(tfname, cfg)], float(pnl.min()))
        short_sig = turned_neg & (c < lower); long_sig = turned_pos & (c > upper)
        sig_bars[tfname] += int((long_sig | short_sig).sum()); tot_bars[tfname] += int(valid.sum())
        state = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
        tf_sig[tfname] = (tsh, state, tsh + tfmin * 60_000_000_000)
        del h, l, c, bid, ask, sma, sd, lower, upper, slope, sp, turned_neg, turned_pos
        gc.collect()
    m5_tsh, m5_state, _ = tf_sig["M5"]
    cl = (m5_state == 1); cs = (m5_state == -1)
    for tfname in ("M30", "H1"):
        ctsh, cstate, cclosed = tf_sig[tfname]
        idx = np.searchsorted(cclosed, m5_tsh, side="right") - 1
        ok = idx >= 0; st = np.where(ok, cstate[np.clip(idx, 0, len(cstate)-1)], 0)
        cl &= (st == 1); cs &= (st == -1)
    yrs = (m5_tsh[-1] - m5_tsh[0]) / YR_NS
    per_pair_conf[p] = (int(cl.sum()), int(cs.sum()), yrs, int((m5_state != 0).sum()))
    conf_long += int(cl.sum()); conf_short += int(cs.sum())
    print(f"  {p} done", flush=True)
    del df, ts_i64, tf_sig, m5_tsh, m5_state, cl, cs; gc.collect()

harness_by_tf = {}
for tfname, _ in TFS:
    print(f"\n========  {tfname}  ========")
    print(f"{'TP/SL':>9} {'net':>8} {'ntr':>6} {'worst':>7} {'eqMaxDD':>8}  EJ/EU/GU/UJ")
    rank = []
    for cfg in GRID:
        tl = sorted(trades[(tfname, cfg)], key=lambda t: t['exit_time'])
        net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
        dd = equity_drawdown(np.cumsum(net))['max_dd'] if N else 0.0
        pp = perpair[(tfname, cfg)]
        rank.append((cfg, net.sum(), N))
        tag = "  <== TP50/SL150" if cfg == HARNESS else ""
        pv = lambda k: pp.get(k, 0.0)
        print(f"{int(cfg[0])}/{int(cfg[1])!s:>3} {net.sum():>8.0f} {N:>6d} {maxloss[(tfname,cfg)]:>6.0f}p "
              f"{dd:>7.0f}p  {pv('EUR_JPY'):.0f}/{pv('EUR_USD'):.0f}/{pv('GBP_USD'):.0f}/{pv('USD_JPY'):.0f}{tag}")
    rank.sort(key=lambda r: -r[1]); bcfg, bns, bN = rank[0]
    print(f"  best net: TP{int(bcfg[0])}/SL{int(bcfg[1])} = {bns:.0f}p (n={bN})")
    N, ns, v = validate(sorted(trades[(tfname, HARNESS)], key=lambda t: t['exit_time']))
    harness_by_tf[tfname] = (N, ns, v)
    if v:
        io, wfp, mc, dd = v
        print(f"  >> TP50/SL150: net={ns:.0f}p n={N} IS {io['is_net']:.0f} OOS {io['oos_net']:.0f} "
              f"(WR {io['oos_wr']:.0f}%) WF {wfp}/6 MC p_net={mc['p_net']:.3f} eqMaxDD {dd:.0f}p")
    else:
        print(f"  >> TP50/SL150: net={ns:.0f}p n={N} (too few to validate)")

print("\n========  RARITY  ========")
tot_yrs = sum(v[2] for v in per_pair_conf.values())
for tfname, _ in TFS:
    print(f"  {tfname}: {sig_bars[tfname]} signal bars = {100.0*sig_bars[tfname]/max(tot_bars[tfname],1):.2f}% "
          f"of valid bars, ~{sig_bars[tfname]/max(tot_yrs,1e-9):.0f}/yr(4pairs)")
print(f"\n  ALL-3-TF simultaneous: long={conf_long} short={conf_short} over ~{tot_yrs:.1f} pair-yrs "
      f"=> {(conf_long+conf_short)/max(tot_yrs,1e-9):.1f}/yr/pair")

print("\n========  ENTRY COMPARISON @ TP50/SL150  ========")
print(f"  SMA-stack baseline:         -344 pips")
print(f"  BB50-PULLBACK M5 (idea #1): +1792 pips (unconfirmed, MC p=0.20)")
for tfname, _ in TFS:
    N, ns, v = harness_by_tf[tfname]
    extra = f"WF {v[1]}/6 MC p_net={v[2]['p_net']:.3f}" if v else "few trades"
    print(f"  BB50-BREAKOUT {tfname:>3} (idea #2): {ns:>6.0f} pips (n={N}, {extra})")
print("Momentum/continuation has failed net-of-spread everywhere in this book — testing if here too.")
