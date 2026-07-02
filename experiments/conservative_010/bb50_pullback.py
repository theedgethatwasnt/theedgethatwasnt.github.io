#!/usr/bin/env python3
"""Entry idea (user, IMG_3494): Bollinger SMA50 ±1σ. If SMA50 slope UP and price touches
the LOWER band -> LONG. Mirror (slope DOWN + touch UPPER band) -> SHORT. Trend-filtered
band pullback. Bounded fade exit. Tested on M5 / M30 / H1.

Also reports RARITY: per-TF signal frequency + how often the condition fires SIMULTANEOUSLY
across M5+M30+H1 (causally aligned on the M5 grid: a coarse TF's signal is the last bar that
CLOSED at/before the M5 bar). Answers 'is multi-TF confluence too rare to trade?'.

- SMA50 + σ50 bands per TF close, causal. Slope := SMA50[i]-SMA50[i-SLOPE_LB].
- Entry(flat): slope>0 & low<=lower -> long@ask ; slope<0 & high>=upper -> short@bid.
- Exit: fixed TP + HARD bounded SL (no PSAR/flip), worse-side fills + 2p slip, real spread.
- Multi-pair (4 harness pairs) + IS/OOS + 6-fold WF + MC per TF.
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
YR_NS = 365.25 * 24 * 3600 * 1e9


@nb.njit(cache=True)
def _kern_bb(h, l, bid, ask, lower, upper, slope, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if slope[i] > 0.0 and l[i] <= lower[i]:
                pos = 1; entry = ask[i]; _eb[nt] = i; continue
            if slope[i] < 0.0 and h[i] >= upper[i]:
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


def resample(df, ts_i64, tf_min):
    bucket = ts_i64 // (tf_min * 60_000_000_000)
    g = df.groupby(bucket, sort=True)
    tsh = g['c'].last().index.to_numpy() * (tf_min * 60_000_000_000)
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(),
            g['bid'].last().to_numpy(), g['ask'].last().to_numpy(), tsh)


trades = {(tf, c): [] for tf, _ in TFS for c in GRID}
perpair = {(tf, c): {} for tf, _ in TFS for c in GRID}
maxloss = {(tf, c): 0.0 for tf, _ in TFS for c in GRID}
# rarity accumulators
sig_bars = {tf: 0 for tf, _ in TFS}; tot_bars = {tf: 0 for tf, _ in TFS}
conf_long = 0; conf_short = 0; m5_total = 0; m5_years = 0.0
per_pair_conf = {}

for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts_i64 = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']})
    del d; gc.collect()
    tf_sig = {}   # tfname -> (tsh, state, closed_ns)  state: +1 long-signal, -1 short, 0 none
    for tfname, tfmin in TFS:
        h, l, c, bid, ask, tsh = resample(df, ts_i64, tfmin)
        n = len(c)
        sma = pd.Series(c).rolling(50).mean().to_numpy()
        sd = pd.Series(c).rolling(50).std(ddof=0).to_numpy()
        valid = ~np.isnan(sma)
        lower = np.where(valid, sma - sd, -1e18); upper = np.where(valid, sma + sd, 1e18)
        slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
        slope = np.where(valid, slope, 0.0)
        # backtest
        for cfg in GRID:
            tp, sl = cfg
            eb, pnl = _kern_bb(h, l, bid, ask, lower, upper, slope, pip, tp, sl, SLIP)
            et = tsh[eb]
            for i in range(len(pnl)):
                trades[(tfname, cfg)].append({'exit_time': int(et[i]),
                                              'pnl_pips': float(pnl[i]), 'pair': p})
            perpair[(tfname, cfg)][p] = float(pnl.sum())
            if len(pnl): maxloss[(tfname, cfg)] = min(maxloss[(tfname, cfg)], float(pnl.min()))
        # rarity: touch-based entry signal per bar
        long_sig = (slope > 0) & (l <= lower)
        short_sig = (slope < 0) & (h >= upper)
        sig_bars[tfname] += int((long_sig | short_sig).sum()); tot_bars[tfname] += int(valid.sum())
        state = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
        tf_sig[tfname] = (tsh, state, tsh + tfmin * 60_000_000_000)
        del h, l, c, bid, ask, sma, sd, lower, upper, slope, long_sig, short_sig
        gc.collect()
    # confluence on the M5 grid (causal: coarse TF = last bar CLOSED at/before m5 bar time)
    m5_tsh, m5_state, _ = tf_sig["M5"]
    cl = (m5_state == 1); cs = (m5_state == -1)
    for tfname in ("M30", "H1"):
        ctsh, cstate, cclosed = tf_sig[tfname]
        idx = np.searchsorted(cclosed, m5_tsh, side="right") - 1
        ok = idx >= 0; idx_c = np.clip(idx, 0, len(cstate) - 1)
        st = np.where(ok, cstate[idx_c], 0)
        cl &= (st == 1); cs &= (st == -1)
    yrs = (m5_tsh[-1] - m5_tsh[0]) / YR_NS
    per_pair_conf[p] = (int(cl.sum()), int(cs.sum()), len(m5_tsh), yrs,
                        int((m5_state != 0).sum()))
    conf_long += int(cl.sum()); conf_short += int(cs.sum()); m5_total += len(m5_tsh); m5_years += yrs
    print(f"  {p} done", flush=True)
    del df, ts_i64, tf_sig, m5_tsh, m5_state, cl, cs; gc.collect()

HARNESS = (50.0, 150.0)   # the fixed exit for the entry comparison (same as SMA-stack baseline)


def validate(tl):
    net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    if N < 30:
        return N, net.sum(), None
    eb = np.arange(N); io = split_is_oos(eb, net, int(N*4/6))
    wf = walk_forward(eb, net, N, n_folds=6); mc = monte_carlo(net, n=300)
    dd = equity_drawdown(np.cumsum(net))['max_dd']
    return N, net.sum(), (io, sum(1 for f in wf if f['net']>0), mc, dd)


# ---- backtest results per TF (grid + validate the TP50/SL150 harness) ----
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
        print(f"{int(cfg[0])}/{int(cfg[1])!s:>3} {net.sum():>8.0f} {N:>6d} {maxloss[(tfname,cfg)]:>6.0f}p "
              f"{dd:>7.0f}p  {pp['EUR_JPY']:.0f}/{pp['EUR_USD']:.0f}/{pp['GBP_USD']:.0f}/{pp['USD_JPY']:.0f}{tag}")
    rank.sort(key=lambda r: -r[1]); bcfg, bns, bN = rank[0]
    print(f"  best net: TP{int(bcfg[0])}/SL{int(bcfg[1])} = {bns:.0f}p (n={bN})")
    # full validation on the TP50/SL150 harness (the entry-comparison config)
    tlh = sorted(trades[(tfname, HARNESS)], key=lambda t: t['exit_time'])
    N, ns, v = validate(tlh)
    harness_by_tf[tfname] = (N, ns, v)
    if v:
        io, wfp, mc, dd = v
        print(f"  >> TP50/SL150: net={ns:.0f}p n={N} IS {io['is_net']:.0f} OOS {io['oos_net']:.0f} "
              f"(WR {io['oos_wr']:.0f}%) WF {wfp}/6 MC p_net={mc['p_net']:.3f} eqMaxDD {dd:.0f}p")
    else:
        print(f"  >> TP50/SL150: net={ns:.0f}p n={N} (too few to validate)")

# ---- RARITY + CONFLUENCE ----
print("\n========  RARITY OF THE ENTRY CONDITION  ========")
print(f"{'TF':>5} {'signal bars':>12} {'of valid bars':>14} {'rate':>16}")
for tfname, tfmin in TFS:
    bars_per_yr = tot_bars[tfname] / max(sum(v[3] for v in per_pair_conf.values()), 1e-9)
    pct = 100.0 * sig_bars[tfname] / max(tot_bars[tfname], 1)
    per_yr = sig_bars[tfname] / max(sum(v[3] for v in per_pair_conf.values()), 1e-9)
    print(f"{tfname:>5} {sig_bars[tfname]:>12d} {pct:>13.2f}% {per_yr:>10.0f}/yr(4pairs)")
print("\n========  SIMULTANEOUS across M5+M30+H1 (causal, on M5 grid)  ========")
tot_yrs = sum(v[3] for v in per_pair_conf.values())
print(f"{'pair':>8} {'M5 sig':>8} {'ALL-3 long':>11} {'ALL-3 short':>12} {'confluence/yr':>14}")
for p in PAIRS:
    cl, csh, nm5, yrs, m5sig = per_pair_conf[p]
    print(f"{p:>8} {m5sig:>8d} {cl:>11d} {csh:>12d} {(cl+csh)/max(yrs,1e-9):>12.1f}/yr")
print(f"\nTOTAL: all-3-aligned long={conf_long} short={conf_short} over ~{tot_yrs:.1f} pair-years "
      f"=> {(conf_long+conf_short)/max(tot_yrs,1e-9):.1f} confluence entries/yr/pair")
print(f"Single-TF M5 entries ~{sig_bars['M5']/max(tot_yrs,1e-9):.0f}/yr/pair -> confluence is "
      f"{(conf_long+conf_short)/max(sig_bars['M5'],1)*100:.2f}% as frequent.")

print("\n========  ENTRY COMPARISON @ SAME EXIT (TP50/SL150)  ========")
print("Isolates the ENTRY: identical bounded fade exit, different entry rule.")
print(f"  SMA-stack entry (baseline):  -344 pips")
for tfname, _ in TFS:
    N, ns, v = harness_by_tf[tfname]
    extra = f"WF {v[1]}/6 MC p_net={v[2]['p_net']:.3f}" if v else "few trades"
    print(f"  BB50-pullback {tfname:>3} entry:    {ns:>6.0f} pips  (n={N}, {extra})")
print("Positive + WF>=4/6 + MC p<0.05 => the BB50 pullback is a REAL entry edge (first on this book).")
