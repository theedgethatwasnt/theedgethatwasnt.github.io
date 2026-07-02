#!/usr/bin/env python3
"""User idea #2: run the pullback + breakout entries on the S5 timeframe with BB SMA200 (±1σ).
Tested across ALL 12 pairs at once (generalization visible immediately — no tune-then-OOS trap).
Band-width / squeeze metrics recorded per trade (band's min-width + expanding vs contracting).

- S5 raw bars (base TF, no resample). Center = SMA200, band = SMA200 ± 1σ (rolling std200).
- PULLBACK: slope>0 & low<=lower -> long ; slope<0 & high>=upper -> short.
- BREAKOUT: slope just turned + & close>upper -> long ; just turned - & close<lower -> short.
- Exit: TP50/SL100 bounded fade (no PSAR/flip), worse-side fills + 2p slip, real spread.
- Band-width per entry: bw=(upper-lower)/pip; bw_vs_min = bw / min(bw, last 200) (~1 = squeeze);
  bw_slope = bw - bw[-20] (>0 expanding). Bucket by these to test squeeze/expansion.
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo, equity_drawdown

S5_DIR = "/path/to/projects/fx-core/data/s5_ohlc"
MAX_ROWS = 5_000_000; SLIP = 2.0; SLOPE_LB = 3; MA = 200; K = 1.0; TP = 50.0; SL = 100.0
PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_JPY", "AUD_USD",
         "CAD_JPY", "CHF_JPY", "EUR_GBP", "GBP_JPY", "NZD_JPY", "NZD_USD"]


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, lower, upper, longsig, shortsig, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if shortsig[i]: pos = -1; entry = bid[i]; _eb[nt] = i; continue
            if longsig[i]: pos = 1; entry = ask[i]; _eb[nt] = i; continue
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
    return _eb[:nt], _pnl[:nt]


def load(pair):
    df = pq.read_table(f"{S5_DIR}/{pair}_S5_BA.parquet",
                       columns=['timestamp', 'high', 'low', 'close', 'bid_c', 'ask_c']).to_pandas()
    df = df.sort_values('timestamp').reset_index(drop=True)
    if len(df) > MAX_ROWS: df = df.iloc[-MAX_ROWS:].reset_index(drop=True)
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    ts = df['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    return df, ts, pip


res = {"PULLBACK": {}, "BREAKOUT": {}}   # entry -> {pair: [trades w/ bw]}
for p in PAIRS:
    df, ts, pip = load(p)
    h = df['high'].to_numpy(); l = df['low'].to_numpy(); c = df['close'].to_numpy()
    bid = df['bid_c'].to_numpy(); ask = df['ask_c'].to_numpy(); del df; gc.collect()
    n = len(c); s = pd.Series(c)
    sma = s.rolling(MA).mean().to_numpy(); sd = s.rolling(MA).std(ddof=0).to_numpy()
    v = ~np.isnan(sma) & ~np.isnan(sd)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]; slope = np.where(v, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    lower = np.where(v, sma - K*sd, -1e18); upper = np.where(v, sma + K*sd, 1e18)
    # band-width metrics (vectorized)
    bw = np.where(v, 2*K*sd/pip, np.nan)
    bwser = pd.Series(bw)
    bw_min = bwser.rolling(MA, min_periods=20).min().to_numpy()
    bw_vs_min = bw / bw_min
    bw_slope = np.full(n, np.nan); bw_slope[20:] = bw[20:] - bw[:-20]
    sigs = {"PULLBACK": ((slope > 0) & (l <= lower) & v, (slope < 0) & (h >= upper) & v),
            "BREAKOUT": (((slope > 0) & (sp <= 0) & (c > upper) & v),
                         ((slope < 0) & (sp >= 0) & (c < lower) & v))}
    for ent, (ls, ss) in sigs.items():
        eb, pnl = _kern(h, l, c, bid, ask, lower, upper, ls, ss, pip, TP, SL, SLIP)
        et = ts[eb]
        res[ent][p] = [{'exit_time': int(et[i]), 'pnl_pips': float(pnl[i]),
                        'bw_vs_min': float(bw_vs_min[eb[i]]), 'bw_slope': float(bw_slope[eb[i]])}
                       for i in range(len(pnl))]
    print(f"  {p} done (PB n={len(res['PULLBACK'][p])}, BO n={len(res['BREAKOUT'][p])})", flush=True)
    del h, l, c, bid, ask, sma, sd, slope, sp, lower, upper, bw, bw_min, bw_vs_min, bw_slope; gc.collect()


def agg(dct):
    tl = []
    for p in PAIRS: tl += dct[p]
    return sorted(tl, key=lambda t: t['exit_time'])


def bucket(rws, key):
    vals = np.array([r[key] for r in rws]); pn = np.array([r['pnl_pips'] for r in rws])
    ok = np.isfinite(vals); vals, pn = vals[ok], pn[ok]
    if len(vals) < 30: return "n/a"
    q1, q2 = np.quantile(vals, [1/3, 2/3])
    parts = []
    for nm, m in (("LOW", vals <= q1), ("MID", (vals > q1) & (vals <= q2)), ("HIGH", vals > q2)):
        parts.append(f"{nm} net={pn[m].sum():.0f}(n{m.sum()},wr{100*(pn[m]>0).mean():.0f})")
    return f"cuts@{q1:.2f}/{q2:.2f}: " + "  ".join(parts)


for ent in ("PULLBACK", "BREAKOUT"):
    tl = agg(res[ent]); net = np.array([t['pnl_pips'] for t in tl]); N = len(net); eb = np.arange(N)
    npos = sum(1 for p in PAIRS if sum(t['pnl_pips'] for t in res[ent][p]) > 0)
    wf = sum(1 for f in walk_forward(eb, net, N, 6) if f['net']>0)
    mc = monte_carlo(net, 300)['p_net']; dd = equity_drawdown(np.cumsum(net))['max_dd']
    print(f"\n======== S5 + SMA200 {ent} (TP50/SL100, K=1, 12 pairs) ========")
    print(f"  12-pair: {npos}/12 positive | net={net.sum():.0f}p n={N} WF {wf}/6 MC p_net={mc:.3f} eqMaxDD={dd:.0f}p")
    print("  per-pair net: " + " ".join(f"{p[:6]}={sum(t['pnl_pips'] for t in res[ent][p]):.0f}" for p in PAIRS))
    print(f"  by bw_vs_min (LOW~=squeeze): {bucket(tl, 'bw_vs_min')}")
    print(f"  by bw_slope (HIGH=expanding): {bucket(tl, 'bw_slope')}")
print("\nReal & broad only if 12-pair net>0 with >=7/12 pairs positive AND MC p<0.05. "
      "Squeeze/expansion helps if a bw bucket is clearly better (then it's a filter to OOS-test).")
