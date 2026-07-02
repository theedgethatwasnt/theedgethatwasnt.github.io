#!/usr/bin/env python3
"""User idea (IMG_3497): S5 + SMA200. Price makes a SHARP move far below the lower band, then
REVERTS to the SMA200. Fade it: fresh penetration >= K2*sigma below SMA200 -> LONG, target =
return to SMA200 (dynamic mean), bounded SL, time cap. Mirror for spikes above -> SHORT.
Contrarian (the surviving category). Tested on ALL 12 pairs. Sweeps K2 x SL. Records sharpness
(drop velocity into entry) + band-width, bucketed. Prior caveat: spike direction is only faintly
reversive net of spread (spread_sigma / zr_bigbar) — this exact framing is new.

Exit priority: SL (bounded, fill level+slip) -> TP=SMA200 (fill at mean level) -> time cap (bid/ask).
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo, equity_drawdown

S5_DIR = "/path/to/projects/fx-core/data/s5_ohlc"
MAX_ROWS = 5_000_000; SLIP = 2.0; MA = 200
MAXHOLD = 4320   # S5 bars = 6h cap
PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_JPY", "AUD_USD",
         "CAD_JPY", "CHF_JPY", "EUR_GBP", "GBP_JPY", "NZD_JPY", "NZD_USD"]
K2S = [2.0, 2.5, 3.0]      # overshoot depth (std devs beyond SMA200)
SLS = [30.0, 60.0, 100.0]  # bounded stop (pips) beyond entry


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, sma, sd, pip, k2, sl, slip, maxhold):
    n = len(h); pos = 0; entry = 0.0; ebar = -1
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if sma[i] != sma[i]:   # nan guard
                continue
            thr_lo = sma[i] - k2 * sd[i]; thr_hi = sma[i] + k2 * sd[i]
            # fresh penetration: was inside last bar, breaches this bar
            if l[i] <= thr_lo and c[i-1] > (sma[i-1] - k2 * sd[i-1]):
                pos = 1; entry = ask[i]; ebar = i; continue
            if h[i] >= thr_hi and c[i-1] < (sma[i-1] + k2 * sd[i-1]):
                pos = -1; entry = bid[i]; ebar = i; continue
        if pos != 0:
            rsn = -1; exf = 0.0
            fc = entry - pos * sl * pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0 and sma[i] == sma[i]:          # TP = revert to SMA200 (mean)
                if pos == 1 and h[i] >= sma[i]: exf = sma[i]; rsn = 2
                elif pos == -1 and l[i] <= sma[i]: exf = sma[i]; rsn = 2
            if rsn < 0 and (i - ebar) >= maxhold:      # time cap
                exf = (bid[i] if pos == 1 else ask[i]); rsn = 3
            if rsn >= 0:
                _pnl[nt] = (exf-entry)/pip if pos == 1 else (entry-exf)/pip
                _eb[nt] = ebar; nt += 1; pos = 0
    return _eb[:nt], _pnl[:nt]


def load(pair):
    df = pq.read_table(f"{S5_DIR}/{pair}_S5_BA.parquet",
                       columns=['timestamp', 'high', 'low', 'close', 'bid_c', 'ask_c']).to_pandas()
    df = df.sort_values('timestamp').reset_index(drop=True)
    if len(df) > MAX_ROWS: df = df.iloc[-MAX_ROWS:].reset_index(drop=True)
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    ts = df['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    return df, ts, pip


res = {(k2, sl): {} for k2 in K2S for sl in SLS}
sharp = {(k2, sl): {} for k2 in K2S for sl in SLS}   # per-trade sharpness (drop vel)
for p in PAIRS:
    df, ts, pip = load(p)
    h = df['high'].to_numpy(); l = df['low'].to_numpy(); c = df['close'].to_numpy()
    bid = df['bid_c'].to_numpy(); ask = df['ask_c'].to_numpy(); del df; gc.collect()
    n = len(c); s = pd.Series(c)
    sma = s.rolling(MA).mean().to_numpy(); sd = s.rolling(MA).std(ddof=0).to_numpy()
    # sharpness proxy: abs move over last 60 S5 bars (5 min) in pips
    vel60 = np.full(n, np.nan); vel60[60:] = np.abs(c[60:] - c[:-60]) / pip
    for k2 in K2S:
        for sl in SLS:
            eb, pnl = _kern(h, l, c, bid, ask, sma, sd, pip, k2, sl, SLIP, MAXHOLD)
            et = ts[eb]
            res[(k2, sl)][p] = [{'exit_time': int(et[i]), 'pnl_pips': float(pnl[i])} for i in range(len(pnl))]
            sharp[(k2, sl)][p] = [(float(vel60[eb[i]]), float(pnl[i])) for i in range(len(pnl))]
    print(f"  {p} done", flush=True)
    del h, l, c, bid, ask, sma, sd, vel60; gc.collect()


def report(dct):
    tl = []
    for p in PAIRS: tl += dct[p]
    tl.sort(key=lambda t: t['exit_time']); net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    if N < 30: return 0, N, 0, 0, 1.0
    eb = np.arange(N)
    npos = sum(1 for p in PAIRS if sum(t['pnl_pips'] for t in dct[p]) > 0)
    wf = sum(1 for f in walk_forward(eb, net, N, 6) if f['net']>0)
    return net.sum(), N, npos, wf, monte_carlo(net, 300)['p_net']


print("\n===== S5 SPIKE-FADE (fresh >=K2σ overshoot -> revert to SMA200, bounded SL, 12 pairs) =====")
print(f"{'K2':>4} {'SL':>5} | {'12p net':>8} {'n':>6} {'pos':>5} {'WF':>4} {'MCp':>6}")
best = None
for k2 in K2S:
    for sl in SLS:
        net, N, npos, wf, mc = report(res[(k2, sl)])
        star = "  <<" if (net > 0 and npos >= 7 and mc < 0.05) else ""
        print(f"{k2:>4} {sl:>5.0f} | {net:>8.0f} {N:>6} {npos:>3}/12 {wf:>3}/6 {mc:>6.3f}{star}")
        if best is None or net > best[0]: best = (net, k2, sl)
# sharpness bucket on the best-net config
_, bk2, bsl = best
pairs_sharp = []
for p in PAIRS: pairs_sharp += sharp[(bk2, bsl)][p]
v = np.array([x[0] for x in pairs_sharp]); pn = np.array([x[1] for x in pairs_sharp])
ok = np.isfinite(v); v, pn = v[ok], pn[ok]
if len(v) >= 30:
    q1, q2 = np.quantile(v, [1/3, 2/3])
    print(f"\nSharpness (5-min drop vel into entry) on best K2={bk2}/SL={bsl}  cuts@{q1:.1f}/{q2:.1f}p:")
    for nm, m in (("LOW(gentle)", v <= q1), ("MID", (v > q1) & (v <= q2)), ("HIGH(sharp)", v > q2)):
        print(f"  {nm:>12}: net={pn[m].sum():.0f} n={m.sum()} mean={pn[m].mean():.2f} wr={100*(pn[m]>0).mean():.0f}%")
print("\nReal & broad only if 12p net>0, >=7/12 pairs, MC p<0.05. Sharper reverting better => "
      "the 'large sharp move' half of the idea holds. Then -> OOS gate.")
