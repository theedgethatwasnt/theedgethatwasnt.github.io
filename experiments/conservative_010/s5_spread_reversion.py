#!/usr/bin/env python3
"""User rule (IMG_3497 + refinements): if price moved >= N spreads within the last 10s (2 S5 bars),
fade the REVERSION with a NIMBLE exit. PLUS condition on WHERE the move started vs SMA200:
did it start AT the mean (overshoot -> should revert) or ALREADY FAR (trend leg -> won't)? and
on WHICH SIDE, and whether it EXTENDS away from the mean or moves toward it.

Move in spread-units (auto-adapts to cost). Nimble exit: TP=F*move, tight time cap, bounded SL.
Worse-side fills + 2p slip, real spread. 12 pairs. Records per trade: dist_start (σ from SMA200
at the move's origin), side (above/below), extending (move away from mean). Buckets the primary
config by these. Contrarian; prior headwind = hft_micro/oracle (fast reversion spread-eaten).
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo, equity_drawdown

S5_DIR = "/path/to/projects/fx-core/data/s5_ohlc"
MAX_ROWS = 5_000_000; SLIP = 2.0; LB = 2; MA = 200
PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_JPY", "AUD_USD",
         "CAD_JPY", "CHF_JPY", "EUR_GBP", "GBP_JPY", "NZD_JPY", "NZD_USD"]
CONFIGS = [(5, 1.0, 20, 12), (5, 0.5, 20, 12), (8, 1.0, 25, 24), (10, 1.0, 30, 24)]
PRIMARY = (5, 1.0, 20, 12)


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, dirn, movep, pip, F, slpips, tcap, slip):
    n = len(h); pos = 0; entry = 0.0; ebar = -1; tgt = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if dirn[i] == 1:
                pos = 1; entry = ask[i]; ebar = i; tgt = entry + F*movep[i]*pip; continue
            if dirn[i] == -1:
                pos = -1; entry = bid[i]; ebar = i; tgt = entry - F*movep[i]*pip; continue
        if pos != 0:
            rsn = -1; exf = 0.0; fc = entry - pos*slpips*pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0:
                if pos == 1 and h[i] >= tgt: exf = tgt; rsn = 2
                elif pos == -1 and l[i] <= tgt: exf = tgt; rsn = 2
            if rsn < 0 and (i - ebar) >= tcap:
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
    return df, (0.01 if pair.endswith("JPY") else 0.0001)


res = {cf: {} for cf in CONFIGS}
cond = {p: [] for p in PAIRS}   # primary-config trades w/ conditioning
for p in PAIRS:
    df, pip = load(p)
    h = df['high'].to_numpy(); l = df['low'].to_numpy(); c = df['close'].to_numpy()
    bid = df['bid_c'].to_numpy(); ask = df['ask_c'].to_numpy(); del df; gc.collect()
    n = len(c); spread = (ask - bid) / pip
    move = np.zeros(n); move[LB:] = (c[LB:] - c[:-LB]) / pip; movep = np.abs(move)
    s = pd.Series(c); sma = s.rolling(MA).mean().to_numpy(); sd = s.rolling(MA).std(ddof=0).to_numpy()
    dist = (c - sma) / sd            # σ from SMA200 at each bar (signed)
    for (N, F, slp, tcap) in CONFIGS:
        trig = movep >= N * spread
        fresh = trig.copy(); fresh[1:] = trig[1:] & ~trig[:-1]
        dirn = np.where(fresh & (move > 0), -1, np.where(fresh & (move < 0), 1, 0)).astype(np.int64)
        eb, pnl = _kern(h, l, c, bid, ask, dirn, movep, pip, F, slp, tcap, SLIP)
        res[(N, F, slp, tcap)][p] = pnl
        if (N, F, slp, tcap) == PRIMARY:
            st = eb - LB                            # move origin bar
            for k in range(len(eb)):
                ds = dist[st[k]]                    # σ from SMA200 at move start
                mv = move[eb[k]]                    # signed move pips
                side = 1.0 if ds > 0 else -1.0
                extending = 1 if (np.sign(mv) == np.sign(ds) and ds == ds) else 0
                cond[p].append((float(ds) if ds == ds else np.nan, int(extending), float(pnl[k])))
    print(f"  {p} done", flush=True)
    del h, l, c, bid, ask, spread, move, movep, sma, sd, dist; gc.collect()


def stats(dct):
    allp = np.concatenate([dct[p] for p in PAIRS]); N = len(allp)
    if N < 30: return 0, N, 0, 0, 1.0, 0
    npos = sum(1 for p in PAIRS if dct[p].sum() > 0)
    wf = sum(1 for f in walk_forward(np.arange(N), allp, N, 6) if f['net']>0)
    return allp.sum(), N, npos, wf, monte_carlo(allp, 300)['p_net'], 100*np.mean(allp > 0)


print("\n===== S5 SPREAD-REVERSION (fade >=N-spread move/10s, nimble exit, 12 pairs) =====")
print(f"{'N':>3} {'F':>4} {'SL':>4} {'cap':>4} | {'net':>9} {'n':>7} {'pos':>5} {'WF':>4} {'MCp':>6} {'WR':>4}")
for cf in CONFIGS:
    net, n, npos, wf, mc, wr = stats(res[cf])
    star = "  <<" if (net > 0 and npos >= 7 and mc < 0.05) else ""
    print(f"{cf[0]:>3} {cf[1]:>4.1f} {cf[2]:>4.0f} {cf[3]:>4} | {net:>9.0f} {n:>7} {npos:>3}/12 {wf:>3}/6 {mc:>6.3f} {wr:>3.0f}%{star}")

# conditioning on PRIMARY: where did the move start + side + extending
allc = [x for p in PAIRS for x in cond[p]]
ds = np.array([x[0] for x in allc]); ext = np.array([x[1] for x in allc]); pn = np.array([x[2] for x in allc])
ok = np.isfinite(ds); ds, ext, pn = ds[ok], ext[ok], pn[ok]
print(f"\n--- PRIMARY {PRIMARY}: conditioning on the move's ORIGIN vs SMA200 (n={len(pn)}) ---")
ads = np.abs(ds); q1, q2 = np.quantile(ads, [1/3, 2/3])
print(f"by |dist_start| (σ from SMA200 at origin) cuts@{q1:.2f}/{q2:.2f}:")
for nm, m in (("NEAR-mean", ads <= q1), ("MID", (ads > q1)&(ads <= q2)), ("FAR", ads > q2)):
    print(f"    {nm:>10}: net={pn[m].sum():>7.0f} n={m.sum():>5} mean={pn[m].mean():>5.2f} wr={100*(pn[m]>0).mean():.0f}%")
print("by move relative to mean:")
for nm, m in (("EXTENDING(away)", ext == 1), ("TOWARD(reverting)", ext == 0)):
    print(f"    {nm:>18}: net={pn[m].sum():>7.0f} n={m.sum():>5} mean={pn[m].mean():>5.2f} wr={100*(pn[m]>0).mean():.0f}%")
print("by side of SMA200 at origin:")
for nm, m in (("ABOVE (short fade)", ds > 0), ("BELOW (long fade)", ds < 0)):
    print(f"    {nm:>18}: net={pn[m].sum():>7.0f} n={m.sum():>5} mean={pn[m].mean():>5.2f} wr={100*(pn[m]>0).mean():.0f}%")
print("\nHypothesis: NEAR-mean origin + EXTENDING-away = clean overshoot -> reverts (net>0). "
      "FAR origin = trend leg -> won't. If a subset is clearly +, that's the tradeable filter -> OOS gate.")
