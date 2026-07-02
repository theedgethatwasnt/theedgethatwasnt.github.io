#!/usr/bin/env python3
"""NEW dimension (user): relative VOLUME + VWAP + prior-run smoothness. Fade dislocations from
rolling VWAP on M5; condition on relative volume (5min vs hour), VWAP-distance, and prior-run
smoothness. VWAP = volume-weighted fair value (better anchor than SMA?). Hypothesis (ties to the
one significant prior edge, low-vol first-touch H4): dislocations on LOW relative volume = no
conviction -> revert; HIGH rvol = real move -> continue. Volume is OANDA tick-volume.

- M5 bars (S5->M5). VWAP over last hour (12 M5 bars) of hlc3. rvol = vol[i]/mean(vol,12).
- Entry: |close-VWAP| >= K*std20 -> fade toward VWAP. Exit: TP=VWAP (dynamic), bounded SL, time cap.
- Record per trade: rvol, prior smoothness (std of M5 returns/20), VWAP-dist (sd), side. Bucket.
- 12 pairs, aggregate + MC. Prior caveat: tick-volume 'no lift' in past studies.
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo

S5_DIR = "/path/to/projects/fx-core/data/s5_ohlc"
MAX_ROWS = 5_000_000; SLIP = 2.0; M5_NS = 300_000_000_000
HOURW = 12; SMOW = 20; K = 1.5; SL = 40.0; TCAP = 24   # 2h cap on M5
PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_JPY", "AUD_USD",
         "CAD_JPY", "CHF_JPY", "EUR_GBP", "GBP_JPY", "NZD_JPY", "NZD_USD"]


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, vwap, thr_sd, pip, K, slp, tcap, slip):
    n = len(h); pos = 0; entry = 0.0; ebar = -1
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if vwap[i] != vwap[i] or thr_sd[i] != thr_sd[i]:
                continue
            up = vwap[i] + K*thr_sd[i]; dn = vwap[i] - K*thr_sd[i]
            if c[i] > up:   pos = -1; entry = bid[i]; ebar = i; continue   # far above VWAP -> short
            if c[i] < dn:   pos = 1;  entry = ask[i]; ebar = i; continue   # far below VWAP -> long
        if pos != 0:
            rsn = -1; exf = 0.0; fc = entry - pos*slp*pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0 and vwap[i] == vwap[i]:            # TP = revert to VWAP
                if pos == 1 and h[i] >= vwap[i]: exf = vwap[i]; rsn = 2
                elif pos == -1 and l[i] <= vwap[i]: exf = vwap[i]; rsn = 2
            if rsn < 0 and (i - ebar) >= tcap:
                exf = (bid[i] if pos == 1 else ask[i]); rsn = 3
            if rsn >= 0:
                _pnl[nt] = (exf-entry)/pip if pos == 1 else (entry-exf)/pip
                _eb[nt] = ebar; nt += 1; pos = 0
    return _eb[:nt], _pnl[:nt]


def load_m5(pair):
    df = pq.read_table(f"{S5_DIR}/{pair}_S5_BA.parquet",
                       columns=['timestamp', 'high', 'low', 'close', 'bid_c', 'ask_c', 'volume']).to_pandas()
    df = df.sort_values('timestamp').reset_index(drop=True)
    if len(df) > MAX_ROWS: df = df.iloc[-MAX_ROWS:].reset_index(drop=True)
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    ts = df['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    b = ts // M5_NS
    g = df.groupby(b, sort=True)
    m = pd.DataFrame({'h': g['high'].max(), 'l': g['low'].min(), 'c': g['close'].last(),
                      'bid': g['bid_c'].last(), 'ask': g['ask_c'].last(), 'vol': g['volume'].sum()})
    tsh = g['close'].last().index.to_numpy() * M5_NS
    return m, tsh, pip


rows = {p: [] for p in PAIRS}
for p in PAIRS:
    m, tsh, pip = load_m5(p)
    h = m['h'].to_numpy(); l = m['l'].to_numpy(); c = m['c'].to_numpy()
    bid = m['bid'].to_numpy(); ask = m['ask'].to_numpy(); vol = m['vol'].to_numpy().astype(float); del m; gc.collect()
    n = len(c); hlc3 = (h + l + c) / 3.0
    pv = pd.Series(hlc3 * vol).rolling(HOURW).sum().to_numpy()
    vv = pd.Series(vol).rolling(HOURW).sum().to_numpy()
    vwap = pv / np.where(vv > 0, vv, np.nan)
    thr_sd = pd.Series(c).rolling(SMOW).std(ddof=0).to_numpy()
    rvol = vol / pd.Series(vol).rolling(HOURW).mean().to_numpy()
    ret = np.zeros(n); ret[1:] = np.abs(np.diff(c)) / pip
    smooth = pd.Series(ret).rolling(SMOW).std(ddof=0).to_numpy()    # LOW = smooth/mild
    vdist = np.abs(c - vwap) / thr_sd
    eb, pnl = _kern(h, l, c, bid, ask, vwap, thr_sd, pip, K, SL, TCAP, SLIP)
    for k in range(len(eb)):
        i = eb[k]
        rows[p].append((float(pnl[k]), float(rvol[i]), float(smooth[i]), float(vdist[i])))
    print(f"  {p} done (n={len(eb)})", flush=True)
    del h, l, c, bid, ask, vol, hlc3, pv, vv, vwap, thr_sd, rvol, ret, smooth, vdist; gc.collect()


allr = [x for p in PAIRS for x in rows[p]]
pnl = np.array([x[0] for x in allr]); rv = np.array([x[1] for x in allr])
sm = np.array([x[2] for x in allr]); vd = np.array([x[3] for x in allr])
N = len(pnl); npos = sum(1 for p in PAIRS if sum(x[0] for x in rows[p]) > 0)
wf = sum(1 for f in walk_forward(np.arange(N), pnl, N, 6) if f['net']>0)
mc = monte_carlo(pnl, 300)['p_net']
print(f"\n===== M5 VWAP-REVERSION (fade |c-VWAP|>={K}σ -> revert to VWAP, SL{SL}, 12 pairs) =====")
print(f"  ALL: {npos}/12 pos | net={pnl.sum():.0f}p n={N} WF {wf}/6 MC p={mc:.3f} WR={100*np.mean(pnl>0):.0f}%")


def buckets(name, metric, hint):
    ok = np.isfinite(metric); v, pn = metric[ok], pnl[ok]
    q1, q2 = np.quantile(v, [1/3, 2/3])
    print(f"  by {name} ({hint}) cuts@{q1:.2f}/{q2:.2f}:")
    for nm, mk in (("LOW", v <= q1), ("MID", (v > q1)&(v <= q2)), ("HIGH", v > q2)):
        print(f"    {nm:>4}: net={pn[mk].sum():>7.0f} n={mk.sum():>5} mean={pn[mk].mean():>5.2f} wr={100*(pn[mk]>0).mean():.0f}%")


buckets("rvol", rv, "LOW=quiet, HIGH=volume spike")
buckets("smoothness", sm, "LOW=smooth/mild prior run")
buckets("vwap_dist", vd, "HIGH=far from VWAP")
print("\nReal only if some (rvol x smoothness) subset is net>0, broad, MC<0.05 -> then OOS gate. "
      "Hypothesis: LOW rvol (quiet) + LOW smoothness (mild run) dislocations revert best.")
