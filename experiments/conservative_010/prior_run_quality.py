#!/usr/bin/env python3
"""Prior-run QUALITY filter: only take entries whose run INTO the signal was MILD-and-SUSTAINED,
not exhausted-sharp. Applies the Oracle 'meandering/mild' finding as an entry gate.

Two candidate configs:
  BREAKOUT (winner): H1, SMA50, K=1, TP50/SL100. Prior run = the uptrend that just rolled over.
  PULLBACK (ruled out): M5, SMA50, K=1, TP50/SL150. Rescue attempt.

Prior-run metrics over the W bars BEFORE the entry bar (all causal, computed from closed bars):
  rstd  = std of per-bar returns (pips)   -> LOW = smooth/mild, HIGH = choppy/violent (rolling std)
  vel   = |SMA[i-1]-SMA[i-1-W]|/W (pips/bar) -> LOW = shallow/mild, HIGH = steep/sharp
  rlen  = consecutive prior same-sign-slope bars -> HIGH = sustained, LOW = brief
  eff   = |net|/path over W -> HIGH = clean one-way, mid ~0.5 = meandering
Bucket trades into terciles by each metric; report net/n/mean/WR. Then validate a
'mild+sustained' filtered subset (WF+MC, 4-pair and ex-EUR_USD).
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba
from validate import walk_forward, monte_carlo, equity_drawdown

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SLIP = 2.0; SLOPE_LB = 3; W = 20
H1_NS = 3_600_000_000_000; M5_NS = 300_000_000_000


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, lower, upper, longsig, shortsig, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if shortsig[i]:
                pos = -1; entry = bid[i]; _eb[nt] = i; continue
            if longsig[i]:
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


def build(c, kind, K=1.0):
    n = len(c); s = pd.Series(c)
    sma = s.rolling(50).mean().to_numpy(); sd = s.rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(sma) & ~np.isnan(sd)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
    slope = np.where(v, slope, 0.0)
    lower = np.where(v, sma - K*sd, -1e18); upper = np.where(v, sma + K*sd, 1e18)
    return sma, slope, lower, upper, v


def prior_metrics(c, sma, slope, eb, pip):
    """Causal prior-run metrics over [i-W, i-1] for each entry bar i."""
    ret = np.zeros(len(c)); ret[1:] = np.abs(np.diff(c)) / pip
    rstd, vel, rlen, eff = [], [], [], []
    for i in eb:
        a = max(i - W, 1)
        seg = c[a:i]                        # prior closes (strictly before entry bar)
        if len(seg) < 3:
            rstd.append(np.nan); vel.append(np.nan); rlen.append(0); eff.append(np.nan); continue
        r = np.diff(seg) / pip
        rstd.append(float(np.std(r)))
        vel.append(abs(sma[i-1] - sma[a]) / max(i-1-a, 1) / pip)
        path = np.sum(np.abs(r)) + 1e-9
        eff.append(abs(seg[-1] - seg[0]) / (path))
        # sustained: consecutive prior bars with slope sign == sign(slope[i-1])
        sg = np.sign(slope[i-1]); k = 0; j = i - 1
        while j >= 1 and np.sign(slope[j]) == sg and sg != 0:
            k += 1; j -= 1
        rlen.append(k)
    return (np.array(rstd), np.array(vel), np.array(rlen, float), np.array(eff))


def breakout_sig(c):
    _, slope, lower, upper, v = build(c, "brk")
    sp = np.empty(len(c)); sp[0] = 0.0; sp[1:] = slope[:-1]
    tneg = (slope < 0) & (sp >= 0) & v; tpos = (slope > 0) & (sp <= 0) & v
    return (tpos & (c > upper)), (tneg & (c < lower))     # long, short


def pullback_sig(h, l, c):
    _, slope, lower, upper, v = build(c, "pb")
    return ((slope > 0) & (l <= lower) & v), ((slope < 0) & (h >= upper) & v)


CFG = {"BREAKOUT": dict(tf=H1_NS, tp=50.0, sl=100.0),
       "PULLBACK": dict(tf=M5_NS, tp=50.0, sl=150.0)}
rows = {k: [] for k in CFG}
for p in PAIRS:
    d = load_pair_ba(p); pip = d['pip']
    ts = np.asarray(d['ts']).astype('datetime64[ns]').astype(np.int64)
    df = pd.DataFrame({'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
                       'bid': d['bid_c'], 'ask': d['ask_c']}); del d; gc.collect()
    for kind, cf in CFG.items():
        h, l, c, bid, ask, tsh = resample(df, ts, cf['tf'])
        sma, slope, lower, upper, v = build(c, kind)
        ls, ss = breakout_sig(c) if kind == "BREAKOUT" else pullback_sig(h, l, c)
        eb, pnl = _kern(h, l, c, bid, ask, lower, upper, ls, ss, pip, cf['tp'], cf['sl'], SLIP)
        rstd, vel, rlen, eff = prior_metrics(c, sma, slope, eb, pip)
        et = tsh[eb]
        for i in range(len(pnl)):
            rows[kind].append(dict(pnl=float(pnl[i]), rstd=rstd[i], vel=vel[i],
                                   rlen=rlen[i], eff=eff[i], pair=p, exit_time=int(et[i])))
    print(f"  {p} done", flush=True)
    del df, ts; gc.collect()


def terciles(rws, metric):
    vals = np.array([r[metric] for r in rws]); pn = np.array([r['pnl'] for r in rws])
    ok = ~np.isnan(vals); vals, pn = vals[ok], pn[ok]
    if len(vals) < 30: return None
    q1, q2 = np.quantile(vals, [1/3, 2/3])
    out = []
    for name, m in (("LOW", vals <= q1), ("MID", (vals > q1) & (vals <= q2)), ("HIGH", vals > q2)):
        s = pn[m]
        out.append((name, len(s), float(s.sum()), float(s.mean()) if len(s) else 0,
                    100*float((s > 0).mean()) if len(s) else 0))
    return q1, q2, out


def valid(rws):
    tl = sorted(rws, key=lambda r: r['exit_time']); net = np.array([r['pnl'] for r in tl]); N = len(net)
    if N < 30: return N, float(net.sum()), 0, 1.0
    eb = np.arange(N)
    return N, float(net.sum()), sum(1 for f in walk_forward(eb, net, N, 6) if f['net']>0), \
        monte_carlo(net, 300)['p_net']


for kind in CFG:
    rws = rows[kind]
    tot = sum(r['pnl'] for r in rws)
    print(f"\n================  {kind}  (n={len(rws)}, total net {tot:.0f}p)  ================")
    for metric, hint in (("rstd", "LOW=smooth/mild"), ("vel", "LOW=shallow/mild"),
                         ("rlen", "HIGH=sustained"), ("eff", "mid~meander")):
        t = terciles(rws, metric)
        if not t: continue
        q1, q2, out = t
        print(f"  by {metric} ({hint}) cuts@{q1:.2f}/{q2:.2f}:  " +
              "  ".join(f"{nm}:n={n} net={s:.0f} mean={mn:.2f} wr={wr:.0f}%" for nm, n, s, mn, wr in out))
    # filter: mild+sustained = rstd LOW (<= median) AND rlen >= median
    vs_r = np.array([r['rstd'] for r in rws]); vs_l = np.array([r['rlen'] for r in rws])
    okm = ~np.isnan(vs_r)
    rmed = np.nanmedian(vs_r); lmed = np.nanmedian(vs_l)
    filt = [r for r in rws if not np.isnan(r['rstd']) and r['rstd'] <= rmed and r['rlen'] >= lmed]
    noeu = [r for r in filt if r['pair'] != "EUR_USD"]
    N, ns, wf, mcp = valid(filt); N3, ns3, wf3, mcp3 = valid(noeu)
    print(f"  >> FILTER (rstd<=med AND rlen>=med): 4p net={ns:.0f} n={N} WF {wf}/6 MC p={mcp:.3f} | "
          f"exEU net={ns3:.0f} n={N3} WF {wf3}/6 MC p={mcp3:.3f}")
print("\nHelps if a bucket is monotone (mild>>sharp) AND the filtered subset keeps net + MC<0.05.")
