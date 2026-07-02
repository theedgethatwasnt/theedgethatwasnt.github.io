#!/usr/bin/env python3
"""
TV ratings Phase 2 — screen the 26 INDIVIDUAL component votes, prioritising
FREQUENT traders (the user wants frequent short bites; longer holds OK but
frequency is the requirement).

Loads the Phase-0 cached TF bars (o/h/l/c/v/bid/ask), recomputes each indicator's
Buy/Sell/Neutral vote, and screens each vote vs forward return net of spread,
follow & fade, scored net p/d + AMDDP5 + trades/day. OOS.

Output ranks by AMDDP5 *among cells that trade frequently* (>= FREQ_FLOOR/day),
and flags cross-pair-consistent components (real signal, not single-pair noise).
"""
import gc, sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

CACHE   = Path(__file__).parent / "cache"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)
PAIRS = sys.argv[1:] or ["EUR_USD", "USD_JPY", "GBP_USD"]
TFS = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]
TF_BPD = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1}
HORIZONS = [1, 3, 6]
IS_FRAC = 0.70
AMDDP_K = 0.05
FREQ_FLOOR = 5.0   # trades/day to count as "frequent"


def pip_of(p): return 0.01 if "JPY" in p else 0.0001
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100-100/(1+rs)

def stoch(h, l, c, k=14, d=3):
    ll = l.rolling(k).min(); hh = h.rolling(k).max()
    ks = (100*(c-ll)/(hh-ll).replace(0, np.nan)).rolling(d).mean()
    return ks, ks.rolling(d).mean()

def cci(h, l, c, n=20):
    tp = (h+l+c)/3; ma = tp.rolling(n).mean()
    return (tp-ma)/(0.015*(tp-ma).abs().rolling(n).mean().replace(0, np.nan))

def adx_di(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1/n, adjust=False).mean()/atr
    ndi = 100*minus.ewm(alpha=1/n, adjust=False).mean()/atr
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, ndi

def macd(c): m = ema(c, 12)-ema(c, 26); return m, ema(m, 9)
def willr(h, l, c, n=14):
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    return -100*(hh-c)/(hh-ll).replace(0, np.nan)
def ultimate(h, l, c):
    pc = c.shift(); tl = pd.concat([l, pc], axis=1).min(axis=1)
    bp = c-tl; tr = pd.concat([h, pc], axis=1).max(axis=1)-tl
    return 100*(4*(bp.rolling(7).sum()/tr.rolling(7).sum())
                + 2*(bp.rolling(14).sum()/tr.rolling(14).sum())
                + (bp.rolling(28).sum()/tr.rolling(28).sum()))/7
def hull(c, n=9):
    wma = lambda s, w: s.rolling(w).apply(lambda x: np.dot(x, np.arange(1, w+1))/(w*(w+1)/2), raw=True)
    return wma(2*wma(c, n//2)-wma(c, n), int(np.sqrt(n)))
def vwma(c, v, n=20): return (c*v).rolling(n).sum()/v.rolling(n).sum().replace(0, np.nan)


def compute_votes(df):
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    V = {}
    for n in (10, 20, 30, 50, 100, 200):
        V[f"ema{n}"] = np.sign(c-ema(c, n)); V[f"sma{n}"] = np.sign(c-sma(c, n))
    V["ichi"] = np.sign(c-(h.rolling(26).max()+l.rolling(26).min())/2)
    V["vwma"] = np.sign(c-vwma(c, v))
    hm = hull(c); V["hull"] = np.sign(hm-hm.shift())
    r = rsi(c); V["rsi"] = np.where((r < 30) & (r > r.shift()), 1, np.where((r > 70) & (r < r.shift()), -1, 0))
    kk, dd = stoch(h, l, c)
    V["stoch"] = np.where((kk < 20) & (dd < 20) & (kk > dd), 1, np.where((kk > 80) & (dd > 80) & (kk < dd), -1, 0))
    cc = cci(h, l, c); V["cci"] = np.where((cc < -100) & (cc > cc.shift()), 1, np.where((cc > 100) & (cc < cc.shift()), -1, 0))
    adx, pdi, ndi = adx_di(h, l, c)
    V["adx"] = np.where((adx > 20) & (pdi > ndi), 1, np.where((adx > 20) & (ndi > pdi), -1, 0))
    ao = sma((h+l)/2, 5)-sma((h+l)/2, 34)
    V["ao"] = np.where((ao > 0) & (ao > ao.shift()), 1, np.where((ao < 0) & (ao < ao.shift()), -1, 0))
    mom = c-c.shift(10); V["mom"] = np.where(mom > mom.shift(), 1, np.where(mom < mom.shift(), -1, 0))
    mac, sig = macd(c); V["macd"] = np.sign(mac-sig)
    rr = rsi(c); sll = rr.rolling(14).min(); shh = rr.rolling(14).max()
    srk = (100*(rr-sll)/(shh-sll).replace(0, np.nan)).rolling(3).mean(); srd = srk.rolling(3).mean()
    V["stochrsi"] = np.where((srk < 20) & (srk > srd), 1, np.where((srk > 80) & (srk < srd), -1, 0))
    wr = willr(h, l, c); V["willr"] = np.where((wr < -80) & (wr > wr.shift()), 1, np.where((wr > -20) & (wr < wr.shift()), -1, 0))
    e13 = ema(c, 13); bbp = (h-e13)+(l-e13)
    V["bbp"] = np.where((bbp > 0) & (bbp > bbp.shift()), 1, np.where((bbp < 0) & (bbp < bbp.shift()), -1, 0))
    uo = ultimate(h, l, c); V["uo"] = np.where(uo > 70, 1, np.where(uo < 30, -1, 0))
    return pd.DataFrame({k: np.asarray(V[k], float) for k in V}, index=c.index)


def screen(vote, mid, bid, ask, pip, H, fade, bpd):
    n = len(mid); sgn = -vote if fade else vote
    idx = np.where((vote != 0) & (np.arange(n) < n-H))[0]
    if len(idx) < 50: return np.nan, np.nan, 0.0, np.nan
    nets = []; amd = []
    for i in idx:
        d = sgn[i]
        entry = mid[i]; spr = (ask[i]-bid[i])/pip
        net = d*(mid[i+H]-entry)/pip - spr
        path = d*(mid[i+1:i+H+1]-entry)/pip
        nets.append(net); amd.append(net-AMDDP_K*float(np.clip(-path, 0, None).sum()))
    nets = np.array(nets); amd = np.array(amd); days = n/bpd
    return nets.sum()/days, amd.sum()/days, len(nets)/days, float((nets > 0).mean()*100)


rows = []
for pair in PAIRS:
    pip = pip_of(pair)
    for tf in TFS:
        f = CACHE / f"{pair}_{tf}.parquet"
        if not f.exists(): continue
        df = pd.read_parquet(f)
        votes = compute_votes(df)
        n = len(df); ie = int(n*IS_FRAC); bpd = TF_BPD[tf]
        mid = df["c"].values; bid = df["bid"].values; ask = df["ask"].values
        for ind in votes.columns:
            vv = votes[ind].values
            for H in HORIZONS:
                for fade in (False, True):
                    npd, apd, tpd, wr = screen(vv[ie:], mid[ie:], bid[ie:], ask[ie:], pip, H, fade, bpd)
                    if tpd > 0 and not np.isnan(npd):
                        rows.append(dict(pair=pair, tf=tf, ind=ind, H=H,
                                         dir="fade" if fade else "follow",
                                         net_ppd=npd, amddp5_ppd=apd, tpd=tpd, wr=wr))
        del df, votes; gc.collect()
    print(f"  {pair} done")

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "phase2_component_edge.csv", index=False)
print(f"\nScreened {len(R)} component cells → {RESULTS}/phase2_component_edge.csv\n")

freq = R[R.tpd >= FREQ_FLOOR]
print("="*100)
print(f"FREQUENT traders only (>= {FREQ_FLOOR} trades/day) — top 20 by AMDDP5")
print("="*100)
print(f"{'pair':<9}{'tf':>4}{'ind':>10}{'dir':>7}{'H':>3}{'tpd':>7}{'net_ppd':>9}{'amddp5':>9}{'wr':>5}")
for _, r in freq.sort_values("amddp5_ppd", ascending=False).head(20).iterrows():
    print(f"{r.pair:<9}{r.tf:>4}{r.ind:>10}{r['dir']:>7}{int(r.H):>3}{r.tpd:>7.1f}"
          f"{r.net_ppd:>+9.1f}{r.amddp5_ppd:>+9.1f}{r.wr:>4.0f}%")

print(f"\n{'='*100}\nFrequent (>= {FREQ_FLOOR}/d) AMDDP5>0 by indicator — cross-pair consistency\n{'='*100}")
fp = freq[freq.amddp5_ppd > 0]
print(f"{'indicator':>10}{'cells+':>8}{'pairs':>7}{'best_amddp5':>13}{'best_net':>10}{'med_tpd':>9}")
for ind, g in fp.groupby("ind"):
    print(f"{ind:>10}{len(g):>8}{g.pair.nunique():>7}{g.amddp5_ppd.max():>+13.1f}"
          f"{g.net_ppd.max():>+10.1f}{g.tpd.median():>9.1f}")
if len(fp) == 0:
    print("  NONE — no frequent (>=5/d) component clears spread on AMDDP5.")
print(f"\nFrequent cells: {len(freq)} | AMDDP5>0: {len(fp)} ({100*len(fp)/max(len(freq),1):.1f}%) "
      f"| net>0: {(freq.net_ppd>0).sum()}")
print("Reminder: frequent + short-bite = max spread exposure; positive here must still clear WF+MC+surrogate-null.")
