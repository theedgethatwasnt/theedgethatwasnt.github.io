#!/usr/bin/env python3
"""
QUIET DRIFT — is there a low-volatility, high-persistence regime whose drift covers spread?
Distinct from all prior momentum/breakout work, which entered INTO volatility and whipsawed.
Hypothesis: in low realized-vol + high efficiency-ratio (quiet, non-noisy) regimes, price
drifts steadily in one direction; low noise => no whipsaw => spread paid once => the drift
can clear cost over a multi-bar hold. Test it DIRECTLY and stationarily.

Method (causal, multi-pair, multi-TF, net of spread):
  - M5 mid -> resample to M15 / H1 / H4.
  - Per bar t over trailing window W: efficiency ratio ER = |c[t]-c[t-W]| / Σ|Δc| (0 chop..1
    pure drift); realized vol = std of TF log-returns over W; drift = sign(c[t]-c[t-W]).
  - Non-overlapping entries (step H). Forward return over hold H in the DRIFT direction,
    in pips, minus the pair's spread.
  - Bucket by vol-quartile × ER-quartile (thresholds from IS only, R5). Expectancy per cell
    IS and OOS. Hypothesis predicts the LOW-VOL × HIGH-ER cell is positive in BOTH.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

PROJECT = Path("/path/to/projects/fx-core"); DATA = PROJECT/"data"/"m5_ohlc"
PAIRS = {  # pip, spread pips (OANDA retail median)
 "USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
TF_RULE={"M15":"15min","H1":"1h","H4":"4h","D":"1D"}
IS_FRAC=0.6


def load_tf(con, pair, rule):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp, close, volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    c=df["close"].resample(rule).last(); v=df["volume"].resample(rule).sum()
    out=pd.DataFrame({"c":c,"v":v}).dropna()
    return out["c"].values.astype(float), out["v"].values.astype(float)


def er_vol(c, W):
    d=np.diff(c,prepend=c[0]); ad=np.abs(d); cs=np.cumsum(ad)
    er=np.full(len(c),np.nan); vol=np.full(len(c),np.nan)
    er[W:]=np.abs(c[W:]-c[:-W])/np.where((cs[W:]-cs[:-W])>0,(cs[W:]-cs[:-W]),np.nan)
    r=d.copy()  # absolute price diffs; use as proxy returns for vol (pip-agnostic ratio buckets)
    cs1=np.cumsum(r); cs2=np.cumsum(r*r)
    vol[W:]=np.sqrt(np.maximum((cs2[W:]-cs2[:-W])/W-((cs1[W:]-cs1[:-W])/W)**2,0))
    return er,vol


def run(TF, W, H):
    con=duckdb.connect()
    rule=TF_RULE[TF]
    # gather pooled samples with IS/OOS flag
    S={'er':[], 'vol':[], 'fwd':[], 'is':[], 'vrel':[]}
    for pair,(pip,sp) in PAIRS.items():
        c,vraw=load_tf(con,pair,rule)
        if len(c)<W+H+50: continue
        er,vol=er_vol(c,W)
        # relative tick-volume at t = vol[t] / trailing-mean(W), causal
        vcs=np.cumsum(vraw); vmean=np.full(len(c),np.nan)
        vmean[W:]=(vcs[W:]-vcs[:-W])/W
        vrel=np.where(vmean>0, vraw/vmean, np.nan)
        is_end=int(len(c)*IS_FRAC)
        for t in range(W, len(c)-H, H):                  # non-overlapping
            if np.isnan(er[t]) or np.isnan(vol[t]): continue
            drift=np.sign(c[t]-c[t-W])
            fwd=drift*(c[t+H]-c[t])/pip - sp             # net of spread, drift direction
            S['er'].append(er[t]); S['vol'].append(vol[t]); S['fwd'].append(fwd)
            S['is'].append(t<is_end); S['vrel'].append(vrel[t])
    con.close()
    er=np.array(S['er']); vol=np.array(S['vol']); fwd=np.array(S['fwd']); ism=np.array(S['is']); vrel=np.array(S['vrel'])
    # quartile thresholds from IS only (R5)
    eq=np.quantile(er[ism],[.25,.5,.75]); vq=np.quantile(vol[ism],[.25,.5,.75])
    def qb(x,q): return np.digitize(x,q)   # 0..3
    eb=qb(er,eq); vb=qb(vol,vq)
    print(f"\n===== {TF}  W={W}  H={H}  (n={len(fwd)}, IS {ism.sum()} / OOS {(~ism).sum()}) =====")
    print("  expectancy (pips, net spread) by VOL row × ER col  [IS // OOS] ; *=low-vol×high-ER cell")
    print(f"  {'':<10}{'ER_Q1(chop)':>16}{'ER_Q2':>14}{'ER_Q3':>14}{'ER_Q4(drift)':>16}")
    for vrow in range(4):
        cells=[]
        for ecol in range(4):
            mi=(vb==vrow)&(eb==ecol)&ism; mo=(vb==vrow)&(eb==ecol)&~ism
            ie=fwd[mi].mean() if mi.sum() else float('nan'); oe=fwd[mo].mean() if mo.sum() else float('nan')
            star='*' if (vrow==0 and ecol==3) else ' '
            cells.append(f"{ie:+6.2f}//{oe:+6.2f}{star}")
        lab=['VOL_Q1(low)','VOL_Q2','VOL_Q3','VOL_Q4(high)'][vrow]
        print(f"  {lab:<10}"+"".join(f"{c:>15}" for c in cells))
    # the headline cell
    mi=(vb==0)&(eb==3)&ism; mo=(vb==0)&(eb==3)&~ism
    print(f"  >>> LOW-VOL×HIGH-ER: IS exp {fwd[mi].mean():+.2f}p (n{mi.sum()}, WR{(fwd[mi]>0).mean()*100:.0f}%) | "
          f"OOS exp {fwd[mo].mean():+.2f}p (n{mo.sum()}, WR{(fwd[mo]>0).mean()*100:.0f}%)")
    # EXTREME quiet-drift corner: vol<=p10(IS) AND er>=p90(IS); then SPLIT by tick-volume
    vlo=np.quantile(vol[ism],.10); ehi=np.quantile(er[ism],.90)
    corner=(vol<=vlo)&(er>=ehi)
    ci=corner&ism; co=corner&~ism
    print(f"  >>> EXTREME corner (vol≤p10 & ER≥p90): IS {fwd[ci].mean():+.2f}p (n{ci.sum()}) | OOS {fwd[co].mean():+.2f}p (n{co.sum()})")
    # PROPER-N volume cut: the QUARTILE low-vol×high-ER cell (n in hundreds), split by
    # tick-volume tercile (thresholds IS-only). Tests 'conviction drift' at adequate n.
    qcell=(vb==0)&(eb==3)
    vt=np.quantile(vrel[qcell&ism&~np.isnan(vrel)],[.33,.66]) if (qcell&ism).sum()>30 else [1,2]
    vbk=np.digitize(vrel,vt)
    for k,vlab in [(0,'loVol'),(1,'midVol'),(2,'hiVol')]:
        ii=qcell&(vbk==k)&ism; oo=qcell&(vbk==k)&~ism
        if ii.sum()>20 and oo.sum()>20:
            print(f"        {vlab:>6} drift: IS {fwd[ii].mean():+.2f}p (n{ii.sum()}, WR{(fwd[ii]>0).mean()*100:.0f}%) | "
                  f"OOS {fwd[oo].mean():+.2f}p (n{oo.sum()}, WR{(fwd[oo]>0).mean()*100:.0f}%)")
    return (fwd[mi].mean() if mi.sum() else 0, fwd[mo].mean() if mo.sum() else 0,
            fwd[ci].mean() if ci.sum() else 0, fwd[co].mean() if co.sum() else 0)


if __name__=="__main__":
    print("QUIET DRIFT test — low-vol + high-ER, bet the drift, net of spread, IS/OOS, 12 pairs.")
    grid=[("M15",20,20),("H1",20,20),("H1",40,20),("H4",10,10),("H4",20,10),("H4",20,20),
          ("D",10,5),("D",20,10),("D",20,20)]
    res=[]
    for TF,W,H in grid:
        i,o,ci,co=run(TF,W,H); res.append((TF,W,H,i,o,ci,co))
    print("\n===== SUMMARY: drift-direction, net spread =====")
    print(f"  {'TF':<5}{'W':>4}{'H':>4}{'Q-cell IS':>10}{'Q OOS':>8}{'extreme IS':>12}{'extreme OOS':>12}  stat+?")
    for TF,W,H,i,o,ci,co in res:
        ok='YES' if (ci>0 and co>0) else 'no'
        print(f"  {TF:<5}{W:>4}{H:>4}{i:>+10.2f}{o:>+8.2f}{ci:>+12.2f}{co:>+12.2f}  {ok}")
    print("\n  LEVER if the extreme quiet-drift corner is positive IS AND OOS across TFs (incl. volume split).")
