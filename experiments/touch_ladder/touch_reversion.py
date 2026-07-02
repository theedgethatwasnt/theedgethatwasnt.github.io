#!/usr/bin/env python3
"""
TOUCH-LADDER, reversion side. Complements touch_ladder.py (which tested the BREAK/continuation
side). User refinement: the more a level is touched, the more its resting orders are CONSUMED,
so touch-count should switch the level's behavior:
  - LOW touch-count (fresh, orders intact)  -> strong level -> price REVERTS off it (fade).
  - HIGH touch-count (depleted)            -> weak level   -> price BREAKS through (continue).

Test the reversion side directly: at a fresh APPROACH to a swing level (touch), bet the bounce
AWAY from the level; condition on prior touch-count; net of spread; multi-pair, multi-TF,
IS/OOS. Hypothesis: reversion expectancy positive & largest at LOW touch-count, decaying (and
ideally flipping to continuation) as touches rise.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

PROJECT=Path("/path/to/projects/fx-core"); DATA=PROJECT/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
TF_RULE={"M15":"15min","H1":"1h","H4":"4h"}
IS_FRAC=0.6


def load_ohlcv(con,pair,rule):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def collect(TF,L,H,EPS_PIPS):
    con=duckdb.connect(); rule=TF_RULE[TF]; K=5    # bars for approach-sharpness
    D={'touch':[], 'vol':[], 'fwd':[], 'is':[], 'sharp':[]}
    for pair,(pip,sp) in PAIRS.items():
        r=load_ohlcv(con,pair,rule)
        if len(r)<L+H+50: continue
        h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values; n=len(c); eps=EPS_PIPS*pip
        Rs=pd.Series(h).rolling(L).max().shift(1).values
        Ss=pd.Series(l).rolling(L).min().shift(1).values
        vmean=pd.Series(v).rolling(L).mean().shift(1).values
        is_end=int(n*IS_FRAC); last=-10
        for t in range(L+1,n-H):
            if np.isnan(Rs[t]) or np.isnan(vmean[t]) or vmean[t]<=0: continue
            if t-last<H: continue
            R=Rs[t]; S=Ss[t]
            up_touch = h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R     # fresh approach to resistance, held
            dn_touch = l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
            if not (up_touch or dn_touch): continue
            if up_touch:
                touches=int(np.sum(h[t-L:t]>=R-eps)); d=-1          # fade: short off resistance
                sharp=(c[t]-c[t-K])/pip                              # run-UP into resistance (sharp=large)
            else:
                touches=int(np.sum(l[t-L:t]<=S+eps)); d=+1          # fade: long off support
                sharp=(c[t-K]-c[t])/pip                              # run-DOWN into support
            fwd=d*(c[t+H]-c[t])/pip - sp                            # reversion-direction, net spread
            D['touch'].append(touches); D['vol'].append(v[t]/vmean[t]); D['fwd'].append(fwd)
            D['is'].append(t<is_end); D['sharp'].append(sharp)
            last=t
    con.close()
    return (np.array(D['touch']),np.array(D['vol']),np.array(D['fwd']),np.array(D['is']),np.array(D['sharp']))


def run(TF,L,H,EPS):
    touch,vol,fwd,ism,sharp=collect(TF,L,H,EPS)
    if len(fwd)<200:
        print(f"  {TF} L{L} H{H}: too few touches ({len(fwd)})"); return
    print(f"\n===== {TF} L={L} H={H} eps={EPS}p — {len(fwd)} touches (IS {ism.sum()}/OOS {(~ism).sum()}) =====")
    ra=pd.Series(touch[ism]).rank().values; rb=pd.Series(fwd[ism]).rank().values
    sc=pd.Series(sharp[ism]).rank().values
    print(f"  IS rank-corr: touches→reversion={np.corrcoef(ra,rb)[0,1]:+.3f} (neg=>fewer touches revert more) | "
          f"sharpness→reversion={np.corrcoef(sc,rb)[0,1]:+.3f} (pos=>sharper approach reverts more)")
    def bk(x): return 0 if x<=1 else 1 if x<=3 else 2 if x<=6 else 3
    tb=np.array([bk(x) for x in touch])
    print(f"  reversion expectancy (pips, net spread) by touch-count:")
    print(f"  {'touches':<10}{'IS_exp':>9}{'IS_WR':>7}{'IS_n':>7}{'OOS_exp':>10}{'OOS_WR':>8}{'OOS_n':>7}")
    for b,lab in [(0,'1'),(1,'2-3'),(2,'4-6'),(3,'7+')]:
        mi=(tb==b)&ism; mo=(tb==b)&~ism
        ie=fwd[mi].mean() if mi.sum()>10 else float('nan'); oe=fwd[mo].mean() if mo.sum()>10 else float('nan')
        iw=(fwd[mi]>0).mean()*100 if mi.sum() else 0; ow=(fwd[mo]>0).mean()*100 if mo.sum() else 0
        print(f"  {lab:<10}{ie:>+9.2f}{iw:>6.0f}%{mi.sum():>7}{oe:>+10.2f}{ow:>7.0f}%{mo.sum():>7}")
    # USER'S sharpest case: FIRST touch (touches<=1) reached by a SHARP approach -> fade
    ft=touch<=1
    if ft.sum()>60:
        shi=np.quantile(sharp[ft&ism],.66) if (ft&ism).sum()>20 else np.inf
        print(f"  FIRST-touch (touches≤1), split by approach sharpness (IS p66={shi:.1f}p run-in):")
        for lab,m in [("sharp(run≥p66)",sharp>=shi),("gentle(run<p66)",sharp<shi)]:
            mi=ft&m&ism; mo=ft&m&~ism
            if mi.sum()>10 and mo.sum()>10:
                print(f"    {lab:<18} IS {fwd[mi].mean():+.2f}p (n{mi.sum()}, WR{(fwd[mi]>0).mean()*100:.0f}%) | "
                      f"OOS {fwd[mo].mean():+.2f}p (n{mo.sum()}, WR{(fwd[mo]>0).mean()*100:.0f}%)")


if __name__=="__main__":
    print("TOUCH-LADDER reversion side — fade a fresh swing-level touch, by touch-count. Net spread, IS/OOS, 12 pairs.")
    for TF,L,H,EPS in [("M15",48,12,5),("H1",24,8,5),("H1",48,12,8),("H4",20,8,10),("H4",30,12,15)]:
        run(TF,L,H,EPS)
    print("\n  LEVER if LOW-touch reversion is positive IS AND OOS (and corr negative: fresh levels revert, rehearsed ones don't).")
