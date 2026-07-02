#!/usr/bin/env python3
"""
ZR-on-big-bars — fade an exceptionally-big (but not rare) H1 bar, 2-leg double-or-bust.
User idea: a big H1 bar tends to be followed by a correction; bet the correction; if wrong,
double once and bet again; if wrong again, bust.

Mechanics (faithful 2-leg Zone Recovery, fade direction s = -sign(big-bar body)):
  entry0 = big bar close. zone Z.
  Phase 1 (1u): WIN if price reaches the correction barrier (entry0 - s*Z*... toward correction)
    → +Z. If price reaches the continuation barrier (Z the other way) → DOUBLE (add 2u).
  Phase 2 (3u): WIN if price returns to entry0 → aggregate +2Z. BUST if price reaches a 2nd
    continuation zone (2Z from entry0) → aggregate -4Z.
  Barriers resolved within an H1 bar by the project's within-bar sequencing (R2: bull=low→high,
  bear=high→low). Time-capped. Net of per-pair spread on every unit traded.

Reports the EMPIRICAL expected value per cycle, p(correct on leg1), double rate, p(recover on
leg2), bust rate — across all big-bar events, 12 pairs, IS/OOS. "big but not rare" = body
percentile (90/95/97). Z = fraction of the big-bar body.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; MAXBARS=24


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()


@nb.njit(cache=True)
def run_cycle(o,h,l,c, start, s, Z, pip):
    """s=-1 short (faded up bar) / s=+1 long. Z in pips. Returns (pnl_pips_gross, units_traded,
    outcome) outcome: 0 win1, 1 win2(recover), 2 bust, 3 timecap."""
    e0=c[start]
    # correction barrier (favorable) and continuation barrier (adverse)
    corr1 = e0 + s*Z*pip            # price moves Z toward correction (s direction)
    cont1 = e0 - s*Z*pip            # price moves Z toward continuation (against)
    cont2 = e0 - s*2*Z*pip          # bust barrier
    doubled=False
    n=len(c)
    for i in range(start+1, min(start+1+MAXBARS, n)):
        bull = c[i]>=o[i]
        # ordered extremes within the bar
        # for s=-1 (short): favorable=price down=low; adverse=price up=high
        # we test both extremes in path order
        ext = (l[i], h[i]) if bull else (h[i], l[i])   # (first, second)
        for px in ext:
            if not doubled:
                # phase1: corr1 favorable, cont1 adverse
                if (s==-1 and px<=corr1) or (s==1 and px>=corr1):
                    return (Z, 1.0, 0)
                if (s==-1 and px>=cont1) or (s==1 and px<=cont1):
                    doubled=True                       # add 2u at cont1
                    continue
            else:
                # phase2: recover to e0 (favorable), bust at cont2 (adverse)
                if (s==-1 and px<=e0) or (s==1 and px>=e0):
                    return (2*Z, 3.0, 1)
                if (s==-1 and px<=cont2) or (s==1 and px>=cont2):
                    return (-4*Z, 3.0, 2)
    # time cap: mark to market aggregate
    P=c[min(start+MAXBARS, n-1)]
    if not doubled:
        return (s*(P-e0)/pip, 1.0, 3)
    agg = s*(e0-P)/pip + 2*s*(cont1-P)/pip   # 1u from e0 + 2u from cont1, short pnl = s*(entry-P)
    return (agg, 3.0, 3)


@nb.njit(cache=True)
def run_all(o,h,l,c, idx, sgn, Z, pip, sp, is_end):
    m=len(idx); pnl=np.empty(m); isf=np.empty(m,np.int64); outc=np.empty(m,np.int64)
    for k in range(m):
        g,units,oc = run_cycle(o,h,l,c, idx[k], sgn[k], Z, pip)
        pnl[k]=g - units*sp           # spread on each unit traded
        isf[k]= 1 if idx[k]<is_end else 0
        outc[k]=oc
    return pnl, isf, outc


def main():
    _o=np.zeros(40); run_cycle(_o,_o+1,_o,_o+0.5,1,-1,10.0,0.01)
    con=duckdb.connect()
    print("ZR on big H1 bars — fade + 2-leg double-or-bust. EV per cycle, net spread, 12 pairs, IS/OOS.")
    print(f"  {'pctile':>7}{'Zfrac':>7}{'events':>8}{'p_corr1':>9}{'dbl%':>7}{'p_rec2':>8}{'bust%':>7}{'IS_EV':>8}{'OOS_EV':>8}")
    for pct in (90,95,97):
        for zf in (0.5,1.0):
            allp=[]; alli=[]; allo=[]
            for pair,(pip,sp) in PAIRS.items():
                r=load(con,pair)
                o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values
                body=np.abs(c-o)/pip
                is_end=int(len(c)*IS_FRAC)
                thr=np.quantile(body[:is_end], pct/100.0)     # IS-only threshold (R5)
                ev_idx=np.where(body>=thr)[0]
                ev_idx=ev_idx[(ev_idx>20)&(ev_idx<len(c)-MAXBARS-1)]
                if len(ev_idx)<30: continue
                sgn=(-np.sign(c[ev_idx]-o[ev_idx])).astype(np.int64)   # fade
                # Z as fraction of THAT bar's body (pips)
                for j in ev_idx:
                    pass
                Zarr=body[ev_idx]*zf
                # run per pair with per-event Z: loop (Z varies) — call run_cycle directly
                for kk,j in enumerate(ev_idx):
                    g,u,oc=run_cycle(o,h,l,c,int(j),int(sgn[kk]),float(Zarr[kk]),pip)
                    allp.append(g-u*sp); alli.append(1 if j<is_end else 0); allo.append(oc)
            allp=np.array(allp); alli=np.array(alli); allo=np.array(allo)
            if len(allp)<100: continue
            dbl=(allo!=0); rec=(allo==1); bust=(allo==2)
            ism=alli==1
            print(f"  {pct:>7}{zf:>7.1f}{len(allp):>8}{(allo==0).mean()*100:>8.0f}%{dbl.mean()*100:>6.0f}%"
                  f"{(rec.sum()/max(dbl.sum(),1))*100:>7.0f}%{bust.mean()*100:>6.0f}%"
                  f"{allp[ism].mean():>+8.2f}{allp[~ism].mean():>+8.2f}")
    con.close()
    print("\n  EV is per cycle in pips, net spread. +EV needs the correction edge to beat the martingale's")
    print("  fair-game baseline AND spread. Prior (spread_sigma): direction after big bars ≈ random,")
    print("  cover anti-selected for chop → expect EV ≤ 0.")


if __name__=="__main__":
    main()
