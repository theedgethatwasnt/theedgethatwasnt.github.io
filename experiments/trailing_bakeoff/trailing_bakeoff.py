#!/usr/bin/env python3
"""
TRAILING-STOP BAKE-OFF — which trail scheme is best for retail FX? Common testbed: a Donchian
N-bar breakout TREND entry (so there's a move to ride), 12 pairs, H4/H1, net spread, IS/OOS.
Only the EXIT varies. PSAR is our current 010 exit and the baseline to beat.

Schemes (exit a long when the bar LOW breaches the stop; symmetric for shorts):
  0 fixedSLTP   : SL=entry-s*ATR0, TP=entry+t*ATR0           (non-trailing reference)
  1 psar        : Parabolic SAR (af 0.02->0.20)
  2 chandelier  : stop = HH(Nc) - m*ATR   (long)             [non-accelerating, vol-buffered]
  3 nbar        : stop = LL(Nb)            (long)             [pure structure step]
  4 atr_ratchet : stop = max(prev, close - m*ATR)            [ratchet, vol]
  5 supertrend  : SuperTrend line flip
  6 giveback    : lock f*MFE once MFE>=arm                   [ZR-style, proven for us]
  7 fixedpip    : stop = peak - P pips                       [OANDA native]
Each scheme reported at its IS-best param; ranked on OOS p/d + MaxDD + capture ratio.
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
TF_RULE={"H1":"1h","H4":"4h"}; IS_FRAC=0.6; DONCH=20


def load(con,pair,rule):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()


def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


@nb.njit(cache=True)
def psar_series(h,l,af0,afmax):
    n=len(h); sar=np.empty(n); up=True; af=af0; ep=h[0]; sar[0]=l[0]
    for i in range(1,n):
        sar[i]=sar[i-1]+af*(ep-sar[i-1])
        if up:
            if l[i]<sar[i]: up=False; sar[i]=ep; ep=l[i]; af=af0
            else:
                if h[i]>ep: ep=h[i]; af=min(af+af0,afmax)
        else:
            if h[i]>sar[i]: up=True; sar[i]=ep; ep=h[i]; af=af0
            else:
                if l[i]<ep: ep=l[i]; af=min(af+af0,afmax)
    return sar


def supertrend(h,l,c,a,m):
    hl2=(h+l)/2; ub=hl2+m*a; lb=hl2-m*a; n=len(c)
    st=np.empty(n); dir_=np.ones(n)
    fu=ub.copy(); fl=lb.copy()
    for i in range(1,n):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
    for i in range(1,n):
        if c[i]>fu[i-1]: dir_[i]=1
        elif c[i]<fl[i-1]: dir_[i]=-1
        else: dir_[i]=dir_[i-1]
    st=np.where(dir_==1,fl,fu)
    return st


@nb.njit(cache=True)
def sim(o,h,l,c, dHH,dLL, atr0, chl,chs,nbl,nbh, pip,sp,
        scheme, p1,p2, is_end):
    """Donchian breakout entry; exit by `scheme`, every stop trade-ANCHORED at entry.
    PSAR & SuperTrend reset per trade (no global-series fake-profit bug); open trades at
    end-of-data are marked to market and counted (no survivorship). Returns (pnl,mfe,entrybar)."""
    n=len(o); pos=0; entry=0.0; ebar=-1; peak=0.0; mfe=0.0; trail=0.0; aentry=0.0
    psar=0.0; ep=0.0; af=0.0                       # per-trade PSAR state
    P=np.empty(n); M=np.empty(n); E=np.empty(n,np.int64); k=0
    for i in range(1,n):
        if pos==0:
            if c[i]>dHH[i] and atr0[i]>0:
                pos=1; entry=c[i]; ebar=i; peak=c[i]; mfe=0.0; aentry=atr0[i]
                trail=entry-p1*aentry; psar=l[i]; ep=h[i]; af=p1
            elif c[i]<dLL[i] and atr0[i]>0:
                pos=-1; entry=c[i]; ebar=i; peak=c[i]; mfe=0.0; aentry=atr0[i]
                trail=entry+p1*aentry; psar=h[i]; ep=l[i]; af=p1
            continue
        if pos==1:
            if h[i]>peak: peak=h[i]
            fav=(h[i]-entry)/pip
        else:
            if l[i]<peak: peak=l[i]
            fav=(entry-l[i])/pip
        if fav>mfe: mfe=fav
        ex=0.0; hit=False
        if scheme==0:      # fixed SL + TP (non-trailing reference)
            if pos==1:
                slv=entry-p1*aentry; tpv=entry+p2*aentry
                if l[i]<=slv: ex=slv; hit=True
                elif h[i]>=tpv: ex=tpv; hit=True
            else:
                slv=entry+p1*aentry; tpv=entry-p2*aentry
                if h[i]>=slv: ex=slv; hit=True
                elif l[i]<=tpv: ex=tpv; hit=True
        elif scheme==1:    # PSAR, per-trade (p1=af0, p2=afmax)
            psar=psar+af*(ep-psar)
            if pos==1:
                if psar>l[i-1]: psar=l[i-1]
                if l[i]<=psar: ex=psar; hit=True
                elif h[i]>ep: ep=h[i]; af=min(af+p1,p2)
            else:
                if psar<h[i-1]: psar=h[i-1]
                if h[i]>=psar: ex=psar; hit=True
                elif l[i]<ep: ep=l[i]; af=min(af+p1,p2)
        elif scheme==2:    # chandelier (precomputed, always on correct side)
            s=chl[i] if pos==1 else chs[i]
            if pos==1 and l[i]<=s: ex=s; hit=True
            elif pos==-1 and h[i]>=s: ex=s; hit=True
        elif scheme==3:    # n-bar swing
            s=nbl[i] if pos==1 else nbh[i]
            if pos==1 and l[i]<=s: ex=s; hit=True
            elif pos==-1 and h[i]>=s: ex=s; hit=True
        elif scheme==4:    # ATR ratchet (close - m*ATR, favorable-only)
            if pos==1:
                t=c[i-1]-p1*atr0[i]
                if t>trail: trail=t
                if l[i]<=trail: ex=trail; hit=True
            else:
                t=c[i-1]+p1*atr0[i]
                if t<trail: trail=t
                if h[i]>=trail: ex=trail; hit=True
        elif scheme==5:    # SuperTrend, per-trade (ratcheting hl2 ± m*ATR band)
            hl2=(h[i]+l[i])/2.0
            if pos==1:
                t=hl2-p1*atr0[i]
                if t>trail: trail=t
                if c[i]<trail: ex=trail; hit=True
            else:
                t=hl2+p1*atr0[i]
                if t<trail: trail=t
                if c[i]>trail: ex=trail; hit=True
        elif scheme==6:    # giveback: lock p1*MFE once MFE>=p2 arm
            if mfe>=p2:
                lock=p1*mfe
                if pos==1:
                    t=entry+lock*pip
                    if l[i]<=t: ex=t; hit=True
                else:
                    t=entry-lock*pip
                    if h[i]>=t: ex=t; hit=True
        elif scheme==7:    # fixed-pip trail from peak
            if pos==1:
                t=peak-p1*pip
                if l[i]<=t: ex=t; hit=True
            else:
                t=peak+p1*pip
                if h[i]>=t: ex=t; hit=True
        if hit:
            P[k]=(ex-entry)/pip*pos - sp; M[k]=mfe; E[k]=ebar; k+=1; pos=0
    if pos!=0:             # close residual at last close (no survivorship)
        P[k]=(c[n-1]-entry)/pip*pos - sp; M[k]=mfe; E[k]=ebar; k+=1
    return P[:k], M[:k], E[:k]


SCHEMES={
 0:("fixedSLTP",[(1.0,3.0),(1.5,4.0),(2.0,6.0)]),
 1:("psar",[(0.02,0.20),(0.01,0.10)]),               # p1=af0,p2=afmax (handled via series; param picks series)
 2:("chandelier",[(2.0,0),(3.0,0),(4.0,0)]),         # p1=m (series picks via Nc fixed)
 3:("nbar",[(0,0)]),                                  # uses LL(Nb)/HH(Nb) series
 4:("atr_ratchet",[(2.0,0),(3.0,0),(4.0,0)]),
 5:("supertrend",[(2.0,0),(3.0,0)]),
 6:("giveback",[(0.5,5.0),(0.6,3.0),(0.7,8.0)]),
 7:("fixedpip",[(20.0,0),(40.0,0),(80.0,0)]),
}


def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--tf",default="H4"); a=ap.parse_args()
    rule=TF_RULE[a.tf]; con=duckdb.connect()
    # warm numba
    z=np.zeros(60)
    sim(z,z,z,z,z,z,z+1,z,z,z,z,0.01,2.0,1,0.02,0.2,30)
    NC=22; NB=10  # chandelier/swing lookbacks (fixed)
    PRE={}
    for pair in PAIRS:
        r=load(con,pair,rule);
        if len(r)<200: continue
        o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values
        a14=atr(h,l,c,14)
        dHH=pd.Series(h).rolling(DONCH).max().shift(1).values; dLL=pd.Series(l).rolling(DONCH).min().shift(1).values
        chl=pd.Series(h).rolling(NC).max().shift(1).values; chs=pd.Series(l).rolling(NC).min().shift(1).values
        PRE[pair]=dict(o=o,h=h,l=l,c=c,a=np.nan_to_num(a14),dHH=np.nan_to_num(dHH,nan=1e18),
            dLL=np.nan_to_num(dLL,nan=-1e18),
            chl=np.nan_to_num(chl - 3*a14, nan=-1e18), chs=np.nan_to_num(chs + 3*a14, nan=1e18),
            nbl=np.nan_to_num(pd.Series(l).rolling(NB).min().shift(1).values, nan=-1e18),
            nbh=np.nan_to_num(pd.Series(h).rolling(NB).max().shift(1).values, nan=1e18),
            n=len(c), is_end=int(len(c)*IS_FRAC))
    con.close()

    def portfolio(scheme,p1,p2):
        isn=[];oosn=[];mfeall=[];capall=[]
        for pair,(pip,sp) in PAIRS.items():
            if pair not in PRE: continue
            d=PRE[pair]
            # rebuild chandelier with this m if scheme==2
            chl=d['chl']; chs=d['chs']
            if scheme==2:
                hh=pd.Series(d['h']).rolling(NC).max().shift(1).values; ll=pd.Series(d['l']).rolling(NC).min().shift(1).values
                chl=np.nan_to_num(hh-p1*d['a'],nan=-1e18); chs=np.nan_to_num(ll+p1*d['a'],nan=1e18)
            P,M,E=sim(d['o'],d['h'],d['l'],d['c'],d['dHH'],d['dLL'],d['a'],chl,chs,d['nbl'],d['nbh'],
                      pip,sp,scheme,p1,p2,d['is_end'])
            ism=E<d['is_end']
            isn.append(P[ism]);oosn.append(P[~ism]); mfeall.append(M)
            cap=P/np.maximum(M,1e-9); capall.append(cap[~ism])
        isn=np.concatenate(isn);oosn=np.concatenate(oosn);cap=np.concatenate(capall)
        # crude p/d: total pips / (#bars*tf/day). use trade count proxy -> per 100 trades
        def stat(x):
            if len(x)==0: return (0,0,0,0)
            cum=x.cumsum(); dd=float((cum-np.maximum.accumulate(cum)).min())
            return (x.mean(), (x>0).mean()*100, dd, len(x))
        ie=stat(isn); oe=stat(oosn)
        return ie,oe,np.nanmedian(cap)

    print(f"TRAILING BAKE-OFF — {a.tf}, Donchian({DONCH}) breakout entry, 12 pairs, net spread, IS/OOS.")
    print(f"  {'scheme':<12}{'param':>12}{'IS_exp':>9}{'OOS_exp':>9}{'OOS_WR':>8}{'OOS_DD':>9}{'OOS_n':>7}{'capR':>7}")
    best={}
    for sc,(name,params) in SCHEMES.items():
        rows=[]
        for (p1,p2) in params:
            ie,oe,capr=portfolio(sc,p1,p2)
            rows.append(((p1,p2),ie,oe,capr))
        # pick IS-best by IS expectancy
        rows.sort(key=lambda r:-r[1][0])
        (p1,p2),ie,oe,capr=rows[0]
        best[name]=(oe[0],oe[2])
        print(f"  {name:<12}{f'{p1}/{p2}':>12}{ie[0]:>+9.2f}{oe[0]:>+9.2f}{oe[1]:>7.0f}%{oe[2]:>9.0f}{oe[3]:>7}{capr:>7.2f}")
    print("\n  IS-best param shown; ranked by OOS_exp. capR = median OOS capture (realized/MFE).")
    print("  Winner = highest OOS_exp with acceptable DD; non-accelerating structure/vol schemes expected to beat PSAR.")


if __name__=="__main__":
    main()
