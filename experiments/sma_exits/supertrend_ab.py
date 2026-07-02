#!/usr/bin/env python3
"""
010 A/B — replace the PSAR trailing exit with SuperTrend, on the ACTUAL 010 SMA-stack entry.
Bake-off found SuperTrend (ratcheting ATR band) beats PSAR on a generic breakout; this tests it
on 010's real signal + exits. Only the TRAIL changes; TP, 200p fence, and arm logic stay identical.
PSAR pairs = EUR_USD, GBP_USD. Exit at bar CLOSE when close crosses the trail (matches live).
~9.6mo S5, net spread, IS/OOS. SuperTrend swept over m and arm (0=from entry, 20=current PSAR arm).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG

S5=H.S5_DIR; MAX_ROWS=5_000_000
ABPAIRS=["EUR_USD","GBP_USD"]    # the PSAR pairs in 010


def atr_wilder(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def supertrend(h,l,c,a,m):
    hl2=(h+l)/2; ub=hl2+m*a; lb=hl2-m*a; n=len(c)
    fu=ub.copy(); fl=lb.copy(); dir_=np.ones(n)
    for i in range(1,n):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        if c[i]>fu[i-1]: dir_[i]=1
        elif c[i]<fl[i-1]: dir_[i]=-1
        else: dir_[i]=dir_[i-1]
    return np.where(dir_==1,fl,fu)


@nb.njit(cache=True)
def kern_ab(o,h,l,c, t1l,t1s,t2l,t2s, trail, pip, tp_pips, act, fence):
    """010 stack exit but the post-arm trail is the projected `trail` line (PSAR or SuperTrend);
    exit at bar close when close crosses it. TP + 200p fence identical to live."""
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; armed=False
    pnl=np.empty(n); ent=np.empty(n,np.int64); rsn=np.empty(n,np.int64); nt=0
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1: pos=1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
            if t1s[i]==1 and t2s[i]==1: pos=-1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
        if pos!=0:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            if fav>mfe: mfe=fav
            if (not armed) and mfe>=act: armed=True
            ex=0.0; r=-1
            fc=entry-pos*fence*pip
            if pos==1 and l[i]<=fc: ex=fc; r=2
            elif pos==-1 and h[i]>=fc: ex=fc; r=2
            if r<0 and tp_pips>0:
                tp=entry+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: ex=tp; r=0
                elif pos==-1 and l[i]<=tp: ex=tp; r=0
            if r<0 and armed and not np.isnan(trail[i]):
                if pos==1 and c[i]<trail[i]: ex=c[i]; r=1
                elif pos==-1 and c[i]>trail[i]: ex=c[i]; r=1
            if r>=0:
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; nt+=1; pos=0
    return pnl[:nt], ent[:nt], rsn[:nt]


def prep2(pair):
    pip,t1m,t2m,(ss,sm,sl),tp,use_psar,af,act,fence = CFG[pair]
    df=H.fast_tail_read(S5/f"{pair}_S5_BA.parquet",MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
    o=df['open'].to_numpy(float); h=df['high'].to_numpy(float); l=df['low'].to_numpy(float); c=df['close'].to_numpy(float)
    ts=df['timestamp'].to_numpy(); n=len(df)
    prev=np.empty_like(ts); prev[0]=ts[0]; prev[1:]=ts[:-1]
    tf1=H.resample_minutes(df,t1m,5/60); tf2=H.resample_minutes(df,t2m,5/60)
    def nov(cc,tt):
        a=sma(cc,ss); b=sma(cc,sm); d=sma(cc,sl)
        return (project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,1))).astype(np.int8),
                project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,0))).astype(np.int8))
    t1l,t1s=nov(tf1['close'].to_numpy(float),tf1['timestamp'].to_numpy())
    t2l,t2s=nov(tf2['close'].to_numpy(float),tf2['timestamp'].to_numpy())
    h1=tf1['high'].to_numpy(float); l1=tf1['low'].to_numpy(float); c1=tf1['close'].to_numpy(float); t1=tf1['timestamp'].to_numpy()
    psar=project_to_m5(prev,t1,psar_series(h1,l1,af,0.10))
    a1=atr_wilder(h1,l1,c1,14)
    st={m:project_to_m5(prev,t1,supertrend(h1,l1,c1,a1,m)) for m in (2.0,3.0)}
    return dict(o=o,h=h,l=l,c=c,t1l=t1l,t1s=t1s,t2l=t2l,t2s=t2s,psar=psar,st=st,
                pip=pip,tp=tp,act=act,fence=fence,n=n,is_end=int(n*IS_FRAC),
                sp=PAIRS[pair][1]*SPREAD_FRAC, days=n*(5/60)/(60*24))


def stats(net,e,d):
    ism=e<d['is_end']; isd=d['is_end']/d['n']*d['days']; oosd=d['days']-isd
    def dd(x):
        if not len(x): return 0.0
        cum=x.cumsum(); return float((cum-np.maximum.accumulate(cum)).min())
    return (net[ism].sum()/max(isd,1), net[~ism].sum()/max(oosd,1), dd(net[~ism]),
            net.mean(), (net>0).mean()*100, len(net))


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kern_ab(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,20.,200.)
    print("010 PSAR→SuperTrend A/B — real SMA-stack entry, only the trail changes. ~9.6mo S5, net spread.")
    print(f"  {'pair':<9}{'exit':<18}{'IS_pd':>8}{'OOS_pd':>8}{'OOS_DD':>9}{'exp':>8}{'WR%':>6}{'n':>6}")
    for pr in ABPAIRS:
        d=prep2(pr)
        # baseline: current PSAR (arm@act)
        p,e,r=kern_ab(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['psar'],
                      d['pip'],d['tp'],d['act'],d['fence'])
        s=stats(p-d['sp'],e,d)
        print(f"  {pr:<9}{'PSAR (current)':<18}{s[0]:>8.2f}{s[1]:>8.2f}{s[2]:>9.0f}{s[3]:>+8.2f}{s[4]:>5.0f}%{s[5]:>6}")
        for m in (2.0,3.0):
            for arm in (d['act'],0.0):
                p,e,r=kern_ab(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['st'][m],
                              d['pip'],d['tp'],arm,d['fence'])
                s=stats(p-d['sp'],e,d)
                lab=f"ST m{m:.0f} arm{arm:.0f}"
                print(f"  {pr:<9}{lab:<18}{s[0]:>8.2f}{s[1]:>8.2f}{s[2]:>9.0f}{s[3]:>+8.2f}{s[4]:>5.0f}%{s[5]:>6}")
        print()
    print("  Winner per pair = higher OOS_pd with DD no worse than PSAR. Deploy only if it beats PSAR IS+OOS.")


if __name__=="__main__":
    main()
