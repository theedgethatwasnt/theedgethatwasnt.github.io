#!/usr/bin/env python3
"""
Less-ruinous catastrophe exit for 010. Sweep the fence width (200p current -> tighter)
and an optional time-stop, on the ~9.6mo window that actually produces fence hits, to
see if the -200p tail shrinks faster than the edge erodes. IS/OOS split so a winner
must hold on both (window-sensitivity is the recurring trap). Signals prepped once per
pair; kernel re-run per (fence, time-stop).
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

S5 = H.S5_DIR; MAX_ROWS = 5_000_000
FENCES = [200.0, 120.0, 80.0, 50.0]
TIMESTOPS_H = [0.0, 24.0, 8.0]   # 0 = none


@nb.njit(cache=True)
def kern2(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, pip, tp_pips, use_psar, act, fence, max_hold):
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; armed=False
    pnl=np.empty(n); ent=np.empty(n,np.int64); rsn=np.empty(n,np.int64); nt=0
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1: pos=1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
            if t1s[i]==1 and t2s[i]==1: pos=-1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
        if pos!=0:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            if fav>mfe: mfe=fav
            if use_psar and (not armed) and mfe>=act: armed=True
            ex=0.0; r=-1
            fc=entry - pos*fence*pip
            if pos==1 and l[i]<=fc: ex=fc; r=2
            elif pos==-1 and h[i]>=fc: ex=fc; r=2
            if r<0 and tp_pips>0:
                tp=entry+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: ex=tp; r=0
                elif pos==-1 and l[i]<=tp: ex=tp; r=0
            if r<0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos==1 and c[i]<sar_b[i]: ex=c[i]; r=1
                elif pos==-1 and c[i]>sar_b[i]: ex=c[i]; r=1
            if r<0 and max_hold>0 and (i-ebar)>=max_hold:   # time-stop at current close
                ex=c[i]; r=3
            if r>=0:
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; nt+=1; pos=0
    return pnl[:nt], ent[:nt], rsn[:nt]


def prep(pair):
    pip,t1m,t2m,(ss,sm,sl),tp,use_psar,af,act,_ = CFG[pair]
    df=H.fast_tail_read(S5/f"{pair}_S5_BA.parquet", MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
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
    sar=project_to_m5(prev,tf1['timestamp'].to_numpy(),psar_series(tf1['high'].to_numpy(float),tf1['low'].to_numpy(float),af,0.10)) if use_psar else np.full(n,np.nan)
    return dict(o=o,h=h,l=l,c=c,t1l=t1l,t1s=t1s,t2l=t2l,t2s=t2s,sar=sar,pip=pip,tp=tp,
                use_psar=use_psar,act=act,n=n,is_end=int(n*IS_FRAC),sp=PAIRS[pair][1]*SPREAD_FRAC,
                days=n*(5/60)/(60*24))


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kern2(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,200.,0)
    P={pr:prep(pr) for pr in CFG}
    bar_per_h=int(60*60/5)  # S5 bars per hour
    print("010 catastrophe-exit sweep (~9.6mo). per-config PORTFOLIO net of spread; IS/OOS split.")
    print(f"  {'fence':>6}{'tstop':>7}{'trades':>8}{'IS_pd':>8}{'OOS_pd':>8}{'expect':>9}{'tail%(<=-fence+5)':>18}{'worst':>8}{'OOS_DD':>9}")
    base=None
    for fence in FENCES:
        for tsh in TIMESTOPS_H:
            mh=int(tsh*bar_per_h)
            allnet=[]; allent=[]; alln=0; isnet=0.0; oosnet=0.0; isd=0.0; oosd=0.0; ntail=0; ntot=0; worst=0.0; oos_all=[]
            for pr,d in P.items():
                p,e,r=kern2(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                            d['pip'],d['tp'],d['use_psar'],d['act'],fence,mh)
                net=p-d['sp']; ism=e<d['is_end']
                isnet+=net[ism].sum(); oosnet+=net[~ism].sum()
                isd+=d['is_end']/d['n']*d['days']; oosd+=d['days']-d['is_end']/d['n']*d['days']
                ntail+=int((r==2).sum()); ntot+=len(p); worst=min(worst, net.min() if len(net) else 0)
                allnet.append(net); oos_all.append(net[~ism])
            allnet=np.concatenate(allnet); oa=np.concatenate(oos_all)
            cum=oa.cumsum(); dd=float((cum-np.maximum.accumulate(cum)).min()) if len(oa) else 0
            row=(fence, tsh, ntot, isnet/max(isd,1), oosnet/max(oosd,1), allnet.mean(), ntail/max(ntot,1)*100, worst, dd)
            if base is None: base=row
            ts_lbl=f"{tsh:.0f}h" if tsh>0 else "none"
            print(f"  {fence:>6.0f}{ts_lbl:>7}{row[2]:>8}{row[3]:>8.2f}{row[4]:>8.2f}{row[5]:>+9.2f}{row[6]:>17.1f}%{row[7]:>8.0f}{row[8]:>9.0f}")
    print(f"\n  baseline = fence 200 / no time-stop. Goal: a config with worst-trade & tail% DOWN and expectancy/OOS_pd NOT worse.")


if __name__=="__main__":
    main()
