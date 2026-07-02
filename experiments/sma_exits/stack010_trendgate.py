"""
stack010_trendgate.py — does a higher-TF (daily / H4) TREND GATE improve 010?
Same engine + configs + 200p fence as stack010_equity, but only allow LONG when the gate TF is in an
uptrend (price > gate-SMA) and SHORT when downtrend. Compares: NONE (baseline) vs DAILY vs H4 gate.
Reports expectancy, WR, fence hits/rate, total net, max drawdown — and crucially whether the gate
KILLS the counter-trend trades that become 200p fence disasters (the negative-skew tail).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, numba as nb
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, SPREAD_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG, S5, MAX_ROWS

@nb.njit(cache=True)
def kern_g(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, tru,trd, pip, tp_pips, use_psar, act, fence):
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; armed=False
    pnl=np.empty(n); ent=np.empty(n,np.int64); rsn=np.empty(n,np.int64); nt=0
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1 and tru[i]==1: pos=1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
            if t1s[i]==1 and t2s[i]==1 and trd[i]==1: pos=-1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
        if pos!=0:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            if fav>mfe: mfe=fav
            if use_psar and (not armed) and mfe>=act: armed=True
            ex=0.0; r=-1; fc=entry - pos*fence*pip
            if pos==1 and l[i]<=fc: ex=fc; r=2
            elif pos==-1 and h[i]>=fc: ex=fc; r=2
            if r<0 and tp_pips>0:
                tp=entry+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: ex=tp; r=0
                elif pos==-1 and l[i]<=tp: ex=tp; r=0
            if r<0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos==1 and c[i]<sar_b[i]: ex=c[i]; r=1
                elif pos==-1 and c[i]>sar_b[i]: ex=c[i]; r=1
            if r>=0:
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; nt+=1; pos=0
    return pnl[:nt], ent[:nt], rsn[:nt]

def gate_arrays(df, prev, n, mode, gate_min, gate_sma):
    if mode=="none": one=np.ones(n,np.int8); return one,one
    g=H.resample_minutes(df, gate_min, 5/60)
    gc=g['close'].to_numpy(float); gts=g['timestamp'].to_numpy(); gs=sma(gc,gate_sma)
    up=(gc>gs).astype(float); dn=(gc<gs).astype(float)
    tru=(project_to_m5(prev,gts,up)>0.5).astype(np.int8)
    trd=(project_to_m5(prev,gts,dn)>0.5).astype(np.int8)
    return tru,trd

def run_pair(pair, mode, gate_min, gate_sma):
    pip,t1m,t2m,(ss,sm,sl),tp,use_psar,af,act,fence=CFG[pair]
    df=H.fast_tail_read(S5/f"{pair}_S5_BA.parquet",MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
    o=df['open'].to_numpy(float);h=df['high'].to_numpy(float);l=df['low'].to_numpy(float);c=df['close'].to_numpy(float)
    ts=df['timestamp'].to_numpy();n=len(df);prev=np.empty_like(ts);prev[0]=ts[0];prev[1:]=ts[:-1]
    tf1=H.resample_minutes(df,t1m,5/60);tf2=H.resample_minutes(df,t2m,5/60)
    def nov(cc,tt):
        a=sma(cc,ss);b=sma(cc,sm);d=sma(cc,sl)
        return (project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,1))).astype(np.int8),
                project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,0))).astype(np.int8))
    t1l,t1s=nov(tf1['close'].to_numpy(float),tf1['timestamp'].to_numpy())
    t2l,t2s=nov(tf2['close'].to_numpy(float),tf2['timestamp'].to_numpy())
    sar=project_to_m5(prev,tf1['timestamp'].to_numpy(),psar_series(tf1['high'].to_numpy(float),tf1['low'].to_numpy(float),af,0.10)) if use_psar else np.full(n,np.nan)
    tru,trd=gate_arrays(df,prev,n,mode,gate_min,gate_sma)
    p,e,r=kern_g(o,h,l,c,t1l,t1s,t2l,t2s,sar,tru,trd,pip,tp,use_psar,act,fence)
    net=p-PAIRS[pair][1]*SPREAD_FRAC
    return pd.DataFrame({'pair':pair,'exit_ts':ts[e],'net':net,'reason':r})

def main():
    _c=np.zeros(50);_s=np.zeros(50,np.int8)
    H.tf_signal(_c,_c,_c,_c,1);psar_series(_c+1,_c+1,.02,.1);kern_g(_c,_c,_c,_c,_s,_s,_s,_s,_c,_s,_s,.01,20.,True,20.,200.)
    scenarios=[("NONE (baseline)","none",0,0),("DAILY SMA20","daily",1440,20),("H4 SMA50","h4",240,50)]
    print(f"  {'gate':>16} {'trades':>7} {'expect':>8} {'WR%':>5} {'fence':>6} {'fence%':>7} {'net':>9} {'maxDD':>8} {'p/day':>7}")
    for nm,mode,gm,gs in scenarios:
        parts=[run_pair(p,mode,gm,gs) for p in CFG]
        allt=pd.concat(parts).sort_values('exit_ts').reset_index(drop=True)
        eq=allt['net'].cumsum().values; dd=(eq-np.maximum.accumulate(eq)).min()
        nf=int((allt.reason==2).sum())
        span=max((pd.Timestamp(allt.exit_ts.iloc[-1])-pd.Timestamp(allt.exit_ts.iloc[0])).days,1)
        print(f"  {nm:>16} {len(allt):>7} {allt.net.mean():>+8.2f} {(allt.net>0).mean()*100:>4.0f}% {nf:>6} {nf/len(allt)*100:>6.2f}% {eq[-1]:>+9.0f} {dd:>+8.0f} {eq[-1]/span:>+7.1f}")

if __name__=="__main__": main()
