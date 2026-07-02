#!/usr/bin/env python3
"""
Reconcile backtest vs live entry frequency. Hypothesis: the backtest projects the
TF-level novelty (newly-aligned) onto EVERY base bar within that TF bar, so after a
quick TP the kernel RE-ENTERS on the still-active signal — multiple trades per
alignment that live never makes (live fires once, on the TF-bar-emit tick). Test:
run each pair with (A) the current projected-novelty entry vs (B) entry only on the
RISING EDGE of the joint signal (once per alignment). Compare trade count + fence rate.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series, psar_kernel, tp_kernel
from stack010_equity import CFG, kern

S5 = H.S5_DIR; MAX_ROWS = 3_000_000


def rising_edge(a):
    e = np.zeros(len(a), np.int8); e[1:] = ((a[1:] == 1) & (a[:-1] == 0)).astype(np.int8); return e


def run(pair, edge):
    pip,t1m,t2m,(ss,sm,sl),tp,use_psar,af,act,fence = CFG[pair]
    df=H.fast_tail_read(S5/f"{pair}_S5_BA.parquet", MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
    o=df['open'].to_numpy(float); h=df['high'].to_numpy(float); l=df['low'].to_numpy(float); c=df['close'].to_numpy(float)
    ts=df['timestamp'].to_numpy(); n=len(df); days=n*(5/60)/(60*24)
    prev=np.empty_like(ts); prev[0]=ts[0]; prev[1:]=ts[:-1]
    tf1=H.resample_minutes(df,t1m,5/60); tf2=H.resample_minutes(df,t2m,5/60)
    def nov(cc,tt):
        a=sma(cc,ss); b=sma(cc,sm); d=sma(cc,sl)
        return (project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,1))).astype(np.int8),
                project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,0))).astype(np.int8))
    t1l,t1s=nov(tf1['close'].to_numpy(float),tf1['timestamp'].to_numpy())
    t2l,t2s=nov(tf2['close'].to_numpy(float),tf2['timestamp'].to_numpy())
    if edge:
        # enter only on the rising edge of the JOINT signal (once per alignment onset)
        jl=((t1l==1)&(t2l==1)).astype(np.int8); js=((t1s==1)&(t2s==1)).astype(np.int8)
        jl=rising_edge(jl); js=rising_edge(js)
        t1l=jl; t2l=jl; t1s=js; t2s=js   # kernel needs both arms 1; mirror the joint edge
    if use_psar:
        sar=project_to_m5(prev,tf1['timestamp'].to_numpy(),psar_series(tf1['high'].to_numpy(float),tf1['low'].to_numpy(float),af,0.10))
    else:
        sar=np.full(n,np.nan)
    p,e,r=kern(o,h,l,c,t1l,t1s,t2l,t2s,sar,pip,tp,use_psar,act,fence)
    sp=PAIRS[pair][1]*SPREAD_FRAC; net=p-sp
    nf=int((r==2).sum())
    return len(p), len(p)/days, net.mean(), (net>0).mean()*100, nf, nf/max(len(p),1)*100, net.sum()/days


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kern(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,200.)
    print("Backtest entry reconciliation: projected-novelty (current) vs rising-edge (once/alignment)")
    print(f"  {'pair':<9}{'entry':>14}{'trades':>8}{'tr/day':>8}{'expect':>9}{'WR%':>6}{'fence':>7}{'fence%':>8}{'p/day':>8}")
    for pr in CFG:
        for edge,lab in [(False,'projected'),(True,'rising-edge')]:
            n,tpd,exp,wr,nf,fr,ppd=run(pr,edge)
            print(f"  {pr:<9}{lab:>14}{n:>8}{tpd:>8.2f}{exp:>+9.2f}{wr:>5.0f}%{nf:>7}{fr:>7.1f}%{ppd:>+8.2f}")


if __name__=="__main__":
    main()
