#!/usr/bin/env python3
"""
Would a SECOND account profitably take the opposing mid-trade signals?
Different from reverse_variant: there we CLOSED the original (ate its loss) and the new
leg inherited mid-trade timing. Here the original rides to its designed exit untouched;
a second account opens the opposing joint-alignment as its OWN independent trade,
managed by the pair's normal exit (TP/PSAR/200p fence). So we measure the standalone
P&L of the signals the intended (HOLD) strategy IGNORES because it was already in a
position. ~9.6mo S5, net of spread.
"""
import sys
from pathlib import Path
import numpy as np
import numba as nb
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG
from fence_timestop_sweep import prep
FENCE=200.0


@nb.njit(cache=True)
def hold_timeline(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, pip, tp_pips, use_psar, act, fence):
    """Per-bar position direction under the intended HOLD strategy (1 pos/pair)."""
    n=len(o); pos=0; entry=0.0; mfe=0.0; armed=False
    dirarr=np.zeros(n,np.int64)
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1: pos=1; entry=o[i]; mfe=0.0; armed=False
            elif t1s[i]==1 and t2s[i]==1: pos=-1; entry=o[i]; mfe=0.0; armed=False
        else:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            if fav>mfe: mfe=fav
            if use_psar and (not armed) and mfe>=act: armed=True
            r=-1; fc=entry-pos*fence*pip
            if pos==1 and l[i]<=fc: r=2
            elif pos==-1 and h[i]>=fc: r=2
            if r<0 and tp_pips>0:
                tp=entry+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: r=0
                elif pos==-1 and l[i]<=tp: r=0
            if r<0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos==1 and c[i]<sar_b[i]: r=1
                elif pos==-1 and c[i]>sar_b[i]: r=1
            if r>=0: pos=0
        dirarr[i]=pos
    return dirarr


@nb.njit(cache=True)
def sim_independent(start, d, o,h,l,c, sar_b, pip, tp_pips, use_psar, act, fence):
    """Walk one independent trade entered at bar `start` dir d to its exit; return pnl pips."""
    entry=o[start]; mfe=0.0; armed=False; n=len(o)
    for i in range(start+1,n):
        fav=(h[i]-entry)/pip if d==1 else (entry-l[i])/pip
        if fav>mfe: mfe=fav
        if use_psar and (not armed) and mfe>=act: armed=True
        fc=entry-d*fence*pip
        if d==1 and l[i]<=fc: return (fc-entry)/pip*d
        if d==-1 and h[i]>=fc: return (fc-entry)/pip*d
        if tp_pips>0:
            tp=entry+d*tp_pips*pip
            if d==1 and h[i]>=tp: return (tp-entry)/pip*d
            if d==-1 and l[i]<=tp: return (tp-entry)/pip*d
        if use_psar and armed and not np.isnan(sar_b[i]):
            if d==1 and c[i]<sar_b[i]: return (c[i]-entry)/pip*d
            if d==-1 and c[i]>sar_b[i]: return (c[i]-entry)/pip*d
    return (c[n-1]-entry)/pip*d


@nb.njit(cache=True)
def run_signals(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, dirarr, pip, tp_pips, use_psar, act, fence):
    """Every joint-novelty -> independent trade. Returns pnl, dir, opposite-to-HOLD flag."""
    n=len(o); pnl=np.empty(n); sig=np.empty(n,np.int64); opp=np.empty(n,np.int64); k=0
    for i in range(1,n):
        d=0
        if t1l[i]==1 and t2l[i]==1: d=1
        elif t1s[i]==1 and t2s[i]==1: d=-1
        if d==0: continue
        pnl[k]=sim_independent(i,d,o,h,l,c,sar_b,pip,tp_pips,use_psar,act,fence)
        sig[k]=d; opp[k]=1 if dirarr[i]==-d else 0   # HOLD held the opposite here
        k+=1
    return pnl[:k], sig[:k], opp[:k]


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    hold_timeline(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,200.)
    sim_independent(1,1,_c+1,_c+1,_c+1,_c+1,_c,.01,20.,True,20.,200.)
    P={pr:prep(pr) for pr in CFG}
    print("Second-account test: independent trade per joint-alignment, pair's normal exit. ~9.6mo S5, net spread.")
    print(f"  {'pair':<9}{'all_sig':>8}{'all_exp':>9}|{'OPPOSING(2nd-acct)':>20}{'n':>6}{'exp':>8}{'WR%':>6}{'total':>9}")
    g_opp=[];
    for pr,d in P.items():
        dirarr=hold_timeline(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],d['pip'],d['tp'],d['use_psar'],d['act'],FENCE)
        pnl,sig,opp=run_signals(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],dirarr,d['pip'],d['tp'],d['use_psar'],d['act'],FENCE)
        net=pnl-d['sp']; om=opp==1
        oexp=net[om].mean() if om.sum() else 0.0
        g_opp.append(net[om])
        print(f"  {pr:<9}{len(net):>8}{net.mean():>+9.2f}|{'':>20}{int(om.sum()):>6}{oexp:>+8.2f}{(net[om]>0).mean()*100 if om.sum() else 0:>5.0f}%{net[om].sum():>+9.0f}")
    allo=np.concatenate(g_opp)
    print(f"\n  ALL opposing 'second-account' trades: n={len(allo)}  expectancy={allo.mean():+.2f}p  WR={(allo>0).mean()*100:.0f}%  total={allo.sum():+.0f}p")
    print(f"  >>> If expectancy > 0, a second account taking the ignored opposing signals would have been profitable.")


if __name__=="__main__":
    main()
