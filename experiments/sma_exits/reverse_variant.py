#!/usr/bin/env python3
"""
Are the opposing signals that caused the flips actually GOOD — worth a DELIBERATE
stop-and-reverse? The flip bug netted positions at a loss by accident; this asks the
real question: when an opposing joint-alignment fires mid-trade, is closing and
reversing better than holding to TP/PSAR/fence? Backtest two variants per pair:
  HOLD (intended): enter only when flat; manage to TP/PSAR/200p fence.
  REVERSE: same, but if an opposing joint-alignment fires while in a position,
           close at market and open the opposite immediately.
~9.6mo S5, IS/OOS split, net of spread. Reuses stack010 signal prep.
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
from fence_timestop_sweep import prep   # signal prep (one resample per pair)

FENCE = 200.0


@nb.njit(cache=True)
def kern_rev(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, pip, tp_pips, use_psar, act, fence, reverse):
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; armed=False
    pnl=np.empty(2*n); ent=np.empty(2*n,np.int64); rsn=np.empty(2*n,np.int64); nt=0
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
            # REVERSE on opposing joint-alignment
            opp = (pos==1 and t1s[i]==1 and t2s[i]==1) or (pos==-1 and t1l[i]==1 and t2l[i]==1)
            if r<0 and reverse==1 and opp:
                ex=c[i]; r=4
            if r>=0:
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; nt+=1
                if r==4:                       # immediately open the opposite
                    pos=-pos; entry=c[i]; ebar=i; mfe=0.0; armed=False
                else:
                    pos=0
    return pnl[:nt], ent[:nt], rsn[:nt]


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kern_rev(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,200.,1)
    P={pr:prep(pr) for pr in CFG}
    print("Stop-and-reverse vs hold (intended). ~9.6mo S5, IS/OOS, net spread.")
    print(f"  {'pair':<9}{'variant':>9}{'trades':>8}{'IS_pd':>8}{'OOS_pd':>8}{'expect':>9}{'WR%':>6}{'n_rev':>7}{'rev_pnl':>9}")
    for pr,d in P.items():
        for rev,lab in [(0,'HOLD'),(1,'REVERSE')]:
            p,e,r=kern_rev(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                           d['pip'],d['tp'],d['use_psar'],d['act'],FENCE,rev)
            net=p-d['sp']; ism=e<d['is_end']
            isd=d['is_end']/d['n']*d['days']; oosd=d['days']-isd
            # for REVERSE: isolate the pnl of trades that were ENTERED via a reverse
            # (a reverse entry's exit is the NEXT trade). Approx: pnl on trades whose
            # entry bar coincides with a prior reverse exit. Simpler: report count + mean of reverse-EXIT legs.
            nrev=int((r==4).sum())
            revpnl = net[r==4].mean() if nrev>0 else 0.0   # pnl of the leg we reversed OUT of
            print(f"  {pr:<9}{lab:>9}{len(p):>8}{net[ism].sum()/max(isd,1):>8.2f}{net[~ism].sum()/max(oosd,1):>8.2f}"
                  f"{net.mean():>+9.2f}{(net>0).mean()*100:>5.0f}%{nrev:>7}{revpnl:>+9.2f}")
    print("\n  If REVERSE expectancy/OOS_pd beats HOLD, the opposing mid-trade signal is worth a deliberate reverse.")
    print("  rev_pnl = mean pnl of the leg we reversed OUT of (negative = we were losing when the opposing signal fired).")


if __name__=="__main__":
    main()
