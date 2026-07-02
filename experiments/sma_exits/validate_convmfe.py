#!/usr/bin/env python3
"""
Validate convMFE γ1 H5 as 010's exit vs the current flat-200, before deploy.
Entries are IDENTICAL (same stack signal) — only the exit differs — so trades pair by
entry bar. Per pair: IS/OOS p/d, MaxDD, worst, expectancy for both; 4-chunk walk-forward;
and a paired-difference MC (bootstrap) on net_convMFE − net_flat200 to test the
improvement is signal not noise. ~9.6mo S5, net spread.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG
from fence_timestop_sweep import prep
from converging_fence_frontier import kernx, A_ARM, BAR_PER_H

RNG = np.random.default_rng(7)


def trades_for(d, mech, F0, Hd, gamma, Tbars):
    p,e,r,mae = kernx(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                      d['pip'],d['tp'],d['use_psar'],d['act'], mech, float(F0), float(Hd), A_ARM, gamma, float(Tbars))
    return p-d['sp'], e, r


def pd_dd(net, ts):
    if not len(net): return 0.0, 0.0
    order=np.argsort(ts); nn=net[order]
    days=max((pd.Timestamp(ts.max())-pd.Timestamp(ts.min())).total_seconds()/86400,1)
    cum=nn.cumsum(); dd=float((cum-np.maximum.accumulate(cum)).min())
    return nn.sum()/days, dd


def main():
    _c=np.zeros(60); _s=np.zeros(60,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kernx(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,1,200.,5.,20.,1.0,1000.)
    P={pr:prep(pr) for pr in CFG}
    for pr in CFG:
        df=H.fast_tail_read(H.S5_DIR/f"{pr}_S5_BA.parquet",5_000_000).sort_values('timestamp').reset_index(drop=True)
        P[pr]['ts']=df['timestamp'].to_numpy()

    print("=== convMFE γ1 H5 vs flat200 — per pair (IS/OOS p/d, MaxDD, worst, exp) ===")
    print(f"{'pair':<9}{'cfg':>9}{'IS_pd':>8}{'OOS_pd':>8}{'MaxDD':>8}{'worst':>8}{'exp':>7}{'WR%':>6}{'n':>6}")
    agg={'flat':[], 'conv':[]}; aggts={'flat':[], 'conv':[]}; paired=[]
    for pr,d in P.items():
        for lab,(mech,F0,Hd,g,T) in [('flat200',(0,200,200,1.0,0)),('convMFE',(1,200,5,1.0,0))]:
            net,e,r=trades_for(d,mech,F0,Hd,g,T*BAR_PER_H if T else 1)
            ts=d['ts'][e]; ism=e<d['is_end']
            ispd,_=pd_dd(net[ism],ts[ism]); oospd,oosdd=pd_dd(net[~ism],ts[~ism])
            key='flat' if lab=='flat200' else 'conv'
            agg[key].append((net,e,ts,ism))
            print(f"{pr:<9}{lab:>9}{ispd:>8.1f}{oospd:>8.1f}{oosdd:>8.0f}{net.min():>8.0f}{net.mean():>+7.2f}{(net>0).mean()*100:>5.0f}%{len(net):>6}")
        # paired diff (match by entry bar)
        nf,ef,_=trades_for(d,0,200,200,1.0,1); ng,eg,_=trades_for(d,1,200,5,1.0,1)
        mb={int(b):v for b,v in zip(ef,nf)}; mg={int(b):v for b,v in zip(eg,ng)}
        common=set(mb)&set(mg)
        paired += [mg[b]-mb[b] for b in common]
        print()
    # portfolio WF (4 chunks by global exit time)
    print("=== portfolio walk-forward (4 time chunks): p/d and MaxDD, both exits ===")
    for key,lab in [('flat','flat200'),('conv','convMFE')]:
        allnet=np.concatenate([a[0] for a in agg[key]]); allts=np.concatenate([a[2] for a in agg[key]])
        order=np.argsort(allts); allnet=allnet[order]; allts=allts[order]
        edges=np.linspace(0,len(allnet),5).astype(int)
        cells=[]
        for j in range(4):
            sl=slice(edges[j],edges[j+1]); pdv,ddv=pd_dd(allnet[sl],allts[sl]); cells.append((pdv,ddv))
        print(f"  {lab:<9}"+"  ".join(f"c{j+1}: {c[0]:+6.1f}p/d DD{c[1]:>5.0f}" for j,c in enumerate(cells)))
    # paired MC
    paired=np.array(paired)
    obs=paired.mean()
    boot=np.array([RNG.choice(paired,len(paired),replace=True).mean() for _ in range(2000)])
    p_gt0=(boot<=0).mean()   # prob improvement <=0
    print(f"\n=== paired-difference MC (net_convMFE − net_flat200, matched entries, n={len(paired)}) ===")
    print(f"  mean improvement = {obs:+.3f} p/trade   bootstrap 95% CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")
    print(f"  P(improvement <= 0) = {p_gt0:.4f}   => {'SIGNIFICANT improvement' if p_gt0<0.05 else 'not significant'}")
    print("\n  Deploy gate: convMFE OOS p/d >0 per pair AND portfolio improvement significant AND DD<=flat200.")


if __name__=="__main__":
    main()
