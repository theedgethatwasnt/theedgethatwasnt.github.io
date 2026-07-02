#!/usr/bin/env python3
"""
Step 2 — refine the converging fence PER PAIR. Step 1 showed it's not uniform: GBP_USD
loves convMFE, EUR_JPY (SL pair) is hurt by a low breakeven-lock. So sweep per pair a
finer grid incl. an ASYMMETRIC 'knee' (stay full-wide until mfe>=knee, then converge):
  d(mfe) = F0                                  if mfe <= knee
         = H + (F0-H)*(1 - (mfe-knee)/(A-knee))^γ   else
Pick each pair's best exit (max OOS p/d s.t. DD <= its flat200 baseline), then assemble
the portfolio with per-pair-best exits and re-test (OOS, WF, paired-MC vs all-flat200).
~9.6mo S5, net spread, IS/OOS.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG
from fence_timestop_sweep import prep

A_ARM=20.0; BAR_PER_H=int(60*60/5); RNG=np.random.default_rng(11)


@nb.njit(cache=True)
def kernr(o,h,l,c,t1l,t1s,t2l,t2s,sar_b,pip,tp_pips,use_psar,act, mech,F0,Hd,A,gamma,knee):
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
            if mech==1:
                if mfe<=knee: d=F0
                else:
                    frac=(mfe-knee)/(A-knee)
                    if frac>1.0: frac=1.0
                    d=Hd+(F0-Hd)*(1.0-frac)**gamma
            else:
                d=F0
            ex=0.0; r=-1
            fc=entry-pos*d*pip
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


def pd_dd(net, ts):
    if not len(net): return 0.0, 0.0
    o=np.argsort(ts); nn=net[o]
    days=max((pd.Timestamp(ts.max())-pd.Timestamp(ts.min())).total_seconds()/86400,1)
    cum=nn.cumsum(); return nn.sum()/days, float((cum-np.maximum.accumulate(cum)).min())


def run(d, mech,F0,Hd,gamma,knee):
    p,e,r=kernr(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                d['pip'],d['tp'],d['use_psar'],d['act'], mech,float(F0),float(Hd),A_ARM,gamma,float(knee))
    return p-d['sp'], e, r


def main():
    _c=np.zeros(60); _s=np.zeros(60,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kernr(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,1,200.,5.,20.,1.0,10.)
    P={pr:prep(pr) for pr in CFG}
    for pr in CFG:
        df=H.fast_tail_read(H.S5_DIR/f"{pr}_S5_BA.parquet",5_000_000).sort_values('timestamp').reset_index(drop=True)
        P[pr]['ts']=df['timestamp'].to_numpy()

    GAMMAS=[0.5,1.0,2.0,3.0]; HS=[5,15,30,60]; KNEES=[0,5,10]
    chosen={}
    # SELECT ON IS ONLY (R8: OOS sealed). Objective: max IS expectancy s.t. IS DD no worse
    # than that pair's flat200 IS DD. OOS reported afterward as held-out confirmation.
    print("=== per-pair refine: SELECTED ON IS, OOS shown as sealed confirmation ===")
    print(f"{'pair':<9}{'chosen exit':<26}{'IS_exp(base)':>16}{'| OOS_pd(base)':>16}{'OOS_DD(base)':>16}")
    for pr,d in P.items():
        bnet,be,_=run(d,0,200,200,1.0,0); bism=be<d['is_end']
        b_ispd,b_isdd=pd_dd(bnet[bism],d['ts'][be][bism]); b_isexp=bnet[bism].mean()
        b_ospd,b_osdd=pd_dd(bnet[~bism],d['ts'][be][~bism])
        best=None
        for g in GAMMAS:
            for Hd in HS:
                for kn in KNEES:
                    net,e,r=run(d,1,200,Hd,g,kn); ism=e<d['is_end']
                    is_exp=net[ism].mean(); _,is_dd=pd_dd(net[ism],d['ts'][e][ism])
                    if is_dd>=b_isdd-1e-9:                       # IS DD shallower-or-equal
                        if best is None or is_exp>best['is_exp']:
                            best=dict(g=g,Hd=Hd,kn=kn,is_exp=is_exp)
        if best is None or best['is_exp']<=b_isexp:
            chosen[pr]=dict(mech=0,F0=200,Hd=200,g=1.0,kn=0,desc='flat200')
            print(f"{pr:<9}{'flat200 (no IS-safe gain)':<26}{b_isexp:>+8.2f}{'':>8}{b_ospd:>+8.1f}{'':>8}{b_osdd:>8.0f}")
        else:
            net,e,r=run(d,1,200,best['Hd'],best['g'],best['kn']); ism=e<d['is_end']
            ospd,osdd=pd_dd(net[~ism],d['ts'][e][~ism])
            chosen[pr]=dict(mech=1,F0=200,Hd=best['Hd'],g=best['g'],kn=best['kn'],
                            desc=f"convMFE γ{best['g']} H{best['Hd']} kn{best['kn']}")
            print(f"{pr:<9}{chosen[pr]['desc']:<26}{best['is_exp']:>+8.2f}({b_isexp:+.2f}){ospd:>+8.1f}({b_ospd:+.1f}){osdd:>8.0f}({b_osdd:.0f})")

    # assemble portfolio: per-pair-best vs all-flat200
    print("\n=== portfolio: per-pair-best exits vs all-flat200 ===")
    paired=[]; agg={'flat':[],'best':[]}
    for pr,d in P.items():
        bnet,be,_=run(d,0,200,200,1.0,0)
        ch=chosen[pr]; cnet,ce,_=run(d,ch['mech'],ch['F0'],ch['Hd'],ch['g'],ch['kn'])
        agg['flat'].append((bnet,d['ts'][be],be<d['is_end'])); agg['best'].append((cnet,d['ts'][ce],ce<d['is_end']))
        mb={int(b):v for b,v in zip(be,bnet)}; mg={int(b):v for b,v in zip(ce,cnet)}
        for b in set(mb)&set(mg): paired.append(mg[b]-mb[b])
    for key,lab in [('flat','all-flat200'),('best','per-pair-best')]:
        net=np.concatenate([a[0] for a in agg[key]]); ts=np.concatenate([a[1] for a in agg[key]])
        ism=np.concatenate([a[2] for a in agg[key]])
        ospd,osdd=pd_dd(net[~ism],ts[~ism]); ispd,_=pd_dd(net[ism],ts[ism])
        print(f"  {lab:<14} IS {ispd:+6.1f}p/d  OOS {ospd:+6.1f}p/d  OOS_DD {osdd:>6.0f}  exp {net.mean():+.2f}  WR {(net>0).mean()*100:.0f}%  worst {net.min():.0f}")
    paired=np.array(paired); boot=np.array([RNG.choice(paired,len(paired),replace=True).mean() for _ in range(2000)])
    print(f"\n  paired improvement = {paired.mean():+.3f} p/tr  95%CI[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}]  "
          f"P(<=0)={ (boot<=0).mean():.4f}  => {'SIGNIFICANT' if (boot<=0).mean()<0.05 else 'not significant'}")


if __name__=="__main__":
    main()
