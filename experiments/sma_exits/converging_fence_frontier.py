#!/usr/bin/env python3
"""
010 robustness — converging-fence FRONTIER (p/d vs MaxDD) across 4 mechanism families.
See DESIGN_converging_fence.md. Reuses H17 prep() (signals cached per pair) + a unified
kernel. ~9.6mo S5, net of per-pair spread, IS/OOS split, calendar-day p/d.

Mechanisms (entry-anchored stop distance d, ratchets since mfe/age monotonic):
  mech 0 flat            : d = F0           (baseline=200; tighter flat = 'step'/early-SL)
  mech 1 converge-MFE    : d = H + (F0-H)*(1-min(mfe/A,1))^g
  mech 2 converge-time   : d = H + (F0-H)*(1-min(age/T,1))^g
  psar-from-entry        : mech 0 + lowered PSAR arm `act` (PSAR pairs only)
Once mfe>=act the PSAR exit coexists (usually tighter); SL pairs have no PSAR.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG
from fence_timestop_sweep import prep

BAR_PER_H = int(60*60/5)   # S5 bars/hour
A_ARM = 20.0               # PSAR arm / convergence target (pips)
# current OANDA pip $ value per unit (JPY≈$0.0000626, USD-quote≈$0.0001) and live ~38u, NAV~$76
NAV = 76.0; UNITS = 38


@nb.njit(cache=True)
def kernx(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, pip, tp_pips, use_psar, act,
          mech, F0, Hd, A, gamma, Tbars):
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; tmae=0.0; armed=False
    pnl=np.empty(n); ent=np.empty(n,np.int64); rsn=np.empty(n,np.int64); mae=np.empty(n); nt=0
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1: pos=1; entry=o[i]; ebar=i; mfe=0.0; tmae=0.0; armed=False; continue
            if t1s[i]==1 and t2s[i]==1: pos=-1; entry=o[i]; ebar=i; mfe=0.0; tmae=0.0; armed=False; continue
        if pos!=0:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            adv=(entry-l[i])/pip if pos==1 else (h[i]-entry)/pip
            if fav>mfe: mfe=fav
            if adv>tmae: tmae=adv
            if use_psar and (not armed) and mfe>=act: armed=True
            if mech==1:
                frac=mfe/A
                if frac>1.0: frac=1.0
                d=Hd+(F0-Hd)*(1.0-frac)**gamma
            elif mech==2:
                frac=(i-ebar)/Tbars
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
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; mae[nt]=tmae; nt+=1; pos=0
    return pnl[:nt], ent[:nt], rsn[:nt], mae[:nt]


def pip_dollar(pair):
    # $/unit/pip. JPY-quote ≈ pip(0.01)/spot; USD-quote = 0.0001*1 (USD account). Approx spots.
    spot = {"EUR_JPY":160.,"GBP_JPY":195.,"USD_JPY":157.,"EUR_USD":1.,"GBP_USD":1.,"AUD_USD":1.}
    pip = CFG[pair][0]
    if pair.endswith("JPY"):
        return pip / spot.get(pair,150.)         # 0.01 JPY converted to USD
    return 0.0001                                # USD-quote, 1 unit


# (name, mech, F0, H, gamma, T_h, act_override)  act_override<0 => use pair default
CONFIGS = [
 ("flat200 (current)",    0, 200, 200, 1.0,  0, -1),
 ("flat120",              0, 120, 120, 1.0,  0, -1),
 ("flat80",               0,  80,  80, 1.0,  0, -1),
 ("flat50",               0,  50,  50, 1.0,  0, -1),
 ("psar_entry act0",      0, 200, 200, 1.0,  0,  0),
 ("psar_entry act5",      0, 200, 200, 1.0,  0,  5),
 ("psar_entry act10",     0, 200, 200, 1.0,  0, 10),
 ("convMFE g0.5 H30",     1, 200,  30, 0.5,  0, -1),
 ("convMFE g1 H30",       1, 200,  30, 1.0,  0, -1),
 ("convMFE g2 H30",       1, 200,  30, 2.0,  0, -1),
 ("convMFE g4 H30",       1, 200,  30, 4.0,  0, -1),
 ("convMFE g1 H5",        1, 200,   5, 1.0,  0, -1),
 ("convMFE g2 H5",        1, 200,   5, 2.0,  0, -1),
 ("convMFE g4 H5",        1, 200,   5, 4.0,  0, -1),
 ("time g1 H30 T24h",     2, 200,  30, 1.0, 24, -1),
 ("time g1 H30 T8h",      2, 200,  30, 1.0,  8, -1),
 ("time g2 H30 T24h",     2, 200,  30, 2.0, 24, -1),
 ("time g1 H5 T24h",      2, 200,   5, 1.0, 24, -1),
]


def run_config(P, cfg):
    name, mech, F0, Hd, gamma, T_h, act_ovr = cfg
    Tbars = max(T_h*BAR_PER_H, 1)
    is_net=[]; oos_net=[]; is_ts=[]; oos_ts=[]; allnet=[]; ntail=0; ntot=0; worst=0.0
    worst_usd = 0.0
    for pr,d in P.items():
        act = act_ovr if (act_ovr>=0 and d['use_psar']) else d['act']
        p,e,r,mae = kernx(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                          d['pip'],d['tp'],d['use_psar'],act, mech, float(F0), float(Hd), A_ARM, gamma, Tbars)
        net=p-d['sp']; ism=e<d['is_end']
        ts=d['ts_exit'][e]
        is_net.append(net[ism]); oos_net.append(net[~ism]); is_ts.append(ts[ism]); oos_ts.append(ts[~ism])
        allnet.append(net); ntail+=int((r==2).sum()); ntot+=len(p)
        if len(net): worst=min(worst, net.min()); worst_usd=min(worst_usd, net.min()*pip_dollar(pr)*UNITS)
    isn=np.concatenate(is_net); oosn=np.concatenate(oos_net); alln=np.concatenate(allnet)
    ist=np.concatenate(is_ts); oost=np.concatenate(oos_ts)
    def days(t): return max((pd.Timestamp(t.max())-pd.Timestamp(t.min())).total_seconds()/86400, 1) if len(t) else 1
    is_pd=isn.sum()/days(ist); oos_pd=oosn.sum()/days(oost)
    oa=oosn[np.argsort(oost)]; cum=oa.cumsum(); maxdd=float((cum-np.maximum.accumulate(cum)).min()) if len(oa) else 0
    return dict(name=name, trades=ntot, is_pd=is_pd, oos_pd=oos_pd, exp=alln.mean(),
                wr=(alln>0).mean()*100, tail=ntail/max(ntot,1)*100, worst=worst,
                worst_usd=worst_usd, maxdd=maxdd, dd_usd=maxdd/ -200.0 * worst_usd if worst_usd else 0)


def main():
    _c=np.zeros(60); _s=np.zeros(60,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kernx(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,1,200.,30.,20.,1.0,1000.)
    P={pr:prep(pr) for pr in CFG}
    # prep() doesn't carry exit timestamps; recompute the ts array alongside (same fast read)
    for pr in CFG:
        df=H.fast_tail_read(H.S5_DIR/f"{pr}_S5_BA.parquet", 5_000_000).sort_values('timestamp').reset_index(drop=True)
        P[pr]['ts_exit']=df['timestamp'].to_numpy()
    rows=[run_config(P,c) for c in CONFIGS]
    base=rows[0]
    print("010 converging-fence FRONTIER (4 pairs, ~9.6mo S5, net spread, IS/OOS).")
    print(f"  {'config':<20}{'trades':>7}{'IS_pd':>8}{'OOS_pd':>8}{'exp':>7}{'WR%':>6}{'tail%':>7}{'worst_p':>9}{'worstUSD':>9}{'OOS_DD':>8}")
    for r in rows:
        flag = '  <-- base' if r['name']==base['name'] else ''
        print(f"  {r['name']:<20}{r['trades']:>7}{r['is_pd']:>8.1f}{r['oos_pd']:>8.1f}{r['exp']:>+7.2f}"
              f"{r['wr']:>5.0f}%{r['tail']:>6.1f}%{r['worst']:>9.0f}{r['worst_usd']:>+9.2f}{r['maxdd']:>8.0f}{flag}")
    # frontier plot: OOS_pd vs |OOS MaxDD|
    fig,ax=plt.subplots(1,2,figsize=(15,7))
    fam_color=lambda nm:( '#888' if nm.startswith('flat200') else '#9c27b0' if nm.startswith('flat')
                          else '#ff9800' if nm.startswith('psar') else '#2196f3' if nm.startswith('convMFE')
                          else '#4caf50')
    for r in rows:
        for k,(xx,yy,xl,t) in enumerate([(abs(r['maxdd']),r['oos_pd'],'OOS MaxDD (pips)','OOS p/d vs MaxDD'),
                                          (abs(r['worst']),r['oos_pd'],'worst single trade (pips)','OOS p/d vs worst-trade')]):
            c=fam_color(r['name']); mk='*' if r['name']==base['name'] else 'o'; sz=320 if mk=='*' else 90
            ax[k].scatter(xx,yy,c=c,marker=mk,s=sz,edgecolors='k',zorder=3)
            ax[k].annotate(r['name'].replace('conv','c').replace(' ',''),(xx,yy),fontsize=6,xytext=(4,3),textcoords='offset points')
            ax[k].set_xlabel(xl); ax[k].set_ylabel('OOS pips/day'); ax[k].set_title(t); ax[k].grid(alpha=.3)
    plt.suptitle('010 converging-fence frontier — ★=current flat200 | grey=flat/step purple=flat-tight orange=psar-entry blue=convMFE green=time')
    plt.tight_layout(); out=Path(__file__).parent/'results'/'converging_fence_frontier.png'; out.parent.mkdir(exist_ok=True)
    plt.savefig(out,dpi=110); print(f"\n  frontier -> {out}")
    print(f"\n  baseline OOS_pd={base['oos_pd']:.1f}  MaxDD={base['maxdd']:.0f}p  worst={base['worst']:.0f}p ({base['worst_usd']:+.2f} @ {UNITS}u/${NAV:.0f}NAV)")
    print("  Want: points UP (>=base p/d) and LEFT (smaller MaxDD/worst) of the star.")


if __name__=="__main__":
    main()
