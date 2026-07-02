#!/usr/bin/env python3
"""
Step 3 — when does the SMA-stack ENTRY work? Both exit studies say the exit isn't the
lever; the entry edge is regime-fragile (IS-neg / OOS-pos in this window). The stack is a
momentum/trend entry, so hypothesis: it wins in TRENDING regimes, bleeds in CHOP.
Condition each entry's net pnl (current flat-200 exit) on, at entry time:
  - Kaufman efficiency ratio ER_n = |c[t]-c[t-n]| / Σ|Δc|  (0=chop .. 1=pure trend)
  - realized vol (std of 5s returns over n) percentile
Bucket trades, show expectancy + WR + p/d per bucket, IS and OOS SEPARATELY. A regime
where the entry is positive in BOTH IS and OOS = a deployable gate (the lever). ~9.6mo S5.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from gbpjpy_h1h4_psar import psar_series
from stack010_equity import CFG
from fence_timestop_sweep import prep
from converging_fence_frontier import kernx, A_ARM

WINS = [240, 720, 2160]   # 20min, 1h, 3h of S5 bars


def er_series(c, n):
    """Kaufman efficiency ratio over trailing n bars, causal (uses c[..t])."""
    absd = np.abs(np.diff(c, prepend=c[0]))
    csum = np.cumsum(absd)
    vol = np.empty_like(c); vol[:n]=np.nan
    vol[n:] = csum[n:] - csum[:-n]                  # Σ|Δc| over window
    direction = np.empty_like(c); direction[:n]=np.nan
    direction[n:] = np.abs(c[n:] - c[:-n])
    er = np.where(vol>0, direction/vol, 0.0)
    er[:n]=np.nan
    return er


def main():
    _c=np.zeros(60); _s=np.zeros(60,np.int8); H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kernx(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,0,200.,200.,20.,1.0,1.)
    P={pr:prep(pr) for pr in CFG}
    for pr in CFG:
        df=H.fast_tail_read(H.S5_DIR/f"{pr}_S5_BA.parquet",5_000_000).sort_values('timestamp').reset_index(drop=True)
        P[pr]['ts']=df['timestamp'].to_numpy()

    # gather every entry's net + regime features + IS/OOS flag, pooled across pairs
    allnet=[]; allism=[]; ER={w:[] for w in WINS}; VOL=[]
    for pr,d in P.items():
        p,e,r,_=kernx(d['o'],d['h'],d['l'],d['c'],d['t1l'],d['t1s'],d['t2l'],d['t2s'],d['sar'],
                      d['pip'],d['tp'],d['use_psar'],d['act'],0,200.,200.,A_ARM,1.0,1.)
        net=p-d['sp']; ism=e<d['is_end']
        c=d['c']
        ers={w:er_series(c,w) for w in WINS}
        # realized vol over 720 bars (1h) at entry, as pip-returns std
        ret=np.abs(np.diff(c, prepend=c[0]))/d['pip']
        rc=np.cumsum(ret); rc2=np.cumsum(ret*ret); w=720
        volstd=np.full_like(c,np.nan)
        volstd[w:]=np.sqrt(np.maximum((rc2[w:]-rc2[:-w])/w - ((rc[w:]-rc[:-w])/w)**2,0))
        for k,bar in enumerate(e):
            if bar< max(WINS): continue
            allnet.append(net[k]); allism.append(ism[k]); VOL.append(volstd[bar])
            for wn in WINS: ER[wn].append(ers[wn][bar])
    allnet=np.array(allnet); allism=np.array(allism); VOL=np.array(VOL)
    for w in WINS: ER[w]=np.array(ER[w])
    print(f"pooled entries with regime features: {len(allnet)}  (IS {allism.sum()}, OOS {(~allism).sum()})")

    def buckets(feat, name, q=4):
        ok=~np.isnan(feat)
        f=feat[ok]; nn=allnet[ok]; im=allism[ok]
        edges=np.quantile(f,[i/q for i in range(q+1)])
        print(f"\n=== entry expectancy by {name} quartile (IS | OOS) ===")
        print(f"  {'bucket':<14}{'range':>16}{'IS_exp':>9}{'IS_WR':>7}{'IS_n':>7}{'OOS_exp':>10}{'OOS_WR':>8}{'OOS_n':>7}")
        for j in range(q):
            lo,hi=edges[j],edges[j+1]
            m=(f>=lo)&(f<=hi if j==q-1 else f<hi)
            mi=m&im; mo=m&~im
            ie=nn[mi].mean() if mi.sum() else float('nan'); oe=nn[mo].mean() if mo.sum() else float('nan')
            iw=(nn[mi]>0).mean()*100 if mi.sum() else 0; ow=(nn[mo]>0).mean()*100 if mo.sum() else 0
            print(f"  Q{j+1:<13}{f'[{lo:.3f},{hi:.3f}]':>16}{ie:>+9.2f}{iw:>6.0f}%{mi.sum():>7}{oe:>+10.2f}{ow:>7.0f}%{mo.sum():>7}")

    for w in WINS: buckets(ER[w], f"ER_{w}({w*5//60}min)")
    buckets(VOL, "realized-vol(1h)")
    print("\n  LEVER if: a high-ER (trend) bucket is positive in BOTH IS and OOS while low-ER (chop) is negative.")


if __name__=="__main__":
    main()
