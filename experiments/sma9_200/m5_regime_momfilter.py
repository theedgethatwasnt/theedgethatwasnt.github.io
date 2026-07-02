"""
m5_regime_momfilter.py — does narrowing the SMA9/200 entry to bars where the 3-bar pre-entry
price momentum is SIGNIFICANT and CONTINUING (each of the 3 bars moving in the trade direction)
produce a spread-clearing subset? Reuses the causal per-trade features from m5_regime_lgbm.

r1=newest bar return, r3=oldest, all signed IN TRADE DIRECTION in spread units. mom3 = total.
"continuing" = r1>0 & r2>0 & r3>0 (sustained thrust). "accelerating" = r1>=r2>=r3 (building).
"""
import numpy as np, pandas as pd, os
from m5_regime_lgbm import build_pair, PAIRS, DATA

def main():
    df = pd.concat([build_pair(p) for p in PAIRS if os.path.exists(DATA.format(p))],
                   ignore_index=True)
    N=len(df); base=df.pnl.mean()
    print(f"SMA9/200 M5 entry — momentum-continuation FILTER ({N} trades). Net of real spread.")
    print(f"baseline (all trades): net {base:+.2f} p/tr, WR {100*(df.pnl>0).mean():.1f}%\n"+"="*84)
    def rep(name, m):
        s=df[m]
        if len(s)==0: print(f"  {name:46s}   n=    0"); return
        print(f"  {name:46s}   n={len(s):5d} ({100*len(s)/N:4.1f}%)   net {s.pnl.mean():+6.2f}   "
              f"WR {100*(s.pnl>0).mean():4.1f}%   tot {s.pnl.sum():+8.0f}p")
    cont = (df.r1>0)&(df.r2>0)&(df.r3>0)
    acc  = cont & (df.r1>=df.r2) & (df.r2>=df.r3)
    print("  -- continuity --")
    rep("continuing (all 3 bars WITH trade)", cont)
    rep("choppy (NOT all 3 with trade)", ~cont)
    rep("continuing & accelerating (r1>=r2>=r3)", acc)
    print("  -- significant AND continuing (mom3 in spread units) --")
    for thr in (1,2,3,4):
        rep(f"continuing & mom3 > {thr} spread(s)", cont & (df.mom3>thr))
    print("  -- significant AND continuing AND accelerating --")
    for thr in (1,2,3):
        rep(f"cont+accel & mom3 > {thr} spread(s)", acc & (df.mom3>thr))
    print("  -- for contrast: strong momentum but AGAINST the trade (price moved against you) --")
    rep("all 3 bars AGAINST trade (r1,r2,r3<0)", (df.r1<0)&(df.r2<0)&(df.r3<0))
    print("="*84)
    print("  Verdict: any subset with net > 0 clears the spread; otherwise momentum-narrowing")
    print("  reduces the bleed at best, and the entry still has no monetizable directional edge.")

if __name__=="__main__": main()
