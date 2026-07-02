"""
tf_regime.py — does the SMA9/200 regime system clear the spread at a HIGHER timeframe?
Same rule, resampled from S5 to M5 / M15 / H1. The spread is paid once per trade regardless
of TF, so a bigger-move TF amortizes the toll better — IF any gross edge survives. Reports the
trend entry (MA-cross exit) and its contrarian FADE (ATR bracket), net of real spread, per TF.
"""
import numpy as np, pyarrow.parquet as pq, os
from m5_regime_exits import blkres, sma, atr, run_exit, entries, PAIRS, DATA, FAST, SLOW

TFS=[(60,"M5"),(180,"M15"),(720,"H1")]

def main():
    rng=np.random.default_rng(0)
    for bars,name in TFS:
        bpd=17280.0/bars                                  # bars per day (86400s / (bars*5s))
        ent=[]; fad=[]; npos_e=npos_f=0; tot_days=0.0
        for p in PAIRS:
            f=DATA.format(p)
            if not os.path.exists(f): continue
            t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
            o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
                ["open","high","low","close","bid_c","ask_c"])
            O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask,bars=bars)
            if len(C) < SLOW+50: continue
            pip=0.01 if "JPY" in p else 0.0001
            sp=float(np.median(ASK-BID)/pip); s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
            ei,di=entries(O,C,s9,s200)
            if len(ei)==0: continue
            ep=run_exit(O,H,L,C,s9,s200,A,s9,ei,di,pip,sp,0,0.,0.)          # trend, MA-cross
            fp=run_exit(O,H,L,C,s9,s200,A,s9,ei,-di,pip,sp,4,1.5,1.5)       # fade, ATR bracket
            ent.append(ep); fad.append(fp); tot_days=max(tot_days,len(C)/bpd)
            npos_e+=ep.mean()>0; npos_f+=fp.mean()>0
        E=np.concatenate(ent); F=np.concatenate(fad)
        def mc(v):
            o=v.mean(); null=np.array([(v*rng.choice(np.array([-1.,1.]),len(v))).mean() for _ in range(1000)])
            return float((np.abs(null)>=abs(o)).mean())
        print(f"\n{name:4s} ({bars} S5/bar)  —  {len(E)} trades ({len(E)/tot_days:.1f}/day)")
        print(f"  TREND (MA-cross): net {E.mean():+6.2f} p/tr  WR {100*(E>0).mean():4.1f}%  "
              f"{E.sum()/tot_days:+7.1f} p/day  {npos_e}/12 pairs+  MC p={mc(E):.3f}")
        print(f"  FADE  (ATR brkt): net {F.mean():+6.2f} p/tr  WR {100*(F>0).mean():4.1f}%  "
              f"{F.sum()/tot_days:+7.1f} p/day  {npos_f}/12 pairs+  MC p={mc(F):.3f}")

if __name__=="__main__": main()
