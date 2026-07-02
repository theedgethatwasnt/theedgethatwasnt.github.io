"""
m5_regime_amddp_rangepos.py — FOCUS on the one signal with real IC: AMDDP5-into-entry.
Build it as a CONTINUOUS time series (S5-cadence pain, fixed W bars, computed at every M5 bar,
per direction), then take its Stochastic-%K position within its own last-N-bar range:
    %K_N = (AMDDP_now − rolling_min_N) / (rolling_max_N − rolling_min_N)   in [0,1]
%K≈1 = the run into THIS entry is near its N-bar PEAK (exceptionally clean/strong vs recent norm);
%K≈0 = near its trough. Question: does range-position predict the trade outcome? All other
indicators dropped. Causal: the series at bar i ends at i's S5 close, entry is i+1 (no leak).
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, os
from numba import njit
from scipy.stats import spearmanr
from m5_regime_exits import blkres, sma, atr, run_exit, entries, PAIRS, DATA, FAST, SLOW

W=5; NS=(100,500,1000)

@njit
def amddp_series(o5,h5,l5,c5,d,pip,n_m5,beta=0.05):
    out=np.full(n_m5,np.nan); n5=len(c5)
    for b in range(W-1,n_m5):
        start=(b-W+1)*60; end=b*60+60
        if end>n5: break
        entry=o5[start]; peak=0.0; ddsum=0.0
        for k in range(start,end):
            best  = d*((h5[k] if d==1 else l5[k])-entry)/pip
            worst = d*((l5[k] if d==1 else h5[k])-entry)/pip
            if best>peak: peak=best
            gb=peak-worst
            if gb>0: ddsum+=gb
        finalp=d*(c5[end-1]-entry)/pip
        a=finalp-beta*ddsum
        if finalp>0 and a<0: a=0.001
        out[b]=a
    return out

def pctK(s,N):
    s=pd.Series(s); mn=s.rolling(N,min_periods=N//2).min(); mx=s.rolling(N,min_periods=N//2).max()
    return ((s-mn)/(mx-mn+1e-12)).to_numpy()

def main():
    g={N:[] for N in NS}; g["lvl"]=[]; g["pnl"]=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
        o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c"])
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); pip=0.01 if "JPY" in p else 0.0001
        sp_pips=float(np.median(ASK-BID)/pip); s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
        nm=len(C)
        sL=amddp_series(o,h,l,c, 1,pip,nm); sS=amddp_series(o,h,l,c,-1,pip,nm)
        kL={N:pctK(sL,N) for N in NS}; kS={N:pctK(sS,N) for N in NS}
        ei,di=entries(O,C,s9,s200); sig=ei-1
        pnl=run_exit(O,H,L,C,s9,s200,A,s9,ei,di,pip,sp_pips,0,0.,0.)
        for t_ in range(len(ei)):
            i=sig[t_]; d=di[t_]
            for N in NS: g[N].append(kL[N][i] if d==1 else kS[N][i])
            g["lvl"].append((sL[i] if d==1 else sS[i])/sp_pips); g["pnl"].append(pnl[t_])
    pnl=np.array(g["pnl"]); base=np.nanmean(pnl)
    print(f"AMDDP5-into-entry range-position (Stochastic-%K), W={W} S5-cadence. {len(pnl)} trades, base net {base:+.2f} p/tr")
    print("="*78)
    def ic(name,x):
        x=np.array(x); ok=~np.isnan(x);
        print(f"  IC({name:12s}, pnl) = {spearmanr(x[ok],pnl[ok]).statistic:+.4f}")
    ic("AMDDP level",g["lvl"])
    for N in NS: ic(f"%K_{N}",g[N])
    for N in NS:
        x=np.array(g[N]); ok=~np.isnan(x); xx=x[ok]; pp=pnl[ok]
        qs=np.quantile(xx,np.linspace(0,1,6))
        print(f"\n  trade outcome by %K_{N} (0=AMDDP at N-bar trough, 1=at N-bar PEAK):")
        for k in range(5):
            lo,hi=qs[k],qs[k+1]; m=(xx>=lo)&(xx<=hi) if k==4 else (xx>=lo)&(xx<hi)
            s=pp[m]; print(f"    Q{k+1} [{lo:.2f},{hi:.2f}] n={int(m.sum()):5d}  net {s.mean():+6.2f}  WR {100*(s>0).mean():4.1f}%")

if __name__=="__main__": main()
