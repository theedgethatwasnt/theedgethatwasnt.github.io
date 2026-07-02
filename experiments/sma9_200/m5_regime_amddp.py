"""
m5_regime_amddp.py — AMDDP5 of the 3 bars INTO the entry as a momentum-QUALITY indicator.

At each SMA9/200 signal (fill bar e=i+1), score the run-up as if we'd entered in the signal
direction at the OPEN of the 3rd bar back (O[e-3]) and held to the signal bar close (C[e-1]):
   AMDDP5 = final_pnl_pips − 0.05 * accumulated_drawdown   (+ profit-protection floor)
accumulated_drawdown = sum over the 3 bars of giveback from the running favorable peak to the
bar's worst excursion. High = profitable AND smooth (sustained/increasing momentum, little
pullback); low = choppy. Then bin the actual trade outcome by this indicator vs raw momentum.
Causal (uses only bars at/<= the signal). Net of real spread on the trade. MA-cross exit.
"""
import numpy as np, pyarrow.parquet as pq, os
from numba import njit
from m5_regime_exits import blkres, sma, atr, run_exit, entries, PAIRS, DATA, FAST, SLOW

@njit
def amddp5_3(O,H,L,C,ei,di,pip,beta=0.05):
    m=len(ei); out=np.full(m,np.nan); rawm=np.full(m,np.nan)
    for t in range(m):
        e=ei[t]; d=di[t]; s=e-3                      # entry at open of 3rd bar back
        if s<1: continue
        entry=O[s]; peak=0.0; ddsum=0.0
        for k in range(s,e):                          # bars s, s+1, s+2 (= i-2,i-1,i)
            best  = d*((H[k] if d==1 else L[k])-entry)/pip   # best unrealized in bar (signal dir)
            worst = d*((L[k] if d==1 else H[k])-entry)/pip   # worst unrealized in bar
            if best>peak: peak=best
            gb = peak-worst
            if gb>0: ddsum+=gb
        finalp = d*(C[e-1]-entry)/pip                 # net move over the 3 bars, signal dir
        a = finalp - beta*ddsum
        if finalp>0 and a<0: a=0.001                  # profit-protection floor
        out[t]=a; rawm[t]=finalp
    return out, rawm

def binrep(label,x,pnl,nb=5):
    x=np.asarray(x); pnl=np.asarray(pnl); ok=~np.isnan(x)
    x=x[ok]; pnl=pnl[ok]; qs=np.quantile(x,np.linspace(0,1,nb+1))
    print(f"\n  {label}  (quintile bins)   range | n | mean p/tr | win%")
    for k in range(nb):
        lo,hi=qs[k],qs[k+1]; m=(x>=lo)&(x<=hi) if k==nb-1 else (x>=lo)&(x<hi)
        if m.sum()==0: continue
        pk=pnl[m]; print(f"    Q{k+1} [{lo:+7.2f},{hi:+7.2f}]  n={int(m.sum()):5d}  {pk.mean():+6.2f}  {100*(pk>0).mean():4.1f}%")

def main():
    g_a=[]; g_m=[]; g_p=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
        o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c"])
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); pip=0.01 if "JPY" in p else 0.0001
        sp_pips=float(np.median(ASK-BID)/pip); s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
        ei,di=entries(O,C,s9,s200)
        a3,m3=amddp5_3(O,H,L,C,ei,di,pip)
        pnl=run_exit(O,H,L,C,s9,s200,A,s9,ei,di,pip,sp_pips,0,0.,0.)   # MA-cross outcome
        g_a.append(a3/sp_pips); g_m.append(m3/sp_pips); g_p.append(pnl)  # indicators in spread units
    A=np.concatenate(g_a); M=np.concatenate(g_m); P=np.concatenate(g_p)
    print(f"AMDDP5-of-last-3-bars as a pre-entry indicator ({(~np.isnan(A)).sum()} trades). "
          f"Trade outcome = MA-cross exit, net of spread. baseline net {np.nanmean(P):+.2f} p/tr")
    print("="*84)
    binrep("RAW 3-bar momentum (signal dir, spread units)", M, P)
    binrep("AMDDP5 of last 3 bars (signal dir, spread units)", A, P)
    print("\n  (Q5 = strongest/cleanest run INTO the trend signal. If Q5 is the WORST outcome,")
    print("   the clean thrust is exhaustion -> a FADE indicator, not a continuation one.)")

if __name__=="__main__": main()
