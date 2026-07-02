"""
m5_regime_exits.py — exit/stop bake-off on the SMA9/200 M5 regime ENTRY.
Same entries (signal bar i, fill at i+1 open, causal), every stop type the project has used,
each compared to the user's baseline exit (SMA9 x SMA200 regime flip). Net of real spread.

Each non-baseline variant exits on whichever comes first: ITS stop OR the regime-flip backstop
(so it answers "does cutting early with stop X beat just holding to the flip?"). Intrabar level
exits (SL/TP/trail) fill at the level using mid H/L; close-based exits (MA-cross/time/supertrend)
fill at the bar close. Full spread deducted once per round trip (fair across variants).
"""
import numpy as np, pyarrow.parquet as pq, os
from numba import njit

PAIRS=["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY",
       "AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
DATA="data/s5_ohlc/{}_S5_BA.parquet"; BARS_PER=60; FAST,SLOW=9,200; MAXH=4000

def blkres(o,h,l,c,bid,ask,bars=BARS_PER):
    n=len(c); nb=n//bars
    def B(a): return a[:nb*bars].reshape(nb,bars)
    return B(o)[:,0],B(h).max(1),B(l).min(1),B(c)[:,-1],B(bid)[:,-1],B(ask)[:,-1]
def sma(c,p):
    out=np.full(len(c),np.nan); cs=np.cumsum(c); out[p-1:]=(cs[p-1:]-np.concatenate([[0.0],cs[:-p]]))/p; return out
@njit
def atr(h,l,c,p=14):
    n=len(c); tr=np.empty(n); tr[0]=h[0]-l[0]; out=np.full(n,np.nan); s=0.0
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    run=0.0
    for i in range(n):
        run+=tr[i]
        if i>=p: run-=tr[i-p]
        if i>=p-1: out[i]=run/p
    return out

@njit
def entries(O,C,s9,s200):
    """Return e_idx (fill bar), dir for each entry (causal first-occurrence regime rule)."""
    n=len(C); pos=0; ei=np.empty(n,np.int64); di=np.empty(n,np.int64); k=0
    for i in range(SLOW,n-1):
        if np.isnan(s9[i]) or np.isnan(s200[i]): continue
        if pos!=0:
            if (pos==-1 and s9[i]>=s200[i]) or (pos==1 and s9[i]<=s200[i]): pos=0
            continue
        down=s9[i]<s200[i]; up=s9[i]>s200[i]
        sc=down and (C[i]<s9[i]) and not((s9[i-1]<s200[i-1]) and (C[i-1]<s9[i-1]))
        lc=up   and (C[i]>s9[i]) and not((s9[i-1]>s200[i-1]) and (C[i-1]>s9[i-1]))
        if sc: ei[k]=i+1; di[k]=-1; k+=1; pos=-1
        elif lc: ei[k]=i+1; di[k]=1; k+=1; pos=1
    return ei[:k], di[:k]

@njit
def run_exit(O,H,L,C,s9,s200,A,M,ei,di,pip,sp_pips,code,p1,p2):
    """pnl (net pips) per entry for a given exit code. p1,p2 = params (ATR mults / pips / bars).
    M = an extra SMA series used by code 12 (exit when that SMA loses favorable slope)."""
    m=len(ei); out=np.empty(m)
    for t in range(m):
        e=ei[t]; d=di[t]; ent=O[e]; aentry=A[e] if not np.isnan(A[e]) else (sp_pips*pip*3)
        best=ent  # most-favorable price reached (for trails/giveback)
        ex=ent; done=False
        end=min(e+MAXH, len(C)-1)
        for k in range(e, end):
            hi=H[k]; lo=L[k]; cl=C[k]
            fav = (ent-lo) if d==-1 else (hi-ent)     # favorable excursion (price units)
            adv = (hi-ent) if d==-1 else (ent-lo)     # adverse excursion
            # update favorable extreme
            if d==-1:
                if lo<best: best=lo
            else:
                if hi>best: best=hi
            # ---- intrabar level exits ----
            if code==1:   # fixed SL p1 pips
                if adv>=p1*pip: ex=ent + d*(-1)*p1*pip if False else (ent + p1*pip if d==-1 else ent - p1*pip); done=True
            elif code==2: # ATR SL p1*ATR
                if adv>=p1*aentry: ex=(ent + p1*aentry) if d==-1 else (ent - p1*aentry); done=True
            elif code==3: # fixed TP p1 pips
                if fav>=p1*pip: ex=(ent - p1*pip) if d==-1 else (ent + p1*pip); done=True
            elif code==4: # ATR bracket TP p1*ATR / SL p2*ATR (SL checked first = conservative)
                if adv>=p2*aentry: ex=(ent + p2*aentry) if d==-1 else (ent - p2*aentry); done=True
                elif fav>=p1*aentry: ex=(ent - p1*aentry) if d==-1 else (ent + p1*aentry); done=True
            elif code==5: # fixed-pip trail p1
                stop = (best + p1*pip) if d==-1 else (best - p1*pip)
                if (d==-1 and hi>=stop) or (d==1 and lo<=stop): ex=stop; done=True
            elif code==6: # ATR trail p1*ATR (entry ATR)
                stop = (best + p1*aentry) if d==-1 else (best - p1*aentry)
                if (d==-1 and hi>=stop) or (d==1 and lo<=stop): ex=stop; done=True
            elif code==7: # chandelier: trail off favorable extreme by p1*current ATR
                ak=A[k] if not np.isnan(A[k]) else aentry
                stop=(best + p1*ak) if d==-1 else (best - p1*ak)
                if (d==-1 and hi>=stop) or (d==1 and lo<=stop): ex=stop; done=True
            elif code==8: # SuperTrend: hl2 ± p1*ATR, exit on CLOSE cross (bar close)
                ak=A[k] if not np.isnan(A[k]) else aentry; hl2=(hi+lo)/2.0
                band=(hl2 + p1*ak) if d==-1 else (hl2 - p1*ak)
                if (d==-1 and cl>band) or (d==1 and cl<band): ex=cl; done=True
            elif code==9: # peak-giveback: after fav>=p1*ATR, exit if give back p2*ATR from extreme
                if fav>=p1*aentry:
                    gb=(best + p2*aentry) if d==-1 else (best - p2*aentry)
                    if (d==-1 and hi>=gb) or (d==1 and lo<=gb): ex=gb; done=True
            elif code==10: # N-bar time stop
                if (k-e)>=p1: ex=cl; done=True
            elif code==11: # breakeven at +p1*ATR then ATR trail p2*ATR
                if fav>=p1*aentry:
                    base = ent
                    tr = (best + p2*aentry) if d==-1 else (best - p2*aentry)
                    stop = min(base,tr) if d==-1 else max(base,tr)
                    if (d==-1 and hi>=stop) or (d==1 and lo<=stop): ex=stop; done=True
            elif code==12: # exit when SMA M loses favorable slope (MA rolls over against trade)
                if k>e and not np.isnan(M[k]) and not np.isnan(M[k-1]):
                    slope = M[k]-M[k-1]
                    if (d==-1 and slope>=0.0) or (d==1 and slope<=0.0): ex=cl; done=True
            elif code==13: # TP+trail together: once fav>=p1 pips, trail a p2-pip step behind the extreme
                if fav>=p1*pip:
                    stop = (best + p2*pip) if d==-1 else (best - p2*pip)
                    if (d==-1 and hi>=stop) or (d==1 and lo<=stop): ex=stop; done=True
            elif code==14: # early-deep stop: SL of p1 pips active ONLY in the first p2 bars (then free)
                if (k-e) < p2 and adv >= p1*pip:
                    ex=(ent + p1*pip) if d==-1 else (ent - p1*pip); done=True
            if done: break
            # ---- regime-flip backstop (all variants incl baseline=code 0) ----
            if (d==-1 and s9[k]>=s200[k]) or (d==1 and s9[k]<=s200[k]):
                ex = O[k+1] if k+1<len(O) else cl; done=True; break
        if not done: ex=C[end]
        out[t] = (d*(ex-ent)/pip) - sp_pips     # net of full spread
    return out

EXITS = [  # (label, code, p1, p2)
 ("baseline MA-cross",        0, 0,    0),
 ("fixed SL 15p",             1, 15,   0),
 ("fixed SL 30p",             1, 30,   0),
 ("ATR SL 2x",                2, 2.0,  0),
 ("fixed TP 15p",             3, 15,   0),
 ("ATR bracket TP2/SL2",      4, 2.0,  2.0),
 ("ATR bracket TP1.5/SL1",    4, 1.5,  1.0),
 ("fixed-pip trail 10p",      5, 10,   0),
 ("ATR trail 1.5x",           6, 1.5,  0),
 ("ATR trail 3x",             6, 3.0,  0),
 ("chandelier 3x ATR",        7, 3.0,  0),
 ("SuperTrend 2x ATR",        8, 2.0,  0),
 ("peak-giveback a1/g0.5",    9, 1.0,  0.5),
 ("N-bar time stop 24",      10, 24,   0),
 ("N-bar time stop 72",      10, 72,   0),
 ("breakeven@1ATR + trail2", 11, 1.0,  2.0),
 ("TP2 + trail-step 2p",     13, 2.0,  2.0),
 ("TP2 + trail-step 3p",     13, 2.0,  3.0),
 ("early-MAE 10p/12bar",     14, 10,   12),
 ("early-MAE 12p/24bar",     14, 12,   24),
 ("early-MAE 15p/24bar",     14, 15,   24),
 ("early-MAE 20p/36bar",     14, 20,   36),
]

MOM_PERIODS=(5,7,9,13,18,22)   # "exit when SMA{p} loses momentum" sweep

def main():
    mom_labels=[f"SMA{p} momentum-loss" for p in MOM_PERIODS]
    all_labels=[lab for lab,_,_,_ in EXITS]+mom_labels
    pooled={lab:[] for lab in all_labels}; tot_days=0.0; nentry=0
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
        o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c"])
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask)
        pip=0.01 if "JPY" in p else 0.0001
        sp_pips=float(np.median(ASK-BID)/pip)
        s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
        ei,di=entries(O,C,s9,s200); nentry+=len(ei); tot_days=max(tot_days,len(C)/288.0)
        for lab,code,p1,p2 in EXITS:
            pnl=run_exit(O,H,L,C,s9,s200,A,s9,ei,di,pip,sp_pips,code,float(p1),float(p2))
            pooled[lab].append(pnl)
        for per in MOM_PERIODS:                       # SMA-momentum-loss exit sweep
            mm=sma(C,per)
            pnl=run_exit(O,H,L,C,s9,s200,A,mm,ei,di,pip,sp_pips,12,float(per),0.0)
            pooled[f"SMA{per} momentum-loss"].append(pnl)
    print(f"SMA9/200 M5 entry — exit bake-off across 12 pairs ({nentry} entries). Net of real spread.\n"+"="*78)
    print(f"  {'exit rule':26s} {'net p/tr':>9}  {'win%':>6}  {'p/day':>8}  {'total':>9}")
    base=np.concatenate(pooled["baseline MA-cross"]); bmean=base.mean()
    rows=[]
    for lab in all_labels:
        v=np.concatenate(pooled[lab]); rows.append((lab,v.mean(),100*(v>0).mean(),v.sum()/tot_days,v.sum()))
    for lab,mean,wr,ppd,tot in sorted(rows,key=lambda r:-r[1]):
        d = mean-bmean
        flag = "  ← baseline" if lab=="baseline MA-cross" else (f"  (+{d:.2f} vs base)" if d>0 else f"  ({d:.2f} vs base)")
        print(f"  {lab:26s} {mean:+9.2f}  {wr:5.1f}%  {ppd:+8.1f}  {tot:+9.0f}{flag}")
    print("="*78)
    print("  (All variants exit on their stop OR the regime-flip backstop, whichever first.)")

if __name__=="__main__": main()
