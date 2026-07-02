"""
m5_regime_inpain.py — how much PAIN must you endure to ride a winning trend?
For each SMA9/200 trend trade, walk to the MA-cross exit tracking the in-trade MAE (max adverse
excursion = deepest drawdown), MFE (best run), accumulated-drawdown (AMDDP pain), hold, pnl.
Then characterize WINNERS vs LOSERS: a winner that zig-zags its way up still spends time
underwater — quantify how much, so you know the drawdown you must tolerate (and why a tight
stop cuts the winners). Mid OHLC, net of real spread.
"""
import numpy as np, pyarrow.parquet as pq, os
from numba import njit
from m5_regime_exits import blkres, sma, atr, entries, PAIRS, DATA, FAST, SLOW

@njit
def walk_pain(O,H,L,C,s9,s200,ei,di,pip,sp_pips):
    m=len(ei); mae=np.empty(m); mfe=np.empty(m); pnl=np.empty(m); hold=np.empty(m); ddsum_a=np.empty(m)
    for t in range(m):
        e=ei[t]; d=di[t]; ent=O[e]; peak=0.0; worst=0.0; best=0.0; ddsum=0.0
        end=min(e+4000,len(C)-1); k=e
        for k in range(e,end):
            fav=d*((H[k] if d==1 else L[k])-ent)/pip          # best within bar
            adv=d*((L[k] if d==1 else H[k])-ent)/pip          # worst within bar (<=0 mostly)
            if fav>best: best=fav
            if adv<worst: worst=adv
            if fav>peak: peak=fav
            gb=peak-adv
            if gb>0: ddsum+=gb
            if (d==-1 and s9[k]>=s200[k]) or (d==1 and s9[k]<=s200[k]): break
        mae[t]=-worst; mfe[t]=best; pnl[t]=(d*(C[k]-ent)/pip)-sp_pips; hold[t]=k-e; ddsum_a[t]=ddsum
    return mae,mfe,pnl,hold,ddsum_a

def pct(a,qs): return [round(float(np.percentile(a,q)),1) for q in qs]

def main():
    MAE=[]; MFE=[]; PNL=[]; HOLD=[]; DD=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
        o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c"])
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); pip=0.01 if "JPY" in p else 0.0001
        sp=float(np.median(ASK-BID)/pip); s9,s200=sma(C,FAST),sma(C,SLOW)
        ei,di=entries(O,C,s9,s200)
        mae,mfe,pnl,hold,dd=walk_pain(O,H,L,C,s9,s200,ei,di,pip,sp)
        MAE.append(mae);MFE.append(mfe);PNL.append(pnl);HOLD.append(hold);DD.append(dd)
    mae=np.concatenate(MAE);mfe=np.concatenate(MFE);pnl=np.concatenate(PNL);hold=np.concatenate(HOLD);dd=np.concatenate(DD)
    win=pnl>0; los=~win
    qs=[25,50,75,90]
    print(f"SMA9/200 M5 trend trades, in-trade pain to MA-cross exit ({len(pnl)} trades, {100*win.mean():.0f}% winners)")
    print("="*84)
    print(f"  {'group':16s} {'n':>6} {'MAE pips (p25/50/75/90)':>30} {'MFE p50':>8} {'hold p50':>9} {'net p/tr':>9}")
    print(f"  {'WINNERS':16s} {win.sum():6d}   {str(pct(mae[win],qs)):>28}   {np.median(mfe[win]):6.1f}   {np.median(hold[win]):7.0f}   {pnl[win].mean():+7.2f}")
    print(f"  {'LOSERS':16s} {los.sum():6d}   {str(pct(mae[los],qs)):>28}   {np.median(mfe[los]):6.1f}   {np.median(hold[los]):7.0f}   {pnl[los].mean():+7.2f}")
    print(f"\n  Accumulated AMDDP drawdown (path pain) — winners p50={np.median(dd[win]):.0f}  losers p50={np.median(dd[los]):.0f}")
    print(f"  Winner MAE / its MFE ratio (pain endured per pip of run, p50): {np.median(mae[win]/(mfe[win]+1e-9)):.2f}")
    # how often does a winner first dip past a given drawdown before paying out?
    print("\n  Of WINNERS, share that first endured a drawdown of at least X pips before the win:")
    for X in (3,5,10,20,30):
        print(f"     >= {X:2d}p drawdown: {100*np.mean(mae[win]>=X):4.1f}%")
    print("\n  Read: a tight stop set inside the winners' MAE band cuts the very trades you need.")

if __name__=="__main__": main()
