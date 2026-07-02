"""
eurusd_portfolio_overlap.py — if the 7 BB-reentry configs ran concurrently on EUR/USD, how much
do their trades OVERLAP (redundant) vs NOVEL, and what % of time is the portfolio in the market?
Generates each config's (entry,exit,dir) intervals, maps to the M5 grid, computes union exposure
and per-config novelty (a trade is 'novel' if no other config is already in-market same-direction
at its entry). Mid prices, fixed 1.7p (matches the original EUR/USD table).
"""
import duckdb, numpy as np, pandas as pd
PIP=1e-4; SPREAD=1.7; SMA=9; K=1.0
CONFIGS=[("5min",288,4.0),("5min",288,10.0),("15min",96,6.0),("1h",24,6.0),("1h",24,10.0),("4h",12,20.0),("1D",8,20.0)]

def gen_iv(o,h,l,c,ts,basis,sd,MIN,tcap):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; entry=0.0; ei=0; stop=0.0; ext=0; ext_peak=0.0; iv=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        up_out=l[i]>up[i]; dn_out=h[i]<lo[i]
        if up_out: ext_peak=h[i] if ext!=1 else max(ext_peak,h[i]); ext=1
        elif dn_out: ext_peak=l[i] if ext!=-1 else min(ext_peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stop: ex=stop
                elif l[i]<=basis[i]: ex=basis[i]
            else:
                if l[i]<stop: ex=stop
                elif h[i]>=basis[i]: ex=basis[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex): iv.append((ts[ei],ts[i],pos)); pos=0
        if pos==0:
            ent=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i]:
                if 0.5*(ent-basis[i])/PIP-SPREAD>=MIN: pos=-1; entry=ent; ei=i+1; stop=ext_peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i]:
                if 0.5*(basis[i]-ent)/PIP-SPREAD>=MIN: pos=1; entry=ent; ei=i+1; stop=ext_peak
    return iv

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    grid=df.index.values.astype("datetime64[ns]"); G=len(grid)
    occ=np.zeros(G,dtype=int)                       # how many configs in-market at each M5 step
    dirsum=np.zeros(G,dtype=int)                    # net direction across configs
    name=lambda tf,m:f"{tf.replace('min','m').replace('1h','H1').replace('4h','H4').replace('1D','D1')}/{m:.0f}"
    masks={}
    allint=[]
    for tf,tcap,meat in CONFIGS:
        d=df.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index.values.astype("datetime64[ns]")
        basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
        iv=gen_iv(o,h,l,c,ts,basis,sd,meat,tcap)
        m=np.zeros(G,dtype=bool); md=np.zeros(G,dtype=int)
        for a,b,dr in iv:
            i0=np.searchsorted(grid,a); i1=np.searchsorted(grid,b)
            m[i0:i1+1]=True; md[i0:i1+1]=dr
        masks[name(tf,meat)]=(m,md,iv); occ+=m.astype(int); dirsum+=md
        allint.append((name(tf,meat),iv))
    inmkt=(occ>0)
    print(f"EUR/USD concurrent portfolio of {len(CONFIGS)} BB-reentry configs — {G} M5 steps, {(grid[-1]-grid[0])/np.timedelta64(365,'D'):.1f} yrs")
    print("="*84)
    print(f"TIME IN MARKET (union of all configs): {100*inmkt.mean():.1f}% of all M5 steps")
    print(f"  flat: {100*(occ==0).mean():.1f}%  |  exactly 1 config: {100*(occ==1).mean():.1f}%  |  2+ stacked: {100*(occ>=2).mean():.1f}%")
    print(f"  avg # configs in-market when not flat: {occ[inmkt].mean():.2f} of {len(CONFIGS)}")
    print()
    # per-config novelty: trade is 'novel' if at its entry no OTHER config already in-market same dir
    print(f"  {'config':>9} {'trades':>7} {'%time in mkt':>12} {'novel%':>7} {'overlap%':>9}")
    tot=0; nov=0
    for nm,(m,md,iv) in masks.items():
        novel=0
        for a,b,dr in iv:
            i0=np.searchsorted(grid,a)
            others=occ[i0]-1  # other configs in-market at this entry (minus self if self started exactly here; approx)
            # same-direction stacking check via dirsum sign
            if occ[i0]<=1 or np.sign(dirsum[i0])!=dr: novel+=1
        tot+=len(iv); nov+=novel
        print(f"  {nm:>9} {len(iv):>7} {100*m.mean():>11.1f}% {100*novel/max(1,len(iv)):>6.0f}% {100*(1-novel/max(1,len(iv))):>8.0f}%")
    print(f"\n  PORTFOLIO: {tot} total config-trades, ~{100*nov/tot:.0f}% novel / ~{100*(1-nov/tot):.0f}% overlapping")
    print(f"  (sum of independent exposures {sum(m.mean() for m,_,_ in masks.values())*100:.0f}% vs actual union {100*inmkt.mean():.1f}% -> overlap compresses it)")

if __name__=="__main__": main()
