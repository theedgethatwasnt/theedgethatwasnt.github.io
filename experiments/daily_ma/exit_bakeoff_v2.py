"""
exit_bakeoff_v2.py — V2 exit refinements on the live BB-fade book (4 TFs x 12 pairs, real spread).
Baseline = opp_band (current live exit, +43.7 pips/day). Test combinations:
  base         : opp-band target + extension stop + full time cap (current)
  +gv50        : same, PLUS a giveback trail — exit if price retraces 50% of MFE (lock profit if it
                 reverts most of the way but stalls short of the far band)
  +cap4 / +cap5: opp-band target but hard 4- / 5-bar time cap (user: most action in 4-5 bars)
  +gv50_cap5   : both combined
Reports full-book pips/day + OOS + MC per variant. Sequential engine, per-pair median real spread.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("5min",288,4.0),("15min",96,6.0),("1h",24,6.0),("4h",12,10.0)]

def bt(o,h,l,c,pip,sp,meat,tcap,variant):
    n=len(c); cl=np.array(c); basis=np.full(n,np.nan); sd=np.full(n,np.nan)
    for i in range(SMA-1,n): w=cl[i-SMA+1:i+1]; basis[i]=w.mean(); sd[i]=w.std()
    up=basis+K*sd; lo=basis-K*sd
    cap = 4 if variant=="+cap4" else 5 if variant in("+cap5","+gv50_cap5") else tcap
    gv = variant in("+gv50","+gv50_cap5")
    pos=0;ei=0;ent=0;stp=0;ext=0;peak=0;mfe=0; out=[]
    for i in range(SMA,n-1):
        if np.isnan(basis[i]): continue
        uo=l[i]>up[i];do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                mfe=max(mfe,(ent-l[i])/pip)
                if h[i]>stp: ex=stp
                elif l[i]<=lo[i]: ex=lo[i]
                elif gv and mfe>=2 and (ent-c[i])/pip<=0.5*mfe: ex=c[i]      # gave back half of MFE
            else:
                mfe=max(mfe,(h[i]-ent)/pip)
                if l[i]<stp: ex=stp
                elif h[i]>=up[i]: ex=up[i]
                elif gv and mfe>=2 and (c[i]-ent)/pip<=0.5*mfe: ex=c[i]
            if np.isnan(ex) and (i-ei)>=cap: ex=c[i]
            if not np.isnan(ex): out.append(pos*(ex-ent)/pip-sp); pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-sp>=meat: pos=-1;ent=e;ei=i+1;stp=peak;mfe=0
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-sp>=meat: pos=1;ent=e;ei=i+1;stp=peak;mfe=0
    return out

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    cells=[]; span=0   # precompute resampled arrays ONCE per (pair,tf)
    for p,pip in PAIRS.items():
        b=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
        b["timestamp"]=pd.to_datetime(b["timestamp"],utc=True); b=b.set_index("timestamp")
        span=max(span,(b.index[-1]-b.index[0]).days)
        for tf,tc,meat in TF:
            d=b.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            cells.append((d.open.values,d.high.values,d.low.values,d.close.values,pip,med[p],meat,tc))
        del b; gc.collect()
    print(f"V2 exit bake-off — full book (4 TF x 12 pairs), real spread, {span}d. Baseline opp_band=+43.7 pd")
    print("="*72)
    print(f"  {'variant':>12} {'pips/day':>9} {'OOS pd':>8} {'p/trade':>8} {'WR':>5} {'trades':>7} {'MCp':>6}")
    for variant in ["base","+gv50","+cap4","+cap5","+gv50_cap5"]:
        allp=[]
        for (o,h,l,c,pip,sp,meat,tc) in cells:
            allp+=bt(o,h,l,c,pip,sp,meat,tc,variant)
        a=np.array(allp); oos=a[int(len(a)*0.6):]
        null=np.array([(a*rng.choice([-1.,1.],len(a))).mean() for _ in range(1500)])
        print(f"  {variant:>12} {a.sum()/span:>+9.1f} {oos.sum()/(span*0.4):>+8.1f} {a.mean():>+8.2f} {100*(a>0).mean():>4.0f}% {len(a):>7} {(np.abs(null)>=abs(a.mean())).mean():>6.3f}",flush=True)

if __name__=="__main__": main()
