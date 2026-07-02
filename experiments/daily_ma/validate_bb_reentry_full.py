"""
validate_bb_reentry_full.py — full validation of the BB re-entry fade + half-distance meat filter.
Part A: 12 pairs x {M5,M15,H1} x REAL per-bar spread (ask_c-bid_c from S5 BA, ~1.5y) x 3-fold
        walk-forward. The strict test (real costs, breadth, temporal robustness).
Part B: 12 pairs x 5.3y mid (m5_ohlc), fixed cost = each pair's MEDIAN real spread. Breadth over
        the long period.
Meat thresholds PRE-FIXED from the EUR/USD IS finding (M5=4,M15=6,H1=6) — applied to all pairs/
folds, no per-pair tuning (SOP R8). One pair in memory at a time.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TFS=[("5min",288,4.0),("15min",96,6.0),("1h",24,6.0)]   # (rule, time_cap_bars, meat_pips)

def gen(o,h,l,c,sp,ts,basis,sd,MIN,tcap):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; entry=0.0; ei=0; es=0.0; stop=0.0; ext=0; ext_peak=0.0; pipv=1.0; out=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]) or np.isnan(sp[i]): continue
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
            if not np.isnan(ex):
                net=pos*(ex-entry)/PIP-0.5*(es+sp[i])   # mid pnl minus half(entry+exit) spread
                out.append((ts[ei],net)); pos=0
        if pos==0:
            ent=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i]:
                if 0.5*(ent-basis[i])/PIP-sp[i]>=MIN: pos=-1; entry=ent; ei=i+1; es=sp[i+1] if not np.isnan(sp[i+1]) else sp[i]; stop=ext_peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i]:
                if 0.5*(basis[i]-ent)/PIP-sp[i]>=MIN: pos=1; entry=ent; ei=i+1; es=sp[i+1] if not np.isnan(sp[i+1]) else sp[i]; stop=ext_peak
    return out

def stats(trades,span_yr,rng,nfold=3):
    if len(trades)<20: return None
    t=sorted(trades,key=lambda x:x[0]); pnl=np.array([p for _,p in t])
    folds=np.array_split(pnl,nfold); fm=[f.mean() for f in folds if len(f)]
    null=np.array([(pnl*rng.choice([-1.,1.],len(pnl))).mean() for _ in range(2000)])
    return dict(n=len(pnl),pt=pnl.mean(),wr=100*(pnl>0).mean(),pyr=pnl.sum()/span_yr,
                mcp=(np.abs(null)>=abs(pnl.mean())).mean(),folds=fm)

def main():
    global PIP
    con=duckdb.connect(); rng=np.random.default_rng(0)
    med_sp={}                          # (pair,tf) -> median real spread
    portA={tf:[] for tf,_,_ in TFS}; perpairA={tf:{} for tf,_,_ in TFS}
    print("PART A — REAL per-bar spread, 12 pairs, walk-forward (S5 BA ~1.5y)"); print("="*78)
    for pair,pip in PAIRS.items():
        PIP=pip
        s5=con.execute(f"SELECT timestamp,open,high,low,close,bid_c,ask_c FROM 'data/s5_ohlc/{pair}_S5_BA.parquet' ORDER BY timestamp").df()
        s5["timestamp"]=pd.to_datetime(s5["timestamp"],utc=True); s5=s5.set_index("timestamp")
        for tf,tcap,meat in TFS:
            d=s5.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last","bid_c":"last","ask_c":"last"}).dropna()
            o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values
            sp=((d.ask_c.values-d.bid_c.values)/pip); ts=d.index
            med_sp[(pair,tf)]=np.nanmedian(sp)
            basis=pd.Series(c).rolling(SMA).mean().values; sdv=pd.Series(c).rolling(SMA).std().values
            tr=gen(o,h,l,c,sp,ts,basis,sdv,meat,tcap)
            portA[tf]+=tr; perpairA[tf][pair]=np.mean([p for _,p in tr]) if len(tr)>=10 else np.nan
        del s5; gc.collect()
    spanA=1.48
    print(f"  {'TF':>5} {'trades':>7} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'MCp':>6} {'WF folds (p/t)':>22} {'pairs+':>7}")
    for tf,_,meat in TFS:
        s=stats(portA[tf],spanA,rng)
        pp=perpairA[tf]; npos=sum(1 for v in pp.values() if v>0); ntot=sum(1 for v in pp.values() if not np.isnan(v))
        if s: print(f"  {tf:>5} {s['n']:>7} {s['pt']:>+8.2f} {s['wr']:>4.0f}% {s['pyr']:>+7.0f} {s['mcp']:>6.3f}  "
                    f"[{' '.join(f'{x:+.1f}' for x in s['folds'])}]  {npos}/{ntot}")
    print("  per-pair p/trade (M5):", {k:round(v,1) for k,v in perpairA['5min'].items() if not np.isnan(v)})

    print("\nPART B — 5.3y mid, fixed cost = per-pair MEDIAN real spread"); print("="*78)
    portB={tf:[] for tf,_,_ in TFS}; perpairB={tf:{} for tf,_,_ in TFS}
    for pair,pip in PAIRS.items():
        PIP=pip
        m=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{pair}_M5.parquet' ORDER BY timestamp").df()
        m["timestamp"]=pd.to_datetime(m["timestamp"],utc=True); m=m.set_index("timestamp")
        for tf,tcap,meat in TFS:
            d=m.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index
            sp=np.full(len(c),med_sp.get((pair,tf),np.nan))   # fixed = median real spread
            basis=pd.Series(c).rolling(SMA).mean().values; sdv=pd.Series(c).rolling(SMA).std().values
            tr=gen(o,h,l,c,sp,ts,basis,sdv,meat,tcap)
            portB[tf]+=tr; perpairB[tf][pair]=np.mean([p for _,p in tr]) if len(tr)>=10 else np.nan
        del m; gc.collect()
    spanB=5.3
    print(f"  {'TF':>5} {'trades':>7} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'MCp':>6} {'WF folds (p/t)':>22} {'pairs+':>7}")
    for tf,_,meat in TFS:
        s=stats(portB[tf],spanB,rng)
        pp=perpairB[tf]; npos=sum(1 for v in pp.values() if v>0); ntot=sum(1 for v in pp.values() if not np.isnan(v))
        if s: print(f"  {tf:>5} {s['n']:>7} {s['pt']:>+8.2f} {s['wr']:>4.0f}% {s['pyr']:>+7.0f} {s['mcp']:>6.3f}  "
                    f"[{' '.join(f'{x:+.1f}' for x in s['folds'])}]  {npos}/{ntot}")
    print("  median real spread used (pips):", {k:round(med_sp[(k,'1h')],1) for k in PAIRS})

if __name__=="__main__": main()
