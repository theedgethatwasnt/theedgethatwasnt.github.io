"""
capital_analysis.py — live logistics for the 4-TF BB re-entry book.
(1) Account separation: peak CONCURRENT positions per instrument across TFs (do the 4 TFs collide
    on the same pair on one netted account?) + peak concurrent per TF (margin driver per account).
(2) Capital: portfolio equity DRAWDOWN (pips) per TF + combined, worst single trade, translated to
    a capital recommendation (DD buffer + margin) at risk-parity $1/pip sizing.
Sequential engine, opp-band exit, 5.3y M15<-M5, per-pair median real spread.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0; TCAP_BARS={"5min":288,"15min":96,"1h":24,"4h":12}; TFS_MEAT=[("5min",4.0),("15min",6.0),("1h",6.0),("4h",10.0)]
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}

def backtest(o,h,l,c,ts,basis,sd,sp,meat,tcap,pip):
    n=len(c); up=basis+K*sd; lo=basis-K*sd; pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0; tr=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        uo=l[i]>up[i]; do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stp: ex=stp
                elif l[i]<=lo[i]: ex=lo[i]
            else:
                if l[i]<stp: ex=stp
                elif h[i]>=up[i]: ex=up[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex): tr.append((ts[ei],ts[i],pos*(ex-ent)/pip-sp)); pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(e-basis[i])/pip-sp>=meat: pos=-1; ent=e; ei=i+1; stp=peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-e)/pip-sp>=meat: pos=1; ent=e; ei=i+1; stp=peak
    return tr

def peak_concurrent(intervals):  # list of (start,end) np.datetime64
    ev=[]
    for a,b in intervals: ev.append((a,1)); ev.append((b,-1))
    ev.sort(); cur=mx=0
    for _,d in ev: cur+=d; mx=max(mx,cur)
    return mx

def maxdd(trades):  # by exit time, cumulative pips
    t=sorted(trades,key=lambda x:x[1]); eq=np.cumsum([p for _,_,p in t]); peak=np.maximum.accumulate(eq)
    return (eq-peak).min(), eq[-1]

def main():
    con=duckdb.connect(); med_sp={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med_sp[p]=float(np.nanmedian(s.s.values)/pip)
    by_tf={tf:[] for tf,_ in TFS_MEAT}; by_tf_pair={}; allt=[]; span=0
    for p,pip in PAIRS.items():
        base=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
        base["timestamp"]=pd.to_datetime(base["timestamp"],utc=True); base=base.set_index("timestamp")
        span=max(span,(base.index[-1]-base.index[0]).days)
        for tf,meat in TFS_MEAT:
            d=base.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
            basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
            tr=backtest(o,h,l,c,ts,basis,sd,med_sp[p],meat,TCAP_BARS[tf],pip)
            by_tf[tf]+=tr; by_tf_pair[(tf,p)]=tr; allt+=tr
        del base; gc.collect()
    print("="*88); print(f"LIVE LOGISTICS — 4-TF BB re-entry book, ~{span} days ({span/365:.1f}y), risk-parity $1/pip"); print("="*88)
    # (1) account separation: same-pair collisions across TFs
    print("\n(1) ACCOUNT SEPARATION — peak concurrent positions on the SAME pair across the 4 TFs:")
    worst=0
    for p in PAIRS:
        iv=[(a,b) for tf,_ in TFS_MEAT for (a,b,_) in by_tf_pair[(tf,p)]]
        pc=peak_concurrent(iv); worst=max(worst,pc)
    print(f"    worst-case simultaneous positions on one instrument across TFs: {worst}")
    print(f"    -> OANDA NETS per instrument, so {worst}>1 means the TFs WOULD collide on one account.")
    # per-TF (per account) concurrency across the 12 pairs
    print("\n    peak concurrent positions WITHIN each TF (across 12 pairs = that account's margin driver):")
    for tf,_ in TFS_MEAT:
        iv=[(a,b) for (a,b,_) in by_tf[tf]]; print(f"      {tf:>5}: {peak_concurrent(iv)} of 12 pairs open at once")
    allpc=peak_concurrent([(a,b) for (a,b,_) in allt])
    print(f"    peak concurrent across ALL 48 instances (4 TF x 12 pairs): {allpc}")
    # (2) drawdown + capital
    print("\n(2) DRAWDOWN (pips, $1/pip) & worst trade:")
    for tf,_ in TFS_MEAT:
        dd,tot=maxdd(by_tf[tf]); print(f"    {tf:>5}: maxDD {dd:7.0f}p | total {tot:+8.0f}p | worst trade {min(p for _,_,p in by_tf[tf]):.0f}p")
    cdd,ctot=maxdd(allt); worst_tr=min(p for _,_,p in allt)
    print(f"    COMBINED: maxDD {cdd:.0f}p | total {ctot:+.0f}p ({ctot/span:+.1f} pips/day) | worst single trade {worst_tr:.0f}p")
    print(f"\n  CAPITAL TRANSLATION (at $1/pip = ~10k units/major, ~$400 margin/position @30:1):")
    print(f"    income $1/pip: {ctot/span*1:.0f} $/day | maxDD ${abs(cdd):.0f} | peak margin ~${allpc*400}")
    print(f"    DD-safe capital (maxDD <=20% of equity): ${abs(cdd)*5:.0f}  | margin-safe (peak <=45%): ${allpc*400/0.45:.0f}")

if __name__=="__main__": main()
