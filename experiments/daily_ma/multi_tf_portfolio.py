"""
multi_tf_portfolio.py — GO-LIVE validation of the FULL deployable book: BB re-entry fade,
opposite-band exit, across the 4 proven timeframes (M5,M15,H1,H4) x 12 pairs, run CONCURRENTLY.
Sequential engine per (pair,TF); per-TF meat thresholds from the proven sweeps. 5.3y M15<-M5 mid,
per-pair MEDIAN real spread. Reports COMBINED portfolio PIPS/DAY + every gate + per-TF contribution.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0; TCAP_BARS={"5min":288,"15min":96,"1h":24,"4h":12}
TFS_MEAT=[("5min",4.0),("15min",6.0),("1h",6.0),("4h",10.0)]
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}

def backtest(o,h,l,c,ts,basis,sd,fixed_sp,meat,tcap,pip):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0; tr=[]
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
            if not np.isnan(ex): tr.append((ts[ei],pos*(ex-ent)/pip-fixed_sp,(ts[i]-ts[ei])/np.timedelta64(1,'D'))); pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(e-basis[i])/pip-fixed_sp>=meat: pos=-1; ent=e; ei=i+1; stp=peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-e)/pip-fixed_sp>=meat: pos=1; ent=e; ei=i+1; stp=peak
    return tr

def gates(tr,span,rng,label):
    t=sorted(tr,key=lambda x:x[0]); pnl=np.array([p for _,p,_ in t]); ets=np.array([x for x,_,_ in t])
    hold=np.array([hd for _,_,hd in t]); is_cut=ets[int(len(ets)*0.6)]; oos=pnl[ets>=is_cut]; isn=pnl[ets<is_cut]
    folds=np.array_split(pnl,4); fppd=[f.sum()/(span/4) for f in folds]
    null=np.array([(pnl*rng.choice([-1.,1.],len(pnl))).mean() for _ in range(5000)]); mcp=(np.abs(null)>=abs(pnl.mean())).mean()
    sqn=np.sqrt(len(pnl))*pnl.mean()/pnl.std(); tstat=pnl.mean()/(pnl.std()/np.sqrt(len(pnl)))
    print(f"  {label}")
    print(f"    PIPS/DAY: all {pnl.sum()/span:+.1f} | IS {isn.sum()/(span*0.6):+.1f} | OOS {oos.sum()/(span*0.4):+.1f}   (per-trade {pnl.mean():+.2f}p, WR {100*(pnl>0).mean():.0f}%)")
    print(f"    trades {len(pnl)} ({len(pnl)/span:.2f}/day), avg hold {hold.mean():.2f}d")
    print(f"    GATES: MC p={mcp:.4f} {'PASS' if mcp<0.05 else 'FAIL'} | SQN={sqn:.2f} {'PASS' if sqn>1 else 'FAIL'} | t={tstat:.1f} {'PASS' if abs(tstat)>2 else 'FAIL'}")
    print(f"    WALK-FWD (4 folds pips/day): [{' '.join(f'{x:+.1f}' for x in fppd)}]  {'ALL+ PASS' if all(x>0 for x in fppd) else 'FAIL'}")

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    med_sp={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med_sp[p]=float(np.nanmedian(s.s.values)/pip)
    combined=[]; per_tf={tf:[] for tf,_ in TFS_MEAT}; spans=[]
    for p,pip in PAIRS.items():
        base=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
        base["timestamp"]=pd.to_datetime(base["timestamp"],utc=True); base=base.set_index("timestamp")
        for tf,meat in TFS_MEAT:
            d=base.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
            basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
            tr=backtest(o,h,l,c,ts,basis,sd,med_sp[p],meat,TCAP_BARS[tf],pip)
            combined+=tr; per_tf[tf]+=tr
        spans.append((base.index[-1]-base.index[0]).days); del base; gc.collect()
    span=np.mean(spans)
    print("="*92); print(f"FULL BOOK — 4 TFs (M5,M15,H1,H4) x 12 pairs, opp-band exit, real spread, ~{span:.0f} days"); print("="*92)
    gates(combined,span,rng,"COMBINED PORTFOLIO (all 4 TFs concurrent)")
    print("\n  per-timeframe contribution (pips/day):")
    for tf,meat in TFS_MEAT:
        pnl=np.array([p for _,p,_ in per_tf[tf]])
        print(f"    {tf:>5} (meat {meat:.0f}): {pnl.sum()/span:+6.1f} pips/day  | {len(pnl)} trades ({len(pnl)/span:.2f}/day) | per-trade {pnl.mean():+.2f}p")

if __name__=="__main__": main()
