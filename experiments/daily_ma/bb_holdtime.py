"""bb_holdtime.py — distribution of BB-fade trade HOLD TIMES (backtest, 4 TF x 12 pairs)."""
import sys, numpy as np, pandas as pd, duckdb, gc
sys.path.insert(0,".")
from lib.bb_fade import backtest
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("M5","5min",5,288,4.0),("M15","15min",15,96,6.0),("H1","1h",60,24,6.0),("H4","4h",240,12,10.0)]

def fmt(mins):
    if mins<60: return f"{mins:.0f}m"
    if mins<1440: return f"{mins/60:.1f}h"
    return f"{mins/1440:.1f}d"

def main():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    allmin=[]
    print(f"  {'TF':>4} {'trades':>7} {'same-bar%':>9} {'median':>8} {'mean':>8} {'p75':>7} {'p90':>7} {'p95':>8} {'max':>8}")
    for tfn,rule,bmin,tcap,meat in TF:
        hold=[]
        for p,pip in PAIRS.items():
            b=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
            b["timestamp"]=pd.to_datetime(b["timestamp"],utc=True); b=b.set_index("timestamp")
            d=b.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            for (eb,xb,dr,pnl) in backtest(d.open.values,d.high.values,d.low.values,d.close.values,pip,med[p],meat,tcap):
                hold.append((xb-eb)*bmin)
            del b,d; gc.collect()
        h=np.array(hold,float); allmin+=hold
        sb=100*np.mean(h<=0.0) if len(h) else 0   # exit on entry bar
        if len(h): print(f"  {tfn:>4} {len(h):>7} {sb:>8.0f}% {fmt(np.median(h)):>8} {fmt(h.mean()):>8} {fmt(np.percentile(h,75)):>7} {fmt(np.percentile(h,90)):>7} {fmt(np.percentile(h,95)):>8} {fmt(h.max()):>8}")
    a=np.array(allmin,float)
    print(f"\n=== ALL {len(a)} trades combined ===")
    print(f"  same-bar exits: {100*np.mean(a<=0):.0f}%   median {fmt(np.median(a))}   mean {fmt(a.mean())}")
    print("  percentiles:", " ".join(f"p{q}={fmt(np.percentile(a,q))}" for q in [10,25,50,75,90,95,99]))
    print("\n  histogram (hold time):")
    bins=[(0,1,"<=entry bar"),(1,15,"<15m"),(15,60,"15-60m"),(60,240,"1-4h"),(240,1440,"4-24h"),(1440,1e9,">1d")]
    for lo,hi,lbl in bins:
        c=int(np.sum((a>=lo)&(a<hi))) if lo>0 else int(np.sum(a<=0))
        print(f"    {lbl:>12}: {c:>6} ({100*c/len(a):>4.1f}%) {'#'*int(60*c/len(a))}")

if __name__=="__main__": main()
