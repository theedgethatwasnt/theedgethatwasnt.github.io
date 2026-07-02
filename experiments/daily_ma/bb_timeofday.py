"""
bb_timeofday.py — does HOUR-OF-DAY / SESSION conditioning hone the BB-fade edge?
Backtest the 4-TF book (12 pairs, real median spread), tag each trade by entry hour-of-day (UTC),
session, and day-of-week. Report expectancy per bucket; then IS-pick the positive hours and validate
on OOS (don't overfit). Sessions (UTC): Asia 22-07, London 07-12, Overlap 12-16, NY 16-21, Thin 21-22.
"""
import sys, numpy as np, pandas as pd, duckdb, gc
sys.path.insert(0,".")
from lib.bb_fade import backtest
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("5min",288,4.0),("15min",96,6.0),("1h",24,6.0),("4h",12,10.0)]
def sess(h):
    if 22<=h or h<7: return "Asia"
    if 7<=h<12: return "London"
    if 12<=h<16: return "Overlap"
    if 16<=h<21: return "NY"
    return "Thin"

def main():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    rows=[]
    for rule,tcap,meat in TF:
        for p,pip in PAIRS.items():
            b=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
            b["timestamp"]=pd.to_datetime(b["timestamp"],utc=True); b=b.set_index("timestamp")
            d=b.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna(); idx=d.index
            for (eb,xb,dr,pnl) in backtest(d.open.values,d.high.values,d.low.values,d.close.values,pip,med[p],meat,tcap):
                t=idx[eb]; rows.append((t,t.hour,t.dayofweek,sess(t.hour),pnl))
            del b,d; gc.collect()
    R=pd.DataFrame(rows,columns=["ts","hour","dow","sess","pnl"]).sort_values("ts")
    print(f"=== {len(R)} trades. By SESSION (UTC): ===")
    print(f"  {'session':>9} {'trades':>7} {'expect':>8} {'WR%':>5} {'%of pnl':>8}")
    tot=R.pnl.sum()
    for s in ["Asia","London","Overlap","NY","Thin"]:
        x=R[R.sess==s]
        if len(x): print(f"  {s:>9} {len(x):>7} {x.pnl.mean():>+8.2f} {100*(x.pnl>0).mean():>4.0f}% {100*x.pnl.sum()/tot:>7.0f}%")
    print("\n=== By HOUR-OF-DAY (UTC), expectancy: ===")
    for h in range(24):
        x=R[R.hour==h]
        if len(x)>=30: print(f"  {h:02d}:00  n={len(x):>5}  exp={x.pnl.mean():>+6.2f}  {'+'*int(max(0,x.pnl.mean()*3))}{'-'*int(max(0,-x.pnl.mean()*3))}")
    print("\n=== By DAY-OF-WEEK (0=Mon): ===")
    for dw in range(5):
        x=R[R.dow==dw]
        if len(x): print(f"  dow{dw} n={len(x):>5} exp={x.pnl.mean():>+6.2f}")
    # IS-pick positive hours, validate OOS
    cut=R.ts.quantile(0.6)
    IS=R[R.ts<cut]; OOS=R[R.ts>=cut]
    goodh=set(IS.groupby("hour").pnl.mean()[lambda s:s>IS.pnl.mean()].index)
    base_oos=OOS.pnl.mean(); filt_oos=OOS[OOS.hour.isin(goodh)].pnl.mean()
    print(f"\n=== IS-picked good hours (exp>IS-mean): {sorted(goodh)} ===")
    print(f"  OOS expectancy: ALL {base_oos:+.2f}  | GOOD-HOURS-ONLY {filt_oos:+.2f}  ({'HELPS' if filt_oos>base_oos else 'no help'})  | trades kept {100*OOS.hour.isin(goodh).mean():.0f}%")

if __name__=="__main__": main()
