"""
ic_markets_analysis.py — how much better on IC Markets (Raw/ECN spread, 500:1, no-FIFO)?
THREE levers, dominant first:
 (1) SPREAD — re-run the 4-BB book at ECN all-in cost levels vs the OANDA per-pair median baseline.
     The edge is thin (~+1p) and spread enters BOTH the meat gate and the P&L, so this is decisive.
 (2) no-FIFO pooling — capital efficiency (computed separately ~1.5-2x).
 (3) 500:1 leverage — margin ceiling moot (DD/blow-up cap still binds); capital-efficiency only.
"""
import sys, numpy as np, pandas as pd, duckdb, gc
sys.path.insert(0,".")
from lib.bb_fade import backtest
DDF=0.10
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("M5","5min",288,4.0),("M15","15min",96,6.0),("H1","1h",24,6.0),("H4","4h",12,10.0)]

def run_at(spread_fn, label):
    con=duckdb.connect(); tot=0.0; n=0; span=0
    for tfn,rule,tcap,meat in TF:
        for p,pip in PAIRS.items():
            b=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/s5_ohlc/{p}_S5_BA.parquet' ORDER BY timestamp").df()
            b["timestamp"]=pd.to_datetime(b["timestamp"],utc=True);b=b.set_index("timestamp")
            d=b.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            sp=spread_fn(p)
            tr=backtest(d.open.values,d.high.values,d.low.values,d.close.values,pip,sp,meat,tcap)
            tot+=sum(t[3] for t in tr); n+=len(tr)
            span=max(span,(b.index[-1]-b.index[0]).days); del b,d; gc.collect()
    return tot/span, n/span, tot/max(n,1)

def main():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    print("avg OANDA median spread across 12 pairs: %.2fp"%np.mean(list(med.values())))
    print("\n(1) SPREAD lever — 4-BB book pips/day at different all-in costs:")
    print(f"  {'scenario':>22} {'spread model':>22} {'pips/day':>9} {'trades/d':>9} {'p/trade':>8}")
    scenarios=[
      ("OANDA retail (baseline)", lambda p: med[p], "per-pair median (1.3-4.1)"),
      ("IC Raw ~1.0p all-in",     lambda p: 1.0,    "1.0p flat"),
      ("IC Raw ~0.7p all-in",     lambda p: 0.7,    "0.7p flat"),
      ("IC Raw ~0.5p all-in",     lambda p: 0.5,    "0.5p flat"),
    ]
    base=None
    for nm,fn,desc in scenarios:
        pd_,td,pt=run_at(fn,nm)
        if base is None: base=pd_
        print(f"  {nm:>22} {desc:>22} {pd_:>+9.1f} {td:>9.1f} {pt:>+8.2f}   ({pd_/base:.2f}x baseline)")

if __name__=="__main__": main()
