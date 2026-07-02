"""
eurusd_d1_sma9_clear.py — EUR/USD DAILY trend-follow on SMA9 "bar fully clears the MA".
Position = +1 if the whole daily bar is above SMA9 (low>SMA9), -1 if fully below (high<SMA9),
0 if the bar touches the MA. Signal on bar close -> act next bar open (causal). Hold while bars
stay clear; flat the moment a bar touches. Net of spread per transaction. Single pair, D1.
"""
import duckdb, numpy as np, pandas as pd

PIP=0.0001; SPREAD=1.7   # EUR_USD fixed spread (pips), per the first-touch PAIRS table
SMA=9

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values; h=d.high.values; l=d.low.values; c=d.close.values; ts=d.index
    n=len(c); sma=pd.Series(c).rolling(SMA).mean().values
    # target position from each closed bar
    tgt=np.zeros(n)
    for i in range(n):
        if np.isnan(sma[i]): continue
        if l[i]>sma[i]: tgt[i]=1
        elif h[i]<sma[i]: tgt[i]=-1
    # walk: position at bar i+1 = tgt[i]; trade on changes, fill at open[i+1]
    pos=0; entry=0.0; ei=0; trades=[]
    for i in range(SMA,n-1):
        want=tgt[i]
        if want!=pos:
            px=o[i+1]
            if pos!=0:                                  # close current
                pnl=pos*(px-entry)/PIP - SPREAD
                trades.append((ts[ei],ts[i+1],pos,pnl,(ts[i+1]-ts[ei]).days))
            if want!=0:                                 # open new
                entry=px; ei=i+1
            pos=want
    if pos!=0: pnl=pos*(c[-1]-entry)/PIP-SPREAD; trades.append((ts[ei],ts[-1],pos,pnl,(ts[-1]-ts[ei]).days))
    T=pd.DataFrame(trades,columns=["entry_ts","exit_ts","dir","pnl","days"])
    span=(ts[-1]-ts[0]).days/365.25
    print(f"EUR/USD D1 SMA9 bar-clear trend-follow — {len(d)} daily bars, {span:.1f} yrs, {len(T)} trades")
    print(f"  net total: {T.pnl.sum():+.0f} pips   per-trade {T.pnl.mean():+.1f}p   WR {100*(T.pnl>0).mean():.0f}%   per-year {T.pnl.sum()/span:+.0f}p")
    print(f"  longs {len(T[T.dir>0])} ({T[T.dir>0].pnl.mean():+.1f}p)  shorts {len(T[T.dir<0])} ({T[T.dir<0].pnl.mean():+.1f}p)")
    print(f"  avg hold {T.days.mean():.0f}d (median {T.days.median():.0f}d, max {T.days.max():.0f}d)   winners {T[T.pnl>0].pnl.mean():+.0f}p / losers {T[T.pnl<=0].pnl.mean():+.0f}p")
    eq=T.pnl.cumsum(); dd=(eq-eq.cummax()).min()
    print(f"  equity: max drawdown {dd:.0f} pips ({dd/T.pnl.mean():.1f}x avg trade)")
    # MC: shuffle signs
    rng=np.random.default_rng(0); o_=T.pnl.mean()
    null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(5000)])
    print(f"  MC sign-flip P(|null|>=obs) = {(np.abs(null)>=abs(o_)).mean():.3f}")
    bh=(c[-1]-c[SMA])/PIP
    print(f"  buy & hold EUR/USD same span: {bh:+.0f} pips (strategy is {'directional, not just long-drift' if abs(T.pnl.sum())>abs(bh) else 'comparable to drift'})")

if __name__=="__main__": main()
