"""
eurusd_d1_sma_sweep.py — EUR/USD D1 trend-follow, MA period sweep, CLOSE-BEYOND exit.
Entry: bar completely above SMA(N) (low>SMA) -> long; completely below (high<SMA) -> short.
Exit/flip: only when a bar CLOSES on the other side of the MA (ignores intrabar wicks -> kills
the touch-whipsaw). Signal on close -> act next open. Net spread per transition. D1, EUR/USD.
"""
import duckdb, numpy as np, pandas as pd

PIP=0.0001; SPREAD=1.7

def run(o,h,l,c,ts,sma):
    n=len(c); pos=0; entry=0.0; ei=0; tr=[]
    start=int(np.argmax(~np.isnan(sma)))+1
    for i in range(start,n-1):
        np_=pos
        if pos==1 and c[i]<sma[i]: np_=0           # close-beyond exit
        elif pos==-1 and c[i]>sma[i]: np_=0
        if np_==0:                                 # (re-)entry on a full bar-clear
            if l[i]>sma[i]: np_=1
            elif h[i]<sma[i]: np_=-1
        if np_!=pos:
            px=o[i+1]
            if pos!=0: tr.append((pos,pos*(px-entry)/PIP-SPREAD,(ts[i+1]-ts[ei]).days))
            if np_!=0: entry=px; ei=i+1
            pos=np_
    if pos!=0: tr.append((pos,pos*(c[-1]-entry)/PIP-SPREAD,(ts[-1]-ts[ei]).days))
    return pd.DataFrame(tr,columns=["dir","pnl","days"])

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    bh=(c[-1]-c[50])/PIP
    print(f"EUR/USD D1 trend-follow, close-beyond exit, MA sweep — {len(d)} daily bars, {span:.1f} yrs")
    print(f"buy&hold over span: {bh:+.0f} pips\n"+"="*86)
    print(f"  {'MA':>4} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'avg hold':>9} {'maxDD':>8} {'MCp':>6}")
    for N in (9,13,20,50):
        sma=pd.Series(c).rolling(N).mean().values
        T=run(o,h,l,c,ts,sma)
        if len(T)<5: print(f"  {N:>4}  too few"); continue
        eq=T.pnl.cumsum(); dd=(eq-eq.cummax()).min()
        obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(4000)])
        mcp=(np.abs(null)>=abs(obs)).mean()
        print(f"  {N:>4} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.1f} {100*(T.pnl>0).mean():>4.0f}% "
              f"{T.pnl.sum()/span:>+7.0f} {T.days.mean():>7.0f}d {dd:>+8.0f} {mcp:>6.3f}")

if __name__=="__main__": main()
