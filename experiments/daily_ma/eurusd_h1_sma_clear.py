"""
eurusd_h1_sma_clear.py — EUR/USD HOURLY trend-follow on SMA(N) "bar fully clears the MA".
Entry: bar completely above SMA(N) (low>SMA) -> long; completely below (high<SMA) -> short.
Exit modes: TOUCH (exit when a bar's wick touches the MA, the user's spec) vs CLOSE-beyond
(exit only when a bar closes through). Signal on close -> act next open, net spread. H1, EUR/USD.
"""
import duckdb, numpy as np, pandas as pd

PIP=0.0001; SPREAD=1.7

def run(o,h,l,c,ts,sma,exit_mode):
    n=len(c); pos=0; entry=0.0; ei=0; tr=[]
    start=int(np.argmax(~np.isnan(sma)))+1
    for i in range(start,n-1):
        np_=pos
        if pos==1:
            hit = (l[i]<=sma[i]) if exit_mode=="touch" else (c[i]<sma[i])
            if hit: np_=0
        elif pos==-1:
            hit = (h[i]>=sma[i]) if exit_mode=="touch" else (c[i]>sma[i])
            if hit: np_=0
        if np_==0:
            if l[i]>sma[i]: np_=1
            elif h[i]<sma[i]: np_=-1
        if np_!=pos:
            px=o[i+1]
            if pos!=0: tr.append((pos,pos*(px-entry)/PIP-SPREAD,(ts[i+1]-ts[ei]).total_seconds()/3600))
            if np_!=0: entry=px; ei=i+1
            pos=np_
    if pos!=0: tr.append((pos,pos*(c[-1]-entry)/PIP-SPREAD,(ts[-1]-ts[ei]).total_seconds()/3600))
    return pd.DataFrame(tr,columns=["dir","pnl","hrs"])

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD H1 trend-follow 'bar clears SMA' — {len(d)} H1 bars, {span:.1f} yrs, buy&hold {(c[-1]-c[50])/PIP:+.0f}p")
    print("="*92)
    print(f"  {'cfg':>18} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(h)':>8} {'maxDD':>8} {'MCp':>6}")
    def show(name,T):
        if len(T)<5: print(f"  {name:>18}  too few"); return
        eq=T.pnl.cumsum(); dd=(eq-eq.cummax()).min(); obs=T.pnl.mean()
        null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(3000)])
        print(f"  {name:>18} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} {100*(T.pnl>0).mean():>4.0f}% "
              f"{T.pnl.sum()/span:>+7.0f} {T.hrs.mean():>7.0f}h {dd:>+8.0f} {(np.abs(null)>=abs(obs)).mean():>6.3f}")
    # the user's spec: SMA13, touch exit  (+ close-beyond comparison)
    s13=pd.Series(c).rolling(13).mean().values
    show("SMA13 touch", run(o,h,l,c,ts,s13,"touch"))
    show("SMA13 close-beyond", run(o,h,l,c,ts,s13,"close"))
    # period sweep with the better (close-beyond) exit
    for N in (20,50,100):
        show(f"SMA{N} close-beyond", run(o,h,l,c,ts,pd.Series(c).rolling(N).mean().values,"close"))

if __name__=="__main__": main()
