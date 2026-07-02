"""
eurusd_h1_fade_to_ma.py — EUR/USD H1 mean-reversion: fade a PEAKED extension back to SMA7.
Let price extend fully clear of SMA7 (don't fade the run). When it stalls (first lower-high in an
up-extension / higher-low in a down-extension), enter the FADE at next open. Exit: target = price
returns to SMA7, OR stop = a new extreme beyond the extension peak (run resumed), OR time cap.
Net spread, EUR/USD H1. Reports the target distance (does the reversion clear the spread?).
"""
import duckdb, numpy as np, pandas as pd

PIP=0.0001; SPREAD=1.7; SMA=7; TCAP=24   # time cap in H1 bars

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index
    n=len(c); sma=pd.Series(c).rolling(SMA).mean().values
    pos=0; ext_dir=0; ext_peak=0.0; entered=False; entry=0.0; ei=0; stop=0.0
    tr=[]
    for i in range(SMA+1,n-1):
        if np.isnan(sma[i]): continue
        above=l[i]>sma[i]; below=h[i]<sma[i]
        # update extension regime
        if above:
            if ext_dir!=1: ext_dir=1; ext_peak=h[i]; entered=False
            else: ext_peak=max(ext_peak,h[i])
        elif below:
            if ext_dir!=-1: ext_dir=-1; ext_peak=l[i]; entered=False
            else: ext_peak=min(ext_peak,l[i])
        else:
            ext_dir=0
        # manage open fade
        if pos!=0:
            exitpx=np.nan
            if pos==-1:                                   # short: target=SMA, stop=new high>peak
                if h[i]>stop: exitpx=stop                 # run resumed (stop)
                elif l[i]<=sma[i]: exitpx=sma[i]          # reverted to mean (target)
            else:
                if l[i]<stop: exitpx=stop
                elif h[i]>=sma[i]: exitpx=sma[i]
            if np.isnan(exitpx) and (i-ei)>=TCAP: exitpx=c[i]   # time cap
            if not np.isnan(exitpx):
                tr.append((pos,pos*(exitpx-entry)/PIP-SPREAD,i-ei)); pos=0
        # entry on peak (first stall while extended)
        if pos==0 and not entered and i>0:
            if ext_dir==1 and above and h[i]<h[i-1]:      # up-extension peaked -> fade short
                pos=-1; entry=o[i+1]; ei=i+1; stop=ext_peak; entered=True
            elif ext_dir==-1 and below and l[i]>l[i-1]:   # down-extension peaked -> fade long
                pos=1; entry=o[i+1]; ei=i+1; stop=ext_peak; entered=True
    T=pd.DataFrame(tr,columns=["dir","pnl","bars"])
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD H1 fade-peaked-extension-to-SMA{SMA} — {len(d)} H1 bars, {span:.1f} yrs, {len(T)} trades")
    obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(4000)])
    print(f"  net {T.pnl.sum():+.0f}p  per-trade {obs:+.2f}p  WR {100*(T.pnl>0).mean():.0f}%  per-yr {T.pnl.sum()/span:+.0f}p  hold {T.bars.mean():.0f}h  MCp {(np.abs(null)>=abs(obs)).mean():.3f}")
    print(f"  winners {T[T.pnl>0].pnl.mean():+.1f}p / losers {T[T.pnl<=0].pnl.mean():+.1f}p   ({len(T[T.dir<0])} short-fades, {len(T[T.dir>0])} long-fades)")

if __name__=="__main__": main()
