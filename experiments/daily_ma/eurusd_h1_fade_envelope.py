"""
eurusd_h1_fade_envelope.py — fade peaked extension, trend-adaptive MA (user refinement):
in an UPtrend the mean is the SMA7-of-LOWS (rising support); in a DOWNtrend the SMA7-of-HIGHS.
Up-extension = bar fully above SMA7(high) (strongly stretched up); peak (lower-high) -> fade short,
target = SMA7(low). Mirror down. Stop = extension peak; time cap. Net spread, EUR/USD H1.
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; SMA=7; TCAP=24

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index; n=len(c)
    maL=pd.Series(l).rolling(SMA).mean().values; maH=pd.Series(h).rolling(SMA).mean().values
    pos=0; ext_dir=0; ext_peak=0.0; entered=False; entry=0.0; ei=0; stop=0.0; tgt=0.0; trd=[]
    for i in range(SMA+1,n-1):
        if np.isnan(maL[i]) or np.isnan(maH[i]): continue
        above=l[i]>maH[i]; below=h[i]<maL[i]                 # fully above HIGH-MA / below LOW-MA
        if above:
            if ext_dir!=1: ext_dir=1; ext_peak=h[i]; entered=False
            else: ext_peak=max(ext_peak,h[i])
        elif below:
            if ext_dir!=-1: ext_dir=-1; ext_peak=l[i]; entered=False
            else: ext_peak=min(ext_peak,l[i])
        else: ext_dir=0
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stop: ex=stop
                elif l[i]<=maL[i]: ex=maL[i]                 # target = SMA7(low)
            else:
                if l[i]<stop: ex=stop
                elif h[i]>=maH[i]: ex=maH[i]                 # target = SMA7(high)
            if np.isnan(ex) and (i-ei)>=TCAP: ex=c[i]
            if not np.isnan(ex): trd.append((pos,pos*(ex-entry)/PIP-SPREAD,i-ei)); pos=0
        if pos==0 and not entered and i>0:
            if ext_dir==1 and above and h[i]<h[i-1]: pos=-1; entry=o[i+1]; ei=i+1; stop=ext_peak; entered=True
            elif ext_dir==-1 and below and l[i]>l[i-1]: pos=1; entry=o[i+1]; ei=i+1; stop=ext_peak; entered=True
    T=pd.DataFrame(trd,columns=["dir","pnl","bars"]); span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD H1 fade peaked ext -> low/high envelope SMA{SMA} — {len(d)} H1 bars, {span:.1f} yrs, {len(T)} trades")
    if len(T)<5: print("  too few"); return
    obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(4000)])
    print(f"  net {T.pnl.sum():+.0f}p  per-trade {obs:+.2f}p  WR {100*(T.pnl>0).mean():.0f}%  per-yr {T.pnl.sum()/span:+.0f}p  hold {T.bars.mean():.0f}h  MCp {(np.abs(null)>=abs(obs)).mean():.3f}")
    print(f"  winners {T[T.pnl>0].pnl.mean():+.1f}p / losers {T[T.pnl<=0].pnl.mean():+.1f}p  (vs close-MA H1-peak baseline -1.53p)")

if __name__=="__main__": main()
