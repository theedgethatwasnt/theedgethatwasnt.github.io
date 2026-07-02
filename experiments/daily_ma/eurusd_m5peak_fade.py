"""
eurusd_m5peak_fade.py — fade a peaked extension back to the H1 SMA7, but detect the PEAK on M5.
Context/mean = H1 SMA7 (causal, mapped to M5). Extension = M5 price >= k * (H1 ATR) above/below
the H1 SMA7 (meaningfully stretched). Peak = first M5 lower-high (up) / higher-low (down) while
extended -> fade at next M5 open. Target = H1 SMA7 (the mean); stop = new M5 extreme beyond the
swing; time cap. Net spread. Sweep the extension distance k. Compare vs H1-peak baseline (-1.53p).
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; TCAP=288   # M5 bars (~24h)

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    o=df.open.values;h=df.high.values;l=df.low.values;c=df.close.values; idx=df.index
    H=df.resample("1h").agg({"high":"max","low":"min","close":"last"}).dropna()
    sma7=H.close.rolling(7).mean()
    pc=H.close.shift(1); tr=np.maximum(H.high-H.low,np.maximum((H.high-pc).abs(),(H.low-pc).abs()))
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    sma_m5=sma7.shift(1).reindex(idx,method="ffill").values      # causal: last completed H1 SMA7
    atr_m5=atr.shift(1).reindex(idx,method="ffill").values
    n=len(c); span=(idx[-1]-idx[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD fade peaked extension -> H1 SMA7, M5 peak detection — {n} M5 bars, {span:.1f} yrs")
    print(f"  (vs H1-peak baseline: -1.53p/trade, -962 p/yr)\n"+"="*82)
    print(f"  {'k_ext(ATR)':>10} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(m5)':>9} {'MCp':>6}")
    for k in (1.0,1.5,2.0,3.0):
        pos=0; entry=0.0; ei=0; stop=0.0; trd=[]
        for i in range(20,n-1):
            if np.isnan(sma_m5[i]) or np.isnan(atr_m5[i]) or atr_m5[i]<=0: continue
            up_ext = (c[i]-sma_m5[i]) >= k*atr_m5[i]
            dn_ext = (sma_m5[i]-c[i]) >= k*atr_m5[i]
            if pos!=0:
                ex=np.nan
                if pos==-1:
                    if h[i]>stop: ex=stop
                    elif l[i]<=sma_m5[i]: ex=sma_m5[i]
                else:
                    if l[i]<stop: ex=stop
                    elif h[i]>=sma_m5[i]: ex=sma_m5[i]
                if np.isnan(ex) and (i-ei)>=TCAP: ex=c[i]
                if not np.isnan(ex): trd.append((pos,pos*(ex-entry)/PIP-SPREAD)); pos=0
            if pos==0 and i>0:
                if up_ext and h[i]<h[i-1]: pos=-1; entry=o[i+1]; ei=i+1; stop=h[i]+0.0
                elif dn_ext and l[i]>l[i-1]: pos=1; entry=o[i+1]; ei=i+1; stop=l[i]-0.0
        T=pd.DataFrame(trd,columns=["dir","pnl"])
        if len(T)<5: print(f"  {k:>10.1f}  too few"); continue
        obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(2000)])
        print(f"  {k:>10.1f} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} {100*(T.pnl>0).mean():>4.0f}% "
              f"{T.pnl.sum()/span:>+7.0f} {0:>8d}  {(np.abs(null)>=abs(obs)).mean():>6.3f}")

if __name__=="__main__": main()
