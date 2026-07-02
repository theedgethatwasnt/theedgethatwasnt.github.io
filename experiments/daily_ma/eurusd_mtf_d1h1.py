"""
eurusd_mtf_d1h1.py — EUR/USD multi-timeframe: DAILY trend filter + HOURLY entry, must AGREE.
Daily trend = sign(daily close - daily SMA(Nd)), causal (uses only days completed before the H1
bar). Hourly entry = bar fully clears SMA13 (low>SMA13 long / high<SMA13 short) AND agrees with
the daily trend. Exit = H1 touches/closes-beyond SMA13 OR the daily regime flips. Net spread,
next-bar-open fills. Compare vs H1-alone (dead) and daily-alone (+17.6/trade) baselines.
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; HSMA=13

def run(o,h,l,c,ts,sma,dt,exit_mode):
    n=len(c); pos=0; entry=0.0; ei=0; tr=[]
    for i in range(HSMA+1,n-1):
        if np.isnan(sma[i]) or np.isnan(dt[i]): continue
        np_=pos
        if pos==1:
            hit=False if exit_mode=="dflip" else ((l[i]<=sma[i]) if exit_mode=="touch" else (c[i]<sma[i]))
            if hit or dt[i]<=0: np_=0
        elif pos==-1:
            hit=False if exit_mode=="dflip" else ((h[i]>=sma[i]) if exit_mode=="touch" else (c[i]>sma[i]))
            if hit or dt[i]>=0: np_=0
        if np_==0:
            if l[i]>sma[i] and dt[i]>0: np_=1
            elif h[i]<sma[i] and dt[i]<0: np_=-1
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
    H=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    D=df["close"].resample("1D").last().dropna()
    o=H.open.values;h=H.high.values;l=H.low.values;c=H.close.values;ts=H.index
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    sma13=pd.Series(c).rolling(HSMA).mean().values
    print(f"EUR/USD MTF (daily trend + hourly entry, must agree) — {len(H)} H1 bars, {span:.1f} yrs")
    print("="*92)
    print(f"  {'cfg':>22} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(h)':>8} {'MCp':>6}")
    for Nd in (20,50,100):
        dsma=D.rolling(Nd).mean()
        dtrend=np.sign(D-dsma)                              # +1/-1 per daily close
        dt_avail=dtrend.shift(1).reindex(ts,method="ffill").values  # causal: prior completed day
        for ex in ("touch","close","dflip"):
            T=run(o,h,l,c,ts,sma13,dt_avail,ex)
            if len(T)<5: print(f"  D-SMA{Nd}/{ex:5}  too few"); continue
            obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(3000)])
            print(f"  {'D-SMA'+str(Nd)+' / H1-'+ex:>22} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} "
                  f"{100*(T.pnl>0).mean():>4.0f}% {T.pnl.sum()/span:>+7.0f} {T.hrs.mean():>7.0f}h "
                  f"{(np.abs(null)>=abs(obs)).mean():>6.3f}")

if __name__=="__main__": main()
