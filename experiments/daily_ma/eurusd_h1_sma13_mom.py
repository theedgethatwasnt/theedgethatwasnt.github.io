"""
eurusd_h1_sma13_mom.py — EUR/USD H1 SMA13 bar-clear trend-follow + a MINIMUM SMA13-MOMENTUM gate.
Entry: bar fully above SMA13 (low>SMA) AND SMA13 sloping UP hard enough -> long; fully below AND
SMA13 sloping DOWN hard enough -> short. slope = (SMA13[i]-SMA13[i-3]) normalized by ATR; gate =
|slope_atr| >= thr in the trade direction. Exit when a bar touches the MA (user's spec). Sweep thr.
Idea: skip the flat-MA chop where the whipsaws live. Net spread, next-bar-open fills.
"""
import duckdb, numpy as np, pandas as pd

PIP=0.0001; SPREAD=1.7; SMA=13

def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values

def run(o,h,l,c,ts,sma,a,thr,norm=True):
    n=len(c); pos=0; entry=0.0; ei=0; tr=[]
    slope=np.full(n,np.nan); slope[3:]=sma[3:]-sma[:-3]
    for i in range(SMA+3,n-1):
        if np.isnan(sma[i]) or np.isnan(a[i]) or a[i]<=0: continue
        sl=(slope[i]/a[i]) if norm else (slope[i]/PIP)   # ATR-normalized vs raw pips (per 3 bars)
        np_=pos
        if pos==1 and l[i]<=sma[i]: np_=0
        elif pos==-1 and h[i]>=sma[i]: np_=0
        if np_==0:
            if l[i]>sma[i] and sl>=thr: np_=1
            elif h[i]<sma[i] and sl<=-thr: np_=-1
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
    sma=pd.Series(c).rolling(SMA).mean().values; a=atr(h,l,c)
    print(f"EUR/USD H1 SMA13 bar-clear + min-SMA-momentum gate — {len(d)} H1 bars, {span:.1f} yrs")
    print(f"  (slope = SMA13 3-bar change / ATR; gate filters flat-MA chop)\n"+"="*84)
    def sweep(label, norm, thrs):
        print(f"  [{label}]   {'thr':>7} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(h)':>8} {'MCp':>6}")
        for thr in thrs:
            T=run(o,h,l,c,ts,sma,a,thr,norm=norm)
            if len(T)<5: print(f"  {'':>13} {thr:>7.2f}  too few"); continue
            obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(3000)])
            print(f"  {'':>13} {thr:>7.2f} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} {100*(T.pnl>0).mean():>4.0f}% "
                  f"{T.pnl.sum()/span:>+7.0f} {T.hrs.mean():>7.0f}h {(np.abs(null)>=abs(obs)).mean():>6.3f}")
    sweep("ATR-normalized slope", True, (0.0,0.10,0.20,0.30,0.50,1.00))
    print()
    sweep("RAW slope (pips/3bars)", False, (0.0,1.0,2.0,3.0,5.0,10.0))

if __name__=="__main__": main()
