"""
eurusd_h1_bb_fade.py — EUR/USD H1 Bollinger-Band fade.
BB: basis = SMA9(close), bands = basis +/- K*std9(close). A bar COMPLETELY outside the band
(low > upper -> overextended up / high < lower -> overextended down) is faded back to the basis
(SMA9). Entry next open, exit when price touches the basis (the mean), time cap as backstop.
Net spread. Sweep band width K (1.0 as specified, + 1.5 / 2.0 — deeper = bigger target vs spread).
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; SMA=9; TCAP=24

def run(o,h,l,c,ts,basis,sd,K):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; entry=0.0; ei=0; trd=[]; tdist=[]
    for i in range(SMA+1,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        if pos!=0:
            ex=np.nan
            if pos==-1 and l[i]<=basis[i]: ex=basis[i]
            elif pos==1 and h[i]>=basis[i]: ex=basis[i]
            if np.isnan(ex) and (i-ei)>=TCAP: ex=c[i]
            if not np.isnan(ex): trd.append((pos,pos*(ex-entry)/PIP-SPREAD,i-ei)); pos=0
        if pos==0:
            if l[i]>up[i]: pos=-1; entry=o[i+1]; ei=i+1; tdist.append((entry-basis[i])/PIP)
            elif h[i]<lo[i]: pos=1; entry=o[i+1]; ei=i+1; tdist.append((basis[i]-entry)/PIP)
    return pd.DataFrame(trd,columns=["dir","pnl","bars"]), np.array(tdist)

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index
    basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD H1 Bollinger fade (basis SMA{SMA}, bar fully outside K-sigma -> fade to basis) — {len(d)} H1 bars, {span:.1f} yrs")
    print("="*92)
    print(f"  {'K-sigma':>8} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(h)':>8} {'tgt~':>6} {'MCp':>6}")
    for K in (1.0,1.5,2.0):
        T,td=run(o,h,l,c,ts,basis,sd,K)
        if len(T)<5: print(f"  {K:>8.1f}  too few"); continue
        obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(3000)])
        print(f"  {K:>8.1f} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} {100*(T.pnl>0).mean():>4.0f}% "
              f"{T.pnl.sum()/span:>+7.0f} {T.bars.mean():>7.0f}h {td.mean():>5.1f}p {(np.abs(null)>=abs(obs)).mean():>6.3f}")
    print("  (tgt~ = avg pips from entry to the basis = the reversion target; compare to ~1.7p spread)")

if __name__=="__main__": main()
