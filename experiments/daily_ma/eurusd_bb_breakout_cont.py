"""
eurusd_bb_breakout_cont.py — BB CONTINUATION (mirror of the re-entry fade).
basis=SMA9, bands=±1sigma. Entry: SMA5>SMA9 (trend up) AND a bar CLOSES out above the upper band
having closed inside the bar before (breakout) -> LONG in trend direction (continuation). Mirror
short. Exit: a bar CLOSES back inside the envelope. Next-open fills, net 1.7p, IS/OOS 60/40.
Sweep M5/M15/H1 (the fade was +ve on all three; continuation is its losing complement?).
"""
import duckdb, numpy as np, pandas as pd
PIP=1e-4; SPREAD=1.7; IS_FRAC=0.6

def gen(o,c,ts,s5,s9,up,lo,tcap):
    n=len(c); pos=0; entry=0.0; ei=0; out=[]
    for i in range(10,n-1):
        if np.isnan(up[i]) or np.isnan(s5[i]): continue
        np_=pos
        if pos==1 and c[i]<up[i]: np_=0          # closed back inside -> exit
        elif pos==-1 and c[i]>lo[i]: np_=0
        if np_==0:
            if c[i]>up[i] and c[i-1]<=up[i-1] and s5[i]>s9[i]: np_=1     # closed out above + uptrend
            elif c[i]<lo[i] and c[i-1]>=lo[i-1] and s5[i]<s9[i]: np_=-1
        if np_==0 and pos!=0 and (i-ei)>=tcap: pass
        if pos!=0 and np_==pos and (i-ei)>=tcap: np_=0   # time-cap backstop
        if np_!=pos:
            px=o[i+1]
            if pos!=0: out.append((ts[ei],pos*(px-entry)/PIP-SPREAD))
            if np_!=0: entry=px; ei=i+1
            pos=np_
    return out

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    print("EUR/USD BB CONTINUATION (SMA5>SMA9 + close-out breakout, exit on close-back-inside)")
    print("="*84)
    print(f"  {'TF':>5} {'trades':>7} {'IS p/t':>7} {'OOS p/t':>8} {'OOS WR':>7} {'OOS p/yr':>9} {'OOS MCp':>8} {'hold':>5}")
    for tf,tcap in [("5min",288),("15min",96),("1h",24)]:
        d=df.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        o=d.open.values;c=d.close.values;ts=d.index.values.astype("datetime64[ns]"); n=len(c)
        s5=pd.Series(c).rolling(5).mean().values; s9=pd.Series(c).rolling(9).mean().values
        sd=pd.Series(c).rolling(9).std().values; up=s9+sd; lo=s9-sd
        span=(d.index[-1]-d.index[0]).days/365.25; is_cut=ts[int(n*IS_FRAC)]
        tr=gen(o,c,ts,s5,s9,up,lo,tcap)
        if len(tr)<20: print(f"  {tf:>5} too few"); continue
        isp=np.array([p for t,p in tr if t<is_cut]); oos=np.array([p for t,p in tr if t>=is_cut])
        null=np.array([(oos*rng.choice([-1.,1.],len(oos))).mean() for _ in range(3000)])
        print(f"  {tf:>5} {len(tr):>7} {isp.mean():>+7.2f} {oos.mean():>+8.2f} {100*(oos>0).mean():>6.0f}% "
              f"{oos.sum()/(span*(1-IS_FRAC)):>+9.0f} {(np.abs(null)>=abs(oos.mean())).mean():>8.3f}")

if __name__=="__main__": main()
