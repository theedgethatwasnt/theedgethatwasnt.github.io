"""
bb_reentry_multitf.py — Bollinger RE-ENTRY fade + half-distance meat filter, across timeframes,
with IS/OOS validation. EUR/USD. Same rule as eurusd_h1_bb_reentry_fade but parameterized by TF
and split 60/40 IS/OOS (the meat threshold is the only knob; OOS must stay positive to be real).
Mechanism check: meat filter = volatility gate, so higher TFs (bigger sigma) should be stronger.
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; SMA=9; K=1.0; IS_FRAC=0.6

def gen(o,h,l,c,basis,sd,MIN,tcap):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; entry=0.0; ei=0; stop=0.0; ext=0; ext_peak=0.0; out=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        up_out=l[i]>up[i]; dn_out=h[i]<lo[i]
        if up_out: ext_peak=h[i] if ext!=1 else max(ext_peak,h[i]); ext=1
        elif dn_out: ext_peak=l[i] if ext!=-1 else min(ext_peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stop: ex=stop
                elif l[i]<=basis[i]: ex=basis[i]
            else:
                if l[i]<stop: ex=stop
                elif h[i]>=basis[i]: ex=basis[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex): out.append((i,pos*(ex-entry)/PIP-SPREAD)); pos=0
        if pos==0:
            ent=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i]:
                if 0.5*(ent-basis[i])/PIP-SPREAD>=MIN: pos=-1; entry=ent; ei=i+1; stop=ext_peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i]:
                if 0.5*(basis[i]-ent)/PIP-SPREAD>=MIN: pos=1; entry=ent; ei=i+1; stop=ext_peak
    return out  # list of (bar_index, pnl)

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    rng=np.random.default_rng(0)
    print("EUR/USD Bollinger RE-ENTRY fade — multi-TF + IS/OOS (60/40).  meat = projected half-dist net spread")
    print("="*100)
    print(f"  {'TF':>4} {'meat':>5} {'trades':>7} {'IS p/t':>7} {'OOS p/t':>8} {'OOS WR':>7} {'OOS p/yr':>9} {'OOS MCp':>8}")
    TFS=[("5min",288),("15min",96),("1h",24),("4h",12),("1D",8)]
    MEATS=[2.0,4.0,6.0,10.0,20.0]
    for tf,tcap in TFS:
        d=df.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; n=len(c)
        basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
        span=(d.index[-1]-d.index[0]).days/365.25; is_cut=int(n*IS_FRAC)
        for MIN in MEATS:
            tr=gen(o,h,l,c,basis,sd,MIN,tcap)
            if len(tr)<40: continue
            isp=np.array([p for bi,p in tr if bi<is_cut]); oos=np.array([p for bi,p in tr if bi>=is_cut])
            if len(oos)<20: continue
            null=np.array([(oos*rng.choice([-1.,1.],len(oos))).mean() for _ in range(2000)])
            print(f"  {tf:>4} {MIN:>5.0f} {len(tr):>7} {isp.mean():>+7.2f} {oos.mean():>+8.2f} "
                  f"{100*(oos>0).mean():>6.0f}% {oos.sum()/(span*(1-IS_FRAC)):>+9.0f} {(np.abs(null)>=abs(oos.mean())).mean():>8.3f}")
        print()

if __name__=="__main__": main()
