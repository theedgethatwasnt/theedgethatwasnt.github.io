"""
eurusd_h1_bb_reentry_fade.py — EUR/USD H1 Bollinger RE-ENTRY fade with half-distance meat filter.
BB: basis=SMA9(close), bands=basis +/- 1*std9. RULE: if the PREVIOUS bar was completely outside the
band AND the current bar touches the band (came back), fade toward the basis. The realistic target
is ~HALF the price->basis gap (mirror convergence: the MA rises toward the falling price, meeting
midway), so gate on (0.5*gap - spread) >= MIN_MEAT. Exit at the live SMA9 touch (captures ~the half
automatically); stop if it re-extends past the extension peak; time cap. Net spread. Sweep MIN_MEAT.
"""
import duckdb, numpy as np, pandas as pd
PIP=0.0001; SPREAD=1.7; SMA=9; K=1.0; TCAP=24

def run(o,h,l,c,ts,basis,sd,MIN):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; entry=0.0; ei=0; stop=0.0; ext=0; ext_peak=0.0; trd=[]; proj=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        up_out=l[i]>up[i]; dn_out=h[i]<lo[i]
        if up_out: ext_peak=h[i] if ext!=1 else max(ext_peak,h[i]); ext=1
        elif dn_out: ext_peak=l[i] if ext!=-1 else min(ext_peak,l[i]); ext=-1
        # manage open
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stop: ex=stop
                elif l[i]<=basis[i]: ex=basis[i]
            else:
                if l[i]<stop: ex=stop
                elif h[i]>=basis[i]: ex=basis[i]
            if np.isnan(ex) and (i-ei)>=TCAP: ex=c[i]
            if not np.isnan(ex): trd.append((pos,pos*(ex-entry)/PIP-SPREAD,i-ei)); pos=0
        # entry: prev bar fully outside, current bar touched the band
        if pos==0:
            ent=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i]:               # was above, came back to touch
                meat=0.5*(ent-basis[i])/PIP
                if meat-SPREAD>=MIN: pos=-1; entry=ent; ei=i+1; stop=ext_peak; proj.append(meat)
                ext=0
            elif h[i-1]<lo[i-1] and h[i]>=lo[i]:
                meat=0.5*(basis[i]-ent)/PIP
                if meat-SPREAD>=MIN: pos=1; entry=ent; ei=i+1; stop=ext_peak; proj.append(meat)
                ext=0
    return pd.DataFrame(trd,columns=["dir","pnl","bars"]), np.array(proj)

def main():
    con=duckdb.connect()
    df=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    d=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;ts=d.index
    basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
    span=(ts[-1]-ts[0]).days/365.25; rng=np.random.default_rng(0)
    print(f"EUR/USD H1 Bollinger RE-ENTRY fade + half-distance meat filter (basis SMA{SMA}, {K}sigma) — {span:.1f} yrs")
    print("="*92)
    print(f"  {'min_meat(p)':>11} {'trades':>7} {'net pips':>9} {'p/trade':>8} {'WR':>5} {'p/yr':>7} {'hold(h)':>8} {'projhalf':>8} {'MCp':>6}")
    for MIN in (0.0,2.0,4.0,6.0,10.0):
        T,proj=run(o,h,l,c,ts,basis,sd,MIN)
        if len(T)<5: print(f"  {MIN:>11.1f}  too few"); continue
        obs=T.pnl.mean(); null=np.array([(T.pnl.values*rng.choice([-1.,1.],len(T))).mean() for _ in range(3000)])
        print(f"  {MIN:>11.1f} {len(T):>7} {T.pnl.sum():>+9.0f} {obs:>+8.2f} {100*(T.pnl>0).mean():>4.0f}% "
              f"{T.pnl.sum()/span:>+7.0f} {T.bars.mean():>7.0f}h {proj.mean():>7.1f}p {(np.abs(null)>=abs(obs)).mean():>6.3f}")

if __name__=="__main__": main()
