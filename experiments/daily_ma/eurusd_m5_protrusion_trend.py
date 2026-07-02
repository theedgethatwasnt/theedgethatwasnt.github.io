"""
eurusd_m5_protrusion_trend.py — trade IN THE DIRECTION of the band-protrusion oscillator.
Per-bar signed protrusion (fraction of bar outside the SMA5 1-sigma envelope, + above / - below).
Indicator = ROLLING SUM over W bars (a raw cumsum drifts, so ±1 wouldn't repeat). Strategy:
LONG when indicator > +T_enter, hold while > +T_exit; SHORT when < -T_enter, hold while < -T_exit.
Next-open fills, net spread, IS/OOS. EUR/USD M5. Sweep W and the exit threshold.
"""
import duckdb, numpy as np, pandas as pd
PIP=1e-4; SPREAD=1.6; SMA=5; K=1.0; IS_FRAC=0.6

def protrusion(h,l,up,lo):
    rng=np.maximum(h-l,1e-12)
    above=np.maximum(0.0,h-np.maximum(up,l)); below=np.maximum(0.0,np.minimum(lo,h)-l)
    t=(above-below)/rng; t[np.isnan(up)]=0.0; return t

def run(o,c,ts,ind,Tin,Tout):
    n=len(c); pos=0; ent=0.0; ei=0; tr=[]
    for i in range(1,n-1):
        if np.isnan(ind[i]): continue
        np_=pos
        if pos==1 and ind[i]<Tout: np_=0
        elif pos==-1 and ind[i]>-Tout: np_=0
        if np_==0:
            if ind[i]>Tin: np_=1
            elif ind[i]<-Tin: np_=-1
        if np_!=pos:
            px=o[i+1]
            if pos!=0: tr.append((ts[ei],pos*(px-ent)/PIP-SPREAD))
            if np_!=0: ent=px; ei=i+1
            pos=np_
    return tr

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    d=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d=d.set_index("timestamp")
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
    basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
    tally=protrusion(h,l,basis+K*sd,basis-K*sd)
    span=(d.index[-1]-d.index[0]).days
    print(f"EUR/USD M5 protrusion-TREND (go with the oscillator). span {span}d, net {SPREAD}p spread")
    print("="*86)
    print(f"  {'W':>4} {'Tin':>4} {'Tout':>5} {'trades':>7} {'p/trade':>8} {'WR':>5} {'pips/day':>9} {'OOS p/t':>8} {'MCp':>6}")
    for W in (10,20,50):
        ind=pd.Series(tally).rolling(W).sum().values
        for Tin,Tout in [(1.0,1.0),(1.0,0.0),(2.0,0.0)]:
            tr=run(o,c,ts,ind,Tin,Tout)
            if len(tr)<30: print(f"  {W:>4} {Tin:>4.1f} {Tout:>5.1f}  too few"); continue
            pnl=np.array([p for _,p in tr]); ets=np.array([x for x,_ in tr]); icut=ets[int(len(ets)*IS_FRAC)]
            oos=pnl[ets>=icut]; null=np.array([(pnl*rng.choice([-1.,1.],len(pnl))).mean() for _ in range(2000)])
            print(f"  {W:>4} {Tin:>4.1f} {Tout:>5.1f} {len(tr):>7} {pnl.mean():>+8.2f} {100*(pnl>0).mean():>4.0f}% "
                  f"{pnl.sum()/span:>+9.1f} {oos.mean():>+8.2f} {(np.abs(null)>=abs(pnl.mean())).mean():>6.3f}")

if __name__=="__main__": main()
