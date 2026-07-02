"""
protrusion_study.py — what is the band-protrusion oscillator good for? EUR/USD M5, net spread.
Two RESOLUTIONS: (A) SMA5/1sigma (fast/fine), (B) SMA20/2sigma (slow/standard BB).
Two INDICATOR forms: cumsum 'accumulation line' + rolling-window oscillator.
Usage tests:
  1. SMA-cross on the cumsum line (signal line) — WITH the cross (momentum) and AGAINST (fade).
  2. Fade rolling-oscillator extremes (short when osc>+T, long when osc<-T) — mean-reversion.
  3. Trend on rolling osc (go with) — already shown dead, included as control.
Reports per config: trades, p/trade, WR, pips/day, OOS p/t, MC sign-flip p.
"""
import duckdb, numpy as np, pandas as pd
PIP=1e-4; SPREAD=1.6; IS_FRAC=0.6

def tally(h,l,basis,sd,k):
    up=basis+k*sd; lo=basis-k*sd; rng=np.maximum(h-l,1e-12)
    a=np.maximum(0.0,h-np.maximum(up,l)); b=np.maximum(0.0,np.minimum(lo,h)-l)
    t=(a-b)/rng; t[np.isnan(up)]=0.0; return t

def trade(o,c,ts,long_sig,short_sig,exit_long,exit_short,span):
    n=len(c); pos=0; ent=0.0; ei=0; tr=[]
    for i in range(1,n-1):
        np_=pos
        if pos==1 and exit_long[i]: np_=0
        elif pos==-1 and exit_short[i]: np_=0
        if np_==0:
            if long_sig[i]: np_=1
            elif short_sig[i]: np_=-1
        if np_!=pos:
            px=o[i+1]
            if pos!=0: tr.append((ts[ei],pos*(px-ent)/PIP-SPREAD))
            if np_!=0: ent=px; ei=i+1
            pos=np_
    return tr

def report(name,tr,span,rng):
    if len(tr)<30: print(f"  {name:<34} too few"); return
    pnl=np.array([p for _,p in tr]); ets=np.array([x for x,_ in tr]); icut=ets[int(len(ets)*IS_FRAC)]
    oos=pnl[ets>=icut]; null=np.array([(pnl*rng.choice([-1.,1.],len(pnl))).mean() for _ in range(2000)])
    print(f"  {name:<34} {len(tr):>6}tr {pnl.mean():>+6.2f}p/t {100*(pnl>0).mean():>3.0f}%WR {pnl.sum()/span:>+7.1f}pd  OOS{oos.mean():>+6.2f}  MCp{(np.abs(null)>=abs(pnl.mean())).mean():>6.3f}")

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    d=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d=d.set_index("timestamp")
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
    span=(d.index[-1]-d.index[0]).days
    for (label,sma,k) in [("A: SMA5/1σ",5,1.0),("B: SMA20/2σ",20,2.0)]:
        basis=pd.Series(c).rolling(sma).mean().values; sd=pd.Series(c).rolling(sma).std().values
        t=tally(h,l,basis,sd,k); line=np.nancumsum(t)
        print(f"\n=== Resolution {label} ===  (mean tally {np.nanmean(t):+.4f}, % bars out {100*np.mean(np.abs(t)>0):.0f})")
        # 1. SMA-cross on cumsum line
        for W in (50,200):
            sig=pd.Series(line).rolling(W).mean().values
            above=line>sig; below=line<sig
            report(f"cumsum>SMA{W} WITH (long above)", trade(o,c,ts,above,below,below,above,span),span,rng)
            report(f"cumsum<SMA{W} AGAINST (fade)",    trade(o,c,ts,below,above,above,below,span),span,rng)
        # 2/3. rolling oscillator
        for W in (20,50):
            osc=pd.Series(t).rolling(W).sum().values; T=np.nanpercentile(np.abs(osc),80)
            hi=osc>T; lo_=osc<-T
            report(f"rollW{W} FADE extreme (|osc|>p80)", trade(o,c,ts,lo_,hi,osc>0,osc<0,span),span,rng)  # long when very negative
            report(f"rollW{W} TREND (go with)",          trade(o,c,ts,hi,lo_,osc<0,osc>0,span),span,rng)

if __name__=="__main__": main()
