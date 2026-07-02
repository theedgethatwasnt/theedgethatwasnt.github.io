"""
eurusd_m5_protrusion.py — EUR/USD M5 "band-protrusion oscillator".
BB on SMA5, 1-sigma bands. For each bar, tally the SIGNED fraction of the bar's range that sticks
OUT of the envelope: +frac if it pokes above the upper band, -frac if below the lower band
(e.g. 10% of range below lower -> -0.1). Running cumulative total plotted under the price+BB.
"""
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SMA=5; K=1.0; N=800   # recent bars to plot (readable window)

def main():
    con=duckdb.connect()
    d=con.execute("SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/EUR_USD_M5.parquet' ORDER BY timestamp").df()
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d=d.set_index("timestamp").tail(N+SMA)
    c=d.close.values;h=d.high.values;l=d.low.values
    basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
    up=basis+K*sd; lo=basis-K*sd
    rng=np.maximum(h-l,1e-12)
    above=np.maximum(0.0, h-np.maximum(up,l))      # portion of bar above upper band
    below=np.maximum(0.0, np.minimum(lo,h)-l)      # portion of bar below lower band
    tally=(above-below)/rng                        # signed fraction outside the envelope
    tally[np.isnan(up)]=0.0
    run=np.nancumsum(tally)
    # plot recent window
    d=d.iloc[SMA:]; x=np.arange(len(d)); ix=d.index
    basis,up,lo,tally,run,c=basis[SMA:],up[SMA:],lo[SMA:],tally[SMA:],run[SMA:],c[SMA:]
    run=run-run[0]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,9),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(x,c,color="#222",lw=0.9,label="EUR/USD close")
    ax1.plot(x,basis,color="#1f77b4",lw=0.8,label=f"SMA{SMA} basis")
    ax1.fill_between(x,lo,up,color="#1f77b4",alpha=0.12,label=f"±{K:g}σ band")
    ax1.plot(x,up,color="#1f77b4",lw=0.5,alpha=0.5); ax1.plot(x,lo,color="#1f77b4",lw=0.5,alpha=0.5)
    ax1.set_title(f"EUR/USD M5 — price + Bollinger(SMA{SMA},{K:g}σ)  [last {len(d)} bars]"); ax1.legend(loc="upper left",fontsize=8); ax1.grid(alpha=0.2)
    ax2.plot(x,run,color="#d62728",lw=1.0)
    ax2.axhline(0,color="#888",lw=0.6)
    ax2.fill_between(x,run,0,where=(run>=0),color="#2ca02c",alpha=0.25)
    ax2.fill_between(x,run,0,where=(run<0),color="#d62728",alpha=0.25)
    ax2.set_title("Running total of signed band-protrusion (Σ of per-bar fraction outside envelope; + above / − below)"); ax2.grid(alpha=0.2)
    step=max(1,len(d)//10); ax2.set_xticks(x[::step]); ax2.set_xticklabels([t.strftime("%m-%d %H:%M") for t in ix[::step]],rotation=45,fontsize=7)
    plt.tight_layout(); out="research/experiments/daily_ma/eurusd_m5_protrusion.png"; plt.savefig(out,dpi=110); print("saved",out)
    print(f"tally stats: mean {np.nanmean(tally):+.4f} | bars poking out {100*np.mean(np.abs(tally)>0):.0f}% | final running total {run[-1]:+.1f}")

if __name__=="__main__": main()
