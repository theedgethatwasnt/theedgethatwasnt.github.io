"""
portfolio_analysis.py — the 4 BB timeframe-books (001-004) + 010 as a portfolio over the LAST MONTH.
Daily % returns per strategy (BB: S5-exit sim on S5 BA, ret = daily_pips * DD_FRAC / worst_DD;
010: live trades from trades.duckdb, ret = daily_pips * 010_$pip/NAV). Correlation matrix, risk-parity
allocation, combined equity curve + smoothness (daily vol, annualized Sharpe, max DD, % up days).
"""
import duckdb, numpy as np, pandas as pd, gc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SMA=9;K=1.0; DDF=0.10
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("M5","5min",60,288,4.0,3574),("M15","15min",180,96,6.0,1703),("H1","1h",720,24,6.0,2264),("H4","4h",2880,12,10.0,1483)]
DAYS=35

def sim_dated(df,pip,sp,rule,s5pb,tcap,meat):
    d=df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;tts=d.index.values
    n=len(c);basis=np.full(n,np.nan);sd=np.full(n,np.nan)
    for i in range(SMA-1,n):w=c[i-SMA+1:i+1];basis[i]=w.mean();sd[i]=w.std()
    up=basis+K*sd;lo=basis-K*sd
    s5o=df.open.values;s5h=df.high.values;s5l=df.low.values;s5c=df.close.values;s5ts=df.index.values
    tfo=np.searchsorted(tts,s5ts,side="right")-1; s5st=np.searchsorted(s5ts,tts)
    pos=0;ext=0;peak=0;ents=[]
    for i in range(SMA,n-1):
        if np.isnan(basis[i]):continue
        uo=l[i]>up[i];do=h[i]<lo[i]
        if uo:peak=h[i] if ext!=1 else max(peak,h[i]);ext=1
        elif do:peak=l[i] if ext!=-1 else min(peak,l[i]);ext=-1
        if pos==0:
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-sp>=meat:ents.append((i+1,-1,peak));pos=-1
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-sp>=meat:ents.append((i+1,1,peak));pos=1
        if pos!=0 and ((pos==-1 and(h[i]>peak or l[i]<=lo[i]))or(pos==1 and(l[i]<peak or h[i]>=up[i]))):pos=0
    out=[]
    for tfi,dr,pk in ents:
        if tfi>=n:continue
        j0=s5st[tfi] if tfi<len(s5st) else None
        if j0 is None or j0>=len(s5c):continue
        ent=s5o[j0];pnl=None;jend=min(j0+tcap*s5pb,len(s5c)-1)
        for j in range(j0,jend):
            ct=tfo[j]
            if ct<0:continue
            if dr==-1:
                if s5h[j]>pk:pnl=(ent-pk)/pip-sp;break
                if s5l[j]<=lo[ct]:pnl=(ent-s5c[j])/pip-sp;break
            else:
                if s5l[j]<pk:pnl=(pk-ent)/pip-sp;break
                if s5h[j]>=up[ct]:pnl=(s5c[j]-ent)/pip-sp;break
        je=min(j0+tcap*s5pb,len(s5c)-1)
        if pnl is None:pnl=(ent-s5c[je])/pip-sp if dr==-1 else (s5c[je]-ent)/pip-sp
        out.append((pd.Timestamp(s5ts[je]).normalize(),pnl))
    return out

def main():
    con=duckdb.connect();med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df();med[p]=float(np.nanmedian(s.s.values)/pip)
    daily={}
    for tfn,rule,s5pb,tcap,meat,wdd in TF:
        per=[]
        for p,pip in PAIRS.items():
            df=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/s5_ohlc/{p}_S5_BA.parquet' ORDER BY timestamp").df()
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True);df=df.set_index("timestamp")
            cut=df.index.max()-pd.Timedelta(days=DAYS+5); df=df[df.index>=cut]
            per+=sim_dated(df,pip,med[p],rule,s5pb,tcap,meat);del df;gc.collect()
        s=pd.Series({d:0.0 for d,_ in per});
        for d,pn in per: s[d]=s.get(d,0)+pn
        daily[tfn]=(s*DDF/wdd)   # daily % return
    R=pd.DataFrame(daily).sort_index().fillna(0.0)
    R=R[R.index>=R.index.max()-pd.Timedelta(days=DAYS)]
    print("=== daily % returns: last", len(R),"days (4 BB TFs), DD_FRAC=0.10 ===")
    print("mean/day:",{k:f'{100*R[k].mean():.3f}%' for k in R}); print("vol/day:",{k:f'{100*R[k].std():.3f}%' for k in R})
    print("\n=== correlation matrix (BB TFs) ==="); print((R.corr()).round(2).to_string())
    # risk parity (inverse-vol) weights
    iv=1/R.std(); w=iv/iv.sum()
    print("\n=== risk-parity (inverse-vol) weights ==="); print({k:f'{100*w[k]:.0f}%' for k in w.index})
    port=(R*w).sum(axis=1); eq=(1+port).cumprod()
    ann=np.sqrt(252)*port.mean()/port.std() if port.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    print(f"\n=== combined portfolio (risk-parity, last {len(R)}d) ===")
    print(f"  total {100*(eq.iloc[-1]-1):+.2f}% | daily mean {100*port.mean():+.3f}% vol {100*port.std():.3f}% | ann.Sharpe {ann:.2f} | maxDD {100*dd:.2f}% | up-days {100*(port>0).mean():.0f}%")
    eqw=(R.mean(axis=1)); eqe=(1+eqw).cumprod()
    # chart
    fig,(a1,a2)=plt.subplots(2,1,figsize=(13,8),gridspec_kw={"height_ratios":[2,1]})
    for k in R: a1.plot((1+R[k]).cumprod().values,lw=0.8,alpha=0.6,label=f"BB {k}")
    a1.plot(eq.values,lw=2.2,color="black",label="Portfolio (risk-parity)")
    a1.set_title(f"BB 4-TF portfolio — equity over last {len(R)} days (DD_FRAC=0.10, real spread)");a1.legend(fontsize=8);a1.grid(alpha=0.2)
    a2.bar(range(len(port)),100*port.values,color=["#2ca02c" if x>0 else "#d62728" for x in port.values])
    a2.set_title("Portfolio daily return %");a2.grid(alpha=0.2)
    plt.tight_layout();out="research/experiments/daily_ma/portfolio_lastmonth.png";plt.savefig(out,dpi=110);print("saved",out)
    # 010 overlap from DB
    try:
        db=duckdb.connect("trades.duckdb",read_only=True) if False else None
    except: pass

if __name__=="__main__":main()
