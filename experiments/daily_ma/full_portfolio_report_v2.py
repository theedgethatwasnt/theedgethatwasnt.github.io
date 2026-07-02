"""
full_portfolio_report_v2.py — PROPER portfolio analysis on the full history.
4 BB-fade books: 5.3y daily returns (lib/bb_fade.backtest on m5_ohlc, per-pair median real spread,
DD_FRAC=0.10) -> robust covariance/correlation/frontier with REAL drawdowns & Sharpe.
010 SMA-Stack: real ~5wk live daily P&L (cannot be backtested over 5.3y) — overlaid, correlation
to BB estimated from overlap (thin, flagged). Charts + stats.
"""
import sys, numpy as np, pandas as pd, duckdb, gc
sys.path.insert(0,".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lib.bb_fade import backtest
DDF=0.10
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
TF=[("M5","5min",288,4.0,3574),("M15","15min",96,6.0,1703),("H1","1h",24,6.0,2264),("H4","4h",12,10.0,1483)]
TEN={"2026-05-19":-39.0,"2026-05-26":-22.6,"2026-06-03":12.9,"2026-06-04":20.3,"2026-06-05":69.7,
     "2026-06-08":-11.1,"2026-06-09":127.7,"2026-06-10":46.5,"2026-06-11":35.5,"2026-06-14":29.3,
     "2026-06-15":4.1,"2026-06-16":27.5,"2026-06-17":64.2,"2026-06-18":-70.1,"2026-06-19":0.0,
     "2026-06-22":21.3,"2026-06-23":36.7,"2026-06-24":30.6}; TEN_PIP_RET=0.0001084

def stats(r):
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sh=ar/av if av>0 else 0
    eq=(1+r).cumprod(); mdd=(eq/eq.cummax()-1).min(); cal=ar/abs(mdd) if mdd<0 else 0
    dn=r[r<0].std()*np.sqrt(252)
    return dict(ar=100*ar,av=100*av,sh=sh,so=ar/dn if dn>0 else 0,mdd=100*mdd,cal=cal,win=100*(r>0).mean(),n=len(r))

def bb_daily():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    daily={}
    for tfn,rule,tcap,meat,wdd in TF:
        acc={}
        for p,pip in PAIRS.items():
            b=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
            b["timestamp"]=pd.to_datetime(b["timestamp"],utc=True); b=b.set_index("timestamp")
            d=b.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            idx=d.index
            for (eb,xb,dr,pnl) in backtest(d.open.values,d.high.values,d.low.values,d.close.values,pip,med[p],meat,tcap):
                dt=idx[xb].normalize(); acc[dt]=acc.get(dt,0.0)+pnl
            del b,d; gc.collect()
        s=pd.Series(acc).sort_index()*DDF/wdd
        daily[tfn]=s
    R=pd.DataFrame(daily).sort_index()
    # full business-day index over the span, fill non-trade days with 0
    full=pd.date_range(R.index.min(),R.index.max(),freq="B")
    return R.reindex(full).fillna(0.0)

def main():
    R=bb_daily(); R.index=R.index.tz_localize(None)
    cols=list(R.columns)
    print(f"=== BB books: {len(R)} business days ({R.index.min().date()}..{R.index.max().date()}) ===")
    print(f"  {'':>6} {'ret%':>7} {'vol%':>7} {'Sharpe':>7} {'Sortino':>8} {'maxDD%':>8} {'Calmar':>7} {'win%':>5}")
    for c in cols:
        s=stats(R[c]); print(f"  {c:>6} {s['ar']:>7.1f} {s['av']:>7.1f} {s['sh']:>7.2f} {s['so']:>8.2f} {s['mdd']:>8.1f} {s['cal']:>7.2f} {s['win']:>4.0f}%")
    print("\n=== correlation (5.3y, ROBUST) ==="); print(R.corr().round(2).to_string())
    mu=R.mean().values*252; cov=R.cov().values*252
    rng=np.random.default_rng(0);N=40000;W=rng.random((N,len(cols)));W/=W.sum(1,keepdims=True)
    pr=W@mu;pv=np.sqrt(np.einsum('ij,jk,ik->i',W,cov,W));psh=pr/pv
    imax=psh.argmax();imin=pv.argmin();iv=1/R.std();rp=(iv/iv.sum()).values
    def pt(w):v=np.sqrt(w@cov@w);r=w@mu;return v,r,r/v
    print("\n=== key portfolios (5.3y, annualized) ===")
    for nm,w in [("Max-Sharpe",W[imax]),("Min-Var",W[imin]),("Risk-Parity",rp)]:
        v,r,sh=pt(w);print(f"  {nm:>12}: ret {100*r:5.1f}% vol {100*v:4.1f}% Sharpe {sh:.2f}  w={dict(zip(cols,(w*100).round(0).astype(int)))}")
    port=(R*rp).sum(1);ps=stats(port)
    print(f"\n4-BB RISK-PARITY (5.3y): ret {ps['ar']:.1f}% vol {ps['av']:.1f}% Sharpe {ps['sh']:.2f} Sortino {ps['so']:.2f} maxDD {ps['mdd']:.1f}% Calmar {ps['cal']:.2f} win {ps['win']:.0f}%")
    # 010 overlay (live ~5wk)
    s10=pd.Series({pd.Timestamp(k):v*TEN_PIP_RET for k,v in TEN.items()}); st10=stats(s10)
    ov=pd.concat([port.rename("bb"),s10.rename("t10")],axis=1).dropna()
    c10=ov.corr().iloc[0,1] if len(ov)>2 else float("nan")
    print(f"010 SMA-Stack (REAL live {st10['n']}d): ret {st10['ar']:.1f}% vol {st10['av']:.1f}% Sharpe {st10['sh']:.2f} maxDD {st10['mdd']:.1f}% | corr to BB-port (overlap {len(ov)}d, THIN) = {c10:+.2f}")
    # charts: frontier, correlation, full equity/DD
    eq=(1+port).cumprod();dd=eq/eq.cummax()-1
    fig,axs=plt.subplots(1,3,figsize=(19,5.4))
    sc=axs[0].scatter(100*pv,100*pr,c=psh,s=3,cmap="viridis",alpha=0.5);plt.colorbar(sc,ax=axs[0],label="Sharpe")
    for nm,w,mk in [("Max-Sharpe",W[imax],"*"),("Min-Var",W[imin],"D"),("Risk-Parity",rp,"s")]:
        v,r,_=pt(w);axs[0].scatter(100*v,100*r,marker=mk,s=170,edgecolor="k",label=nm,zorder=5)
    for c in cols: axs[0].scatter(100*R[c].std()*np.sqrt(252),100*R[c].mean()*252,s=60,edgecolor="k",alpha=.7);axs[0].annotate(c,(100*R[c].std()*np.sqrt(252),100*R[c].mean()*252),fontsize=7)
    axs[0].set_xlabel("Annualized vol %");axs[0].set_ylabel("Annualized return %");axs[0].set_title(f"Efficient Frontier — 4 BB books (5.3y, ROBUST)");axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    cm=R.corr().values;im=axs[1].imshow(cm,cmap="RdBu_r",vmin=-1,vmax=1)
    axs[1].set_xticks(range(len(cols)));axs[1].set_yticks(range(len(cols)));axs[1].set_xticklabels(cols);axs[1].set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):axs[1].text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",fontsize=9)
    axs[1].set_title("Correlation (5.3y)");plt.colorbar(im,ax=axs[1])
    axs[2].plot(R.index,100*(eq.values-1),color="navy",lw=1.0,label="Risk-parity equity %")
    axs[2].fill_between(R.index,100*dd.values,0,color="red",alpha=.3,label="Drawdown %")
    axs[2].set_title("4-BB portfolio equity & drawdown (5.3y)");axs[2].legend(fontsize=8);axs[2].grid(alpha=.2)
    plt.tight_layout();out="research/experiments/daily_ma/full_portfolio_frontier_5y.png";plt.savefig(out,dpi=110);print("\nsaved",out)

if __name__=="__main__": main()
