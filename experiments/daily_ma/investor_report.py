"""
investor_report.py — analytical report assets for the BB-fade timeframe portfolio.
Efficient frontier, risk metrics, correlation, equity+drawdown. HYPOTHETICAL/BACKTEST inputs
(30d, real spread, DD_FRAC=0.10) — illustrative, NOT a track record. Writes charts + a stats table.
"""
import sys, numpy as np, pandas as pd, duckdb, gc
sys.path.insert(0,"research/experiments/daily_ma")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from portfolio_analysis import sim_dated, PAIRS, TF, DDF, DAYS

def daily_returns():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    daily={}
    for tfn,rule,s5pb,tcap,meat,wdd in TF:
        per=[]
        for p,pip in PAIRS.items():
            df=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/s5_ohlc/{p}_S5_BA.parquet' ORDER BY timestamp").df()
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True);df=df.set_index("timestamp")
            cut=df.index.max()-pd.Timedelta(days=DAYS+5);df=df[df.index>=cut]
            per+=sim_dated(df,pip,med[p],rule,s5pb,tcap,meat);del df;gc.collect()
        s=pd.Series(0.0,index=sorted({d for d,_ in per}))
        for d,pn in per: s[d]+=pn
        daily[tfn]=s*DDF/wdd
    R=pd.DataFrame(daily).sort_index().fillna(0.0)
    return R[R.index>=R.index.max()-pd.Timedelta(days=DAYS)]

def stats(r):
    ann_r=r.mean()*252; ann_v=r.std()*np.sqrt(252); sh=ann_r/ann_v if ann_v>0 else 0
    dn=r[r<0].std()*np.sqrt(252); so=ann_r/dn if dn>0 else 0
    eq=(1+r).cumprod(); mdd=(eq/eq.cummax()-1).min(); cal=ann_r/abs(mdd) if mdd<0 else 0
    return dict(ann_ret=100*ann_r, ann_vol=100*ann_v, sharpe=sh, sortino=so, maxdd=100*mdd,
                calmar=cal, win=100*(r>0).mean(), best=100*r.max(), worst=100*r.min())

def main():
    R=daily_returns(); cols=list(R.columns)
    mu=R.mean().values*252; cov=R.cov().values*252
    print(f"Inputs: {len(R)} trading days, {len(cols)} strategies (BB {cols})\n")
    print("Per-strategy (annualized, hypothetical):")
    print(f"  {'':>6} {'ret%':>7} {'vol%':>7} {'Sharpe':>7} {'Sortino':>8} {'maxDD%':>7} {'Calmar':>7} {'win%':>5}")
    for c in cols:
        s=stats(R[c]); print(f"  {c:>6} {s['ann_ret']:>7.1f} {s['ann_vol']:>7.1f} {s['sharpe']:>7.2f} {s['sortino']:>8.2f} {s['maxdd']:>7.2f} {s['calmar']:>7.2f} {s['win']:>4.0f}%")
    # efficient frontier: random long-only weights
    rng=np.random.default_rng(0); N=20000; W=rng.random((N,len(cols))); W/=W.sum(1,keepdims=True)
    pr=W@mu; pv=np.sqrt(np.einsum('ij,jk,ik->i',W,cov,W)); psh=pr/pv
    imax=psh.argmax(); imin=pv.argmin()
    iv=1/R.std(); rp=(iv/iv.sum()).values; ew=np.ones(len(cols))/len(cols)
    def pt(w): r=w@mu; v=np.sqrt(w@cov@w); return v,r,r/v
    print("\nKey portfolios (annualized, hypothetical):")
    for nm,w in [("Max-Sharpe",W[imax]),("Min-Var",W[imin]),("Risk-Parity",rp),("Equal-Wt",ew)]:
        v,r,s=pt(w); print(f"  {nm:>12}: ret {100*r:5.1f}%  vol {100*v:4.1f}%  Sharpe {s:.2f}  weights {dict(zip(cols,(w*100).round(0).astype(int)))}")
    # charts
    fig,axs=plt.subplots(1,3,figsize=(18,5.2))
    sc=axs[0].scatter(100*pv,100*pr,c=psh,s=4,cmap="viridis",alpha=0.5); plt.colorbar(sc,ax=axs[0],label="Sharpe")
    for nm,w,mk in [("Max-Sharpe",W[imax],"*"),("Min-Var",W[imin],"D"),("Risk-Parity",rp,"s"),("Equal-Wt",ew,"^")]:
        v,r,_=pt(w); axs[0].scatter(100*v,100*r,marker=mk,s=160,edgecolor="k",label=nm,zorder=5)
    axs[0].set_xlabel("Annualized vol %");axs[0].set_ylabel("Annualized return %");axs[0].set_title("Efficient Frontier (HYPOTHETICAL, 30d inputs)");axs[0].legend(fontsize=8);axs[0].grid(alpha=0.2)
    im=axs[1].imshow(R.corr().values,cmap="RdBu_r",vmin=-1,vmax=1)
    axs[1].set_xticks(range(len(cols)));axs[1].set_yticks(range(len(cols)));axs[1].set_xticklabels(cols);axs[1].set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)): axs[1].text(j,i,f"{R.corr().values[i,j]:.2f}",ha="center",va="center",fontsize=9)
    axs[1].set_title("Correlation matrix");plt.colorbar(im,ax=axs[1])
    port=(R*rp).sum(1);eq=(1+port).cumprod();ddc=eq/eq.cummax()-1
    axs[2].plot(100*(eq.values-1),color="navy",lw=1.8,label="Risk-parity equity %")
    axs[2].fill_between(range(len(ddc)),100*ddc.values,0,color="red",alpha=0.3,label="Drawdown %")
    axs[2].set_title("Portfolio equity & drawdown (30d)");axs[2].legend(fontsize=8);axs[2].grid(alpha=0.2)
    plt.tight_layout();out="research/experiments/daily_ma/investor_frontier.png";plt.savefig(out,dpi=110);print("\nsaved",out)
    s=stats(port); print(f"\nRISK-PARITY PORTFOLIO (annualized, hypothetical): ret {s['ann_ret']:.1f}% vol {s['ann_vol']:.1f}% Sharpe {s['sharpe']:.2f} Sortino {s['sortino']:.2f} maxDD {s['maxdd']:.2f}% Calmar {s['calmar']:.1f} win {s['win']:.0f}%")

if __name__=="__main__": main()
