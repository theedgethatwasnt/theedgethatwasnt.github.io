"""
full_portfolio_report.py — FULL 5-strategy portfolio (educational/hypothetical):
010 SMA-Stack (momentum, REAL live daily P&L) + 4 BB-fade timeframe books (backtest, S5-exit).
Efficient frontier, correlation, risk-parity allocation, combined equity/drawdown. Mixed bases
(010 live ~5wk vs BB 30d backtest); 010 cross-correlations are thin-sample — illustrative only.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"research/experiments/daily_ma")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from investor_report import daily_returns, stats

# 010 SMA-Stack REAL live daily pips (trades.duckdb, acct ...010, is_paper=FALSE)
TEN={"2026-05-19":-39.0,"2026-05-26":-22.6,"2026-06-03":12.9,"2026-06-04":20.3,"2026-06-05":69.7,
     "2026-06-08":-11.1,"2026-06-09":127.7,"2026-06-10":46.5,"2026-06-11":35.5,"2026-06-14":29.3,
     "2026-06-15":4.1,"2026-06-16":27.5,"2026-06-17":64.2,"2026-06-18":-70.1,"2026-06-19":0.0,
     "2026-06-22":21.3,"2026-06-23":36.7,"2026-06-24":30.6}
TEN_PIP_RET=0.0001084   # 010 $/pip per $NAV (units=NAV*1.3, avg pip-value across its 4 pairs)

def main():
    Rbb=daily_returns()                                   # 4 BB cols, ~30 trading days
    s10=pd.Series({pd.Timestamp(k,tz="UTC").normalize():v*TEN_PIP_RET for k,v in TEN.items()})
    s10.index=s10.index.tz_localize(None)
    Rbb.index=pd.to_datetime(Rbb.index).tz_localize(None)
    R=Rbb.copy(); R["SMA-Stk(010)"]=s10
    R=R.sort_index()
    cols=list(R.columns)
    print("Per-strategy (annualized; 010=REAL live ~5wk, BB=30d backtest):")
    print(f"  {'':>12} {'ret%':>7} {'vol%':>7} {'Sharpe':>7} {'maxDD%':>7} {'win%':>5} {'days':>5}")
    for c in cols:
        r=R[c].dropna(); st=stats(r)
        print(f"  {c:>12} {st['ann_ret']:>7.1f} {st['ann_vol']:>7.1f} {st['sharpe']:>7.2f} {st['maxdd']:>7.2f} {st['win']:>4.0f}% {len(r):>5}")
    # correlation on overlapping dates (010 vs BB = thin)
    Rc=R.dropna()
    print(f"\nCorrelation (full overlap = {len(Rc)} days — 010 cross-corr THIN/illustrative):")
    print(R.corr().round(2).to_string())
    # frontier inputs: standalone annualized mean/vol; cov from 0-filled union
    Rf=R.fillna(0.0); mu=Rf.mean().values*252; cov=Rf.cov().values*252
    rng=np.random.default_rng(0);N=30000;W=rng.random((N,len(cols)));W/=W.sum(1,keepdims=True)
    pr=W@mu; pv=np.sqrt(np.einsum('ij,jk,ik->i',W,cov,W)); psh=np.divide(pr,pv,out=np.zeros_like(pr),where=pv>0)
    imax=psh.argmax(); imin=pv.argmin(); iv=1/Rf.std().replace(0,np.nan); rp=(iv/iv.sum()).fillna(0).values
    def pt(w): r=w@mu;v=np.sqrt(w@cov@w);return v,r,(r/v if v>0 else 0)
    print("\nKey portfolios (annualized, hypothetical):")
    for nm,w in [("Max-Sharpe",W[imax]),("Min-Var",W[imin]),("Risk-Parity",rp)]:
        v,r,sh=pt(w); print(f"  {nm:>12}: ret {100*r:6.1f}% vol {100*v:5.1f}% Sharpe {sh:.2f}  w={dict(zip(cols,(w*100).round(0).astype(int)))}")
    # charts
    fig,axs=plt.subplots(1,3,figsize=(19,5.4))
    sc=axs[0].scatter(100*pv,100*pr,c=psh,s=4,cmap="viridis",alpha=0.5);plt.colorbar(sc,ax=axs[0],label="Sharpe")
    for nm,w,mk in [("Max-Sharpe",W[imax],"*"),("Min-Var",W[imin],"D"),("Risk-Parity",rp,"s")]:
        v,r,_=pt(w);axs[0].scatter(100*v,100*r,marker=mk,s=170,edgecolor="k",label=nm,zorder=5)
    for i,c in enumerate(cols):
        axs[0].scatter(100*Rf[c].std()*np.sqrt(252),100*Rf[c].mean()*252,marker="o",s=70,edgecolor="k",alpha=0.7)
        axs[0].annotate(c,(100*Rf[c].std()*np.sqrt(252),100*Rf[c].mean()*252),fontsize=7)
    axs[0].set_xlabel("Annualized vol %");axs[0].set_ylabel("Annualized return %");axs[0].set_title("Efficient Frontier — 5 strategies (HYPOTHETICAL)");axs[0].legend(fontsize=8);axs[0].grid(alpha=0.2)
    cm=R.corr().values
    im=axs[1].imshow(cm,cmap="RdBu_r",vmin=-1,vmax=1);axs[1].set_xticks(range(len(cols)));axs[1].set_yticks(range(len(cols)))
    axs[1].set_xticklabels(cols,rotation=45,ha="right",fontsize=8);axs[1].set_yticklabels(cols,fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)): axs[1].text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",fontsize=8)
    axs[1].set_title("Correlation (010 cross = thin)");plt.colorbar(im,ax=axs[1])
    port=(Rf*rp).sum(1);eq=(1+port).cumprod();dd=eq/eq.cummax()-1
    axs[2].plot(100*(eq.values-1),color="navy",lw=1.8,label="Risk-parity equity %")
    axs[2].fill_between(range(len(dd)),100*dd.values,0,color="red",alpha=0.3,label="Drawdown %")
    axs[2].set_title("5-strategy portfolio equity & drawdown");axs[2].legend(fontsize=8);axs[2].grid(alpha=0.2)
    plt.tight_layout();out="research/experiments/daily_ma/full_portfolio_frontier.png";plt.savefig(out,dpi=110);print("\nsaved",out)
    st=stats(port);print(f"\n5-STRATEGY RISK-PARITY (annualized, hypothetical): ret {st['ann_ret']:.1f}% vol {st['ann_vol']:.1f}% Sharpe {st['sharpe']:.2f} maxDD {st['maxdd']:.2f}% win {st['win']:.0f}%")
    print("RP weights:",{c:f'{100*w:.0f}%' for c,w in zip(cols,rp)})

if __name__=="__main__": main()
