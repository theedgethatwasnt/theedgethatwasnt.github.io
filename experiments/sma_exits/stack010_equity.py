#!/usr/bin/env python3
"""
010 SMA-stack — long-run equity curve INCLUDING the 200p catastrophe fence.
Live mostly exits with profit (TP / PSAR trail), but a 200-pip outer-fence stop can
be hit and the short live sample hasn't seen one yet. Question: with those rare deep
stops included, does 010 still have positive expectancy, and what does the equity
curve look like? Backtests the 4 live pairs with their EXACT configs + fence, over
~9.6 months of S5, deducts spread, and builds the portfolio equity curve.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series

S5 = H.S5_DIR; MAX_ROWS = 5_000_000
# pair -> (pip, tf1_min, tf2_min, sma, tp_pips, use_psar, af_start, act, fence)
CFG = {
 "EUR_JPY": (0.01,   2.0, 10.0, (5,15,35), 20.0, False, 0.0,   0.0, 200.0),
 "EUR_USD": (0.0001, 1.0,  5.0, (5,15,35), 30.0, True,  0.020, 20.0, 200.0),
 "GBP_USD": (0.0001, 0.5,  1.0, (7,22,50),  0.0, True,  0.020, 20.0, 200.0),
 "USD_JPY": (0.01,   1.0,  5.0, (5,10,22), 15.0, False, 0.0,   0.0, 200.0),
}


@nb.njit(cache=True)
def kern(o,h,l,c, t1l,t1s,t2l,t2s, sar_b, pip, tp_pips, use_psar, act, fence):
    n=len(o); pos=0; entry=0.0; ebar=-1; mfe=0.0; armed=False
    pnl=np.empty(n); ent=np.empty(n,np.int64); rsn=np.empty(n,np.int64); nt=0
    for i in range(1,n):
        if pos==0:
            if t1l[i]==1 and t2l[i]==1: pos=1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
            if t1s[i]==1 and t2s[i]==1: pos=-1; entry=o[i]; ebar=i; mfe=0.0; armed=False; continue
        if pos!=0:
            fav=(h[i]-entry)/pip if pos==1 else (entry-l[i])/pip
            if fav>mfe: mfe=fav
            if use_psar and (not armed) and mfe>=act: armed=True
            ex=0.0; r=-1
            fc=entry - pos*fence*pip
            if pos==1 and l[i]<=fc: ex=fc; r=2
            elif pos==-1 and h[i]>=fc: ex=fc; r=2
            if r<0 and tp_pips>0:
                tp=entry+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: ex=tp; r=0
                elif pos==-1 and l[i]<=tp: ex=tp; r=0
            if r<0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos==1 and c[i]<sar_b[i]: ex=c[i]; r=1
                elif pos==-1 and c[i]>sar_b[i]: ex=c[i]; r=1
            if r>=0:
                pnl[nt]=(ex-entry)/pip*pos; ent[nt]=ebar; rsn[nt]=r; nt+=1; pos=0
    return pnl[:nt], ent[:nt], rsn[:nt]


def run_pair(pair):
    pip,t1m,t2m,(ss,sm,sl),tp,use_psar,af,act,fence = CFG[pair]
    df=H.fast_tail_read(S5/f"{pair}_S5_BA.parquet", MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
    o=df['open'].to_numpy(float); h=df['high'].to_numpy(float); l=df['low'].to_numpy(float); c=df['close'].to_numpy(float)
    ts=df['timestamp'].to_numpy(); n=len(df)
    prev=np.empty_like(ts); prev[0]=ts[0]; prev[1:]=ts[:-1]
    tf1=H.resample_minutes(df,t1m,5/60); tf2=H.resample_minutes(df,t2m,5/60)
    def nov(cc,tt):
        a=sma(cc,ss); b=sma(cc,sm); d=sma(cc,sl)
        return (project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,1))).astype(np.int8),
                project_to_m5(prev,tt,H.novelty(H.tf_signal(cc,a,b,d,0))).astype(np.int8))
    t1l,t1s=nov(tf1['close'].to_numpy(float),tf1['timestamp'].to_numpy())
    t2l,t2s=nov(tf2['close'].to_numpy(float),tf2['timestamp'].to_numpy())
    if use_psar:
        sar=project_to_m5(prev, tf1['timestamp'].to_numpy(),
                          psar_series(tf1['high'].to_numpy(float),tf1['low'].to_numpy(float),af,0.10))
    else:
        sar=np.full(n,np.nan)
    p,e,r=kern(o,h,l,c,t1l,t1s,t2l,t2s,sar,pip,tp,use_psar,act,fence)
    spcost=PAIRS[pair][1]*SPREAD_FRAC
    net=p-spcost
    return pd.DataFrame({'pair':pair,'exit_ts':ts[e],'net':net,'reason':r})


def main():
    _c=np.zeros(50); _s=np.zeros(50,np.int8)
    H.tf_signal(_c,_c,_c,_c,1); psar_series(_c+1,_c+1,.02,.1)
    kern(_c,_c,_c,_c,_s,_s,_s,_s,_c,.01,20.,True,20.,200.)
    parts=[]
    print(f"010 4-pair backtest WITH 200p fence (~9.6mo S5). per-trade pips net of spread:")
    print(f"  {'pair':<9}{'trades':>7}{'expectancy':>11}{'WR%':>6}{'fence_hits':>11}{'fence_rate':>11}{'worst':>8}")
    for pr in CFG:
        d=run_pair(pr); parts.append(d)
        nf=int((d.reason==2).sum())
        print(f"  {pr:<9}{len(d):>7}{d.net.mean():>+11.2f}{(d.net>0).mean()*100:>5.0f}%{nf:>11}{nf/max(len(d),1)*100:>10.2f}%{d.net.min():>8.0f}")
    allt=pd.concat(parts).sort_values('exit_ts').reset_index(drop=True)
    allt['cum']=allt['net'].cumsum()
    nf=int((allt.reason==2).sum())
    eq=allt['cum'].values; dd=(eq-np.maximum.accumulate(eq)); maxdd=dd.min()
    span_days=(pd.Timestamp(allt['exit_ts'].iloc[-1])-pd.Timestamp(allt['exit_ts'].iloc[0])).days
    print(f"\n  PORTFOLIO: {len(allt)} trades over {span_days}d  net {eq[-1]:+.0f}p  expectancy {allt.net.mean():+.2f}p/trade")
    print(f"  fence(200p) hits: {nf} ({nf/len(allt)*100:.2f}% of trades)  worst single trade {allt.net.min():.0f}p")
    print(f"  equity max drawdown: {maxdd:.0f}p   final/day: {eq[-1]/max(span_days,1):+.1f}p/d")
    print(f"  POSITIVE EXPECTANCY including fence? {'YES' if allt.net.mean()>0 else 'NO'}")
    # dump daily net pips (for portfolio / correlation analysis)
    allt['date']=pd.to_datetime(allt['exit_ts']).dt.normalize()
    daily=allt.groupby('date')['net'].sum()
    odir=Path(__file__).parent/'results'; odir.mkdir(exist_ok=True)
    daily.to_csv(odir/'stack010_daily.csv'); print(f"  daily 010 net -> results/stack010_daily.csv ({len(daily)} days, {daily.index.min().date()}..{daily.index.max().date()})")
    # plot
    fig,ax=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[3,1]})
    t=pd.to_datetime(allt['exit_ts'])
    ax[0].plot(t,eq,color='#2196f3',lw=1.3)
    fh=allt[allt.reason==2]
    if len(fh): ax[0].scatter(pd.to_datetime(fh['exit_ts']), fh['cum'], color='#f44336', s=30, zorder=5, label=f'200p fence hit (n={len(fh)})')
    ax[0].axhline(0,color='#888',lw=0.6); ax[0].set_title(f'010 SMA-stack (4 pairs) — equity WITH 200p catastrophe fence | net {eq[-1]:+.0f}p, expectancy {allt.net.mean():+.2f}p/tr, fence {nf/len(allt)*100:.1f}%'); ax[0].set_ylabel('cumulative pips'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].fill_between(t, dd, 0, color='#f44336', alpha=.4); ax[1].set_ylabel('drawdown (pips)'); ax[1].grid(alpha=.3)
    plt.tight_layout(); out=Path(__file__).parent/'results'/'stack010_equity.png'; out.parent.mkdir(exist_ok=True)
    plt.savefig(out,dpi=110); print(f"\n  equity curve -> {out}")


if __name__=="__main__":
    main()
