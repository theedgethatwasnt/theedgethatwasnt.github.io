"""
Split shocks by HIGHER-TF trend (daily 10-day return, like the Markov filter;
also H1 SMA slope). For each shock measure BOTH:
  - fade_pnl  (enter counter to shock, TP20/SL30)
  - cont_pnl  (enter WITH shock, TP20/SL30)
align = sign(shock) * sign(trend):
  align=+1  shock WITH daily trend  -> fading fights the trend (expect fade bad, cont good)
  align=-1  shock AGAINST daily trend -> fading rides trend resumption (expect fade good)
Then test a regime strategy: against-trend shocks -> FADE; with-trend shocks -> CONTINUE.
WF check (3 IS chunks) on the winning split. Real spread, causal trend labels.
"""
import numpy as np, pandas as pd, sys
from numba import njit
from pathlib import Path
sys.path.insert(0, '/path/to/projects/fx-core')

PROJ = Path('/path/to/projects/fx-core')
PAIRS = ['GBP_JPY', 'USD_JPY', 'EUR_JPY', 'AUD_JPY']
PIP = 0.01
THR=2.5; PEAK=44; TP=20.0; SL=30.0; HORIZON=600; MAD_WIN=2048


def compute_shock_z(close, pip, w=6, mad_win=2048):
    n=len(close); vel=np.empty(n); vel[:w]=0.0
    vel[w:]=(close[w:]-close[:n-w])/pip
    vs=pd.Series(vel); rm=vs.rolling(mad_win,min_periods=50).median()
    ad=(vs-rm).abs(); rmad=ad.rolling(mad_win,min_periods=50).median()
    z=((vs-rm)/(1.4826*rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


@njit
def scan(bid, ask, close, shock, vel, td, peak, tp, sl, horizon, pip):
    n=len(close); mx=n//8
    fade=np.zeros(mx); cont=np.zeros(mx); trd=np.zeros(mx); tix=np.zeros(mx,dtype=np.int64)
    ev=0; cd=0
    for t in range(MAD_WIN, n-peak-horizon-2):
        if cd>0: cd-=1; continue
        if shock[t]!=1: continue
        ws=t+peak+1; sp=(ask[t]-bid[t])/pip
        sh = 1 if vel[t]>0 else -1          # shock direction
        # FADE: d=sh -> trade opposite (short if up-shock). pnl convention from retrace
        for mode in range(2):
            d = sh if mode==0 else -sh       # mode0 fade(enter opposite shock => pos dir = -sh? )
            # define position dir: fade pos = -sh ; cont pos = +sh
            posdir = -sh if mode==0 else sh
            if posdir==1:  fill=ask[ws]      # long pays ask
            else:          fill=bid[ws]      # short fills bid
            tp_lvl = fill + tp*pip*posdir
            sl_lvl = fill - sl*pip*posdir
            pnl=0.0; done=False
            for j in range(ws, min(ws+horizon, n-1)):
                lo=bid[j]; hi=ask[j]
                if posdir==1:
                    if hi>=tp_lvl: pnl=tp-sp; done=True; break
                    if lo<=sl_lvl: pnl=-sl-sp; done=True; break
                else:
                    if lo<=tp_lvl: pnl=tp-sp; done=True; break
                    if hi>=sl_lvl: pnl=-sl-sp; done=True; break
            if not done:
                j=min(ws+horizon,n-1)
                if posdir==1: pnl=(bid[j]-fill)/pip-sp
                else:         pnl=(fill-ask[j])/pip-sp
            if mode==0: fade[ev]=pnl
            else:       cont[ev]=pnl
        trd[ev]=sh*td[t]      # align: +1 with-trend, -1 against-trend, 0 flat
        tix[ev]=t; ev+=1
        cd=horizon
    return fade[:ev], cont[:ev], trd[:ev], tix[:ev]


def daily_trend(df, pip):
    """causal daily 10-day return sign, mapped to each S5 bar (known at day start)."""
    c=df.set_index('timestamp')['close']
    dc=c.resample('1D').last()
    r10=dc.pct_change(10).shift(1)            # prev completed day's 10d return
    tr=np.sign(r10).fillna(0.0)
    # map to S5 bars by date
    dates=df['timestamp'].dt.floor('1D')
    return dates.map(tr).fillna(0.0).to_numpy()


def h1_trend(df):
    c=df.set_index('timestamp')['close']
    hc=c.resample('1h').last()
    sma=hc.rolling(20).mean()
    slope=np.sign(sma.diff(5)).shift(1).fillna(0.0)
    hours=df['timestamp'].dt.floor('1h')
    return hours.map(slope).fillna(0.0).to_numpy()


def report(tag, fade, cont, align, days):
    print(f'  --- {tag} ---')
    print(f'    {"bucket":14s} {"N":>7} {"fade mean":>10} {"fadeWR":>7} {"cont mean":>10} {"contWR":>7}')
    for lab,m in [('AGAINST trend', align<-0.5), ('FLAT', np.abs(align)<0.5), ('WITH trend', align>0.5)]:
        if m.sum()<50: continue
        f=fade[m]; c=cont[m]
        print(f'    {lab:14s} {m.sum():>7} {f.mean():>+9.2f}p {(f>0).mean()*100:>6.0f}% {c.mean():>+9.2f}p {(c>0).mean()*100:>6.0f}%')
    # regime strategy: against->fade, with->cont, flat->skip
    strat=np.where(align<-0.5, fade, np.where(align>0.5, cont, np.nan))
    s=strat[np.isfinite(strat)]
    print(f'    REGIME STRAT (against->fade, with->cont): N={len(s)} mean={s.mean():+.2f}p WR={(s>0).mean()*100:.0f}% p/d={s.sum()/days:+.1f}')


ALL={}
for pair in PAIRS:
    df=pd.read_parquet(PROJ/'data'/'s5_ba'/f'{pair}_S5_BA.parquet',
                       columns=['timestamp','close','bid_c','ask_c'])
    close=df.close.to_numpy(float); bid=df.bid_c.to_numpy(float); ask=df.ask_c.to_numpy(float)
    n=len(close); days=n*5/(60*60*24*5/7)
    z,vel=compute_shock_z(close,PIP); shock=(np.abs(z)>THR).astype(np.int8)
    tD=daily_trend(df,PIP); tH=h1_trend(df)
    _=scan(bid[:60000],ask[:60000],close[:60000],shock[:60000],vel[:60000],tD[:60000],PEAK,TP,SL,HORIZON,PIP)
    fade,cont,alD,tix=scan(bid,ask,close,shock,vel,tD,PEAK,TP,SL,HORIZON,PIP)
    # H1 align: recompute align with tH (reuse same shocks via tix)
    sh=np.sign(vel[tix]); alH=sh*tH[tix]
    print('='*72)
    print(f'{pair}: {len(fade):,} shocks  fade-all={fade.mean():+.2f}p  cont-all={cont.mean():+.2f}p')
    report('DAILY 10d trend', fade, cont, alD, days)
    report('H1 SMA20 slope', fade, cont, alH, days)
    ALL[pair]=(fade,cont,alD,alH,tix,days,n)

print('\n'+'='*72+'\nAGGREGATE (4 JPY):')
F=np.concatenate([ALL[p][0] for p in PAIRS]); C=np.concatenate([ALL[p][1] for p in PAIRS])
AD=np.concatenate([ALL[p][2] for p in PAIRS]); AH=np.concatenate([ALL[p][3] for p in PAIRS])
totdays=sum(ALL[p][5] for p in PAIRS)/4
print(f'  fade-all={F.mean():+.2f}p  cont-all={C.mean():+.2f}p  N={len(F):,}')
report('DAILY 10d trend', F, C, AD, totdays)
report('H1 SMA20 slope', F, C, AH, totdays)
