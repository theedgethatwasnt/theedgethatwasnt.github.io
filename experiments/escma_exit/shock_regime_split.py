"""
Do |z|>2.5 momentum shocks happen in FLAT/compressed regimes or in TRENDS?
Are they separable, and does the split make the retrace fade more profitable?

For each shock at bar t (S5):
  - pre-shock EFFICIENCY RATIO (trendiness) over Wpre bars before t:
        ER = |c[t]-c[t-Wpre]| / sum|diff|     (1=clean trend, 0=chop/flat)
  - pre-shock VOL COMPRESSION: std(ret, last 120) / std(ret, last 2048)
        (<1 = quiet/compressed before the burst)
  - FADE outcome: enter counter-trend at t+peak+1 (market), TP=20 SL=30 horizon=600,
        real spread. pnl in pips.
Then bucket fade pnl by regime and test if "fade only flat-regime shocks" separates.
"""
import numpy as np, pandas as pd, sys
from numba import njit
from pathlib import Path
sys.path.insert(0, '/path/to/projects/fx-core')

PROJ = Path('/path/to/projects/fx-core')
PAIRS = ['GBP_JPY', 'USD_JPY', 'EUR_JPY', 'AUD_JPY']
PIP = 0.01
THR = 2.5; PEAK = 44; TP = 20.0; SL = 30.0; HORIZON = 600
WPRE = 360          # 30 min pre-shock window for efficiency ratio
Z_WINDOW = 6; MAD_WIN = 2048


def compute_shock_z(close, pip, w=6, mad_win=2048):
    n=len(close); vel=np.empty(n); vel[:w]=0.0
    vel[w:]=(close[w:]-close[:n-w])/pip
    vs=pd.Series(vel); rm=vs.rolling(mad_win,min_periods=50).median()
    ad=(vs-rm).abs(); rmad=ad.rolling(mad_win,min_periods=50).median()
    z=((vs-rm)/(1.4826*rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


@njit
def scan(bid, ask, close, shock, vel, pip, peak, tp, sl, horizon, wpre):
    n=len(close); mx=n//8
    fp=np.zeros(mx); er=np.zeros(mx); comp=np.zeros(mx); tix=np.zeros(mx,dtype=np.int64)
    ev=0; cd=0
    for t in range(MAD_WIN, n-peak-horizon-2):
        if cd>0: cd-=1; continue
        if shock[t]!=1: continue
        # efficiency ratio over [t-wpre, t]
        net=abs(close[t]-close[t-wpre])/pip
        s=0.0
        for k in range(t-wpre+1, t+1): s+=abs(close[k]-close[k-1])/pip
        ef = net/s if s>1e-9 else 0.0
        # vol compression: std last 120 / std last 2048
        m1=0.0
        for k in range(t-120+1,t+1): m1+=(close[k]-close[k-1])/pip
        m1/=120; v1=0.0
        for k in range(t-120+1,t+1):
            d=(close[k]-close[k-1])/pip-m1; v1+=d*d
        v1=(v1/120)**0.5
        m2=0.0
        for k in range(t-2048+1,t+1): m2+=(close[k]-close[k-1])/pip
        m2/=2048; v2=0.0
        for k in range(t-2048+1,t+1):
            d=(close[k]-close[k-1])/pip-m2; v2+=d*d
        v2=(v2/2048)**0.5
        cp = v1/v2 if v2>1e-9 else 1.0
        # fade: direction d=sign(vel); trade opposite. d=1(upshock)->short
        d = 1 if vel[t]>0 else -1
        ws=t+peak+1
        sp=(ask[t]-bid[t])/pip
        # market entry at ws
        if d==1:  fill=bid[ws]      # short
        else:     fill=ask[ws]
        tp_lvl = fill - tp*pip*d
        sl_lvl = fill + sl*pip*d
        pnl=0.0; done=False
        for j in range(ws, min(ws+horizon, n-1)):
            lo=bid[j]; hi=ask[j]
            if d==1:   # short: profit when price falls
                if lo<=tp_lvl: pnl=tp-sp; done=True; break
                if hi>=sl_lvl: pnl=-sl-sp; done=True; break
            else:      # long
                if hi>=tp_lvl: pnl=tp-sp; done=True; break
                if lo<=sl_lvl: pnl=-sl-sp; done=True; break
        if not done:
            j=min(ws+horizon,n-1)
            if d==1: pnl=(fill-bid[j])/pip-sp
            else:    pnl=(ask[j]-fill)/pip-sp
        fp[ev]=pnl; er[ev]=ef; comp[ev]=cp; tix[ev]=t; ev+=1
        cd=horizon
    return fp[:ev], er[:ev], comp[:ev], tix[:ev]


def bucket_report(name, vals, pnl, edges):
    print(f'    by {name}:')
    labels=['LOW','MID','HIGH']
    qs=np.quantile(vals,[0,1/3,2/3,1.0])
    for i in range(3):
        m=(vals>=qs[i])&(vals<=qs[i+1]) if i==2 else (vals>=qs[i])&(vals<qs[i+1])
        if m.sum()<20: continue
        pp=pnl[m]
        print(f'      {labels[i]:4s} [{qs[i]:.2f}-{qs[i+1]:.2f}]  N={m.sum():>6}  mean={pp.mean():+6.2f}p  WR={ (pp>0).mean()*100:4.0f}%  sum={pp.sum():+8.0f}')


all_fp=[]; all_er=[]; all_cp=[]; all_days=[]
print(f"Shock regime split  thr={THR} peak={PEAK} fade TP={TP} SL={SL}\n")
for pair in PAIRS:
    df=pd.read_parquet(PROJ/'data'/'s5_ba'/f'{pair}_S5_BA.parquet',
                       columns=['timestamp','close','bid_c','ask_c'])
    close=df.close.to_numpy(float); bid=df.bid_c.to_numpy(float); ask=df.ask_c.to_numpy(float)
    n=len(close); days=n*5/(60*60*24*5/7)
    z,vel=compute_shock_z(close,PIP)
    shock=(np.abs(z)>THR).astype(np.int8)
    _=scan(bid[:60000],ask[:60000],close[:60000],shock[:60000],vel[:60000],PIP,PEAK,TP,SL,HORIZON,WPRE)
    fp,er,cp,tix=scan(bid,ask,close,shock,vel,PIP,PEAK,TP,SL,HORIZON,WPRE)
    print('='*70)
    print(f'{pair}: {len(fp):,} shocks ({len(fp)/n*100:.2f}% of bars)  fade all: mean={fp.mean():+.2f}p WR={(fp>0).mean()*100:.0f}% p/d={fp.sum()/days:+.1f}')
    print(f'  ER(trendiness) median at shocks={np.median(er):.3f}  | compression median={np.median(cp):.2f}')
    bucket_report('EFFICIENCY RATIO (low=flat/chop, high=trend)', er, fp, None)
    bucket_report('VOL COMPRESSION (low=quiet pre-burst)', cp, fp, None)
    all_fp.append(fp); all_er.append(er); all_cp.append(cp); all_days.append(days)

print('\n'+'='*70)
print('AGGREGATE (4 JPY pairs):')
FP=np.concatenate(all_fp); ER=np.concatenate(all_er); CP=np.concatenate(all_cp)
tot_days=sum(all_days)
print(f'  fade ALL shocks: N={len(FP):,} mean={FP.mean():+.2f}p WR={(FP>0).mean()*100:.0f}% p/d(sum)={FP.sum()/ (tot_days/4):+.1f}')
bucket_report('EFFICIENCY RATIO', ER, FP, None)
bucket_report('VOL COMPRESSION', CP, FP, None)
