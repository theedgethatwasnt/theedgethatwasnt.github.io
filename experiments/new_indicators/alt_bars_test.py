#!/usr/bin/env python3
"""
Volume/tick-bar (event-sampled) indicator screen.
==================================================
Lopez de Prado idea: sample bars by ACTIVITY (cumulative tick-volume) not clock
time. Returns become closer to IID and signals can sharpen. OANDA `volume` = tick
count, so a volume bar ≈ a fixed-tick-count bar.

Builds event bars at two activity scales (≈daily and ≈4h sampling), then runs the
same honest gate as the time-bar screen: spread-deducted avg pnl/trade at a bounded
1-bar hold + IC. TREND and CONTRARIAN framings.

Read-only on data/m5_ba.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70
HOLD = 1
TARGET_BARS = {"vbar~D1": 1400, "vbar~H4": 5600}   # event-bar counts over full history

def pip_sz(p): return 0.01 if p in JPY else 0.0001

def ind_trix(c, n=14):
    e = c.ewm(span=n, adjust=False).mean().ewm(span=n, adjust=False).mean().ewm(span=n, adjust=False).mean()
    return e.pct_change() * 1e4
def ind_vortex(h,l,c, n=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    vip = (h-l.shift()).abs().rolling(n).sum()/tr.rolling(n).sum()
    vim = (l-h.shift()).abs().rolling(n).sum()/tr.rolling(n).sum()
    return vip - vim
def ind_fisher(h,l, n=10):
    med=(h+l)/2; mn=med.rolling(n).min(); mx=med.rolling(n).max()
    x=(2*(med-mn)/(mx-mn).replace(0,np.nan)-1).clip(-.999,.999)
    x=x.ewm(alpha=.33,adjust=False).mean().clip(-.999,.999)
    return (0.5*np.log((1+x)/(1-x))).ewm(alpha=.5,adjust=False).mean()
def ind_rsi2(c, n=2):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

def build_signals(rs):
    o,h,l,c = rs["open"],rs["high"],rs["low"],rs["close"]
    sig = {}
    sig["TRIX14"]   = np.sign(ind_trix(c)).fillna(0)
    sig["Vortex14"] = np.sign(ind_vortex(h,l,c)).fillna(0)
    sig["Fisher10"] = np.sign(ind_fisher(h,l)).fillna(0)
    r2 = ind_rsi2(c); s2 = pd.Series(0.0,index=c.index); s2[r2<10]=1.0; s2[r2>90]=-1.0
    sig["RSI2"] = s2
    return sig

def eval_signal(rs, sig, pip, n_is, hold=HOLD):
    c=rs["close"].values; bid=rs["bid_c"].values; ask=rs["ask_c"].values
    s=sig.values.astype(float); n=len(c)
    fwd=np.full(n,np.nan); fwd[:-hold]=(c[hold:]-c[:-hold])/pip
    bid_f=np.full(n,np.nan); ask_f=np.full(n,np.nan); bid_f[:-hold]=bid[hold:]; ask_f[:-hold]=ask[hold:]
    pnl=np.full(n,np.nan); lo=s==1; sh=s==-1
    pnl[lo]=(bid_f[lo]-ask[lo])/pip; pnl[sh]=(bid[sh]-ask_f[sh])/pip
    m=(s!=0)&~np.isnan(fwd); m[:n_is]=False
    if m.sum()<30: return dict(ic_oos=np.nan,avg_oos=np.nan,t_oos=np.nan,n_oos=0)
    ic=np.corrcoef(s[m],fwd[m])[0,1] if s[m].std()>0 else np.nan
    pm=(s!=0)&~np.isnan(pnl); pm[:n_is]=False; pv=pnl[pm]
    t=pv.mean()/(pv.std(ddof=1)/np.sqrt(len(pv))) if pv.std()>0 else np.nan
    return dict(ic_oos=ic,avg_oos=pv.mean(),t_oos=t,n_oos=len(pv))

def make_volume_bars(df, threshold):
    cum = df["volume"].cumsum()
    bar_id = (cum // threshold).astype("int64")
    g = df.groupby(bar_id)
    vb = pd.DataFrame({
        "open":  g["open"].first(),  "high": g["high"].max(),
        "low":   g["low"].min(),     "close":g["close"].last(),
        "bid_c": g["bid_c"].last(),  "ask_c":g["ask_c"].last(),
    })
    return vb.reset_index(drop=True)

rows=[]
for pair in PAIRS:
    pip=pip_sz(pair)
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    tot=df["volume"].sum()
    for scale,target in TARGET_BARS.items():
        thr=max(1.0, tot/target)
        vb=make_volume_bars(df, thr)
        if len(vb)<500: continue
        n_is=int(len(vb)*IS_FRAC)
        sigs=build_signals(vb)
        for iname,base in sigs.items():
            for framing,sgn in (("trend",1),("contra",-1)):
                r=eval_signal(vb,(base*sgn).fillna(0),pip,n_is)
                rows.append(dict(pair=pair,scale=scale,bars=len(vb),ind=iname,framing=framing,**r))

res=pd.DataFrame(rows); res.to_csv(RESULTS/"alt_bars_screen.csv",index=False)
print("="*92)
print("VOLUME/TICK-BAR SCREEN — aggregate across 12 pairs.  Gate: OOS avg pnl/trade > 0 (net spread)")
print("="*92)
print(f"{'scale':<9} {'indicator':<9} {'framing':<7} {'meanIC':>8} {'mean_avgPnl':>12} {'pairs>0':>8} {'pairs_t>2':>9}")
agg=[]
for (sc,ind,fr),g in res.groupby(["scale","ind","framing"]):
    p_pos=int((g["avg_oos"]>0).sum()); p_sig=int(((g["avg_oos"]>0)&(g["t_oos"]>2)).sum())
    agg.append(dict(scale=sc,ind=ind,framing=fr,meanIC=g["ic_oos"].mean(),
                    mean_avgPnl=g["avg_oos"].mean(),pairs_pos=p_pos,pairs_sig=p_sig))
agg=pd.DataFrame(agg).sort_values("mean_avgPnl",ascending=False)
for _,r in agg.iterrows():
    print(f"{r['scale']:<9} {r['ind']:<9} {r['framing']:<7} {r['meanIC']:>8.4f} "
          f"{r['mean_avgPnl']:>12.3f} {r['pairs_pos']:>8d} {r['pairs_sig']:>9d}")
agg.to_csv(RESULTS/"alt_bars_screen_agg.csv",index=False)
print("\nSaved → results/alt_bars_screen.csv + alt_bars_screen_agg.csv")
