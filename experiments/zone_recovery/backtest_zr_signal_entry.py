"""
Directional signal filter on ZR 1st-leg entry.
Tests several momentum signals as entry gates on top-3 pairs.

Signal modes:
  filter: keep alternating, skip entry when signal contradicts direction
  only:   enter in signal direction whenever signal fires (drops alternating)

Signals:
  h1_smaP   — H1 SMA-P slope (P=5,10,20)
  m5_smaP   — M5 SMA-P slope (P=5,10,20)
  m5_atrK   — M5 Wilder ATR14 range burst > K× (K=1.0,1.5,2.0,2.5,3.0), dir from bar
  h1m5_PQ   — H1 SMA-P AND M5 SMA-Q must agree (combo filter)

Per-pair sweetspot + trail (from prior IS/trail sweep):
  CHF_JPY: N=1, ZW=40, tgt=20, ta=5,  td=3
  GBP_USD: N=6, ZW=30, tgt=15, ta=10, td=7
  USD_JPY: N=1, ZW=40, tgt=20, ta=10, td=5
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PIP_MAP  = {"CHF_JPY":0.01,"GBP_USD":0.0001,"USD_JPY":0.01}
SPREAD=1.4; MAX_LEGS=10; PF=1.25

PAIR_CFG = {
    "CHF_JPY": dict(N=1, zw=40.0, tgt=20.0, ta=5,  td=3),
    "GBP_USD": dict(N=6, zw=30.0, tgt=15.0, ta=10, td=7),
    "USD_JPY": dict(N=1, zw=40.0, tgt=20.0, ta=10, td=5),
}

# ─── Signal computation (vectorised numpy/pandas) ────────────────────────────

def sig_h1_sma(cl, period):
    """H1 SMA-period slope at each M5 bar. +1 above SMA, -1 below."""
    # Resample M5 to H1 (group 12-bar windows)
    n = len(cl)
    h1_n = n // 12
    h1_cl = np.array([cl[i*12+11] for i in range(h1_n)])  # last M5 of each H1
    sma = pd.Series(h1_cl).rolling(period, min_periods=period).mean().values
    slope = np.where(np.isnan(sma), 0, np.sign(h1_cl - sma)).astype(np.int8)
    # Upsample: each H1 bar covers 12 M5 bars
    sig = np.zeros(n, dtype=np.int8)
    for i in range(h1_n):
        sig[i*12:i*12+12] = slope[i]
    return sig

def sig_m5_sma(cl, period):
    """M5 SMA-period slope. +1 above SMA, -1 below."""
    sma = pd.Series(cl).rolling(period, min_periods=period).mean().values
    return np.where(np.isnan(sma), 0, np.sign(cl - sma)).astype(np.int8)

def sig_m5_atr(op, hi, lo, cl, threshold):
    """M5 Wilder ATR14 range burst. Direction from bar close vs open."""
    n = len(cl)
    tr = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    # Wilder smoothing (EMA with alpha=1/14)
    atr = np.empty(n)
    atr[0] = tr[0]
    alpha = 1.0/14
    for i in range(1, n):
        atr[i] = atr[i-1]*(1-alpha) + tr[i]*alpha
    rng = hi - lo
    burst = rng > threshold * atr
    direction = np.where(cl >= op, np.int8(1), np.int8(-1))
    return np.where(burst, direction, np.int8(0)).astype(np.int8)

def sig_h1m5_combo(cl, op, hi, lo, h1_p, m5_p):
    """H1 SMA-h1_p AND M5 SMA-m5_p must agree. +1 both up, -1 both down, 0 conflict."""
    h1 = sig_h1_sma(cl, h1_p)
    m5 = sig_m5_sma(cl, m5_p)
    return np.where(h1 == m5, h1, np.int8(0)).astype(np.int8)

# ─── Numba simulation ────────────────────────────────────────────────────────

@njit
def sim_zr_sig(op,hi,lo,cl,sig,pip,spread,pf,ml,N,zw,tgt,ta,td,sig_mode):
    """
    sig_mode=0: filter  — alternating, skip when sig contradicts d
    sig_mode=1: sig_only — enter in sig direction, drop alternating
    sig[i]: +1 long, -1 short, 0 no signal
    Returns: total_pips, n_cycles, n_entries_tried, n_trail_exit, n_zr_exit, avg_legs
    """
    n=len(cl); total=0.0; nc=0; nt_entries=0
    n_trail=0; n_zr=0; legs_acc=0.0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    i=0; d=1
    while i<n:
        nt_entries+=1
        s=sig[i]
        if sig_mode==0:    # filter: skip if signal contradicts
            if s==0 or (s!=0 and s!=d):
                i+=N; continue
        else:              # signal_only: enter in signal direction
            if s==0:
                i+=N; continue
            d=int(s)

        e=cl[i]
        if d==1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:    lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i+=1
        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl==1:
                cur_mfe=(h-e)/pip if d==1 else (e-l)/pip
                if cur_mfe>peak_mfe: peak_mfe=cur_mfe
                if peak_mfe>=ta: trail_on=True
                if trail_on:
                    if d==1:
                        ts=e+(peak_mfe-td)*pip
                        if l<=ts:
                            total+=(ts-e)/pip-spread; nc+=1; n_trail+=1
                            legs_acc+=1.0; ex=True
                    else:
                        ts=e-(peak_mfe-td)*pip
                        if h>=ts:
                            total+=(e-ts)/pip-spread; nc+=1; n_trail+=1
                            legs_acc+=1.0; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1
                    legs_acc+=float(nl); n_zr+=(nl>1); ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1
                    legs_acc+=float(nl); n_zr+=(nl>1); ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; nc+=1; legs_acc+=float(nl); n_zr+=(nl>1); ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; legs_acc+=float(nl); ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; nc+=1; legs_acc+=float(nl); n_zr+=(nl>1); ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; legs_acc+=float(nl); ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        if sig_mode==0: d=-d
        i+=N-1
    return total, nc, nt_entries, n_trail, n_zr, legs_acc/max(nc,1)


# Warm-up compile
_df0=pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(float); _h=_df0.high.values[:2000].astype(float)
_l=_df0.low.values[:2000].astype(float); _c=_df0.close.values[:2000].astype(float)
_sig=np.ones(2000,dtype=np.int8)
sim_zr_sig(_o,_h,_l,_c,_sig,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.,10.,5.,0)
sim_zr_sig(_o,_h,_l,_c,_sig,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.,10.,5.,1)
print("JIT compiled\n")

OOS_FRAC=0.30; WF_CHUNKS=3

rows=[]

for pair, cfg in PAIR_CFG.items():
    pip=PIP_MAP[pair]
    N=cfg['N']; zw=cfg['zw']; tgt=cfg['tgt']
    ta=float(cfg['ta']); td=float(cfg['td'])

    df=pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(float); hi=df.high.values.astype(float)
    lo=df.low.values.astype(float); cl=df.close.values.astype(float)
    nb=len(cl)

    oos_start=int(nb*(1-OOS_FRAC))
    oo=op[oos_start:]; oh=hi[oos_start:]; ol=lo[oos_start:]; oc=cl[oos_start:]
    oos_bars=len(oc); oos_td=oos_bars/(24*12)
    chunk_sz=oos_bars//WF_CHUNKS

    def run_one(sig_oos, sig_mode, label):
        tot,nc,nte,ntr,nzr,avgl = sim_zr_sig(
            oo,oh,ol,oc,sig_oos,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,ta,td,sig_mode)
        ppd=tot/oos_td; ppc=tot/max(nc,1)
        entry_rate=100*nc/max(nte,1)
        trail_pct=100*ntr/max(nc,1); zr_pct=100*nzr/max(nc,1)
        wf=0
        for ch in range(WF_CHUNKS):
            s=ch*chunk_sz; e2=(ch+1)*chunk_sz if ch<WF_CHUNKS-1 else oos_bars
            sv=sig_oos[s:e2]
            ct,cnc,_,_,_,_ = sim_zr_sig(
                oo[s:e2],oh[s:e2],ol[s:e2],oc[s:e2],sv,
                pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,ta,td,sig_mode)
            wf+=(ct>0)
        rows.append(dict(pair=pair,signal=label,
                         ppd=round(ppd,1),ppc=round(ppc,1),
                         entry_rate=round(entry_rate,1),
                         trail_pct=round(trail_pct,1),zr_pct=round(zr_pct,1),
                         avg_legs=round(avgl,2),wf=wf,n_cycles=nc))
        return ppd, ppc, entry_rate, wf

    print(f"\n{'='*70}\n{pair}  N={N} ZW={zw} tgt={tgt} ta={ta} td={td}\n{'='*70}")

    # Baseline (no signal = all ones, filter mode = always agree)
    full_sig=np.ones(oos_bars,dtype=np.int8)
    ppd,ppc,er,wf=run_one(full_sig,0,'no_signal')
    print(f"  {'no_signal':<22} ppd={ppd:7.1f} ppc={ppc:7.1f} entry%={er:5.1f} trail%={rows[-1]['trail_pct']:5.1f} zr%={rows[-1]['zr_pct']:5.1f} wf={wf}/3")

    # --- H1 SMA slopes ---
    for p in [5, 10, 20]:
        # Compute on full data, then slice OOS
        s_full=sig_h1_sma(cl, p)
        s_oos=s_full[oos_start:]
        for mode, mlabel in [(0,'filter'),(1,'only')]:
            lbl=f'h1_sma{p}_{mlabel}'
            ppd,ppc,er,wf=run_one(s_oos,mode,lbl)
            print(f"  {lbl:<22} ppd={ppd:7.1f} ppc={ppc:7.1f} entry%={er:5.1f} trail%={rows[-1]['trail_pct']:5.1f} zr%={rows[-1]['zr_pct']:5.1f} wf={wf}/3")

    # --- M5 SMA slopes ---
    for p in [5, 10, 20]:
        s_full=sig_m5_sma(cl, p)
        s_oos=s_full[oos_start:]
        for mode, mlabel in [(0,'filter'),(1,'only')]:
            lbl=f'm5_sma{p}_{mlabel}'
            ppd,ppc,er,wf=run_one(s_oos,mode,lbl)
            print(f"  {lbl:<22} ppd={ppd:7.1f} ppc={ppc:7.1f} entry%={er:5.1f} trail%={rows[-1]['trail_pct']:5.1f} zr%={rows[-1]['zr_pct']:5.1f} wf={wf}/3")

    # --- M5 ATR burst ---
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        s_full=sig_m5_atr(op,hi,lo,cl,k)
        s_oos=s_full[oos_start:]
        for mode, mlabel in [(0,'filter'),(1,'only')]:
            lbl=f'm5_atr{k:.1f}_{mlabel}'
            ppd,ppc,er,wf=run_one(s_oos,mode,lbl)
            print(f"  {lbl:<22} ppd={ppd:7.1f} ppc={ppc:7.1f} entry%={er:5.1f} trail%={rows[-1]['trail_pct']:5.1f} zr%={rows[-1]['zr_pct']:5.1f} wf={wf}/3")

    # --- H1+M5 combo ---
    for h1p,m5p in [(10,5),(20,5),(10,10),(20,10)]:
        s_full=sig_h1m5_combo(cl,op,hi,lo,h1p,m5p)
        s_oos=s_full[oos_start:]
        for mode, mlabel in [(0,'filter'),(1,'only')]:
            lbl=f'h1{h1p}_m5{m5p}_{mlabel}'
            ppd,ppc,er,wf=run_one(s_oos,mode,lbl)
            print(f"  {lbl:<22} ppd={ppd:7.1f} ppc={ppc:7.1f} entry%={er:5.1f} trail%={rows[-1]['trail_pct']:5.1f} zr%={rows[-1]['zr_pct']:5.1f} wf={wf}/3")

df_res=pd.DataFrame(rows)
out='/path/to/projects/fx-core/research/experiments/zone_recovery/zr_signal_entry_results.csv'
df_res.to_csv(out,index=False)
print(f"\nSaved {len(df_res)} rows → {out}")

print("\n\n=== TOP PERFORMERS PER PAIR (WF=3, vs baseline) ===")
for pair in PAIR_CFG:
    sub=df_res[df_res.pair==pair].copy()
    base=sub[sub.signal=='no_signal'].iloc[0]
    wf3=sub[sub.wf==3].sort_values('ppd',ascending=False)
    print(f"\n{pair} | BASELINE {base.ppd:.1f} p/d")
    print(wf3[['signal','ppd','ppc','entry_rate','trail_pct','zr_pct','avg_legs','wf']].head(10).to_string(index=False))
