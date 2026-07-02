"""
Regime drift diagnostic: for top-3 pairs, split OOS into 6 ~95-day windows.
For each window run full grid sweep (ZW x N x tgt_f x trail_act x trail_dist).
Report winning params per window to measure how much sweetspot drifts over time.
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path
from itertools import product

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PIP_MAP     = {"CHF_JPY":0.01,"GBP_USD":0.0001,"USD_JPY":0.01}
PIP_USD_MAP = {"CHF_JPY":0.000107,"GBP_USD":0.0001,"USD_JPY":0.000064}
SPREAD=1.4; MAX_LEGS=10; PF=1.25

PAIR_CFG = {
    "CHF_JPY": dict(N_ref=1, zw_ref=40.0, tgt_ref=20.0, ta_ref=5, td_ref=3),
    "GBP_USD": dict(N_ref=6, zw_ref=30.0, tgt_ref=15.0, ta_ref=10, td_ref=7),
    "USD_JPY": dict(N_ref=1, zw_ref=40.0, tgt_ref=20.0, ta_ref=10, td_ref=5),
}

# Search grid
ZWS    = [20, 25, 30, 35, 40, 45, 50, 55, 60]
NS     = [1, 2, 3, 6, 12]
TGT_FS = [0.25, 0.50, 1.00]
TAS    = [5, 7, 10, 14, 20, 30]
TDS    = [3, 5, 7, 10]
# Precompute valid (ta, td) combos
TATD = [(ta, td) for ta in TAS for td in TDS if td < ta]

@njit
def sim_zr_trail(op, hi, lo, cl, pip, spread, pf, ml, N, zw, tgt, ta, td):
    n = len(cl)
    total = 0.0; nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        e = cl[i]
        if d == 1:
            uz = e; lz = e - zw*pip; ut = e + tgt*pip; lt = lz - tgt*pip
        else:
            lz = e; uz = e + zw*pip; lt = e - tgt*pip; ut = uz + tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False
        peak_mfe=0.0; trail_on=False
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl == 1:
                cur_mfe = (h-e)/pip if d==1 else (e-l)/pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if d == 1:
                        ts = e + (peak_mfe-td)*pip
                        if l <= ts:
                            total += (ts-e)/pip - spread
                            nc += 1; ex = True
                    else:
                        ts = e - (peak_mfe-td)*pip
                        if h >= ts:
                            total += (e-ts)/pip - spread
                            nc += 1; ex = True
            if ex: break
            for pn in range(2):
                if ex: break
                dh = (bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2 -= tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2 -= tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1
        d=-d; i+=N-1
    return total, nc

# Warm-up compile
_df0=pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(float); _h=_df0.high.values[:2000].astype(float)
_l=_df0.low.values[:2000].astype(float); _c=_df0.close.values[:2000].astype(float)
sim_zr_trail(_o,_h,_l,_c,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.,10.,5.)
print("JIT compiled\n")

OOS_FRAC = 0.30
N_WINDOWS = 6
MIN_CYCLES = 10

all_rows = []

for pair, cfg in PAIR_CFG.items():
    pip = PIP_MAP[pair]
    df = pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(float); hi=df.high.values.astype(float)
    lo=df.low.values.astype(float); cl=df.close.values.astype(float)
    nb = len(cl)

    oos_start = int(nb*(1-OOS_FRAC))
    oo=op[oos_start:]; oh=hi[oos_start:]; ol=lo[oos_start:]; oc=cl[oos_start:]
    oos_bars = len(oc)
    win_sz = oos_bars // N_WINDOWS

    print(f"\n{'='*60}")
    print(f"{pair} | OOS bars={oos_bars} | window_sz={win_sz} ({win_sz/(24*12):.0f} days)")
    print(f"Ref config: N={cfg['N_ref']} ZW={cfg['zw_ref']} tgt={cfg['tgt_ref']} ta={cfg['ta_ref']} td={cfg['td_ref']}")
    print(f"{'='*60}")

    for w in range(N_WINDOWS):
        s = w*win_sz
        e2 = (w+1)*win_sz if w<N_WINDOWS-1 else oos_bars
        wo=oo[s:e2]; wh=oh[s:e2]; wl=ol[s:e2]; wc=oc[s:e2]
        w_days = (e2-s)/(24*12)
        best_ppd=-1e9; best_cfg=None

        for N in NS:
            for zw in ZWS:
                for tgt_f in TGT_FS:
                    tgt = zw*tgt_f
                    if tgt < 5: continue  # skip degenerate
                    for (ta,td) in TATD:
                        tot, nc = sim_zr_trail(wo,wh,wl,wc,pip,SPREAD,PF,MAX_LEGS,
                                               N,float(zw),tgt,float(ta),float(td))
                        if nc < MIN_CYCLES: continue
                        ppd = tot/(e2-s)*(24*12)
                        if ppd > best_ppd:
                            best_ppd=ppd; best_cfg=(N,zw,tgt_f,tgt,ta,td,nc)

        # Also check reference config
        N_r=cfg['N_ref']; zw_r=cfg['zw_ref']; tgt_r=cfg['tgt_ref']
        ta_r=float(cfg['ta_ref']); td_r=float(cfg['td_ref'])
        ref_tot, ref_nc = sim_zr_trail(wo,wh,wl,wc,pip,SPREAD,PF,MAX_LEGS,N_r,zw_r,tgt_r,ta_r,td_r)
        ref_ppd = ref_tot/(e2-s)*(24*12) if ref_nc>0 else 0

        if best_cfg:
            bN,bZW,btf,btgt,bta,btd,bnc = best_cfg
            print(f"  W{w+1} ({w*win_sz/(24*12):.0f}–{e2/(24*12):.0f}d): "
                  f"WINNER N={bN} ZW={bZW} tgt_f={btf} ta={bta} td={btd} "
                  f"→ {best_ppd:.0f} p/d | REF: {ref_ppd:.0f} p/d (Δ={best_ppd-ref_ppd:+.0f})")
            all_rows.append(dict(pair=pair, window=w+1,
                                 N=bN, ZW=bZW, tgt_f=btf, tgt=btgt, ta=bta, td=btd,
                                 ppd=round(best_ppd,1), ref_ppd=round(ref_ppd,1),
                                 delta=round(best_ppd-ref_ppd,1), n_cycles=bnc))
        else:
            print(f"  W{w+1}: no valid config found")

    # Drift stats across windows
    sub = [r for r in all_rows if r['pair']==pair]
    if len(sub)>=2:
        zws = [r['ZW'] for r in sub]
        tas = [r['ta'] for r in sub]
        tds = [r['td'] for r in sub]
        ns  = [r['N']  for r in sub]
        print(f"\n  DRIFT STATS (across {len(sub)} windows):")
        print(f"    ZW:  mean={np.mean(zws):.1f}  std={np.std(zws):.1f}  range=[{min(zws)},{max(zws)}]")
        print(f"    ta:  mean={np.mean(tas):.1f}  std={np.std(tas):.1f}  range=[{min(tas)},{max(tas)}]")
        print(f"    td:  mean={np.mean(tds):.1f}  std={np.std(tds):.1f}  range=[{min(tds)},{max(tds)}]")
        print(f"    N:   mean={np.mean(ns):.1f}  std={np.std(ns):.1f}  range=[{min(ns)},{max(ns)}]")

df_out = pd.DataFrame(all_rows)
out = '/path/to/projects/fx-core/research/experiments/zone_recovery/zr_regime_drift.csv'
df_out.to_csv(out, index=False)
print(f"\nSaved {len(df_out)} rows → {out}")
