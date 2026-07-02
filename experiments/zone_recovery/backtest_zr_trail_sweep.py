"""
Trailing stop sweep on top-3 pairs' IS-validated sweetspots.
Per-pair config (from IS sweep):
  CHF_JPY: N=1, ZW=40, tgt=20p
  GBP_USD: N=6, ZW=30, tgt=15p
  USD_JPY: N=1, ZW=40, tgt=20p
Sweep: trail_act x trail_dist (td < ta), OOS data, WF 3-chunk.
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PIP_MAP     = {"CHF_JPY":0.01,"GBP_USD":0.0001,"USD_JPY":0.01}
PIP_USD_MAP = {"CHF_JPY":0.000107,"GBP_USD":0.0001,"USD_JPY":0.000064}
SPREAD=1.4; MAX_LEGS=10; PF=1.25

PAIR_CFG = {
    "CHF_JPY": dict(N=1, zw=40.0, tgt=20.0),
    "GBP_USD": dict(N=6, zw=30.0, tgt=15.0),
    "USD_JPY": dict(N=1, zw=40.0, tgt=20.0),
}

TRAIL_ACTS  = [5, 7, 10, 14, 20, 30]
TRAIL_DISTS = [3, 5, 7, 10]

@njit
def sim_zr_trail(op, hi, lo, cl, pip, spread, pf, ml, N, zw, tgt, ta, td):
    n = len(cl)
    total = 0.0; nc = 0; nm = 0; legs_acc = 0.0
    n_trail = 0; n_zr = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        e = cl[i]
        if d == 1:
            uz = e; lz = e - zw*pip; ut = e + tgt*pip; lt = lz - tgt*pip
        else:
            lz = e; uz = e + zw*pip; lt = e - tgt*pip; ut = uz + tgt*pip
        lv[0] = 1.0; ld[0] = float(d); lp[0] = e
        nl = 1; lu = ll = -1; ex = False
        peak_mfe = 0.0; trail_on = False
        i += 1
        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; bull = c >= op[i]

            # Trailing stop: single-leg only
            if nl == 1:
                cur_mfe = (h - e) / pip if d == 1 else (e - l) / pip
                if cur_mfe > peak_mfe:
                    peak_mfe = cur_mfe
                if peak_mfe >= ta:
                    trail_on = True
                if trail_on:
                    if d == 1:
                        ts = e + (peak_mfe - td) * pip
                        if l <= ts:
                            total += (ts - e) / pip - spread
                            nc += 1; n_trail += 1; legs_acc += 1.0; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts:
                            total += (e - ts) / pip - spread
                            nc += 1; n_trail += 1; legs_acc += 1.0; ex = True
            if ex:
                break

            # Target exits + ZR leg additions
            for pn in range(2):
                if ex:
                    break
                dh = (bull and pn == 0) or (not bull and pn == 1)
                if l <= ut <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    total += net - tv*spread; nc += 1
                    legs_acc += float(nl); n_zr += (nl > 1); ex = True; break
                if l <= lt <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    total += net - tv*spread; nc += 1
                    legs_acc += float(nl); n_zr += (nl > 1); ex = True; break
                if dh and h >= uz and lu != i:
                    lu = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c >= ut:
                            total += nt2; nc += 1
                            legs_acc += float(nl); n_zr += (nl > 1); ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            total += net - tv2*spread; nc += 1; nm += 1
                            legs_acc += float(nl); ex = True; break
                        lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1
                if not dh and l <= lz and ll != i:
                    ll = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c <= lt:
                            total += nt2; nc += 1
                            legs_acc += float(nl); n_zr += (nl > 1); ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            total += net - tv2*spread; nc += 1; nm += 1
                            legs_acc += float(nl); ex = True; break
                        lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1
            i += 1
        d = -d
        i += N - 1
    return total, nc, nm, legs_acc / max(nc, 1), n_trail, n_zr


# Warm-up compile
_df0 = pd.read_parquet(DATA_DIR / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(float); _h=_df0.high.values[:2000].astype(float)
_l=_df0.low.values[:2000].astype(float); _c=_df0.close.values[:2000].astype(float)
sim_zr_trail(_o,_h,_l,_c,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.,10.,5.)
print("JIT compiled\n")

OOS_FRAC = 0.30
WF_CHUNKS = 3

rows = []
for pair, cfg in PAIR_CFG.items():
    pip = PIP_MAP[pair]
    N = cfg['N']; zw = cfg['zw']; tgt = cfg['tgt']

    df = pd.read_parquet(DATA_DIR / f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op = df.open.values.astype(float); hi = df.high.values.astype(float)
    lo = df.low.values.astype(float); cl = df.close.values.astype(float)
    nb = len(cl)

    oos_start = int(nb * (1 - OOS_FRAC))
    oo = op[oos_start:]; oh = hi[oos_start:]
    ol = lo[oos_start:]; oc = cl[oos_start:]
    oos_bars = len(oc)
    oos_td = oos_bars / (24*12)  # 24h * 12 bars/h

    chunk_sz = oos_bars // WF_CHUNKS

    # Baseline: ta=9999 disables trail
    bt,bnc,bnm,bal,_,_ = sim_zr_trail(oo,oh,ol,oc,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,9999.,9998.)
    base_ppd = bt / oos_td
    bwf = 0
    for ch in range(WF_CHUNKS):
        s=ch*chunk_sz; e2=(ch+1)*chunk_sz if ch<WF_CHUNKS-1 else oos_bars
        ct,_,_,_,_,_ = sim_zr_trail(oo[s:e2],oh[s:e2],ol[s:e2],oc[s:e2],pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,9999.,9998.)
        bwf += (ct > 0)
    rows.append(dict(pair=pair,ta='BASE',td='BASE',
                     ppd=round(base_ppd,1),ppc=round(bt/max(bnc,1),1),
                     avg_legs=round(bal,2),trail_pct=0,zr_pct=100,wf=bwf))
    print(f"{pair} BASE ppd={base_ppd:.1f} ppc={bt/max(bnc,1):.1f} wf={bwf}/3")

    for ta in TRAIL_ACTS:
        for td in TRAIL_DISTS:
            if td >= ta:
                continue
            tot,nc,nm,avg_l,n_tr,n_zr = sim_zr_trail(
                oo,oh,ol,oc,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,float(ta),float(td))
            ppd=tot/oos_td; ppc=tot/max(nc,1)
            trail_pct=100*n_tr/max(nc,1); zr_pct=100*n_zr/max(nc,1)
            wf=0
            for ch in range(WF_CHUNKS):
                s=ch*chunk_sz; e2=(ch+1)*chunk_sz if ch<WF_CHUNKS-1 else oos_bars
                ct,_,_,_,_,_ = sim_zr_trail(oo[s:e2],oh[s:e2],ol[s:e2],oc[s:e2],pip,SPREAD,PF,MAX_LEGS,N,zw,tgt,float(ta),float(td))
                wf += (ct > 0)
            rows.append(dict(pair=pair,ta=ta,td=td,
                             ppd=round(ppd,1),ppc=round(ppc,1),
                             avg_legs=round(avg_l,2),trail_pct=round(trail_pct,1),
                             zr_pct=round(zr_pct,1),wf=wf))

df_res = pd.DataFrame(rows)
out = '/path/to/projects/fx-core/research/experiments/zone_recovery/zr_trail_sweep_results.csv'
df_res.to_csv(out, index=False)
print(f"\nSaved {len(df_res)} rows → {out}")
print("\n=== TOP CONFIGS PER PAIR (WF=3, sorted ppd) ===")
for pair in PAIR_CFG:
    sub = df_res[df_res.pair==pair].copy()
    base = sub[sub.ta=='BASE'].iloc[0]
    wf3 = sub[sub.wf==3].sort_values('ppd', ascending=False)
    print(f"\n{pair} | BASE: {base.ppd} p/d | WF3 configs ({len(wf3)}):")
    print(wf3[['ta','td','ppd','ppc','avg_legs','trail_pct','zr_pct','wf']].head(8).to_string(index=False))
    if len(wf3) == 0:
        best = sub.sort_values(['wf','ppd'], ascending=[False,False])
        print("  (none WF=3) best:")
        print(best[['ta','td','ppd','ppc','avg_legs','trail_pct','zr_pct','wf']].head(5).to_string(index=False))
