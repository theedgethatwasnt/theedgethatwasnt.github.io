"""
ZR MAX_LEGS design sweep.

Insight from param sweep: bounded progression requires TGT >> ZW, but large TGT
causes deep cycles (escape rarely hit) that explode at max_legs=10.

Resolution: engineer MAX_LEGS as a hard design constraint.
  - Choose (ZW, TGT) so that vol AT max_legs is physically achievable
  - Accept: cycles that reach max_legs close at market (small bounded loss)
  - Deep cycles become less frequent as TGT grows (price escapes more reliably)
  - Trade-off: fewer deep cycles vs larger vol when they DO occur

For each (ZW, TGT, max_legs) triple:
  - Vol at max_legs (the "engineered max exposure")
  - Expected loss per deep cycle (vol_at_max_legs × ZW/2 avg pips from target)
  - Fraction of cycles reaching max_legs
  - Full IS/OOS WF + MC validation on EUR_JPY M5
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'backtest_zr_ml_sweep_results.csv'

PF        = 1.25
OOS_FRAC  = 0.30
IS_CHUNKS = 3
OOS_CHUNKS= 3
N_BOOT    = 1000

PAIR = "EUR_JPY"
PIP  = 0.01
TA   = 6.0
TD   = 1.0


def compute_progression(zw, tgt, sp, pf, ml):
    uzb, lzb = 0.0, -zw
    uetgt, letgt = tgt, -(zw + tgt)
    legs = [(1.0, +1.0, uzb)]
    vols = [1]
    for leg_n in range(1, ml):
        is_lzb = (leg_n % 2 == 1)
        target = letgt if is_lzb else uetgt
        new_dir = -1.0 if is_lzb else +1.0
        new_entry = lzb if is_lzb else uzb
        net = sum(v * d * (target - e) for v, d, e in legs)
        net -= sum(v for v, d, e in legs) * sp
        if net >= 0:
            vols.append(0)
        else:
            npu = max(tgt - sp, 1e-8)
            v = max(1.0, math.ceil(-net / npu * pf))
            vols.append(int(v))
            legs.append((v, new_dir, new_entry))
    return vols, sum(vols)


@njit
def sim_zr_ml(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td, max_entry_spread):
    n = len(cl)
    cycle_pnl  = np.zeros(n, dtype=np.float64)
    cycle_legs = np.zeros(n, dtype=np.int32)
    nc = 0; n_trail = 0; n_zr = 0; n_ml = 0; n_skipped = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_entry = spread_arr[i]
        if max_entry_spread > 0 and sp_entry > max_entry_spread:
            n_skipped += 1; i += 1; continue
        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=(c >= op[i])
            sp = spread_arr[i]
            if nl == 1:
                cur_mfe = (h-e)/pip if d==1 else (e-l)/pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if d == 1:
                        be = e + sp * pip
                        ts = e + (peak_mfe - td) * pip
                        if ts < be: ts = be
                        if l <= ts:
                            cycle_pnl[nc]=(ts-e)/pip-sp; cycle_legs[nc]=nl
                            nc+=1; n_trail+=1; ex=True
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip-sp; cycle_legs[nc]=nl
                            nc+=1; n_trail+=1; ex=True
            if ex: break
            for pass_idx in range(2):
                if ex: break
                is_high = (bull == (pass_idx == 0))
                if is_high and h >= uz and lu != i:
                    lu = i
                    net_at_ut = 0.0; tv = 0.0
                    for k in range(nl):
                        net_at_ut += lv[k]*ld[k]*(ut-lp[k])/pip; tv += lv[k]
                    net_at_ut -= tv * sp
                    if net_at_ut < 0:
                        npu = tgt - sp
                        if npu <= 1e-8: npu = 1e-8
                        v = max(1.0, math.ceil(-net_at_ut / npu * pf))
                        if nl >= ml:
                            net_cl = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net_cl += lv[k]*ld[k]*(c-lp[k])/pip; tv2 += lv[k]
                            cycle_pnl[nc]=net_cl-tv2*sp; cycle_legs[nc]=nl
                            nc+=1; n_ml+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_high) and l <= lz and ll != i:
                    ll = i
                    net_at_lt = 0.0; tv = 0.0
                    for k in range(nl):
                        net_at_lt += lv[k]*ld[k]*(lt-lp[k])/pip; tv += lv[k]
                    net_at_lt -= tv * sp
                    if net_at_lt < 0:
                        npu = tgt - sp
                        if npu <= 1e-8: npu = 1e-8
                        v = max(1.0, math.ceil(-net_at_lt / npu * pf))
                        if nl >= ml:
                            net_cl = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net_cl += lv[k]*ld[k]*(c-lp[k])/pip; tv2 += lv[k]
                            cycle_pnl[nc]=net_cl-tv2*sp; cycle_legs[nc]=nl
                            nc+=1; n_ml+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(ut-lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc]=net_exit-tv*sp; cycle_legs[nc]=nl
                    nc+=1; n_zr+=1; ex=True; break
                if l <= lt <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(lt-lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc]=net_exit-tv*sp; cycle_legs[nc]=nl
                    nc+=1; n_zr+=1; ex=True; break
            i += 1
        d = -d
    return cycle_pnl[:nc], cycle_legs[:nc], nc, n_trail, n_zr, n_ml, n_skipped


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, 1.4)
_o = _df0.open.values[:2000].astype(np.float64)
_h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64)
_c = _df0.close.values[:2000].astype(np.float64)
sim_zr_ml(_o, _h, _l, _c, _sp0, 0.0001, PF, 10, 30., 15., 5., 1., 0.)
print("done.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
df_mid = pd.read_parquet(DATA_DIR_MID/f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
df_ba  = pd.read_parquet(DATA_DIR_BA /f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
df_mid['ts_key'] = df_mid['timestamp'].astype(str).str[:19]
df_ba['ts_key']  = df_ba['timestamp'].astype(str).str[:19]
merged = df_mid.merge(df_ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner').sort_values('ts_key').reset_index(drop=True)

nb = len(merged)
is_end = int(nb * (1 - OOS_FRAC))
is_chunk_sz  = is_end // IS_CHUNKS
oos_len      = nb - is_end
oos_chunk_sz = oos_len // OOS_CHUNKS
oos_days     = oos_len / (24 * 12)

op = merged.open.values.astype(np.float64)
hi = merged.high.values.astype(np.float64)
lo = merged.low.values.astype(np.float64)
cl = merged.close.values.astype(np.float64)
sp = ((merged.ask_c - merged.bid_c) / PIP).clip(lower=0.3).values.astype(np.float64)

sp_is    = sp[:is_end]
gate_thr = float(np.percentile(sp_is, 90))
sp_med   = float(np.median(sp_is))

print(f"{PAIR}  IS spread: med={sp_med:.2f}p  p90(gate)={gate_thr:.2f}p")
print(f"IS bars: {is_end}  OOS bars: {oos_len} ({oos_days:.1f} days)\n")

# ── Grid: ZW × TGT × MAX_LEGS ────────────────────────────────────────────────
# Focus on TGT ≥ ZW (bounded-progression zone) with ml=3,4,5,6
# Plus reference ZW=50 TGT=25 at all ml values
grid = []

# Bounded-progression candidates (TGT ≥ 1.3×ZW gives ratio ≤ 2.2)
candidates = [
    (20, 30), (20, 40), (20, 50),
    (25, 35), (25, 50), (25, 60),
    (30, 45), (30, 60), (30, 75),
    (40, 60), (40, 80), (40, 100),
    (50, 75), (50, 100), (50, 125),
]
ml_values = [3, 4, 5, 6, 8]

for (zw, tgt) in candidates:
    if tgt <= gate_thr: continue
    for ml in ml_values:
        grid.append((zw, tgt, ml))

# Reference: current config with varying ml
for ml in ml_values:
    grid.append((50, 25, ml))

rng  = np.random.default_rng(42)
rows = []

sep = "─" * 120
print(sep)
print(f"  {'ZW':>4} {'TGT':>5} {'ml':>3} | "
      f"{'vols at ml':>20} | {'total':>6} | "
      f"{'p/d':>8} {'minT':>7} | "
      f"{'IS':>3} {'OOS':>4} | "
      f"{'P5':>7} {'P(+)':>6} | "
      f"{'ml%':>5} {'5+%':>5} | status")
print(sep)

for zw, tgt, ml in grid:
    vols, total = compute_progression(zw, tgt, sp_med, PF, ml)
    vols_str = "→".join(str(v) for v in vols[-3:])  # show last 3 legs

    cyc, legs_arr, nc, nt, nz, nml, ns = sim_zr_ml(
        op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
        sp[is_end:], PIP, PF, ml, zw, tgt, TA, TD, gate_thr)

    if nc == 0:
        continue

    ppd  = cyc.sum() / oos_days
    minT = float(cyc.min())
    ml_pct = nml / nc * 100
    l5_pct = float(np.mean(legs_arr >= 5)) * 100

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s = ch * is_chunk_sz
        e2 = (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end
        c2, _, nc2, *_ = sim_zr_ml(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                                    sp[s:e2], PIP, PF, ml, zw, tgt, TA, TD, gate_thr)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s = is_end + ch * oos_chunk_sz
        e2 = is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb
        c2, _, nc2, *_ = sim_zr_ml(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                                    sp[s:e2], PIP, PF, ml, zw, tgt, TA, TD, gate_thr)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and not math.isnan(p5) and p5 > 0 and prob > 0.95:
        status = "🟢 PASS"
    elif ppd > 0 and is_wf >= 2 and oos_wf >= 2:
        status = "🟡 near"
    elif ppd < 0:
        status = "🔴 neg"
    else:
        status = f"IS={is_wf}/3 OOS={oos_wf}/3"

    print(f"  {zw:>4.0f} {tgt:>5.0f} {ml:>3} | "
          f"{vols_str:>20} | {total:>6} | "
          f"{ppd:>8.1f} {minT:>7.2f} | "
          f"{is_wf:>3} {oos_wf:>4} | "
          f"{p5:>7.1f} {prob:>6.3f} | "
          f"{ml_pct:>5.1f} {l5_pct:>5.2f} | {status}")
    sys.stdout.flush()

    rows.append(dict(
        zw=zw, tgt=tgt, ml=ml,
        total_vol=total,
        ppd=round(ppd, 1), minT=round(minT, 2),
        is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5, 1) if not math.isnan(p5) else None,
        prob=round(prob, 3) if not math.isnan(prob) else None,
        ml_pct=round(ml_pct, 2),
        l5_pct=round(l5_pct, 2),
    ))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

print("\n=== PASSING CONFIGS ===")
print(f"  {'ZW':>4} {'TGT':>5} {'ml':>3} | {'total_vol':>9} | "
      f"{'p/d':>8} | {'IS':>3} {'OOS':>4} | {'P5':>7} {'P(+)':>6} | {'ml%':>5} {'5+%':>5}")
for r in sorted([x for x in rows if x.get('p5') and x['p5']>0 and x.get('prob',0)>0.95
                  and x['is_wf']==IS_CHUNKS and x['oos_wf']==OOS_CHUNKS],
                key=lambda x: x['ppd'], reverse=True):
    print(f"  {r['zw']:>4.0f} {r['tgt']:>5.0f} {r['ml']:>3} | {r['total_vol']:>9} | "
          f"{r['ppd']:>8.1f} | {r['is_wf']:>3} {r['oos_wf']:>4} | "
          f"{r['p5']:>7.1f} {r['prob']:>6.3f} | {r['ml_pct']:>5.2f} {r['l5_pct']:>5.2f}")
