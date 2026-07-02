"""
ZR Parameter Sweep — ZW × TGT space.

Design philosophy: engineer (ZW, TGT) so that the volume at MAX_LEGS=10
is a known, physically achievable number — not "survivable because it's
rare." Most cycles should resolve in ≤4 legs; an occasional 5-leg cycle
is acceptable. The worst-case (10-leg) must B/E or better at the escape
target while staying within margin limits.

Key math: per-leg volume ratio ≈ (ZW/TGT + 1) × PF
  Current deployed: ZW=50 TGT=25 → ratio ≈ 3.0 → leg-10 = 645K units (unusable)
  Target: ratio ≤ 1.9 → TGT ≥ ~1.4×ZW

Sweep:
  ZW:  [10, 15, 20, 25, 30, 40, 50]
  TGT: for each ZW, try fractions 0.5..3.0 × ZW (round to nearest 5p)
  ta=6, td=1 (best from fixed-sim single-ZW sweep)

For each combo reports:
  - Exact volume at each leg 1-10 (iterative, not approximation)
  - Total volume and ratio
  - IS/OOS WF walk-forward + MC bootstrap (EUR_JPY M5)
  - Leg-depth: fraction of cycles that reach 1/2/3/4/5+ legs
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'backtest_zr_param_sweep_results.csv'

MAX_LEGS  = 10
PF        = 1.25
OOS_FRAC  = 0.30
IS_CHUNKS = 3
OOS_CHUNKS= 3
N_BOOT    = 1000     # faster; increase to 2000 for final validation

PAIR   = "EUR_JPY"
PIP    = 0.01
TA     = 6.0
TD     = 1.0

# ── Volume progression calculator ────────────────────────────────────────────
def compute_progression(zw, tgt, sp, pf, ml=10):
    """
    Exact iterative computation of volume at each leg for worst-case
    alternating crossing sequence (LZB, UZB, LZB, ...).
    Returns: (vols list, total_vol, ratio per leg list)
    """
    uzb, lzb = 0.0, -zw
    uetgt, letgt = tgt, -(zw + tgt)
    legs = [(1.0, +1.0, uzb)]   # vol, dir, entry_rel
    vols = [1]
    for leg_n in range(1, ml):
        is_lzb = (leg_n % 2 == 1)  # odd legs cross LZB, even legs cross UZB
        if is_lzb:
            target = letgt; new_dir = -1.0; new_entry = lzb
        else:
            target = uetgt; new_dir = +1.0; new_entry = uzb
        net = sum(v * d * (target - e) for v, d, e in legs)
        net -= sum(v for v, d, e in legs) * sp
        if net >= 0:
            vols.append(0)
        else:
            npu = tgt - sp
            if npu <= 0: npu = 1e-8
            v = max(1.0, math.ceil(-net / npu * pf))
            vols.append(int(v))
            legs.append((v, new_dir, new_entry))
    total = sum(vols)
    ratios = [vols[i]/vols[i-1] if vols[i-1] > 0 and vols[i] > 0 else 0
              for i in range(1, len(vols))]
    return vols, total, ratios


# ── Fixed ZR simulation (same as backtest_zr_fixed.py) ───────────────────────
@njit
def sim_zr_fixed(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td, max_entry_spread):
    n = len(cl)
    cycle_pnl  = np.zeros(n, dtype=np.float64)
    cycle_legs = np.zeros(n, dtype=np.int32)
    nc = 0; n_trail = 0; n_zr = 0; n_skipped = 0
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
                            cycle_pnl[nc] = net_cl - tv2*sp; cycle_legs[nc]=nl
                            nc+=1; ex=True; break
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
                            cycle_pnl[nc] = net_cl - tv2*sp; cycle_legs[nc]=nl
                            nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(ut-lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc] = net_exit - tv*sp; cycle_legs[nc]=nl
                    nc+=1; n_zr+=1; ex=True; break
                if l <= lt <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(lt-lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc] = net_exit - tv*sp; cycle_legs[nc]=nl
                    nc+=1; n_zr+=1; ex=True; break
            i += 1
        d = -d
    return cycle_pnl[:nc], cycle_legs[:nc], nc, n_trail, n_zr, n_skipped


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, 1.4)
_o = _df0.open.values[:2000].astype(np.float64)
_h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64)
_c = _df0.close.values[:2000].astype(np.float64)
sim_zr_fixed(_o, _h, _l, _c, _sp0, 0.0001, PF, MAX_LEGS, 30., 15., 5., 1., 0.)
print("done.\n")

# ── Load EUR_JPY data ─────────────────────────────────────────────────────────
df_mid = pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
df_ba  = pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
df_mid['ts_key'] = df_mid['timestamp'].astype(str).str[:19]
df_ba['ts_key']  = df_ba['timestamp'].astype(str).str[:19]
merged = df_mid.merge(df_ba[['ts_key', 'bid_c', 'ask_c']], on='ts_key', how='inner')
merged = merged.sort_values('ts_key').reset_index(drop=True)

nb = len(merged)
is_end       = int(nb * (1 - OOS_FRAC))
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
gate_thr = float(np.percentile(sp_is, 90))   # p90 entry spread gate
sp_med   = float(np.median(sp_is))

print(f"{PAIR}  IS spread: med={sp_med:.2f}p  p90(gate)={gate_thr:.2f}p")
print(f"IS bars: {is_end}  OOS bars: {oos_len} ({oos_days:.1f} days)\n")


# ── Parameter grid ────────────────────────────────────────────────────────────
def _tgt_candidates(zw):
    """Round TGT values covering 0.4×ZW to 3.0×ZW, step 5p, min 10p."""
    cands = set()
    for frac in [0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        v = max(10.0, round(zw * frac / 5) * 5.0)
        cands.add(v)
    return sorted(cands)

ZWS = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

grid = [(zw, tgt) for zw in ZWS for tgt in _tgt_candidates(zw)
        if tgt > gate_thr]       # TGT must exceed spread gate to be viable

print(f"Grid: {len(grid)} (ZW, TGT) combinations\n")


# ── Progression table header ──────────────────────────────────────────────────
print("=" * 120)
print("VOLUME PROGRESSION (theoretical, sp=median, worst-case alternating crossings)")
print(f"{'ZW':>4} {'TGT':>5} {'ratio':>6} | {'L1':>4} {'L2':>5} {'L3':>6} {'L4':>7} {'L5':>8} "
      f"{'L6':>9} {'L7':>10} {'L8':>11} | {'Total':>8} | {'viable?':>8}")
print("-" * 120)
for zw, tgt in grid:
    vols, total, ratios = compute_progression(zw, tgt, sp_med, PF, MAX_LEGS)
    r = ratios[0] if ratios else 0
    # "Engineered viable" = leg-10 total ≤ 500 units (manageable with ~$500 account, 20 base_units)
    viable = "🟢 OK" if total <= 500 else ("🟡 ~OK" if total <= 2000 else "🔴 high")
    row = " ".join(f"{v:>{5+i}}" for i, v in enumerate(vols))
    print(f"{zw:>4.0f} {tgt:>5.0f} {r:>6.2f} | {row} | {total:>8} | {viable}")

print("=" * 120)
print()


# ── Backtest sweep ────────────────────────────────────────────────────────────
rng  = np.random.default_rng(42)
rows = []

sep2 = "─" * 105
print(sep2)
print(f"  {'ZW':>4} {'TGT':>5} {'ratio':>6} | "
      f"{'p/d':>8} {'minT':>7} | "
      f"{'IS-wf':>5} {'OOS-wf':>6} | "
      f"{'P5':>7} {'P(+)':>6} | "
      f"{'1leg%':>6} {'≤4leg%':>7} {'5+leg%':>7} | "
      f"{'L10vol':>7} | status")
print(sep2)

for zw, tgt in grid:
    vols, total_vol, ratios = compute_progression(zw, tgt, sp_med, PF, MAX_LEGS)
    r = ratios[0] if ratios else 0

    cyc, legs_arr, nc, nt, nz, ns = sim_zr_fixed(
        op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
        sp[is_end:], PIP, PF, MAX_LEGS, zw, tgt, TA, TD, gate_thr)

    if nc == 0:
        continue

    ppd  = cyc.sum() / oos_days
    minT = float(cyc.min())

    # leg-depth stats
    l1pct  = float(np.mean(legs_arr == 1)) * 100
    l4pct  = float(np.mean(legs_arr <= 4)) * 100
    l5pct  = float(np.mean(legs_arr >= 5)) * 100

    # IS walk-forward
    is_wf = 0
    for ch in range(IS_CHUNKS):
        s = ch * is_chunk_sz
        e2 = (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end
        c2, _, nc2, *_ = sim_zr_fixed(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                                       sp[s:e2], PIP, PF, MAX_LEGS, zw, tgt, TA, TD, gate_thr)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    # OOS walk-forward
    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s = is_end + ch * oos_chunk_sz
        e2 = is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb
        c2, _, nc2, *_ = sim_zr_fixed(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                                       sp[s:e2], PIP, PF, MAX_LEGS, zw, tgt, TA, TD, gate_thr)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    # Status
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and not math.isnan(p5) and p5 > 0 and prob > 0.95:
        if total_vol <= 500:
            status = "🟢 PASS+VIABLE"
        elif total_vol <= 2000:
            status = "🟡 PASS+~OK"
        else:
            status = "✅ PASS (high vol)"
    elif ppd < 0:
        status = "🔴 neg"
    else:
        status = f"IS={is_wf}/3 OOS={oos_wf}/3"

    print(f"  {zw:>4.0f} {tgt:>5.0f} {r:>6.2f} | "
          f"{ppd:>8.1f} {minT:>7.2f} | "
          f"{is_wf:>5} {oos_wf:>6} | "
          f"{p5:>7.1f} {prob:>6.3f} | "
          f"{l1pct:>6.1f} {l4pct:>7.1f} {l5pct:>7.2f} | "
          f"{total_vol:>7} | {status}")
    sys.stdout.flush()

    rows.append(dict(
        zw=zw, tgt=tgt, ratio=round(r, 2),
        ppd=round(ppd, 1), minT=round(minT, 2),
        is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5, 1) if not math.isnan(p5) else None,
        prob=round(prob, 3) if not math.isnan(prob) else None,
        l1pct=round(l1pct, 1), l4pct=round(l4pct, 1), l5pct=round(l5pct, 2),
        total_vol_leg10=total_vol,
    ))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}")

print("\n=== ENGINEERED VIABLE CONFIGURATIONS (total_vol ≤ 500, all WF gates) ===")
print(f"  {'ZW':>4} {'TGT':>5} {'ratio':>6} | {'p/d':>8} | "
      f"{'IS-wf':>5} {'OOS-wf':>6} | {'P5':>7} {'P(+)':>6} | "
      f"{'1leg%':>6} {'≤4leg%':>7} {'5+leg%':>7} | {'L10vol':>7}")
for r in sorted(rows, key=lambda x: x['ppd'], reverse=True):
    if (r.get('p5') and r['p5'] > 0 and r.get('prob', 0) > 0.95
            and r['is_wf'] == IS_CHUNKS and r['oos_wf'] == OOS_CHUNKS
            and r['total_vol_leg10'] <= 500):
        print(f"  {r['zw']:>4.0f} {r['tgt']:>5.0f} {r['ratio']:>6.2f} | "
              f"{r['ppd']:>8.1f} | "
              f"{r['is_wf']:>5} {r['oos_wf']:>6} | "
              f"{r['p5']:>7.1f} {r['prob']:>6.3f} | "
              f"{r['l1pct']:>6.1f} {r['l4pct']:>7.1f} {r['l5pct']:>7.2f} | "
              f"{r['total_vol_leg10']:>7}")
