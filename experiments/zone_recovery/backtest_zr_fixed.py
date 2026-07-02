"""
Backtest with all three ZR fixes applied:
  Fix 1 — Volume denominator: (tgt - spread) not tgt
  Fix 2 — Zone crossings processed BEFORE target exits in same bar
  Fix 3 — Both escape targets always active (no direction gate)
  + be_floor on trail stop (carried over from bev)

Compares sim_zr_bev (old, buggy) vs sim_zr_fixed (corrected) side-by-side.
Runs full IS/OOS WF + MC bootstrap validation on EUR_JPY.
Sweeps ta/td.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'backtest_zr_fixed_results.csv'

MAX_LEGS  = 10
PF        = 1.25
OOS_FRAC  = 0.30
IS_CHUNKS = 3
OOS_CHUNKS= 3
N_BOOT    = 2000

PAIR      = "EUR_JPY"
ZW        = 50.0
TGT       = 25.0
PIP       = 0.01

TRAIL_ACTS  = [2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 30]
TRAIL_DISTS = [1, 2, 3, 5, 7]


# ── Old sim (bev only, still has sequencing + denominator bugs) ───────────────
@njit
def sim_zr_bev(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td, max_entry_spread):
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
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
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
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
                            cycle_pnl[nc]=(ts-e)/pip-sp; nc+=1; n_trail+=1; ex=True
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip-sp; nc+=1; n_trail+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                # BUG: targets checked before zone crossings
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*sp; nc+=1; n_zr+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*sp; nc+=1; n_zr+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*sp
                    if nt2>=0:
                        if c>=ut: cycle_pnl[nc]=nt2; nc+=1; n_zr+=1; ex=True; break
                    else:
                        # BUG: denominator is tgt, not (tgt - spread)
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*sp
                    if nt2>=0:
                        if c<=lt: cycle_pnl[nc]=nt2; nc+=1; n_zr+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1
        d = -d
    return cycle_pnl[:nc], nc, n_trail, n_zr, n_skipped


# ── Fixed sim: zones-first, both targets always active, (tgt-sp) denominator ──
@njit
def sim_zr_fixed(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td, max_entry_spread):
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
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

            # ─ Trail stop (single leg only) with be_floor ─────────────────────
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
                            cycle_pnl[nc]=(ts-e)/pip - sp; nc+=1; n_trail+=1; ex=True
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip - sp; nc+=1; n_trail+=1; ex=True
            if ex: break

            # ─ FIX 2: zone crossings BEFORE targets ───────────────────────────
            # Process primary extreme first: bullish→hi first, bearish→lo first.
            # This ensures the hedge leg is placed before any target can fire
            # in the same bar (geometry: price can't reach UETGT without first
            # crossing UZB, same bar guarantees hedge fires first).
            for pass_idx in range(2):
                if ex: break
                # is_high=True when inspecting bar's high extreme
                is_high = (bull == (pass_idx == 0))

                # ── Zone crossings ───────────────────────────────────────────
                if is_high and h >= uz and lu != i:
                    lu = i
                    net_at_ut = 0.0; tv = 0.0
                    for k in range(nl):
                        net_at_ut += lv[k]*ld[k]*(ut - lp[k])/pip
                        tv += lv[k]
                    net_at_ut -= tv * sp
                    if net_at_ut < 0:
                        # FIX 1: each new unit earns (tgt - spread) net pips
                        net_per_unit = tgt - sp
                        if net_per_unit <= 1e-8: net_per_unit = 1e-8
                        v = max(1.0, math.ceil(-net_at_ut / net_per_unit * pf))
                        if nl >= ml:
                            net_cl = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net_cl += lv[k]*ld[k]*(c - lp[k])/pip; tv2 += lv[k]
                            cycle_pnl[nc] = net_cl - tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1

                if (not is_high) and l <= lz and ll != i:
                    ll = i
                    net_at_lt = 0.0; tv = 0.0
                    for k in range(nl):
                        net_at_lt += lv[k]*ld[k]*(lt - lp[k])/pip
                        tv += lv[k]
                    net_at_lt -= tv * sp
                    if net_at_lt < 0:
                        net_per_unit = tgt - sp
                        if net_per_unit <= 1e-8: net_per_unit = 1e-8
                        v = max(1.0, math.ceil(-net_at_lt / net_per_unit * pf))
                        if nl >= ml:
                            net_cl = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net_cl += lv[k]*ld[k]*(c - lp[k])/pip; tv2 += lv[k]
                            cycle_pnl[nc] = net_cl - tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1

                if ex: break

                # ── FIX 3: both targets always active (no direction gate) ────
                # Geometry guarantees the zone fires before target in the same
                # bar, so hedge is always in place before exit is triggered.
                if l <= ut <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(ut - lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc] = net_exit - tv*sp; nc+=1; n_zr+=1; ex=True; break
                if l <= lt <= h:
                    net_exit = 0.0; tv = 0.0
                    for k in range(nl):
                        net_exit += lv[k]*ld[k]*(lt - lp[k])/pip; tv += lv[k]
                    cycle_pnl[nc] = net_exit - tv*sp; nc+=1; n_zr+=1; ex=True; break

            i += 1
        d = -d
    return cycle_pnl[:nc], nc, n_trail, n_zr, n_skipped


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, 1.4)
_o = _df0.open.values[:2000].astype(np.float64)
_h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64)
_c = _df0.close.values[:2000].astype(np.float64)
sim_zr_bev(_o, _h, _l, _c, _sp0, 0.0001, PF, MAX_LEGS, 30., 15., 5., 1., 0.)
sim_zr_fixed(_o, _h, _l, _c, _sp0, 0.0001, PF, MAX_LEGS, 30., 15., 5., 1., 0.)
print("done.\n")

# ── Load EUR_JPY data ─────────────────────────────────────────────────────────
ba_path  = DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet'
mid_path = DATA_DIR_MID / f'{PAIR}_M5.parquet'

df_mid = pd.read_parquet(mid_path).sort_values('timestamp').reset_index(drop=True)
df_ba  = pd.read_parquet(ba_path).sort_values('timestamp').reset_index(drop=True)
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
gate_thr = float(np.percentile(sp_is, 90))
sp_med   = float(np.median(sp_is))

print(f"EUR_JPY  ZW={ZW} tgt={TGT}  IS spread: med={sp_med:.2f}p  p90(gate)={gate_thr:.2f}p")
print(f"IS bars: {is_end}  OOS bars: {oos_len} ({oos_days:.1f} days)")
print()

rng  = np.random.default_rng(42)
rows = []

sep = "─" * 115
print(sep)
print(f"  {'ta':>4} {'td':>4} | "
      f"{'── OLD (bev, buggy) ────────────────────────────':^47} | "
      f"{'── FIXED (zones-first, both-tgt, tgt-sp denom) ─':^47}")
print(f"  {'ta':>4} {'td':>4} | "
      f"{'p/d':>8} {'minT':>7} {'IS-wf':>5} {'OOS-wf':>6} {'P5':>7} {'P(+)':>6} | "
      f"{'p/d':>8} {'minT':>7} {'IS-wf':>5} {'OOS-wf':>6} {'P5':>7} {'P(+)':>6} | status")
print(sep)


def _wf_score(sim_fn, arr_op, arr_hi, arr_lo, arr_cl, arr_sp, s, e2):
    res = sim_fn(arr_op[s:e2], arr_hi[s:e2], arr_lo[s:e2], arr_cl[s:e2],
                 arr_sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT,
                 float(ta), float(td), gate_thr)
    cyc, nc_ = res[0], res[1]
    return nc_ > 0 and cyc.sum() > 0


for ta in TRAIL_ACTS:
    for td in TRAIL_DISTS:
        if td >= ta: continue
        min_net = ta - td - gate_thr
        if min_net < 0: continue

        ta_f = float(ta); td_f = float(td)

        # ── Old (bev, buggy) ──────────────────────────────────────────────────
        cyc_o, nc_o, nt_o, nz_o, ns_o = sim_zr_bev(
            op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
            sp[is_end:], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)
        ppd_o = cyc_o.sum() / oos_days if nc_o > 0 else 0.0
        minT_o = float(cyc_o.min()) if nc_o > 0 else 0.0

        is_wf_o = sum(
            1 for ch in range(IS_CHUNKS)
            for s, e2 in [(ch*is_chunk_sz, (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end)]
            if _wf_score(sim_zr_bev, op, hi, lo, cl, sp, s, e2))
        oos_wf_o = sum(
            1 for ch in range(OOS_CHUNKS)
            for s, e2 in [(is_end + ch*oos_chunk_sz,
                           is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb)]
            if _wf_score(sim_zr_bev, op, hi, lo, cl, sp, s, e2))

        p5_o = prob_o = float('nan')
        if is_wf_o == IS_CHUNKS and oos_wf_o == OOS_CHUNKS and nc_o > 0:
            boot = np.array([rng.choice(cyc_o, nc_o, replace=True).sum() / oos_days
                             for _ in range(N_BOOT)])
            p5_o   = float(np.percentile(boot, 5))
            prob_o = float(np.mean(boot > 0))

        # ── Fixed ─────────────────────────────────────────────────────────────
        cyc_f, nc_f, nt_f, nz_f, ns_f = sim_zr_fixed(
            op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
            sp[is_end:], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)
        ppd_f = cyc_f.sum() / oos_days if nc_f > 0 else 0.0
        minT_f = float(cyc_f.min()) if nc_f > 0 else 0.0

        is_wf_f = sum(
            1 for ch in range(IS_CHUNKS)
            for s, e2 in [(ch*is_chunk_sz, (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end)]
            if _wf_score(sim_zr_fixed, op, hi, lo, cl, sp, s, e2))
        oos_wf_f = sum(
            1 for ch in range(OOS_CHUNKS)
            for s, e2 in [(is_end + ch*oos_chunk_sz,
                           is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb)]
            if _wf_score(sim_zr_fixed, op, hi, lo, cl, sp, s, e2))

        if nc_f == 0:
            print(f"  {ta:>4} {td:>4} | {'(no cycles)':>72}")
            sys.stdout.flush()
            continue

        p5_f = prob_f = float('nan')
        if is_wf_f == IS_CHUNKS and oos_wf_f == OOS_CHUNKS:
            boot_f = np.array([rng.choice(cyc_f, nc_f, replace=True).sum() / oos_days
                               for _ in range(N_BOOT)])
            p5_f   = float(np.percentile(boot_f, 5))
            prob_f = float(np.mean(boot_f > 0))

        if is_wf_f == IS_CHUNKS and oos_wf_f == OOS_CHUNKS and not math.isnan(p5_f) and p5_f > 0 and prob_f > 0.95:
            status = "🟢 FIXED PASS"
        elif is_wf_o == IS_CHUNKS and oos_wf_o == OOS_CHUNKS and not math.isnan(p5_o) and p5_o > 0 and prob_o > 0.95:
            status = "🟡 OLD pass / fixed diff"
        elif ppd_f < 0:
            status = "🔴 neg"
        else:
            status = f"❌ IS={is_wf_f}/3 OOS={oos_wf_f}/3"

        print(f"  {ta:>4} {td:>4} | "
              f"{ppd_o:>8.1f} {minT_o:>7.2f} {is_wf_o:>5} {oos_wf_o:>6} "
              f"{p5_o:>7.1f} {prob_o:>6.3f} | "
              f"{ppd_f:>8.1f} {minT_f:>7.2f} {is_wf_f:>5} {oos_wf_f:>6} "
              f"{p5_f:>7.1f} {prob_f:>6.3f} | {status}")
        sys.stdout.flush()

        rows.append(dict(
            ta=ta, td=td,
            ppd_old=round(ppd_o, 1), minT_old=round(minT_o, 2),
            is_wf_old=is_wf_o, oos_wf_old=oos_wf_o,
            p5_old=round(p5_o, 1) if not math.isnan(p5_o) else None,
            prob_old=round(prob_o, 3) if not math.isnan(prob_o) else None,
            ppd_fixed=round(ppd_f, 1), minT_fixed=round(minT_f, 2),
            is_wf_fixed=is_wf_f, oos_wf_fixed=oos_wf_f,
            p5_fixed=round(p5_f, 1) if not math.isnan(p5_f) else None,
            prob_fixed=round(prob_f, 3) if not math.isnan(prob_f) else None,
        ))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}")

print("\n=== FIXED VALIDATED (IS-wf=3 OOS-wf=3 P5>0 P(+)>95%) ===")
print(f"  {'ta':>4} {'td':>4} | {'ppd_old':>9} {'ppd_fixed':>10} | "
      f"{'minT_old':>9} {'minT_fixed':>10} | {'P5_fixed':>9} {'P(+)_fixed':>11}")
found = False
for r in rows:
    if r.get('p5_fixed') and r['p5_fixed'] > 0 and r.get('prob_fixed', 0) > 0.95:
        print(f"  {r['ta']:>4} {r['td']:>4} | "
              f"{r['ppd_old']:>9.1f} {r['ppd_fixed']:>10.1f} | "
              f"{r['minT_old']:>9.2f} {r['minT_fixed']:>10.2f} | "
              f"{r['p5_fixed']:>9.1f} {r['prob_fixed']:>11.3f}")
        found = True
if not found:
    print("  (none passed all gates)")

print("\n=== DEPLOYED CONFIG: ta=5 td=1 ===")
dep = [r for r in rows if r['ta'] == 5 and r['td'] == 1]
if dep:
    r = dep[0]
    print(f"  old  → p/d={r['ppd_old']:+.1f}  IS={r['is_wf_old']}/3  OOS={r['oos_wf_old']}/3  "
          f"P5={r['p5_old']}  P(+)={r['prob_old']}")
    print(f"  fixed→ p/d={r['ppd_fixed']:+.1f}  IS={r['is_wf_fixed']}/3  OOS={r['oos_wf_fixed']}/3  "
          f"P5={r['p5_fixed']}  P(+)={r['prob_fixed']}")
else:
    print("  ta=5 td=1 did not appear in sweep (check min_net gate)")
