"""
Validate Option B (break-even floor on trail stop) for EUR_JPY.

sim_zr_bev adds: trail stop price clamped to at least break-even (entry ± spread).
Runs full IS/OOS WF + MC bootstrap for both old and new sim side-by-side.
Reports: p/d, cycles/day, min-trail-pnl, IS-wf, OOS-wf, P5, P(+).
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'backtest_zr_bev_results.csv'

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


# ── Original sim (no be_floor) ────────────────────────────────────────────────
@njit
def sim_zr(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td, max_entry_spread):
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
                        ts = e + (peak_mfe-td)*pip
                        if l <= ts:
                            cycle_pnl[nc]=(ts-e)/pip-sp; nc+=1; n_trail+=1; ex=True
                    else:
                        ts = e - (peak_mfe-td)*pip
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip-sp; nc+=1; n_trail+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
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


# ── Option B sim (be_floor on trail) ─────────────────────────────────────────
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
                        be = e + sp * pip               # break-even floor
                        ts = e + (peak_mfe - td) * pip
                        if ts < be: ts = be
                        if l <= ts:
                            cycle_pnl[nc]=(ts-e)/pip-sp; nc+=1; n_trail+=1; ex=True
                    else:
                        be = e - sp * pip               # break-even floor
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip-sp; nc+=1; n_trail+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
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


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, 1.4)
_o = _df0.open.values[:2000].astype(np.float64)
_h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64)
_c = _df0.close.values[:2000].astype(np.float64)
sim_zr(_o, _h, _l, _c, _sp0, 0.0001, PF, MAX_LEGS, 30., 15., 5., 1., 0.)
sim_zr_bev(_o, _h, _l, _c, _sp0, 0.0001, PF, MAX_LEGS, 30., 15., 5., 1., 0.)
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

hdr = (f"  {'ta':>4} {'td':>4} | {'p/d':>8} {'cday':>6} {'minT':>7} | "
       f"{'IS-wf':>5} {'OOS-wf':>6} | {'P5':>7} {'P(+)':>6} | "
       f"{'p/d_b':>8} {'minT_b':>7} | {'P5_b':>7} {'P(+)_b':>7} | status")
print(f"{'─'*len(hdr)}")
print(f"  {'ta':>4} {'td':>4} | ── ORIGINAL ─────────────────────────── | ── BEV ─────────────────────────────── | status")
print(f"{'─'*len(hdr)}")

for ta in TRAIL_ACTS:
    for td in TRAIL_DISTS:
        if td >= ta: continue
        min_net = ta - td - gate_thr
        if min_net < 0: continue

        ta_f = float(ta); td_f = float(td)

        # ── Original ──────────────────────────────────────────────────────────
        cyc, nc, nt, nz, ns = sim_zr(
            op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
            sp[is_end:], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)
        if nc == 0: continue
        ppd_o  = cyc.sum() / oos_days
        minT_o = float(cyc[cyc < 999].min()) if nc > 0 else 0.0

        is_wf = sum(
            1 for ch in range(IS_CHUNKS)
            for s, e2 in [(ch * is_chunk_sz,
                           (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end)]
            if sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                      sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[1] > 0
            and sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                       sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[0].sum() > 0)

        oos_wf = sum(
            1 for ch in range(OOS_CHUNKS)
            for s, e2 in [(is_end + ch * oos_chunk_sz,
                           is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb)]
            if sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                      sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[1] > 0
            and sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                       sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[0].sum() > 0)

        both_wf = (is_wf == IS_CHUNKS) and (oos_wf == OOS_CHUNKS)
        p5_o = prob_o = float('nan')
        if both_wf and nc > 0:
            boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                             for _ in range(N_BOOT)])
            p5_o   = float(np.percentile(boot, 5))
            prob_o = float(np.mean(boot > 0))

        # ── BEV ───────────────────────────────────────────────────────────────
        cyc_b, nc_b, nt_b, nz_b, ns_b = sim_zr_bev(
            op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
            sp[is_end:], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)
        if nc_b == 0: continue
        ppd_b  = cyc_b.sum() / oos_days
        minT_b = float(cyc_b.min())

        is_wf_b = sum(
            1 for ch in range(IS_CHUNKS)
            for s, e2 in [(ch * is_chunk_sz,
                           (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end)]
            if sim_zr_bev(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                          sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[1] > 0
            and sim_zr_bev(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                           sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[0].sum() > 0)

        oos_wf_b = sum(
            1 for ch in range(OOS_CHUNKS)
            for s, e2 in [(is_end + ch * oos_chunk_sz,
                           is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb)]
            if sim_zr_bev(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                          sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[1] > 0
            and sim_zr_bev(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                           sp[s:e2], PIP, PF, MAX_LEGS, ZW, TGT, ta_f, td_f, gate_thr)[0].sum() > 0)

        both_wf_b = (is_wf_b == IS_CHUNKS) and (oos_wf_b == OOS_CHUNKS)
        p5_b = prob_b = float('nan')
        if both_wf_b and nc_b > 0:
            boot_b = np.array([rng.choice(cyc_b, nc_b, replace=True).sum() / oos_days
                               for _ in range(N_BOOT)])
            p5_b   = float(np.percentile(boot_b, 5))
            prob_b = float(np.mean(boot_b > 0))

        if both_wf_b and not math.isnan(p5_b) and p5_b > 0 and prob_b > 0.95:
            status = "✅ BEV PASS"
        elif both_wf and not math.isnan(p5_o) and p5_o > 0 and prob_o > 0.95:
            status = "🟡 orig pass / bev diff"
        elif ppd_b < 0:
            status = "❌ neg"
        else:
            status = f"❌ IS={is_wf_b}/3 OOS={oos_wf_b}/3"

        print(f"  {ta:>4} {td:>4} | "
              f"{ppd_o:>8.1f} {minT_o:>7.2f} | {is_wf:>5} {oos_wf:>6} | "
              f"{p5_o:>7.1f} {prob_o:>6.3f} | "
              f"{ppd_b:>8.1f} {minT_b:>7.2f} | "
              f"{p5_b:>7.1f} {prob_b:>6.3f} | {status}")
        sys.stdout.flush()

        rows.append(dict(
            ta=ta, td=td,
            ppd_orig=round(ppd_o, 1), min_trail_orig=round(minT_o, 2),
            is_wf=is_wf, oos_wf=oos_wf,
            p5_orig=round(p5_o, 1) if not math.isnan(p5_o) else None,
            prob_orig=round(prob_o, 3) if not math.isnan(prob_o) else None,
            ppd_bev=round(ppd_b, 1), min_trail_bev=round(minT_b, 2),
            is_wf_b=is_wf_b, oos_wf_b=oos_wf_b,
            p5_bev=round(p5_b, 1) if not math.isnan(p5_b) else None,
            prob_bev=round(prob_b, 3) if not math.isnan(prob_b) else None,
        ))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}")

print("\n=== BEV VALIDATED (IS-wf=3 OOS-wf=3 P5>0 P(+)>95%) ===")
print(f"{'ta':>4} {'td':>4} | {'ppd_orig':>9} {'ppd_bev':>9} | "
      f"{'minT_orig':>10} {'minT_bev':>9} | {'P5_bev':>8} {'P(+)_bev':>9}")
for r in rows:
    if r.get('p5_bev') and r['p5_bev'] > 0 and r.get('prob_bev', 0) > 0.95:
        print(f"  {r['ta']:>2} {r['td']:>2} | "
              f"{r['ppd_orig']:>9.1f} {r['ppd_bev']:>9.1f} | "
              f"{r['min_trail_orig']:>10.2f} {r['min_trail_bev']:>9.2f} | "
              f"{r['p5_bev']:>8.1f} {r['prob_bev']:>9.3f}")
