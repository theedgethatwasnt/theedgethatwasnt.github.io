"""
ZR ta/td sweep with DOUBLE walk-forward gate:
  - IS  WF: IS split into 3 chunks → all must be profitable   (same as sweep)
  - OOS WF: OOS split into 3 sub-windows → all must be profitable  (NEW)
  - MC bootstrap: P5 > 0 AND P(+) > 95%  on full OOS           (same)

Only configs that survive IS-WF AND OOS-WF AND MC are reported.
This is the strictest possible out-of-sample validation.

Uses real OANDA bid/ask spreads (BA parquets), IS-p90 gate per pair.
All 11 pairs with BA data.

Output: zr_oos_wf_sweep_results.csv
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'zr_oos_wf_sweep_results.csv'

MAX_LEGS    = 10
PF          = 1.25
OOS_FRAC    = 0.30
IS_CHUNKS   = 3
OOS_CHUNKS  = 3
N_BOOT      = 2000

PAIR_CFG = {
    "AUD_USD": dict(zw=30.0, tgt=15.0, pip=0.0001),
    "NZD_USD": dict(zw=25.0, tgt=12.5, pip=0.0001),
    "EUR_GBP": dict(zw=40.0, tgt=20.0, pip=0.0001),
    "EUR_USD": dict(zw=30.0, tgt=15.0, pip=0.0001),
    "USD_JPY": dict(zw=40.0, tgt=20.0, pip=0.01),
    "GBP_USD": dict(zw=30.0, tgt=15.0, pip=0.0001),
    "AUD_JPY": dict(zw=50.0, tgt=25.0, pip=0.01),
    "CAD_JPY": dict(zw=50.0, tgt=12.5, pip=0.01),
    "NZD_JPY": dict(zw=40.0, tgt=20.0, pip=0.01),
    "EUR_JPY": dict(zw=50.0, tgt=25.0, pip=0.01),
    "CHF_JPY": dict(zw=40.0, tgt=20.0, pip=0.01),
}

TRAIL_ACTS  = [2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 30]
TRAIL_DISTS = [1, 2, 3, 5, 7]


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


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, 1.4)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
sim_zr(_o,_h,_l,_c,_sp0,0.0001,PF,MAX_LEGS,30.,15.,5.,1.,0.)
print("done.\n")

rng  = np.random.default_rng(42)
rows = []

for pair, cfg in PAIR_CFG.items():
    zw=cfg['zw']; tgt=cfg['tgt']; pip=cfg['pip']

    ba_path  = DATA_DIR_BA  / f'{pair}_M5_BA.parquet'
    mid_path = DATA_DIR_MID / f'{pair}_M5.parquet'
    if not ba_path.exists() or not mid_path.exists():
        print(f"[{pair}] missing data — skip"); continue

    df_mid = pd.read_parquet(mid_path).sort_values('timestamp').reset_index(drop=True)
    df_ba  = pd.read_parquet(ba_path).sort_values('timestamp').reset_index(drop=True)
    df_mid['ts_key'] = df_mid['timestamp'].astype(str).str[:19]
    df_ba['ts_key']  = df_ba['timestamp'].astype(str).str[:19]
    merged = df_mid.merge(df_ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner')
    merged = merged.sort_values('ts_key').reset_index(drop=True)

    nb = len(merged)
    if nb < 5000:
        print(f"[{pair}] only {nb} aligned bars — skip"); continue

    is_end   = int(nb * (1 - OOS_FRAC))
    is_chunk_sz  = is_end // IS_CHUNKS
    oos_len  = nb - is_end
    oos_chunk_sz = oos_len // OOS_CHUNKS
    oos_days = oos_len / (24 * 12)

    op = merged.open.values.astype(np.float64)
    hi = merged.high.values.astype(np.float64)
    lo = merged.low.values.astype(np.float64)
    cl = merged.close.values.astype(np.float64)
    sp = ((merged.ask_c - merged.bid_c) / pip).clip(lower=0.3).values.astype(np.float64)

    sp_is    = sp[:is_end]
    gate_thr = float(np.percentile(sp_is, 90))
    sp_med_r = float(np.median(sp_is))

    print(f"\n{'═'*72}")
    print(f"{pair}  ZW={zw} tgt={tgt}  IS spread: med={sp_med_r:.2f}p  p90(gate)={gate_thr:.2f}p")
    print(f"  {'ta':>4} {'td':>4} | {'p/d':>9} {'c/day':>7} {'ppc':>7} | "
          f"{'IS-wf':>6} {'OOS-wf':>7} | {'P5':>8} {'P(+)':>6} | status")
    print(f"  {'─'*90}")

    best_ppd = -1e9
    for ta in TRAIL_ACTS:
        for td in TRAIL_DISTS:
            if td >= ta: continue
            min_net = ta - td - gate_thr
            if min_net < 0: continue

            ta_f = float(ta); td_f = float(td)

            # Full OOS
            cyc, nc, nt, nz, nskip = sim_zr(
                op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:],
                sp[is_end:], pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, gate_thr)
            if nc == 0: continue
            ppd      = cyc.sum() / oos_days
            ppc      = cyc.mean()
            cday     = nc / oos_days

            # IS walk-forward (3 chunks)
            is_wf = 0
            for ch in range(IS_CHUNKS):
                s = ch * is_chunk_sz
                e2 = (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end
                c2, nc2, *_ = sim_zr(
                    op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                    sp[s:e2], pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, gate_thr)
                if nc2 > 0 and c2.sum() > 0: is_wf += 1

            # OOS walk-forward (3 sub-windows)
            oos_wf = 0
            for ch in range(OOS_CHUNKS):
                s = is_end + ch * oos_chunk_sz
                e2 = is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb
                c2, nc2, *_ = sim_zr(
                    op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                    sp[s:e2], pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, gate_thr)
                if nc2 > 0 and c2.sum() > 0: oos_wf += 1

            both_wf = (is_wf == IS_CHUNKS) and (oos_wf == OOS_CHUNKS)

            p5 = prob_pos = float('nan')
            if both_wf:
                boot = np.array([
                    rng.choice(cyc, size=nc, replace=True).sum() / oos_days
                    for _ in range(N_BOOT)])
                p5 = np.percentile(boot, 5)
                prob_pos = np.mean(boot > 0)

            if both_wf and not math.isnan(p5) and p5 > 0 and prob_pos > 0.95:
                status = f"✅ P5={p5:.0f} P(+)={prob_pos:.3f}"
                if ppd > best_ppd: best_ppd = ppd
            elif is_wf == IS_CHUNKS and oos_wf < OOS_CHUNKS:
                status = f"🟡 IS-wf=3 OOS-wf={oos_wf}/3"
            elif both_wf and not math.isnan(p5):
                status = f"🟡 both-wf=3 P5={p5:.0f} P(+)={prob_pos:.3f}"
            elif ppd < 0:
                status = "❌ neg"
            else:
                status = f"❌ IS={is_wf}/3 OOS={oos_wf}/3"

            star = " ◄ BEST" if both_wf and ppd == best_ppd and ppd > 0 else ""
            print(f"  {ta:>4} {td:>4} | {ppd:>9.1f} {cday:>7.1f} {ppc:>7.2f} | "
                  f"{is_wf:>6} {oos_wf:>7} | {p5:>8.1f} {prob_pos:>6.3f} | {status}{star}")
            sys.stdout.flush()

            rows.append(dict(
                pair=pair, ta=ta, td=td, zw=zw, tgt=tgt,
                sp_med=round(sp_med_r, 2), gate=round(gate_thr, 2),
                ppd=round(ppd, 1), cday=round(cday, 1), ppc=round(ppc, 2),
                is_wf=is_wf, oos_wf=oos_wf,
                boot_p5=round(p5, 1) if not math.isnan(p5) else None,
                prob_pos=round(prob_pos, 3) if not math.isnan(prob_pos) else None,
            ))

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved {len(rows)} rows → {OUT_PATH}")

print("\n=== BEST VALIDATED CONFIG PER PAIR (IS-wf=3 + OOS-wf=3 + P5>0 + P(+)>95%) ===")
print(f"{'pair':<10} {'ta':>4} {'td':>4} {'ppd':>9} {'c/day':>7} {'P5':>8} {'P(+)':>6} "
      f"{'sp_med':>7} {'gate':>6}")
for pair in PAIR_CFG:
    sub = df_out[(df_out.pair==pair) & (df_out.is_wf==IS_CHUNKS) &
                 (df_out.oos_wf==OOS_CHUNKS) &
                 (df_out.boot_p5.notna()) & (df_out.boot_p5>0) &
                 (df_out.prob_pos>0.95)]
    if sub.empty:
        print(f"{pair:<10}  — no validated config"); continue
    best = sub.loc[sub.ppd.idxmax()]
    print(f"{pair:<10} {int(best.ta):>4} {int(best.td):>4} {best.ppd:>9.1f} "
          f"{best.cday:>7.1f} {best.boot_p5:>8.1f} {best.prob_pos:>6.3f} "
          f"{best.sp_med:>7.2f}p {best.gate:>5.2f}p")

# Also show IS-wf=3 passed but OOS-wf<3 (to understand what the OOS gate filters out)
print("\n=== CONFIGS THAT PASS IS-WF=3 BUT FAIL OOS-WF (for comparison) ===")
print(f"{'pair':<10} {'ta':>4} {'td':>4} {'ppd':>9} {'IS-wf':>6} {'OOS-wf':>7}")
for pair in PAIR_CFG:
    sub = df_out[(df_out.pair==pair) & (df_out.is_wf==IS_CHUNKS) &
                 (df_out.oos_wf<OOS_CHUNKS) & (df_out.ppd>0)]
    if sub.empty: continue
    best = sub.loc[sub.ppd.idxmax()]
    print(f"{pair:<10} {int(best.ta):>4} {int(best.td):>4} {best.ppd:>9.1f} "
          f"{int(best.is_wf):>6} {int(best.oos_wf):>7}")
