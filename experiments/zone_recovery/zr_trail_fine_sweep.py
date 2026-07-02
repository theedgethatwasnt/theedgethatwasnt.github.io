"""
Fine-grained ta/td sweep for CHF_JPY random-trail strategy.
Original sweep used td in {3,5,7,10} and missed td<3.
Catastrophic zones: ta=7, ta=20 (-18K p/d). Need to map full landscape.
Runs bootstrap MC inline on every wf=3 config.
"""
import math
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
SPREAD   = 1.4
MAX_LEGS = 10
PF       = 1.25
OOS_FRAC = 0.30
N_BOOT   = 2000
WF_CHUNKS = 3

PAIR     = "CHF_JPY"
PIP      = 0.01
ZW       = 40.0
TGT      = 20.0

# Fine grid around ta=5 sweet spot + exploratory smaller ta
TRAIL_ACTS  = [2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 30]
TRAIL_DISTS = [1, 2, 3, 5, 7, 10]   # td must be < ta


@njit
def sim_zr_trail_full(op, hi, lo, cl, pip, spread, pf, ml, zw, tgt, ta, td):
    """Returns (total_pnl, n_cycles, n_trail, n_zr, legs_acc, cycle_pnl_array)."""
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
    nc = 0; n_trail = 0; n_zr = 0; legs_acc = 0.0
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
                            pnl = (ts - e) / pip - spread
                            cycle_pnl[nc] = pnl; nc += 1; n_trail += 1
                            legs_acc += 1.0; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts:
                            pnl = (e - ts) / pip - spread
                            cycle_pnl[nc] = pnl; nc += 1; n_trail += 1
                            legs_acc += 1.0; ex = True
            if ex:
                break
            for pn in range(2):
                if ex:
                    break
                dh = (bull and pn == 0) or (not bull and pn == 1)
                if l <= ut <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    cycle_pnl[nc] = net - tv*spread; nc += 1
                    legs_acc += float(nl); n_zr += 1; ex = True; break
                if l <= lt <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    cycle_pnl[nc] = net - tv*spread; nc += 1
                    legs_acc += float(nl); n_zr += 1; ex = True; break
                if dh and h >= uz and lu != i:
                    lu = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c >= ut:
                            cycle_pnl[nc] = nt2; nc += 1
                            legs_acc += float(nl); n_zr += 1; ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            cycle_pnl[nc] = net - tv2*spread; nc += 1
                            legs_acc += float(nl); ex = True; break
                        lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1
                if not dh and l <= lz and ll != i:
                    ll = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c <= lt:
                            cycle_pnl[nc] = nt2; nc += 1
                            legs_acc += float(nl); n_zr += 1; ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            cycle_pnl[nc] = net - tv2*spread; nc += 1
                            legs_acc += float(nl); ex = True; break
                        lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1
            i += 1
        d = -d
    return cycle_pnl[:nc], nc, n_trail, n_zr, legs_acc


print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o = _df0.open.values[:2000].astype(np.float64); _h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64);  _c = _df0.close.values[:2000].astype(np.float64)
sim_zr_trail_full(_o,_h,_l,_c,0.0001,SPREAD,PF,MAX_LEGS,30.,15.,5.,3.)
print("done.\n")

df = pd.read_parquet(DATA_DIR/f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
op = df.open.values.astype(np.float64); hi = df.high.values.astype(np.float64)
lo = df.low.values.astype(np.float64); cl = df.close.values.astype(np.float64)
nb = len(cl)
is_end   = int(nb * (1 - OOS_FRAC))
chunk_sz = is_end // WF_CHUNKS

rng = np.random.default_rng(42)
rows = []

print(f"{'ta':>4} {'td':>4} | {'ppd':>9} {'c/day':>7} {'ppc':>7} {'trail%':>7} {'zr%':>6} {'avgl':>6} | "
      f"{'wf':>3} | {'P5':>8} {'P(+)':>6} | notes")
print("─" * 100)

for ta in TRAIL_ACTS:
    for td in TRAIL_DISTS:
        if td >= ta:
            continue

        # OOS eval
        oos_op=op[is_end:]; oos_hi=hi[is_end:]; oos_lo=lo[is_end:]; oos_cl=cl[is_end:]
        oos_days = len(oos_cl) / (24*12)
        cyc, nc, nt, nz, la = sim_zr_trail_full(
            oos_op, oos_hi, oos_lo, oos_cl, PIP, SPREAD, PF, MAX_LEGS, ZW, TGT, float(ta), float(td))
        if nc == 0:
            continue
        ppd  = cyc.sum() / oos_days
        ppc  = cyc.mean()
        cday = nc / oos_days
        trail_pct = 100 * nt / nc
        zr_pct    = 100 * nz / nc
        avgl      = la / nc

        # WF: IS split into 3 chunks, all must be positive
        wf = 0
        for ch in range(WF_CHUNKS):
            s = ch * chunk_sz; e2 = (ch+1)*chunk_sz if ch < WF_CHUNKS-1 else is_end
            c2, nc2, *_ = sim_zr_trail_full(
                op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2], PIP, SPREAD, PF, MAX_LEGS, ZW, TGT, float(ta), float(td))
            if nc2 > 0 and c2.sum() > 0:
                wf += 1
        wf_pass = wf == WF_CHUNKS

        # Bootstrap MC only if wf=3
        p5 = prob_pos = float('nan')
        if wf_pass:
            boot = np.array([rng.choice(cyc, size=nc, replace=True).sum() / oos_days
                             for _ in range(N_BOOT)])
            p5       = np.percentile(boot, 5)
            prob_pos = np.mean(boot > 0)

        # Flags
        note = ""
        if ppd < 0:      note = "❌ negative"
        elif not wf_pass: note = f"❌ wf={wf}/3"
        elif p5 < 0:      note = f"🟡 wf=3 P5={p5:.0f}"
        elif prob_pos < 0.95: note = f"🟡 P(+)={prob_pos:.3f}"
        else:             note = f"✅ P5={p5:.0f} P(+)={prob_pos:.3f}"

        star = " ◄ LIVE" if ta == 5 and td == 3 else ""
        print(f"{ta:>4} {td:>4} | {ppd:>9.1f} {cday:>7.1f} {ppc:>7.2f} {trail_pct:>7.1f} {zr_pct:>6.1f} {avgl:>6.3f} | "
              f"{wf:>3} | {p5:>8.1f} {prob_pos:>6.3f} | {note}{star}")

        rows.append(dict(ta=ta, td=td, ppd=round(ppd,1), cday=round(cday,1),
                         ppc=round(ppc,2), trail_pct=round(trail_pct,1),
                         zr_pct=round(zr_pct,1), avgl=round(avgl,3),
                         wf=wf, boot_p5=round(p5,1) if not math.isnan(p5) else None,
                         prob_pos=round(prob_pos,3) if not math.isnan(prob_pos) else None))

df_out = pd.DataFrame(rows)
out_path = Path(__file__).parent / 'zr_trail_fine_sweep_results.csv'
df_out.to_csv(out_path, index=False)
print(f"\nSaved {len(rows)} rows → {out_path}")

print("\n=== wf=3 CONFIGS RANKED BY ppd ===")
wf3 = df_out[df_out.wf == 3].sort_values('ppd', ascending=False)
print(wf3[['ta','td','ppd','cday','ppc','trail_pct','zr_pct','boot_p5','prob_pos']].to_string(index=False))
