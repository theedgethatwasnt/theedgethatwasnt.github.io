"""
Permutation test + Bootstrap MC for top P&F-timed ZR configs.

Permutation test (2000 shuffles):
  H0: P&F reversal timing provides no edge over random scheduling at the
      same entry frequency.
  Method: shuffle P&F reversal bar positions uniformly within OOS window,
          re-run ZR sim. p = fraction(shuffled_ppd >= observed_ppd).

Bootstrap MC (2000 samples):
  Resample per-cycle P&L with replacement. Reports p5/p25/p50/p75/p95,
  Sharpe, and P(p/d > 0) across bootstrap draws.

Configs tested (WF=3 from zr_pnf_sweep_results.csv):
  Alternating: USD_JPY(5,2), NZD_JPY(5,4), CHF_JPY(10,3), NZD_JPY(10,3),
               EUR_JPY(5,4), AUD_JPY(5,2), AUD_JPY(5,3)
  Column:      AUD_JPY(15,2), AUD_JPY(10,3)
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
OUT_PATH  = Path('/path/to/projects/fx-core/research/experiments/zone_recovery/zr_pnf_permtest_results.csv')

SPREAD = 1.4; MAX_LEGS = 10; PF = 1.25
N_PERM  = 2000
N_BOOT  = 2000
OOS_FRAC = 0.30

PIP_MAP = {
    "AUD_JPY": 0.01, "CHF_JPY": 0.01, "EUR_JPY": 0.01,
    "NZD_JPY": 0.01, "USD_JPY": 0.01, "NZD_USD": 0.0001,
}

PAIR_CFG = {
    "CHF_JPY": dict(zw=40.0, tgt=20.0, ta=5.0,  td=3.0),
    "USD_JPY": dict(zw=40.0, tgt=20.0, ta=10.0, td=5.0),
    "EUR_JPY": dict(zw=50.0, tgt=25.0, ta=5.0,  td=3.0),
    "NZD_JPY": dict(zw=40.0, tgt=20.0, ta=5.0,  td=3.0),
    "AUD_JPY": dict(zw=50.0, tgt=25.0, ta=5.0,  td=3.0),
    "NZD_USD": dict(zw=25.0, tgt=12.5, ta=5.0,  td=3.0),
}

CONFIGS = [
    # (pair, box_pips, reversal, dir_mode, label)
    ("USD_JPY", 5,  2, 0, "alt"),
    ("NZD_JPY", 5,  4, 0, "alt"),
    ("CHF_JPY", 10, 3, 0, "alt"),
    ("NZD_JPY", 10, 3, 0, "alt"),
    ("EUR_JPY", 5,  4, 0, "alt"),
    ("AUD_JPY", 5,  2, 0, "alt"),
    ("AUD_JPY", 5,  3, 0, "alt"),
    ("AUD_JPY", 15, 2, 1, "col"),
    ("AUD_JPY", 10, 3, 1, "col"),
]


# ─── P&F reversal builder ────────────────────────────────────────────────────

@njit
def build_pnf_reversals(hi, lo, box_size, rev_n):
    n = len(hi)
    rev_bars = np.zeros(n, dtype=np.int64)
    rev_dirs = np.zeros(n, dtype=np.int8)
    n_rev = 0
    if n < 2:
        return rev_bars[:0], rev_dirs[:0]
    col_dir = np.int8(1)
    col_extreme = (hi[0] + lo[0]) * 0.5
    for i in range(1, n):
        h = hi[i]; l = lo[i]
        if col_dir == 1:
            if h >= col_extreme + box_size:
                col_extreme += math.floor((h - col_extreme) / box_size) * box_size
            elif l <= col_extreme - rev_n * box_size:
                col_dir = np.int8(-1)
                col_extreme -= box_size
                rev_bars[n_rev] = i; rev_dirs[n_rev] = np.int8(-1); n_rev += 1
        else:
            if l <= col_extreme - box_size:
                col_extreme -= math.floor((col_extreme - l) / box_size) * box_size
            elif h >= col_extreme + rev_n * box_size:
                col_dir = np.int8(1)
                col_extreme += box_size
                rev_bars[n_rev] = i; rev_dirs[n_rev] = np.int8(1); n_rev += 1
    return rev_bars[:n_rev], rev_dirs[:n_rev]


# ─── ZR sim: returns per-cycle P&L array ────────────────────────────────────

@njit
def sim_zr_pnf_cycles(op, hi, lo, cl, rev_bars, rev_dirs, dir_mode,
                       pip, spread, pf, ml, zw, tgt, ta, td):
    """
    Same as sim_zr_pnf but records per-cycle P&L for bootstrap.
    Returns (cycle_pnl array, n_cycles).
    """
    n = len(cl); n_rev = len(rev_bars)
    cycle_pnl = np.zeros(n_rev, dtype=np.float64)
    nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    d = np.int8(1); ri = 0

    while ri < n_rev:
        entry_bar = rev_bars[ri]; col_d = rev_dirs[ri]; ri += 1
        if entry_bar >= n: break

        direction = float(d) if dir_mode == 0 else float(col_d)
        e = cl[entry_bar]
        if direction == 1.0:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip

        lv[0]=1.0; ld[0]=direction; lp[0]=e
        nl=1; lu=-1; ll=-1; ex=False
        peak_mfe=0.0; trail_on=False
        cycle_gross = 0.0; cycle_cost = 0.0
        i = entry_bar + 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]

            if nl == 1:
                cur_mfe = (h-e)/pip if direction==1.0 else (e-l)/pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if direction == 1.0:
                        ts = e + (peak_mfe - td) * pip
                        if l <= ts:
                            cycle_pnl[nc] = (ts-e)/pip - spread
                            nc += 1; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts:
                            cycle_pnl[nc] = (e-ts)/pip - spread
                            nc += 1; ex = True
            if ex: break

            for pn in range(2):
                if ex: break
                dh = (bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*spread; nc+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*spread; nc+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: cycle_pnl[nc]=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: cycle_pnl[nc]=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1

        if dir_mode == 0: d = np.int8(-d)
        while ri < n_rev and rev_bars[ri] <= i: ri += 1

    return cycle_pnl[:nc], nc


# ─── ZR sim: lightweight (no cycle storage), for permutation runs ────────────

@njit
def sim_zr_entry_bars(op, hi, lo, cl, rev_bars, pip, spread, pf, ml, zw, tgt, ta, td):
    """
    Alternating ZR sim from arbitrary entry bar array.
    rev_bars must be sorted. Column direction ignored (alternating only).
    """
    n=len(cl); n_rev=len(rev_bars)
    total=0.0; lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    d=np.int8(1); ri=0

    while ri < n_rev:
        entry_bar=rev_bars[ri]; ri+=1
        if entry_bar >= n: break
        direction=float(d)
        e=cl[entry_bar]
        if direction==1.0:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=direction; lp[0]=e
        nl=1; lu=-1; ll=-1; ex=False
        peak_mfe=0.0; trail_on=False
        i=entry_bar+1

        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl==1:
                cur_mfe=(h-e)/pip if direction==1.0 else (e-l)/pip
                if cur_mfe>peak_mfe: peak_mfe=cur_mfe
                if peak_mfe>=ta: trail_on=True
                if trail_on:
                    if direction==1.0:
                        ts=e+(peak_mfe-td)*pip
                        if l<=ts: total+=(ts-e)/pip-spread; ex=True
                    else:
                        ts=e-(peak_mfe-td)*pip
                        if h>=ts: total+=(e-ts)/pip-spread; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        d=np.int8(-d)
        while ri<n_rev and rev_bars[ri]<=i: ri+=1

    return total


# ─── Warm-up compile ─────────────────────────────────────────────────────────

print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
_rb,_rd = build_pnf_reversals(_h,_l,0.0010,3)
_dummy_dirs = np.ones(max(len(_rb),1), dtype=np.int8)
sim_zr_pnf_cycles(_o,_h,_l,_c,_rb,_rd,0,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
sim_zr_entry_bars(_o,_h,_l,_c,_rb,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
print("done.\n")

rng = np.random.default_rng(42)
rows = []

for cfg in CONFIGS:
    pair, box_pips, rev_n, dir_mode, dir_label = cfg
    pip  = PIP_MAP[pair]
    pcfg = PAIR_CFG[pair]
    zw=pcfg['zw']; tgt=pcfg['tgt']; ta=pcfg['ta']; td=pcfg['td']
    box_size = box_pips * pip

    df = pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(np.float64); hi=df.high.values.astype(np.float64)
    lo=df.low.values.astype(np.float64);  cl=df.close.values.astype(np.float64)
    nb = len(cl)
    is_end  = int(nb*(1-OOS_FRAC))
    oos_op=op[is_end:]; oos_hi=hi[is_end:]; oos_lo=lo[is_end:]; oos_cl=cl[is_end:]
    oos_bars = len(oos_cl)
    oos_td   = oos_bars / (24*12)

    oos_rb, oos_rd = build_pnf_reversals(oos_hi, oos_lo, box_size, rev_n)
    n_rev = len(oos_rb)

    # Observed OOS p/d
    cyc_pnl, nc = sim_zr_pnf_cycles(
        oos_op, oos_hi, oos_lo, oos_cl, oos_rb, oos_rd, dir_mode,
        pip, SPREAD, PF, MAX_LEGS, zw, tgt, ta, td)
    obs_ppd = cyc_pnl.sum() / oos_td

    tag = f"{pair} box={box_pips} rev={rev_n} {dir_label}"
    print(f"\n{'─'*60}")
    print(f"{tag}  |  obs={obs_ppd:.1f} p/d  n_cycles={nc}  n_rev={n_rev}")

    # ── Permutation test ─────────────────────────────────────────────────────
    print(f"  Permutation test ({N_PERM} shuffles)...", end=' ', flush=True)
    all_bar_positions = np.arange(oos_bars, dtype=np.int64)
    perm_ppd = np.empty(N_PERM)

    for k in range(N_PERM):
        shuffled = rng.choice(all_bar_positions, size=n_rev, replace=False)
        shuffled.sort()
        tot = sim_zr_entry_bars(
            oos_op, oos_hi, oos_lo, oos_cl, shuffled,
            pip, SPREAD, PF, MAX_LEGS, zw, tgt, ta, td)
        perm_ppd[k] = tot / oos_td

    p_val  = np.mean(perm_ppd >= obs_ppd)
    perm_median = np.median(perm_ppd)
    perm_p95    = np.percentile(perm_ppd, 95)
    print(f"done.")
    print(f"  Null median={perm_median:.1f}  p95={perm_p95:.1f}  p-value={p_val:.4f}  "
          f"({'PASS' if p_val < 0.05 else 'FAIL'} @ α=0.05)")

    # ── Bootstrap MC ─────────────────────────────────────────────────────────
    print(f"  Bootstrap MC ({N_BOOT} samples)...", end=' ', flush=True)
    boot_ppd = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sample = rng.choice(cyc_pnl, size=nc, replace=True)
        boot_ppd[k] = sample.sum() / oos_td

    p5,p25,p50,p75,p95 = np.percentile(boot_ppd, [5,25,50,75,95])
    sharpe = boot_ppd.mean() / (boot_ppd.std() + 1e-9)
    prob_pos = np.mean(boot_ppd > 0)
    print(f"done.")
    print(f"  Bootstrap p/d:  p5={p5:.0f}  p25={p25:.0f}  median={p50:.0f}  p75={p75:.0f}  p95={p95:.0f}")
    print(f"  Sharpe={sharpe:.2f}  P(p/d>0)={prob_pos:.3f}")

    gate_perm = p_val < 0.05
    gate_p5   = p5 > 0
    gate_prob = prob_pos > 0.95
    gates_passed = sum([gate_perm, gate_p5, gate_prob])
    gate_str = (f"perm={'✅' if gate_perm else '❌'}  "
                f"p5>0={'✅' if gate_p5 else '❌'}  "
                f"P(+)>95%={'✅' if gate_prob else '❌'}  "
                f"→ {gates_passed}/3 gates")
    print(f"  Gates: {gate_str}")

    rows.append(dict(
        pair=pair, box_pips=box_pips, reversal=rev_n, direction=dir_label,
        obs_ppd=round(obs_ppd,1), n_cycles=nc,
        perm_null_median=round(perm_median,1), perm_null_p95=round(perm_p95,1),
        p_value=round(p_val,4),
        boot_p5=round(p5,0), boot_p25=round(p25,0),
        boot_median=round(p50,0), boot_p75=round(p75,0), boot_p95=round(p95,0),
        sharpe=round(sharpe,2), prob_pos=round(prob_pos,3),
        gate_perm=int(gate_perm), gate_p5=int(gate_p5), gate_prob=int(gate_prob),
        gates=gates_passed,
    ))
    sys.stdout.flush()

# ─── Summary ─────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(rows)
df_res.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved {len(df_res)} rows → {OUT_PATH}")

print("\n=== SUMMARY ===")
print(f"{'Config':35} {'obs_ppd':>8} {'p-val':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'P(+)':>6} {'gates':>5}")
print("─"*90)
for _, r in df_res.iterrows():
    tag = f"{r.pair} b={r.box_pips} rv={r.reversal} {r.direction}"
    pf_flag = "✅" if r.p_value < 0.05 else "❌"
    print(f"{tag:35} {r.obs_ppd:>8.0f} {r.p_value:>7.4f}{pf_flag} "
          f"{r.boot_p5:>7.0f} {r.boot_median:>7.0f} {r.boot_p95:>7.0f} "
          f"{r.prob_pos:>6.3f} {r.gates:>5}/3")
