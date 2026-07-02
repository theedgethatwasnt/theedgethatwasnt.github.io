"""
ZR Partial Hedge — Fixed-Target Baseline (GBP_USD)
====================================================

Baseline: backtest_zr_tight.py validated design (+7,319 p/d, IS=3/3, OOS=3/3).
Exit mechanism: 1-leg ta/td trailing stop + PSAR at ut/lt target cross.
This is the design that works; trail-lock baseline was broken on current data.

Partial hedge addition:
  Level 1 (at f_partial × ZW from entry): equal-volume delta-neutral hedge.
             Freezes further loss accumulation between entry and zone boundary.
  Level 2 (at ZW): PF-sized recovery leg, sized smaller because partial hedge
             reduces net-at-target by its contribution.

Three unwind modes for Level 1 (only while nl==1):
  A (mode=0): close when price crosses back through ph_level  (near-zero cost)
  B (mode=1): close when net aggregate P&L >= 0 at bar close  (usually never)
  C (mode=2): close when price crosses back through entry price  (-f*ZW cost)

Once nl >= 2, partial hedge is frozen (no new opens, existing stays).
Partial hedge P&L is always included in PSAR exits and emergency closes.

Fixed params (validated GBP_USD best config from backtest_zr_tight.py):
  ZW=30, TGT=21, PF=1.25, body=0.5
  ta=6, td=1, af0=0.01, af_step=0.01, af_max=0.20

Sweep: f_partial ∈ {0.0, 0.25, 0.33, 0.50, 0.67, 0.75}
       × unwind_mode ∈ {A, B, C}
       f_partial=0.0 = baseline (no partial hedge)

Gates: IS=3/3 WF chunks, OOS=3/3 WF chunks, P5>0, P(+)>0.95
"""

import math, time
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR     = "GBP_USD"
PIP      = 0.0001
ZW       = 30.0
TGT      = 21.0
PF       = 1.25
BODY     = 0.5
TA       = 6.0    # 1-leg trail activation (pips MFE)
TD       = 1.0    # 1-leg trail distance from peak
AF0      = 0.01   # PSAR initial AF
AFST     = 0.01   # PSAR AF step
AFMX     = 0.20   # PSAR max AF
ML       = 10     # max legs
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT   = 2000

F_PARTIALS   = [0.0, 0.25, 0.33, 0.50, 0.67, 0.75]
UNWIND_MODES = [0, 1, 2]   # 0=A, 1=B, 2=C
MODE_LABELS  = {0: 'A', 1: 'B', 2: 'C'}

OUT_CSV = Path(__file__).parent / "zr_ph_tight_results.csv"


# ── Numba kernel ──────────────────────────────────────────────────────────────

@njit
def sim_zr_ph_tight(op, hi, lo, cl, sp_arr,
                    pip, pf, zw, tgt, body_thresh,
                    ta, td, af0, af_step, af_max,
                    ml, f_partial, unwind_mode):
    """
    ZR fixed-target (PSAR at ut/lt) + optional partial hedge at f_partial*ZW.

    Exit hierarchy:
      1. PSAR (if active): fires when PSAR level is crossed
      2. 1-leg ta/td trailing stop (nl==1 only)
      3. Intra-bar: mode A/C unwind, hedge open, zone crossings, target → PSAR

    f_partial=0 → no partial hedge (baseline matches backtest_zr_tight.py).
    unwind_mode: 0=A (ph_level), 1=B (agg>=0 bar close), 2=C (entry)

    Returns (pnl[:nc], nlegs[:nc], ph_ops[:nc], nc)
    """
    n = len(cl)
    pnl_out    = np.zeros(n, dtype=np.float64)
    nlegs_out  = np.zeros(n, dtype=np.int32)
    ph_ops_out = np.zeros(n, dtype=np.int32)
    nc = 0

    lv = np.zeros(ml, dtype=np.float64)   # permanent leg volumes
    ld = np.zeros(ml, dtype=np.float64)   # permanent leg directions
    lp = np.zeros(ml, dtype=np.float64)   # permanent leg entry prices

    i = 0
    d = 1  # alternating entry direction

    while i < n:
        # ── Body absorption gate ────────────────────────────────────────────
        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i
        if body_thresh > 0.0 and rng_i > 1e-10:
            adv = (op_i - cl_i) if (d == 1 and op_i > cl_i) else \
                  (cl_i - op_i) if (d == -1 and cl_i > op_i) else 0.0
            if adv / rng_i > body_thresh:
                i += 1; continue

        # ── Open cycle ─────────────────────────────────────────────────────
        e = cl_i; fd = float(d)
        if d == 1:
            uz = e;              lz = e - zw * pip
            ut = e + tgt * pip;  lt = lz - tgt * pip
            ph_level = e - f_partial * zw * pip
        else:
            lz = e;              uz = e + zw * pip
            lt = e - tgt * pip;  ut = uz + tgt * pip
            ph_level = e + f_partial * zw * pip

        lv[0] = 1.0; ld[0] = fd; lp[0] = e
        nl = 1
        lu = -1; ll = -1  # zone-cross guard (bar index of last cross)

        # Partial hedge state
        ph_active   = False
        ph_dir      = 0.0
        ph_px       = 0.0
        realized    = 0.0
        ph_op_count = 0

        # 1-leg trailing stop state
        peak_mfe = 0.0
        ton      = False

        # PSAR state
        psar_on  = False
        psar_val = 0.0
        ep_val   = 0.0
        af_cur   = af0
        net_dir  = 0.0

        ex = False
        i += 1

        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = (c >= op[i])

            # ── 1. PSAR exit (highest priority) ───────────────────────────
            if psar_on:
                if net_dir > 0:
                    if h > ep_val:
                        ep_val = h
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl):
                            net += lv[k] * ld[k] * (psar_val - lp[k]) / pip
                            tv  += lv[k]
                        net -= tv * sp
                        if ph_active:
                            net += ph_dir * (psar_val - ph_px) / pip - sp
                        net += realized
                        pnl_out[nc] = net; nlegs_out[nc] = nl; ph_ops_out[nc] = ph_op_count
                        nc += 1; ex = True; break
                else:
                    if l < ep_val:
                        ep_val = l
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl):
                            net += lv[k] * ld[k] * (psar_val - lp[k]) / pip
                            tv  += lv[k]
                        net -= tv * sp
                        if ph_active:
                            net += ph_dir * (psar_val - ph_px) / pip - sp
                        net += realized
                        pnl_out[nc] = net; nlegs_out[nc] = nl; ph_ops_out[nc] = ph_op_count
                        nc += 1; ex = True; break
                i += 1; continue

            # ── 2. 1-leg ta/td trailing stop ──────────────────────────────
            if nl == 1:
                mfe = (h - e) / pip if d == 1 else (e - l) / pip
                if mfe > peak_mfe: peak_mfe = mfe
                if peak_mfe >= ta: ton = True
                if ton:
                    if d == 1:
                        be = e + sp * pip
                        ts = e + (peak_mfe - td) * pip
                        if ts < be: ts = be
                        if l <= ts:
                            net = (ts - e) / pip - sp
                            if ph_active:
                                net += ph_dir * (ts - ph_px) / pip - sp
                            net += realized
                            pnl_out[nc] = net; nlegs_out[nc] = 1; ph_ops_out[nc] = ph_op_count
                            nc += 1; ex = True; break
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            net = (e - ts) / pip - sp
                            if ph_active:
                                net += ph_dir * (ts - ph_px) / pip - sp
                            net += realized
                            pnl_out[nc] = net; nlegs_out[nc] = 1; ph_ops_out[nc] = ph_op_count
                            nc += 1; ex = True; break
            if ex: break

            # ── 3. Mode B unwind at bar close (nl==1) ─────────────────────
            if ph_active and nl == 1 and unwind_mode == 1:
                # Aggregate = permanent leg + hedge + realized at bar close
                agg_perm = ld[0] * (c - lp[0]) / pip
                agg_ph   = ph_dir * (c - ph_px) / pip
                agg_tot  = agg_perm + agg_ph + realized - 2.0 * sp
                if agg_tot >= 0.0:
                    realized += ph_dir * (c - ph_px) / pip - sp
                    ph_active = False

            # ── 4. Intra-bar events ────────────────────────────────────────
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                px    = h if is_hi else l

                # 4a. Mode A/C unwind (nl==1 only, not mode B)
                if ph_active and nl == 1 and unwind_mode != 1:
                    should_unwind = False; unwind_px = 0.0
                    if unwind_mode == 0:   # A: back through ph_level
                        if d == 1 and is_hi and px >= ph_level:
                            should_unwind = True; unwind_px = ph_level
                        elif d == -1 and (not is_hi) and px <= ph_level:
                            should_unwind = True; unwind_px = ph_level
                    else:                  # C: back through entry
                        if d == 1 and is_hi and px >= e:
                            should_unwind = True; unwind_px = e
                        elif d == -1 and (not is_hi) and px <= e:
                            should_unwind = True; unwind_px = e
                    if should_unwind:
                        realized += ph_dir * (unwind_px - ph_px) / pip - sp
                        ph_active = False

                # 4b. Open partial hedge (nl==1, f>0, not already active)
                if f_partial > 0.0 and not ph_active and nl == 1:
                    if d == 1 and (not is_hi) and px <= ph_level:
                        ph_active = True; ph_dir = -1.0
                        ph_px = ph_level; ph_op_count += 1
                    elif d == -1 and is_hi and px >= ph_level:
                        ph_active = True; ph_dir = 1.0
                        ph_px = ph_level; ph_op_count += 1

                # 4c. Zone crossings — add recovery legs
                # Upper crossing (LONG recovery, only if net short)
                if is_hi and px >= uz and lu != i:
                    lu = i
                    net_at = 0.0; tv_at = 0.0
                    for k in range(nl):
                        net_at += lv[k] * ld[k] * (ut - lp[k]) / pip
                        tv_at  += lv[k]
                    net_at -= tv_at * sp
                    if ph_active:
                        net_at += ph_dir * (ut - ph_px) / pip - sp
                    net_at += realized
                    if nl >= ml:
                        net2 = 0.0; tv2 = 0.0
                        for k in range(nl):
                            net2 += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                        net2 -= tv2 * sp
                        if ph_active: net2 += ph_dir * (c - ph_px) / pip - sp
                        net2 += realized
                        pnl_out[nc] = net2; nlegs_out[nc] = nl; ph_ops_out[nc] = ph_op_count
                        nc += 1; ex = True; break
                    if net_at < 0.0:
                        npu = max(tgt - sp, 1e-8)
                        v = max(1.0, math.ceil(-net_at / npu * pf))
                        lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1

                if ex: break

                # Lower crossing (SHORT recovery, only if net long)
                if (not is_hi) and px <= lz and ll != i:
                    ll = i
                    net_at = 0.0; tv_at = 0.0
                    for k in range(nl):
                        net_at += lv[k] * ld[k] * (lt - lp[k]) / pip
                        tv_at  += lv[k]
                    net_at -= tv_at * sp
                    if ph_active:
                        net_at += ph_dir * (lt - ph_px) / pip - sp
                    net_at += realized
                    if nl >= ml:
                        net2 = 0.0; tv2 = 0.0
                        for k in range(nl):
                            net2 += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                        net2 -= tv2 * sp
                        if ph_active: net2 += ph_dir * (c - ph_px) / pip - sp
                        net2 += realized
                        pnl_out[nc] = net2; nlegs_out[nc] = nl; ph_ops_out[nc] = ph_op_count
                        nc += 1; ex = True; break
                    if net_at < 0.0:
                        npu = max(tgt - sp, 1e-8)
                        v = max(1.0, math.ceil(-net_at / npu * pf))
                        lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1

                if ex: break

                # 4d. Target cross → activate PSAR
                if l <= ut <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    if ph_active: net_v += ph_dir
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = ut
                    psar_val = ut - tgt * pip if net_dir > 0 else ut + tgt * pip
                    break
                if l <= lt <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    if ph_active: net_v += ph_dir
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = lt
                    psar_val = lt - tgt * pip if net_dir > 0 else lt + tgt * pip
                    break

            i += 1
        # ── end of cycle ───────────────────────────────────────────────────
        d = -d

    return pnl_out[:nc], nlegs_out[:nc], ph_ops_out[:nc], nc


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([
        rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


# ── Run one config ────────────────────────────────────────────────────────────

def run_config(op, hi, lo, cl, sp, is_end, nb, oos_days, f_partial, unwind_mode, rng):

    def _call(s, e):
        return sim_zr_ph_tight(
            op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
            PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML,
            f_partial, unwind_mode
        )

    # IS walk-forward
    is_csz = is_end // IS_CHUNKS
    is_wf  = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
        p, _, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: is_wf += 1

    # Full OOS
    p_oos, nl_oos, ph_oos, nc_oos = _call(is_end, nb)
    ppd_oos = p_oos.sum() / oos_days if nc_oos > 0 else 0.0

    # OOS walk-forward
    oos_len = nb - is_end
    oos_csz = oos_len // OOS_CHUNKS
    oos_wf  = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
        p, _, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: oos_wf += 1

    # Bootstrap
    p5 = p_pos = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_oos > 0:
        p5, p_pos = _boot_stats(p_oos, oos_days, N_BOOT, rng)

    mode_lbl = MODE_LABELS.get(unwind_mode, '?') if f_partial > 0 else '-'

    l1_pct   = float(np.mean(nl_oos == 1) * 100) if nc_oos > 0 else float('nan')
    l2_pct   = float(np.mean(nl_oos == 2) * 100) if nc_oos > 0 else float('nan')
    l3p_pct  = float(np.mean(nl_oos >= 3) * 100) if nc_oos > 0 else float('nan')
    avg_legs = float(nl_oos.mean())               if nc_oos > 0 else float('nan')
    avg_ph   = float(ph_oos.mean())               if nc_oos > 0 else float('nan')

    passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0
              and not math.isnan(p_pos) and p_pos > 0.95)

    return dict(
        f=f_partial, mode=mode_lbl,
        ppd=round(ppd_oos, 1), nc=nc_oos,
        is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5, 1)     if not math.isnan(p5)    else float('nan'),
        p_pos=round(p_pos, 4) if not math.isnan(p_pos) else float('nan'),
        l1=round(l1_pct, 1), l2=round(l2_pct, 1), l3p=round(l3p_pct, 1),
        avg_legs=round(avg_legs, 2), avg_ph=round(avg_ph, 2),
        passed=passed,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("Loading data...", flush=True)
    mid = (pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts'] = mid['timestamp'].astype(str).str[:19]
    ba['ts']  = ba['timestamp'].astype(str).str[:19]
    df = mid.merge(ba[['ts', 'bid_c', 'ask_c']], on='ts', how='inner').reset_index(drop=True)
    print(f"  merged rows={len(df):,}  {df.timestamp.min()} → {df.timestamp.max()}")

    op = df.open.values.astype(np.float64)
    hi = df.high.values.astype(np.float64)
    lo = df.low.values.astype(np.float64)
    cl = df.close.values.astype(np.float64)
    sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)

    nb       = len(df)
    is_end   = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24.0 * 12.0)

    print(f"  is_end={is_end:,}  oos_bars={nb-is_end:,}  oos_days={oos_days:.1f}")
    print()

    print("JIT compile...", end=' ', flush=True)
    sim_zr_ph_tight(op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
                    PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML, 0.5, 0)
    print("done.\n")

    rng = np.random.default_rng(42)

    # Build sweep: baseline first, then all f × mode combos
    configs = [(0.0, 0)]  # baseline (f=0, mode irrelevant)
    for f in [fp for fp in F_PARTIALS if fp > 0]:
        for m in UNWIND_MODES:
            configs.append((f, m))

    results = []
    for fp, um in configs:
        lbl = f"f={fp:.2f} mode={MODE_LABELS[um] if fp > 0 else '-':1s}"
        print(f"  {lbl}...", end=' ', flush=True)
        t1 = time.time()
        row = run_config(op, hi, lo, cl, sp, is_end, nb, oos_days, fp, um, rng)
        results.append(row)
        tag = "🟢PASS" if row['passed'] else "🔴    "
        print(f"{tag}  p/d={row['ppd']:8.1f}  IS={row['is_wf']}/{IS_CHUNKS}  "
              f"OOS={row['oos_wf']}/{OOS_CHUNKS}  P5={row['p5']}  "
              f"L1={row['l1']:.1f}%  avg_ph={row['avg_ph']:.2f}  "
              f"({time.time()-t1:.1f}s)")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUT_CSV, index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    sep = "─" * 120
    print(f"\n{'═'*120}")
    print(f"  ZR Partial Hedge (Fixed-Target) — {PAIR}")
    print(f"  ZW={ZW}p  TGT={TGT}p  PF={PF}  body={BODY}  ta={TA}  td={TD}")
    print(f"  OOS={oos_days:.0f} days  IS={IS_CHUNKS}/OOS={OOS_CHUNKS} WF  boot={N_BOOT}")
    print(f"{'═'*120}")
    hdr = (f"  {'f':>5} {'mode':>4} | {'p/d':>8} {'cyc':>6} | "
           f"{'IS':>2} {'OOS':>3} | {'P5':>8} {'P(+)':>6} | "
           f"{'L1%':>5} {'L2%':>5} {'L3+%':>5} {'avgL':>5} {'avgPH':>5} | result")
    print(sep); print(hdr); print(sep)

    for row in results:
        p5s  = f"{row['p5']:8.1f}"    if not math.isnan(row['p5'])    else "     nan"
        pps  = f"{row['p_pos']:6.3f}" if not math.isnan(row['p_pos']) else "   nan"
        wf_ok = row['is_wf'] == IS_CHUNKS and row['oos_wf'] == OOS_CHUNKS
        tag  = "🟢 PASS" if row['passed'] else ("🟡 near" if wf_ok else "       ")
        print(f"  {row['f']:5.2f} {row['mode']:>4} | {row['ppd']:8.1f} {row['nc']:6d} | "
              f"{row['is_wf']:>2}/{IS_CHUNKS} {row['oos_wf']:>3}/{OOS_CHUNKS} | "
              f"{p5s} {pps} | "
              f"{row['l1']:5.1f} {row['l2']:5.1f} {row['l3p']:5.1f} "
              f"{row['avg_legs']:5.2f} {row['avg_ph']:5.2f} | {tag}")

    print(sep)
    n_pass = sum(1 for r in results if r['passed'])
    print(f"\n  Total configs: {len(results)}  |  Passing: {n_pass}")
    print(f"  Results → {OUT_CSV}")
    print(f"  Total runtime: {time.time()-t0:.1f}s\n")

    baseline = results[0]
    passing  = [r for r in results[1:] if r['passed']]
    best     = max(passing, key=lambda r: r['p5']) if passing else None

    print(f"  Baseline (no hedge):  p/d={baseline['ppd']:.1f}  "
          f"L1={baseline['l1']:.1f}%  P5={baseline['p5']}  P(+)={baseline['p_pos']}")
    if best:
        delta = best['ppd'] - baseline['ppd']
        sign  = '+' if delta >= 0 else ''
        print(f"  Best partial hedge:   p/d={best['ppd']:.1f} ({sign}{delta:.1f})  "
              f"f={best['f']}  mode={best['mode']}  P5={best['p5']}")
        print(f"  L1: {baseline['l1']:.1f}% → {best['l1']:.1f}%  "
              f"avgL: {baseline['avg_legs']} → {best['avg_legs']}")
    else:
        print("  No partial hedge config beats baseline with full-pass gates.")
    print()


if __name__ == "__main__":
    main()
