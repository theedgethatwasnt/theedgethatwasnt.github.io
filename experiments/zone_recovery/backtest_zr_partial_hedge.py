"""
ZR Partial Hedge Experiment — GBP_USD
======================================

Two-level hedge design:
  Level 1 (at f_partial × ZW from entry): equal-volume delta-neutral hedge.
             Freezes loss at −f*ZW when price enters the zone.
  Level 2 (at ZW from entry): PF-sized recovery leg, sized to profit at TGT.
             Sizing accounts for the partial hedge already open.

Three unwind modes for Level 1:
  A (mode=0): close when price crosses back through the partial trigger level
  B (mode=1): close when net aggregate P&L >= 0  (bar-close check)
  C (mode=2): close when price crosses back through original entry price

Partial hedge can open and close multiple times per cycle (while np_legs==1).
Once a full recovery leg is added (np_legs>=2), partial hedge state is frozen.

Sweep: f_partial ∈ {0.0, 0.25, 0.33, 0.50, 0.67, 0.75}
       × unwind_mode ∈ {A, B, C}
       f_partial=0.0 = baseline (no partial hedge; just standard trail-lock ZR)

Fixed: ZW=30, TGT=21, PF=1.25, body=0.5, ML=10
       min_lock=2.0, trail_base=3.0, trail_max=30.0  (deployed config)

Gates: IS=3/3 WF chunks, OOS=3/3 WF chunks, P5>0, P(+)>0.95
"""

import math, time
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR       = "GBP_USD"
PIP        = 0.0001
ZW         = 30.0
TGT        = 21.0
PF         = 1.25
BODY       = 0.5
ML         = 10
MIN_LOCK   = 2.0
TRAIL_BASE = 3.0
TRAIL_MAX  = 30.0
OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000

F_PARTIALS   = [0.0, 0.25, 0.33, 0.50, 0.67, 0.75]
UNWIND_MODES = [0, 1, 2]   # 0=A, 1=B, 2=C
MODE_LABELS  = {0: 'A', 1: 'B', 2: 'C'}

OUT_CSV = Path(__file__).parent / "zr_partial_hedge_results.csv"


# ── Numba kernel ──────────────────────────────────────────────────────────────

@njit
def sim_zr_partial(op, hi, lo, cl, sp_arr,
                   pip, pf, zw, tgt, body_thresh,
                   min_lock, trail_base, trail_max_pnl,
                   gate, ml, f_partial, unwind_mode):
    """
    ZR with optional partial hedge at f_partial*ZW from entry.
    f_partial=0  → no partial hedge (baseline).
    unwind_mode: 0=A (ph_level), 1=B (agg>=0 at bar close), 2=C (entry_px)

    Returns (pnl_arr[:nc], nlegs_arr[:nc], ph_ops_arr[:nc], nc)
    nlegs    = number of permanent legs when cycle closed
    ph_ops   = number of times partial hedge was opened during the cycle
    """
    n = len(cl)
    pnl_out    = np.zeros(n, dtype=np.float64)
    nlegs_out  = np.zeros(n, dtype=np.int32)
    ph_ops_out = np.zeros(n, dtype=np.int32)
    nc = 0

    # Permanent leg storage
    pv = np.zeros(ml, dtype=np.float64)   # volume
    pd_ = np.zeros(ml, dtype=np.float64)  # direction (+1/-1)
    pp = np.zeros(ml, dtype=np.float64)   # entry price

    i = 0
    d = 1  # alternating entry direction

    while i < n:
        # ── Entry gate ─────────────────────────────────────────────────────
        sp_e = sp_arr[i]
        if gate > 0.0 and sp_e > gate:
            i += 1; continue

        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i
        if body_thresh > 0.0 and rng_i > 1e-10:
            adv = (op_i - cl_i) if (d == 1 and op_i > cl_i) else \
                  (cl_i - op_i) if (d == -1 and cl_i > op_i) else 0.0
            if adv / rng_i > body_thresh:
                i += 1; continue

        # ── Open cycle ─────────────────────────────────────────────────────
        e = cl_i
        fd = float(d)

        if d == 1:
            uz = e;              lz = e - zw * pip
            ut = e + tgt * pip;  lt = lz - tgt * pip
            ph_level = e - f_partial * zw * pip   # partial trigger: below entry
        else:
            lz = e;              uz = e + zw * pip
            lt = e - tgt * pip;  ut = uz + tgt * pip
            ph_level = e + f_partial * zw * pip   # partial trigger: above entry

        # Permanent leg 0: original entry
        pv[0] = 1.0; pd_[0] = fd; pp[0] = e
        np_legs = 1

        # Partial hedge state
        ph_active   = False
        ph_dir      = 0.0          # direction of partial hedge (-d)
        ph_px       = 0.0          # partial hedge entry price
        realized    = 0.0          # realized P&L from closed partial hedges
        ph_op_count = 0            # stats

        last_zone = 0              # zone crossing guard: 1=upper, -1=lower, 0=none
        locked    = False
        peak_pnl  = 0.0
        stop_pnl  = 0.0
        ex        = False
        i        += 1

        # ── Inner bar loop ─────────────────────────────────────────────────
        while i < n and not ex:
            h  = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = c >= op[i]

            # 1. Aggregate P&L at bar close (permanent legs + partial hedge)
            agg = realized
            tv  = 0.0
            for k in range(np_legs):
                agg += pv[k] * pd_[k] * (c - pp[k]) / pip
                tv  += pv[k]
            if ph_active:
                agg += ph_dir * (c - ph_px) / pip
                tv  += 1.0
            agg -= tv * sp

            # 2. Trail-lock update and exit check
            if not locked and agg >= min_lock:
                locked   = True
                peak_pnl = agg
                stop_pnl = min_lock

            if locked:
                if agg > peak_pnl:
                    peak_pnl = agg
                    if peak_pnl >= trail_max_pnl:
                        tdist = min_lock
                    else:
                        span = trail_max_pnl - min_lock
                        t = (peak_pnl - min_lock) / span
                        if t < 0.0: t = 0.0
                        if t > 1.0: t = 1.0
                        tdist = trail_base + t * (min_lock - trail_base)
                    ns = peak_pnl - tdist
                    if ns > stop_pnl: stop_pnl = ns
                if agg <= stop_pnl:
                    pnl_out[nc]    = agg
                    nlegs_out[nc]  = np_legs
                    ph_ops_out[nc] = ph_op_count
                    nc += 1; ex = True; break

            # 3. Mode B unwind (bar-close): only while np_legs==1
            if ph_active and np_legs == 1 and unwind_mode == 1 and agg >= 0.0:
                realized += ph_dir * (c - ph_px) / pip - sp
                ph_active = False

            # 4. Intra-bar events (bull=high first, bear=low first)
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                px = h if is_hi else l

                # 4a. Unwind partial hedge (Mode A or C, np_legs==1 only)
                if ph_active and np_legs == 1 and unwind_mode != 1:
                    should_unwind = False
                    unwind_px     = 0.0
                    if unwind_mode == 0:   # A: price crosses back through ph_level
                        if d == 1 and is_hi and px >= ph_level:
                            should_unwind = True; unwind_px = ph_level
                        elif d == -1 and (not is_hi) and px <= ph_level:
                            should_unwind = True; unwind_px = ph_level
                    else:                  # C: price crosses back through entry
                        if d == 1 and is_hi and px >= e:
                            should_unwind = True; unwind_px = e
                        elif d == -1 and (not is_hi) and px <= e:
                            should_unwind = True; unwind_px = e
                    if should_unwind:
                        realized += ph_dir * (unwind_px - ph_px) / pip - sp
                        ph_active = False

                # 4b. Open partial hedge (np_legs==1, not already active)
                if f_partial > 0.0 and not ph_active and np_legs == 1:
                    if d == 1 and (not is_hi) and px <= ph_level:
                        ph_active = True
                        ph_dir    = -1.0     # SHORT hedge for LONG original
                        ph_px     = ph_level
                        ph_op_count += 1
                    elif d == -1 and is_hi and px >= ph_level:
                        ph_active = True
                        ph_dir    = 1.0      # LONG hedge for SHORT original
                        ph_px     = ph_level
                        ph_op_count += 1

                # 4c. Zone boundary crossings
                # Upper crossing → LONG recovery (only when net SHORT)
                if is_hi and px >= uz and last_zone != 1:
                    last_zone = 1
                    net_vol = 0.0
                    for k in range(np_legs): net_vol += pv[k] * pd_[k]
                    if ph_active: net_vol += ph_dir
                    if net_vol < 0.0:
                        if np_legs >= ml:
                            net2 = realized; tv2 = 0.0
                            for k in range(np_legs):
                                net2 += pv[k] * pd_[k] * (c - pp[k]) / pip
                                tv2  += pv[k]
                            if ph_active:
                                net2 += ph_dir * (c - ph_px) / pip; tv2 += 1.0
                            pnl_out[nc]    = net2 - tv2 * sp
                            nlegs_out[nc]  = np_legs
                            ph_ops_out[nc] = ph_op_count
                            nc += 1; ex = True; break
                        # Size recovery at ut
                        nat = 0.0; tvt = 0.0
                        for k in range(np_legs):
                            nat += pv[k] * pd_[k] * (ut - pp[k]) / pip; tvt += pv[k]
                        if ph_active:
                            nat += ph_dir * (ut - ph_px) / pip; tvt += 1.0
                        nat -= tvt * sp
                        if nat < 0.0:
                            npu = tgt - sp
                            if npu < 1e-8: npu = 1e-8
                            v = math.ceil(-nat / npu * pf)
                            if v < 1.0: v = 1.0
                            pv[np_legs] = v; pd_[np_legs] = 1.0; pp[np_legs] = uz
                            np_legs += 1

                if ex: break

                # Lower crossing → SHORT recovery (only when net LONG)
                if (not is_hi) and px <= lz and last_zone != -1:
                    last_zone = -1
                    net_vol = 0.0
                    for k in range(np_legs): net_vol += pv[k] * pd_[k]
                    if ph_active: net_vol += ph_dir
                    if net_vol > 0.0:
                        if np_legs >= ml:
                            net2 = realized; tv2 = 0.0
                            for k in range(np_legs):
                                net2 += pv[k] * pd_[k] * (c - pp[k]) / pip
                                tv2  += pv[k]
                            if ph_active:
                                net2 += ph_dir * (c - ph_px) / pip; tv2 += 1.0
                            pnl_out[nc]    = net2 - tv2 * sp
                            nlegs_out[nc]  = np_legs
                            ph_ops_out[nc] = ph_op_count
                            nc += 1; ex = True; break
                        # Size recovery at lt
                        nat = 0.0; tvt = 0.0
                        for k in range(np_legs):
                            nat += pv[k] * pd_[k] * (lt - pp[k]) / pip; tvt += pv[k]
                        if ph_active:
                            nat += ph_dir * (lt - ph_px) / pip; tvt += 1.0
                        nat -= tvt * sp
                        if nat < 0.0:
                            npu = tgt - sp
                            if npu < 1e-8: npu = 1e-8
                            v = math.ceil(-nat / npu * pf)
                            if v < 1.0: v = 1.0
                            pv[np_legs] = v; pd_[np_legs] = -1.0; pp[np_legs] = lz
                            np_legs += 1

            i += 1
        # ── End of cycle ───────────────────────────────────────────────────
        d = -d

    return pnl_out[:nc], nlegs_out[:nc], ph_ops_out[:nc], nc


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([
        rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


# ── Run one config ────────────────────────────────────────────────────────────

def run_config(op, hi, lo, cl, sp, is_end, nb, oos_days,
               gate, f_partial, unwind_mode, rng):
    mode_lbl = MODE_LABELS.get(unwind_mode, '?') if f_partial > 0 else '-'

    def _call(s, e):
        return sim_zr_partial(
            op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
            PIP, PF, ZW, TGT, BODY, MIN_LOCK, TRAIL_BASE, TRAIL_MAX,
            gate, ML, f_partial, unwind_mode
        )

    # IS WF
    is_csz = is_end // IS_CHUNKS
    is_wf  = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
        p, _, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: is_wf += 1

    # OOS full
    p_oos, nl_oos, ph_oos, nc_oos = _call(is_end, nb)
    ppd_oos = p_oos.sum() / oos_days if nc_oos > 0 else 0.0

    # OOS WF
    oos_len = nb - is_end
    oos_csz = oos_len // OOS_CHUNKS
    oos_wf  = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
        p, _, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: oos_wf += 1

    # Bootstrap on OOS if WF passes
    p5 = p_pos = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_oos > 0:
        p5, p_pos = _boot_stats(p_oos, oos_days, N_BOOT, rng)

    # Leg distribution (OOS)
    l1_pct  = float(np.mean(nl_oos == 1) * 100) if nc_oos > 0 else float('nan')
    l2_pct  = float(np.mean(nl_oos == 2) * 100) if nc_oos > 0 else float('nan')
    l3p_pct = float(np.mean(nl_oos >= 3) * 100) if nc_oos > 0 else float('nan')
    avg_legs = float(nl_oos.mean()) if nc_oos > 0 else float('nan')
    avg_ph  = float(ph_oos.mean()) if nc_oos > 0 else float('nan')

    passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0
              and not math.isnan(p_pos) and p_pos > 0.95)

    return dict(
        f=f_partial, mode=mode_lbl,
        ppd=round(ppd_oos, 1), nc=nc_oos,
        is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5, 1) if not math.isnan(p5) else float('nan'),
        p_pos=round(p_pos, 4) if not math.isnan(p_pos) else float('nan'),
        l1=round(l1_pct, 1), l2=round(l2_pct, 1),
        l3p=round(l3p_pct, 1), avg_legs=round(avg_legs, 2),
        avg_ph=round(avg_ph, 2),
        passed=passed,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

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

    nb      = len(df)
    is_end  = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24.0 * 12.0)
    gate    = float(np.percentile(sp[:is_end], 90))

    print(f"  is_end={is_end:,}  oos_bars={nb-is_end:,}  oos_days={oos_days:.1f}  gate={gate:.2f}p")
    print()

    print("JIT compile...", end=' ', flush=True)
    sim_zr_partial(op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
                   PIP, PF, ZW, TGT, BODY, MIN_LOCK, TRAIL_BASE, TRAIL_MAX,
                   gate, ML, 0.5, 0)
    print("done.\n")

    rng = np.random.default_rng(42)

    # Build sweep: f_partial=0 (baseline, mode irrelevant) + all f × mode combos
    configs = [(0.0, 0)]   # baseline
    for f in [fp for fp in F_PARTIALS if fp > 0]:
        for m in UNWIND_MODES:
            configs.append((f, m))

    results = []
    for fp, um in configs:
        lbl = f"f={fp:.2f} mode={MODE_LABELS[um] if fp>0 else '-':1s}"
        print(f"  {lbl}...", end=' ', flush=True)
        t1 = time.time()
        row = run_config(op, hi, lo, cl, sp, is_end, nb, oos_days,
                         gate, fp, um, rng)
        results.append(row)
        tag = "🟢PASS" if row['passed'] else "🔴    "
        print(f"{tag}  p/d={row['ppd']:8.1f}  IS={row['is_wf']}/{IS_CHUNKS}  "
              f"OOS={row['oos_wf']}/{OOS_CHUNKS}  P5={row['p5']}  "
              f"L1={row['l1']:.1f}%  avg_ph={row['avg_ph']:.2f}  "
              f"({time.time()-t1:.1f}s)")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUT_CSV, index=False)

    # ── Print table ───────────────────────────────────────────────────────────
    sep = "─" * 115
    print(f"\n{'═'*115}")
    print(f"  ZR Partial Hedge — {PAIR}  ZW={ZW}p TGT={TGT}p PF={PF} body={BODY}")
    print(f"  min_lock={MIN_LOCK} trail_base={TRAIL_BASE} trail_max={TRAIL_MAX}")
    print(f"  OOS={oos_days:.0f} days  gate={gate:.2f}p  IS={IS_CHUNKS}/OOS={OOS_CHUNKS} WF  boot={N_BOOT}")
    print(f"{'═'*115}")
    hdr = (f"  {'f':>5} {'mode':>4} | {'p/d':>8} {'cyc':>6} | "
           f"{'IS':>2} {'OOS':>3} | {'P5':>8} {'P(+)':>6} | "
           f"{'L1%':>5} {'L2%':>5} {'L3+%':>5} {'avgL':>5} {'avgPH':>5} | result")
    print(sep); print(hdr); print(sep)

    for row in results:
        p5s   = f"{row['p5']:8.1f}" if not math.isnan(row['p5']) else "     nan"
        pps   = f"{row['p_pos']:6.3f}" if not math.isnan(row['p_pos']) else "   nan"
        tag   = "🟢 PASS" if row['passed'] else ("🟡 near" if row['is_wf']==IS_CHUNKS and row['oos_wf']==OOS_CHUNKS else "       ")
        print(f"  {row['f']:5.2f} {row['mode']:>4} | {row['ppd']:8.1f} {row['nc']:6d} | "
              f"{row['is_wf']:>2}/{IS_CHUNKS} {row['oos_wf']:>3}/{OOS_CHUNKS} | "
              f"{p5s} {pps} | "
              f"{row['l1']:5.1f} {row['l2']:5.1f} {row['l3p']:5.1f} {row['avg_legs']:5.2f} {row['avg_ph']:5.2f} | {tag}")

    print(sep)
    n_pass = sum(1 for r in results if r['passed'])
    print(f"\n  Total configs: {len(results)}  |  Passing: {n_pass}")
    print(f"  Results → {OUT_CSV}")
    print(f"  Total runtime: {time.time()-t0:.1f}s\n")

    # ── Baseline vs best comparison ───────────────────────────────────────────
    baseline = results[0]
    passing  = [r for r in results[1:] if r['passed']]
    best     = max(passing, key=lambda r: r['p5']) if passing else None

    print(f"  Baseline (no partial hedge): p/d={baseline['ppd']}  L1={baseline['l1']}%  P5={baseline['p5']}")
    if best:
        delta = best['ppd'] - baseline['ppd']
        print(f"  Best partial hedge:          p/d={best['ppd']} (+{delta:.1f})  "
              f"f={best['f']}  mode={best['mode']}  P5={best['p5']}")
        print(f"  L1: {baseline['l1']}% → {best['l1']}%  "
              f"avg_legs: {baseline['avg_legs']} → {best['avg_legs']}")
    else:
        print("  No config beats baseline with full-pass gates.")
    print()


if __name__ == "__main__":
    main()
