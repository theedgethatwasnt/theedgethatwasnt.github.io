"""
ZR Trail-Lock Backtest — GBP_USD, redesigned exit mechanism.

New design (Session 052): aggregate P/L trailing stop replaces PSAR escape.
- Entry: alternating L/S every bar, body_thresh=0.5 filter, spread gate
- Legs:  zone-boundary crossings add recovery legs (same as before)
         last_zone_crossed guard prevents re-fires on same boundary
- Exit (primary): aggregate P/L trailing stop:
           lock activates when agg >= min_lock
           trail shrinks from trail_base → min_lock as peak grows → trail_max_pnl
           fires when agg <= stop_pnl  → close all legs flat at bar close
- Exit (emergency): max_legs cap — close at market if cycle exceeds ML legs.
           This enforces the practical margin constraint (same ML=10 used in old design).
           Not a strategy exit — just prevents runaway cycles in ranging markets.

P/L unit: volume-weighted pips = sum_k(vol_k * dir_k * price_diff_k / pip) - total_vol * spread.
Financial P/L in USD = pnl_unit × base_units × pip_usd.

Parameter sweep:
  min_lock       ∈ [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
  trail_base     ∈ [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
  trail_max_pnl  ∈ [10.0, 20.0, 30.0, 50.0]

Fixed params: ZW=30, TGT=21, PF=1.10, body_thresh=0.5, N=1, ML=10

Gates (all required):
  IS_WF  : all 3 IS chunks positive
  OOS_WF : all 3 OOS chunks positive
  P5     : bootstrap 5th-pct OOS p/d > 0   (n_boot=2000)
  P_POS  : bootstrap P(p/d > 0) > 0.95     (MC bootstrap — correct for ZR cycle samples)
"""

import math, sys, time
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR         = "GBP_USD"
PIP          = 0.0001
ZW           = 30.0
TGT          = 21.0
PF           = 1.10
BODY_THRESH  = 0.5
ML           = 10     # max legs per cycle — emergency close if exceeded (margin constraint)
OOS_FRAC     = 0.30
IS_CHUNKS    = 3
OOS_CHUNKS   = 3
N_BOOT       = 2000

MIN_LOCK_VALUES    = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]  # all >= spread (~2.4p)
TRAIL_BASE_VALUES  = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
TRAIL_MAX_VALUES   = [10.0, 20.0, 30.0, 50.0]

OUT_CSV = Path(__file__).parent / "zr_trail_lock_results_v2.csv"


# ── Numba kernel ─────────────────────────────────────────────────────────────
@njit
def sim_zr_trail_lock(op, hi, lo, cl, sp_arr,
                      pip, pf, zw, tgt, body_thresh,
                      min_lock, trail_base, trail_max_pnl,
                      gate, ml):
    """
    Full ZR simulation with aggregate P/L trail-lock exit.
    Returns (pnl_arr[:nc], nlegs_arr[:nc], nc).

    Sequencing (R2):
      bull bar  → high processed before low
      bear bar  → low processed before high
    Zone crossings (R4a):
      last_zone tracks which boundary was last crossed (1=upper, -1=lower, 0=none)
      new LONG recovery only when last_zone != 1
      new SHORT recovery only when last_zone != -1
    Exit:
      bar close P/L checked against stop_pnl after locked
      No targets, no single-leg trail, no PSAR
    """
    n = len(cl)
    pnl_out   = np.zeros(n, dtype=np.float64)
    nlegs_out = np.zeros(n, dtype=np.int32)
    nc = 0

    lv = np.zeros(ml, dtype=np.float64)
    ld = np.zeros(ml, dtype=np.float64)
    lp = np.zeros(ml, dtype=np.float64)

    i = 0
    d = 1  # alternating direction

    while i < n:
        # ── Entry gate ─────────────────────────────────────────────────────
        sp_e = sp_arr[i]
        if gate > 0.0 and sp_e > gate:
            i += 1
            continue

        # Body absorption filter (skip without flipping direction)
        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i
        if body_thresh > 0.0 and rng_i > 1e-10:
            if d == 1:
                adv = (op_i - cl_i) if op_i > cl_i else 0.0
            else:
                adv = (cl_i - op_i) if cl_i > op_i else 0.0
            if adv / rng_i > body_thresh:
                i += 1
                continue

        # ── Open cycle ─────────────────────────────────────────────────────
        e = cl_i
        if d == 1:
            uz = e;             lz = e - zw * pip
            ut = e + tgt * pip; lt = lz - tgt * pip
        else:
            lz = e;             uz = e + zw * pip
            lt = e - tgt * pip; ut = uz + tgt * pip

        lv[0] = 1.0; ld[0] = float(d); lp[0] = e
        nl = 1
        last_zone = 0   # 0=none, 1=upper last, -1=lower last
        locked   = False
        peak_pnl = 0.0
        stop_pnl = 0.0
        ex = False
        i += 1

        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = c >= op[i]

            # ── Aggregate P/L at bar close ──────────────────────────────────
            agg = 0.0; tv = 0.0
            for k in range(nl):
                agg += lv[k] * ld[k] * (c - lp[k]) / pip
                tv  += lv[k]
            agg -= tv * sp

            # Lock-in trigger
            if not locked and agg >= min_lock:
                locked   = True
                peak_pnl = agg
                stop_pnl = min_lock

            # Trail update and exit check
            if locked:
                if agg > peak_pnl:
                    peak_pnl = agg
                    # Trail distance shrinks from trail_base → min_lock
                    if peak_pnl >= trail_max_pnl:
                        tdist = min_lock
                    else:
                        span = trail_max_pnl - min_lock
                        if span < 1e-9:
                            tdist = min_lock
                        else:
                            t = (peak_pnl - min_lock) / span
                            if t < 0.0: t = 0.0
                            if t > 1.0: t = 1.0
                            tdist = trail_base + t * (min_lock - trail_base)
                    new_stop = peak_pnl - tdist
                    if new_stop > stop_pnl:
                        stop_pnl = new_stop
                if agg <= stop_pnl:
                    pnl_out[nc]   = agg
                    nlegs_out[nc] = nl
                    nc += 1; ex = True; break

            # ── Zone crossings ──────────────────────────────────────────────
            # Bull bar: pass_idx=0 → is_hi=True (high first)
            # Bear bar: pass_idx=0 → is_hi=False (low first)
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))

                # Upper zone crossing → LONG recovery (when net short)
                if is_hi and h >= uz and last_zone != 1:
                    last_zone = 1
                    net_vol = 0.0
                    for k in range(nl): net_vol += lv[k] * ld[k]
                    if net_vol < 0.0:   # net short → LONG recovery valid
                        if nl >= ml:    # emergency: max legs reached — close at market
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            pnl_out[nc]   = net2 - tv2 * sp
                            nlegs_out[nc] = nl
                            nc += 1; ex = True; break
                        net_at = 0.0; tv_at = 0.0
                        for k in range(nl):
                            net_at += lv[k] * ld[k] * (ut - lp[k]) / pip
                            tv_at  += lv[k]
                        net_at -= tv_at * sp
                        if net_at < 0.0:
                            npu = tgt - sp
                            if npu < 1e-8: npu = 1e-8
                            v = math.ceil(-net_at / npu * pf)
                            if v < 1.0: v = 1.0
                            lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1

                if ex: break

                # Lower zone crossing → SHORT recovery (when net long)
                if (not is_hi) and l <= lz and last_zone != -1:
                    last_zone = -1
                    net_vol = 0.0
                    for k in range(nl): net_vol += lv[k] * ld[k]
                    if net_vol > 0.0:   # net long → SHORT recovery valid
                        if nl >= ml:    # emergency: max legs reached — close at market
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            pnl_out[nc]   = net2 - tv2 * sp
                            nlegs_out[nc] = nl
                            nc += 1; ex = True; break
                        net_at = 0.0; tv_at = 0.0
                        for k in range(nl):
                            net_at += lv[k] * ld[k] * (lt - lp[k]) / pip
                            tv_at  += lv[k]
                        net_at -= tv_at * sp
                        if net_at < 0.0:
                            npu = tgt - sp
                            if npu < 1e-8: npu = 1e-8
                            v = math.ceil(-net_at / npu * pf)
                            if v < 1.0: v = 1.0
                            lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1

            i += 1

        # Open cycle at end-of-data → discard (not counted)
        d = -d

    return pnl_out[:nc], nlegs_out[:nc], nc


# ── Numba parallel sweep ──────────────────────────────────────────────────────
@njit(parallel=True)
def sweep_trail_lock(op, hi, lo, cl, sp_arr, s, e2,
                     pip, pf, zw, tgt, body_thresh, gate, ml,
                     min_locks, trail_bases, trail_maxes):
    """
    Run all param combos for a given slice [s:e2].
    Returns ppd_arr[n_combos], nc_arr[n_combos].
    """
    n_ml = len(min_locks)
    n_tb = len(trail_bases)
    n_tm = len(trail_maxes)
    n_combos = n_ml * n_tb * n_tm
    days = (e2 - s) / (24.0 * 12.0)

    ppd_out = np.zeros(n_combos, dtype=np.float64)
    nc_out  = np.zeros(n_combos, dtype=np.int32)

    for idx in prange(n_combos):
        i_ml = idx // (n_tb * n_tm)
        i_tb = (idx // n_tm) % n_tb
        i_tm = idx % n_tm
        ml_ = min_locks[i_ml]
        tb  = trail_bases[i_tb]
        tm  = trail_maxes[i_tm]
        p, _, nc = sim_zr_trail_lock(
            op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2], sp_arr[s:e2],
            pip, pf, zw, tgt, body_thresh, ml_, tb, tm, gate, ml
        )
        ppd_out[idx] = p.sum() / days if days > 0 else 0.0
        nc_out[idx]  = nc

    return ppd_out, nc_out


# ── Bootstrap (MC) helpers ────────────────────────────────────────────────────
def _boot_stats(pnl: np.ndarray, oos_days: float, n_boot: int, rng) -> tuple:
    """
    Bootstrap MC: resample cycle P/Ls with replacement.
    Returns (p5, p_positive) where:
      p5        = 5th percentile of bootstrapped p/d  (robustness floor)
      p_positive = fraction of boots with p/d > 0     (confidence)
    """
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([
        rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...", flush=True)
    mid = (pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    df = (mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner')
          .reset_index(drop=True))
    print(f"  merged rows={len(df):,}  {df.timestamp.min()} → {df.timestamp.max()}")

    op = df.open.values.astype(np.float64)
    hi = df.high.values.astype(np.float64)
    lo = df.low.values.astype(np.float64)
    cl = df.close.values.astype(np.float64)
    sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)

    nb      = len(df)
    is_end  = int(nb * (1 - OOS_FRAC))
    is_csz  = is_end // IS_CHUNKS
    oos_len = nb - is_end
    oos_csz = oos_len // OOS_CHUNKS
    oos_days = oos_len / (24.0 * 12.0)

    gate = float(np.percentile(sp[:is_end], 90))
    print(f"  is_end={is_end:,}  oos_bars={oos_len:,}  oos_days={oos_days:.1f}  gate={gate:.2f}p")
    print()

    # Convert param lists to numpy arrays for Numba
    ml_arr = np.array(MIN_LOCK_VALUES,   dtype=np.float64)
    tb_arr = np.array(TRAIL_BASE_VALUES, dtype=np.float64)
    tm_arr = np.array(TRAIL_MAX_VALUES,  dtype=np.float64)
    n_ml, n_tb, n_tm = len(ml_arr), len(tb_arr), len(tm_arr)
    n_combos = n_ml * n_tb * n_tm

    print(f"Compiling Numba...", end=' ', flush=True)
    # JIT warm-up
    _p, _, _nc = sim_zr_trail_lock(
        op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
        PIP, PF, ZW, TGT, BODY_THRESH, 2.0, 10.0, 50.0, gate, ML
    )
    sweep_trail_lock(op, hi, lo, cl, sp, 0, 2000,
                     PIP, PF, ZW, TGT, BODY_THRESH, gate, ML,
                     ml_arr, tb_arr, tm_arr)
    print("done.\n", flush=True)

    rng = np.random.default_rng(42)

    # ── IS WF per combo ────────────────────────────────────────────────────
    print(f"Running IS WF ({IS_CHUNKS} chunks × {n_combos} combos)...", flush=True)
    is_wf_counts = np.zeros(n_combos, dtype=np.int32)
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
        ppd, nc_ = sweep_trail_lock(op, hi, lo, cl, sp, s_, e_,
                                    PIP, PF, ZW, TGT, BODY_THRESH, gate, ML,
                                    ml_arr, tb_arr, tm_arr)
        for idx in range(n_combos):
            if nc_[idx] > 0 and ppd[idx] > 0:
                is_wf_counts[idx] += 1
        print(f"  IS chunk {ch+1}/{IS_CHUNKS} done", flush=True)

    # ── OOS: full OOS p/d ──────────────────────────────────────────────────
    print(f"Running OOS ({OOS_CHUNKS} chunks)...", flush=True)
    oos_ppd, oos_nc = sweep_trail_lock(op, hi, lo, cl, sp, is_end, nb,
                                        PIP, PF, ZW, TGT, BODY_THRESH, gate, ML,
                                        ml_arr, tb_arr, tm_arr)
    oos_wf_counts = np.zeros(n_combos, dtype=np.int32)
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
        ppd, nc_ = sweep_trail_lock(op, hi, lo, cl, sp, s_, e_,
                                    PIP, PF, ZW, TGT, BODY_THRESH, gate, ML,
                                    ml_arr, tb_arr, tm_arr)
        for idx in range(n_combos):
            if nc_[idx] > 0 and ppd[idx] > 0:
                oos_wf_counts[idx] += 1
        print(f"  OOS chunk {ch+1}/{OOS_CHUNKS} done", flush=True)

    # ── Bootstrap + MC for passing combos ─────────────────────────────────
    print(f"\nBootstrap + MC for IS=3/3 & OOS=3/3 combos...", flush=True)
    results = []
    n_passing_wf = int(np.sum((is_wf_counts == IS_CHUNKS) & (oos_wf_counts == OOS_CHUNKS)))
    print(f"  {n_passing_wf} combos pass IS=3/3 & OOS=3/3", flush=True)

    for idx in range(n_combos):
        i_ml = idx // (n_tb * n_tm)
        i_tb = (idx // n_tm) % n_tb
        i_tm = idx % n_tm
        ml_ = ml_arr[i_ml]
        tb  = tb_arr[i_tb]
        tm  = tm_arr[i_tm]

        is_wf  = int(is_wf_counts[idx])
        oos_wf = int(oos_wf_counts[idx])
        ppd    = float(oos_ppd[idx])
        nc_    = int(oos_nc[idx])

        p5 = p_pos = float('nan')
        if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_ > 0:
            pnl_oos, _, _ = sim_zr_trail_lock(
                op[is_end:nb], hi[is_end:nb], lo[is_end:nb], cl[is_end:nb], sp[is_end:nb],
                PIP, PF, ZW, TGT, BODY_THRESH, ml_, tb, tm, gate, ML
            )
            p5, p_pos = _boot_stats(pnl_oos, oos_days, N_BOOT, rng)

        passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                  and not math.isnan(p5) and p5 > 0
                  and not math.isnan(p_pos) and p_pos > 0.95)

        results.append(dict(
            min_lock=ml_, trail_base=tb, trail_max_pnl=tm,
            ppd=round(ppd, 2), nc=nc_,
            is_wf=is_wf, oos_wf=oos_wf,
            p5=round(p5, 2) if not math.isnan(p5) else float('nan'),
            p_pos=round(p_pos, 4) if not math.isnan(p_pos) else float('nan'),
            passed=passed,
        ))

    # ── Sort and print ─────────────────────────────────────────────────────
    df_r = pd.DataFrame(results).sort_values('p5', ascending=False).reset_index(drop=True)
    df_r.to_csv(OUT_CSV, index=False)

    sep = "─" * 105
    hdr = (f"  {'ml':>5} {'tb':>5} {'tm':>6} | "
           f"{'p/d':>8} {'cycles':>7} | "
           f"{'IS':>2} {'OS':>2} | "
           f"{'P5':>8} {'P(+)':>6} | result")
    print(f"\n{'═'*105}")
    print(f"  ZR Trail-Lock — {PAIR}  ZW={ZW}p  TGT={TGT}p  PF={PF}  body={BODY_THRESH}")
    print(f"  OOS={oos_days:.0f} trading-days  gate={gate:.2f}p  IS={IS_CHUNKS}/OOS={OOS_CHUNKS} WF  boot={N_BOOT} (MC bootstrap)")
    print(f"{'═'*105}")
    print(sep); print(hdr); print(sep)

    n_pass = 0
    for _, row in df_r.iterrows():
        if row.is_wf < IS_CHUNKS or row.oos_wf < OOS_CHUNKS:
            continue  # skip clean failures from printed table
        p5_str   = f"{row.p5:8.1f}" if not math.isnan(row.p5) else "     nan"
        ppos_str = f"{row.p_pos:6.3f}" if not math.isnan(row.p_pos) else "   nan"
        tag = "🟢 PASS" if row.passed else "🟡 near"
        if row.passed: n_pass += 1
        print(f"  {row.min_lock:5.1f} {row.trail_base:5.1f} {row.trail_max_pnl:6.1f} | "
              f"  {row.ppd:8.1f} {row.nc:7d} | "
              f"{row.is_wf:>2}/{IS_CHUNKS} {row.oos_wf:>2}/{OOS_CHUNKS} | "
              f"{p5_str} {ppos_str} | {tag}")

    print(sep)
    print(f"\n  Total combos: {n_combos}  |  WF-stable (IS=3/3 & OOS=3/3): {int(np.sum((is_wf_counts==IS_CHUNKS)&(oos_wf_counts==OOS_CHUNKS)))}  |  Full-pass: {n_pass}")
    print(f"  Results saved → {OUT_CSV}\n")

    # ── Best config summary ────────────────────────────────────────────────
    passing = df_r[df_r.passed].head(5)
    if not passing.empty:
        print("  TOP PASSING CONFIGS (by P5):")
        for _, row in passing.iterrows():
            print(f"    min_lock={row.min_lock}  trail_base={row.trail_base}  "
                  f"trail_max_pnl={row.trail_max_pnl}  "
                  f"p/d={row.ppd:.1f}  P5={row.p5:.1f}  P(+)={row.p_pos:.3f}")
    else:
        near = df_r[(df_r.is_wf >= IS_CHUNKS) & (df_r.oos_wf >= OOS_CHUNKS)].head(5)
        if not near.empty:
            print("  NO FULL-PASS CONFIGS. Best WF-stable by P5:")
            for _, row in near.iterrows():
                p5_str   = f"{row.p5:.1f}" if not math.isnan(row.p5) else "nan"
                ppos_str = f"{row.p_pos:.3f}" if not math.isnan(row.p_pos) else "nan"
                print(f"    min_lock={row.min_lock}  trail_base={row.trail_base}  "
                      f"trail_max_pnl={row.trail_max_pnl}  "
                      f"p/d={row.ppd:.1f}  P5={p5_str}  P(+)={ppos_str}")
        else:
            print("  🔴 NO WF-STABLE CONFIGS FOUND. New exit mechanism does not produce edge.")
    print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time()-t0:.1f}s")
