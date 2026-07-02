"""
ZR Double-Down Long (DDL) — GBP_USD
=====================================

Standard ZR with one key change: the FIRST zone-boundary crossing doubles
down on the original direction instead of adding a counter-recovery leg.

3-Phase design (LONG first-entry example — SHORT is mirror):
  Phase 1 (nl=1): 1u LONG at e. ta/td trail. Body filter.
                  Zone: uz=e, lz=e-ZW. PSAR at ut=e+TGT or lt=lz-TGT.
  Phase 2 (nl=2): if price hits lz → add 1u LONG (same direction, double-down).
                  avg = (e + lz)/2 = e - ZW/2.
                  Rebase zone: uz=avg, lz=avg-ZW, ut=avg+TGT, lt=avg-ZW-TGT.
                  Position is 2u LONG, profitable at avg+TGT = e + (TGT - ZW/2).
  Phase 3+:       if price hits lz2=avg-ZW → standard ZR SHORT recovery from
                  2u LONG avg position. Alternating legs, PF-sized. Targets ut/lt.

Key math (ZW=30, TGT=21): after double-down target shifts from e+21 → e+6.
Phase 3 SHORT recovery sizing at avg-ZW (e-45): ~7u (vs standard ZR's 4u at e-30).

Gates: IS=3/3 WF, OOS=3/3 WF, P5>0, P(+)>0.95
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
TA       = 6.0
TD       = 1.0
AF0      = 0.01
AFST     = 0.01
AFMX     = 0.20
ML       = 10
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT   = 2000

OUT_CSV = Path(__file__).parent / "zr_ddlong_results.csv"


@njit
def sim_zr_ddlong(op, hi, lo, cl, sp_arr,
                  pip, pf, zw, tgt, body_thresh,
                  ta, td, af0, af_step, af_max, ml):
    """
    ZR with double-down first crossing.

    Phase 1 (nl=1): standard 1-leg ta/td trail.
    Phase 2 (nl=2, dd_done=True): first zone-boundary hit → adds same-direction
                                   leg (LONG for LONG entry, SHORT for SHORT).
                                   Zone rebased to new avg price.
    Phase 3+ (nl>2): standard ZR alternating recovery from rebased zone.

    Returns (pnl[:nc], nlegs[:nc], nc)
    """
    n = len(cl)
    pnl_out   = np.zeros(n, dtype=np.float64)
    nlegs_out = np.zeros(n, dtype=np.int32)
    nc = 0

    lv = np.zeros(ml, dtype=np.float64)
    ld = np.zeros(ml, dtype=np.float64)
    lp = np.zeros(ml, dtype=np.float64)

    i = 0
    d = 1  # alternating entry direction

    while i < n:
        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i
        if body_thresh > 0.0 and rng_i > 1e-10:
            adv = (op_i - cl_i) if (d == 1 and op_i > cl_i) else \
                  (cl_i - op_i) if (d == -1 and cl_i > op_i) else 0.0
            if adv / rng_i > body_thresh:
                i += 1; continue

        # ── Open cycle ─────────────────────────────────────────────────
        e = cl_i; fd = float(d)
        if d == 1:
            uz = e;              lz = e - zw * pip
            ut = e + tgt * pip;  lt = lz - tgt * pip
        else:
            lz = e;              uz = e + zw * pip
            lt = e - tgt * pip;  ut = uz + tgt * pip

        lv[0] = 1.0; ld[0] = fd; lp[0] = e
        nl = 1
        lu = -1; ll = -1

        dd_done = False  # whether the double-down leg has been placed

        # 1-leg trail state
        peak_mfe = 0.0; ton = False

        # PSAR state
        psar_on  = False; psar_val = 0.0; ep_val = 0.0
        af_cur   = af0; net_dir = 0.0

        ex = False
        i += 1

        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = (c >= op[i])

            # ── 1. PSAR exit ───────────────────────────────────────────
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
                        pnl_out[nc] = net; nlegs_out[nc] = nl
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
                        pnl_out[nc] = net; nlegs_out[nc] = nl
                        nc += 1; ex = True; break
                i += 1; continue

            # ── 2. 1-leg ta/td trail (Phase 1 only) ───────────────────
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
                            pnl_out[nc] = net; nlegs_out[nc] = 1
                            nc += 1; ex = True; break
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            net = (e - ts) / pip - sp
                            pnl_out[nc] = net; nlegs_out[nc] = 1
                            nc += 1; ex = True; break
            if ex: break

            # ── 3. Intra-bar events ────────────────────────────────────
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                px    = h if is_hi else l

                # ── 3a. Upper crossing ─────────────────────────────────
                if is_hi and px >= uz and lu != i:
                    lu = i
                    if not dd_done and nl == 1 and ld[0] < 0:
                        # SHORT first entry: double-down SHORT at uz
                        lv[nl] = 1.0; ld[nl] = -1.0; lp[nl] = uz
                        nl += 1; dd_done = True
                        avg = (lp[0] + lp[1]) * 0.5
                        lz = avg; uz = avg + zw * pip
                        lt = avg - tgt * pip; ut = uz + tgt * pip
                    else:
                        # Standard ZR: LONG recovery at uz (net_at at ut)
                        net_at = 0.0; tv_at = 0.0
                        for k in range(nl):
                            net_at += lv[k] * ld[k] * (ut - lp[k]) / pip
                            tv_at  += lv[k]
                        net_at -= tv_at * sp
                        if nl >= ml:
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            net2 -= tv2 * sp
                            pnl_out[nc] = net2; nlegs_out[nc] = nl
                            nc += 1; ex = True; break
                        if net_at < 0.0:
                            npu = max(tgt - sp, 1e-8)
                            v = max(1.0, math.ceil(-net_at / npu * pf))
                            lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1

                if ex: break

                # ── 3b. Lower crossing ─────────────────────────────────
                if (not is_hi) and px <= lz and ll != i:
                    ll = i
                    if not dd_done and nl == 1 and ld[0] > 0:
                        # LONG first entry: double-down LONG at lz
                        lv[nl] = 1.0; ld[nl] = 1.0; lp[nl] = lz
                        nl += 1; dd_done = True
                        avg = (lp[0] + lp[1]) * 0.5
                        uz = avg; lz = avg - zw * pip
                        ut = avg + tgt * pip; lt = lz - tgt * pip
                    else:
                        # Standard ZR: SHORT recovery at lz (net_at at lt)
                        net_at = 0.0; tv_at = 0.0
                        for k in range(nl):
                            net_at += lv[k] * ld[k] * (lt - lp[k]) / pip
                            tv_at  += lv[k]
                        net_at -= tv_at * sp
                        if nl >= ml:
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            net2 -= tv2 * sp
                            pnl_out[nc] = net2; nlegs_out[nc] = nl
                            nc += 1; ex = True; break
                        if net_at < 0.0:
                            npu = max(tgt - sp, 1e-8)
                            v = max(1.0, math.ceil(-net_at / npu * pf))
                            lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1

                if ex: break

                # ── 3c. Target cross → activate PSAR ──────────────────
                if l <= ut <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = ut
                    psar_val = ut - tgt * pip if net_dir > 0 else ut + tgt * pip
                    break
                if l <= lt <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = lt
                    psar_val = lt - tgt * pip if net_dir > 0 else lt + tgt * pip
                    break

            i += 1
        d = -d

    return pnl_out[:nc], nlegs_out[:nc], nc


def _boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([
        rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


def run_one(op, hi, lo, cl, sp, is_end, nb, oos_days, rng, label):
    def _call(s, e_):
        return sim_zr_ddlong(
            op[s:e_], hi[s:e_], lo[s:e_], cl[s:e_], sp[s:e_],
            PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML
        )

    is_csz = is_end // IS_CHUNKS
    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
        p, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: is_wf += 1

    p_oos, nl_oos, nc_oos = _call(is_end, nb)
    ppd_oos = p_oos.sum() / oos_days if nc_oos > 0 else 0.0

    oos_len = nb - is_end
    oos_csz = oos_len // OOS_CHUNKS
    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
        p, _, nc = _call(s_, e_)
        days = (e_ - s_) / (24.0 * 12.0)
        if nc > 0 and p.sum() / days > 0: oos_wf += 1

    p5 = p_pos = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_oos > 0:
        p5, p_pos = _boot_stats(p_oos, oos_days, N_BOOT, rng)

    l1_pct  = float(np.mean(nl_oos == 1) * 100) if nc_oos > 0 else float('nan')
    l2_pct  = float(np.mean(nl_oos == 2) * 100) if nc_oos > 0 else float('nan')
    l3p_pct = float(np.mean(nl_oos >= 3) * 100) if nc_oos > 0 else float('nan')
    avg_pnl = float(p_oos.mean()) if nc_oos > 0 else float('nan')

    passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0
              and not math.isnan(p_pos) and p_pos > 0.95)

    return dict(
        label=label, ppd=round(ppd_oos, 1), nc=nc_oos,
        is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5, 1) if not math.isnan(p5) else float('nan'),
        p_pos=round(p_pos, 4) if not math.isnan(p_pos) else float('nan'),
        l1=round(l1_pct, 1), l2=round(l2_pct, 1), l3p=round(l3p_pct, 1),
        avg_pnl=round(avg_pnl, 2) if not math.isnan(avg_pnl) else float('nan'),
        passed=passed,
    )


BASE_PPD = 7180.2; BASE_P5 = 1644.9  # from backtest_zr_ph_tight.py baseline (f=0)


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
    print(f"  rows={len(df):,}  {df.timestamp.min()} → {df.timestamp.max()}")

    op = df.open.values.astype(np.float64)
    hi = df.high.values.astype(np.float64)
    lo = df.low.values.astype(np.float64)
    cl = df.close.values.astype(np.float64)
    sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)

    nb       = len(df)
    is_end   = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24.0 * 12.0)
    print(f"  is_end={is_end:,}  oos_bars={nb-is_end:,}  oos_days={oos_days:.1f}")

    print("\nJIT compile...", end=' ', flush=True)
    sim_zr_ddlong(op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
                  PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML)
    print("done.\n")

    rng = np.random.default_rng(42)

    print("  ddlong...", end=' ', flush=True)
    t1 = time.time()
    row = run_one(op, hi, lo, cl, sp, is_end, nb, oos_days, rng, 'ddlong')
    results = [row]
    tag = "🟢PASS" if row['passed'] else "🔴    "
    p5s = f"{row['p5']:.1f}" if not math.isnan(row['p5']) else 'nan'
    print(f"{tag}  p/d={row['ppd']:8.1f}  IS={row['is_wf']}/{IS_CHUNKS}  "
          f"OOS={row['oos_wf']}/{OOS_CHUNKS}  P5={p5s}  "
          f"L1={row['l1']:.1f}%  L2={row['l2']:.1f}%  ({time.time()-t1:.1f}s)")

    sep = "─" * 110
    print(f"\n{'═'*110}")
    print(f"  ZR Double-Down Long (DDL) vs Baseline — {PAIR}")
    print(f"  ZW={ZW}p  TGT={TGT}p  PF={PF}  body={BODY}  ta={TA}  td={TD}")
    print(f"  OOS={oos_days:.0f} days  IS={IS_CHUNKS}/OOS={OOS_CHUNKS} WF  boot={N_BOOT}")
    print(f"{'═'*110}")
    hdr = (f"  {'label':>20} | {'p/d':>8} {'cyc':>6} | "
           f"{'IS':>2} {'OOS':>3} | {'P5':>8} {'P(+)':>6} | "
           f"{'L1%':>5} {'L2%':>5} {'L3+%':>5} {'avg/t':>6} | result")
    # baseline row (from zr_ph_tight.py)
    print(sep); print(hdr); print(sep)
    print(f"  {'baseline_tight':>20} | {BASE_PPD:8.1f}  {'3753':>5} | "
          f" 3/3   3/3 | {'1644.9':>8} {'1.000':>6} | "
          f"{'83.8':>5} {'8.1':>5} {'8.1':>5} {'   ---':>6} | 🟢 PASS  [reference]")
    for row in results:
        p5s  = f"{row['p5']:8.1f}"    if not math.isnan(row['p5'])    else "     nan"
        pps  = f"{row['p_pos']:6.3f}" if not math.isnan(row['p_pos']) else "   nan"
        avgs = f"{row['avg_pnl']:6.2f}" if not math.isnan(row['avg_pnl']) else "   nan"
        tag  = "🟢 PASS" if row['passed'] else "🔴     "
        print(f"  {row['label']:>20} | {row['ppd']:8.1f} {row['nc']:6d} | "
              f"{row['is_wf']:>2}/{IS_CHUNKS} {row['oos_wf']:>3}/{OOS_CHUNKS} | "
              f"{p5s} {pps} | "
              f"{row['l1']:5.1f} {row['l2']:5.1f} {row['l3p']:5.1f} {avgs} | {tag}")
    print(sep)

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    ddl = results[0]
    delta = ddl['ppd'] - BASE_PPD
    sign = '+' if delta >= 0 else ''
    print(f"\n  Baseline (tight ZR)  p/d={BASE_PPD:.1f}  P5={BASE_P5}  L1=83.8%  L3+=8.1%")
    print(f"  DDL (double-down)    p/d={ddl['ppd']:.1f}  P5={ddl['p5']}  L1={ddl['l1']:.1f}%  L3+={ddl['l3p']:.1f}%  ({sign}{delta:.1f} vs baseline)")
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
    print(f"  Results → {OUT_CSV}\n")


if __name__ == "__main__":
    main()
