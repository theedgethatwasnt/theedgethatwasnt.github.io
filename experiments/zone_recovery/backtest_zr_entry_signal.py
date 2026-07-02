"""
ZR Entry Signal Test — body-vs-SMA10 directional filter, M5 and H1.

Signal at entry bar i:
  M5: prev M5 bar's high < SMA10(highs[-10:]) → force LONG
      prev M5 bar's low  > SMA10(lows[-10:])  → force SHORT
      else                                     → alternating (baseline)
  H1: same logic on last completed H1 bar vs SMA10 of H1 highs/lows
      (H1 built from M5 via proper timestamp resampling to handle gaps)

Fixed params (best validated v2): min_lock=2.0, trail_base=3.0, trail_max_pnl=30.0
Gates: IS=3/3 + OOS=3/3 + P5>0 + P(+)>0.95
"""

import math, time
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR        = "GBP_USD"
PIP         = 0.0001
ZW          = 30.0
TGT         = 21.0
PF          = 1.10
BODY_THRESH = 0.5
ML          = 10
MIN_LOCK    = 2.0
TRAIL_BASE  = 3.0
TRAIL_MAX   = 30.0
OOS_FRAC    = 0.30
IS_CHUNKS   = 3
OOS_CHUNKS  = 3
N_BOOT      = 2000
SMA_W       = 10

OUT_CSV = Path(__file__).parent / "zr_entry_signal_results.csv"


# ── Signal computation ────────────────────────────────────────────────────────

def make_m5_signal(hi: np.ndarray, lo: np.ndarray, w: int = SMA_W) -> np.ndarray:
    """
    At entry bar i: compare prev bar's high/low against SMA-w of highs/lows
    ending at bar i-1 (causal — no lookahead).
      sig=1  → LONG   (prev high below average → price in a low area)
      sig=-1 → SHORT  (prev low above average  → price in a high area)
      sig=0  → no signal, fall back to alternating
    """
    # rolling(w).mean() at i-1 = mean(hi[i-w:i]) — 10 bars ending at i-1
    sma_hi = pd.Series(hi).rolling(w, min_periods=w).mean().shift(1).values
    sma_lo = pd.Series(lo).rolling(w, min_periods=w).mean().shift(1).values
    prev_hi = np.concatenate([[np.nan], hi[:-1]])
    prev_lo = np.concatenate([[np.nan], lo[:-1]])

    sig = np.zeros(len(hi), dtype=np.int8)
    valid = ~np.isnan(sma_hi)
    sig[valid & (prev_hi < sma_hi)] = 1
    sig[valid & (sig == 0) & (prev_lo > sma_lo)] = -1
    return sig


def make_h1_signal(df_m5: pd.DataFrame, w: int = SMA_W) -> np.ndarray:
    """
    Resample M5 to H1 via timestamps (handles weekend/holiday gaps).
    At each M5 bar: use the signal of the current H1 bucket, which is derived
    from the PREVIOUS completed H1 bar vs SMA-w of H1 highs/lows.
    """
    df_h1 = (df_m5.set_index('timestamp')[['high', 'low']]
             .resample('1h').agg({'high': 'max', 'low': 'min'})
             .dropna())

    sma_hi = df_h1['high'].rolling(w, min_periods=w).mean().shift(1)
    sma_lo = df_h1['low'].rolling(w, min_periods=w).mean().shift(1)
    prev_hi = df_h1['high'].shift(1)
    prev_lo = df_h1['low'].shift(1)

    h1_sig = pd.Series(0, index=df_h1.index, dtype='int8')
    valid = sma_hi.notna() & prev_hi.notna()
    h1_sig[valid & (prev_hi < sma_hi)] = 1
    h1_sig[valid & (h1_sig == 0) & (prev_lo > sma_lo)] = -1

    # Map each M5 bar to its H1 bucket (floor to hour)
    m5_h1_ts = df_m5['timestamp'].dt.floor('1h')
    sig_series = m5_h1_ts.map(h1_sig).fillna(0).astype('int8')
    return sig_series.values


# ── Numba simulation kernel ───────────────────────────────────────────────────

@njit
def sim_zr_signal(op, hi, lo, cl, sp_arr, sig,
                  pip, pf, zw, tgt, body_thresh,
                  min_lock, trail_base, trail_max_pnl,
                  gate, ml):
    """
    ZR trail-lock sim with optional entry signal.
    sig[i]: 1=force LONG, -1=force SHORT, 0=use alternating.
    Identical mechanics to backtest_zr_trail_lock.py except direction source.
    """
    n = len(cl)
    pnl_out   = np.zeros(n, dtype=np.float64)
    nlegs_out = np.zeros(n, dtype=np.int32)
    nc = 0

    lv = np.zeros(ml, dtype=np.float64)
    ld = np.zeros(ml, dtype=np.float64)
    lp = np.zeros(ml, dtype=np.float64)

    i = 0
    d = 1  # alternating baseline direction

    while i < n:
        sp_e = sp_arr[i]
        if gate > 0.0 and sp_e > gate:
            i += 1; continue

        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i

        s = sig[i]
        d_use = s if s != 0 else d

        if body_thresh > 0.0 and rng_i > 1e-10:
            adv = (op_i - cl_i) if (d_use == 1 and op_i > cl_i) else \
                  ((cl_i - op_i) if (d_use == -1 and cl_i > op_i) else 0.0)
            if adv / rng_i > body_thresh:
                i += 1; continue

        e = cl_i
        if d_use == 1:
            uz = e;             lz = e - zw * pip
            ut = e + tgt * pip; lt = lz - tgt * pip
        else:
            lz = e;             uz = e + zw * pip
            lt = e - tgt * pip; ut = uz + tgt * pip

        lv[0] = 1.0; ld[0] = float(d_use); lp[0] = e
        nl = 1; last_zone = 0
        locked = False; peak_pnl = 0.0; stop_pnl = 0.0
        ex = False; i += 1

        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = c >= op[i]

            agg = 0.0; tv = 0.0
            for k in range(nl):
                agg += lv[k] * ld[k] * (c - lp[k]) / pip
                tv  += lv[k]
            agg -= tv * sp

            if not locked and agg >= min_lock:
                locked = True; peak_pnl = agg; stop_pnl = min_lock

            if locked:
                if agg > peak_pnl:
                    peak_pnl = agg
                    span = trail_max_pnl - min_lock
                    if peak_pnl >= trail_max_pnl or span < 1e-9:
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
                    pnl_out[nc] = agg; nlegs_out[nc] = nl; nc += 1
                    ex = True; break

            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))

                if is_hi and h >= uz and last_zone != 1:
                    last_zone = 1
                    net_vol = 0.0
                    for k in range(nl): net_vol += lv[k] * ld[k]
                    if net_vol < 0.0:
                        if nl >= ml:
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            pnl_out[nc] = net2 - tv2 * sp; nlegs_out[nc] = nl
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

                if (not is_hi) and l <= lz and last_zone != -1:
                    last_zone = -1
                    net_vol = 0.0
                    for k in range(nl): net_vol += lv[k] * ld[k]
                    if net_vol > 0.0:
                        if nl >= ml:
                            net2 = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                                tv2  += lv[k]
                            pnl_out[nc] = net2 - tv2 * sp; nlegs_out[nc] = nl
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

        d = -d

    return pnl_out[:nc], nlegs_out[:nc], nc


# ── Bootstrap helper ──────────────────────────────────────────────────────────

def boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


# ── Validation run for one variant ───────────────────────────────────────────

def validate(label, sig, op, hi, lo, cl, sp,
             nb, is_end, is_csz, oos_csz, oos_days,
             gate, rng):
    def run(s, e2):
        return sim_zr_signal(
            op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2], sp[s:e2], sig[s:e2],
            PIP, PF, ZW, TGT, BODY_THRESH, MIN_LOCK, TRAIL_BASE, TRAIL_MAX, gate, ML
        )

    # IS WF
    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
        p, _, nc = run(s_, e_)
        if nc > 0 and p.sum() > 0:
            is_wf += 1

    # OOS full
    pnl_oos, legs_oos, nc_oos = run(is_end, nb)
    ppd = pnl_oos.sum() / oos_days if nc_oos > 0 else 0.0

    # OOS WF chunks
    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
        p, _, nc = run(s_, e_)
        if nc > 0 and p.sum() > 0:
            oos_wf += 1

    p5 = p_pos = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_oos > 0:
        p5, p_pos = boot_stats(pnl_oos, oos_days, N_BOOT, rng)

    passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0
              and not math.isnan(p_pos) and p_pos > 0.95)

    # Signal coverage (% of OOS entries driven by signal, not fallback)
    sig_oos = sig[is_end:nb]
    sig_pct = float(np.mean(sig_oos != 0)) * 100

    # Cycle stats
    cpd = nc_oos / oos_days
    med_pnl = float(np.median(pnl_oos)) if nc_oos > 0 else float('nan')
    win_rate = float(np.mean(pnl_oos > 0)) * 100 if nc_oos > 0 else float('nan')

    return dict(label=label, ppd=round(ppd, 1), nc=nc_oos,
                cpd=round(cpd, 1), med=round(med_pnl, 1),
                win=round(win_rate, 1),
                is_wf=is_wf, oos_wf=oos_wf,
                p5=round(p5, 1) if not math.isnan(p5) else float('nan'),
                p_pos=round(p_pos, 4) if not math.isnan(p_pos) else float('nan'),
                sig_pct=round(sig_pct, 1), passed=passed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    mid = (pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    df = (mid.merge(ba[['ts_key', 'bid_c', 'ask_c']], on='ts_key', how='inner')
          .reset_index(drop=True))
    print(f"  {len(df):,} bars  {df.timestamp.min()} → {df.timestamp.max()}")

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
    print(f"  is_end={is_end:,}  oos_bars={oos_len:,}  oos_days={oos_days:.1f}  gate={gate:.2f}p\n")

    print("Computing signals...", flush=True)
    sig_base = np.zeros(nb, dtype=np.int8)
    sig_m5   = make_m5_signal(hi, lo, SMA_W)
    sig_h1   = make_h1_signal(df[['timestamp', 'high', 'low']], SMA_W)
    for name, sig in [('baseline', sig_base), ('m5_sma10', sig_m5), ('h1_sma10', sig_h1)]:
        oos_nz = np.sum(sig[is_end:] != 0)
        print(f"  {name}: {oos_nz/oos_len*100:.1f}% of OOS bars have signal")

    print("\nJIT warmup...", end=' ', flush=True)
    sim_zr_signal(op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
                  sig_base[:2000], PIP, PF, ZW, TGT, BODY_THRESH,
                  MIN_LOCK, TRAIL_BASE, TRAIL_MAX, gate, ML)
    print("done.\n")

    rng = np.random.default_rng(42)
    results = []
    for label, sig in [('baseline', sig_base), ('m5_sma10', sig_m5), ('h1_sma10', sig_h1)]:
        print(f"Validating {label}...", flush=True)
        r = validate(label, sig, op, hi, lo, cl, sp,
                     nb, is_end, is_csz, oos_csz, oos_days, gate, rng)
        results.append(r)
        tag = "🟢 PASS" if r['passed'] else ("🟡 near" if r['ppd'] > 0 and r['is_wf'] >= 2 and r['oos_wf'] >= 2 else "🔴")
        p5_s   = f"{r['p5']:6.1f}" if not math.isnan(r['p5']) else "   nan"
        ppos_s = f"{r['p_pos']:.3f}" if not math.isnan(r['p_pos']) else " nan"
        print(f"  {label:<12} p/d={r['ppd']:>7.1f}  c/d={r['cpd']:>5.1f}  "
              f"med={r['med']:>5.1f}p  win={r['win']:>4.1f}%  "
              f"IS={r['is_wf']}/{IS_CHUNKS} OOS={r['oos_wf']}/{OOS_CHUNKS}  "
              f"P5={p5_s}  P(+)={ppos_s}  sig={r['sig_pct']:>4.1f}%  {tag}")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUT_CSV, index=False)

    sep = "─" * 115
    print(f"\n{'═'*115}")
    print(f"  ZR Entry Signal Comparison — {PAIR}  ZW={ZW}p TGT={TGT}p PF={PF}  "
          f"min_lock={MIN_LOCK} trail_base={TRAIL_BASE} trail_max={TRAIL_MAX}")
    print(f"  OOS={oos_days:.0f} d  gate={gate:.2f}p  IS=3/3 OOS=3/3  boot={N_BOOT}")
    print(f"{'═'*115}")
    print(sep)
    hdr = (f"  {'variant':<12} | {'p/d':>8} {'c/d':>5} {'med':>6} {'win%':>5} | "
           f"{'IS':>2} {'OS':>2} | {'P5':>8} {'P(+)':>6} | {'sig%':>5} | result")
    print(hdr); print(sep)
    for r in results:
        tag = "🟢 PASS" if r['passed'] else ("🟡 near" if r['ppd'] > 0 and r['is_wf'] >= 2 and r['oos_wf'] >= 2 else "🔴")
        p5_s   = f"{r['p5']:8.1f}" if not math.isnan(r['p5']) else "     nan"
        ppos_s = f"{r['p_pos']:6.3f}" if not math.isnan(r['p_pos']) else "   nan"
        print(f"  {r['label']:<12} | {r['ppd']:>8.1f} {r['cpd']:>5.1f} {r['med']:>6.1f} {r['win']:>5.1f} | "
              f"{r['is_wf']:>2}/{IS_CHUNKS} {r['oos_wf']:>2}/{OOS_CHUNKS} | "
              f"{p5_s} {ppos_s} | {r['sig_pct']:>5.1f}% | {tag}")
    print(sep)
    print(f"\n  Results saved → {OUT_CSV}")
    print(f"  Done in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
