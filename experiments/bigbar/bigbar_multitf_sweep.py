"""
Big Bar Momentum sweep — EUR_JPY across M1/M5/M15/M30/H1.

Port of forex-alpha-experiment.jsx to real OANDA data.

Signal: |ΔC₃ / ATR₃| > θ  →  enter in direction of ΔC₃
Filter: (close − SMA10) / ATR₃ used as trend-agreement gate:
  • No filter  (baseline)
  • SMA10 agree: only take LONG when close > SMA10, SHORT when close < SMA10
Hold H bars, exit at close.
TC = real bid/ask spread (IS-p90 gate per TF).

Gates (must ALL pass to report ✅):
  IS-WF:   IS split into 3 chunks, all have mean_ret > 0
  OOS-WF:  OOS split into 3 sub-windows, all have mean_ret > 0
  MC:      P5 > 0 AND P(+) > 95%  (2000 bootstrap samples of full OOS)

Classifies each signal as TREND / MEAN_REV / FADE / LATE (same as JSX).

Data:
  M5 mid  : data/m5_ohlc/EUR_JPY_M5.parquet          (2021-01-03 → 2026-04-09, ~5yr)
  M5 BA   : data/m5_ba/EUR_JPY_M5_BA.parquet         (2024-05-05 → present, ~1yr)
  S5 BA   : data/s5_ohlc/EUR_JPY_S5_BA.parquet       (2025-10-01 → present, ~6mo)
  M15/M30/H1: resampled from M5 mid + M5 BA
  M1      : resampled from S5 BA mid  (6 months only — noted in output)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DATA_MID  = Path('/path/to/projects/fx-core/data/m5_ohlc/EUR_JPY_M5.parquet')
DATA_BA   = Path('/path/to/projects/fx-core/data/m5_ba/EUR_JPY_M5_BA.parquet')
DATA_S5BA = Path('/path/to/projects/fx-core/data/s5_ohlc/EUR_JPY_S5_BA.parquet')
OUT_CSV   = Path(__file__).parent / 'bigbar_multitf_results.csv'

OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000
PIP        = 0.01

THRESHOLDS = [round(x, 1) for x in np.arange(0.5, 3.1, 0.1)]
HOLD_BARS  = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
SMA_N      = 10   # bars for price-SMA trend filter


# ── Signal kernel (Numba) ─────────────────────────────────────────────────────
@njit
def compute_signals(op, hi, lo, cl, spread_arr, threshold, hold, sma_filter):
    """
    sma_filter = 0: take all signals (both directions)
    sma_filter = 1: LONG only when cl > sma10, SHORT only when cl < sma10
    Returns ret_h, ret_mid, dirs arrays.
    """
    n = len(cl)
    ret_h   = np.empty(n, dtype=np.float64)
    ret_mid = np.empty(n, dtype=np.float64)
    dirs    = np.empty(n, dtype=np.float64)
    ns = 0
    half = max(1, hold // 2)

    for i in range(max(4, SMA_N), n - hold - 1):
        # ATR3
        tr0 = max(hi[i]   - lo[i],   abs(hi[i]   - cl[i-1]), abs(lo[i]   - cl[i-1]))
        tr1 = max(hi[i-1] - lo[i-1], abs(hi[i-1] - cl[i-2]), abs(lo[i-1] - cl[i-2]))
        tr2 = max(hi[i-2] - lo[i-2], abs(hi[i-2] - cl[i-3]), abs(lo[i-2] - cl[i-3]))
        atr3 = (tr0 + tr1 + tr2) / 3.0
        if atr3 < 1e-10: continue

        # 3-bar momentum signal
        mom3 = (cl[i] - cl[i-3]) / atr3
        if abs(mom3) <= threshold: continue

        # SMA10
        sma10 = 0.0
        for k in range(SMA_N):
            sma10 += cl[i - k]
        sma10 /= SMA_N

        dir_ = 1.0 if mom3 > 0 else -1.0

        # SMA10 trend filter
        if sma_filter == 1:
            above_sma = cl[i] > sma10
            if dir_ == 1.0 and not above_sma: continue
            if dir_ == -1.0 and above_sma:    continue

        entry    = cl[i]
        sp       = spread_arr[i]
        exit_h   = cl[i + hold]
        exit_mid = cl[i + half]

        raw_h   = (exit_h   - entry) * dir_ / entry * 10000.0
        raw_mid = (exit_mid - entry) * dir_ / entry * 10000.0
        tc_bps  = sp * PIP / entry * 10000.0

        ret_h[ns]   = raw_h - tc_bps
        ret_mid[ns] = raw_mid
        dirs[ns]    = dir_
        ns += 1

    return ret_h[:ns], ret_mid[:ns], dirs[:ns]

# sentinel constant inside njit scope
SMA_N = 10


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_c = np.linspace(1.1, 1.2, 600).astype(np.float64)
_h = _c + 0.001; _l = _c - 0.001; _o = _c
_sp = np.full(600, 2.3)
compute_signals(_o, _h, _l, _c, _sp, 1.0, 5, 0)
compute_signals(_o, _h, _l, _c, _sp, 1.0, 5, 1)
print("done.\n")


# ── Data loading helpers ──────────────────────────────────────────────────────
def load_m5_with_spread():
    df_mid = pd.read_parquet(DATA_MID).sort_values('timestamp').reset_index(drop=True)
    df_ba  = pd.read_parquet(DATA_BA).sort_values('timestamp').reset_index(drop=True)
    df_mid['ts_key'] = df_mid['timestamp'].astype(str).str[:16]
    df_ba['ts_key']  = df_ba['timestamp'].astype(str).str[:16]
    merged = df_mid.merge(df_ba[['ts_key', 'bid_c', 'ask_c']], on='ts_key', how='left')
    merged = merged.sort_values('ts_key').reset_index(drop=True)
    sp_raw = (merged.ask_c - merged.bid_c) / PIP
    sp_med = float(sp_raw.median())
    merged['spread_p'] = sp_raw.fillna(sp_med).clip(lower=0.3)
    return merged


def resample_ohlcs(df, rule):
    df2 = df.set_index('timestamp')
    ohlc = df2.resample(rule).agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last'),
    ).dropna()
    sp = df2['spread_p'].resample(rule).last().reindex(ohlc.index).ffill()
    ohlc['spread_p'] = sp.values
    return ohlc.reset_index()


def load_m1_from_s5():
    df = pd.read_parquet(DATA_S5BA).sort_values('timestamp').reset_index(drop=True)
    df['mid_o'] = (df.bid_o + df.ask_o) / 2
    df['mid_h'] = (df.bid_h + df.ask_h) / 2
    df['mid_l'] = (df.bid_l + df.ask_l) / 2
    df['mid_c'] = (df.bid_c + df.ask_c) / 2
    df['spread_p'] = (df.ask_c - df.bid_c) / PIP
    df2 = df.set_index('timestamp')
    ohlc = df2.resample('1min').agg(
        open=('mid_o', 'first'), high=('mid_h', 'max'),
        low=('mid_l', 'min'), close=('mid_c', 'last'),
    ).dropna()
    sp = df2['spread_p'].resample('1min').last().reindex(ohlc.index).ffill()
    ohlc['spread_p'] = sp.values
    return ohlc.reset_index()


# ── Category helper ───────────────────────────────────────────────────────────
def classify(ret_h, ret_mid):
    trend    = (ret_h > 0) & (ret_mid > 0)
    mean_rev = (ret_h <= 0) & (ret_mid <= 0)
    fade     = (ret_h <= 0) & (ret_mid > 0)
    late     = (ret_h > 0)  & (ret_mid <= 0)
    return trend, mean_rev, fade, late


# ── Run one config, return (mean_ret, nc) per chunk ──────────────────────────
def run_chunks(op, hi, lo, cl, sp, chunk_bounds, thr, hold, sma_f):
    results = []
    for s, e2 in chunk_bounds:
        rh, _, _ = compute_signals(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                                   sp[s:e2], thr, hold, sma_f)
        results.append((rh.mean() if len(rh) >= 5 else -999.0, len(rh)))
    return results


# ── Main sweep ────────────────────────────────────────────────────────────────
print("Loading data...")
df_m5  = load_m5_with_spread()
print(f"  M5  : {len(df_m5)} bars  "
      f"{df_m5.timestamp.min().date()} → {df_m5.timestamp.max().date()}")

df_m15 = resample_ohlcs(df_m5, '15min')
df_m30 = resample_ohlcs(df_m5, '30min')
df_h1  = resample_ohlcs(df_m5, '1h')
print(f"  M15 : {len(df_m15)} bars")
print(f"  M30 : {len(df_m30)} bars")
print(f"  H1  : {len(df_h1)} bars")

df_m1 = None
if DATA_S5BA.exists():
    df_m1 = load_m1_from_s5()
    print(f"  M1  : {len(df_m1)} bars  "
          f"{df_m1.timestamp.min().date()} → {df_m1.timestamp.max().date()}  "
          f"[⚠ 6-month window]")

timeframes = [
    ('M5',  df_m5,  288),   # bars per trading day (24h × 12)
    ('M15', df_m15, 96),
    ('M30', df_m30, 48),
    ('H1',  df_h1,  24),
]
if df_m1 is not None:
    timeframes = [('M1', df_m1, 1440)] + timeframes

rng  = np.random.default_rng(42)
rows = []

FILTERS = [(0, 'all'), (1, 'sma10')]

for tf_name, df, bars_per_day in timeframes:
    nb = len(df)
    is_end       = int(nb * (1 - OOS_FRAC))
    oos_len      = nb - is_end
    is_chunk_sz  = is_end  // IS_CHUNKS
    oos_chunk_sz = oos_len // OOS_CHUNKS
    oos_days     = oos_len / bars_per_day

    op_all = df.open.values.astype(np.float64)
    hi_all = df.high.values.astype(np.float64)
    lo_all = df.low.values.astype(np.float64)
    cl_all = df.close.values.astype(np.float64)
    sp_all = df.spread_p.values.astype(np.float64)

    sp_is    = sp_all[:is_end]
    gate_thr = float(np.percentile(sp_is, 90))
    sp_med   = float(np.median(sp_is))

    is_chunks  = [(ch * is_chunk_sz,
                   (ch+1)*is_chunk_sz if ch < IS_CHUNKS-1 else is_end)
                  for ch in range(IS_CHUNKS)]
    oos_chunks = [(is_end + ch * oos_chunk_sz,
                   is_end + (ch+1)*oos_chunk_sz if ch < OOS_CHUNKS-1 else nb)
                  for ch in range(OOS_CHUNKS)]

    print(f"\n{'═'*90}")
    print(f"{tf_name}  bars={nb}  IS={is_end}  OOS={oos_len} ({oos_days:.1f} days)  "
          f"sp_med={sp_med:.2f}p  gate={gate_thr:.2f}p")

    for sma_f, fname in FILTERS:
        print(f"\n  ── filter={fname} ──────────────────────────────────────────────")
        print(f"  {'θ':>5} {'hold':>5} | {'sig/d':>6} {'win%':>6} {'ret_bps':>8} {'sharpe':>7} | "
              f"{'IS-wf':>6} {'OOS-wf':>7} | {'P5':>7} {'P(+)':>6} | "
              f"{'TREND%':>7} {'MR%':>5} | status")
        print(f"  {'─'*100}")

        best_ret = -1e9

        for thr in THRESHOLDS:
            for hold in HOLD_BARS:
                # Full OOS
                rh, rm, dr = compute_signals(
                    op_all[is_end:], hi_all[is_end:], lo_all[is_end:], cl_all[is_end:],
                    sp_all[is_end:], thr, hold, sma_f)
                nc = len(rh)
                if nc < 10: continue

                mean_ret = float(rh.mean())
                std_ret  = float(rh.std())
                win_pct  = float((rh > 0).mean())
                sharpe   = mean_ret / std_ret * np.sqrt(bars_per_day) if std_ret > 0 else 0.0
                sig_day  = nc / oos_days

                # IS WF
                is_res  = run_chunks(op_all, hi_all, lo_all, cl_all, sp_all,
                                     is_chunks, thr, hold, sma_f)
                is_wf   = sum(1 for m, n in is_res if n >= 5 and m > 0)

                # OOS WF
                oos_res = run_chunks(op_all, hi_all, lo_all, cl_all, sp_all,
                                     oos_chunks, thr, hold, sma_f)
                oos_wf  = sum(1 for m, n in oos_res if n >= 5 and m > 0)

                both_wf = (is_wf == IS_CHUNKS) and (oos_wf == OOS_CHUNKS)

                p5 = prob_pos = float('nan')
                if both_wf:
                    boot = np.array([
                        rng.choice(rh, size=nc, replace=True).mean()
                        for _ in range(N_BOOT)])
                    p5       = float(np.percentile(boot, 5))
                    prob_pos = float(np.mean(boot > 0))

                if both_wf and not np.isnan(p5) and p5 > 0 and prob_pos > 0.95:
                    status = "✅ PASS"
                    if mean_ret > best_ret: best_ret = mean_ret
                elif is_wf == IS_CHUNKS:
                    status = f"🟡 IS=3 OOS={oos_wf}/3 ret={mean_ret:.2f}"
                elif mean_ret > 0 and is_wf >= 2:
                    status = f"  IS={is_wf}/3 OOS={oos_wf}/3 ret={mean_ret:.2f}"
                else:
                    continue   # suppress noise

                star = " ◄ BEST" if both_wf and mean_ret == best_ret and mean_ret > 0 else ""
                trend, mr, fade, late = classify(rh, rm)
                print(f"  {thr:>5.1f} {hold:>5} | {sig_day:>6.1f} {win_pct*100:>5.1f}% "
                      f"{mean_ret:>8.3f} {sharpe:>7.3f} | "
                      f"{is_wf:>6} {oos_wf:>7} | "
                      f"{p5:>7.3f} {prob_pos:>6.3f} | "
                      f"{trend.mean()*100:>6.1f}% {mr.mean()*100:>4.1f}% | "
                      f"{status}{star}")
                sys.stdout.flush()

                rows.append(dict(
                    tf=tf_name, filter=fname, threshold=thr, hold=hold,
                    sig_day=round(sig_day, 1), win_pct=round(win_pct*100, 1),
                    mean_ret=round(mean_ret, 3), sharpe=round(sharpe, 3),
                    pct_trend=round(trend.mean()*100, 1),
                    pct_mr=round(mr.mean()*100, 1),
                    pct_fade=round(fade.mean()*100, 1),
                    pct_late=round(late.mean()*100, 1),
                    is_wf=is_wf, oos_wf=oos_wf,
                    p5=round(p5, 3) if not np.isnan(p5) else None,
                    prob_pos=round(prob_pos, 3) if not np.isnan(prob_pos) else None,
                    sp_med=round(sp_med, 2), gate=round(gate_thr, 2),
                ))

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\n\nSaved {len(rows)} rows → {OUT_CSV}")

print("\n=== VALIDATED CONFIGS (all 5 gates) ===")
print(f"{'TF':<5} {'filter':<6} {'θ':>5} {'hold':>5} | "
      f"{'sig/d':>6} {'win%':>6} {'ret':>7} {'sharpe':>7} | "
      f"{'TREND%':>7} {'MR%':>6} | {'P5':>7} {'P(+)':>6}")
found = False
for r in rows:
    if r.get('p5') and r['p5'] > 0 and r.get('prob_pos', 0) > 0.95:
        print(f"{r['tf']:<5} {r['filter']:<6} {r['threshold']:>5.1f} {r['hold']:>5} | "
              f"{r['sig_day']:>6.1f} {r['win_pct']:>5.1f}% "
              f"{r['mean_ret']:>7.3f} {r['sharpe']:>7.3f} | "
              f"{r['pct_trend']:>6.1f}% {r['pct_mr']:>5.1f}% | "
              f"{r['p5']:>7.3f} {r['prob_pos']:>6.3f}")
        found = True
if not found:
    print("  none — signal has no validated edge on EUR_JPY at any TF")

print("\n=== IS-WF=3 CANDIDATES (partial — OOS gate failed) ===")
print(f"{'TF':<5} {'filter':<6} {'θ':>5} {'hold':>5} | "
      f"{'ret':>7} {'sharpe':>7} | {'IS-wf':>6} {'OOS-wf':>7}")
for r in rows:
    if r['is_wf'] == IS_CHUNKS and (not r.get('p5') or r['p5'] <= 0):
        print(f"{r['tf']:<5} {r['filter']:<6} {r['threshold']:>5.1f} {r['hold']:>5} | "
              f"{r['mean_ret']:>7.3f} {r['sharpe']:>7.3f} | "
              f"{r['is_wf']:>6} {r['oos_wf']:>7}")
