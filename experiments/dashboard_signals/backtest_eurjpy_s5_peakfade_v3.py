#!/usr/bin/env python3
"""
EUR_JPY S5 peak-fade v3 — TP/SL expressed as multiples of spread at entry.

Per user spec: "exit when position uPnL > 1× or 2× spread."

Mechanics
---------
On entry, capture spread_at_entry = ask_at_entry - bid_at_entry (in pips).
Then exits trigger on NET unrealized pnl crossing K × spread_at_entry:
  TP_K = 1 → exit when net uPnL ≥ 1 × spread_at_entry  (≈ 1 spread of profit)
  TP_K = 2 → exit when net uPnL ≥ 2 × spread_at_entry
SL similarly in spread units (SL_K ∈ {1, 2, 3}).

Also fixes the v1 spread double-counting bug (in v1 pnl had spread
subtracted from an already-net calculation).

Peak-detection modes from v2 preserved:
  decel    : enter when |s5_mom| drops below peak_frac × peak_magnitude
  signflip : enter when s5_mom flips sign vs the spike direction

Sweep
-----
spike_thr  ∈ {30, 50, 80, 120}    pips/min
m1_min     ∈ {0, 5, 10}           pips/min
peak_mode  ∈ {decel, signflip}
peak_frac  ∈ {0.3, 0.5, 0.7}      (decel only)
TP_K       ∈ {1, 2, 3}            spread multiples (NET)
SL_K       ∈ {1, 2, 3}            spread multiples (NET)
max_hold   ∈ {12, 60, 120}        S5 bars (1, 5, 10 min)
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
S5_DIR  = PROJECT / "data" / "s5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

PAIR = "EUR_JPY"
PIP  = 0.01
IS_FRAC = 0.70

SPIKE_THRS = [30.0, 50.0, 80.0, 120.0]
M1_MINS    = [0.0, 5.0, 10.0]
PEAK_MODES = [0, 1]                  # 0=decel, 1=signflip
PEAK_FRACS = [0.3, 0.5, 0.7]
TP_K_LIST  = [1.0, 2.0, 3.0]
SL_K_LIST  = [1.0, 2.0, 3.0]
MAX_HOLDS  = [12, 60, 120]


@njit(cache=True)
def sim_peak_fade_v3(close, bid, ask, sp, s5_mom, m1_mom, pip,
                     spike_thr, m1_min, peak_mode, peak_frac,
                     tp_K, sl_K, max_hold):
    """TP/SL in spread multiples (captured at entry). NET pnl semantics."""
    n = len(close)
    pnl_arr  = np.empty(n, dtype=np.float64)
    hold_arr = np.empty(n, dtype=np.int32)
    type_arr = np.empty(n, dtype=np.int8)
    dir_arr  = np.empty(n, dtype=np.int8)
    sp_entry_arr = np.empty(n, dtype=np.float64)
    count = 0
    state = 0
    spike_dir = 0
    spike_peak = 0.0
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0
    sp_entry = 0.0       # captured at trade open

    for i in range(13, n):
        if in_trade:
            if direction == 1:
                net_pnl = (bid[i] - entry_price) / pip
            else:
                net_pnl = (entry_price - ask[i]) / pip
            # TP/SL thresholds (in net pips) scale with entry spread
            tp_thr = tp_K * sp_entry
            sl_thr = sl_K * sp_entry
            if net_pnl >= tp_thr:
                pnl_arr[count]  = tp_thr
                hold_arr[count] = i - entry_bar
                type_arr[count] = 0
                dir_arr[count]  = direction
                sp_entry_arr[count] = sp_entry
                count += 1; in_trade = False
            elif net_pnl <= -sl_thr:
                pnl_arr[count]  = -sl_thr
                hold_arr[count] = i - entry_bar
                type_arr[count] = 1
                dir_arr[count]  = direction
                sp_entry_arr[count] = sp_entry
                count += 1; in_trade = False
            elif i - entry_bar >= max_hold:
                pnl_arr[count]  = net_pnl
                hold_arr[count] = i - entry_bar
                type_arr[count] = 2
                dir_arr[count]  = direction
                sp_entry_arr[count] = sp_entry
                count += 1; in_trade = False
            continue

        s5 = s5_mom[i]
        m1 = m1_mom[i]
        abs_s5 = abs(s5)

        if state == 0:
            if abs_s5 >= spike_thr:
                if (s5 > 0 and m1 >= m1_min) or (s5 < 0 and m1 <= -m1_min):
                    state = 1
                    spike_dir = 1 if s5 > 0 else -1
                    spike_peak = abs_s5
        elif state == 1:
            same_dir = (spike_dir == 1 and s5 > 0) or (spike_dir == -1 and s5 < 0)
            entered = False
            if peak_mode == 0:
                if not same_dir:
                    state = 0
                    continue
                if abs_s5 > spike_peak:
                    spike_peak = abs_s5
                    continue
                if abs_s5 <= spike_peak * peak_frac:
                    entered = True
            else:
                # signflip
                if not same_dir:
                    entered = True
            if entered:
                entry_dir = -spike_dir
                entry_price = ask[i] if entry_dir == 1 else bid[i]
                entry_bar = i
                direction = entry_dir
                sp_entry = sp[i]
                in_trade = True
                state = 0

    return pnl_arr[:count], hold_arr[:count], type_arr[:count], dir_arr[:count], sp_entry_arr[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 5000
    c = np.linspace(180.0, 180.5, n)
    b = c - 0.005; a = c + 0.005
    sp = np.ones(n)
    s5 = np.zeros(n); m1 = np.zeros(n)
    s5[1000] = 100.0; m1[1000] = 30.0
    sim_peak_fade_v3(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, 0, 0.5, 1.0, 2.0, 12)
    sim_peak_fade_v3(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, 1, 0.5, 1.0, 2.0, 12)


def main():
    warmup_jit()
    print(f"EUR_JPY S5 peak-fade v3 (TP/SL in spread multiples; NET pnl)")
    print(f"  spike thr : {SPIKE_THRS} pips/min")
    print(f"  m1 min    : {M1_MINS}")
    print(f"  peak mode : 0=decel, 1=signflip")
    print(f"  peak frac : {PEAK_FRACS}  (decel only)")
    print(f"  TP K      : {TP_K_LIST} × spread_at_entry (NET profit)")
    print(f"  SL K      : {SL_K_LIST} × spread_at_entry (NET loss)")
    print(f"  max hold  : {MAX_HOLDS} S5 bars")

    print(f"\nLoading {PAIR} S5 BA…")
    t0 = time.time()
    df = (pd.read_parquet(S5_DIR / f"{PAIR}_S5_BA.parquet")
            .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    print(f"  {len(df):,} S5 rows  ({time.time()-t0:.1f}s)")

    close = df["close"].values.astype(np.float64)
    bid   = df["bid_c"].values.astype(np.float64)
    ask   = df["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / PIP).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 17280.0

    s5_mom = np.empty(n_total, dtype=np.float64)
    s5_mom[0] = 0.0
    s5_mom[1:] = (close[1:] - close[:-1]) / PIP * 12.0
    m1_mom = np.empty(n_total, dtype=np.float64)
    m1_mom[:12] = 0.0
    m1_mom[12:] = (close[12:] - close[:-12]) / PIP

    print(f"  mean spread (OOS): {sp[n_is:].mean():.2f} pips")
    print(f"  median spread:     {np.median(sp[n_is:]):.2f}")
    print(f"  s5_mom 99.0%ile:   {np.percentile(np.abs(s5_mom), 99.0):.1f} pips/min")

    close_o = close[n_is:]
    bid_o   = bid[n_is:]
    ask_o   = ask[n_is:]
    sp_o    = sp[n_is:]
    s5_o    = s5_mom[n_is:]
    m1_o    = m1_mom[n_is:]

    print(f"\nOOS bars: {len(close_o):,}  (~{oos_days:.0f} days)")

    rows = []
    t0 = time.time()
    last_report = time.time()
    config_idx = 0

    for spike in SPIKE_THRS:
        for m1m in M1_MINS:
            for pm in PEAK_MODES:
                pfs = PEAK_FRACS if pm == 0 else [0.5]
                for pf in pfs:
                    for tp in TP_K_LIST:
                        for sl in SL_K_LIST:
                            for mh in MAX_HOLDS:
                                config_idx += 1
                                p, h, t, d, sp_e = sim_peak_fade_v3(
                                    close_o, bid_o, ask_o, sp_o,
                                    s5_o, m1_o, PIP,
                                    float(spike), float(m1m),
                                    int(pm), float(pf),
                                    float(tp), float(sl), int(mh))
                                n = len(p)
                                if n == 0:
                                    rows.append(dict(
                                        spike=spike, m1_min=m1m, peak_mode=pm,
                                        peak_frac=pf, tp_K=tp, sl_K=sl, max_hold=mh,
                                        n=0, sum_pips=0.0, ppd=0.0,
                                        wr=0.0, tp_pct=0.0, sl_pct=0.0,
                                        time_pct=0.0, mean_pnl=0.0, mdd=0.0,
                                        mean_sp_entry=0.0))
                                    continue
                                wr = (p > 0).sum() / n * 100
                                ppd = p.sum() / oos_days
                                mdd = max_dd(p)
                                tp_pct = (t == 0).sum() / n * 100
                                sl_pct = (t == 1).sum() / n * 100
                                tm_pct = (t == 2).sum() / n * 100
                                rows.append(dict(
                                    spike=spike, m1_min=m1m, peak_mode=pm,
                                    peak_frac=pf, tp_K=tp, sl_K=sl, max_hold=mh,
                                    n=n, sum_pips=round(float(p.sum()), 1),
                                    ppd=round(ppd, 2), wr=round(wr, 1),
                                    tp_pct=round(tp_pct, 1),
                                    sl_pct=round(sl_pct, 1),
                                    time_pct=round(tm_pct, 1),
                                    mean_pnl=round(float(p.mean()), 2),
                                    mdd=round(mdd, 1),
                                    mean_sp_entry=round(float(sp_e.mean()), 2),
                                ))
                                if time.time() - last_report > 5:
                                    print(f"  {config_idx} configs done", flush=True)
                                    last_report = time.time()

    df_res = pd.DataFrame(rows)
    out_csv = OUT / "eurjpy_s5_peakfade_v3.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n{len(df_res)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    df_res = df_res.sort_values("sum_pips", ascending=False)
    print("\n" + "="*125)
    print(f"  Top 20 configs by Σ pips (OOS, {oos_days:.0f} days)")
    print("="*125)
    print(f"  {'spike':>5} {'m1':>3} {'pm':>3} {'pf':>4} "
          f"{'TPx':>4} {'SLx':>4} {'hold':>5}  {'n':>5} {'Σ pips':>9} "
          f"{'ppd':>8} {'WR%':>5} {'TP%':>5} {'SL%':>5} {'tm%':>5} "
          f"{'mean':>7} {'sp_en':>6} {'MDD':>7}")
    for _, r in df_res.head(20).iterrows():
        pm_lbl = "DEC" if r['peak_mode'] == 0 else "SGN"
        print(f"  {r['spike']:>5.0f} {r['m1_min']:>3.0f} {pm_lbl:>3} "
              f"{r['peak_frac']:>4.1f} {r['tp_K']:>4.1f} {r['sl_K']:>4.1f} "
              f"{int(r['max_hold']):>5d}  {int(r['n']):>5d} "
              f"{r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
              f"{r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['time_pct']:>5.1f} "
              f"{r['mean_pnl']:>+7.2f} {r['mean_sp_entry']:>6.2f} {r['mdd']:>7.1f}")

    print(f"\n  positive configs: {(df_res.sum_pips > 0).sum()} / {len(df_res)}")

    print("\n" + "="*125)
    print(f"  Top 15 by ppd among configs with n >= 200")
    print("="*125)
    filt = df_res[df_res.n >= 200].sort_values("ppd", ascending=False)
    if len(filt):
        print(f"  {'spike':>5} {'m1':>3} {'pm':>3} {'pf':>4} "
              f"{'TPx':>4} {'SLx':>4} {'hold':>5}  {'n':>5} {'Σ pips':>9} "
              f"{'ppd':>8} {'WR%':>5} {'mean':>7}")
        for _, r in filt.head(15).iterrows():
            pm_lbl = "DEC" if r['peak_mode'] == 0 else "SGN"
            print(f"  {r['spike']:>5.0f} {r['m1_min']:>3.0f} {pm_lbl:>3} "
                  f"{r['peak_frac']:>4.1f} {r['tp_K']:>4.1f} {r['sl_K']:>4.1f} "
                  f"{int(r['max_hold']):>5d}  {int(r['n']):>5d} "
                  f"{r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
                  f"{r['mean_pnl']:>+7.2f}")
    else:
        print("  (no configs with n>=200)")


if __name__ == "__main__":
    main()
