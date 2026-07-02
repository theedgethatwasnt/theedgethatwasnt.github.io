#!/usr/bin/env python3
"""
EUR_JPY S5 peak-fade v4.

v1/v2/v3 were all 0/N positive with fade direction. The 41% WR result
suggests the trade direction might be backwards — moves after spikes
CONTINUE more often than they revert. v4 tests this and slower peak
detection.

New axes in v4:
  direction_mode ∈ {fade, continuation}
    fade         : enter OPPOSITE the spike direction (the v3 behaviour)
    continuation : enter SAME direction as spike (trend continuation)
  peak_tf        ∈ {s5, m1}
    s5 : detect peak on S5 momentum (v3 behaviour)
    m1 : detect peak on M1 momentum (slower, less noise)

Same sweep as v3 for the rest: spike_thr / m1_min / peak_frac /
TP_K / SL_K / max_hold. TP/SL in spread multiples (NET pnl semantics).

Sweep size: 1296 × 2 (direction) × 2 (peak_tf) = 5184 configs.
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
DIRECTIONS = [0, 1]                  # 0=fade, 1=continuation
PEAK_TFS   = [0, 1]                  # 0=s5, 1=m1


@njit(cache=True)
def sim_v4(close, bid, ask, sp, s5_mom, m1_mom, pip,
           spike_thr, m1_min, peak_mode, peak_frac,
           tp_K, sl_K, max_hold, direction_mode, peak_tf):
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
    sp_entry = 0.0

    for i in range(13, n):
        if in_trade:
            if direction == 1:
                net_pnl = (bid[i] - entry_price) / pip
            else:
                net_pnl = (entry_price - ask[i]) / pip
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
            # IDLE — find spike on S5
            if abs_s5 >= spike_thr:
                if (s5 > 0 and m1 >= m1_min) or (s5 < 0 and m1 <= -m1_min):
                    state = 1
                    spike_dir = 1 if s5 > 0 else -1
                    # Initialize peak tracker on the chosen TF
                    if peak_tf == 0:
                        spike_peak = abs_s5
                    else:
                        spike_peak = abs(m1)
        elif state == 1:
            # SPIKING — detect peak on chosen TF
            if peak_tf == 0:
                mom_val = s5
                mom_mag = abs_s5
            else:
                mom_val = m1
                mom_mag = abs(m1)

            same_dir = (spike_dir == 1 and mom_val > 0) or (spike_dir == -1 and mom_val < 0)
            entered = False
            if peak_mode == 0:
                if not same_dir:
                    state = 0
                    continue
                if mom_mag > spike_peak:
                    spike_peak = mom_mag
                    continue
                if mom_mag <= spike_peak * peak_frac:
                    entered = True
            else:
                if not same_dir:
                    entered = True
            if entered:
                # direction_mode: 0=fade, 1=continuation
                entry_dir = -spike_dir if direction_mode == 0 else spike_dir
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
    for dm in (0, 1):
        for tf in (0, 1):
            for pm in (0, 1):
                sim_v4(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, pm, 0.5, 1.0, 2.0, 12, dm, tf)


def main():
    warmup_jit()
    print(f"EUR_JPY S5 peak-fade v4 (direction flip + multi-TF peak detection)")
    print(f"  direction : 0=fade, 1=continuation")
    print(f"  peak_tf   : 0=s5, 1=m1")
    print(f"  TP/SL in spread multiples (NET pnl)")
    print(f"  full sweep: {len(SPIKE_THRS) * len(M1_MINS) * len(PEAK_MODES) * len(TP_K_LIST) * len(SL_K_LIST) * len(MAX_HOLDS) * len(DIRECTIONS) * len(PEAK_TFS)} (varies — peak_frac only for decel)")

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

    for dm in DIRECTIONS:
        for ptf in PEAK_TFS:
            for spike in SPIKE_THRS:
                for m1m in M1_MINS:
                    for pm in PEAK_MODES:
                        pfs = PEAK_FRACS if pm == 0 else [0.5]
                        for pf in pfs:
                            for tp in TP_K_LIST:
                                for sl in SL_K_LIST:
                                    for mh in MAX_HOLDS:
                                        config_idx += 1
                                        p, h, t, d, sp_e = sim_v4(
                                            close_o, bid_o, ask_o, sp_o,
                                            s5_o, m1_o, PIP,
                                            float(spike), float(m1m),
                                            int(pm), float(pf),
                                            float(tp), float(sl), int(mh),
                                            int(dm), int(ptf))
                                        n = len(p)
                                        if n == 0:
                                            rows.append(dict(
                                                direction=dm, peak_tf=ptf,
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
                                        rows.append(dict(
                                            direction=dm, peak_tf=ptf,
                                            spike=spike, m1_min=m1m, peak_mode=pm,
                                            peak_frac=pf, tp_K=tp, sl_K=sl, max_hold=mh,
                                            n=n, sum_pips=round(float(p.sum()), 1),
                                            ppd=round(ppd, 2), wr=round(wr, 1),
                                            tp_pct=round((t == 0).sum() / n * 100, 1),
                                            sl_pct=round((t == 1).sum() / n * 100, 1),
                                            time_pct=round((t == 2).sum() / n * 100, 1),
                                            mean_pnl=round(float(p.mean()), 2),
                                            mdd=round(mdd, 1),
                                            mean_sp_entry=round(float(sp_e.mean()), 2),
                                        ))
                                        if time.time() - last_report > 8:
                                            print(f"  {config_idx} configs done ({config_idx/(time.time()-t0):.0f}/s)", flush=True)
                                            last_report = time.time()

    df_res = pd.DataFrame(rows)
    out_csv = OUT / "eurjpy_s5_peakfade_v4.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n{len(df_res)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    df_res = df_res.sort_values("sum_pips", ascending=False)
    print(f"\n  positive configs: {(df_res.sum_pips > 0).sum()} / {len(df_res)}")

    # Headline split by direction + peak_tf
    print("\n" + "="*100)
    print("  Best Σ pips by direction × peak_tf")
    print("="*100)
    for dm in DIRECTIONS:
        for ptf in PEAK_TFS:
            sub = df_res[(df_res.direction == dm) & (df_res.peak_tf == ptf)]
            if not len(sub): continue
            n_pos = (sub.sum_pips > 0).sum()
            best = sub.iloc[0]
            dlbl = "FADE" if dm == 0 else "CONT"
            tflbl = "S5" if ptf == 0 else "M1"
            print(f"  dir={dlbl}  peak_tf={tflbl}  positive {n_pos}/{len(sub)}  "
                  f"best Σ={best['sum_pips']:+.1f}p (ppd={best['ppd']:+.2f}, n={int(best['n'])}, WR={best['wr']}%)")

    print("\n" + "="*125)
    print(f"  Top 25 by Σ pips overall")
    print("="*125)
    print(f"  {'dir':>4} {'tf':>3} {'spike':>5} {'m1':>3} {'pm':>3} {'pf':>4} "
          f"{'TPx':>4} {'SLx':>4} {'hold':>5}  {'n':>5} {'Σ pips':>9} "
          f"{'ppd':>8} {'WR%':>5} {'TP%':>5} {'SL%':>5} {'tm%':>5} {'mean':>7}")
    for _, r in df_res.head(25).iterrows():
        dlbl = "FADE" if r['direction'] == 0 else "CONT"
        tflbl = "S5" if r['peak_tf'] == 0 else "M1"
        pmlbl = "DEC" if r['peak_mode'] == 0 else "SGN"
        print(f"  {dlbl:>4} {tflbl:>3} {r['spike']:>5.0f} {r['m1_min']:>3.0f} "
              f"{pmlbl:>3} {r['peak_frac']:>4.1f} {r['tp_K']:>4.1f} {r['sl_K']:>4.1f} "
              f"{int(r['max_hold']):>5d}  {int(r['n']):>5d} "
              f"{r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
              f"{r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['time_pct']:>5.1f} "
              f"{r['mean_pnl']:>+7.2f}")


if __name__ == "__main__":
    main()
