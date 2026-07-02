#!/usr/bin/env python3
"""
EUR_JPY S5 peak-fade v2.

Fixes
-----
1. Spread double-counting bug from v1. In v1: pnl_arr[count] = tp_pips - sp[i].
   But (bid[t] - ask_entry)/pip already nets the spread (long sells at bid,
   bought at ask) — subtracting spread again under-counted by ~spread per
   trade. v2: pnl = tp_pips when bid - ask_entry >= tp_pips × pip.
   That means TP values in this sweep are NET targets (after spread).
2. Wider TP grid focused on user's clarified ask: 1-3 pips NET profit.
3. Added a second peak-detection mode: SIGN_FLIP (wait for s5_mom to cross
   zero, cleaner than single-bar deceleration).

Sweep
-----
spike_thr   ∈ {30, 50, 80, 120} pips/min  (S5 momentum spike threshold)
m1_min      ∈ {0, 5, 10} pips/min          (M1 continuation requirement)
peak_mode   ∈ {decel, signflip}             (how we detect peak)
peak_frac   ∈ {0.3, 0.5, 0.7}               (decel mode only — |s5|/|peak|)
TP_pips     ∈ {1, 2, 3, 5}                  (NET profit target after spread)
SL_pips     ∈ {2, 3, 5}                     (NET loss cap)
max_hold_S5 ∈ {12, 60, 120}                 (1min / 5min / 10min)
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

SPIKE_THRS   = [30.0, 50.0, 80.0, 120.0]
M1_MINS      = [0.0, 5.0, 10.0]
# 0 = deceleration, 1 = sign flip
PEAK_MODES   = [0, 1]
PEAK_FRACS   = [0.3, 0.5, 0.7]              # only matters when peak_mode == 0
TP_NETS      = [1.0, 2.0, 3.0, 5.0]
SL_NETS      = [2.0, 3.0, 5.0]
MAX_HOLDS    = [12, 60, 120]


@njit(cache=True)
def sim_peak_fade_v2(close, bid, ask, sp, s5_mom, m1_mom, pip,
                     spike_thr, m1_min, peak_mode, peak_frac,
                     tp_net, sl_net, max_hold):
    """v2 — NET pnl semantics (no double-count of spread).

    Long enters at ask_entry. Exits via:
      TP: bid[t] - ask_entry >= tp_net × pip  →  net pnl = tp_net
      SL: bid[t] - ask_entry <= -sl_net × pip →  net pnl = -sl_net
      Time: market sell at bid[t]            →  net pnl = (bid - ask_entry)/pip

    Short symmetric (entry at bid, exits via ask).

    peak_mode = 0 (decel)    : enter when |s5_mom| < peak_mag × peak_frac
    peak_mode = 1 (signflip) : enter when s5_mom changes sign vs spike_dir
    """
    n = len(close)
    pnl_arr  = np.empty(n, dtype=np.float64)
    hold_arr = np.empty(n, dtype=np.int32)
    type_arr = np.empty(n, dtype=np.int8)
    dir_arr  = np.empty(n, dtype=np.int8)
    count = 0
    state = 0
    spike_dir = 0
    spike_peak = 0.0
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0

    for i in range(13, n):
        if in_trade:
            if direction == 1:
                net_pnl = (bid[i] - entry_price) / pip
            else:
                net_pnl = (entry_price - ask[i]) / pip
            if net_pnl >= tp_net:
                pnl_arr[count]  = tp_net
                hold_arr[count] = i - entry_bar
                type_arr[count] = 0
                dir_arr[count]  = direction
                count += 1; in_trade = False
            elif net_pnl <= -sl_net:
                pnl_arr[count]  = -sl_net
                hold_arr[count] = i - entry_bar
                type_arr[count] = 1
                dir_arr[count]  = direction
                count += 1; in_trade = False
            elif i - entry_bar >= max_hold:
                pnl_arr[count]  = net_pnl
                hold_arr[count] = i - entry_bar
                type_arr[count] = 2
                dir_arr[count]  = direction
                count += 1; in_trade = False
            continue

        s5 = s5_mom[i]
        m1 = m1_mom[i]
        abs_s5 = abs(s5)

        if state == 0:
            # IDLE
            if abs_s5 >= spike_thr:
                if (s5 > 0 and m1 >= m1_min) or (s5 < 0 and m1 <= -m1_min):
                    state = 1
                    spike_dir = 1 if s5 > 0 else -1
                    spike_peak = abs_s5
        elif state == 1:
            # SPIKING — check for peak per the chosen mode
            same_dir = (spike_dir == 1 and s5 > 0) or (spike_dir == -1 and s5 < 0)
            if peak_mode == 0:
                # Deceleration peak detector
                if not same_dir:
                    state = 0   # sign flipped before clean decel; abort
                    continue
                if abs_s5 > spike_peak:
                    spike_peak = abs_s5
                    continue
                if abs_s5 <= spike_peak * peak_frac:
                    # PEAK FOUND
                    entry_dir = -spike_dir
                    entry_price = ask[i] if entry_dir == 1 else bid[i]
                    entry_bar = i
                    direction = entry_dir
                    in_trade = True
                    state = 0
            else:
                # Sign-flip peak detector — wait for s5_mom to flip sign
                if not same_dir:
                    # Sign flipped → peak by definition; enter opposite the spike
                    entry_dir = -spike_dir
                    entry_price = ask[i] if entry_dir == 1 else bid[i]
                    entry_bar = i
                    direction = entry_dir
                    in_trade = True
                    state = 0
                # else continue tracking; nothing else needed (no peak_frac)

    return pnl_arr[:count], hold_arr[:count], type_arr[:count], dir_arr[:count]


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
    sim_peak_fade_v2(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, 0, 0.5, 1.0, 2.0, 12)
    sim_peak_fade_v2(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, 1, 0.5, 1.0, 2.0, 12)


def main():
    warmup_jit()
    print(f"EUR_JPY S5 peak-fade v2 (NET pnl semantics, sign-flip mode added)")
    print(f"  spike thr  : {SPIKE_THRS} pips/min")
    print(f"  m1 min     : {M1_MINS}")
    print(f"  peak mode  : 0=decel, 1=signflip")
    print(f"  peak frac  : {PEAK_FRACS}  (decel mode only)")
    print(f"  TP (net)   : {TP_NETS} pips")
    print(f"  SL (net)   : {SL_NETS} pips")
    print(f"  max hold   : {MAX_HOLDS} S5 bars")

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

    print(f"  s5_mom 99.5%ile: {np.percentile(np.abs(s5_mom), 99.5):.1f} pips/min")
    print(f"  s5_mom 99.0%ile: {np.percentile(np.abs(s5_mom), 99.0):.1f}")
    print(f"  s5_mom 95.0%ile: {np.percentile(np.abs(s5_mom), 95.0):.1f}")
    print(f"  EUR_JPY mean spread: {sp[n_is:].mean():.2f} pips")

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
                pfs = PEAK_FRACS if pm == 0 else [0.5]   # placeholder for signflip
                for pf in pfs:
                    for tp in TP_NETS:
                        for sl in SL_NETS:
                            for mh in MAX_HOLDS:
                                config_idx += 1
                                p, h, t, d = sim_peak_fade_v2(
                                    close_o, bid_o, ask_o, sp_o,
                                    s5_o, m1_o, PIP,
                                    float(spike), float(m1m),
                                    int(pm), float(pf),
                                    float(tp), float(sl), int(mh))
                                n = len(p)
                                if n == 0:
                                    rows.append(dict(
                                        spike=spike, m1_min=m1m, peak_mode=pm,
                                        peak_frac=pf, tp=tp, sl=sl, max_hold=mh,
                                        n=0, sum_pips=0.0, ppd=0.0,
                                        wr=0.0, tp_pct=0.0, sl_pct=0.0,
                                        time_pct=0.0, mean_pnl=0.0, mdd=0.0))
                                    continue
                                wr = (p > 0).sum() / n * 100
                                ppd = p.sum() / oos_days
                                mdd = max_dd(p)
                                tp_pct = (t == 0).sum() / n * 100
                                sl_pct = (t == 1).sum() / n * 100
                                tm_pct = (t == 2).sum() / n * 100
                                rows.append(dict(
                                    spike=spike, m1_min=m1m, peak_mode=pm,
                                    peak_frac=pf, tp=tp, sl=sl, max_hold=mh,
                                    n=n, sum_pips=round(float(p.sum()), 1),
                                    ppd=round(ppd, 2), wr=round(wr, 1),
                                    tp_pct=round(tp_pct, 1),
                                    sl_pct=round(sl_pct, 1),
                                    time_pct=round(tm_pct, 1),
                                    mean_pnl=round(float(p.mean()), 2),
                                    mdd=round(mdd, 1),
                                ))
                                if time.time() - last_report > 5:
                                    print(f"  {config_idx} configs done", flush=True)
                                    last_report = time.time()

    df_res = pd.DataFrame(rows)
    out_csv = OUT / "eurjpy_s5_peakfade_v2.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n{len(df_res)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Top by Σ pips ─────────────────────────────────────────────────────
    df_res = df_res.sort_values("sum_pips", ascending=False)
    print("\n" + "="*120)
    print(f"  Top 20 configs by Σ pips (OOS, {oos_days:.0f} days)")
    print("="*120)
    print(f"  {'spike':>5} {'m1':>3} {'pm':>3} {'pf':>4} {'TP':>4} {'SL':>4} "
          f"{'hold':>5}  {'n':>5} {'Σ pips':>9} {'ppd':>8} {'WR%':>5} "
          f"{'TP%':>5} {'SL%':>5} {'tm%':>5} {'mean':>7} {'MDD':>7}")
    for _, r in df_res.head(20).iterrows():
        pm_lbl = "DEC" if r['peak_mode'] == 0 else "SGN"
        print(f"  {r['spike']:>5.0f} {r['m1_min']:>3.0f} {pm_lbl:>3} "
              f"{r['peak_frac']:>4.1f} {r['tp']:>4.1f} {r['sl']:>4.1f} "
              f"{int(r['max_hold']):>5d}  {int(r['n']):>5d} "
              f"{r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
              f"{r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['time_pct']:>5.1f} "
              f"{r['mean_pnl']:>+7.2f} {r['mdd']:>7.1f}")

    print(f"\n  positive configs: {(df_res.sum_pips > 0).sum()} / {len(df_res)}")

    # Filter for high-n, positive WR
    print("\n" + "="*120)
    print(f"  Top 15 by ppd among configs with n >= 200 (broader signal — more reliable)")
    print("="*120)
    filt = df_res[df_res.n >= 200].sort_values("ppd", ascending=False)
    if len(filt):
        print(f"  {'spike':>5} {'m1':>3} {'pm':>3} {'pf':>4} {'TP':>4} {'SL':>4} "
              f"{'hold':>5}  {'n':>5} {'Σ pips':>9} {'ppd':>8} {'WR%':>5} {'mean':>7}")
        for _, r in filt.head(15).iterrows():
            pm_lbl = "DEC" if r['peak_mode'] == 0 else "SGN"
            print(f"  {r['spike']:>5.0f} {r['m1_min']:>3.0f} {pm_lbl:>3} "
                  f"{r['peak_frac']:>4.1f} {r['tp']:>4.1f} {r['sl']:>4.1f} "
                  f"{int(r['max_hold']):>5d}  {int(r['n']):>5d} "
                  f"{r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
                  f"{r['mean_pnl']:>+7.2f}")
    else:
        print("  (no configs with n>=200)")


if __name__ == "__main__":
    main()
