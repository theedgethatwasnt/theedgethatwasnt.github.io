#!/usr/bin/env python3
"""
EUR_JPY S5 peak-fade — catch the correction after a high-momentum spike.

Pattern (user's spec):
  1. S5 momentum spikes exceptionally high in one direction
  2. M1 momentum confirms — same-direction continuation
  3. Sharp correction in the opposite direction follows

Strategy:
  - Detect spike: |s5_mom[t]| >= spike_threshold (pips/min)
  - Require M1 same-sign continuation: sign(m1_mom) == sign(s5_mom)
                                       AND |m1_mom| >= m1_min
  - Wait for peak: |s5_mom[t]| < |s5_mom[t-1]| × peak_frac    (deceleration)
                   AND s5_mom[t] still has same sign as spike
  - Enter opposite direction at that bar
  - Exit at TP (1-2 pips), SL, or max-hold time stop

All units: pips/min (matches the dashboard).
  s5_mom = (c[t] - c[t-1]) / pip × (60/5)   = 12 × Δpips per 5s bar
  m1_mom = (c[t] - c[t-12]) / pip / 1       = pips over the last 60s

Sweep
-----
spike_threshold  ∈ {30, 50, 80, 120, 180} pips/min
m1_min           ∈ {0, 5, 10} pips/min
peak_frac        ∈ {0.5, 0.7, 0.9}        (drop fraction signalling peak)
TP_pips          ∈ {1, 2, 3}
SL_pips          ∈ {3, 5, 10}
max_hold_S5      ∈ {12, 60, 120}          (1min, 5min, 10min)

OOS-only (last 30% of S5 BA).
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
PIP  = 0.01     # JPY pair
IS_FRAC = 0.70

# Grid
SPIKE_THRS   = [30.0, 50.0, 80.0, 120.0, 180.0]   # pips/min
M1_MINS      = [0.0, 5.0, 10.0]                    # pips/min
PEAK_FRACS   = [0.5, 0.7, 0.9]                     # drop fraction
TP_PIPS_LIST = [1.0, 2.0, 3.0]
SL_PIPS_LIST = [3.0, 5.0, 10.0]
MAX_HOLDS    = [12, 60, 120]                       # S5 bars: 1min, 5min, 10min


@njit(cache=True)
def sim_peak_fade(close, bid, ask, sp, s5_mom, m1_mom, pip,
                  spike_thr, m1_min, peak_frac,
                  tp_pips, sl_pips, max_hold):
    """Detect peak after spike-with-M1-confirmation; enter opposite direction.

    State machine per bar i:
      IDLE: scan for |s5_mom[i]| >= spike_thr with matching M1 → SPIKING.
      SPIKING: track the peak magnitude. If |s5_mom[i]| < peak_mag * peak_frac
               and current direction still matches spike sign → PEAK FOUND →
               enter opposite-direction trade. If sign of s5_mom flips before
               peak, abort and return to IDLE.
      IN_TRADE: exit on TP / SL / max_hold (priority in that order).
    """
    n = len(close)
    pnl_arr  = np.empty(n, dtype=np.float64)
    hold_arr = np.empty(n, dtype=np.int32)
    type_arr = np.empty(n, dtype=np.int8)   # 0=TP, 1=SL, 2=time
    dir_arr  = np.empty(n, dtype=np.int8)
    count = 0

    # State: 0=idle, 1=spiking
    state = 0
    spike_dir = 0      # +1 / -1
    spike_peak = 0.0   # max |s5_mom| seen during this spike
    spike_start_bar = 0

    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0

    for i in range(13, n):
        if in_trade:
            # Exits use the bid/ask appropriate to the exit side
            if direction == 1:
                # LONG: TP when bid >= entry+TP, SL when ask <= entry-SL
                curr_high_pnl = (bid[i] - entry_price) / pip   # favourable check
                curr_low_pnl  = (ask[i] - entry_price) / pip
            else:
                # SHORT: TP when ask <= entry-TP, SL when bid >= entry+SL
                curr_high_pnl = (entry_price - ask[i]) / pip
                curr_low_pnl  = (entry_price - bid[i]) / pip
            # TP check
            if curr_high_pnl >= tp_pips:
                pnl_arr[count]  = tp_pips - sp[i]
                hold_arr[count] = i - entry_bar
                type_arr[count] = 0
                dir_arr[count]  = direction
                count += 1; in_trade = False
            elif curr_low_pnl <= -sl_pips:
                pnl_arr[count]  = -sl_pips - sp[i]
                hold_arr[count] = i - entry_bar
                type_arr[count] = 1
                dir_arr[count]  = direction
                count += 1; in_trade = False
            elif i - entry_bar >= max_hold:
                # Time stop: exit at market
                if direction == 1:
                    pnl = (bid[i] - entry_price) / pip - sp[i]
                else:
                    pnl = (entry_price - ask[i]) / pip - sp[i]
                pnl_arr[count]  = pnl
                hold_arr[count] = i - entry_bar
                type_arr[count] = 2
                dir_arr[count]  = direction
                count += 1; in_trade = False
            continue

        # No active trade — process the state machine
        s5 = s5_mom[i]
        m1 = m1_mom[i]
        abs_s5 = abs(s5)

        if state == 0:
            # IDLE — look for spike
            if abs_s5 >= spike_thr:
                # Need M1 in same direction
                if (s5 > 0 and m1 >= m1_min) or (s5 < 0 and m1 <= -m1_min):
                    state = 1
                    spike_dir = 1 if s5 > 0 else -1
                    spike_peak = abs_s5
                    spike_start_bar = i
        elif state == 1:
            # SPIKING — track peak; check for deceleration → enter
            same_dir = (spike_dir == 1 and s5 > 0) or (spike_dir == -1 and s5 < 0)
            if not same_dir:
                # Momentum sign flipped before clean peak detection — abort
                state = 0
                continue
            if abs_s5 > spike_peak:
                spike_peak = abs_s5
                continue   # still climbing
            # abs_s5 <= spike_peak (deceleration possible)
            if abs_s5 <= spike_peak * peak_frac:
                # PEAK FOUND — enter opposite direction
                entry_dir = -spike_dir
                if entry_dir == 1:
                    entry_price = ask[i]
                else:
                    entry_price = bid[i]
                entry_bar = i
                direction = entry_dir
                in_trade = True
                state = 0   # reset for next spike

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
    sim_peak_fade(c, b, a, sp, s5, m1, PIP, 30.0, 5.0, 0.5, 1.0, 5.0, 12)


def main():
    warmup_jit()
    print("EUR_JPY S5 peak-fade backtest")
    print(f"  spike thr  : {SPIKE_THRS} pips/min")
    print(f"  m1 min     : {M1_MINS} pips/min")
    print(f"  peak frac  : {PEAK_FRACS}")
    print(f"  TP         : {TP_PIPS_LIST} pips")
    print(f"  SL         : {SL_PIPS_LIST} pips")
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

    # Compute s5_mom and m1_mom in pips/min
    # s5_mom: 1-bar (5s) change in pips/min = pips × (60/5) = pips × 12
    s5_mom = np.empty(n_total, dtype=np.float64)
    s5_mom[0] = 0.0
    s5_mom[1:] = (close[1:] - close[:-1]) / PIP * 12.0
    # m1_mom: 12-bar (1 min) change in pips/min = pips / 1
    m1_mom = np.empty(n_total, dtype=np.float64)
    m1_mom[:12] = 0.0
    m1_mom[12:] = (close[12:] - close[:-12]) / PIP

    print(f"  s5_mom 99.5%ile: {np.percentile(np.abs(s5_mom), 99.5):.1f} pips/min")
    print(f"  s5_mom 99.0%ile: {np.percentile(np.abs(s5_mom), 99.0):.1f}")
    print(f"  s5_mom 95.0%ile: {np.percentile(np.abs(s5_mom), 95.0):.1f}")
    print(f"  s5_mom 90.0%ile: {np.percentile(np.abs(s5_mom), 90.0):.1f}")

    close_o = close[n_is:]
    bid_o   = bid[n_is:]
    ask_o   = ask[n_is:]
    sp_o    = sp[n_is:]
    s5_o    = s5_mom[n_is:]
    m1_o    = m1_mom[n_is:]

    print(f"\nOOS bars: {len(close_o):,}  (~{oos_days:.0f} days)")
    print(f"Sweep size: {len(SPIKE_THRS) * len(M1_MINS) * len(PEAK_FRACS) * len(TP_PIPS_LIST) * len(SL_PIPS_LIST) * len(MAX_HOLDS)} configs")

    rows = []
    t0 = time.time()
    config_idx = 0
    total = (len(SPIKE_THRS) * len(M1_MINS) * len(PEAK_FRACS)
             * len(TP_PIPS_LIST) * len(SL_PIPS_LIST) * len(MAX_HOLDS))
    last_report = time.time()

    for spike in SPIKE_THRS:
        for m1m in M1_MINS:
            for pf in PEAK_FRACS:
                for tp in TP_PIPS_LIST:
                    for sl in SL_PIPS_LIST:
                        for mh in MAX_HOLDS:
                            config_idx += 1
                            p, h, t, d = sim_peak_fade(
                                close_o, bid_o, ask_o, sp_o,
                                s5_o, m1_o, PIP,
                                float(spike), float(m1m), float(pf),
                                float(tp), float(sl), int(mh))
                            n = len(p)
                            if n == 0:
                                rows.append(dict(spike=spike, m1_min=m1m, peak_frac=pf,
                                                 tp=tp, sl=sl, max_hold=mh,
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
                                spike=spike, m1_min=m1m, peak_frac=pf,
                                tp=tp, sl=sl, max_hold=mh,
                                n=n, sum_pips=round(float(p.sum()), 1),
                                ppd=round(ppd, 2), wr=round(wr, 1),
                                tp_pct=round(tp_pct, 1),
                                sl_pct=round(sl_pct, 1),
                                time_pct=round(tm_pct, 1),
                                mean_pnl=round(float(p.mean()), 2),
                                mdd=round(mdd, 1),
                            ))
                            if time.time() - last_report > 5:
                                print(f"  {config_idx}/{total} configs, "
                                      f"{config_idx/(time.time()-t0):.0f} cfg/s", flush=True)
                                last_report = time.time()

    df_res = pd.DataFrame(rows)
    out_csv = OUT / "eurjpy_s5_peakfade.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n{len(df_res)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Top configs ───────────────────────────────────────────────────────
    print("\n" + "="*110)
    print(f"  Top 20 configs by Σ pips (OOS, {oos_days:.0f} days)")
    print("="*110)
    df_res = df_res.sort_values("sum_pips", ascending=False)
    print(f"  {'spike':>6} {'m1':>4} {'pf':>4} {'TP':>4} {'SL':>4} {'hold':>5}  "
          f"{'n':>6} {'Σ pips':>9} {'ppd':>8} {'WR%':>5} "
          f"{'TP%':>5} {'SL%':>5} {'tm%':>5} {'mean':>7} {'MDD':>7}")
    for _, r in df_res.head(20).iterrows():
        print(f"  {r['spike']:>6.0f} {r['m1_min']:>4.0f} {r['peak_frac']:>4.1f} "
              f"{r['tp']:>4.1f} {r['sl']:>4.1f} {int(r['max_hold']):>5d}  "
              f"{int(r['n']):>6d} {r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f} "
              f"{r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['time_pct']:>5.1f} "
              f"{r['mean_pnl']:>+7.2f} {r['mdd']:>7.1f}")

    # ── Best WR configs (with min trade count) ─────────────────────────────
    print("\n" + "="*110)
    print(f"  Top 15 by WR (filtering n >= 100 trades)")
    print("="*110)
    filt = df_res[df_res.n >= 100].sort_values("wr", ascending=False)
    print(f"  {'spike':>6} {'m1':>4} {'pf':>4} {'TP':>4} {'SL':>4} {'hold':>5}  "
          f"{'n':>6} {'Σ pips':>9} {'ppd':>8} {'WR%':>5}  {'mean':>7}")
    for _, r in filt.head(15).iterrows():
        print(f"  {r['spike']:>6.0f} {r['m1_min']:>4.0f} {r['peak_frac']:>4.1f} "
              f"{r['tp']:>4.1f} {r['sl']:>4.1f} {int(r['max_hold']):>5d}  "
              f"{int(r['n']):>6d} {r['sum_pips']:>+9.1f} {r['ppd']:>+8.2f} {r['wr']:>5.1f}  "
              f"{r['mean_pnl']:>+7.2f}")


if __name__ == "__main__":
    main()
