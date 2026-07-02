#!/usr/bin/env python3
"""
H1 Cross-Breakout v3 — Combined-move entry + ATR trailing stop exit.

Refinements (vs v2)
-------------------
Entry simplified:
  Cross above SMA at lag = N_small + 2 (lag ∈ {3, 4}), then a COMBINED
  two-bar momentum check (no leg-by-leg structure), then current bar still
  rising. Acceleration / per-leg consolidation checks dropped — they were
  filtering out most of the valid signal.

  LONG (N_small = 1):
    c[t-4] < sma[t-4]    AND   c[t-3] > sma[t-3]    (cross)
    c[t-1] - c[t-3] > thld                          (combined two-bar move up)
    c[t]   - c[t-1] > 0                             (current bar rising)

  LONG (N_small = 2):
    c[t-5] < sma[t-5]    AND   c[t-4] > sma[t-4]    (cross)
    c[t-1] - c[t-4] > thld                          (combined three-bar move)
    c[t]   - c[t-1] > 0

  Short = mirror.

Exit: ATR trailing stop (unchanged from v2).
  • atr_at_entry captured at entry bar.
  • Track MFE in pips. Arm trail when MFE ≥ activate_mult × ATR_at_entry/pip.
  • Trail width = trail_mult × ATR_at_entry, anchored to HWM_price.
  • Exit at trail_stop_px (bar low ≤ stop for long).
  • Fallback max-hold cap.

Sweep
-----
N_SMALL        ∈ {1, 2}   (cross lag ∈ {3, 4})
SMA_N          ∈ {7, 10}
ENTRY mode     ∈ {pip, atr}
THLD_PIPS      ∈ {5, 10, 20, 30}        # combined-move threshold (pip mode)
THLD_MULTS     ∈ {0.5, 1.0, 1.5, 2.0}   # combined-move threshold (× ATR)
ACTIVATE_MULT  ∈ {0.5, 1.0, 1.5}
TRAIL_MULT     ∈ {0.5, 1.0, 1.5}
MAX_HOLD_H1    = 96  (4 days)
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}

N_SMALLS       = [1, 2]
SMA_NS         = [7, 10]
ATR_PERIOD     = 14
MAX_HOLD_H1    = 96      # 4 days fallback cap

# Entry — combined-move threshold (pip mode)
THLD_PIPS  = [5.0, 10.0, 20.0, 30.0]

# Entry — combined-move threshold (ATR mode, × ATR_at_t)
THLD_MULTS = [0.5, 1.0, 1.5, 2.0]

# Exit — trail
ACTIVATE_MULTS = [0.5, 1.0, 1.5]
TRAIL_MULTS    = [0.5, 1.0, 1.5]

IS_FRAC = 0.70


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def resample_h1(df_m5):
    o = df_m5["open"].resample("1h").first()
    h = df_m5["high"].resample("1h").max()
    l = df_m5["low"].resample("1h").min()
    c = df_m5["close"].resample("1h").last()
    bc = df_m5["bid_c"].resample("1h").last()
    ac = df_m5["ask_c"].resample("1h").last()
    out = pd.concat([o, h, l, c, bc, ac], axis=1)
    out.columns = ["open","high","low","close","bid_c","ask_c"]
    return out.dropna()


def wilder_atr(high, low, close, period):
    n = len(close)
    tr  = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i-1])
        c_ = abs(low[i]  - close[i-1])
        tr[i] = max(a, max(b, c_))
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ── Numba simulator: loosened entry + ATR trailing stop exit ─────────────────

@njit(cache=True)
def _sim_atrail(close, sma, high, low, bid, ask, sp, atr, pip,
                thld_arr, n_small,
                activate_mult, trail_mult, max_hold, sp_gate):
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)   # 0=trail 1=maxhold 2=eod
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0    # in price units
    hwm_pips = 0.0     # max favorable excursion in pips
    armed = False
    min_start = n_small + 3  # need closes at t-n_small-3 .. t for entry check

    for t in range(min_start, n):
        if in_trade:
            # Update HWM (MFE in pips, signed dir)
            excur = (close[t] - ep) / pip * dir_
            if excur > hwm_pips:
                hwm_pips = excur
            # Arm trail once MFE crosses activation threshold
            if not armed and atr_entry > 0.0:
                act_thr_pips = activate_mult * atr_entry / pip
                if hwm_pips >= act_thr_pips:
                    armed = True
            exited = False
            if armed:
                # HWM_price for long = ep + hwm_pips * pip
                # HWM_price for short = ep - hwm_pips * pip
                hwm_price = ep + dir_ * hwm_pips * pip
                trail_px  = hwm_price - dir_ * trail_mult * atr_entry
                if dir_ == 1 and low[t] <= trail_px:
                    pnl_out[count]  = (trail_px - ep) / pip - sp[t]
                    hold_out[count] = t - ei
                    type_out[count] = 0
                    count += 1; in_trade = False; armed = False; hwm_pips = 0.0
                    exited = True
                elif dir_ == -1 and high[t] >= trail_px:
                    pnl_out[count]  = (ep - trail_px) / pip - sp[t]
                    hold_out[count] = t - ei
                    type_out[count] = 0
                    count += 1; in_trade = False; armed = False; hwm_pips = 0.0
                    exited = True
            # Fallback max-hold cap
            if (not exited) and (t - ei) >= max_hold:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei
                type_out[count] = 1
                count += 1; in_trade = False; armed = False; hwm_pips = 0.0
            continue

        # ── Entry check ─────────────────────────────────────────────────────
        thld = thld_arr[t]
        if np.isnan(thld) or thld <= 0.0:
            continue
        if np.isnan(atr[t]):
            continue
        if sp[t] > sp_gate:
            continue

        c_t  = close[t]
        c_1  = close[t-1]   # breakout bar

        # Cross occurred at lag (n_small + 2)
        below_idx = t - n_small - 3
        cross_idx = t - n_small - 2
        if below_idx < 0:
            continue
        combined_move = c_1 - close[cross_idx]   # signed move from cross to t-1

        # ── Long ───────────────────────────────────────────────────────────
        long_xover = (close[below_idx] < sma[below_idx]) and \
                     (close[cross_idx] > sma[cross_idx])
        long_move  = combined_move > thld
        long_curr  = (c_t - c_1) > 0.0
        long_ok = long_xover and long_move and long_curr

        # ── Short (mirror) ─────────────────────────────────────────────────
        short_xover = (close[below_idx] > sma[below_idx]) and \
                      (close[cross_idx] < sma[cross_idx])
        short_move  = combined_move < -thld
        short_curr  = (c_t - c_1) < 0.0
        short_ok = short_xover and short_move and short_curr

        if long_ok:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False
        elif short_ok:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False

    # End-of-data: honest market close of any open position
    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei
        type_out[count] = 2
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 300
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0015)
    thld = np.full(n, 10.0 * 0.0001)
    _sim_atrail(c, s, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.0, 1.0, 48, 2.0)
    _sim_atrail(c, s, h, l, b, a, sp, atr, 0.0001, thld, 2, 1.0, 1.0, 48, 2.0)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    h1 = resample_h1(df_m5)
    del df_m5
    pip = pip_sz(pair); sg = SP_GATES[pair]

    high  = h1["high"].values.astype(np.float64)
    low   = h1["low"].values.astype(np.float64)
    close = h1["close"].values.astype(np.float64)
    bid   = h1["bid_c"].values.astype(np.float64)
    ask   = h1["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)

    n_total = len(close)
    n_is    = int(n_total * IS_FRAC)
    oos_h1  = n_total - n_is
    oos_days = oos_h1 / 24.0

    atr_price = wilder_atr(high, low, close, ATR_PERIOD)

    def _eval(thld_arr_full, mode, thld_label, thld_val,
              sma_n_val, n_small, act, trail):
        sma_arr = pd.Series(close).rolling(sma_n_val).mean().values.astype(np.float64)
        p, h, t = _sim_atrail(
            close[n_is:], sma_arr[n_is:],
            high[n_is:], low[n_is:],
            bid[n_is:], ask[n_is:], sp[n_is:], atr_price[n_is:],
            pip, thld_arr_full[n_is:],
            int(n_small), float(act), float(trail), int(MAX_HOLD_H1), float(sg))
        n = len(p)
        row = dict(pair=pair, mode=mode, sma_n=sma_n_val, n_small=n_small,
                   activate=act, trail=trail,
                   thld_pip=thld_val if thld_label == 'thld_pip' else None,
                   thld_mult=thld_val if thld_label == 'thld_mult' else None,
                   n=n, days=round(oos_days,1))
        if n == 0:
            row.update(ppd=0.0, wr=0.0, mdd=0.0, calmar=0.0,
                       mean_hold_h1=0.0, max_loss=0.0, max_win=0.0,
                       trail_pct=0.0, maxhold_pct=0.0)
        else:
            wr  = (p>0).sum() / n * 100
            ppd = p.sum() / oos_days
            mdd = max_dd(p)
            cal = ppd / mdd if mdd > 0 else 0.0
            row.update(ppd=round(ppd,2), wr=round(wr,1),
                       mdd=round(mdd,1), calmar=round(cal,2),
                       mean_hold_h1=round(float(h.mean()),1),
                       max_loss=round(float(p.min()),1),
                       max_win=round(float(p.max()),1),
                       trail_pct=round((t==0).sum()/n*100,1),
                       maxhold_pct=round((t==1).sum()/n*100,1))
        all_rows.append(row)

    for sma_n in SMA_NS:
        for n_small in N_SMALLS:
            # ── Mode 1: fixed pips ──
            for thld_p in THLD_PIPS:
                thld_arr = np.full(n_total, thld_p * pip, dtype=np.float64)
                for act in ACTIVATE_MULTS:
                    for trail in TRAIL_MULTS:
                        _eval(thld_arr, "pip", "thld_pip", thld_p,
                              sma_n, n_small, act, trail)

            # ── Mode 2: ATR multiples ──
            for thld_m in THLD_MULTS:
                thld_arr_atr = thld_m * atr_price
                for act in ACTIVATE_MULTS:
                    for trail in TRAIL_MULTS:
                        _eval(thld_arr_atr, "atr", "thld_mult", thld_m,
                              sma_n, n_small, act, trail)

    del close, bid, ask, sp, h1, high, low, atr_price
    gc.collect()


def _summary_block(df, mode):
    sub_df = df[df["mode"] == mode]
    if sub_df.empty: return
    if mode == "pip":
        thld_col = "thld_pip"; lbl = "thld_p"
    else:
        thld_col = "thld_mult"; lbl = "thldM"
    group_cols = ["sma_n","n_small",thld_col,"activate","trail"]
    g = (sub_df.groupby(group_cols)
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"),
                 mean_hold=("mean_hold_h1","mean"),
                 mean_trail_pct=("trail_pct","mean"))
            .reset_index().sort_values("sum_ppd", ascending=False))
    print("\n" + "="*110)
    print(f"  Mode: {mode.upper()}  —  Σ OOS p/d across all 12 pairs (top 15)")
    print("="*110)
    print(f"  {'sma':>4}  {'ns':>3}  {lbl:>7}  {'act':>4}  {'trail':>5}  "
          f"{'Σ ppd':>7}  {'pos':>3}/12 {'Σn':>6}  {'WR%':>5}  {'hold':>5}  {'trl%':>5}")
    for _, r in g.head(15).iterrows():
        print(f"  {int(r['sma_n']):>4d}  {int(r['n_small']):>3d}  "
              f"{r[thld_col]:>7.2f}  "
              f"{r['activate']:>4.1f}  {r['trail']:>5.1f}  "
              f"{r['sum_ppd']:>+7.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>6d}  "
              f"{r['mean_wr']:>5.1f}  {r['mean_hold']:>5.1f}  {r['mean_trail_pct']:>5.1f}")

    top = g.iloc[0]
    mask = ((sub_df.sma_n == top['sma_n']) &
            (sub_df.n_small == top['n_small']) &
            (sub_df[thld_col] == top[thld_col]) &
            (sub_df.activate == top['activate']) &
            (sub_df.trail == top['trail']))
    sub2 = sub_df[mask].sort_values("ppd", ascending=False)
    print(f"\n  --- {mode.upper()} top: sma={int(top['sma_n'])} ns={int(top['n_small'])} "
          f"{lbl}={top[thld_col]:.2f} act={top['activate']:.1f} trail={top['trail']:.1f} ---")
    print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'Calmar':>7}"
          f"{'hold':>6}{'trl%':>6}{'max_loss':>10}{'max_win':>9}")
    for _, r in sub2.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['mdd']:>7.1f}{r['calmar']:>7.2f}{r['mean_hold_h1']:>6.1f}"
              f"{r['trail_pct']:>6.1f}{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


def main():
    warmup_jit()
    print("H1 Cross-Breakout v3 — combined-move entry + ATR trailing exit")
    print(f"  N_SMALL={N_SMALLS}  SMA_N={SMA_NS}  max_hold_H1={MAX_HOLD_H1}")
    print(f"  PIP entry thld: {THLD_PIPS}")
    print(f"  ATR entry thld: {THLD_MULTS} × ATR{ATR_PERIOD}")
    print(f"  trail: activate={ACTIVATE_MULTS}  width={TRAIL_MULTS}")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "h1_xbreak_atrail.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")
    _summary_block(df, "pip")
    _summary_block(df, "atr")


if __name__ == "__main__":
    main()
