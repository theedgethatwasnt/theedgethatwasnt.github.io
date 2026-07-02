#!/usr/bin/env python3
"""
v6 — Classic SMA7/SMA50 crossover, confirmed on both M5 and H1.

Entry (LONG)
------------
  M5 SMA7[t-1] <= M5 SMA50[t-1]  AND  M5 SMA7[t] >  M5 SMA50[t]   (cross at t)
  H1 SMA7 > H1 SMA50 at the latest CLOSED H1 bar  (regime gate, R1)
  spread <= sp_gate

Entry (SHORT) = mirror.

Exit modes (sweep)
------------------
  TRAIL  : ATR trailing stop using M5 ATR_at_entry.
           Arm at MFE ≥ activate × ATR. Trail width = trail_mult × ATR.
           Fallback max-hold M5 cap.
  XBACK  : exit on opposite M5 crossover (SMA7 crosses back below SMA50 for long;
           above for short). Fallback max-hold cap.

Sweep
-----
EXIT_MODE     ∈ {TRAIL, XBACK}
ACTIVATE_MULT ∈ {0.5, 1.0, 1.5}        (TRAIL only)
TRAIL_MULT    ∈ {0.5, 1.0, 1.5}        (TRAIL only)
MAX_HOLD_M5   ∈ {288, 576}             (24h, 48h)

All 12 pairs, OOS-only (last 30%).
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

SMA_FAST  = 7
SMA_SLOW  = 50

ACTIVATE_MULTS = [0.5, 1.0, 1.5]
TRAIL_MULTS    = [0.5, 1.0, 1.5]
MAX_HOLDS_M5   = [288, 576]   # 24h, 48h
ATR_PERIOD_M5  = 14
IS_FRAC        = 0.70


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def resample_h1(df_m5):
    h = df_m5["high"].resample("1h").max()
    l = df_m5["low"].resample("1h").min()
    c = df_m5["close"].resample("1h").last()
    out = pd.concat([h,l,c], axis=1)
    out.columns = ["high","low","close"]
    return out.dropna()


def wilder_atr(high, low, close, period):
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
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


@njit(cache=True)
def _sim_trail(close, fast_m5, slow_m5, h1_long, h1_short,
               high, low, bid, ask, sp, atr_m5, pip,
               activate_mult, trail_mult, max_hold, sp_gate):
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)   # 0=trail 1=maxhold 2=eod
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0; hwm_pips = 0.0; armed = False

    for t in range(1, n):
        if in_trade:
            excur = (close[t] - ep) / pip * dir_
            if excur > hwm_pips: hwm_pips = excur
            if not armed and atr_entry > 0.0:
                if hwm_pips >= activate_mult * atr_entry / pip:
                    armed = True
            exited = False
            if armed:
                hwm_price = ep + dir_ * hwm_pips * pip
                trail_px  = hwm_price - dir_ * trail_mult * atr_entry
                if dir_ == 1 and low[t] <= trail_px:
                    pnl_out[count]  = (trail_px - ep) / pip - sp[t]
                    hold_out[count] = t - ei; type_out[count] = 0
                    count += 1; in_trade=False; armed=False; hwm_pips=0.0
                    exited = True
                elif dir_ == -1 and high[t] >= trail_px:
                    pnl_out[count]  = (ep - trail_px) / pip - sp[t]
                    hold_out[count] = t - ei; type_out[count] = 0
                    count += 1; in_trade=False; armed=False; hwm_pips=0.0
                    exited = True
            if (not exited) and (t - ei) >= max_hold:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei; type_out[count] = 1
                count += 1; in_trade=False; armed=False; hwm_pips=0.0
            continue

        # Cross detection: SMA7 crosses above (long) or below (short) SMA50 at t
        if np.isnan(fast_m5[t]) or np.isnan(slow_m5[t]) or \
           np.isnan(fast_m5[t-1]) or np.isnan(slow_m5[t-1]) or \
           np.isnan(atr_m5[t]):
            continue
        if sp[t] > sp_gate:
            continue

        long_cross_m5  = (fast_m5[t-1] <= slow_m5[t-1]) and (fast_m5[t] > slow_m5[t])
        short_cross_m5 = (fast_m5[t-1] >= slow_m5[t-1]) and (fast_m5[t] < slow_m5[t])

        if long_cross_m5 and h1_long[t]:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
            atr_entry = atr_m5[t]; hwm_pips = 0.0; armed = False
        elif short_cross_m5 and h1_short[t]:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True
            atr_entry = atr_m5[t]; hwm_pips = 0.0; armed = False

    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei; type_out[count] = 2
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count]


@njit(cache=True)
def _sim_xback(close, fast_m5, slow_m5, h1_long, h1_short,
               bid, ask, sp, atr_m5, pip,
               max_hold, sp_gate):
    """Exit on opposite M5 crossover, fallback max-hold."""
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)   # 0=xback 1=maxhold 2=eod
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0

    for t in range(1, n):
        if in_trade:
            cross_back = False
            if dir_ == 1:
                if fast_m5[t-1] >= slow_m5[t-1] and fast_m5[t] < slow_m5[t]:
                    cross_back = True
            else:
                if fast_m5[t-1] <= slow_m5[t-1] and fast_m5[t] > slow_m5[t]:
                    cross_back = True
            if cross_back:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei; type_out[count] = 0
                count += 1; in_trade=False
                continue
            if (t - ei) >= max_hold:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei; type_out[count] = 1
                count += 1; in_trade=False
            continue

        if np.isnan(fast_m5[t]) or np.isnan(slow_m5[t]) or \
           np.isnan(fast_m5[t-1]) or np.isnan(slow_m5[t-1]):
            continue
        if sp[t] > sp_gate:
            continue

        long_cross_m5  = (fast_m5[t-1] <= slow_m5[t-1]) and (fast_m5[t] > slow_m5[t])
        short_cross_m5 = (fast_m5[t-1] >= slow_m5[t-1]) and (fast_m5[t] < slow_m5[t])

        if long_cross_m5 and h1_long[t]:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
        elif short_cross_m5 and h1_short[t]:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True

    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei; type_out[count] = 2
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 200
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    f = c.copy(); s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0010)
    h1l = np.ones(n, dtype=np.bool_); h1s = np.zeros(n, dtype=np.bool_)
    _sim_trail(c, f, s, h1l, h1s, h, l, b, a, sp, atr, 0.0001, 1.0, 1.0, 96, 2.0)
    _sim_xback(c, f, s, h1l, h1s, b, a, sp, atr, 0.0001, 96, 2.0)


def build_h1_gates(df_m5):
    """Build per-M5 booleans:  h1_long[t]  ↔  latest closed H1 had SMA7 > SMA50.
                                h1_short[t] ↔  latest closed H1 had SMA7 < SMA50."""
    h1 = resample_h1(df_m5)
    fast = h1["close"].rolling(SMA_FAST).mean()
    slow = h1["close"].rolling(SMA_SLOW).mean()
    long_state  = (fast > slow)
    short_state = (fast < slow)
    # R1: shift by 1 H1 so we use the previously-closed bar's state
    long_state  = long_state.shift(1).reindex(df_m5.index, method="ffill").fillna(False).astype(np.bool_)
    short_state = short_state.shift(1).reindex(df_m5.index, method="ffill").fillna(False).astype(np.bool_)
    return long_state.values, short_state.values


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]

    high  = df_m5["high"].values.astype(np.float64)
    low   = df_m5["low"].values.astype(np.float64)
    close = df_m5["close"].values.astype(np.float64)
    bid   = df_m5["bid_c"].values.astype(np.float64)
    ask   = df_m5["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    atr_m5 = wilder_atr(high, low, close, ATR_PERIOD_M5)

    fast_m5 = pd.Series(close).rolling(SMA_FAST).mean().values.astype(np.float64)
    slow_m5 = pd.Series(close).rolling(SMA_SLOW).mean().values.astype(np.float64)
    h1_long, h1_short = build_h1_gates(df_m5)

    n_total = len(close)
    n_is    = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 288.0

    def _record(mode, p, h, t, **kw):
        n = len(p)
        row = dict(pair=pair, mode=mode, n=n, days=round(oos_days,1), **kw)
        if n == 0:
            row.update(ppd=0.0, wr=0.0, mdd=0.0, calmar=0.0,
                       mean_hold_m5=0.0, max_loss=0.0, max_win=0.0)
        else:
            wr  = (p>0).sum() / n * 100
            ppd = p.sum() / oos_days
            mdd = max_dd(p)
            cal = ppd / mdd if mdd > 0 else 0.0
            row.update(ppd=round(ppd,2), wr=round(wr,1),
                       mdd=round(mdd,1), calmar=round(cal,2),
                       mean_hold_m5=round(float(h.mean()),1),
                       max_loss=round(float(p.min()),1),
                       max_win=round(float(p.max()),1))
        all_rows.append(row)

    for max_hold in MAX_HOLDS_M5:
        for act in ACTIVATE_MULTS:
            for trail in TRAIL_MULTS:
                p, h, t = _sim_trail(
                    close[n_is:], fast_m5[n_is:], slow_m5[n_is:],
                    h1_long[n_is:], h1_short[n_is:],
                    high[n_is:], low[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
                    atr_m5[n_is:], pip,
                    float(act), float(trail), int(max_hold), float(sg))
                _record("TRAIL", p, h, t, activate=act, trail=trail, max_hold=max_hold)

        # XBACK doesn't use trail params — emit just once per max_hold
        p, h, t = _sim_xback(
            close[n_is:], fast_m5[n_is:], slow_m5[n_is:],
            h1_long[n_is:], h1_short[n_is:],
            bid[n_is:], ask[n_is:], sp[n_is:], atr_m5[n_is:], pip,
            int(max_hold), float(sg))
        _record("XBACK", p, h, t, activate=0.0, trail=0.0, max_hold=max_hold)

    del close, bid, ask, sp, high, low, atr_m5, df_m5, fast_m5, slow_m5
    gc.collect()


def main():
    warmup_jit()
    print("v6 — SMA7/SMA50 dual-TF crossover")
    print(f"  Entry: M5 SMA{SMA_FAST}/{SMA_SLOW} cross + H1 SMA{SMA_FAST}/{SMA_SLOW} same direction")
    print(f"  Exit modes: TRAIL (act={ACTIVATE_MULTS}, trail={TRAIL_MULTS}) | XBACK (opposite M5 cross)")
    print(f"  max_hold_M5 ∈ {MAX_HOLDS_M5}")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "v6_sma_xover.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Per (mode, params): Σ p/d across 12 pairs ──────────────────────────
    for mode in ("TRAIL", "XBACK"):
        sub = df[df["mode"] == mode]
        if sub.empty: continue
        if mode == "TRAIL":
            cols = ["activate","trail","max_hold"]
        else:
            cols = ["max_hold"]
        g = (sub.groupby(cols)
                .agg(sum_ppd=("ppd","sum"),
                     n_pos=("ppd", lambda x: int((x>0).sum())),
                     total_n=("n","sum"),
                     mean_wr=("wr","mean"),
                     mean_hold=("mean_hold_m5","mean"))
                .reset_index().sort_values("sum_ppd", ascending=False))
        print("\n" + "="*100)
        print(f"  Mode: {mode}  —  Σ OOS p/d across 12 pairs")
        print("="*100)
        print(g.to_string(index=False))

        top = g.iloc[0]
        mask = sub.copy()
        for col in cols:
            mask = mask[mask[col] == top[col]]
        sub2 = mask.sort_values("ppd", ascending=False)
        print(f"\n  --- {mode} top: " +
              ", ".join(f"{c}={top[c]}" for c in cols) + " ---")
        print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>8}{'Calmar':>7}"
              f"{'hold(M5)':>9}{'max_loss':>10}{'max_win':>9}")
        for _, r in sub2.iterrows():
            print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
                  f"{r['mdd']:>8.1f}{r['calmar']:>7.2f}{r['mean_hold_m5']:>9.1f}"
                  f"{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


if __name__ == "__main__":
    main()
