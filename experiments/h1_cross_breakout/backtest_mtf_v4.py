#!/usr/bin/env python3
"""
H1 + M5 MTF v4 — M5 cross-breakout entry, H1 regime gate, ATR trail exit.

Idea
----
Run the v3 pattern (cross → combined two-bar move → current bar confirms)
on M5 instead of H1, and require H1 to agree on direction. M5 gives finer
timing (~5 min vs 1 h delay); H1 filters out fast-noise direction.

M5 entry rule (LONG, mirror for SHORT)
--------------------------------------
At each completed M5 bar t:
  • c[t - n_small - 3] < sma_m5[t - n_small - 3]
  • c[t - n_small - 2] > sma_m5[t - n_small - 2]     (M5 cross)
  • c[t - 1] - c[t - n_small - 2] > thld             (combined move)
  • c[t] - c[t - 1] > 0                              (current bar still up)
  • spread gate
H1 gate:
  • At the M5 bar's timestamp, look up the latest CLOSED H1 bar.
  • Long allowed only if h1.close > h1.sma_h1 AND h1.sma slope up over the
    last `H1_TREND_BARS` H1 bars (sma_h1[t] - sma_h1[t-H1_TREND_BARS] > 0).
  • Short symmetric.

Exit
----
Same ATR trailing stop as v3 but with M5 ATR.
  • atr_m5_at_entry captured at entry.
  • Track MFE in pips. Arm trail when MFE ≥ activate × atr_m5_at_entry/pip.
  • Trail width = trail_mult × atr_m5_at_entry.
  • Fallback max-hold cap (M5 bars).

Sweep
-----
SMA_M5       ∈ {12, 24, 50}   # 1h / 2h / ~4h-equivalent
SMA_H1       ∈ {7, 14}        # H1 regime SMA
H1_TREND_BARS = 5             # H1 SMA slope window (5h)
N_SMALL      ∈ {1, 2}
THLD_PIPS    ∈ {3, 5, 8, 12}
THLD_MULTS   ∈ {0.5, 1.0, 1.5}    # × M5 ATR
ACTIVATE_MULT∈ {0.5, 1.0, 1.5}
TRAIL_MULT   ∈ {0.5, 1.0, 1.5}
MAX_HOLD_M5  = 288  (24 h)
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

# Sweep grids
N_SMALLS       = [1, 2]
SMA_M5S        = [12, 24, 50]
SMA_H1S        = [7, 14]
H1_TREND_BARS  = 5
THLD_PIPS      = [3.0, 5.0, 8.0, 12.0]
THLD_MULTS     = [0.5, 1.0, 1.5]
ACTIVATE_MULTS = [0.5, 1.0, 1.5]
TRAIL_MULTS    = [0.5, 1.0, 1.5]
MAX_HOLD_M5    = 288
ATR_PERIOD_M5  = 14
IS_FRAC        = 0.70


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def resample_h1(df_m5):
    o = df_m5["open"].resample("1h").first()
    h = df_m5["high"].resample("1h").max()
    l = df_m5["low"].resample("1h").min()
    c = df_m5["close"].resample("1h").last()
    out = pd.concat([o,h,l,c], axis=1)
    out.columns = ["open","high","low","close"]
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
def _sim(close, sma_m5, h1_regime,
         high, low, bid, ask, sp, atr_m5, pip,
         thld_arr, n_small,
         activate_mult, trail_mult, max_hold, sp_gate):
    """
    h1_regime[t] :   +1 long-allowed, -1 short-allowed, 0 no entries
    """
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0; hwm_pips = 0.0; armed = False
    min_start = n_small + 3

    for t in range(min_start, n):
        if in_trade:
            excur = (close[t] - ep) / pip * dir_
            if excur > hwm_pips:
                hwm_pips = excur
            if not armed and atr_entry > 0.0:
                act_thr_pips = activate_mult * atr_entry / pip
                if hwm_pips >= act_thr_pips:
                    armed = True
            exited = False
            if armed:
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
            if (not exited) and (t - ei) >= max_hold:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei
                type_out[count] = 1
                count += 1; in_trade = False; armed = False; hwm_pips = 0.0
            continue

        thld = thld_arr[t]
        if np.isnan(thld) or thld <= 0.0:
            continue
        if np.isnan(atr_m5[t]):
            continue
        if sp[t] > sp_gate:
            continue
        regime = h1_regime[t]
        if regime == 0:
            continue

        below_idx = t - n_small - 3
        cross_idx = t - n_small - 2
        if below_idx < 0:
            continue
        c_t = close[t]; c_1 = close[t-1]
        combined_move = c_1 - close[cross_idx]

        if regime == 1:
            long_xover = (close[below_idx] < sma_m5[below_idx]) and \
                         (close[cross_idx] > sma_m5[cross_idx])
            long_move  = combined_move > thld
            long_curr  = (c_t - c_1) > 0.0
            if long_xover and long_move and long_curr:
                ep = ask[t]; dir_ = 1; ei = t; in_trade = True
                atr_entry = atr_m5[t]; hwm_pips = 0.0; armed = False
                continue
        else:  # regime == -1
            short_xover = (close[below_idx] > sma_m5[below_idx]) and \
                          (close[cross_idx] < sma_m5[cross_idx])
            short_move  = combined_move < -thld
            short_curr  = (c_t - c_1) < 0.0
            if short_xover and short_move and short_curr:
                ep = bid[t]; dir_ = -1; ei = t; in_trade = True
                atr_entry = atr_m5[t]; hwm_pips = 0.0; armed = False

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
    n = 600
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0010)
    thld = np.full(n, 5.0 * 0.0001)
    reg = np.ones(n, dtype=np.int8)
    _sim(c, s, reg, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.0, 1.0, 96, 2.0)
    _sim(c, s, reg, h, l, b, a, sp, atr, 0.0001, thld, 2, 1.0, 1.0, 96, 2.0)


def build_h1_regime(df_m5, sma_h1_n, trend_bars):
    """Build per-M5-bar regime by forward-filling H1 SMA & slope.

    +1 long-allowed: c_h1 > sma_h1 AND sma_h1[t] - sma_h1[t - trend_bars] > 0
    -1 short-allowed: c_h1 < sma_h1 AND slope < 0
     0 none
    Aligned to M5 grid via reindex(forward-fill).
    """
    h1 = resample_h1(df_m5)
    sma = h1["close"].rolling(sma_h1_n).mean()
    slope = sma - sma.shift(trend_bars)
    reg = pd.Series(0, index=h1.index, dtype=np.int8)
    reg[(h1["close"] > sma) & (slope > 0)] = 1
    reg[(h1["close"] < sma) & (slope < 0)] = -1
    # Crucial: shift by 1 H1 bar so we use the previously-closed H1 bar (R1).
    reg = reg.shift(1).reindex(df_m5.index, method="ffill").fillna(0).astype(np.int8)
    return reg.values


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

    n_total = len(close)
    n_is    = int(n_total * IS_FRAC)
    oos_m5  = n_total - n_is
    oos_days = oos_m5 / 288.0

    # Pre-build H1 regimes per SMA setting
    regimes = {sh: build_h1_regime(df_m5, sh, H1_TREND_BARS) for sh in SMA_H1S}

    def _eval(thld_arr_full, mode, thld_val,
              sma_m5_val, sma_h1_val, n_small, act, trail):
        sma_arr = pd.Series(close).rolling(sma_m5_val).mean().values.astype(np.float64)
        reg = regimes[sma_h1_val]
        p, h, t = _sim(
            close[n_is:], sma_arr[n_is:], reg[n_is:],
            high[n_is:], low[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
            atr_m5[n_is:],
            pip, thld_arr_full[n_is:],
            int(n_small), float(act), float(trail), int(MAX_HOLD_M5), float(sg))
        n = len(p)
        row = dict(pair=pair, mode=mode, sma_m5=sma_m5_val, sma_h1=sma_h1_val,
                   n_small=n_small, activate=act, trail=trail,
                   thld_pip=thld_val if mode == 'pip' else None,
                   thld_mult=thld_val if mode == 'atr' else None,
                   n=n, days=round(oos_days,1))
        if n == 0:
            row.update(ppd=0.0, wr=0.0, mdd=0.0, calmar=0.0,
                       mean_hold_m5=0.0, max_loss=0.0, max_win=0.0,
                       trail_pct=0.0, maxhold_pct=0.0)
        else:
            wr  = (p>0).sum() / n * 100
            ppd = p.sum() / oos_days
            mdd = max_dd(p)
            cal = ppd / mdd if mdd > 0 else 0.0
            row.update(ppd=round(ppd,2), wr=round(wr,1),
                       mdd=round(mdd,1), calmar=round(cal,2),
                       mean_hold_m5=round(float(h.mean()),1),
                       max_loss=round(float(p.min()),1),
                       max_win=round(float(p.max()),1),
                       trail_pct=round((t==0).sum()/n*100,1),
                       maxhold_pct=round((t==1).sum()/n*100,1))
        all_rows.append(row)

    for sma_m5_val in SMA_M5S:
        for sma_h1_val in SMA_H1S:
            for n_small in N_SMALLS:
                # PIP mode
                for thld_p in THLD_PIPS:
                    thld_arr = np.full(n_total, thld_p * pip, dtype=np.float64)
                    for act in ACTIVATE_MULTS:
                        for trail in TRAIL_MULTS:
                            _eval(thld_arr, "pip", thld_p,
                                  sma_m5_val, sma_h1_val, n_small, act, trail)
                # ATR mode
                for thld_m in THLD_MULTS:
                    thld_arr_atr = thld_m * atr_m5
                    for act in ACTIVATE_MULTS:
                        for trail in TRAIL_MULTS:
                            _eval(thld_arr_atr, "atr", thld_m,
                                  sma_m5_val, sma_h1_val, n_small, act, trail)

    del close, bid, ask, sp, high, low, atr_m5, df_m5
    gc.collect()


def _summary_block(df, mode):
    sub_df = df[df["mode"] == mode]
    if sub_df.empty: return
    thld_col = "thld_pip" if mode == "pip" else "thld_mult"
    lbl = "thld_p" if mode == "pip" else "thldM"
    group_cols = ["sma_m5","sma_h1","n_small",thld_col,"activate","trail"]
    g = (sub_df.groupby(group_cols)
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"),
                 mean_hold=("mean_hold_m5","mean"),
                 mean_trail_pct=("trail_pct","mean"))
            .reset_index().sort_values("sum_ppd", ascending=False))
    print("\n" + "="*120)
    print(f"  Mode: {mode.upper()}  —  Σ OOS p/d across 12 pairs (top 15)")
    print("="*120)
    print(f"  {'smaM5':>5}  {'smaH1':>5}  {'ns':>3}  {lbl:>7}  {'act':>4}  {'trail':>5}  "
          f"{'Σ ppd':>7}  {'pos':>3}/12 {'Σn':>6}  {'WR%':>5}  {'hold(M5)':>8}  {'trl%':>5}")
    for _, r in g.head(15).iterrows():
        print(f"  {int(r['sma_m5']):>5d}  {int(r['sma_h1']):>5d}  "
              f"{int(r['n_small']):>3d}  {r[thld_col]:>7.2f}  "
              f"{r['activate']:>4.1f}  {r['trail']:>5.1f}  "
              f"{r['sum_ppd']:>+7.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>6d}  "
              f"{r['mean_wr']:>5.1f}  {r['mean_hold']:>8.1f}  {r['mean_trail_pct']:>5.1f}")

    top = g.iloc[0]
    mask = ((sub_df.sma_m5 == top['sma_m5']) &
            (sub_df.sma_h1 == top['sma_h1']) &
            (sub_df.n_small == top['n_small']) &
            (sub_df[thld_col] == top[thld_col]) &
            (sub_df.activate == top['activate']) &
            (sub_df.trail == top['trail']))
    sub2 = sub_df[mask].sort_values("ppd", ascending=False)
    print(f"\n  --- {mode.upper()} top: sma_M5={int(top['sma_m5'])} sma_H1={int(top['sma_h1'])} "
          f"ns={int(top['n_small'])} {lbl}={top[thld_col]:.2f} "
          f"act={top['activate']:.1f} trail={top['trail']:.1f} ---")
    print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>8}{'Calmar':>7}"
          f"{'hold(M5)':>9}{'trl%':>6}{'max_loss':>10}{'max_win':>9}")
    for _, r in sub2.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['mdd']:>8.1f}{r['calmar']:>7.2f}{r['mean_hold_m5']:>9.1f}"
              f"{r['trail_pct']:>6.1f}{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


def main():
    warmup_jit()
    print("H1+M5 MTF v4 — M5 entry × H1 regime gate × ATR trail exit")
    print(f"  SMA_M5={SMA_M5S}  SMA_H1={SMA_H1S}  H1_TREND_BARS={H1_TREND_BARS}")
    print(f"  N_SMALL={N_SMALLS}")
    print(f"  PIP thld: {THLD_PIPS}  ATR thld: {THLD_MULTS} × M5_ATR{ATR_PERIOD_M5}")
    print(f"  trail: act={ACTIVATE_MULTS} width={TRAIL_MULTS}  max_hold_M5={MAX_HOLD_M5}")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "mtf_v4_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")
    _summary_block(df, "pip")
    _summary_block(df, "atr")


if __name__ == "__main__":
    main()
