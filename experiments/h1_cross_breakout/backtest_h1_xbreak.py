#!/usr/bin/env python3
"""
H1 Cross-Consolidation-Breakout — backtest sweep.

Entry rule (long; short is mirror)
----------------------------------
All conditions on completed H1 closes; t = current bar:
  c[t-4]   - sma_N[t-4] <  0
  c[t-3]   - sma_N[t-3] >  0                # crossed above SMA at t-3
  0 < c[t-2] - c[t-3]   <  thld_pip * pip   # small same-direction move
       c[t-1] - c[t-2]   >  thld_pip * pip   # large same-direction move
  |c[t-1] - c[t-2]|     >  |c[t-2] - c[t-3]|   # acceleration safety
  c[t]     - c[t-1]     >  0                # current bar still rising

Exit rule
---------
Close at H1 bar close when:
  |c[t-1] - sma_N[t-1]|  <  |c[t-2] - sma_N[t-2]|     (gap shrinking)
  AND if exit_strict:
      |c[t]   - sma_N[t]  |  <  |c[t-1] - sma_N[t-1]| (still shrinking)

No fixed TP or SL. Exit signal IS the risk control.

Data + sampling
---------------
M5 BA resampled to H1 (last close per H1 window). Spread per H1 entry
sampled from the M5 bar at the H1 boundary timestamp.

Pairs   : 12 standard FX-Core set
SMA_N   : {5, 7, 10}
thld    : {3, 5, 8, 12, 20} pips
exit    : {False, True}      # whether to require current-bar gap also shrinks
IS/OOS  : 70 / 30
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

SMA_NS         = [5, 7, 10]
EXIT_STRICTS   = [0, 1]   # 0 = single-bar shrink rule; 1 = also require current bar
IS_FRAC        = 0.70

# Mode 1 — fixed-pip thresholds
SMALL_PIPS = [1.0, 2.0, 3.0, 5.0, 8.0]
LARGE_PIPS = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0]

# Mode 2 — ATR-multiple thresholds (per-bar ATR14 on H1)
ATR_PERIOD     = 14
SMALL_MULTS    = [0.05, 0.10, 0.20, 0.30, 0.50]
LARGE_MULTS    = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00]


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def resample_h1(df_m5):
    """Resample M5 BA to H1 OHLC + bid_c + ask_c (last close per H1 window)."""
    o = df_m5["open"].resample("1h").first()
    h = df_m5["high"].resample("1h").max()
    l = df_m5["low"].resample("1h").min()
    c = df_m5["close"].resample("1h").last()
    bc = df_m5["bid_c"].resample("1h").last()
    ac = df_m5["ask_c"].resample("1h").last()
    out = pd.concat([o, h, l, c, bc, ac], axis=1)
    out.columns = ["open","high","low","close","bid_c","ask_c"]
    return out.dropna()


@njit(cache=True)
def _sim(close, sma, bid, ask, sp, pip, small_arr, large_arr, exit_strict, sp_gate):
    """Iterate H1 bars from t=4 to end. Track in_trade state.

    small_arr[t] : per-bar upper bound on |c[t-2] - c[t-3]| (PRICE units)
    large_arr[t] : per-bar lower bound on |c[t-1] - c[t-2]| (PRICE units)

    For fixed-pip mode, pass constant arrays. For ATR mode, pass mult * atr.

    Returns:
      pnl (pips/trade net of spread), hold_bars, exit_type (0=signal, 1=eod).
    """
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0

    for t in range(4, n):
        if in_trade:
            # Exit check at current bar close
            gap1 = abs(close[t-1] - sma[t-1])
            gap2 = abs(close[t-2] - sma[t-2])
            shrink_1 = gap1 < gap2
            gap0 = abs(close[t] - sma[t])
            shrink_0 = gap0 < gap1
            do_exit = shrink_1 if exit_strict == 0 else (shrink_1 and shrink_0)
            if do_exit:
                # Exit at this bar's close-side bid/ask
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei
                type_out[count] = 0
                count += 1
                in_trade = False
            # If still in trade, fall through (no re-entry this bar)
            if in_trade:
                continue

        # Entry check (after possible same-bar exit)
        # Need closes at t, t-1, t-2, t-3, t-4 and sma at t-3, t-4
        c_t  = close[t];   c_1 = close[t-1]; c_2 = close[t-2]
        c_3  = close[t-3]; c_4 = close[t-4]
        s_3  = sma[t-3];   s_4 = sma[t-4]
        d3 = c_2 - c_3   # small-bar candidate
        d2 = c_1 - c_2   # big-bar candidate
        sp_thr = small_arr[t]
        lg_thr = large_arr[t]
        if np.isnan(sp_thr) or np.isnan(lg_thr) or lg_thr <= sp_thr:
            continue
        # LONG
        long_xover = (c_4 < s_4) and (c_3 > s_3)
        long_cons  = (0.0 < d3 < sp_thr)
        long_break = (d2 > lg_thr)
        long_accel = abs(d2) > abs(d3)
        long_curr  = (c_t - c_1) > 0.0
        long_ok = long_xover and long_cons and long_break and long_accel and long_curr
        # SHORT (mirror)
        short_xover = (c_4 > s_4) and (c_3 < s_3)
        short_cons  = (-sp_thr < d3 < 0.0)
        short_break = (d2 < -lg_thr)
        short_accel = abs(d2) > abs(d3)
        short_curr  = (c_t - c_1) < 0.0
        short_ok = short_xover and short_cons and short_break and short_accel and short_curr

        # Spread gate
        if sp[t] > sp_gate:
            continue

        if long_ok:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
        elif short_ok:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True

    # Close any open position at end-of-OOS at market for honest accounting
    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei
        type_out[count] = 1   # end-of-data exit
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def wilder_atr(high, low, close, period):
    """Wilder's ATR on H1 OHLC."""
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


def warmup_jit():
    n = 200
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n)
    sm = np.full(n, 3.0 * 0.0001); lg = np.full(n, 8.0 * 0.0001)
    _sim(c, s, b, a, sp, 0.0001, sm, lg, 0, 2.0)
    _sim(c, s, b, a, sp, 0.0001, sm, lg, 1, 2.0)


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
    oos_days = oos_h1 / 24.0   # H1 bars per day = 24

    # Per-bar ATR for ATR-mode (in price units)
    atr_price = wilder_atr(high, low, close, ATR_PERIOD)

    def _record(mode, p1_label, p1_val, p2_label, p2_val,
                sma_n, es, p, h):
        n = len(p)
        if n == 0:
            all_rows.append(dict(pair=pair, mode=mode, sma_n=sma_n,
                                 small=p1_val if p1_label=='small' else None,
                                 large=p2_val if p2_label=='large' else None,
                                 small_mult=p1_val if p1_label=='small_mult' else None,
                                 large_mult=p2_val if p2_label=='large_mult' else None,
                                 exit_strict=es, n=0, ppd=0.0, wr=0.0,
                                 mdd=0.0, calmar=0.0, mean_hold_h1=0.0,
                                 max_loss=0.0, max_win=0.0,
                                 days=round(oos_days,1)))
            return
        wr = (p>0).sum() / n * 100
        ppd = p.sum() / oos_days
        mdd = max_dd(p)
        cal = ppd / mdd if mdd > 0 else 0.0
        all_rows.append(dict(pair=pair, mode=mode, sma_n=sma_n,
                             small=p1_val if p1_label=='small' else None,
                             large=p2_val if p2_label=='large' else None,
                             small_mult=p1_val if p1_label=='small_mult' else None,
                             large_mult=p2_val if p2_label=='large_mult' else None,
                             exit_strict=es, n=n, ppd=round(ppd,2),
                             wr=round(wr,1),
                             mdd=round(mdd,1), calmar=round(cal,2),
                             mean_hold_h1=round(float(h.mean()),1),
                             max_loss=round(float(p.min()),1),
                             max_win=round(float(p.max()),1),
                             days=round(oos_days,1)))

    for sma_n in SMA_NS:
        sma = pd.Series(close).rolling(sma_n).mean().values.astype(np.float64)

        # ── Mode 1: fixed pips ───────────────────────────────────────────
        for small_p in SMALL_PIPS:
            small_arr = np.full(n_total, small_p * pip, dtype=np.float64)
            for large_p in LARGE_PIPS:
                if large_p <= small_p:
                    continue
                large_arr = np.full(n_total, large_p * pip, dtype=np.float64)
                for es in EXIT_STRICTS:
                    p, h, t = _sim(close[n_is:], sma[n_is:],
                                   bid[n_is:], ask[n_is:], sp[n_is:],
                                   pip, small_arr[n_is:], large_arr[n_is:],
                                   int(es), float(sg))
                    _record("pip", "small", small_p, "large", large_p,
                            sma_n, es, p, h)

        # ── Mode 2: ATR multiples ────────────────────────────────────────
        for small_m in SMALL_MULTS:
            small_arr_atr = small_m * atr_price
            for large_m in LARGE_MULTS:
                if large_m <= small_m:
                    continue
                large_arr_atr = large_m * atr_price
                for es in EXIT_STRICTS:
                    p, h, t = _sim(close[n_is:], sma[n_is:],
                                   bid[n_is:], ask[n_is:], sp[n_is:],
                                   pip, small_arr_atr[n_is:], large_arr_atr[n_is:],
                                   int(es), float(sg))
                    _record("atr", "small_mult", small_m, "large_mult", large_m,
                            sma_n, es, p, h)

    del close, bid, ask, sp, h1, atr_price, high, low
    gc.collect()


def main():
    warmup_jit()
    print("H1 Cross-Consolidation-Breakout sweep")
    print(f"  SMA_N={SMA_NS}  exit_strict={EXIT_STRICTS}")
    print(f"  PIP-mode: small={SMALL_PIPS} large={LARGE_PIPS}")
    print(f"  ATR-mode: small_mult={SMALL_MULTS} large_mult={LARGE_MULTS} (ATR{ATR_PERIOD} on H1)")
    print(f"  pairs={len(ALL_PAIRS)}  OOS=30%")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "h1_xbreak_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Portfolio summary by config, split by mode ────────────────────────
    for mode in ("pip", "atr"):
        sub_df = df[df["mode"] == mode]
        if sub_df.empty: continue
        if mode == "pip":
            group_cols = ["sma_n", "small", "large", "exit_strict"]
            lbl_s, lbl_l = "small", "large"
        else:
            group_cols = ["sma_n", "small_mult", "large_mult", "exit_strict"]
            lbl_s, lbl_l = "smallM", "largeM"
        g = (sub_df.groupby(group_cols)
                .agg(sum_ppd=("ppd","sum"),
                     n_pos=("ppd", lambda x: int((x>0).sum())),
                     total_n=("n","sum"),
                     mean_wr=("wr","mean"),
                     mean_hold=("mean_hold_h1","mean"))
                .reset_index().sort_values("sum_ppd", ascending=False))
        print("\n" + "="*100)
        print(f"  Mode: {mode.upper()}  —  Σ OOS p/d across all 12 pairs (top 15)")
        print("="*100)
        print(f"  {'sma':>4}  {lbl_s:>7}  {lbl_l:>7}  {'strict':>6}  "
              f"{'Σ ppd':>7}  {'pos':>3}/12 {'Σn':>6}  {'WR%':>5}  {'hold(H1)':>8}")
        for _, r in g.head(15).iterrows():
            print(f"  {int(r['sma_n']):>4d}  {r[group_cols[1]]:>7.2f}  {r[group_cols[2]]:>7.2f}  "
                  f"{int(r['exit_strict']):>6d}  "
                  f"{r['sum_ppd']:>+7.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>6d}  "
                  f"{r['mean_wr']:>5.1f}  {r['mean_hold']:>8.1f}")
        print("  ── worst 5 ──")
        for _, r in g.tail(5).iterrows():
            print(f"  {int(r['sma_n']):>4d}  {r[group_cols[1]]:>7.2f}  {r[group_cols[2]]:>7.2f}  "
                  f"{int(r['exit_strict']):>6d}  "
                  f"{r['sum_ppd']:>+7.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>6d}  "
                  f"{r['mean_wr']:>5.1f}  {r['mean_hold']:>8.1f}")

        # Per-pair detail at top config for this mode
        top = g.iloc[0]
        mask = ((sub_df.sma_n == top['sma_n']) &
                (sub_df[group_cols[1]] == top[group_cols[1]]) &
                (sub_df[group_cols[2]] == top[group_cols[2]]) &
                (sub_df.exit_strict == top['exit_strict']))
        sub2 = sub_df[mask].sort_values("ppd", ascending=False)
        print(f"\n  --- {mode.upper()} top: sma={int(top['sma_n'])} "
              f"{lbl_s}={top[group_cols[1]]:.2f} {lbl_l}={top[group_cols[2]]:.2f} "
              f"strict={int(top['exit_strict'])} ---")
        print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'Calmar':>7}{'hold':>6}{'max_loss':>10}{'max_win':>9}")
        for _, r in sub2.iterrows():
            print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
                  f"{r['mdd']:>7.1f}{r['calmar']:>7.2f}{r['mean_hold_h1']:>6.1f}"
                  f"{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


if __name__ == "__main__":
    main()
