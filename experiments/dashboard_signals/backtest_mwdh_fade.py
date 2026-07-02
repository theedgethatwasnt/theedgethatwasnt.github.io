#!/usr/bin/env python3
"""
M/W/D/H/M5 alignment-score FADE strategy backtest.

Bin study (backtest_mwdh_nesting.py) showed contrarian forward returns:
  score=-5 → +10.1 pips avg over 24h (55.6% WR, n=34,403)
  score=+5 →  -0.7 pips avg over 24h (50.1% WR, n=62,348)
Trend approximately monotonic from -5 to +5.

This script trades that signal:
  Entry: LONG when alignment_score CROSSES INTO -5 (prev != -5, now = -5)
         SHORT when crosses into +5 (prev != +5, now = +5)
  Hold:  fixed N M5 bars (sweep 96 / 288 / 576 = 8h / 24h / 48h)
  Exit:  market at hold-window end, bid (long) / ask (short)
  Spread: round-trip cost via bid/ask at entry + exit
  Re-entry: only after current trade closes

Sweep:
  HOLD_BARS ∈ {96, 288, 576}     (8h, 24h, 48h on M5)
  MODE      ∈ {long_only, both}  (whether to take short side too —
                                  bin study showed short side ~flat-to-neg)

All 12 pairs. IS/OOS 70/30. OOS-only reported.
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
IS_FRAC = 0.70

HOLD_BARS_LIST = [96, 288, 576]


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def per_tf_sign(df_m5, freq):
    """Sign of (last completed TF bar's close - prior TF bar's close), aligned
    to M5 grid. R1-clean: shift by 1 before reindex so we use only completed
    bars on the higher TF."""
    closes = df_m5["close"].resample(freq).last().dropna()
    diffs  = closes.diff()
    sign_shifted = np.sign(diffs).shift(1)
    return sign_shifted.reindex(df_m5.index, method="ffill").fillna(0).values.astype(np.int8)


@njit(cache=True)
def sim_fade(close, bid, ask, sp, score, pip, sp_gate, hold_bars, mode_long_only):
    """Cross-into-extreme entry, fixed-hold exit.

    mode_long_only = 1 → only enter on score==-5; skip score==+5 events
                     0 → both directions (short on +5)
    """
    n = len(close)
    pnl_arr  = np.empty(n, dtype=np.float64)
    hold_arr = np.empty(n, dtype=np.int32)
    dir_arr  = np.empty(n, dtype=np.int8)
    score_arr= np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0
    entry_score = 0
    prev_score = 99   # sentinel — first bar won't trigger

    for i in range(1, n):
        s = score[i]
        if in_trade:
            if i - entry_bar >= hold_bars:
                exit_px = bid[i] if direction == 1 else ask[i]
                pnl_arr[count]  = (exit_px - entry_price) / pip * direction - sp[i]
                hold_arr[count] = i - entry_bar
                dir_arr[count]  = direction
                score_arr[count]= entry_score
                count += 1
                in_trade = False
        else:
            # cross INTO ±5
            if s == -5 and prev_score != -5 and sp[i] <= sp_gate:
                entry_price = ask[i]
                entry_bar = i
                direction = 1   # LONG fade
                entry_score = -5
                in_trade = True
            elif (not mode_long_only) and s == 5 and prev_score != 5 and sp[i] <= sp_gate:
                entry_price = bid[i]
                entry_bar = i
                direction = -1  # SHORT fade
                entry_score = 5
                in_trade = True
        prev_score = s

    # If still open at end-of-OOS, close at market
    if in_trade:
        i = n - 1
        exit_px = bid[i] if direction == 1 else ask[i]
        pnl_arr[count]  = (exit_px - entry_price) / pip * direction - sp[i]
        hold_arr[count] = i - entry_bar
        dir_arr[count]  = direction
        score_arr[count]= entry_score
        count += 1

    return pnl_arr[:count], hold_arr[:count], dir_arr[:count], score_arr[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 1000
    c = np.linspace(1.0, 1.05, n)
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n)
    score = np.zeros(n, dtype=np.int8)
    score[100] = -5; score[500] = 5
    sim_fade(c, b, a, sp, score, 0.0001, 2.0, 96, 1)
    sim_fade(c, b, a, sp, score, 0.0001, 2.0, 96, 0)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]

    m_sign  = per_tf_sign(df_m5, "1MS")
    w_sign  = per_tf_sign(df_m5, "1W")
    d_sign  = per_tf_sign(df_m5, "1D")
    h_sign  = per_tf_sign(df_m5, "1h")
    m5_sign = per_tf_sign(df_m5, "5min")
    score = (m_sign + w_sign + d_sign + h_sign + m5_sign).astype(np.int8)

    close = df_m5["close"].values.astype(np.float64)
    bid   = df_m5["bid_c"].values.astype(np.float64)
    ask   = df_m5["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 288.0

    for hold in HOLD_BARS_LIST:
        for mode in ("long_only", "both"):
            p, h, d, sc = sim_fade(
                close[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
                score[n_is:], pip, float(sg), int(hold),
                1 if mode == "long_only" else 0)
            n = len(p)
            row = dict(pair=pair, hold_bars=hold, mode=mode,
                       n=n, days=round(oos_days, 1))
            if n == 0:
                row.update(ppd=0.0, wr=0.0, mdd=0.0, mean_pnl=0.0,
                           max_loss=0.0, max_win=0.0, sum_pips=0.0,
                           n_long=0, n_short=0,
                           wr_long=0.0, wr_short=0.0,
                           ppd_long=0.0, ppd_short=0.0)
            else:
                wr = (p > 0).sum() / n * 100
                ppd = p.sum() / oos_days
                mdd = max_dd(p)
                long_mask  = (d == 1)
                short_mask = (d == -1)
                n_long  = int(long_mask.sum())
                n_short = int(short_mask.sum())
                p_long  = p[long_mask] if n_long else np.array([0.0])
                p_short = p[short_mask] if n_short else np.array([0.0])
                row.update(
                    ppd=round(ppd, 2), wr=round(wr, 1),
                    mdd=round(mdd, 1),
                    mean_pnl=round(float(p.mean()), 2),
                    max_loss=round(float(p.min()), 1),
                    max_win=round(float(p.max()), 1),
                    sum_pips=round(float(p.sum()), 1),
                    n_long=n_long, n_short=n_short,
                    wr_long=round((p_long > 0).mean() * 100, 1) if n_long else 0.0,
                    wr_short=round((p_short > 0).mean() * 100, 1) if n_short else 0.0,
                    ppd_long=round(float(p_long.sum()) / oos_days, 2),
                    ppd_short=round(float(p_short.sum()) / oos_days, 2),
                )
            all_rows.append(row)
    del df_m5
    gc.collect()


def main():
    warmup_jit()
    print("M/W/D/H/M5 alignment-score FADE strategy")
    print(f"  hold bars:  {HOLD_BARS_LIST}  (96=8h, 288=24h, 576=48h)")
    print(f"  modes:      long_only (fade -5 only) + both (fade -5 long, +5 short)")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "mwdh_fade.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Portfolio summary by hold × mode ───────────────────────────────────
    print("\n" + "="*110)
    print("  Portfolio totals (12 pairs) by hold × mode")
    print("="*110)
    print(f"  {'hold':>5} {'mode':<10}  {'Σ pips':>9}  {'ppd':>8}  "
          f"{'Σ n':>6}  {'pairs+':>7}  {'mean WR%':>9}  "
          f"{'long Σ':>8}  {'short Σ':>9}")
    for hold in HOLD_BARS_LIST:
        for mode in ("long_only", "both"):
            sub = df[(df.hold_bars == hold) & (df["mode"] == mode)]
            total = sub.sum_pips.sum()
            avg_days = sub.days.mean()
            ppd = total / avg_days
            n_total = sub.n.sum()
            wr_mean = sub.wr.mean() if n_total else 0
            pos = int((sub.ppd > 0).sum())
            ppd_long_total  = sub.ppd_long.sum()
            ppd_short_total = sub.ppd_short.sum()
            print(f"  {hold:>5} {mode:<10}  {total:>+9.1f}  {ppd:>+8.2f}  "
                  f"{int(n_total):>6}  {pos:>3}/12   {wr_mean:>8.1f}%  "
                  f"{ppd_long_total:>+8.2f}  {ppd_short_total:>+9.2f}")

    # ── Per-pair detail at best config ────────────────────────────────────
    sums = (df.groupby(["hold_bars","mode"])["sum_pips"].sum()
              .reset_index().sort_values("sum_pips", ascending=False))
    best_h = int(sums.iloc[0]["hold_bars"])
    best_m = sums.iloc[0]["mode"]
    print("\n" + "="*110)
    print(f"  Per-pair detail at best portfolio config: hold={best_h} mode={best_m}")
    print("="*110)
    sub = df[(df.hold_bars == best_h) & (df["mode"] == best_m)].sort_values("ppd", ascending=False)
    print(f"  {'pair':<10}{'n':>5}{'n_lng':>6}{'n_sht':>6}"
          f"{'Σ pips':>10}{'ppd':>8}{'WR%':>6}"
          f"{'wr_lng':>8}{'wr_sht':>8}"
          f"{'mean pnl':>10}{'MDD':>8}{'max_loss':>10}{'max_win':>10}")
    for _, r in sub.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>5}{int(r['n_long']):>6}{int(r['n_short']):>6}"
              f"{r['sum_pips']:>+10.1f}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['wr_long']:>8.1f}{r['wr_short']:>8.1f}"
              f"{r['mean_pnl']:>+10.2f}{r['mdd']:>8.1f}"
              f"{r['max_loss']:>+10.1f}{r['max_win']:>+10.1f}")


if __name__ == "__main__":
    main()
