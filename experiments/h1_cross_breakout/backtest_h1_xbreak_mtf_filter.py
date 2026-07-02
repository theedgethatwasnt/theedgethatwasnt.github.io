#!/usr/bin/env python3
"""
(c) v3 H1 cross-breakout WITH MTF momentum-agreement filter at entry.

Gate the v3 entry on the momentum-agreement state at the entry bar:
  Long allowed:  m5_mom > 0 AND h1_mom > 0
  Short allowed: m5_mom < 0 AND h1_mom < 0

At H1 resolution:
  m5_mom = close[i] - close[i-1]   (no M5 close here; use H1 close — 1 H1 = 12 M5)
  h1_mom = close[i] - close[i-1]   ← wait this is same as m5 at H1 resolution
  Use h1_mom = close[i] - close[i-1]  (1 H1 back)
       h4_mom = close[i] - close[i-4]  (4 H1 back)
Filter requires both signs to match the entry direction.

(For an actually-multi-TF filter at H1 entry, the natural is H1 + H4 — same
as the (b) experiment. So this (c) really tests: does requiring H1+H4
sign-agreement at the v3 entry improve the per-pair p/d?)

Fixed entry: sma=7, n_small=1, thld=2.0×ATR, act=1.5, trail=0.05, no iSL,
max_hold=96 H1 bars. All 12 pairs.
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_h1_xbreak_atrail import (
    resample_h1, wilder_atr, max_dd,
    ALL_PAIRS, JPY, SP_GATES, pip_sz, IS_FRAC, MAX_HOLD_H1, ATR_PERIOD,
)

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

WINNERS_7 = {"AUD_JPY","NZD_JPY","CHF_JPY","EUR_GBP","EUR_JPY","EUR_USD","GBP_USD"}

SMA_N         = 7
N_SMALL       = 1
THLD_MULT     = 2.0
ACTIVATE_MULT = 1.5
TRAIL_MULT    = 0.05
H1_LAG        = 1
H4_LAG        = 4


@njit(cache=True)
def _sim(close, sma, high, low, bid, ask, sp, atr, pip,
         thld_arr, n_small,
         activate_mult, trail_mult, max_hold, sp_gate,
         h1_lag, h4_lag, filter_on):
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    skipped_filter = 0
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0; hwm_pips = 0.0; armed = False
    min_start = max(n_small + 3, h4_lag + 1)

    for t in range(min_start, n):
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

        thld = thld_arr[t]
        if np.isnan(thld) or thld <= 0.0:
            continue
        if np.isnan(atr[t]):
            continue
        if sp[t] > sp_gate:
            continue

        c_t = close[t]; c_1 = close[t-1]
        below_idx = t - n_small - 3
        cross_idx = t - n_small - 2
        if below_idx < 0:
            continue
        combined_move = c_1 - close[cross_idx]

        long_xover = (close[below_idx] < sma[below_idx]) and \
                     (close[cross_idx] > sma[cross_idx])
        long_move  = combined_move > thld
        long_curr  = (c_t - c_1) > 0.0
        long_ok = long_xover and long_move and long_curr

        short_xover = (close[below_idx] > sma[below_idx]) and \
                      (close[cross_idx] < sma[cross_idx])
        short_move  = combined_move < -thld
        short_curr  = (c_t - c_1) < 0.0
        short_ok = short_xover and short_move and short_curr

        # ── MTF filter at entry bar ──
        if filter_on:
            h1_diff = close[t] - close[t - h1_lag]
            h4_diff = close[t] - close[t - h4_lag]
            if long_ok:
                if not (h1_diff > 0.0 and h4_diff > 0.0):
                    skipped_filter += 1; continue
            elif short_ok:
                if not (h1_diff < 0.0 and h4_diff < 0.0):
                    skipped_filter += 1; continue

        if long_ok:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False
        elif short_ok:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False

    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei; type_out[count] = 2
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count], skipped_filter


def warmup_jit():
    n = 300
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0015); thld = np.full(n, 30.0 * 0.0001)
    _sim(c, s, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.5, 0.05, 96, 2.0, 1, 4, 0)
    _sim(c, s, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.5, 0.05, 96, 2.0, 1, 4, 1)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    h1 = resample_h1(df_m5); del df_m5
    pip = pip_sz(pair); sg = SP_GATES[pair]
    high  = h1["high"].values.astype(np.float64)
    low   = h1["low"].values.astype(np.float64)
    close = h1["close"].values.astype(np.float64)
    bid   = h1["bid_c"].values.astype(np.float64)
    ask   = h1["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 24.0
    atr_price = wilder_atr(high, low, close, ATR_PERIOD)
    sma = pd.Series(close).rolling(SMA_N).mean().values.astype(np.float64)
    thld_arr = THLD_MULT * atr_price

    for filter_on in (0, 1):
        p, h, t, sk = _sim(
            close[n_is:], sma[n_is:], high[n_is:], low[n_is:],
            bid[n_is:], ask[n_is:], sp[n_is:], atr_price[n_is:],
            pip, thld_arr[n_is:], int(N_SMALL),
            float(ACTIVATE_MULT), float(TRAIL_MULT),
            int(MAX_HOLD_H1), float(sg), H1_LAG, H4_LAG, int(filter_on))
        n = len(p)
        if n == 0:
            all_rows.append(dict(pair=pair, filter_on=filter_on, n=0, ppd=0.0,
                                 wr=0.0, mdd=0.0, mean_hold_h1=0.0, mean_pnl=0.0,
                                 max_loss=0.0, max_win=0.0, sum_pips=0.0,
                                 skipped=sk, days=round(oos_days, 1)))
            continue
        wr = (p>0).sum() / n * 100
        ppd = p.sum() / oos_days
        mdd = max_dd(p)
        all_rows.append(dict(pair=pair, filter_on=filter_on, n=n,
                             ppd=round(ppd, 2), wr=round(wr, 1),
                             mdd=round(mdd, 1),
                             mean_hold_h1=round(float(h.mean()), 1),
                             mean_pnl=round(float(p.mean()), 2),
                             max_loss=round(float(p.min()), 1),
                             max_win=round(float(p.max()), 1),
                             sum_pips=round(float(p.sum()), 1),
                             skipped=int(sk),
                             days=round(oos_days, 1)))


def main():
    warmup_jit()
    print("(c) v3 H1 cross-breakout + MTF (H1+H4) momentum-agreement filter at entry")
    print(f"  Filter: long requires h1_mom>0 AND h4_mom>0; short mirror")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "h1_xbreak_mtf_filter.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")

    print("\n" + "="*100)
    print("  Effect of MTF (H1+H4) agreement filter at v3 entry")
    print("="*100)
    print(f"  {'pair':<10}{'n_base':>8}{'n_filt':>8}{'ppd_base':>10}{'ppd_filt':>10}"
          f"{'Δ ppd':>9}{'wr_base':>9}{'wr_filt':>9}{'skipped':>9}")
    for pair in ALL_PAIRS:
        b = df[(df.pair == pair) & (df.filter_on == 0)].iloc[0]
        f = df[(df.pair == pair) & (df.filter_on == 1)].iloc[0]
        marker = "*" if pair in WINNERS_7 else " "
        print(f"  {marker}{pair:<9}{int(b['n']):>8}{int(f['n']):>8}"
              f"{b['ppd']:>+10.2f}{f['ppd']:>+10.2f}"
              f"{f['ppd']-b['ppd']:>+9.2f}{b['wr']:>9.1f}{f['wr']:>9.1f}"
              f"{int(f['skipped']):>9}")

    # Portfolio totals
    print("\n  " + "─"*60)
    for tag, lbl in [(0, "baseline"), (1, "with MTF filter")]:
        sub = df[df.filter_on == tag]
        total = sub.sum_pips.sum()
        days = sub.days.mean()
        ppd = total / days
        n = sub.n.sum()
        pos = int((sub.ppd > 0).sum())
        # 7-winner subset
        sub7 = sub[sub.pair.isin(WINNERS_7)]
        total7 = sub7.sum_pips.sum()
        ppd7 = total7 / days
        pos7 = int((sub7.ppd > 0).sum())
        print(f"  {lbl:<18}  12-pair Σ={total:+.1f}p ppd={ppd:+.2f}  pairs+={pos}/12  "
              f"|  7-pair Σ={total7:+.1f}p ppd={ppd7:+.2f}  pairs+={pos7}/7")


if __name__ == "__main__":
    main()
