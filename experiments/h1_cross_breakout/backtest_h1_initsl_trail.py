#!/usr/bin/env python3
"""
v3 + initial SL — protect ALL pairs from adverse move before trail arms.

Mechanics
---------
Entry  : v3 winning entry — sma=7, n_small=1, thld=2.0×ATR (combined-move)
Initial SL : at entry, hard stop at entry ∓ init_mult × ATR_at_entry
             (long: below entry, short: above entry)
Trail  : armed when MFE ≥ activate × ATR_at_entry.
         Once armed, trail_px = HWM ∓ trail_mult × ATR_at_entry.
         Trail is always ≥ entry after activation (because activate >> trail
         in this grid).
Exit precedence per bar:
   1. Trail stop (if armed) hit
   2. Initial SL hit  (relevant only before activation; after, trail is tighter)
   3. Max-hold cap (96 H1 bars)
   4. End-of-OOS market close

Sweep
-----
ACTIVATE   ∈ {1.0, 1.5, 2.0}
TRAIL      ∈ {0.05, 0.075, 0.10, 0.15, 0.20}      ← extends below v3
INIT_SL    ∈ {0.50, 1.00, 1.50, 2.00}             ← NEW
All 12 pairs, OOS-only.
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

SMA_N      = 7
N_SMALL    = 1
THLD_MULT  = 2.0

ACTIVATE_MULTS  = [1.0, 1.5, 2.0]
TRAIL_MULTS     = [0.05, 0.075, 0.10, 0.15, 0.20]
INIT_SL_MULTS   = [0.0, 0.5, 1.0, 1.5, 2.0]   # 0.0 = no init SL


@njit(cache=True)
def _sim(close, sma, high, low, bid, ask, sp, atr, pip,
         thld_arr, n_small,
         activate_mult, trail_mult, init_sl_mult,
         max_hold, sp_gate):
    n = len(close)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)  # 0=trail 1=init_sl 2=maxhold 3=eod
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0; hwm_pips = 0.0; armed = False
    init_sl_px = 0.0
    min_start = n_small + 3

    for t in range(min_start, n):
        if in_trade:
            excur = (close[t] - ep) / pip * dir_
            if excur > hwm_pips: hwm_pips = excur

            # Arm trail when MFE crosses activation
            if not armed and atr_entry > 0.0:
                if hwm_pips >= activate_mult * atr_entry / pip:
                    armed = True

            exited = False

            # 1) Trail stop (if armed) — checked first since it's the tightest
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

            # 2) Initial SL (only relevant before trail arms; once armed,
            #    trail is always tighter under this grid)
            if (not exited) and init_sl_mult > 0.0:
                if dir_ == 1 and low[t] <= init_sl_px:
                    pnl_out[count]  = (init_sl_px - ep) / pip - sp[t]
                    hold_out[count] = t - ei; type_out[count] = 1
                    count += 1; in_trade=False; armed=False; hwm_pips=0.0
                    exited = True
                elif dir_ == -1 and high[t] >= init_sl_px:
                    pnl_out[count]  = (ep - init_sl_px) / pip - sp[t]
                    hold_out[count] = t - ei; type_out[count] = 1
                    count += 1; in_trade=False; armed=False; hwm_pips=0.0
                    exited = True

            # 3) Max-hold
            if (not exited) and (t - ei) >= max_hold:
                exit_px = bid[t] if dir_ == 1 else ask[t]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
                hold_out[count] = t - ei; type_out[count] = 2
                count += 1; in_trade=False; armed=False; hwm_pips=0.0
            continue

        # ── Entry check ─────────────────────────────────────────────────────
        thld = thld_arr[t]
        if np.isnan(thld) or thld <= 0.0:
            continue
        if np.isnan(atr[t]):
            continue
        if sp[t] > sp_gate:
            continue

        c_t  = close[t]; c_1 = close[t-1]
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

        if long_ok:
            ep = ask[t]; dir_ = 1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False
            init_sl_px = ep - init_sl_mult * atr_entry
        elif short_ok:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False
            init_sl_px = ep + init_sl_mult * atr_entry

    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei; type_out[count] = 3
        count += 1

    return pnl_out[:count], hold_out[:count], type_out[:count]


def warmup_jit():
    n = 300
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0015)
    thld = np.full(n, 30.0 * 0.0001)
    _sim(c, s, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.5, 0.10, 1.0, 96, 2.0)
    _sim(c, s, h, l, b, a, sp, atr, 0.0001, thld, 1, 1.5, 0.05, 2.0, 96, 2.0)


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
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 24.0
    atr_price = wilder_atr(high, low, close, ATR_PERIOD)
    sma = pd.Series(close).rolling(SMA_N).mean().values.astype(np.float64)
    thld_arr = THLD_MULT * atr_price

    for act in ACTIVATE_MULTS:
        for trail in TRAIL_MULTS:
            for isl in INIT_SL_MULTS:
                p, h, t = _sim(
                    close[n_is:], sma[n_is:],
                    high[n_is:], low[n_is:],
                    bid[n_is:], ask[n_is:], sp[n_is:], atr_price[n_is:],
                    pip, thld_arr[n_is:], int(N_SMALL),
                    float(act), float(trail), float(isl),
                    int(MAX_HOLD_H1), float(sg))
                n = len(p)
                if n == 0:
                    all_rows.append(dict(pair=pair, activate=act, trail=trail, init_sl=isl,
                                         n=0, ppd=0.0, wr=0.0, mdd=0.0, calmar=0.0,
                                         mean_hold_h1=0.0, max_loss=0.0, max_win=0.0,
                                         trail_pct=0.0, init_sl_pct=0.0,
                                         days=round(oos_days,1)))
                    continue
                wr  = (p>0).sum() / n * 100
                ppd = p.sum() / oos_days
                mdd = max_dd(p)
                cal = ppd / mdd if mdd > 0 else 0.0
                all_rows.append(dict(pair=pair, activate=act, trail=trail, init_sl=isl,
                                     n=n, ppd=round(ppd,2), wr=round(wr,1),
                                     mdd=round(mdd,1), calmar=round(cal,2),
                                     mean_hold_h1=round(float(h.mean()),1),
                                     max_loss=round(float(p.min()),1),
                                     max_win=round(float(p.max()),1),
                                     trail_pct=round((t==0).sum()/n*100,1),
                                     init_sl_pct=round((t==1).sum()/n*100,1),
                                     days=round(oos_days,1)))
    del close, bid, ask, sp, h1, atr_price, high, low, sma, thld_arr
    gc.collect()


def main():
    warmup_jit()
    print("v3 + initial SL — extended trail × activate × init_sl sweep")
    print(f"  entry: sma={SMA_N} ns={N_SMALL} thld={THLD_MULT}×ATR (fixed)")
    print(f"  activate ∈ {ACTIVATE_MULTS}")
    print(f"  trail    ∈ {TRAIL_MULTS}")
    print(f"  init_sl  ∈ {INIT_SL_MULTS}  ×ATR_at_entry")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "h1_initsl_trail_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── 12-pair portfolio ────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  Top configs ALL 12 pairs (sorted by Σ p/d, top 20)")
    print("="*100)
    g = (df.groupby(["activate","trail","init_sl"])
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"),
                 worst_max_loss=("max_loss","min"),
                 mean_trl_pct=("trail_pct","mean"),
                 mean_isl_pct=("init_sl_pct","mean"))
            .reset_index().sort_values("sum_ppd", ascending=False))
    print(f"  {'act':>4}  {'trail':>5}  {'iSL':>4}  {'Σ ppd':>7}  {'pos':>3}/12 {'Σn':>5}  {'WR%':>5}  {'worst':>8}  {'trl%':>5}  {'iSL%':>5}")
    for _, r in g.head(20).iterrows():
        print(f"  {r['activate']:>4.2f}  {r['trail']:>5.3f}  {r['init_sl']:>4.2f}  "
              f"{r['sum_ppd']:>+7.2f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>5d}  "
              f"{r['mean_wr']:>5.1f}  {r['worst_max_loss']:>+8.1f}  "
              f"{r['mean_trl_pct']:>5.1f}  {r['mean_isl_pct']:>5.1f}")

    # ── 7-winner subset ──────────────────────────────────────────────────
    df_sub = df[df.pair.isin(WINNERS_7)]
    g_sub = (df_sub.groupby(["activate","trail","init_sl"])
                .agg(sum_ppd=("ppd","sum"),
                     n_pos=("ppd", lambda x: int((x>0).sum())),
                     total_n=("n","sum"),
                     mean_wr=("wr","mean"),
                     worst_max_loss=("max_loss","min"))
                .reset_index().sort_values("sum_ppd", ascending=False))
    print("\n" + "="*100)
    print("  Top configs on 7-WINNER subset (top 15)")
    print("="*100)
    print(f"  {'act':>4}  {'trail':>5}  {'iSL':>4}  {'Σ ppd':>7}  {'pos':>3}/7  {'Σn':>5}  {'WR%':>5}  {'worst':>8}")
    for _, r in g_sub.head(15).iterrows():
        print(f"  {r['activate']:>4.2f}  {r['trail']:>5.3f}  {r['init_sl']:>4.2f}  "
              f"{r['sum_ppd']:>+7.2f}  {int(r['n_pos']):>3d}/7  {int(r['total_n']):>5d}  "
              f"{r['mean_wr']:>5.1f}  {r['worst_max_loss']:>+8.1f}")

    # ── Per-pair detail at top 12-pair config ────────────────────────────
    top = g.iloc[0]
    print("\n" + "="*100)
    print(f"  Per-pair OOS at top 12-config: act={top['activate']} trail={top['trail']} iSL={top['init_sl']}")
    print("="*100)
    sub = df[(df.activate==top['activate'])&(df.trail==top['trail'])&(df.init_sl==top['init_sl'])]
    sub = sub.sort_values("ppd", ascending=False)
    print(f"  {'pair':<10}{'n':>4}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'hold':>6}"
          f"{'trl%':>6}{'iSL%':>6}{'max_loss':>10}{'max_win':>9}")
    for _, r in sub.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>4}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['mdd']:>7.1f}{r['mean_hold_h1']:>6.1f}{r['trail_pct']:>6.1f}"
              f"{r['init_sl_pct']:>6.1f}{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


if __name__ == "__main__":
    main()
