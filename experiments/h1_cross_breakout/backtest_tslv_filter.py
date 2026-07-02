#!/usr/bin/env python3
"""
TSLV (Time-Since-Last-Visit) + SMA(TSLV) as predictor for v3 H1 cross-breakout.

For each H1 bar, compute TSLV: the smallest k>0 such that some prior bar i-k
had its [low, high] range containing the current bar's close, within a
tolerance of ε = TOUCH_FRAC × ATR14_at_current_bar (scale-aware).

  TSLV high   => price hasn't been here in a long time (breakout territory)
  TSLV low    => price oscillating, level recently touched

We also compute SMA(N) of TSLV for N ∈ {5, 10, 20}. A high SMA means TSLV
has been *sustained* high for several bars (structural breakout) — versus
a single-bar spike (could be noise).

Backtest: re-detect every v3 cross-breakout entry on each pair, capture
TSLV and SMA-TSLV at entry, run the v3 simulator, then bin trades by
quantile to see if TSLV (or its SMA) predicts win-rate / per-trade pnl.

Fixed v3 config:
  sma=7, n_small=1, thld=2.0×ATR, activate=1.5×ATR, trail=0.05×ATR,
  init_sl=NONE, max_hold=96 H1 bars

Pairs: 7-winner subset (AUD_JPY, NZD_JPY, CHF_JPY, EUR_GBP,
                       EUR_JPY, EUR_USD, GBP_USD)
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
    JPY, SP_GATES, pip_sz, IS_FRAC, MAX_HOLD_H1, ATR_PERIOD,
)

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

WINNERS_7 = ["AUD_JPY","NZD_JPY","CHF_JPY","EUR_GBP","EUR_JPY","EUR_USD","GBP_USD"]

# v3 winning config (fixed)
SMA_N         = 7
N_SMALL       = 1
THLD_MULT     = 2.0
ACTIVATE_MULT = 1.5
TRAIL_MULT    = 0.05
INIT_SL       = 0.0   # 0 = no init SL (confirmed best in prior sweep)

# TSLV parameters — pip-grid bucketing
TSLV_GRID_PIPS_LIST = [5, 10]      # grid sizes to sweep (pips per bin)
TSLV_SMA_NS         = [5, 10, 20]  # SMA windows on the TSLV time series


# ── TSLV via pip-grid bucketing ──────────────────────────────────────────────
#
# Bin price space into fixed-pip cells (e.g., 5-pip grid). Each cell holds
# the index of the most-recent bar that visited it (where "visit" = the
# bar's [low, high] range covers the cell). At each new bar i:
#   1. Query: find the bin containing close[i]; TSLV[i] = i − last_visit[bin]
#      (0 if the bin has never been touched before bar i).
#   2. Update: mark every bin in [low[i], high[i]] as visited at i.
# Order matters — query BEFORE update so the current bar doesn't self-touch.

@njit(cache=True)
def _tslv_grid_kernel(high, low, close, grid_step, base_price, n_bins):
    n = len(close)
    out = np.zeros(n, dtype=np.int32)
    last_visit = np.full(n_bins, -1, dtype=np.int32)

    for i in range(n):
        # ── Query bin of current close
        b_close = int((close[i] - base_price) / grid_step)
        if 0 <= b_close < n_bins:
            lv = last_visit[b_close]
            if lv >= 0:
                out[i] = i - lv
        # ── Update: mark all bins in [low, high]
        b_lo = int((low[i]  - base_price) / grid_step)
        b_hi = int((high[i] - base_price) / grid_step)
        if b_lo < 0: b_lo = 0
        if b_hi >= n_bins: b_hi = n_bins - 1
        for b in range(b_lo, b_hi + 1):
            last_visit[b] = i
    return out


def compute_tslv_grid(high, low, close, grid_pips, pip):
    """TSLV with a fixed-pip grid. Returns int32 array of bars-since-last-visit."""
    grid_step = grid_pips * pip
    # Pad the grid so all bars fit even if there's a drift
    base_price = float(low.min()) - 100 * grid_step
    top_price  = float(high.max()) + 100 * grid_step
    n_bins = int((top_price - base_price) / grid_step) + 2
    return _tslv_grid_kernel(high, low, close, grid_step, base_price, n_bins)


# ── v3 simulator capturing TSLV at entry ─────────────────────────────────────

@njit(cache=True)
def _sim_capture_tslv(close, sma, high, low, bid, ask, sp, atr, pip,
                      thld_arr, tslv, tslv_sma, n_small,
                      activate_mult, trail_mult, max_hold, sp_gate):
    n = len(close)
    pnl_out      = np.empty(n, dtype=np.float64)
    hold_out     = np.empty(n, dtype=np.int32)
    type_out     = np.empty(n, dtype=np.int8)
    dir_out      = np.empty(n, dtype=np.int8)
    tslv_at_e    = np.empty(n, dtype=np.float64)
    tslv_sma_at_e= np.empty(n, dtype=np.float64)
    count = 0
    in_trade = False
    dir_ = 0; ep = 0.0; ei = 0
    atr_entry = 0.0; hwm_pips = 0.0; armed = False
    min_start = n_small + 3

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
            dir_out[count]      = 1
            tslv_at_e[count]    = float(tslv[t])
            tslv_sma_at_e[count]= tslv_sma[t]
        elif short_ok:
            ep = bid[t]; dir_ = -1; ei = t; in_trade = True
            atr_entry = atr[t]; hwm_pips = 0.0; armed = False
            dir_out[count]      = -1
            tslv_at_e[count]    = float(tslv[t])
            tslv_sma_at_e[count]= tslv_sma[t]

    if in_trade:
        t = n - 1
        exit_px = bid[t] if dir_ == 1 else ask[t]
        pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[t]
        hold_out[count] = t - ei; type_out[count] = 2
        count += 1

    return (pnl_out[:count], hold_out[:count], type_out[:count],
            dir_out[:count], tslv_at_e[:count], tslv_sma_at_e[:count])


def warmup_jit():
    n = 300
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    s = c.copy(); h = c + 0.0005; l = c - 0.0005
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); atr = np.full(n, 0.0015)
    thld = np.full(n, 30.0 * 0.0001)
    tslv = np.zeros(n, dtype=np.int32)
    tslv_sma = np.zeros(n)
    _tslv_grid_kernel(h, l, c, 0.0005, 0.5, 5000)
    _sim_capture_tslv(c, s, h, l, b, a, sp, atr, 0.0001, thld,
                      tslv, tslv_sma, 1, 1.5, 0.05, 96, 2.0)


def run_pair(pair):
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

    sweep_rows = []
    # Sweep grid size × SMA window
    for grid_pips in TSLV_GRID_PIPS_LIST:
        tslv = compute_tslv_grid(high, low, close, grid_pips, pip)
        for sma_n_tslv in TSLV_SMA_NS:
            tslv_sma = pd.Series(tslv.astype(np.float64)).rolling(sma_n_tslv).mean().values

            p, h, t, d, t_e, t_sma_e = _sim_capture_tslv(
                close[n_is:], sma[n_is:], high[n_is:], low[n_is:],
                bid[n_is:], ask[n_is:], sp[n_is:], atr_price[n_is:],
                pip, thld_arr[n_is:],
                tslv[n_is:], tslv_sma[n_is:],
                int(N_SMALL), float(ACTIVATE_MULT), float(TRAIL_MULT),
                int(MAX_HOLD_H1), float(sg))
            n = len(p)
            for k in range(n):
                sweep_rows.append(dict(
                    pair=pair,
                    grid_pips=grid_pips,
                    tslv_sma_n=sma_n_tslv,
                    tslv_at_entry=int(t_e[k]),
                    tslv_sma_at_entry=round(float(t_sma_e[k]), 1),
                    pnl=round(float(p[k]), 2),
                    win=int(p[k] > 0),
                    hold=int(h[k]),
                    direction=int(d[k]),
                    exit_type=int(t[k]),
                ))

    del high, low, close, bid, ask, sp, atr_price, sma, thld_arr
    gc.collect()
    return sweep_rows, oos_days


def main():
    warmup_jit()
    print("TSLV + SMA(TSLV) as predictor on v3 H1 cross-breakout entries")
    print(f"  pairs: {WINNERS_7}")
    print(f"  v3 fixed config: sma={SMA_N} ns={N_SMALL} thld={THLD_MULT}xATR "
          f"act={ACTIVATE_MULT}xATR trail={TRAIL_MULT}xATR no-iSL")
    print(f"  TSLV: pip-grid bucket, range-touch (low..high marks all bins)")
    print(f"  TSLV grid (pips): {TSLV_GRID_PIPS_LIST}")
    print(f"  SMA(TSLV) windows: {TSLV_SMA_NS}")
    all_rows = []
    days_per_pair = {}
    t0 = time.time()
    for pair in WINNERS_7:
        ts = time.time()
        rows, oos_days = run_pair(pair)
        all_rows.extend(rows)
        days_per_pair[pair] = oos_days
        per_combo = len(rows) // (len(TSLV_GRID_PIPS_LIST) * len(TSLV_SMA_NS))
        print(f"  {pair}: {time.time()-ts:.1f}s  trades={per_combo}", flush=True)
    df = pd.DataFrame(all_rows)
    out_csv = OUT / "tslv_filter.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")
    oos_days_avg = np.mean(list(days_per_pair.values()))

    # ── (1) TSLV at entry vs outcome, per grid size ───────────────────────
    print("\n" + "="*100)
    print(f"  TSLV-at-entry vs outcome, by GRID SIZE  (all 7 pairs combined, OOS)")
    print("="*100)
    for grid_pips in TSLV_GRID_PIPS_LIST:
        # One copy per (pair, trade); take the canonical sma_n_tslv to dedup
        base = df[(df.grid_pips == grid_pips) & (df.tslv_sma_n == TSLV_SMA_NS[0])].copy()
        if len(base) < 10:
            print(f"\n  --- grid={grid_pips}p --- too few trades ({len(base)})")
            continue
        try:
            base["tslv_bin"] = pd.qcut(base.tslv_at_entry, q=5,
                                        labels=["Q1 low","Q2","Q3","Q4","Q5 high"],
                                        duplicates="drop")
        except Exception:
            base["tslv_bin"] = "all"
        g = base.groupby("tslv_bin", observed=False).agg(
            n=("pnl","size"), wr=("win","mean"),
            mean_pnl=("pnl","mean"), sum_pnl=("pnl","sum"),
            mean_tslv=("tslv_at_entry","mean")
        ).reset_index()
        g["sum_ppd"] = g["sum_pnl"] / oos_days_avg
        g["wr"] = g["wr"] * 100
        print(f"\n  --- grid={grid_pips}p   total trades = {len(base)} ---")
        print(f"  {'bin':<10}  {'n':>5}  {'mean_TSLV':>10}  {'WR%':>5}  {'mean pnl':>9}  {'Σ ppd':>7}")
        for _, r in g.iterrows():
            print(f"  {str(r.tslv_bin):<10}  {int(r.n):>5}  {r.mean_tslv:>10.0f}  "
                  f"{r.wr:>5.1f}  {r.mean_pnl:>+9.2f}  {r.sum_ppd:>+7.2f}")

    # ── (2) SMA(TSLV) at entry, per grid x sma_n ──────────────────────────
    print("\n" + "="*100)
    print(f"  SMA(TSLV)-at-entry vs outcome  (per grid_pips × SMA window)")
    print("="*100)
    for grid_pips in TSLV_GRID_PIPS_LIST:
        for sma_n_tslv in TSLV_SMA_NS:
            sub = df[(df.grid_pips == grid_pips) & (df.tslv_sma_n == sma_n_tslv)].copy()
            if len(sub) < 10:
                continue
            try:
                sub["sma_bin"] = pd.qcut(sub.tslv_sma_at_entry, q=5,
                                          labels=["Q1 low","Q2","Q3","Q4","Q5 high"],
                                          duplicates="drop")
            except Exception:
                sub["sma_bin"] = "all"
            gs = sub.groupby("sma_bin", observed=False).agg(
                n=("pnl","size"), wr=("win","mean"),
                mean_pnl=("pnl","mean"), sum_pnl=("pnl","sum"),
                mean_sma=("tslv_sma_at_entry","mean")
            ).reset_index()
            gs["sum_ppd"] = gs["sum_pnl"] / oos_days_avg
            gs["wr"] = gs["wr"] * 100
            print(f"\n  --- grid={grid_pips}p   SMA(TSLV, N={sma_n_tslv}) ---")
            print(f"  {'bin':<10}  {'n':>5}  {'mean_SMA':>9}  {'WR%':>5}  {'mean pnl':>9}  {'Σ ppd':>7}")
            for _, r in gs.iterrows():
                print(f"  {str(r.sma_bin):<10}  {int(r.n):>5}  {r.mean_sma:>9.0f}  "
                      f"{r.wr:>5.1f}  {r.mean_pnl:>+9.2f}  {r.sum_ppd:>+7.2f}")

    # ── (3) Top-40% filter (≥ q60) — does it improve the strategy? ────────
    print("\n" + "="*100)
    print("  Filter test: keep only trades where SMA(TSLV) >= q60 (top 40%)")
    print("="*100)
    for grid_pips in TSLV_GRID_PIPS_LIST:
        for sma_n_tslv in TSLV_SMA_NS:
            sub = df[(df.grid_pips == grid_pips) & (df.tslv_sma_n == sma_n_tslv)]
            if len(sub) < 10: continue
            q60 = np.quantile(sub.tslv_sma_at_entry, 0.60)
            kept = sub[sub.tslv_sma_at_entry >= q60]
            baseline_pnl = sub.pnl.sum()
            kept_pnl     = kept.pnl.sum() if len(kept) else 0
            baseline_wr  = (sub.pnl > 0).mean() * 100
            kept_wr      = (kept.pnl > 0).mean() * 100 if len(kept) else 0
            print(f"\n  --- grid={grid_pips}p  SMA(N={sma_n_tslv})  threshold >= {q60:.0f} bars ---")
            print(f"  baseline (no filter): n={len(sub):>4}  Σ={baseline_pnl:>+7.1f}p  ppd={baseline_pnl/oos_days_avg:>+5.2f}  WR={baseline_wr:.1f}%")
            print(f"  kept (top 40%):       n={len(kept):>4}  Σ={kept_pnl:>+7.1f}p  ppd={kept_pnl/oos_days_avg:>+5.2f}  WR={kept_wr:.1f}%")


if __name__ == "__main__":
    main()
