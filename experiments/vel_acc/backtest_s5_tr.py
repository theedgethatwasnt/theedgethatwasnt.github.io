"""
S5 Rolling-Window TR Momentum — Backtest Sweep
===============================================
Replicates the M5 TR momentum strategy (v2) using S5 bars.

CRITICAL DESIGN:
  Entry: evaluated ONLY at M5 bar boundaries (every 5 min = 60 S5 bars)
         Same signal frequency as M5 backtest — identical entry conditions.
  Trail: updated at EVERY S5 bar close (0.1-0.5 pip gap vs M5's 3-7 pip gap).
  Exit:  filled at S5 bar close when trail triggered or max_hold reached.

M5-equivalent signals from S5 rolling windows (all causal, all at bar close):
  tr_m5(t)   = max(high[t-59:t+1]) - min(low[t-59:t+1])   [pips, N=M5_WIN=60]
  vel_m5(t)  = (close[t] - close[t-60]) / pip               [pips, direction]
  vel_s30(t) = (close[t] - close[t-6]) / pip                [pips, N=S30_WIN=6]
  vel_15m(t) = (close[t] - close[t-180]) / pip              [pips, N=P15_WIN=180]

Why this is the correct replication:
  At every M5 bar close (bar t), the rolling windows spanning t-59..t
  cover EXACTLY the same bars as that M5 bar → tr_m5 == M5_bar.high - M5_bar.low.
  M5 entry = once every 60 S5 bars. S5 trail = every 5 seconds. Fill gap → 0.1-0.5p.

Live-backtest consistency (SOP R6):
  Backtest: Numba loops with M5-boundary mask derived from timestamps.
  Live: same arithmetic on deque of S5 bars, entry evaluated each time
        a new M5 bar timestamp arrives.
"""

import gc
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT     = Path(__file__).resolve().parents[3]
BA_DIR   = ROOT / "data" / "s5_ba"    # full-history (when available)
OHLC_DIR = ROOT / "data" / "s5_ohlc"  # partial existing files

PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
    "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "NZD_JPY", "CHF_JPY", "EUR_GBP",
]
PIP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
    "USD_JPY": 0.01,   "EUR_JPY": 0.01,   "GBP_JPY": 0.01,   "AUD_JPY": 0.01,
    "CAD_JPY": 0.01,   "NZD_JPY": 0.01,   "CHF_JPY": 0.01,   "EUR_GBP": 0.0001,
}

# Rolling window sizes in S5 bars
M5_WIN  = 60    # 5 min  — M5 equivalent (entry evaluated here)
S30_WIN = 6     # 30 sec — S30 fast confluence
P15_WIN = 180   # 15 min — macro context

IS_FRAC              = 0.70
N_WF                 = 3
N_MC                 = 300
MIN_TRADES_PER_CHUNK = 5
S5_PER_TRADING_DAY   = 17280.0   # 24h × 3600/5
M5_SECONDS           = 300       # 5 minutes in seconds

# Sweep parameters
TR_THRESHOLDS  = [5, 6, 7, 8, 10, 12, 15]  # pips — M5-equivalent TR threshold
TRAIL_PIPS     = [2]                         # pips — trail distance (matches v2)
FAST_CONF      = [False, True]               # require S30 velocity agrees
SLOW_CONF      = [False, True]               # require 15m velocity agrees
MAX_HOLD_BARS  = [720, 1440, 2880]          # S5 bars: 1h, 2h, 4h safety cap
DIRECTIONS     = [1, -1]                     # 1=LONG, -1=SHORT


def build_configs():
    rows = []
    for tr, trail, fc, sc, mh, d in product(
        TR_THRESHOLDS, TRAIL_PIPS, FAST_CONF, SLOW_CONF, MAX_HOLD_BARS, DIRECTIONS,
    ):
        rows.append((float(tr), float(trail), float(fc), float(sc), float(mh), float(d)))
    return np.array(rows, dtype=np.float64)


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit
def _rolling_max(arr, n):
    """Rolling max over n bars (causal). O(n×N) — fine for n≤180, N≤8M."""
    result = np.empty(len(arr))
    for t in range(len(arr)):
        if t < n - 1:
            result[t] = np.nan
            continue
        m = arr[t - n + 1]
        for i in range(t - n + 2, t + 1):
            if arr[i] > m: m = arr[i]
        result[t] = m
    return result


@njit
def _rolling_min(arr, n):
    result = np.empty(len(arr))
    for t in range(len(arr)):
        if t < n - 1:
            result[t] = np.nan
            continue
        m = arr[t - n + 1]
        for i in range(t - n + 2, t + 1):
            if arr[i] < m: m = arr[i]
        result[t] = m
    return result


@njit
def _sim_pnl(close, hi, lo, spread, pip,
             roll_tr,           # precomputed M5 TR in pips
             m5_mask,           # bool: True = this S5 bar is an M5 bar close
             tr_thresh, trail_p, fast_c, slow_c, max_hold, di,
             sp_gate, start, end):
    """
    Simulate one config over [start, end).

    Entry: only when m5_mask[t] is True (M5 bar close).
    Trail: updated every S5 bar.
    Exit: S5 bar close when trail triggered or max_hold reached.
    """
    warmup  = P15_WIN + M5_WIN  # worst-case warmup
    mh      = int(max_hold)
    direction = int(di)   # +1 LONG, -1 SHORT

    in_trade = False
    entry_px = 0.0
    hw       = 0.0   # high-water mark (best close seen since entry)
    hold     = 0
    total_pnl = 0.0
    n_trades  = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        h  = hi[t]
        l  = lo[t]
        sp = spread[t] / pip   # spread in pips

        if in_trade:
            hold += 1
            # Update HWM and trail at S5 resolution
            if direction == 1:
                if h > hw: hw = h
                trail = hw - trail_p * pip
                should_exit = (l <= trail) or (hold >= mh)
            else:
                if l < hw: hw = l
                trail = hw + trail_p * pip
                should_exit = (h >= trail) or (hold >= mh)

            if should_exit:
                # S5 bar-close fill: gap ≤ 0.1-0.5 pip (vs M5's 3-7 pip)
                pnl        = direction * (c - entry_px) / pip - sp
                total_pnl += pnl
                n_trades  += 1
                in_trade   = False

        else:
            # Entry: only at M5 bar boundaries
            if not m5_mask[t]: continue
            if sp > sp_gate: continue
            if np.isnan(roll_tr[t]): continue

            tr     = roll_tr[t]   # M5-equivalent TR in pips
            vel_m5 = (c - close[t - M5_WIN]) / pip   # net M5 velocity in pips

            if tr < tr_thresh: continue

            # Fast S30 confluence (N=6 S5 bars = 30 seconds)
            if fast_c > 0.0:
                vel_s30 = (c - close[t - S30_WIN]) / pip
                if direction == 1  and vel_s30 <= 0.0: continue
                if direction == -1 and vel_s30 >= 0.0: continue

            # Slow 15m confluence (N=180 S5 bars)
            if slow_c > 0.0:
                vel_15m = (c - close[t - P15_WIN]) / pip
                if direction == 1  and vel_15m <= 0.0: continue
                if direction == -1 and vel_15m >= 0.0: continue

            # Direction check: M5 net velocity
            if direction == 1  and vel_m5 <= 0.0: continue
            if direction == -1 and vel_m5 >= 0.0: continue

            # Enter at S5 bar close
            entry_px = c
            hw       = h if direction == 1 else l
            in_trade = True
            hold     = 0

    if in_trade and end > 0:
        sp        = spread[end - 1] / pip
        pnl       = direction * (close[end - 1] - entry_px) / pip - sp
        total_pnl += pnl
        n_trades  += 1

    return total_pnl, n_trades


@njit(parallel=True)
def wf_sweep(close, hi, lo, spread, pip, roll_tr, m5_mask,
             configs, sp_gate, chunk_ends):
    n_cfg     = len(configs)
    nc        = len(chunk_ends)
    wf_pnl    = np.zeros((n_cfg, nc))
    wf_trades = np.zeros((n_cfg, nc))

    for ci in prange(n_cfg):
        tr    = configs[ci, 0]
        trail = configs[ci, 1]
        fc    = configs[ci, 2]
        sc    = configs[ci, 3]
        mh    = int(configs[ci, 4])
        di    = int(configs[ci, 5])

        s = 0
        for k in range(nc):
            e     = int(chunk_ends[k])
            pnl, nt = _sim_pnl(close, hi, lo, spread, pip,
                                roll_tr, m5_mask,
                                tr, trail, fc, sc, mh, di,
                                sp_gate, s, e)
            wf_pnl[ci, k]    = pnl
            wf_trades[ci, k] = nt
            s = e

    return wf_pnl, wf_trades


@njit
def sim_collect(close, hi, lo, spread, pip, roll_tr, m5_mask,
                tr_thresh, trail_p_price, fast_c, slow_c, max_hold, di,
                sp_gate, start, end):
    warmup  = P15_WIN + M5_WIN
    mh      = int(max_hold)
    direction = int(di)

    max_t   = (end - start) // 60 + 10   # at most 1 trade per M5 bar
    pnls    = np.empty(max_t)
    in_trade = False
    entry_px = 0.0
    hw       = 0.0
    hold     = 0
    nt = 0; nw = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        h  = hi[t]
        l  = lo[t]
        sp = spread[t] / pip

        if in_trade:
            hold += 1
            if direction == 1:
                if h > hw: hw = h
                trail = hw - trail_p_price
                should_exit = (l <= trail) or (hold >= mh)
            else:
                if l < hw: hw = l
                trail = hw + trail_p_price
                should_exit = (h >= trail) or (hold >= mh)

            if should_exit:
                p = direction * (c - entry_px) / pip - sp
                pnls[nt] = p; nt += 1
                if p > 0.0: nw += 1
                in_trade = False

        else:
            if not m5_mask[t]: continue
            if sp > sp_gate: continue
            if np.isnan(roll_tr[t]): continue

            tr     = roll_tr[t]
            vel_m5 = (c - close[t - M5_WIN]) / pip

            if tr < tr_thresh: continue

            if fast_c > 0.0:
                vel_s30 = (c - close[t - S30_WIN]) / pip
                if direction == 1  and vel_s30 <= 0.0: continue
                if direction == -1 and vel_s30 >= 0.0: continue

            if slow_c > 0.0:
                vel_15m = (c - close[t - P15_WIN]) / pip
                if direction == 1  and vel_15m <= 0.0: continue
                if direction == -1 and vel_15m >= 0.0: continue

            if direction == 1  and vel_m5 <= 0.0: continue
            if direction == -1 and vel_m5 >= 0.0: continue

            entry_px = c
            hw       = h if direction == 1 else l
            in_trade = True
            hold     = 0

    if in_trade and end > 0:
        sp = spread[end - 1] / pip
        p  = direction * (close[end - 1] - entry_px) / pip - sp
        pnls[nt] = p; nt += 1
        if p > 0.0: nw += 1

    return pnls[:nt], nt, nw


@njit
def mc_pvalue(trade_pnls, n_mc):
    observed = 0.0
    n = len(trade_pnls)
    for i in range(n): observed += trade_pnls[i]
    if n == 0: return 1.0
    count = 0
    for _ in range(n_mc):
        s = 0.0
        for i in range(n):
            s += trade_pnls[i] if np.random.random() > 0.5 else -trade_pnls[i]
        if s >= observed: count += 1
    return count / n_mc


# ── Data loader ────────────────────────────────────────────────────────────────

def find_parquet(pair):
    for d, label in [(BA_DIR, "s5_ba"), (OHLC_DIR, "s5_ohlc")]:
        p = d / f"{pair}_S5_BA.parquet"
        if p.exists():
            return p, label
    return None, None


def load_pair(pair):
    path, src = find_parquet(pair)
    if path is None:
        return None, None, None, None, None, None

    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "close" in df.columns:
        cl = df["close"].astype(np.float64).values
        hi = df["high"].astype(np.float64).values
        lo = df["low"].astype(np.float64).values
    else:
        cl = ((df["bid_c"] + df["ask_c"]) / 2).astype(np.float64).values
        hi = ((df["bid_h"] + df["ask_h"]) / 2).astype(np.float64).values
        lo = ((df["bid_l"] + df["ask_l"]) / 2).astype(np.float64).values

    sp = (df["ask_c"] - df["bid_c"]).astype(np.float64).values

    # Build M5-bar-close mask from timestamps
    ts_s = pd.to_datetime(df["timestamp"]).astype(np.int64).values // 1_000_000_000  # unix seconds numpy array
    m5_period = ts_s // M5_SECONDS   # which M5 bar each S5 bar belongs to
    # A bar is the last in its M5 period when the next bar is in a different period
    m5_mask = np.zeros(len(df), dtype=np.bool_)
    m5_mask[:-1] = m5_period[:-1] != m5_period[1:]
    m5_mask[-1]  = True   # last bar is always "M5 close"

    n_m5 = m5_mask.sum()
    m5_frac = m5_mask.mean()
    print(f"    M5 boundaries: {n_m5:,} ({m5_frac:.3f} of bars = {1/m5_frac:.1f}s avg spacing, expected 60)")

    return cl, hi, lo, sp, m5_mask.astype(np.int8), src


# ── Per-pair runner ────────────────────────────────────────────────────────────

def run_pair(pair, configs):
    pip                           = PIP[pair]
    close, hi, lo, sp, m5_mask, src = load_pair(pair)
    if close is None:
        return []

    n        = len(close)
    is_end   = int(n * IS_FRAC)
    oos_days = (n - is_end) / S5_PER_TRADING_DAY

    if oos_days < 30:
        print(f"  {pair} [{src}]: only {oos_days:.0f} OOS days — skip")
        return []

    sp_gate = float(np.percentile(sp[:is_end] / pip, 90))
    print(f"  {pair} [{src}]: n={n:,}  IS={is_end:,}  OOS={oos_days:.0f}d  sp_gate={sp_gate:.2f}p")

    # Precompute rolling M5-equivalent TR (max-min over 60 S5 bars)
    print("    Precomputing rolling max/min...")
    roll_hi  = _rolling_max(hi, M5_WIN)
    roll_lo  = _rolling_min(lo, M5_WIN)
    roll_tr  = (roll_hi - roll_lo) / pip   # M5 TR in pips
    tr_at_m5 = roll_tr[m5_mask.astype(bool)]
    print(f"    TR at M5 closes: mean={np.nanmean(tr_at_m5):.2f}p  P90={np.nanpercentile(tr_at_m5, 90):.2f}p")

    # Convert m5_mask to float64 for Numba (bool arrays tricky across prange)
    m5_f = m5_mask.astype(np.float64)

    chunk_sz   = is_end // N_WF
    chunk_ends = np.array([(k+1)*chunk_sz for k in range(N_WF)], dtype=np.int64)
    chunk_ends[-1] = is_end

    # Gate 1: WF sweep — trail passed in pips, sim converts to price
    configs_price = configs.copy()
    configs_price[:, 1] *= pip   # trail: pips → price

    wf_pnl, wf_trades = wf_sweep(close, hi, lo, sp, pip, roll_tr, m5_f,
                                  configs_price, sp_gate, chunk_ends)

    wf_pass = (np.all(wf_pnl > 0, axis=1) &
               np.all(wf_trades >= MIN_TRADES_PER_CHUNK, axis=1))
    wf_idx  = np.where(wf_pass)[0]
    print(f"    Gate1 WF:       {len(wf_idx):>5}/{len(configs):>5} pass")

    if len(wf_idx) == 0:
        # Show distribution of best-chunk performance for diagnosis
        best_chunk_pd = wf_pnl.max(axis=1) / (is_end / 3 / S5_PER_TRADING_DAY)
        all_chunks_pd = wf_pnl.sum(axis=1) / (is_end / S5_PER_TRADING_DAY)
        print(f"    Best IS config: all_chunks_pd={all_chunks_pd.max():.1f}p/d  "
              f"best_chunk_pd={best_chunk_pd.max():.1f}p/d  "
              f"all_pos_chunks={np.sum(np.all(wf_pnl > 0, axis=1))}")
        return []

    # Gates 2+3: OOS + MC
    survivors = []
    for ci in wf_idx:
        tr    = configs[ci, 0]
        trail = configs[ci, 1]
        fc    = configs[ci, 2]
        sc    = configs[ci, 3]
        mh    = int(configs[ci, 4])
        di    = int(configs[ci, 5])

        oos_pnls, nt, nw = sim_collect(
            close, hi, lo, sp, pip, roll_tr, m5_f,
            tr, trail * pip, fc, sc, mh, di, sp_gate, is_end, n,
        )

        if nt == 0 or oos_pnls.sum() <= 0:
            continue

        p_val  = mc_pvalue(oos_pnls, N_MC)
        if p_val >= 0.05:
            continue

        oos_pd = oos_pnls.sum() / oos_days
        fc_str = "S30+15m" if fc > 0 and sc > 0 else ("S30" if fc > 0 else ("15m" if sc > 0 else "M5only"))
        survivors.append({
            "pair":       pair,
            "data":       src,
            "tr_thresh":  tr,
            "trail":      trail,
            "confluence": fc_str,
            "fast_conf":  bool(fc),
            "slow_conf":  bool(sc),
            "max_hold":   mh,
            "direction":  di,
            "sp_gate":    round(sp_gate, 2),
            "oos_pnl":    round(oos_pnls.sum(), 1),
            "oos_pd":     round(oos_pd, 1),
            "oos_trades": nt,
            "oos_wr":     round(nw / nt, 3),
            "mc_pval":    round(p_val, 4),
            "wf_pnl":     [round(x, 1) for x in wf_pnl[ci]],
            "wf_trades":  [int(x) for x in wf_trades[ci]],
        })

    print(f"    Gate2+3 OOS+MC: {len(survivors):>5}/{len(wf_idx):>5} pass")
    return survivors


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  S5 Rolling-Window TR Momentum — Backtest Sweep")
    print(f"  Entry: M5 bar closes only | Trail: every S5 bar")
    print(f"  M5_WIN={M5_WIN}bars  S30_WIN={S30_WIN}bars  P15_WIN={P15_WIN}bars")
    print("=" * 72)

    configs = build_configs()
    print(f"\nConfigs per pair: {len(configs)}")
    print(f"TR thresholds:  {TR_THRESHOLDS} pips")
    print(f"Confluence:     S30={FAST_CONF}  15m={SLOW_CONF}")
    print(f"Max hold (S5):  {MAX_HOLD_BARS} bars = {[x//720 for x in MAX_HOLD_BARS]} hrs")

    print("\nWarming up Numba JIT...")
    dummy_c = np.cumsum(np.random.randn(2000)) * 0.0001 + 1.1
    dummy_h = dummy_c + np.abs(np.random.randn(2000)) * 0.0001
    dummy_l = dummy_c - np.abs(np.random.randn(2000)) * 0.0001
    dummy_s = np.full(2000, 0.0002)
    dummy_m = (np.arange(2000) % 60 == 59).astype(np.float64)
    rh = _rolling_max(dummy_h, M5_WIN)
    rl = _rolling_min(dummy_l, M5_WIN)
    rt = (rh - rl) / 0.0001
    _sim_pnl(dummy_c, dummy_h, dummy_l, dummy_s, 0.0001, rt, dummy_m,
             8.0, 0.0002, 0.0, 0.0, 720, 1, 1.70, 0, 1500)
    print("  Done.\n")

    all_results = []

    for pair in PAIRS:
        path, src = find_parquet(pair)
        if path is None:
            print(f"  {pair}: no S5 data — skip")
            continue
        results = run_pair(pair, configs)
        all_results.extend(results)
        gc.collect()

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)

    if not all_results:
        print("  0 survivors.")
        print("\nDiagnosis: even with S5 exit resolution (0.1-0.5p fill gap),")
        print("the TR momentum signal has insufficient edge to overcome spread costs.")
        print("Consider: alternative entry (bar-direction confirmation), wider trail,")
        print("or different confluence filters.")
        return

    df = pd.DataFrame(all_results)
    print(f"\nTotal survivors: {len(df)} across {df['pair'].nunique()} pairs\n")

    print("Best config per pair (by OOS p/d):\n")
    hdr = (f"  {'Pair':<10} {'TR':>4} {'trail':>5} {'confluence':<9} "
           f"{'hold_h':>6} {'dir':>5} {'p/d':>7} {'trades':>7} {'WR':>6} {'MC':>7}")
    print(hdr)
    print("  " + "-" * 74)

    for pair in PAIRS:
        sub = df[df["pair"] == pair]
        if sub.empty: continue
        row = sub.sort_values("oos_pd", ascending=False).iloc[0]
        d_str = "LONG" if row["direction"] == 1 else "SHORT"
        hold_h = row["max_hold"] / 720
        print(f"  {pair:<10} {row['tr_thresh']:>4.0f} {row['trail']:>5.0f} "
              f"{row['confluence']:<9} {hold_h:>6.1f} {d_str:>5} "
              f"{row['oos_pd']:>7.1f} {row['oos_trades']:>7} "
              f"{row['oos_wr']:>6.1%} {row['mc_pval']:>7.4f}")

    # Confluence effect
    print("\nConfluence effect (mean OOS p/d by filter level):")
    for conf in ["M5only", "S30", "15m", "S30+15m"]:
        sub = df[df["confluence"] == conf]
        if sub.empty: continue
        print(f"  {conf:<9}: n={len(sub):>4}  mean_pd={sub['oos_pd'].mean():>7.1f}  "
              f"median_pd={sub['oos_pd'].median():>7.1f}")

    out = ROOT / "research" / "experiments" / "vel_acc" / "results_s5_tr.csv"
    df.to_csv(out, index=False)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
