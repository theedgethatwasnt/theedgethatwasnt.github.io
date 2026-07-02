#!/usr/bin/env python3
"""
S5 Counter-Trend Momentum Backtest
===================================
Hypothesis: when all rolling lags agree on direction, the move is over-extended
            and the market fades back — trade OPPOSITE to momentum confluence.

    lag_short ≈ S5–S30  (1–6 bars  = 5–30s)
    lag_mid   ≈ S30–M2  (6–30 bars = 30s–2.5m)
    lag_long  ≈ M2–M15  (30–180 bars = 2.5–15m)

Signal (long):  all three lags show DOWNWARD momentum → fade/counter-trend long.
Signal (short): all three lags show UPWARD momentum   → fade/counter-trend short.
Optional accel: momentum DECELERATING (losing steam) — abs(ms) < abs(ms_prev).
ATR variant:    require |momentum| > atr_mult × ATR14 (strong move = better fade candidate).

Motivation: trend sweep found WR≈22% (trend) → counter-trend WR≈78%.
            This script tests whether that counter-trend edge survives WF + MC.

Fill model (SOP-compliant):
    Entry : close[i] ± half_spread  (market order, R3)
    TP/SL : broker-side exact levels (R2 within-bar sequencing)
    Spread gate: spread_pips[i] > tp_pips × 0.5 → skip entry
    Timeout: 120 S5 bars (10 min) → exit at close price

Validation:
    IS WF    : 3 temporal chunks, all p/d > 0
    Trade gate: IS trades ≥ 200
    Spread gate: P90 IS spread as hard entry cap  (R5)
    MC       : 1000 sign-shuffles, mc_p < 0.05
    OOS      : sealed — reported only for MC survivors  (R8)

Usage:
    python3 backtest_s5_ctr_mom.py [--pair EUR_USD] [--fast]
"""

import argparse
import itertools
import time
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT  = Path("/path/to/projects/fx-core")
DATA_DIR = PROJECT / "data/s5_ohlc"
RES_DIR  = PROJECT / "research/experiments/s5_momentum_sweep/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Pair registry ──────────────────────────────────────────────────────────────
PAIRS = {
    "EUR_USD": {"file": "EUR_USD_S5_BA.parquet", "pip": 0.0001},
    "EUR_JPY": {"file": "EUR_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file": "GBP_JPY_S5_BA.parquet", "pip": 0.01},
}

# ── Backtest constants ─────────────────────────────────────────────────────────
IS_FRAC        = 0.70
N_WF_CHUNKS    = 3
MIN_IS_TRADES  = 200
MC_SHUFFLES    = 1000
TIMEOUT_BARS   = 120    # 120 × 5s = 10 min max hold
ATR_PERIOD     = 14

# ── Sweep space ────────────────────────────────────────────────────────────────
LAG_SHORT_OPT  = [1,  2,  3,  6]
LAG_MID_OPT    = [6,  12, 18, 30]
LAG_LONG_OPT   = [30, 60, 90, 180]
ACCEL_OPT      = [False, True]
TP_PIPS_OPT    = [2, 3, 5, 10]
SL_PIPS_OPT    = [2, 3, 5]
ATR_MULT_OPT   = [0.0, 0.3, 0.5]


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_atr14(high, low, close, period):
    n = len(close)
    tr  = np.empty(n, dtype=np.float64)
    atr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    for i in range(period - 1):
        atr[i] = atr[period - 1]
    return atr


@njit(cache=True, fastmath=True)
def find_exit(close, high, low, entry_bar, entry_px,
              direction, tp_level, sl_level,
              tp_pips_f, sl_pips_f, pip_size, timeout):
    """R2: bull bar → check high first for long, low first for short."""
    n   = len(close)
    end = min(entry_bar + timeout + 1, n)
    for j in range(entry_bar + 1, end):
        bull = close[j] >= close[j - 1]
        if direction == 1:
            if bull:
                if high[j] >= tp_level: return tp_pips_f,  np.int64(j - entry_bar)
                if low[j]  <= sl_level: return -sl_pips_f, np.int64(j - entry_bar)
            else:
                if low[j]  <= sl_level: return -sl_pips_f, np.int64(j - entry_bar)
                if high[j] >= tp_level: return tp_pips_f,  np.int64(j - entry_bar)
        else:
            if not bull:
                if low[j]  <= tp_level: return tp_pips_f,  np.int64(j - entry_bar)
                if high[j] >= sl_level: return -sl_pips_f, np.int64(j - entry_bar)
            else:
                if high[j] >= sl_level: return -sl_pips_f, np.int64(j - entry_bar)
                if low[j]  <= tp_level: return tp_pips_f,  np.int64(j - entry_bar)
    last = min(entry_bar + timeout, n - 1)
    raw  = (close[last] - entry_px) * np.float64(direction) / pip_size
    return raw, np.int64(last - entry_bar)


@njit(cache=True, fastmath=True)
def run_segment(close, high, low, spread_pips, atr, pip_size,
                lag_short, lag_mid, lag_long,
                accel_req, tp_pips, sl_pips, atr_mult,
                seg_start, seg_end, timeout, sp_gate):
    """
    Counter-trend: trade OPPOSITE to momentum confluence.
    Long  when all lags are DOWN (fade the decline).
    Short when all lags are UP   (fade the rise).
    Accel gate: momentum must be DECELERATING (abs(ms) < abs(ms_prev)).
    """
    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)

    warmup = lag_long * 2 + ATR_PERIOD + 2
    start  = max(seg_start, warmup)
    next_entry = start

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)

    i = start
    while i < seg_end - 1:
        if i < next_entry:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > tp_f * 0.5 or sp > sp_gate:
            i += 1
            continue

        atr_i = atr[i]

        for direction in (np.int64(1), np.int64(-1)):
            ms = close[i] - close[i - lag_short]
            mm = close[i] - close[i - lag_mid]
            ml = close[i] - close[i - lag_long]

            # Counter-trend: long when all lags DOWN, short when all lags UP
            if direction == 1:   # long → need all lags negative (price fell)
                if ms >= 0.0 or mm >= 0.0 or ml >= 0.0:
                    continue
            else:                # short → need all lags positive (price rose)
                if ms <= 0.0 or mm <= 0.0 or ml <= 0.0:
                    continue

            # ATR-strength gate: require the move to be large enough to fade
            if atr_mult > 0.0:
                thresh = atr_mult * atr_i
                if abs(ms) < thresh or abs(mm) < thresh or abs(ml) < thresh:
                    continue

            # Deceleration gate: momentum losing steam (better reversal candidate)
            if accel_req:
                ms_prev = close[i - lag_short] - close[i - 2 * lag_short]
                # direction=1: ms<0, ms_prev<0; deceleration = abs(ms) < abs(ms_prev)
                #              i.e. ms > ms_prev (less negative)
                # direction=-1: ms>0, ms_prev>0; deceleration = abs(ms) < abs(ms_prev)
                #               i.e. ms < ms_prev (less positive)
                if direction == 1:
                    if ms <= ms_prev:   # still accelerating downward → skip
                        continue
                else:
                    if ms >= ms_prev:   # still accelerating upward → skip
                        continue

            entry_px = close[i] + np.float64(direction) * 0.5 * sp * pip_size
            tp_level = entry_px + np.float64(direction) * tp_f * pip_size
            sl_level = entry_px - np.float64(direction) * sl_f * pip_size

            pnl, bars = find_exit(close, high, low, i, entry_px,
                                   direction, tp_level, sl_level,
                                   tp_f, sl_f, pip_size, timeout)

            total_pips += pnl
            n_trades   += np.int64(1)
            if pnl > 0.0:
                n_wins += np.int64(1)
            next_entry = i + bars + np.int64(1)
            break

        i += 1

    return total_pips, n_trades, n_wins


@njit(parallel=True, cache=True)
def sweep_parallel(close, high, low, spread_pips, atr, pip_size,
                   configs, is_end, wf_starts, wf_ends, timeout, sp_gate):
    """Parallel sweep over configs. configs[c] = [ls, lm, ll, accel, tp, sl, atr_mult_idx]."""
    ATR_MULTS = np.array([0.0, 0.3, 0.5], dtype=np.float64)
    n_cfg = len(configs)
    n_wf  = len(wf_starts)
    n_col = (n_wf + 2) * 3
    out   = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all = len(close)

    for c in prange(n_cfg):
        ls = configs[c, 0];  lm = configs[c, 1];  ll = configs[c, 2]
        ac = bool(configs[c, 3])
        tp = configs[c, 4];  sl = configs[c, 5]
        am = ATR_MULTS[configs[c, 6]]

        p, n, w = run_segment(close, high, low, spread_pips, atr, pip_size,
                               ls, lm, ll, ac, tp, sl, am,
                               0, is_end, timeout, sp_gate)
        out[c, 0] = p;  out[c, 1] = n;  out[c, 2] = w

        for k in range(n_wf):
            p, n, w = run_segment(close, high, low, spread_pips, atr, pip_size,
                                   ls, lm, ll, ac, tp, sl, am,
                                   wf_starts[k], wf_ends[k], timeout, sp_gate)
            out[c, 3 + k * 3    ] = p
            out[c, 3 + k * 3 + 1] = n
            out[c, 3 + k * 3 + 2] = w

        p, n, w = run_segment(close, high, low, spread_pips, atr, pip_size,
                               ls, lm, ll, ac, tp, sl, am,
                               is_end, n_all, timeout, sp_gate)
        out[c, 3 + n_wf * 3    ] = p
        out[c, 3 + n_wf * 3 + 1] = n
        out[c, 3 + n_wf * 3 + 2] = w

    return out


@njit(cache=True, fastmath=True)
def run_segment_pnl(close, high, low, spread_pips, atr, pip_size,
                     lag_short, lag_mid, lag_long,
                     accel_req, tp_pips, sl_pips, atr_mult,
                     seg_start, seg_end, timeout, sp_gate):
    """Same as run_segment but returns full pnl array (for MC)."""
    MAX_T = 100_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t = np.int64(0)

    warmup = lag_long * 2 + ATR_PERIOD + 2
    start  = max(seg_start, warmup)
    next_entry = start

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)

    i = start
    while i < seg_end - 1 and n_t < MAX_T:
        if i < next_entry:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > tp_f * 0.5 or sp > sp_gate:
            i += 1
            continue

        atr_i = atr[i]

        for direction in (np.int64(1), np.int64(-1)):
            ms = close[i] - close[i - lag_short]
            mm = close[i] - close[i - lag_mid]
            ml = close[i] - close[i - lag_long]

            if direction == 1:
                if ms >= 0.0 or mm >= 0.0 or ml >= 0.0:
                    continue
            else:
                if ms <= 0.0 or mm <= 0.0 or ml <= 0.0:
                    continue

            if atr_mult > 0.0:
                thresh = atr_mult * atr_i
                if abs(ms) < thresh or abs(mm) < thresh or abs(ml) < thresh:
                    continue

            if accel_req:
                ms_prev = close[i - lag_short] - close[i - 2 * lag_short]
                if direction == 1:
                    if ms <= ms_prev:
                        continue
                else:
                    if ms >= ms_prev:
                        continue

            entry_px = close[i] + np.float64(direction) * 0.5 * sp * pip_size
            tp_level = entry_px + np.float64(direction) * tp_f * pip_size
            sl_level = entry_px - np.float64(direction) * sl_f * pip_size

            pnl, bars = find_exit(close, high, low, i, entry_px,
                                   direction, tp_level, sl_level,
                                   tp_f, sl_f, pip_size, timeout)

            pnl_arr[n_t] = pnl
            n_t += np.int64(1)
            next_entry = i + bars + np.int64(1)
            break

        i += 1

    return pnl_arr[:n_t]


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def run_mc(pnl_arr, is_days, n_shuffles=MC_SHUFFLES, seed=42):
    """Sign-shuffle MC. Returns fraction of shuffles with p/d ≥ actual."""
    if len(pnl_arr) < 50:
        return np.nan
    actual_pd = pnl_arr.sum() / is_days
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]),
                       size=(n_shuffles, len(pnl_arr)))
    shuffled_pd = (np.abs(pnl_arr) * signs).sum(axis=1) / is_days
    return float((shuffled_pd >= actual_pd).mean())


# ── Config builder ─────────────────────────────────────────────────────────────

def build_configs():
    rows = []
    atr_idx_map = {v: i for i, v in enumerate(ATR_MULT_OPT)}
    for ls, lm, ll in itertools.product(LAG_SHORT_OPT, LAG_MID_OPT, LAG_LONG_OPT):
        if not (ls < lm < ll):
            continue
        for ac in ACCEL_OPT:
            for tp in TP_PIPS_OPT:
                for sl in SL_PIPS_OPT:
                    if tp < sl:
                        continue
                    for am in ATR_MULT_OPT:
                        rows.append((ls, lm, ll, int(ac), tp, sl, atr_idx_map[am]))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(r[0], r[1], r[2], bool(r[3]), r[4], r[5], ATR_MULT_OPT[r[6]])
            for r in rows]
    return arr, meta


# ── Main ───────────────────────────────────────────────────────────────────────

def process_pair(pair_name, pair_cfg, configs_arr, configs_meta):
    path = DATA_DIR / pair_cfg["file"]
    pip  = pair_cfg["pip"]

    print(f"\n{'='*64}")
    print(f"  {pair_name}  |  {path.name}")

    probe = duckdb.query(f'SELECT * FROM "{path}" LIMIT 1').df().columns.tolist()
    if "open" in probe:
        df    = duckdb.query(
            f'SELECT timestamp, open, high, low, close, bid_c, ask_c '
            f'FROM "{path}" ORDER BY timestamp'
        ).df()
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        sp    = ((df["ask_c"] - df["bid_c"]) / pip).clip(0.3, 20.0).values.astype(np.float64)
    else:
        df    = duckdb.query(
            f'SELECT timestamp, bid_h, bid_l, bid_c, ask_h, ask_l, ask_c '
            f'FROM "{path}" ORDER BY timestamp'
        ).df()
        close = ((df["bid_c"] + df["ask_c"]) / 2).values.astype(np.float64)
        high  = ((df["bid_h"] + df["ask_h"]) / 2).values.astype(np.float64)
        low   = ((df["bid_l"] + df["ask_l"]) / 2).values.astype(np.float64)
        sp    = ((df["ask_c"] - df["bid_c"]) / pip).clip(0.3, 20.0).values.astype(np.float64)

    # Contiguous arrays for Numba
    close = np.ascontiguousarray(close)
    high  = np.ascontiguousarray(high)
    low   = np.ascontiguousarray(low)
    sp    = np.ascontiguousarray(sp)

    atr   = compute_atr14(high, low, close, ATR_PERIOD)

    n      = len(close)
    is_end = int(n * IS_FRAC)
    sp_gate = float(np.percentile(sp[:is_end], 90))   # R5

    ts_dates = pd.to_datetime(df["timestamp"]).dt.normalize()
    is_days  = max(1, int(ts_dates.iloc[:is_end].nunique()))
    oos_days = max(1, int(ts_dates.iloc[is_end:].nunique()))

    chunk     = is_end // N_WF_CHUNKS
    wf_starts = np.array([k * chunk       for k in range(N_WF_CHUNKS)], dtype=np.int64)
    wf_ends   = np.array([(k+1) * chunk   for k in range(N_WF_CHUNKS)], dtype=np.int64)

    print(f"  Bars {n:,}  IS {is_end:,} ({is_days}d)  OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p")
    print(f"  Configs: {len(configs_arr)}")

    print("  Compiling Numba ...", end="", flush=True)
    t0 = time.time()
    _ws = wf_starts.copy(); _we = np.minimum(wf_ends, 2000)
    _ = sweep_parallel(close[:3000], high[:3000], low[:3000], sp[:3000],
                        atr[:3000], np.float64(pip),
                        configs_arr[:4], 2000, _ws, _we, TIMEOUT_BARS, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0 = time.time()
    raw = sweep_parallel(close, high, low, sp, atr, np.float64(pip),
                          configs_arr, is_end, wf_starts, wf_ends,
                          TIMEOUT_BARS, sp_gate)
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    n_wf = N_WF_CHUNKS
    rows = []
    for c, meta in enumerate(configs_meta):
        ls, lm, ll, ac, tp, sl, am = meta

        is_pips  = raw[c, 0];  is_n  = int(raw[c, 1]);  is_w  = int(raw[c, 2])
        oos_pips = raw[c, 3 + n_wf*3]
        oos_n    = int(raw[c, 3 + n_wf*3 + 1])
        oos_w    = int(raw[c, 3 + n_wf*3 + 2])

        wf_pds = []
        for k in range(n_wf):
            pk = raw[c, 3 + k*3];  nk = int(raw[c, 3 + k*3 + 1])
            wf_pds.append(pk / (is_days / n_wf) if nk > 0 else 0.0)

        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0
        wr_is  = is_w  / is_n  if is_n  > 0 else 0.0
        wr_oos = oos_w / oos_n if oos_n > 0 else 0.0

        wf_pass = (is_n >= MIN_IS_TRADES) and all(p > 0.0 for p in wf_pds)

        rows.append({
            "pair":     pair_name,
            "ls": ls, "lm": lm, "ll": ll,
            "accel":    ac, "tp": tp, "sl": sl, "atr_mult": am,
            "is_pd":    round(is_pd,   2),
            "is_wr":    round(wr_is,   3),
            "is_n":     is_n,
            "wf1":      round(wf_pds[0], 2),
            "wf2":      round(wf_pds[1], 2),
            "wf3":      round(wf_pds[2], 2),
            "wf_pass":  wf_pass,
            "oos_pd":   round(oos_pd,  2),
            "oos_wr":   round(wr_oos,  3),
            "oos_n":    oos_n,
            "mc_p":     np.nan,
        })

    survivors = [r for r in rows if r["wf_pass"]]
    print(f"  WF survivors: {len(survivors)}  (running MC ...)", end="", flush=True)
    t0 = time.time()
    for r in survivors:
        pnl_arr = run_segment_pnl(
            close, high, low, sp, atr, np.float64(pip),
            r["ls"], r["lm"], r["ll"], bool(r["accel"]),
            r["tp"], r["sl"], np.float64(r["atr_mult"]),
            0, is_end, TIMEOUT_BARS, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)

    mc_pass = sum(1 for r in survivors
                  if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
    print(f" {time.time()-t0:.1f}s  mc_p<0.05: {mc_pass}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Quick test: 6 configs only")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    print(f"Config space: {len(configs_arr)} configs per pair "
          f"(counter-trend / fade signal)")

    if args.fast:
        configs_arr  = configs_arr[:6]
        configs_meta = configs_meta[:6]
        print("  [fast mode: 6 configs only]")

    pairs_to_run = {args.pair: PAIRS[args.pair]} if args.pair else PAIRS

    all_rows = []
    for pair_name, pair_cfg in pairs_to_run.items():
        rows = process_pair(pair_name, pair_cfg, configs_arr, configs_meta)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    out_path = RES_DIR / "ctr_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df_out)} rows)")

    top = (df_out[df_out["wf_pass"] & (df_out["mc_p"] < 0.05)]
           .sort_values("oos_pd", ascending=False))

    print(f"\n{'='*80}")
    if len(top) > 0:
        print(f"PASSED WF + MC  ({len(top)} configs)  — ranked by OOS p/d:")
        cols = ["pair","ls","lm","ll","accel","tp","sl","atr_mult",
                "is_pd","oos_pd","oos_wr","oos_n","mc_p"]
        print(top[cols].head(30).to_string(index=False))
    else:
        print("No configs passed WF + MC.")
        wf_only = (df_out[df_out["wf_pass"]]
                   .sort_values("is_pd", ascending=False).head(20))
        if len(wf_only):
            print(f"\nWF-only survivors (top 20 IS p/d):")
            cols = ["pair","ls","lm","ll","accel","tp","sl","atr_mult",
                    "is_pd","wf1","wf2","wf3","mc_p"]
            print(wf_only[cols].to_string(index=False))
        else:
            print("No WF survivors — counter-trend hypothesis also rejected.")

    # Diagnostic: best IS p/d per pair regardless of gates
    print(f"\n{'─'*80}")
    print("Best IS p/d per pair (top 5, no gate filter):")
    for pair in df_out["pair"].unique():
        sub = df_out[df_out["pair"]==pair].sort_values("is_pd", ascending=False).head(5)
        print(f"\n  {pair}:")
        print(sub[["ls","lm","ll","accel","tp","sl","atr_mult",
                    "is_pd","is_wr","is_n","wf_pass","mc_p"]].to_string(index=False))

    # Show trade count distribution to understand signal frequency
    print(f"\n{'─'*80}")
    print("Trade count distribution for IS (all configs, per pair):")
    for pair in df_out["pair"].unique():
        sub = df_out[df_out["pair"]==pair]
        print(f"  {pair}: median={sub['is_n'].median():.0f}  "
              f"p25={sub['is_n'].quantile(0.25):.0f}  "
              f"p75={sub['is_n'].quantile(0.75):.0f}  "
              f"max={sub['is_n'].max():.0f}")


if __name__ == "__main__":
    main()
