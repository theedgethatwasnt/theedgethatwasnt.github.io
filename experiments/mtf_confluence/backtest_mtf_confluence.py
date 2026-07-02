#!/usr/bin/env python3
"""
Multi-Resolution Two-Layer Momentum Confluence Backtest
========================================================
Sweeps three bar resolutions — S5, S30, M1 — all derived from S5 parquets
via pandas resampling. No new OANDA fetches needed.

Two-layer signal (trend-following):
  Fast layer: close[i] - close[i - lag_fast]   (short-window momentum)
  Slow layer: close[i] - close[i - lag_slow]   (longer-window confirmation)
  Entry: both layers agree in direction.

Goal: find which resolution + lag combination predicts 2–3 pip continuation
      reliably (the "wave forming and strong enough to continue" question).

Exit: TP / SL / timeout on bars of the analysis resolution (SOP R2 sequencing).
      For small TPs (2–3p), S5 resolution gives the fastest TP detection.

Spread gate: entry skipped when spread > tp * 0.5 OR > P90 IS spread (SOP R5).
             → EUR_USD TP=2 will naturally show 0 trades (spread ≈ 1.7p).
             → USD_JPY TP=2 viable (spread ≈ 0.3p).

Validation:
  IS WF    : 3 temporal chunks, all p/d > 0
  Trade gate: IS trades ≥ 100
  Spread gate: P90 IS (SOP R5)
  MC       : 1000 sign-shuffles, mc_p < 0.05
  OOS      : sealed (SOP R8)

Usage:
    python3 backtest_mtf_confluence.py [--pairs EUR_USD EUR_JPY] [--fast]
"""

import argparse
import itertools
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT  = Path("/path/to/projects/fx-core")
DATA_DIR = PROJECT / "data/s5_ohlc"
RES_DIR  = PROJECT / "research/experiments/mtf_confluence/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Pairs ──────────────────────────────────────────────────────────────────────
PAIRS = {
    "EUR_USD": {"file": "EUR_USD_S5_BA.parquet", "pip": 0.0001},
    "EUR_JPY": {"file": "EUR_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file": "GBP_JPY_S5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file": "USD_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file": "GBP_USD_S5_BA.parquet", "pip": 0.0001},
}

# ── Resolutions derived from S5 via resampling ────────────────────────────────
# bars_per_day: approximate S5 trading bars per day × downfactor
RESOLUTIONS = {
    "s5":  {"freq": None,   "bars_per_day": 2073},   # raw S5
    "s30": {"freq": "30s",  "bars_per_day":  345},   # 2073 / 6
    "m1":  {"freq": "1min", "bars_per_day":  173},   # 2073 / 12
}

# ── Constants ─────────────────────────────────────────────────────────────────
IS_FRAC        = 0.70
N_WF_CHUNKS    = 3
MIN_IS_TRADES  = 100
MC_SHUFFLES    = 1000
TIMEOUT_BARS   = 24   # 24 bars at each resolution (2min S5, 12min S30, 24min M1)
ATR_PERIOD     = 500  # long-period ATR for magnitude gate — stable across all resolutions
                      # S5:500=42min  S30:500=4.2h  M1:500=8.3h

# Active trading session filter (UTC hours, inclusive start / exclusive end)
# London + early NY: typical JPY spread 0.3-0.7p vs 3-10p off-hours.
SESSION_UTC_START = 7
SESSION_UTC_END   = 17

# ── Sweep space ───────────────────────────────────────────────────────────────
# Each group uses 3 lags: base, 2×base, 3×base — ALL must agree in direction,
# AND each lag's momentum must exceed atr_mult × ATR500.
LAG_FAST_BASE_OPT = [1, 2, 3]        # fast group base (bars at chosen resolution)
LAG_SLOW_BASE_OPT = [6, 12, 18]      # slow group base (bars at chosen resolution)
TP_PIPS_OPT       = [2, 3, 5, 10]
SL_PIPS_OPT       = [1, 2, 3, 5]
ACCEL_OPT         = [False, True]    # accel checked per-lag in both groups
ATR_MULT_OPT      = [0.0, 0.5, 1.0, 2.0]  # min momentum = atr_mult × ATR500


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_s5_raw(pair_meta: dict) -> pd.DataFrame:
    """Load S5 parquet, normalise to mid OHLC schema."""
    path = DATA_DIR / pair_meta["file"]
    pip  = pair_meta["pip"]

    probe = duckdb.query(f'SELECT * FROM "{path}" LIMIT 1').df().columns.tolist()
    if "open" in probe:
        df = duckdb.query(
            f'SELECT timestamp, open, high, low, close, bid_c, ask_c '
            f'FROM "{path}" ORDER BY timestamp'
        ).df()
    else:
        df = duckdb.query(
            f'SELECT timestamp, '
            f'(bid_o+ask_o)/2.0 AS open,  (bid_h+ask_h)/2.0 AS high, '
            f'(bid_l+ask_l)/2.0 AS low,   (bid_c+ask_c)/2.0 AS close, '
            f'bid_c, ask_c '
            f'FROM "{path}" ORDER BY timestamp'
        ).df()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["spread_pips"] = ((df["ask_c"] - df["bid_c"]).astype(np.float64) / pip).clip(0.3, 20.0)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(np.float64)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_bars(pair_meta: dict, freq: str | None) -> pd.DataFrame:
    """Return OHLC + spread_pips at the requested frequency (None = S5 raw)."""
    df = _load_s5_raw(pair_meta)
    if freq is None:
        return df

    df = df.set_index("timestamp")
    resampled = pd.DataFrame({
        "open":        df["open"].resample(freq).first(),
        "high":        df["high"].resample(freq).max(),
        "low":         df["low"].resample(freq).min(),
        "close":       df["close"].resample(freq).last(),
        "spread_pips": df["spread_pips"].resample(freq).mean(),
    }).dropna().reset_index()
    return resampled


# ── Numba kernels ─────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def find_exit(close, high, low, entry_bar, entry_px,
              direction, tp_level, sl_level,
              tp_pips_f, sl_pips_f, pip_size, timeout):
    """SOP R2: bull bar → H first for long, L first for short."""
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
def compute_atr(high, low, close, period):
    """Wilder ATR of given period. Long period → stable volatility baseline."""
    n   = len(close)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = high[0] - low[0]
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        if i < period:
            atr[i] = (atr[i-1] * (i - 1) + tr) / i
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr) / period
    return atr


@njit(cache=True, fastmath=True)
def _check_group(close, atr, i, base_lag, n_lags, direction, accel_req, atr_mult):
    """
    Check that all n_lags lags in [base, 2*base, ..., n_lags*base] agree
    in the given direction. If accel_req, each lag must also be accelerating.
    Returns True if all agree.
    """
    threshold = atr_mult * atr[i]
    for k in range(1, n_lags + 1):
        lag  = base_lag * k
        mom  = close[i] - close[i - lag]
        if direction == 1:
            if mom <= 0.0:
                return False
            if atr_mult > 0.0 and mom < threshold:
                return False
            if accel_req:
                mom_prev = close[i - lag] - close[i - 2 * lag]
                if mom <= mom_prev:
                    return False
        else:
            if mom >= 0.0:
                return False
            if atr_mult > 0.0 and -mom < threshold:
                return False
            if accel_req:
                mom_prev = close[i - lag] - close[i - 2 * lag]
                if mom >= mom_prev:
                    return False
    return True


@njit(cache=True, fastmath=True)
def run_segment(close, high, low, spread_pips, atr, in_session, pip_size,
                lag_fast_base, lag_slow_base, n_lags,
                accel_req, atr_mult, tp_pips, sl_pips,
                seg_start, seg_end, timeout, sp_gate):
    """
    Trend-following confluence:
      Fast group: [lf, 2*lf, ..., n_lags*lf] — all agree in direction
      Slow group: [ls, 2*ls, ..., n_lags*ls] — all agree in direction
      Accel: checked independently per lag in both groups (if accel_req).
      in_session: uint8 mask — entry skipped when 0 (off-hours filter).
    """
    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)

    warmup     = lag_slow_base * n_lags * 2 + 2
    start      = max(seg_start, warmup)
    next_entry = start

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)

    i = start
    while i < seg_end - 1:
        if i < next_entry:
            i += 1
            continue

        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > tp_f * 0.5 or sp > sp_gate:
            i += 1
            continue

        direction = np.int64(0)
        for d in (np.int64(1), np.int64(-1)):
            if (_check_group(close, atr, i, lag_fast_base, n_lags, d, accel_req, atr_mult) and
                    _check_group(close, atr, i, lag_slow_base, n_lags, d, accel_req, atr_mult)):
                direction = d
                break

        if direction != 0:
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

        i += 1

    return total_pips, n_trades, n_wins


@njit(cache=True, fastmath=True)
def run_segment_pnl(close, high, low, spread_pips, atr, in_session, pip_size,
                    lag_fast_base, lag_slow_base, n_lags,
                    accel_req, atr_mult, tp_pips, sl_pips,
                    seg_start, seg_end, timeout, sp_gate):
    """Same as run_segment but returns per-trade pnl array for MC."""
    MAX_T   = 50_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup     = lag_slow_base * n_lags * 2 + 2
    start      = max(seg_start, warmup)
    next_entry = start

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)

    i = start
    while i < seg_end - 1 and n_t < MAX_T:
        if i < next_entry:
            i += 1
            continue

        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > tp_f * 0.5 or sp > sp_gate:
            i += 1
            continue

        direction = np.int64(0)
        for d in (np.int64(1), np.int64(-1)):
            if (_check_group(close, atr, i, lag_fast_base, n_lags, d, accel_req, atr_mult) and
                    _check_group(close, atr, i, lag_slow_base, n_lags, d, accel_req, atr_mult)):
                direction = d
                break

        if direction != 0:
            entry_px = close[i] + np.float64(direction) * 0.5 * sp * pip_size
            tp_level = entry_px + np.float64(direction) * tp_f * pip_size
            sl_level = entry_px - np.float64(direction) * sl_f * pip_size

            pnl, bars = find_exit(close, high, low, i, entry_px,
                                   direction, tp_level, sl_level,
                                   tp_f, sl_f, pip_size, timeout)

            pnl_arr[n_t] = pnl
            n_t          += np.int64(1)
            next_entry   = i + bars + np.int64(1)

        i += 1

    return pnl_arr[:n_t]


ATR_MULTS = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)

@njit(parallel=True, cache=True)
def sweep_parallel(close, high, low, spread_pips, atr, in_session, pip_size,
                   configs, n_lags, is_end, wf_starts, wf_ends, timeout, sp_gate):
    """configs[c] = [lag_fast_base, lag_slow_base, accel, tp, sl, atr_mult_idx]."""
    n_cfg = len(configs)
    n_wf  = len(wf_starts)
    n_col = (n_wf + 2) * 3
    out   = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all = len(close)

    for c in prange(n_cfg):
        lf  = np.int64(configs[c, 0])
        ls  = np.int64(configs[c, 1])
        ac  = configs[c, 2] > 0
        tp  = np.float64(configs[c, 3])
        sl_ = np.float64(configs[c, 4])
        nl  = np.int64(n_lags)
        am  = ATR_MULTS[configs[c, 5]]

        p, n, w = run_segment(close, high, low, spread_pips, atr, in_session, pip_size,
                               lf, ls, nl, ac, am, tp, sl_,
                               0, is_end, timeout, sp_gate)
        out[c, 0] = p;  out[c, 1] = n;  out[c, 2] = w

        for k in range(n_wf):
            p, n, w = run_segment(close, high, low, spread_pips, atr, in_session, pip_size,
                                   lf, ls, nl, ac, am, tp, sl_,
                                   wf_starts[k], wf_ends[k], timeout, sp_gate)
            out[c, 3 + k*3    ] = p
            out[c, 3 + k*3 + 1] = n
            out[c, 3 + k*3 + 2] = w

        p, n, w = run_segment(close, high, low, spread_pips, atr, in_session, pip_size,
                               lf, ls, nl, ac, am, tp, sl_,
                               is_end, n_all, timeout, sp_gate)
        out[c, 3 + n_wf*3    ] = p
        out[c, 3 + n_wf*3 + 1] = n
        out[c, 3 + n_wf*3 + 2] = w

    return out


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def run_mc(pnl_arr, is_days, n_shuffles=MC_SHUFFLES, seed=42):
    if len(pnl_arr) < 50:
        return np.nan
    actual_pd = pnl_arr.sum() / is_days
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]),
                       size=(n_shuffles, len(pnl_arr)))
    shuffled = (np.abs(pnl_arr) * signs).sum(axis=1) / is_days
    return float((shuffled >= actual_pd).mean())


# ── Config builder ─────────────────────────────────────────────────────────────
N_LAGS = 3   # lags per group: base, 2×base, 3×base
_ATR_MULT_IDX = {v: i for i, v in enumerate(ATR_MULT_OPT)}

def build_configs():
    rows = []
    for lf, ls, ac, tp, sl, am in itertools.product(
            LAG_FAST_BASE_OPT, LAG_SLOW_BASE_OPT, ACCEL_OPT,
            TP_PIPS_OPT, SL_PIPS_OPT, ATR_MULT_OPT):
        if lf * N_LAGS >= ls:   # fast group top lag must be shorter than slow base
            continue
        if tp < sl:
            continue
        rows.append((lf, ls, int(ac), tp, sl, _ATR_MULT_IDX[am]))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(r[0], r[1], bool(r[2]), r[3], r[4], ATR_MULT_OPT[r[5]]) for r in rows]
    return arr, meta


# ── Per-resolution pair processor ─────────────────────────────────────────────
def process(pair_name, pair_cfg, res_name, res_cfg,
            configs_arr, configs_meta, compiled_already, session_filter=False):
    path = DATA_DIR / pair_cfg["file"]
    pip  = pair_cfg["pip"]

    if not path.exists():
        print(f"  [{pair_name}/{res_name}] parquet missing — skip")
        return []

    print(f"\n{'─'*60}")
    print(f"  {pair_name} @ {res_name}  |  {path.name}")

    bars = load_bars(pair_cfg, res_cfg["freq"])
    close = np.ascontiguousarray(bars["close"].values.astype(np.float64))
    high  = np.ascontiguousarray(bars["high"].values.astype(np.float64))
    low   = np.ascontiguousarray(bars["low"].values.astype(np.float64))
    sp    = np.ascontiguousarray(bars["spread_pips"].values.astype(np.float64))
    atr   = np.ascontiguousarray(compute_atr(high, low, close, ATR_PERIOD))

    ts_col = "timestamp" if "timestamp" in bars.columns else bars.columns[0]
    ts     = pd.to_datetime(bars[ts_col])
    hours  = ts.dt.hour.values

    # Session mask: entries only during active London+NY hours
    sess_mask = ((hours >= SESSION_UTC_START) & (hours < SESSION_UTC_END)).astype(np.uint8)
    if not session_filter:
        sess_mask[:] = 1   # all bars in-session when filter disabled

    n       = len(close)
    is_end  = int(n * IS_FRAC)

    # sp_gate from IS session bars only (SOP R5 — don't leak OOS spread distribution)
    is_sess = sess_mask[:is_end].astype(bool)
    sp_is_sess = sp[:is_end][is_sess] if session_filter and is_sess.any() else sp[:is_end]
    sp_gate = float(np.percentile(sp_is_sess, 90))

    atr_p50  = float(np.median(atr[:is_end]) / pip)   # informational: ATR500 in pips
    ts_idx   = ts.dt.normalize()
    is_days  = max(1, int(ts_idx.iloc[:is_end].nunique()))
    oos_days = max(1, int(ts_idx.iloc[is_end:].nunique()))

    chunk     = is_end // N_WF_CHUNKS
    wf_starts = np.array([k * chunk     for k in range(N_WF_CHUNKS)], dtype=np.int64)
    wf_ends   = np.array([(k+1) * chunk for k in range(N_WF_CHUNKS)], dtype=np.int64)

    bars_per_day = n / max(1, is_days + oos_days)
    sess_label   = f"  [session 07-17 UTC]" if session_filter else ""
    print(f"  {n:,} bars ({bars_per_day:.0f}/day)  IS {is_end:,} ({is_days}d)  "
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  ATR500={atr_p50:.2f}p{sess_label}")

    in_sess = np.ascontiguousarray(sess_mask)

    # Warm up Numba on first call only
    if not compiled_already[0]:
        print("  Compiling Numba ...", end="", flush=True)
        t0 = time.time()
        _ws = wf_starts.copy()
        _we = np.minimum(wf_ends, 2000)
        _ = sweep_parallel(close[:3000], high[:3000], low[:3000], sp[:3000],
                            atr[:3000], in_sess[:3000], np.float64(pip), configs_arr[:4],
                            np.int64(N_LAGS), 2000, _ws, _we, TIMEOUT_BARS, sp_gate)
        compiled_already[0] = True
        print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0  = time.time()
    raw = sweep_parallel(close, high, low, sp, atr, in_sess, np.float64(pip),
                          configs_arr, np.int64(N_LAGS), is_end,
                          wf_starts, wf_ends, TIMEOUT_BARS, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    n_wf  = N_WF_CHUNKS
    rows  = []
    for c, (lf, ls, ac, tp, sl, am) in enumerate(configs_meta):
        is_pips  = raw[c, 0];  is_n  = int(raw[c, 1]);  is_w = int(raw[c, 2])
        oos_pips = raw[c, 3 + n_wf*3]
        oos_n    = int(raw[c, 3 + n_wf*3 + 1])
        oos_w    = int(raw[c, 3 + n_wf*3 + 2])

        wf_pds = []
        for k in range(n_wf):
            pk = raw[c, 3 + k*3];  nk = int(raw[c, 3 + k*3 + 1])
            wf_pds.append(pk / (is_days / n_wf) if nk > 0 else 0.0)

        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0

        wf_pass = (is_n >= MIN_IS_TRADES) and all(p > 0.0 for p in wf_pds)

        rows.append({
            "pair":          pair_name,
            "res":           res_name,
            "lag_fast_base": lf,
            "lag_slow_base": ls,
            "fast_lags":     f"[{lf},{2*lf},{3*lf}]",
            "slow_lags":     f"[{ls},{2*ls},{3*ls}]",
            "atr_mult":      am,
            "accel":      ac,
            "tp":         tp,
            "sl":         sl,
            "is_pd":      round(is_pd,  2),
            "is_wr":      round(is_w / is_n if is_n > 0 else 0.0, 3),
            "is_n":       is_n,
            "wf1":        round(wf_pds[0], 2),
            "wf2":        round(wf_pds[1], 2),
            "wf3":        round(wf_pds[2], 2),
            "wf_pass":    wf_pass,
            "oos_pd":     round(oos_pd, 2),
            "oos_wr":     round(oos_w / oos_n if oos_n > 0 else 0.0, 3),
            "oos_n":      oos_n,
            "mc_p":       np.nan,
        })

    survivors = [r for r in rows if r["wf_pass"]]
    print(f"  WF survivors: {len(survivors)}", end="  (MC ...)" if survivors else "\n", flush=True)
    t0 = time.time()
    for r in survivors:
        pnl_arr = run_segment_pnl(
            close, high, low, sp, atr, in_sess, np.float64(pip),
            r["lag_fast_base"], r["lag_slow_base"], np.int64(N_LAGS),
            bool(r["accel"]), float(r["atr_mult"]), float(r["tp"]), float(r["sl"]),
            0, is_end, TIMEOUT_BARS, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)
    if survivors:
        mc_pass = sum(1 for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
        print(f" {time.time()-t0:.1f}s  mc_p<0.05: {mc_pass}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",   nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--res",     nargs="+", default=list(RESOLUTIONS.keys()),
                        help="Resolutions to test: s5 s30 m1")
    parser.add_argument("--fast",    action="store_true")
    parser.add_argument("--session", action="store_true",
                        help="Only enter during active hours (07-17 UTC). "
                             "Filters off-hours wide-spread bars from entry + sp_gate.")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    print(f"Config space: {len(configs_arr)} per resolution per pair")
    print(f"Resolutions : {args.res}")
    print(f"Pairs       : {args.pairs}")
    print(f"Session flt : {'07-17 UTC' if args.session else 'off (all hours)'}")

    if args.fast:
        configs_arr  = configs_arr[:4]
        configs_meta = configs_meta[:4]

    compiled = [False]   # shared across calls so Numba only compiles once
    all_rows = []

    for res_name in args.res:
        if res_name not in RESOLUTIONS:
            print(f"Unknown resolution {res_name}, skip")
            continue
        res_cfg = RESOLUTIONS[res_name]
        print(f"\n{'='*60}")
        print(f"  RESOLUTION: {res_name.upper()}")

        for pair_name in args.pairs:
            if pair_name not in PAIRS:
                continue
            rows = process(pair_name, PAIRS[pair_name], res_name, res_cfg,
                           configs_arr, configs_meta, compiled, args.session)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = RES_DIR / "mtf_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    mc_pass = df[df["wf_pass"] & (df["mc_p"] < 0.05)]
    print(f"\n{'='*80}")
    if len(mc_pass) > 0:
        print(f"PASSED WF + MC  ({len(mc_pass)} configs)  ranked by OOS p/d:")
        cols = ["pair","res","lag_fast_base","lag_slow_base","fast_lags","slow_lags","accel","tp","sl",
                "is_pd","oos_pd","oos_wr","oos_n","mc_p"]
        print(mc_pass.sort_values("oos_pd", ascending=False)[cols].head(40).to_string(index=False))
    else:
        print("No WF + MC survivors.")
        wf_only = df[df["wf_pass"]].sort_values("is_pd", ascending=False).head(20)
        if len(wf_only):
            print(f"\nWF-only top 20:")
            print(wf_only[["pair","res","lag_fast_base","lag_slow_base","fast_lags","slow_lags","accel","tp","sl",
                            "is_pd","wf1","wf2","wf3","mc_p"]].to_string(index=False))
        else:
            print("No WF survivors at any resolution.")

    # ── Per-resolution breakdown ───────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("WF survivors by resolution:")
    for res in df["res"].unique():
        sub = df[(df["res"] == res) & df["wf_pass"]]
        mc_n = int((sub["mc_p"] < 0.05).sum())
        print(f"  {res}: WF={len(sub)}  MC={mc_n}")

    # ── Small-TP breakdown ─────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("TP=2 and TP=3 results (all configs, best IS p/d per pair×res):")
    for tp_target in [2, 3]:
        sub = df[df["tp"] == tp_target]
        if sub.empty:
            continue
        print(f"\n  TP={tp_target}:")
        best = (sub.groupby(["pair","res"])
                   .apply(lambda g: g.nlargest(1, "is_pd"), include_groups=False)
                   .reset_index(level=[0,1]))
        print(best[["pair","res","lag_fast_base","lag_slow_base","fast_lags","slow_lags","accel","sl",
                     "is_pd","is_wr","is_n","wf_pass","mc_p"]].to_string(index=False))

    # ── Trade frequency ───────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("Median IS trade count by resolution × pair:")
    pivot = df.groupby(["res","pair"])["is_n"].median().unstack(level="pair")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
