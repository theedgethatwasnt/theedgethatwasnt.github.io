#!/usr/bin/env python3
"""
Harvester v2 — Variable TP/SL + D3@S5 Wavelet IC Study
=========================================================
Key change from v1: TP and SL are proportional to the distance from SMA at entry.
  tp_pips = dist_pips × tp_frac   (0.3–0.6)
  sl_pips = dist_pips × sl_frac   (1.5–2.5)

This decouples the strategy from the fixed-TP-vs-spread trap in v1.
At dist=4p, tp_frac=0.5 → TP=2p (enough to cover spread). R:R = 0.5/1.5 needs 75% WR.
At dist=6p, tp_frac=0.5 → TP=3p. Same R:R, larger absolute room.

Entry logic unchanged:
  1. SMA(period) on M5 close — causal rolling.
  2. Distance gate: |close - SMA| ∈ [dist_mult_min, dist_mult_max] × sp_gate.
  3. Bar-exhaustion: last n_consec bars same-direction AND price on same side of SMA.
  4. ATR gate (optional): n-bar price move ≥ atr_mult × ATR14.
  5. TP quality gate: tp_f > 0.5 × sp (adaptive, replaces fixed sp > tp×0.5).

Wavelet IC study (--ic flag):
  After finding WF+MC survivors, re-run them on the available S5 data period
  (data/s5_ohlc/ typically covers 10–13 months). At each trade entry, extract:
    • D3@S5 = Haar level-3 on last 8 S5 bars (40s intra-bar pattern)
    • D4@S5 = Haar level-4 on last 16 S5 bars (80s intra-bar)
  Correlate wavelet alignment with trade outcome to score wave quality at entry.

Usage:
    python3 backtest_harvester_v2.py [--pairs USD_JPY EUR_USD] [--fast] [--ic]
"""

import argparse
import itertools
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT   = Path("/path/to/projects/fx-core")
M5_DIR    = PROJECT / "data/m5_ba"
S5_DIR    = PROJECT / "data/s5_ohlc"
RES_DIR   = PROJECT / "research/experiments/harvester/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "EUR_JPY": {"file_m5": "EUR_JPY_M5_BA.parquet", "file_s5": "EUR_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file_m5": "GBP_JPY_M5_BA.parquet", "file_s5": "GBP_JPY_S5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file_m5": "USD_JPY_M5_BA.parquet", "file_s5": "USD_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file_m5": "GBP_USD_M5_BA.parquet", "file_s5": "GBP_USD_S5_BA.parquet", "pip": 0.0001},
    "EUR_USD": {"file_m5": "EUR_USD_M5_BA.parquet", "file_s5": "EUR_USD_S5_BA.parquet", "pip": 0.0001},
}

IS_FRAC       = 0.70
N_WF_CHUNKS   = 3
MIN_IS_TRADES = 50
MC_SHUFFLES   = 1000
MAX_HOLD_BARS = 24       # 2h timeout at M5
ATR_PERIOD    = 14
SESSION_START = 7
SESSION_END   = 21

# ── Sweep space ────────────────────────────────────────────────────────────────
SMA_OPT          = [5, 7, 10, 14]
N_CONSEC_OPT     = [2, 3, 4]
DIST_MULT_MIN_OPT = [1.5, 2.0, 2.5]    # min dist = mult × sp_gate
DIST_MULT_MAX_OPT = [4.0, 6.0, 10.0]   # max dist = mult × sp_gate
TP_FRAC_OPT      = [0.3, 0.4, 0.5, 0.6]  # TP = dist × frac
SL_FRAC_OPT      = [1.5, 2.0, 2.5]    # SL = dist × frac  (always > tp_frac)
ATR_MULT_OPT     = [0.0, 0.5, 1.0]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_m5(pair_cfg: dict) -> pd.DataFrame:
    path = M5_DIR / pair_cfg["file_m5"]
    pip  = pair_cfg["pip"]
    df = duckdb.query(
        f'SELECT timestamp, open, high, low, close, bid_c, ask_c '
        f'FROM "{path}" ORDER BY timestamp'
    ).df()
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)
    df["spread_pips"] = ((df["ask_c"] - df["bid_c"]).astype(np.float64) / pip).clip(0.1, 30.0)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(np.float64)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_s5(pair_cfg: dict) -> pd.DataFrame | None:
    path = S5_DIR / pair_cfg["file_s5"]
    if not path.exists():
        return None
    pip = pair_cfg["pip"]
    probe = duckdb.query(f'SELECT * FROM "{path}" LIMIT 1').df().columns.tolist()
    if "open" in probe:
        df = duckdb.query(f'SELECT timestamp, close FROM "{path}" ORDER BY timestamp').df()
    else:
        df = duckdb.query(
            f'SELECT timestamp, (bid_c+ask_c)/2.0 AS close FROM "{path}" ORDER BY timestamp'
        ).df()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["close"]     = df["close"].astype(np.float64)
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_sma(close, period):
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    s = np.float64(0.0)
    for i in range(n):
        s += close[i]
        if i >= period:
            s -= close[i - period]
        out[i] = s / min(i + 1, period)
    return out


@njit(cache=True, fastmath=True)
def compute_atr14(high, low, close):
    n   = len(close)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = high[0] - low[0]
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr) / ATR_PERIOD \
                 if i >= ATR_PERIOD else (atr[i-1] * (i-1) + tr) / i
    return atr


@njit(cache=True, fastmath=True)
def run_trade(close, high, low, spread_pips, entry_bar, entry_px,
              direction, tp_price, sl_price, pip_size, max_hold):
    """Returns (pnl_pips, bars_held, exit_reason). tp_price/sl_price in price units."""
    n      = len(close)
    tp_lev = entry_px + np.float64(direction) * tp_price
    sl_lev = entry_px - np.float64(direction) * sl_price

    end = min(entry_bar + max_hold + 1, n)
    for j in range(entry_bar + 1, end):
        bull = close[j] >= close[j-1]
        hit_tp = False; hit_sl = False
        if direction == 1:
            if bull:
                hit_tp = high[j] >= tp_lev
                hit_sl = low[j]  <= sl_lev
            else:
                hit_sl = low[j]  <= sl_lev
                hit_tp = high[j] >= tp_lev
        else:
            if not bull:
                hit_tp = low[j]  <= tp_lev
                hit_sl = high[j] >= sl_lev
            else:
                hit_sl = high[j] >= sl_lev
                hit_tp = low[j]  <= tp_lev

        if hit_tp:
            return tp_price / pip_size,  np.int64(j - entry_bar), np.int64(0)
        if hit_sl:
            return -sl_price / pip_size, np.int64(j - entry_bar), np.int64(1)

    last = min(entry_bar + max_hold, n - 1)
    pnl  = (close[last] - entry_px) * np.float64(direction) / pip_size \
           - np.float64(0.5) * spread_pips[last]
    return pnl, np.int64(last - entry_bar), np.int64(2)


@njit(cache=True, fastmath=True)
def run_segment_v2(close, high, low, open_, spread_pips, sma, atr,
                   in_session, pip_size,
                   sma_period, n_consec, dist_mult_min, dist_mult_max,
                   tp_frac, sl_frac, atr_mult,
                   seg_start, seg_end, sp_gate):
    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, np.int64(warmup))
    next_entry = start

    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)
    n_tp       = np.int64(0)

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate

    i = start
    while i < seg_end - 1:
        if i < next_entry:
            i += 1
            continue
        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > sp_gate:
            i += 1
            continue

        dist = abs(close[i] - sma[i]) / pip_size
        if dist < dist_lo or dist > dist_hi:
            i += 1
            continue

        # Compute variable TP and SL from current distance (in price units)
        tp_price = dist * tp_frac * pip_size
        sl_price = dist * sl_frac * pip_size

        # Adaptive TP quality gate: TP must exceed half the spread
        if tp_price < np.float64(0.5) * sp * pip_size:
            i += 1
            continue

        above_sma = close[i] > sma[i]

        all_bull = True
        all_bear = True
        for k in range(np.int64(1), n_consec + np.int64(1)):
            j = i - k
            if close[j] < open_[j]:
                all_bull = False
            if close[j] >= open_[j]:
                all_bear = False
            if not all_bull and not all_bear:
                break

        direction = np.int64(0)
        if all_bull and above_sma:
            direction = np.int64(-1)
        elif all_bear and not above_sma:
            direction = np.int64(1)

        if direction == np.int64(0):
            i += 1
            continue

        if atr_mult > np.float64(0.0):
            n_bar_move = abs(close[i] - close[i - n_consec])
            if n_bar_move < atr_mult * atr[i]:
                i += 1
                continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl_pips, bars, reason = run_trade(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        total_pips += pnl_pips
        n_trades   += np.int64(1)
        if pnl_pips > np.float64(0.0): n_wins += np.int64(1)
        if reason == np.int64(0):       n_tp   += np.int64(1)

        next_entry = i + bars + np.int64(1)
        i += 1

    return total_pips, n_trades, n_wins, n_tp


@njit(cache=True, fastmath=True)
def run_segment_pnl_v2(close, high, low, open_, spread_pips, sma, atr,
                       in_session, pip_size,
                       sma_period, n_consec, dist_mult_min, dist_mult_max,
                       tp_frac, sl_frac, atr_mult,
                       seg_start, seg_end, sp_gate):
    MAX_T   = 10_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, np.int64(warmup))
    next_entry = start

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate

    i = start
    while i < seg_end - 1 and n_t < MAX_T:
        if i < next_entry:
            i += 1
            continue
        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > sp_gate:
            i += 1
            continue

        dist = abs(close[i] - sma[i]) / pip_size
        if dist < dist_lo or dist > dist_hi:
            i += 1
            continue

        tp_price = dist * tp_frac * pip_size
        sl_price = dist * sl_frac * pip_size

        if tp_price < np.float64(0.5) * sp * pip_size:
            i += 1
            continue

        above_sma = close[i] > sma[i]

        all_bull = True
        all_bear = True
        for k in range(np.int64(1), n_consec + np.int64(1)):
            j = i - k
            if close[j] < open_[j]:
                all_bull = False
            if close[j] >= open_[j]:
                all_bear = False
            if not all_bull and not all_bear:
                break

        direction = np.int64(0)
        if all_bull and above_sma:
            direction = np.int64(-1)
        elif all_bear and not above_sma:
            direction = np.int64(1)

        if direction == np.int64(0):
            i += 1
            continue

        if atr_mult > np.float64(0.0):
            n_bar_move = abs(close[i] - close[i - n_consec])
            if n_bar_move < atr_mult * atr[i]:
                i += 1
                continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl_pips, bars, reason = run_trade(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        pnl_arr[n_t] = pnl_pips
        n_t          += np.int64(1)
        next_entry   = i + bars + np.int64(1)
        i += 1

    return pnl_arr[:n_t]


# ── Config builder ─────────────────────────────────────────────────────────────
# cols: si  nc  dmi  dxi  tfi  sfi  ai
SMA_ARR      = np.array(SMA_OPT,          dtype=np.int32)
DIST_MIN_ARR = np.array(DIST_MULT_MIN_OPT, dtype=np.float64)
DIST_MAX_ARR = np.array(DIST_MULT_MAX_OPT, dtype=np.float64)
TP_FRAC_ARR  = np.array(TP_FRAC_OPT,      dtype=np.float64)
SL_FRAC_ARR  = np.array(SL_FRAC_OPT,      dtype=np.float64)
ATR_MULT_ARR = np.array(ATR_MULT_OPT,     dtype=np.float64)


def build_configs():
    rows = []
    for si, nc, dmi, dxi, tfi, sfi, ai in itertools.product(
            range(len(SMA_OPT)), N_CONSEC_OPT,
            range(len(DIST_MULT_MIN_OPT)), range(len(DIST_MULT_MAX_OPT)),
            range(len(TP_FRAC_OPT)), range(len(SL_FRAC_OPT)),
            range(len(ATR_MULT_OPT))):
        if DIST_MULT_MAX_OPT[dxi] <= DIST_MULT_MIN_OPT[dmi]:
            continue
        rows.append((si, nc, dmi, dxi, tfi, sfi, ai))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(SMA_OPT[r[0]], r[1],
             DIST_MULT_MIN_OPT[r[2]], DIST_MULT_MAX_OPT[r[3]],
             TP_FRAC_OPT[r[4]], SL_FRAC_OPT[r[5]],
             ATR_MULT_OPT[r[6]]) for r in rows]
    return arr, meta


@njit(parallel=True, cache=True)
def sweep_parallel_v2(close, high, low, open_, spread_pips, sma_v, sma_period,
                      atr, in_session, pip_size,
                      configs, is_end, wf_starts, wf_ends, sp_gate):
    n_cfg  = len(configs)
    n_wf   = len(wf_starts)
    n_stat = 4
    n_col  = (n_wf + 2) * n_stat
    out    = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all  = len(close)

    for c in prange(n_cfg):
        nc   = np.int64(configs[c, 1])
        dmi  = configs[c, 2]
        dxi  = configs[c, 3]
        tfi  = configs[c, 4]
        sfi  = configs[c, 5]
        ai   = configs[c, 6]

        dmin = DIST_MIN_ARR[dmi]
        dmax = DIST_MAX_ARR[dxi]
        tf   = TP_FRAC_ARR[tfi]
        sf   = SL_FRAC_ARR[sfi]
        am   = ATR_MULT_ARR[ai]

        p0,p1,p2,p3 = run_segment_v2(
            close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
            np.int64(sma_period), nc, dmin, dmax, tf, sf, am, 0, is_end, sp_gate)
        out[c,0]=p0; out[c,1]=p1; out[c,2]=p2; out[c,3]=p3

        for k in range(n_wf):
            p0,p1,p2,p3 = run_segment_v2(
                close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
                np.int64(sma_period), nc, dmin, dmax, tf, sf, am,
                wf_starts[k], wf_ends[k], sp_gate)
            b = n_stat + k * n_stat
            out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3

        p0,p1,p2,p3 = run_segment_v2(
            close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
            np.int64(sma_period), nc, dmin, dmax, tf, sf, am, is_end, n_all, sp_gate)
        b = n_stat + n_wf * n_stat
        out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3

    return out


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def run_mc(pnl_arr, is_days, n_shuffles=MC_SHUFFLES, seed=42):
    if len(pnl_arr) < 30:
        return np.nan
    actual_pd = pnl_arr.sum() / is_days
    rng    = np.random.default_rng(seed)
    signs  = rng.choice(np.array([-1.0, 1.0]), size=(n_shuffles, len(pnl_arr)))
    shuffl = (np.abs(pnl_arr) * signs).sum(axis=1) / is_days
    return float((shuffl >= actual_pd).mean())


# ── Haar D3/D4 at S5 resolution (IC study) ────────────────────────────────────
@njit(cache=True, fastmath=True)
def haar_detail_s5(close, i, level):
    """Causal Haar D_level on S5 data (stride=1)."""
    window = np.int64(1) << np.int64(level)
    half   = window >> np.int64(1)
    if i < window - np.int64(1):
        return np.float64(0.0)
    left_sum  = np.float64(0.0)
    right_sum = np.float64(0.0)
    for k in range(half):
        left_sum  += close[i - (window - np.int64(1) - np.int64(k))]
        right_sum += close[i - (half  - np.int64(1) - np.int64(k))]
    return (right_sum - left_sum) / np.float64(half)


def ic_study_wavelet(pair_name, m5_df, is_end, survivor_meta, s5_df, pip):
    """
    For each trade entry in the IS period (where S5 data overlaps),
    extract D3@S5 and D4@S5 at the M5 entry bar and correlate with outcome.

    survivor_meta: (sma_p, nc, dmin_mult, dmax_mult, tp_frac, sl_frac, am)
    """
    if s5_df is None or len(s5_df) == 0:
        return None

    # Align S5 with M5 by timestamp
    m5_ts  = m5_df["timestamp"].values
    s5_ts  = s5_df["timestamp"].values
    s5_cl  = s5_df["close"].values.astype(np.float64)

    # Find S5 range that overlaps with M5 IS period
    m5_is_end_ts = m5_ts[min(is_end, len(m5_ts)-1)]

    # Re-run entries in Python to collect wavelet features
    sma_p, nc, dmin_mult, dmax_mult, tp_frac, sl_frac, am = survivor_meta
    close  = m5_df["close"].values.astype(np.float64)
    high   = m5_df["high"].values.astype(np.float64)
    low    = m5_df["low"].values.astype(np.float64)
    open_  = m5_df["open"].values.astype(np.float64)
    sp     = m5_df["spread_pips"].values.astype(np.float64)
    hours  = m5_df["timestamp"].dt.hour.values
    sess   = (hours >= SESSION_START) & (hours < SESSION_END)

    # Compute SMA and ATR14
    from numba import njit as _njit
    sma_arr = compute_sma(close, sma_p)
    atr_arr = compute_atr14(high, low, close)

    is_sess_sp = sp[:is_end][sess[:is_end]]
    sp_gate    = float(np.percentile(is_sess_sp, 90)) if len(is_sess_sp) > 0 \
                 else float(np.percentile(sp[:is_end], 90))

    dist_lo = dmin_mult * sp_gate
    dist_hi = dmax_mult * sp_gate

    records = []
    warmup     = sma_p + nc + 2
    next_entry = warmup

    for i in range(warmup, is_end - 1):
        # S5 data only covers recent period — skip if M5 bar before S5 start
        if m5_ts[i] < s5_ts[0]:
            continue

        if i < next_entry or not sess[i]:
            continue

        sp_i = sp[i]
        if sp_i > sp_gate:
            continue

        dist = abs(close[i] - sma_arr[i]) / pip
        if dist < dist_lo or dist > dist_hi:
            continue

        tp_price = dist * tp_frac * pip
        sl_price = dist * sl_frac * pip
        if tp_price < 0.5 * sp_i * pip:
            continue

        above_sma = close[i] > sma_arr[i]
        all_bull   = all(close[i-k] >= open_[i-k] for k in range(1, nc+1))
        all_bear   = all(close[i-k] <  open_[i-k] for k in range(1, nc+1))

        direction = 0
        if all_bull and above_sma:
            direction = -1
        elif all_bear and not above_sma:
            direction = 1
        if direction == 0:
            continue

        if am > 0.0:
            if abs(close[i] - close[i - nc]) < am * atr_arr[i]:
                continue

        # Find the S5 bar closest to (and ≤) this M5 bar timestamp
        s5_idx = np.searchsorted(s5_ts, m5_ts[i], side='right') - 1
        if s5_idx < 16:  # need at least 16 bars for D4
            continue

        d3 = haar_detail_s5(s5_cl, s5_idx, 3)  # 8 bars × 5s = 40s
        d4 = haar_detail_s5(s5_cl, s5_idx, 4)  # 16 bars × 5s = 80s

        # Alignment: >0 if wavelet direction agrees with the run (confirming exhaustion)
        # d3_align=+1 means wave same direction as the n-bar run we're fading
        # d3_align=-1 means wave already reversing (early reversion signal)
        d3_align = 1 if (direction == -1 and d3 > 0) or (direction == 1 and d3 < 0) else -1
        d4_align = 1 if (direction == -1 and d4 > 0) or (direction == 1 and d4 < 0) else -1

        # Run the trade and record outcome
        entry_px = close[i] + direction * 0.5 * sp_i * pip
        pnl, bars, reason = run_trade(
            close, high, low, sp,
            i, entry_px, direction,
            tp_price, sl_price, MAX_HOLD_BARS)
        pnl_pips = pnl / pip

        records.append({
            "pair":      pair_name,
            "m5_bar":    i,
            "timestamp": m5_ts[i],
            "direction": direction,
            "dist_pips": round(dist, 3),
            "tp_pips":   round(tp_price / pip, 3),
            "sl_pips":   round(sl_price / pip, 3),
            "d3_raw":    round(float(d3) / pip, 4),
            "d4_raw":    round(float(d4) / pip, 4),
            "d3_energy": round(float(d3)**2 / pip**2, 6),
            "d4_energy": round(float(d4)**2 / pip**2, 6),
            "d3_align":  d3_align,   # +1 = run confirmed, -1 = run reversing
            "d4_align":  d4_align,
            "pnl_pips":  round(pnl_pips, 4),
            "win":       int(pnl_pips > 0),
            "reason":    int(reason),
        })
        next_entry = i + bars + 1

    if not records:
        return None

    df_ic = pd.DataFrame(records)
    n = len(df_ic)

    # IC: does d3_align predict win?
    if df_ic["d3_align"].std() > 0 and df_ic["win"].std() > 0:
        ic_d3, pval_d3 = stats.pearsonr(df_ic["d3_align"], df_ic["win"])
        ic_d4, pval_d4 = stats.pearsonr(df_ic["d4_align"], df_ic["win"])
        # t-stat for significance
        t_d3 = ic_d3 * np.sqrt(n - 2) / np.sqrt(max(1e-9, 1 - ic_d3**2))
        t_d4 = ic_d4 * np.sqrt(n - 2) / np.sqrt(max(1e-9, 1 - ic_d4**2))
    else:
        ic_d3 = ic_d4 = pval_d3 = pval_d4 = t_d3 = t_d4 = 0.0

    # WR split by d3_align
    wr_d3_pos = df_ic[df_ic["d3_align"] == 1]["win"].mean() if (df_ic["d3_align"]==1).any() else 0.0
    wr_d3_neg = df_ic[df_ic["d3_align"] ==-1]["win"].mean() if (df_ic["d3_align"]==-1).any() else 0.0
    n_d3_pos  = (df_ic["d3_align"] == 1).sum()
    n_d3_neg  = (df_ic["d3_align"] ==-1).sum()

    print(f"\n    IC Study — {pair_name} ({n} trades overlap with S5 data):")
    print(f"    D3@S5 IC={ic_d3:+.4f}  t={t_d3:+.2f}  p={pval_d3:.3f}")
    print(f"    D4@S5 IC={ic_d4:+.4f}  t={t_d4:+.2f}  p={pval_d4:.3f}")
    print(f"    WR when D3 run-confirmed (+1): {wr_d3_pos:.1%}  n={n_d3_pos}")
    print(f"    WR when D3 run-reversing (-1): {wr_d3_neg:.1%}  n={n_d3_neg}")

    return {"ic_d3": ic_d3, "t_d3": t_d3, "ic_d4": ic_d4, "t_d4": t_d4,
            "wr_d3_pos": wr_d3_pos, "wr_d3_neg": wr_d3_neg,
            "n_ic": n, "records": df_ic}


# ── Helper: run sweep for one SMA group ───────────────────────────────────────
def _run_sweep(close, high, low, open_, sp, smas_list, atr, in_sess, pip,
               configs_arr, is_end, wf_starts, wf_ends, sp_gate):
    n_cfg  = len(configs_arr)
    n_wf   = len(wf_starts)
    n_stat = 4
    out    = np.zeros((n_cfg, (n_wf + 2) * n_stat), dtype=np.float64)

    for si, sma_arr in enumerate(smas_list):
        mask = configs_arr[:, 0] == si
        if not mask.any():
            continue
        idx = np.where(mask)[0].astype(np.int64)
        sub = configs_arr[mask]
        sub_out = sweep_parallel_v2(
            close, high, low, open_, sp, sma_arr, np.int64(SMA_OPT[si]),
            atr, in_sess, pip,
            sub, is_end, wf_starts, wf_ends, sp_gate)
        out[idx] = sub_out

    return out


# ── Process one pair ───────────────────────────────────────────────────────────
def process(pair_name, pair_cfg, configs_arr, configs_meta,
            compiled_already, run_ic=False):
    m5_path = M5_DIR / pair_cfg["file_m5"]
    if not m5_path.exists():
        print(f"  {pair_name}: M5 parquet missing — skip")
        return []

    print(f"\n{'─'*60}")
    print(f"  {pair_name}  |  {pair_cfg['file_m5']}")
    pip = pair_cfg["pip"]

    df    = load_m5(pair_cfg)
    close = np.ascontiguousarray(df["close"].values.astype(np.float64))
    high  = np.ascontiguousarray(df["high"].values.astype(np.float64))
    low   = np.ascontiguousarray(df["low"].values.astype(np.float64))
    open_ = np.ascontiguousarray(df["open"].values.astype(np.float64))
    sp    = np.ascontiguousarray(df["spread_pips"].values.astype(np.float64))
    atr   = np.ascontiguousarray(compute_atr14(high, low, close))

    hours    = df["timestamp"].dt.hour.values
    sess_msk = ((hours >= SESSION_START) & (hours < SESSION_END)).astype(np.uint8)
    in_sess  = np.ascontiguousarray(sess_msk)

    smas_list = [np.ascontiguousarray(compute_sma(close, p)) for p in SMA_OPT]

    n      = len(close)
    is_end = int(n * IS_FRAC)

    is_sess_sp = sp[:is_end][sess_msk[:is_end].astype(bool)]
    sp_gate    = float(np.percentile(is_sess_sp, 90)) if len(is_sess_sp) > 0 \
                 else float(np.percentile(sp[:is_end], 90))

    ts_idx   = df["timestamp"].dt.normalize()
    is_days  = max(1, int(ts_idx.iloc[:is_end].nunique()))
    oos_days = max(1, int(ts_idx.iloc[is_end:].nunique()))
    bpd      = n / max(1, is_days + oos_days)
    atr_med  = float(np.median(atr[:is_end]) / pip)

    chunk     = is_end // N_WF_CHUNKS
    wf_starts = np.array([k * chunk     for k in range(N_WF_CHUNKS)], dtype=np.int64)
    wf_ends   = np.array([(k+1) * chunk for k in range(N_WF_CHUNKS)], dtype=np.int64)

    print(f"  {n:,} bars ({bpd:.0f}/day)  IS {is_end:,} ({is_days}d)  "
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  ATR14={atr_med:.2f}p")

    if not compiled_already[0]:
        print("  Compiling Numba ...", end="", flush=True)
        t0 = time.time()
        _ws = wf_starts.copy(); _we = np.minimum(wf_ends, 2000)
        _ = _run_sweep(close[:3000], high[:3000], low[:3000], open_[:3000], sp[:3000],
                       smas_list, atr[:3000], in_sess[:3000], np.float64(pip),
                       configs_arr[:4], 2000, _ws, _we, sp_gate)
        compiled_already[0] = True
        print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0  = time.time()
    raw = _run_sweep(close, high, low, open_, sp, smas_list, atr, in_sess, np.float64(pip),
                     configs_arr, is_end, wf_starts, wf_ends, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    n_stat = 4
    n_wf   = N_WF_CHUNKS
    rows   = []

    for c, (sma_p, nc, dmin_m, dmax_m, tf, sf, am) in enumerate(configs_meta):
        def _g(seg, s):
            if   seg == "is":  return raw[c, s]
            elif seg == "oos": return raw[c, n_stat + n_wf*n_stat + s]
            else:              return raw[c, n_stat + seg*n_stat + s]

        is_pips  = _g("is",0); is_n = int(_g("is",1)); is_w = int(_g("is",2)); is_tp = int(_g("is",3))
        oos_pips = _g("oos",0); oos_n = int(_g("oos",1)); oos_w = int(_g("oos",2))

        wf_pds = []
        for k in range(n_wf):
            pk = _g(k,0); nk = int(_g(k,1))
            wf_pds.append(pk / (is_days / n_wf) if nk > 0 else 0.0)

        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0
        wf_pass = (is_n >= MIN_IS_TRADES) and all(p > 0 for p in wf_pds)

        rows.append({
            "pair":       pair_name,
            "sma":        sma_p,   "n_consec":  nc,
            "dist_min_m": dmin_m,  "dist_max_m": dmax_m,
            "tp_frac":    tf,      "sl_frac":    sf,
            "atr_mult":   am,
            "is_pd":      round(is_pd,  2),
            "is_wr":      round(is_w / is_n if is_n > 0 else 0.0, 3),
            "is_n":       is_n,
            "tp_pct":     round(is_tp / is_n if is_n > 0 else 0.0, 3),
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
    print(f"  WF survivors: {len(survivors)}", end="  (MC ...)\n" if survivors else "\n", flush=True)

    for r in survivors:
        si       = SMA_OPT.index(r["sma"])
        sma_arr  = smas_list[si]
        pnl_arr  = run_segment_pnl_v2(
            close, high, low, open_, sp, sma_arr, atr, in_sess, np.float64(pip),
            int(r["sma"]), int(r["n_consec"]),
            float(r["dist_min_m"]), float(r["dist_max_m"]),
            float(r["tp_frac"]), float(r["sl_frac"]), float(r["atr_mult"]),
            0, is_end, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)

        # Wavelet IC study on survivors if --ic flag given and S5 data available
        if run_ic:
            s5_df = load_s5(pair_cfg)
            meta  = (r["sma"], r["n_consec"], r["dist_min_m"], r["dist_max_m"],
                     r["tp_frac"], r["sl_frac"], r["atr_mult"])
            ic_result = ic_study_wavelet(pair_name, df, is_end, meta, s5_df, pip)
            if ic_result:
                r["ic_d3"]     = round(ic_result["ic_d3"], 4)
                r["t_d3"]      = round(ic_result["t_d3"], 2)
                r["ic_d4"]     = round(ic_result["ic_d4"], 4)
                r["t_d4"]      = round(ic_result["t_d4"], 2)
                r["wr_d3_pos"] = round(ic_result["wr_d3_pos"], 3)
                r["wr_d3_neg"] = round(ic_result["wr_d3_neg"], 3)
                r["n_ic"]      = ic_result["n_ic"]
                # Save per-trade IC records
                ic_path = RES_DIR / f"harvester_v2_ic_{pair_name}.csv"
                ic_result["records"].to_csv(ic_path, index=False)
                print(f"  IC records → {ic_path}")

    if survivors:
        mc_ok = sum(1 for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
        print(f"  mc_p<0.05: {mc_ok}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--fast",  action="store_true")
    parser.add_argument("--ic",    action="store_true", help="Run D3/D4@S5 IC study on WF survivors")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    print(f"Config space  : {len(configs_arr)} per pair")
    print(f"Pairs         : {args.pairs}")
    print(f"Entry gate    : |close-SMA| ∈ [dm_min, dm_max]×sp_gate "
          f"+ n_consec bars + ATR gate")
    print(f"Exit          : TP=dist×tp_frac | SL=dist×sl_frac | {MAX_HOLD_BARS}-bar timeout")
    print(f"TP fracs      : {TP_FRAC_OPT}   SL fracs: {SL_FRAC_OPT}")
    print(f"Session       : {SESSION_START:02d}:00–{SESSION_END:02d}:00 UTC")
    if args.ic:
        print(f"IC mode       : D3/D4@S5 wavelet features on WF survivors")

    if args.fast:
        configs_arr  = configs_arr[:8]
        configs_meta = configs_meta[:8]

    compiled = [False]
    all_rows = []

    for pair_name in args.pairs:
        if pair_name not in PAIRS:
            continue
        rows = process(pair_name, PAIRS[pair_name], configs_arr, configs_meta,
                       compiled, run_ic=args.ic)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    out_path = RES_DIR / "harvester_v2_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df_out)} rows)")

    mc_pass = df_out[df_out["wf_pass"] & (df_out["mc_p"] < 0.05)].copy() \
              if len(df_out) > 0 else pd.DataFrame()
    print(f"\n{'='*80}")
    if len(mc_pass) > 0:
        print(f"PASSED WF + MC  ({len(mc_pass)} configs):")
        cols = ["pair","sma","n_consec","dist_min_m","dist_max_m",
                "tp_frac","sl_frac","atr_mult",
                "is_pd","is_wr","is_n","tp_pct","oos_pd","oos_wr","oos_n","mc_p"]
        print(mc_pass.sort_values("oos_pd", ascending=False)[cols].head(40).to_string(index=False))
    else:
        print("No WF + MC survivors.")
        if len(df_out) > 0:
            wf_only = df_out[df_out["wf_pass"]].sort_values("is_pd", ascending=False).head(20)
            if len(wf_only):
                print(f"\nWF-only survivors ({len(wf_only)}):")
                print(wf_only[["pair","sma","n_consec","dist_min_m","dist_max_m",
                               "tp_frac","sl_frac","atr_mult",
                               "is_pd","is_wr","is_n","tp_pct",
                               "wf1","wf2","wf3"]].to_string(index=False))
            else:
                top = df_out[df_out["is_n"] >= 10].sort_values("is_pd", ascending=False).head(20)
                if len(top):
                    print(f"\nTop IS p/d (diagnostic):")
                    print(top[["pair","sma","n_consec","dist_min_m","dist_max_m",
                               "tp_frac","sl_frac","atr_mult",
                               "is_pd","is_wr","is_n","tp_pct",
                               "wf1","wf2","wf3"]].to_string(index=False))

    print(f"\n{'─'*80}")
    print("WF survivors by pair:")
    for p in args.pairs:
        sub = df_out[df_out["pair"] == p]
        wf  = sub["wf_pass"].sum()
        mc  = (sub["wf_pass"] & (sub["mc_p"] < 0.05)).sum()
        best_pd = sub[sub["is_n"] >= 10]["is_pd"].max() if len(sub[sub["is_n"] >= 10]) > 0 else 0.0
        print(f"  {p}: WF={wf}  MC={mc}  best IS p/d={best_pd:.2f}")


if __name__ == "__main__":
    main()
