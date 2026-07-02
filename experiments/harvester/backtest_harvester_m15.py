#!/usr/bin/env python3
"""
Harvester M15 — SMA mean-reversion + bar-exhaustion on M15 bars
================================================================
Same logic as v2 (proportional TP/SL) but using M15 bars resampled from M5 BA.

Why M15:
  On M5, strict distance filters (dist >= 3×sp) give WR ~85% but median SMA
  deviation is only 2.6p — so dist >= 3×sp (= 5.4p EUR_USD) is a rare tail.
  On M15, median SMA14 deviation is 4.6p, so the same absolute threshold is
  near-median: conditions fire 3-5× more often, without regime clustering.

  A 2-bar M15 exhaustion = 30min one-direction pressure.
  A 3-bar M15 exhaustion = 45min. Much stronger signal than M5.

  TP/spread at M15 strict dist: 3.4-4.3× (vs 1.3× on M5 relaxed, 2.8× strict).

Entry:
  1. SMA(period) on M15 close.
  2. Distance gate: |close - SMA| in [dist_min, dist_max] × sp_gate pips.
  3. Bar-exhaustion: last n_consec M15 bars same body direction, price same
     side as SMA (all-bull + above SMA → SHORT; all-bear + below SMA → LONG).
  4. ATR gate (optional): n-bar move >= atr_mult × ATR14.
  5. TP quality gate: tp_price > 0.5 × spread (adaptive).

Exit:
  TP = dist × tp_frac | SL = dist × sl_frac | 8-bar timeout (2h)

Usage:
    python3 backtest_harvester_m15.py [--pairs USD_JPY EUR_USD] [--fast]
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
PROJECT = Path("/path/to/projects/fx-core")
M5_DIR  = PROJECT / "data/m5_ba"
RES_DIR = PROJECT / "research/experiments/harvester/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "EUR_JPY": {"file_m5": "EUR_JPY_M5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file_m5": "GBP_JPY_M5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file_m5": "USD_JPY_M5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file_m5": "GBP_USD_M5_BA.parquet", "pip": 0.0001},
    "EUR_USD": {"file_m5": "EUR_USD_M5_BA.parquet", "pip": 0.0001},
}

IS_FRAC       = 0.70
N_WF_CHUNKS   = 3
MIN_IS_TRADES = 50
MC_SHUFFLES   = 1000
MAX_HOLD_BARS = 8        # 2h timeout at M15 (8 × 15min)
ATR_PERIOD    = 14
SESSION_START = 7
SESSION_END   = 21

# ── Sweep space ────────────────────────────────────────────────────────────────
SMA_OPT           = [7, 10, 14, 20]
N_CONSEC_OPT      = [2, 3, 4]
DIST_MULT_MIN_OPT = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]   # up to 6× sp_gate
DIST_MULT_MAX_OPT = [5.0, 8.0, 12.0, 16.0]
TP_FRAC_OPT       = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SL_FRAC_OPT       = [1.5, 2.0, 2.5]
ATR_MULT_OPT      = [0.0, 0.5, 1.0]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_m15(pair_cfg: dict) -> pd.DataFrame:
    """Load M5 BA parquet and resample to M15 bars."""
    path = M5_DIR / pair_cfg["file_m5"]
    pip  = pair_cfg["pip"]

    df = duckdb.query(
        f'SELECT timestamp, open, high, low, close, bid_c, ask_c '
        f'FROM "{path}" ORDER BY timestamp'
    ).df()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close", "bid_c", "ask_c"]:
        df[col] = df[col].astype(np.float64)

    # Resample M5 → M15 (closed=left, label=left = start of 15min window)
    m15 = df.resample("15min", closed="left", label="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "bid_c": "last",
        "ask_c": "last",
    }).dropna(subset=["open", "close"])

    m15 = m15.reset_index()
    m15["spread_pips"] = ((m15["ask_c"] - m15["bid_c"]).astype(np.float64) / pip
                          ).clip(0.1, 30.0)
    return m15


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
def run_trade_m15(close, high, low, spread_pips, entry_bar,
                  entry_px, direction, tp_price, sl_price, pip_size, max_hold):
    """TP/SL in price units. Returns (pnl_pips, bars_held, exit_reason)."""
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
def run_segment_m15(close, high, low, open_, spread_pips, sma, atr,
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
        pnl_pips, bars, reason = run_trade_m15(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        total_pips += pnl_pips
        n_trades   += np.int64(1)
        if pnl_pips > np.float64(0.0): n_wins += np.int64(1)
        if reason  == np.int64(0):     n_tp   += np.int64(1)

        next_entry = i + bars + np.int64(1)
        i += 1

    return total_pips, n_trades, n_wins, n_tp


@njit(cache=True, fastmath=True)
def run_segment_pnl_m15(close, high, low, open_, spread_pips, sma, atr,
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
        pnl_pips, bars, reason = run_trade_m15(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        pnl_arr[n_t] = pnl_pips
        n_t          += np.int64(1)
        next_entry   = i + bars + np.int64(1)
        i += 1

    return pnl_arr[:n_t]


# ── Config builder ─────────────────────────────────────────────────────────────
DIST_MIN_ARR  = np.array(DIST_MULT_MIN_OPT, dtype=np.float64)
DIST_MAX_ARR  = np.array(DIST_MULT_MAX_OPT, dtype=np.float64)
TP_FRAC_ARR   = np.array(TP_FRAC_OPT,      dtype=np.float64)
SL_FRAC_ARR   = np.array(SL_FRAC_OPT,      dtype=np.float64)
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
def sweep_parallel_m15(close, high, low, open_, spread_pips, sma_v, sma_period,
                       atr, in_session, pip_size,
                       configs, is_end, wf_starts, wf_ends, sp_gate):
    n_cfg  = len(configs)
    n_wf   = len(wf_starts)
    n_stat = 4
    n_col  = (n_wf + 2) * n_stat
    out    = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all  = len(close)

    for c in prange(n_cfg):
        nc  = np.int64(configs[c, 1])
        dmi = configs[c, 2]
        dxi = configs[c, 3]
        tfi = configs[c, 4]
        sfi = configs[c, 5]
        ai  = configs[c, 6]

        dmin = DIST_MIN_ARR[dmi]
        dmax = DIST_MAX_ARR[dxi]
        tf   = TP_FRAC_ARR[tfi]
        sf   = SL_FRAC_ARR[sfi]
        am   = ATR_MULT_ARR[ai]

        p0,p1,p2,p3 = run_segment_m15(
            close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
            np.int64(sma_period), nc, dmin, dmax, tf, sf, am, 0, is_end, sp_gate)
        out[c,0]=p0; out[c,1]=p1; out[c,2]=p2; out[c,3]=p3

        for k in range(n_wf):
            p0,p1,p2,p3 = run_segment_m15(
                close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
                np.int64(sma_period), nc, dmin, dmax, tf, sf, am,
                wf_starts[k], wf_ends[k], sp_gate)
            b = n_stat + k * n_stat
            out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3

        p0,p1,p2,p3 = run_segment_m15(
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


# ── Sweep helper ───────────────────────────────────────────────────────────────
def _run_sweep(close, high, low, open_, sp, smas_list, atr, in_sess, pip,
               configs_arr, is_end, wf_starts, wf_ends, sp_gate):
    n_cfg  = len(configs_arr)
    n_wf   = len(wf_starts)
    out    = np.zeros((n_cfg, (n_wf + 2) * 4), dtype=np.float64)

    for si, sma_arr in enumerate(smas_list):
        mask = configs_arr[:, 0] == si
        if not mask.any():
            continue
        idx = np.where(mask)[0].astype(np.int64)
        sub = configs_arr[mask]
        sub_out = sweep_parallel_m15(
            close, high, low, open_, sp, sma_arr, np.int64(SMA_OPT[si]),
            atr, in_sess, pip,
            sub, is_end, wf_starts, wf_ends, sp_gate)
        out[idx] = sub_out

    return out


# ── Process one pair ───────────────────────────────────────────────────────────
def process(pair_name, pair_cfg, configs_arr, configs_meta, compiled_already):
    m5_path = M5_DIR / pair_cfg["file_m5"]
    if not m5_path.exists():
        print(f"  {pair_name}: M5 parquet missing — skip")
        return []

    print(f"\n{'─'*60}")
    print(f"  {pair_name}  |  {pair_cfg['file_m5']} → M15")
    pip = pair_cfg["pip"]

    df    = load_m15(pair_cfg)
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
    dev_p50  = float(np.median(np.abs(close[:is_end] - smas_list[2][:is_end])) / pip)

    chunk     = is_end // N_WF_CHUNKS
    wf_starts = np.array([k * chunk     for k in range(N_WF_CHUNKS)], dtype=np.int64)
    wf_ends   = np.array([(k+1) * chunk for k in range(N_WF_CHUNKS)], dtype=np.int64)

    print(f"  {n:,} bars ({bpd:.0f}/day)  IS {is_end:,} ({is_days}d)  "
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  "
          f"ATR14={atr_med:.2f}p  SMA14_dev_p50={dev_p50:.1f}p")

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
            "sma":        sma_p,  "n_consec": nc,
            "dist_min_m": dmin_m, "dist_max_m": dmax_m,
            "tp_frac":    tf,     "sl_frac":   sf,
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

    if survivors:
        for r in survivors:
            sma_p, nc = r["sma"], r["n_consec"]
            dmin_m, dmax_m = r["dist_min_m"], r["dist_max_m"]
            tf, sf, am = r["tp_frac"], r["sl_frac"], r["atr_mult"]
            pnl_arr = run_segment_pnl_m15(
                close, high, low, open_, sp,
                compute_sma(close, sma_p), atr, in_sess, np.float64(pip),
                np.int64(sma_p), np.int64(nc), dmin_m, dmax_m,
                tf, sf, am, 0, is_end, sp_gate)
            mc_p = run_mc(pnl_arr, is_days)
            r["mc_p"] = round(mc_p, 4) if not np.isnan(mc_p) else np.nan
        mc_pass = [r for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05]
        print(f"  mc_p<0.05: {len(mc_pass)}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    n_cfg = len(configs_arr)

    print(f"Config space  : {n_cfg} per pair")
    print(f"Pairs         : {args.pairs}")
    print(f"Entry gate    : |close-SMA| ∈ [dm_min, dm_max]×sp_gate + n_consec M15 bars + ATR gate")
    print(f"Exit          : TP=dist×tp_frac | SL=dist×sl_frac | {MAX_HOLD_BARS}-bar timeout (2h)")
    print(f"TP fracs      : {TP_FRAC_OPT}   SL fracs: {SL_FRAC_OPT}")
    print(f"Session       : {SESSION_START:02d}:00–{SESSION_END:02d}:00 UTC  (M15 bars, resampled from M5 BA)")

    if args.fast:
        configs_arr  = configs_arr[:8]
        configs_meta = configs_meta[:8]

    compiled = [False]
    all_rows = []

    for pair_name in args.pairs:
        if pair_name not in PAIRS:
            continue
        rows = process(pair_name, PAIRS[pair_name], configs_arr, configs_meta, compiled)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    out_path = RES_DIR / "harvester_m15_results.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\nResults → {out_path}  ({len(df_out)} rows)")
    print()
    print("=" * 80)

    wf_mc  = [r for r in all_rows if r.get("wf_pass") and not np.isnan(r.get("mc_p", np.nan)) and r["mc_p"] < 0.05]
    wf_only = [r for r in all_rows if r.get("wf_pass")]

    if wf_mc:
        print("WF + MC survivors:")
        df_s = pd.DataFrame(wf_mc).sort_values("is_pd", ascending=False)
        print(df_s[["pair","sma","n_consec","dist_min_m","dist_max_m","tp_frac","sl_frac",
                     "atr_mult","is_pd","is_wr","is_n","wf1","wf2","wf3","oos_pd","mc_p"]
                   ].to_string(index=False))
    else:
        print("No WF + MC survivors.")
        print()
        print("Top IS p/d (diagnostic):")
        df_diag = pd.DataFrame(all_rows).nlargest(20, "is_pd")
        print(df_diag[["pair","sma","n_consec","dist_min_m","dist_max_m","tp_frac","sl_frac",
                        "atr_mult","is_pd","is_wr","is_n","wf1","wf2","wf3"]
                      ].to_string(index=False))

    if wf_only and not wf_mc:
        print()
        print("WF-only survivors:")
        df_wf = pd.DataFrame(wf_only).sort_values("is_pd", ascending=False)
        print(df_wf[["pair","sma","n_consec","dist_min_m","dist_max_m","tp_frac","sl_frac",
                      "atr_mult","is_pd","is_wr","is_n","tp_pct","wf1","wf2","wf3"]
                    ].to_string(index=False))

    print()
    print("─" * 80)
    print("WF survivors by pair:")
    for pair in args.pairs:
        if pair not in PAIRS:
            continue
        p_rows = [r for r in all_rows if r["pair"] == pair]
        wf_n  = sum(1 for r in p_rows if r.get("wf_pass"))
        mc_n  = sum(1 for r in p_rows if r.get("wf_pass") and not np.isnan(r.get("mc_p", np.nan)) and r["mc_p"] < 0.05)
        best  = max((r["is_pd"] for r in p_rows), default=0.0)
        print(f"  {pair}: WF={wf_n}  MC={mc_n}  best IS p/d={best:.2f}")


if __name__ == "__main__":
    main()
