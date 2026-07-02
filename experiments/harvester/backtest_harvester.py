#!/usr/bin/env python3
"""
Harvester — SMA Mean-Reversion + Bar-Exhaustion Backtest
==========================================================
Strategy logic:
  1. SMA(period) on M5 close — rolling causal.
  2. Distance gate: |close - SMA| must be in [dist_mult_min, dist_mult_max] × sp_gate.
     Extended enough to expect reversion; not so far it's a breakout.
  3. Bar-exhaustion filter (two conditions must both hold):
       a. Direction: last n_consec bars all BULL (close >= open) AND close > SMA → SHORT
                     last n_consec bars all BEAR (close < open) AND close < SMA → LONG
       b. ATR gate (optional): cumulative move of the last n_consec bars
          = |close[i] - close[i - n_consec]| ≥ atr_mult × ATR(period=14).
          Ensures the n-bar run is a "real push," not noise.
  4. TP: fixed pips from fill. SL: hard stop. Timeout: MAX_HOLD_BARS bars.

"Harvester" = systematically harvesting small mean-reversion snaps after candle exhaustion.

Lookahead audit:
  - SMA: running sum, window causal ✅
  - ATR14: Wilder causal ✅
  - Bar bodies: close[j] >= open[j] for j = i-n_consec..i-1 ✅
  - Entry fill at bar-i close ± half_spread ✅ (market order R1)
  - Exit loop starts at bar i+1 ✅

Usage:
    python3 backtest_harvester.py [--pairs GBP_JPY USD_JPY EUR_USD] [--fast]
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
DATA_DIR = PROJECT / "data/m5_ba"
RES_DIR  = PROJECT / "research/experiments/harvester/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "EUR_JPY": {"file": "EUR_JPY_M5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file": "GBP_JPY_M5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file": "USD_JPY_M5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file": "GBP_USD_M5_BA.parquet", "pip": 0.0001},
    "EUR_USD": {"file": "EUR_USD_M5_BA.parquet", "pip": 0.0001},
}

IS_FRAC       = 0.70
N_WF_CHUNKS   = 3
MIN_IS_TRADES = 50
MC_SHUFFLES   = 1000
MAX_HOLD_BARS = 24       # 2 hours at M5
ATR_PERIOD    = 14       # short ATR for recent-bars magnitude check
SESSION_START = 7
SESSION_END   = 21

# ── Sweep space ────────────────────────────────────────────────────────────────
SMA_OPT          = [5, 7, 10, 14]       # SMA lookback
N_CONSEC_OPT     = [2, 3, 4]           # consecutive same-direction bars required
DIST_MIN_OPT     = [1.0, 1.5, 2.0]    # lower bound = mult × sp_gate (pips)
DIST_MAX_OPT     = [2.5, 3.0, 4.5]    # upper bound = mult × sp_gate (pips)
TP_PIPS_OPT      = [1, 2, 3, 5]       # TP in pips
SL_PIPS_OPT      = [5, 10, 20]        # hard SL in pips
ATR_MULT_OPT     = [0.0, 0.5, 1.0]   # 0=off; >0: n-bar move ≥ mult × ATR14


# ── Data loading ───────────────────────────────────────────────────────────────
def load_m5(pair_cfg: dict) -> pd.DataFrame:
    path = DATA_DIR / pair_cfg["file"]
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


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_sma(close, period):
    n   = len(close)
    sma = np.empty(n, dtype=np.float64)
    s   = np.float64(0.0)
    for i in range(n):
        s += close[i]
        if i >= period:
            s -= close[i - period]
        sma[i] = s / min(i + 1, period)
    return sma


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
              direction, tp_pips, sl_pips, pip_size, max_hold):
    """
    Returns (pnl_pips, bars_held, exit_reason).
      0=TP  1=SL  2=timeout
    """
    n      = len(close)
    tp_lev = entry_px + np.float64(direction) * tp_pips * pip_size
    sl_lev = entry_px - np.float64(direction) * sl_pips * pip_size

    end = min(entry_bar + max_hold + 1, n)
    for j in range(entry_bar + 1, end):
        sp_j = spread_pips[j]
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
            return tp_pips, np.int64(j - entry_bar), np.int64(0)
        if hit_sl:
            return -sl_pips, np.int64(j - entry_bar), np.int64(1)

    last = min(entry_bar + max_hold, n - 1)
    pnl  = (close[last] - entry_px) * np.float64(direction) / pip_size \
           - np.float64(0.5) * spread_pips[last]
    return pnl, np.int64(last - entry_bar), np.int64(2)


@njit(cache=True, fastmath=True)
def run_segment(close, high, low, open_, spread_pips, sma, atr,
                in_session, pip_size,
                sma_period, n_consec, dist_mult_min, dist_mult_max,
                tp_pips, sl_pips, atr_mult,
                seg_start, seg_end, sp_gate):
    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, warmup)
    next_entry = start

    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)
    n_tp       = np.int64(0)

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate
    tp_f    = np.float64(tp_pips)
    sl_f    = np.float64(sl_pips)

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

        # Distance gate: |close - SMA| in pips
        dist = abs(close[i] - sma[i]) / pip_size
        if dist < dist_lo or dist > dist_hi:
            i += 1
            continue

        # Direction: close vs SMA
        above_sma = close[i] > sma[i]

        # Bar-exhaustion: check last n_consec bars have same body direction
        all_bull = True
        all_bear = True
        for k in range(np.int64(1), n_consec + np.int64(1)):
            j = i - k
            if close[j] < open_[j]:   # bear body
                all_bull = False
            if close[j] >= open_[j]:  # bull body
                all_bear = False
            if not all_bull and not all_bear:
                break

        direction = np.int64(0)
        if all_bull and above_sma:
            direction = np.int64(-1)   # n bull bars + price above SMA → SHORT
        elif all_bear and not above_sma:
            direction = np.int64(1)    # n bear bars + price below SMA → LONG

        if direction == np.int64(0):
            i += 1
            continue

        # ATR gate: n-bar move ≥ atr_mult × ATR14 (in raw price units)
        if atr_mult > np.float64(0.0):
            n_bar_move = abs(close[i] - close[i - n_consec])
            if n_bar_move < atr_mult * atr[i]:
                i += 1
                continue

        # Spread gate vs TP
        if sp > tp_f * 0.5:
            i += 1
            continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl, bars, reason = run_trade(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_f, sl_f, pip_size, np.int64(MAX_HOLD_BARS))

        total_pips += pnl
        n_trades   += np.int64(1)
        if pnl > np.float64(0.0): n_wins += np.int64(1)
        if reason == np.int64(0):  n_tp   += np.int64(1)

        next_entry = i + bars + np.int64(1)
        i += 1

    return total_pips, n_trades, n_wins, n_tp


@njit(cache=True, fastmath=True)
def run_segment_pnl(close, high, low, open_, spread_pips, sma, atr,
                    in_session, pip_size,
                    sma_period, n_consec, dist_mult_min, dist_mult_max,
                    tp_pips, sl_pips, atr_mult,
                    seg_start, seg_end, sp_gate):
    MAX_T   = 10_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, warmup)
    next_entry = start

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate
    tp_f    = np.float64(tp_pips)
    sl_f    = np.float64(sl_pips)

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

        if sp > tp_f * 0.5:
            i += 1
            continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl, bars, _ = run_trade(
            close, high, low, spread_pips,
            i, entry_px, direction,
            tp_f, sl_f, pip_size, np.int64(MAX_HOLD_BARS))
        pnl_arr[n_t] = pnl
        n_t          += np.int64(1)
        next_entry   = i + bars + np.int64(1)
        i += 1

    return pnl_arr[:n_t]


# ── Config builder ─────────────────────────────────────────────────────────────
# cols: sma_idx  n_consec  dmin_idx  dmax_idx  tp  sl  atr_idx
SMA_ARR      = np.array(SMA_OPT,      dtype=np.int32)
DIST_MIN_ARR = np.array(DIST_MIN_OPT, dtype=np.float64)
DIST_MAX_ARR = np.array(DIST_MAX_OPT, dtype=np.float64)
ATR_MULT_ARR = np.array(ATR_MULT_OPT, dtype=np.float64)


def build_configs():
    rows = []
    for si, nc, dmi, dxi, tp, sl, ai in itertools.product(
            range(len(SMA_OPT)), N_CONSEC_OPT,
            range(len(DIST_MIN_OPT)), range(len(DIST_MAX_OPT)),
            TP_PIPS_OPT, SL_PIPS_OPT,
            range(len(ATR_MULT_OPT))):
        if sl <= tp:
            continue
        if DIST_MAX_OPT[dxi] <= DIST_MIN_OPT[dmi]:
            continue
        rows.append((si, nc, dmi, dxi, tp, sl, ai))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(SMA_OPT[r[0]], r[1], DIST_MIN_OPT[r[2]], DIST_MAX_OPT[r[3]],
             r[4], r[5], ATR_MULT_OPT[r[6]]) for r in rows]
    return arr, meta


@njit(parallel=True, cache=True)
def sweep_parallel(close, high, low, open_, spread_pips, sma_v, sma_period, atr,
                   in_session, pip_size,
                   configs, is_end, wf_starts, wf_ends, sp_gate):
    """
    All configs in `configs` share the same SMA (sma_v / sma_period).
    _run_sweep_one_pair groups by SMA index before calling here.
    """
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
        tp   = np.float64(configs[c, 4])
        sl   = np.float64(configs[c, 5])
        ai   = configs[c, 6]

        sma_p = np.int64(sma_period)
        dmin  = DIST_MIN_ARR[dmi]
        dmax  = DIST_MAX_ARR[dxi]
        am    = ATR_MULT_ARR[ai]

        p0,p1,p2,p3 = run_segment(
            close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
            sma_p, nc, dmin, dmax, tp, sl, am, 0, is_end, sp_gate)
        out[c,0]=p0; out[c,1]=p1; out[c,2]=p2; out[c,3]=p3

        for k in range(n_wf):
            p0,p1,p2,p3 = run_segment(
                close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
                sma_p, nc, dmin, dmax, tp, sl, am, wf_starts[k], wf_ends[k], sp_gate)
            b = n_stat + k * n_stat
            out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3

        p0,p1,p2,p3 = run_segment(
            close, high, low, open_, spread_pips, sma_v, atr, in_session, pip_size,
            sma_p, nc, dmin, dmax, tp, sl, am, is_end, n_all, sp_gate)
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


# ── Pre-compute all SMAs ───────────────────────────────────────────────────────
def precompute_smas(close: np.ndarray) -> list:
    return [np.ascontiguousarray(compute_sma(close, p)) for p in SMA_OPT]


# ── Process one pair ───────────────────────────────────────────────────────────
def process(pair_name, pair_cfg, configs_arr, configs_meta, compiled_already):
    path = DATA_DIR / pair_cfg["file"]
    if not path.exists():
        print(f"  {pair_name}: parquet missing — skip")
        return []

    print(f"\n{'─'*60}")
    print(f"  {pair_name}  |  {path.name}")
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

    smas_list = precompute_smas(close)
    # pack into object array for Numba (passed by index via configs)
    smas_nb = smas_list   # list of contiguous arrays

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
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  ATR14med={atr_med:.2f}p")

    # Compile warmup — pass each SMA array separately via tuple-unpacking trick
    # (Numba can't take a list of arrays as a single arg, use the first SMA for warmup)
    if not compiled_already[0]:
        print("  Compiling Numba ...", end="", flush=True)
        t0 = time.time()
        _ws = wf_starts.copy(); _we = np.minimum(wf_ends, 2000)
        _ = _run_sweep_one_pair(
            close[:3000], high[:3000], low[:3000], open_[:3000], sp[:3000],
            smas_nb, atr[:3000], in_sess[:3000], np.float64(pip),
            configs_arr[:4], 2000, _ws, _we, sp_gate)
        compiled_already[0] = True
        print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0  = time.time()
    raw = _run_sweep_one_pair(
        close, high, low, open_, sp, smas_nb, atr, in_sess, np.float64(pip),
        configs_arr, is_end, wf_starts, wf_ends, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    n_stat = 4
    n_wf   = N_WF_CHUNKS
    rows   = []

    for c, (sma_p, nc, dmin, dmax, tp, sl, am) in enumerate(configs_meta):
        def _g(seg, s):
            if   seg == "is":  return raw[c, s]
            elif seg == "oos": return raw[c, n_stat + n_wf*n_stat + s]
            else:              return raw[c, n_stat + seg*n_stat + s]

        is_pips  = _g("is",0); is_n = int(_g("is",1)); is_w = int(_g("is",2))
        is_tp    = int(_g("is",3))
        oos_pips = _g("oos",0); oos_n = int(_g("oos",1)); oos_w = int(_g("oos",2))

        wf_pds = []
        for k in range(n_wf):
            pk = _g(k,0); nk = int(_g(k,1))
            wf_pds.append(pk / (is_days / n_wf) if nk > 0 else 0.0)

        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0
        wf_pass = (is_n >= MIN_IS_TRADES) and all(p > 0 for p in wf_pds)

        rows.append({
            "pair":      pair_name,
            "sma":       sma_p,  "n_consec": nc,
            "dist_min":  dmin,   "dist_max":  dmax,
            "tp":        tp,     "sl":        sl,
            "atr_mult":  am,
            "is_pd":     round(is_pd,  2),
            "is_wr":     round(is_w / is_n if is_n > 0 else 0.0, 3),
            "is_n":      is_n,
            "tp_pct":    round(is_tp / is_n if is_n > 0 else 0.0, 3),
            "wf1":       round(wf_pds[0], 2),
            "wf2":       round(wf_pds[1], 2),
            "wf3":       round(wf_pds[2], 2),
            "wf_pass":   wf_pass,
            "oos_pd":    round(oos_pd, 2),
            "oos_wr":    round(oos_w / oos_n if oos_n > 0 else 0.0, 3),
            "oos_n":     oos_n,
            "mc_p":      np.nan,
        })

    survivors = [r for r in rows if r["wf_pass"]]
    print(f"  WF survivors: {len(survivors)}", end="  (MC ...)\n" if survivors else "\n",
          flush=True)
    for r in survivors:
        sma_arr = smas_list[SMA_OPT.index(r["sma"])]
        pnl_arr = run_segment_pnl(
            close, high, low, open_, sp, sma_arr, atr, in_sess, np.float64(pip),
            int(r["sma"]), int(r["n_consec"]),
            float(r["dist_min"]), float(r["dist_max"]),
            float(r["tp"]), float(r["sl"]), float(r["atr_mult"]),
            0, is_end, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)
    if survivors:
        mc_ok = sum(1 for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
        print(f"  mc_p<0.05: {mc_ok}")

    return rows


def _run_sweep_one_pair(close, high, low, open_, sp, smas_list, atr, in_sess, pip,
                         configs_arr, is_end, wf_starts, wf_ends, sp_gate):
    """Run sweep for one pair, iterating over each SMA index group to feed Numba."""
    n_cfg  = len(configs_arr)
    n_wf   = len(wf_starts)
    n_stat = 4
    out    = np.zeros((n_cfg, (n_wf + 2) * n_stat), dtype=np.float64)

    for si, sma_arr in enumerate(smas_list):
        # Select configs that use this SMA index
        mask = configs_arr[:, 0] == si
        if not mask.any():
            continue
        idx   = np.where(mask)[0].astype(np.int64)
        sub   = configs_arr[mask]
        sub_out = sweep_parallel(
            close, high, low, open_, sp, sma_arr, np.int64(SMA_OPT[si]),
            atr, in_sess, pip,
            sub, is_end, wf_starts, wf_ends, sp_gate)
        out[idx] = sub_out

    return out


# Rewrite sweep_parallel to take a single SMA array (not a list):
# Already done above — sweep_parallel takes `sma_v` directly.


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    print(f"Config space  : {len(configs_arr)} per pair")
    print(f"Pairs         : {args.pairs}")
    print(f"Entry gate    : |close - SMA| in [dist_min, dist_max] × sp_gate "
          f"+ last n_consec same-direction bars + ATR n-bar move gate")
    print(f"Exit          : TP (fixed pips) | hard SL | {MAX_HOLD_BARS}-bar timeout (2h)")
    print(f"SMA options   : {SMA_OPT}")
    print(f"Session       : {SESSION_START:02d}:00–{SESSION_END:02d}:00 UTC")

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

    df = pd.DataFrame(all_rows)
    out_path = RES_DIR / "harvester_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df)} rows)")

    mc_pass = df[df["wf_pass"] & (df["mc_p"] < 0.05)].copy() if len(df) > 0 else pd.DataFrame()
    print(f"\n{'='*80}")
    if len(mc_pass) > 0:
        print(f"PASSED WF + MC  ({len(mc_pass)} configs):")
        cols = ["pair","sma","n_consec","dist_min","dist_max","tp","sl","atr_mult",
                "is_pd","is_wr","is_n","tp_pct","oos_pd","oos_wr","oos_n","mc_p"]
        print(mc_pass.sort_values("oos_pd", ascending=False)[cols].head(40).to_string(index=False))
    else:
        print("No WF + MC survivors.")
        if len(df) > 0:
            wf_only = df[df["wf_pass"]].sort_values("is_pd", ascending=False).head(20)
            if len(wf_only):
                print(f"\nWF-only survivors:")
                print(wf_only[["pair","sma","n_consec","dist_min","dist_max",
                               "tp","sl","atr_mult","is_pd","is_wr","is_n",
                               "tp_pct","wf1","wf2","wf3"]].to_string(index=False))
            else:
                top = df[df["is_n"] >= 10].sort_values("is_pd", ascending=False).head(20)
                if len(top):
                    print(f"\nTop IS p/d (diagnostic):")
                    print(top[["pair","sma","n_consec","dist_min","dist_max",
                               "tp","sl","atr_mult","is_pd","is_wr","is_n",
                               "tp_pct","wf1","wf2","wf3"]].to_string(index=False))

    # Summary by pair
    if len(df) > 0:
        print(f"\n{'─'*80}")
        print("WF survivors by pair:")
        for p in args.pairs:
            sub = df[df["pair"] == p]
            wf = sub["wf_pass"].sum()
            mc = (sub["wf_pass"] & (sub["mc_p"] < 0.05)).sum() if wf > 0 else 0
            best = sub[sub["is_n"] >= 10]["is_pd"].max() if len(sub[sub["is_n"] >= 10]) > 0 else 0
            print(f"  {p}: WF={wf}  MC={mc}  best IS p/d={best:.2f}")


if __name__ == "__main__":
    main()
