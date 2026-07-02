#!/usr/bin/env python3
"""
Wave-Exit Momentum Backtest
============================
Hypothesis: multi-scale S5 momentum confluence identifies waves in progress.
            Entering at M5 close + exiting when the wave breaks (fast group reverses)
            gives tight losses on bad entries and full rides on good ones.

Entry:  M5 bar close (S5 index i where i % M5_BARS == M5_BARS-1).
        Signal: S5 fast [1,2,3] AND slow [12,24,36] all agree in direction.
        Optional: ATR magnitude gate + acceleration requirement.

Exit (in priority order):
  1. Wave-break: N_REVERSAL fast lags reverse direction at any S5 bar post-entry.
  2. TP ceiling : price moves tp_pips in direction.
  3. Hard SL    : price moves sl_pips against (0 = disabled, wave-break only).
  4. Timeout    : max_bars S5 bars held.

No lookahead: M5 bars derived from S5 by sampling at i%12==11. All features
computed from close[0..i] only. Session filter: entries 07-17 UTC only.

Usage:
    python3 backtest_wave_exit.py [--pairs USD_JPY EUR_JPY] [--fast]
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
RES_DIR  = PROJECT / "research/experiments/wave_exit/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Pairs ──────────────────────────────────────────────────────────────────────
PAIRS = {
    "EUR_JPY": {"file": "EUR_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file": "GBP_JPY_S5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file": "USD_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file": "GBP_USD_S5_BA.parquet", "pip": 0.0001},
    "EUR_USD": {"file": "EUR_USD_S5_BA.parquet", "pip": 0.0001},
}

# ── Constants ──────────────────────────────────────────────────────────────────
IS_FRAC         = 0.70
N_WF_CHUNKS     = 3
MIN_IS_TRADES   = 50     # lower bar — wave-break exits are selective
MC_SHUFFLES     = 1000
M5_BARS         = 12     # S5 bars per M5
ATR_PERIOD      = 500    # long-period Wilder ATR for magnitude gate
MAX_HOLD_BARS   = 120    # S5 bars = 10 min max hold
SESSION_START   = 7      # UTC hour inclusive
SESSION_END     = 17     # UTC hour exclusive

# ── Fast/slow lags (fixed — derived from M5 alignment) ─────────────────────────
# Fast: 5s/10s/15s context — detects sub-minute wave
# Slow: M5/2×M5/3×M5 — confirms multi-minute trend direction
FAST_LAGS  = np.array([1,  2,  3],  dtype=np.int64)
SLOW_LAGS  = np.array([12, 24, 36], dtype=np.int64)
N_LAGS     = 3

# ── Sweep space ────────────────────────────────────────────────────────────────
TP_PIPS_OPT    = [5, 10, 20, 30]       # TP ceiling
SL_PIPS_OPT    = [0, 5, 10, 20]        # hard SL (0 = wave-break only)
N_REVERSAL_OPT = [1, 2, 3]             # fast lags that must reverse to exit
ATR_MULT_OPT   = [0.0, 0.5, 1.0, 2.0] # entry magnitude threshold
ACCEL_OPT      = [False, True]         # require acceleration at all fast lags


# ── Data loading ───────────────────────────────────────────────────────────────
def load_s5(pair_cfg: dict) -> pd.DataFrame:
    path = DATA_DIR / pair_cfg["file"]
    pip  = pair_cfg["pip"]
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
    df["timestamp"]    = pd.to_datetime(df["timestamp"], utc=True)
    df["spread_pips"]  = ((df["ask_c"] - df["bid_c"]).astype(np.float64) / pip).clip(0.1, 20.0)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(np.float64)
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_atr(high, low, close, period):
    n   = len(close)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = high[0] - low[0]
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * (period - 1) + tr) / period if i >= period \
                 else (atr[i-1] * (i - 1) + tr) / i
    return atr


@njit(cache=True, fastmath=True)
def _check_confluence(close, atr, i, direction, atr_mult, accel_req):
    """
    Both fast [1,2,3] and slow [12,24,36] groups must all agree in direction.
    Optional: each fast lag must be accelerating (mom[lag] > mom_prev[lag]).
    Optional: each lag's |momentum| >= atr_mult × ATR500.
    """
    threshold = atr_mult * atr[i]
    # fast group
    for k in range(N_LAGS):
        lag = FAST_LAGS[k]
        mom = close[i] - close[i - lag]
        if direction == 1:
            if mom <= 0.0:                              return False
            if atr_mult > 0.0 and mom < threshold:     return False
            if accel_req:
                if mom <= (close[i - lag] - close[i - 2*lag]): return False
        else:
            if mom >= 0.0:                              return False
            if atr_mult > 0.0 and -mom < threshold:    return False
            if accel_req:
                if mom >= (close[i - lag] - close[i - 2*lag]): return False
    # slow group
    for k in range(N_LAGS):
        lag = SLOW_LAGS[k]
        mom = close[i] - close[i - lag]
        if direction == 1:
            if mom <= 0.0:                              return False
        else:
            if mom >= 0.0:                              return False
    return True


@njit(cache=True, fastmath=True)
def _count_fast_reversals(close, i, entry_direction):
    """Count how many fast lags have reversed against the entry direction."""
    n_rev = np.int64(0)
    for k in range(N_LAGS):
        lag = FAST_LAGS[k]
        if i < lag:
            continue
        mom = close[i] - close[i - lag]
        if entry_direction == 1 and mom < 0.0:
            n_rev += 1
        elif entry_direction == -1 and mom > 0.0:
            n_rev += 1
    return n_rev


@njit(cache=True, fastmath=True)
def run_wave_trade(close, high, low, spread_pips, entry_bar, entry_px,
                   direction, tp_pips, sl_pips, n_reversal_thresh,
                   pip_size, max_hold):
    """
    Run one trade from entry_bar, monitoring every S5 bar.
    Exit on: wave-break (n_reversal_thresh fast lags reverse) OR TP OR hard SL OR timeout.
    Returns (pnl_pips, bars_held, exit_reason).
      exit_reason: 0=wave-break  1=TP  2=hard-SL  3=timeout
    """
    n      = len(close)
    tp_lev = entry_px + np.float64(direction) * tp_pips * pip_size
    sl_lev = entry_px - np.float64(direction) * sl_pips * pip_size if sl_pips > 0.0 \
             else np.float64(-1e18) * np.float64(direction)

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)

    end = min(entry_bar + max_hold + 1, n)
    for j in range(entry_bar + 1, end):
        sp_j = spread_pips[j]
        bull = close[j] >= close[j - 1]

        # Check TP / hard-SL with SOP R2 sequencing
        hit_tp = False
        hit_sl = False
        if direction == 1:
            if bull:
                hit_tp = high[j] >= tp_lev
                hit_sl = sl_pips > 0.0 and low[j] <= sl_lev
            else:
                hit_sl = sl_pips > 0.0 and low[j] <= sl_lev
                hit_tp = high[j] >= tp_lev
        else:
            if not bull:
                hit_tp = low[j] <= tp_lev
                hit_sl = sl_pips > 0.0 and high[j] >= sl_lev
            else:
                hit_sl = sl_pips > 0.0 and high[j] >= sl_lev
                hit_tp = low[j] <= tp_lev

        if hit_tp:
            return tp_f - np.float64(0.0), np.int64(j - entry_bar), np.int64(1)
        if hit_sl:
            return -sl_f, np.int64(j - entry_bar), np.int64(2)

        # Wave-break exit: check fast group reversal count
        n_rev = _count_fast_reversals(close, j, direction)
        if n_rev >= n_reversal_thresh:
            pnl = (close[j] - entry_px) * np.float64(direction) / pip_size \
                  - np.float64(0.5) * sp_j   # pay exit half-spread
            return pnl, np.int64(j - entry_bar), np.int64(0)

    # Timeout
    last = min(entry_bar + max_hold, n - 1)
    sp_l = spread_pips[last]
    pnl  = (close[last] - entry_px) * np.float64(direction) / pip_size \
           - np.float64(0.5) * sp_l
    return pnl, np.int64(last - entry_bar), np.int64(3)


@njit(cache=True, fastmath=True)
def run_segment(close, high, low, spread_pips, atr, in_session, pip_size,
                tp_pips, sl_pips, n_reversal_thresh, atr_mult, accel_req,
                seg_start, seg_end, sp_gate):
    """
    Iterate segment bar by bar. At each M5 close in-session, check confluence.
    If signal fires, run one wave trade. No re-entry until trade closes.
    Returns (total_pips, n_trades, n_wins, n_wave_exits, n_tp_exits, n_sl_exits, n_timeouts).
    """
    warmup = SLOW_LAGS[N_LAGS - 1] * 2 + 2   # need 72 S5 bars of history
    start  = max(seg_start, warmup)

    total_pips  = np.float64(0.0)
    n_trades    = np.int64(0)
    n_wins      = np.int64(0)
    n_wave      = np.int64(0)
    n_tp        = np.int64(0)
    n_sl        = np.int64(0)
    n_to        = np.int64(0)

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)
    nr   = np.int64(n_reversal_thresh)

    next_entry = start
    i = start

    while i < seg_end - 1:
        if i < next_entry:
            i += 1
            continue

        # Entry only at M5 bar close
        if i % M5_BARS != M5_BARS - 1:
            i += 1
            continue

        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > sp_gate or sp > tp_f * 0.4:   # spread must be < 40% of TP
            i += 1
            continue

        direction = np.int64(0)
        for d in (np.int64(1), np.int64(-1)):
            if _check_confluence(close, atr, i, d, atr_mult, accel_req):
                direction = d
                break

        if direction != 0:
            entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
            pnl, bars, reason = run_wave_trade(
                close, high, low, spread_pips,
                i, entry_px, direction,
                tp_f, sl_f, nr, pip_size, MAX_HOLD_BARS)

            total_pips += pnl
            n_trades   += np.int64(1)
            if pnl > 0.0:
                n_wins += np.int64(1)
            if   reason == 0: n_wave += np.int64(1)
            elif reason == 1: n_tp   += np.int64(1)
            elif reason == 2: n_sl   += np.int64(1)
            else:             n_to   += np.int64(1)

            next_entry = i + bars + np.int64(1)

        i += 1

    return total_pips, n_trades, n_wins, n_wave, n_tp, n_sl, n_to


@njit(cache=True, fastmath=True)
def run_segment_pnl(close, high, low, spread_pips, atr, in_session, pip_size,
                    tp_pips, sl_pips, n_reversal_thresh, atr_mult, accel_req,
                    seg_start, seg_end, sp_gate):
    """Same as run_segment but returns per-trade pnl array for MC."""
    MAX_T   = 20_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup = SLOW_LAGS[N_LAGS - 1] * 2 + 2
    start  = max(seg_start, warmup)

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)
    nr   = np.int64(n_reversal_thresh)

    next_entry = start
    i = start

    while i < seg_end - 1 and n_t < MAX_T:
        if i < next_entry:
            i += 1
            continue
        if i % M5_BARS != M5_BARS - 1:
            i += 1
            continue
        if not in_session[i]:
            i += 1
            continue

        sp = spread_pips[i]
        if sp > sp_gate or sp > tp_f * 0.4:
            i += 1
            continue

        direction = np.int64(0)
        for d in (np.int64(1), np.int64(-1)):
            if _check_confluence(close, atr, i, d, atr_mult, accel_req):
                direction = d
                break

        if direction != 0:
            entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
            pnl, bars, _ = run_wave_trade(
                close, high, low, spread_pips,
                i, entry_px, direction,
                tp_f, sl_f, nr, pip_size, MAX_HOLD_BARS)
            pnl_arr[n_t] = pnl
            n_t          += np.int64(1)
            next_entry   = i + bars + np.int64(1)

        i += 1

    return pnl_arr[:n_t]


# ── Config builder ─────────────────────────────────────────────────────────────
def build_configs():
    rows = []
    for tp, sl, nr, am, ac in itertools.product(
            TP_PIPS_OPT, SL_PIPS_OPT, N_REVERSAL_OPT, ATR_MULT_OPT, ACCEL_OPT):
        if sl > 0 and sl >= tp:   # hard SL must be < TP
            continue
        rows.append((tp, sl, nr, ATR_MULT_OPT.index(am), int(ac)))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(r[0], r[1], r[2], ATR_MULT_OPT[r[3]], bool(r[4])) for r in rows]
    return arr, meta


ATR_MULTS_ARR = np.array(ATR_MULT_OPT, dtype=np.float64)


@njit(parallel=True, cache=True)
def sweep_parallel(close, high, low, spread_pips, atr, in_session, pip_size,
                   configs, is_end, wf_starts, wf_ends, sp_gate):
    """
    configs[c] = [tp, sl, n_reversal, atr_mult_idx, accel].
    Output columns per config: IS(7) + N_WF×WF(7 each) + OOS(7).
    7 stats: total_pips, n_trades, n_wins, n_wave, n_tp, n_sl, n_to
    """
    n_cfg  = len(configs)
    n_wf   = len(wf_starts)
    n_stat = 7
    n_col  = (n_wf + 2) * n_stat
    out    = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all  = len(close)

    for c in prange(n_cfg):
        tp  = np.float64(configs[c, 0])
        sl  = np.float64(configs[c, 1])
        nr  = np.int64(configs[c, 2])
        am  = ATR_MULTS_ARR[configs[c, 3]]
        ac  = configs[c, 4] > 0

        # IS full
        p0,p1,p2,p3,p4,p5,p6 = run_segment(close, high, low, spread_pips, atr, in_session,
                                            pip_size, tp, sl, nr, am, ac, 0, is_end, sp_gate)
        out[c,0]=p0; out[c,1]=p1; out[c,2]=p2; out[c,3]=p3
        out[c,4]=p4; out[c,5]=p5; out[c,6]=p6

        # WF chunks
        for k in range(n_wf):
            p0,p1,p2,p3,p4,p5,p6 = run_segment(close, high, low, spread_pips, atr,
                                                in_session, pip_size, tp, sl, nr, am, ac,
                                                wf_starts[k], wf_ends[k], sp_gate)
            b = n_stat + k * n_stat
            out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3
            out[c,b+4]=p4; out[c,b+5]=p5; out[c,b+6]=p6

        # OOS
        p0,p1,p2,p3,p4,p5,p6 = run_segment(close, high, low, spread_pips, atr, in_session,
                                            pip_size, tp, sl, nr, am, ac, is_end, n_all, sp_gate)
        b = n_stat + n_wf * n_stat
        out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3
        out[c,b+4]=p4; out[c,b+5]=p5; out[c,b+6]=p6

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


# ── Process one pair ───────────────────────────────────────────────────────────
def process(pair_name, pair_cfg, configs_arr, configs_meta, compiled_already):
    path = DATA_DIR / pair_cfg["file"]
    if not path.exists():
        print(f"  {pair_name}: parquet missing — skip")
        return []

    print(f"\n{'─'*60}")
    print(f"  {pair_name}  |  {path.name}")
    pip = pair_cfg["pip"]

    df    = load_s5(pair_cfg)
    close = np.ascontiguousarray(df["close"].values.astype(np.float64))
    high  = np.ascontiguousarray(df["high"].values.astype(np.float64))
    low   = np.ascontiguousarray(df["low"].values.astype(np.float64))
    sp    = np.ascontiguousarray(df["spread_pips"].values.astype(np.float64))
    atr   = np.ascontiguousarray(compute_atr(high, low, close, ATR_PERIOD))

    hours    = df["timestamp"].dt.hour.values
    sess_msk = ((hours >= SESSION_START) & (hours < SESSION_END)).astype(np.uint8)
    in_sess  = np.ascontiguousarray(sess_msk)

    n       = len(close)
    is_end  = int(n * IS_FRAC)

    # sp_gate from IS session bars only
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
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  ATR500={atr_med:.2f}p  "
          f"[session 07-17 UTC]")

    if not compiled_already[0]:
        print("  Compiling Numba ...", end="", flush=True)
        t0 = time.time()
        _ws = wf_starts.copy(); _we = np.minimum(wf_ends, 2000)
        _ = sweep_parallel(close[:3000], high[:3000], low[:3000], sp[:3000],
                           atr[:3000], in_sess[:3000], np.float64(pip),
                           configs_arr[:4], 2000, _ws, _we, sp_gate)
        compiled_already[0] = True
        print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0  = time.time()
    raw = sweep_parallel(close, high, low, sp, atr, in_sess, np.float64(pip),
                         configs_arr, is_end, wf_starts, wf_ends, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    n_stat = 7
    n_wf   = N_WF_CHUNKS
    rows   = []

    for c, (tp, sl, nr, am, ac) in enumerate(configs_meta):
        def _get(segment, stat):
            if segment == "is":
                return raw[c, stat]
            elif segment == "oos":
                return raw[c, n_stat + n_wf * n_stat + stat]
            else:   # wf chunk k
                return raw[c, n_stat + segment * n_stat + stat]

        is_pips = _get("is", 0); is_n = int(_get("is", 1)); is_w = int(_get("is", 2))
        is_wave = int(_get("is", 3)); is_tp = int(_get("is", 4))
        is_sl   = int(_get("is", 5)); is_to = int(_get("is", 6))

        oos_pips = _get("oos", 0); oos_n = int(_get("oos", 1)); oos_w = int(_get("oos", 2))

        wf_pds = []
        for k in range(n_wf):
            pk = _get(k, 0); nk = int(_get(k, 1))
            wf_pds.append(pk / (is_days / n_wf) if nk > 0 else 0.0)

        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0
        is_wr  = is_w / is_n         if is_n  > 0 else 0.0
        oos_wr = oos_w / oos_n       if oos_n > 0 else 0.0

        wave_pct = is_wave / is_n if is_n > 0 else 0.0
        tp_pct   = is_tp   / is_n if is_n > 0 else 0.0

        wf_pass = (is_n >= MIN_IS_TRADES) and all(p > 0 for p in wf_pds)

        rows.append({
            "pair":       pair_name,
            "tp":         tp,  "sl":    sl,  "n_rev":  nr,
            "atr_mult":   am,  "accel": ac,
            "is_pd":      round(is_pd,  2),
            "is_wr":      round(is_wr,  3),
            "is_n":       is_n,
            "wave_pct":   round(wave_pct, 3),
            "tp_pct":     round(tp_pct,   3),
            "wf1": round(wf_pds[0], 2),
            "wf2": round(wf_pds[1], 2),
            "wf3": round(wf_pds[2], 2),
            "wf_pass":    wf_pass,
            "oos_pd":     round(oos_pd, 2),
            "oos_wr":     round(oos_wr, 3),
            "oos_n":      oos_n,
            "mc_p":       np.nan,
        })

    survivors = [r for r in rows if r["wf_pass"]]
    print(f"  WF survivors: {len(survivors)}", end="  (MC ...)" if survivors else "\n",
          flush=True)
    t0 = time.time()
    for r in survivors:
        pnl_arr = run_segment_pnl(
            close, high, low, sp, atr, in_sess, np.float64(pip),
            float(r["tp"]), float(r["sl"]), int(r["n_rev"]),
            float(r["atr_mult"]), bool(r["accel"]),
            0, is_end, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)
    if survivors:
        mc_ok = sum(1 for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
        print(f" {time.time()-t0:.1f}s  mc_p<0.05: {mc_ok}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()

    configs_arr, configs_meta = build_configs()
    print(f"Config space  : {len(configs_arr)} per pair")
    print(f"Pairs         : {args.pairs}")
    print(f"Entry         : M5 close, S5 fast[1,2,3] + slow[12,24,36] confluence")
    print(f"Exit          : wave-break (n_rev fast lags reverse) | TP ceiling | hard SL | timeout")
    print(f"Session filter: {SESSION_START:02d}:00-{SESSION_END:02d}:00 UTC")

    if args.fast:
        configs_arr  = configs_arr[:8]
        configs_meta = configs_meta[:8]

    compiled = [False]
    all_rows = []

    for pair_name in args.pairs:
        if pair_name not in PAIRS:
            print(f"Unknown pair {pair_name}, skip")
            continue
        rows = process(pair_name, PAIRS[pair_name], configs_arr, configs_meta, compiled)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = RES_DIR / "wave_exit_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    mc_pass = df[df["wf_pass"] & (df["mc_p"] < 0.05)].copy() if len(df) > 0 else pd.DataFrame()
    print(f"\n{'='*80}")
    if len(mc_pass) > 0:
        print(f"PASSED WF + MC  ({len(mc_pass)} configs)  ranked by OOS p/d:")
        cols = ["pair","tp","sl","n_rev","atr_mult","accel",
                "is_pd","is_wr","is_n","wave_pct","tp_pct",
                "oos_pd","oos_wr","oos_n","mc_p"]
        print(mc_pass.sort_values("oos_pd", ascending=False)[cols].head(40).to_string(index=False))
    else:
        print("No WF + MC survivors.")
        if len(df) > 0:
            wf_only = df[df["wf_pass"]].sort_values("is_pd", ascending=False).head(20)
            if len(wf_only):
                print(f"\nWF-only top 20 (no MC yet):")
                print(wf_only[["pair","tp","sl","n_rev","atr_mult","accel",
                                "is_pd","is_wr","is_n","wave_pct","tp_pct",
                                "wf1","wf2","wf3"]].to_string(index=False))
            else:
                print("No WF survivors.")
                # Show top IS p/d regardless — diagnose signal quality
                top = df.sort_values("is_pd", ascending=False).head(20)
                print(f"\nTop IS p/d (diagnostic):")
                print(top[["pair","tp","sl","n_rev","atr_mult","accel",
                            "is_pd","is_wr","is_n","wave_pct","tp_pct",
                            "wf1","wf2","wf3"]].to_string(index=False))

    # ── Exit reason breakdown ──────────────────────────────────────────────────
    if len(df) > 0 and df["is_n"].sum() > 0:
        print(f"\n{'─'*80}")
        print("Exit reason breakdown (IS, all configs aggregated, top 10 by is_pd):")
        top10 = df.sort_values("is_pd", ascending=False).head(10)
        for _, r in top10.iterrows():
            if r["is_n"] > 0:
                print(f"  {r['pair']} tp={r['tp']} sl={r['sl']} n_rev={r['n_rev']} "
                      f"atr_mult={r['atr_mult']} accel={r['accel']}  "
                      f"is_pd={r['is_pd']:.1f}  WR={r['is_wr']:.0%}  n={r['is_n']}  "
                      f"wave_exit={r['wave_pct']:.0%}  tp_exit={r['tp_pct']:.0%}")


if __name__ == "__main__":
    main()
