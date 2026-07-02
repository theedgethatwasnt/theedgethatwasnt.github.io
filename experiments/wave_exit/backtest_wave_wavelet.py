#!/usr/bin/env python3
"""
Wave-Wavelet Exit Backtest
===========================
Same entry gate as wave_exit: M5 bar close + S5 fast[1,2,3] & slow[12,24,36] confluence.

Exit: Haar wavelet D_L signal reversal instead of raw fast-group flip.
  - D3 (level 3): compares avg(last 4 bars) vs avg(prev 4 bars)  →  40-second scale
  - D4 (level 4): compares avg(last 8 bars) vs avg(prev 8 bars)  →  80-second scale
  - D5 (level 5): compares avg(last 16 bars) vs avg(prev 16 bars) → 160-second (2.7 min)

The wavelet signal at level L is the Haar detail coefficient:
  D_L = avg(close[i-2^(L-1)+1 : i+1]) - avg(close[i-2^L+1 : i-2^(L-1)+1])

Positive D_L = upward trend at that scale. Negative = downward.

At entry: multi-scale confluence ensures D3/D4 are aligned with direction.
Post-entry: monitor D_L at every S5 bar. When D_L opposes direction for
n_confirm consecutive bars → wave at that scale has ended → exit.

This avoids the noise problem in wave_exit.py (S5 fast group fires every 1-2 bars).
The Haar DWT is causal and runs in O(2^L) per bar — fast in Numba.

Usage:
    python3 backtest_wave_wavelet.py [--pairs USD_JPY EUR_JPY] [--fast]
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

PAIRS = {
    "EUR_JPY": {"file": "EUR_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_JPY": {"file": "GBP_JPY_S5_BA.parquet", "pip": 0.01},
    "USD_JPY": {"file": "USD_JPY_S5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file": "GBP_USD_S5_BA.parquet", "pip": 0.0001},
    "EUR_USD": {"file": "EUR_USD_S5_BA.parquet", "pip": 0.0001},
}

IS_FRAC        = 0.70
N_WF_CHUNKS    = 3
MIN_IS_TRADES  = 50
MC_SHUFFLES    = 1000
M5_BARS        = 12
ATR_PERIOD     = 500
MAX_HOLD_BARS  = 240   # 20 min max hold (longer — wavelet exit is more selective)
SESSION_START  = 7
SESSION_END    = 17

FAST_LAGS = np.array([1,  2,  3],  dtype=np.int64)
SLOW_LAGS = np.array([12, 24, 36], dtype=np.int64)
N_LAGS    = 3

# ── Sweep space ────────────────────────────────────────────────────────────────
TP_PIPS_OPT   = [5, 10, 20, 30]      # TP ceiling
SL_PIPS_OPT   = [0, 10, 20]          # hard fallback SL (0 = wavelet-only)
WT_LEVEL_OPT  = [3, 4]               # Haar level to monitor (D3/D4)
WT_STRIDE_OPT = [6, 12, 24]          # S5 stride: 6=S30, 12=M1, 24=M2
                                      # D3@S30=4min  D3@M1=8min  D3@M2=16min
N_CONFIRM_OPT = [1, 2, 3]            # consecutive reversed bars to trigger exit
ATR_MULT_OPT  = [0.0, 0.5, 1.0]     # entry magnitude gate (vs ATR500)
ACCEL_OPT     = [False, True]        # entry acceleration requirement


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
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)
    df["spread_pips"] = ((df["ask_c"] - df["bid_c"]).astype(np.float64) / pip).clip(0.1, 20.0)
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
def haar_detail_strided(close, i, level, stride):
    """
    Haar D_level coefficient, sampling close every `stride` S5 bars.
    Effective time scale per sample: stride × 5 seconds.
      stride=1  → S5  raw    D3=40s   D4=80s   D5=160s
      stride=6  → S30        D3=4min  D4=8min  D5=16min
      stride=12 → M1         D3=8min  D4=16min D5=32min
      stride=24 → M2         D3=16min D4=32min
    Window spans 2^level × stride S5 bars total.
    Fully causal: only uses close[i - (2^level-1)*stride .. i].
    """
    window  = np.int64(1) << np.int64(level)   # 2^level samples
    half    = window >> np.int64(1)             # 2^(level-1)
    needed  = (window - np.int64(1)) * np.int64(stride)  # S5 bars needed before i

    if i < needed:
        return np.float64(0.0)

    # Left half: older samples (indices window-1 .. half in the strided series)
    # Right half: newer samples (indices half-1 .. 0 in strided series)
    left_sum  = np.float64(0.0)
    right_sum = np.float64(0.0)
    for k in range(half):
        # left sample k: i - (window-1-k)*stride
        left_sum  += close[i - (window - np.int64(1) - np.int64(k)) * np.int64(stride)]
        # right sample k: i - (half-1-k)*stride
        right_sum += close[i - (half - np.int64(1) - np.int64(k)) * np.int64(stride)]

    return (right_sum - left_sum) / np.float64(half)


@njit(cache=True, fastmath=True)
def _check_confluence(close, atr, i, direction, atr_mult, accel_req):
    """Fast[1,2,3] + slow[12,24,36] — all must agree in direction."""
    threshold = atr_mult * atr[i]
    for k in range(N_LAGS):
        lag = FAST_LAGS[k]
        mom = close[i] - close[i - lag]
        if direction == 1:
            if mom <= 0.0:                                     return False
            if atr_mult > 0.0 and mom < threshold:            return False
            if accel_req and mom <= close[i-lag] - close[i-2*lag]: return False
        else:
            if mom >= 0.0:                                     return False
            if atr_mult > 0.0 and -mom < threshold:           return False
            if accel_req and mom >= close[i-lag] - close[i-2*lag]: return False
    for k in range(N_LAGS):
        lag = SLOW_LAGS[k]
        mom = close[i] - close[i - lag]
        if direction == 1:
            if mom <= 0.0: return False
        else:
            if mom >= 0.0: return False
    return True


@njit(cache=True, fastmath=True)
def run_wave_trade_wt(close, high, low, spread_pips, entry_bar, entry_px,
                      direction, tp_pips, sl_pips, wt_level, wt_stride, n_confirm,
                      pip_size, max_hold):
    """
    Run one trade from entry_bar.
    Exit on: wavelet D_level reversal (n_confirm bars) | TP | hard SL | timeout.
      wt_stride: S5 bars between samples (6=S30, 12=M1, 24=M2).
    Returns (pnl_pips, bars_held, exit_reason).
      0=wavelet-exit  1=TP  2=hard-SL  3=timeout
    """
    n      = len(close)
    tp_lev = entry_px + np.float64(direction) * tp_pips * pip_size
    sl_lev = entry_px - np.float64(direction) * sl_pips * pip_size \
             if sl_pips > 0.0 else np.float64(-1e18) * np.float64(direction)

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)
    n_rev_count = np.int64(0)

    end = min(entry_bar + max_hold + 1, n)
    for j in range(entry_bar + 1, end):
        sp_j = spread_pips[j]
        bull = close[j] >= close[j - 1]

        # TP / hard SL first (SOP R2 sequencing)
        hit_tp = False; hit_sl = False
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
            return tp_f, np.int64(j - entry_bar), np.int64(1)
        if hit_sl:
            return -sl_f, np.int64(j - entry_bar), np.int64(2)

        # Wavelet exit: track D_level reversal count at strided resolution
        d_sig = haar_detail_strided(close, j, wt_level, wt_stride)
        if (direction == 1 and d_sig < 0.0) or (direction == -1 and d_sig > 0.0):
            n_rev_count += np.int64(1)
        else:
            n_rev_count = np.int64(0)   # reset on any pro-direction bar

        if n_rev_count >= n_confirm:
            pnl = (close[j] - entry_px) * np.float64(direction) / pip_size \
                  - np.float64(0.5) * sp_j
            return pnl, np.int64(j - entry_bar), np.int64(0)

    last = min(entry_bar + max_hold, n - 1)
    pnl  = (close[last] - entry_px) * np.float64(direction) / pip_size \
           - np.float64(0.5) * spread_pips[last]
    return pnl, np.int64(last - entry_bar), np.int64(3)


@njit(cache=True, fastmath=True)
def run_segment_wt(close, high, low, spread_pips, atr, in_session, pip_size,
                   tp_pips, sl_pips, wt_level, wt_stride, n_confirm, atr_mult, accel_req,
                   seg_start, seg_end, sp_gate):
    warmup     = SLOW_LAGS[N_LAGS - 1] * 2 + 2   # 74 bars
    start      = max(seg_start, warmup)
    next_entry = start

    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)
    n_wt_exit  = np.int64(0)
    n_tp_exit  = np.int64(0)
    n_sl_exit  = np.int64(0)
    n_to_exit  = np.int64(0)

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)
    wl   = np.int64(wt_level)
    ws   = np.int64(wt_stride)
    nc   = np.int64(n_confirm)

    i = start
    while i < seg_end - 1:
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
            pnl, bars, reason = run_wave_trade_wt(
                close, high, low, spread_pips,
                i, entry_px, direction,
                tp_f, sl_f, wl, ws, nc, pip_size, MAX_HOLD_BARS)

            total_pips += pnl
            n_trades   += np.int64(1)
            if pnl > 0.0: n_wins    += np.int64(1)
            if   reason == 0: n_wt_exit += np.int64(1)
            elif reason == 1: n_tp_exit += np.int64(1)
            elif reason == 2: n_sl_exit += np.int64(1)
            else:             n_to_exit += np.int64(1)

            next_entry = i + bars + np.int64(1)
        i += 1

    return total_pips, n_trades, n_wins, n_wt_exit, n_tp_exit, n_sl_exit, n_to_exit


@njit(cache=True, fastmath=True)
def run_segment_pnl_wt(close, high, low, spread_pips, atr, in_session, pip_size,
                       tp_pips, sl_pips, wt_level, wt_stride, n_confirm, atr_mult, accel_req,
                       seg_start, seg_end, sp_gate):
    MAX_T   = 20_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup     = SLOW_LAGS[N_LAGS - 1] * 2 + 2
    start      = max(seg_start, warmup)
    next_entry = start

    tp_f = np.float64(tp_pips)
    sl_f = np.float64(sl_pips)
    wl   = np.int64(wt_level)
    ws   = np.int64(wt_stride)
    nc   = np.int64(n_confirm)

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
            pnl, bars, _ = run_wave_trade_wt(
                close, high, low, spread_pips,
                i, entry_px, direction,
                tp_f, sl_f, wl, ws, nc, pip_size, MAX_HOLD_BARS)
            pnl_arr[n_t] = pnl
            n_t          += np.int64(1)
            next_entry   = i + bars + np.int64(1)
        i += 1

    return pnl_arr[:n_t]


# ── Config builder ─────────────────────────────────────────────────────────────
def build_configs():
    rows = []
    for tp, sl, wl, ws, nc, am_idx, ac in itertools.product(
            TP_PIPS_OPT, SL_PIPS_OPT,
            WT_LEVEL_OPT, WT_STRIDE_OPT, N_CONFIRM_OPT,
            range(len(ATR_MULT_OPT)), ACCEL_OPT):
        sl_val = sl
        if sl_val > 0 and sl_val >= tp:
            continue
        # cols: tp sl wl ws nc am_idx ac
        rows.append((tp, sl_val, wl, ws, nc, am_idx, int(ac)))
    arr  = np.array(rows, dtype=np.int32)
    meta = [(r[0], r[1], r[2], r[3], r[4], ATR_MULT_OPT[r[5]], bool(r[6])) for r in rows]
    return arr, meta


ATR_MULTS_ARR = np.array(ATR_MULT_OPT, dtype=np.float64)


@njit(parallel=True, cache=True)
def sweep_parallel_wt(close, high, low, spread_pips, atr, in_session, pip_size,
                      configs, is_end, wf_starts, wf_ends, sp_gate):
    n_cfg  = len(configs)
    n_wf   = len(wf_starts)
    n_stat = 7
    n_col  = (n_wf + 2) * n_stat
    out    = np.zeros((n_cfg, n_col), dtype=np.float64)
    n_all  = len(close)

    for c in prange(n_cfg):
        tp  = np.float64(configs[c, 0])
        sl  = np.float64(configs[c, 1])
        wl  = np.int64(configs[c, 2])
        ws  = np.int64(configs[c, 3])
        nc  = np.int64(configs[c, 4])
        am  = ATR_MULTS_ARR[configs[c, 5]]
        ac  = configs[c, 6] > 0

        p0,p1,p2,p3,p4,p5,p6 = run_segment_wt(
            close, high, low, spread_pips, atr, in_session, pip_size,
            tp, sl, wl, ws, nc, am, ac, 0, is_end, sp_gate)
        out[c,0]=p0; out[c,1]=p1; out[c,2]=p2; out[c,3]=p3
        out[c,4]=p4; out[c,5]=p5; out[c,6]=p6

        for k in range(n_wf):
            p0,p1,p2,p3,p4,p5,p6 = run_segment_wt(
                close, high, low, spread_pips, atr, in_session, pip_size,
                tp, sl, wl, ws, nc, am, ac, wf_starts[k], wf_ends[k], sp_gate)
            b = n_stat + k * n_stat
            out[c,b]=p0; out[c,b+1]=p1; out[c,b+2]=p2; out[c,b+3]=p3
            out[c,b+4]=p4; out[c,b+5]=p5; out[c,b+6]=p6

        p0,p1,p2,p3,p4,p5,p6 = run_segment_wt(
            close, high, low, spread_pips, atr, in_session, pip_size,
            tp, sl, wl, ws, nc, am, ac, is_end, n_all, sp_gate)
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
          f"OOS {n-is_end:,} ({oos_days}d)  SP_P90={sp_gate:.2f}p  ATR500={atr_med:.2f}p")

    if not compiled_already[0]:
        print("  Compiling Numba ...", end="", flush=True)
        t0 = time.time()
        _ws = wf_starts.copy(); _we = np.minimum(wf_ends, 2000)
        _ = sweep_parallel_wt(close[:3000], high[:3000], low[:3000], sp[:3000],
                              atr[:3000], in_sess[:3000], np.float64(pip),
                              configs_arr[:4], 2000, _ws, _we, sp_gate)
        compiled_already[0] = True
        print(f" {time.time()-t0:.1f}s")

    print("  Sweeping ...", end="", flush=True)
    t0  = time.time()
    raw = sweep_parallel_wt(close, high, low, sp, atr, in_sess, np.float64(pip),
                            configs_arr, is_end, wf_starts, wf_ends, sp_gate)
    print(f" {time.time()-t0:.1f}s")

    n_stat = 7
    n_wf   = N_WF_CHUNKS
    rows   = []

    for c, (tp, sl, wl, ws, nc, am, ac) in enumerate(configs_meta):
        def _g(seg, s):
            if   seg == "is":  return raw[c, s]
            elif seg == "oos": return raw[c, n_stat + n_wf*n_stat + s]
            else:              return raw[c, n_stat + seg*n_stat + s]

        is_pips = _g("is",0); is_n = int(_g("is",1)); is_w = int(_g("is",2))
        is_wt   = int(_g("is",3)); is_tp = int(_g("is",4))

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
            "tp":        tp,  "sl":        sl,
            "wt_level":  wl,  "wt_stride": ws,
            "n_confirm": nc,
            "atr_mult":  am,  "accel":     ac,
            "is_pd":    round(is_pd,  2),
            "is_wr":    round(is_w / is_n if is_n > 0 else 0.0, 3),
            "is_n":     is_n,
            "wt_pct":   round(is_wt / is_n if is_n > 0 else 0.0, 3),
            "tp_pct":   round(is_tp / is_n if is_n > 0 else 0.0, 3),
            "avg_win":  round(is_pips / is_w if is_w > 0 else 0.0, 2),
            "avg_loss": round((is_pips - is_w*(is_pips/is_n if is_n>0 else 0)) /
                              max(1, is_n - is_w), 2) if is_n > is_w > 0 else 0.0,
            "wf1":      round(wf_pds[0], 2),
            "wf2":      round(wf_pds[1], 2),
            "wf3":      round(wf_pds[2], 2),
            "wf_pass":  wf_pass,
            "oos_pd":   round(oos_pd, 2),
            "oos_wr":   round(oos_w / oos_n if oos_n > 0 else 0.0, 3),
            "oos_n":    oos_n,
            "mc_p":     np.nan,
        })

    survivors = [r for r in rows if r["wf_pass"]]
    print(f"  WF survivors: {len(survivors)}", end="  (MC ...)" if survivors else "\n",
          flush=True)
    for r in survivors:
        pnl_arr = run_segment_pnl_wt(
            close, high, low, sp, atr, in_sess, np.float64(pip),
            float(r["tp"]), float(r["sl"]), int(r["wt_level"]), int(r["wt_stride"]),
            int(r["n_confirm"]), float(r["atr_mult"]), bool(r["accel"]), 0, is_end, sp_gate)
        r["mc_p"] = round(run_mc(pnl_arr, is_days), 4)
    if survivors:
        mc_ok = sum(1 for r in survivors if not np.isnan(r["mc_p"]) and r["mc_p"] < 0.05)
        print(f"  mc_p<0.05: {mc_ok}")

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
    print(f"Entry gate    : M5 close, S5 fast[1,2,3] + slow[12,24,36] confluence")
    print(f"Exit          : Haar D_L reversal × n_confirm | TP | hard SL  (strided)")
    print(f"WT levels     : L=3,4  Strides=6(S30),12(M1),24(M2)")
    print(f"  D3@S30=4min  D3@M1=8min  D3@M2=16min  D4@S30=8min  D4@M1=16min")
    print(f"Session       : {SESSION_START:02d}:00-{SESSION_END:02d}:00 UTC")

    if args.fast:
        configs_arr  = configs_arr[:6]
        configs_meta = configs_meta[:6]

    compiled = [False]
    all_rows = []

    for pair_name in args.pairs:
        if pair_name not in PAIRS:
            continue
        rows = process(pair_name, PAIRS[pair_name], configs_arr, configs_meta, compiled)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = RES_DIR / "wave_wavelet_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df)} rows)")

    mc_pass = df[df["wf_pass"] & (df["mc_p"] < 0.05)].copy() if len(df) > 0 else pd.DataFrame()
    print(f"\n{'='*80}")
    if len(mc_pass) > 0:
        print(f"PASSED WF + MC  ({len(mc_pass)} configs):")
        cols = ["pair","tp","sl","wt_level","wt_stride","n_confirm","atr_mult","accel",
                "is_pd","is_wr","is_n","wt_pct","tp_pct","avg_win",
                "oos_pd","oos_wr","oos_n","mc_p"]
        print(mc_pass.sort_values("oos_pd", ascending=False)[cols].head(40).to_string(index=False))
    else:
        print("No WF + MC survivors.")
        if len(df) > 0:
            wf_only = df[df["wf_pass"]].sort_values("is_pd", ascending=False).head(20)
            if len(wf_only):
                print(f"\nWF-only survivors:")
                print(wf_only[["pair","tp","sl","wt_level","wt_stride","n_confirm","atr_mult","accel",
                               "is_pd","is_wr","is_n","wt_pct","tp_pct",
                               "wf1","wf2","wf3"]].to_string(index=False))
            else:
                top = df[df["is_n"] > 0].sort_values("is_pd", ascending=False).head(20)
                print(f"\nTop IS p/d (diagnostic — no WF pass):")
                if len(top):
                    print(top[["pair","tp","sl","wt_level","wt_stride","n_confirm","atr_mult","accel",
                               "is_pd","is_wr","is_n","wt_pct","tp_pct","avg_win",
                               "wf1","wf2","wf3"]].to_string(index=False))

    # Exit reason breakdown for top configs
    if len(df) > 0 and df["is_n"].sum() > 0:
        print(f"\n{'─'*80}")
        print("Exit mix for top-10 IS p/d configs:")
        top10 = df[df["is_n"] > 0].sort_values("is_pd", ascending=False).head(10)
        for _, r in top10.iterrows():
            print(f"  {r['pair']} tp={r['tp']} sl={r['sl']} L={r['wt_level']} "
                  f"st={r['wt_stride']} nc={r['n_confirm']} am={r['atr_mult']} ac={r['accel']}  "
                  f"pd={r['is_pd']:.1f}  WR={r['is_wr']:.0%}  n={r['is_n']}  "
                  f"wt_exit={r['wt_pct']:.0%}  tp_exit={r['tp_pct']:.0%}  "
                  f"avg_win={r['avg_win']:.1f}p")


if __name__ == "__main__":
    main()
