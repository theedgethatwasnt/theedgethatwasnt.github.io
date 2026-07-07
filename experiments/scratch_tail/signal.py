#!/usr/bin/env python3
"""
signal.py — SMA-Scratch Tail-Bounding Test (scratch_tail): the entry SIGNAL and per-pair
scratch-window config, ported VERBATIM (R6) from `services/strategy_sma_scratch_paper/main.py`
(live paper service). `sma_n`, `six_of_six`, `wilder_atr`, `PairCfg`, `CONFIGS`, `SMA_N`,
`LAGS`, `TP_PIPS` below are byte-for-byte the same logic as the deployed file (only the
`deque` import path and docstring differ) — copied, not fitted, per PREREGISTRATION.md
"Frozen strategy parameters".

`build_pair_signal()` is new (not in the live service): it replays the verbatim per-bar
functions over a pair's full H1/M30 history to produce vectorized per-M5-bar arrays
(`dir_signal`, `atr_h1_price`) for the backtest harness, using the exact same bounded
`deque(maxlen=64)` buffers the live service uses (so ATR's re-seed-from-buffer quirk — see
docstring on `wilder_atr` below — is reproduced exactly, not "cleaned up": R7 parity requires
matching the live service's actual arithmetic, not a more theoretically-correct one).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bars import m5_to_h1, m5_to_m30

# ── verbatim from services/strategy_sma_scratch_paper/main.py ────────────────
SMA_N = 16
LAGS = (8, 10, 15)
TP_PIPS = 20.0
MIN_FETCH = SMA_N + max(LAGS) + 4   # warmup minimum (live-service parity constant, unused here)


@dataclass
class PairCfg:
    pair: str
    pip: float
    T_s_bars: int          # scratch activation bars (M5 bars)
    k_atr: float = None    # if set, W = k_atr * ATR_H1_at_entry (pips)
    W_pips: float = None   # else fixed-W mode
    T_q_bars: int = None   # quality-filter check bar (None = disabled)
    X_pips: float = None   # min MFE required at T_q


# Frozen 6-pair set (H7 sweep, 2026-06-01) — verbatim from the deployed CONFIGS list.
# NOTE (documented per PREREGISTRATION.md "Frozen strategy parameters" — a discrepancy between
# the pre-registration's prose and the live module, caught while porting): the live module's
# top-of-file DOCSTRING still describes a 3-pair "USD_JPY quality exit (T_q=2h, X=3p)" — but
# the actual deployed `CONFIGS` list below leaves T_q_bars/X_pips unset (None) for ALL 6 pairs,
# incl. USD_JPY, so the quality-filter branch is a no-op on the live service today. Ported
# verbatim (CONFIGS values control behavior, not the stale docstring prose) — the quality-filter
# code path is still implemented faithfully below (never exercised by these CONFIGS, exactly
# matching live).
CONFIGS = [
    PairCfg('USD_JPY', 0.01,   T_s_bars=48*12, k_atr=0.5),
    PairCfg('NZD_USD', 0.0001, T_s_bars=24*12, k_atr=1.5),
    PairCfg('GBP_USD', 0.0001, T_s_bars=12*12, k_atr=1.5),
    PairCfg('CAD_JPY', 0.01,   T_s_bars= 8*12, k_atr=0.5),
    PairCfg('AUD_USD', 0.0001, T_s_bars=12*12, k_atr=2.0),
    PairCfg('GBP_JPY', 0.01,   T_s_bars= 6*12, k_atr=1.5),
]
CONFIG_BY_PAIR = {c.pair: c for c in CONFIGS}
PAIRS = [c.pair for c in CONFIGS]


def sma_n(deq, n):
    if len(deq) < n: return None
    arr = list(deq)[-n:]
    return sum(arr) / len(arr)


def six_of_six(closes_deq, lags):
    cur = sma_n(closes_deq, SMA_N)
    if cur is None: return 0
    ups = dns = 0
    for lg in lags:
        past = None
        if len(closes_deq) >= SMA_N + lg:
            arr = list(closes_deq)
            past_window = arr[-(SMA_N + lg):-lg]
            past = sum(past_window) / len(past_window)
        if past is None: return 0
        if cur > past: ups += 1
        elif cur < past: dns += 1
    if ups == len(lags): return 1
    if dns == len(lags): return -1
    return 0


def wilder_atr(highs: deque, lows: deque, closes: deque, period: int = 14):
    """Compute Wilder ATR over the last (period+1) bars. Returns price units.
    NOTE (documented, not "fixed"): this recomputes the seed + RMA chain from whatever is
    CURRENTLY in the (bounded, maxlen=64 live) buffer every call — not a single continuous
    unbounded recursion. Reproduced exactly for R7 parity; see module docstring."""
    n = len(closes)
    if n < period + 1:
        return None
    h = list(highs); l = list(lows); c = list(closes)
    tr = [h[0] - l[0]]
    for i in range(1, n):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    atr_val = sum(tr[1:period+1]) / period
    for i in range(period+1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
    return atr_val


# ── backtest-only: vectorized-per-M5-bar signal builder (new, not in the live service) ──
_BUFLEN = 64  # matches live service's deque(maxlen=64)


def _replay_tf_signal(closes):
    """Replay six_of_six over one timeframe's close series using a maxlen=64 deque exactly
    like warmup()+process_pair() incrementally do. Returns an int8 array, one value per bar
    (aligned to that bar's OWN close — i.e. index k uses only bars 0..k, causal)."""
    buf = deque(maxlen=_BUFLEN)
    out = np.empty(len(closes), dtype=np.int8)
    for i, c in enumerate(closes):
        buf.append(c)
        out[i] = six_of_six(buf, LAGS)
    return out


def _replay_atr_h1(highs, lows, closes, period=14):
    """Replay wilder_atr over the H1 series using the same maxlen=64 deque convention.
    Returns a float array (price units), NaN before period+1 bars of history exist."""
    hbuf, lbuf, cbuf = deque(maxlen=_BUFLEN), deque(maxlen=_BUFLEN), deque(maxlen=_BUFLEN)
    out = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(len(closes)):
        hbuf.append(highs[i]); lbuf.append(lows[i]); cbuf.append(closes[i])
        v = wilder_atr(hbuf, lbuf, cbuf, period)
        if v is not None:
            out[i] = v
    return out


def build_pair_signal(pair, m5_df):
    """Precompute, for one pair's full M5-BA dataframe, the per-M5-bar arrays the harness
    needs: `dir_signal` (int8: +1/-1/0, the 6-of-6 H1+M30 agreement AS OF each M5 bar's own
    close — i.e. using the latest H1/M30 bar with close_time <= this M5 bar's close_time,
    exactly matching process_pair()'s "fetch fresh, use whatever's complete" live semantics)
    and `atr_h1_price` (float, Wilder ATR(14,H1) in price units, same as-of convention, used
    both for the scratch window W and, in stop-bearing arms, the disaster-stop level).

    Forward-fill is via `np.searchsorted` (backward/causal, side='right'-1) — an as-of join,
    the same "R4-adjacent but analysis-only" convention already used by
    research/experiments/multiday_contrarian/harness.py for M5<->H4 alignment (this module is
    a backtest/analysis harness, not itself a live feature builder — R4's incremental-state
    rule governs `lib/incremental_features.py`, the live curator; this harness's own R7 parity
    gate is what proves the as-of join reproduces the live service's actual behavior)."""
    h1 = m5_to_h1(m5_df)
    m30 = m5_to_m30(m5_df)

    h1_sig = _replay_tf_signal(h1["close"].to_numpy())
    m30_sig = _replay_tf_signal(m30["close"].to_numpy())
    atr_h1 = _replay_atr_h1(h1["high"].to_numpy(), h1["low"].to_numpy(), h1["close"].to_numpy())

    m5_ts = m5_df["timestamp"].to_numpy()
    # bars.py's "timestamp" column is the BIN START (same convention as multiday_contrarian/
    # bars.py's H4/D1) — must add the bar's own duration to get its CLOSE time before comparing
    # against M5 timestamps (multiday_contrarian/harness.py does the identical
    # `bar["timestamp"] + timedelta(hours=4)` step for H4; here it's +1h / +30min).
    h1_close_ts = (h1["timestamp"] + pd.Timedelta(hours=1)).to_numpy()
    m30_close_ts = (m30["timestamp"] + pd.Timedelta(minutes=30)).to_numpy()

    # as-of (backward, causal): for each M5 bar, the index of the latest H1/M30 bar whose
    # CLOSE time <= this M5 bar's close time — "the latest H1/M30 bar that has already closed
    # at or before this M5 close", matching process_pair()'s "fetch fresh, use whatever's
    # complete" live semantics (gate 2 empirically validates this choice; see PREREGISTRATION.md).
    h1_idx = np.searchsorted(h1_close_ts, m5_ts, side="right") - 1
    m30_idx = np.searchsorted(m30_close_ts, m5_ts, side="right") - 1

    n = len(m5_df)
    h_sig_ffill = np.zeros(n, dtype=np.int8)
    m_sig_ffill = np.zeros(n, dtype=np.int8)
    atr_ffill = np.full(n, np.nan, dtype=np.float64)

    valid_h1 = h1_idx >= 0
    h_sig_ffill[valid_h1] = h1_sig[h1_idx[valid_h1]]
    atr_ffill[valid_h1] = atr_h1[h1_idx[valid_h1]]
    valid_m30 = m30_idx >= 0
    m_sig_ffill[valid_m30] = m30_sig[m30_idx[valid_m30]]

    dir_signal = np.zeros(n, dtype=np.int8)
    dir_signal[(h_sig_ffill == 1) & (m_sig_ffill == 1)] = 1
    dir_signal[(h_sig_ffill == -1) & (m_sig_ffill == -1)] = -1

    return dir_signal, atr_ffill
