"""
ASI (Accumulative Swing Index) + SMA5 + MC(D)/MC(dD) — all JIT compiled.

Pipeline: OHLC -> ASI (Wilder) -> SMA5 -> MC(D), MC(dD)

The SMA5(ASI) is treated as a "synthetic price series" fed into
the standard multi-timeframe EMA3-EMA5 momentum consensus.
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def compute_asi(o, h, l, c, n, atr_period=14, atr_mult=3.0):
    """Compute Accumulative Swing Index on OHLC arrays."""
    EPSILON = 1e-10
    atr = np.zeros(n, dtype=np.float64)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    atr[0] = h[0] - l[0]
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        if i < atr_period:
            atr[i] = atr[i - 1] + (tr - atr[i - 1]) / (i + 1)
        else:
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr) / atr_period

    asi = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        C2, O2, H2, L2 = c[i], o[i], h[i], l[i]
        C1, O1 = c[i - 1], o[i - 1]

        N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
        t1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
        t2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
        t3 = (H2 - L2) + 0.25 * abs(C1 - O1)
        R = max(t1, max(t2, t3))
        if R <= 0:
            R = EPSILON
        K = max(abs(H2 - C1), abs(L2 - C1))
        limit = atr_mult * atr[i]
        if limit < EPSILON:
            limit = EPSILON
        SI = 50.0 * (N / R) * (K / limit)
        asi[i] = asi[i - 1] + SI

    return asi


@njit(cache=True)
def sma_jit(arr, period, n):
    """Simple Moving Average."""
    out = np.zeros(n, dtype=np.float64)
    cumsum = 0.0
    for i in range(n):
        cumsum += arr[i]
        if i >= period:
            cumsum -= arr[i - period]
        if i >= period - 1:
            out[i] = cumsum / period
        else:
            out[i] = cumsum / (i + 1)
    return out


@njit(cache=True)
def _ema_diff_mc(series, n, n_lags=5):
    """MC(D) and MC(dD) on a single-TF series via EMA3-EMA5."""
    mc_d = np.zeros(n, dtype=np.float64)
    mc_dd = np.zeros(n, dtype=np.float64)
    if n < n_lags + 3:
        return mc_d, mc_dd

    alpha3 = 2.0 / 4.0
    alpha5 = 2.0 / 6.0
    e3 = series[0]
    e5 = series[0]
    d_vals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        e3 = alpha3 * series[i] + (1.0 - alpha3) * e3
        e5 = alpha5 * series[i] + (1.0 - alpha5) * e5
        d_vals[i] = e3 - e5

    for i in range(n_lags + 1, n):
        pos = neg = 0
        for lag in range(n_lags):
            change = d_vals[i - lag] - d_vals[i - lag - 1]
            if change > 0:
                pos += 1
            elif change < 0:
                neg += 1
        mc_d[i] = (pos - neg) / n_lags

    for i in range(n_lags + 2, n):
        pos = neg = 0
        for lag in range(n_lags):
            j = i - lag
            if j >= 3:
                dd_now = d_vals[j] - 2.0 * d_vals[j - 1] + d_vals[j - 2]
                dd_prev = d_vals[j - 1] - 2.0 * d_vals[j - 2] + d_vals[j - 3]
                change = dd_now - dd_prev
                if change > 0:
                    pos += 1
                elif change < 0:
                    neg += 1
        mc_dd[i] = (pos - neg) / n_lags

    return mc_d, mc_dd


@njit(cache=True)
def compute_mc_on_series(series, n, tf_bars, weights, n_tfs, n_lags=5):
    """Multi-TF MC(D) and MC(dD) on a price-like series."""
    mc_d_out = np.zeros(n, dtype=np.float64)
    mc_dd_out = np.zeros(n, dtype=np.float64)
    tw = 0.0

    for tf_idx in range(n_tfs):
        bp = tf_bars[tf_idx]
        w = weights[tf_idx]
        n_tf = n // bp
        if n_tf < n_lags + 5:
            continue
        tw += w

        tf_series = np.empty(n_tf, dtype=np.float64)
        for j in range(n_tf):
            tf_series[j] = series[(j + 1) * bp - 1]

        tf_mc_d, tf_mc_dd = _ema_diff_mc(tf_series, n_tf, n_lags)

        for i in range(n):
            tf_i = min(i // bp, n_tf - 1)
            mc_d_out[i] += w * tf_mc_d[tf_i]
            mc_dd_out[i] += w * tf_mc_dd[tf_i]

    if tw > 0:
        for i in range(n):
            mc_d_out[i] /= tw
            mc_dd_out[i] /= tw

    return mc_d_out, mc_dd_out


# TFs matching existing MTFMC
TF_BARS_S5 = np.array([1, 2, 6, 12, 24, 60, 120, 360, 720], dtype=np.int64)
TF_SEC = [5, 10, 30, 60, 120, 300, 600, 1800, 3600]
TF_WEIGHTS = np.array([math.log2(max(s / 5, 1)) + 1 for s in TF_SEC], dtype=np.float64)
N_TFS = len(TF_BARS_S5)


def compute_asi_mc(o, h, l, c, n, sma_period=5):
    """Full pipeline: OHLC -> ASI -> SMA5 -> MC(D), MC(dD)."""
    asi = compute_asi(o, h, l, c, n)
    smooth = sma_jit(asi, sma_period, n)
    mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d, mc_dd
