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


@njit(cache=True)
def compute_mc_causal_batch(series, n, indices, tf_bars, weights, n_tfs, n_lags=5):
    """Causal mc_d/mc_dd at arbitrary M5 indices — O(n + k*n_tfs), not O(k*n).

    For each index i in `indices`, returns the value that
    compute_mc_on_series(series[:i+1], i+1, ...)[-1] would return — i.e. the
    causal value available at bar i, using only series[0..i].

    Causal TF-window mapping (equivalent to compute_mc_on_series truncation):
      tf_series[j] = series[(j+1)*bp - 1]   (same formula as compute_mc_on_series)
      j_causal = i//bp  if i%bp == bp-1      (bar i is the last bar of window j)
               = i//bp - 1  otherwise         (window i//bp is incomplete — use prev)
      This is proven equivalent to compute_mc_on_series(series[:i+1], i+1)[-1].

    Kept in the same file as compute_mc_on_series so any change to the TF-series
    formula (line above) is immediately visible here. Both must use the same
    tf_series[j] = series[(j+1)*bp - 1] construction.

    Args:
        series:  smooth M5 price series (e.g. sma5(ASI)), length n
        n:       len(series)
        indices: int64 array of M5 bar indices to evaluate
        tf_bars, weights, n_tfs: same as compute_mc_on_series
    Returns:
        mc_d[k], mc_dd[k] for each index k in indices
    """
    k = len(indices)
    mc_d_out  = np.zeros(k, dtype=np.float64)
    mc_dd_out = np.zeros(k, dtype=np.float64)
    tw_out    = np.zeros(k, dtype=np.float64)

    for tf_idx in range(n_tfs):
        bp = int(tf_bars[tf_idx])
        w  = float(weights[tf_idx])

        n_tf_full = n // bp
        if n_tf_full < 1:
            continue

        # TF series — identical formula to compute_mc_on_series
        tf_series = np.empty(n_tf_full, dtype=np.float64)
        for j in range(n_tf_full):
            tf_series[j] = series[(j + 1) * bp - 1]

        # Run EMA diff MC on full TF series once
        tf_mc_d, tf_mc_dd = _ema_diff_mc(tf_series, n_tf_full, n_lags)

        for ki in range(k):
            i = indices[ki]
            n_tf_causal = (i + 1) // bp
            if n_tf_causal < n_lags + 5:
                continue

            # Last complete TF window at bar i
            j_here = i // bp
            j_causal = j_here if (i % bp == bp - 1) else j_here - 1
            if j_causal < n_lags + 1 or j_causal >= n_tf_full:
                continue

            mc_d_out[ki]  += w * tf_mc_d[j_causal]
            mc_dd_out[ki] += w * tf_mc_dd[j_causal]
            tw_out[ki]    += w

    for ki in range(k):
        if tw_out[ki] > 0.0:
            mc_d_out[ki]  /= tw_out[ki]
            mc_dd_out[ki] /= tw_out[ki]

    return mc_d_out, mc_dd_out


# TFs matching existing MTFMC
TF_BARS_S5 = np.array([1, 2, 6, 12, 24, 60, 120, 360, 720], dtype=np.int64)
TF_SEC = [5, 10, 30, 60, 120, 300, 600, 1800, 3600]
TF_WEIGHTS = np.array([math.log2(max(s / 5, 1)) + 1 for s in TF_SEC], dtype=np.float64)
N_TFS = len(TF_BARS_S5)


def compute_asi_mc(o, h, l, c, n):
    """Variant A: OHLC -> ASI -> SMA5 -> MC(D), MC(dD). Single-TF, virtual resample."""
    asi = compute_asi(o, h, l, c, n)
    smooth = sma_jit(asi, 5, n)
    mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d, mc_dd


@njit(cache=True)
def _resample_s5_to_tf(o_s5, h_s5, l_s5, c_s5, n_s5, bars_per):
    """Resample S5 OHLC to a higher TF."""
    n_tf = n_s5 // bars_per
    o = np.empty(n_tf, dtype=np.float64)
    h = np.empty(n_tf, dtype=np.float64)
    l = np.empty(n_tf, dtype=np.float64)
    c = np.empty(n_tf, dtype=np.float64)
    for j in range(n_tf):
        start = j * bars_per
        o[j] = o_s5[start]
        c[j] = c_s5[start + bars_per - 1]
        hh = h_s5[start]
        ll = l_s5[start]
        for k in range(start + 1, start + bars_per):
            if h_s5[k] > hh:
                hh = h_s5[k]
            if l_s5[k] < ll:
                ll = l_s5[k]
        h[j] = hh
        l[j] = ll
    return o, h, l, c, n_tf


@njit(cache=True)
def _mc_on_mapped_series(mapped, n_m5, w, mc_d, mc_dd, n_lags=5):
    """EMA3-EMA5 MC on a mapped series, accumulate weighted into mc_d/mc_dd."""
    alpha3 = 2.0 / 4.0
    alpha5 = 2.0 / 6.0
    e3 = mapped[0]
    e5 = mapped[0]
    d_vals = np.zeros(n_m5, dtype=np.float64)
    for i in range(n_m5):
        e3 = alpha3 * mapped[i] + (1 - alpha3) * e3
        e5 = alpha5 * mapped[i] + (1 - alpha5) * e5
        d_vals[i] = e3 - e5

    for i in range(n_lags + 1, n_m5):
        pos = neg = 0
        for lag in range(n_lags):
            change = d_vals[i - lag] - d_vals[i - lag - 1]
            if change > 0:
                pos += 1
            elif change < 0:
                neg += 1
        mc_d[i] += w * (pos - neg) / n_lags

    for i in range(n_lags + 2, n_m5):
        pos = neg = 0
        for lag in range(n_lags):
            ji = i - lag
            if ji >= 3:
                dd_now = d_vals[ji] - 2 * d_vals[ji - 1] + d_vals[ji - 2]
                dd_prev = d_vals[ji - 1] - 2 * d_vals[ji - 2] + d_vals[ji - 3]
                change = dd_now - dd_prev
                if change > 0:
                    pos += 1
                elif change < 0:
                    neg += 1
        mc_dd[i] += w * (pos - neg) / n_lags


@njit(cache=True)
def _map_tf_to_m5(smooth_tf, n_tf, bars_per_m5_in_tf, n_m5):
    """Map TF SMA5(ASI) values to M5 cadence (latest available TF bar)."""
    mapped = np.zeros(n_m5, dtype=np.float64)
    for i in range(n_m5):
        tf_i = min(i // bars_per_m5_in_tf, n_tf - 1)
        mapped[i] = smooth_tf[tf_i]
    return mapped


def compute_asi_mc_multitf(o_s5, h_s5, l_s5, c_s5, n_s5):
    """Variant B: Multi-TF ASI → SMA5 each → EMA3-EMA5 consensus at M5 cadence.

    Computes ASI on S5/S30/M1/M5/H1 from raw S5 OHLC, maps each to M5,
    runs weighted EMA3-EMA5 MC consensus.

    Args: S5 OHLC arrays + length
    Returns: mc_d[n_m5], mc_dd[n_m5] at M5 cadence
    """
    # TF definitions: (bars_per_s5, bars_per_m5_in_tf, tf_seconds)
    # S5=1 bar, S30=6, M1=12, M5=60, H1=720
    tf_configs = [
        (1,    60,    5),      # S5: 1 S5 bar, 60 S5 per M5
        (6,    10,    30),     # S30: 6 S5 bars, 10 S30 per M5
        (12,   5,     60),     # M1: 12 S5, 5 M1 per M5
        (60,   1,     300),    # M5: 60 S5, 1:1
        (720,  1,     3600),   # H1: 720 S5, 1 H1 per 12 M5
    ]
    # For H1: bars_per_m5_in_tf = 1/12, but we need int → use special handling
    # Actually: M5 index / (720/60) = M5 index / 12

    n_m5 = n_s5 // 60  # 60 S5 bars = 1 M5 bar
    if n_m5 < 50:
        return np.zeros(n_m5, dtype=np.float64), np.zeros(n_m5, dtype=np.float64)

    mc_d = np.zeros(n_m5, dtype=np.float64)
    mc_dd = np.zeros(n_m5, dtype=np.float64)
    tw = 0.0

    for bars_per, m5_per_tf, tf_sec in tf_configs:
        w = math.log2(max(tf_sec / 5, 1)) + 1

        if bars_per == 1:
            # S5: ASI on raw S5, then map to M5 (take every 60th)
            asi = compute_asi(o_s5, h_s5, l_s5, c_s5, n_s5)
            smooth = sma_jit(asi, 5, n_s5)
            # Map: for M5 bar i, use S5 bar (i+1)*60 - 1
            mapped = np.zeros(n_m5, dtype=np.float64)
            for i in range(n_m5):
                s5_idx = min((i + 1) * 60 - 1, n_s5 - 1)
                mapped[i] = smooth[s5_idx]
        else:
            # Resample S5 → TF
            o_tf, h_tf, l_tf, c_tf, n_tf = _resample_s5_to_tf(o_s5, h_s5, l_s5, c_s5, n_s5, bars_per)
            if n_tf < 20:
                continue
            asi = compute_asi(o_tf, h_tf, l_tf, c_tf, n_tf)
            smooth = sma_jit(asi, 5, n_tf)

            if bars_per == 60:
                # M5: 1:1 mapping
                mapped = smooth[:n_m5] if n_tf >= n_m5 else np.zeros(n_m5, dtype=np.float64)
                if n_tf >= n_m5:
                    mapped = smooth[:n_m5].copy()
                else:
                    mapped = np.zeros(n_m5, dtype=np.float64)
                    mapped[:n_tf] = smooth
            elif bars_per == 720:
                # H1: 1 H1 = 12 M5 bars
                mapped = np.zeros(n_m5, dtype=np.float64)
                for i in range(n_m5):
                    h1_i = min(i // 12, n_tf - 1)
                    mapped[i] = smooth[h1_i]
            else:
                # S30, M1: m5_per_tf TF bars per M5
                mapped = np.zeros(n_m5, dtype=np.float64)
                for i in range(n_m5):
                    tf_i = min((i + 1) * m5_per_tf - 1, n_tf - 1)
                    mapped[i] = smooth[tf_i]

        _mc_on_mapped_series(mapped, n_m5, w, mc_d, mc_dd)
        tw += w

    if tw > 0:
        for i in range(n_m5):
            mc_d[i] /= tw
            mc_dd[i] /= tw

    return mc_d, mc_dd
