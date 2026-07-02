"""Elementary MA feature library — 9 causal features from {bid, SMA5, SMA50}.

Causality contract
------------------
Every feature at bar ``t`` uses ONLY bars ``[0, t]``. No cross-resolution
merges, no future windows. Regression tests in
``tests/unit/test_elementary_ma_causality.py`` verify by mutating bars
``[t+1, end]`` and asserting outputs unchanged for indices ``[0, t]``.

Raw signals (pips, signed)
    d_bid_sma5[t]   = (bid[t] - SMA5(close)[t])   / pip_size
    d_bid_sma50[t]  = (bid[t] - SMA50(close)[t])  / pip_size
    d_sma5_sma50[t] = (SMA5[t] - SMA50[t])        / pip_size

Encodings (three ways to present the same raw signal)
    E-atan:    (2/π) * arctan(raw / S)     where S = 2 * rolling_MAD_500
    E-pz-norm: 2 * Φ(rolling_z) - 1        where z uses mean/std over 500 bars
    E-pz-rank: 2 * (rank/N) - 1            empirical percentile over 500 bars
    E-mom:     (2/π) * arctan(Δraw / S)    one-bar change of raw, arctan-scaled

Convention: bid ≈ mid_close here (training data has no separate bid/ask).
Live deployment must replace ``bid`` with the real curator bid.
"""
from __future__ import annotations

from math import pi

import numpy as np
from numba import njit


WINDOW_MAD = 500
WINDOW_PZ = 500
ARCTAN_K = 2.0        # S = ARCTAN_K * rolling_MAD
ATAN_SCALE = 2.0 / pi


# ── SMAs ─────────────────────────────────────────────────────────────────
@njit(cache=True)
def sma(x: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average, causal, pre-warmup bars = NaN."""
    out = np.empty(len(x))
    out[:] = np.nan
    if len(x) < n:
        return out
    s = 0.0
    for i in range(n):
        s += x[i]
    out[n - 1] = s / n
    for i in range(n, len(x)):
        s += x[i] - x[i - n]
        out[i] = s / n
    return out


# ── Rolling MAD ──────────────────────────────────────────────────────────
@njit(cache=True)
def _median_of_window(buf: np.ndarray) -> float:
    """Median of a buffer via insertion sort copy (small windows, fine)."""
    tmp = buf.copy()
    tmp.sort()
    m = len(tmp)
    if m % 2 == 1:
        return tmp[m // 2]
    return 0.5 * (tmp[m // 2 - 1] + tmp[m // 2])


@njit(cache=True)
def rolling_mad(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling median absolute deviation, causal. Pre-warmup = NaN.

    MAD_t = median( |x[i] - median(x[t-W+1..t])| for i in t-W+1..t )

    O(N * W log W). For N=400k W=500 this is roughly 30-60s per pair.
    Precompute once per pair and cache to .npy.
    """
    n = len(x)
    out = np.empty(n)
    out[:] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        win = x[i - window + 1:i + 1]
        med = _median_of_window(win)
        dev = np.abs(win - med)
        out[i] = _median_of_window(dev)
    return out


# ── Rolling mean / std for z-score ───────────────────────────────────────
@njit(cache=True)
def rolling_mean_std(x: np.ndarray, window: int):
    """Rolling mean & std (ddof=0), causal. Pre-warmup entries = NaN."""
    n = len(x)
    mean = np.empty(n); mean[:] = np.nan
    std = np.empty(n); std[:] = np.nan
    if n < window:
        return mean, std
    s = 0.0
    s2 = 0.0
    for i in range(window):
        s += x[i]
        s2 += x[i] * x[i]
    mean[window - 1] = s / window
    var = s2 / window - (s / window) ** 2
    std[window - 1] = np.sqrt(var) if var > 0 else 0.0
    for i in range(window, n):
        old = x[i - window]
        new = x[i]
        s += new - old
        s2 += new * new - old * old
        m = s / window
        mean[i] = m
        v = s2 / window - m * m
        std[i] = np.sqrt(v) if v > 0 else 0.0
    return mean, std


# ── Rolling empirical-rank percentile ────────────────────────────────────
@njit(cache=True)
def rolling_rank_percentile(x: np.ndarray, window: int) -> np.ndarray:
    """Map x[t] to 2*(rank/N) - 1 over the trailing window (causal).

    rank = position of x[t] in sorted window, 1-indexed. Middle → 0, top → ~+1,
    bottom → ~-1. Ties broken by left position (not significant for this use).

    O(N * W) (linear scan per bar). ~15-30s per 400k-bar pair. Cache to .npy.
    """
    n = len(x)
    out = np.empty(n); out[:] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        target = x[i]
        cnt = 0
        for k in range(window):
            if x[i - window + 1 + k] < target:
                cnt += 1
        # rank = cnt+1 (1-indexed); map (cnt+1)/window to [-1,+1]
        out[i] = 2.0 * (cnt + 0.5) / window - 1.0
    return out


# ── Normal CDF (numba-friendly) ──────────────────────────────────────────
@njit(cache=True)
def _phi(z: float) -> float:
    """Normal CDF via erf-free approximation (Abramowitz 7.1.26)."""
    # Use sign-preserving approx good to ~1.5e-7
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if z >= 0 else -1.0
    zz = abs(z) / 1.4142135623730951
    t = 1.0 / (1.0 + p * zz)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-zz * zz)
    return 0.5 * (1.0 + sign * y)


# ── Feature computation ──────────────────────────────────────────────────
def compute_raw_pips(bid: np.ndarray, close: np.ndarray, pip: float):
    """Returns (d_bid_sma5, d_bid_sma50, d_sma5_sma50) in pips."""
    s5 = sma(close, 5)
    s50 = sma(close, 50)
    d_bid_sma5 = (bid - s5) / pip
    d_bid_sma50 = (bid - s50) / pip
    d_sma5_sma50 = (s5 - s50) / pip
    return d_bid_sma5, d_bid_sma50, d_sma5_sma50, s5, s50


@njit(cache=True)
def atan_scale_with_mad(raw: np.ndarray, mad: np.ndarray,
                        k: float = ARCTAN_K,
                        floor: float = 1.0) -> np.ndarray:
    """(2/π) * arctan(raw / (k * mad)), with floor on mad to avoid div-by-0."""
    n = len(raw)
    out = np.empty(n); out[:] = np.nan
    for i in range(n):
        if np.isnan(raw[i]) or np.isnan(mad[i]):
            continue
        scale = k * mad[i]
        if scale < floor:
            scale = floor
        out[i] = ATAN_SCALE * np.arctan(raw[i] / scale)
    return out


@njit(cache=True)
def z_to_cdf_signed(mean: np.ndarray, std: np.ndarray,
                    raw: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Compute 2 * Φ((raw - mean) / std) - 1. Causal if inputs are causal."""
    n = len(raw)
    out = np.empty(n); out[:] = np.nan
    for i in range(n):
        if np.isnan(mean[i]) or np.isnan(std[i]) or np.isnan(raw[i]):
            continue
        s = std[i] if std[i] > floor else floor
        z = (raw[i] - mean[i]) / s
        out[i] = 2.0 * _phi(z) - 1.0
    return out


def compute_all_features(bid: np.ndarray, close: np.ndarray, pip: float):
    """Compute all 9 features. Returns dict keyed by feature name.

    9 features:
      atan_d_bid_sma5, atan_d_bid_sma50, atan_d_sma5_sma50
      pznorm_d_bid_sma5, pznorm_d_bid_sma50, pznorm_d_sma5_sma50
      pzrank_d_bid_sma5, pzrank_d_bid_sma50, pzrank_d_sma5_sma50
      mom_atan_d_bid_sma5, mom_atan_d_bid_sma50, mom_atan_d_sma5_sma50
    """
    d1, d2, d3, _, _ = compute_raw_pips(bid, close, pip)
    out = {}

    # E-atan
    mad1 = rolling_mad(d1, WINDOW_MAD)
    mad2 = rolling_mad(d2, WINDOW_MAD)
    mad3 = rolling_mad(d3, WINDOW_MAD)
    a1 = atan_scale_with_mad(d1, mad1)
    a2 = atan_scale_with_mad(d2, mad2)
    a3 = atan_scale_with_mad(d3, mad3)
    out["atan_d_bid_sma5"] = a1
    out["atan_d_bid_sma50"] = a2
    out["atan_d_sma5_sma50"] = a3

    # E-pz-norm — compute on NaN-stripped slice, then slot back
    for name, raw in [("pznorm_d_bid_sma5", d1),
                       ("pznorm_d_bid_sma50", d2),
                       ("pznorm_d_sma5_sma50", d3)]:
        mask = ~np.isnan(raw)
        first_valid = int(np.argmax(mask)) if mask.any() else len(raw)
        sub = raw[first_valid:]
        m, s = rolling_mean_std(sub, WINDOW_PZ)
        cdf = z_to_cdf_signed(m, s, sub)
        full = np.empty_like(raw); full[:] = np.nan
        full[first_valid:] = cdf
        out[name] = full

    # E-pz-rank — same NaN-strip trick
    for name, raw in [("pzrank_d_bid_sma5", d1),
                       ("pzrank_d_bid_sma50", d2),
                       ("pzrank_d_sma5_sma50", d3)]:
        mask = ~np.isnan(raw)
        first_valid = int(np.argmax(mask)) if mask.any() else len(raw)
        sub = raw[first_valid:]
        ranks = rolling_rank_percentile(sub, WINDOW_PZ)
        full = np.empty_like(raw); full[:] = np.nan
        full[first_valid:] = ranks
        out[name] = full

    # E-mom (one-bar change of raw, arctan-scaled against same MAD)
    dd1 = np.empty_like(d1); dd1[:] = np.nan
    dd2 = np.empty_like(d2); dd2[:] = np.nan
    dd3 = np.empty_like(d3); dd3[:] = np.nan
    dd1[1:] = d1[1:] - d1[:-1]
    dd2[1:] = d2[1:] - d2[:-1]
    dd3[1:] = d3[1:] - d3[:-1]
    out["mom_atan_d_bid_sma5"] = atan_scale_with_mad(dd1, mad1)
    out["mom_atan_d_bid_sma50"] = atan_scale_with_mad(dd2, mad2)
    out["mom_atan_d_sma5_sma50"] = atan_scale_with_mad(dd3, mad3)

    return out


ARM_FEATURES = {
    "E-atan":    ["atan_d_bid_sma5", "atan_d_bid_sma50", "atan_d_sma5_sma50"],
    "E-pz-norm": ["pznorm_d_bid_sma5", "pznorm_d_bid_sma50", "pznorm_d_sma5_sma50"],
    "E-pz-rank": ["pzrank_d_bid_sma5", "pzrank_d_bid_sma50", "pzrank_d_sma5_sma50"],
    "E-mom":     ["mom_atan_d_bid_sma5", "mom_atan_d_bid_sma50", "mom_atan_d_sma5_sma50"],
    "E-all": [
        "atan_d_bid_sma5", "atan_d_bid_sma50", "atan_d_sma5_sma50",
        "pznorm_d_bid_sma5", "pznorm_d_bid_sma50", "pznorm_d_sma5_sma50",
        "pzrank_d_bid_sma5", "pzrank_d_bid_sma50", "pzrank_d_sma5_sma50",
    ],
}
