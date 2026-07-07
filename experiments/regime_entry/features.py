"""Regime feature vector on a closed 60-sample S5 window (master doc Part I §3).

Zigzag rule (PREREGISTRATION.md): plain alternating extrema on closes
(TopsBots Stage 1+2 pattern, lib/swing_indicators.py lineage, no Stage-3 gate),
window endpoints included as leg boundaries. Legal here despite right-neighbor
comparisons: the whole window predates the decision point (closed window).
"""
import numpy as np
from numba import njit

FEATURE_NAMES = ["drift", "er", "rv", "peak_slope", "trough_slope",
                 "leg_expansion", "vr4", "n_legs", "vr2", "vr8"]


@njit(cache=True)
def _ols_slope(x, y):
    n = len(x)
    if n < 2:
        return np.nan
    mx = np.mean(x); my = np.mean(y)
    denom = np.sum((x - mx) ** 2)
    if denom == 0.0:
        return np.nan
    return np.sum((x - mx) * (y - my)) / denom


@njit(cache=True)
def _variance_ratio(diffs, k):
    """Lo-MacKinlay VR(k): Var(k-step)/(k*Var(1-step)), overlapping k-sums."""
    n = len(diffs)
    if n < k + 2:
        return np.nan
    v1 = np.var(diffs)
    if v1 == 0.0:
        return np.nan
    m = n - k + 1
    ks = np.empty(m)
    for i in range(m):
        s = 0.0
        for j in range(k):
            s += diffs[i + j]
        ks[i] = s
    return np.var(ks) / (k * v1)


@njit(cache=True)
def compute_regime_features(closes, pip):
    """Returns (drift, er, rv, peak_slope, trough_slope, leg_expansion,
    vr4, n_legs, vr2, vr8); pip-denominated where dimensional."""
    n = len(closes)
    drift = (closes[-1] - closes[0]) / pip
    diffs = np.diff(closes)
    rv = np.std(diffs) / pip

    # Stage 1: strict local extrema on closes
    idx = np.empty(n, dtype=np.int64)
    typ = np.empty(n, dtype=np.int8)          # +1 peak, -1 trough
    m = 0
    for i in range(1, n - 1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            idx[m] = i; typ[m] = 1; m += 1
        elif closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            idx[m] = i; typ[m] = -1; m += 1

    # Stage 2: alternation — keep most extreme of same-type runs
    aidx = np.empty(m, dtype=np.int64)
    atyp = np.empty(m, dtype=np.int8)
    a = 0
    i = 0
    while i < m:
        t = typ[i]
        best = i
        j = i
        while j < m and typ[j] == t:
            if (t == 1 and closes[idx[j]] > closes[idx[best]]) or \
               (t == -1 and closes[idx[j]] < closes[idx[best]]):
                best = j
            j += 1
        aidx[a] = idx[best]; atyp[a] = t; a += 1
        i = j

    # Envelope slopes: interior alternating extrema only, in pips per S5 bar
    npk = 0; ntr = 0
    for q in range(a):
        if atyp[q] == 1: npk += 1
        else: ntr += 1
    peak_slope = np.nan; trough_slope = np.nan
    if npk >= 2:
        px = np.empty(npk); py = np.empty(npk); p = 0
        for q in range(a):
            if atyp[q] == 1:
                px[p] = aidx[q]; py[p] = closes[aidx[q]] / pip; p += 1
        peak_slope = _ols_slope(px, py)
    if ntr >= 2:
        tx = np.empty(ntr); ty = np.empty(ntr); p = 0
        for q in range(a):
            if atyp[q] == -1:
                tx[p] = aidx[q]; ty[p] = closes[aidx[q]] / pip; p += 1
        trough_slope = _ols_slope(tx, ty)

    # Legs: endpoints + alternating extrema tile the path
    nz = a + 2
    zval = np.empty(nz)
    zval[0] = closes[0]
    for q in range(a):
        zval[q + 1] = closes[aidx[q]]
    zval[nz - 1] = closes[-1]
    nlegs = nz - 1
    legs = np.empty(nlegs)
    path = 0.0
    for q in range(nlegs):
        legs[q] = (zval[q + 1] - zval[q]) / pip
        path += abs(legs[q])

    er = np.nan
    if path > 0.0:
        er = abs(drift) / path

    leg_expansion = np.nan
    if nlegs >= 4:
        lx = np.arange(nlegs).astype(np.float64)
        ly = np.abs(legs)
        leg_expansion = _ols_slope(lx, ly)

    vr2 = _variance_ratio(diffs, 2)
    vr4 = _variance_ratio(diffs, 4)
    vr8 = _variance_ratio(diffs, 8)

    return (drift, er, rv, peak_slope, trough_slope, leg_expansion,
            vr4, float(nlegs), vr2, vr8)
