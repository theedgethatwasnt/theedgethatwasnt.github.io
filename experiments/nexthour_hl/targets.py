"""Forward-rolling 60-min high/low excursions (the prediction target).

The window is strictly forward and left-open, (t, t+horizon], so the bar at t is
never included. Future bars are used for LABELS only — features (computed elsewhere)
remain strictly trailing.
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _forward_extremes(ts_ns, high, low, horizon_ns):
    """For each t: max(high) and min(low) over indices j with t < j and
    ts[j] <= ts[t]+horizon. Returns (fmax, fmin, rcover) where rcover is the last
    forward index (or -1 if none). O(N) via monotonic deques; both window edges
    (left = t+1, right) advance monotonically."""
    n = ts_ns.shape[0]
    fmax = np.full(n, np.nan)
    fmin = np.full(n, np.nan)
    rcover = np.full(n, -1, np.int64)
    maxdq = np.empty(n + 1, np.int64); mf = 0; mb = 0     # decreasing-high deque [mf, mb)
    mindq = np.empty(n + 1, np.int64); nf = 0; nb = 0     # increasing-low deque [nf, nb)
    r = -1
    for t in range(n):
        limit = ts_ns[t] + horizon_ns
        while r + 1 < n and ts_ns[r + 1] <= limit:
            r += 1
            while mb > mf and high[maxdq[mb - 1]] <= high[r]:
                mb -= 1
            maxdq[mb] = r; mb += 1
            while nb > nf and low[mindq[nb - 1]] >= low[r]:
                nb -= 1
            mindq[nb] = r; nb += 1
        while mb > mf and maxdq[mf] <= t:     # window is (t, r] → drop indices <= t
            mf += 1
        while nb > nf and mindq[nf] <= t:
            nf += 1
        if r > t and mb > mf:
            fmax[t] = high[maxdq[mf]]
            fmin[t] = low[mindq[nf]]
            rcover[t] = r
    return fmax, fmin, rcover


def forward_excursions(ts, high, low, close, pip, horizon_min=60, min_cover_min=30):
    ts_ns = pd.DatetimeIndex(ts).as_unit("ns").asi8
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    horizon_ns = np.int64(horizon_min) * 60 * 1_000_000_000
    cover_ns = np.int64(min_cover_min) * 60 * 1_000_000_000
    fmax, fmin, rcover = _forward_extremes(ts_ns, high, low, horizon_ns)
    safe_r = np.where(rcover >= 0, rcover, 0)
    valid = (rcover >= 0) & (ts_ns[safe_r] >= ts_ns + cover_ns)
    up_pips = (fmax - close) / pip
    dn_pips = (close - fmin) / pip
    up_pips[~valid] = np.nan
    dn_pips[~valid] = np.nan
    return up_pips, dn_pips, valid
