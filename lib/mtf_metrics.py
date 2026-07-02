"""Causal multi-timeframe trailing-rolling-window metrics from S5.

See docs/superpowers/specs/2026-06-29-causal-mtf-metrics-design.md.
Every window is strictly trailing, so each output row depends only on bars at or
before its own timestamp — no lookahead.
"""
import gc

import numpy as np
import pandas as pd

TF_MINUTES = {"m5": 5, "h1": 60, "h4": 240, "h8": 480, "d1": 1440, "w1": 10080}


def position_in_range(c, low, high):
    """Unclamped position of price `c` within reference range [low, high]:
    0=low, 1=high, 0.5=mid, >1 above the range, <0 below (the fraction beyond is the
    % of range penetrated). 0.5 where high==low (degenerate); NaN where the range is
    NaN (e.g. no reference window yet). Scalars or numpy arrays."""
    c = np.asarray(c, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    rng = high - low
    out = np.where(rng > 0, (c - low) / np.where(rng > 0, rng, 1.0), 0.5)
    return np.where(np.isnan(rng) | np.isnan(c), np.nan, out)


def metrics_from_ohlc(o, h, l, c, pip, nominal_min):
    """pm/eff/loc from a window's OHLC. Scalars or numpy arrays.

    pm  = (c - o) / pip / nominal_min          signed pips per minute
    eff = |c - o| / (h - l)                     0 where h == l
    loc = position_in_range(c, l, h)            position of c in [l,h] (see that fn).

    Note: build_mtf overrides loc to reference the PRIOR window (so price can
    penetrate it); here loc is the general position of c within this window's [l,h].
    """
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    rng = h - l
    safe = rng > 0
    denom = np.where(safe, rng, 1.0)            # avoid divide-by-zero warnings
    pm = (c - o) / pip / nominal_min
    eff = np.where(safe, np.abs(c - o) / denom, 0.0)
    loc = position_in_range(c, l, h)
    return pm, eff, loc


def rolling_window_ohlc(ts, open_, high, low, close, minutes):
    """Trailing, time-based, right-closed window OHLC for one timeframe.

    Window at step i = (ts[i] - minutes, ts[i]]. Returns four float64 arrays:
    open of the oldest bar in the window, rolling max(high), rolling min(low),
    and the current close. Strictly trailing ⇒ causal.
    """
    # normalize to ns: asi8 returns the index's OWN unit, so a [us]/[ms] input would
    # silently make every window 1000x/1e6x too long. Force ns so delta_ns matches.
    ts_idx = pd.DatetimeIndex(ts).as_unit("ns")
    ts_ns = ts_idx.asi8                                  # int64 nanoseconds
    delta_ns = np.int64(minutes) * 60 * 1_000_000_000
    # oldest bar in window: first index with ts > ts[i] - minutes
    j = np.searchsorted(ts_ns, ts_ns - delta_ns, side="right")
    open_win = np.asarray(open_, dtype=np.float64)[j]
    win = pd.Timedelta(minutes=minutes)                  # closed='right' by default
    high_win = pd.Series(np.asarray(high, dtype=np.float64), index=ts_idx) \
        .rolling(win, min_periods=1).max().to_numpy()
    low_win = pd.Series(np.asarray(low, dtype=np.float64), index=ts_idx) \
        .rolling(win, min_periods=1).min().to_numpy()
    close_win = np.asarray(close, dtype=np.float64)
    return open_win, high_win, low_win, close_win


def build_mtf(ts, open_, high, low, close, pip):
    """All six TFs' pm/eff/loc. Returns dict[str, float32 ndarray] keyed
    '<tf>_pm'/'<tf>_eff'/'<tf>_loc'. One TF at a time to bound memory.

    pm/eff describe the CURRENT trailing window (t-N, t]. loc references the PRIOR
    window — the trailing window as it stood N minutes ago, ~(t-2N, t-N] — so the
    current price can break out of it: loc>1 above the prior bar, <0 below. NaN until
    a prior window exists (warmup). Strictly trailing ⇒ causal."""
    out = {}
    ts_ns = pd.DatetimeIndex(ts).as_unit("ns").asi8
    for tf, minutes in TF_MINUTES.items():
        ow, hw, lw, cw = rolling_window_ohlc(ts, open_, high, low, close, minutes)
        pm, eff, _ = metrics_from_ohlc(ow, hw, lw, cw, pip, minutes)   # current window
        # prior window = the current window as it stood `minutes` ago, ~(t-2N, t-N]
        delta_ns = np.int64(minutes) * 60 * 1_000_000_000
        j = np.searchsorted(ts_ns, ts_ns - delta_ns, side="right") - 1
        ok = j >= 0
        safe_j = np.where(ok, j, 0)
        prior_high = np.where(ok, hw[safe_j], np.nan)
        prior_low = np.where(ok, lw[safe_j], np.nan)
        loc = position_in_range(cw, prior_low, prior_high)            # cw = price now
        out[f"{tf}_pm"] = pm.astype(np.float32)
        out[f"{tf}_eff"] = eff.astype(np.float32)
        out[f"{tf}_loc"] = loc.astype(np.float32)
        del ow, hw, lw, cw, pm, eff, loc, j, ok, safe_j, prior_high, prior_low
        gc.collect()
    return out
