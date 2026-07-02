"""
Swing Structure Indicators for NEAT Training
=============================================
TopsBots 3-stage algorithm applied to price H/L and ASI series.

Computes (all at M5 cadence, pre-computed for training):
  SB_A        : ASI structural breakout state {-1,-0.5,0,+0.5,+1}
  ERP_A       : Extended Range Position on ASI (unclamped continuous, replaces SB_A + RangePos_ASI)
  ERP_P       : Extended Range Position on price (unclamped continuous, replaces SB_P + RangePos_price)
  HH_ASI      : higher_highs binary {0,1} for HHHL decomposition on ASI
  HL_ASI      : higher_lows  binary {0,1} for HHHL decomposition on ASI
  HH_PRICE    : higher_highs binary {0,1} for HHHL decomposition on price
  HL_PRICE    : higher_lows  binary {0,1} for HHHL decomposition on price

Encoding summary:
  SB_A / SB_P : {-2,-1,0,+1,+2} raw int → normalized {-1,-0.5,0,+0.5,+1}
  ERP_*       : (close - active_lsp) / (active_hsp - active_lsp), unclamped
                > 1.0  = above HSP  (SB = +2 zone)
                [0, 1] = between    (SB = ±1 zone, with depth)
                < 0.0  = below LSP  (SB = -2 zone)
  HH_*, HL_*  : binary 0/1 — HHHL broken into 2 independent components

Algorithm (TopsBots 3-stage):
  Stage 1: N=1 local extremes (h[i] > h[i-1] AND h[i] > h[i+1])
  Stage 2: Alternation — within same-type runs keep most extreme
  Stage 3: Exceeding-extremes gate — new HSP only if higher than last OR new LSP since

References:
  <GITHUB_USER>/TopsBots_Python, <GITHUB_USER>/TopsBots_MQL4
  research/charts/joint_states.py (original 5-state implementation)
"""

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1+2+3: TopsBots core — returns list of (idx, type, value) confirmed swings
# Pure Python: runs once at export time, not during per-genome eval
# ─────────────────────────────────────────────────────────────────────────────

def topsbots_swings(hi, lo):
    """TopsBots 3-stage swing detection on hi/lo series.
    Returns sorted list of (bar_idx, 'H'|'L', value) confirmed swing points.
    hi, lo: numpy arrays of equal length N.
    Requires 1-bar lookahead — last bar never generates a swing.
    """
    N = len(hi)
    # Stage 1: N=1 local extremes
    raw = []
    for i in range(1, N - 1):
        if hi[i] > hi[i-1] and hi[i] > hi[i+1]:
            raw.append((i, 'H', float(hi[i])))
        if lo[i] < lo[i-1] and lo[i] < lo[i+1]:
            raw.append((i, 'L', float(lo[i])))
    raw.sort(key=lambda x: x[0])

    # Stage 2: alternation — keep most extreme of consecutive same-type run
    alt = []
    i = 0
    while i < len(raw):
        t = raw[i][1]
        j = i
        while j < len(raw) and raw[j][1] == t:
            j += 1
        run = raw[i:j]
        best = max(run, key=lambda x: x[2]) if t == 'H' else min(run, key=lambda x: x[2])
        alt.append(best)
        i = j

    # Stage 3: exceeding-extremes gate
    sig = []
    lh = ll = None
    glh = glh2 = False   # glh=gate_for_low_unlock, glh2=gate_for_high_unlock
    for (idx, t, v) in alt:
        if t == 'H':
            if lh is None or v > lh or glh:
                sig.append((idx, 'H', v))
                lh = v
                glh = False
                glh2 = True
        else:
            if ll is None or v < ll or glh2:
                sig.append((idx, 'L', v))
                ll = v
                glh2 = False
                glh = True
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 5-state encoder + Extended Range Position — both from confirmed swings
# ─────────────────────────────────────────────────────────────────────────────

def compute_swing_features(series_hi, series_lo, mid_series):
    """
    Given hi/lo detection series and a mid_series for position measurement,
    compute 5-state breakout indicator AND extended range position.

    Args:
        series_hi : array used for swing HIGH detection (h[] for price, asi[] for ASI)
        series_lo : array used for swing LOW  detection (l[] for price, asi[] for ASI)
        mid_series: array used for position measurement (close[] for price, asi[] for ASI)

    Returns:
        state      : int8 array, {-2,-1,0,+1,+2}
        erp        : float64 array, unclamped (close-lsp)/(hsp-lsp)
        act_h      : float64 array, active HSP at each bar (nan until established)
        act_l      : float64 array, active LSP at each bar (nan until established)
        sig        : list of (idx,'H'|'L',value) confirmed swings
    """
    N = len(series_hi)
    sig = topsbots_swings(series_hi, series_lo)
    sw = {s[0]: (s[1], s[2]) for s in sig}

    act_h = np.full(N, np.nan, dtype=np.float64)
    act_l = np.full(N, np.nan, dtype=np.float64)
    state = np.zeros(N, dtype=np.int8)
    erp   = np.full(N, np.nan, dtype=np.float64)

    lhv = lvv = np.nan
    cur_last = 0   # +1 if last confirmed was LSP (ascending), -1 if HSP (descending)

    for i in range(N):
        if i in sw:
            t, v = sw[i]
            if t == 'H':
                lhv = v
                cur_last = -1   # last confirmed = HSP → descending context
            else:
                lvv = v
                cur_last = +1   # last confirmed = LSP → ascending context
        act_h[i] = lhv
        act_l[i] = lvv

        m = float(mid_series[i])

        if not np.isnan(act_h[i]) and not np.isnan(act_l[i]):
            rng = act_h[i] - act_l[i]
            if m > act_h[i]:
                state[i] = +2
                erp[i] = (m - act_l[i]) / rng if rng > 0 else 1.0
            elif m < act_l[i]:
                state[i] = -2
                erp[i] = (m - act_l[i]) / rng if rng > 0 else 0.0
            else:
                state[i] = cur_last  # +1 or -1
                erp[i] = (m - act_l[i]) / rng if rng > 0 else 0.5
        elif not np.isnan(act_h[i]):
            state[i] = -1 if m <= act_h[i] else +2
            erp[i] = np.nan
        elif not np.isnan(act_l[i]):
            state[i] = +1 if m >= act_l[i] else -2
            erp[i] = np.nan
        else:
            state[i] = 0
            erp[i] = np.nan

    return state, erp, act_h, act_l, sig


def state_to_normalized(state):
    """Map int8 state {-2,-1,0,+1,+2} → float {-1,-0.5,0,+0.5,+1}."""
    return state.astype(np.float64) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# HHHL binary decomposition — 2 independent binary inputs
# ─────────────────────────────────────────────────────────────────────────────

def compute_hhhl_binary(sig, N):
    """
    From confirmed swing list, compute rolling HHHL binary decomposition.
    At each bar, uses the 2 most recent confirmed HSPs and 2 most recent LSPs.

    Returns:
        hh : float64 array {0.0, 1.0} — last HSP > prev HSP (higher high)
        hl : float64 array {0.0, 1.0} — last LSP > prev LSP (higher low)
        Both NaN until 2 HSPs + 2 LSPs confirmed.
    """
    hh = np.full(N, np.nan, dtype=np.float64)
    hl = np.full(N, np.nan, dtype=np.float64)

    hsps = []   # (idx, val) confirmed highs in order
    lsps = []   # (idx, val) confirmed lows in order
    sw = {s[0]: (s[1], s[2]) for s in sig}

    for i in range(N):
        if i in sw:
            t, v = sw[i]
            if t == 'H':
                hsps.append((i, v))
            else:
                lsps.append((i, v))

        if len(hsps) >= 2 and len(lsps) >= 2:
            hh[i] = 1.0 if hsps[-1][1] > hsps[-2][1] else 0.0
            hl[i] = 1.0 if lsps[-1][1] > lsps[-2][1] else 0.0

    return hh, hl


# ─────────────────────────────────────────────────────────────────────────────
# IC pre-screen helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic(feature, close, lag=1):
    """Pearson IC: correlation of feature[i] with sign(close[i+lag] - close[i]).
    Ignores NaN values. Returns (ic, n_valid).
    """
    N = len(close)
    target = np.where(close[lag:] > close[:N-lag], 1.0, -1.0)
    feat = feature[:N-lag]
    valid = np.isfinite(feat) & np.isfinite(target)
    if valid.sum() < 100:
        return np.nan, 0
    ic = np.corrcoef(feat[valid], target[valid])[0, 1]
    return float(ic), int(valid.sum())


def compute_corr(a, b):
    """Pearson correlation ignoring NaN."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute all swing features from M5 OHLC + ASI
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_swing_features(o, h, l, c, asi):
    """
    Compute all swing structure indicators from M5 OHLC and pre-computed ASI.

    Args:
        o, h, l, c : M5 OHLC arrays (float64)
        asi        : Wilder ASI array (float64), same length

    Returns dict with keys:
        sb_a_raw   : int8  {-2,-1,0,+1,+2} — ASI breakout state
        sb_a       : float {-1,-0.5,0,+0.5,+1} — normalized
        erp_a      : float unclamped (asi - lsp_asi) / (hsp_asi - lsp_asi)
        hh_asi     : float {0,1} — higher highs on ASI
        hl_asi     : float {0,1} — higher lows on ASI
        sb_p_raw   : int8  {-2,-1,0,+1,+2} — price breakout state
        sb_p       : float {-1,-0.5,0,+0.5,+1} — normalized
        erp_p      : float unclamped (close - lsp_price) / (hsp_price - lsp_price)
        hh_price   : float {0,1} — higher highs on price
        hl_price   : float {0,1} — higher lows on price
        act_h_a    : float — active ASI HSP level
        act_l_a    : float — active ASI LSP level
        act_h_p    : float — active price HSP level
        act_l_p    : float — active price LSP level
    """
    N = len(c)
    mid = (h + l) / 2.0

    # SB-A: TopsBots on ASI series (hi=lo=ASI)
    sb_a_raw, erp_a, act_h_a, act_l_a, sig_a = compute_swing_features(asi, asi, asi)
    hh_asi, hl_asi = compute_hhhl_binary(sig_a, N)

    # SB-P: TopsBots on price h[]/l[]
    sb_p_raw, erp_p, act_h_p, act_l_p, sig_p = compute_swing_features(h, l, mid)
    hh_price, hl_price = compute_hhhl_binary(sig_p, N)

    return {
        'sb_a_raw':  sb_a_raw,
        'sb_a':      state_to_normalized(sb_a_raw),
        'erp_a':     erp_a,
        'hh_asi':    hh_asi,
        'hl_asi':    hl_asi,
        'sb_p_raw':  sb_p_raw,
        'sb_p':      state_to_normalized(sb_p_raw),
        'erp_p':     erp_p,
        'hh_price':  hh_price,
        'hl_price':  hl_price,
        'act_h_a':   act_h_a,
        'act_l_a':   act_l_a,
        'act_h_p':   act_h_p,
        'act_l_p':   act_l_p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fill NaN forward for training (startup NaNs → last valid → 0 if none)
# ─────────────────────────────────────────────────────────────────────────────

def ffill_zero(arr):
    """Forward-fill NaN, then fill remaining with 0. Returns float64 copy."""
    out = arr.copy().astype(np.float64)
    last = 0.0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out
