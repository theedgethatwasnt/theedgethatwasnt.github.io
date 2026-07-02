"""
Inline numba indicator computations for the CMA-NN encyclopedia extension.

These are indicators NOT present in data/unified_indicators/*.parquet that we
want to test in greedy Phase D. All take M5 OHLC and produce a float64 array
of the same length, normalized to roughly [-1,+1] or [0,1].
"""
import numpy as np
from numba import njit


# ── Wilder RSI 14 ──────────────────────────────────────────────────────
@njit(cache=True)
def compute_rsi_14(close):
    n = len(close)
    out = np.full(n, 0.5)
    if n < 15:
        return out
    period = 14
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = close[i] - close[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    if avg_l == 0:
        out[period] = 1.0
    else:
        rs = avg_g / avg_l
        out[period] = 1.0 - 1.0 / (1.0 + rs)
    for i in range(period + 1, n):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_g = (avg_g * (period - 1) + gain) / period
        avg_l = (avg_l * (period - 1) + loss) / period
        if avg_l == 0:
            out[i] = 1.0
        else:
            rs = avg_g / avg_l
            out[i] = 1.0 - 1.0 / (1.0 + rs)
    # Map [0,1] → [-1,+1] (centered RSI)
    for i in range(n):
        out[i] = (out[i] - 0.5) * 2.0
    return out


# ── EMA ratio (close / EMA - 1, tanh-normalized) ───────────────────────
@njit(cache=True)
def compute_ema_ratio(close, period):
    n = len(close)
    out = np.zeros(n)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    ema = close[0]
    out[0] = 0.0
    for i in range(1, n):
        ema = alpha * close[i] + (1 - alpha) * ema
        if ema > 0:
            r = close[i] / ema - 1.0
            # Tanh-normalize: typical FX bar moves ~0.001, scale x100
            out[i] = np.tanh(r * 100.0)
    return out


# ── ATR(14) / close ratio (volatility measure) ─────────────────────────
@njit(cache=True)
def compute_atr_ratio(high, low, close):
    n = len(close)
    out = np.zeros(n)
    if n < 15:
        return out
    period = 14
    tr_sum = 0.0
    for i in range(1, period + 1):
        h = high[i]
        l = low[i]
        c1 = close[i - 1]
        tr = max(h - l, max(abs(h - c1), abs(l - c1)))
        tr_sum += tr
    atr = tr_sum / period
    if close[period] > 0:
        out[period] = np.tanh(atr / close[period] * 1000.0)
    for i in range(period + 1, n):
        h = high[i]
        l = low[i]
        c1 = close[i - 1]
        tr = max(h - l, max(abs(h - c1), abs(l - c1)))
        atr = (atr * (period - 1) + tr) / period
        if close[i] > 0:
            out[i] = np.tanh(atr / close[i] * 1000.0)
    return out


# ── Two-bar momentum: tanh((close[t] - close[t-2]) / atr) ──────────────
@njit(cache=True)
def compute_two_bar_momentum(close):
    n = len(close)
    out = np.zeros(n)
    for i in range(2, n):
        if close[i - 2] > 0:
            r = (close[i] - close[i - 2]) / close[i - 2]
            out[i] = np.tanh(r * 500.0)
    return out


# ── Rate of change 10 bars ─────────────────────────────────────────────
@njit(cache=True)
def compute_roc_10(close):
    n = len(close)
    out = np.zeros(n)
    for i in range(10, n):
        if close[i - 10] > 0:
            r = (close[i] - close[i - 10]) / close[i - 10]
            out[i] = np.tanh(r * 200.0)
    return out


# ── Donchian position (20-bar): (c - lo20) / (hi20 - lo20) → [0,1] ────
@njit(cache=True)
def compute_donchian_pos(high, low, close, period=20):
    n = len(close)
    out = np.full(n, 0.5)
    for i in range(period, n):
        hi = high[i - period + 1]
        lo = low[i - period + 1]
        for k in range(i - period + 2, i + 1):
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]
        rng = hi - lo
        if rng > 0:
            out[i] = (close[i] - lo) / rng
    return out


# ── Bollinger position (20-bar): (c - sma20) / (2*std20) → ~[-1,+1] ───
@njit(cache=True)
def compute_bb_pos(close, period=20):
    n = len(close)
    out = np.zeros(n)
    for i in range(period, n):
        s = 0.0
        for k in range(i - period + 1, i + 1):
            s += close[k]
        mean = s / period
        var = 0.0
        for k in range(i - period + 1, i + 1):
            d = close[k] - mean
            var += d * d
        std = np.sqrt(var / period)
        if std > 0:
            out[i] = np.tanh((close[i] - mean) / (2.0 * std))
    return out


# ── Stochastic %K (14-bar): (c - lo14) / (hi14 - lo14) → [0,1] ────────
@njit(cache=True)
def compute_stoch_k(high, low, close, period=14):
    n = len(close)
    out = np.full(n, 0.5)
    for i in range(period, n):
        hi = high[i - period + 1]
        lo = low[i - period + 1]
        for k in range(i - period + 2, i + 1):
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]
        rng = hi - lo
        if rng > 0:
            out[i] = (close[i] - lo) / rng
    return out


# ── Range expansion: current TR / avg TR(20) ──────────────────────────
@njit(cache=True)
def compute_range_expansion(high, low, close, period=20):
    n = len(close)
    out = np.zeros(n)
    if n < period + 2:
        return out
    trs = np.zeros(n)
    for i in range(1, n):
        trs[i] = max(high[i] - low[i],
                     max(abs(high[i] - close[i - 1]),
                         abs(low[i] - close[i - 1])))
    for i in range(period + 1, n):
        s = 0.0
        for k in range(i - period + 1, i + 1):
            s += trs[k]
        avg = s / period
        if avg > 0:
            out[i] = np.tanh(trs[i] / avg - 1.0)
    return out


# ── Body ratio: |c - o| / (h - l) → [0,1] ─────────────────────────────
@njit(cache=True)
def compute_body_ratio(o, h, l, c):
    n = len(c)
    out = np.zeros(n)
    for i in range(n):
        rng = h[i] - l[i]
        if rng > 0:
            out[i] = abs(c[i] - o[i]) / rng
    return out


# ── Aroon oscillator at H1 cadence (25 H1 bars = 25h lookback) ────────
@njit(cache=True)
def compute_aroon_osc_h1(high, low, bar_period=12, aroon_period=25):
    """Aroon oscillator computed on H1-spaced samples from M5 OHLC.

    At each M5 bar i, looks at the last `aroon_period` H1-spaced highs/lows
    (positions i, i-12, i-24, ..., i-12*(aroon_period-1)). The H1 high is
    max(M5 high over the 12-bar window); H1 low is min over the window.

    Output range: [-1, +1] (Aroon up - Aroon down) / 100.
    """
    n = len(high)
    out = np.zeros(n)
    needed = bar_period * aroon_period
    if n < needed:
        return out
    for i in range(needed, n):
        # Build the H1-cadence high/low windows by max/min over each 12-M5 chunk
        max_h_idx = -1
        max_h_val = -1e18
        min_l_idx = -1
        min_l_val = 1e18
        # H1 bar k corresponds to M5 bars [i - bar_period*(k+1) + 1 .. i - bar_period*k]
        for k in range(aroon_period):
            end = i - bar_period * k
            start = end - bar_period + 1
            if start < 0:
                start = 0
            local_h = high[start]
            local_l = low[start]
            for m in range(start + 1, end + 1):
                if high[m] > local_h:
                    local_h = high[m]
                if low[m] < local_l:
                    local_l = low[m]
            if local_h > max_h_val:
                max_h_val = local_h
                max_h_idx = k    # k=0 is most recent H1 bar
            if local_l < min_l_val:
                min_l_val = local_l
                min_l_idx = k
        # Aroon up: % closer to most recent the highest high is
        # bars_since_high in H1 units = max_h_idx
        aroon_up = (aroon_period - max_h_idx) / aroon_period
        aroon_dn = (aroon_period - min_l_idx) / aroon_period
        out[i] = aroon_up - aroon_dn   # already in [-1, +1]
    return out


# ── ADX 14 (Wilder) — directional movement strength ──────────────────
@njit(cache=True)
def compute_adx_14(high, low, close):
    n = len(close)
    out = np.zeros(n)
    if n < 30:
        return out
    period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                    max(abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1])))
    tr_smooth = 0.0
    pdm_smooth = 0.0
    mdm_smooth = 0.0
    for i in range(1, period + 1):
        tr_smooth += tr[i]
        pdm_smooth += plus_dm[i]
        mdm_smooth += minus_dm[i]
    dx_sum = 0.0
    for i in range(period + 1, 2 * period + 1):
        tr_smooth = tr_smooth - tr_smooth / period + tr[i]
        pdm_smooth = pdm_smooth - pdm_smooth / period + plus_dm[i]
        mdm_smooth = mdm_smooth - mdm_smooth / period + minus_dm[i]
        if tr_smooth > 0:
            pdi = pdm_smooth / tr_smooth
            mdi = mdm_smooth / tr_smooth
            denom = pdi + mdi
            if denom > 0:
                dx_sum += abs(pdi - mdi) / denom
    adx = dx_sum / period
    out[2 * period] = adx
    for i in range(2 * period + 1, n):
        tr_smooth = tr_smooth - tr_smooth / period + tr[i]
        pdm_smooth = pdm_smooth - pdm_smooth / period + plus_dm[i]
        mdm_smooth = mdm_smooth - mdm_smooth / period + minus_dm[i]
        if tr_smooth > 0:
            pdi = pdm_smooth / tr_smooth
            mdi = mdm_smooth / tr_smooth
            denom = pdi + mdi
            dx = abs(pdi - mdi) / denom if denom > 0 else 0.0
            adx = (adx * (period - 1) + dx) / period
        out[i] = adx
    # Already in [0,1]
    return out


# ── Dispatcher: name → compute function ────────────────────────────────
EXTRA_COMPUTE = {
    "rsi_14":             ("close",   lambda o, h, l, c: compute_rsi_14(c)),
    "ema8_ratio":         ("close",   lambda o, h, l, c: compute_ema_ratio(c, 8)),
    "ema21_ratio":        ("close",   lambda o, h, l, c: compute_ema_ratio(c, 21)),
    "atr_ratio":          ("ohlc",    lambda o, h, l, c: compute_atr_ratio(h, l, c)),
    "two_bar_momentum":   ("close",   lambda o, h, l, c: compute_two_bar_momentum(c)),
    "roc_10":             ("close",   lambda o, h, l, c: compute_roc_10(c)),
    "donchian_pos":       ("ohlc",    lambda o, h, l, c: compute_donchian_pos(h, l, c)),
    "bb_pos":             ("close",   lambda o, h, l, c: compute_bb_pos(c)),
    "stoch_k":            ("ohlc",    lambda o, h, l, c: compute_stoch_k(h, l, c)),
    "range_expansion":    ("ohlc",    lambda o, h, l, c: compute_range_expansion(h, l, c)),
    "body_ratio":         ("ohlc",    lambda o, h, l, c: compute_body_ratio(o, h, l, c)),
    "adx_14":             ("ohlc",    lambda o, h, l, c: compute_adx_14(h, l, c)),
    "aroon_osc_h1":       ("ohlc",    lambda o, h, l, c: compute_aroon_osc_h1(h, l)),
}


def is_inline_computable(name: str) -> bool:
    return name in EXTRA_COMPUTE


def compute_inline(name: str, o, h, l, c):
    return EXTRA_COMPUTE[name][1](o, h, l, c)
