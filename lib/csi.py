"""
CSI — Welles Wilder's Commodity Selection Index, adapted for FX.

CSI ranks *pairs* by how worth-trading they are right now:

    CSI = ADXR × ATR_pips × (V / √M) × (1 / (150 + C)) × 100

where
    ADXR     directional-movement rating  (trendiness, 0–100, direction-agnostic)
    ATR_pips average true range in pips    (volatility / opportunity size)
    V        USD value of a 1-pip move on a standard lot   (usd_per_pip)
    M        required margin for a standard lot             (margin efficiency)
    C        commission in USD ≈ spread × V                 (transaction cost)

Two entry points:
  - `wilder_csi_from_ohlc(...)` — the classic candle-based form (M5/H1/D).
  - `tick_csi(efficiency, range_pips, V, M, C)` — a live tick-window analog that
    replaces ADXR with a directional-efficiency ratio and ATR with the window's
    pip range (ADX14 needs ~28 bars, which a 5–30 s tick window cannot form).

`compute_econ` derives V and M per pair from OANDA instrument metadata + a price
snapshot (ported from the batch section of oandacsi.py).
"""
import math


def compute_econ(instruments, prices, lot=100_000):
    """V (usd_per_pip) and M (required margin) per instrument.

    instruments: iterable with .name, .pipLocation, .marginRate
    prices: {instrument_name: bid_price_float}
    Returns {name: {"V": float, "M": float}} (pairs with un-derivable V skipped).
    """
    out = {}
    for inst in instruments:
        name = inst.name
        if name not in prices:
            continue
        bid = prices[name]
        if not bid or bid <= 0:
            continue
        base = name[:3]
        pip_loc = inst.pipLocation
        mrate = float(inst.marginRate)
        cross_per_pip = lot * (10 ** pip_loc)
        if name.endswith("USD"):
            V = cross_per_pip
            M = mrate * lot * bid
        else:
            base_per_pip = cross_per_pip / bid
            rate = prices.get(base + "_USD")
            if rate is None:
                u = prices.get("USD_" + base)
                rate = (1.0 / u) if u else None
            V = (base_per_pip * rate) if rate else None
            u = prices.get("USD_" + base)
            M = (mrate * lot / u) if u else (mrate * lot)
        if V is None or V <= 0 or M is None or M <= 0:
            continue
        out[name] = {"V": float(V), "M": float(M)}
    return out


def econ_factor(V, M, C):
    """The margin/cost factor (V/√M)·1/(150+C). Direction-agnostic scalar."""
    if M <= 0:
        return 0.0
    return (V / math.sqrt(M)) * (1.0 / (150.0 + C))


def tick_csi(efficiency, range_pips, V, M, C, scale=100.0):
    """Live tick-window CSI analog.

    efficiency  |net move| / Σ|tick-to-tick move|  ∈ [0,1]  (≈ ADXR/100)
    range_pips  (window high − low) / pip                   (≈ ATR_pips)
    """
    return efficiency * range_pips * econ_factor(V, M, C) * scale


def _wilder_adxr_atr(o, h, l, c, period=14):
    """Return (ADXR, ATR) on the last bar from OHLC arrays, or (nan, nan)."""
    import numpy as np
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    n = len(c)
    if n < 2 * period + 2:
        return float("nan"), float("nan")
    pc = np.roll(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    up = h - np.roll(h, 1)
    dn = np.roll(l, 1) - l
    pDM = np.where((up > dn) & (up > 0), up, 0.0)
    nDM = np.where((dn > up) & (dn > 0), dn, 0.0)

    def _wsum(x):
        s = np.full(n, np.nan)
        if n >= period:
            s[period - 1] = x[:period].sum()
            for i in range(period, n):
                s[i] = s[i - 1] - s[i - 1] / period + x[i]
        return s

    tr14, pdm14, ndm14 = _wsum(tr), _wsum(pDM), _wsum(nDM)
    with np.errstate(divide="ignore", invalid="ignore"):
        pDI = 100.0 * pdm14 / tr14
        nDI = 100.0 * ndm14 / tr14
        dx = 100.0 * np.abs(pDI - nDI) / (pDI + nDI)
    adx = np.full(n, np.nan)
    valid = np.where(~np.isnan(dx))[0]
    if len(valid) >= period:
        start = valid[period - 1]
        adx[start] = np.nanmean(dx[valid[0]:valid[period - 1] + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    adxr = (adx + np.roll(adx, period)) / 2.0
    atr = tr14 / period
    return float(adxr[-1]), float(atr[-1])


def wilder_csi_from_ohlc(o, h, l, c, pip_loc, V, M, C, scale=100.0):
    """Classic candle-based CSI on the last bar. Returns float or nan."""
    adxr, atr = _wilder_adxr_atr(o, h, l, c)
    if adxr != adxr or atr != atr:   # nan
        return float("nan")
    atr_pips = atr / (10 ** pip_loc)
    return adxr * atr_pips * econ_factor(V, M, C) * scale
