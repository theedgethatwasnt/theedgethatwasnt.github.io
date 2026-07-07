#!/usr/bin/env python3
"""Generate the glossary's per-term indicator charts that no existing
author plate covered.

These complement the encyclopedia plates copied in from the book's figure
set (ATR, RSI, MACD, PSAR, ASI, Bollinger/KAMA, Aroon, Vortex, ZigZag,
efficiency-ratio, squeeze, combo-geometric, log-return-deltas, cyclic
encoding, realized-vol, AMDDP). Everything here is drawn from the small,
committed EUR/USD (and 4-pair) M5 samples in figures/glossary/ so the site
is reproducible without the fx-core data repo.

    python3 build_term_charts.py

Writes color PNGs into figures/glossary/ (gen_*.png). Style mirrors the
book's encyclopedia plates: cream ground, navy price, teal/red accents,
serif type. One figure can serve several related glossary terms; the
slug->figure map lives in build_glossary.py (CHART_MAP).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GLOSS = HERE / "figures" / "glossary"

# ---- palette (matches the book encyclopedia plates) ------------------------
BG    = "#faf6ee"
NAVY  = "#1e2a44"
TEAL  = "#2f8f7f"
RED   = "#9e2f2f"
GOLD  = "#b5793a"
BLUE  = "#37599a"
GREY  = "#8a857c"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelsize": 10.5, "axes.labelcolor": NAVY,
    "axes.edgecolor": "#cfc8ba", "axes.linewidth": 0.8,
    "xtick.color": NAVY, "ytick.color": NAVY,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e7e0d2", "grid.alpha": 0.7,
    "legend.fontsize": 9, "legend.frameon": True,
    "legend.facecolor": BG, "legend.edgecolor": "#cfc8ba",
})

PIP = 0.0001

# ---- data ------------------------------------------------------------------
df = pd.read_csv(GLOSS / "sample_eurusd_m5.csv")
O = df["open"].to_numpy(float)
H = df["high"].to_numpy(float)
L = df["low"].to_numpy(float)
C = df["close"].to_numpy(float)
N = len(C)


# ---- pure-numpy indicator library -----------------------------------------
def ema(src, p):
    a = 2 / (p + 1)
    r = np.empty_like(src, float)
    r[0] = src[0]
    for i in range(1, len(src)):
        r[i] = a * src[i] + (1 - a) * r[i - 1]
    return r


def sma(src, p):
    r = np.full(len(src), np.nan)
    cs = np.cumsum(np.insert(src, 0, 0.0))
    r[p - 1:] = (cs[p:] - cs[:-p]) / p
    return r


def wilder(src, p):
    r = np.full(len(src), np.nan)
    if len(src) <= p:
        return r
    r[p] = np.nanmean(src[1:p + 1])
    for i in range(p + 1, len(src)):
        r[i] = (r[i - 1] * (p - 1) + src[i]) / p
    return r


def true_range():
    r = np.empty(N)
    r[0] = H[0] - L[0]
    for i in range(1, N):
        r[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    return r


def atr(p=14):
    return wilder(true_range(), p)


def bollinger(p=20, m=2):
    mid = sma(C, p)
    std = pd.Series(C).rolling(p).std(ddof=0).to_numpy()
    return mid, mid + m * std, mid - m * std


def donchian(p=40):
    up = pd.Series(H).rolling(p).max().to_numpy()
    lo = pd.Series(L).rolling(p).min().to_numpy()
    return up, lo


def adx(p=14):
    tr = true_range()
    up = np.diff(H, prepend=H[0])
    dn = -np.diff(L, prepend=L[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_ = wilder(tr, p)
    sp, sm = wilder(pdm, p), wilder(mdm, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * sp / atr_
        mdi = 100 * sm / atr_
        s = pdi + mdi
        dx = np.where(s > 0, 100 * np.abs(pdi - mdi) / s, 0.0)
    a = wilder(dx, p)
    a[:p * 2] = np.nan
    return a, pdi, mdi


def asi():
    out = np.zeros(N)
    for i in range(1, N):
        c1, c2 = C[i - 1], C[i]
        h2, l2 = H[i], L[i]
        o1 = O[i - 1]
        K = max(abs(h2 - c1), abs(l2 - c1))
        T = max(abs(h2 - c1), abs(l2 - c1), 1e-9)
        R = max(abs(h2 - c1), abs(l2 - c1), abs(h2 - l2)) + 0.25 * abs(c1 - o1)
        if R == 0:
            R = 1e-9
        si = 50 * ((c2 - c1) + 0.5 * (c2 - O[i]) + 0.25 * (c1 - o1)) / R * K / T
        out[i] = out[i - 1] + si
    return out


def supertrend(period=10, mult=3.0):
    a = atr(period)
    hl2 = (H + L) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = np.full(N, np.nan)
    dir_ = np.ones(N)  # 1 = up (support below), -1 = down
    for i in range(period + 1, N):
        if np.isnan(st[i - 1]):
            st[i] = lower[i]
            dir_[i] = 1
            continue
        if dir_[i - 1] == 1:
            lower[i] = max(lower[i], st[i - 1]) if C[i - 1] > st[i - 1] else lower[i]
            if C[i] < lower[i]:
                dir_[i] = -1
                st[i] = upper[i]
            else:
                dir_[i] = 1
                st[i] = lower[i]
        else:
            upper[i] = min(upper[i], st[i - 1]) if C[i - 1] < st[i - 1] else upper[i]
            if C[i] > upper[i]:
                dir_[i] = 1
                st[i] = lower[i]
            else:
                dir_[i] = -1
                st[i] = upper[i]
    return st, dir_


def kalman(q=1e-6, r=2e-4):
    """Scalar random-walk Kalman level smoother."""
    xhat = np.empty(N)
    p = 1.0
    xhat[0] = C[0]
    for i in range(1, N):
        xpred = xhat[i - 1]
        ppred = p + q
        k = ppred / (ppred + r)
        xhat[i] = xpred + k * (C[i] - xpred)
        p = (1 - k) * ppred
    return xhat


def efficiency_ratio(p=10):
    out = np.full(N, np.nan)
    for i in range(p, N):
        change = abs(C[i] - C[i - p])
        vol = np.sum(np.abs(np.diff(C[i - p:i + 1])))
        out[i] = change / vol if vol > 0 else 0.0
    return out


def trix(p=15):
    e1 = ema(np.log(C), p)
    e2 = ema(e1, p)
    e3 = ema(e2, p)
    t = np.full(N, np.nan)
    t[1:] = (e3[1:] - e3[:-1]) * 10000
    return t


def fisher(p=10):
    out = np.zeros(N)
    val = np.zeros(N)
    med = (H + L) / 2
    for i in range(p, N):
        hh = med[i - p + 1:i + 1].max()
        ll = med[i - p + 1:i + 1].min()
        rng = max(hh - ll, 1e-9)
        val[i] = 0.66 * (2 * (med[i] - ll) / rng - 0.5) + 0.34 * val[i - 1]
        v = min(max(val[i], -0.999), 0.999)
        out[i] = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * out[i - 1]
    return out


def williams_r(p=14):
    out = np.full(N, np.nan)
    for i in range(p - 1, N):
        hh = H[i - p + 1:i + 1].max()
        ll = L[i - p + 1:i + 1].min()
        rng = max(hh - ll, 1e-9)
        out[i] = -100 * (hh - C[i]) / rng
    return out


def rolling_hurst(win=128):
    lr = np.diff(np.log(C), prepend=np.log(C[0]))
    out = np.full(N, np.nan)
    lags = np.array([2, 4, 8, 16, 32])
    for i in range(win, N):
        w = lr[i - win:i]
        tau = []
        for lag in lags:
            d = w[lag:] - w[:-lag]
            tau.append(np.sqrt(np.mean(d * d)) + 1e-12)
        h = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        out[i] = h
    return out


def rolling_vr(win=120, k=4):
    lr = np.diff(np.log(C), prepend=np.log(C[0]))
    out = np.full(N, np.nan)
    for i in range(win, N):
        w = lr[i - win:i]
        v1 = np.var(w)
        kk = np.convolve(w, np.ones(k), "valid")
        vk = np.var(kk)
        out[i] = vk / (k * v1) if v1 > 0 else np.nan
    return out


def pnf(box_pips=10, reversal=3):
    """Classic Point & Figure: alternating X (up) / O (down) columns, a new
    box added on each ≥1-box move, a new column on a ≥`reversal`-box reversal.
    Returns list of (kind, lo_box, hi_box) and the box size."""
    box = box_pips * PIP
    b = C / box
    cols = []
    kind = "X"
    lo = hi = int(np.floor(b[0]))
    for v in b[1:]:
        fv, cv = int(np.floor(v)), int(np.ceil(v))
        if kind == "X":
            if fv > hi:
                hi = fv
            elif cv <= hi - reversal:
                cols.append(("X", lo, hi))
                kind, hi, lo = "O", hi - 1, cv
        else:
            if cv < lo:
                lo = cv
            elif fv >= lo + reversal:
                cols.append(("O", lo, hi))
                kind, lo, hi = "X", lo + 1, fv
    cols.append((kind, lo, hi))
    return cols, box


# ---- plotting helpers ------------------------------------------------------
def finish(fig, name):
    fig.tight_layout()
    fig.savefig(GLOSS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def price_ax(ax, sl, label="EUR/USD"):
    x = np.arange(sl.stop - sl.start)
    ax.plot(x, C[sl], color=NAVY, lw=1.4, label=label)
    ax.set_xticks([])
    return x


# ===========================================================================
def fig_trend_overlays():
    sl = slice(240, N)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = price_ax(ax, sl)
    ax.plot(x, sma(C, 20)[sl], color=GOLD, lw=1.5, label="SMA(20)")
    ax.plot(x, ema(C, 20)[sl], color=TEAL, lw=1.5, ls="--", label="EMA(20)")
    mid, up, lo = bollinger(20, 2)
    ax.fill_between(x, lo[sl], up[sl], color=BLUE, alpha=0.10, label="Bollinger ±2σ(20)")
    ax.plot(x, up[sl], color=BLUE, lw=0.9, ls=":")
    ax.plot(x, lo[sl], color=BLUE, lw=0.9, ls=":")
    du, dl = donchian(40)
    ax.plot(x, du[sl], color=RED, lw=0.9, ls="--", label="Donchian(40)")
    ax.plot(x, dl[sl], color=RED, lw=0.9, ls="--")
    ax.set_title("Moving averages, Bollinger Bands & Donchian channel — trend overlays")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", ncol=2)
    finish(fig, "gen_trend_overlays.png")


def fig_adx_dmi():
    sl = slice(200, N)
    x = np.arange(sl.stop - sl.start)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.6]})
    a1.plot(x, C[sl], color=NAVY, lw=1.4)
    a1.set_title("ADX / DMI — trend strength (ADX) and direction (+DI vs −DI)")
    a1.set_ylabel("EUR/USD"); a1.set_xticks([])
    a, pdi, mdi = adx(14)
    a2.plot(x, a[sl], color=NAVY, lw=1.6, label="ADX(14)")
    a2.plot(x, pdi[sl], color=TEAL, lw=1.3, ls="--", label="+DI")
    a2.plot(x, mdi[sl], color=RED, lw=1.3, ls="-.", label="−DI")
    a2.axhline(25, color=GREY, lw=0.9, ls=":")
    a2.text(1, 26, "ADX > 25 = trending", color=GREY, fontsize=8)
    a2.set_ylabel("0–100"); a2.set_xticks([])
    a2.legend(loc="upper left", ncol=3)
    finish(fig, "gen_adx_dmi.png")


def fig_mc_confluence():
    sl = slice(240, N)
    x = np.arange(sl.stop - sl.start)
    a = asi()
    asm = sma(a, 5)
    # MC(D): multi-timeframe momentum confluence = mean sign of ASI-SMA slope
    # across several lookbacks (−1..+1). MC(dD): its bar-to-bar acceleration.
    scales = [5, 10, 20, 40]
    slopes = np.zeros((len(scales), N))
    for j, s in enumerate(scales):
        sh = np.full(N, np.nan)
        sh[s:] = asm[s:] - asm[:-s]
        slopes[j] = np.tanh(sh / (np.nanstd(sh) + 1e-12))
    mc_d = np.nanmean(slopes, axis=0)
    mc_dd = np.full(N, np.nan)
    mc_dd[1:] = mc_d[1:] - mc_d[:-1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.8]})
    a1.plot(x, C[sl], color=NAVY, lw=1.4, label="EUR/USD")
    a1b = a1.twinx()
    a1b.plot(x, asm[sl], color=GOLD, lw=1.4, label="ASI → SMA(5)")
    a1b.set_ylabel("ASI", color=GOLD); a1b.tick_params(axis="y", colors=GOLD)
    a1b.grid(False)
    a1.set_title("ASIMC / MTF Momentum Confluence — MC(D) & MC(dD)")
    a1.set_ylabel("EUR/USD"); a1.set_xticks([])
    a2.axhline(0, color=GREY, lw=0.9)
    a2.fill_between(x, 0, mc_d[sl], where=mc_d[sl] >= 0, color=TEAL, alpha=0.55,
                    label="MC(D) > 0  (multi-TF up-confluence)")
    a2.fill_between(x, 0, mc_d[sl], where=mc_d[sl] < 0, color=RED, alpha=0.5,
                    label="MC(D) < 0  (down-confluence)")
    a2.plot(x, mc_dd[sl] * 3, color=NAVY, lw=1.0, ls=":", label="MC(dD) ×3 (acceleration)")
    a2.set_ylabel("confluence (−1..+1)"); a2.set_xticks([])
    a2.legend(loc="upper left", ncol=1)
    finish(fig, "gen_mc_confluence.png")


def fig_pnf(box_pips=5, reversal=3):
    cols, box = pnf(box_pips=box_pips, reversal=reversal)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    cx = 0
    for kind, bot, top in cols:
        if top < bot:
            bot, top = top, bot
        ys = [rr * box for rr in range(bot, top + 1)]
        xs = [cx] * len(ys)
        if kind == "X":
            ax.scatter(xs, ys, marker="x", s=34, color=TEAL, linewidths=1.5)
        else:
            ax.scatter(xs, ys, marker="o", s=34, facecolors="none",
                       edgecolors=RED, linewidths=1.4)
        cx += 1
    ax.set_xlim(-1, cx)
    ax.set_title(f"Point & Figure (P&F) — X = up-column, O = down-column "
                 f"({box_pips}-pip box, {reversal}-box reversal)")
    ax.set_ylabel("EUR/USD")
    ax.set_xlabel("column (time collapses; only ≥1-box moves advance the chart)")
    ax.set_xticks([])
    ax.grid(True, axis="y")
    finish(fig, "gen_pnf.png")


def fig_supertrend():
    sl = slice(200, N)
    x = np.arange(sl.stop - sl.start)
    st, dir_ = supertrend(10, 3.0)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x, C[sl], color=NAVY, lw=1.3, label="EUR/USD")
    up = dir_[sl] == 1
    sts = st[sl]
    stu = np.where(up, sts, np.nan)
    std = np.where(~up, sts, np.nan)
    ax.plot(x, stu, color=TEAL, lw=1.8, label="SuperTrend — support (long)")
    ax.plot(x, std, color=RED, lw=1.8, label="SuperTrend — resistance (short)")
    flips = np.where(np.diff(dir_[sl]) != 0)[0]
    ax.scatter(flips, sts[flips], color=GOLD, s=22, zorder=5, label="flip (exit/reverse)")
    ax.set_title("SuperTrend — ATR-band ratchet trailing stop (flips on close through the band)")
    ax.set_ylabel("EUR/USD"); ax.set_xticks([])
    ax.legend(loc="upper left")
    finish(fig, "gen_supertrend.png")


def fig_kalman():
    sl = slice(300, N)
    x = np.arange(sl.stop - sl.start)
    kf = kalman()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
    a1.plot(x, C[sl], color=GREY, lw=0.9, label="EUR/USD (raw)")
    a1.plot(x, kf[sl], color=BLUE, lw=1.8, label="Kalman-filtered level")
    a1.set_title("Kalman filter — recursive noise-adaptive smoother (fair-value estimate)")
    a1.set_ylabel("EUR/USD"); a1.set_xticks([]); a1.legend(loc="upper left")
    resid = (C - kf) / PIP
    a2.axhline(0, color=GREY, lw=0.8)
    a2.fill_between(x, 0, resid[sl], color=GOLD, alpha=0.5)
    a2.set_ylabel("resid (pips)"); a2.set_xticks([])
    a2.set_title("Price − Kalman residual — mean-reverting stretch used as a signal")
    finish(fig, "gen_kalman.png")


def fig_currency_strength():
    s = pd.read_csv(GLOSS / "sample_strength_m5.csv")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    lr = {p: np.diff(np.log(s[p].to_numpy(float)), prepend=np.log(s[p].to_numpy(float)[0]))
          for p in pairs}
    # per-currency rolling-sum log return, USD from the quote side
    win = 60
    def roll(a):
        return pd.Series(a).rolling(win).sum().to_numpy()
    eur, gbp, aud = roll(lr["EUR_USD"]), roll(lr["GBP_USD"]), roll(lr["AUD_USD"])
    usdjpy = roll(lr["USD_JPY"])
    # USD strength ≈ −(EUR/USD, GBP/USD, AUD/USD moves) + USD/JPY move
    usd = (-(eur + gbp + aud) + usdjpy) / 4
    jpy = -usdjpy - usd  # JPY weakens when USD/JPY up
    def z(a):
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)
    zeur, zusd, zgbp, zjpy = z(eur), z(usd), z(gbp), z(jpy)
    m = len(s)
    sl = slice(win + 60, m)
    x = np.arange(sl.stop - sl.start)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.4]})
    a1.axhline(0, color=GREY, lw=0.8)
    a1.plot(x, zusd[sl], color=NAVY, lw=1.6, label="USD")
    a1.plot(x, zeur[sl], color=TEAL, lw=1.4, label="EUR")
    a1.plot(x, zgbp[sl], color=GOLD, lw=1.4, label="GBP")
    a1.plot(x, zjpy[sl], color=RED, lw=1.4, label="JPY")
    a1.set_title("Currency Strength Index (CSI) — z-scored rolling returns per currency")
    a1.set_ylabel("strength z"); a1.set_xticks([]); a1.legend(loc="upper left", ncol=4)
    spread = zeur[sl] - zusd[sl]
    a2.axhline(0, color=GREY, lw=0.8)
    a2.fill_between(x, 0, spread, where=spread >= 0, color=TEAL, alpha=0.5)
    a2.fill_between(x, 0, spread, where=spread < 0, color=RED, alpha=0.5)
    a2.set_title("StrengthSpread — z(EUR) − z(USD): the tradeable strong-minus-weak gap")
    a2.set_ylabel("EUR − USD"); a2.set_xticks([])
    finish(fig, "gen_currency_strength.png")


def fig_oscillators():
    sl = slice(240, N)
    x = np.arange(sl.stop - sl.start)
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.4]})
    axes[0].plot(x, C[sl], color=NAVY, lw=1.3)
    axes[0].set_title("Screened oscillators — TRIX, Fisher Transform, Williams %R")
    axes[0].set_ylabel("EUR/USD"); axes[0].set_xticks([])
    tr = trix(15)
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].plot(x, tr[sl], color=GOLD, lw=1.4)
    axes[1].set_ylabel("TRIX")
    axes[1].set_title("TRIX(15) — triple-smoothed log-price rate of change")
    fi = fisher(10)
    axes[2].axhline(0, color=GREY, lw=0.8)
    axes[2].plot(x, fi[sl], color=BLUE, lw=1.4)
    axes[2].set_ylabel("Fisher")
    axes[2].set_title("Fisher Transform(10) — Gaussianized price, sharpens turning points")
    wr = williams_r(14)
    axes[3].plot(x, wr[sl], color=TEAL, lw=1.4)
    axes[3].axhline(-20, color=GREY, lw=0.8, ls=":")
    axes[3].axhline(-80, color=GREY, lw=0.8, ls=":")
    axes[3].set_ylim(-100, 0)
    axes[3].set_ylabel("%R")
    axes[3].set_title("Williams %R(14) — overbought (−20) / oversold (−80)")
    for a in axes:
        a.set_xticks([])
    finish(fig, "gen_oscillators.png")


def fig_meanrev_stats():
    sl = slice(280, N)
    x = np.arange(sl.stop - sl.start)
    hu = rolling_hurst(128)
    vr = rolling_vr(120, 4)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4]})
    axes[0].plot(x, C[sl], color=NAVY, lw=1.3)
    axes[0].set_title("Mean-reversion diagnostics — Hurst exponent & Variance Ratio")
    axes[0].set_ylabel("EUR/USD"); axes[0].set_xticks([])
    axes[1].axhline(0.5, color=GREY, lw=0.9, ls=":")
    axes[1].plot(x, hu[sl], color=BLUE, lw=1.4)
    axes[1].text(1, 0.52, "0.5 = random walk; <0.5 mean-revert; >0.5 trend", color=GREY, fontsize=8)
    axes[1].set_ylabel("Hurst H"); axes[1].set_xticks([])
    axes[2].axhline(1.0, color=GREY, lw=0.9, ls=":")
    axes[2].plot(x, vr[sl], color=GOLD, lw=1.4)
    axes[2].text(1, 1.02, "VR<1 = mean-reverting; VR>1 = trending", color=GREY, fontsize=8)
    axes[2].set_ylabel("VR(4)"); axes[2].set_xticks([])
    finish(fig, "gen_meanrev_stats.png")


def fig_session_levels():
    sl = slice(0, N)
    ts = pd.to_datetime(df["timestamp"])
    hod = ts.dt.hour.to_numpy()
    day = ts.dt.date.to_numpy()
    x = np.arange(N)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(x, C, color=NAVY, lw=1.2, label="EUR/USD")
    # Asian session high/low (00–07 UTC) drawn forward into the day
    uniq = sorted(set(day))
    shown = {"asia_h": False, "pdh": False}
    for d in uniq:
        dmask = day == d
        asia = dmask & (hod >= 0) & (hod < 7)
        if asia.sum() > 3:
            ah, al = H[asia].max(), L[asia].min()
            fwd = dmask & (hod >= 7)
            if fwd.sum() > 0:
                xs = x[fwd]
                ax.hlines([ah, al], xs.min(), xs.max(), color=GOLD, lw=1.1, ls="--",
                          label=("Asian session high/low" if not shown["asia_h"] else None))
                shown["asia_h"] = True
        # prior-day high/low
        idx = uniq.index(d)
        if idx > 0:
            pmask = day == uniq[idx - 1]
            pdh, pdl = H[pmask].max(), L[pmask].min()
            xs = x[dmask]
            ax.hlines([pdh, pdl], xs.min(), xs.max(), color=RED, lw=1.0, ls=":",
                      label=("Prior-day high/low (PDH/PDL)" if not shown["pdh"] else None))
            shown["pdh"] = True
    ax.set_title("Session & pivot levels — Asian range, prior-day high/low, session pivots")
    ax.set_ylabel("EUR/USD"); ax.set_xticks([])
    ax.legend(loc="upper left")
    finish(fig, "gen_session_levels.png")


def main():
    GLOSS.mkdir(parents=True, exist_ok=True)
    fig_trend_overlays()
    fig_adx_dmi()
    fig_mc_confluence()
    fig_pnf()
    fig_supertrend()
    fig_kalman()
    fig_currency_strength()
    fig_oscillators()
    fig_meanrev_stats()
    fig_session_levels()
    print("done.")


if __name__ == "__main__":
    main()
