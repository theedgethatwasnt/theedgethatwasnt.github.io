#!/usr/bin/env python3
"""
PantasticSMA multi-timeframe chart.

Subplots (shared x-axis):
  S5 / S30 / M5 / H1  — price line + colored SMA (red=rising, blue=falling, gray=neutral)
  6 pair agreement strips  — S5+S30, S5+M5, S5+H1, S30+M5, S30+H1, M5+H1
  ALL-4 strip              — all four TFs agree

WhiteMethod (price/min, timeframe-neutral): rate = Δsma / (lookback × bar_minutes)
Threshold: 0.003 price/min  (original cTrader default)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

ROOT = Path(__file__).parents[3]

SMA_P    = 7
LOOKBACK = 5
THR      = 0.003     # price per minute

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = '#0d1117'
PANEL_BG = '#161b22'
PRICE_C  = '#8b949e'
UP_C     = '#e74c3c'   # rising  (cTrader Red)
DN_C     = '#2e86de'   # falling (cTrader DodgerBlue)
NEU_C    = '#4a5568'   # neutral
AGR_UP   = '#22863a'   # both-up  — green
AGR_DN   = '#b91c1c'   # both-dn  — red
AGR_OPP  = '#d97706'   # opposing — amber

# ── Pantastic signal ──────────────────────────────────────────────────────────
def pantastic(prices: np.ndarray, bar_minutes: float):
    """Return (sma, signal) arrays. signal ∈ {-1, 0, 1}."""
    sma = pd.Series(prices).rolling(SMA_P, min_periods=SMA_P).mean().values
    sig = np.zeros(len(prices), dtype=np.int8)
    warm = SMA_P + LOOKBACK
    for i in range(warm, len(prices)):
        if np.isnan(sma[i]) or np.isnan(sma[i - LOOKBACK]):
            continue
        r = (sma[i] - sma[i - LOOKBACK]) / (LOOKBACK * bar_minutes)
        sig[i] = 1 if r > THR else (-1 if r < -THR else 0)
    return sma, sig


# ── Data loading — compute signals on FULL history, then window for display ────
print("Loading EUR_JPY S5 (full dataset for warmup) …")
s5_all = (pd.read_parquet(ROOT / "data" / "s5_ohlc" / "EUR_JPY_S5_BA.parquet")
            .sort_values("timestamp").reset_index(drop=True))
s5_all["mid_c"] = (s5_all.bid_c + s5_all.ask_c) / 2
print(f"  Full S5: {len(s5_all):,} bars  {s5_all.timestamp.iloc[0].date()}  →  {s5_all.timestamp.iloc[-1].date()}")

s5_all_i = s5_all.set_index("timestamp")

def resample_close(dfidx, freq):
    return dfidx[["mid_c"]].resample(freq).last().dropna().reset_index()

s30_all = resample_close(s5_all_i, "30s")
m5_all  = resample_close(s5_all_i, "5min")
h1_all  = resample_close(s5_all_i, "1h")
print(f"  Full resampled: S30={len(s30_all):,}  M5={len(m5_all):,}  H1={len(h1_all):,}")

# ── Compute Pantastic on each TF (full history) ────────────────────────────────
print("Computing Pantastic signals on full history …")
s5_sma_all,  s5_sig_all  = pantastic(s5_all.mid_c.values,   5 / 60)
s30_sma_all, s30_sig_all = pantastic(s30_all.mid_c.values,  0.5)
m5_sma_all,  m5_sig_all  = pantastic(m5_all.mid_c.values,   5.0)
h1_sma_all,  h1_sig_all  = pantastic(h1_all.mid_c.values,  60.0)

# Attach computed columns
s5_all["sma"]  = s5_sma_all;  s5_all["sig"]  = s5_sig_all
s30_all["sma"] = s30_sma_all; s30_all["sig"] = s30_sig_all
m5_all["sma"]  = m5_sma_all;  m5_all["sig"]  = m5_sig_all
h1_all["sma"]  = h1_sma_all;  h1_all["sig"]  = h1_sig_all

# ── Window for plotting (~4 days, Oct 6–9 2025 — best H1 volatility) ──────────
START = pd.Timestamp("2025-10-05 22:00:00+00:00")
END   = pd.Timestamp("2025-10-09 22:00:00+00:00")

def window(df, ts_col="timestamp"):
    mask = (df[ts_col] >= START) & (df[ts_col] < END)
    return df[mask].reset_index(drop=True)

s5  = window(s5_all)
s30 = window(s30_all)
m5  = window(m5_all)
h1  = window(h1_all)
print(f"  Windowed: S5={len(s5):,}  S30={len(s30):,}  M5={len(m5):,}  H1={len(h1):,}")

s5_sma,  s5_sig  = s5.sma.values,  s5.sig.values.astype(np.int8)
s30_sma, s30_sig = s30.sma.values, s30.sig.values.astype(np.int8)
m5_sma,  m5_sig  = m5.sma.values,  m5.sig.values.astype(np.int8)
h1_sma,  h1_sig  = h1.sma.values,  h1.sig.values.astype(np.int8)
print(f"  H1 signals in window: up={(h1_sig==1).sum()}  dn={(h1_sig==-1).sum()}  neu={(h1_sig==0).sum()}")


# ── Forward-fill higher-TF signals onto S5 grid ────────────────────────────────
def ff_to_s5(src_ts, src_sig, tgt_ts):
    src = pd.DataFrame({"ts": src_ts, "sig": src_sig.astype(float)})
    tgt = pd.DataFrame({"ts": tgt_ts})
    m   = pd.merge_asof(tgt, src, on="ts", direction="backward")
    return m["sig"].fillna(0).values.astype(np.int8)

ts5    = s5.timestamp
s5_ff  = s5_sig
s30_ff = ff_to_s5(s30.timestamp, s30_sig, ts5)
m5_ff  = ff_to_s5(m5.timestamp,  m5_sig,  ts5)
h1_ff  = ff_to_s5(h1.timestamp,  h1_sig,  ts5)


# ── Agreement panels ───────────────────────────────────────────────────────────
def agree(a, b):
    """1=both-up, -1=both-dn, 2=opposing, 0=one-neutral."""
    r = np.zeros(len(a), dtype=np.int8)
    r[(a == 1)  & (b == 1)]  = 1
    r[(a == -1) & (b == -1)] = -1
    r[(a == 1)  & (b == -1)] = 2
    r[(a == -1) & (b == 1)]  = 2
    return r

all4 = np.zeros(len(ts5), dtype=np.int8)
all4[(s5_ff == 1)  & (s30_ff == 1)  & (m5_ff == 1)  & (h1_ff == 1)]  = 1
all4[(s5_ff == -1) & (s30_ff == -1) & (m5_ff == -1) & (h1_ff == -1)] = -1

agr_panels = [
    ("S5 + S30",  agree(s5_ff,  s30_ff)),
    ("S5 + M5",   agree(s5_ff,  m5_ff)),
    ("S5 + H1",   agree(s5_ff,  h1_ff)),
    ("S30 + M5",  agree(s30_ff, m5_ff)),
    ("S30 + H1",  agree(s30_ff, h1_ff)),
    ("M5 + H1",   agree(m5_ff,  h1_ff)),
    ("ALL 4",     all4),
]


# ── Figure layout ──────────────────────────────────────────────────────────────
N_PRICE = 4
N_AGR   = len(agr_panels)
N_TOTAL = N_PRICE + N_AGR

fig = plt.figure(figsize=(20, N_PRICE * 3.0 + N_AGR * 1.1), facecolor=BG)
gs  = gridspec.GridSpec(N_TOTAL, 1, figure=fig,
                        hspace=0.04,
                        height_ratios=[4] * N_PRICE + [1.4] * N_AGR,
                        top=0.955, bottom=0.045, left=0.09, right=0.985)

axes = [fig.add_subplot(gs[i], facecolor=PANEL_BG) for i in range(N_TOTAL)]
for ax in axes[1:]:
    ax.sharex(axes[0])


# ── Colored SMA line via LineCollection ────────────────────────────────────────
def draw_sma(ax, ts, price, sma, sig):
    xdt  = pd.DatetimeIndex(ts).to_pydatetime()
    # faint raw-price underlay
    ax.plot(xdt, price, color=PRICE_C, linewidth=0.35, alpha=0.35, zorder=1)

    valid = ~np.isnan(sma)
    xv = mdates.date2num(xdt[valid])
    yv = sma[valid]
    sv = sig[valid]
    if len(xv) < 2:
        return

    pts  = np.c_[xv, yv].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = [UP_C if s == 1 else (DN_C if s == -1 else NEU_C) for s in sv[:-1]]

    lc = LineCollection(segs, colors=cols, linewidths=1.8, zorder=3)
    ax.add_collection(lc)

    pad = max((yv.max() - yv.min()) * 0.12, 0.05)
    ax.set_ylim(yv.min() - pad, yv.max() + pad)


price_layers = [
    ("S5",  s5.timestamp,  s5.mid_c.values,  s5_sma,  s5_sig),
    ("S30", s30.timestamp, s30.mid_c.values, s30_sma, s30_sig),
    ("M5",  m5.timestamp,  m5.mid_c.values,  m5_sma,  m5_sig),
    ("H1",  h1.timestamp,  h1.mid_c.values,  h1_sma,  h1_sig),
]

for i, (label, ts, price, sma, sig) in enumerate(price_layers):
    ax = axes[i]
    draw_sma(ax, ts, price, sma, sig)
    ax.set_ylabel(label, color="white", fontsize=10, fontweight="bold",
                  rotation=0, labelpad=38, va="center")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.tick_params(colors="#6e7681", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color("#21262d")
        sp.set_linewidth(0.6)
    # subtle horizontal gridlines
    ax.yaxis.grid(True, color="#21262d", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ── Agreement strips ───────────────────────────────────────────────────────────
xdt_s5 = pd.DatetimeIndex(ts5).to_pydatetime()

for i, (label, asig) in enumerate(agr_panels):
    ax = axes[N_PRICE + i]
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    ax.fill_between(xdt_s5, 0, 1, where=(asig == 1),  color=AGR_UP,  alpha=0.92, step="post")
    ax.fill_between(xdt_s5, 0, 1, where=(asig == -1), color=AGR_DN,  alpha=0.92, step="post")
    ax.fill_between(xdt_s5, 0, 1, where=(asig == 2),  color=AGR_OPP, alpha=0.75, step="post")

    ax.set_ylabel(label, color="#8b949e", fontsize=8.5, rotation=0,
                  labelpad=58, va="center")
    ax.tick_params(colors="#6e7681", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#21262d")
        sp.set_linewidth(0.6)


# ── X-axis — bottom subplot only ───────────────────────────────────────────────
for ax in axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(bottom=False)

ax_bot = axes[-1]
ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d\n%H:%M UTC"))
plt.setp(ax_bot.get_xticklabels(), color="#8b949e", fontsize=8)
ax_bot.tick_params(colors="#6e7681", labelsize=8)

# ── Title ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    "PantasticSMA · EUR/JPY · S5 / S30 / M5 / H1 "
    "· SMA(7)  lookback=5  thr=0.003 price/min",
    color="white", fontsize=13, fontweight="bold", y=0.972,
)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    Patch(color=UP_C,   label="Rising  (SMA momentum ↑)"),
    Patch(color=DN_C,   label="Falling (SMA momentum ↓)"),
    Patch(color=NEU_C,  label="Neutral"),
    Patch(color=AGR_UP, label="Agree ↑ (both TFs rising)"),
    Patch(color=AGR_DN, label="Agree ↓ (both TFs falling)"),
    Patch(color=AGR_OPP,label="Opposing (TFs conflict)"),
]
fig.legend(
    handles=legend_handles,
    loc="upper right", bbox_to_anchor=(0.985, 0.955),
    facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
    fontsize=8.5, framealpha=0.92, ncol=3, columnspacing=1.0,
    handlelength=1.2,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "pantastic_mtf_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
plt.close(fig)
