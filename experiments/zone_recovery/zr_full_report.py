"""
ZR Full Report — All Zone Recovery / Ping-Pong Variants
Generates a multi-page PDF covering every experiment from Phase 1 classic cBot
through P&F-timed sweep and permutation validation.

Usage:
    python3 zr_full_report.py
Output:
    zr_full_report.pdf
"""

import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import os, sys, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, "results")
OUT  = os.path.join(BASE, "zr_full_report.pdf")

# ── colour palette ────────────────────────────────────────────────────────────
C_RED    = "#e74c3c"
C_YEL    = "#f39c12"
C_GRN    = "#27ae60"
C_BLU    = "#2980b9"
C_PUR    = "#8e44ad"
C_GREY   = "#7f8c8d"
C_DARK   = "#2c3e50"
C_BG     = "#f8f9fa"

def clr(gates):
    if gates >= 3: return C_GRN
    if gates >= 2: return C_YEL
    return C_RED

def fig_style(fig):
    fig.patch.set_facecolor(C_BG)

def ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:   ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK, pad=7)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10, color=C_GREY)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10, color=C_GREY)
    ax.tick_params(colors=C_GREY, labelsize=9)

# ── data loading ──────────────────────────────────────────────────────────────
def load(name, sub=None):
    path = os.path.join(sub or BASE, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

p1  = load("phase1_classic_grid.csv",         RES)
p2  = load("phase2_atr_grid.csv",             RES)
p5  = load("phase5_sweep.csv",                RES)
p6  = load("phase6_ez_sweep.csv",             RES)
p7  = load("phase7_boundary_vs_center.csv",   RES)
p8  = load("phase8_multipair_boundary.csv",   RES)
p9  = load("phase9_multipair_breakeven.csv",  RES)
trail  = load("zr_trail_sweep_results.csv")
signal = load("zr_signal_entry_results.csv")
perp   = load("zr_perpair_isoos_results.csv")
pnf    = load("zr_pnf_sweep_results.csv")
perm   = load("zr_pnf_permtest_results.csv")

# Hardcoded summary stats for variants without per-config CSVs
# (from JOURNEY-README.md; OOS 70/30 split, break-even sizing PF=1.25)
H1_PAIRS = dict(
    GBP_JPY=72360, USD_JPY=43142, CHF_JPY=60886, EUR_JPY=39671,
    AUD_JPY=29685, NZD_JPY=20965, CAD_JPY=68181, GBP_USD=39512,
    EUR_USD=31246, AUD_USD=13568, NZD_USD=16833, EUR_GBP=6267,
)
H4_PAIRS = dict(
    GBP_JPY=126237, USD_JPY=90598, CHF_JPY=127178, EUR_JPY=59815,
    AUD_JPY=49579, NZD_JPY=41023, CAD_JPY=61071, GBP_USD=80292,
    EUR_USD=69426, AUD_USD=44516, NZD_USD=36889, EUR_GBP=33198,
)
RANDOM_BASELINE = dict(
    GBP_JPY=28400, USD_JPY=3084,  CHF_JPY=27439, EUR_JPY=1977,
    AUD_JPY=-659987, NZD_JPY=3185, CAD_JPY=4851, GBP_USD=7148,
    EUR_USD=1828, AUD_USD=1286, NZD_USD=1489, EUR_GBP=620,
)

VARIANT_SUMMARY = [
    # label, phase, entry, sizing, gates, sharpe, sqn, oos_pnl_$, max_dd_pips, avg_legs, note
    ("Ph1: Classic cBot",     1, "Random",      "Dynamic",     0, -0.022, -2.9, -24278, 25981, 3.58, "E/Z=0.29 — all configs negative"),
    ("Ph2: ATR-grid best",    2, "Random",      "Dynamic",     2,  0.055,  7.34, 57624,  2649, 3.81, "E/Z≈8-10, 2 configs pass"),
    ("Ph4: Convex sizing",    4, "Random",      "Convex n^1.5",4,  0.128, 10.18, 462000,  2600, 3.8, "+3.8× vs dynamic"),
    ("Ph5: Large zone best",  5, "Random",      "Convex n^1.5",5,  0.356,  8.45, 26525,   0.0,  9.46, "hz=35 tgt=35"),
    ("Ph6: E/Z sweep best",   6, "Random",      "Convex n^1.5",5,  0.408,  9.70,  3767,   109, 9.84, "hz=35 E/Z=0.85"),
    ("Ph7b: Boundary zw=25",  7, "Random",      "Break-even",  5,  0.268, 17.90, 28400,    0.0, 1.50, "zw=25 tgt=50 PF=1.25"),
    ("Ph9: GBP_JPY live",     9, "Random",      "Break-even",  5,  0.031,  2.52, 16771,    0.0, 1.50, "zw=56 tgt=28 PF=1.25"),
    ("Trail: act=14 td=7",    10, "Random",     "Break-even",  4,  0.065,  3.10, 41495,    0.0, 1.16, "CHF_JPY +147% vs baseline"),
    ("H1 S/R directional",    11, "H1 TopsBots","Break-even",  5,  0.070,  5.20, 442317,   0.0, 1.08, "12-pair agg, tgt=0.25×ZW"),
    ("H4 S/R directional",    12, "H4 TopsBots","Break-even",  5,  0.095,  7.80, 819820,   0.0, 1.08, "12-pair agg — DEPLOYED"),
    ("Random N-bar (CHF_JPY)",13, "Random",      "Break-even",  3,  0.050,  2.80,  1080,    0.0, 1.44, "ta=5 td=3 N=1 — LIVE"),
    ("P&F alt (USD_JPY b5r2)",14, "P&F Reversal","Break-even",  1,  1.18,   0.0, 1335,    0.0, 0.0,   "perm p=0.006, boot p5<0"),
    ("P&F alt (NZD_JPY b5r4)",14, "P&F Reversal","Break-even",  3,  1.11,   0.0, 1207,    0.0, 0.0,   "3/3 gates"),
    ("P&F alt (CHF_JPY b10r3)",14,"P&F Reversal","Break-even",  3,  1.08,   0.0, 1113,    0.0, 0.0,   "3/3 gates"),
]

# ── page helpers ──────────────────────────────────────────────────────────────
def page_title(pdf, title, subtitle=""):
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.72), 0.90, 0.22,
                  boxstyle="round,pad=0.01", fc="#2c3e50", ec="none"))
    ax.text(0.5, 0.845, title, ha="center", va="center",
            fontsize=24, fontweight="bold", color="white")
    if subtitle:
        ax.text(0.5, 0.785, subtitle, ha="center", va="center",
                fontsize=13, color="#bdc3c7")
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
def p_title():
    fig, ax = page_title(
        None,
        "Zone Recovery / Ping-Pong — Complete Experiment Report",
        "All variants from classic cBot (Phase 1) through P&F-timed ZR · May 2026"
    )
    ax.text(0.5, 0.66, "14 Experiment Families  ·  12 Currency Pairs  ·  5+ Years OOS Data",
            ha="center", va="center", fontsize=11, color=C_GREY)

    legend_items = [
        ("🔴  0-1 gates (FAILED)", C_RED),
        ("🟡  2 gates (MARGINAL)", C_YEL),
        ("🟢  3+ gates (VALID / DEPLOYED)", C_GRN),
    ]
    for i, (lbl, c) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((0.12, 0.52 - i*0.06), 0.76, 0.045,
                      boxstyle="round,pad=0.005", fc=c+"22", ec=c, lw=1.5))
        ax.text(0.5, 0.543 - i*0.06, lbl, ha="center", va="center",
                fontsize=10, color=C_DARK)

    # Timeline bar
    phases = [
        ("Phase 1\nClassic", C_RED),
        ("Phase 2-3\nATR", C_YEL),
        ("Phase 4\nConvex", C_GRN),
        ("Phase 5-6\nZone opt", C_GRN),
        ("Phase 7-9\nMulti-pair", C_GRN),
        ("Trail\nHybrid", C_GRN),
        ("H1 S/R\nDir.", C_GRN),
        ("H4 S/R\nLIVE", C_GRN),
        ("Random\nN-bar", C_GRN),
        ("P&F\ntimed", C_GRN),
    ]
    x0, bw, gap = 0.07, 0.078, 0.005
    for i, (lbl, c) in enumerate(phases):
        xp = x0 + i*(bw+gap)
        ax.add_patch(FancyBboxPatch((xp, 0.12), bw, 0.20,
                      boxstyle="round,pad=0.005", fc=c+"44", ec=c, lw=1.5))
        ax.text(xp + bw/2, 0.22, lbl, ha="center", va="center",
                fontsize=7.5, color=C_DARK, fontweight="bold")

    ax.text(0.5, 0.06,
            "Break-even sizing (PF=1.25)  ·  H4 TopsBots S/R  ·  Split-account 011 (LONG) / 012 (SHORT)  ·  CHF_JPY live since 2026-05-04",
            ha="center", va="center", fontsize=9, color=C_GREY, style="italic")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MASTER RANKING: All variants by OOS p/d
# ══════════════════════════════════════════════════════════════════════════════
def p_master_ranking():
    variants = [
        ("H4 S/R dir. 12-pair agg",   819820, 0.095, 5,  "Break-even", True),
        ("H1 S/R dir. 12-pair agg",   442317, 0.070, 5,  "Break-even", False),
        ("Ph4: Convex sizing (EUR_USD)",462000, 0.128, 4, "Convex n^1.5", False),
        ("Trail hybrid GBP_JPY",        41495, 0.065, 4,  "Break-even", False),
        ("Ph7b: Boundary zw=25",        28400, 0.268, 5,  "Break-even", False),
        ("Ph9: GBP_JPY random",         16771, 0.031, 5,  "Break-even", False),
        ("Ph5: Large zone best",         26525, 0.356, 5, "Convex n^1.5", False),
        ("Ph6: E/Z best (EUR_USD)",       3767, 0.408, 5, "Convex n^1.5", False),
        ("Random N-bar CHF_JPY (LIVE)",   1080, 0.050, 3,  "Break-even", True),
        ("P&F NZD_JPY b5r4",             1207, 1.11,  3,  "Break-even", False),
        ("P&F USD_JPY b5r2",             1335, 1.18,  1,  "Break-even", False),
        ("Ph2: ATR best (EUR_USD)",       57624, 0.055, 2, "Dynamic", False),
        ("Ph1: Classic cBot best",       -24278, -0.022, 0, "Dynamic", False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("All ZR Variants — OOS Performance Ranking", fontsize=13,
                 fontweight="bold", color=C_DARK, y=0.97)

    # Left: OOS P&L bar
    ax = axes[0]
    ax_style(ax, "OOS P&L ($@1,000u)", "Variant", "OOS USD")
    names  = [v[0] for v in variants]
    pnls   = [v[1] for v in variants]
    gates  = [v[3] for v in variants]
    live   = [v[5] for v in variants]
    colors = [C_GRN if g >= 3 else (C_YEL if g >= 2 else C_RED) for g in gates]
    y = range(len(names))
    bars = ax.barh(y, pnls, color=colors, alpha=0.82, edgecolor="white", height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(0, color=C_DARK, lw=0.8)
    for i, (bar, lv) in enumerate(zip(bars, live)):
        if lv:
            ax.text(bar.get_width() + 5000, i, " ★LIVE", va="center",
                    fontsize=7, color=C_GRN, fontweight="bold")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # Right: Sharpe comparison
    ax = axes[1]
    ax_style(ax, "Sharpe Ratio", "Variant", "Sharpe")
    sharpes = [v[2] for v in variants]
    bars2 = ax.barh(y, sharpes, color=colors, alpha=0.82, edgecolor="white", height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(["" for _ in names], fontsize=7.5)
    ax.axvline(0, color=C_DARK, lw=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(fc=C_GRN, label="3+ gates (Valid)"),
        Patch(fc=C_YEL, label="2 gates (Marginal)"),
        Patch(fc=C_RED, label="0-1 gates (Failed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.15))
    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Each bar represents the aggregate OOS P&L ($@1,000 units) or Sharpe ratio for one experiment family or top configuration. "
              "Colour encodes validation gate count: green = 3+ gates passed (statistically valid), yellow = 2 gates (marginal), "
              "red = 0–1 gates (failed). Note that OOS P&L is not comparable across different lot-size contexts — "
              "variants tested on a single pair (EUR_USD, CHF_JPY) will show lower absolute USD than 12-pair aggregates. "
              "Sharpe is the cleaner cross-variant comparison. The H4 directional variants dominate both panels.",
        finding="The H4 S/R directional entry is the unambiguous leader: +$819K aggregate, 12/12 pairs, perm p=0.0000. "
                "All single-pair convex-sizing variants (Ph4–Ph6) are inflated because 97% of cycles hit max_legs — "
                "they are momentum pyramids, not true ZR recovery. Break-even sizing (Ph7-onward) is the correct benchmark.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PHASE 1: Classic cBot ping-pong anatomy
# ══════════════════════════════════════════════════════════════════════════════
def p_phase1_classic():
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Phase 1: Classic cBot Zone Recovery — The Ping-Pong Origin",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not p1.empty:
        # E/Z ratio vs Sharpe
        ax = axes[0]
        ax_style(ax, "E/Z Ratio vs Sharpe (all 320 configs)", "E/Z Ratio", "Sharpe")
        sc = ax.scatter(p1["ez_ratio"], p1["sharpe"], c=p1["sharpe"],
                       cmap="RdYlGn", vmin=-0.05, vmax=0.05, alpha=0.7, s=30)
        ax.axhline(0, color=C_DARK, lw=1, ls="--")
        ax.axvline(0.29, color=C_RED, lw=1.5, ls="--", label="Original cBot E/Z=0.29")
        ax.legend(fontsize=7, frameon=False)
        plt.colorbar(sc, ax=ax, label="Sharpe")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Win rate distribution
        ax = axes[1]
        ax_style(ax, "Win Rate Distribution", "Win Rate", "Count")
        ax.hist(p1["win_rate"], bins=25, color=C_RED, alpha=0.75, edgecolor="white")
        ax.axvline(p1["win_rate"].mean(), color=C_DARK, lw=1.5, ls="--",
                   label=f'Mean {p1["win_rate"].mean():.1%}')
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Avg legs distribution
        ax = axes[2]
        ax_style(ax, "Average Legs per Cycle", "Avg Legs", "Count")
        ax.hist(p1["avg_legs"], bins=25, color=C_RED, alpha=0.75, edgecolor="white")
        ax.axvline(p1["avg_legs"].mean(), color=C_DARK, lw=1.5, ls="--",
                   label=f'Mean {p1["avg_legs"].mean():.1f} legs')
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

    # Annotation boxes
    notes = [
        "All 320 configs negative",
        "E/Z = 0.29 (original cBot)\nTarget = 29% of zone width\nInsufficient to recover legs",
        "High win rate (87%) but\nlarge losing trades dwarf winners",
        "3.5 avg legs = deep grids\nroutinely activated",
        "Root cause: E/Z too low —\ntarget < spread × leg_count",
    ]
    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="The 'Ping-Pong' strategy originated in the cBot (cAlgo) retail trading community. The original implementation "
              "sets a fixed target-to-zone ratio (E/Z) of just 0.29 — meaning the escape target is only 29% of the zone width. "
              "We tested every combination of zone width, target, and profit-factor across 320 configurations. Every single "
              "config lost money in OOS. The left chart shows why: at E/Z=0.29 (red dashed line), Sharpe is uniformly negative "
              "regardless of other parameters. The middle chart shows win rate of 84–92% — the strategy wins most individual trades "
              "but the rare deep-grid cycles produce losses so large they overwhelm the small winners. The right chart shows "
              "3.5 average legs per cycle: the grid activates constantly because the target is too close to recover costs.",
        finding="ROOT CAUSE: E/Z=0.29 means target_pips < spread × average_legs. Every deep-grid cycle is a structural loss. "
                "Fix: raise E/Z to 8–10 via ATR calibration (Phase 2). This single change turns all 80 ATR configs positive.",
        finding_color=C_RED)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PHASE 2-4: ATR calibration + sizing discovery
# ══════════════════════════════════════════════════════════════════════════════
def p_phase24():
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Phase 2–4: ATR Calibration + Convex Sizing Discovery",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Phase 2: ATR grid — Sharpe by hz_mult × tgt_mult
    ax = fig.add_subplot(gs[0, :2])
    if not p2.empty:
        ax_style(ax, "Phase 2: ATR Grid — Sharpe by (hz_mult, tgt_mult)", "Target Mult", "HZ Mult")
        pivot = p2.pivot_table(values="sharpe", index="half_zone_mult",
                               columns="target_mult", aggfunc="max")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-0.03, vmax=0.07, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=7, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
        plt.colorbar(im, ax=ax, label="Sharpe")
        ax.set_facecolor("white")
    else:
        ax.text(0.5, 0.5, "Phase 2 data not available\n(ATR grid: best Sharpe 0.055, E/Z~8-10)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color=C_GREY)
        ax.axis("off")

    # Sizing comparison bar chart
    ax = fig.add_subplot(gs[0, 2])
    ax_style(ax, "Dynamic vs Convex Sizing\n(OOS pips, EUR_USD)", "Sizing Mode", "OOS Net Pips")
    modes = ["Dynamic\n(Phase 2)", "Convex n^1.5\n(Phase 4)"]
    vals  = [57624, 462000]
    colors = [C_YEL, C_GRN]
    bars = ax.bar(modes, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f"{v/1000:.0f}K", ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # Metrics comparison table
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    rows = [
        ["Metric",          "Ph1: Classic", "Ph2: ATR Dynamic", "Ph4: ATR Convex", "Ph5: Large Zone", "Ph6: E/Z Opt"],
        ["E/Z Ratio",       "0.29",         "8.7 – 9.7",        "—",               "0.19 – 0.50",      "0.30 – 0.90"],
        ["Gates Passed",    "0 / 5",        "2 / 5",            "4 / 5",           "5 / 5",            "38/39 pass 5/5"],
        ["OOS Sharpe",      "−0.022",       "+0.055",           "+0.128",          "+0.356 (best)",    "+0.408 (best)"],
        ["OOS SQN",         "−2.9",         "7.3",              "10.2",            "8.4",              "9.7"],
        ["Avg Legs/Cycle",  "3.58",         "3.81",             "3.81",            "9.5",              "9.8"],
        ["Max DD (pips)",   "25,981",       "2,649",            "~2,600",          "—",                "109"],
        ["Win Rate",        "84.4%",        "83.7%",            "~84%",            "72%",              "68.6%"],
    ]
    col_w = [0.18, 0.13, 0.16, 0.15, 0.16, 0.18]
    x0 = 0.02
    colors_row0 = [C_DARK] * 6
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            xp = x0 + sum(col_w[:ci])
            bg = C_BG if ri % 2 == 0 else "white"
            if ri == 0: bg = C_DARK
            fc = "white" if ri == 0 else C_DARK
            ax.add_patch(FancyBboxPatch((xp, 0.85 - ri*0.115), col_w[ci]-0.005, 0.105,
                          boxstyle="round,pad=0.002", fc=bg, ec="none"))
            ax.text(xp + col_w[ci]/2, 0.9 - ri*0.115 + 0.01, cell,
                    ha="center", va="center", fontsize=8,
                    color=fc, fontweight="bold" if ri == 0 else "normal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Phase 2 recalibrates zone width to 0.7–0.9× a short-term ATR and the target to 1.5–3.25× a long-term ATR, "
              "pushing E/Z to 8–10. All 80 configs turn OOS-positive. The heatmap shows Sharpe peaks in the top-right "
              "(high hz_mult, high tgt_mult) — wider targets relative to zone width consistently outperform. "
              "Phase 4 discovers the sizing breakthrough: replacing dynamic sizing (each leg sized to break even at its own "
              "target) with convex sizing (volume ∝ leg_number^1.5) multiplies OOS pips by 3.8×. The comparison bar is stark. "
              "The metrics table consolidates Phases 1–6 showing how each design change improved Sharpe, SQN, and win rate.",
        finding="SIZING IS EVERYTHING (within correct E/Z): Convex n^1.5 sizing = +462K pips vs Dynamic +57K pips on same "
                "ATR config. However, convex is NOT true ZR — 97% of cycles hit max_legs (it's a pyramid). "
                "Phase 7+ switches to break-even sizing, which is the correct production model.",
        finding_color=C_YEL)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PHASE 5-6: Zone sweep + E/Z optimization
# ══════════════════════════════════════════════════════════════════════════════
def p_phase56():
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Phase 5–6: Large-Zone Sweep & E/Z Ratio Optimization",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    # P5: hz vs tgt heatmap
    ax = axes[0, 0]
    if not p5.empty and "hz" in p5.columns:
        pv = p5[p5["verdict"]=="PASS"].pivot_table(
            values="sharpe", index="hz", columns="tgt", aggfunc="max")
        im = ax.imshow(pv.values if not pv.empty else np.zeros((4, 4)),
                       aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.40, origin="lower")
        if not pv.empty:
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels([str(v) for v in pv.columns], fontsize=7, rotation=45)
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels([str(v) for v in pv.index], fontsize=7)
        plt.colorbar(im, ax=ax, label="Sharpe")
    ax_style(ax, "Ph5: hz vs tgt → max Sharpe\n(Convex sizing, PASS configs)", "Target (pips)", "Half-Zone (pips)")

    # P5: avg_legs distribution
    ax = axes[0, 1]
    if not p5.empty and "avg_legs" in p5.columns:
        ax_style(ax, "Ph5: Avg Legs Distribution", "Avg Legs/Cycle", "Count")
        passed = p5[p5["verdict"] == "PASS"]
        failed = p5[p5["verdict"] != "PASS"]
        ax.hist(failed["avg_legs"], bins=15, color=C_RED, alpha=0.6, label="FAIL")
        ax.hist(passed["avg_legs"], bins=15, color=C_GRN, alpha=0.6, label="PASS")
        ax.axvline(passed["avg_legs"].mean() if not passed.empty else 0,
                   color=C_DARK, lw=1.5, ls="--",
                   label=f'PASS mean={passed["avg_legs"].mean():.1f}' if not passed.empty else "")
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 5 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # P6: E/Z vs Sharpe
    ax = axes[1, 0]
    if not p6.empty:
        ax_style(ax, "Ph6: E/Z Ratio → Sharpe (monotonic rise)", "E/Z Ratio", "Sharpe")
        for hz_val, grp in p6.groupby("hz"):
            grp_sorted = grp.sort_values("ez")
            c = {30: C_BLU, 35: C_GRN, 40: C_PUR}.get(hz_val, C_GREY)
            ax.plot(grp_sorted["ez"], grp_sorted["sharpe"], "o-",
                    color=c, alpha=0.8, ms=4, label=f"hz={hz_val}")
        ax.axhline(0, color=C_DARK, lw=0.8, ls="--")
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 6 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # P6: max_dd vs sharpe scatter
    ax = axes[1, 1]
    if not p6.empty and "max_dd" in p6.columns:
        ax_style(ax, "Ph6: Risk-Return (MaxDD vs Sharpe)", "Max DD (pips)", "Sharpe")
        passed6 = p6[p6["verdict"] == "PASS"]
        sc = ax.scatter(passed6["max_dd"].abs(), passed6["sharpe"],
                       c=passed6["ez"], cmap="viridis", alpha=0.8, s=40)
        plt.colorbar(sc, ax=ax, label="E/Z ratio")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 6 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Phase 5 sweeps zone width (hz=25–75p) and target (tgt=25–45p) using random entries and convex sizing on EUR/USD. "
              "The heatmap shows Sharpe peaks around hz=35, tgt=35 (E/Z≈0.5) — larger zones with proportional targets perform best. "
              "Phase 6 then fixes hz at 30/35/40p and sweeps E/Z from 0.30 to 0.90. The critical finding: Sharpe rises "
              "monotonically with E/Z — every increase in target-to-zone ratio improves performance, up to the tested limit of "
              "E/Z=0.85 (Sharpe 0.408, SQN 9.7). This confirms that the target size is the primary Sharpe lever: wider escape "
              "targets reduce the frequency of deep grid cycles and improve per-cycle expectancy.",
        finding="E/Z RISES MONOTONICALLY WITH SHARPE: From E/Z=0.30 (Sharpe 0.22) to E/Z=0.85 (Sharpe 0.408). "
                "38/39 configs across all hz values pass 5/5 gates. However, these use convex sizing on single-pair EUR/USD. "
                "Phase 7+ moves to break-even sizing and multi-pair, which changes the optimal E/Z to 0.5 (tgt = 0.5×ZW).",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PHASE 7-9: Multi-pair expansion
# ══════════════════════════════════════════════════════════════════════════════
def p_phase79():
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Phase 7–9: Boundary Engine, Multi-Pair Expansion & Break-Even Sizing",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    # P7: boundary vs center
    ax = axes[0, 0]
    if not p7.empty:
        center = p7[p7["variant"]=="center"]
        bdry   = p7[p7["variant"]=="boundary"]
        ax_style(ax, "Ph7: Boundary vs Center — Sharpe dist.", "Sharpe", "Count")
        ax.hist(center["sharpe"].clip(-0.1, 0.5), bins=20, color=C_BLU, alpha=0.6, label="Center")
        ax.hist(bdry["sharpe"].clip(-0.1, 0.5), bins=20, color=C_GRN, alpha=0.6, label="Boundary")
        ax.axvline(center["sharpe"].median(), color=C_BLU, lw=2, ls="--",
                   label=f'Center med={center["sharpe"].median():.3f}')
        ax.axvline(bdry["sharpe"].median(), color=C_GRN, lw=2, ls="--",
                   label=f'Boundary med={bdry["sharpe"].median():.3f}')
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 7 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # P8: multi-pair convex best per pair
    ax = axes[0, 1]
    if not p8.empty:
        best8 = p8[p8["gates"] >= 4].groupby("pair")["pnl_hr_1u"].max().reset_index()
        best8 = best8.sort_values("pnl_hr_1u", ascending=True)
        ax_style(ax, "Ph8: Best $/hr@1u per pair (Convex)", "$/hr@1u", "Pair")
        ax.barh(range(len(best8)), best8["pnl_hr_1u"]*24, color=C_BLU, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best8)))
        ax.set_yticklabels(best8["pair"].str.replace("_", "/"), fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlabel("$/day@1u (convex — inflated)", fontsize=8, color=C_GREY)
    else:
        ax.text(0.5, 0.5, "Phase 8 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # P9: break-even multi-pair best per pair
    ax = axes[1, 0]
    if not p9.empty:
        best9 = p9[p9["gates"] >= 4].groupby("pair").apply(
            lambda g: g.loc[g["sharpe"].idxmax()]).reset_index(drop=True)
        best9 = best9.sort_values("sharpe", ascending=True)
        colors9 = [C_GRN if r >= 5 else C_YEL for r in best9["gates"]]
        ax_style(ax, "Ph9: Best Sharpe per pair (Break-even)", "Sharpe", "Pair")
        ax.barh(range(len(best9)), best9["sharpe"], color=colors9, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best9)))
        ax.set_yticklabels(best9["pair"].str.replace("_", "/"), fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 9 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # P9: ZW vs Sharpe scatter (avg_legs not in p9; use zw as x-axis)
    ax = axes[1, 1]
    if not p9.empty:
        passed9 = p9[p9["gates"] >= 4]
        ax_style(ax, "Ph9: Zone Width vs Sharpe\n(Break-even sizing, passed configs)", "Zone Width (pips)", "Sharpe")
        sc = ax.scatter(passed9["zw"], passed9["sharpe"],
                       c=passed9["ez"], cmap="viridis", alpha=0.6, s=30)
        plt.colorbar(sc, ax=ax, label="E/Z ratio")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Phase 9 data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Phase 7 compares two zone geometries: boundary-entry (enter at zone boundary, target beyond it) vs center-entry "
              "(enter at zone midpoint). Center wins on per-trade Sharpe; boundary doubles the cycle count. Both pass 63/64 "
              "or 5/5 gates — the geometry is less important than sizing and E/Z. Phase 8 extends to all 12 pairs using convex "
              "sizing and ATR-calibrated zone widths. Phase 9 switches to break-even sizing on 12 pairs, which is the "
              "production model: the optimal zone width shifts from 0.12×daily_range (Phase 8 convex) to 0.53×daily_range "
              "(Phase 9 break-even). The right scatter shows ZW vs Sharpe — wider zones with E/Z=0.5–1.0 dominate.",
        finding="BREAK-EVEN SIZING NEEDS WIDER ZONES: Phase 8 (convex) optimum zw ≈ 0.12×daily_range. "
                "Phase 9 (break-even) optimum zw ≈ 0.53×daily_range. The analytical prior (zw = 0.30×ATR_daily) "
                "was off by 45–62% in Phase 8 and undershoots Phase 9. Empirical calibration per pair is required.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — TRAILING HYBRID SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def p_trail():
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Trailing Stop Hybrid — First-Leg Trail Sweep",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not trail.empty:
        pairs = trail["pair"].unique()
        # P/D by (ta, td) for CHF_JPY (most data)
        ax = axes[0]
        ax_style(ax, "CHF_JPY: P/D by (ta, td)", "Trail Dist td (pips)", "Activation ta (pips)")
        chf = trail[trail["pair"]=="CHF_JPY"].copy()
        chf_real = chf[chf["ta"] != "BASE"].copy()
        if not chf_real.empty:
            chf_real["ta"] = chf_real["ta"].astype(float)
            chf_real["td"] = chf_real["td"].astype(float)
            chf_real["ppd"] = pd.to_numeric(chf_real["ppd"], errors="coerce")
            try:
                pv = chf_real.pivot_table(values="ppd", index="ta", columns="td", aggfunc="mean")
                im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn",
                               vmin=pv.values.min(), vmax=pv.values.max(), origin="lower")
                ax.set_xticks(range(len(pv.columns)))
                ax.set_xticklabels([str(int(v)) for v in pv.columns], fontsize=8)
                ax.set_yticks(range(len(pv.index)))
                ax.set_yticklabels([str(int(v)) for v in pv.index], fontsize=8)
                plt.colorbar(im, ax=ax, label="P/D (pips)")
            except Exception:
                ax.text(0.5, 0.5, "Pivot failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_facecolor("white")

        # Trail % by pair
        ax = axes[1]
        ax_style(ax, "Trail Exit % by Pair\n(best ta/td config)", "Pair", "Trail Exit %")
        best_trail = trail[trail["ta"] != "BASE"].copy()
        best_trail["ppd"] = pd.to_numeric(best_trail["ppd"], errors="coerce")
        best_trail["trail_pct"] = pd.to_numeric(best_trail["trail_pct"], errors="coerce")
        bt = best_trail.groupby("pair").apply(lambda g: g.loc[g["ppd"].idxmax()]).reset_index(drop=True)
        ax.bar(bt["pair"].str.replace("_", "/"), bt["trail_pct"],
               color=[C_GRN if w == 3 else C_YEL for w in bt["wf"]], alpha=0.8, edgecolor="white")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # P/D improvement vs baseline
        ax = axes[2]
        ax_style(ax, "Trail vs Baseline P/D improvement", "Pair", "P/D improvement (×)")
        baselines = trail[trail["ta"] == "BASE"].set_index("pair")["ppd"].to_dict()
        bt["base"] = bt["pair"].map(baselines)
        bt["improvement"] = bt["ppd"].astype(float) / bt["base"].astype(float)
        colors_im = [C_GRN if v > 1 else C_RED for v in bt["improvement"]]
        ax.bar(bt["pair"].str.replace("_", "/"), bt["improvement"],
               color=colors_im, alpha=0.8, edgecolor="white")
        ax.axhline(1, color=C_DARK, lw=1.5, ls="--", label="Baseline")
        ax.legend(fontsize=7, frameon=False)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Trail sweep data not available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="The trailing hybrid attaches a trailing stop to the primary leg only: once MFE (maximum favourable excursion) "
              "reaches 'ta' pips, the trail activates and locks in profit at peak − 'td' pips. If price reaches the trail stop "
              "before the zone is crossed, the cycle closes profitably in one leg. This recycled capital immediately starts "
              "a new cycle, increasing throughput without changing risk per cycle. The left heatmap shows CHF_JPY p/d by "
              "(ta, td) — the sweet spot is ta=5, td=3: activates quickly enough to catch short bounces but doesn't trail "
              "so tightly it gets stopped on noise. The middle chart compares trail exit percentage by pair: CHF_JPY achieves "
              "90% trail exits (90% of cycles close profitably in one leg without ever triggering the grid).",
        finding="BEST CONFIG: CHF_JPY ta=5, td=3 → +1,080 p/d vs +500 p/d baseline (+116%). 4/5 gates (perm p=0.05 marginal). "
                "The trail is NOT a stop — it harvests short bounces, which are the dominant price behaviour at S/R levels. "
                "This is the configuration currently deployed live on accounts 011/012.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SIGNAL ENTRY SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def p_signal():
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Signal Entry Sweep — SMA Filters, Crosses & Direction Gates",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not signal.empty:
        signal["ppd"] = pd.to_numeric(signal["ppd"], errors="coerce")

        # P/D by signal mode (mean across pairs)
        ax = axes[0]
        ax_style(ax, "Mean P/D by Signal Mode\n(across all pairs)", "Signal Mode", "Mean P/D")
        agg = signal.groupby("signal")["ppd"].mean().sort_values(ascending=False)
        colors_s = [C_GRN if v > 0 else C_RED for v in agg.values]
        ax.bar(range(len(agg)), agg.values, color=colors_s, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color=C_DARK, lw=1, ls="--")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Entry rate by signal mode
        ax = axes[1]
        ax_style(ax, "Entry Rate by Signal Mode\n(% of bars with signal)", "Signal Mode", "Entry Rate %")
        signal["entry_rate"] = pd.to_numeric(signal["entry_rate"], errors="coerce")
        agg_er = signal.groupby("signal")["entry_rate"].mean().reindex(agg.index)
        ax.bar(range(len(agg_er)), agg_er.values, color=C_BLU, alpha=0.7, edgecolor="white")
        ax.set_xticks(range(len(agg_er)))
        ax.set_xticklabels(agg_er.index, rotation=45, ha="right", fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # P/D by pair × signal heatmap (best signal per pair)
        ax = axes[2]
        ax_style(ax, "P/D Heatmap: Pair × Signal Mode", "Signal Mode", "Pair")
        try:
            pv = signal.pivot_table(values="ppd", index="pair", columns="signal", aggfunc="mean")
            max_abs = max(abs(pv.values.min()), abs(pv.values.max()), 1)
            im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn",
                           vmin=-max_abs, vmax=max_abs, origin="lower")
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels(pv.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels(pv.index.str.replace("_", "/"), fontsize=7)
            plt.colorbar(im, ax=ax, label="P/D")
        except Exception:
            ax.text(0.5, 0.5, "Heatmap failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_facecolor("white")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Signal entry data not available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="This experiment asked: can a directional signal (price vs SMA, SMA vs its own prior value, SMA cross) "
              "improve ZR returns by only entering when the signal agrees with the primary leg direction? Modes tested: "
              "no_signal (random), h1_sma5_filter (only enter if H1 SMA5 agrees, 2.5% entry rate), h1_sma5_only "
              "(enter in direction of SMA5, 100% entry rate), h1_sma10_filter (1.5% entry rate), and similar for SMA20. "
              "The left chart shows mean p/d by mode. The middle shows entry rate — filter modes block 97–99% of entries. "
              "The heatmap on the right shows pair × signal p/d. The pattern is consistent: _only modes are competitive, "
              "_filter modes are catastrophically worse because they kill L/S symmetry and reduce n_cycles to near-zero.",
        finding="FILTERS DESTROY THE EDGE: m5_sma10_filter = 111 p/d vs random_alt = 1,080 p/d on CHF_JPY. "
                "Filters block 89% of entries and break the L/S complementarity that makes alternating ZR work. "
                "The _only (direction-as-entry) modes are competitive but not better than P&F timing — use P&F instead.",
        finding_color=C_RED)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — H1/H4 DIRECTIONAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def p_h1h4():
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("H1 / H4 TopsBots S/R Directional Entry — 12-Pair OOS Results",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    pairs  = list(H4_PAIRS.keys())
    h1_usd = [H1_PAIRS[p] for p in pairs]
    h4_usd = [H4_PAIRS[p] for p in pairs]
    base   = [RANDOM_BASELINE[p] for p in pairs]
    yp     = range(len(pairs))

    # H1 vs H4 improvement
    ax = axes[0]
    ax_style(ax, "OOS P&L: Random vs H1 vs H4\n($@1,000u, tgt=0.25×ZW)", "USD", "Pair")
    ax.barh([y + 0.22 for y in yp], base, height=0.22, color=C_GREY, alpha=0.7, label="Random")
    ax.barh([y + 0.0  for y in yp], h1_usd, height=0.22, color=C_BLU, alpha=0.8, label="H1 dir.")
    ax.barh([y - 0.22 for y in yp], h4_usd, height=0.22, color=C_GRN, alpha=0.9, label="H4 dir.")
    ax.set_yticks(list(yp))
    ax.set_yticklabels([p.replace("_", "/") for p in pairs], fontsize=7)
    ax.axvline(0, color=C_DARK, lw=0.8)
    ax.legend(fontsize=7, frameon=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Improvement ratio H4 vs random
    ax = axes[1]
    ax_style(ax, "H4 vs Random: Improvement Factor\n(>1 = H4 beats random)", "Improvement ×", "Pair")
    improv = []
    for p in pairs:
        b = RANDOM_BASELINE[p]
        h = H4_PAIRS[p]
        if abs(b) < 100:
            improv.append(0)
        elif b < 0:
            improv.append(abs(h)/abs(b) if h > 0 else -1)
        else:
            improv.append(h / b)
    colors_imp = [C_GRN if v > 1 else C_RED for v in improv]
    ax.barh(list(yp), improv, color=colors_imp, alpha=0.8, edgecolor="white")
    ax.set_yticks(list(yp))
    ax.set_yticklabels([p.replace("_", "/") for p in pairs], fontsize=7)
    ax.axvline(1, color=C_DARK, lw=1.5, ls="--", label="Break-even")
    ax.legend(fontsize=7, frameon=False)
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Aggregate bars + permutation significance
    ax = axes[2]
    ax_style(ax, "Aggregate OOS + Significance\n(12-pair sum, perm p=0.0000)", "Config", "Agg OOS USD")
    configs = ["Random\nBaseline", "H1 dir.\n(12 pairs)", "H4 dir.\n(12 pairs)"]
    agg_vals = [sum(RANDOM_BASELINE.values()), sum(H1_PAIRS.values()), sum(H4_PAIRS.values())]
    colors_agg = [C_GREY, C_BLU, C_GRN]
    bars = ax.bar(configs, agg_vals, color=colors_agg, alpha=0.85, edgecolor="white", width=0.55)
    for bar, v in zip(bars, agg_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(agg_vals) * 0.02),
                f"${v/1000:.0f}K", ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.axhline(0, color=C_DARK, lw=0.8)
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.text(2, sum(H4_PAIRS.values()) * 0.55, "perm p=0.0000\n12/12 pairs ✓",
            ha="center", fontsize=8, color=C_GRN, fontweight="bold")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="The single most impactful finding of the entire research program: using H4 TopsBots swing high/low levels "
              "as directional entry signals — entering LONG at confirmed H4 support, SHORT at confirmed H4 resistance — "
              "transforms ZR from a marginal strategy into a high-edge one. Mechanically: TopsBots detects H4 swing points "
              "with a 1-bar lag (swing at bar N−1 confirmed when bar N closes past it — zero lookahead by construction). "
              "When price reaches a confirmed H4 support, we enter LONG expecting a bounce; the ZR grid is insurance if "
              "price continues down. The left chart shows OOS P&L by pair for random vs H1 vs H4 directional. The middle "
              "chart shows the improvement factor (H4 vs random). The right chart shows the permutation-tested aggregate.",
        finding="H4 DIRECTIONAL: +$819K vs +$43K random (19× uplift). 12/12 pairs positive. Perm p=0.0000 (0/2000 shuffles beat real). "
                "H4 > H1 > random on every pair but CAD_JPY. H1+H4 consensus is WORSE than H4 alone — the H4 signal is sufficient.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — RANDOM N-BAR PER-PAIR BASELINE
# ══════════════════════════════════════════════════════════════════════════════
def p_random_perp():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Random N-Bar Alternating Entry — Per-Pair IS/OOS Baseline",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not perp.empty:
        perp["oos_ppd"] = pd.to_numeric(perp["oos_ppd"], errors="coerce")
        perp["is_ppd"]  = pd.to_numeric(perp["is_ppd"],  errors="coerce")
        perp["avgl"]    = pd.to_numeric(perp["avgl"],    errors="coerce")
        best_perp = perp.groupby("pair").apply(
            lambda g: g.loc[g["oos_ppd"].idxmax()]).reset_index(drop=True)
        best_perp = best_perp.sort_values("oos_ppd", ascending=True)

        # OOS p/d by pair
        ax = axes[0]
        ax_style(ax, "Best OOS P/D per Pair\n(random alternating N-bar)", "OOS P/D (pips)", "Pair")
        colors_pp = [C_GRN if v > 0 else C_RED for v in best_perp["oos_ppd"]]
        ax.barh(range(len(best_perp)), best_perp["oos_ppd"], color=colors_pp, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best_perp)))
        ax.set_yticklabels(best_perp["pair"].str.replace("_", "/"), fontsize=8)
        ax.axvline(0, color=C_DARK, lw=0.8)
        for i, row in best_perp.iterrows():
            ax.text(max(best_perp["oos_ppd"]) * 0.02, list(best_perp.index).index(i),
                    f'N={row["N"]} ZW={row["zw"]:.0f}p tgt={row["tf"]:.0f}× WF={int(row["wf"])}',
                    va="center", fontsize=6.5, color=C_DARK)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # IS vs OOS scatter
        ax = axes[1]
        ax_style(ax, "IS vs OOS P/D — Overfitting Check", "IS P/D (pips)", "OOS P/D (pips)")
        colors_sc = [C_GRN if row["wf"] >= 3 else C_YEL if row["wf"] >= 2 else C_RED
                     for _, row in best_perp.iterrows()]
        ax.scatter(best_perp["is_ppd"], best_perp["oos_ppd"], c=colors_sc, s=80, alpha=0.85, edgecolors="white")
        for _, row in best_perp.iterrows():
            ax.annotate(row["pair"].replace("_", "/"),
                        (row["is_ppd"], row["oos_ppd"]),
                        fontsize=6.5, color=C_GREY,
                        xytext=(5, 5), textcoords="offset points")
        lim = max(abs(best_perp["is_ppd"].max()), abs(best_perp["oos_ppd"].max())) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "--", color=C_GREY, lw=1, alpha=0.5, label="IS=OOS")
        ax.axhline(0, color=C_DARK, lw=0.8); ax.axvline(0, color=C_DARK, lw=0.8)
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Per-pair baseline data not available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Before testing more complex signals, we characterised the 'floor' return of ZR with pure random alternating "
              "entry across all 12 pairs. Parameters swept per pair: N (entry interval 1–20 bars), ZW (10–50p), tgt_frac "
              "(0.25–2.0). The left chart shows the best OOS p/d achievable per pair with random entry — some pairs are "
              "structurally not suited (AUD_JPY is highly negative even optimised, suggesting its volatility structure "
              "overwhelms the break-even guarantee). The IS vs OOS scatter on the right is the overfitting check: points "
              "on the IS=OOS line are not overfit; points below it lost IS-to-OOS transfer. Most pairs with WF=3 (green) "
              "sit on or near the line. AUD_JPY's negative OOS despite positive IS is a clear warning sign.",
        finding="RANDOM BASELINE BY PAIR: CHF_JPY, AUD_USD, NZD_USD, GBP_USD have positive OOS p/d with WF=3. "
                "AUD_JPY is strongly negative — exclude from random-entry ZR. USD_JPY and NZD_JPY are borderline. "
                "These baselines define where P&F timing and H4 direction need to add value.",
        finding_color=C_BLU)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — P&F-TIMED SWEEP RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def p_pnf_sweep():
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("P&F-Timed ZR Sweep — 240 Configs (10 pairs × 4 boxes × 3 reversals × 2 directions)",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not pnf.empty:
        pnf["oos_ppd"]    = pd.to_numeric(pnf["oos_ppd"],    errors="coerce")
        pnf["vs_random"]  = pd.to_numeric(pnf["vs_random"],  errors="coerce")
        pnf["wf"]         = pd.to_numeric(pnf["wf"],         errors="coerce")
        pnf["n_cycles"]   = pd.to_numeric(pnf["n_cycles"],   errors="coerce")
        pnf["avg_legs"]   = pd.to_numeric(pnf["avg_legs"],   errors="coerce")
        pnf["trail_pct"]  = pd.to_numeric(pnf["trail_pct"],  errors="coerce")

        alt = pnf[pnf["direction"] == "alternating"]
        col = pnf[pnf["direction"] == "column"]

        # OOS ppd by pair (best alternating config)
        ax = axes[0, 0]
        ax_style(ax, "Best OOS P/D per Pair — Alternating", "OOS P/D (pips/day)", "Pair")
        best_alt = alt.groupby("pair").apply(
            lambda g: g.loc[g["oos_ppd"].idxmax()]).reset_index(drop=True)
        best_alt = best_alt.sort_values("oos_ppd", ascending=True)
        colors_ba = [C_GRN if v >= 3 else C_YEL if v >= 2 else C_RED for v in best_alt["wf"]]
        ax.barh(range(len(best_alt)), best_alt["oos_ppd"], color=colors_ba, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best_alt)))
        ax.set_yticklabels(best_alt["pair"].str.replace("_", "/"), fontsize=8)
        ax.axvline(0, color=C_DARK, lw=0.8)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # vs_random lift by pair
        ax = axes[0, 1]
        ax_style(ax, "P&F Lift Over Random Baseline\n(alternating, best config)", "Lift (pips/day)", "Pair")
        best_alt_s = best_alt.sort_values("vs_random", ascending=True)
        colors_lift = [C_GRN if v > 0 else C_RED for v in best_alt_s["vs_random"]]
        ax.barh(range(len(best_alt_s)), best_alt_s["vs_random"], color=colors_lift, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best_alt_s)))
        ax.set_yticklabels(best_alt_s["pair"].str.replace("_", "/"), fontsize=8)
        ax.axvline(0, color=C_DARK, lw=1.5, ls="--")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Box size vs OOS ppd scatter
        ax = axes[1, 0]
        ax_style(ax, "Box Size vs OOS P/D\n(alternating, all pairs)", "Box Size (pips)", "OOS P/D")
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(alt))
        sc = ax.scatter(alt["box_pips"].astype(float) + jitter, alt["oos_ppd"],
                       c=alt["wf"], cmap="RdYlGn", vmin=0, vmax=3, alpha=0.6, s=25)
        ax.axhline(0, color=C_DARK, lw=0.8, ls="--")
        plt.colorbar(sc, ax=ax, label="WF score")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Alternating vs Column direction
        ax = axes[1, 1]
        ax_style(ax, "Alternating vs Column Direction\n(best OOS P/D per pair × direction)", "OOS P/D (pips/day)", "Pair")
        pairs_shared = sorted(set(alt["pair"]) & set(col["pair"]))
        ba2 = alt.groupby("pair")["oos_ppd"].max().reindex(pairs_shared)
        bc2 = col.groupby("pair")["oos_ppd"].max().reindex(pairs_shared)
        yp = range(len(pairs_shared))
        ax.barh([y + 0.2 for y in yp], ba2.values, height=0.38, color=C_BLU, alpha=0.8, label="Alternating")
        ax.barh([y - 0.2 for y in yp], bc2.values, height=0.38, color=C_PUR, alpha=0.8, label="Column dir.")
        ax.set_yticks(list(yp))
        ax.set_yticklabels([p.replace("_", "/") for p in pairs_shared], fontsize=7)
        ax.axvline(0, color=C_DARK, lw=0.8)
        ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "P&F sweep data not available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Point & Figure (P&F) charts record price in terms of box-size pip movements, ignoring time entirely. "
              "An X column extends upward by box_size pips; when price falls reversal × box_size pips from the column peak, "
              "a new O column begins. A P&F reversal is a structurally earned price event — price had to actually travel "
              "reversal × box_size pips to trigger it. Since ZR parameters (ZW, tgt) are denominated in pips not time, "
              "P&F is the natural entry clock. We swept 10 pairs × 4 box sizes × 3 reversals × 2 directions (alternating vs "
              "column-direction). Top-left: best OOS p/d per pair (alternating). Top-right: lift over random baseline — "
              "8/10 pairs improve. Bottom-left: box size scatter — small boxes (5p) generate more signals, large (15p) fewer "
              "but higher quality. Bottom-right: alternating vs column direction — both competitive, pair-dependent.",
        finding="P&F IMPROVES TIMING ON 8/10 PAIRS: USD_JPY b5r2 +1,335 p/d vs +296 random (+351%). NZD_JPY b5r4 +1,207 p/d. "
                "CHF_JPY b10r3 +1,113 p/d vs +358 random (+211%). Only EUR_USD and EUR_GBP show negative lift. "
                "Alternating direction dominates column direction on 7/10 pairs — the contrarian bounce matters more than trend.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — PERMUTATION TEST + BOOTSTRAP MC
# ══════════════════════════════════════════════════════════════════════════════
def p_permtest():
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("P&F Permutation Test + Bootstrap MC — 9 Top Configs",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    if not perm.empty:
        perm_sorted = perm.sort_values("obs_ppd", ascending=False)

        for i, (_, row) in enumerate(perm_sorted.iterrows()):
            if i >= 6: break
            ax = axes[i // 3, i % 3]
            label = f'{row["pair"].replace("_","/")} b{int(row["box_pips"])}r{int(row["reversal"])} {row["direction"][:3]}'
            gates = int(row["gates"])
            gc = C_GRN if gates >= 3 else (C_YEL if gates >= 2 else C_RED)

            # Bootstrap distribution boxes
            p5_  = float(row["boot_p5"])
            p25_ = float(row["boot_p25"])
            p50_ = float(row["boot_median"])
            p75_ = float(row["boot_p75"])
            p95_ = float(row["boot_p95"])
            obs_ = float(row["obs_ppd"])
            pv_  = float(row["p_value"])

            # Draw bootstrap violin-style
            ax.barh(0, p95_ - p5_, left=p5_, height=0.4, color=gc, alpha=0.2, edgecolor="none")
            ax.barh(0, p75_ - p25_, left=p25_, height=0.4, color=gc, alpha=0.4, edgecolor="none")
            ax.plot([p50_, p50_], [-0.25, 0.25], color=gc, lw=2.5)
            ax.axvline(obs_, color=C_DARK, lw=2, ls="--", label=f"obs={obs_:.0f}")
            ax.axvline(0, color=C_RED, lw=1, ls=":")
            ax.set_yticks([]); ax.set_xlim(min(p5_, -200), max(p95_, obs_ * 1.1))
            ax.set_title(f"{label}\np={pv_:.3f} | Gates={gates}/3", fontsize=8,
                         fontweight="bold", color=gc, pad=4)
            ax.set_xlabel("P/D (pips/day)", fontsize=7, color=C_GREY)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Gates legend
            gate_lbl = []
            if row["gate_perm"]: gate_lbl.append("✓perm")
            else: gate_lbl.append("✗perm")
            if row["gate_p5"]: gate_lbl.append("✓p5>0")
            else: gate_lbl.append("✗p5>0")
            if row["gate_prob"]: gate_lbl.append("✓P(+)>0.95")
            else: gate_lbl.append("✗P(+)>0.95")
            ax.text(0.98, 0.05, "  ".join(gate_lbl),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color=gc)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Permtest data not available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Showing that P&F timing improves OOS p/d is necessary but not sufficient — the improvement could be a "
              "statistical artifact of the specific OOS period. Two tests confirm real edge: (1) Permutation test: shuffle "
              "the 2,000 P&F reversal bar positions randomly 2,000 times, keeping frequency constant, and measure how often "
              "the shuffled result beats the observed. p=0 means no shuffle ever matched. (2) Bootstrap MC: resample "
              "per-cycle P&L 2,000 times with replacement to estimate the confidence interval. Each panel below shows the "
              "bootstrap distribution (dark box = p25–p75, whiskers = p5–p95) with the observed p/d as a dashed vertical "
              "line. Green panels pass all 3 gates (perm p<0.05, boot p5>0, P(+)>0.95); red pass only 1.",
        finding="6/9 TOP CONFIGS PASS ALL 3 GATES: NZD_JPY b5r4, CHF_JPY b10r3, AUD_JPY b10r2 (alt and col), plus 2 more. "
                "USD_JPY b5r2 passes perm (p=0.006) but boot p5 < 0 — fat-tail cycle variance exceeds bootstrap confidence. "
                "This means the edge is real but a few large recovery cycles dominate the p/d, making bootstrap fragile.",
        finding_color=C_GRN)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 13 — RISK & DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def p_risk():
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig_style(fig)
    fig.suptitle("Risk Profile & Drawdown Analysis — Cross-Variant Comparison",
                 fontsize=12, fontweight="bold", color=C_DARK, y=0.98)

    # Leg distribution comparison (from data or hardcoded)
    ax = axes[0]
    ax_style(ax, "Leg Distribution by Variant\n(% of cycles by leg count)", "% of Cycles", "Leg Count")
    # Hardcoded from JOURNEY analysis:
    leg_variants = {
        "H4 dir. (live)": [96.5, 2.0, 0.8, 0.5, 0.2],
        "Ph9 GBP_JPY":    [67.5, 20.9, 7.5, 3.1, 1.0],
        "Ph6 E/Z best":   [3.0,  7.0, 13.0, 24.0, 53.0],
        "Ph5 Large zone": [3.0,  7.0, 14.0, 25.0, 51.0],
    }
    colors_lv = [C_GRN, C_BLU, C_YEL, C_PUR]
    for yi, (vname, dist) in enumerate(leg_variants.items()):
        x = dist[:5]
        legs = [1, 2, 3, 4, "5+"]
        for xi, (pct, leg) in enumerate(zip(x, legs)):
            pass
        ax.barh([f"L{l}" for l in legs], x[::-1],
                color=colors_lv[yi], alpha=0.5, label=vname)

    # Actually do a proper stacked bar
    ax.cla()
    ax_style(ax, "Leg Distribution by Variant\n(% per leg count 1→5+)", "Leg Count", "Fraction of Cycles (%)")
    y_variants = list(leg_variants.keys())
    leg_counts = ["1 leg", "2 legs", "3 legs", "4 legs", "5+ legs"]
    data = np.array(list(leg_variants.values()))
    x = np.arange(len(y_variants))
    bar_colors = [C_GRN, C_BLU, C_YEL, C_PUR, C_RED]
    bottom = np.zeros(len(y_variants))
    for li, (lcnt, bc) in enumerate(zip(leg_counts, bar_colors)):
        ax.bar(x, data[:, li], bottom=bottom, color=bc, alpha=0.8, label=lcnt, edgecolor="white")
        bottom += data[:, li]
    ax.set_xticks(x)
    ax.set_xticklabels(y_variants, rotation=20, ha="right", fontsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_ylabel("% of Cycles", fontsize=9, color=C_GREY)
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Max DD comparison
    ax = axes[1]
    ax_style(ax, "Max Drawdown (pips) by Variant\n(negative = adverse float)", "Variant", "Max DD (pips, abs)")
    dd_variants = [
        ("Ph1 Classic",   25981, 0),
        ("Ph2 ATR dyn.",   2649, 2),
        ("Ph4 Convex",     2600, 4),
        ("Ph6 E/Z best",    109, 5),
        ("Ph7b Boundary",    50, 5),
        ("Ph9 GBP_JPY",   ~50, 5),
        ("H4 dir. (live)",   75, 5),
        ("Trail hybrid",     60, 4),
    ]
    dd_labels = [v[0] for v in dd_variants]
    dd_vals   = [abs(v[1]) for v in dd_variants]
    dd_gates  = [v[2] for v in dd_variants]
    dd_colors = [C_GRN if g >= 3 else (C_YEL if g >= 2 else C_RED) for g in dd_gates]
    ax.bar(range(len(dd_labels)), dd_vals, color=dd_colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(dd_labels)))
    ax.set_xticklabels(dd_labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylabel("Max DD (pips)", fontsize=9, color=C_GREY)

    # Phase 9 Calmar ratio by pair
    ax = axes[2]
    if not p9.empty and "calmar" in p9.columns:
        ax_style(ax, "Ph9 Break-Even: Calmar Ratio by Pair\n(best config per pair)", "Calmar", "Pair")
        best9c = p9[p9["gates"] >= 4].groupby("pair").apply(
            lambda g: g.loc[g["calmar"].idxmax()]).reset_index(drop=True)
        best9c = best9c.sort_values("calmar", ascending=True)
        ax.barh(range(len(best9c)), best9c["calmar"].clip(-5, 50),
                color=C_BLU, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(best9c)))
        ax.set_yticklabels(best9c["pair"].str.replace("_", "/"), fontsize=7)
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    else:
        # Hardcoded Calmar from JOURNEY
        ax_style(ax, "Break-Even: Sharpe vs SQN by Variant", "Sharpe", "SQN")
        variants_rs = [
            ("Ph1 Classic", -0.022, -2.9, 0),
            ("Ph2 ATR",      0.055,  7.3,  2),
            ("Ph4 Convex",   0.128, 10.2,  4),
            ("Ph5 best",     0.356,  8.4,  5),
            ("Ph6 best",     0.408,  9.7,  5),
            ("Ph7b bndry",   0.268, 17.9,  5),
            ("Ph9 GBP/JPY",  0.031,  2.5,  5),
            ("Trail",        0.065,  3.1,  4),
            ("H4 dir.",      0.095,  7.8,  5),
        ]
        for vn, sh, sq, gt in variants_rs:
            c = C_GRN if gt >= 3 else (C_YEL if gt >= 2 else C_RED)
            ax.scatter([sh], [sq], s=80, color=c, alpha=0.85, edgecolors="white")
            ax.annotate(vn, (sh, sq), fontsize=6.5, color=C_GREY,
                        xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color=C_DARK, lw=0.5, ls="--")
        ax.axvline(0, color=C_DARK, lw=0.5, ls="--")
        ax.set_facecolor("white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    pass  # layout handled by add_page_text
    add_page_text(fig,
        intro="Understanding how legs distribute is critical for capital sizing. The left stacked bar compares four variants: "
              "H4 directional (96.5% of cycles are 1-leg bounces — the grid almost never activates), Phase 9 GBP_JPY random "
              "(67.5% one-leg, 20.9% two-leg), Phase 5-6 large-zone convex (only 3% one-leg — the grid fires constantly). "
              "This shows the directional signal quality directly: H4 S/R is so good at identifying reversal points that "
              "the hedge grid is almost never needed. The middle chart compares max drawdown in pips across all variants — "
              "Phase 1 classic is catastrophic (25,981p DD), ATR-calibrated and later variants are far more controlled. "
              "The right Sharpe vs SQN scatter shows the risk-return landscape: Ph5-6 convex configs have high Sharpe but "
              "lower SQN (fewer trades), while boundary configs (Ph7b) have high SQN with moderate Sharpe.",
        finding="LEG DISTRIBUTION IS THE SIGNAL QUALITY PROXY: H4 directional → 96.5% 1-leg exits. Random → 67.5%. "
                "Convex Phase 5-6 → 3% (97% hit max_legs, confirming it is a pyramid not true ZR). "
                "The closer you get to 100% 1-leg exits, the better your directional signal quality.",
        finding_color=C_BLU)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 14 — LIVE STATUS & DEPLOYMENT DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
def p_live():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig_style(fig)
    ax.axis("off")
    fig.suptitle("Live Deployment Status & Variant Decision Summary",
                 fontsize=13, fontweight="bold", color=C_DARK, y=0.97)

    rows = [
        ["Variant",              "Config",                          "OOS P/D",    "Gates", "Status",    "Notes"],
        ["Ph1: Classic cBot",    "E/Z=0.29, Dynamic sizing",        "−42p",       "0/5",   "KILLED",    "All configs negative"],
        ["Ph2-3: ATR grid",      "hz=0.9×ATR, tgt=3×ATR, Dynamic", "+3.3p",      "2/5",   "EVOLVED",   "Foundation: E/Z discovered"],
        ["Ph4: Convex sizing",   "Same + Convex n^1.5",             "+800p",      "4/5",   "EVOLVED",   "+3.8× pips vs dynamic"],
        ["Ph5: Large zone",      "hz=35, tgt=35, Convex",           "+46p/day",   "5/5",   "RESEARCHED","EUR_USD only, Sharpe 0.356"],
        ["Ph6: E/Z best",        "hz=35, tgt=60, E/Z=0.85, Convex","$3.8K OOS",  "38/39", "RESEARCHED","Sharpe 0.408"],
        ["Ph7b: Boundary",       "zw=25, tgt=50, Break-even",       "$28K OOS",   "5/5",   "VALIDATED", "EUR_USD"],
        ["Ph9: GBP_JPY random",  "zw=56, tgt=28, PF=1.25",         "+88p/day",   "5/5",   "WAS LIVE",  "Stopped 2026-05-04"],
        ["Trail hybrid",         "act=14, trail=7, CHF_JPY",        "+240p/day",  "4/5",   "CANDIDATE", "perm p=0.05 marginal"],
        ["H1 S/R directional",   "tgt=0.25×ZW, 12 pairs",          "+442K OOS",  "5/5",   "VALIDATED", "perm p=0.0000"],
        ["H4 S/R directional",   "tgt=0.25×ZW, 12 pairs",          "+820K OOS",  "5/5",   "WAS LIVE",  "Stopped; replaced by ZR-random"],
        ["Random N-bar CHF_JPY", "N=1, ZW=40, ta=5, td=3",         "+1080p/day", "3/5",   "🟢 LIVE",   "fx-zr-random, acct 011/012"],
        ["P&F NZD_JPY b5r4",     "alternating, box=5, rev=4",       "+1207p/day", "3/3",   "CANDIDATE", "3/3 gates pass"],
        ["P&F CHF_JPY b10r3",    "alternating, box=10, rev=3",      "+1113p/day", "3/3",   "CANDIDATE", "3/3 gates — upgrade from N-bar"],
        ["P&F USD_JPY b5r2",     "alternating, box=5, rev=2",       "+1335p/day", "1/3",   "MONITOR",   "perm OK, boot p5 < 0"],
    ]

    col_w = [0.18, 0.22, 0.10, 0.07, 0.10, 0.28]
    row_h = 0.053
    x0, y0 = 0.01, 0.93

    status_colors = {
        "KILLED": C_RED+"33", "EVOLVED": C_GREY+"33", "RESEARCHED": C_GREY+"22",
        "VALIDATED": C_BLU+"33", "WAS LIVE": C_YEL+"33", "CANDIDATE": C_YEL+"55",
        "🟢 LIVE": C_GRN+"55", "MONITOR": C_YEL+"33",
    }

    for ri, row in enumerate(rows):
        status = row[4] if ri > 0 else ""
        bg = status_colors.get(status, C_BG)
        if ri == 0: bg = C_DARK
        for ci, cell in enumerate(row):
            xp = x0 + sum(col_w[:ci])
            ax.add_patch(FancyBboxPatch((xp, y0 - ri*row_h), col_w[ci]-0.003, row_h-0.002,
                          boxstyle="round,pad=0.001", fc=bg, ec="none"))
            fc = "white" if ri == 0 else C_DARK
            fw = "bold" if ri == 0 or ci == 0 else "normal"
            fs = 7.5 if ri == 0 else 7
            ax.text(xp + col_w[ci]/2, y0 - ri*row_h + row_h/2, cell,
                    ha="center", va="center", fontsize=fs, color=fc, fontweight=fw)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.016,
            "CHF_JPY live at B=20 units (acct 011=LONG, 012=SHORT)  ·  GBP_USD (NAV≥$28)  ·  USD_JPY (NAV≥$40)  ·  P&F b10r3 upgrade: immediate candidate",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 15 — EVOLUTION NARRATIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def p_evolution():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig_style(fig)
    ax.axis("off")
    fig.suptitle("Zone Recovery: Key Insights & Evolution Narrative",
                 fontsize=13, fontweight="bold", color=C_DARK, y=0.97)

    insights = [
        ("1", C_RED,   "E/Z Ratio is everything (Phase 1→2)",
         "Classic cBot E/Z=0.29 is a guaranteed loss machine — target too small to recover spread×legs.\n"
         "ATR-calibrated E/Z~8-10 flips all 80 configs positive. The E/Z ratio controls the fundamental break-even economics."),

        ("2", C_YEL,   "Convex sizing is 3.8× more efficient than dynamic (Phase 4)",
         "Dynamic sizing (size to break even at each leg's own target) leaves excess capital idle.\n"
         "Convex n^1.5 pyramids aggressively — bigger positions on deeper legs amplify recovery. +462K vs +57K pips."),

        ("3", C_GRN,   "Break-even sizing is the production model (Phase 7-9)",
         "Convex sizing inflated Phase 8 metrics: 97% of cycles hit max_legs (it's a pyramid, not ZR).\n"
         "Break-even sizing correctly prices each leg to recover the full net deficit at target — true ZR logic."),

        ("4", C_GRN,   "H4 S/R direction is the primary alpha source",
         "Random-direction H4 ZR: +$43K aggregate. H4 directional (LONG at support, SHORT at resistance): +$819K.\n"
         "96.5% of cycles are 1-leg bounces. The grid fires on 1-in-29 trades — directional signal is the edge."),

        ("5", C_GRN,   "P&F timing improves over random N-bar on 8/10 pairs",
         "P&F reversals are structurally earned price events (not time-based) — the natural clock for pip-denominated ZR.\n"
         "USD_JPY: 1,335 vs 296 p/d. NZD_JPY: 1,207 vs 136. CHF_JPY: 1,113 vs 358. 6/9 top configs pass 3/3 gates."),

        ("6", C_BLU,   "All stops make ZR worse — the grid is the stop",
         "Every tested stop (time, equity, max_legs early) reduces P&L. Break-even sizing GUARANTEES profit if target hit.\n"
         "Stops exit before the target can be reached, converting guaranteed-win recovery cycles into losses."),

        ("7", C_PUR,   "Asymmetric recovery structure: 1% of cycles = 82% of profit",
         "At H4 config: 96.5% of cycles are 1-leg bounces contributing 99% of P&L at low risk.\n"
         "1% of cycles (5+ legs) contribute negligible P&L but generate the fat-tail bootstrap variance.\n"
         "⇒ Do NOT apply equity stops to these — they are the structural break-even guarantee events."),
    ]

    y = 0.90
    for num, color, title, body in insights:
        ax.add_patch(FancyBboxPatch((0.02, y-0.095), 0.96, 0.090,
                      boxstyle="round,pad=0.008", fc=color+"15", ec=color, lw=1.5))
        ax.add_patch(FancyBboxPatch((0.02, y-0.095), 0.035, 0.090,
                      boxstyle="round,pad=0.000", fc=color, ec="none"))
        ax.text(0.037, y-0.095+0.045, num, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(0.07, y-0.014, title, va="top", fontsize=9.5,
                fontweight="bold", color=C_DARK)
        ax.text(0.07, y-0.038, body, va="top", fontsize=8, color=C_GREY,
                wrap=True)
        y -= 0.115

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS (book-chapter style overlays)
# ══════════════════════════════════════════════════════════════════════════════
def _wrap(text, width=100):
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))

# Fixed layout zones for chart pages (portrait 8.5×11")
# These are figure-fraction coordinates (0=bottom, 1=top)
_CHART_TOP  = 0.755   # chart subplot area top
_CHART_BOT  = 0.135   # chart subplot area bottom
_TBAR_H     = 0.025   # page title bar height
_INTRO_BOT  = 0.785   # intro text zone bottom (= _CHART_TOP + _TBAR_H + gap)
_FIND_TOP   = 0.125   # finding box top

def add_page_text(fig, intro, finding="", finding_color=None, intro_top=None):
    """Non-overlapping layout: fixed zones, no tight_layout."""
    FC = finding_color or C_GRN

    # Extract suptitle and hide it — we redraw it as a title bar
    page_title = ""
    if hasattr(fig, '_suptitle') and fig._suptitle is not None:
        page_title = fig._suptitle.get_text().strip()
        fig._suptitle.set_visible(False)

    # Direct subplot placement — no tight_layout fight
    fig.subplots_adjust(
        top=_CHART_TOP, bottom=_CHART_BOT,
        left=0.095, right=0.965,
        hspace=0.45, wspace=0.35,
    )

    # Page title bar (thin dark strip just above charts)
    if page_title:
        bar_bot = _CHART_TOP + 0.005
        ax_bar = fig.add_axes([0.03, bar_bot, 0.94, _TBAR_H])
        ax_bar.axis('off')
        ax_bar.add_patch(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle='round,pad=0.01',
            fc=C_DARK, ec='none', transform=ax_bar.transAxes))
        ax_bar.text(0.5, 0.5, page_title, ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white',
                    transform=ax_bar.transAxes)

    # Intro text (top zone, above title bar)
    ax_h = fig.add_axes([0.03, _INTRO_BOT, 0.94, 1.0 - _INTRO_BOT - 0.008])
    ax_h.axis('off')
    ax_h.text(0.0, 1.0, _wrap(intro, 92),
              fontsize=11, va='top', ha='left', linespacing=1.5,
              color=C_DARK, transform=ax_h.transAxes, clip_on=True)

    # Finding box (bottom zone)
    if finding:
        ax_f = fig.add_axes([0.03, 0.010, 0.94, _FIND_TOP - 0.018])
        ax_f.axis('off')
        ax_f.add_patch(FancyBboxPatch(
            (0, 0.04), 1, 0.92, boxstyle='round,pad=0.015',
            fc=FC + '22', ec=FC, lw=2.0, transform=ax_f.transAxes))
        ax_f.text(0.5, 0.52, _wrap(finding, 92),
                  fontsize=11, color=C_DARK, va='center', ha='center',
                  fontweight='bold', linespacing=1.4, transform=ax_f.transAxes,
                  clip_on=True)

# ══════════════════════════════════════════════════════════════════════════════
# NEW PAGE: TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════════════════
def p_toc():
    fig = plt.figure(figsize=(8.5, 11)); fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.905), 0.94, 0.080,
                  boxstyle="round,pad=0.01", fc=C_DARK, ec="none"))
    ax.text(0.50, 0.945, "Table of Contents", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white")
    toc = [
        ( 1, "Title Page",                       "Research overview, evolution timeline, validation legend"),
        ( 2, "Table of Contents",                "This page"),
        ( 3, "Executive Summary",                "What is ZR? Objectives, 5 key findings, headline metrics"),
        ( 4, "ZR Mechanics — How It Works",      "Cycle anatomy, parameter definitions, split-account setup"),
        ( 5, "Glossary",                          "All technical terms, shorthand, and abbreviations defined"),
        ( 6, "Phase 1 — Classic cBot Ping-Pong", "320 configs, E/Z=0.29, all negative — root-cause analysis"),
        ( 7, "Phase 2–4 — ATR + Sizing",         "E/Z calibration, dynamic vs convex sizing: the 3.8× breakthrough"),
        ( 8, "Phase 5–6 — Zone Optimization",    "Large-zone sweep, E/Z ratio sweep, Sharpe rises monotonically"),
        ( 9, "Phase 7–9 — Multi-pair Expansion", "Boundary vs center, 12-pair convex, break-even multi-pair"),
        (10, "Trailing Stop Hybrid",              "First-leg trail: +147% vs baseline via capital recycling"),
        (11, "Signal Entry Sweep",               "SMA cross / price-vs-SMA gates — why all filters hurt"),
        (12, "H1 / H4 TopsBots Directional",     "The primary alpha: +$819K aggregate OOS, perm p=0.0000"),
        (13, "Random N-bar Alternating",          "Per-pair IS/OOS baseline, IS vs OOS overfitting check"),
        (14, "P&F-timed ZR",                     "240-config sweep, 8/10 pairs beat random timing"),
        (15, "Permutation Test + Bootstrap MC",   "Statistical significance: 6/9 top configs pass all 3 gates"),
        (16, "Risk & Drawdown Analysis",          "Leg distribution, max DD comparison, Sharpe vs SQN landscape"),
        (17, "Master Performance Ranking",        "All 14 families ranked by OOS P&L and Sharpe"),
        (18, "Live Deployment Status",            "Current config, account mapping, NAV-gated rollout plan"),
        (19, "Evolution Narrative",               "7 pivotal insights that reshaped the research direction"),
        (20, "Bottom Line — Deployment Decision", "What to run on 011/012 now and why — risk-adjusted verdict"),
    ]
    col_x = [0.055, 0.095, 0.305]
    ax.text(col_x[0], 0.895, "Pg",  fontsize=8, fontweight="bold", color=C_GREY, va="bottom", ha="center")
    ax.text(col_x[2], 0.895, "Section / Description", fontsize=8, fontweight="bold", color=C_GREY, va="bottom")
    ax.axhline(0.893, xmin=0.03, xmax=0.97, color=C_GREY, lw=0.7)
    row_h = 0.038
    new_pgs = {2, 3, 4, 5, 20}
    for i, (pg, section, desc) in enumerate(toc):
        y = 0.890 - (i + 1) * row_h
        bg = C_BG if i % 2 == 0 else "white"
        ax.add_patch(FancyBboxPatch((0.03, y - 0.003), 0.945, row_h - 0.002,
                      boxstyle="square,pad=0", fc=bg, ec="none", zorder=0))
        c_sec = C_BLU if pg in new_pgs else C_DARK
        ax.text(col_x[0], y + row_h/2, str(pg), ha="center", va="center", fontsize=8, color=C_GREY)
        ax.text(col_x[2], y + row_h/2, f"{section}  —  {desc}",
                va="center", fontsize=8, color=c_sec,
                fontweight="bold" if pg in new_pgs else "normal")
    ax.text(0.50, 0.006, "Blue entries are new pages added in this report edition.",
            ha="center", fontsize=7.5, color=C_BLU, style="italic")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# NEW PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def p_exec_summary():
    fig = plt.figure(figsize=(8.5, 11)); fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.905), 0.94, 0.080,
                  boxstyle="round,pad=0.01", fc=C_DARK, ec="none"))
    ax.text(0.50, 0.946, "Executive Summary", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white")
    ax.text(0.50, 0.917, "Zone Recovery / Ping-Pong Research Program  ·  FX-Core  ·  May 2026",
            ha="center", va="center", fontsize=9, color="#bdc3c7")

    ax.text(0.05, 0.893, "WHAT IS ZONE RECOVERY?", fontsize=9, fontweight="bold", color=C_BLU)
    ax.text(0.05, 0.875, _wrap(
        "Zone Recovery (ZR) — nicknamed 'Ping-Pong' in retail cBot communities — is a break-even hedging "
        "strategy. When price crosses a zone boundary in the adverse direction, an additional position is "
        "opened in the opposing direction, sized so that if price later reaches a defined escape target, the "
        "combined position closes at zero or better. Additional 'legs' are opened each time price re-crosses "
        "the zone. The key insight: if the target is set far enough beyond the zone (high E/Z ratio) and "
        "positions are correctly sized, every cycle is mathematically guaranteed to close profitably at the "
        "target — the only question is when. To work around OANDA's no-hedging rule, LONG legs go to account "
        "011 and SHORT legs to account 012.", 88),
        fontsize=10, color=C_DARK, va="top", linespacing=1.4)

    ax.text(0.05, 0.738, "RESEARCH OBJECTIVE", fontsize=9, fontweight="bold", color=C_BLU)
    ax.text(0.05, 0.720, _wrap(
        "Starting from the classic cBot implementation (which loses money reliably), we systematically "
        "examined every design dimension: zone geometry, target ratio, sizing algorithm, entry timing "
        "(random, SMA-filtered, H1/H4 directional, P&F-timed), trailing stops, multi-pair applicability, "
        "and stop mechanisms. Validation requires passing 3–5 statistical gates: OOS positive, WF=3 "
        "(all walk-forward sub-periods profitable), permutation p<0.05, bootstrap p5>0, SQN>1. "
        "All results reported on the held-out 30% OOS period never used during parameter search.", 88),
        fontsize=10, color=C_DARK, va="top", linespacing=1.4)

    ax.text(0.05, 0.612, "5 KEY FINDINGS", fontsize=9, fontweight="bold", color=C_BLU)
    findings = [
        (C_RED,  "1", "E/Z=0.29 guarantees structural loss.",
                       "Classic cBot target is too close to recover spread × leg_count. Must calibrate E/Z to 8–10+ via ATR. This single fix turns all 80 ATR configs positive."),
        (C_YEL,  "2", "Convex sizing (n^1.5) is 3.8× more efficient — but it is NOT true ZR.",
                       "Convex pyramids aggressively on deeper legs. In production, use break-even sizing (correctly prices each leg to recover the full deficit at target)."),
        (C_GRN,  "3", "H4 TopsBots S/R direction is the primary alpha — a 19× uplift over random.",
                       "+$819K vs +$43K aggregate 12-pair OOS. Permutation p=0.0000. 96.5% of cycles are 1-leg bounces. The directional signal — not the grid — is the edge."),
        (C_GRN,  "4", "P&F timing beats random on 8/10 pairs; 6/9 top configs pass all 3 gates.",
                       "P&F reversals are structurally earned price events, not arbitrary time bars. The natural clock for a pip-denominated system. USD_JPY +1,335 p/d vs +296 random."),
        (C_GRN,  "5", "All stops (time, equity, max-leg early) make ZR worse.",
                       "Break-even sizing guarantees profit if target hit. Stops exit before target — converting guaranteed-recovery cycles into realised losses. The grid IS the stop."),
    ]
    y = 0.595
    for clr_f, num, bold_txt, body_txt in findings:
        ax.add_patch(FancyBboxPatch((0.04, y - 0.068), 0.92, 0.066,
                      boxstyle="round,pad=0.005", fc=clr_f+"15", ec=clr_f, lw=1.2))
        ax.add_patch(FancyBboxPatch((0.04, y - 0.068), 0.028, 0.066,
                      boxstyle="round,pad=0", fc=clr_f, ec="none"))
        ax.text(0.054, y - 0.035, num, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(0.082, y - 0.018, bold_txt, va="top", fontsize=10, fontweight="bold", color=C_DARK)
        ax.text(0.082, y - 0.038, _wrap(body_txt, 88), va="top", fontsize=9, color=C_GREY, linespacing=1.3)
        y -= 0.075

    kpis = [
        ("Phase 1 Result",    "All Negative",    C_RED),
        ("H4 Directional\nAggregate OOS", "+$819K", C_GRN),
        ("Perm p-value",      "0.0000",           C_GRN),
        ("Pairs Validated",   "12 / 12",          C_GRN),
        ("Current Live",      "ZR-Random\nCHF_JPY", C_BLU),
    ]
    bw = 0.163
    for i, (lbl, val, c) in enumerate(kpis):
        xp = 0.04 + i*(bw + 0.010)
        ax.add_patch(FancyBboxPatch((xp, 0.005), bw, 0.100,
                      boxstyle="round,pad=0.008", fc=c+"22", ec=c, lw=1.5))
        ax.text(xp+bw/2, 0.068, val, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_DARK)
        ax.text(xp+bw/2, 0.022, lbl, ha="center", va="center",
                fontsize=9, color=C_GREY)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# NEW PAGE: ZR MECHANICS
# ══════════════════════════════════════════════════════════════════════════════
def p_zr_mechanics():
    fig = plt.figure(figsize=(8.5, 11)); fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.905), 0.94, 0.080,
                  boxstyle="round,pad=0.01", fc=C_DARK, ec="none"))
    ax.text(0.50, 0.946, "ZR Mechanics — Anatomy of a Cycle", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white")

    steps = [
        ("① Primary Leg Enters",
         "A LONG or SHORT position (the 'primary leg') opens at a zone boundary. Entry logic varies by variant: "
         "H4 TopsBots → LONG at confirmed H4 support, SHORT at resistance (mean-reversion). Random N-bar → "
         "alternate L/S every N minutes, no signal. P&F → enter at each Point & Figure column reversal."),
        ("② Optional Trail Phase (single leg only)",
         "While only one leg is open, an optional trailing stop is active: if MFE ≥ ta pips, lock profit at "
         "peak − td pips. If trail fires before the zone is crossed, the cycle closes with one leg at a "
         "profit — the most common outcome in H4 directional ZR (96.5% of all cycles are 1-leg exits)."),
        ("③ Zone Crossed → Hedge Leg Added",
         "If price moves ZW pips against the primary, a second position opens in the opposite direction. "
         "Break-even sizing: volume = ceil(−net_pips / tgt × PF). PF=1.25 means we target 25% above "
         "break-even. The combined open P&L is now negative, but closing both at tgt will produce a net gain."),
        ("④ Additional Legs (recovery grid)",
         "Each subsequent zone crossing adds another leg, again break-even sized for the full cumulative "
         "deficit. With max_legs=10 as circuit-breaker, the worst case requires capital for 10 simultaneous "
         "positions. In practice this fires once in 5 years of GBP_JPY data (0.05% of cycles)."),
        ("⑤ Escape Target Reached — All Legs Close",
         "When price reaches tgt_frac × ZW beyond the entry boundary, ALL legs close simultaneously. "
         "The net P&L is guaranteed ≥ 0 by the break-even sizing formula. A new primary leg opens "
         "immediately in the alternating direction, recycling capital and beginning the next cycle."),
    ]
    y = 0.892
    for title, body in steps:
        ax.text(0.04, y, title, fontsize=9, fontweight="bold", color=C_BLU, va="top")
        ax.text(0.04, y - 0.023, _wrap(body, 65), fontsize=9, color=C_DARK,
                va="top", linespacing=1.35)
        y -= 0.117

    # Price path schematic
    ax_d = fig.add_axes([0.545, 0.290, 0.415, 0.600])
    ax_d.set_facecolor("white")
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
    ax_d.set_title("ZR Cycle — Schematic Price Path", fontsize=9.5,
                   fontweight="bold", color=C_DARK, pad=5)
    t = np.arange(21)
    price = np.array([1.0000,1.0003,1.0007,1.0005,1.0000,
                      0.9998,0.9993,0.9985,0.9970,0.9963,
                      0.9960,0.9968,0.9975,0.9983,0.9990,
                      0.9995,0.9998,1.0003,1.0008,1.0010,1.0010])
    ax_d.plot(t, price, color=C_BLU, lw=2.2, zorder=3)
    ax_d.fill_between(t, price, 0.9955, where=(price < 1.0000), color=C_RED, alpha=0.07)
    ax_d.fill_between(t, price, 1.0012, where=(price >= 1.0010), color=C_GRN, alpha=0.15)
    ax_d.axhline(1.0000, color=C_DARK, lw=1.4, ls="--", alpha=0.7)
    ax_d.axhline(0.9960, color=C_RED,  lw=1.4, ls="--", alpha=0.7)
    ax_d.axhline(1.0010, color=C_GRN,  lw=1.4, ls="--", alpha=0.7)
    ax_d.text(20.4, 1.0000, "Entry\nboundary",  va="center", fontsize=6.5, color=C_DARK)
    ax_d.text(20.4, 0.9960, "Zone bottom\n(−ZW=40p)", va="center", fontsize=6.5, color=C_RED)
    ax_d.text(20.4, 1.0010, "Escape target\n(+10p = 25%×ZW)", va="center", fontsize=6.5, color=C_GRN)
    ax_d.annotate("① LONG\nopens", xy=(0, 1.0000), xytext=(1.2, 1.0018), fontsize=6.5, color=C_GRN,
                  arrowprops=dict(arrowstyle="->", color=C_GRN, lw=0.8))
    ax_d.annotate("③ HEDGE SHORT\nadded", xy=(9, 0.9963), xytext=(3.5, 0.9956), fontsize=6.5, color=C_RED,
                  arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))
    ax_d.annotate("⑤ ALL close\nat target", xy=(19, 1.0010), xytext=(13.5, 1.0017), fontsize=6.5, color=C_GRN,
                  arrowprops=dict(arrowstyle="->", color=C_GRN, lw=0.8))
    ax_d.set_xlim(-0.5, 25.5); ax_d.set_ylim(0.9945, 1.0025)
    ax_d.set_xlabel("Time (M5 bars)", fontsize=8, color=C_GREY)
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v-1)*10000:+.0f}p"))
    ax_d.tick_params(colors=C_GREY, labelsize=7)

    # Parameters box
    params = [
        ("ZW",       "Zone Width (pips). Distance between entry boundary and opposite boundary. Dynamic: H4 swing high − low. Fixed: e.g. 40p."),
        ("tgt_frac", "Escape target = tgt_frac × ZW beyond entry boundary. 0.25 = target is 25% of ZW past the boundary."),
        ("PF",       "Profit Factor markup on break-even sizing. PF=1.25 → target 25% above mathematical break-even."),
        ("ta / td",  "Trail activation (ta pips MFE) and trail distance (td pips behind peak). First leg only."),
        ("B",        "Base units: primary leg size. B=20 OANDA units ≈ $0.002/pip. Recovery legs are multiples of B."),
        ("ML",       "max_legs: circuit-breaker. Force-close all at market if ML legs open without hitting target."),
        ("N",        "Random entry interval in M5 bars. N=1 = every 5-min bar. Alternates L/S each cycle."),
    ]
    ax_p = fig.add_axes([0.04, 0.010, 0.92, 0.260])
    ax_p.axis("off"); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1)
    ax_p.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                    fc=C_DARK+"18", ec=C_DARK, lw=1.2))
    ax_p.text(0.5, 0.955, "KEY PARAMETERS", ha="center", va="top",
              fontsize=9, fontweight="bold", color=C_DARK)
    rh = 0.82 / len(params)
    for i, (name, defn) in enumerate(params):
        yp = 0.88 - i * rh
        col = 0 if i < 4 else 1
        row = i if i < 4 else i - 4
        xb = 0.01 if col == 0 else 0.52
        yp = 0.88 - row * (0.88 / 4 if col == 0 else 0.88 / 3)
        ax_p.text(xb + 0.005, yp, name + ":", fontsize=9, fontweight="bold",
                  color=C_BLU, va="top")
        ax_p.text(xb + 0.08, yp, _wrap(defn, 55), fontsize=9, color=C_DARK,
                  va="top", linespacing=1.25)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# NEW PAGE: GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════
def p_glossary():
    fig = plt.figure(figsize=(8.5, 11)); fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.905), 0.94, 0.080,
                  boxstyle="round,pad=0.01", fc=C_DARK, ec="none"))
    ax.text(0.50, 0.946, "Glossary — Technical Terms, Shorthand & Abbreviations",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    terms = [
        ("ZR / Zone Recovery", "The hedge-grid system. Also 'Ping-Pong'. Adverse zone crossing triggers a counter-position sized to break even at an escape target."),
        ("E/Z Ratio", "Escape-to-Zone ratio: target_pips / zone_width_pips. Classic cBot E/Z=0.29 (catastrophic). ATR-calibrated optimal: 8–10. Boundary mode optimal: 0.5–2.0."),
        ("ZW", "Zone Width in pips. Dynamic: H4 swing high − low (avg 103p). Fixed: chosen constant (e.g. 40p)."),
        ("tgt / tgt_frac", "Escape target. tgt_frac=0.25 → target is 25% of ZW beyond entry boundary."),
        ("PF", "Profit Factor: break-even sizing multiplier. PF=1.25 → position sized to net 25% above mathematical break-even at target."),
        ("ML / max_legs", "Maximum legs before force-close (circuit-breaker). Fires in <0.05% of H4-dir cycles over 5 years."),
        ("ta / td", "Trail activation (ta pips MFE) and trail distance (td pips behind peak). First-leg trailing stop only."),
        ("B (Base Units)", "Primary leg size in OANDA units. B=20 ≈ $0.002/pip. Recovery legs are multiples of B by break-even formula."),
        ("N (entry interval)", "Random ZR: open primary leg every N M5 bars. N=1 = every 5 minutes. Alternates LONG/SHORT each cycle."),
        ("IS / OOS", "In-Sample (70%, used for parameter search) / Out-of-Sample (30%, held out, never touched during tuning)."),
        ("WF (Walk-Forward)", "WF score: number of consecutive IS sub-periods that are individually profitable. WF=3 = strictest filter."),
        ("Sharpe Ratio", "Mean daily P&L / std dev, annualised. >0.1 good, >0.3 excellent for FX strategies."),
        ("SQN", "System Quality Number (Van Tharp): mean_trade / std_trade × √N. >1.0 tradeable, >5.0 excellent, >10.0 world-class."),
        ("Perm p / perm_p", "Permutation test p-value: fraction of 2,000 timing-shuffled runs that beat the observed result. p<0.05 required."),
        ("Boot p5", "5th percentile of 2,000 bootstrap resamples of per-cycle P&L. boot p5 > 0 required (robustly positive)."),
        ("P(+)", "Bootstrap probability that p/d > 0. Gate: P(+) > 0.95."),
        ("Gates", "Count of statistical validation gates passed (out of 3 or 5 depending on test)."),
        ("TopsBots", "Causal swing detector. H1/H4 swing confirmed at bar N−1 when bar N closes past it. Zero lookahead."),
        ("H4 S/R", "Support/Resistance from H4 OHLC swing highs/lows via TopsBots. LONG at support, SHORT at resistance."),
        ("P&F (Point & Figure)", "Time-agnostic chart. X column: price moves box_size pips up. Reversal: price drops rev × box_size pips → new O column."),
        ("ppd / p/d", "Pips per day — throughput metric for P&F and random variants, independent of lot size."),
        ("OOS USD", "Out-of-sample P&L in US dollars at a specified lot size (e.g. 1,000 OANDA units)."),
        ("Calmar", "Return / max drawdown. Higher = better. Measures return per unit of max adverse capital exposure."),
        ("OANDA units", "OANDA position denomination. 1,000 units ≈ 1 micro-lot ≈ $0.10/pip. B=20 units ≈ $0.002/pip."),
        ("acct 011 / 012", "Split-account workaround: 011 holds LONG legs, 012 holds SHORT legs (OANDA prohibits simultaneous hedging)."),
        ("conv. / break-even sizing", "Convex: volume ∝ n^1.5 (momentum pyramid, inflated metrics). Break-even: volume = ceil(−net_pips / tgt × PF) — true ZR recovery logic."),
    ]
    col1 = terms[:len(terms)//2+1]
    col2 = terms[len(terms)//2+1:]
    rh = 0.86 / max(len(col1), len(col2))
    for ci, col_terms in enumerate([col1, col2]):
        xb = 0.04 if ci==0 else 0.535
        xt = 0.155 if ci==0 else 0.645
        for ri, (term, defn) in enumerate(col_terms):
            y = 0.892 - ri * rh
            bg = C_BG if ri % 2 == 0 else "white"
            ax.add_patch(FancyBboxPatch((xb-0.01, y-rh+0.003), 0.455, rh-0.004,
                          boxstyle="square,pad=0", fc=bg, ec="none"))
            ax.text(xb, y-0.004, term+":", fontsize=9, fontweight="bold", color=C_BLU, va="top")
            ax.text(xt, y-0.004, _wrap(defn, 52), fontsize=9, color=C_DARK,
                    va="top", linespacing=1.2)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# NEW PAGE: BOTTOM LINE — DEPLOYMENT DECISION
# ══════════════════════════════════════════════════════════════════════════════
def p_bottom_line():
    fig = plt.figure(figsize=(8.5, 11)); fig_style(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.905), 0.94, 0.080,
                  boxstyle="round,pad=0.01", fc="#1a252f", ec="none"))
    ax.text(0.50, 0.949, "Bottom Line — What Should Run on Accounts 011 / 012?",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.50, 0.917, "Maximising profit given risk, capital constraints, and statistical evidence",
            ha="center", va="center", fontsize=9, color="#bdc3c7")

    ax.text(0.05, 0.893, "WHAT THE RESEARCH SHOWS", fontsize=9, fontweight="bold", color=C_BLU)
    ax.text(0.05, 0.875, _wrap(
        "Across 14 experiment families and 240+ configurations, one signal dominates all others: entering "
        "LONG at H4 TopsBots support and SHORT at H4 resistance. This lifted aggregate 12-pair OOS P&L from "
        "+$43K (random direction) to +$819K — a 19× improvement — with permutation p=0.0000. The signal is "
        "real, causal, and statistically unambiguous. All other variants (random N-bar, P&F, SMA filters) are "
        "approximations of the same mean-reversion phenomenon: entering near levels where price is structurally "
        "likely to reverse. The grid is insurance; the directional entry is the edge.", 88),
        fontsize=10, color=C_DARK, va="top", linespacing=1.4)

    ax.text(0.05, 0.730, "WHY WE'RE RUNNING RANDOM N-BAR (NOT H4 DIRECTIONAL) NOW", fontsize=9, fontweight="bold", color=C_YEL)
    ax.text(0.05, 0.712, _wrap(
        "The H4 directional strategy was live and generating strong results but was replaced by the simpler "
        "random-alternating service because: (1) dynamic ZW (H4 swing range 50–200p) demands larger capital "
        "buffers per leg at USD17 NAV; (2) strategy_zr_random is a single clean service with no TopsBots "
        "dependency, faster to iterate; (3) random alternating still generates ~1,080 p/d on CHF_JPY and "
        "is provably profitable with WF=3. The simplification trades maximum alpha for operational robustness "
        "at micro-capital scale. This is the correct tradeoff until NAV grows.", 88),
        fontsize=10, color=C_DARK, va="top", linespacing=1.4)

    ax.text(0.05, 0.640, "RECOMMENDED DEPLOYMENT ROADMAP  (risk-adjusted, in priority order)", fontsize=9, fontweight="bold", color=C_GRN)
    road = [
        ("NOW  (NAV ~USD17)", C_GRN,
         "Stay on random N-bar CHF_JPY (N=1, ZW=40, ta=5, td=3). It is profitable with 3/3 gates. "
         "UPGRADE immediately: swap entry clock to P&F timing (box=10, rev=3) — 3/3 gates, +1,113 vs "
         "+1,080 p/d, identical risk (same ZW/tgt/B). Only the entry timing changes; no other risk shifts."),
        ("NEAR-TERM  (NAV ≥ $28)", C_BLU,
         "Add GBP_USD (random alternating, N=6, ZW=30, ta=10, td=7 — 812 p/d OOS). At USD28 NAV, "
         "2-leg max margin ≈ 22% — within the 45% OANDA margin gate. Do NOT add before $28."),
        ("NEAR-TERM  (NAV ≥ $40)", C_BLU,
         "Add USD_JPY (N=1, ZW=40, ta=10, td=5 — 647 p/d OOS). P&F b5r2 gives +1,335 p/d "
         "but bootstrap p5 < 0 (fat-tail cycle risk). Use random alternating until live data confirms."),
        ("STRATEGIC  (1–2 weeks dev)", C_PUR,
         "Re-integrate H4 TopsBots directional entry into strategy_zr_random. Run with fixed ZW=40p "
         "(not dynamic) to control margin at current NAV. Expected outcome: 3–5× higher p/d than random "
         "on same pairs. Perm p=0.0000 across 12 pairs — this is the highest-confidence signal we have."),
    ]
    y = 0.622
    for header, c, body in road:
        ax.add_patch(FancyBboxPatch((0.04, y-0.082), 0.92, 0.080,
                      boxstyle="round,pad=0.006", fc=c+"18", ec=c, lw=1.5))
        ax.text(0.055, y-0.012, header, va="top", fontsize=10, fontweight="bold", color=c)
        ax.text(0.055, y-0.033, _wrap(body, 88), va="top", fontsize=9, color=C_DARK, linespacing=1.35)
        y -= 0.088

    ax.text(0.05, 0.200, "RISK GUARD-RAILS", fontsize=9, fontweight="bold", color=C_RED)
    risks = [
        ("Max float at B=20u",  "Random ZW=40p, 2 legs: ~USD0.80 adverse float = 4.7% of USD17 NAV. Acceptable."),
        ("Adding pairs",        "Each pair is independently split across 011/012. No cross-pair margin interaction."),
        ("H4 + fixed ZW=40",    "2-leg max float ≈ USD0.80. Safe at USD17. Dynamic ZW (50–200p) needs NAV ≥ USD50 first."),
        ("P(+) gate",           "CHF_JPY b10r3 and NZD_JPY b5r4 pass boot p5>0. USD_JPY b5r2 fails p5 gate — hold off."),
        ("max_legs=10",         "Worst historical cycle (GBP_JPY): 10 legs at B=20u ≈ USD15 float. Keep B=20u. Do not increase."),
    ]
    y = 0.185
    for rh, rb in risks:
        ax.add_patch(FancyBboxPatch((0.04, y-0.028), 0.92, 0.028,
                      boxstyle="square,pad=0", fc=C_RED+"08", ec="none"))
        ax.text(0.055, y-0.014, rh+":", va="center", fontsize=9, fontweight="bold", color=C_RED)
        ax.text(0.230, y-0.014, rb,    va="center", fontsize=9, color=C_DARK)
        y -= 0.030

    ax.add_patch(FancyBboxPatch((0.03, 0.005), 0.94, 0.040,
                  boxstyle="round,pad=0.005", fc=C_GRN+"33", ec=C_GRN, lw=1.8))
    ax.text(0.50, 0.025,
            "VERDICT:  Upgrade CHF_JPY to P&F b10r3 NOW.  Add GBP_USD at USD28 NAV.  Add USD_JPY at USD40 NAV.  "
            "Target: H4 directional (fixed ZW) once integrated — the unambiguous #1 signal.",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_DARK)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MASTER EXPERIMENT CATALOG
# ══════════════════════════════════════════════════════════════════════════════
def p_catalog_overview():
    """One-page overview table: all 10 experiment families with status."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_DARK, ec="none"))
    ax.text(0.5, 0.955, "Master Experiment Catalog — All ZR Variants",
            ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "10 experiment families · What was swept · Gates applied · Pass/Fail/Live",
            ha="center", va="center", fontsize=9, color="#bdc3c7")

    families = [
        # (Family#, Name, Swept, Gate, n_configs, top_result, status_color, status)
        ("1", "ZR Mechanics\n(Phases 1–9)", "hz, tgt, PF, EZ,\nsizing variants",
         "Sharpe/SQN/WR", "1,400+", "Convex sizing 4× > dynamic\nBoundary entry dominates", C_BLU, "Foundation"),
        ("2", "Per-Pair\nRandom IS/OOS", "8 pairs × ZW ×\nN × tf",
         "wf=3 only", "12 best", "CHF_JPY 358 p/d\nNZD_JPY 137 p/d", C_YEL, "Baseline"),
        ("3", "Signal-Filtered\nEntry", "H1/M5 SMA, ATR\n3 pairs × 31 signals",
         "wf=3 only", "93", "Random beats ALL filters\nfor CHF_JPY (1,080 > 1,292 IS)", C_YEL, "Rejected"),
        ("4", "Trail Exit\nSweep (coarse)", "ta={5,7,10,14,20,30}\ntd={3,5,7,10}", "wf=3 only",
         "57", "ta=5,td=3=1,080 p/d\nta=7,ta=20: −18K CLIFF", C_YEL, "Partial"),
        ("5", "Fine Trail\n+ Bootstrap", "ta={2..30}\ntd={1..10}",
         "wf=3+P5>0+P(+)>95%", "45", "ta=5,td=1: 1,118 p/d P5=+464\nta=6,td=5: 1,659 p/d P5=+255", C_GRN, "🟢 LIVE"),
        ("6", "P&F Rev\nSweep", "10 pairs × box{5,10,15}\n× rev{1–3} × ALT/COL",
         "wf=3 only", "300", "AUD_JPY b15r2 COL 1,470 p/d\nNZD_JPY b5r4 ALT 1,207 p/d", C_YEL, "Screener"),
        ("7", "P&F Permtest\n+ Bootstrap MC", "20 best P&F configs\nfull 3-gate validation",
         "perm p<0.05\nP5>0, P(+)>95%", "20", "12/20 fully 3/3 validated\nNZD_JPY/AUD_JPY/EUR_JPY/GBP_USD", C_GRN, "Validated"),
        ("8", "Hybrid Fallback\n(P&F + Random)", "4 pairs × n_fb{1,5,12,24,48}\n× override thresh",
         "wf=3 only", "88", "Never strongly beats pure P&F\nShort fallback catastrophic", C_RED, "Rejected"),
        ("9", "Random Trail\nBootstrap MC", "CHF/NZD/USD_JPY\nOOS cycle P&L resample",
         "P5>0, P(+)>95%", "3 pairs", "CHF_JPY 2/2 PASS P5=+409\nNZD_JPY 0/2 FAIL P5=−2", C_GRN, "Validated"),
        ("10", "H4 Directional\n(Multi-TF)", "H1/H2/H4 S/R bias\n12 pairs",
         "perm p=0.000\n0/2000 shuffles", "4 TFs", "H4: +$919K vs −$578K base\np=0.0000 permtest", C_GRN, "Was live"),
        ("11", "All-Pairs Random\nTrail Sweep", "9 pairs × ta{2..30}\n× td{1..7}",
         "wf=3+P5>0+P(+)>95%", "378", "EUR_JPY ta=30: 2,364 p/d P5=+408\nAUD_JPY ta=4: 1,599 p/d 12c/day", C_GRN, "Validated"),
        ("12", "P&F Hi-Freq\nPermtest", "66 configs wf=3\nb∈{5,10,15} r≤2.5",
         "perm p<0.05\nP5>0, P(+)>95%", "66", "0/66 passed 3/3 — perm fails all\nEdge in ZR mechanics, NOT P&F timing", C_RED, "Negative"),
    ]

    y = 0.88
    row_h = 0.082
    cols = [0.00, 0.04, 0.19, 0.36, 0.52, 0.62, 0.82, 0.93]
    hdrs = ["#", "Family", "Swept", "Gate", "N", "Top result", "Status"]
    for ci, (hdr, x) in enumerate(zip(hdrs, cols)):
        ax.text(x + 0.005, y + 0.005, hdr, fontsize=7.5, fontweight="bold",
                color=C_DARK, va="center")
    ax.axhline(y - 0.008, color=C_GREY, lw=0.5, xmin=0, xmax=1)
    y -= 0.015

    for i, (num, name, swept, gate, n, result, sc, status) in enumerate(families):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - row_h + 0.004), 1, row_h,
                                    fc=bg, ec="none", zorder=0))
        cy = y - row_h / 2 + 0.004
        ax.text(cols[0] + 0.005, cy, num, fontsize=8, va="center", color=C_GREY)
        ax.text(cols[1] + 0.005, cy, name, fontsize=7.5, va="center", color=C_DARK)
        ax.text(cols[2] + 0.005, cy, swept, fontsize=6.5, va="center", color=C_GREY)
        ax.text(cols[3] + 0.005, cy, gate, fontsize=6.5, va="center", color=C_GREY)
        ax.text(cols[4] + 0.005, cy, n, fontsize=7.5, va="center", color=C_DARK, ha="center")
        ax.text(cols[5] + 0.005, cy, result, fontsize=6.5, va="center", color=C_DARK)
        ax.add_patch(FancyBboxPatch((cols[6] + 0.002, cy - 0.018), 0.10, 0.036,
                      boxstyle="round,pad=0.003", fc=sc, ec="none", alpha=0.85))
        ax.text(cols[6] + 0.052, cy, status, fontsize=6.5, va="center",
                ha="center", color="white", fontweight="bold")
        y -= row_h

    # Legend
    ax.axhline(y - 0.005, color=C_GREY, lw=0.4, xmin=0, xmax=1)
    y -= 0.025
    ax.text(0.01, y, "Gate hierarchy:", fontsize=8, fontweight="bold", color=C_DARK)
    ax.text(0.01, y - 0.022,
            "wf=3 only → wf=3 + bootstrap (P5>0 + P(+)>95%) → wf=3 + permutation test (p<0.05) + bootstrap = full 3/3",
            fontsize=7.5, color=C_GREY)
    ax.text(0.01, y - 0.044,
            "Live: CHF_JPY random-alt ta=5 td=1 ZW=40 | Top random: EUR_JPY 2,364 p/d · AUD_JPY 1,599 p/d · NZD_JPY 752 p/d",
            fontsize=7.5, color=C_DARK, fontstyle="italic")

    return fig


def p_catalog_validated():
    """All fully validated configs across all families — the deployable universe."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_DARK, ec="none"))
    ax.text(0.5, 0.955, "Validated Strategy Universe — All Configs Passing All Gates",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "Sorted by OOS p/d. Random-trail: wf=3+boot. P&F: wf=3+perm+boot.",
            ha="center", va="center", fontsize=9, color="#bdc3c7")

    rows = [
        # (Fam, Pair, Entry, Config, ppd, c/day, P5, P(+), Gates, Live)
        ("11", "EUR_JPY", "Random-alt", "ta=30 td=3 ZW=50 tgt=25",      2364,  3.8, "+408", "100%", "wf+boot", "🔵pending"),
        ("11", "AUD_JPY", "Random-alt", "ta=4  td=1 ZW=50 tgt=25",      1599, 12.0, "+123", "100%", "wf+boot", "🔵pending"),
        ("5",  "CHF_JPY", "Random-alt", "ta=6  td=5 ZW=40 tgt=20",      1659, 16.8, "+255", "100%", "wf+boot", "⚠️cliff"),
        ("5",  "CHF_JPY", "Random-alt", "ta=5  td=1 ZW=40 tgt=20",      1118, 20.1, "+464", "100%", "wf+boot", "🟢LIVE"),
        ("7",  "AUD_JPY", "P&F b15r2",  "COL ZW=50 tgt=25 ta=5 td=3",  1470,  2.5, "+18",  "100%", "3/3", ""),
        ("7",  "AUD_JPY", "P&F b10r3",  "COL ZW=50 tgt=25 ta=5 td=3",  1467,  2.8, "+19",  "100%", "3/3", ""),
        ("7",  "NZD_JPY", "P&F b5r4",   "ALT ZW=40 tgt=20 ta=5 td=3",  1207,  3.8, "+43",  "100%", "3/3", "🔵pending"),
        ("11", "NZD_JPY", "Random-alt", "ta=2  td=1 ZW=40 tgt=20",       752, 19.4, "+170", "100%", "wf+boot", "🔵pending"),
        ("11", "EUR_JPY", "Random-alt", "ta=3  td=1 ZW=50 tgt=25",       767, 26.8, "+273", "100%", "wf+boot", "🔵alt"),
        ("7",  "NZD_JPY", "P&F b5r4",   "COL ZW=40 tgt=20 ta=5 td=3",  1134,  3.6, "+31",  "100%", "3/3", ""),
        ("7",  "NZD_JPY", "P&F b10r3",  "ALT ZW=40 tgt=20 ta=5 td=3",  1120,  2.1, "+19",  "100%", "3/3", ""),
        ("7",  "NZD_JPY", "P&F b10r3",  "COL ZW=40 tgt=20 ta=5 td=3",  1108,  2.1, "+20",  "100%", "3/3", ""),
        ("7",  "NZD_JPY", "P&F b10r4",  "COL ZW=40 tgt=20 ta=5 td=3",  1089,  1.4, "+8",   "100%", "3/3", ""),
        ("7",  "NZD_JPY", "P&F b15r2.5","ALT ZW=40 tgt=20 ta=5 td=3",  1087,  1.5, "+5",   "99.8%","3/3", ""),
        ("9",  "USD_JPY", "Random-alt", "ta=10 td=5 ZW=40 tgt=20",       591, 10.9, "+179", "100%", "wf+boot", ""),
        ("11", "USD_JPY", "Random-alt", "ta=7  td=1 ZW=40 tgt=20",       373, 15.0, "+198", "100%", "wf+boot", "🔵pending"),
        ("7",  "EUR_JPY", "P&F b5r4",   "COL ZW=50 tgt=25 ta=5 td=3",   534,  7.5, "+106", "100%", "3/3", "🔵pending"),
        ("7",  "EUR_JPY", "P&F b5r4",   "ALT ZW=50 tgt=25 ta=5 td=3",   511,  7.6, "+96",  "100%", "3/3", ""),
        ("7",  "AUD_JPY", "P&F b5r2",   "ALT ZW=50 tgt=25 ta=5 td=3",   495,  7.3, "+80",  "100%", "3/3", ""),
        ("11", "CAD_JPY", "Random-alt", "ta=2  td=1 ZW=50 tgt=12.5",     295, 25.6, "+89",  "100%", "wf+boot", "🔵pending"),
        ("7",  "GBP_USD", "P&F b5r2",   "COL ZW=30 tgt=15 ta=10 td=7",  264,  5.2, "+60",  "100%", "3/3", "🔵pending"),
        ("11", "GBP_USD", "Random-alt", "ta=5  td=1 ZW=30 tgt=15",       240, 13.0, "+130", "100%", "wf+boot", "🔵pending"),
        ("11", "AUD_USD", "Random-alt", "ta=10 td=5 ZW=30 tgt=15",       168,  3.4, "+59",  "100%", "wf+boot", ""),
        ("11", "NZD_USD", "Random-alt", "ta=6  td=5 ZW=25 tgt=12.5",      98,  4.7, "+46",  "100%", "wf+boot", ""),
        ("10", "12-pair",  "H4 TopsBots","directional ZW=56 tgt=0.25×ZW",None, 0.7, "—",    "—",    "perm p=0", "⏹stopped"),
    ]

    y = 0.88
    row_h = 0.032
    cx = [0.00, 0.03, 0.11, 0.19, 0.37, 0.47, 0.53, 0.60, 0.67, 0.76, 0.86]
    hdrs = ["F", "Pair", "Entry", "Config", "p/d", "c/d", "P5", "P(+)", "Gates", "Status"]
    for hdr, x in zip(hdrs, cx):
        ax.text(x + 0.002, y + 0.003, hdr, fontsize=6.5, fontweight="bold",
                color=C_DARK, va="center")
    ax.axhline(y - 0.005, color=C_GREY, lw=0.5)
    y -= 0.010

    for i, (fam, pair, entry, cfg, ppd, cday, p5, ppos, gates, status) in enumerate(rows):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - row_h + 0.001), 1, row_h,
                                    fc=bg, ec="none", zorder=0))
        cy = y - row_h / 2 + 0.001

        sc = C_GRN if "LIVE" in status else (C_BLU if "pending" in status or "alt" in status else
             (C_GREY if "stopped" in status else (C_YEL if "cliff" in status else "none")))
        ppd_str = f"{ppd:,}" if ppd else "—"
        vals = [fam, pair, entry, cfg, ppd_str, f"{cday}", p5, ppos, gates, status]
        for vi, (val, x) in enumerate(zip(vals, cx)):
            color = C_DARK
            if vi == 4 and ppd:
                color = C_GRN if ppd > 800 else C_DARK
            ax.text(x + 0.002, cy, str(val), fontsize=5.8, va="center", color=color)
        if status and sc != "none":
            ax.add_patch(FancyBboxPatch((cx[9] + 0.001, cy - 0.010), 0.10, 0.020,
                          boxstyle="round,pad=0.002", fc=sc, ec="none", alpha=0.8))
            ax.text(cx[9] + 0.051, cy, status, fontsize=5.0, va="center",
                    ha="center", color="white", fontweight="bold")
        y -= row_h

    ax.axhline(y - 0.001, color=C_GREY, lw=0.4)
    y -= 0.014
    ax.text(0.01, y,
            "⚠️ ta=6,td=5 CHF_JPY omitted — sits 1 pip below catastrophic ta=7 cliff (−18K p/d). Shadow-validate before live use.",
            fontsize=6.5, color=C_YEL)
    ax.text(0.01, y - 0.016,
            "P&F hi-freq configs (b≤10, r≤2.5): 0/66 passed permtest — P&F timing adds NO edge over random for fast reversals.",
            fontsize=6.5, color=C_RED)
    ax.text(0.01, y - 0.032,
            "Aggregate potential (top 7 random-trail pairs): EUR_JPY+AUD_JPY+CHF_JPY+NZD_JPY+USD_JPY+CAD_JPY+GBP_USD = 6,741 p/d at 1 unit each.",
            fontsize=6.5, color=C_GRN)

    return fig


def p_catalog_failed():
    """Configs that passed wf=3 but failed permtest or bootstrap — and why."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc="#7f0000", ec="none"))
    ax.text(0.5, 0.955, "Failed / Partially Validated Configs — wf=3 But Rejected",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "Passed walk-forward but failed permutation test or bootstrap. Do NOT deploy.",
            ha="center", va="center", fontsize=9, color="#ffcccc")

    failed = [
        # (Fam, Pair, Entry, Config, ppd, gates, failure_reason)
        ("7", "USD_JPY", "P&F b5r2", "ALT ZW=40 ta=10 td=5", 1335, "1/3",
         "perm PASS p=0.006 BUT boot P5=−62, P(+)=0.928. High c/day (8.3) inflates mean; fat left tail."),
        ("7", "CHF_JPY", "P&F b10r3","ALT ZW=40 ta=5 td=3",  1113, "1/3",
         "perm PASS p=0.031 BUT boot P5=−16, P(+)=0.921. Same 1/3 as b15r1.5. Random entry dominates."),
        ("7", "CHF_JPY", "P&F b15r1.5","ALT ZW=40 ta=5 td=3",1163, "1/3",
         "perm PASS p=0.020 BUT boot P5=−10, P(+)=0.935. Slightly better than b10r3 but same gate count."),
        ("7", "CHF_JPY", "P&F b5r3", "COL ZW=40 ta=5 td=3",   680, "2/3",
         "perm FAIL p=0.076. Too frequent (10.5 c/day) — null distribution wide. boot P5=+249 strong."),
        ("7", "CHF_JPY", "P&F b5r2", "COL ZW=40 ta=5 td=3",   652, "2/3",
         "perm FAIL p=0.095. Same as above — r=2 at b=5 still too frequent for permtest to separate."),
        ("7", "EUR_JPY", "P&F b5r1.5","ALT ZW=50 ta=5 td=3",  557, "2/3",
         "perm FAIL p=0.065. 5,232 cycles — null median 169 p/d, too close to observed."),
        ("7", "EUR_JPY", "P&F b5r1.5","COL ZW=50 ta=5 td=3",  553, "2/3",
         "perm FAIL p=0.074. Same reasoning as ALT above."),
        ("7", "AUD_JPY", "P&F b5r3", "ALT ZW=50 ta=5 td=3",   426, "2/3",
         "perm FAIL p=0.116. Not significant."),
        ("9", "NZD_JPY", "Random-alt","ta=5 td=3 ZW=40 tgt=20", 72, "0/2",
         "boot P5=−2, P(+)=94.6%. Random entry has no edge for NZD_JPY — P&F timing IS the edge."),
        ("3", "CHF_JPY", "m5_sma10", "filter ZW=40 ta=5 td=3", 1292, "IS only",
         "IS-optimistic: OOS A/B vs random = 111 p/d vs 1,080 p/d. Signal filter destroys complementarity."),
        ("4", "CHF_JPY", "Random-alt","ta=7 any-td ZW=40",    -18000, "wf=2",
         "Catastrophic cliff. ta=7 resonates with CHF_JPY M5 volatility. ALL td values fail."),
        ("4", "CHF_JPY", "Random-alt","ta=20 any-td ZW=40",   -18000, "wf=2+",
         "Second catastrophic zone. ta=20 wf=2 regardless of td."),
        ("12", "ALL PAIRS", "P&F hi-freq","b∈{5,10,15} r≤2.5 (66 configs)", None, "0/3",
         "0/66 passed perm test. P5>0 and P(+)=100% on all 62/66 (ZR mechanics profitable). "
         "P&F timing itself adds NO edge — entry timing is irrelevant. Random entry is optimal."),
    ]

    y = 0.88
    row_h = 0.062
    for i, (fam, pair, entry, cfg, ppd, gates, reason) in enumerate(failed):
        bg = "#fff5f5" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - row_h + 0.002), 1, row_h,
                                    fc=bg, ec="none", zorder=0))
        cy = y - row_h / 2 + 0.005
        gc = C_RED if gates in ("1/3","0/2","wf=2","wf=2+","IS only") else C_YEL
        ax.text(0.005, cy + 0.012, f"F{fam} · {pair} · {entry} · {cfg}",
                fontsize=8, fontweight="bold", color=C_DARK, va="center")
        ppd_str = f"{ppd:,}" if ppd is not None else "—"
        ax.text(0.005, cy - 0.010, f"ppd={ppd_str}  gates={gates}",
                fontsize=7.5, color=gc, va="center")
        wrapped = textwrap.fill(reason, width=110)
        ax.text(0.005, cy - 0.028, wrapped, fontsize=6.5, color=C_GREY, va="center")
        y -= row_h

    return fig


def p_allpairs_sweep():
    """All-pairs random trail sweep — best validated config per pair + key insight."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_DARK, ec="none"))
    ax.text(0.5, 0.955, "Family 11: All-Pairs Random Trail Sweep",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "ta ∈ {2,3,4,5,6,7,8,10,14,20,30} × td ∈ {1,2,3,5,7} — 9 pairs — 378 configs total",
            ha="center", va="center", fontsize=9, color="#bdc3c7")

    # Summary insight box
    y = 0.88
    ax.add_patch(FancyBboxPatch((0.01, 0.80), 0.98, 0.075,
                  boxstyle="round,pad=0.005", fc="#eafaf1", ec=C_GRN, lw=1.5))
    ax.text(0.5, 0.862, "KEY FINDING: Random alternating entry is validated on ALL 9 pairs.",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C_GRN)
    ax.text(0.5, 0.838, "189 configs pass wf=3 + P5>0 + P(+)≥99.9%. Aggregate potential: 6,741 p/d across 7 top pairs at 1 unit each.",
            ha="center", va="center", fontsize=8.5, color=C_DARK)
    ax.text(0.5, 0.817, "P&F timing tested on 66 hi-freq configs: 0/66 pass permtest. The edge is in ZR mechanics, not entry timing.",
            ha="center", va="center", fontsize=8.5, color=C_RED)

    y = 0.79
    # Table: best per pair
    cols = [0.01, 0.12, 0.22, 0.30, 0.38, 0.50, 0.60, 0.70, 0.78, 0.88]
    hdrs = ["Pair", "ZW/tgt", "Best ta", "Best td", "p/d", "c/day", "ppc", "P5", "P(+)", "Deploy"]
    for hdr, x in zip(hdrs, cols):
        ax.text(x, y, hdr, fontsize=8, fontweight="bold", color=C_DARK, va="center")
    ax.axhline(y - 0.008, color=C_GREY, lw=0.5)
    y -= 0.025

    data = [
        # (pair, zw_tgt, ta, td, ppd, cday, ppc, p5, ppos, priority)
        ("EUR_JPY", "50/25", 30, 3, 2364,  3.8, 622, 408, "100%", "HIGH"),
        ("AUD_JPY", "50/25",  4, 1, 1599, 12.0, 133, 123, "100%", "HIGH"),
        ("CHF_JPY", "40/20",  5, 1, 1118, 20.1,  54, 464, "100%", "🟢LIVE"),
        ("NZD_JPY", "40/20",  2, 1,  752, 19.4,  39, 170, "100%", "HIGH"),
        ("EUR_JPY*","50/25",  3, 1,  767, 26.8,  29, 273, "100%", "ALT"),
        ("USD_JPY", "40/20",  7, 1,  373, 15.0,  25, 198, "100%", "HIGH"),
        ("CAD_JPY", "50/12",  2, 1,  295, 25.6,  12,  89, "100%", "MED"),
        ("GBP_USD", "30/15",  5, 1,  240, 13.0,  18, 130, "100%", "MED"),
        ("AUD_USD", "30/15", 10, 5,  168,  3.4,  49,  59, "100%", "LOW"),
        ("NZD_USD", "25/12",  6, 5,   98,  4.7,  21,  46, "100%", "LOW"),
        ("EUR_GBP", "40/20",  3, 2,   95,  2.9,  32,  14, "100%", "SKIP"),
    ]

    row_h = 0.050
    for i, (pair, zwtgt, ta, td, ppd, cday, ppc, p5, ppos, priority) in enumerate(data):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - row_h + 0.004), 1, row_h,
                                    fc=bg, ec="none", zorder=0))
        cy = y - row_h / 2 + 0.004
        pc = {"HIGH": C_GRN, "🟢LIVE": C_GRN, "ALT": C_BLU, "MED": C_YEL,
              "LOW": C_GREY, "SKIP": C_RED}.get(priority, C_DARK)
        vals = [pair, zwtgt, str(ta), str(td), f"{ppd:,}", f"{cday}", f"{ppc}", f"+{p5}", ppos]
        for vi, (val, x) in enumerate(zip(vals, cols)):
            c = C_GRN if vi == 4 and ppd > 800 else C_DARK
            ax.text(x, cy, val, fontsize=8, va="center", color=c)
        ax.add_patch(FancyBboxPatch((cols[9], cy - 0.012), 0.095, 0.024,
                      boxstyle="round,pad=0.003", fc=pc, ec="none", alpha=0.85))
        ax.text(cols[9] + 0.048, cy, priority, fontsize=6.5, va="center",
                ha="center", color="white", fontweight="bold")
        y -= row_h

    # Fast-cycler insight
    y -= 0.010
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.020
    ax.text(0.5, y, "★ FASTEST CYCLERS (≥15 c/day, all validated)",
            ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    y -= 0.025
    fast = [
        "GBP_USD ta=2: 28.4 c/day · 163 p/d · P5=+124",
        "EUR_JPY ta=3: 26.8 c/day · 767 p/d · P5=+273",
        "CAD_JPY ta=2: 25.6 c/day · 295 p/d · P5=+89",
        "AUD_JPY ta=2: 22.1 c/day · 263 p/d · P5=+130",
        "NZD_JPY ta=2: 19.4 c/day · 752 p/d · P5=+170",
        "USD_JPY ta=7: 15.0 c/day · 373 p/d · P5=+198",
    ]
    for j, line in enumerate(fast):
        ax.text(0.1 + (j % 2) * 0.45, y - (j // 2) * 0.028, f"● {line}",
                fontsize=8, color=C_DARK)
    y -= 0.090

    # Note on EUR_JPY
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.018
    ax.text(0.01, y,
            "* EUR_JPY appears twice: ta=30 (2,364 p/d, 3.8 c/day, ppc=623) vs ta=3 (767 p/d, 26.8 c/day, ppc=29). "
            "Different profiles: ta=30 is infrequent with huge multi-leg ZR payouts; ta=3 is fast small-trail exits.",
            fontsize=7, color=C_GREY, wrap=True)

    return fig


def p_spread_sensitivity():
    """Spread sensitivity — TOD model results + gate analysis."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_DARK, ec="none"))
    ax.text(0.5, 0.958, "Spread Sensitivity Analysis — Time-of-Day (TOD) Model",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "Session 026 · Replaces volume-based model (calibration artifacts). "
            "TOD: dead zone (21-22h) → 2.7p · Lon/NY (13-15h) → 1.26p",
            ha="center", va="center", fontsize=8.5, color="#bdc3c7")

    # TOD profile mini bar chart
    ax2 = fig.add_axes([0.04, 0.79, 0.45, 0.10])
    tod = [2.24,2.17,2.10,2.03,1.96,1.96,1.82,1.68,1.47,1.40,1.40,1.40,
           1.33,1.26,1.26,1.26,1.33,1.47,1.54,1.68,1.82,2.66,2.73,2.45]
    colors = [C_RED if v > 2.5 else (C_YEL if v > 1.6 else C_GRN) for v in tod]
    ax2.bar(range(24), tod, color=colors, width=0.8, edgecolor='none')
    ax2.axhline(1.4, color=C_GREY, lw=0.8, ls='--', label='1.4p baseline')
    ax2.axhline(2.5, color=C_RED, lw=0.8, ls='--', label='gate threshold')
    ax2.set_xlim(-0.5, 23.5); ax2.set_ylim(1.0, 3.0)
    ax2.set_xticks([0,6,12,18,23]); ax2.set_xticklabels(['00h','06h','12h','18h','23h'], fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.set_title("TOD Spread Profile (UTC hour)", fontsize=8, color=C_DARK)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=6, loc='upper left')

    # Legend box
    ax3 = fig.add_axes([0.52, 0.79, 0.45, 0.10])
    ax3.axis('off')
    entries = [
        (C_GRN, "✅ robust: |Δ| < 5% and neg% < 3%"),
        (C_YEL, "⚠️ mild: |Δ| < 15%"),
        (C_RED, "❌ sensitive: |Δ| ≥ 15%"),
    ]
    for i, (c, txt) in enumerate(entries):
        ax3.add_patch(FancyBboxPatch((0.02, 0.70 - i*0.28), 0.12, 0.18,
                       boxstyle="round,pad=0.02", fc=c, ec="none", alpha=0.85,
                       transform=ax3.transAxes))
        ax3.text(0.17, 0.79 - i*0.28, txt, fontsize=8, color=C_DARK,
                 va='center', transform=ax3.transAxes)

    y = 0.76
    cols = [0.01, 0.12, 0.19, 0.26, 0.37, 0.49, 0.62, 0.73, 0.82, 0.91]
    hdrs = ["Pair", "ta", "td", "const p/d", "tod p/d", "gate p/d", "skip%", "neg%", "verdict", ""]
    for hdr, x in zip(hdrs, cols):
        ax.text(x, y, hdr, fontsize=7.5, fontweight="bold", color=C_DARK, va="center")
    ax.axhline(y - 0.007, color=C_GREY, lw=0.5)
    y -= 0.030

    rows = [
        # (pair, ta, td, const, tod, gate, skip_pct, neg_pct, verdict, note)
        ("CHF_JPY", 5, 1, 1118, 970, 1122, 22.6, 0.5,  "⚠️ mild", "gate restores baseline ✓"),
        ("CHF_JPY", 3, 1,  248, 220,  287, 18.4, 1.7,  "⚠️ mild", "gate +16%"),
        ("CHF_JPY", 6, 5, 1659,1725, 1798, 25.4,15.0,  "⚠️ mild", "cliff: high neg%"),
        ("AUD_JPY", 4, 1, 1599,1675, 1670, 21.5, 0.2,  "✅ robust","strong — deploy"),
        ("EUR_JPY", 3, 1,  767, 714,  941, 17.1, 1.8,  "⚠️ mild", "gate +23% free alpha"),
        ("EUR_JPY",30, 3, 2364,1739,  731, 35.4, 2.0,  "❌ sens.", "gate hurts — fragile"),
        ("NZD_JPY", 2, 1,  752, 717,  120, 19.5,21.3,  "⚠️ mild", "gate collapses — ta too thin"),
        ("USD_JPY", 7, 1,  373, 345,  263, 20.5, 1.0,  "⚠️ mild", "mild -7%"),
        ("GBP_USD", 5, 1,  240, 190,  177, 21.7, 1.0,  "❌ sens.", "spread costly on small tgt"),
        ("CAD_JPY", 2, 1,  295, 286,  292, 17.1,19.6,  "⚠️ mild", "ta=2 thin — 20% neg"),
    ]

    rh = 0.045
    for i, (pair, ta, td, const, tod, gate, skip, neg, verdict, note) in enumerate(rows):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - rh + 0.002), 1, rh, fc=bg, ec="none", zorder=0))
        cy = y - rh / 2 + 0.002
        vc = C_GRN if "robust" in verdict else (C_YEL if "mild" in verdict else C_RED)
        delta_g = gate - const
        gc = C_GRN if delta_g >= 0 else C_RED
        vals = [pair, str(ta), str(td), f"{const:,}", f"{tod:,}", f"{gate:,}",
                f"{skip:.1f}%", f"{neg:.1f}%"]
        for vi, (val, x) in enumerate(zip(vals, cols)):
            c = C_DARK
            if vi == 5:
                c = gc  # gate p/d colored
            ax.text(x, cy, val, fontsize=7.5, va="center", color=c)
        ax.text(cols[8], cy, verdict, fontsize=7, va="center", color=vc, fontweight="bold")
        ax.text(cols[9], cy, note, fontsize=6, va="center", color=C_GREY)
        y -= rh

    y -= 0.010
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.020

    # Key findings
    findings = [
        (C_GRN, "AUD_JPY ta=4 is genuinely robust — +4.8% under TOD. Earlier -7,242 p/d was volume-model artifact. Deploy."),
        (C_GRN, "CHF_JPY ta=5 (live): spread gate restores full baseline. Gate blocks ~22% dead-zone entries at no cost."),
        (C_GRN, "EUR_JPY ta=3 + gate: gate boosts from 767 → 941 p/d (+23%). Best EUR_JPY deployment config."),
        (C_RED, "EUR_JPY ta=30: -26% under TOD, gate collapses to 731 p/d. Real-world p/d likely ~1,700 not 2,364."),
        (C_RED, "NZD_JPY ta=2 & CAD_JPY ta=2: ~20% neg cycles even with gate — min net too thin vs dead-zone spread."),
    ]
    for fc, txt in findings:
        ax.add_patch(FancyBboxPatch((0.01, y - 0.022), 0.98, 0.025,
                      boxstyle="round,pad=0.003", fc=fc, ec="none", alpha=0.12))
        ax.text(0.03, y - 0.009, txt, fontsize=7.5, color=C_DARK, va="center")
        y -= 0.030

    y -= 0.005
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.018
    ax.text(0.5, y,
            "Live gate: services/strategy_zr_random/main.py — reads actual bid_c/ask_c from ZMQ bar, "
            "skips new cycle if (ask-bid)/pip > MAX_ENTRY_SPREAD (2.5). Blocks Asian dead-zone entries.",
            ha="center", fontsize=7.5, color=C_GREY)

    return fig


def p_real_spread():
    """Session 027 — real OANDA bid/ask spread data + 3-way backtest."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc="#c0392b", ec="none"))
    ax.text(0.5, 0.958, "🚨 Real OANDA Spread — Critical Finding",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.921, "Session 027 · 2y M5 bid/ask data from OANDA API · ~149K bars per pair",
            ha="center", va="center", fontsize=8.5, color="#f5b7b1")

    # Spread data by pair
    y = 0.895
    ax.add_patch(FancyBboxPatch((0.01, 0.835), 0.98, 0.055,
                  boxstyle="round,pad=0.003", fc="#fdf2f8", ec=C_RED, lw=1.5))
    ax.text(0.5, 0.878, "OANDA CHF_JPY spread = 3.6-4.1p ALL DAY (21 UTC peak = 17.1p)",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_RED)
    ax.text(0.5, 0.850, "Backtest assumed 1.4p. Actual spread is 3× higher — trail minimum "
            "(ta=5,td=1) net = 4−3.7 = +0.3p. CHF_JPY edge is wiped.",
            ha="center", va="center", fontsize=8.5, color=C_DARK)

    y = 0.825
    sp_data = [
        ("CHF_JPY", 4.1, 5.2, 99.3, C_RED, "❌ 3× too wide — unprofitable"),
        ("NZD_JPY", 2.7, 3.5, 79.8, C_RED, "❌ 2× too wide"),
        ("EUR_JPY", 2.8, 3.2, 52.0, C_RED, "❌ 2× too wide"),
        ("CAD_JPY", 2.3, 3.2, 34.5, C_YEL, "⚠️ 1.6× wider"),
        ("AUD_JPY", 2.4, 2.8, 19.5, C_YEL, "⚠️ 1.7× wider — gate helps"),
        ("GBP_USD", 1.9, 2.2,  6.0, C_GRN, "✅ Viable with gate"),
        ("USD_JPY", 1.7, 2.0,  5.6, C_GRN, "✅ Best — closest to 1.4p"),
    ]
    cols_sp = [0.01, 0.14, 0.26, 0.36, 0.49, 0.61]
    hdrs_sp = ["Pair", "med spread", "p90", "%>2.5p", "1.4p assume", "verdict"]
    for hdr, x in zip(hdrs_sp, cols_sp):
        ax.text(x, y, hdr, fontsize=7.5, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.006, color=C_GREY, lw=0.4)
    y -= 0.025
    for pair, med, p90, gt25, c, verdict in sp_data:
        ratio = med / 1.4
        ax.add_patch(plt.Rectangle((0, y - 0.018), 1, 0.022, fc="#f8f9fa", ec="none"))
        ax.text(cols_sp[0], y - 0.006, pair, fontsize=8, color=C_DARK)
        ax.text(cols_sp[1], y - 0.006, f"{med:.1f}p", fontsize=8, color=c, fontweight="bold")
        ax.text(cols_sp[2], y - 0.006, f"{p90:.1f}p", fontsize=8, color=C_DARK)
        ax.text(cols_sp[3], y - 0.006, f"{gt25:.0f}%", fontsize=8, color=c)
        ax.text(cols_sp[4], y - 0.006, f"{ratio:.1f}×", fontsize=8, color=c)
        ax.text(cols_sp[5], y - 0.006, verdict, fontsize=7.5, color=c)
        y -= 0.022

    y -= 0.006
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.018

    # 3-way comparison table
    ax.text(0.5, y, "3-Way Backtest: A=const 1.4p · B=real spread no-gate · C=real+gate(2.5p)",
            ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    y -= 0.022
    cols3 = [0.01, 0.12, 0.18, 0.28, 0.40, 0.52, 0.65, 0.75, 0.88]
    hdrs3 = ["Pair", "ta", "td", "A const", "B real", "C+gate", "skip%", "neg%B", "best"]
    for hdr, x in zip(hdrs3, cols3):
        ax.text(x, y, hdr, fontsize=7.5, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.006, color=C_GREY, lw=0.4)
    y -= 0.025

    rows3 = [
        ("CHF_JPY", 5, 1,  988, -1189, -1907, 99.9,  11.1, "A", C_RED),
        ("CHF_JPY", 3, 1,  202, -2007,     2, 99.9,  46.6, "C", C_RED),
        ("AUD_JPY", 4, 1,   84,    63,    84, 57.2,   4.7, "C", C_GRN),
        ("EUR_JPY", 3, 1,  220,    26,    76, 89.3,  22.0, "C", C_YEL),
        ("EUR_JPY",30, 3, 1154,   265,    35, 91.5,   2.6, "A", C_RED),
        ("NZD_JPY", 2, 1,   88,  -478,    34, 90.2,  52.1, "C", C_YEL),
        ("USD_JPY", 7, 1,  320,   280,   102, 13.5,   1.1, "A", C_GRN),
        ("GBP_USD", 5, 1,  282,    62,   140, 16.2,   2.2, "C", C_GRN),
        ("CAD_JPY", 2, 1,   53,    27,   -67, 50.2,  40.3, "A", C_YEL),
    ]
    rh3 = 0.038
    for i, (pair, ta, td, pa, pb, pc, skip, neg, best, rc) in enumerate(rows3):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - rh3 + 0.002), 1, rh3, fc=bg, ec="none"))
        cy = y - rh3 / 2 + 0.002
        gc = C_GRN if pc > pa else (C_YEL if pc > 0 else C_RED)
        for v, x in zip([pair, str(ta), str(td), f"{pa:,}", f"{pb:,}", f"{pc:,}",
                          f"{skip:.1f}%", f"{neg:.1f}%"], cols3):
            c = C_DARK
            if x == cols3[4]: c = C_RED if pb < 0 else (C_GRN if pb > pa else C_YEL)
            if x == cols3[5]: c = gc
            ax.text(x, cy, v, fontsize=7.5, va="center", color=c)
        bc = C_GRN if best == "C" else (C_BLU if best == "B" else C_YEL)
        ax.add_patch(FancyBboxPatch((cols3[8], cy - 0.01), 0.06, 0.020,
                      boxstyle="round,pad=0.002", fc=bc, ec="none"))
        ax.text(cols3[8] + 0.03, cy, best, fontsize=7.5, va="center", ha="center",
                color="white", fontweight="bold")
        y -= rh3

    y -= 0.008
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.018

    # Corrected deployment plan
    ax.text(0.5, y, "Corrected Deployment Plan (based on real spread data)",
            ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    y -= 0.022
    plan = [
        (C_GRN,  "USD_JPY ta=7 td=1 ZW=40 — deploy, no gate. Real p/d ~280. Spread 1.7p ≈ assumption."),
        (C_GRN,  "GBP_USD ta=5 td=1 ZW=30 — deploy with gate=2.5p. Real p/d ~140. Viable."),
        (C_GRN,  "AUD_JPY ta=4 td=1 ZW=50 — deploy with gate=2.5p. Real p/d ~84. Viable."),
        (C_RED,  "CHF_JPY — STOP. Median 4.1p spread eliminates trail edge entirely. Real p/d < 0."),
        (C_RED,  "EUR_JPY, NZD_JPY — spread 2.7-2.8p gates out 80-90% of entries. Marginal at best."),
    ]
    for fc, txt in plan:
        ax.add_patch(FancyBboxPatch((0.01, y - 0.020), 0.98, 0.022,
                      boxstyle="round,pad=0.002", fc=fc, ec="none", alpha=0.12))
        ax.text(0.03, y - 0.009, txt, fontsize=7.5, color=C_DARK, va="center")
        y -= 0.028

    y -= 0.005
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.015
    ax.text(0.5, y,
            "Data: data/m5_ba/{pair}_M5_BA.parquet (2y, ~149K M5 bars, OANDA API price=MBA). "
            "Live fix: oanda_adapter.get_candles now requests price=MBA. "
            "Hedge sizing uses live bid/ask spread (live_spread passed to _breakeven_volume).",
            ha="center", fontsize=7, color=C_GREY, wrap=True)

    return fig


def p_allpairs_realspread():
    """Session 028 — definitive real-spread ta/td sweep across ALL 11 pairs."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_BLU, ec="none"))
    ax.text(0.5, 0.958, "Real-Spread ta/td Sweep — All 11 Pairs",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.921,
            "Session 028 · 2y BA parquets · gate = IS p90 spread per pair · "
            "wf=3 + bootstrap P5>0 + P(+)>95%",
            ha="center", va="center", fontsize=8.5, color="#aed6f1")

    # Summary box
    ax.add_patch(FancyBboxPatch((0.01, 0.850), 0.98, 0.055,
                  boxstyle="round,pad=0.003", fc="#eaf4fb", ec=C_BLU, lw=1.0))
    ax.text(0.5, 0.884,
            "9 of 11 pairs validated  ·  EUR_JPY leads at 591 p/d  ·  "
            "CHF_JPY and USD_JPY have no passing config",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.text(0.5, 0.860,
            "Ranking: EUR_JPY 591 > AUD_USD 194 > NZD_JPY 190 > EUR_USD 150 > "
            "NZD_USD 146 > GBP_USD 125 > CAD_JPY 124 > AUD_JPY 106 > EUR_GBP 17",
            ha="center", va="center", fontsize=7.5, color=C_GREY)

    # Best-per-pair table
    y = 0.838
    cols = [0.01, 0.11, 0.17, 0.23, 0.31, 0.40, 0.49, 0.57, 0.66, 0.75, 0.87]
    hdrs = ["Pair", "ta", "td", "p/d", "c/day", "P5", "P(+)", "sp_med", "gate", "skip%", "verdict"]
    for hdr, x in zip(hdrs, cols):
        ax.text(x, y, hdr, fontsize=7, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.005, color=C_GREY, lw=0.4)
    y -= 0.022

    best_rows = [
        ("EUR_JPY",  5, 1,  590.5, 11.3, 124.2, 1.000, 2.30, 3.10, 30.0, C_GRN,  "✅ STAR"),
        ("AUD_USD", 10, 5,  194.3,  3.6,  34.6, 1.000, 1.30, 1.50, 25.1, C_GRN,  "✅ tight sp"),
        ("NZD_JPY", 30, 1,  190.2,  2.0,  50.2, 0.994, 2.80, 3.60, 34.4, C_GRN,  "✅ ta=30 clears sp"),
        ("EUR_USD", 20, 1,  150.1,  2.6,  42.3, 1.000, 1.60, 1.70, 34.2, C_GRN,  "✅ solid"),
        ("NZD_USD", 10, 1,  145.5,  2.8,  29.2, 0.983, 1.50, 1.70, 34.0, C_GRN,  "✅ good"),
        ("GBP_USD", 10, 5,  124.8,  6.7,  46.8, 0.997, 1.90, 2.20, 29.9, C_GRN,  "✅ viable"),
        ("CAD_JPY",  7, 1,  123.5,  5.7,  44.1, 1.000, 2.40, 3.20, 26.5, C_GRN,  "✅ good"),
        ("AUD_JPY", 20, 5,  106.1,  2.6,  78.4, 1.000, 2.10, 2.60, 49.1, C_YEL,  "⚠️ skip=49%"),
        ("EUR_GBP",  3, 1,   16.7,  2.9,   9.4, 1.000, 1.40, 1.60, 37.2, C_YEL,  "⚠️ marginal"),
        ("USD_JPY",  0, 0,    0.0,  0.0,   0.0, 0.000, 1.70, 2.10,  0.0, C_RED,  "❌ no config (WF fail)"),
        ("CHF_JPY",  0, 0,    0.0,  0.0,   0.0, 0.000, 3.50, 5.30,  0.0, C_RED,  "❌ sp=3.5p kills edge"),
    ]
    rh = 0.034
    for i, (pair, ta, td, ppd, cday, p5, pp, sp_med, gate, skip, c, verdict) in enumerate(best_rows):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - rh + 0.001), 1, rh, fc=bg, ec="none"))
        cy = y - rh / 2 + 0.001
        ta_s  = str(ta)  if ta  else "—"
        td_s  = str(td)  if td  else "—"
        ppd_s = f"{ppd:.0f}"  if ppd  else "—"
        cday_s= f"{cday:.1f}" if cday else "—"
        p5_s  = f"{p5:.0f}"   if p5   else "—"
        pp_s  = f"{pp:.3f}"   if pp   else "—"
        vals  = [pair, ta_s, td_s, ppd_s, cday_s, p5_s, pp_s,
                 f"{sp_med:.2f}p", f"{gate:.2f}p", f"{skip:.0f}%", verdict]
        for v, x in zip(vals, cols):
            fc = c if x in (cols[3], cols[10]) else C_DARK
            ax.text(x, cy, v, fontsize=7, va="center", color=fc)
        y -= rh

    y -= 0.008
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.016

    # Key insights block
    ax.text(0.5, y, "Key Insights", ha="center", fontsize=9,
            fontweight="bold", color=C_DARK)
    y -= 0.020

    insights = [
        (C_GRN, "EUR_JPY 591 p/d — surprising star. ZW=50/tgt=25 absorbs 2.3p spread. "
                 "Min trail net = 5−1−3.1 = +0.9p. Prior 3-way test (ta=3) missed optimal config."),
        (C_GRN, "NZD_JPY validates at ta=30,td=1 — very wide trail (29p net min) defeats 2.8p spread. "
                 "Only 2 cycles/day but P5=50, P(+)=99.4%."),
        (C_GRN, "AUD_USD ta=10,td=5 — 194 p/d. Tight spread (1.3p med, 1.5p gate). Best p/d of "
                 "USD pairs. Gate skips only 25% of entries."),
        (C_YEL, "USD_JPY — no validated config despite 1.7p spread. WF fails on 3 chunks (regime "
                 "instability). Positive in OOS (134-242 p/d) but inconsistent across IS periods."),
        (C_RED, "CHF_JPY fully dead — median 3.5p, p90=5.3p. All ta/td combos deeply negative. "
                 "Live strategy on accts 011/012 MUST BE STOPPED immediately."),
        (C_YEL, "AUD_JPY skip=49% — gate at 2.6p blocks half of all entries. High-skip pairs "
                 "have lower capital efficiency despite positive expected value per trade."),
    ]
    for fc, txt in insights:
        wrapped = textwrap.fill(txt, width=110)
        lines   = wrapped.split("\n")
        height  = 0.016 * len(lines) + 0.008
        ax.add_patch(FancyBboxPatch((0.01, y - height), 0.98, height,
                      boxstyle="round,pad=0.002", fc=fc, ec="none", alpha=0.10))
        for li, line in enumerate(lines):
            ax.text(0.025, y - 0.010 - li * 0.016, line,
                    fontsize=7.5, va="center", color=C_DARK)
        y -= height + 0.006

    y -= 0.004
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.014

    ax.text(0.5, y,
            "Recommended live pairs: EUR_JPY (ta=5,td=1), AUD_USD (ta=10,td=5), "
            "EUR_USD (ta=20,td=1), NZD_JPY (ta=30,td=1) — all wf=3 validated with real spreads",
            ha="center", fontsize=7.5, fontweight="bold", color=C_BLU)
    y -= 0.016
    ax.text(0.5, y,
            "Data: zr_realspread_sweep_results.csv (352 rows · 11 pairs · "
            "real OANDA bid/ask spread · IS p90 gate per pair)",
            ha="center", fontsize=7, color=C_GREY)

    return fig


def p_double_wf():
    """Session 029 — double walk-forward gate: IS-wf=3 AND OOS-wf=3 AND MC."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_PUR, ec="none"))
    ax.text(0.5, 0.958, "Double Walk-Forward Gate — IS-wf=3 AND OOS-wf=3",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.921,
            "Session 029 · Strictest validation: IS split 3 chunks + OOS split 3 sub-windows "
            "+ bootstrap P5>0 + P(+)>95%",
            ha="center", va="center", fontsize=8.5, color="#d7bde2")

    # Method box
    ax.add_patch(FancyBboxPatch((0.01, 0.850), 0.98, 0.055,
                  boxstyle="round,pad=0.003", fc="#f5eef8", ec=C_PUR, lw=1.0))
    ax.text(0.5, 0.884,
            "Gate: IS-wf=3 (all IS chunks profitable) AND OOS-wf=3 (all OOS sub-windows profitable)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_DARK)
    ax.text(0.5, 0.860,
            "This catches configs that pass IS validation but show OOS regime instability. "
            "8 of 11 pairs have fully validated configs.",
            ha="center", va="center", fontsize=7.5, color=C_GREY)

    # Double-WF validated table
    y = 0.838
    cols = [0.01, 0.11, 0.17, 0.23, 0.31, 0.40, 0.49, 0.57, 0.66, 0.75, 0.87]
    hdrs = ["Pair", "ta", "td", "p/d", "c/day", "P5", "P(+)", "sp_med", "gate", "IS-wf", "OOS-wf"]
    for hdr, x in zip(hdrs, cols):
        ax.text(x, y, hdr, fontsize=7, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.005, color=C_GREY, lw=0.4)
    y -= 0.022

    validated = [
        ("EUR_JPY",  5, 1,  590.5, 11.3, 121.0, 1.000, 2.30, 3.10, 3, 3, C_GRN,  "✅ STAR — full double WF"),
        ("NZD_JPY", 30, 1,  190.2,  2.0,  46.4, 0.992, 2.80, 3.60, 3, 3, C_GRN,  "✅ solid"),
        ("AUD_USD",  6, 1,  172.7,  5.4,  60.4, 1.000, 1.30, 1.50, 3, 3, C_GRN,  "✅ tight sp"),
        ("EUR_USD", 20, 1,  150.1,  2.6,  43.2, 1.000, 1.60, 1.70, 3, 3, C_GRN,  "✅ solid"),
        ("GBP_USD", 10, 5,  124.8,  6.7,  44.4, 0.997, 1.90, 2.20, 3, 3, C_GRN,  "✅ viable"),
        ("CAD_JPY",  7, 1,  123.5,  5.7,  44.8, 1.000, 2.40, 3.20, 3, 3, C_GRN,  "✅ good"),
        ("AUD_JPY", 20, 5,  106.1,  2.6,  77.2, 1.000, 2.10, 2.60, 3, 3, C_YEL,  "⚠️ skip=49%"),
        ("NZD_USD", 30, 1,  102.6,  2.1,  59.4, 1.000, 1.50, 1.70, 3, 3, C_GRN,  "✅ ta=30 robust"),
        ("EUR_GBP",  3, 1,   16.7,  2.9,   9.1, 1.000, 1.40, 1.60, 3, 3, C_YEL,  "⚠️ marginal"),
        ("USD_JPY",  0, 0,    0.0,  0.0,   0.0, 0.000, 1.70, 2.10, 0, 0, C_RED,  "❌ regime instability"),
        ("CHF_JPY",  0, 0,    0.0,  0.0,   0.0, 0.000, 3.50, 5.30, 0, 0, C_RED,  "❌ spread fatal"),
    ]
    rh = 0.034
    for i, row in enumerate(validated):
        pair, ta, td, ppd, cday, p5, pp, sp_med, gate, is_wf, oos_wf, c, verdict = row
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - rh + 0.001), 1, rh, fc=bg, ec="none"))
        cy = y - rh / 2 + 0.001
        ta_s  = str(ta)  if ta  else "—"
        td_s  = str(td)  if td  else "—"
        ppd_s = f"{ppd:.0f}"  if ppd  else "—"
        cday_s= f"{cday:.1f}" if cday else "—"
        p5_s  = f"{p5:.0f}"   if p5   else "—"
        pp_s  = f"{pp:.3f}"   if pp   else "—"
        iswf_s= f"{is_wf}/3"  if is_wf else "0/3"
        ooswf_s= f"{oos_wf}/3" if oos_wf else "0/3"
        vals  = [pair, ta_s, td_s, ppd_s, cday_s, p5_s, pp_s,
                 f"{sp_med:.2f}p", f"{gate:.2f}p", iswf_s, ooswf_s]
        for v, x in zip(vals, cols):
            fc = c if x in (cols[3], cols[11] if len(cols)>11 else cols[10]) else C_DARK
            if x == cols[3]: fc = c
            ax.text(x, cy, v, fontsize=7, va="center", color=fc)
        y -= rh

    y -= 0.008
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.016

    # What OOS gate filtered out
    ax.text(0.5, y, "IS-wf=3 passed but OOS-wf failed — filtered by new gate",
            ha="center", fontsize=8.5, fontweight="bold", color=C_DARK)
    y -= 0.022

    filtered = [
        ("AUD_USD", 10, 5,  194.3, 3, 2,
         "Best prior config (194 p/d) — OOS sub-window 2 was negative. ta=6 replaces it."),
        ("NZD_USD", 10, 1,  145.5, 3, 2,
         "Prior best 146 p/d — OOS instability. ta=30 is slower but passes both gates."),
        ("EUR_USD",  6, 2,  516.2, 3, 2,
         "Very high 516 p/d but OOS regime failure — ta=20 is the robust alternative."),
        ("NZD_JPY", 20, 7,  116.8, 3, 2,
         "ta=30 replaces — wide trail (29p net min) is more regime-stable."),
        ("EUR_JPY", 14, 1,  107.4, 3, 2,
         "Filtered, but ta=5 still passes. EUR_JPY is robust at the smaller activation."),
    ]
    cols_f = [0.01, 0.11, 0.17, 0.23, 0.30, 0.37, 0.45]
    hdrs_f = ["Pair", "ta", "td", "p/d", "IS-wf", "OOS-wf", "Why filtered"]
    for hdr, x in zip(hdrs_f, cols_f):
        ax.text(x, y, hdr, fontsize=7, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.005, color=C_GREY, lw=0.4)
    y -= 0.020
    for pair, ta, td, ppd, iswf, ooswf, note in filtered:
        bg = "#fff9e6"
        ax.add_patch(plt.Rectangle((0, y - 0.022), 1, 0.024, fc=bg, ec="none"))
        ax.text(cols_f[0], y - 0.011, pair,         fontsize=7, va="center")
        ax.text(cols_f[1], y - 0.011, str(ta),      fontsize=7, va="center")
        ax.text(cols_f[2], y - 0.011, str(td),      fontsize=7, va="center")
        ax.text(cols_f[3], y - 0.011, f"{ppd:.0f}", fontsize=7, va="center", color=C_YEL)
        ax.text(cols_f[4], y - 0.011, f"{iswf}/3",  fontsize=7, va="center", color=C_GRN)
        ax.text(cols_f[5], y - 0.011, f"{ooswf}/3", fontsize=7, va="center", color=C_RED)
        ax.text(cols_f[6], y - 0.011, note,         fontsize=6.5, va="center", color=C_GREY)
        y -= 0.026

    y -= 0.008
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.015

    ax.text(0.5, y,
            "Verdict: EUR_JPY ta=5 td=1 at 591 p/d survives the strictest gate. "
            "Priority deployment: EUR_JPY → NZD_JPY → AUD_USD → EUR_USD → GBP_USD → CAD_JPY → AUD_JPY → NZD_USD",
            ha="center", fontsize=7.5, fontweight="bold", color=C_PUR, wrap=True)
    y -= 0.016
    ax.text(0.5, y,
            "Data: zr_oos_wf_sweep_results.csv · 352 rows · "
            "gate=IS-p90 per pair · IS split 3 chunks · OOS split 3 sub-windows · 2000 bootstrap draws",
            ha="center", fontsize=7, color=C_GREY)

    return fig


def p_deployment_030():
    """Session 030 — EUR_JPY deployment decision + leg-depth analysis."""
    fig = plt.figure(figsize=(8.5, 11))
    fig_style(fig)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Header
    ax.add_patch(FancyBboxPatch((0.0, 0.91), 1.0, 0.09,
                  boxstyle="round,pad=0.005", fc=C_GRN, ec="none"))
    ax.text(0.5, 0.958, "Session 030 — EUR_JPY Deployed (2026-05-05)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.921,
            "CHF_JPY retired (spread fatal) · EUR_JPY: best of 11 pairs on all criteria · "
            "LIVE on accounts 011 (LONG) / 012 (SHORT)",
            ha="center", va="center", fontsize=8.5, color="#d5f5e3")

    # Decision rationale box
    ax.add_patch(FancyBboxPatch((0.01, 0.845), 0.98, 0.060,
                  boxstyle="round,pad=0.003", fc="#eafaf1", ec=C_GRN, lw=1.0))
    ax.text(0.5, 0.882,
            "Deployment config: EUR_JPY  ZW=50  tgt=25  ta=5  td=1  gate=3.1p  "
            "base_units=20  signal=none",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.text(0.5, 0.858,
            "591 p/d OOS · IS-wf=3 · OOS-wf=3 · P5=+121 · P(+)=100% · "
            "1.54% of cycles go 5+ legs (lowest of all 11 pairs)",
            ha="center", va="center", fontsize=8, color=C_GREY)

    # Why CHF_JPY retired
    y = 0.832
    ax.text(0.5, y, "WHY CHF_JPY RETIRED", ha="center", fontsize=9,
            fontweight="bold", color=C_RED)
    y -= 0.020
    ax.text(0.5, y,
            "CHF_JPY spread: median 3.5p, p90 gate = 5.3p.  "
            "Min trail net = ta − td − gate = 5 − 1 − 5.3 = −1.3p < 0.  "
            "ALL configs negative after real spread cost.  "
            "Not a parameter problem — structurally undeployable at current OANDA spread.",
            ha="center", va="center", fontsize=7.5, color=C_DARK, wrap=True)

    # Leg-depth table
    y -= 0.030
    ax.text(0.5, y, "LEG-DEPTH DISTRIBUTION — ALL DOUBLE-WF VALIDATED CONFIGS (OOS cycles)",
            ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    y -= 0.008
    ax.text(0.5, y,
            "Key question: how often does the strategy actually need 5+ legs?  "
            "5+leg cycles generate most of the profit but are rare.",
            ha="center", fontsize=7.5, color=C_GREY)
    y -= 0.022

    cols = [0.01, 0.11, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.73, 0.85]
    hdrs = ["Pair (config)", "1-leg%", "2-leg%", "3-leg%", "4-leg%", "5+leg%",
            "pnl 5+", "pnl 1-4", "total p/d", "note"]
    for hdr, x in zip(hdrs, cols):
        ax.text(x, y, hdr, fontsize=6.5, fontweight="bold", color=C_DARK)
    ax.axhline(y - 0.005, color=C_GREY, lw=0.4)
    y -= 0.022

    depth_data = [
        # pair_label, l1%, l2%, l3%, l4%, l5p%, pnl5p, pnl14, total, note, color
        ("EUR_JPY ta=5 td=1", 92.1, 4.4, 1.5, 0.5, 1.54, 531.9, 58.6,  590.5, "★ DEPLOYED",  C_GRN),
        ("AUD_USD ta=6 td=1", 85.0, 6.8, 5.6, 0.7, 1.86, 139.8, 32.8,  172.7, "tight spread", C_GRN),
        ("CAD_JPY ta=7 td=1", 86.3, 7.8, 2.6, 1.2, 2.13,  88.6, 34.9,  123.5, "low 5+leg",   C_GRN),
        ("EUR_USD ta=20 td=1",64.7,20.5, 7.2, 3.3, 4.35, 123.7, 26.3,  150.1, "high 2-leg",  C_YEL),
        ("GBP_USD ta=10 td=5",75.0,13.4, 5.4, 2.5, 3.76,  99.9, 24.9,  124.8, "viable",      C_GRN),
        ("NZD_JPY ta=30 td=1",67.3,17.5, 5.6, 3.6, 5.94, 184.0,  6.2,  190.2, "high 5+leg%", C_YEL),
        ("AUD_JPY ta=20 td=5",69.4,14.0, 8.6, 3.9, 4.16,  57.4, 48.7,  106.1, "49% skip",    C_YEL),
        ("NZD_USD ta=30 td=1",60.4,20.5, 7.8, 5.5, 5.84,  76.4, 26.2,  102.6, "high 5+leg%", C_YEL),
    ]
    rh = 0.030
    for i, row in enumerate(depth_data):
        lbl, l1, l2, l3, l4, l5p, pnl5p, pnl14, tot, note, c = row
        bg = "#f0fff4" if c == C_GRN else ("#fff9e6" if c == C_YEL else "#fff5f5")
        ax.add_patch(plt.Rectangle((0, y - rh + 0.001), 1, rh, fc=bg, ec="none"))
        cy = y - rh / 2 + 0.001
        l5_color = C_GRN if l5p < 2.5 else (C_YEL if l5p < 4.5 else C_RED)
        vals = [lbl, f"{l1:.1f}%", f"{l2:.1f}%", f"{l3:.1f}%", f"{l4:.1f}%",
                f"{l5p:.2f}%", f"{pnl5p:.1f}", f"{pnl14:.1f}", f"{tot:.0f}", note]
        colors = [C_DARK, C_GREY, C_GREY, C_GREY, C_GREY, l5_color,
                  C_DARK, C_DARK, c, C_GREY]
        for v, x, vc in zip(vals, cols, colors):
            ax.text(x, cy, v, fontsize=6.5, va="center", color=vc,
                    fontweight="bold" if x == cols[0] else "normal")
        y -= rh

    y -= 0.008
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.018

    # Structural conclusions
    ax.text(0.5, y, "STRUCTURAL FINDINGS FROM LEG-DEPTH ANALYSIS",
            ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    y -= 0.020
    findings = [
        "90% of EUR_JPY profit comes from 5+leg cycles (1.54% of cycles) — unavoidable ZR mechanic",
        "98.4% of EUR_JPY cycles resolve at ≤4 legs — user's '≤4 legs' preference is satisfied in practice",
        "Pure trail (ML=1) is always negative for all pairs — ZR profitability requires full gearing",
        "Flat +3p gearing is always negative — standard PF=1.25 breakeven sizing is the only working model",
        "Signal gates (SMA10/SMA20) destroy edge — random alternating beats all tested signals",
    ]
    for f in findings:
        ax.text(0.03, y, f"• {f}", fontsize=7.5, color=C_DARK)
        y -= 0.018

    y -= 0.010
    ax.axhline(y, color=C_GREY, lw=0.4)
    y -= 0.016

    # Live status footer
    ax.add_patch(FancyBboxPatch((0.01, y - 0.055), 0.98, 0.050,
                  boxstyle="round,pad=0.003", fc="#eaf4fb", ec=C_BLU, lw=0.8))
    ax.text(0.5, y - 0.010,
            "🟢  LIVE 2026-05-05 — fx-zr-random container  |  ZR_PAIRS=EUR_JPY",
            ha="center", fontsize=9, fontweight="bold", color=C_BLU)
    ax.text(0.5, y - 0.028,
            "Account 011 (LONG legs)  +  Account 012 (SHORT legs)",
            ha="center", fontsize=8, color=C_DARK)
    ax.text(0.5, y - 0.044,
            "Next: add GBP_USD at NAV≥$28 · add USD_JPY at NAV≥$40",
            ha="center", fontsize=7.5, color=C_GREY)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Building ZR full report…")
    np.random.seed(42)

    pages = [
        ("01 Title",               p_title),
        ("02 Table of Contents",   p_toc),
        ("03 Executive Summary",   p_exec_summary),
        ("04 ZR Mechanics",        p_zr_mechanics),
        ("05 Glossary",            p_glossary),
        ("06 Phase 1: Classic",    p_phase1_classic),
        ("07 Phase 2-4",           p_phase24),
        ("08 Phase 5-6",           p_phase56),
        ("09 Phase 7-9",           p_phase79),
        ("10 Trail sweep",         p_trail),
        ("11 Signal sweep",        p_signal),
        ("12 H1/H4 directional",   p_h1h4),
        ("13 Random per-pair",     p_random_perp),
        ("14 P&F sweep",           p_pnf_sweep),
        ("15 Permtest + MC",       p_permtest),
        ("16 Risk & DD",           p_risk),
        ("17 Master ranking",      p_master_ranking),
        ("18 Live status",         p_live),
        ("19 Evolution narrative", p_evolution),
        ("20 Bottom Line",         p_bottom_line),
        ("21 Catalog: Overview",   p_catalog_overview),
        ("22 Catalog: Validated",  p_catalog_validated),
        ("23 Catalog: Failed",     p_catalog_failed),
        ("24 All-Pairs Sweep",     p_allpairs_sweep),
        ("25 Spread Sensitivity",  p_spread_sensitivity),
        ("26 Real Spread (OANDA)", p_real_spread),
        ("27 All-Pairs Real Spread Sweep", p_allpairs_realspread),
        ("28 Double WF Gate (IS+OOS)",    p_double_wf),
        ("29 EUR_JPY Deployment (030)",   p_deployment_030),
    ]

    with PdfPages(OUT) as pdf:
        for i, (name, fn) in enumerate(pages):
            print(f"  {name}…", end=" ", flush=True)
            try:
                fig = fn()
                fig.text(0.97, 0.005, f"{i+1}", fontsize=9, ha='right', va='bottom', color=C_GREY)
                pdf.savefig(fig, bbox_inches=None, facecolor=C_BG)
                plt.close(fig)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                plt.close("all")

        d = pdf.infodict()
        d["Title"]        = "ZR Full Report — All Ping-Pong Variants"
        d["Author"]       = "FX-Core Research"
        d["Subject"]      = "Zone Recovery experiment archive"
        d["CreationDate"] = datetime.now()

    print(f"\nSaved: {OUT}")

if __name__ == "__main__":
    main()
