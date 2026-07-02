#!/usr/bin/env python3
"""
Session 060 — PantasticSMA: Confluence edge + P&F application.

Part 1 — MTF Confluence:
  Does requiring TF agreement (pair / triple / quad) improve edge over single TF?
  Entry: all TFs in combo show same direction. Exit: any TF breaks agreement.
  Combos: S5, S30, M5, H1 singles + 6 pairs + 4 triples + ALL-4.
  Data: EUR_JPY S5 (full ~6 months) for signals; IS/OOS 70/30.

Part 2 — PantasticSMA on P&F level series:
  Apply Pantastic SMA momentum to P&F-filtered price (step-function).
  Sweep: box_pips ∈ {5, 10, 20} × rev_mult ∈ {1, 1.5, 2, 2.5, 3, 3.5}.
  Both trend-follow and counter-trend.
  P&F built with R2 within-bar ordering (bull: H→L, bear: L→H).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from numba import njit

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

PIP      = 0.01    # EUR/JPY
SMA_P    = 7
LOOKBACK = 5
THR      = 0.003   # price / minute

# ══════════════════════════════════════════════════════════════════════════════
# Stats helper
# ══════════════════════════════════════════════════════════════════════════════

def stats(trades, ts_ns, start_i, end_i):
    if len(trades) == 0:
        return None
    ts0   = pd.Timestamp(ts_ns[start_i],   unit='ns', tz='UTC')
    ts1   = pd.Timestamp(ts_ns[end_i - 1], unit='ns', tz='UTC')
    t_d   = (ts1 - ts0).total_seconds() / 86400 * 5 / 7
    if t_d <= 0:
        return None
    wins = trades[trades > 0]; loss = trades[trades <= 0]
    return dict(
        n   = len(trades),
        tpd = len(trades) / t_d,
        wr  = len(wins) / len(trades) * 100,
        aw  = wins.mean() if len(wins) else 0.0,
        al  = loss.mean() if len(loss) else 0.0,
        pf  = abs(wins.sum() / loss.sum()) if loss.sum() != 0 else np.inf,
        ppd = trades.sum() / t_d,
        tot = trades.sum(),
    )


def row(label, si, so):
    def f(s, tag):
        if s is None:
            return f"  {tag}  —"
        sign = "🟢" if s['ppd'] > 0 else "🔴"
        return (f"  {tag} n={s['n']:5d}({s['tpd']:5.1f}/d)  "
                f"WR={s['wr']:4.1f}%  W={s['aw']:+5.2f}  L={s['al']:+5.2f}  "
                f"PF={s['pf']:5.2f}  p/d={s['ppd']:+7.2f} {sign}")
    print(f"  {label}")
    print(f(si, "IS "))
    print(f(so, "OOS"))


# ══════════════════════════════════════════════════════════════════════════════
# Numba kernels
# ══════════════════════════════════════════════════════════════════════════════

@njit
def backtest_sig(mc, sp, sig, pip, s, e):
    """Trade a pre-computed signal array aligned to mc."""
    n = e - s
    tp = np.empty(n, np.float64); nt = 0
    eq = np.zeros(n, np.float64); cum = 0.0
    pos = 0; epx = 0.0
    for ii in range(n):
        i = s + ii; eq[ii] = cum
        ns = sig[i]
        if ns == pos: continue
        h = sp[i] / 2.0
        if pos != 0:
            x = mc[i] - h if pos == 1 else mc[i] + h
            p = (x - epx) / pip if pos == 1 else (epx - x) / pip
            cum += p; tp[nt] = p; nt += 1
        if ns == 1:  epx = mc[i] + h
        elif ns == -1: epx = mc[i] - h
        pos = ns; eq[ii] = cum
    if pos != 0:
        h = sp[e-1] / 2.0
        x = mc[e-1] - h if pos == 1 else mc[e-1] + h
        p = (x - epx) / pip if pos == 1 else (epx - x) / pip
        cum += p; tp[nt] = p; nt += 1; eq[n-1] = cum
    return tp[:nt], eq, cum


@njit
def run_pnf_pantastic(
    mid_o, mid_h, mid_l, mid_c, spread,
    box, rev_mult,
    sma_p, lookback, thr, bar_min,
    pip, counter_trend,
    start_i, end_i
):
    """
    Build P&F level series from full OHLC, apply Pantastic SMA momentum,
    backtest the signal against real spread.
    """
    n_total  = len(mid_c)
    rev_size = rev_mult * box

    # ── P&F level series ─────────────────────────────────────────────────────
    pnf_lvl   = np.zeros(n_total, np.float64)
    pnf_dir   = 0        # 0=uninit 1=X -1=O
    col_h_idx = 0        # integer box index of column high
    col_l_idx = 0        # integer box index of column low
    anc_idx   = int(mid_c[0] / box)   # anchor box index

    for i in range(n_total):
        is_bull = mid_c[i] >= mid_o[i]
        for tick in range(2):
            if is_bull:
                p = mid_h[i] if tick == 0 else mid_l[i]
            else:
                p = mid_l[i] if tick == 0 else mid_h[i]

            p_idx = int(p / box)

            if pnf_dir == 0:
                d = p_idx - anc_idx
                if d >= 1:   col_h_idx = p_idx; pnf_dir = 1
                elif d <= -1: col_l_idx = p_idx; pnf_dir = -1

            elif pnf_dir == 1:
                if p_idx - col_h_idx >= 1:
                    col_h_idx = p_idx                        # extend X
                elif p < col_h_idx * box - rev_size:
                    col_l_idx = p_idx; pnf_dir = -1          # reverse to O

            else:  # pnf_dir == -1
                if col_l_idx - p_idx >= 1:
                    col_l_idx = p_idx                        # extend O
                elif p > col_l_idx * box + rev_size:
                    col_h_idx = p_idx; pnf_dir = 1           # reverse to X

        if   pnf_dir == 1:  pnf_lvl[i] = col_h_idx * box
        elif pnf_dir == -1: pnf_lvl[i] = col_l_idx * box
        else:               pnf_lvl[i] = anc_idx * box

    # ── Rolling SMA on P&F level ──────────────────────────────────────────────
    sma_buf = np.zeros(sma_p, np.float64)
    sma_sum = 0.0; sma_filled = 0
    sma_v   = np.zeros(n_total, np.float64)
    for i in range(n_total):
        sma_sum = sma_sum - sma_buf[i % sma_p] + pnf_lvl[i]
        sma_buf[i % sma_p] = pnf_lvl[i]
        if sma_filled < sma_p: sma_filled += 1
        if sma_filled >= sma_p: sma_v[i] = sma_sum / sma_p

    # ── Backtest ──────────────────────────────────────────────────────────────
    n    = end_i - start_i
    warm = sma_p + lookback
    tp   = np.empty(n, np.float64); nt = 0
    eq   = np.zeros(n, np.float64); cum = 0.0
    pos  = 0; epx = 0.0

    for ii in range(n):
        i = start_i + ii
        eq[ii] = cum
        if ii < warm or sma_v[i] == 0.0: continue
        prev_i = i - lookback
        if sma_v[prev_i] == 0.0: continue

        rate = (sma_v[i] - sma_v[prev_i]) / (lookback * bar_min)
        if counter_trend:
            ns = -1 if rate > thr else (1 if rate < -thr else 0)
        else:
            ns =  1 if rate > thr else (-1 if rate < -thr else 0)

        if ns == pos: continue
        h = spread[i] / 2.0
        if pos != 0:
            x = mid_c[i] - h if pos == 1 else mid_c[i] + h
            p = (x - epx) / pip if pos == 1 else (epx - x) / pip
            cum += p; tp[nt] = p; nt += 1
        if ns == 1:   epx = mid_c[i] + h
        elif ns == -1: epx = mid_c[i] - h
        pos = ns; eq[ii] = cum

    if pos != 0:
        h = spread[end_i-1] / 2.0
        x = mid_c[end_i-1] - h if pos == 1 else mid_c[end_i-1] + h
        p = (x - epx) / pip if pos == 1 else (epx - x) / pip
        cum += p; tp[nt] = p; nt += 1; eq[n-1] = cum

    return tp[:nt], eq, cum


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading EUR_JPY S5 (full dataset) …")
raw = (pd.read_parquet(ROOT / "data" / "s5_ohlc" / "EUR_JPY_S5_BA.parquet")
         .sort_values("timestamp").reset_index(drop=True))
raw["mid_o"] = (raw.bid_o + raw.ask_o) / 2
raw["mid_h"] = (raw.bid_h + raw.ask_h) / 2
raw["mid_l"] = (raw.bid_l + raw.ask_l) / 2
raw["mid_c"] = (raw.bid_c + raw.ask_c) / 2
raw["spread"] = raw.ask_c - raw.bid_c

mc  = raw.mid_c.values.astype(np.float64)
mo  = raw.mid_o.values.astype(np.float64)
mh  = raw.mid_h.values.astype(np.float64)
ml  = raw.mid_l.values.astype(np.float64)
sp  = raw.spread.values.astype(np.float64)
ts  = raw.timestamp.values.astype(np.int64)

N     = len(mc)
SPLIT = int(N * 0.70)
print(f"  {N:,} S5 bars  {raw.timestamp.iloc[0].date()} → {raw.timestamp.iloc[-1].date()}")
print(f"  IS: 0–{SPLIT:,}  OOS: {SPLIT:,}–{N:,}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Confluence edge
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*88)
print("PART 1 — MTF CONFLUENCE (EUR_JPY S5, thr=0.003 price/min, SMA7, lookback5)")
print("═"*88)

# ── Compute Pantastic signal on each TF ───────────────────────────────────────
def compute_pantastic_signal(prices, bar_min):
    sma = pd.Series(prices).rolling(SMA_P, min_periods=SMA_P).mean().values
    sig = np.zeros(len(prices), dtype=np.int8)
    warm = SMA_P + LOOKBACK
    for i in range(warm, len(prices)):
        if np.isnan(sma[i]) or np.isnan(sma[i - LOOKBACK]):
            continue
        r = (sma[i] - sma[i - LOOKBACK]) / (LOOKBACK * bar_min)
        sig[i] = np.int8(1) if r > THR else (np.int8(-1) if r < -THR else np.int8(0))
    return sig


def ff_to_s5(src_ts, src_sig, tgt_ts):
    src = pd.DataFrame({"ts": src_ts, "sig": src_sig.astype(float)})
    tgt = pd.DataFrame({"ts": tgt_ts})
    m   = pd.merge_asof(tgt, src, on="ts", direction="backward")
    return m["sig"].fillna(0).values.astype(np.int8)


raw_i = raw.set_index("timestamp")

s30_raw = raw_i[["mid_c"]].resample("30s").last().dropna().reset_index()
m5_raw  = raw_i[["mid_c"]].resample("5min").last().dropna().reset_index()
h1_raw  = raw_i[["mid_c"]].resample("1h").last().dropna().reset_index()

print("Computing TF signals …")
s5_sig  = compute_pantastic_signal(mc,                    5 / 60)
s30_sig = compute_pantastic_signal(s30_raw.mid_c.values,  0.5)
m5_sig  = compute_pantastic_signal(m5_raw.mid_c.values,   5.0)
h1_sig  = compute_pantastic_signal(h1_raw.mid_c.values,  60.0)

ts5 = raw.timestamp
s30_ff = ff_to_s5(s30_raw.timestamp, s30_sig, ts5)
m5_ff  = ff_to_s5(m5_raw.timestamp,  m5_sig,  ts5)
h1_ff  = ff_to_s5(h1_raw.timestamp,  h1_sig,  ts5)

tf_signals = {"S5": s5_sig, "S30": s30_ff, "M5": m5_ff, "H1": h1_ff}

# Build agreement signal for a set of TF keys
def agreement_signal(keys):
    arrs = [tf_signals[k].astype(np.int16) for k in keys]
    up   = np.ones(N, dtype=bool)
    dn   = np.ones(N, dtype=bool)
    for a in arrs:
        up &= (a == 1)
        dn &= (a == -1)
    sig = np.zeros(N, dtype=np.int8)
    sig[up] = 1; sig[dn] = -1
    return sig


print()
print(f"  {'Config':20s}  IS n(tpd) WR% p/d    OOS n(tpd) WR% p/d")
print("  " + "-"*84)

all_keys = ["S5", "S30", "M5", "H1"]

# Singles
for k in all_keys:
    sig = tf_signals[k]
    tr_is,  _, _ = backtest_sig(mc, sp, sig, PIP, 0, SPLIT)
    tr_oos, _, _ = backtest_sig(mc, sp, sig, PIP, SPLIT, N)
    si = stats(tr_is,  ts, 0, SPLIT)
    so = stats(tr_oos, ts, SPLIT, N)
    row(k, si, so)

print()

# Pairs (C(4,2)=6)
for r in range(2, 5):
    label = f"--- {r}-TF combos ---"
    print(f"  {label}")
    for keys in combinations(all_keys, r):
        sig = agreement_signal(list(keys))
        tr_is,  _, _ = backtest_sig(mc, sp, sig, PIP, 0, SPLIT)
        tr_oos, _, _ = backtest_sig(mc, sp, sig, PIP, SPLIT, N)
        si = stats(tr_is,  ts, 0, SPLIT)
        so = stats(tr_oos, ts, SPLIT, N)
        label_str = "+".join(keys)
        row(label_str, si, so)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — PantasticSMA on P&F level series
# ══════════════════════════════════════════════════════════════════════════════

print("═"*88)
print("PART 2 — PANTASTIC ON P&F LEVEL SERIES (EUR_JPY S5, SMA7, lookback5, thr=0.003)")
print("═"*88)
print()

BOX_PIPS = [5, 10, 20]
REV_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
BAR_MIN  = 5 / 60   # S5 bars: 5 seconds = 5/60 minutes

# Warm-up compile
print("  JIT compiling …", end=" ", flush=True)
run_pnf_pantastic(mo[:500], mh[:500], ml[:500], mc[:500], sp[:500],
                  0.05, 1.0, SMA_P, LOOKBACK, THR, BAR_MIN, PIP, False, 0, 500)
print("done")
print()

results = []

for box_pips in BOX_PIPS:
    box = box_pips * PIP
    for rev_mult in REV_MULTS:
        for ct, dir_label in [(False, "trend"), (True, "counter")]:
            tr_is,  _, _ = run_pnf_pantastic(
                mo, mh, ml, mc, sp, box, rev_mult,
                SMA_P, LOOKBACK, THR, BAR_MIN, PIP, ct, 0, SPLIT)
            tr_oos, _, _ = run_pnf_pantastic(
                mo, mh, ml, mc, sp, box, rev_mult,
                SMA_P, LOOKBACK, THR, BAR_MIN, PIP, ct, SPLIT, N)

            si = stats(tr_is,  ts, 0, SPLIT)
            so = stats(tr_oos, ts, SPLIT, N)

            oos_ppd = so["ppd"] if so else -999
            results.append((oos_ppd, box_pips, rev_mult, dir_label, si, so))

# Sort by OOS p/d descending
results.sort(key=lambda x: x[0], reverse=True)

print(f"  {'Config':30s}  {'IS p/d':>8}  {'IS n':>6}  {'OOS p/d':>9}  {'OOS n':>6}  {'OOS WR':>7}  {'OOS PF':>7}")
print("  " + "-"*86)

for oos_ppd, bx, rv, dl, si, so in results:
    cfg = f"box={bx:2d}p rev={rv:.1f} {dl:7s}"
    is_ppd  = f"{si['ppd']:+8.2f}" if si else "      —"
    is_n    = f"{si['n']:6d}"       if si else "     —"
    oos_p   = f"{so['ppd']:+9.2f}" if so else "       —"
    oos_n   = f"{so['n']:6d}"       if so else "     —"
    oos_wr  = f"{so['wr']:6.1f}%"  if so else "      —"
    oos_pf  = f"{so['pf']:7.2f}"   if so else "      —"
    flag = "🟢" if (so and so['ppd'] > 0) else "🔴"
    print(f"  {cfg:30s}  {is_ppd}  {is_n}  {oos_p}  {oos_n}  {oos_wr}  {oos_pf} {flag}")

print()

# Summary
pos_oos = sum(1 for r in results if r[5] and r[5]['ppd'] > 0)
print(f"  OOS positive: {pos_oos}/{len(results)} configs")

# Best by box size
for bx in BOX_PIPS:
    sub = [(r[0], r[2], r[3]) for r in results if r[1] == bx and r[5] is not None]
    best = max(sub, key=lambda x: x[0]) if sub else None
    if best:
        print(f"  Best box={bx}p: rev={best[1]:.1f} {best[2]:7s}  OOS p/d={best[0]:+.2f}")
