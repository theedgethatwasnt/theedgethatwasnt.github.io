#!/usr/bin/env python3
"""
Session 061 — Currency Strength Lower-TF Viability Sweep

Does the H4 contrarian StrengthSpread alpha (Sharpe 0.59 from csi_factor_study,
H4-signal × 64-bar hold) persist at M5 / M15 / H1 signal granularity?

StrengthSpread_{A/B} = z(strength_A) - z(strength_B)
  strength_c[i] = Σ_p sign_cp × roll_sum(log_ret_p, sig_tf)[i]
  z-scored cross-sectionally across 8 currencies at each bar.

Sweep:
  sig_tf  ∈ {1, 3, 12, 48}     M5-bars  (M5, ~M15, ~H1, ~H4)
  hold    ∈ {1,2,4,8,16,32,64,128,256,512,3072}   M5-bars
  direction ∈ {trend, counter}
  pairs: all 12 M5 BA pairs

Also: top-3 vs bottom-3 currency portfolio at best config.
IS/OOS: 70/30 on common M5 timestamp intersection.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
DATA = ROOT / "data" / "m5_ba"

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY",
    "EUR_GBP", "EUR_JPY", "EUR_USD",
    "GBP_JPY", "GBP_USD",
    "NZD_JPY", "NZD_USD", "USD_JPY",
]
CURRENCIES = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"]
CUR_IDX    = {c: i for i, c in enumerate(CURRENCIES)}

PIP = {
    "AUD_JPY": 0.01,   "CAD_JPY": 0.01,   "CHF_JPY": 0.01,  "EUR_JPY": 0.01,
    "GBP_JPY": 0.01,   "NZD_JPY": 0.01,   "USD_JPY": 0.01,
    "AUD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_USD": 0.0001,
    "GBP_USD": 0.0001, "NZD_USD": 0.0001,
}

SIG_TFS = [1, 3, 12, 48]
HOLDS   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 3072]
IS_FRAC = 0.70

# Sign matrix: SIGN_MAT[curr_idx, pair_idx] ∈ {-1, 0, +1}
SIGN_MAT = np.zeros((len(CURRENCIES), len(PAIRS)), dtype=np.float64)
for _pi, _p in enumerate(PAIRS):
    _b, _q = _p.split("_")
    SIGN_MAT[CUR_IDX[_b], _pi] = +1.0
    SIGN_MAT[CUR_IDX[_q], _pi] = -1.0


# ══════════════════════════════════════════════════════════════════════════════
# Load & align data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading 12 M5 BA parquets ...")
pair_df = {}
for p in PAIRS:
    df = (pd.read_parquet(DATA / f"{p}_M5_BA.parquet")
            .sort_values("timestamp").reset_index(drop=True))
    df["mid_c"] = (df.bid_c + df.ask_c) / 2.0
    df["sp"]    = df.ask_c - df.bid_c
    pair_df[p]  = df
    print(f"  {p}: {len(df):,} bars  {df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()}")

# Intersection of all 12 pair timestamps → common grid
print("\nBuilding common timestamp grid (intersection) ...")
common_ts_set = set(pair_df[PAIRS[0]].timestamp.values.astype(np.int64))
for p in PAIRS[1:]:
    common_ts_set &= set(pair_df[p].timestamp.values.astype(np.int64))
common_ts_ns = np.array(sorted(common_ts_set), dtype=np.int64)
N = len(common_ts_ns)
print(f"  {N:,} common bars")

# Build aligned (N, 12) arrays
mid_mat = np.zeros((N, len(PAIRS)), dtype=np.float64)
sp_mat  = np.zeros((N, len(PAIRS)), dtype=np.float64)
lr_mat  = np.zeros((N, len(PAIRS)), dtype=np.float64)

common_ts_ns_set = set(common_ts_ns)
for pi, p in enumerate(PAIRS):
    df  = pair_df[p]
    ts  = df.timestamp.values.astype(np.int64)
    mask = np.isin(ts, common_ts_ns)
    df_c = df[mask].reset_index(drop=True)
    mid  = df_c.mid_c.values
    sp   = df_c.sp.values
    mid_mat[:, pi] = mid
    sp_mat[:, pi]  = sp
    lr_mat[1:, pi] = np.log(mid[1:] / mid[:-1])

split = int(N * IS_FRAC)
print(f"  IS: idx 0–{split-1}  "
      f"({pd.Timestamp(common_ts_ns[0], unit='ns', tz='UTC').date()} → "
      f"{pd.Timestamp(common_ts_ns[split-1], unit='ns', tz='UTC').date()})")
print(f"  OOS: idx {split}–{N-1}  "
      f"({pd.Timestamp(common_ts_ns[split], unit='ns', tz='UTC').date()} → "
      f"{pd.Timestamp(common_ts_ns[-1], unit='ns', tz='UTC').date()})")


# ══════════════════════════════════════════════════════════════════════════════
# Compute StrengthSpread signals  (N × 12 array for each sig_tf)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing StrengthSpread signals ...")
lr_df = pd.DataFrame(lr_mat, columns=PAIRS)

ss_all = {}  # sig_tf → (N, 12) float64

for sig_tf in SIG_TFS:
    roll = lr_df.rolling(sig_tf, min_periods=sig_tf).sum().values  # (N, 12)

    # strength[bar, curr] = dot(roll[bar, :], SIGN_MAT[curr, :])
    str_mat = roll @ SIGN_MAT.T     # (N, 8)

    # Cross-sectional z-score at each bar
    mu = np.nanmean(str_mat, axis=1, keepdims=True)
    sd = np.nanstd (str_mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-15, 1e-15, sd)
    z  = (str_mat - mu) / sd        # (N, 8)  — NaN during warmup

    # StrengthSpread per pair
    ss = np.zeros((N, len(PAIRS)), dtype=np.float64)
    for pi, p in enumerate(PAIRS):
        b, q = p.split("_")
        ss[:, pi] = z[:, CUR_IDX[b]] - z[:, CUR_IDX[q]]

    ss = np.where(np.isnan(ss), 0.0, ss)   # warmup → neutral
    ss_all[sig_tf] = ss

    oos_nonzero = (ss[split:] != 0.0).sum()
    print(f"  sig_tf={sig_tf:3d}bars ({sig_tf*5:4d}min)  "
          f"OOS non-zero bars: {oos_nonzero:,}  "
          f"spread range [{ss[split:].min():.2f}, {ss[split:].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# Numba fixed-hold backtest kernel
# ══════════════════════════════════════════════════════════════════════════════
@njit
def backtest_fixed_hold(mc, sp, sig, hold, counter, pip, s, e):
    """Enter on each new signal, hold for `hold` bars, exit, repeat.
    counter=True: fade signal (short when sig>0, long when sig<0)."""
    n       = e - s
    max_tr  = n // max(hold, 1) + 2
    trades  = np.empty(max_tr, np.float64)
    n_tr    = 0
    cum_pnl = 0.0
    equity  = np.zeros(n, np.float64)

    ii = 0
    while ii < n:
        i = s + ii
        v = sig[i]
        if v == 0.0:
            ii += 1
            continue

        pos = 1 if v > 0.0 else -1
        if counter:
            pos = -pos

        half_e = sp[i] / 2.0
        entry  = mc[i] + half_e if pos == 1 else mc[i] - half_e

        exit_ii = min(ii + hold, n - 1)
        j       = s + exit_ii
        half_x  = sp[j] / 2.0
        exit_p  = mc[j] - half_x if pos == 1 else mc[j] + half_x

        pnl = (exit_p - entry) / pip if pos == 1 else (entry - exit_p) / pip
        cum_pnl      += pnl
        trades[n_tr]  = pnl
        n_tr += 1
        for jj in range(ii, exit_ii + 1):
            equity[jj] = cum_pnl

        ii = exit_ii + 1

    return trades[:n_tr], equity


def stats(trades, equity, s, e):
    if len(trades) == 0:
        return None
    t0  = pd.Timestamp(common_ts_ns[s],     unit="ns", tz="UTC")
    t1  = pd.Timestamp(common_ts_ns[e - 1], unit="ns", tz="UTC")
    t_d = (t1 - t0).total_seconds() / 86400 * 5 / 7
    wins   = trades[trades > 0];  losses = trades[trades <= 0]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
    ppd    = trades.sum() / t_d if t_d > 0 else 0.0
    dd     = (equity - np.maximum.accumulate(equity)).min()
    return dict(n=len(trades), wr=wr, pf=pf, ppd=ppd, dd=dd,
                aw=wins.mean()   if len(wins)   else 0.0,
                al=losses.mean() if len(losses) else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Main sweep  (4 × 11 × 2 × 12 = 1056 configs)
# ══════════════════════════════════════════════════════════════════════════════
n_total = len(SIG_TFS) * len(HOLDS) * 2 * len(PAIRS)
print(f"\nSweeping {n_total} configs ...")

rows = []
for pi, pair in enumerate(PAIRS):
    mc  = mid_mat[:, pi]
    sp  = sp_mat[:, pi]
    pip = PIP[pair]

    for sig_tf in SIG_TFS:
        sig = ss_all[sig_tf][:, pi]

        for hold in HOLDS:
            for counter in [False, True]:
                ti, ei = backtest_fixed_hold(mc, sp, sig, hold, counter, pip, 0, split)
                to, eo = backtest_fixed_hold(mc, sp, sig, hold, counter, pip, split, N)
                si = stats(ti, ei, 0, split)
                so = stats(to, eo, split, N)
                rows.append(dict(
                    pair    = pair,
                    sig_tf  = sig_tf,
                    hold    = hold,
                    d       = "ctr" if counter else "trd",
                    is_n    = si["n"]   if si else 0,
                    is_wr   = si["wr"]  if si else 0.0,
                    is_ppd  = si["ppd"] if si else 0.0,
                    oos_n   = so["n"]   if so else 0,
                    oos_wr  = so["wr"]  if so else 0.0,
                    oos_ppd = so["ppd"] if so else 0.0,
                    oos_dd  = so["dd"]  if so else 0.0,
                ))
    print(f"  {pair} ✓")

rdf = pd.DataFrame(rows).sort_values("oos_ppd", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio: top-3 vs bottom-3 at best config
# ══════════════════════════════════════════════════════════════════════════════
print("\nTop-3 vs Bottom-3 portfolio test ...")

# At each bar, which 3 currencies are strongest / weakest?
# Trade only pairs where one currency is in top-3 AND other is in bottom-3.
# Signal = ctr or trd based on which side each currency is.

@njit
def backtest_portfolio(mc_all, sp_all, pips, pair_b, pair_q, z_all, hold, counter, s, e):
    """
    mc_all: (N, 12), sp_all: (N, 12), z_all: (N, 8)
    pair_b[pi], pair_q[pi]: base/quote currency indices for pair pi
    Only enter when base ∈ top-3 AND quote ∈ bottom-3 (trend) or reversed (counter).
    Aggregate P&L across all active pairs (normalized per trade).
    """
    n      = e - s
    max_tr = (n // max(hold, 1) + 2) * 12
    trades = np.empty(max_tr, np.float64)
    n_tr   = 0
    cum_pnl = 0.0
    equity  = np.zeros(n, np.float64)
    n_pairs = mc_all.shape[1]

    # Per-pair hold counter
    hold_left = np.zeros(n_pairs, np.int64)
    positions = np.zeros(n_pairs, np.int64)  # +1=long, -1=short, 0=flat
    entry_p   = np.zeros(n_pairs, np.float64)

    ii = 0
    while ii < n:
        i = s + ii

        # Rank 8 currencies by strength at this bar
        z = z_all[i]
        # argsort ascending → top-3 are last 3 indices, bottom-3 are first 3
        # Use simple O(8) rank since numba can do it inline
        ranks = np.zeros(8, np.int64)
        for c in range(8):
            r = 0
            for cc in range(8):
                if z[cc] < z[c]:
                    r += 1
            ranks[c] = r   # rank 0=weakest, 7=strongest
        # top3: rank >= 5,  bottom3: rank <= 2

        bar_pnl = 0.0

        for pi in range(n_pairs):
            bi = pair_b[pi]
            qi = pair_q[pi]

            # Exit if hold expired
            if hold_left[pi] > 0:
                hold_left[pi] -= 1
                if hold_left[pi] == 0 and positions[pi] != 0:
                    pos = positions[pi]
                    half_x = sp_all[i, pi] / 2.0
                    ep = mc_all[i, pi] - half_x if pos == 1 else mc_all[i, pi] + half_x
                    pnl = (ep - entry_p[pi]) / pips[pi] if pos == 1 else (entry_p[pi] - ep) / pips[pi]
                    bar_pnl += pnl
                    trades[n_tr] = pnl; n_tr += 1
                    cum_pnl += pnl
                    positions[pi] = 0

            # Enter if flat and signal qualifies
            if positions[pi] == 0:
                rb, rq = ranks[bi], ranks[qi]
                # trend: base strong (top3) + quote weak (bottom3) → LONG
                in_top3_b    = rb >= 5
                in_bottom3_q = rq <= 2
                in_bottom3_b = rb <= 2
                in_top3_q    = rq >= 5

                pos = 0
                if not counter:
                    if in_top3_b and in_bottom3_q:   pos = 1
                    elif in_bottom3_b and in_top3_q: pos = -1
                else:
                    if in_top3_b and in_bottom3_q:   pos = -1
                    elif in_bottom3_b and in_top3_q: pos = 1

                if pos != 0:
                    half_e = sp_all[i, pi] / 2.0
                    entry_p[pi]   = mc_all[i, pi] + half_e if pos == 1 else mc_all[i, pi] - half_e
                    positions[pi] = pos
                    hold_left[pi] = hold

        equity[ii] = cum_pnl
        ii += 1

    return trades[:n_tr], equity


pair_b_idx = np.array([CUR_IDX[p.split("_")[0]] for p in PAIRS], dtype=np.int64)
pair_q_idx = np.array([CUR_IDX[p.split("_")[1]] for p in PAIRS], dtype=np.int64)
pips_arr   = np.array([PIP[p] for p in PAIRS], dtype=np.float64)

# Test best sig_tf from single-pair sweep at each direction
best_by_dir = {}
for d in ["trd", "ctr"]:
    sub = rdf[rdf.d == d]
    best_row = sub.loc[sub.oos_ppd.idxmax()]
    best_by_dir[d] = (int(best_row.sig_tf), int(best_row.hold))

pf_rows = []
for d, counter in [("trd", False), ("ctr", True)]:
    for sig_tf in SIG_TFS:
        for hold in [8, 32, 128, 512]:
            z_mat = ss_all[sig_tf]   # (N, 12)  — actually we need raw z scores
            # Recompute raw z per bar for portfolio
            roll = lr_df.rolling(sig_tf, min_periods=sig_tf).sum().values
            str_mat = roll @ SIGN_MAT.T
            mu2 = np.nanmean(str_mat, axis=1, keepdims=True)
            sd2 = np.nanstd (str_mat, axis=1, keepdims=True)
            sd2 = np.where(sd2 < 1e-15, 1e-15, sd2)
            z_raw = np.where(np.isnan(str_mat), 0.0, (str_mat - mu2) / sd2)

            for seg, (s, e) in [("IS", (0, split)), ("OOS", (split, N))]:
                tp, ep = backtest_portfolio(
                    mid_mat, sp_mat, pips_arr, pair_b_idx, pair_q_idx,
                    z_raw, hold, counter, s, e)
                st = stats(tp, ep, s, e)
                pf_rows.append(dict(
                    seg=seg, sig_tf=sig_tf, hold=hold, d=d,
                    n=st["n"] if st else 0,
                    wr=st["wr"] if st else 0.0,
                    ppd=st["ppd"] if st else 0.0,
                    dd=st["dd"] if st else 0.0,
                ))

# Deduplicate z_raw recompute (minor redundancy above is fine for research)
pf_df = pd.DataFrame(pf_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════
W = 122
bar = "━" * W

def fr(r, pfx=""):
    return (f"{pfx}{r['pair']:10s} tf={r['sig_tf']:3d} hold={r['hold']:5d} {r['d']:4s}  "
            f"IS: n={r['is_n']:5.0f} WR={r['is_wr']:5.1f}% p/d={r['is_ppd']:+8.2f}  "
            f"OOS: n={r['oos_n']:5.0f} WR={r['oos_wr']:5.1f}% p/d={r['oos_ppd']:+9.2f} DD={r['oos_dd']:+8.1f}")

print(f"\n{bar}")
print("TOP 50 CONFIGS BY OOS p/d")
print(bar)
for _, r in rdf.head(50).iterrows():
    print(fr(r))

print(f"\n{bar}")
print("WORST 10 CONFIGS BY OOS p/d")
print(bar)
for _, r in rdf.tail(10).iterrows():
    print(fr(r))

TF_LABEL = {1: "M5 ( 1bar)", 3: "M15( 3bar)", 12: "H1 (12bar)", 48: "H4 (48bar)"}
print(f"\n{bar}")
print("AGGREGATE: mean OOS p/d by [sig_tf, direction]")
print(bar)
for sig_tf in SIG_TFS:
    for d in ["trd", "ctr"]:
        sub = rdf[(rdf.sig_tf == sig_tf) & (rdf.d == d)]
        best = sub.loc[sub.oos_ppd.idxmax()]
        print(f"  {TF_LABEL[sig_tf]} {d}  "
              f"mean={sub.oos_ppd.mean():+8.2f}  "
              f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
              f"best={sub.oos_ppd.max():+8.2f} "
              f"({best['pair']} hold={best['hold']})")

print(f"\n{bar}")
print("AGGREGATE: mean OOS p/d by hold (across all sig_tf, pairs, dirs)")
print(bar)
for h in HOLDS:
    sub = rdf[rdf.hold == h]
    print(f"  hold={h:5d}bars = {h*5//60:3d}h{(h*5)%60:02d}m  "
          f"mean={sub.oos_ppd.mean():+8.2f}  "
          f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
          f"best={sub.oos_ppd.max():+8.2f}")

print(f"\n{bar}")
print("AGGREGATE: mean OOS p/d by pair (across all sig_tf, holds, dirs)")
print(bar)
for p in PAIRS:
    sub = rdf[rdf.pair == p]
    best = sub.loc[sub.oos_ppd.idxmax()]
    print(f"  {p:12s}  mean={sub.oos_ppd.mean():+8.2f}  "
          f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
          f"best={sub.oos_ppd.max():+8.2f} "
          f"(tf={best['sig_tf']} hold={best['hold']} {best['d']})")

print(f"\n{bar}")
print("PORTFOLIO: top-3 vs bottom-3 currencies — IS/OOS by [sig_tf, hold, direction]")
print(bar)
pf_is  = pf_df[pf_df.seg == "IS"].copy()
pf_oos = pf_df[pf_df.seg == "OOS"].copy()
merged_pf = pf_is.merge(pf_oos, on=["sig_tf", "hold", "d"], suffixes=("_is", "_oos"))
merged_pf = merged_pf.sort_values("ppd_oos", ascending=False)

print(f"  {'tf':3s} {'hold':5s} {'d':4s}  "
      f"IS: {'n':5s} {'WR%':6s} {'p/d':7s}  "
      f"OOS: {'n':5s} {'WR%':6s} {'p/d':8s} {'DD':7s}")
for _, r in merged_pf.iterrows():
    print(f"  {r.sig_tf:3d} {r.hold:5d} {r.d:4s}  "
          f"IS: {r.n_is:5.0f} {r.wr_is:5.1f}% {r.ppd_is:+7.2f}  "
          f"OOS: {r.n_oos:5.0f} {r.oos_wr:5.1f}% {r.ppd_oos:+8.2f} {r.dd_oos:+7.1f}"
          if 'oos_wr' in r else
          f"  {r.sig_tf:3d} {r.hold:5d} {r.d:4s}  "
          f"IS: {r.n_is:5.0f} {r.wr_is:5.1f}% {r.ppd_is:+7.2f}  "
          f"OOS: {r.n_oos:5.0f} {r.wr_oos:5.1f}% {r.ppd_oos:+8.2f} {r.dd_oos:+7.1f}")

out = Path(__file__).parent / "results.csv"
rdf.to_csv(out, index=False)
print(f"\nFull results → {out}")
