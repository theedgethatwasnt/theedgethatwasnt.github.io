#!/usr/bin/env python3
"""
Price Momentum Confluence Sweep
================================
Signal: raw close momentum at multiple lags across TF pairs
  mom_k = close_resampled[t] - close_resampled[t-k]  (NO SMA smoothing)
  LONG when all N indicators > 0, SHORT when all N < 0

Sweep:
  - TF pairs: H1+M30, M30+M15, M15+M5
  - All C(8,3)=56 lag triplets from {1,2,3,5,8,10,15,20}
  - TP levels: 5, 10, 15, 20, 30p
  - 12 pairs, IS 3-fold validation
  - Reports: p/d, t/d, IS3 count, mean/P50/P90 hold times
  - Flags configs with mean_h <= 12h (short-hold target) vs baseline ~150h
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70

LAG_POOL = [1, 2, 3, 5, 8, 10, 15, 20]
LAG_TRIPLETS = list(combinations(LAG_POOL, 3))   # 56 combos

TF_PAIRS = [
    ("1h",    "30min", "H1+M30"),
    ("30min", "15min", "M30+M15"),
    ("15min", "5min",  "M15+M5"),
]
TP_LEVELS = [5, 10, 15, 20, 30]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


def simulate_tp(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    """Returns (pnls, hold_bars) arrays."""
    n = len(mid)
    pnls = []; holds = []
    in_trade = False; dir_ = 0; ep = 0.0; ei = 0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) / tp_pips * dir_ >= tp_pips:
                # compute actual exit
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - ep) / tp_pips * dir_ - sp[i])
                holds.append(i - ei)
                in_trade = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True
    return np.array(pnls, dtype=np.float64), np.array(holds, dtype=np.int32)


def simulate_tp_fast(mid, bid, ask, sp_arr, sig, pip, tp_pips, sp_gate):
    """Correct p/d calc: exit at bid/ask, p/d in pips."""
    n = len(mid)
    pnls = []; holds = []
    in_trade = False; dir_ = 0; ep = 0.0; ei = 0
    for i in range(1, n):
        if in_trade:
            cur = (mid[i] - ep) / pip * dir_
            if cur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - ep) / pip * dir_ - sp_arr[i])
                holds.append(i - ei)
                in_trade = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp_arr[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True
    return np.array(pnls, dtype=np.float64), np.array(holds, dtype=np.int32)


def build_price_signal(df, lags, tf1, tf2):
    """Pure price momentum: close[t] - close[t-k] on each TF."""
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        rs_shifted = rs.shift(1)  # causal: use prev-closed bar
        rs_s = rs_shifted.reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


def is3_pass(df, sig, pip, sp_gate, n_is, tp):
    fold = n_is // 3; passes = 0
    for f in range(3):
        s = f * fold; e = s + fold if f < 2 else n_is
        days = (e - s) / 288
        mid  = df["close"].values[s:e].astype(np.float64)
        bid  = df["bid_c"].values[s:e].astype(np.float64)
        ask  = df["ask_c"].values[s:e].astype(np.float64)
        sp   = (ask - bid) / pip
        sv   = sig.values[s:e]
        p, _ = simulate_tp_fast(mid, bid, ask, sp, sv, pip, tp, sp_gate)
        if len(p) > 0 and p.sum() / days > 0:
            passes += 1
    return passes


# ── Pre-load all pair data ──────────────────────────────────────────────────
print("Loading 12-pair M5 BA data …")
cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    p    = pip_sz(pair)
    n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / p, 90))
    cache[pair] = dict(df=df, pip=p, sp_gate=sp_gate, n_is=n_is,
                       oos_days=len(df.iloc[n_is:]) / 288)

# ── Pre-cache resampled close series per (pair, tf) ─────────────────────────
print("Pre-caching resampled close series …")
rs_cache = {}
all_tfs = set()
for tf1, tf2, _ in TF_PAIRS:
    all_tfs.add(tf1); all_tfs.add(tf2)
for pair in PAIRS:
    df = cache[pair]["df"]
    for tf in all_tfs:
        rs_cache[(pair, tf)] = df["close"].resample(tf).last().dropna()

# ── Main sweep ────────────────────────────────────────────────────────────────
print(f"\nSweeping {len(TF_PAIRS)} TF-pairs × {len(LAG_TRIPLETS)} lag triplets "
      f"× {len(TP_LEVELS)} TP levels = "
      f"{len(TF_PAIRS)*len(LAG_TRIPLETS)*len(TP_LEVELS)} configs …\n")

SMA_BASELINE_PPD = 29.8  # SMA16 lags=(8,10,15) TP=20p
SHORT_HOLD_TARGET = 12.0  # hours

rows = []
n_total = len(TF_PAIRS) * len(LAG_TRIPLETS) * len(TP_LEVELS)
n_done  = 0

for tf1, tf2, tflabel in TF_PAIRS:
    for lags in LAG_TRIPLETS:
        for tp in TP_LEVELS:
            ppd_sum = 0.0; tpd_sum = 0.0
            all_h = []; n_is3 = 0

            for pair in PAIRS:
                c   = cache[pair]
                df  = c["df"]
                pip = c["pip"]
                sg  = c["sp_gate"]
                n_is = c["n_is"]

                sig = build_price_signal(df, lags, tf1, tf2)

                # OOS
                oos = df.iloc[n_is:]
                mid = oos["close"].values.astype(np.float64)
                bid = oos["bid_c"].values.astype(np.float64)
                ask = oos["ask_c"].values.astype(np.float64)
                sp  = (ask - bid) / pip
                sv  = sig.values[n_is:]

                p, h = simulate_tp_fast(mid, bid, ask, sp, sv, pip, tp, sg)
                if len(p) > 0:
                    ppd_sum += p.sum() / c["oos_days"]
                    tpd_sum += len(p) / c["oos_days"]
                    all_h.extend(h.tolist())

                n_is3 += 1 if is3_pass(df, sig, pip, sg, n_is, tp) == 3 else 0

            if not all_h:
                n_done += 1
                continue

            h_arr  = np.array(all_h, dtype=np.float32) * 5 / 60  # bars→hours
            mean_h = float(h_arr.mean())
            p50_h  = float(np.percentile(h_arr, 50))
            p90_h  = float(np.percentile(h_arr, 90))

            rows.append(dict(
                tfs=tflabel, lags=str(lags), tp=tp,
                ppd=round(ppd_sum, 2), tpd=round(tpd_sum, 3),
                n_is3=n_is3,
                mean_h=round(mean_h, 1), p50_h=round(p50_h, 1), p90_h=round(p90_h, 1),
                short_hold=(mean_h <= SHORT_HOLD_TARGET),
                beats_baseline=(ppd_sum > SMA_BASELINE_PPD),
            ))
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done}/{n_total} done …", flush=True)

df_out = pd.DataFrame(rows)
df_out.to_csv(RESULTS / "price_mom_sweep.csv", index=False)

# ── Results ─────────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"PRICE MOMENTUM CONFLUENCE — Full Sweep Results")
print(f"{'='*78}")
print(f"Total configs: {len(rows)}  |  Configs beating SMA16 baseline (+{SMA_BASELINE_PPD}p/d): "
      f"{df_out['beats_baseline'].sum()}  |  Short-hold (≤12h mean): "
      f"{df_out['short_hold'].sum()}")

print(f"\n── Top 20 by p/d (all IS3 levels) ──────────────────────────────────────")
print(f"{'TFs':>10} {'Lags':>12} {'TP':>4} {'p/d':>8} {'t/d':>6} "
      f"{'IS3':>5} {'mean_h':>7} {'P50h':>6} {'P90h':>6}  flags")
print("-" * 78)
for _, r in df_out.sort_values("ppd", ascending=False).head(20).iterrows():
    flags = ""
    if r.beats_baseline: flags += " ★BEAT"
    if r.short_hold:     flags += " ⏱SHORT"
    print(f"{r.tfs:>10} {r.lags:>12} {int(r.tp):>3}p "
          f"{r.ppd:>+7.1f} {r.tpd:>6.3f} "
          f"{int(r.n_is3):>3}/12 {r.mean_h:>6.1f}h {r.p50_h:>5.1f}h {r.p90_h:>5.1f}h"
          f"  {flags}")

print(f"\n── Configs with IS3≥10 pairs, sorted by p/d ─────────────────────────────")
good = df_out[df_out["n_is3"] >= 10].sort_values("ppd", ascending=False)
print(f"{'TFs':>10} {'Lags':>12} {'TP':>4} {'p/d':>8} {'t/d':>6} "
      f"{'IS3':>5} {'mean_h':>7} {'P50h':>6} {'P90h':>6}  flags")
print("-" * 78)
for _, r in good.head(30).iterrows():
    flags = ""
    if r.beats_baseline: flags += " ★BEAT"
    if r.short_hold:     flags += " ⏱SHORT"
    print(f"{r.tfs:>10} {r.lags:>12} {int(r.tp):>3}p "
          f"{r.ppd:>+7.1f} {r.tpd:>6.3f} "
          f"{int(r.n_is3):>3}/12 {r.mean_h:>6.1f}h {r.p50_h:>5.1f}h {r.p90_h:>5.1f}h"
          f"  {flags}")

print(f"\n── Short-hold configs (mean_h ≤ 12h), sorted by p/d ───────────────────")
short = df_out[df_out["short_hold"]].sort_values("ppd", ascending=False)
if len(short) > 0:
    for _, r in short.head(20).iterrows():
        print(f"{r.tfs:>10} {r.lags:>12} {int(r.tp):>3}p "
              f"{r.ppd:>+7.1f} {r.tpd:>6.3f} "
              f"{int(r.n_is3):>3}/12 {r.mean_h:>6.1f}h {r.p50_h:>5.1f}h {r.p90_h:>5.1f}h")
else:
    print("  (none found — minimum mean hold exceeds 12h)")

print(f"\n── Hold time summary by TF combo ────────────────────────────────────────")
for tfs, g in df_out.groupby("tfs"):
    print(f"  {tfs}: mean_h range [{g.mean_h.min():.1f}h – {g.mean_h.max():.1f}h]"
          f"  |  P90 range [{g.p90_h.min():.1f}h – {g.p90_h.max():.1f}h]"
          f"  |  best p/d {g.ppd.max():+.1f}")

print(f"\nSMA16 baseline: mean_h ≈ 150h, p/d = +{SMA_BASELINE_PPD}p")
print(f"\nSaved → {RESULTS/'price_mom_sweep.csv'}")
