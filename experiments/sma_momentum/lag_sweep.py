#!/usr/bin/env python3
"""
SMA Momentum — Lag combination + SMA period sweep
==================================================
Lags [1,5,10] and SMA22 were initial guesses.
Sweep all C(8,3)=56 lag triplets × 4 SMA periods to find the
combination that maximises portfolio p/day with IS robustness.

Strategy: cache the SMA series per (sma_n, pair, tf) first —
then building each signal is just lag-diffs (fast).
Numba-compiled simulation loop keeps the inner hot path fast.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from numba import njit
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "EUR_GBP", "AUD_USD",
    "NZD_JPY", "CHF_JPY", "NZD_USD", "CAD_JPY",
]
JPY_PAIRS = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

LAG_CANDIDATES = [1, 2, 3, 5, 8, 10, 15, 20]
SMA_PERIODS    = [16, 22, 34, 55]
TP_LEVELS      = [20, 40]          # benchmark + frontier peak
IS_FRAC        = 0.70


def pip_size(pair):
    return 0.01 if pair in JPY_PAIRS else 0.0001


@njit
def simulate_tp_nb(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    """Numba-compiled TP simulation. Returns (pnl_sum, n_trades, n_wins)."""
    n = len(mid)
    pnl_sum = 0.0
    n_trades = 0
    n_wins   = 0
    in_trade = False
    dir_     = 0
    entry_px = 0.0

    for i in range(1, n):
        if in_trade:
            if (mid[i] - entry_px) * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                p = (exit_px - entry_px) * dir_ - sp[i]
                pnl_sum += p
                n_trades += 1
                if p > 0.0:
                    n_wins += 1
                in_trade = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                entry_px = ask[i] if nd == 1 else bid[i]
                dir_     = nd
                in_trade = True

    return pnl_sum, n_trades, n_wins


def run_sim(df_slice, sig_slice, pip, tp_pips, sp_gate):
    """Wrapper: converts units and calls Numba kernel."""
    bid = df_slice["bid_c"].values.astype(np.float64) / pip
    ask = df_slice["ask_c"].values.astype(np.float64) / pip
    mid = df_slice["close"].values.astype(np.float64) / pip
    sp  = (df_slice["ask_c"].values.astype(np.float64)
           - df_slice["bid_c"].values.astype(np.float64)) / pip
    s   = sig_slice.values.astype(np.int8)
    return simulate_tp_nb(mid, bid, ask, sp, s, tp_pips, sp_gate)


# ── Pre-load raw data ────────────────────────────────────────────────────────

print("Loading pair data …")
raw_data = {}
for pair in PAIRS:
    path = DATA / f"{pair}_M5_BA.parquet"
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_size(pair)
    n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile(
        (df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    oos_days = len(df) / 288 * (1 - IS_FRAC)
    raw_data[pair] = dict(df=df, pip=pip, sp_gate=sp_gate,
                          n_is=n_is, oos_days=oos_days)

# JIT warm-up (compile on a small dummy call)
_d = raw_data["GBP_JPY"]
run_sim(_d["df"].iloc[:1000], pd.Series(np.zeros(1000, dtype=np.int8),
        index=_d["df"].index[:1000]), _d["pip"], 20.0, _d["sp_gate"])
print("Numba kernel compiled.")

# ── Pre-compute SMA series per (sma_n, pair, tf) ────────────────────────────

print("Pre-computing SMA series …")
sma_cache = {}   # key: (sma_n, pair, tf) → pd.Series on M5 index
for sma_n in SMA_PERIODS:
    for pair in PAIRS:
        df = raw_data[pair]["df"]
        for tf in ["1h", "30min"]:
            rs  = df["close"].resample(tf).last().dropna()
            sma = rs.rolling(sma_n, min_periods=sma_n).mean()
            sma = sma.shift(1)                        # causality
            sma_cache[(sma_n, pair, tf)] = sma.reindex(df.index, method="ffill")

print(f"Cached {len(sma_cache)} SMA series.")

# ── Sweep ────────────────────────────────────────────────────────────────────

lag_combos = list(combinations(LAG_CANDIDATES, 3))
total_cfgs = len(SMA_PERIODS) * len(lag_combos)
print(f"\nSweeping {len(SMA_PERIODS)} SMA periods × "
      f"{len(lag_combos)} lag combos = {total_cfgs} configs "
      f"× {len(PAIRS)} pairs × {len(TP_LEVELS)} TPs …\n")

rows = []
done = 0

for sma_n in SMA_PERIODS:
    for lags in lag_combos:
        done += 1
        if done % 56 == 0:
            print(f"  SMA{sma_n} done ({done}/{total_cfgs}) …", flush=True)

        # Build signal for each pair using cached SMA
        pair_signals = {}
        for pair in PAIRS:
            df = raw_data[pair]["df"]
            mom_cols = []
            for tf in ["1h", "30min"]:
                sma = sma_cache[(sma_n, pair, tf)]
                for k in lags:
                    mom_cols.append(sma - sma.shift(k))

            all_moms = pd.concat(mom_cols, axis=1)
            n_ind    = len(mom_cols)           # 2 TFs × 3 lags = 6
            score    = (all_moms > 0).sum(axis=1)
            sig = pd.Series(np.int8(0), index=df.index)
            sig[score >= n_ind] = np.int8(1)
            sig[score <= 0]     = np.int8(-1)
            pair_signals[pair]  = sig

        # Simulate for each pair and TP
        for tp in TP_LEVELS:
            ppd_sum = tpd_sum = 0.0
            n_is3   = n_pos   = 0

            for pair in PAIRS:
                d       = raw_data[pair]
                df      = d["df"]
                pip     = d["pip"]
                sp_gate = d["sp_gate"]
                n_is    = d["n_is"]
                oos_days= d["oos_days"]
                sig     = pair_signals[pair]

                # IS 3-fold
                fold_size = n_is // 3
                is_pass   = 0
                for f in range(3):
                    s = f * fold_size
                    e = s + fold_size if f < 2 else n_is
                    days_f = (e - s) / 288
                    pnl_sum, nt, _ = run_sim(
                        df.iloc[s:e], sig.iloc[s:e], pip, tp, sp_gate)
                    if nt > 0 and pnl_sum / days_f > 0:
                        is_pass += 1

                # OOS
                pnl_sum, nt, nw = run_sim(
                    df.iloc[n_is:], sig.iloc[n_is:], pip, tp, sp_gate)
                ppd = pnl_sum / oos_days if nt > 0 else 0.0
                tpd = nt / oos_days

                ppd_sum += ppd
                tpd_sum += tpd
                if ppd > 0:
                    n_pos += 1
                if is_pass == 3:
                    n_is3 += 1

            rows.append(dict(
                sma_n=sma_n,
                lags=str(lags),
                tp=tp,
                ppd=round(ppd_sum, 2),
                tpd=round(tpd_sum, 3),
                avg_p=round(ppd_sum / tpd_sum, 2) if tpd_sum > 0 else 0,
                n_is3=n_is3,
                n_pos=n_pos,
            ))

df_res = pd.DataFrame(rows)
df_res.to_csv(RESULTS / "sma_lag_sweep.csv", index=False)

# ── Report ───────────────────────────────────────────────────────────────────

baseline_lags = str((1, 5, 10))

for tp in TP_LEVELS:
    sub = df_res[df_res["tp"] == tp].copy()
    # Only configs where all 12 pairs are IS3 (or at least 10)
    sub10 = sub[sub["n_is3"] >= 10].sort_values("ppd", ascending=False)

    print(f"\n{'='*72}")
    print(f"TOP 20 — TP={tp}p — portfolio p/day (IS3≥10/12)")
    print(f"{'='*72}")
    print(f"{'SMA':>5} {'Lags':>18} {'p/day':>8} {'t/day':>7} "
          f"{'IS3':>6} {'pos':>5}")
    print("-"*72)
    for _, r in sub10.head(20).iterrows():
        star = " ◄ baseline" if r["lags"] == baseline_lags and r["sma_n"] == 22 else ""
        print(f"  {int(r.sma_n):>3}  {r.lags:>18}  {r.ppd:>+7.1f}  "
              f"{r.tpd:>6.3f}  {int(r.n_is3):>4}/12  {int(r.n_pos):>4}/12{star}")

    # Baseline row even if outside top 20
    bl = sub[(sub["lags"] == baseline_lags) & (sub["sma_n"] == 22)]
    if not bl.empty:
        r = bl.iloc[0]
        rank = (sub["ppd"] > r["ppd"]).sum() + 1
        print(f"\n  Baseline (SMA22, {baseline_lags}): "
              f"{r.ppd:+.1f} p/d  rank {rank}/{len(sub)}")

print(f"\nFull results → {RESULTS / 'sma_lag_sweep.csv'}")
