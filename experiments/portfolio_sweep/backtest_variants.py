#!/usr/bin/env python3
"""
Momentum Signal Variants — Full TF/Lag/SMA Sweep (Fast)
=========================================================
Sweeps two signal families across TF combinations, lag triplets, SMA periods,
and TP levels. Signal building uses pre-computed numpy diff matrices (~100x
faster than per-config pandas operations).

Signal families:
  pmom     : raw close momentum (close[t] - close[t-k])       sma_n=0
  sma_mom  : SMA-smoothed momentum (SMA[t] - SMA[t-k])        sma_n=8,12,16,22

TF combos (4 speed levels):
  M15+M5  : fast  (~30min-3h holds)
  M30+M15 : medium (~1-8h)
  H1+M30  : slow  (~4-24h)
  H4+H1   : very slow (~12-96h)

Output: results/variant_sweep.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from numba import njit
import time
import warnings; warnings.filterwarnings("ignore")

PROJECT  = Path(__file__).resolve().parents[3]
DATA     = PROJECT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC  = 0.70

LAG_POOL     = [1, 2, 3, 5, 8, 10, 15, 20]
LAG_TRIPLETS = list(combinations(range(len(LAG_POOL)), 3))  # 56 triplets of indices
LAG_TRIPLETS_VAL = [(LAG_POOL[a], LAG_POOL[b], LAG_POOL[c]) for a,b,c in LAG_TRIPLETS]

TF_COMBOS = [
    ("15min", "5min",  "M15+M5"),
    ("30min", "15min", "M30+M15"),
    ("1h",    "30min", "H1+M30"),
    ("4h",    "1h",    "H4+H1"),
]
SMA_PERIODS = [0, 8, 12, 16, 22]   # 0 = raw pmom
TP_LEVELS   = [5, 10, 15, 20, 30]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


@njit(cache=True)
def sim_tp(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    n = len(mid); pnl_sum = 0.0; n_trades = 0; n_wins = 0
    in_trade = False; dir_ = 0; ep = 0.0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) * dir_ >= tp_pips:
                ex = bid[i] if dir_ == 1 else ask[i]
                p  = (ex - ep) * dir_ - sp[i]
                pnl_sum += p; n_trades += 1
                if p > 0.0: n_wins += 1
                in_trade = False
        else:
            nd = sig[i-1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_trade = True
    return pnl_sum, n_trades, n_wins


@njit(cache=True)
def score_to_sig(score_arr, n_ind):
    n = len(score_arr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if score_arr[i] == n_ind:
            sig[i] = 1
        elif score_arr[i] == 0:
            sig[i] = -1
    return sig


@njit(cache=True)
def compute_score_from_diffs(d1a, d1b, d1c, d2a, d2b, d2c):
    """Count how many of 6 momentum values are positive."""
    n = len(d1a)
    score = np.zeros(n, dtype=np.int8)
    for i in range(n):
        s = np.int8(0)
        if d1a[i] > 0: s += 1
        if d1b[i] > 0: s += 1
        if d1c[i] > 0: s += 1
        if d2a[i] > 0: s += 1
        if d2b[i] > 0: s += 1
        if d2c[i] > 0: s += 1
        score[i] = s
    return score


def run_wf(mid, bid, ask, sp, sig_arr, tp, sp_gate, n_is):
    fold = n_is // 3; passes = 0
    for f in range(3):
        s = f * fold; e = s + fold if f < 2 else n_is
        days = (e - s) / 288.0
        p_sum, n_t, _ = sim_tp(mid[s:e], bid[s:e], ask[s:e], sp[s:e],
                                sig_arr[s:e], tp, sp_gate)
        if n_t > 0 and days > 0 and p_sum / days > 0.0: passes += 1
    return passes


# ── Pre-load data ─────────────────────────────────────────────────────────────
print("Loading 12-pair M5 BA data …")
cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    p = pip_sz(pair); n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / p, 90))
    mid = (df["close"].values / p).astype(np.float64)
    bid = (df["bid_c"].values / p).astype(np.float64)
    ask = (df["ask_c"].values / p).astype(np.float64)
    sp  = (ask - bid)
    cache[pair] = dict(df=df, pip=p, sp_gate=sp_gate, n_is=n_is,
                       oos_days=len(df.iloc[n_is:]) / 288,
                       mid=mid, bid=bid, ask=ask, sp=sp)

# ── Pre-compute diff matrices: (pair, tf, sma_n) → ndarray (n_bars, n_lags) ──
print("Pre-computing diff matrices …")
# diff_mat[(pair, tf, sma_n)][i, k] = series[i] - series[i - LAG_POOL[k]]
# series = raw close or SMA(sma_n) on resampled TF, shift(1) → reindex to M5
all_tfs = set(tf for tf1, tf2, _ in TF_COMBOS for tf in (tf1, tf2))
diff_cache = {}
for pair in PAIRS:
    df = cache[pair]["df"]
    for tf in all_tfs:
        rs_raw = df["close"].resample(tf).last().dropna()
        for sma_n in SMA_PERIODS:
            series = rs_raw if sma_n == 0 else rs_raw.rolling(sma_n, min_periods=sma_n).mean()
            # shift(1) for causality, reindex to M5 bars
            rs_m5 = series.shift(1).reindex(df.index, method="ffill")
            arr   = rs_m5.values.astype(np.float64)
            # Pre-compute all lag diffs as numpy columns
            n = len(arr)
            mat = np.empty((n, len(LAG_POOL)), dtype=np.float32)
            for ki, k in enumerate(LAG_POOL):
                diff = arr - np.concatenate([np.full(k, np.nan), arr[:-k]])
                mat[:, ki] = diff.astype(np.float32)
            diff_cache[(pair, tf, sma_n)] = mat

print(f"  Cached {len(diff_cache)} diff matrices")

# Warm up Numba
_d = np.zeros(100, dtype=np.float64); _s = np.zeros(100, dtype=np.int8)
sim_tp(_d, _d, _d, _d, _s, 5.0, 2.0)
_sc = np.zeros(100, dtype=np.int8)
compute_score_from_diffs(_d, _d, _d, _d, _d, _d)

# ── Main sweep ────────────────────────────────────────────────────────────────
n_total = len(SMA_PERIODS) * len(TF_COMBOS) * len(LAG_TRIPLETS) * len(TP_LEVELS)
print(f"\nSweeping {len(SMA_PERIODS)} sma × {len(TF_COMBOS)} TF × "
      f"{len(LAG_TRIPLETS)} lags × {len(TP_LEVELS)} TP = {n_total} configs …")

rows = []; n_done = 0; t0 = time.time()

for sma_n in SMA_PERIODS:
    sig_label = "pmom" if sma_n == 0 else f"sma{sma_n}"
    for tf1, tf2, tflabel in TF_COMBOS:
        for (ai, bi, ci) in LAG_TRIPLETS:
            la, lb, lc = LAG_POOL[ai], LAG_POOL[bi], LAG_POOL[ci]
            for tp in TP_LEVELS:
                ppd_sum = 0.0; tpd_sum = 0.0; n_is3 = 0

                for pair in PAIRS:
                    c = cache[pair]
                    mat1 = diff_cache[(pair, tf1, sma_n)]
                    mat2 = diff_cache[(pair, tf2, sma_n)]
                    # Pull diff columns as float64 for Numba
                    d1a = mat1[:, ai].astype(np.float64)
                    d1b = mat1[:, bi].astype(np.float64)
                    d1c = mat1[:, ci].astype(np.float64)
                    d2a = mat2[:, ai].astype(np.float64)
                    d2b = mat2[:, bi].astype(np.float64)
                    d2c = mat2[:, ci].astype(np.float64)
                    score = compute_score_from_diffs(d1a, d1b, d1c, d2a, d2b, d2c)
                    sv = score_to_sig(score, 6)

                    wf3 = run_wf(c["mid"], c["bid"], c["ask"], c["sp"],
                                 sv, tp, c["sp_gate"], c["n_is"])
                    n_is3 += int(wf3 == 3)

                    p_sum, n_t, n_w = sim_tp(
                        c["mid"][c["n_is"]:], c["bid"][c["n_is"]:],
                        c["ask"][c["n_is"]:], c["sp"][c["n_is"]:],
                        sv[c["n_is"]:], tp, c["sp_gate"])

                    if c["oos_days"] > 0:
                        ppd_sum += p_sum / c["oos_days"]
                        tpd_sum += n_t   / c["oos_days"]

                rows.append(dict(
                    signal=sig_label, tfs=tflabel, sma_n=sma_n,
                    lags=str((la,lb,lc)), tp=tp,
                    n_is3=n_is3, wf_pass=int(n_is3==12),
                    portfolio_pd=round(ppd_sum, 2),
                    portfolio_tpd=round(tpd_sum, 2),
                ))

                n_done += 1
                if n_done % 500 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"  {n_done}/{n_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

df_out = pd.DataFrame(rows)
df_out.to_csv(RESULTS / "variant_sweep.csv", index=False)
print(f"\nSaved {len(df_out)} rows → results/variant_sweep.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
survivors = df_out[df_out["n_is3"] >= 10].sort_values("portfolio_pd", ascending=False)
print(f"\nConfigs with n_is3 >= 10 (out of 12): {len(survivors)}")
print(survivors.head(25)[["signal","tfs","sma_n","lags","tp","n_is3","portfolio_pd","portfolio_tpd"]].to_string(index=False))
print(f"\nAll-12 WF-passing configs: {df_out['wf_pass'].sum()}")
print(df_out[df_out['wf_pass']==1].sort_values('portfolio_pd', ascending=False).head(25)[
    ["signal","tfs","sma_n","lags","tp","n_is3","portfolio_pd"]].to_string(index=False))
