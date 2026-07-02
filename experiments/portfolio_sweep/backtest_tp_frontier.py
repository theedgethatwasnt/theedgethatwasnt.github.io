#!/usr/bin/env python3
"""
TP Frontier — Current Signals
==============================
Tests the two deployed signals across a TP ladder (3-50p) to map
the full profit-per-day frontier and find the minimum viable TP.

Signals:
  pmom   : raw close momentum M15+M5 lags=(1,3,8)   [deployed @ TP=10p]
  sma16  : SMA16 H1+M30      lags=(8,10,15)          [deployed @ TP=20p]

Output: results/tp_frontier.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT  = Path(__file__).resolve().parents[3]
DATA     = PROJECT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC  = 0.70

TP_LEVELS = [3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]

SIGNALS = [
    dict(label="pmom_m15m5",  tf1="15min", tf2="5min",  sma_n=0,  lags=(1,3,8)),
    dict(label="sma16_h1m30", tf1="1h",    tf2="30min", sma_n=16, lags=(8,10,15)),
]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


@njit(cache=True)
def sim_tp(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    n = len(mid)
    pnl_sum = 0.0; n_trades = 0; n_wins = 0
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


def build_signal(df, tf1, tf2, sma_n, lags):
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        if sma_n > 0:
            rs = rs.rolling(sma_n, min_periods=sma_n).mean()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


def run_wf(df, sig, pip, tp_pips, sp_gate, n_is):
    fold = n_is // 3; passes = 0
    for f in range(3):
        s = f * fold; e = s + fold if f < 2 else n_is
        days = (e - s) / 288
        mid = df["close"].values[s:e].astype(np.float64) / pip
        bid = df["bid_c"].values[s:e].astype(np.float64) / pip
        ask = df["ask_c"].values[s:e].astype(np.float64) / pip
        sp  = (ask - bid)
        sv  = sig.values[s:e].astype(np.int8)
        p, n, _ = sim_tp(mid, bid, ask, sp, sv, tp_pips, sp_gate)
        if n > 0 and p / days > 0: passes += 1
    return passes


print("Loading data …")
cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    p = pip_sz(pair); n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / p, 90))
    cache[pair] = dict(df=df, pip=p, sp_gate=sp_gate, n_is=n_is,
                       oos_days=len(df.iloc[n_is:]) / 288)

rows = []
for cfg in SIGNALS:
    print(f"\n{cfg['label']} …")
    for pair in PAIRS:
        c = cache[pair]
        df = c["df"]; pip = c["pip"]; sg = c["sp_gate"]; n_is = c["n_is"]
        sig = build_signal(df, cfg["tf1"], cfg["tf2"], cfg["sma_n"], cfg["lags"])

        for tp in TP_LEVELS:
            wf3 = run_wf(df, sig, pip, tp, sg, n_is)

            oos = df.iloc[n_is:]
            mid = oos["close"].values.astype(np.float64) / pip
            bid = oos["bid_c"].values.astype(np.float64) / pip
            ask = oos["ask_c"].values.astype(np.float64) / pip
            sp  = (ask - bid)
            sv  = sig.values[n_is:].astype(np.int8)
            p_sum, n_t, n_w = sim_tp(mid, bid, ask, sp, sv, tp, sg)

            oos_pd = p_sum / c["oos_days"] if c["oos_days"] > 0 else 0.0
            oos_wr = 100.0 * n_w / n_t if n_t > 0 else 0.0
            tpd    = n_t / c["oos_days"] if c["oos_days"] > 0 else 0.0

            rows.append(dict(
                signal=cfg["label"], pair=pair, tp=tp,
                wf3=wf3, wf_pass=int(wf3==3),
                oos_pd=round(oos_pd, 3), oos_wr=round(oos_wr, 1),
                tpd=round(tpd, 3), n_oos=n_t,
            ))

    # Portfolio totals per TP
    df_cfg = pd.DataFrame([r for r in rows if r["signal"] == cfg["label"]])
    print(f"  {'TP':>4}  {'n_is3':>5}  {'portf_pd':>9}  {'tpd':>6}  {'wf_pass_pct':>11}")
    for tp in TP_LEVELS:
        sub = df_cfg[df_cfg["tp"] == tp]
        n3  = sub["wf_pass"].sum()
        ppd = sub["oos_pd"].sum()
        tpd = sub["tpd"].sum()
        pct = 100*n3/len(PAIRS)
        print(f"  {tp:>4}  {n3:>5}  {ppd:>+9.1f}  {tpd:>6.1f}  {pct:>10.0f}%")

df_out = pd.DataFrame(rows)
df_out.to_csv(RESULTS / "tp_frontier.csv", index=False)
print(f"\nSaved {len(df_out)} rows → results/tp_frontier.csv")
