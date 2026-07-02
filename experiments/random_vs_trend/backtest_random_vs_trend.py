#!/usr/bin/env python3
"""
If direction can't be predicted, what does money-management alone do?
2:1 risk:reward (TP=T, SL=2T) under three entry rules, net of spread:
  A. RANDOM direction
  B. WITH the current H1 trend  (sign of mid[t]-mid[t-12 M5])
  C. AGAINST the H1 trend       (contrarian — our data says this should beat B)
One position at a time, re-enter when flat. Intrabar fills, SL checked first
(conservative). Mid OHLC for the path, full spread deducted at entry (SOP R3).
Reports net pips, win rate, expectancy/trade and MAX DRAWDOWN per arm, multi-pair.

usage: backtest_random_vs_trend.py [TP_pips=20]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
PAIRS = ["AUD_JPY", "CAD_JPY", "CHF_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "USD_JPY"]
PIP = 0.01
TP = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
H1 = 12   # M5 bars per hour


@njit(cache=True)
def sim(o, h, l, c, sp, dir_stream, tp_px, tp_pips, sl_mult):
    """dir_stream[i] in {-1,0,+1} = intended entry direction if flat at bar i.
       tp_px = TP in PRICE units (for level checks); tp_pips = TP in PIPS (for P&L).
       Returns per-trade pnl (PIPS) array."""
    n = len(c)
    pnl = np.empty(n, np.float64); k = 0
    pos = 0; tp = 0.0; sl = 0.0; spread_paid = 0.0
    for i in range(n):
        if pos == 0:
            d = dir_stream[i]
            if d != 0:
                pos = d; entry = c[i]; spread_paid = sp[i]
                if d == 1:
                    tp = entry + tp_px; sl = entry - tp_px * sl_mult
                else:
                    tp = entry - tp_px; sl = entry + tp_px * sl_mult
        else:
            hit = 0
            if pos == 1:
                if l[i] <= sl: hit = -1            # SL first (conservative)
                elif h[i] >= tp: hit = 1
            else:
                if h[i] >= sl: hit = -1
                elif l[i] <= tp: hit = 1
            if hit != 0:
                gross_pips = (tp_pips if hit == 1 else -tp_pips * sl_mult)
                pnl[k] = gross_pips - spread_paid; k += 1
                pos = 0
    return pnl[:k]


def maxdd(pnl):
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq)
    return float((peak - eq).max()) if len(eq) else 0.0


def arm_stats(pnl):
    if len(pnl) == 0:
        return dict(n=0, net=0, wr=0, exp=0, dd=0)
    return dict(n=len(pnl), net=pnl.sum(), wr=(pnl > 0).mean() * 100,
                exp=pnl.mean(), dd=maxdd(pnl))


def main():
    tp_px = TP * PIP
    print(f"Random vs H1-trend, TP={TP:.0f}p SL={2*TP:.0f}p (2:1 risk), net of spread, {len(PAIRS)} JPY crosses.\n")
    agg = {"RANDOM": [], "WITH_H1": [], "AGAINST_H1": []}
    perpair_dd = {"RANDOM": [], "WITH_H1": [], "AGAINST_H1": []}
    rng = np.random.default_rng(7)
    for p in PAIRS:
        df = pd.read_parquet(PROJECT / "data" / "m5_ba" / f"{p}_M5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
        o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
        sp = (df["ask_c"].values - df["bid_c"].values) / PIP
        n = len(c)
        # H1 trend sign, causal
        h1 = np.zeros(n, np.int64)
        h1[H1:] = np.sign(c[H1:] - c[:-H1]).astype(np.int64)
        # random arm: average over 5 seeds for a stable DD estimate
        for arm, ds in (("WITH_H1", h1), ("AGAINST_H1", -h1)):
            pnl = sim(o, h, l, c, sp, ds, tp_px, TP, 2.0)
            agg[arm].append(pnl); perpair_dd[arm].append(maxdd(pnl))
        rnd_pnls = []
        for s in range(5):
            r = (rng.integers(0, 2, n) * 2 - 1).astype(np.int64)
            rnd_pnls.append(sim(o, h, l, c, sp, r, tp_px, TP, 2.0))
        # use the median-DD random seed for this pair
        rnd_pnls.sort(key=lambda x: maxdd(x))
        agg["RANDOM"].append(rnd_pnls[2]); perpair_dd["RANDOM"].append(maxdd(rnd_pnls[2]))

    print(f"  {'arm':<12}{'trades':>8}{'net pips':>10}{'WR%':>7}{'exp/trade':>11}{'maxDD':>9}{'worst-pair DD':>15}")
    for arm in ("RANDOM", "WITH_H1", "AGAINST_H1"):
        allpnl = np.concatenate(agg[arm])
        st = arm_stats(allpnl)
        wdd = max(perpair_dd[arm])
        print(f"  {arm:<12}{st['n']:>8}{st['net']:>10.0f}{st['wr']:>7.1f}{st['exp']:>11.3f}{st['dd']:>9.0f}{wdd:>15.0f}")
    print(f"\n  (no-edge expectation with 2:1 + driftless walk: WR~66.7%, exp ~= -spread/trade;"
          f" DD grows with -spread accumulation + variance)")
    print("  per-pair max DD (pips):")
    for arm in ("RANDOM", "WITH_H1", "AGAINST_H1"):
        dds = ", ".join(f"{p.split('_')[0]}={d:.0f}" for p, d in zip(PAIRS, perpair_dd[arm]))
        print(f"    {arm:<12} {dds}")


if __name__ == "__main__":
    main()
