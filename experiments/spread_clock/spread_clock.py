#!/usr/bin/env python3
"""
STUDY 1 -- Spread clock: hour-of-week spread profiles per pair.

Method
------
- Data: 12-pair OANDA M5 bid/ask parquets (full history, ~7.4y), columns
  timestamp (UTC), bid_c, ask_c.
- Spread per bar: (ask_c - bid_c) / pip, pip = 0.01 for *_JPY pairs else 0.0001.
  ask_c/bid_c are bar-close quotes -- this is the same spread definition used
  by the live traders (R3a field contract) and by all recorded backtests.
- Bucket: hour-of-week = dow*24 + hour, UTC, dow: Monday=0 .. Sunday=6.
  168 buckets. Per bucket: median, p25, p75, n bars.
- Liquid bucket: n >= 25% of the pair's max bucket count (kills the weekend
  gap + Fri/Sun edge buckets with a handful of stale bars).
- Cheapest / most expensive 8h window: circular scan over the 168 buckets;
  a window is eligible only if all 8 of its buckets are liquid. Score =
  mean of the 8 bucket medians.
- Toll model: one round trip costs one full spread (buy at ask = mid + s/2,
  sell at bid = mid - s/2). Entry and exit assumed at the chosen hour class,
  so toll = median spread of that bucket.
    (a) uniform-random hour  = mean of medians over liquid buckets
    (b) cheapest liquid hour = min median over liquid buckets
    (c) worst liquid hour    = max median over liquid buckets

Outputs (written next to this script on the compute box):
  profiles.parquet  -- pair, how, dow, hour, n, median, p25, p75, liquid
  summary.csv       -- per-pair cheapest/most-expensive 8h window + ratio
  toll.csv          -- per-pair toll at (a)/(b)/(c), savings, % of 1.6p ref
"""
import gc
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("M5BA_DIR", "/root/work/data/m5_ba")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
    "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY",
]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LIQ_FRAC = 0.25   # bucket liquid if n >= 25% of pair's max bucket n
WIN = 8           # window length in hours
REF_SPREAD = 1.6  # reference round-trip toll in pips (EUR_USD median)


def hw_label(hw: int) -> str:
    return f"{DOW_NAMES[hw // 24]} {hw % 24:02d}:00"


def window_label(start_hw: int) -> str:
    end_hw = (start_hw + WIN) % 168
    return f"{hw_label(start_hw)}-{hw_label(end_hw)}"


def main() -> None:
    prof_rows, summ_rows, toll_rows = [], [], []

    for pair in PAIRS:
        path = os.path.join(DATA_DIR, f"{pair}_M5_BA.parquet")
        tbl = pq.read_table(path, columns=["timestamp", "bid_c", "ask_c"])
        ts = tbl.column("timestamp").to_pandas()
        bid = tbl.column("bid_c").to_numpy(zero_copy_only=False).astype(np.float64)
        ask = tbl.column("ask_c").to_numpy(zero_copy_only=False).astype(np.float64)
        del tbl

        pip = 0.01 if pair.endswith("_JPY") else 0.0001
        sp = (ask - bid) / pip
        # sanity: drop non-positive / absurd spreads (>50 pips = bad tick)
        ok = (sp > 0) & (sp < 50)
        n_dropped = int((~ok).sum())
        sp = sp[ok]
        ts = ts[ok]

        hw = (ts.dt.dayofweek.to_numpy() * 24 + ts.dt.hour.to_numpy()).astype(np.int64)
        t0, t1 = ts.iloc[0], ts.iloc[-1]
        del ts

        med = np.full(168, np.nan)
        p25 = np.full(168, np.nan)
        p75 = np.full(168, np.nan)
        nb = np.zeros(168, dtype=np.int64)
        order = np.argsort(hw, kind="stable")
        hw_s, sp_s = hw[order], sp[order]
        bounds = np.searchsorted(hw_s, np.arange(169))
        for b in range(168):
            lo, hi = bounds[b], bounds[b + 1]
            nb[b] = hi - lo
            if hi > lo:
                med[b], p25[b], p75[b] = np.percentile(sp_s[lo:hi], [50, 25, 75])
        del order, hw_s, sp_s, hw, sp

        liquid = nb >= LIQ_FRAC * nb.max()
        for b in range(168):
            prof_rows.append(dict(
                pair=pair, how=b, dow=DOW_NAMES[b // 24], hour=b % 24,
                n=int(nb[b]), median=med[b], p25=p25[b], p75=p75[b],
                liquid=bool(liquid[b]),
            ))

        # circular 8h windows, all buckets must be liquid
        med_ext = np.concatenate([med, med[:WIN]])
        liq_ext = np.concatenate([liquid, liquid[:WIN]])
        win_mean = np.full(168, np.nan)
        for s in range(168):
            seg_liq = liq_ext[s:s + WIN]
            if seg_liq.all():
                win_mean[s] = np.nanmean(med_ext[s:s + WIN])
        cheap_s = int(np.nanargmin(win_mean))
        exp_s = int(np.nanargmax(win_mean))

        liq_meds = med[liquid]
        toll_uniform = float(np.mean(liq_meds))
        toll_cheap = float(np.min(liq_meds))
        toll_worst = float(np.max(liq_meds))
        cheap_hw = int(np.where(liquid)[0][np.argmin(liq_meds)])
        worst_hw = int(np.where(liquid)[0][np.argmax(liq_meds)])

        summ_rows.append(dict(
            pair=pair,
            cheapest_8h=window_label(cheap_s),
            cheapest_8h_med=round(float(win_mean[cheap_s]), 3),
            expensive_8h=window_label(exp_s),
            expensive_8h_med=round(float(win_mean[exp_s]), 3),
            ratio_exp_cheap=round(float(win_mean[exp_s] / win_mean[cheap_s]), 2),
            overall_median=round(float(np.median(liq_meds)), 3),
            n_liquid_buckets=int(liquid.sum()),
            n_bars=int(nb.sum()),
            n_dropped=n_dropped,
            span=f"{t0.date()}..{t1.date()}",
        ))
        toll_rows.append(dict(
            pair=pair,
            toll_uniform=round(toll_uniform, 3),
            toll_cheapest=round(toll_cheap, 3),
            cheapest_hour=hw_label(cheap_hw),
            toll_worst=round(toll_worst, 3),
            worst_hour=hw_label(worst_hw),
            saving_vs_uniform=round(toll_uniform - toll_cheap, 3),
            saving_pct_of_1p6=round(100 * (toll_uniform - toll_cheap) / REF_SPREAD, 1),
            worst_penalty=round(toll_worst - toll_uniform, 3),
        ))
        print(f"{pair}: uniform={toll_uniform:.2f}p cheap={toll_cheap:.2f}p "
              f"worst={toll_worst:.2f}p ratio8h={win_mean[exp_s]/win_mean[cheap_s]:.2f}")
        gc.collect()

    pd.DataFrame(prof_rows).to_parquet(os.path.join(OUT_DIR, "profiles.parquet"), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    pd.DataFrame(toll_rows).to_csv(os.path.join(OUT_DIR, "toll.csv"), index=False)
    print("wrote profiles.parquet, summary.csv, toll.csv ->", OUT_DIR)


if __name__ == "__main__":
    main()
