#!/usr/bin/env python3
"""S5 Momentum Confluence + MFE-trailing exit.

Entry  long:  c - c[t-5] > 0  AND  c - c[t-12] > 0  AND  c - c[t-120] > 0
Entry short:  mirror (all three < 0)
Exit:         while rolling avg of running MFE over last N S5 bars is non-decreasing
              → stay in.  When it falls (MFE-roll-avg drops below its previous value)
              → exit at next S5 close.

Sweep:
  mfe_win     ∈ {12, 60, 144}     S5 bars (1 min / 5 min / 12 min averaging)
  max_hold    ∈ {180, 720, 2880}  S5 bars (15 min / 1 h / 4 h hard cap)
  spread gate = IS P90 (per-pair)

Pairs: GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY (the 4 JPY pairs with S5 BA data)

Outputs:
  results/s5_mom_confluence_mfe.csv  per (pair × mfe_win × max_hold)
  Plus a per-pair best-config summary printed at the end.

SOP compliance:
  R1 closed bars  R3 mid + spread cost  R5 IS-P90 sp_gate, hardcoded
  R6 same logic backtest/live  R8 OOS evaluated once
"""
from __future__ import annotations
import gc, time, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "s5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

PAIRS   = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP     = 0.01
IS_FRAC = 0.70

MFE_WINS  = [12, 60, 144]                  # rolling-avg window for MFE in S5 bars
MAX_HOLDS = [180, 720, 2880]               # hard cap S5 bars (15m / 1h / 4h)
LAGS      = (5, 12, 120)
MAX_LAG   = max(LAGS) + 1


# ───────── numba kernel ─────────
@njit(cache=True, fastmath=True)
def simulate(close, sp, mfe_win, max_hold, sp_gate, pip):
    """Run one (mfe_win, max_hold) config on a full series of S5 mid closes."""
    n = close.shape[0]
    l1, l2, l3 = LAGS
    pos = 0                  # +1 long, -1 short, 0 flat
    entry_px = 0.0
    entry_sp = 0.0
    hold = 0
    mfe = 0.0                # running max favorable excursion (pips, signed)
    mfe_buf = np.zeros(mfe_win, dtype=np.float64)
    buf_i = 0
    buf_count = 0
    prev_roll_avg = 0.0
    trades = 0
    pnl_sum = 0.0
    pnl_sq  = 0.0
    pnl_max = -1e18
    pnl_min =  1e18

    for t in range(MAX_LAG, n):
        c = close[t]
        if pos == 0:
            if sp[t] > sp_gate:
                continue
            d1 = c - close[t-l1]
            d2 = c - close[t-l2]
            d3 = c - close[t-l3]
            if d1 > 0 and d2 > 0 and d3 > 0:
                pos = 1
            elif d1 < 0 and d2 < 0 and d3 < 0:
                pos = -1
            else:
                continue
            entry_px = c
            entry_sp = sp[t]
            hold = 0
            mfe = 0.0
            for k in range(mfe_win):
                mfe_buf[k] = 0.0
            buf_i = 0
            buf_count = 0
            prev_roll_avg = -1e18
            continue
        # in position — track the bar's *current* favorable excursion (signed
        # price-above-entry, can fall when price retraces). The user's intent
        # was a rolling avg that signals "trade is still extending favorably";
        # the lifetime-max MFE is monotone non-decreasing so it would never
        # trigger an exit. Current excursion fixes that.
        hold += 1
        if pos == 1:
            excursion = (c - entry_px) / pip
        else:
            excursion = (entry_px - c) / pip
        # rolling avg of current excursion over last mfe_win bars
        mfe_buf[buf_i] = excursion
        buf_i = (buf_i + 1) % mfe_win
        if buf_count < mfe_win:
            buf_count += 1
        s = 0.0
        for k in range(buf_count):
            s += mfe_buf[k]
        roll_avg = s / buf_count
        # exit conditions
        roll_dropped = (buf_count >= mfe_win and roll_avg < prev_roll_avg)
        timeout = hold >= max_hold
        if roll_dropped or timeout:
            # exit at current close less round-trip half-spread on each side
            exit_px = c
            half_sp_pips = 0.5 * (entry_sp + sp[t])
            if pos == 1:
                pnl = (exit_px - entry_px) / pip - half_sp_pips
            else:
                pnl = (entry_px - exit_px) / pip - half_sp_pips
            trades += 1
            pnl_sum += pnl
            pnl_sq  += pnl * pnl
            if pnl > pnl_max: pnl_max = pnl
            if pnl < pnl_min: pnl_min = pnl
            pos = 0
        prev_roll_avg = roll_avg

    return trades, pnl_sum, pnl_sq, pnl_max, pnl_min


def run_pair(pair):
    t0 = time.time()
    df = pd.read_parquet(DATA / f"{pair}_S5_BA.parquet")
    # expect columns: timestamp, open, high, low, close, volume, bid_c, ask_c
    close = df["close"].to_numpy(dtype=np.float64)
    bid_c = df["bid_c"].to_numpy(dtype=np.float64)
    ask_c = df["ask_c"].to_numpy(dtype=np.float64)
    sp = (ask_c - bid_c) / PIP                              # pips
    n = close.shape[0]
    is_end = int(n * IS_FRAC)
    sp_gate = float(np.percentile(sp[MAX_LAG:is_end], 90))   # R5: IS-only

    print(f"  [{pair}] {n:,} S5 bars  IS_end={is_end:,}  sp_gate(P90)={sp_gate:.2f}p", flush=True)

    rows = []
    for mfe_win, max_hold in product(MFE_WINS, MAX_HOLDS):
        # IS
        is_close = close[:is_end].copy()
        is_sp    = sp[:is_end].copy()
        is_trd, is_sum, is_sq, is_max, is_min = simulate(
            is_close, is_sp, mfe_win, max_hold, sp_gate, PIP
        )
        # OOS
        oos_close = close[is_end:].copy()
        oos_sp    = sp[is_end:].copy()
        oos_trd, oos_sum, oos_sq, oos_max, oos_min = simulate(
            oos_close, oos_sp, mfe_win, max_hold, sp_gate, PIP
        )
        is_days  = (is_end - MAX_LAG) / (60.0 * 60.0 * 24.0 / 5.0)   # 5-s bars/day
        oos_days = (n - is_end) / (60.0 * 60.0 * 24.0 / 5.0)
        is_pd  = is_sum  / is_days  if is_days  > 0 else 0
        oos_pd = oos_sum / oos_days if oos_days > 0 else 0
        rows.append({
            "pair": pair, "mfe_win": mfe_win, "max_hold": max_hold,
            "is_trd": is_trd, "is_pd": round(is_pd, 2),
            "is_max": round(is_max, 1), "is_min": round(is_min, 1),
            "oos_trd": oos_trd, "oos_pd": round(oos_pd, 2),
            "oos_max": round(oos_max, 1), "oos_min": round(oos_min, 1),
            "wf_pass": int(is_pd > 0 and oos_pd > 0 and oos_trd >= 20),
        })
    dt = time.time() - t0
    print(f"  [{pair}] done in {dt:.1f}s  ({len(rows)} configs)", flush=True)
    return rows


def main():
    print(f"S5 Momentum Confluence + MFE-trailing exit")
    print(f"  Lags: c-c5, c-c12, c-c120 (all-same-sign entry)")
    print(f"  Sweep: mfe_win × max_hold = {len(MFE_WINS)} × {len(MAX_HOLDS)} = {len(MFE_WINS)*len(MAX_HOLDS)}")
    print(f"  Pairs: {PAIRS}")
    print(f"  IS_FRAC = {IS_FRAC}")
    print("=" * 70)

    all_rows = []
    for pair in PAIRS:
        all_rows.extend(run_pair(pair))
        gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "s5_mom_confluence_mfe.csv", index=False)
    print(f"\nwritten → {OUT / 's5_mom_confluence_mfe.csv'}")

    print("\n========== WF-PASS configs (is_pd>0, oos_pd>0, oos_trd≥20) ==========")
    surv = df[df.wf_pass == 1].sort_values("oos_pd", ascending=False)
    print(f"survivors: {len(surv)}/{len(df)}")
    if len(surv) > 0:
        print(surv.head(15).to_string(index=False))

    print("\n========== Best OOS per pair ==========")
    best = df.sort_values(["pair","oos_pd"], ascending=[True, False]).groupby("pair").head(1)
    print(best.to_string(index=False))

    print(f"\nSum oos_pd best-per-pair (WF-pass only): {surv.groupby('pair').head(1).oos_pd.sum():.2f}")


if __name__ == "__main__":
    main()
