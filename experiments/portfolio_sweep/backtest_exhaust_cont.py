#!/usr/bin/env python3
"""
Exhaustion Continuation — Ride the Momentum Peak
=================================================
Bar-exhaustion fires at momentum peaks (n_consec same-direction bars + SMA distance).
Instead of fading (Harvester), we enter IN the exhaustion direction.

Signal (M5):
  LONG  when last n_consec bars all BULL (close>open) AND close > SMA14
        AND (close - SMA14) / pip >= dist_mult × sp_gate
  SHORT when last n_consec bars all BEAR (close<open) AND close < SMA14
        AND (SMA14 - close) / pip >= dist_mult × sp_gate
Exit: broker-side TP only (no SL)

Sweep:
  n_consec : 2, 3, 4
  dist_mult: 0.5, 1.0, 1.5, 2.0, 2.5
  tp_pips  : 5, 8, 10, 15, 20

Output: results/exhaust_cont.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT  = Path(__file__).resolve().parents[3]
DATA     = PROJECT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC  = 0.70
SMA_N    = 14

N_CONSEC_OPTS  = [2, 3, 4]
DIST_MULT_OPTS = [0.5, 1.0, 1.5, 2.0, 2.5]
TP_OPTS        = [5.0, 8.0, 10.0, 15.0, 20.0]


def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


@njit(cache=True)
def compute_sma(close, period):
    n = len(close); out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(close[i - period + 1 : i + 1])
    return out


@njit(cache=True)
def build_exhaust_sig(close, open_, sma, n_consec, dist_mult, sp_gate):
    """Returns int8 array: +1=long, -1=short, 0=flat."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(n_consec - 1, n - 1):
        if np.isnan(sma[i]): continue
        # Check last n_consec bars all same direction
        all_bull = True; all_bear = True
        for j in range(i - n_consec + 1, i + 1):
            if close[j] <= open_[j]: all_bull = False
            if close[j] >= open_[j]: all_bear = False
        dist = (close[i] - sma[i])   # in pip units
        if all_bull and dist >= dist_mult * sp_gate:
            sig[i] = 1
        elif all_bear and (-dist) >= dist_mult * sp_gate:
            sig[i] = -1
    return sig


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


@njit(parallel=True, cache=True)
def sweep_all(mid, bid, ask, sp, close_p, open_p, sma,
              n_is, sp_gate, oos_days,
              nc_arr, dm_arr, tp_arr):
    """Parallel sweep over all (n_consec, dist_mult, tp) combos."""
    n_cfg = len(nc_arr)
    out = np.zeros((n_cfg, 8), dtype=np.float64)  # wf1,wf2,wf3,oos_pd,oos_wr,tpd,n_is3,oos_n
    fold = n_is // 3

    for ci in prange(n_cfg):
        nc  = nc_arr[ci]; dm = dm_arr[ci]; tp = tp_arr[ci]
        sig = build_exhaust_sig(close_p, open_p, sma, nc, dm, sp_gate)

        # IS 3-fold
        n_is3 = 0
        for f in range(3):
            s = f * fold; e = s + fold if f < 2 else n_is
            days = (e - s) / 288.0
            p_sum, n_t, _ = sim_tp(mid[s:e], bid[s:e], ask[s:e], sp[s:e],
                                    sig[s:e], tp, sp_gate)
            if n_t > 0 and days > 0 and p_sum / days > 0.0: n_is3 += 1

        # OOS
        p_sum, n_t, n_w = sim_tp(mid[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
                                   sig[n_is:], tp, sp_gate)
        oos_pd = p_sum / oos_days if oos_days > 0 else 0.0
        oos_wr = 100.0 * n_w / n_t if n_t > 0 else 0.0

        out[ci, 0] = n_is3
        out[ci, 1] = oos_pd
        out[ci, 2] = oos_wr
        out[ci, 3] = n_t / oos_days if oos_days > 0 else 0.0  # tpd
        out[ci, 4] = n_t

    return out


# ── Build parameter arrays for prange ─────────────────────────────────────────
nc_arr = []; dm_arr = []; tp_arr = []
for nc in N_CONSEC_OPTS:
    for dm in DIST_MULT_OPTS:
        for tp in TP_OPTS:
            nc_arr.append(nc); dm_arr.append(dm); tp_arr.append(tp)
nc_arr = np.array(nc_arr, dtype=np.int32)
dm_arr = np.array(dm_arr, dtype=np.float64)
tp_arr = np.array(tp_arr, dtype=np.float64)
N_CFG  = len(nc_arr)

print(f"Exhaustion continuation sweep: {N_CFG} configs × {len(PAIRS)} pairs …")

# ── Warm up Numba ─────────────────────────────────────────────────────────────
_dummy = np.zeros(10, dtype=np.float64)
_ds    = np.zeros(10, dtype=np.int8)
sim_tp(_dummy, _dummy, _dummy, _dummy, _ds, 5.0, 2.0)
compute_sma(_dummy, 3)
build_exhaust_sig(_dummy, _dummy, _dummy, 2, 1.0, 2.0)

rows = []
for pair in PAIRS:
    print(f"  {pair} …", flush=True)
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    oos_days = len(df.iloc[n_is:]) / 288.0

    close_p = (df["close"].values / pip).astype(np.float64)
    open_p  = (df["open"].values  / pip).astype(np.float64) if "open" in df.columns \
              else close_p.copy()
    mid     = close_p.copy()
    bid     = (df["bid_c"].values / pip).astype(np.float64)
    ask     = (df["ask_c"].values / pip).astype(np.float64)
    sp      = (ask - bid)
    sma     = compute_sma(close_p, SMA_N)

    out = sweep_all(mid, bid, ask, sp, close_p, open_p, sma,
                    n_is, sp_gate, oos_days, nc_arr, dm_arr, tp_arr)

    for ci in range(N_CFG):
        rows.append(dict(
            signal="exhaust_cont", pair=pair,
            n_consec=int(nc_arr[ci]), dist_mult=round(float(dm_arr[ci]),2),
            tp=float(tp_arr[ci]),
            n_is3=int(out[ci, 0]), wf_pass=int(out[ci, 0] == 3),
            oos_pd=round(float(out[ci, 1]), 3),
            oos_wr=round(float(out[ci, 2]), 1),
            tpd=round(float(out[ci, 3]), 3),
            n_oos=int(out[ci, 4]),
        ))

    import gc; gc.collect()

df_out = pd.DataFrame(rows)
df_out.to_csv(RESULTS / "exhaust_cont.csv", index=False)
print(f"\nSaved {len(df_out)} rows → results/exhaust_cont.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
survivors = df_out[df_out["wf_pass"] == 1]
print(f"\nWF-passing configs (individual pairs): {len(survivors)}")
# Portfolio view: configs where all 12 pairs pass
port = df_out.groupby(["n_consec","dist_mult","tp"]).agg(
    n_pairs_wf=("wf_pass","sum"),
    portfolio_pd=("oos_pd","sum"),
).reset_index()
port12 = port[port["n_pairs_wf"] == 12].sort_values("portfolio_pd", ascending=False)
print(f"Configs with all 12 pairs WF-passing: {len(port12)}")
print(port12.head(20).to_string(index=False))
