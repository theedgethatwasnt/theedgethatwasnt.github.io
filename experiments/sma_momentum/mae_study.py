#!/usr/bin/env python3
"""
MAE Study — How badly do winning trades drawdown before reaching TP?
====================================================================
For each OOS trade on the deployed config (M15+M5 lags=(1,3,8) TP=10p),
tracks bar-by-bar excursion and records:
  MAE  = maximum adverse excursion (worst drawdown before TP)
  MFE  = maximum favourable excursion at TP hit bar

Reports the MAE distribution so we can answer:
  "Do trades immediately go our way, or do they sit in the red first?"

Also reports by signal episode position (entry on bar 1 vs bar 5 vs bar 10+
of a continuous signal run) to see if early-episode entries have smaller MAE.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70

SP_GATES = {
    "GBP_JPY":4.00,"CAD_JPY":2.60,"EUR_JPY":2.50,"AUD_JPY":2.30,
    "USD_JPY":2.10,"NZD_JPY":3.10,"CHF_JPY":3.70,"NZD_USD":2.00,
    "EUR_USD":1.70,"AUD_USD":1.60,"GBP_USD":2.40,"EUR_GBP":2.00,
}

def pip_sz(p): return 0.01 if p in JPY else 0.0001

def build_signal(df, lags=(1,3,8), tf1="15min", tf2="5min"):
    moms = []
    for tf in [tf1, tf2]:
        rs   = df["close"].resample(tf).last().dropna()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n] = np.int8(1)
    sig[score == 0] = np.int8(-1)
    return sig

@njit
def simulate_with_mae(bid, ask, mid, sp, sig, pip, tp_pips, sp_gate, ep_pos):
    """
    Returns per-trade arrays:
      mae_pips  — worst adverse excursion (always >= 0)
      mfe_pips  — best favourable excursion at moment of TP
      hold_bars — bars held
      ep_bar    — position within signal episode (1=first bar, 2=second, etc.)
    ep_pos[i] = how many consecutive bars the signal has been active through bar i
    """
    n = len(mid)
    mae_out  = np.empty(n, dtype=np.float64)
    mfe_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    epb_out  = np.empty(n, dtype=np.int32)
    count = 0

    in_trade = False; dir_ = 0; ep = 0.0; ei = 0
    mae = 0.0; mfe_peak = 0.0

    for i in range(1, n):
        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur < -mae: mae = -excur      # track worst adverse
            if excur > mfe_peak: mfe_peak = excur
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                mae_out[count]  = mae
                mfe_out[count]  = mfe_peak
                hold_out[count] = i - ei
                epb_out[count]  = ep_pos[ei]
                count += 1; in_trade = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True
                mae = 0.0; mfe_peak = 0.0

    return mae_out[:count], mfe_out[:count], hold_out[:count], epb_out[:count]


print("Loading OOS data and computing MAE…\n")
all_mae=[]; all_mfe=[]; all_hold=[]; all_epb=[]

for pair in PAIRS:
    df = (pd.read_parquet(DATA/f"{pair}_M5_BA.parquet")
          .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip  = pip_sz(pair)
    n_is = int(len(df)*IS_FRAC)
    sg   = SP_GATES[pair]

    sig_full = build_signal(df)
    oos_df   = df.iloc[n_is:]
    oos_sig  = sig_full.iloc[n_is:]

    # compute episode position (how many bars into current signal run)
    s_arr = oos_sig.values
    ep_pos = np.zeros(len(s_arr), dtype=np.int32)
    run = 0
    for i in range(len(s_arr)):
        if s_arr[i] != 0:
            run += 1
        else:
            run = 0
        ep_pos[i] = run

    bid = oos_df["bid_c"].values.astype(np.float64)
    ask = oos_df["ask_c"].values.astype(np.float64)
    mid = oos_df["close"].values.astype(np.float64)
    sp  = ((ask-bid)/pip).astype(np.float64)
    s   = s_arr.astype(np.float64)

    mae, mfe, hold, epb = simulate_with_mae(
        bid, ask, mid, sp, s, pip, 10.0, sg, ep_pos)
    all_mae.extend(mae.tolist())
    all_mfe.extend(mfe.tolist())
    all_hold.extend((hold * 5 / 60).tolist())  # bars → hours
    all_epb.extend(epb.tolist())
    print(f"  {pair}: {len(mae)} trades  "
          f"MAE P50={np.percentile(mae,50):.1f}p  "
          f"MAE P90={np.percentile(mae,90):.1f}p  "
          f"MAE P99={np.percentile(mae,99):.1f}p")

mae_arr  = np.array(all_mae)
mfe_arr  = np.array(all_mfe)
hold_arr = np.array(all_hold)
epb_arr  = np.array(all_epb)
n_total  = len(mae_arr)

print(f"\n{'='*62}")
print(f"MAE DISTRIBUTION — {n_total} OOS winning trades (TP=10p)")
print(f"{'='*62}")
print(f"\n  Trades with MAE = 0p (never in the red) : "
      f"{(mae_arr==0).sum()} ({(mae_arr==0).mean()*100:.1f}%)")
print(f"  Trades with MAE < 2p                   : "
      f"{(mae_arr<2).sum()} ({(mae_arr<2).mean()*100:.1f}%)")
print(f"  Trades with MAE < 5p                   : "
      f"{(mae_arr<5).sum()} ({(mae_arr<5).mean()*100:.1f}%)")
print(f"  Trades with MAE < 10p                  : "
      f"{(mae_arr<10).sum()} ({(mae_arr<10).mean()*100:.1f}%)")
print(f"  Trades with MAE < 20p                  : "
      f"{(mae_arr<20).sum()} ({(mae_arr<20).mean()*100:.1f}%)")
print(f"  Trades with MAE >= 20p                 : "
      f"{(mae_arr>=20).sum()} ({(mae_arr>=20).mean()*100:.1f}%)")

print(f"\n  Percentiles:")
for p in [10,25,50,75,90,95,99,100]:
    print(f"    P{p:3d} MAE = {np.percentile(mae_arr,p):6.1f}p  "
          f"hold = {np.percentile(hold_arr,p):5.1f}h")

print(f"\n{'='*62}")
print(f"MAE BY EPISODE POSITION (when during signal run we entered)")
print(f"{'='*62}")
print(f"  {'Ep.bar':>8}  {'n':>5}  {'MAE_P50':>7}  {'MAE_P90':>7}  "
      f"{'MAE_P99':>7}  {'MAE_mean':>8}  {'hold_P50':>8}")
for lo, hi, lbl in [(1,1,"1"),(2,2,"2"),(3,3,"3"),(4,5,"4-5"),(6,10,"6-10"),(11,999,"11+")]:
    mask = (epb_arr >= lo) & (epb_arr <= hi)
    if mask.sum() < 5:
        continue
    m = mae_arr[mask]; h = hold_arr[mask]
    print(f"  {lbl:>8}  {mask.sum():>5}  "
          f"{np.percentile(m,50):>7.1f}  {np.percentile(m,90):>7.1f}  "
          f"{np.percentile(m,99):>7.1f}  {m.mean():>8.1f}  "
          f"{np.percentile(h,50):>8.1f}h")

print(f"\n{'='*62}")
print(f"KEY QUESTION: If we only entered on bar 1 of signal episode,")
print(f"how much MAE protection do we get vs all-bar entry?")
print(f"{'='*62}")
bar1 = mae_arr[epb_arr == 1]
all_  = mae_arr
print(f"  All entries  : n={len(all_):4d}  MAE_P50={np.percentile(all_,50):.1f}p  "
      f"MAE_P90={np.percentile(all_,90):.1f}p  mean={all_.mean():.1f}p")
print(f"  Bar-1 only   : n={len(bar1):4d}  MAE_P50={np.percentile(bar1,50):.1f}p  "
      f"MAE_P90={np.percentile(bar1,90):.1f}p  mean={bar1.mean():.1f}p")
print(f"  Reduction    :       "
      f"P50 {np.percentile(all_,50)-np.percentile(bar1,50):+.1f}p  "
      f"P90 {np.percentile(all_,90)-np.percentile(bar1,90):+.1f}p  "
      f"mean {all_.mean()-bar1.mean():+.1f}p")
