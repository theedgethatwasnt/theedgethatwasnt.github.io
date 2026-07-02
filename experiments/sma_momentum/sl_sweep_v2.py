#!/usr/bin/env python3
"""
Stop-Loss Sweep v2 — "First Formation Only" entry rule
=======================================================
Key change vs v1: after ANY exit (TP or SL), the strategy sits flat
until the signal resets to ZERO, then waits for a brand new signal
formation before entering again.

This prevents whipsaw re-entry into the same momentum episode after SL.

Both strategies tested:
  A — fx-price-mom-live: M15+M5 lags=(1,3,8) TP=10p
  B — fx-sma-live:       SMA16 lags=(8,10,15) TP=20p

SL levels swept: None, 10, 15, 20, 25, 30, 40, 50, 75, 100p
"""
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY       = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC   = 0.70

SP_GATES_PM = {
    "GBP_JPY":4.00,"CAD_JPY":2.60,"EUR_JPY":2.50,"AUD_JPY":2.30,
    "USD_JPY":2.10,"NZD_JPY":3.10,"CHF_JPY":3.70,"NZD_USD":2.00,
    "EUR_USD":1.70,"AUD_USD":1.60,"GBP_USD":2.40,"EUR_GBP":2.00,
}
SP_GATES_SMA = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,
}

SL_LEVELS = [None, 10, 15, 20, 25, 30, 40, 50, 75, 100]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


# ── Signal builders (identical to deployed code) ───────────────────────────────

def build_pm_signal(df, lags=(1,3,8), tf1="15min", tf2="5min"):
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


def build_sma_signal(df, sma_n=16, lags=(8,10,15), tf1="1h", tf2="30min"):
    moms = []
    for tf in [tf1, tf2]:
        rs   = df["close"].resample(tf).last().dropna()
        sm   = rs.rolling(sma_n).mean()
        sm_s = sm.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(sm_s - sm_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n] = np.int8(1)
    sig[score == 0] = np.int8(-1)
    return sig


# ── Numba simulator — first-formation-only ─────────────────────────────────────

@njit
def simulate_first_formation(bid, ask, mid, sp, sig, pip,
                              tp_pips, sl_pips, sp_gate):
    """
    Entry rule: only on rising/falling edge of signal (0→±1).
    After any exit: wait for signal=0, then enter on next non-zero bar.

    sl_pips <= 0 → no SL.
    exit_type: 0=TP  1=SL
    """
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count    = 0

    in_trade     = False
    wait_for_zero = False   # True after exit: must see sig=0 before re-entry
    dir_ = 0; ep = 0.0; ei = 0
    use_sl = sl_pips > 0.0
    prev_sig = 0.0

    for i in range(1, n):
        cur_sig = sig[i - 1]   # signal on bar i-1 (closed), determines entry at i

        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1
                in_trade = False; wait_for_zero = True
            elif use_sl and excur <= -sl_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 1
                count += 1
                in_trade = False; wait_for_zero = True
        else:
            if wait_for_zero:
                # waiting for signal to reset to 0
                if cur_sig == 0:
                    wait_for_zero = False
            else:
                # only enter on fresh edge: previous bar was 0, this bar is ±1
                is_fresh = (prev_sig == 0 and cur_sig != 0)
                if is_fresh and sp[i] <= sp_gate:
                    ep = ask[i] if cur_sig == 1 else bid[i]
                    dir_ = int(cur_sig); ei = i; in_trade = True

        prev_sig = cur_sig

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_drawdown_pips(pnl_arr):
    if len(pnl_arr) == 0: return 0.0
    eq   = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(eq)
    return (peak - eq).max()


def run_strategy(name, pairs, sp_gates, signal_fn, tp_pips):
    print(f"\n{'='*62}")
    print(f"Strategy: {name}  TP={tp_pips}p")
    print(f"  Entry rule: first signal formation only (edge trigger)")
    print(f"  After exit: wait for signal=0 before re-entry")
    print(f"{'='*62}")

    pair_data = {}
    for pair in pairs:
        df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
              .set_index("timestamp").sort_index())
        df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
        pip   = pip_sz(pair)
        n_is  = int(len(df) * IS_FRAC)
        sg    = sp_gates[pair]
        sig   = signal_fn(df)
        oos_df  = df.iloc[n_is:]
        oos_sig = sig.iloc[n_is:]
        bid  = oos_df["bid_c"].values.astype(np.float64)
        ask  = oos_df["ask_c"].values.astype(np.float64)
        mid  = oos_df["close"].values.astype(np.float64)
        sp   = ((ask - bid) / pip).astype(np.float64)
        s    = oos_sig.values.astype(np.float64)
        pair_data[pair] = dict(bid=bid, ask=ask, mid=mid, sp=sp,
                               sig=s, pip=pip, sp_gate=sg,
                               oos_days=len(oos_df)/288)
        print(f"  {pair}: {len(oos_df)} OOS bars")

    avg_days = np.mean([v["oos_days"] for v in pair_data.values()])

    rows = []
    for sl in SL_LEVELS:
        sl_arg = float(sl) if sl is not None else -1.0
        label  = f"{sl}p" if sl is not None else "None"

        all_pnl = []; all_holds = []; all_types = []
        for pair, d in pair_data.items():
            pnl, holds, types = simulate_first_formation(
                d["bid"], d["ask"], d["mid"], d["sp"], d["sig"],
                d["pip"], float(tp_pips), sl_arg, d["sp_gate"]
            )
            all_pnl.extend(pnl.tolist())
            all_holds.extend(holds.tolist())
            all_types.extend(types.tolist())

        if not all_pnl:
            continue

        pnl_arr   = np.array(all_pnl)
        hold_arr  = np.array(all_holds)
        type_arr  = np.array(all_types)
        n         = len(pnl_arr)
        n_tp      = (type_arr == 0).sum()
        n_sl      = (type_arr == 1).sum()
        wr        = n_tp / n * 100
        sl_pct    = n_sl / n * 100
        total_p   = pnl_arr.sum()
        ppd       = total_p / avg_days
        avg_hold_h= hold_arr.mean() * 5 / 60        # M5 bars → hours
        avg_sl_loss = pnl_arr[type_arr==1].mean() if n_sl > 0 else 0.0
        mdd       = max_drawdown_pips(pnl_arr)
        calmar    = ppd / mdd if mdd > 0 else 0.0

        rows.append(dict(SL=label, n=n, WR=wr, SL_pct=sl_pct,
                         total_p=total_p, ppd=ppd, avg_hold_h=avg_hold_h,
                         avg_sl_loss=avg_sl_loss, mdd=mdd, calmar=calmar))

    base = next((r for r in rows if r["SL"]=="None"), rows[0])
    print(f"\n{'SL':>6}  {'n':>5}  {'WR%':>5}  {'SL%':>5}  "
          f"{'p/d':>7}  {'vs base':>7}  {'hold_h':>6}  "
          f"{'avg_SL':>7}  {'MaxDD':>7}  {'Calmar':>6}")
    print("-" * 80)
    for r in rows:
        delta = r["ppd"] - base["ppd"]
        print(f"{r['SL']:>6}  {r['n']:>5}  {r['WR']:>5.1f}  {r['SL_pct']:>5.1f}  "
              f"{r['ppd']:>7.1f}  {delta:>+7.1f}  {r['avg_hold_h']:>6.1f}h  "
              f"{r['avg_sl_loss']:>7.1f}  {r['mdd']:>7.1f}  {r['calmar']:>6.2f}")
    return rows, avg_days


print("Warming up Numba...")
_df = pd.read_parquet(DATA/f"{ALL_PAIRS[0]}_M5_BA.parquet").set_index("timestamp")
_df = _df.astype({c:"float64" for c in _df.select_dtypes("float32").columns})
_pip = pip_sz(ALL_PAIRS[0])
_b = _df["bid_c"].values[:500].astype(np.float64)
_a = _df["ask_c"].values[:500].astype(np.float64)
_m = _df["close"].values[:500].astype(np.float64)
_s = ((_a-_b)/_pip).astype(np.float64)
_sig = np.zeros(500, dtype=np.float64); _sig[10:20]=1; _sig[30:40]=-1
simulate_first_formation(_b,_a,_m,_s,_sig,_pip,10.0,30.0,4.0)
simulate_first_formation(_b,_a,_m,_s,_sig,_pip,10.0,-1.0,4.0)
print("Done.\n")

rows_pm,  days_pm  = run_strategy(
    "fx-price-mom-live  M15+M5 lags=(1,3,8) TP=10p",
    ALL_PAIRS, SP_GATES_PM,
    lambda df: build_pm_signal(df, lags=(1,3,8), tf1="15min", tf2="5min"),
    tp_pips=10.0
)

SMA_PAIRS = [p for p in ALL_PAIRS if p in SP_GATES_SMA]
rows_sma, days_sma = run_strategy(
    "fx-sma-live  SMA16 lags=(8,10,15) TP=20p",
    SMA_PAIRS, SP_GATES_SMA,
    lambda df: build_sma_signal(df, sma_n=16, lags=(8,10,15), tf1="1h", tf2="30min"),
    tp_pips=20.0
)

# ── Recommendation ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("RECOMMENDATION")
print(f"{'='*62}")
for name, rows in [("pm-live  (TP=10p)", rows_pm),
                   ("sma-live (TP=20p)", rows_sma)]:
    base = next(r for r in rows if r["SL"]=="None")
    print(f"\n  {name}:")
    print(f"    Baseline (no SL): {base['ppd']:.1f} p/d  "
          f"n={base['n']}  MaxDD={base['mdd']:.1f}p")
    for threshold in [0.95, 0.90, 0.80]:
        candidates = [r for r in rows
                      if r["SL"] != "None" and r["ppd"] >= base["ppd"] * threshold]
        if candidates:
            best = max(candidates, key=lambda r: r["calmar"])
            print(f"    Best SL retaining ≥{threshold*100:.0f}% p/d → "
                  f"SL={best['SL']}  ppd={best['ppd']:.1f}  "
                  f"WR={best['WR']:.1f}%  SL_hit={best['SL_pct']:.1f}%  "
                  f"MaxDD={best['mdd']:.1f}p  Calmar={best['calmar']:.2f}")
            break
    else:
        print(f"    No SL level retains ≥80% p/d")
