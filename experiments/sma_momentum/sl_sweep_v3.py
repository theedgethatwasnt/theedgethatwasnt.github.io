#!/usr/bin/env python3
"""
Stop-Loss Sweep v3 — Trailing Stop + First Formation Only
==========================================================
Tests three protection approaches on OOS data:

  A) Fixed SL   — exits when price falls SL pips below ENTRY
  B) Trailing SL — exits when price falls TRAIL pips below HIGH WATERMARK
     (trail starts from entry bar 0, no activation threshold needed)

Entry rule: first signal formation only + wait-for-zero before re-entry.
After ANY exit the strategy waits for signal=0 then a brand-new formation.

Both deployed strategies:
  011 — M15+M5 lags=(1,3,8) TP=10p
  012 — SMA16 lags=(8,10,15) TP=20p
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

FIXED_SL_LEVELS   = [None, 20, 30, 50, 75, 100, 150]
TRAIL_DIST_LEVELS = [15, 20, 25, 30, 40, 50, 75, 100]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


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


@njit
def simulate_fixed_sl(bid, ask, mid, sp, sig, pip,
                       tp_pips, sl_pips, sp_gate):
    """First-formation entry + fixed SL. sl_pips<=0 → no SL."""
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0; prev_sig = 0.0
    use_sl = sl_pips > 0.0

    for i in range(1, n):
        cur_sig = sig[i - 1]
        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True
            elif use_sl and excur <= -sl_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 1
                count += 1; in_trade = False; wait_zero = True
        else:
            if wait_zero:
                if cur_sig == 0: wait_zero = False
            else:
                if prev_sig == 0 and cur_sig != 0 and sp[i] <= sp_gate:
                    ep = ask[i] if cur_sig == 1 else bid[i]
                    dir_ = int(cur_sig); ei = i; in_trade = True
        prev_sig = cur_sig

    return pnl_out[:count], hold_out[:count], type_out[:count]


@njit
def simulate_trailing_sl(bid, ask, mid, sp, sig, pip,
                          tp_pips, trail_pips, sp_gate):
    """First-formation entry + trailing SL from high-water mark."""
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0; prev_sig = 0.0; hwm = 0.0

    for i in range(1, n):
        cur_sig = sig[i - 1]
        if in_trade:
            # update high-water mark
            excur = (mid[i] - ep) / pip * dir_
            if excur > hwm: hwm = excur
            trail_sl = hwm - trail_pips   # pips below HWM

            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True
            elif excur <= trail_sl:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 1
                count += 1; in_trade = False; wait_zero = True
        else:
            if wait_zero:
                if cur_sig == 0: wait_zero = False
            else:
                if prev_sig == 0 and cur_sig != 0 and sp[i] <= sp_gate:
                    ep = ask[i] if cur_sig == 1 else bid[i]
                    dir_ = int(cur_sig); ei = i; in_trade = True; hwm = 0.0
        prev_sig = cur_sig

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl_arr):
    if len(pnl_arr) == 0: return 0.0
    eq = np.cumsum(pnl_arr)
    return (np.maximum.accumulate(eq) - eq).max()


def run_strategy(name, pairs, sp_gates, signal_fn, tp_pips):
    print(f"\n{'='*66}")
    print(f"  {name}  TP={tp_pips}p")
    print(f"{'='*66}")

    pair_data = {}
    for pair in pairs:
        df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
              .set_index("timestamp").sort_index())
        df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
        pip   = pip_sz(pair)
        n_is  = int(len(df) * IS_FRAC)
        sg    = sp_gates[pair]
        sig   = signal_fn(df)
        oos   = df.iloc[n_is:]
        osig  = sig.iloc[n_is:]
        bid   = oos["bid_c"].values.astype(np.float64)
        ask   = oos["ask_c"].values.astype(np.float64)
        mid   = oos["close"].values.astype(np.float64)
        sp    = ((ask - bid) / pip).astype(np.float64)
        s     = osig.values.astype(np.float64)
        pair_data[pair] = dict(bid=bid, ask=ask, mid=mid, sp=sp,
                               sig=s, pip=pip, sp_gate=sg,
                               days=len(oos)/288)
    avg_days = np.mean([v["days"] for v in pair_data.values()])

    # ── Fixed SL ────────────────────────────────────────────────────────────────
    print(f"\n  ── Fixed SL (first-formation entry) ──")
    print(f"  {'SL':>6}  {'n':>5}  {'WR%':>5}  {'SL%':>5}  "
          f"{'p/d':>7}  {'vs_base':>7}  {'avg_SL':>7}  {'MaxDD':>7}  {'Calmar':>6}")
    print(f"  {'-'*68}")

    base_ppd = None
    fixed_rows = []
    for sl in FIXED_SL_LEVELS:
        sl_arg = float(sl) if sl is not None else -1.0
        label  = f"{sl}p" if sl is not None else "None"
        all_p=[]; all_t=[]
        for d in pair_data.values():
            p, h, t = simulate_fixed_sl(
                d["bid"],d["ask"],d["mid"],d["sp"],d["sig"],
                d["pip"],float(tp_pips),sl_arg,d["sp_gate"])
            all_p.extend(p.tolist()); all_t.extend(t.tolist())
        if not all_p: continue
        pnl=np.array(all_p); typ=np.array(all_t)
        n=len(pnl); ntp=(typ==0).sum(); nsl=(typ==1).sum()
        wr=ntp/n*100; sl_pct=nsl/n*100
        ppd=pnl.sum()/avg_days
        if base_ppd is None: base_ppd=ppd
        avg_sl=pnl[typ==1].mean() if nsl>0 else 0.0
        mdd=max_dd(pnl); cal=ppd/mdd if mdd>0 else 0.0
        delta=ppd-base_ppd
        print(f"  {label:>6}  {n:>5}  {wr:>5.1f}  {sl_pct:>5.1f}  "
              f"{ppd:>7.1f}  {delta:>+7.1f}  {avg_sl:>7.1f}  {mdd:>7.1f}  {cal:>6.2f}")
        fixed_rows.append(dict(label=label,n=n,wr=wr,sl_pct=sl_pct,
                               ppd=ppd,avg_sl=avg_sl,mdd=mdd,cal=cal))

    # ── Trailing SL ─────────────────────────────────────────────────────────────
    print(f"\n  ── Trailing SL from HWM (first-formation entry) ──")
    print(f"  {'Trail':>6}  {'n':>5}  {'WR%':>5}  {'SL%':>5}  "
          f"{'p/d':>7}  {'vs_base':>7}  {'avg_SL':>7}  {'MaxDD':>7}  {'Calmar':>6}")
    print(f"  {'-'*68}")

    base_ppd2 = fixed_rows[0]["ppd"] if fixed_rows else None
    trail_rows = []
    for trail in TRAIL_DIST_LEVELS:
        all_p=[]; all_t=[]
        for d in pair_data.values():
            p, h, t = simulate_trailing_sl(
                d["bid"],d["ask"],d["mid"],d["sp"],d["sig"],
                d["pip"],float(tp_pips),float(trail),d["sp_gate"])
            all_p.extend(p.tolist()); all_t.extend(t.tolist())
        if not all_p: continue
        pnl=np.array(all_p); typ=np.array(all_t)
        n=len(pnl); ntp=(typ==0).sum(); nsl=(typ==1).sum()
        wr=ntp/n*100; sl_pct=nsl/n*100
        ppd=pnl.sum()/avg_days
        avg_sl=pnl[typ==1].mean() if nsl>0 else 0.0
        mdd=max_dd(pnl); cal=ppd/mdd if mdd>0 else 0.0
        delta=ppd-(base_ppd2 or ppd)
        print(f"  {trail:>5}p  {n:>5}  {wr:>5.1f}  {sl_pct:>5.1f}  "
              f"{ppd:>7.1f}  {delta:>+7.1f}  {avg_sl:>7.1f}  {mdd:>7.1f}  {cal:>6.2f}")
        trail_rows.append(dict(trail=trail,n=n,wr=wr,sl_pct=sl_pct,
                               ppd=ppd,avg_sl=avg_sl,mdd=mdd,cal=cal))
    return fixed_rows, trail_rows, base_ppd2


# ── Warmup ─────────────────────────────────────────────────────────────────────
print("Warming up Numba...")
_df = pd.read_parquet(DATA/f"{ALL_PAIRS[0]}_M5_BA.parquet").set_index("timestamp")
_df = _df.astype({c:"float64" for c in _df.select_dtypes("float32").columns})
_pip=pip_sz(ALL_PAIRS[0])
_b=_df["bid_c"].values[:500].astype(np.float64)
_a=_df["ask_c"].values[:500].astype(np.float64)
_m=_df["close"].values[:500].astype(np.float64)
_s=((_a-_b)/_pip).astype(np.float64)
_sig=np.zeros(500,dtype=np.float64); _sig[10:20]=1.0; _sig[30:40]=-1.0
simulate_fixed_sl(_b,_a,_m,_s,_sig,_pip,10.0,30.0,4.0)
simulate_fixed_sl(_b,_a,_m,_s,_sig,_pip,10.0,-1.0,4.0)
simulate_trailing_sl(_b,_a,_m,_s,_sig,_pip,10.0,20.0,4.0)
print("Done.")

fr_pm, tr_pm, base_pm = run_strategy(
    "fx-price-mom-live  M15+M5 lags=(1,3,8)",
    ALL_PAIRS, SP_GATES_PM,
    lambda df: build_pm_signal(df,(1,3,8),"15min","5min"),
    tp_pips=10.0
)

SMA_PAIRS = [p for p in ALL_PAIRS if p in SP_GATES_SMA]
fr_sma, tr_sma, base_sma = run_strategy(
    "fx-sma-live  SMA16 lags=(8,10,15)",
    SMA_PAIRS, SP_GATES_SMA,
    lambda df: build_sma_signal(df,16,(8,10,15),"1h","30min"),
    tp_pips=20.0
)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("SUMMARY — Best protection per strategy")
print(f"{'='*66}")
for sname, base, fixed_rows, trail_rows in [
    ("pm-live  TP=10p", base_pm,  fr_pm,  tr_pm),
    ("sma-live TP=20p", base_sma, fr_sma, tr_sma),
]:
    print(f"\n  {sname}  (no-SL baseline: {base:.1f} p/d)")
    print(f"  {'Type':12s}  {'Param':>6}  {'p/d':>7}  {'retain%':>7}  "
          f"{'WR%':>5}  {'SL%':>4}  {'MaxDD':>7}  {'Calmar':>6}")
    best_fixed = max((r for r in fixed_rows if r["label"]!="None"),
                     key=lambda r: r["ppd"], default=None)
    best_trail = max(trail_rows, key=lambda r: r["ppd"], default=None)
    if best_fixed:
        pct = best_fixed["ppd"]/base*100
        print(f"  {'Fixed SL':12s}  {best_fixed['label']:>6}  "
              f"{best_fixed['ppd']:>7.1f}  {pct:>7.1f}%  "
              f"{best_fixed['wr']:>5.1f}  {best_fixed['sl_pct']:>4.1f}  "
              f"{best_fixed['mdd']:>7.1f}  {best_fixed['cal']:>6.2f}")
    if best_trail:
        pct = best_trail["ppd"]/base*100
        print(f"  {'Trailing SL':12s}  {best_trail['trail']:>5}p  "
              f"{best_trail['ppd']:>7.1f}  {pct:>7.1f}%  "
              f"{best_trail['wr']:>5.1f}  {best_trail['sl_pct']:>4.1f}  "
              f"{best_trail['mdd']:>7.1f}  {best_trail['cal']:>6.2f}")
