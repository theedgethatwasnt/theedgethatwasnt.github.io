#!/usr/bin/env python3
"""
Stop-Loss Sweep — both live momentum strategies
=================================================
Tests SL levels: None, 10, 15, 20, 25, 30, 40, 50, 75, 100p
Against the exact deployed configs (OOS period only).

Strategy A — fx-price-mom-live (acct 011):
  M15+M5, lags=(1,3,8), TP=10p, 12 pairs, 25u

Strategy B — fx-sma-live (acct 012):
  SMA16, lags=(8,10,15), TP=20p, 10 pairs, 25u

Reports per SL level:
  p/d, WR, n_trades, SL_hit%, avg_SL_loss, max_DD, Calmar
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

# Deployed sp_gates (IS P90, from validation)
SP_GATES_PM = {
    "GBP_JPY":4.00,"CAD_JPY":2.60,"EUR_JPY":2.50,"AUD_JPY":2.30,
    "USD_JPY":2.10,"NZD_JPY":3.10,"CHF_JPY":3.70,"NZD_USD":2.00,
    "EUR_USD":1.70,"AUD_USD":1.60,"GBP_USD":2.40,"EUR_GBP":2.00,
}
SP_GATES_SMA = {  # SMA live 10-pair subset
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,
}

SL_LEVELS = [None, 10, 15, 20, 25, 30, 40, 50, 75, 100]  # pips; None = no SL

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


# ── Signal builders ────────────────────────────────────────────────────────────

def build_pm_signal(df, lags=(1,3,8), tf1="15min", tf2="5min"):
    """Price momentum: raw close diff on two TFs. 6/6 strict."""
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
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
    """SMA-smoothed momentum on two TFs. 6/6 strict."""
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        sm = rs.rolling(sma_n).mean()
        sm_s = sm.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(sm_s - sm_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n] = np.int8(1)
    sig[score == 0] = np.int8(-1)
    return sig


# ── Numba simulator ────────────────────────────────────────────────────────────

@njit
def simulate_tp_sl(bid, ask, mid, sp, sig, pip, tp_pips, sl_pips, sp_gate):
    """
    Returns parallel arrays: pnl_pips[], hold_bars[], exit_type[]
      exit_type: 0=TP  1=SL  (no SL => only 0s)
    sl_pips < 0 means no SL (use -1.0 as sentinel).
    """
    n = len(mid)
    pnl_out   = np.empty(n, dtype=np.float64)
    hold_out  = np.empty(n, dtype=np.int32)
    type_out  = np.empty(n, dtype=np.int8)
    count = 0

    in_trade = False; dir_ = 0; ep = 0.0; ei = 0
    use_sl = sl_pips > 0.0

    for i in range(1, n):
        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False
            elif use_sl and excur <= -sl_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 1
                count += 1; in_trade = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_drawdown_pips(pnl_arr):
    eq = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return dd.max() if len(dd) else 0.0


def run_strategy(strategy_name, pairs, sp_gates, signal_fn, tp_pips):
    """Load OOS data, build signal, run all SL levels, return summary dict."""
    print(f"\n{'='*62}")
    print(f"Strategy: {strategy_name}  TP={tp_pips}p")
    print(f"{'='*62}")

    # Pre-load all pair data (OOS slice only)
    pair_data = {}
    for pair in pairs:
        df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
              .set_index("timestamp").sort_index())
        df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
        pip  = pip_sz(pair)
        n_is = int(len(df) * IS_FRAC)
        sg   = sp_gates[pair]

        sig_full = signal_fn(df)
        oos_df   = df.iloc[n_is:]
        oos_sig  = sig_full.iloc[n_is:]

        bid  = oos_df["bid_c"].values
        ask  = oos_df["ask_c"].values
        mid  = oos_df["close"].values
        sp   = ((ask - bid) / pip).astype(np.float64)
        s    = oos_sig.values.astype(np.float64)
        oos_days = len(oos_df) / 288

        pair_data[pair] = dict(bid=bid, ask=ask, mid=mid, sp=sp,
                               sig=s, pip=pip, sp_gate=sg,
                               oos_days=oos_days)
        print(f"  Loaded {pair}: {len(oos_df)} OOS bars ({oos_days:.0f} trading days)")

    avg_oos_days = np.mean([v["oos_days"] for v in pair_data.values()])

    # ── Sweep SL levels ────────────────────────────────────────────────────────
    rows = []
    for sl in SL_LEVELS:
        sl_arg = float(sl) if sl is not None else -1.0
        label  = f"{sl}p" if sl is not None else "None"

        all_pnl = []; all_types = []
        for pair, d in pair_data.items():
            pnl, holds, types = simulate_tp_sl(
                d["bid"], d["ask"], d["mid"], d["sp"], d["sig"],
                d["pip"], float(tp_pips), sl_arg, d["sp_gate"]
            )
            all_pnl.extend(pnl.tolist())
            all_types.extend(types.tolist())

        if not all_pnl:
            continue

        pnl_arr   = np.array(all_pnl)
        type_arr  = np.array(all_types)
        n         = len(pnl_arr)
        n_tp      = (type_arr == 0).sum()
        n_sl      = (type_arr == 1).sum()
        wr        = n_tp / n * 100
        sl_pct    = n_sl / n * 100
        total_p   = pnl_arr.sum()
        ppd       = total_p / avg_oos_days
        avg_sl_loss = pnl_arr[type_arr == 1].mean() if n_sl > 0 else 0.0
        mdd       = max_drawdown_pips(pnl_arr)
        calmar    = ppd / mdd if mdd > 0 else 0.0

        rows.append(dict(
            SL=label, n=n, WR=wr, SL_pct=sl_pct,
            total_p=total_p, ppd=ppd,
            avg_SL_loss=avg_sl_loss, mdd=mdd, calmar=calmar
        ))

    # ── Print table ────────────────────────────────────────────────────────────
    base = next((r for r in rows if r["SL"] == "None"), rows[0])
    print(f"\n{'SL':>6}  {'n':>5}  {'WR%':>5}  {'SL%':>5}  "
          f"{'p/d':>7}  {'vs base':>7}  {'avg_SL':>7}  {'MaxDD':>6}  {'Calmar':>6}")
    print("-" * 72)
    for r in rows:
        delta = r["ppd"] - base["ppd"]
        print(f"{r['SL']:>6}  {r['n']:>5}  {r['WR']:>5.1f}  {r['SL_pct']:>5.1f}  "
              f"{r['ppd']:>7.1f}  {delta:>+7.1f}  {r['avg_SL_loss']:>7.1f}  "
              f"{r['mdd']:>6.1f}  {r['calmar']:>6.2f}")
    return rows


# ── Run both strategies ───────────────────────────────────────────────────────

# Warm up Numba
_d = list(list({p:SP_GATES_PM for p in ALL_PAIRS[:1]}.items()))
_df = pd.read_parquet(DATA / f"{ALL_PAIRS[0]}_M5_BA.parquet").set_index("timestamp")
_df = _df.astype({c:"float64" for c in _df.select_dtypes("float32").columns})
_pip = pip_sz(ALL_PAIRS[0])
_bid = _df["bid_c"].values[:500]; _ask = _df["ask_c"].values[:500]
_mid = _df["close"].values[:500]; _sp = ((_ask-_bid)/_pip)
_sig = np.ones(500, dtype=np.float64)
simulate_tp_sl(_bid,_ask,_mid,_sp,_sig,_pip,10.0,30.0,4.0)
simulate_tp_sl(_bid,_ask,_mid,_sp,_sig,_pip,10.0,-1.0,4.0)
print("Numba warmed up.")

results_pm = run_strategy(
    "fx-price-mom-live  (M15+M5 lags=(1,3,8) TP=10p)",
    ALL_PAIRS, SP_GATES_PM,
    lambda df: build_pm_signal(df, lags=(1,3,8), tf1="15min", tf2="5min"),
    tp_pips=10.0
)

SMA_PAIRS = [p for p in ALL_PAIRS if p in SP_GATES_SMA]
results_sma = run_strategy(
    "fx-sma-live  (SMA16 lags=(8,10,15) TP=20p)",
    SMA_PAIRS, SP_GATES_SMA,
    lambda df: build_sma_signal(df, sma_n=16, lags=(8,10,15), tf1="1h", tf2="30min"),
    tp_pips=20.0
)

# ── Recommendation ────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("RECOMMENDATION")
print(f"{'='*62}")
for name, rows, tp in [
    ("pm-live  (TP=10p)", results_pm, 10),
    ("sma-live (TP=20p)", results_sma, 20),
]:
    base_ppd = next(r["ppd"] for r in rows if r["SL"]=="None")
    # Best SL = highest Calmar that loses < 5% p/d vs baseline
    candidates = [r for r in rows if r["SL"] != "None"
                  and r["ppd"] >= base_ppd * 0.95]
    if candidates:
        best = max(candidates, key=lambda r: r["calmar"])
        print(f"\n  {name}:")
        print(f"    Baseline (no SL): {base_ppd:.1f} p/d")
        print(f"    Best SL={best['SL']}: {best['ppd']:.1f} p/d  "
              f"({best['ppd']-base_ppd:+.1f} vs base)  "
              f"WR={best['WR']:.1f}%  SL_hit={best['SL_pct']:.1f}%  "
              f"MaxDD={best['mdd']:.1f}p  Calmar={best['calmar']:.2f}")
    else:
        print(f"\n  {name}: no SL level retains ≥95% p/d — keep None")
