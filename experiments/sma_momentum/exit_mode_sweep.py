#!/usr/bin/env python3
"""
exit_mode_sweep.py — Time-cap and signal-flip exits on the 6 momentum-book
live strategies (001 002 003 004 011 012).

Background
----------
We already know (mae_stop_sweep.csv, sl_sweep_v3.py) that any FIXED-PIP stop
collapses the live momentum book: +27 p/d → −200..−800 p/d, 0/12 pairs pass.
The user asked: are TIME-STOP or MOVING-AVERAGE / SIGNAL-FLIP exits any
better?  This script answers that for each strategy and pair.

Exit modes tested
-----------------
  TP_ONLY     baseline — current live behavior: hold until TP fires (or end).
  TIME_T      close at i+T regardless of P/L  (T in M5 bars)
  SIGNAL_FLIP close when the entry signal has gone to 0 or to opposite_dir.

Reusing the v3 pattern: first-formation entry, wait-for-zero before re-entry,
per-pair IS-P90 spread gate, OOS-only evaluation.

R6 compliance: signal builders mirror live `compute_signal()` byte-for-byte.

Run:
  python3 exit_mode_sweep.py
Output:
  results/exit_mode_sweep.csv   (per strategy × pair × exit mode)
"""
import gc
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY       = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC   = 0.70

# Per-pair IS-P90 spread gates copied from the live services.
SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}

# Time stops in M5 bars (12 bars/hour). 6=30m, 12=1h, 24=2h, 48=4h, 144=12h, 288=24h.
TIME_STOPS_M5 = [6, 12, 24, 48, 144, 288]


def pip_sz(p): return 0.01 if p in JPY else 0.0001


# ── Signal builders — one per live strategy. R6: must match compute_signal(). ──

def build_pmom_h1m30(df):
    """001 — H1 + M30 raw price momentum, lags=(8,10,20)."""
    return _raw_mom(df, lags=(8,10,20), tf1="1h",   tf2="30min")


def build_sma16_m30m15(df):
    """002 — SMA16 momentum, M30 + M15, lags=(1,10,20)."""
    return _sma_mom(df, sma_n=16, lags=(1,10,20), tf1="30min", tf2="15min")


def build_price_mom_m15m5(df):
    """011 — Raw price momentum, M15 + M5, lags=(1,3,8)."""
    return _raw_mom(df, lags=(1,3,8), tf1="15min", tf2="5min")


def build_sma16_h1m30(df):
    """012 — SMA16 momentum, H1 + M30, lags=(8,10,15)."""
    return _sma_mom(df, sma_n=16, lags=(8,10,15), tf1="1h", tf2="30min")


def _raw_mom(df, lags, tf1, tf2):
    moms = []
    for tf in (tf1, tf2):
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


def _sma_mom(df, sma_n, lags, tf1, tf2):
    moms = []
    for tf in (tf1, tf2):
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


def build_exhaust(df, n_consec, dist_mult, pair):
    """003/004 — M5 N-consec same-direction + SMA14 distance gated by sp_gate.

    Mirrors strategy_exhaust_{a,b}_live/main.py:104.
    """
    sma_n = 14
    pip   = pip_sz(pair)
    sp_g  = SP_GATES[pair]
    c = df["close"].values
    o = df["open"].values
    n = len(c)
    sma = pd.Series(c).rolling(sma_n).mean().values
    sig = np.zeros(n, dtype=np.int8)
    # all_bull window: closes > opens over last N_CONSEC
    bull = (c > o).astype(np.int8)
    bear = (c < o).astype(np.int8)
    # convolve-style: window of size n_consec ending at i
    csum_bull = np.zeros(n, dtype=np.int32)
    csum_bear = np.zeros(n, dtype=np.int32)
    csum_bull[:n_consec-1] = -1
    csum_bear[:n_consec-1] = -1
    s_bu = bull[:n_consec].sum(); s_be = bear[:n_consec].sum()
    csum_bull[n_consec-1] = s_bu; csum_bear[n_consec-1] = s_be
    for i in range(n_consec, n):
        s_bu = s_bu + bull[i] - bull[i-n_consec]
        s_be = s_be + bear[i] - bear[i-n_consec]
        csum_bull[i] = s_bu
        csum_bear[i] = s_be
    dist = (c - sma) / pip
    thr  = dist_mult * sp_g
    long_mask  = (csum_bull == n_consec) & (dist >=  thr)
    short_mask = (csum_bear == n_consec) & ((-dist) >= thr)
    sig[long_mask]  = 1
    sig[short_mask] = -1
    return pd.Series(sig, index=df.index)


def build_exhaust_a(df, pair):
    """003 — exhaust A, N_CONSEC=4, DIST_MULT=2.0, TP=15p."""
    return build_exhaust(df, n_consec=4, dist_mult=2.0, pair=pair)


def build_exhaust_b(df, pair):
    """004 — exhaust B, N_CONSEC=2, DIST_MULT=1.0, TP=10p."""
    return build_exhaust(df, n_consec=2, dist_mult=1.0, pair=pair)


# ── Numba simulators ──────────────────────────────────────────────────────────

@njit(cache=True)
def simulate_tp_only(bid, ask, mid, sp, sig, pip, tp_pips, sp_gate, time_cap):
    """First-formation entry. Exit at TP. If time_cap > 0, also exit at i+time_cap.

    time_cap=0 ⇒ pure TP-only (matches live).
    time_cap>0 ⇒ exit at min(TP, time-stop).
    Returns: pnl, hold_bars, exit_type (0=TP, 1=TIME).
    """
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0; prev_sig = 0.0

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
            elif time_cap > 0 and (i - ei) >= time_cap:
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


@njit(cache=True)
def simulate_signal_flip(bid, ask, mid, sp, sig, pip, tp_pips, sp_gate):
    """First-formation entry. Exit at TP OR when signal returns to 0 / opposite.

    Returns: pnl, hold_bars, exit_type (0=TP, 2=FLIP).
    """
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0; prev_sig = 0.0

    for i in range(1, n):
        cur_sig = sig[i - 1]
        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            # Exit on TP first
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True
            # Else exit when signal no longer matches direction
            elif (dir_ == 1 and cur_sig != 1) or (dir_ == -1 and cur_sig != -1):
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 2
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


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


# ── Strategy registry ─────────────────────────────────────────────────────────

STRATEGIES = [
    dict(name="001 pmom_h1m30",        tp=15.0, sig_fn=build_pmom_h1m30,        per_pair=False),
    dict(name="002 sma16_m30m15",      tp=20.0, sig_fn=build_sma16_m30m15,      per_pair=False),
    dict(name="003 exhaust_a",         tp=15.0, sig_fn=build_exhaust_a,         per_pair=True),
    dict(name="004 exhaust_b",         tp=10.0, sig_fn=build_exhaust_b,         per_pair=True),
    dict(name="011 price_mom_m15m5",   tp=10.0, sig_fn=build_price_mom_m15m5,   per_pair=False),
    dict(name="012 sma16_h1m30",       tp=20.0, sig_fn=build_sma16_h1m30,       per_pair=False),
]


def warmup_jit():
    n = 500
    z  = np.zeros(n);  s = np.zeros(n)
    simulate_tp_only(z, z, z, z, s, 0.0001, 10.0, 2.0, 0)
    simulate_tp_only(z, z, z, z, s, 0.0001, 10.0, 2.0, 12)
    simulate_signal_flip(z, z, z, z, s, 0.0001, 10.0, 2.0)


def run_one_pair(strat, pair, all_rows):
    sig_fn = strat["sig_fn"]
    tp_pips = strat["tp"]
    df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
          .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip  = pip_sz(pair); sg = SP_GATES[pair]
    sig  = sig_fn(df, pair) if strat["per_pair"] else sig_fn(df)
    n_is = int(len(df) * IS_FRAC)
    oos  = df.iloc[n_is:]
    osig = sig.iloc[n_is:]
    bid  = oos["bid_c"].values.astype(np.float64)
    ask  = oos["ask_c"].values.astype(np.float64)
    mid  = oos["close"].values.astype(np.float64)
    sp   = ((ask - bid) / pip).astype(np.float64)
    sv   = osig.values.astype(np.float64)
    days = len(oos) / 288.0

    def _record(mode_label, time_cap, p, h, t):
        if len(p) == 0:
            all_rows.append(dict(strategy=strat["name"], pair=pair, mode=mode_label,
                                 n=0, ppd=0.0, wr=0.0, tp_pct=0.0, alt_pct=0.0,
                                 avg_alt=0.0, mdd=0.0, calmar=0.0,
                                 mean_hold=0.0, days=days))
            return
        pnl = p
        n = len(pnl)
        tp_n   = int((t==0).sum())
        alt_n  = int((t!=0).sum())
        wr     = (pnl>0).sum() / n * 100
        tp_pct = tp_n / n * 100
        alt_pct= alt_n / n * 100
        ppd    = pnl.sum() / days
        mdd    = max_dd(pnl)
        cal    = ppd / mdd if mdd > 0 else 0.0
        avg_alt= pnl[t!=0].mean() if alt_n > 0 else 0.0
        all_rows.append(dict(strategy=strat["name"], pair=pair, mode=mode_label,
                             n=n, ppd=round(ppd,2), wr=round(wr,1),
                             tp_pct=round(tp_pct,1), alt_pct=round(alt_pct,1),
                             avg_alt=round(float(avg_alt),1), mdd=round(mdd,1),
                             calmar=round(cal,2),
                             mean_hold=round(float(h.mean()),1) if n else 0.0,
                             days=round(days,1)))

    # Baseline TP-only
    p,h,t = simulate_tp_only(bid,ask,mid,sp,sv,pip,tp_pips,sg,0)
    _record("TP_ONLY", 0, p, h, t)

    # Time caps
    for T in TIME_STOPS_M5:
        p,h,t = simulate_tp_only(bid,ask,mid,sp,sv,pip,tp_pips,sg,T)
        _record(f"TIME_{T}b", T, p, h, t)

    # Signal flip
    p,h,t = simulate_signal_flip(bid,ask,mid,sp,sv,pip,tp_pips,sg)
    _record("SIG_FLIP", -1, p, h, t)


def main():
    warmup_jit()
    print("exit_mode_sweep: TP-only baseline + time-stop sweep + signal-flip")
    print(f"  strategies={len(STRATEGIES)}  pairs={len(ALL_PAIRS)}")
    all_rows = []
    t0 = time.time()
    for strat in STRATEGIES:
        print(f"\n=== {strat['name']}  TP={strat['tp']}p ===", flush=True)
        ts = time.time()
        for pair in ALL_PAIRS:
            run_one_pair(strat, pair, all_rows)
            gc.collect()
        print(f"  {time.time()-ts:.1f}s  ({len(ALL_PAIRS)} pairs)", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT / "exit_mode_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Summary: per strategy total p/d across pairs, per mode ───────────────
    print("\n" + "="*86)
    print("  Σ p/d across all 12 pairs, by strategy × mode (positive ⇒ portfolio survives)")
    print("="*86)
    modes = ["TP_ONLY"] + [f"TIME_{T}b" for T in TIME_STOPS_M5] + ["SIG_FLIP"]
    print(f"  {'strategy':<22}", end="")
    for m in modes:
        print(f"{m:>10}", end="")
    print()
    print("  " + "-"*84)
    for strat in STRATEGIES:
        sub = df[df.strategy == strat["name"]]
        print(f"  {strat['name']:<22}", end="")
        for m in modes:
            s = sub[sub["mode"] == m]
            total = s["ppd"].sum()
            print(f"{total:>+10.1f}", end="")
        print()

    # ── pair pass count: how many of 12 pairs have ppd>0 in each mode ─────
    print("\n" + "="*86)
    print("  pairs with OOS ppd > 0  (out of 12)")
    print("="*86)
    print(f"  {'strategy':<22}", end="")
    for m in modes:
        print(f"{m:>10}", end="")
    print()
    print("  " + "-"*84)
    for strat in STRATEGIES:
        sub = df[df.strategy == strat["name"]]
        print(f"  {strat['name']:<22}", end="")
        for m in modes:
            s = sub[sub["mode"] == m]
            n_pos = int((s["ppd"] > 0).sum())
            print(f"{n_pos:>10d}", end="")
        print()


if __name__ == "__main__":
    main()
