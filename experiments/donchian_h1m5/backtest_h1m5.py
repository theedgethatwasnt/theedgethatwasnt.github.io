"""
H1+M5 Dual-TF Donchian Trend-Follow
=====================================
Entry: M5 close penetrates BOTH the M5 Donchian upper band AND the H1 Donchian
  upper band simultaneously.  Both timeframes confirming the same breakout =
  genuine momentum, not noise.

Motivation:
  H4 Donchian trend-follow works (live acct 010, Calmar ~100-150).
  Counter-trend H4 fades fail (breakouts are real trend starts).
  Goal: same proven edge at H1 scale → more trades/day (>2 target).
  M5 confirmation on top of H1 reduces false signals from short squeezes.

Entry:
  LONG:  M5 close > M5_Donchian_upper(N_M5) AND M5 close > H1_Donchian_upper(N_H1)
  SHORT: M5 close < M5_Donchian_lower(N_M5) AND M5 close < H1_Donchian_lower(N_H1)
  Gate:  at most one trade entry per H1 bar (prevents re-entry churn within bar)

Exit:
  H1 ATR trail (forward-filled to M5): peak ± trail × H1_ATR(14)
  OR max hold in H1 bars

All features are causal (SOP R1, R4).
Spread gate: IS P90 → bars above sentinel-999.  SOP R5.

Usage:
  cd /path/to/projects/fx-core
  python3 research/experiments/donchian_h1m5/backtest_h1m5.py
"""

import time, gc
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC        = 0.70
M5_PER_H1      = 12
M5_PER_DAY     = 288        # calendar day M5 bars
ATR_PER        = 14

PAIRS = [
    ("GBP_JPY", 0.01), ("USD_JPY", 0.01), ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001), ("AUD_JPY", 0.01),
    ("CHF_JPY", 0.01), ("NZD_JPY", 0.01), ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001), ("NZD_USD", 0.0001), ("EUR_GBP", 0.0001),
]

N_M5_VALS  = [10, 20, 40]        # M5 Donchian period (50 min / 100 min / 200 min)
N_H1_VALS  = [5, 10, 20]         # H1 Donchian period (5h / 10h / 20h)
TRAIL_VALS = [1.0, 1.5, 2.0, 2.5]
HOLD_VALS  = [12, 24, 48]        # max hold in H1 bars

MIN_OOS_TRADES = 20
MIN_CALMAR     = 0.5


# ─────────────────────────────────────────────────────────────────────────────
def load_m5(pair: str, pip: float) -> pd.DataFrame:
    df = pd.read_parquet(BA_DIR / f"{pair}_M5_BA.parquet")
    df["spread"] = (df["ask_c"] - df["bid_c"]) / pip
    return df


# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _wilder_atr(high, low, close, period):
    n = len(close); atr = np.empty(n); atr[:] = np.nan
    seed = 0.0
    for j in range(1, period + 1):
        seed += max(high[j] - low[j],
                    abs(high[j] - close[j-1]),
                    abs(low[j]  - close[j-1]))
    atr[period] = seed / period
    for i in range(period + 1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * (period - 1) + tr) / period
    return atr


def make_m5_bands(df: pd.DataFrame, n_m5: int, pip: float):
    """Causal M5 Donchian upper/lower in pip units (shift 1 = no current bar)."""
    hi = df["high"] / pip
    lo = df["low"]  / pip
    upper = hi.rolling(n_m5, min_periods=n_m5).max().shift(1).values
    lower = lo.rolling(n_m5, min_periods=n_m5).min().shift(1).values
    return upper, lower


def make_h1_features(df: pd.DataFrame, n_h1: int, pip: float):
    """
    H1 Donchian bands + ATR at H1 level, forward-filled to M5 resolution.
    Causal: H1 bar j's value uses H1 bars [j-n_h1 : j] (does not include j).
    All values in pip units.
    """
    n_m5 = len(df)
    df = df.copy()
    df["_g"] = np.arange(n_m5) // M5_PER_H1

    # H1 OHLC
    h1 = df.groupby("_g").agg(
        hi=("high", "max"),
        lo=("low",  "min"),
        cl=("close","last"),
    )

    h1_hi = h1["hi"].values / pip
    h1_lo = h1["lo"].values / pip
    h1_cl = h1["cl"].values / pip

    # Causal Donchian at H1 level (shift 1)
    h1u_s = pd.Series(h1_hi).rolling(n_h1, min_periods=n_h1).max().shift(1).values
    h1l_s = pd.Series(h1_lo).rolling(n_h1, min_periods=n_h1).min().shift(1).values

    # H1 Wilder ATR
    h1_atr = _wilder_atr(h1_hi, h1_lo, h1_cl, ATR_PER)

    # Forward-fill H1 values to M5 resolution
    n_h1_bars = len(h1)
    h1u_ff  = np.full(n_m5, np.nan)
    h1l_ff  = np.full(n_m5, np.nan)
    h1a_ff  = np.full(n_m5, np.nan)

    for g in range(n_h1_bars):
        s = g * M5_PER_H1
        e = min(s + M5_PER_H1, n_m5)
        h1u_ff[s:e] = h1u_s[g]
        h1l_ff[s:e] = h1l_s[g]
        h1a_ff[s:e] = h1_atr[g]

    return h1u_ff, h1l_ff, h1a_ff


# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _run(close_p, spread_p,
         m5_upper, m5_lower,
         h1_upper, h1_lower, h1_atr,
         trail, max_hold_m5, is_end):
    """
    Single-config simulation at M5 resolution.  All prices in pip units.

    Entry gate: at most 1 entry per H1 bar (tracks last H1 group index).
    Exit: H1 ATR trail on close prices, or max hold.
    PnL: directional move – spread at exit bar.
    """
    n      = len(close_p)
    warmup = ATR_PER * M5_PER_H1 + 200   # warmup for H1 ATR + N bands

    pos        = 0;     ep  = 0.0;   peak_p = 0.0;  stop_p = 0.0
    held_m5    = 0
    eq         = 0.0;   peak_eq = 0.0;   max_dd = 0.0
    oos_p      = 0.0;   oos_t  = 0;     is_t   = 0
    last_h1g   = -1     # H1 group index of last entry

    for i in range(warmup, n):
        cl  = close_p[i]
        sp  = spread_p[i]
        m5u = m5_upper[i]
        m5l = m5_lower[i]
        h1u = h1_upper[i]
        h1l = h1_lower[i]
        h1a = h1_atr[i]

        if m5u != m5u or h1u != h1u or h1a != h1a or h1a <= 0.0:
            continue

        h1g = i // M5_PER_H1     # current H1 group

        # ── Exit ─────────────────────────────────────────────────────────────
        if pos != 0:
            held_m5 += 1
            if pos == 1:   # LONG: trail below peak
                if cl > peak_p: peak_p = cl
                stop_p = peak_p - trail * h1a
                exit_now = cl <= stop_p or held_m5 >= max_hold_m5
                if exit_now:
                    pnl = cl - ep - sp
            else:           # SHORT: trail above peak
                if cl < peak_p: peak_p = cl
                stop_p = peak_p + trail * h1a
                exit_now = cl >= stop_p or held_m5 >= max_hold_m5
                if exit_now:
                    pnl = ep - cl - sp

            if exit_now:
                pos = 0
                eq += pnl
                if eq > peak_eq: peak_eq = eq
                dd = peak_eq - eq
                if dd > max_dd: max_dd = dd
                if i >= is_end:
                    oos_p += pnl;  oos_t += 1
                else:
                    is_t += 1

        # ── Entry ────────────────────────────────────────────────────────────
        if pos == 0 and sp < 900.0 and h1g != last_h1g:
            if cl > m5u and cl > h1u:    # LONG: both TFs broken out upward
                pos     = 1
                ep      = cl;  peak_p = cl;  held_m5 = 0
                stop_p  = cl - trail * h1a
                last_h1g = h1g
            elif cl < m5l and cl < h1l:  # SHORT: both TFs broken down
                pos     = -1
                ep      = cl;  peak_p = cl;  held_m5 = 0
                stop_p  = cl + trail * h1a
                last_h1g = h1g

    oos_days = (n - is_end) / M5_PER_DAY
    return oos_p, float(oos_t), max_dd, oos_days, float(is_t)


@nb.njit(parallel=True, cache=True)
def _sweep(close_p, spread_p,
           m5_upper, m5_lower,
           h1_upper, h1_lower, h1_atr,
           c_trail, c_hold, is_end):
    """Parallel sweep over (trail × hold) for fixed (N_M5, N_H1)."""
    nc       = len(c_trail)
    oos_pips = np.zeros(nc)
    oos_t    = np.zeros(nc)
    max_dds  = np.zeros(nc)
    oos_days = np.zeros(nc)

    for k in prange(nc):
        r = _run(close_p, spread_p,
                 m5_upper, m5_lower,
                 h1_upper, h1_lower, h1_atr,
                 c_trail[k], c_hold[k], is_end)
        oos_pips[k] = r[0]
        oos_t[k]    = r[1]
        max_dds[k]  = r[2]
        oos_days[k] = r[3]

    return oos_pips, oos_t, max_dds, oos_days


# ─────────────────────────────────────────────────────────────────────────────
def run_pair(pair: str, pip: float) -> pd.DataFrame:
    df     = load_m5(pair, pip)
    n      = len(df)
    is_end = int(n * IS_FRAC)

    cl = (df["close"].values / pip).astype(np.float64)
    sp_raw  = df["spread"].values.astype(np.float64)
    sp_gate = np.percentile(sp_raw[:is_end], 90)
    sp      = np.where(sp_raw > sp_gate, 999.0, sp_raw)

    oos_days = (n - is_end) / M5_PER_DAY

    # Config arrays (sweep over trail × hold)
    c_trail = np.array([t for t in TRAIL_VALS for _ in HOLD_VALS], dtype=np.float64)
    c_hold  = np.array([h * M5_PER_H1 for _ in TRAIL_VALS for h in HOLD_VALS],
                       dtype=np.int64)
    c_hold_h1 = np.array([h for _ in TRAIL_VALS for h in HOLD_VALS], dtype=np.int64)
    nc = len(c_trail)

    rows = []

    for n_m5 in N_M5_VALS:
        m5u, m5l = make_m5_bands(df, n_m5, pip)

        for n_h1 in N_H1_VALS:
            h1u, h1l, h1a = make_h1_features(df, n_h1, pip)

            op, ot, om, od = _sweep(
                cl, sp, m5u, m5l, h1u, h1l, h1a,
                c_trail, c_hold, is_end,
            )

            for k in range(nc):
                if ot[k] < MIN_OOS_TRADES:
                    continue
                pd_val = op[k] / oos_days if oos_days > 0 else 0.0
                if pd_val <= 0.0:
                    continue
                calmar = (pd_val * 252) / om[k] if om[k] > 0 else 0.0
                if calmar < MIN_CALMAR:
                    continue
                rows.append({
                    "pair":   pair,
                    "N_M5":   n_m5,
                    "N_H1":   n_h1,
                    "trail":  c_trail[k],
                    "hold_h1": c_hold_h1[k],
                    "config": f"m{n_m5}_h{n_h1}_tr{c_trail[k]}_hold{c_hold_h1[k]}",
                    "p_d":    round(pd_val, 2),
                    "t_d":    round(ot[k] / oos_days, 3),
                    "MaxDD":  round(om[k], 1),
                    "Calmar": round(calmar, 2),
                    "OOS_t":  int(ot[k]),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    total_configs = len(N_M5_VALS) * len(N_H1_VALS) * len(TRAIL_VALS) * len(HOLD_VALS)

    print("H1+M5 Dual-TF Donchian Trend-Follow")
    print(f"Pairs: {len(PAIRS)}  |  Configs per pair: {total_configs}")
    print(f"IS={IS_FRAC:.0%}  OOS={1-IS_FRAC:.0%}  "
          f"Min OOS trades={MIN_OOS_TRADES}  Min Calmar={MIN_CALMAR}\n")

    print("Compiling Numba kernels...", end=" ", flush=True)
    _d = np.ones(100, dtype=np.float64)
    _wilder_atr(_d, _d, _d, 14)
    print("done.\n")

    all_surv      = []
    best_per_pair = []

    for pair, pip in PAIRS:
        t1 = time.time()
        print(f"  {pair}...", end=" ", flush=True)
        try:
            df_surv = run_pair(pair, pip)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        if df_surv.empty:
            print("0 survivors")
            continue

        all_surv.append(df_surv)
        best = df_surv.sort_values("Calmar", ascending=False).iloc[0]
        best_per_pair.append(best)
        print(f"{len(df_surv):4d} survivors | "
              f"best Calmar={best['Calmar']:.2f} p/d={best['p_d']:.1f} "
              f"t/d={best['t_d']:.3f} MaxDD={best['MaxDD']:.0f} "
              f"[{best['config']}]  ({time.time()-t1:.1f}s)")
        gc.collect()

    print()
    print("─" * 90)
    print(f"PORTFOLIO — best per pair (Calmar-ranked), aggregate t/d (target > 2/day)")
    print(f"{'Pair':<12} {'Config':<35} {'p/d':>6} {'t/d':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("─" * 90)

    total_pd = total_td = 0.0
    for r in sorted(best_per_pair, key=lambda x: x["Calmar"], reverse=True):
        print(f"{r['pair']:<12} {r['config']:<35} {r['p_d']:>6.1f} "
              f"{r['t_d']:>7.3f} {r['MaxDD']:>7.1f} {r['Calmar']:>7.2f}")
        total_pd += r["p_d"]
        total_td += r["t_d"]

    print("─" * 90)
    ok = "✅" if total_td >= 2.0 else "❌"
    print(f"{'TOTAL':<12} {'':35} {total_pd:>6.1f} {total_td:>7.3f}")
    print(f"> 2 t/d target: {ok} ({total_td:.3f} t/d)\n")
    print(f"Total runtime: {time.time()-t0:.1f}s")

    if all_surv:
        out = Path(__file__).parent / "results_h1m5.csv"
        pd.concat(all_surv, ignore_index=True).to_csv(out, index=False)
        print(f"Survivors → {out}")


if __name__ == "__main__":
    main()
