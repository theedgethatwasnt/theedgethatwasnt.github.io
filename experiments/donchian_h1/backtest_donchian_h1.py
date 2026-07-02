"""
H1 Donchian Trend-Follow — Full Parameter Sweep
=================================================
Direct extension of the live H4 Donchian (acct 010) down to H1 for higher
trade frequency.

Logic (identical to H4, just faster TF):
  LONG:  H1 close > highest high of last N H1 bars (shift 1, causal)
  SHORT: H1 close < lowest low  of last N H1 bars
  Exit:  ATR trail from peak, OR max-hold limit
  Optional H4 filter: only enter long when H1 close > H4 Donchian midline
                      only enter short when H1 close < H4 Donchian midline

Sweep:
  N_H1       ∈ {5, 10, 20, 40}       (H1 lookback)
  trail_atr  ∈ {0.5, 1.0, 1.5, 2.0, 2.5}
  max_hold_h ∈ {6, 12, 24, 48, 96}   (max hold hours)
  h4_filter  ∈ {0=off, 1=on}
  → 4×5×5×2 = 200 configs per pair × 12 pairs = 2,400 total

Data: M5 BA parquets (grouped to H1/H4 by 12/48-bar windows).
Spread: H1 mean spread from underlying M5 bars.  IS P90 gate (R5).
IS/OOS: 70/30 split at H1 bar level.
Selection: OOS p/d > 0  AND  OOS trades ≥ 20  AND  Calmar ≥ 0.5

SOP: R1 (closed bars only), R3 (mid price, explicit spread), R4 (causal),
     R5 (IS P90 spread gate), R8 (OOS sealed).

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/donchian_h1/backtest_donchian_h1.py
"""

import time, gc
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC         = 0.70
M5_PER_H1       = 12
M5_PER_H4       = 48
ATR_PERIOD      = 14
M5_PER_CAL_DAY  = 288
H1_PER_CAL_DAY  = 24

PAIRS = [
    ("GBP_JPY", 0.01), ("USD_JPY", 0.01), ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001), ("AUD_JPY", 0.01),
    ("CHF_JPY", 0.01), ("NZD_JPY", 0.01), ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001), ("NZD_USD", 0.0001), ("EUR_GBP", 0.0001),
]

N_H1_VALS    = np.array([5, 10, 20, 40],          dtype=np.int32)
TRAIL_VALS   = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
HOLD_VALS    = np.array([6, 12, 24, 48, 96],       dtype=np.int32)
H4FILT_VALS  = np.array([0, 1],                    dtype=np.int32)

MIN_OOS_TRADES = 20
MIN_CALMAR     = 0.5


# ── Data loading + aggregation ────────────────────────────────────────────────

def load_h1(pair: str, pip: float):
    """
    Load M5 BA parquet, group to H1.
    Returns:
        h1_open, h1_high, h1_low, h1_close  (pip units)
        h1_spread                             (pip units, mean of M5 spreads)
        n_m5                                  (total M5 bars, for IS/OOS split)
    """
    df = pd.read_parquet(BA_DIR / f"{pair}_M5_BA.parquet")
    n_m5 = len(df)
    df["_g"] = np.arange(n_m5) // M5_PER_H1
    df["spread_p"] = (df["ask_c"] - df["bid_c"]) / pip

    h1 = df.groupby("_g").agg(
        open  = ("open",     "first"),
        high  = ("high",     "max"),
        low   = ("low",      "min"),
        close = ("close",    "last"),
        sp    = ("spread_p", "mean"),
    )
    h1_open  = h1["open"].values.astype(np.float64)  / pip
    h1_high  = h1["high"].values.astype(np.float64)  / pip
    h1_low   = h1["low"].values.astype(np.float64)   / pip
    h1_close = h1["close"].values.astype(np.float64) / pip
    h1_sp    = h1["sp"].values.astype(np.float64)

    return h1_open, h1_high, h1_low, h1_close, h1_sp, n_m5


def make_donchian_h1(h1_high: np.ndarray, h1_low: np.ndarray, n: int):
    """Causal H1 Donchian bands (shift 1). Returns upper, lower in same units."""
    upper = pd.Series(h1_high).rolling(n, min_periods=n).max().shift(1).values
    lower = pd.Series(h1_low).rolling(n, min_periods=n).min().shift(1).values
    return upper, lower


def make_h4_midline(h1_high: np.ndarray, h1_low: np.ndarray, n_h4: int = 10):
    """
    Causal H4 Donchian midline, forward-filled to H1 resolution.
    Each H4 bar = 4 H1 bars.
    """
    n_h1 = len(h1_high)
    H1_PER_H4 = 4
    n_h4_bars = n_h1 // H1_PER_H4

    h4_hi = np.array([h1_high[g*H1_PER_H4 : (g+1)*H1_PER_H4].max() for g in range(n_h4_bars)])
    h4_lo = np.array([h1_low[g*H1_PER_H4  : (g+1)*H1_PER_H4].min() for g in range(n_h4_bars)])

    h4_don_hi = pd.Series(h4_hi).rolling(n_h4, min_periods=n_h4).max().shift(1).values
    h4_don_lo = pd.Series(h4_lo).rolling(n_h4, min_periods=n_h4).min().shift(1).values
    h4_mid_g  = np.where(np.isnan(h4_don_hi), np.nan, (h4_don_hi + h4_don_lo) / 2.0)

    # Forward-fill to H1 resolution
    h4_mid_h1 = np.full(n_h1, np.nan)
    for g in range(n_h4_bars):
        s = g * H1_PER_H4
        e = min(s + H1_PER_H4, n_h1)
        h4_mid_h1[s:e] = h4_mid_g[g]

    return h4_mid_h1


@nb.njit(cache=True)
def _wilder_atr(high, low, close, period):
    n = len(close)
    atr = np.empty(n)
    atr[:] = np.nan
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


# ── Numba simulation kernel ───────────────────────────────────────────────────

@nb.njit(cache=True)
def _run_h1(close_p, spread_p, atr_p,
            don_upper, don_lower, h4_mid,
            trail, max_hold, h4_filter,
            is_end):
    """
    Single-config H1 Donchian simulation.
    All arrays at H1 resolution, pip units.
    spread_p: sentinel 999.0 = spread-gated bar (no entry).
    h4_filter: 0=off, 1=require H4 midline direction match.
    Returns (is_trades, oos_pips, oos_trades, max_dd, oos_days).
    """
    n       = len(close_p)
    warmup  = ATR_PERIOD + 50

    pos     = 0;    ep  = 0.0;  peak = 0.0;  stop = 0.0
    held    = 0
    eq      = 0.0;  peak_eq = 0.0;  max_dd = 0.0
    oos_pip = 0.0;  oos_t = 0;  is_t = 0

    for i in range(warmup, n):
        cl  = close_p[i]
        sp  = spread_p[i]
        at  = atr_p[i]
        du  = don_upper[i]
        dl  = don_lower[i]
        h4m = h4_mid[i]

        if at != at or at <= 0.0:
            continue
        if du != du or dl != dl:
            continue

        # ── Exit ─────────────────────────────────────────────────────────────
        if pos != 0:
            held += 1
            if pos == 1:
                if cl > peak: peak = cl
                stop = peak - trail * at
                exit_now = cl <= stop or held >= max_hold
                pnl = cl - ep - sp if exit_now else 0.0
            else:
                if cl < peak: peak = cl
                stop = peak + trail * at
                exit_now = cl >= stop or held >= max_hold
                pnl = ep - cl - sp if exit_now else 0.0

            if exit_now:
                pos = 0
                eq += pnl
                if eq > peak_eq: peak_eq = eq
                dd = peak_eq - eq
                if dd > max_dd: max_dd = dd
                if i >= is_end:
                    oos_pip += pnl;  oos_t += 1
                else:
                    is_t += 1

        # ── Entry ────────────────────────────────────────────────────────────
        if pos == 0 and sp < 900.0:
            h4_ok_long  = (h4_filter == 0) or (h4m == h4m and cl > h4m)
            h4_ok_short = (h4_filter == 0) or (h4m == h4m and cl < h4m)

            if cl > du and h4_ok_long:
                pos  = 1;  ep = cl;  peak = cl;  held = 0
                stop = cl - trail * at
            elif cl < dl and h4_ok_short:
                pos  = -1; ep = cl;  peak = cl;  held = 0
                stop = cl + trail * at

    oos_days = float(n - is_end) / H1_PER_CAL_DAY
    return float(is_t), oos_pip, float(oos_t), max_dd, oos_days


@nb.njit(parallel=True, cache=True)
def _sweep_h1(close_p, spread_p, atr_p,
              all_upper, all_lower, h4_mid,
              n_h1_vals, trail_vals, hold_vals, h4filt_vals,
              is_end):
    """
    Parallel sweep: N_H1 × trail × hold × h4_filter.
    all_upper/all_lower: shape (n_N_H1, n_h1_bars).
    Returns out array: (idx_n, idx_trail, idx_hold, idx_h4f,
                        is_t, oos_pip, oos_t, max_dd, oos_days)
    """
    nN = len(n_h1_vals); nT = len(trail_vals)
    nH = len(hold_vals); nF = len(h4filt_vals)
    total = nN * nT * nH * nF
    out   = np.empty((total, 9), dtype=np.float64)

    for k in prange(total):
        fi = k % nF
        hi = (k // nF) % nH
        ti = (k // (nF * nH)) % nT
        ni = k // (nF * nH * nT)

        res = _run_h1(
            close_p, spread_p, atr_p,
            all_upper[ni], all_lower[ni], h4_mid,
            trail_vals[ti], hold_vals[hi], h4filt_vals[fi],
            is_end,
        )
        out[k, 0] = ni; out[k, 1] = ti; out[k, 2] = hi; out[k, 3] = fi
        out[k, 4] = res[0]; out[k, 5] = res[1]; out[k, 6] = res[2]
        out[k, 7] = res[3]; out[k, 8] = res[4]

    return out


# ── Per-pair runner ───────────────────────────────────────────────────────────

def run_pair(pair: str, pip: float) -> pd.DataFrame:
    h1_open, h1_high, h1_low, h1_close, h1_sp, _ = load_h1(pair, pip)
    n_h1  = len(h1_close)
    is_end = int(n_h1 * IS_FRAC)

    # IS spread gate (R5)
    sp_gate = float(np.nanpercentile(h1_sp[:is_end], 90))
    sp_filt = np.where(h1_sp > sp_gate, 999.0, h1_sp)

    atr_p = _wilder_atr(h1_high, h1_low, h1_close, ATR_PERIOD)

    # Precompute Donchian bands for all N values
    nN = len(N_H1_VALS)
    all_upper = np.full((nN, n_h1), np.nan)
    all_lower = np.full((nN, n_h1), np.nan)
    for i, n in enumerate(N_H1_VALS):
        u, l = make_donchian_h1(h1_high, h1_low, int(n))
        all_upper[i] = np.where(np.isnan(u), np.nan, u)
        all_lower[i] = np.where(np.isnan(l), np.nan, l)

    # H4 midline (N_H4=10 fixed — same as live H4 Donchian)
    h4_mid_raw = make_h4_midline(h1_high, h1_low, n_h4=10)
    h4_mid     = np.where(np.isnan(h4_mid_raw), np.nan, h4_mid_raw)

    raw = _sweep_h1(
        h1_close, sp_filt, atr_p,
        all_upper, all_lower, h4_mid,
        N_H1_VALS.astype(np.int32),
        TRAIL_VALS,
        HOLD_VALS.astype(np.int32),
        H4FILT_VALS.astype(np.int32),
        is_end,
    )

    oos_days = float(n_h1 - is_end) / H1_PER_CAL_DAY

    rows = []
    for row in raw:
        ni, ti, hi, fi = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        is_t   = int(row[4])
        oos_p  = row[5]
        oos_t  = int(row[6])
        max_dd = row[7]

        if oos_t < MIN_OOS_TRADES or oos_p <= 0.0:
            continue

        pd_val = oos_p / oos_days if oos_days > 0 else 0.0
        if pd_val <= 0.0:
            continue

        calmar = (pd_val * 365.0) / max_dd if max_dd > 0 else 0.0
        if calmar < MIN_CALMAR:
            continue

        tpd = oos_t / oos_days if oos_days > 0 else 0.0

        rows.append({
            "pair":     pair,
            "N_H1":     int(N_H1_VALS[ni]),
            "trail":    float(TRAIL_VALS[ti]),
            "hold_h":   int(HOLD_VALS[hi]),
            "h4_filt":  int(H4FILT_VALS[fi]),
            "is_t":     is_t,
            "oos_t":    oos_t,
            "oos_p":    round(oos_p, 1),
            "p_d":      round(pd_val, 2),
            "t_d":      round(tpd, 3),
            "MaxDD":    round(max_dd, 1),
            "Calmar":   round(calmar, 2),
            "sp_gate":  round(sp_gate, 2),
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0    = time.time()
    ncfg  = len(N_H1_VALS) * len(TRAIL_VALS) * len(HOLD_VALS) * len(H4FILT_VALS)
    print("H1 Donchian Trend-Follow — Parameter Sweep")
    print(f"Pairs: {len(PAIRS)}  |  Configs/pair: {ncfg}  |  Total: {ncfg*len(PAIRS)}")
    print(f"IS={IS_FRAC:.0%}  OOS={1-IS_FRAC:.0%}  "
          f"min_oos_t={MIN_OOS_TRADES}  min_calmar={MIN_CALMAR}")
    print(f"H4 filter uses N_H4=10 (matches live acct 010 Donchian period)")
    print()

    print("Compiling Numba...", end=" ", flush=True)
    _d = np.ones(100, dtype=np.float64)
    _wilder_atr(_d, _d, _d, 14)
    print("done.")
    print()

    all_dfs = []
    best_rows = []

    for pair, pip in PAIRS:
        t1 = time.time()
        print(f"  {pair}...", end=" ", flush=True)
        try:
            df = run_pair(pair, pip)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if df.empty:
            print("0 survivors")
            continue

        all_dfs.append(df)
        best = df.sort_values("Calmar", ascending=False).iloc[0]
        best_rows.append({**best.to_dict(), "pair": pair})
        print(
            f"{len(df):4d} survivors | best Calmar={best['Calmar']:.2f} "
            f"p/d={best['p_d']:.1f} t/d={best['t_d']:.3f} "
            f"MaxDD={best['MaxDD']:.0f} "
            f"N={best['N_H1']} tr={best['trail']} hold={best['hold_h']}h "
            f"h4f={'Y' if best['h4_filt'] else 'N'} "
            f"({time.time()-t1:.1f}s)"
        )
        gc.collect()

    print()
    if not all_dfs:
        print("❌ No survivors across any pair. H1 Donchian has no deployable edge.")
        return

    # Save full results
    out_dir = Path(__file__).parent
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(out_dir / "results_h1_donchian.csv", index=False)

    # Summary table
    print("─" * 85)
    print("BEST PER PAIR (Calmar-ranked)")
    print(f"{'Pair':<10} {'N':>4} {'Tr':>5} {'Hold':>6} {'H4f':>4} "
          f"{'p/d':>7} {'t/d':>6} {'MaxDD':>7} {'Calmar':>7}")
    print("─" * 85)

    total_pd = total_td = 0.0
    best_rows.sort(key=lambda r: r["Calmar"], reverse=True)
    for r in best_rows:
        print(
            f"{r['pair']:<10} {r['N_H1']:>4} {r['trail']:>5.1f} {r['hold_h']:>5}h "
            f"{'Y' if r['h4_filt'] else 'N':>4} "
            f"{r['p_d']:>7.1f} {r['t_d']:>6.3f} {r['MaxDD']:>7.0f} {r['Calmar']:>7.2f}"
        )
        total_pd += r["p_d"]
        total_td += r["t_d"]

    print("─" * 85)
    ok = "✅" if total_td >= 2.0 else "❌"
    print(f"{'TOTAL':<30} {total_pd:>7.1f} {total_td:>6.3f}")
    print(f"> 2 t/d target: {ok} ({total_td:.3f} t/d)\n")

    # H4-filter vs no-filter comparison
    print("H4 FILTER BREAKDOWN (all survivors):")
    filt_grp = full_df.groupby("h4_filt").agg(
        n=("pair", "count"),
        mean_pd=("p_d", "mean"),
        mean_calmar=("Calmar", "mean"),
        mean_td=("t_d", "mean"),
    )
    print(filt_grp.to_string())

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print(f"Full results: {out_dir}/results_h1_donchian.csv")


if __name__ == "__main__":
    main()
