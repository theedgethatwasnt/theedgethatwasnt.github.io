"""
AMDDP5 Scoring Kernel — reusable Numba simulator for deterministic-rule sweeps
=============================================================================

Metric: AMDDP5 (Accumulated MDD-Penalized P&L, k=0.05)
---------------------------------------------------------
    AMDDP5_trade = pnl_pips − 0.05 × Σ_t max(0, −u_t)
where u_t is the trade's running unrealized P&L in pips at the close of every
held bar (from the bar AFTER entry through the exit bar, inclusive).

A trade that goes straight to TP without dipping below entry pays zero penalty.
A trade that sits 10 pips underwater for 5 bars pays 0.05 × 50 = 2.5 pips off
its final P&L — even if the trade eventually wins.

Source: "From Fork to Forward Test" (Bandy / NN-based optimization criterion).

------------------------------------------------------------------
Interface
------------------------------------------------------------------
score_signal(close, hi, lo, spread, pip, sig, tp_pips, sl_pips, max_hold,
             sp_gate, start, end)
    sig[t] ∈ {-1, 0, +1}  — signal known at bar close of t, traded at t+1.
    Returns four flat arrays of length n_trades:
        entry_t, exit_t, direction, pnl_pips, acc_dd_pips, hold_bars
    plus aggregate (sum_pnl, sum_amddp5, n_wins).

Exit logic:
    1. SL: bar-close fill if low/high crosses entry ∓ sl (or never if sl<=0).
    2. TP: bar-close fill if mid touched entry ± tp_pips (using bar mid close).
    3. max_hold: forced close at bar close (mid).

Entries:
    sig[t-1] != 0  (signal computed at close of t-1)
    spread[t] / pip <= sp_gate  (SOP R5 — gate hardcoded from IS-only)
    Fill: ask[t] (long) / bid[t] (short), bar t open-or-close depending on
    simulator design. We follow lag_sweep convention: ask_c/bid_c at the next
    bar (t), i.e. trade is filled at the close of the bar following the signal.
    Spread cost is deducted once at exit as `spread[exit_t] / pip`.

SOP compliance:
    R1: All decisions at bar close; sig[t-1] consumed at bar t.
    R3: Mid prices for the running unrealized P&L; spread deducted at exit.
    R5: sp_gate is passed in (computed IS-only by caller).
    R6: Same kernel for backtest. (For live: feed bar-by-bar with same logic.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

AMDDP_K = 0.05   # default penalty coefficient (AMDDP5)
                 # Override by passing lambda_pct=1/5/10 to score_signal()
                 # AMDDP1  (λ=0.01): mild penalty, allows pain in winners
                 # AMDDP5  (λ=0.05): moderate (default)
                 # AMDDP10 (λ=0.10): strong, encourages conservative behavior


# ── Core per-trade kernel ─────────────────────────────────────────────────────

@njit(cache=True)
def _run_kernel(close, hi, lo, bid_c, ask_c, spread, pip, sig,
                tp_pips, sl_pips, max_hold, sp_gate, start, end,
                bar_minutes):
    # bar_minutes: minutes per bar (e.g. 1/12 for S5, 5.0 for M5). Used to
    # convert the running drawdown integral from pip-bars → pip-minutes so
    # the same λ=0.05 means the same thing on S5 and M5 data.
    """Run the deterministic backtest. Returns flat per-trade arrays.

    Trade entry: at bar t, if sig[t-1] != 0 and not in trade and sp[t] <= gate,
                 fill at ask_c[t] (long) / bid_c[t] (short).
    Bar-by-bar running unrealized P&L (in pips, mid-based) tracked for AMDDP.
    Exits:  SL (lo/hi crosses entry ∓ sl) → bar close fill (mid).
            TP (mid extreme touches entry ± tp) → bar close fill (mid).
            max_hold reached → bar close fill (mid).
    Exit spread cost deducted once at exit bar.
    """
    n = end - start
    if n <= 1:
        empty = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, empty_i, empty, empty, empty_i

    # Preallocate worst case: one trade per bar
    entry_t = np.empty(n, dtype=np.int64)
    exit_t  = np.empty(n, dtype=np.int64)
    dir_    = np.empty(n, dtype=np.int64)
    pnl     = np.empty(n, dtype=np.float64)
    acc_dd  = np.empty(n, dtype=np.float64)
    hold_b  = np.empty(n, dtype=np.int64)

    nt = 0
    in_trade = False
    e_idx = 0
    e_px  = 0.0
    di    = 0
    hold  = 0
    cum_dd = 0.0   # accumulated underwater pip-minutes for current trade
                   # (NB: NOT acceleration — this is a time integral, units pip·min)

    for t in range(start + 1, end):
        sp_t = spread[t] / pip

        if in_trade:
            hold += 1

            # Running unrealized P&L at this bar's close (mid-based, no spread)
            u = di * (close[t] - e_px) / pip
            if u < 0.0:
                # Cadence-invariant: |drawdown| × bar duration in minutes
                # → cum_dd in pip·minutes (same units on S5 and M5)
                cum_dd += -u * bar_minutes

            should_exit = False
            exit_px_mid = close[t]   # default: bar-close mid

            # (1) SL — bar-close fill if intra-bar crossed
            if sl_pips > 0.0:
                if di == 1 and lo[t] <= e_px - sl_pips * pip:
                    should_exit = True
                elif di == -1 and hi[t] >= e_px + sl_pips * pip:
                    should_exit = True

            # (2) TP — bar-close fill if intra-bar mid extreme touched
            if not should_exit and tp_pips > 0.0:
                if di == 1 and hi[t] >= e_px + tp_pips * pip:
                    should_exit = True
                elif di == -1 and lo[t] <= e_px - tp_pips * pip:
                    should_exit = True

            # (3) max_hold
            if not should_exit and hold >= max_hold:
                should_exit = True

            if should_exit:
                # Exit at bar close mid, deduct full spread of exit bar
                p = di * (exit_px_mid - e_px) / pip - sp_t
                entry_t[nt] = e_idx
                exit_t[nt]  = t
                dir_[nt]    = di
                pnl[nt]     = p
                acc_dd[nt]  = cum_dd
                hold_b[nt]  = hold
                nt += 1
                in_trade = False

        else:
            nd = sig[t - 1]
            if nd != 0 and sp_t <= sp_gate:
                # Fill at bar t close ask/bid
                e_px  = ask_c[t] if nd == 1 else bid_c[t]
                di    = nd
                e_idx = t
                hold  = 0
                cum_dd = 0.0
                in_trade = True

    # Force-close any open trade at the boundary
    if in_trade and end > start + 1:
        t_last = end - 1
        sp_t = spread[t_last] / pip
        p = di * (close[t_last] - e_px) / pip - sp_t
        entry_t[nt] = e_idx
        exit_t[nt]  = t_last
        dir_[nt]    = di
        pnl[nt]     = p
        acc_dd[nt]  = cum_dd
        hold_b[nt]  = hold
        nt += 1

    return entry_t[:nt], exit_t[:nt], dir_[:nt], pnl[:nt], acc_dd[:nt], hold_b[:nt]


# ── Aggregates ────────────────────────────────────────────────────────────────

@njit(cache=True)
def amddp5_from_arrays(pnl, acc_dd):
    """AMDDP5_trade = pnl − k × acc_dd."""
    n = len(pnl)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = pnl[i] - AMDDP_K * acc_dd[i]
    return out


@njit(cache=True)
def mc_pvalue_amddp(amddp_arr, n_mc):
    """Sign-shuffle MC test on AMDDP5. P(shuffle_sum >= observed_sum)."""
    n = len(amddp_arr)
    if n == 0:
        return 1.0
    observed = 0.0
    for i in range(n):
        observed += amddp_arr[i]
    count = 0
    for _ in range(n_mc):
        s = 0.0
        for i in range(n):
            s += amddp_arr[i] if np.random.random() > 0.5 else -amddp_arr[i]
        if s >= observed:
            count += 1
    return count / n_mc


# ── Python wrapper ────────────────────────────────────────────────────────────

def score_signal(close, hi, lo, bid_c, ask_c, spread, pip, sig,
                 tp_pips, sl_pips, max_hold, sp_gate, start=0, end=None,
                 bar_minutes=5.0, lambda_pct=5):
    """lambda_pct: 1 → AMDDP1 (mild), 5 → AMDDP5 (default), 10 → AMDDP10 (harsh).
    Internal λ = lambda_pct / 100."""
    """Run the kernel and return a trades DataFrame plus aggregates.

    Parameters
    ----------
    close, hi, lo, bid_c, ask_c, spread : np.ndarray[float64]
    pip      : float
    sig      : np.ndarray[int8] in {-1, 0, +1}; sig[t] used at bar t+1
    tp_pips  : float — TP in pips (mid-based intrabar extreme)
    sl_pips  : float — SL in pips (lo/hi crosses); <=0 disables
    max_hold : int   — bars
    sp_gate  : float — IS-only P90 spread (R5)
    start,end: int   — sim range

    Returns
    -------
    trades_df : pd.DataFrame, columns =
        [entry_t, exit_t, direction, pnl_pips, max_dd_pips,
         accumulated_dd_pips, amddp5, hold_bars]
    summary   : dict with sum_pnl, sum_amddp5, n_trades, n_wins
    """
    n = len(close)
    if end is None:
        end = n
    sig_i8 = np.ascontiguousarray(sig, dtype=np.int8)

    e_t, x_t, dr, pnl, acc, hb = _run_kernel(
        close.astype(np.float64, copy=False),
        hi.astype(np.float64, copy=False),
        lo.astype(np.float64, copy=False),
        bid_c.astype(np.float64, copy=False),
        ask_c.astype(np.float64, copy=False),
        spread.astype(np.float64, copy=False),
        float(pip),
        sig_i8,
        float(tp_pips),
        float(sl_pips),
        int(max_hold),
        float(sp_gate),
        int(start),
        int(end),
        float(bar_minutes),
    )

    k_used = float(lambda_pct) / 100.0
    amddp = pnl - k_used * acc

    # max_dd_pips: walk each trade's bars and find min running U.
    # Cheap to do here (vectorised per trade) since n_trades << n_bars.
    max_dd = np.zeros_like(pnl)
    for i in range(len(pnl)):
        s = e_t[i] + 1
        e = x_t[i] + 1
        if e <= s:
            max_dd[i] = 0.0
            continue
        u = dr[i] * (close[s:e] - (ask_c[e_t[i]] if dr[i] == 1 else bid_c[e_t[i]])) / pip
        max_dd[i] = float(-u.min()) if u.size else 0.0
        if max_dd[i] < 0.0:
            max_dd[i] = 0.0

    df = pd.DataFrame({
        "entry_t":             e_t,
        "exit_t":              x_t,
        "direction":           dr,
        "pnl_pips":            pnl,
        "max_dd_pips":         max_dd,
        "accumulated_dd_pips": acc,
        "amddp5":              amddp,
        "hold_bars":           hb,
    })
    summary = {
        "sum_pnl":    float(pnl.sum()),
        "sum_amddp5": float(amddp.sum()),
        "n_trades":   int(len(pnl)),
        "n_wins":     int((pnl > 0).sum()),
    }
    return df, summary


# ── Light self-test (executed only when run as a script) ─────────────────────

def _selftest():
    rng = np.random.default_rng(0)
    n = 5000
    px = 1.10 + np.cumsum(rng.normal(0, 1e-5, n))
    close = px
    hi    = px + 1e-5
    lo    = px - 1e-5
    bid_c = px - 5e-6
    ask_c = px + 5e-6
    spread = ask_c - bid_c
    pip = 0.0001
    sig = np.zeros(n, dtype=np.int8)
    # Random entries every 50 bars
    sig[::50] = 1
    df, agg = score_signal(close, hi, lo, bid_c, ask_c, spread, pip, sig,
                           tp_pips=10.0, sl_pips=10.0, max_hold=20, sp_gate=2.0)
    assert {"pnl_pips", "amddp5", "accumulated_dd_pips"}.issubset(df.columns)
    # AMDDP5 must be <= pnl_pips for every trade (penalty is non-negative)
    assert (df["amddp5"] <= df["pnl_pips"] + 1e-9).all(), "AMDDP5 must not exceed raw pnl"
    # Clean trades (acc==0) must have amddp5 == pnl_pips
    clean = df[df["accumulated_dd_pips"] == 0.0]
    assert np.allclose(clean["amddp5"], clean["pnl_pips"]), "Clean trades must score equal"
    print(f"[selftest] OK — {agg['n_trades']} trades, "
          f"sum_pnl={agg['sum_pnl']:.1f}, sum_amddp5={agg['sum_amddp5']:.1f}")


if __name__ == "__main__":
    _selftest()
