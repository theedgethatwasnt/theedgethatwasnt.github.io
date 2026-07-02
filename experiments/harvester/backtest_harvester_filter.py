#!/usr/bin/env python3
"""
Harvester Anti-Entry Filter on SMA16 + PriceMom Strategies
============================================================
Test whether the bar-exhaustion condition (validated in Harvester experiments)
can improve the live SMA16 and PriceMom strategies by skipping entries where
M5 price is in an exhausted state likely to reverse.

Background:
  - Harvester (M5 + M15) found genuine bar-exhaustion signal (70-77% WR for
    mean-reversion). Spread killed it as a standalone strategy.
  - As an anti-entry filter: when SMA16/PriceMom fires a momentum entry AND
    M5 bar-exhaustion fires in the SAME direction (price stretched ≥ dist_mult×sp_gate
    from SMA14 after n_consec same-direction bars) → skip the entry.
  - The ~75% WR means 3/4 probability the next bars will pull back against the
    momentum direction, making this a high-confidence skip signal.

Filter logic:
  exhaust_dir[i] = direction of bar exhaustion at M5 bar i:
    +1 = upward (last n_consec all-BULL, close > SMA14, |dist| ≥ dist_mult × sp_gate)
    -1 = downward (last n_consec all-BEAR, close < SMA14, |dist| ≥ dist_mult × sp_gate)
    0  = no exhaustion signal

  Skip momentum entry when: exhaust_dir[entry_bar] == signal_direction
  (momentum is pushing into an already-exhausted state → mean-reversion likely)

Sweep parameters:
  n_consec ∈ {2, 3}         (bars needed for exhaustion condition)
  dist_mult ∈ {2.0, 2.5, 3.0}  (distance threshold in multiples of sp_gate pips)

Strategies tested:
  A. SMA16 momentum: SMA16, H1+H30, lags=(8,10,15), TP=20p, 10 pairs (deployed acct 012)
  B. PriceMom: M15+M5, lags=(1,3,8), TP=10p, 12 pairs (deployed acct 011)

For each filter config, report:
  1. Trade count reduction (% filtered)
  2. WR of filtered vs kept trades (IC of filter)
  3. IS WF 3-fold pass count vs baseline
  4. OOS p/d vs baseline
  5. MC p-value vs baseline
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IS_FRAC  = 0.70
MC_N     = 1000

PAIRS_SMA16 = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "EUR_GBP", "AUD_USD",
    "NZD_JPY", "CHF_JPY",
]
PAIRS_PMOM = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "EUR_GBP", "AUD_USD",
    "NZD_JPY", "CHF_JPY", "NZD_USD", "CAD_JPY",
]
JPY_PAIRS = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

# Deployed strategy params
SMA_N   = 16
SMA_LAGS = (8, 10, 15)
SMA_TFS  = ("1h", "30min")
SMA_TP   = 20.0

PMOM_LAGS = (1, 3, 8)
PMOM_TFS  = ("15min", "5min")
PMOM_TP   = 10.0

# Harvester filter sweep space
NCONSEC_OPT  = [2, 3]
DISTMULT_OPT = [2.0, 2.5, 3.0]
EXHAUST_SMA  = 14   # M5 SMA period for bar-exhaustion detection


def pip_sz(pair): return 0.01 if pair in JPY_PAIRS else 0.0001


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_sma_nb(close, period):
    n = len(close); out = np.empty(n, dtype=np.float64); s = 0.0
    for i in range(n):
        s += close[i]
        if i >= period: s -= close[i - period]
        out[i] = s / min(i + 1, period)
    return out


@njit(cache=True, fastmath=True)
def compute_exhaust_dir(close_pip, open_pip, sma14_pip, sp_arr, n_consec, dist_mult, sp_gate):
    """
    Returns bar-exhaustion direction array at M5 resolution (in pips).
    +1 = upward exhaustion → Harvester would SHORT, skip LONG momentum
    -1 = downward exhaustion → Harvester would LONG, skip SHORT momentum
     0 = no signal
    """
    n       = len(close_pip)
    exhaust = np.zeros(n, dtype=np.int8)
    thresh  = dist_mult * sp_gate   # fixed IS P90 threshold

    for i in range(n_consec, n):
        dist = abs(close_pip[i] - sma14_pip[i])
        if dist < thresh:
            continue

        above    = close_pip[i] > sma14_pip[i]
        all_bull = True
        all_bear = True
        for k in range(1, n_consec + 1):
            j = i - k
            if close_pip[j] < open_pip[j]: all_bull = False
            if close_pip[j] >= open_pip[j]: all_bear = False
            if not all_bull and not all_bear:
                break

        if all_bull and above:
            exhaust[i] = np.int8(1)      # upward exhaustion
        elif all_bear and not above:
            exhaust[i] = np.int8(-1)     # downward exhaustion

    return exhaust


@njit(cache=True, fastmath=True)
def simulate_tp_base(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    """Baseline TP simulation. Returns per-trade (pnl, direction) arrays."""
    MAX_T  = 20_000
    pnls   = np.empty(MAX_T, dtype=np.float64)
    dirs   = np.empty(MAX_T, dtype=np.int8)
    n_t    = 0
    n      = len(mid)
    in_t   = False; dir_ = 0; ep = 0.0

    for i in range(1, n):
        if in_t:
            if (mid[i] - ep) * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl     = (exit_px - ep) * dir_ - sp[i]
                if n_t < MAX_T:
                    pnls[n_t] = pnl; dirs[n_t] = np.int8(dir_); n_t += 1
                in_t = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep   = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_t = True

    return pnls[:n_t], dirs[:n_t]


@njit(cache=True, fastmath=True)
def simulate_tp_filt(mid, bid, ask, sp, sig, tp_pips, sp_gate, exhaust_dir):
    """Filtered TP simulation: skip entries where exhaust_dir[i-1] == sig[i-1].
    Returns (pnls, dirs, n_filtered) — pnls/dirs are only the KEPT trades."""
    MAX_T    = 20_000
    pnls     = np.empty(MAX_T, dtype=np.float64)
    dirs     = np.empty(MAX_T, dtype=np.int8)
    n_t      = 0
    n_filt   = np.int64(0)
    n        = len(mid)
    in_t     = False; dir_ = 0; ep = 0.0

    for i in range(1, n):
        if in_t:
            if (mid[i] - ep) * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl     = (exit_px - ep) * dir_ - sp[i]
                if n_t < MAX_T:
                    pnls[n_t] = pnl; dirs[n_t] = np.int8(dir_); n_t += 1
                in_t = False
        else:
            nd = sig[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                if exhaust_dir[i - 1] != 0 and exhaust_dir[i - 1] == nd:
                    n_filt += 1     # would-be entry is blocked
                    continue
                ep   = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_t = True

    return pnls[:n_t], dirs[:n_t], n_filt


def run_mc_pvalue(pnl_arr, is_days, n_shuffles=MC_N, seed=42):
    if len(pnl_arr) < 20:
        return np.nan
    actual = pnl_arr.sum() / is_days
    rng    = np.random.default_rng(seed)
    signs  = rng.choice(np.array([-1.0, 1.0]), size=(n_shuffles, len(pnl_arr)))
    shuffl = (np.abs(pnl_arr) * signs).sum(axis=1) / is_days
    return float((shuffl >= actual).mean())


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pair(pair):
    path = DATA / f"{pair}_M5_BA.parquet"
    df   = pd.read_parquet(path).set_index("timestamp").sort_index()
    df   = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip  = pip_sz(pair)
    n    = len(df)
    n_is = int(n * IS_FRAC)
    sp   = (df["ask_c"] - df["bid_c"]) / pip
    sp_gate = float(np.percentile(sp.iloc[:n_is], 90))
    return df, pip, n_is, sp_gate


# ── Signal builders ────────────────────────────────────────────────────────────

def build_sma16_signal(df):
    """Deployed SMA16 signal: H1+H30, lags=(8,10,15), LONG if all 6 > 0."""
    moms = []
    for tf in SMA_TFS:
        rs  = df["close"].resample(tf).last().dropna()
        sma = rs.rolling(SMA_N, min_periods=SMA_N).mean().shift(1)
        sma = sma.reindex(df.index, method="ffill")
        for k in SMA_LAGS:
            moms.append(sma - sma.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig   = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


def build_pmom_signal(df):
    """Deployed PriceMom signal: M15+M5, lags=(1,3,8), LONG if all 6 > 0."""
    moms = []
    for tf in PMOM_TFS:
        rs  = df["close"].resample(tf).last().dropna()
        rs  = rs.shift(1).reindex(df.index, method="ffill")
        for k in PMOM_LAGS:
            moms.append(rs - rs.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig   = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


# ── Stats computation ─────────────────────────────────────────────────────────

def sim_stats(df, sig, pip, sp_gate, n_is, tp, exhaust=None):
    """
    Run simulation and return stats dict.
    exhaust: None → baseline; array → filtered
    """
    mid_full = (df["close"].values / pip).astype(np.float64)
    bid_full = (df["bid_c"].values / pip).astype(np.float64)
    ask_full = (df["ask_c"].values / pip).astype(np.float64)
    sp_full  = ((df["ask_c"].values - df["bid_c"].values) / pip).astype(np.float64)
    sv_full  = sig.values.astype(np.int8)

    n_days_is  = n_is / 288
    n_days_oos = (len(df) - n_is) / 288

    def _slice(s, e):
        return (mid_full[s:e], bid_full[s:e], ask_full[s:e],
                sp_full[s:e], sv_full[s:e])

    def _run(s, e):
        m, b, a, sp, sv = _slice(s, e)
        if exhaust is not None:
            ex = exhaust[s:e]
            p, d, nf = simulate_tp_filt(m, b, a, sp, sv, tp, sp_gate, ex)
        else:
            p, d = simulate_tp_base(m, b, a, sp, sv, tp, sp_gate)
            nf   = np.int64(0)
        return p, d, nf

    # IS stats
    p_is, d_is, nf_is = _run(0, n_is)
    is_pd   = p_is.sum() / n_days_is if len(p_is) > 0 else 0.0
    is_wr   = (p_is > 0).mean() if len(p_is) > 0 else 0.0
    is_n    = len(p_is)

    # IS 3-fold WF
    fold    = n_is // 3
    wf_pds  = []
    for f in range(3):
        s = f * fold; e = s + fold if f < 2 else n_is
        p_f, _, _ = _run(s, e)
        days_f = (e - s) / 288
        wf_pds.append(p_f.sum() / days_f if len(p_f) > 0 else 0.0)
    wf_pass = sum(1 for p in wf_pds if p > 0)

    # OOS stats
    p_oos, d_oos, nf_oos = _run(n_is, len(df))
    oos_pd  = p_oos.sum() / n_days_oos if len(p_oos) > 0 else 0.0
    oos_wr  = (p_oos > 0).mean() if len(p_oos) > 0 else 0.0
    oos_n   = len(p_oos)

    # MC on IS trades
    mc_p = run_mc_pvalue(p_is, n_days_is)

    return {
        "is_pd":    round(is_pd,  2),
        "is_wr":    round(is_wr,  3),
        "is_n":     is_n,
        "wf1":      round(wf_pds[0], 2),
        "wf2":      round(wf_pds[1], 2),
        "wf3":      round(wf_pds[2], 2),
        "wf_pass":  wf_pass,
        "oos_pd":   round(oos_pd, 2),
        "oos_wr":   round(oos_wr, 3),
        "oos_n":    oos_n,
        "mc_p":     round(float(mc_p), 4) if not np.isnan(mc_p) else np.nan,
        "n_filtered": int(nf_is),  # IS filtered entries
    }


# ── IC study: per-entry analysis ──────────────────────────────────────────────

def ic_study(df, sig, pip, sp_gate, n_is, tp, exhaust):
    """
    Compare the trades that WOULD be filtered vs those that PASS.
    Returns dict with WR of filtered vs kept trades.
    """
    mid = (df["close"].values[:n_is] / pip).astype(np.float64)
    bid = (df["bid_c"].values[:n_is] / pip).astype(np.float64)
    ask = (df["ask_c"].values[:n_is] / pip).astype(np.float64)
    sp  = ((df["ask_c"].values[:n_is] - df["bid_c"].values[:n_is]) / pip).astype(np.float64)
    sv  = sig.values[:n_is].astype(np.int8)
    ex  = exhaust[:n_is]

    # What would happen to the "filtered" trades if we let them through?
    # Invert the filter: only trade when exhaust would block
    pnls_blocked = []
    pnls_kept    = []

    # Full baseline simulation tracking which were blocked
    n = len(mid)
    in_t = False; dir_ = 0; ep = 0.0; was_blocked = False

    # Run baseline to get all potential entries
    all_entries = []  # (bar_i, dir, was_blocked_by_filter)
    for i in range(1, n):
        nd = sv[i - 1]
        if not in_t and nd != 0 and sp[i] <= sp_gate:
            blocked = (ex[i - 1] != 0 and ex[i - 1] == nd)
            all_entries.append((i, nd, blocked))
            if not blocked:
                in_t = True; dir_ = nd; ep = ask[i] if nd == 1 else bid[i]
        elif in_t:
            if (mid[i] - ep) * dir_ >= tp:
                in_t = False

    # Simplified IC: run blocked-only and kept-only simulations
    # "Blocked" trades: what would have been their outcome?
    @njit
    def sim_specific(mid, bid, ask, sp, sig, tp, sp_gate, exhaust, take_blocked):
        n   = len(mid); pnls = np.empty(10000, dtype=np.float64); n_t = 0
        in_t = False; dir_ = 0; ep = 0.0
        for i in range(1, n):
            if in_t:
                if (mid[i] - ep) * dir_ >= tp:
                    exit_px = bid[i] if dir_ == 1 else ask[i]
                    if n_t < 10000:
                        pnls[n_t] = (exit_px - ep) * dir_ - sp[i]; n_t += 1
                    in_t = False
            else:
                nd = sig[i - 1]
                if nd != 0 and sp[i] <= sp_gate:
                    blocked = exhaust[i - 1] != 0 and exhaust[i - 1] == nd
                    if blocked == take_blocked:
                        ep = ask[i] if nd == 1 else bid[i]; dir_ = nd; in_t = True
        return pnls[:n_t]

    p_blocked = sim_specific(mid, bid, ask, sp, sv, tp, sp_gate, ex, True)
    p_kept    = sim_specific(mid, bid, ask, sp, sv, tp, sp_gate, ex, False)

    return {
        "n_blocked":   len(p_blocked),
        "n_kept":      len(p_kept),
        "wr_blocked":  round((p_blocked > 0).mean(), 3) if len(p_blocked) > 0 else np.nan,
        "wr_kept":     round((p_kept    > 0).mean(), 3) if len(p_kept)    > 0 else np.nan,
        "pd_blocked":  round(p_blocked.sum() / (n_is / 288), 2) if len(p_blocked) > 0 else np.nan,
        "pd_kept":     round(p_kept.sum()    / (n_is / 288), 2) if len(p_kept)    > 0 else np.nan,
    }


# ── Process one strategy ───────────────────────────────────────────────────────

def process_strategy(label, pairs, build_signal, tp):
    print(f"\n{'═'*70}")
    print(f"Strategy: {label}  TP={tp}p  {len(pairs)} pairs")
    print(f"{'═'*70}")

    # Pre-load all pair data
    pair_data = {}
    for pair in pairs:
        df, pip, n_is, sp_gate = load_pair(pair)
        sig = build_signal(df)
        # Precompute M5 SMA14 and close/open in pips
        close_pip = (df["close"].values / pip).astype(np.float64)
        open_pip  = (df["open"].values  / pip).astype(np.float64)
        sma14_pip = compute_sma_nb(close_pip, EXHAUST_SMA)
        sp_arr    = ((df["ask_c"].values - df["bid_c"].values) / pip).astype(np.float64)
        pair_data[pair] = {
            "df": df, "pip": pip, "n_is": n_is, "sp_gate": sp_gate,
            "sig": sig, "close_pip": close_pip, "open_pip": open_pip,
            "sma14_pip": sma14_pip, "sp_arr": sp_arr,
        }

    all_rows = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n  Computing baseline …")
    baseline_rows = {}
    for pair in pairs:
        pd_ = pair_data[pair]
        s = sim_stats(pd_["df"], pd_["sig"], pd_["pip"], pd_["sp_gate"],
                      pd_["n_is"], tp, exhaust=None)
        s["pair"]     = pair
        s["strategy"] = label
        s["filter"]   = "baseline"
        s["nc"]       = 0
        s["dm"]       = 0.0
        baseline_rows[pair] = s
        all_rows.append(s)

    # Aggregate baseline
    bl_ppd  = sum(r["oos_pd"] for r in baseline_rows.values())
    bl_wf3  = sum(1 for r in baseline_rows.values() if r["wf_pass"] == 3)
    bl_n_is = sum(r["is_n"]   for r in baseline_rows.values())
    print(f"  BASELINE: portfolio OOS p/d={bl_ppd:+.1f}  WF3_pairs={bl_wf3}/{len(pairs)}  IS_trades={bl_n_is}")
    print(f"  {'Pair':>10}  {'OOS p/d':>8}  {'IS WF':>6}  {'OOS WR':>7}  {'MC p':>6}")
    for pair in pairs:
        r = baseline_rows[pair]
        print(f"  {pair:>10}  {r['oos_pd']:>+8.2f}  {r['wf_pass']:>6}/3  "
              f"{r['oos_wr']:>7.1%}  {r['mc_p']:>6.3f}")

    # ── IC study (first filter param only, for signal quality check) ──────────
    print(f"\n  IC study (filter impact on trade quality) …")
    print(f"  {'Pair':>10}  n_consec=2 dist_mult=2.5:")
    print(f"  {'Pair':>10}  {'n_blk':>6}  {'n_kpt':>6}  {'WR_blk':>7}  {'WR_kpt':>7}  "
          f"{'pd_blk':>8}  {'pd_kpt':>8}")
    for pair in pairs:
        pd_ = pair_data[pair]
        exhaust = compute_exhaust_dir(
            pd_["close_pip"], pd_["open_pip"], pd_["sma14_pip"],
            pd_["sp_arr"], np.int64(2), np.float64(2.5), pd_["sp_gate"])
        ic = ic_study(pd_["df"], pd_["sig"], pd_["pip"], pd_["sp_gate"],
                      pd_["n_is"], tp, exhaust)
        print(f"  {pair:>10}  {ic['n_blocked']:>6}  {ic['n_kept']:>6}  "
              f"{ic['wr_blocked']:>7.1%}  {ic['wr_kept']:>7.1%}  "
              f"{ic['pd_blocked']:>+8.2f}  {ic['pd_kept']:>+8.2f}")

    # ── Filtered sweep ────────────────────────────────────────────────────────
    print(f"\n  Filter sweep …")
    filter_rows = []
    for nc in NCONSEC_OPT:
        for dm in DISTMULT_OPT:
            port_oos_pd  = 0.0
            port_wf3     = 0
            port_is_n    = 0
            port_n_filt  = 0

            for pair in pairs:
                pd_ = pair_data[pair]
                exhaust = compute_exhaust_dir(
                    pd_["close_pip"], pd_["open_pip"], pd_["sma14_pip"],
                    pd_["sp_arr"], np.int64(nc), np.float64(dm), pd_["sp_gate"])
                s = sim_stats(pd_["df"], pd_["sig"], pd_["pip"], pd_["sp_gate"],
                              pd_["n_is"], tp, exhaust=exhaust)
                s["pair"]     = pair
                s["strategy"] = label
                s["filter"]   = f"nc{nc}_dm{dm}"
                s["nc"]       = nc
                s["dm"]       = dm
                all_rows.append(s)
                filter_rows.append(s)
                port_oos_pd += s["oos_pd"]
                if s["wf_pass"] == 3: port_wf3 += 1
                port_is_n   += s["is_n"]
                port_n_filt += s["n_filtered"]

            delta_ppd  = port_oos_pd - bl_ppd
            delta_wf3  = port_wf3    - bl_wf3
            filt_pct   = port_n_filt / (bl_n_is + port_n_filt) * 100 if bl_n_is > 0 else 0

            marker = " 🟢" if port_oos_pd > bl_ppd and port_wf3 >= bl_wf3 else (
                     " 🟡" if port_oos_pd > bl_ppd else "")
            print(f"  nc={nc} dm={dm:.1f}: port_OOS={port_oos_pd:+.1f}  "
                  f"Δppd={delta_ppd:+.1f}  WF3={port_wf3}/{len(pairs)}(Δ{delta_wf3:+d})  "
                  f"filt={filt_pct:.0f}%  IS_n={port_is_n}{marker}")

    return all_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Compile Numba
    print("Compiling Numba kernels …", end="", flush=True)
    t0 = time.time()
    _c  = np.ones(500, dtype=np.float64)
    _o  = np.ones(500, dtype=np.float64)
    _sm = compute_sma_nb(_c, 14)
    _ex = compute_exhaust_dir(_c, _o, _sm, _c * 0.1, np.int64(2), np.float64(2.5), np.float64(2.0))
    _m, _b, _a, _s = _c, _c * 0.9999, _c * 1.0001, _c * 0.0002
    _sig = np.zeros(500, dtype=np.int8); _sig[50:100] = 1; _sig[200:250] = -1
    simulate_tp_base(_m, _b, _a, _s, _sig, 20.0, 2.0)
    simulate_tp_filt(_m, _b, _a, _s, _sig, 20.0, 2.0, _ex.astype(np.int8))
    print(f" {time.time()-t0:.1f}s")

    all_rows = []

    all_rows += process_strategy("SMA16",   PAIRS_SMA16, build_sma16_signal, SMA_TP)
    all_rows += process_strategy("PriceMom", PAIRS_PMOM, build_pmom_signal,  PMOM_TP)

    # Save results
    df_out = pd.DataFrame(all_rows)
    out_path = RESULTS / "harvester_filter_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df_out)} rows)")

    # Summary table
    print("\n" + "═" * 70)
    print("SUMMARY — Portfolio OOS p/d by filter config")
    print("═" * 70)
    for strat in ["SMA16", "PriceMom"]:
        sub  = df_out[df_out["strategy"] == strat]
        bl   = sub[sub["filter"] == "baseline"]
        bl_p = bl["oos_pd"].sum()
        bl_w = (bl["wf_pass"] == 3).sum()
        bl_m = bl["mc_p"].mean()
        print(f"\n{strat} (baseline: OOS={bl_p:+.1f}  WF3={bl_w}  MC_avg={bl_m:.3f}):")
        filt = sub[sub["filter"] != "baseline"].groupby("filter").agg(
            oos_pd=("oos_pd","sum"),
            wf3=("wf_pass", lambda x: (x==3).sum()),
            is_n=("is_n","sum"),
            mc_avg=("mc_p","mean"),
            filt_pct=("n_filtered", lambda x: x.sum()),
        ).reset_index()
        # Add baseline is_n for filt%
        bl_is = sub[sub["filter"]=="baseline"]["is_n"].sum()
        filt["filt_pct"] = filt["filt_pct"] / (filt["filt_pct"] + filt["is_n"]) * 100
        filt["delta"]    = filt["oos_pd"] - bl_p
        filt             = filt.sort_values("oos_pd", ascending=False)
        print(f"  {'Filter':>14}  {'OOS p/d':>8}  {'Δ':>6}  WF3  IS_n  filt%  MC_avg")
        for _, r in filt.iterrows():
            mark = " 🟢" if r["oos_pd"] > bl_p else ""
            print(f"  {r['filter']:>14}  {r['oos_pd']:>+8.1f}  {r['delta']:>+6.1f}  "
                  f"{int(r['wf3']):>3}  {int(r['is_n']):>4}  "
                  f"{r['filt_pct']:>4.0f}%  {r['mc_avg']:.3f}{mark}")

    # Best filter recommendation
    print("\n" + "─" * 70)
    for strat in ["SMA16", "PriceMom"]:
        sub = df_out[(df_out["strategy"] == strat) & (df_out["filter"] != "baseline")]
        bl  = df_out[(df_out["strategy"] == strat) & (df_out["filter"] == "baseline")]
        bl_ppd = bl["oos_pd"].sum()

        best_port = sub.groupby("filter")["oos_pd"].sum().idxmax()
        best_ppd  = sub.groupby("filter")["oos_pd"].sum().max()
        delta     = best_ppd - bl_ppd

        print(f"\n{strat}: best filter = '{best_port}', "
              f"OOS {bl_ppd:+.1f} → {best_ppd:+.1f} (Δ{delta:+.1f} p/d)")

        if delta > 0:
            print(f"  🟢 Filter improves portfolio OOS performance")
        else:
            print(f"  🔴 No filter improves portfolio OOS performance")


if __name__ == "__main__":
    main()
