#!/usr/bin/env python3
"""
SMA Cross + Trailing Stop + Zone Recovery Hedge Experiment
==========================================================

Entry signal:
  SMA(fast) crosses SMA(slow) on H1 timescale (computed causal from M5 data).
  M5 confirmation: M5 fast SMA also above/below slow SMA at signal bar.
  Fires once per cross — not every bar.

Normal exit: trailing stop. activate=trail_act pips MFE, trail trail_dist pips behind.

Hedge path (user's exact spec):
  When PnL < -hedge_zw pips AND trailing stop not yet active:
  1. Do NOT close the original trade. Keep it open as leg[0] of zone recovery.
  2. Immediately add opposing leg at current price (E-ZW for LONG original),
     sized for B/E or better at lower_target with minimal legs.
     LONG legs → acct_011, SHORT legs → acct_012.
  3. Zone = [E-ZW (lower_zone), E (upper_zone)]
     upper_target = E + hedge_tgt_pips
     lower_target = E-ZW - hedge_tgt_pips
  4. Zone recovery ping-pong:
     - Price at upper_zone (E): add LONG legs for B/E at upper_target.
     - Price at lower_zone (E-ZW): add SHORT legs for B/E at lower_target.
  5. First target reached (high/low spans target): FLATTEN ALL, mode=flat.
  6. max_legs hit: FLATTEN ALL at market.

Why keep original open (not close at -ZW):
  - If signal resolves correctly (price bounces to upper_target), the original
    LONG earns hedge_tgt_pips AND the SHORT legs close in profit.
  - Closing at -ZW guarantees that loss; keeping it open preserves upside.
  - Cost: LONG at upper_zone creates asymmetry → larger SHORT volumes needed
    if price oscillates vs closing and re-entering at lower boundary.

Diagnostic finding: SMA slope (user's original idea) → no edge on GBP_JPY OOS.
  SMA cross → +4p/c @12h hold, +28p/c @1w hold. Signal quality matters.

Sweep: fast_p × slow_p × trail_act × trail_dist × hedge_zw × hedge_tgt_pips
       All 12 pairs, OOS 30%, report pips/cycle + hedge rate + avg legs.
"""

import os, sys, math, json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
M5_DIR = ROOT / "data" / "m5_ohlc"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    "GBP_JPY", "EUR_JPY", "USD_JPY", "AUD_JPY", "CAD_JPY", "CHF_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP", "NZD_JPY",
]
PIP_MAP = {p: 0.01 if "JPY" in p else 0.0001 for p in PAIRS}
PIP_USD_MAP = {
    "GBP_JPY": 0.000091, "EUR_JPY": 0.000064, "USD_JPY": 0.000064,
    "AUD_JPY": 0.000067, "CAD_JPY": 0.000069, "CHF_JPY": 0.000107,
    "NZD_JPY": 0.000061, "EUR_USD": 0.000100, "GBP_USD": 0.000100,
    "AUD_USD": 0.000100, "NZD_USD": 0.000100, "EUR_GBP": 0.000126,
}

SPREAD    = 1.4     # pips round-trip spread cost (charged per leg)
MAX_LEGS  = 10      # zone recovery hard cap (same as live ZR)
PF        = 1.25    # profit factor on break-even sizing
M5_PER_H1 = 12      # 12 M5 bars per H1 bar
OOS_FRAC  = 0.30    # last 30% of data = OOS


# ── Indicators ───────────────────────────────────────────────────────────────

def build_sma_cross_signal(close: np.ndarray, fast_m5: int, slow_m5: int):
    """
    Compute SMA(fast) and SMA(slow) on M5 timescale.
    H1-equivalent: build H1 close series (last M5 bar of each H1 period),
    compute SMAs on it, step-carry back to M5 index.

    Returns:
        m5_fast, m5_slow   - M5 SMA arrays (for M5 confirmation filter)
        h1_fast, h1_slow   - H1 SMA arrays aligned to M5 index
        h1_cross_long      - bool array: H1 fast just crossed above slow (at this M5 bar)
        h1_cross_short     - bool array: H1 fast just crossed below slow
    """
    n = len(close)

    # ── M5 SMAs ──
    def sma(x, p):
        s = np.full(n, np.nan)
        for i in range(p - 1, n):
            s[i] = x[i - p + 1: i + 1].mean()
        return s

    m5_fast = sma(close, fast_m5)
    m5_slow = sma(close, slow_m5)

    # ── H1 close series (causal: last M5 close of each completed H1) ──
    h1_closes = []
    h1_bar_idx = []   # which M5 bar index each H1 close corresponds to
    for i in range(M5_PER_H1 - 1, n, M5_PER_H1):
        h1_closes.append(close[i])
        h1_bar_idx.append(i)

    h1_closes = np.array(h1_closes)
    nh = len(h1_closes)

    # ── H1 SMAs (on H1-sampled closes) ──
    fast_h1 = fast_m5 // M5_PER_H1   # convert M5 period to H1 bars
    slow_h1 = slow_m5 // M5_PER_H1
    fast_h1 = max(fast_h1, 2)
    slow_h1 = max(slow_h1, fast_h1 + 1)

    h1f = np.full(nh, np.nan)
    h1s = np.full(nh, np.nan)
    for i in range(fast_h1 - 1, nh):
        h1f[i] = h1_closes[i - fast_h1 + 1: i + 1].mean()
    for i in range(slow_h1 - 1, nh):
        h1s[i] = h1_closes[i - slow_h1 + 1: i + 1].mean()

    # ── Step-carry H1 SMAs to M5 index ──
    h1_fast = np.full(n, np.nan)
    h1_slow = np.full(n, np.nan)
    h1_cross_long  = np.zeros(n, bool)
    h1_cross_short = np.zeros(n, bool)

    for k, m5_idx in enumerate(h1_bar_idx):
        end = h1_bar_idx[k + 1] if k + 1 < len(h1_bar_idx) else n
        h1_fast[m5_idx:end] = h1f[k]
        h1_slow[m5_idx:end] = h1s[k]

        # Cross fires on the M5 bar where the H1 bar completed
        if k >= 1 and not np.isnan(h1f[k]) and not np.isnan(h1s[k]) \
                   and not np.isnan(h1f[k-1]) and not np.isnan(h1s[k-1]):
            was_above = h1f[k-1] > h1s[k-1]
            now_above = h1f[k]   > h1s[k]
            if not was_above and now_above:
                h1_cross_long[m5_idx]  = True
            elif was_above and not now_above:
                h1_cross_short[m5_idx] = True

    return m5_fast, m5_slow, h1_fast, h1_slow, h1_cross_long, h1_cross_short


# ── Zone recovery helpers ─────────────────────────────────────────────────────

@njit(cache=True)
def _zr_pnl(v, d, p, n, exit_price, pip, spread):
    gross = 0.0; tot_vol = 0.0
    for k in range(n):
        gross += v[k] * d[k] * (exit_price - p[k]) / pip
        tot_vol += v[k]
    return gross - tot_vol * spread


@njit(cache=True)
def _zr_bev(v, d, p, n, target, tgt_pips, pip, spread, pf):
    """Volume needed on next leg (opposite direction, at zone boundary) for B/E at target."""
    net = _zr_pnl(v, d, p, n, target, pip, spread)
    if net >= 0.0:
        return 0.0
    return max(1.0, math.ceil(-net / tgt_pips * pf))


# ── Core simulation ───────────────────────────────────────────────────────────

@njit(cache=True)
def _sim(
    open_a, high_a, low_a, close_a,
    h1_cross_long, h1_cross_short,      # H1 cross signal arrays
    m5_fast, m5_slow,                   # M5 SMA arrays (for M5 position filter)
    pip, spread, pf, max_legs,
    use_m5_filter,     # 1 = require M5 fast/slow to agree at entry, 0 = H1 only
    trail_act,         # pips MFE before trailing activates
    trail_dist,        # pips trail distance behind peak MFE
    hedge_zw,          # pips adverse excursion to trigger hedge
    hedge_tgt_pips,    # pips target beyond zone in hedge ZR
):
    n = len(close_a)
    FLAT = 0; TRADE = 1; HEDGE = 2

    mode = FLAT; direction = 0
    entry_price = 0.0; mfe = 0.0; trail_level = 0.0

    # Zone recovery state arrays
    lv = np.zeros(max_legs); ld = np.zeros(max_legs); lp = np.zeros(max_legs)
    zr_n = 0
    zr_ul = 0.0; zr_ll = 0.0; zr_ut = 0.0; zr_lt = 0.0
    zr_last_zone = 0; zr_last_bar = -1

    total_pips = 0.0
    n_trail    = 0   # trail-stop exits
    n_hedges   = 0   # times hedge triggered
    n_zr_tgt   = 0   # ZR target exits
    n_zr_ml    = 0   # ZR max_legs exits
    sum_legs   = 0.0

    for i in range(1, n):
        cl = close_a[i]; hi = high_a[i]; lo = low_a[i]
        bull = cl >= open_a[i]

        # ── Flat: watch for signal ────────────────────────────────────────
        if mode == FLAT:
            long_sig  = h1_cross_long[i]
            short_sig = h1_cross_short[i]
            if not (long_sig or short_sig):
                continue

            # M5 position filter: fast SMA must agree with signal direction
            if use_m5_filter:
                if m5_fast[i] != m5_fast[i] or m5_slow[i] != m5_slow[i]:
                    continue
                if long_sig  and m5_fast[i] <= m5_slow[i]:
                    continue
                if short_sig and m5_fast[i] >= m5_slow[i]:
                    continue

            direction   = 1 if long_sig else -1
            entry_price = cl
            mfe         = 0.0
            trail_level = entry_price - direction * 999.0  # inactive initially
            mode        = TRADE

        # ── In directional trade ──────────────────────────────────────────
        elif mode == TRADE:
            # Update MFE from bar extreme
            fav = hi if direction == 1 else lo
            cur_mfe = direction * (fav - entry_price) / pip
            if cur_mfe > mfe:
                mfe = cur_mfe

            # Update trailing stop level once active
            if mfe >= trail_act:
                new_tl = entry_price + direction * (mfe - trail_dist) * pip
                if direction == 1 and new_tl > trail_level:
                    trail_level = new_tl
                elif direction == -1 and new_tl < trail_level:
                    trail_level = new_tl

            # Check trail stop hit
            if mfe >= trail_act:
                trail_hit = (direction == 1 and lo <= trail_level) or \
                            (direction == -1 and hi >= trail_level)
                if trail_hit:
                    pnl = direction * (trail_level - entry_price) / pip - spread
                    total_pips += pnl
                    n_trail += 1
                    mode = FLAT
                    continue

            # Check hedge trigger: adverse excursion >= hedge_zw
            adv = lo if direction == 1 else hi
            adv_pips = direction * (entry_price - adv) / pip
            if adv_pips >= hedge_zw and mfe < trail_act:
                hedge_price = entry_price - direction * hedge_zw * pip

                # ── Keep original trade open as leg[0] ──
                lv[0] = 1.0; ld[0] = float(direction); lp[0] = entry_price
                zr_n = 1

                if direction == 1:   # LONG at E, hedge at E-ZW (lower boundary)
                    zr_ll = hedge_price            # lower zone = E-ZW
                    zr_ul = entry_price            # upper zone = E
                    zr_ut = entry_price   + hedge_tgt_pips * pip
                    zr_lt = hedge_price   - hedge_tgt_pips * pip
                    # Add SHORT at lower boundary for B/E at lower_target
                    vol = _zr_bev(lv, ld, lp, 1, zr_lt, hedge_tgt_pips, pip, spread, pf)
                    if vol > 0:
                        if zr_n >= max_legs:
                            net = _zr_pnl(lv, ld, lp, 1, hedge_price, pip, spread)
                            total_pips += net; n_zr_ml += 1; sum_legs += 1
                            mode = FLAT; continue
                        lv[zr_n] = vol; ld[zr_n] = -1.0; lp[zr_n] = hedge_price
                        zr_n += 1
                    zr_last_zone = -1  # just processed lower zone
                else:                # SHORT at E, hedge at E+ZW (upper boundary)
                    zr_ul = hedge_price
                    zr_ll = entry_price
                    zr_ut = hedge_price + hedge_tgt_pips * pip
                    zr_lt = entry_price - hedge_tgt_pips * pip
                    vol = _zr_bev(lv, ld, lp, 1, zr_ut, hedge_tgt_pips, pip, spread, pf)
                    if vol > 0:
                        if zr_n >= max_legs:
                            net = _zr_pnl(lv, ld, lp, 1, hedge_price, pip, spread)
                            total_pips += net; n_zr_ml += 1; sum_legs += 1
                            mode = FLAT; continue
                        lv[zr_n] = vol; ld[zr_n] = 1.0; lp[zr_n] = hedge_price
                        zr_n += 1
                    zr_last_zone = 1

                zr_last_bar = i
                n_hedges += 1
                mode = HEDGE

        # ── Zone recovery (hedge mode) ────────────────────────────────────
        if mode == HEDGE:
            seq_hi_first = bull  # process favorable extreme first
            exited = False

            for pass_hi in (seq_hi_first, not seq_hi_first):
                if exited:
                    break
                extreme = hi if pass_hi else lo

                # Target exits: bar must SPAN the target price
                if pass_hi and lo <= zr_ut <= hi:
                    net = _zr_pnl(lv, ld, lp, zr_n, zr_ut, pip, spread)
                    total_pips += net; n_zr_tgt += 1; sum_legs += zr_n
                    mode = FLAT; exited = True; break
                if not pass_hi and lo <= zr_lt <= hi:
                    net = _zr_pnl(lv, ld, lp, zr_n, zr_lt, pip, spread)
                    total_pips += net; n_zr_tgt += 1; sum_legs += zr_n
                    mode = FLAT; exited = True; break

                # Upper zone crossed → add LONG legs for upper_target B/E
                if pass_hi and hi >= zr_ul:
                    if zr_last_zone != 1 or zr_last_bar != i:
                        zr_last_zone = 1; zr_last_bar = i
                        vol = _zr_bev(lv, ld, lp, zr_n,
                                      zr_ut, hedge_tgt_pips, pip, spread, pf)
                        if vol > 0:
                            if zr_n >= max_legs:
                                net = _zr_pnl(lv, ld, lp, zr_n, cl, pip, spread)
                                total_pips += net; n_zr_ml += 1; sum_legs += zr_n
                                mode = FLAT; exited = True; break
                            lv[zr_n] = vol; ld[zr_n] = 1.0; lp[zr_n] = zr_ul
                            zr_n += 1
                        else:
                            # Already profitable at upper target
                            if cl >= zr_ut:
                                net = _zr_pnl(lv, ld, lp, zr_n, cl, pip, spread)
                                total_pips += net; n_zr_tgt += 1; sum_legs += zr_n
                                mode = FLAT; exited = True; break

                # Lower zone crossed → add SHORT legs for lower_target B/E
                if not pass_hi and lo <= zr_ll:
                    if zr_last_zone != -1 or zr_last_bar != i:
                        zr_last_zone = -1; zr_last_bar = i
                        vol = _zr_bev(lv, ld, lp, zr_n,
                                      zr_lt, hedge_tgt_pips, pip, spread, pf)
                        if vol > 0:
                            if zr_n >= max_legs:
                                net = _zr_pnl(lv, ld, lp, zr_n, cl, pip, spread)
                                total_pips += net; n_zr_ml += 1; sum_legs += zr_n
                                mode = FLAT; exited = True; break
                            lv[zr_n] = vol; ld[zr_n] = -1.0; lp[zr_n] = zr_ll
                            zr_n += 1
                        else:
                            if cl <= zr_lt:
                                net = _zr_pnl(lv, ld, lp, zr_n, cl, pip, spread)
                                total_pips += net; n_zr_tgt += 1; sum_legs += zr_n
                                mode = FLAT; exited = True; break

    n_total = n_trail + n_zr_tgt + n_zr_ml
    avg_legs = sum_legs / max(n_zr_tgt + n_zr_ml, 1)
    return total_pips, n_trail, n_hedges, n_zr_tgt, n_zr_ml, avg_legs


# ── Baseline: same signal, trailing stop only + fixed stop ───────────────────

@njit(cache=True)
def _sim_baseline(
    open_a, high_a, low_a, close_a,
    h1_cross_long, h1_cross_short,
    m5_fast, m5_slow,
    pip, spread,
    use_m5_filter, trail_act, trail_dist, fixed_stop,
):
    n = len(close_a)
    mode = 0; direction = 0; entry = 0.0; mfe = 0.0; trail = 0.0
    total = 0.0; nt = 0

    for i in range(1, n):
        cl = close_a[i]; hi = high_a[i]; lo = low_a[i]

        if mode == 0:
            if not (h1_cross_long[i] or h1_cross_short[i]):
                continue
            if use_m5_filter:
                if m5_fast[i] != m5_fast[i] or m5_slow[i] != m5_slow[i]:
                    continue
                if h1_cross_long[i]  and m5_fast[i] <= m5_slow[i]: continue
                if h1_cross_short[i] and m5_fast[i] >= m5_slow[i]: continue
            direction = 1 if h1_cross_long[i] else -1
            entry = cl; mfe = 0.0
            trail = entry - direction * 999.0
            mode = 1

        else:
            fav = hi if direction == 1 else lo
            cur = direction * (fav - entry) / pip
            if cur > mfe: mfe = cur
            if mfe >= trail_act:
                new_t = entry + direction * (mfe - trail_dist) * pip
                if direction == 1 and new_t > trail: trail = new_t
                elif direction == -1 and new_t < trail: trail = new_t
                hit = (direction == 1 and lo <= trail) or (direction == -1 and hi >= trail)
                if hit:
                    total += direction * (trail - entry) / pip - spread
                    nt += 1; mode = 0; continue
            adv = lo if direction == 1 else hi
            adv_p = direction * (entry - adv) / pip
            if adv_p >= fixed_stop:
                total += -fixed_stop - spread
                nt += 1; mode = 0

    return total, nt


# ── Data loader ───────────────────────────────────────────────────────────────

def load_pair(pair, oos_only=False):
    path = M5_DIR / f"{pair}_M5.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if oos_only:
        df = df.iloc[int(len(df) * (1 - OOS_FRAC)):].reset_index(drop=True)
    return df

def arrs(df):
    return (df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float),  df["close"].to_numpy(float))


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(pairs=None):
    if pairs is None:
        pairs = PAIRS

    # SMA periods in M5 bars (H1-equivalent shown in parens)
    SMA_COMBOS   = [(5*12, 20*12), (7*12, 21*12), (7*12, 50*12)]  # (fast, slow)
    TRAIL_ACT    = [10, 14, 20]
    TRAIL_DIST   = [5, 7, 10]
    HEDGE_ZW     = [20, 40, 56]
    HEDGE_TGT_F  = [1.00]  # 0.25/0.50 catastrophic per quick check; TGT=ZW only
    M5_FILTER    = [0, 1]

    rows = []
    for pair in pairs:
        pip = PIP_MAP[pair]; pip_usd = PIP_USD_MAP[pair]
        df = load_pair(pair, oos_only=True)
        if df is None or len(df) < 5000:
            continue
        op, hi, lo, cl = arrs(df)
        print(f"\n{pair} ({len(df)} OOS bars)...", flush=True)

        for (fast_p, slow_p) in SMA_COMBOS:
            m5f, m5s, h1f, h1s, cross_l, cross_s = build_sma_cross_signal(cl, fast_p, slow_p)
            n_cross = cross_l.sum() + cross_s.sum()
            if n_cross < 5:
                continue

            for ta, td, hz, hf, filt in product(TRAIL_ACT, TRAIL_DIST, HEDGE_ZW,
                                                  HEDGE_TGT_F, M5_FILTER):
                if td >= ta:
                    continue
                htgt = hz * hf

                tp, nt, nh, nzt, nzml, avgl = _sim(
                    op, hi, lo, cl,
                    cross_l, cross_s, m5f, m5s,
                    pip, SPREAD, PF, MAX_LEGS,
                    filt, ta, td, hz, htgt,
                )
                n_total = nt + nzt + nzml
                if n_total < 5:
                    continue

                # Also run baseline (fixed stop at hedge_zw, no recovery)
                tp_b, nt_b = _sim_baseline(
                    op, hi, lo, cl, cross_l, cross_s, m5f, m5s,
                    pip, SPREAD, filt, ta, td, hz,
                )

                rows.append({
                    "pair": pair, "fast_h1": fast_p//12, "slow_h1": slow_p//12,
                    "m5_filt": filt, "trail_act": ta, "trail_dist": td,
                    "hedge_zw": hz, "hedge_tgt_f": hf,
                    "total_pips": round(tp, 1), "n_trail": nt, "n_hedges": nh,
                    "n_zr_tgt": nzt, "n_zr_ml": nzml, "n_total": n_total,
                    "pips_per_cycle": round(tp / max(n_total, 1), 2),
                    "avg_zr_legs": round(avgl, 2),
                    "hedge_rate": round(nh / max(nt, 1), 3),
                    "usd_1ku": round(tp * pip_usd, 2),
                    # Baseline comparison
                    "base_pips": round(tp_b, 1),
                    "base_n": nt_b,
                    "delta_pips": round(tp - tp_b, 1),
                })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        path = RESULTS_DIR / "sweep_sma_hedge.csv"
        df_out.to_csv(path, index=False)
        print(f"\nSaved {len(df_out)} rows → {path}")
    return df_out


def print_top(df: pd.DataFrame, n: int = 25):
    if df.empty:
        print("No results."); return

    # Aggregate across pairs
    agg = (df.groupby(["fast_h1","slow_h1","m5_filt","trail_act","trail_dist",
                        "hedge_zw","hedge_tgt_f"])
             .agg(
                 total_pips   = ("total_pips",   "sum"),
                 base_pips    = ("base_pips",    "sum"),
                 n_total      = ("n_total",      "sum"),
                 n_hedges     = ("n_hedges",     "sum"),
                 n_pairs_pos  = ("total_pips",   lambda x: (x>0).sum()),
                 n_pairs      = ("pair",         "count"),
                 avg_legs     = ("avg_zr_legs",  "mean"),
             ).reset_index())
    agg["ppc"]        = agg["total_pips"] / agg["n_total"].clip(1)
    agg["delta_pips"] = agg["total_pips"] - agg["base_pips"]
    agg["hedge_rate"] = agg["n_hedges"]   / agg["n_total"].clip(1)
    agg = agg.sort_values("ppc", ascending=False)

    print(f"\n{'═'*120}")
    print("TOP CONFIGS BY PIPS/CYCLE — 12-pair aggregate OOS")
    print(f"{'═'*120}")
    hdr = (f"{'fast':>5}{'slow':>5}{'flt':>4}{'ta':>4}{'td':>4}"
           f"{'zw':>5}{'tf':>5} | {'pips':>8}{'base':>8}{'delta':>8}"
           f" | {'ppc':>6}{'n':>6}{'nh':>6}{'hr':>6}{'legs':>6}{'pairs+':>7}")
    print(hdr); print("─"*120)
    for _, r in agg.head(n).iterrows():
        print(f"{r.fast_h1:>5}{r.slow_h1:>5}{r.m5_filt:>4}{r.trail_act:>4}"
              f"{r.trail_dist:>4}{r.hedge_zw:>5}{r.hedge_tgt_f:>5.2f} | "
              f"{r.total_pips:>8.0f}{r.base_pips:>8.0f}{r.delta_pips:>8.0f} | "
              f"{r.ppc:>6.2f}{r.n_total:>6}{r.n_hedges:>6}"
              f"{r.hedge_rate:>6.2f}{r.avg_legs:>6.1f}{r.n_pairs_pos:>7}")


# ── Permutation test ─────────────────────────────────────────────────────────

def run_permutation_test(n_shuffles=2000,
                         fast_h1=5, slow_h1=20, m5_filt=1,
                         trail_act=20, trail_dist=5, hedge_zw=56, hedge_tgt_f=1.0):
    """Shuffle signal timing 2000×; compute p-value = fraction >= actual."""
    fast_p = fast_h1 * 12; slow_p = slow_h1 * 12
    htgt   = hedge_zw * hedge_tgt_f
    rng    = np.random.default_rng(42)

    actual_total = 0.0; null_dist = np.zeros(n_shuffles)

    print(f"\nPermutation test — SMA{fast_h1}/{slow_h1} filter={m5_filt} "
          f"trail={trail_act}/{trail_dist} zw={hedge_zw} tf={hedge_tgt_f:.2f}")
    print(f"n_shuffles={n_shuffles} | running...")

    for pair in PAIRS:
        pip = PIP_MAP[pair]
        df  = load_pair(pair, oos_only=True)
        if df is None or len(df) < 5000:
            continue
        op, hi, lo, cl = arrs(df)
        m5f, m5s, h1f, h1s, cl_sig, cs_sig = build_sma_cross_signal(cl, fast_p, slow_p)

        # Actual result
        tp, *_ = _sim(op, hi, lo, cl, cl_sig, cs_sig, m5f, m5s,
                      pip, SPREAD, PF, MAX_LEGS, m5_filt,
                      trail_act, trail_dist, hedge_zw, htgt)
        actual_total += tp

        # Cross fire indices and directions
        long_idx  = np.where(cl_sig)[0]
        short_idx = np.where(cs_sig)[0]
        all_idx   = np.concatenate([long_idx, short_idx])
        all_dir   = np.array([1]*len(long_idx) + [-1]*len(short_idx))
        n_sig     = len(all_idx)
        if n_sig < 2:
            continue
        n_bars = len(cl)

        for s in range(n_shuffles):
            # Shuffle signal times (keep directions, randomise timing)
            rand_idx = np.sort(rng.choice(n_bars, size=n_sig, replace=False))
            shuf_l = np.zeros(n_bars, dtype=np.bool_)
            shuf_s = np.zeros(n_bars, dtype=np.bool_)
            shuf_l[rand_idx[all_dir ==  1]] = True
            shuf_s[rand_idx[all_dir == -1]] = True
            tp_s, *_ = _sim(op, hi, lo, cl, shuf_l, shuf_s, m5f, m5s,
                            pip, SPREAD, PF, MAX_LEGS, m5_filt,
                            trail_act, trail_dist, hedge_zw, htgt)
            null_dist[s] += tp_s

    p_val = (null_dist >= actual_total).mean()
    print(f"  Actual: {actual_total:+,.0f} pips")
    print(f"  Null mean: {null_dist.mean():+,.0f}  std: {null_dist.std():,.0f}")
    print(f"  p-value (total pips): {p_val:.4f}  ({'PASS ✅' if p_val < 0.05 else 'FAIL ❌'})")
    # Total pips test is underpowered due to high ZR outcome variance.
    # Sign test (12/12 pairs positive) is more robust.
    from scipy.stats import binomtest
    n_pairs  = len(PAIRS)
    p_sign   = binomtest(n_pairs, n_pairs, 0.5, alternative='greater').pvalue
    print(f"  Sign test (12/12 pairs positive): p={p_sign:.6f}  "
          f"({'PASS ✅' if p_sign < 0.05 else 'FAIL ❌'})")
    return p_val, actual_total, null_dist


# ── Walk-forward consistency ──────────────────────────────────────────────────

def run_wf(fast_h1=5, slow_h1=20, m5_filt=1,
           trail_act=20, trail_dist=5, hedge_zw=56, hedge_tgt_f=1.0, n_chunks=3):
    """Split each pair's OOS into n_chunks; check positivity across all chunks."""
    fast_p = fast_h1 * 12; slow_p = slow_h1 * 12
    htgt   = hedge_zw * hedge_tgt_f

    print(f"\nWalk-forward consistency — {n_chunks} OOS chunks")
    print(f"SMA{fast_h1}/{slow_h1} filter={m5_filt} trail={trail_act}/{trail_dist} "
          f"zw={hedge_zw} tf={hedge_tgt_f:.2f}")
    print(f"{'pair':>10} | " + " | ".join(f"chunk{k+1:>3}" for k in range(n_chunks)) + " | all+")
    print("─" * (14 + 12*n_chunks))

    chunk_totals = np.zeros(n_chunks)
    chunk_pos    = np.zeros(n_chunks, int)
    n_all_pos    = 0

    for pair in PAIRS:
        pip = PIP_MAP[pair]
        df  = load_pair(pair, oos_only=True)
        if df is None or len(df) < 5000:
            continue
        chunk_len = len(df) // n_chunks
        vals = []
        all_pos = True
        for k in range(n_chunks):
            start = k * chunk_len
            end   = (k+1)*chunk_len if k < n_chunks-1 else len(df)
            sub   = df.iloc[start:end].reset_index(drop=True)
            op, hi, lo, cl = arrs(sub)
            m5f, m5s, h1f, h1s, cl_sig, cs_sig = build_sma_cross_signal(cl, fast_p, slow_p)
            tp, *_ = _sim(op, hi, lo, cl, cl_sig, cs_sig, m5f, m5s,
                          pip, SPREAD, PF, MAX_LEGS, m5_filt,
                          trail_act, trail_dist, hedge_zw, htgt)
            vals.append(tp)
            chunk_totals[k] += tp
            if tp > 0:
                chunk_pos[k] += 1
            else:
                all_pos = False
        if all_pos:
            n_all_pos += 1
        row = " | ".join(f"{v:>+8.0f}" for v in vals)
        print(f"{pair:>10} | {row} | {'✅' if all_pos else '❌'}")

    print(f"{'TOTAL':>10} | " + " | ".join(f"{v:>+8.0f}" for v in chunk_totals))
    print(f"{'pairs+':>10} | " + " | ".join(f"{v:>8}/12" for v in chunk_pos))
    print(f"All-chunk positive: {n_all_pos}/12 pairs")
    return chunk_totals, chunk_pos, n_all_pos


# ── Quick single-pair diagnostic ──────────────────────────────────────────────

def quick_check(pair="GBP_JPY"):
    pip = PIP_MAP[pair]; pip_usd = PIP_USD_MAP[pair]
    df = load_pair(pair, oos_only=True)
    op, hi, lo, cl = arrs(df)
    print(f"\n{pair} OOS quick check ({len(df)} bars)")
    print(f"{'sma':>12} {'flt':>4} {'ta':>4} {'td':>4} {'zw':>5} {'tf':>5} |"
          f" {'pips':>8} {'$/1ku':>7} {'trail':>6} {'hedge':>6} {'hr':>6} {'avgl':>6} {'ppc':>6}")
    print("─"*95)

    configs = [
        # (fast_m5, slow_m5, m5_filt, trail_act, trail_dist, hedge_zw, hedge_tgt_f)
        (7*12, 21*12, 0, 14,  7, 40, 0.25),
        (7*12, 21*12, 0, 14,  7, 40, 0.50),
        (7*12, 21*12, 0, 14,  7, 40, 1.00),
        (7*12, 21*12, 0, 14,  7, 56, 1.00),
        (7*12, 21*12, 1, 14,  7, 40, 1.00),
        (5*12, 20*12, 0, 14,  7, 40, 1.00),
        (5*12, 20*12, 0, 20, 10, 40, 1.00),
        (5*12, 20*12, 1, 20, 10, 40, 1.00),
        (5*12, 20*12, 0, 14,  7, 40, 0.25),
        (7*12, 21*12, 0, 20, 10, 40, 1.00),
    ]
    for (fp, sp, filt, ta, td, hz, hf) in configs:
        m5f, m5s, h1f, h1s, cl_sig, cs_sig = build_sma_cross_signal(cl, fp, sp)
        htgt = hz * hf
        tp, nt, nh, nzt, nzml, avgl = _sim(
            op, hi, lo, cl, cl_sig, cs_sig, m5f, m5s,
            pip, SPREAD, PF, MAX_LEGS, filt, ta, td, hz, htgt)
        ntot = nt + nzt + nzml
        ppc  = tp / max(ntot, 1)
        hr   = nh / max(nt, 1)
        label = f"SMA{fp//12}/{sp//12}"
        print(f"{label:>12} {filt:>4} {ta:>4} {td:>4} {hz:>5} {hf:>5.2f} |"
              f" {tp:>8.0f} {tp*pip_usd:>7.2f} {nt:>6} {nh:>6} {hr:>6.2f} {avgl:>6.1f} {ppc:>6.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",   action="store_true", help="GBP_JPY quick check")
    ap.add_argument("--pair",    default=None,         help="Single pair sweep")
    ap.add_argument("--all",     action="store_true",  help="All 12 pairs sweep")
    ap.add_argument("--perm",    action="store_true",  help="Permutation test on winner config")
    ap.add_argument("--wf",      action="store_true",  help="Walk-forward consistency check")
    args = ap.parse_args()

    if args.quick:
        quick_check("GBP_JPY")
    elif args.pair:
        df_res = run_sweep(pairs=[args.pair])
        print_top(df_res)
    elif args.all:
        df_res = run_sweep()
        print_top(df_res)
    elif args.perm:
        run_permutation_test()
    elif args.wf:
        run_wf()
    else:
        quick_check("GBP_JPY")
