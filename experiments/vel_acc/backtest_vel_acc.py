"""
Velocity-Acceleration Confluence Strategy — M5 Backtest Sweep
==============================================================
Entry: fast_vel > vel_thresh AND fast_acc > 0 AND slow_vel agrees (three-way confluence)
Exit:  fast_acc < acc_thresh  OR  hold >= max_hold  OR  SL bar-close fill

Design motivation:
  TR momentum failed because trailing stops fire intra-bar (when lo <= trail),
  but live fills at bar close → 3-7 pip gap kills edge. This strategy exits on
  ACCELERATION TURNING NEGATIVE, detected at bar close, exiting at that same
  bar close. Fill = signal price. Zero intra-bar gap by construction.

Velocity:    vel_fast(t) = (close[t] - close[t-N]) / (N * pip)   [pips/bar]
Acceleration: acc_fast(t) = vel_fast(t) - vel_fast(t-N)           [pips]
Confluence:  slow vel context must agree with fast direction.

SOP compliance:
  R1: Closed bars only (all decisions at bar close)
  R3: Mid price + spread deducted
  R5: sp_gate = IS P90 spread, hardcoded, never recomputed from OOS
  R6: Same logic in backtest and live (no divergence)
  R8: OOS evaluated exactly once after WF gate passes
"""

import gc
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "m5_ba"

PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
    "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "NZD_JPY", "CHF_JPY", "EUR_GBP",
]
PIP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
    "USD_JPY": 0.01,   "EUR_JPY": 0.01,   "GBP_JPY": 0.01,   "AUD_JPY": 0.01,
    "CAD_JPY": 0.01,   "NZD_JPY": 0.01,   "CHF_JPY": 0.01,   "EUR_GBP": 0.0001,
}

IS_FRAC              = 0.70
N_WF                 = 3
N_MC                 = 300
MIN_TRADES_PER_CHUNK = 5
BARS_PER_DAY         = 288.0   # M5 bars per trading day

# Sweep parameters
FAST_WINS  = [1, 2, 3]          # M5 bars — fast velocity window
SLOW_WINS  = [5, 8, 12, 24]     # M5 bars — slow context window (must > fast)
VEL_THRESH = [0.0, 0.5, 1.0]    # pips — fast velocity minimum at entry
ACC_THRESH = [0.0, -0.5]        # pips — exit when fast_acc < this (0 = immediate)
MAX_HOLDS  = [6, 12, 24]        # M5 bars — safety time exit
SL_PIPS    = [0, 5, 8]          # pips — hard SL bar-close fill (0 = disabled)
DIRECTIONS = [1, -1]            # 1=long, -1=short


# ── Config builder ─────────────────────────────────────────────────────────────

def build_configs():
    rows = []
    for fw, sw, vt, at, mh, sl, d in product(
        FAST_WINS, SLOW_WINS, VEL_THRESH, ACC_THRESH, MAX_HOLDS, SL_PIPS, DIRECTIONS,
    ):
        if sw <= fw:    # slow must be strictly larger than fast
            continue
        rows.append((float(fw), float(sw), float(vt), float(at),
                     float(mh), float(sl), float(d)))
    return np.array(rows, dtype=np.float64)


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit
def _sim_pnl(close, hi, lo, spread, pip,
             fn, sn, vel_thresh, acc_thresh, mh, sl, di, sp_gate,
             start, end):
    """Simulate one config over [start, end). Returns (total_pnl, n_trades).
    All exits at bar close (R1). Spread deducted at exit bar (R3)."""
    warmup   = 2 * fn + sn
    in_trade = False
    entry_px = 0.0
    hold     = 0
    total_pnl = 0.0
    n_trades  = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        h  = hi[t]
        l  = lo[t]
        sp = spread[t] / pip   # spread in pips at this bar

        if in_trade:
            hold += 1
            # Velocity and acceleration at bar close (causal — uses only past closes)
            vel_cur = (c - close[t - fn]) / (fn * pip)
            vel_prv = (close[t - fn] - close[t - 2*fn]) / (fn * pip)
            acc     = vel_cur - vel_prv  # pips

            should_exit = False

            # (1) SL: bar-close fill when bar's low/high crosses SL level
            if sl > 0.0:
                if di == 1  and l <= entry_px - sl * pip:
                    should_exit = True
                elif di == -1 and h >= entry_px + sl * pip:
                    should_exit = True

            # (2) Acceleration exit: momentum exhausting
            if di == 1  and acc < acc_thresh:
                should_exit = True
            elif di == -1 and acc > -acc_thresh:
                should_exit = True

            # (3) Max hold safety
            if hold >= mh:
                should_exit = True

            if should_exit:
                # Bar-close fill — matches live market-order execution
                pnl        = di * (c - entry_px) / pip - sp
                total_pnl += pnl
                n_trades  += 1
                in_trade   = False

        else:
            # Entry check (all at bar close, SOP R1)
            if sp > sp_gate:
                continue

            vel_f = (c - close[t - fn]) / (fn * pip)
            vel_p = (close[t - fn] - close[t - 2*fn]) / (fn * pip)
            acc_f = vel_f - vel_p
            vel_s = (c - close[t - sn]) / (sn * pip)   # slow context

            if di == 1:
                enter = (vel_f > vel_thresh and acc_f > 0.0 and vel_s > 0.0)
            else:   # short
                enter = (vel_f < -vel_thresh and acc_f < 0.0 and vel_s < 0.0)

            if enter:
                entry_px = c
                in_trade = True
                hold     = 0

    # Force-close any open trade at boundary (IS→OOS handoff)
    if in_trade and end > 0:
        sp        = spread[end - 1] / pip
        pnl       = di * (close[end - 1] - entry_px) / pip - sp
        total_pnl += pnl
        n_trades  += 1

    return total_pnl, n_trades


@njit(parallel=True)
def wf_sweep(close, hi, lo, spread, pip, configs, sp_gate, chunk_ends):
    """Parallel WF sweep. Returns wf_pnl[n_cfg, N_WF], wf_trades[n_cfg, N_WF]."""
    n_cfg  = len(configs)
    nc     = len(chunk_ends)
    wf_pnl    = np.zeros((n_cfg, nc))
    wf_trades = np.zeros((n_cfg, nc))

    for ci in prange(n_cfg):
        fn  = int(configs[ci, 0])
        sn  = int(configs[ci, 1])
        vt  = configs[ci, 2]
        at  = configs[ci, 3]
        mh  = int(configs[ci, 4])
        sl  = configs[ci, 5]
        di  = int(configs[ci, 6])

        s = 0
        for k in range(nc):
            e     = int(chunk_ends[k])
            pnl, nt = _sim_pnl(close, hi, lo, spread, pip, fn, sn, vt, at, mh, sl, di, sp_gate, s, e)
            wf_pnl[ci, k]    = pnl
            wf_trades[ci, k] = nt
            s = e

    return wf_pnl, wf_trades


@njit
def sim_collect(close, hi, lo, spread, pip,
                fn, sn, vel_thresh, acc_thresh, mh, sl, di, sp_gate,
                start, end):
    """Collect individual trade P/Ls for OOS stats and MC validation."""
    warmup  = 2 * fn + sn
    max_t   = (end - start) // 2 + 10
    pnls    = np.empty(max_t)
    in_trade = False
    entry_px = 0.0
    hold     = 0
    nt = 0
    nw = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        h  = hi[t]
        l  = lo[t]
        sp = spread[t] / pip

        if in_trade:
            hold += 1
            vel_cur = (c - close[t - fn]) / (fn * pip)
            vel_prv = (close[t - fn] - close[t - 2*fn]) / (fn * pip)
            acc     = vel_cur - vel_prv

            should_exit = False
            if sl > 0.0:
                if di == 1  and l <= entry_px - sl * pip: should_exit = True
                elif di == -1 and h >= entry_px + sl * pip: should_exit = True
            if di == 1  and acc < acc_thresh:  should_exit = True
            elif di == -1 and acc > -acc_thresh: should_exit = True
            if hold >= mh: should_exit = True

            if should_exit:
                p = di * (c - entry_px) / pip - sp
                pnls[nt] = p
                nt += 1
                if p > 0.0: nw += 1
                in_trade = False

        else:
            if sp > sp_gate: continue
            vel_f = (c - close[t - fn]) / (fn * pip)
            vel_p = (close[t - fn] - close[t - 2*fn]) / (fn * pip)
            acc_f = vel_f - vel_p
            vel_s = (c - close[t - sn]) / (sn * pip)

            if di == 1:
                enter = (vel_f > vel_thresh and acc_f > 0.0 and vel_s > 0.0)
            else:
                enter = (vel_f < -vel_thresh and acc_f < 0.0 and vel_s < 0.0)

            if enter:
                entry_px = c; in_trade = True; hold = 0

    if in_trade and end > 0:
        sp = spread[end - 1] / pip
        p  = di * (close[end - 1] - entry_px) / pip - sp
        pnls[nt] = p; nt += 1
        if p > 0.0: nw += 1

    return pnls[:nt], nt, nw


@njit
def mc_pvalue(trade_pnls, n_mc):
    """Sign-shuffle MC test. Returns p-value = P(shuffle >= observed)."""
    observed = 0.0
    n = len(trade_pnls)
    for i in range(n):
        observed += trade_pnls[i]
    if n == 0:
        return 1.0
    count = 0
    for _ in range(n_mc):
        s = 0.0
        for i in range(n):
            s += trade_pnls[i] if np.random.random() > 0.5 else -trade_pnls[i]
        if s >= observed:
            count += 1
    return count / n_mc


# ── Data loader ────────────────────────────────────────────────────────────────

def load_pair(pair):
    path = DATA_DIR / f"{pair}_M5_BA.parquet"
    df   = pd.read_parquet(path)
    df   = df.sort_values("timestamp").reset_index(drop=True)
    close  = df["close"].astype(np.float64).values
    hi     = df["high"].astype(np.float64).values
    lo     = df["low"].astype(np.float64).values
    spread = (df["ask_c"] - df["bid_c"]).astype(np.float64).values
    return close, hi, lo, spread


# ── Per-pair runner ────────────────────────────────────────────────────────────

def run_pair(pair, configs):
    pip              = PIP[pair]
    close, hi, lo, sp = load_pair(pair)
    n                = len(close)
    is_end           = int(n * IS_FRAC)
    oos_days         = (n - is_end) / BARS_PER_DAY

    # SOP R5: spread gate from IS only
    sp_gate = float(np.percentile(sp[:is_end] / pip, 90))

    print(f"  {pair}: n={n:,}  IS={is_end:,}  OOS={n-is_end:,} ({oos_days:.0f}d)  sp_gate={sp_gate:.2f}p")

    # WF chunk boundaries (3 equal IS slices)
    chunk_sz   = is_end // N_WF
    chunk_ends = np.array([(k+1)*chunk_sz for k in range(N_WF)], dtype=np.int64)
    chunk_ends[-1] = is_end   # last chunk → exact IS boundary

    # ── Gate 1: Walk-Forward ──────────────────────────────────────────────────
    wf_pnl, wf_trades = wf_sweep(close, hi, lo, sp, pip, configs, sp_gate, chunk_ends)

    wf_pass = (np.all(wf_pnl > 0, axis=1) &
               np.all(wf_trades >= MIN_TRADES_PER_CHUNK, axis=1))
    wf_idx  = np.where(wf_pass)[0]
    print(f"    Gate1 WF:    {len(wf_idx):>5}/{len(configs):>5} pass")

    if len(wf_idx) == 0:
        return []

    # ── Gates 2+3: OOS p/d > 0 + MC ──────────────────────────────────────────
    survivors = []
    for ci in wf_idx:
        fn  = int(configs[ci, 0])
        sn  = int(configs[ci, 1])
        vt  = configs[ci, 2]
        at  = configs[ci, 3]
        mh  = int(configs[ci, 4])
        sl  = configs[ci, 5]
        di  = int(configs[ci, 6])

        oos_pnls, nt, nw = sim_collect(
            close, hi, lo, sp, pip, fn, sn, vt, at, mh, sl, di, sp_gate,
            is_end, n,
        )

        if nt == 0 or oos_pnls.sum() <= 0:
            continue

        p_val = mc_pvalue(oos_pnls, N_MC)
        if p_val >= 0.05:
            continue

        oos_pd = oos_pnls.sum() / oos_days
        survivors.append({
            "pair":       pair,
            "fast_n":     fn,
            "slow_n":     sn,
            "vel_thresh": vt,
            "acc_thresh": at,
            "max_hold":   mh,
            "sl_pips":    sl,
            "direction":  di,
            "sp_gate":    round(sp_gate, 2),
            "oos_pnl":    round(oos_pnls.sum(), 1),
            "oos_pd":     round(oos_pd, 1),
            "oos_trades": nt,
            "oos_wr":     round(nw / nt, 3),
            "mc_pval":    round(p_val, 4),
            "wf_pnl":     [round(x, 1) for x in wf_pnl[ci]],
            "wf_trades":  [int(x) for x in wf_trades[ci]],
        })

    print(f"    Gate2+3 OOS+MC: {len(survivors):>5}/{len(wf_idx):>5} pass")
    return survivors


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  Velocity-Acceleration Confluence — M5 Backtest Sweep")
    print("=" * 68)

    configs = build_configs()
    print(f"\nConfigs per pair: {len(configs)}")
    print(f"Parameter grid: fast={FAST_WINS} slow={SLOW_WINS} vel={VEL_THRESH}")
    print(f"               acc={ACC_THRESH} hold={MAX_HOLDS} sl={SL_PIPS} dir={DIRECTIONS}")

    # Warm Numba JIT on small array before full run
    print("\nWarming up Numba JIT...")
    dummy = np.ones(200, dtype=np.float64)
    _sim_pnl(dummy, dummy, dummy, dummy*0.0001, 0.0001, 1, 5, 0.0, 0.0, 6, 0.0, 1, 999.0, 0, 150)
    print("  Done.\n")

    all_results = []

    for pair in PAIRS:
        path = DATA_DIR / f"{pair}_M5_BA.parquet"
        if not path.exists():
            print(f"  {pair}: data not found, skipping")
            continue
        results = run_pair(pair, configs)
        all_results.extend(results)
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RESULTS SUMMARY")
    print("=" * 68)

    if not all_results:
        print("  No configs passed all 3 gates on any pair.")
        print("  Strategy has no detectable edge with realistic bar-close fills.")
        return

    df = pd.DataFrame(all_results)

    # Best config per pair (by OOS p/d)
    print(f"\nTotal survivors: {len(df)}")
    print(f"\nBest config per pair (by OOS p/d):\n")
    print(f"  {'Pair':<10} {'fast':>5} {'slow':>5} {'vel':>5} {'acc':>6} "
          f"{'hold':>5} {'sl':>4} {'dir':>4} {'p/d':>7} {'trades':>7} {'WR':>6} {'MC':>7}")
    print("  " + "-" * 76)

    for pair in PAIRS:
        sub = df[df["pair"] == pair]
        if sub.empty:
            continue
        row = sub.sort_values("oos_pd", ascending=False).iloc[0]
        d_str = "LONG" if row["direction"] == 1 else "SHORT"
        print(f"  {pair:<10} {int(row['fast_n']):>5} {int(row['slow_n']):>5} "
              f"{row['vel_thresh']:>5.1f} {row['acc_thresh']:>6.1f} "
              f"{int(row['max_hold']):>5} {int(row['sl_pips']):>4} "
              f"{d_str:>5} {row['oos_pd']:>7.1f} {row['oos_trades']:>7} "
              f"{row['oos_wr']:>6.1%} {row['mc_pval']:>7.4f}")

    # Distribution stats
    print(f"\nOOS p/d stats (all survivors):")
    print(f"  mean={df['oos_pd'].mean():.1f}  median={df['oos_pd'].median():.1f}  "
          f"P5={df['oos_pd'].quantile(0.05):.1f}  P95={df['oos_pd'].quantile(0.95):.1f}")

    # Pairs with any survivor
    n_pairs_pass = df["pair"].nunique()
    print(f"\nPairs with at least 1 survivor: {n_pairs_pass}/12")

    # Parameter patterns among survivors
    print(f"\nWinning parameter distribution (survivors only):")
    for col in ["fast_n", "slow_n", "vel_thresh", "acc_thresh", "max_hold", "sl_pips"]:
        counts = df[col].value_counts().head(3)
        print(f"  {col}: {dict(counts)}")

    # Save full results
    out = ROOT / "research" / "experiments" / "vel_acc" / "results_vel_acc.csv"
    df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
