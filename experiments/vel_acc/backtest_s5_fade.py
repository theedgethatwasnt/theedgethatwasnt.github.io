"""
S5 Rolling-Window Fade Strategy — Backtest Sweep
=================================================
IC study finding: high velocity predicts mean-REVERSION, not continuation.
    mom_z_1h → 10m target IC = -0.026, t = -62 (EUR/USD, 5.8M IS bars)

Strategy: FADE overextended moves, entry confirmed by reversal acceleration.

Signal (S5 rolling windows — all timeframes baked in by N offset):
  vel_slow(t) = (close[t] - close[t - slow_n]) / (slow_n * pip)   [pips/bar]
  vel_fast(t) = (close[t] - close[t - fast_n]) / (fast_n * pip)
  acc_fast(t) = vel_fast(t) - vel_fast(t - fast_n)                 [pips]

Entry LONG (fade downmove):
  vel_slow < -vel_thresh   — sustained downward overextension
  vel_fast > acc_thresh    — fast window starting to reverse upward (optional filter)
  sp <= sp_gate            — spread within IS P90 limit

Entry SHORT (fade upmove):
  vel_slow > +vel_thresh   — sustained upward overextension
  vel_fast < -acc_thresh   — fast window starting to reverse downward (optional)
  sp <= sp_gate

Exit (all at S5 bar close — zero intra-bar fill gap):
  sign(vel_fast) ≠ sign(entry direction)  — fast reversal fully materialised
  OR hold >= max_hold_bars                — safety cap

All fills at S5 bar close (SOP R1, R6). Spread deducted at exit (SOP R3).
sp_gate = IS P90 spread, never recomputed from OOS (SOP R5).
"""

import gc
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT     = Path(__file__).resolve().parents[3]

# Will load from s5_ba/ (full history) if available, else s5_ohlc/ (partial)
BA_DIR   = ROOT / "data" / "s5_ba"
OHLC_DIR = ROOT / "data" / "s5_ohlc"

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
S5_PER_TRADING_DAY   = 17280.0   # 24hr × 3600s / 5s per bar

# Sweep parameters — all windows in S5 bars
SLOW_WINS  = [180, 360, 720]      # 15m, 30m, 1h look-back for overextension
FAST_WINS  = [6, 12, 60]          # 30s, 1m, 5m short-term reversal confirmation
VEL_THRESH = [0.5, 1.0, 2.0, 3.0]  # pips/bar — overextension threshold (slow window)
MAX_HOLDS  = [60, 120, 240, 720]  # 5m, 10m, 20m, 1h max hold in S5 bars
DIRECTIONS = [1, -1]              # 1=LONG (fade down), -1=SHORT (fade up)

# Constraint: slow_win > fast_win


# ── Config builder ─────────────────────────────────────────────────────────────

def build_configs():
    rows = []
    for sw, fw, vt, mh, d in product(
        SLOW_WINS, FAST_WINS, VEL_THRESH, MAX_HOLDS, DIRECTIONS,
    ):
        if sw <= fw:
            continue
        rows.append((float(sw), float(fw), float(vt), float(mh), float(d)))
    return np.array(rows, dtype=np.float64)


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit
def _sim_pnl(close, hi, lo, spread, pip,
             slow_n, fast_n, vel_thresh, max_hold, direction,
             sp_gate, start, end):
    """
    Simulate one fade config over [start, end).
    LONG: enter when slow_vel < -thresh AND fast_vel starting positive.
    SHORT: enter when slow_vel > +thresh AND fast_vel starting negative.
    Exit: fast_vel crosses zero in exit direction OR max_hold.
    Returns (total_pnl_pips, n_trades).
    """
    warmup   = slow_n + fast_n
    di       = int(direction)   # +1 LONG, -1 SHORT
    sn       = int(slow_n)
    fn       = int(fast_n)
    mh       = int(max_hold)

    in_trade = False
    entry_px = 0.0
    hold     = 0
    total_pnl = 0.0
    n_trades  = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        sp = spread[t] / pip

        if in_trade:
            hold += 1
            vel_fast = (c - close[t - fn]) / (fn * pip)

            should_exit = False
            # Exit when fast velocity confirms reversal is done (crosses back)
            if di == 1  and vel_fast < 0.0:   should_exit = True
            elif di == -1 and vel_fast > 0.0:  should_exit = True
            if hold >= mh: should_exit = True

            if should_exit:
                pnl        = di * (c - entry_px) / pip - sp
                total_pnl += pnl
                n_trades  += 1
                in_trade   = False

        else:
            if sp > sp_gate: continue

            vel_slow = (c - close[t - sn]) / (sn * pip)
            vel_fast = (c - close[t - fn]) / (fn * pip)

            if di == 1:
                # Fade downmove: slow going down, fast starting to turn up
                enter = (vel_slow < -vel_thresh and vel_fast > 0.0)
            else:
                # Fade upmove: slow going up, fast starting to turn down
                enter = (vel_slow > vel_thresh and vel_fast < 0.0)

            if enter:
                entry_px = c
                in_trade = True
                hold     = 0

    if in_trade and end > 0:
        sp        = spread[end - 1] / pip
        pnl       = di * (close[end - 1] - entry_px) / pip - sp
        total_pnl += pnl
        n_trades  += 1

    return total_pnl, n_trades


@njit(parallel=True)
def wf_sweep(close, hi, lo, spread, pip, configs, sp_gate, chunk_ends):
    n_cfg     = len(configs)
    nc        = len(chunk_ends)
    wf_pnl    = np.zeros((n_cfg, nc))
    wf_trades = np.zeros((n_cfg, nc))

    for ci in prange(n_cfg):
        sn  = int(configs[ci, 0])
        fn  = int(configs[ci, 1])
        vt  = configs[ci, 2]
        mh  = int(configs[ci, 3])
        di  = int(configs[ci, 4])

        s = 0
        for k in range(nc):
            e     = int(chunk_ends[k])
            pnl, nt = _sim_pnl(close, hi, lo, spread, pip, sn, fn, vt, mh, di, sp_gate, s, e)
            wf_pnl[ci, k]    = pnl
            wf_trades[ci, k] = nt
            s = e

    return wf_pnl, wf_trades


@njit
def sim_collect(close, hi, lo, spread, pip,
                slow_n, fast_n, vel_thresh, max_hold, direction,
                sp_gate, start, end):
    warmup  = int(slow_n) + int(fast_n)
    di      = int(direction)
    sn      = int(slow_n)
    fn      = int(fast_n)
    mh      = int(max_hold)

    max_t   = (end - start) // 2 + 10
    pnls    = np.empty(max_t)
    in_trade = False
    entry_px = 0.0
    hold     = 0
    nt = 0; nw = 0

    t_start = start if start > warmup else warmup

    for t in range(t_start, end):
        c  = close[t]
        sp = spread[t] / pip

        if in_trade:
            hold += 1
            vel_fast = (c - close[t - fn]) / (fn * pip)
            should_exit = False
            if di == 1  and vel_fast < 0.0:  should_exit = True
            elif di == -1 and vel_fast > 0.0: should_exit = True
            if hold >= mh: should_exit = True

            if should_exit:
                p = di * (c - entry_px) / pip - sp
                pnls[nt] = p; nt += 1
                if p > 0.0: nw += 1
                in_trade = False

        else:
            if sp > sp_gate: continue
            vel_slow = (c - close[t - sn]) / (sn * pip)
            vel_fast = (c - close[t - fn]) / (fn * pip)

            if di == 1:
                enter = (vel_slow < -vel_thresh and vel_fast > 0.0)
            else:
                enter = (vel_slow > vel_thresh and vel_fast < 0.0)

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
    observed = 0.0
    n = len(trade_pnls)
    for i in range(n): observed += trade_pnls[i]
    if n == 0: return 1.0
    count = 0
    for _ in range(n_mc):
        s = 0.0
        for i in range(n):
            s += trade_pnls[i] if np.random.random() > 0.5 else -trade_pnls[i]
        if s >= observed: count += 1
    return count / n_mc


# ── Data loader ────────────────────────────────────────────────────────────────

def find_parquet(pair):
    """Find best available S5 parquet for pair. Prefer full-history s5_ba/."""
    ba_path = BA_DIR / f"{pair}_S5_BA.parquet"
    if ba_path.exists():
        return ba_path, "s5_ba"
    # Fallback to existing partial files
    for name in [f"{pair}_S5_BA.parquet"]:
        p = OHLC_DIR / name
        if p.exists():
            return p, "s5_ohlc"
    return None, None


def load_pair(pair):
    path, source = find_parquet(pair)
    if path is None:
        return None, None, None, None, None

    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Handle both schema variants
    if "close" in df.columns:
        close  = df["close"].astype(np.float64).values
        hi     = df["high"].astype(np.float64).values
        lo     = df["low"].astype(np.float64).values
    else:
        # bid_o/bid_h/... schema
        close = ((df["bid_c"] + df["ask_c"]) / 2).astype(np.float64).values
        hi    = ((df["bid_h"] + df["ask_h"]) / 2).astype(np.float64).values
        lo    = ((df["bid_l"] + df["ask_l"]) / 2).astype(np.float64).values

    spread = (df["ask_c"] - df["bid_c"]).astype(np.float64).values
    return close, hi, lo, spread, source


# ── Per-pair runner ────────────────────────────────────────────────────────────

def run_pair(pair, configs):
    pip                    = PIP[pair]
    close, hi, lo, sp, src = load_pair(pair)
    if close is None:
        print(f"  {pair}: no data, skipping")
        return []

    n       = len(close)
    is_end  = int(n * IS_FRAC)
    oos_days = (n - is_end) / S5_PER_TRADING_DAY

    sp_gate  = float(np.percentile(sp[:is_end] / pip, 90))

    print(f"  {pair} [{src}]: n={n:,}  IS={is_end:,}  OOS={oos_days:.0f}d  sp_gate={sp_gate:.2f}p")

    if oos_days < 30:
        print(f"    SKIP: OOS only {oos_days:.0f} days (need ≥30)")
        return []

    chunk_sz   = is_end // N_WF
    chunk_ends = np.array([(k+1)*chunk_sz for k in range(N_WF)], dtype=np.int64)
    chunk_ends[-1] = is_end

    # Gate 1: WF sweep
    wf_pnl, wf_trades = wf_sweep(close, hi, lo, sp, pip, configs, sp_gate, chunk_ends)

    wf_pass = (np.all(wf_pnl > 0, axis=1) &
               np.all(wf_trades >= MIN_TRADES_PER_CHUNK, axis=1))
    wf_idx  = np.where(wf_pass)[0]
    print(f"    Gate1 WF:       {len(wf_idx):>5}/{len(configs):>5} pass")

    if len(wf_idx) == 0:
        return []

    # Gates 2+3: OOS + MC
    survivors = []
    for ci in wf_idx:
        sn  = int(configs[ci, 0])
        fn  = int(configs[ci, 1])
        vt  = configs[ci, 2]
        mh  = int(configs[ci, 3])
        di  = int(configs[ci, 4])

        oos_pnls, nt, nw = sim_collect(
            close, hi, lo, sp, pip, sn, fn, vt, mh, di, sp_gate, is_end, n,
        )

        if nt == 0 or oos_pnls.sum() <= 0:
            continue

        p_val  = mc_pvalue(oos_pnls, N_MC)
        if p_val >= 0.05:
            continue

        oos_pd = oos_pnls.sum() / oos_days
        survivors.append({
            "pair":       pair,
            "data":       src,
            "slow_n":     sn,
            "fast_n":     fn,
            "vel_thresh": vt,
            "max_hold":   mh,
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
    print("  S5 Rolling-Window Fade Strategy — Backtest Sweep")
    print("=" * 68)

    configs = build_configs()
    print(f"\nConfigs per pair: {len(configs)}")
    print(f"Slow windows (S5 bars): {SLOW_WINS} = 15m/30m/1h")
    print(f"Fast windows (S5 bars): {FAST_WINS} = 30s/1m/5m")
    print(f"Vel thresholds (p/bar): {VEL_THRESH}")
    print(f"Max holds (S5 bars):    {MAX_HOLDS} = 5m/10m/20m/1h")

    print("\nWarming up Numba JIT...")
    dummy = np.ones(2000, dtype=np.float64)
    _sim_pnl(dummy, dummy, dummy, dummy*0.0001, 0.0001, 180, 6, 0.5, 60, 1, 999.0, 0, 1000)
    print("  Done.\n")

    all_results = []
    pairs_run = []

    for pair in PAIRS:
        path, src = find_parquet(pair)
        if path is None:
            print(f"  {pair}: no S5 data found, skipping")
            continue
        pairs_run.append(pair)
        results = run_pair(pair, configs)
        all_results.extend(results)
        gc.collect()

    print("\n" + "=" * 68)
    print("  RESULTS SUMMARY")
    print("=" * 68)

    if not all_results:
        print("  No configs passed all 3 gates.")
        print(f"\nPairs tested: {pairs_run}")
        print("Fade strategy has no detectable edge with these parameters.")
        return

    df = pd.DataFrame(all_results)
    print(f"\nTotal survivors: {len(df)} across {df['pair'].nunique()} pairs")

    print(f"\nBest config per pair (by OOS p/d):\n")
    hdr = (f"  {'Pair':<10} {'src':<8} {'slow':>5} {'fast':>5} {'vel':>5} "
           f"{'hold':>6} {'dir':>5} {'p/d':>7} {'trades':>7} {'WR':>6} {'MC':>7}")
    print(hdr)
    print("  " + "-" * 74)

    for pair in PAIRS:
        sub = df[df["pair"] == pair]
        if sub.empty: continue
        row = sub.sort_values("oos_pd", ascending=False).iloc[0]
        d_str = "LONG" if row["direction"] == 1 else "SHORT"
        print(f"  {pair:<10} {row['data']:<8} {int(row['slow_n']):>5} {int(row['fast_n']):>5} "
              f"{row['vel_thresh']:>5.1f} {int(row['max_hold']):>6} {d_str:>5} "
              f"{row['oos_pd']:>7.1f} {row['oos_trades']:>7} "
              f"{row['oos_wr']:>6.1%} {row['mc_pval']:>7.4f}")

    print(f"\nOOS p/d stats: mean={df['oos_pd'].mean():.1f}  "
          f"median={df['oos_pd'].median():.1f}  P5={df['oos_pd'].quantile(0.05):.1f}")

    print(f"\nWinning parameter distribution:")
    for col in ["slow_n", "fast_n", "vel_thresh", "max_hold"]:
        counts = df[col].value_counts().head(3)
        print(f"  {col}: {dict(counts)}")

    out = ROOT / "research" / "experiments" / "vel_acc" / "results_s5_fade.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
