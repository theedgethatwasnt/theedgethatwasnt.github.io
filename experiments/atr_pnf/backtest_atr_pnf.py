"""
ATR-based P&F Backtest — multi-pair sweep.
Box size = atr_mult × ATR(atr_period), reanchored at each column start.

Design:
  ATR computed incrementally per bar (Wilder smoothing — R4 compliant).
  Box size is FROZEN PER COLUMN: at each reversal, new bs = atr_mult × atr,
  then pnf_idx re-anchored to current price with new bs. The chart restarts
  fresh each column — no float drift, no stale scale.

  Reversal requirement also adapts: rev boxes × (mult × ATR) = volatility-scaled
  retracement threshold. At mult=1.0, a 1-rev reversal requires 1 full ATR — far
  above the 5-pip fixed threshold that spread+slippage consumed live.

SOP compliance (CLAUDE.md §Backtest-Live Consistency SOP):
  R1  Closed bars only — bar[i] consumed only after it closes
  R2  Within-bar sequence — bull=(close>=open) → HIGH then LOW
  R3  Mid OHLC for signals; spread deducted at P&L
  R3a bid_h/l ask_h/l NOT used
  R3b BA data fetched before run; no fallback spread
  R4  ATR = Wilder incremental (single running value); ring buffer for col_hist
  R4a col_count (in-progress column) never pushed to col_hist ring
  R5  IS P90 spread gate hardcoded per pair
  R6  Kernel logic is the reference for any future live adaptation
  R8  OOS touched exactly once

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/atr_pnf/backtest_atr_pnf.py [PAIR]
  # e.g. python3 research/experiments/atr_pnf/backtest_atr_pnf.py GBP_JPY
  # omit PAIR to run all 4 top pairs
"""

import math, sys, os, time
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE    = Path(__file__).resolve().parents[3]
BA_DIR  = BASE / "data/m5_ba"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

IS_FRAC    = 0.70
MAX_TRADES = 20_000
MAX_K      = 10       # ring-buffer depth for completed column heights (R4a)

# ── Top pairs: (pair, pip, IS-P90-spread-gate) ────────────────────────────────
# Spread gates = IS P90 hardcoded from fixed-box sweep (R5 — same gate reused;
# ATR boxes don't change spread, only chart scale).
PAIRS = [
    ("GBP_JPY", 0.01,   4.0),
    ("USD_JPY", 0.01,   2.1),
    ("EUR_JPY", 0.01,   2.5),
    ("GBP_USD", 0.0001, 2.4),
]

# ── Parameter space ────────────────────────────────────────────────────────────
# atr_mult stored ×10 as int (Numba int32 array): 5→0.5, 10→1.0, 15→1.5, 20→2.0, 30→3.0
ATR_PERIODS = np.array([7, 14, 20],             dtype=np.int32)
ATR_MULTS10 = np.array([5, 10, 15, 20, 30],    dtype=np.int32)  # ÷10 = actual mult
REVERSALS   = np.array([1, 2, 3],              dtype=np.int32)
MIN_COLS    = np.array([2, 3, 4, 5, 6],        dtype=np.int32)
ENTRY_TYPES = [0, 1]   # 0=E1 (immediate), 1=E2 (confirmation)

# Exit variants — same encoding as fixed-box sweep.
# Focus on trail-based exits that won in the original sweep, plus X2 for coverage.
EXIT_DEFS = [
    (4,  1, 0),   # X3b_1   : 1-box trail
    (5,  2, 0),   # X3b_2   : 2-box trail (the live fix candidate)
    (6,  3, 0),   # X3b_3   : 3-box trail
    (3,  0, 0),   # X2      : first adverse reversal
    (11, 1, 5),   # X3c_1_5 : 1-box trail + col-SMA(5)
    (13, 2, 5),   # X3c_2_5 : 2-box trail + col-SMA(5)
    (10, 1, 3),   # X3c_1_3 : 1-box trail + col-SMA(3)
    (12, 2, 3),   # X3c_2_3 : 2-box trail + col-SMA(3)
]
EXIT_NAMES = {
    4: "X3b_1", 5: "X3b_2", 6: "X3b_3", 3: "X2",
    11: "X3c_1_5", 13: "X3c_2_5", 10: "X3c_1_3", 12: "X3c_2_3",
}

# Total configs: 3 × 5 × 3 × 5 × 2 × 8 = 3,600


def build_configs():
    rows = []
    for ap in ATR_PERIODS:
        for am in ATR_MULTS10:
            for rv in REVERSALS:
                for nc in MIN_COLS:
                    for et in ENTRY_TYPES:
                        for (xt, xp1, xp2) in EXIT_DEFS:
                            rows.append((ap, am, rv, nc, et, xt, xp1, xp2))
    return np.array(rows, dtype=np.int32)


def config_name(cfg_row):
    ap, am, rv, nc, et, xt, xp1, xp2 = cfg_row
    mult_str = f"{am/10:.1f}".replace(".", "p")  # e.g. 1.0 → "1p0"
    return f"atr{ap}_m{mult_str}_r{rv}_n{nc}_E{et+1}_{EXIT_NAMES[xt]}"


def load_data(pair, pip):
    path = BA_DIR / f"{pair}_M5_BA.parquet"
    assert path.exists(), f"Missing BA parquet: {path}"
    df = pd.read_parquet(path)

    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)

    n      = len(df)
    is_end = int(n * IS_FRAC)
    p90    = float(np.percentile(spreads[:is_end], 90))

    chunk0 = is_end // 3
    chunk1 = 2 * (is_end // 3)
    chunks = np.zeros(n, dtype=np.int8)
    chunks[chunk0:chunk1] = 1
    chunks[chunk1:is_end] = 2
    chunks[is_end:]       = 3

    return opens, highs, lows, closes, spreads, chunks, is_end, n, p90


# ── Numba kernel ───────────────────────────────────────────────────────────────

@nb.njit(inline='always')
def _col_sma(hist, ptr, n_valid, k):
    """Mean of last min(k, n_valid) completed column heights (R4a)."""
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        idx = (ptr - 1 - j) % MAX_K
        total += hist[idx]
    return total / count


@nb.njit(parallel=True)
def run_kernel(
    opens, highs, lows, closes, spreads, bar_chunks,
    configs,        # (N, 8): atr_period, atr_mult10, rev, n_min, entry_t, exit_t, xp1, xp2
    spread_gate,    # scalar pips
    pip,            # price per pip
    is_end,         # last IS bar index + 1
    min_bs,         # minimum box size in price units (floor at 2 pips)
    trade_pnl,      # (N, MAX_TRADES) float32 output
    trade_chunk,    # (N, MAX_TRADES) int8 output
    trade_cnt,      # (N,) int32 output
):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        atr_period = configs[ci, 0]
        atr_mult   = configs[ci, 1] * 0.1   # restore float multiplier
        rev        = configs[ci, 2]
        n_min      = configs[ci, 3]
        entry_t    = configs[ci, 4]
        exit_t     = configs[ci, 5]
        xp1        = configs[ci, 6]
        xp2        = configs[ci, 7]

        # ── ATR state ────────────────────────────────────────────────────────
        atr = 0.0    # running Wilder ATR (price units, not pips)
        bs  = 0.0    # current column's box size (frozen per column)

        # ── P&F state ────────────────────────────────────────────────────────
        pnf_idx   = 0
        pnf_level = 0.0
        pnf_dir   = 0   # 0=uninit, +1=up, -1=down
        col_count = 0
        prev_col  = 0

        col_hist     = np.zeros(MAX_K, dtype=np.float64)
        col_hist_ptr = 0
        col_hist_n   = 0

        # ── Position state ───────────────────────────────────────────────────
        pos      = 0
        entry_px = 0.0
        hw_level = 0.0
        pending  = 0

        t_cnt = 0

        for i in range(N_BARS):
            opn = opens[i]
            hi  = highs[i]
            lo  = lows[i]
            cl  = closes[i]
            sp  = spreads[i]
            ck  = bar_chunks[i]

            # ── ATR update (Wilder, R4) ───────────────────────────────────────
            if i == 0:
                tr  = hi - lo
                atr = tr
            else:
                tr1 = hi - lo
                tr2 = hi - closes[i-1]
                if tr2 < 0.0: tr2 = -tr2
                tr3 = lo - closes[i-1]
                if tr3 < 0.0: tr3 = -tr3
                tr  = tr1 if tr1 >= tr2 else tr2
                if tr3 > tr: tr = tr3
                atr = (atr * (atr_period - 1) + tr) / atr_period

            # Effective box size for new column (may differ from current bs)
            new_bs = atr_mult * atr
            if new_bs < min_bs:
                new_bs = min_bs

            # ── Within-bar price sequence (R2) ────────────────────────────────
            bull = (cl >= opn)
            p1   = hi if bull else lo
            p2   = lo if bull else hi

            # ── P&F update (two ticks per bar) ───────────────────────────────
            did_reverse_p1 = False
            did_reverse_p2 = False
            prev_col_p1    = 0
            prev_col_p2    = 0

            for tick in range(2):
                px = p1 if tick == 0 else p2

                if pnf_dir == 0:
                    # First initialization — use current ATR for box size
                    bs        = new_bs
                    pnf_idx   = int(px / bs)
                    pnf_level = pnf_idx * bs
                    pnf_dir   = 1
                    col_count = 1
                    continue

                delta = int(px / bs) - pnf_idx

                if pnf_dir == 1:
                    if delta >= 1:
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count += delta
                    elif delta <= -rev:
                        # ── Reversal down ─────────────────────────────────────
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1

                        # Freeze NEW box size and re-anchor (key ATR-P&F step)
                        bs        = new_bs
                        pnf_dir   = -1
                        pnf_idx   = int(px / bs)
                        pnf_level = pnf_idx * bs
                        col_count = 1

                        if tick == 0:
                            did_reverse_p1 = True
                            prev_col_p1    = prev_col
                        else:
                            did_reverse_p2 = True
                            prev_col_p2    = prev_col

                elif pnf_dir == -1:
                    if delta <= -1:
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count += (-delta)
                    elif delta >= rev:
                        # ── Reversal up ───────────────────────────────────────
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1

                        bs        = new_bs
                        pnf_dir   = 1
                        pnf_idx   = int(px / bs)
                        pnf_level = pnf_idx * bs
                        col_count = 1

                        if tick == 0:
                            did_reverse_p1 = True
                            prev_col_p1    = prev_col
                        else:
                            did_reverse_p2 = True
                            prev_col_p2    = prev_col

            did_reverse     = did_reverse_p1 or did_reverse_p2
            prev_col_at_rev = prev_col_p1 if did_reverse_p1 else prev_col_p2

            # ── High-water update ─────────────────────────────────────────────
            if pos == 1:
                if pnf_dir == 1 and pnf_level > hw_level:
                    hw_level = pnf_level
            elif pos == -1:
                if pnf_dir == -1 and pnf_level < hw_level:
                    hw_level = pnf_level

            # ── EXIT ──────────────────────────────────────────────────────────
            exit_triggered = False
            exit_px_val    = 0.0

            if pos != 0:
                if exit_t == 3:
                    # X2: first adverse reversal
                    if did_reverse and pnf_dir != pos:
                        exit_px_val    = cl
                        exit_triggered = True

                elif 4 <= exit_t <= 6:
                    # X3b: box-quantized trail (d boxes from high-water)
                    d = float(xp1)
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail:
                            exit_px_val    = trail
                            exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail:
                            exit_px_val    = trail
                            exit_triggered = True

                else:
                    # X3c: X3b trail OR X7 col-SMA — whichever fires first
                    d = float(xp1)
                    k = xp2
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail:
                            exit_px_val    = trail
                            exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail:
                            exit_px_val    = trail
                            exit_triggered = True
                    if not exit_triggered and pnf_dir != pos:
                        sma_k = _col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                        if sma_k > 0.0 and col_count >= sma_k:
                            exit_px_val    = cl
                            exit_triggered = True

            if exit_triggered and t_cnt < MAX_TRADES:
                pnl_pips = (exit_px_val - entry_px) * pos / pip - sp
                trade_pnl[ci, t_cnt]   = np.float32(pnl_pips)
                trade_chunk[ci, t_cnt] = ck
                t_cnt += 1
                pos      = 0
                entry_px = 0.0
                hw_level = 0.0

            # ── ENTRY ─────────────────────────────────────────────────────────
            if pos == 0:
                can_enter = (sp <= spread_gate)

                if can_enter:
                    if entry_t == 0:
                        # E1: immediate entry on qualifying reversal
                        if did_reverse and prev_col_at_rev >= n_min:
                            pos      = pnf_dir
                            entry_px = cl
                            hw_level = pnf_level
                    else:
                        # E2: wait for confirmation
                        if did_reverse and prev_col_at_rev >= n_min:
                            pending = pnf_dir
                        if did_reverse and pending != 0 and pnf_dir != pending:
                            pending = 0
                        if pending != 0 and pnf_dir == pending and col_count > rev:
                            pos      = pending
                            entry_px = cl
                            hw_level = pnf_level
                            pending  = 0
                else:
                    if did_reverse and pending != 0 and pnf_dir != pending:
                        pending = 0

        trade_cnt[ci] = t_cnt


# ── Validation pipeline ────────────────────────────────────────────────────────

def stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, config_names):
    N = len(trade_cnt)
    results = []
    for ci in range(N):
        tc = trade_cnt[ci]
        if tc == 0:
            continue
        pnl   = trade_pnl[ci, :tc]
        chunk = trade_chunk[ci, :tc].astype(np.int32)
        is_mask = chunk <= 2
        is_pnl  = pnl[is_mask]
        if len(is_pnl) < 30:
            continue
        c0 = float(pnl[chunk == 0].sum())
        c1 = float(pnl[chunk == 1].sum())
        c2 = float(pnl[chunk == 2].sum())
        if c0 <= 0 or c1 <= 0 or c2 <= 0:
            continue
        results.append({
            "ci": ci, "name": config_names[ci],
            "is_pnl": round(float(is_pnl.sum()), 1),
            "is_ntrd": int(is_mask.sum()),
            "c0": round(c0, 1), "c1": round(c1, 1), "c2": round(c2, 1),
        })
    return pd.DataFrame(results).sort_values("is_pnl", ascending=False) if results \
           else pd.DataFrame(columns=["ci","name","is_pnl","is_ntrd","c0","c1","c2"])


def mc_permutation_test(pnl_arr, n_shuffles=1000, seed=42):
    rng    = np.random.default_rng(seed)
    actual = float(pnl_arr.sum())
    perms  = np.empty(n_shuffles)
    for k in range(n_shuffles):
        signs  = rng.choice([-1.0, 1.0], size=len(pnl_arr))
        perms[k] = float((np.abs(pnl_arr) * signs).sum())
    return float((perms >= actual).mean()), float(np.percentile(perms, 95))


def bootstrap_p5(pnl_arr, n_boot=2000, days_oos=610, seed=99):
    rng  = np.random.default_rng(seed)
    n    = len(pnl_arr)
    if n == 0:
        return 0.0
    sums = np.empty(n_boot)
    for k in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        sums[k] = float(pnl_arr[idx].sum()) / days_oos
    return float(np.percentile(sums, 5))


def stage2_mc_bootstrap(stage1_df, trade_pnl, trade_chunk, trade_cnt, top_n=200):
    candidates = stage1_df.head(top_n)
    results = []
    for _, row in candidates.iterrows():
        ci   = int(row["ci"])
        tc   = trade_cnt[ci]
        pnl  = trade_pnl[ci, :tc].astype(np.float64)
        ck   = trade_chunk[ci, :tc].astype(np.int32)
        is_pnl = pnl[ck <= 2]
        p_val, pct95 = mc_permutation_test(is_pnl)
        p5           = bootstrap_p5(is_pnl)
        results.append({
            "ci": ci, "name": row["name"],
            "is_pnl": row["is_pnl"], "is_ntrd": row["is_ntrd"],
            "mc_pval": round(p_val, 3), "mc_p95": round(pct95, 1),
            "bootstrap_p5": round(p5, 2), "passed_mc": int(p_val < 0.05),
        })
    df = pd.DataFrame(results).sort_values("bootstrap_p5", ascending=False)
    return df


def stage3_oos(stage2_df, trade_pnl, trade_chunk, trade_cnt, days_oos=610):
    survivors = stage2_df[stage2_df["passed_mc"] == 1]
    results = []
    for _, row in survivors.iterrows():
        ci  = int(row["ci"])
        tc  = trade_cnt[ci]
        pnl = trade_pnl[ci, :tc].astype(np.float64)
        ck  = trade_chunk[ci, :tc].astype(np.int32)
        oos_pnl  = pnl[ck == 3]
        oos_tot  = float(oos_pnl.sum())
        oos_ntrd = len(oos_pnl)
        oos_pd   = oos_tot / days_oos if days_oos > 0 else 0.0
        results.append({
            "ci": ci, "name": row["name"],
            "is_pnl": row["is_pnl"], "is_ntrd": row["is_ntrd"],
            "mc_pval": row["mc_pval"], "bootstrap_p5": row["bootstrap_p5"],
            "oos_pnl": round(oos_tot, 1), "oos_ntrd": oos_ntrd,
            "oos_pd": round(oos_pd, 2), "oos_pass": int(oos_tot > 0 and oos_ntrd >= 10),
        })
    return pd.DataFrame(results).sort_values("oos_pd", ascending=False) if results \
           else pd.DataFrame(columns=["ci","name","is_pnl","is_ntrd","mc_pval",
                                      "bootstrap_p5","oos_pnl","oos_ntrd","oos_pd","oos_pass"])


# ── Per-pair runner ────────────────────────────────────────────────────────────

def run_pair(pair, pip, sp_gate, configs, config_names, compiled=False):
    print(f"\n{'='*60}")
    print(f"  {pair}  (pip={pip}, spread_gate={sp_gate}p)")
    print(f"{'='*60}")

    opens, highs, lows, closes, spreads, chunks, is_end, n_bars, p90 = \
        load_data(pair, pip)
    print(f"  Bars={n_bars:,}  IS={is_end:,}  OOS={n_bars-is_end:,}")
    print(f"  IS spread P90={p90:.2f}p  gate={sp_gate}p")
    if abs(p90 - sp_gate) > 0.5:
        print(f"  ⚠ Spread gate differs from IS P90 by >{abs(p90-sp_gate):.2f}p")

    N_CONFIGS  = len(configs)
    min_bs_val = 2.0 * pip   # floor: 2 pips in price units

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS,               dtype=np.int32)

    if not compiled:
        print("  Warming up Numba JIT (500 bars, 1 config)...")
        dp = np.zeros((1, MAX_TRADES), dtype=np.float32)
        dc = np.zeros((1, MAX_TRADES), dtype=np.int8)
        dk = np.zeros(1, dtype=np.int32)
        t0 = time.time()
        run_kernel(
            opens[:500], highs[:500], lows[:500], closes[:500],
            spreads[:500], chunks[:500], configs[:1],
            sp_gate, pip, min(500, is_end), min_bs_val,
            dp, dc, dk,
        )
        print(f"  Compiled in {time.time()-t0:.1f}s")

    print(f"  Running {N_CONFIGS:,} configs × {n_bars:,} bars...")
    t0 = time.time()
    run_kernel(
        opens, highs, lows, closes, spreads, chunks,
        configs, sp_gate, pip, is_end, min_bs_val,
        trade_pnl, trade_chunk, trade_cnt,
    )
    elapsed = time.time() - t0
    total_t = int(trade_cnt.sum())
    print(f"  Done in {elapsed:.1f}s  |  {total_t:,} total trades  "
          f"({total_t/N_CONFIGS:.0f}/config avg)")

    # Validation pipeline
    print("\n  Stage 1: IS walk-forward screen...")
    s1 = stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, config_names)
    s1.to_csv(OUT_DIR / f"{pair}_stage1.csv", index=False)
    print(f"    {len(s1)}/{N_CONFIGS} passed IS WF")

    if len(s1) == 0:
        print("    No configs passed — no edge found.")
        return None, None, None

    print(f"  Stage 2: MC + bootstrap (top 200)...")
    s2 = stage2_mc_bootstrap(s1, trade_pnl, trade_chunk, trade_cnt)
    s2.to_csv(OUT_DIR / f"{pair}_stage2.csv", index=False)
    mc_n = s2["passed_mc"].sum()
    print(f"    {mc_n}/{len(s2)} passed MC (p<0.05)")

    if mc_n == 0:
        print("    No configs passed MC.")
        return s1, s2, None

    print(f"  Stage 3: OOS (sealed — one-time evaluation)...")
    days_oos = (n_bars - is_end) / 288.0   # 288 M5 bars per trading day
    s3 = stage3_oos(s2, trade_pnl, trade_chunk, trade_cnt, days_oos=days_oos)
    s3.to_csv(OUT_DIR / f"{pair}_final.csv", index=False)
    oos_n = s3["oos_pass"].sum()
    print(f"    {oos_n}/{len(s3)} passed OOS")

    if oos_n > 0:
        print(f"\n  🟢 Top OOS configs for {pair}:")
        cols = ["name","oos_pd","oos_ntrd","is_pnl","bootstrap_p5","mc_pval"]
        print(s3[s3["oos_pass"]==1][cols].head(8).to_string(index=False))
    else:
        print(f"  🔴 No configs passed all stages for {pair}.")

    return s1, s2, s3


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    arg_pair = sys.argv[1].upper() if len(sys.argv) > 1 else None
    pairs_to_run = [(p, pip, sg) for p, pip, sg in PAIRS
                    if arg_pair is None or p == arg_pair]

    if not pairs_to_run:
        print(f"Unknown pair: {arg_pair}. Valid: {[p for p,_,_ in PAIRS]}")
        sys.exit(1)

    configs      = build_configs()
    config_names = [config_name(row) for row in configs]
    N_CONFIGS    = len(configs)
    print(f"ATR P&F Sweep — {N_CONFIGS} configs per pair")
    print(f"ATR periods: {ATR_PERIODS.tolist()}")
    print(f"ATR mults:   {[m/10 for m in ATR_MULTS10.tolist()]}")
    print(f"Reversals:   {REVERSALS.tolist()}")
    print(f"Min cols:    {MIN_COLS.tolist()}")
    print(f"Exits:       {list(EXIT_NAMES.values())}")
    print(f"Pairs:       {[p for p,_,_ in pairs_to_run]}")

    summary = []
    compiled = False
    for pair, pip, sp_gate in pairs_to_run:
        s1, s2, s3 = run_pair(pair, pip, sp_gate, configs, config_names,
                               compiled=compiled)
        compiled = True   # JIT warm-up only on first pair

        oos_winners = 0
        best_pd     = 0.0
        best_name   = "—"
        if s3 is not None and len(s3) > 0:
            winners = s3[s3["oos_pass"] == 1]
            oos_winners = len(winners)
            if oos_winners > 0:
                best = winners.iloc[0]
                best_pd   = float(best["oos_pd"])
                best_name = str(best["name"])

        summary.append({
            "pair":        pair,
            "s1_pass":     len(s1) if s1 is not None else 0,
            "mc_pass":     int(s2["passed_mc"].sum()) if s2 is not None else 0,
            "oos_winners": oos_winners,
            "best_oos_pd": round(best_pd, 2),
            "best_config": best_name,
        })

    print("\n" + "="*60)
    print("SUMMARY — ATR P&F sweep")
    print("="*60)
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    df_sum.to_csv(OUT_DIR / "summary.csv", index=False)

    # Fixed-box reference for comparison
    ref = {"GBP_JPY": 71.6, "USD_JPY": 68.5, "EUR_JPY": 39.9, "GBP_USD": 15.4}
    print("\nComparison vs fixed-box OOS baseline:")
    for row in summary:
        p = row["pair"]
        base = ref.get(p, 0.0)
        delta = row["best_oos_pd"] - base
        sym = "🟢" if delta > 0 else ("🟡" if delta > -10 else "🔴")
        print(f"  {sym} {p}: ATR best={row['best_oos_pd']:.1f} p/d  "
              f"vs fixed={base:.1f}  Δ={delta:+.1f}")


if __name__ == "__main__":
    main()
