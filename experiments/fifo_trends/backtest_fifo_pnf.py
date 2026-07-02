"""
FIFO-Trends P&F Backtest — EUR/JPY M5
Single-pass Numba prange kernel over all 2,700 parameter combinations.

SOP adherence (CLAUDE.md §Backtest-Live Consistency SOP):
  R1  Closed bars only — bar[i] consumed after it closes; no future indexing
  R2  Within-bar sequence — bull=(close>=open) → HIGH then LOW; bear → LOW then HIGH
  R3  Mid OHLC for signals; spread = (ask_c - bid_c) / pip deducted explicitly
  R3a bid_h/bid_l/ask_h/ask_l NOT used — live get_candles never sends those
  R3b BA data fetched in full (fetch_m5_ba.py --years 5.5 --pairs EUR_JPY); no fallback
  R4  Incremental-only features — ring buffers only; no rolling over full array
  R4a col_count (in-progress column) never used in completed-column SMA (X7)
  R5  Spread gate = IS P90 = 2.5p hardcoded (computed from real IS data once)
  R6  P&F update logic (update_pnf) is the single source of truth for both
      this backtest and any future live deployment
  R8  OOS evaluated exactly once, after all IS/WF/MC gates pass
  R9  Divergences documented inline

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/fifo_trends/backtest_fifo_pnf.py
"""

import math, time, sys, os
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3]
BA_PATH  = BASE / "data/m5_ba/EUR_JPY_M5_BA.parquet"
OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAIR   = "EUR_JPY"
PIP    = 0.01          # JPY pair: 1 pip = 0.01
IS_FRAC = 0.70         # 70% IS, 30% OOS

# ─── Spread gate (R5) ─────────────────────────────────────────────────────────
# Computed once from IS data: np.percentile(sp[:is_end], 90) = 2.5 pips
# Hardcoded here — do NOT recompute from live or OOS data.
SPREAD_GATE_PIPS = 2.5

# ─── Parameter space → 2,700 configs ─────────────────────────────────────────
#   box sizes (pips): 5 values
#   reversals:        3 values
#   min_col:          6 values
#   entry type:       2 values  (0=E1 immediate, 1=E2 confirmation)
#   exit type:        15 variants
#   total: 5×3×6×2×15 = 2,700
BOX_SIZES  = np.array([5, 10, 15, 20, 30], dtype=np.int32)   # pips
REVERSALS  = np.array([1, 2, 3],            dtype=np.int32)
MIN_COLS   = np.array([2, 3, 4, 5, 6, 8],  dtype=np.int32)
ENTRY_TYPES = [0, 1]

# Exit variants (exit_type, exit_p1, exit_p2):
#   X1_3:  (0, 3, 2)   TP=3×box SL=2×box
#   X1_5:  (1, 5, 3)   TP=5×box SL=3×box
#   X1_8:  (2, 8, 4)   TP=8×box SL=4×box
#   X2:    (3, 0, 0)   exit on first adverse reversal
#   X3b_1: (4, 1, 0)   box-quantized trail d=1
#   X3b_2: (5, 2, 0)   box-quantized trail d=2
#   X3b_3: (6, 3, 0)   box-quantized trail d=3
#   X7_3:  (7, 3, 0)   col-SMA k=3 threshold exit
#   X7_5:  (8, 5, 0)   col-SMA k=5
#   X7_8:  (9, 8, 0)   col-SMA k=8
#   X3c_1_3:(10,1, 3)  X3b d=1 + X7 k=3 (whichever fires first)
#   X3c_1_5:(11,1, 5)  X3b d=1 + X7 k=5
#   X3c_2_3:(12,2, 3)  X3b d=2 + X7 k=3
#   X3c_2_5:(13,2, 5)  X3b d=2 + X7 k=5
#   X3c_3_5:(14,3, 5)  X3b d=3 + X7 k=5
EXIT_DEFS = [
    (0, 3, 2), (1, 5, 3), (2, 8, 4),
    (3, 0, 0),
    (4, 1, 0), (5, 2, 0), (6, 3, 0),
    (7, 3, 0), (8, 5, 0), (9, 8, 0),
    (10,1, 3),(11,1, 5),(12,2, 3),(13,2, 5),(14,3, 5),
]
EXIT_NAMES = [
    "X1_3","X1_5","X1_8",
    "X2",
    "X3b_1","X3b_2","X3b_3",
    "X7_3","X7_5","X7_8",
    "X3c_1_3","X3c_1_5","X3c_2_3","X3c_2_5","X3c_3_5",
]

def build_configs():
    rows = []
    for bi, bs in enumerate(BOX_SIZES):
        for ri, rv in enumerate(REVERSALS):
            for ni, nc in enumerate(MIN_COLS):
                for et in ENTRY_TYPES:
                    for (xt, xp1, xp2) in EXIT_DEFS:
                        rows.append((bs, rv, nc, et, xt, xp1, xp2))
    return np.array(rows, dtype=np.int32)

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load BA parquet; return mid OHLC + spread arrays.

    R3a: use only bid_c/ask_c for spread. Do not use bid_h/l ask_h/l.
    R3b: BA data must exist; no fallback spread.
    """
    assert BA_PATH.exists(), f"BA parquet missing: {BA_PATH}. Run fetch_m5_ba.py first."
    df = pd.read_parquet(BA_PATH)

    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = ((df["ask_c"] - df["bid_c"]) / PIP).values.astype(np.float64)  # pips

    n = len(df)
    is_end = int(n * IS_FRAC)

    # Verify IS P90 spread matches documented value (sanity check R5)
    p90 = float(np.percentile(spreads[:is_end], 90))
    print(f"  Bars: {n}  IS={is_end}  OOS={n-is_end}")
    print(f"  IS spread P90={p90:.2f}p  (gate={SPREAD_GATE_PIPS}p)")
    if abs(p90 - SPREAD_GATE_PIPS) > 0.3:
        print(f"  WARNING: IS P90 spread {p90:.2f} differs from hardcoded gate {SPREAD_GATE_PIPS}")

    # WF chunk boundaries (3 IS chunks + 1 OOS)
    chunk0_end = is_end // 3
    chunk1_end = 2 * (is_end // 3)
    chunk2_end = is_end
    # chunk3 = OOS

    chunks = np.zeros(n, dtype=np.int8)
    chunks[chunk0_end:chunk1_end] = 1
    chunks[chunk1_end:chunk2_end] = 2
    chunks[chunk2_end:]           = 3

    return opens, highs, lows, closes, spreads, chunks, is_end

# ─── Numba kernel ─────────────────────────────────────────────────────────────

MAX_K        = 10      # max ring buffer for completed column heights
MAX_TRADES   = 20000   # max trades per config

@nb.njit(inline='always')
def col_sma(hist, ptr, n_valid, k):
    """Mean of last min(k, n_valid) completed column heights from ring buffer.

    R4a: ring buffer contains only COMPLETED columns. Current in-progress
    column is never pushed here until it completes on reversal.
    """
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
    configs,          # (N_CONFIGS, 7): bs_pips, rev, n_min, entry_t, exit_t, xp1, xp2
    spread_gate,      # scalar pips
    pip,              # scalar price per pip
    is_end,           # last IS bar index + 1
    trade_pnl,        # (N_CONFIGS, MAX_TRADES) float32 output
    trade_chunk,      # (N_CONFIGS, MAX_TRADES) int8 output
    trade_cnt,        # (N_CONFIGS,) int32 output
):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        bs_pips  = configs[ci, 0]
        rev      = configs[ci, 1]
        n_min    = configs[ci, 2]
        entry_t  = configs[ci, 3]
        exit_t   = configs[ci, 4]
        xp1      = configs[ci, 5]   # tp_boxes or d or k
        xp2      = configs[ci, 6]   # sl_boxes or k2

        bs = bs_pips * pip          # box size in price units

        # P&F state
        pnf_idx   = 0              # integer box index — authoritative, no float drift
        pnf_level = 0.0            # derived: pnf_idx * bs — kept in sync for hw_level
        pnf_dir   = 0              # 0=uninit, +1=X(up), -1=O(down)
        col_count = 0
        prev_col  = 0

        # Completed-column ring buffer (R4a: in-progress col never pushed here)
        col_hist    = np.zeros(MAX_K, dtype=np.float64)
        col_hist_ptr = 0
        col_hist_n   = 0

        # Position state
        pos       = 0              # 0=flat, +1=long, -1=short
        entry_px  = 0.0            # mid price at entry (+ half-spread already applied)
        hw_level  = 0.0            # high-water P&F level for trailing stop

        # E2 pending entry
        pending   = 0              # 0=none, +1/-1 direction

        # Trade recording
        t_cnt = 0

        for i in range(N_BARS):
            opn = opens[i]
            hi  = highs[i]
            lo  = lows[i]
            cl  = closes[i]
            sp  = spreads[i]
            ck  = bar_chunks[i]

            # Within-bar price sequence (R2): bull=(close>=open) → HIGH first
            bull = (cl >= opn)
            p1   = hi if bull else lo
            p2   = lo if bull else hi

            # ── P&F update: process p1 then p2 ───────────────────────────
            did_reverse_p1 = False
            did_reverse_p2 = False
            prev_col_p1    = 0
            prev_col_p2    = 0

            for tick in range(2):
                px = p1 if tick == 0 else p2

                if pnf_dir == 0:
                    # First bar — initialize chart (R1: only on first bar)
                    pnf_idx   = int(px / bs)   # floor for positive prices
                    pnf_level = pnf_idx * bs   # derived
                    pnf_dir   = 1
                    col_count = 1
                    continue

                # Float-point fix: absolute box index comparison eliminates
                # accumulated drift from repeated pnf_level += delta*bs.
                delta = int(px / bs) - pnf_idx  # exact integer subtraction

                if pnf_dir == 1:
                    if delta >= 1:
                        # Continuation up
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count += delta
                    elif delta <= -rev:
                        # Reversal down — save completed column BEFORE resetting
                        prev_col = col_count
                        # Push completed column to ring buffer (R4a)
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K:
                            col_hist_n += 1
                        # Start new O column
                        pnf_dir   = -1
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count  = -delta    # abs(delta) boxes in new column
                        if tick == 0:
                            did_reverse_p1 = True
                            prev_col_p1    = prev_col
                        else:
                            did_reverse_p2 = True
                            prev_col_p2    = prev_col

                elif pnf_dir == -1:
                    if delta <= -1:
                        # Continuation down
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count += (-delta)
                    elif delta >= rev:
                        # Reversal up — save completed column
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K:
                            col_hist_n += 1
                        # Start new X column
                        pnf_dir   = 1
                        pnf_idx   += delta
                        pnf_level  = pnf_idx * bs
                        col_count  = delta
                        if tick == 0:
                            did_reverse_p1 = True
                            prev_col_p1    = prev_col
                        else:
                            did_reverse_p2 = True
                            prev_col_p2    = prev_col

            # Combined reversal flags (either tick)
            did_reverse = did_reverse_p1 or did_reverse_p2
            prev_col_at_rev = prev_col_p1 if did_reverse_p1 else prev_col_p2

            # ── Update high-water level for trailing stop ─────────────────
            if pos == 1:
                # Long: favorable direction is up (pnf_dir == +1)
                if pnf_dir == 1 and pnf_level > hw_level:
                    hw_level = pnf_level
            elif pos == -1:
                # Short: favorable direction is down (pnf_dir == -1)
                if pnf_dir == -1 and pnf_level < hw_level:
                    hw_level = pnf_level

            # ── EXIT logic ────────────────────────────────────────────────
            exit_triggered = False
            exit_px_val    = 0.0

            if pos != 0:
                if exit_t <= 2:
                    # X1: fixed TP / SL in box multiples
                    # Conservative: SL checked first (worst case within bar)
                    tp_b = float(xp1)
                    sl_b = float(xp2)
                    if pos == 1:
                        sl_price = entry_px - sl_b * bs
                        tp_price = entry_px + tp_b * bs
                        if lo <= sl_price:
                            exit_px_val   = sl_price
                            exit_triggered = True
                        elif hi >= tp_price:
                            exit_px_val   = tp_price
                            exit_triggered = True
                    else:
                        sl_price = entry_px + sl_b * bs
                        tp_price = entry_px - tp_b * bs
                        if hi >= sl_price:
                            exit_px_val   = sl_price
                            exit_triggered = True
                        elif lo <= tp_price:
                            exit_px_val   = tp_price
                            exit_triggered = True

                elif exit_t == 3:
                    # X2: exit on first adverse reversal
                    if did_reverse and pnf_dir != pos:
                        exit_px_val   = cl
                        exit_triggered = True

                elif 4 <= exit_t <= 6:
                    # X3b: box-quantized trailing stop
                    d = float(xp1)
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail:
                            exit_px_val   = trail
                            exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail:
                            exit_px_val   = trail
                            exit_triggered = True

                elif 7 <= exit_t <= 9:
                    # X7: exit when adverse column height >= col-SMA(k)
                    # R4a: col_hist contains only completed columns; col_count is in-progress
                    k = xp1
                    if pnf_dir != pos:
                        sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                        if sma_k > 0.0 and col_count >= sma_k:
                            exit_px_val   = cl
                            exit_triggered = True

                else:
                    # X3c: X3b trail OR X7 — whichever fires first
                    d = float(xp1)
                    k = xp2
                    # Trail check (like X3b)
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail:
                            exit_px_val   = trail
                            exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail:
                            exit_px_val   = trail
                            exit_triggered = True
                    # X7 check (if trail not yet triggered)
                    if not exit_triggered and pnf_dir != pos:
                        sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                        if sma_k > 0.0 and col_count >= sma_k:
                            exit_px_val   = cl
                            exit_triggered = True

            if exit_triggered and t_cnt < MAX_TRADES:
                # P&L: (exit - entry) × direction / pip − 1 full spread at entry (R3)
                pnl_pips = (exit_px_val - entry_px) * pos / pip - sp
                trade_pnl[ci, t_cnt]   = np.float32(pnl_pips)
                trade_chunk[ci, t_cnt] = ck
                t_cnt += 1
                pos       = 0
                entry_px  = 0.0
                hw_level  = 0.0

            # ── ENTRY logic ───────────────────────────────────────────────
            if pos == 0:
                can_enter = (sp <= spread_gate)   # R5

                if can_enter:
                    if entry_t == 0:
                        # E1: enter immediately on reversal if prev column ≥ n_min
                        if did_reverse and prev_col_at_rev >= n_min:
                            pos      = pnf_dir   # new column direction
                            entry_px = cl        # mid close; spread deducted in pnl (R3)
                            hw_level = pnf_level
                    else:
                        # E2: set pending on qualifying reversal
                        if did_reverse and prev_col_at_rev >= n_min:
                            pending = pnf_dir
                        # Clear pending on adverse reversal
                        if did_reverse and pending != 0 and pnf_dir != pending:
                            pending = 0
                        # Enter when pending confirmed (col_count > rev: at least 1 extra box)
                        if pending != 0 and pnf_dir == pending and col_count > rev:
                            pos      = pending
                            entry_px = cl        # mid close; spread deducted in pnl (R3)
                            hw_level = pnf_level
                            pending  = 0
                else:
                    # Spread too wide — clear any pending (wait for better spread)
                    if did_reverse and pending != 0 and pnf_dir != pending:
                        pending = 0

        trade_cnt[ci] = t_cnt

# ─── Post-processing ──────────────────────────────────────────────────────────

def stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, configs, config_names):
    """IS walk-forward screen: all 3 IS chunks profitable + min 30 IS trades total."""
    print("\n=== Stage 1: IS Walk-Forward Screen ===")
    N = len(trade_cnt)
    results = []

    for ci in range(N):
        tc = trade_cnt[ci]
        if tc == 0:
            continue
        pnl   = trade_pnl[ci, :tc]
        chunk = trade_chunk[ci, :tc].astype(np.int32)

        is_mask = chunk <= 2   # chunks 0,1,2 = IS
        is_pnl  = pnl[is_mask]
        if len(is_pnl) < 30:
            continue

        # All 3 IS chunks must be profitable
        chunk0_sum = pnl[chunk == 0].sum()
        chunk1_sum = pnl[chunk == 1].sum()
        chunk2_sum = pnl[chunk == 2].sum()
        if chunk0_sum <= 0 or chunk1_sum <= 0 or chunk2_sum <= 0:
            continue

        is_total = is_pnl.sum()
        is_mean  = is_pnl.mean()
        is_ntrd  = int(is_mask.sum())
        results.append({
            "ci": ci, "name": config_names[ci],
            "is_pnl": round(float(is_total), 1),
            "is_mean": round(float(is_mean), 3),
            "is_ntrd": is_ntrd,
            "c0": round(float(chunk0_sum), 1),
            "c1": round(float(chunk1_sum), 1),
            "c2": round(float(chunk2_sum), 1),
        })

    if results:
        df = pd.DataFrame(results).sort_values("is_pnl", ascending=False)
    else:
        df = pd.DataFrame(columns=["ci","name","is_pnl","is_mean","is_ntrd","c0","c1","c2"])
    out = OUT_DIR / "eur_jpy_stage1.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} / {N} passed  →  {out}")
    return df


def mc_permutation_test(pnl_arr, n_shuffles=1000, seed=42):
    """MC permutation: shuffle trade signs, return p-value and 95th pct."""
    rng   = np.random.default_rng(seed)
    actual = float(pnl_arr.sum())
    perm_sums = np.empty(n_shuffles)
    for k in range(n_shuffles):
        signs = rng.choice([-1.0, 1.0], size=len(pnl_arr))
        perm_sums[k] = float((np.abs(pnl_arr) * signs).sum())
    p_val  = float((perm_sums >= actual).mean())
    pct95  = float(np.percentile(perm_sums, 95))
    return p_val, pct95


def bootstrap_p5(pnl_arr, n_boot=2000, days_oos=610, seed=99):
    """Bootstrap daily P&L distribution, return P5 pips/day."""
    rng = np.random.default_rng(seed)
    n   = len(pnl_arr)
    if n == 0:
        return 0.0
    sums = np.empty(n_boot)
    for k in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        sums[k] = float(pnl_arr[idx].sum()) / days_oos
    return float(np.percentile(sums, 5))


def stage2_mc_bootstrap(stage1_df, trade_pnl, trade_chunk, trade_cnt,
                        config_names, top_n=200):
    """MC permutation + bootstrap P5 on IS data of top configs."""
    print(f"\n=== Stage 2: MC + Bootstrap (top {top_n}) ===")
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
            "ci": ci, "name": config_names[ci],
            "is_pnl": row["is_pnl"],
            "is_ntrd": row["is_ntrd"],
            "mc_pval": round(p_val, 3),
            "mc_p95": round(pct95, 1),
            "bootstrap_p5": round(p5, 2),
            "passed_mc": int(p_val < 0.05),
        })

    df = pd.DataFrame(results).sort_values("bootstrap_p5", ascending=False)
    out = OUT_DIR / "eur_jpy_stage2.csv"
    df.to_csv(out, index=False)
    mc_passed = df["passed_mc"].sum()
    print(f"  {mc_passed} / {len(df)} passed MC (p<0.05)  →  {out}")
    return df


def stage3_oos(stage2_df, trade_pnl, trade_chunk, trade_cnt,
               config_names, days_oos=610):
    """OOS evaluation — touched exactly once (R8)."""
    print("\n=== Stage 3: OOS (sealed — one-time evaluation) ===")
    survivors = stage2_df[stage2_df["passed_mc"] == 1]
    print(f"  Evaluating {len(survivors)} survivors on OOS")

    results = []
    for _, row in survivors.iterrows():
        ci  = int(row["ci"])
        tc  = trade_cnt[ci]
        pnl = trade_pnl[ci, :tc].astype(np.float64)
        ck  = trade_chunk[ci, :tc].astype(np.int32)

        oos_pnl  = pnl[ck == 3]
        oos_ntrd = len(oos_pnl)
        oos_tot  = float(oos_pnl.sum())
        oos_pd   = oos_tot / days_oos if days_oos > 0 else 0.0
        oos_pass = int(oos_tot > 0 and oos_ntrd >= 10)

        results.append({
            "ci": ci, "name": config_names[ci],
            "is_pnl": row["is_pnl"],
            "is_ntrd": row["is_ntrd"],
            "mc_pval": row["mc_pval"],
            "bootstrap_p5": row["bootstrap_p5"],
            "oos_pnl": round(oos_tot, 1),
            "oos_ntrd": oos_ntrd,
            "oos_pd": round(oos_pd, 2),
            "oos_pass": oos_pass,
        })

    df = pd.DataFrame(results).sort_values("oos_pd", ascending=False)
    out = OUT_DIR / "eur_jpy_final.csv"
    df.to_csv(out, index=False)
    oos_pass = df["oos_pass"].sum()
    print(f"  {oos_pass} / {len(df)} passed OOS  →  {out}")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"FIFO-Trends P&F Backtest — {PAIR}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    opens, highs, lows, closes, spreads, bar_chunks, is_end = load_data()
    N_BARS = len(opens)

    # Build configs
    configs = build_configs()
    N_CONFIGS = len(configs)
    print(f"  Configs: {N_CONFIGS}  Bars: {N_BARS}")

    # Build config name strings for reporting
    entry_label = ["E1", "E2"]
    config_names = []
    for ci in range(N_CONFIGS):
        bs, rv, nc, et, xt, xp1, xp2 = configs[ci]
        config_names.append(
            f"b{bs}_r{rv}_n{nc}_{entry_label[et]}_{EXIT_NAMES[xt]}"
        )

    # Pre-allocate output arrays
    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS,               dtype=np.int32)

    # Warm up Numba (compile on a tiny slice)
    print("\nWarm-up JIT compilation (single config, 500 bars)...")
    dummy_cfg = configs[:1].copy()
    dp = np.zeros((1, MAX_TRADES), dtype=np.float32)
    dc = np.zeros((1, MAX_TRADES), dtype=np.int8)
    dk = np.zeros(1, dtype=np.int32)
    t0 = time.time()
    run_kernel(
        opens[:500], highs[:500], lows[:500], closes[:500],
        spreads[:500], bar_chunks[:500],
        dummy_cfg, SPREAD_GATE_PIPS, PIP, min(500, is_end),
        dp, dc, dk
    )
    print(f"  Compiled in {time.time()-t0:.1f}s")

    # Full run
    print(f"\nRunning {N_CONFIGS} configs × {N_BARS} bars (prange parallel)...")
    t0 = time.time()
    run_kernel(
        opens, highs, lows, closes, spreads, bar_chunks,
        configs, SPREAD_GATE_PIPS, PIP, is_end,
        trade_pnl, trade_chunk, trade_cnt
    )
    elapsed = time.time() - t0
    total_trades = int(trade_cnt.sum())
    print(f"  Done in {elapsed:.1f}s  |  {total_trades:,} total trades recorded")
    print(f"  Avg {total_trades/N_CONFIGS:.0f} trades/config")

    # Quick sanity: check a few configs
    for ci in [0, 100, 500, 1000, 2699]:
        tc = trade_cnt[ci]
        if tc > 0:
            sub = trade_pnl[ci, :tc]
            print(f"  Config {ci} ({config_names[ci]}): {tc} trades  "
                  f"sum={sub.sum():.0f}p  mean={sub.mean():.2f}p/trade")

    # ─── Validation pipeline ───────────────────────────────────────────────
    stage1_df = stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, configs, config_names)

    if len(stage1_df) == 0:
        print("\nNo configs passed Stage 1. Experiment complete — no edge found.")
        return

    stage2_df = stage2_mc_bootstrap(stage1_df, trade_pnl, trade_chunk, trade_cnt, config_names)

    if stage2_df["passed_mc"].sum() == 0:
        print("\nNo configs passed MC test. Experiment complete — no edge found.")
        return

    # OOS — R8: sealed, evaluated exactly once
    stage3_df = stage3_oos(stage2_df, trade_pnl, trade_chunk, trade_cnt, config_names)

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total configs:      {N_CONFIGS}")
    print(f"  Stage 1 (IS WF):    {len(stage1_df)}")
    print(f"  Stage 2 (MC+boot):  {stage2_df['passed_mc'].sum()}")
    print(f"  Stage 3 (OOS):      {stage3_df['oos_pass'].sum()}")

    oos_winners = stage3_df[stage3_df["oos_pass"] == 1]
    if len(oos_winners) > 0:
        print("\nTop OOS configs:")
        print(oos_winners[["name","is_pnl","oos_pnl","oos_pd","bootstrap_p5","mc_pval"]].head(10).to_string(index=False))
    else:
        print("\n🔴 No configs passed all stages. No deployable edge found.")

    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
