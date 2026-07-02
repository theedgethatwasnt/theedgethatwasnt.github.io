"""
GBP_USD small-box FIFO-Trends sweep — box sizes [2, 3, 4, 5] pips.

Motivation: ATR sweep (atr_pnf experiment) found GBP_USD ATR best = 24.9 p/d
(vs 15.4 p/d for fixed b=5) with mult=0.5 × ATR(20) ≈ 2.5-pip boxes.
This sweep tests whether the gain is real and ATR-specific, or whether a
simple fixed small-box config captures the same or better edge.

Box sizes [2, 3, 4] are new territory — original 12-pair sweep only tested
[5, 10, 15, 20, 30]. Box=5 is included as the known baseline.

IS P90 spread gate = 2.4p (same as original sweep, hardcoded per R5).
P50 spread = 1.8p — with box=2, the spread is 90% of box size, so the
sweep must find trail_d ≥ 2 or col-SMA exits to be profitable.

SOP compliance: same as backtest_fifo_pnf.py (R1-R9).

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/fifo_trends/backtest_gbpusd_small_box.py
"""

import time, sys, os
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE     = Path(__file__).resolve().parents[3]
BA_PATH  = BASE / "data/m5_ba/GBP_USD_M5_BA.parquet"
OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAIR     = "GBP_USD"
PIP      = 0.0001
IS_FRAC  = 0.70
SPREAD_GATE_PIPS = 2.4   # IS P90, hardcoded (R5)

# ── Parameter space ────────────────────────────────────────────────────────────
BOX_SIZES  = np.array([2, 3, 4, 5],           dtype=np.int32)  # pips (new: 2,3,4)
REVERSALS  = np.array([1, 2, 3],              dtype=np.int32)
MIN_COLS   = np.array([2, 3, 4, 5, 6, 8],    dtype=np.int32)
ENTRY_TYPES = [0, 1]

EXIT_DEFS = [
    (0, 3, 2), (1, 5, 3), (2, 8, 4),
    (3, 0, 0),
    (4, 1, 0), (5, 2, 0), (6, 3, 0),
    (7, 3, 0), (8, 5, 0), (9, 8, 0),
    (10,1, 3),(11,1, 5),(12,2, 3),(13,2, 5),(14,3, 5),
]
EXIT_NAMES = [
    "X1_3","X1_5","X1_8","X2",
    "X3b_1","X3b_2","X3b_3",
    "X7_3","X7_5","X7_8",
    "X3c_1_3","X3c_1_5","X3c_2_3","X3c_2_5","X3c_3_5",
]
# Total: 4 × 3 × 6 × 2 × 15 = 2,160

MAX_TRADES = 20_000
MAX_K      = 10


def build_configs():
    rows = []
    for bs in BOX_SIZES:
        for rv in REVERSALS:
            for nc in MIN_COLS:
                for et in ENTRY_TYPES:
                    for (xt, xp1, xp2) in EXIT_DEFS:
                        rows.append((bs, rv, nc, et, xt, xp1, xp2))
    return np.array(rows, dtype=np.int32)


def load_data():
    assert BA_PATH.exists(), f"BA parquet missing: {BA_PATH}"
    df      = pd.read_parquet(BA_PATH)
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = ((df["ask_c"] - df["bid_c"]) / PIP).values.astype(np.float64)
    n       = len(df)
    is_end  = int(n * IS_FRAC)
    p90     = float(np.percentile(spreads[:is_end], 90))
    print(f"  Bars={n:,}  IS={is_end:,}  OOS={n-is_end:,}")
    print(f"  IS spread P90={p90:.2f}p  P50={float(np.percentile(spreads[:is_end],50)):.2f}p  gate={SPREAD_GATE_PIPS}p")
    chunk0 = is_end // 3
    chunk1 = 2 * (is_end // 3)
    chunks = np.zeros(n, dtype=np.int8)
    chunks[chunk0:chunk1] = 1
    chunks[chunk1:is_end] = 2
    chunks[is_end:]       = 3
    return opens, highs, lows, closes, spreads, chunks, is_end, n


# ── Numba kernel (identical logic to run_all_pairs.py — R6) ───────────────────

@nb.njit(inline='always')
def col_sma(hist, ptr, n_valid, k):
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
    configs, spread_gate, pip, is_end,
    trade_pnl, trade_chunk, trade_cnt,
):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        bs_pips = configs[ci, 0]
        rev     = configs[ci, 1]
        n_min   = configs[ci, 2]
        entry_t = configs[ci, 3]
        exit_t  = configs[ci, 4]
        xp1     = configs[ci, 5]
        xp2     = configs[ci, 6]

        bs = bs_pips * pip

        pnf_idx   = 0
        pnf_level = 0.0
        pnf_dir   = 0
        col_count = 0
        prev_col  = 0

        col_hist     = np.zeros(MAX_K, dtype=np.float64)
        col_hist_ptr = 0
        col_hist_n   = 0

        pos      = 0
        entry_px = 0.0
        hw_level = 0.0
        pending  = 0
        t_cnt    = 0

        for i in range(N_BARS):
            opn = opens[i]; hi = highs[i]; lo = lows[i]
            cl  = closes[i]; sp = spreads[i]; ck = bar_chunks[i]

            bull = (cl >= opn)
            p1   = hi if bull else lo
            p2   = lo if bull else hi

            did_reverse_p1 = False; did_reverse_p2 = False
            prev_col_p1 = 0;        prev_col_p2 = 0

            for tick in range(2):
                px = p1 if tick == 0 else p2
                if pnf_dir == 0:
                    pnf_idx = int(px / bs); pnf_level = pnf_idx * bs
                    pnf_dir = 1; col_count = 1
                    continue
                delta = int(px / bs) - pnf_idx
                if pnf_dir == 1:
                    if delta >= 1:
                        pnf_idx += delta; pnf_level = pnf_idx * bs; col_count += delta
                    elif delta <= -rev:
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col; col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1
                        pnf_dir = -1; pnf_idx += delta; pnf_level = pnf_idx * bs
                        col_count = -delta
                        if tick == 0: did_reverse_p1 = True; prev_col_p1 = prev_col
                        else:         did_reverse_p2 = True; prev_col_p2 = prev_col
                elif pnf_dir == -1:
                    if delta <= -1:
                        pnf_idx += delta; pnf_level = pnf_idx * bs; col_count += (-delta)
                    elif delta >= rev:
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col; col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1
                        pnf_dir = 1; pnf_idx += delta; pnf_level = pnf_idx * bs
                        col_count = delta
                        if tick == 0: did_reverse_p1 = True; prev_col_p1 = prev_col
                        else:         did_reverse_p2 = True; prev_col_p2 = prev_col

            did_reverse     = did_reverse_p1 or did_reverse_p2
            prev_col_at_rev = prev_col_p1 if did_reverse_p1 else prev_col_p2

            if pos == 1:
                if pnf_dir == 1 and pnf_level > hw_level: hw_level = pnf_level
            elif pos == -1:
                if pnf_dir == -1 and pnf_level < hw_level: hw_level = pnf_level

            exit_triggered = False; exit_px_val = 0.0

            if pos != 0:
                if exit_t <= 2:
                    tp_b = float(xp1); sl_b = float(xp2)
                    if pos == 1:
                        sl_p = entry_px - sl_b * bs; tp_p = entry_px + tp_b * bs
                        if lo <= sl_p: exit_px_val = sl_p; exit_triggered = True
                        elif hi >= tp_p: exit_px_val = tp_p; exit_triggered = True
                    else:
                        sl_p = entry_px + sl_b * bs; tp_p = entry_px - tp_b * bs
                        if hi >= sl_p: exit_px_val = sl_p; exit_triggered = True
                        elif lo <= tp_p: exit_px_val = tp_p; exit_triggered = True
                elif exit_t == 3:
                    if did_reverse and pnf_dir != pos:
                        exit_px_val = cl; exit_triggered = True
                elif 4 <= exit_t <= 6:
                    d = float(xp1)
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail: exit_px_val = trail; exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail: exit_px_val = trail; exit_triggered = True
                elif 7 <= exit_t <= 9:
                    k = xp1
                    if pnf_dir != pos:
                        sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                        if sma_k > 0.0 and col_count >= sma_k:
                            exit_px_val = cl; exit_triggered = True
                else:
                    d = float(xp1); k = xp2
                    if pos == 1:
                        trail = hw_level - d * bs
                        if lo <= trail: exit_px_val = trail; exit_triggered = True
                    else:
                        trail = hw_level + d * bs
                        if hi >= trail: exit_px_val = trail; exit_triggered = True
                    if not exit_triggered and pnf_dir != pos:
                        sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                        if sma_k > 0.0 and col_count >= sma_k:
                            exit_px_val = cl; exit_triggered = True

            if exit_triggered and t_cnt < MAX_TRADES:
                pnl_pips = (exit_px_val - entry_px) * pos / pip - sp
                trade_pnl[ci, t_cnt] = np.float32(pnl_pips)
                trade_chunk[ci, t_cnt] = ck
                t_cnt += 1
                pos = 0; entry_px = 0.0; hw_level = 0.0

            if pos == 0:
                if sp <= spread_gate:
                    if entry_t == 0:
                        if did_reverse and prev_col_at_rev >= n_min:
                            pos = pnf_dir; entry_px = cl; hw_level = pnf_level
                    else:
                        if did_reverse and prev_col_at_rev >= n_min: pending = pnf_dir
                        if did_reverse and pending != 0 and pnf_dir != pending: pending = 0
                        if pending != 0 and pnf_dir == pending and col_count > rev:
                            pos = pending; entry_px = cl; hw_level = pnf_level; pending = 0
                else:
                    if did_reverse and pending != 0 and pnf_dir != pending: pending = 0

        trade_cnt[ci] = t_cnt


# ── Validation pipeline (identical to backtest_fifo_pnf.py) ───────────────────

def stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, config_names):
    N = len(trade_cnt); results = []
    for ci in range(N):
        tc = trade_cnt[ci]
        if tc == 0: continue
        pnl = trade_pnl[ci, :tc]; chunk = trade_chunk[ci, :tc].astype(np.int32)
        is_mask = chunk <= 2; is_pnl = pnl[is_mask]
        if len(is_pnl) < 30: continue
        c0 = float(pnl[chunk==0].sum()); c1 = float(pnl[chunk==1].sum()); c2 = float(pnl[chunk==2].sum())
        if c0 <= 0 or c1 <= 0 or c2 <= 0: continue
        results.append({"ci": ci, "name": config_names[ci],
                        "is_pnl": round(float(is_pnl.sum()), 1),
                        "is_ntrd": int(is_mask.sum()),
                        "c0": round(c0,1), "c1": round(c1,1), "c2": round(c2,1)})
    return pd.DataFrame(results).sort_values("is_pnl", ascending=False) if results \
           else pd.DataFrame(columns=["ci","name","is_pnl","is_ntrd","c0","c1","c2"])


def mc_permutation_test(pnl_arr, n_shuffles=1000, seed=42):
    rng = np.random.default_rng(seed); actual = float(pnl_arr.sum())
    perms = np.empty(n_shuffles)
    for k in range(n_shuffles):
        signs = rng.choice([-1.0, 1.0], size=len(pnl_arr))
        perms[k] = float((np.abs(pnl_arr) * signs).sum())
    return float((perms >= actual).mean()), float(np.percentile(perms, 95))


def bootstrap_p5(pnl_arr, n_boot=2000, days_oos=610, seed=99):
    rng = np.random.default_rng(seed); n = len(pnl_arr)
    if n == 0: return 0.0
    sums = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sums[k] = float(pnl_arr[idx].sum()) / days_oos
    return float(np.percentile(sums, 5))


def stage2_mc_bootstrap(stage1_df, trade_pnl, trade_chunk, trade_cnt, top_n=200):
    candidates = stage1_df.head(top_n); results = []
    for _, row in candidates.iterrows():
        ci = int(row["ci"]); tc = trade_cnt[ci]
        pnl = trade_pnl[ci, :tc].astype(np.float64); ck = trade_chunk[ci, :tc].astype(np.int32)
        is_pnl = pnl[ck <= 2]
        p_val, pct95 = mc_permutation_test(is_pnl); p5 = bootstrap_p5(is_pnl)
        results.append({"ci": ci, "name": row["name"], "is_pnl": row["is_pnl"],
                        "is_ntrd": row["is_ntrd"], "mc_pval": round(p_val, 3),
                        "mc_p95": round(pct95, 1), "bootstrap_p5": round(p5, 2),
                        "passed_mc": int(p_val < 0.05)})
    return pd.DataFrame(results).sort_values("bootstrap_p5", ascending=False)


def stage3_oos(stage2_df, trade_pnl, trade_chunk, trade_cnt, days_oos):
    survivors = stage2_df[stage2_df["passed_mc"] == 1]; results = []
    for _, row in survivors.iterrows():
        ci = int(row["ci"]); tc = trade_cnt[ci]
        pnl = trade_pnl[ci, :tc].astype(np.float64); ck = trade_chunk[ci, :tc].astype(np.int32)
        oos_pnl = pnl[ck == 3]; oos_tot = float(oos_pnl.sum())
        oos_ntrd = len(oos_pnl); oos_pd = oos_tot / days_oos if days_oos > 0 else 0.0
        results.append({"ci": ci, "name": row["name"], "is_pnl": row["is_pnl"],
                        "is_ntrd": row["is_ntrd"], "mc_pval": row["mc_pval"],
                        "bootstrap_p5": row["bootstrap_p5"],
                        "oos_pnl": round(oos_tot, 1), "oos_ntrd": oos_ntrd,
                        "oos_pd": round(oos_pd, 2), "oos_pass": int(oos_tot > 0 and oos_ntrd >= 10)})
    return pd.DataFrame(results).sort_values("oos_pd", ascending=False) if results \
           else pd.DataFrame(columns=["ci","name","is_pnl","is_ntrd","mc_pval",
                                      "bootstrap_p5","oos_pnl","oos_ntrd","oos_pd","oos_pass"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"GBP_USD Small-Box Sweep  (boxes={BOX_SIZES.tolist()}p)")
    print("=" * 60)

    configs = build_configs(); N_CONFIGS = len(configs)
    config_names = []
    entry_label = ["E1", "E2"]
    for ci in range(N_CONFIGS):
        bs, rv, nc, et, xt, xp1, xp2 = configs[ci]
        config_names.append(f"b{bs}_r{rv}_n{nc}_{entry_label[et]}_{EXIT_NAMES[xt]}")
    print(f"  Configs: {N_CONFIGS}")

    print("\nLoading data...")
    opens, highs, lows, closes, spreads, chunks, is_end, n_bars = load_data()
    days_oos = (n_bars - is_end) / 288.0

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS,               dtype=np.int32)

    print("\nWarm-up JIT compilation...")
    dp = np.zeros((1, MAX_TRADES), dtype=np.float32)
    dc = np.zeros((1, MAX_TRADES), dtype=np.int8)
    dk = np.zeros(1, dtype=np.int32)
    t0 = time.time()
    run_kernel(opens[:500], highs[:500], lows[:500], closes[:500],
               spreads[:500], chunks[:500], configs[:1],
               SPREAD_GATE_PIPS, PIP, min(500, is_end), dp, dc, dk)
    print(f"  Compiled in {time.time()-t0:.1f}s")

    print(f"\nRunning {N_CONFIGS} configs × {n_bars:,} bars...")
    t0 = time.time()
    run_kernel(opens, highs, lows, closes, spreads, chunks,
               configs, SPREAD_GATE_PIPS, PIP, is_end,
               trade_pnl, trade_chunk, trade_cnt)
    elapsed = time.time() - t0
    total_t = int(trade_cnt.sum())
    print(f"  Done in {elapsed:.1f}s  |  {total_t:,} trades  ({total_t/N_CONFIGS:.0f}/config avg)")

    # Per-box-size trade count sanity check
    print("\n  Trades by box size:")
    bs_vals = BOX_SIZES.tolist()
    n_per_box = N_CONFIGS // len(bs_vals)
    for i, b in enumerate(bs_vals):
        tc_slice = trade_cnt[i*n_per_box:(i+1)*n_per_box]
        print(f"    b={b}p : {tc_slice.sum():,} total trades, {tc_slice.sum()/n_per_box:.0f}/config avg")

    # ── Validation pipeline ───────────────────────────────────────────────────
    print("\n=== Stage 1: IS Walk-Forward Screen ===")
    s1 = stage1_is_screen(trade_pnl, trade_chunk, trade_cnt, config_names)
    s1.to_csv(OUT_DIR / "gbpusd_small_box_stage1.csv", index=False)
    print(f"  {len(s1)}/{N_CONFIGS} passed IS WF")

    # Stage 1 breakdown by box size
    for b in bs_vals:
        sub = s1[s1["name"].str.startswith(f"b{b}_")]
        print(f"    b={b}p : {len(sub)} passed")

    if len(s1) == 0:
        print("No configs passed Stage 1. No edge found."); return

    print(f"\n=== Stage 2: MC + Bootstrap (top 200) ===")
    s2 = stage2_mc_bootstrap(s1, trade_pnl, trade_chunk, trade_cnt)
    s2.to_csv(OUT_DIR / "gbpusd_small_box_stage2.csv", index=False)
    mc_n = s2["passed_mc"].sum()
    print(f"  {mc_n}/{len(s2)} passed MC (p<0.05)")

    if mc_n == 0:
        print("No configs passed MC."); return

    print(f"\n=== Stage 3: OOS (sealed — one-time evaluation) ===")
    s3 = stage3_oos(s2, trade_pnl, trade_chunk, trade_cnt, days_oos)
    s3.to_csv(OUT_DIR / "gbpusd_small_box_final.csv", index=False)
    oos_n = s3["oos_pass"].sum()
    print(f"  {oos_n}/{len(s3)} passed OOS")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if oos_n > 0:
        winners = s3[s3["oos_pass"] == 1]
        cols = ["name","oos_pd","oos_ntrd","is_pnl","bootstrap_p5","mc_pval"]
        print(winners[cols].head(15).to_string(index=False))

        print(f"\nOOS winners by box size:")
        for b in bs_vals:
            sub = winners[winners["name"].str.startswith(f"b{b}_")]
            if len(sub):
                best = sub.iloc[0]
                print(f"  🟢 b={b}p : {len(sub)} winners, best={best['oos_pd']:.2f} p/d  [{best['name']}]")
            else:
                print(f"  🔴 b={b}p : 0 winners")
    else:
        print("  🔴 No configs passed all stages.")

    print(f"\nFixed-box baseline (b=5, original sweep): 15.37 p/d")
    print(f"ATR sweep best (mult=0.5×ATR20):          24.91 p/d")
    if oos_n > 0:
        best = s3[s3["oos_pass"]==1].iloc[0]
        delta_vs_fixed = best["oos_pd"] - 15.37
        delta_vs_atr   = best["oos_pd"] - 24.91
        sym = "🟢" if best["oos_pd"] > 24.91 else ("🟡" if best["oos_pd"] > 15.37 else "🔴")
        print(f"{sym} Small-box best:                         {best['oos_pd']:.2f} p/d"
              f"  (Δ vs fixed={delta_vs_fixed:+.2f}, Δ vs ATR={delta_vs_atr:+.2f})")

    print(f"\nResults → {OUT_DIR}/gbpusd_small_box_*.csv")


if __name__ == "__main__":
    main()
