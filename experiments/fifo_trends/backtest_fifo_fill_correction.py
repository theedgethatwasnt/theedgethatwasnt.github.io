"""
FIFO-Trends P&F — Bar-Close Fill Correction Test
=================================================
Compares trail-level fills (current backtest assumption) vs bar-close fills
(realistic for live manual management) for the validated ft and ft2 configs.

MOTIVATION:
  X3b/X3c trail exit currently: exit_px_val = trail  (stop-order fill)
  Live manual management:        exit_px_val = cl     (M5 bar close when lo<=trail)
  Gap: M5 bar's close is typically 1-4 pips below trail when trail triggers.

KEY QUESTION:
  Do the ft2 configs (2-box = 10-pip trail) retain positive OOS p/d after
  correcting for bar-close fills? ft configs (1-box = 5-pip trail) are expected
  to fail — they already failed live. ft2 is the candidate for re-deployment.

CONFIGS TESTED:
  GBP_JPY ft:   b5_r1_n4_E2_X3c_1_5 → OOS 71.6 p/d (trail-level)
  GBP_JPY ft2:  b5_r1_n4_E2_X3c_2_5 → OOS 35.3 p/d (trail-level)
  USD_JPY ft:   b5_r1_n3_E2_X3c_1_5 → OOS 68.5 p/d (trail-level)
  USD_JPY ft2:  b5_r1_n3_E2_X3c_2_5 → OOS 32.7 p/d (trail-level)
  EUR_JPY ft:   b5_r1_n3_E2_X3c_1_5 → OOS 39.9 p/d (trail-level)
  GBP_USD sb:   b2_r3_n8_E2_X3c_1_5 → OOS 54.5 p/d (trail-level)

Also sweeps nearby trail distances (d=1,2,3,4) with bar-close fills to find
the optimal trail_d after correcting fills.

Run:
  python3 research/experiments/fifo_trends/backtest_fifo_fill_correction.py
"""

import sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from numba import prange

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"
OUT    = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

IS_FRAC  = 0.70
MAX_K    = 10
N_WF     = 3
MIN_IS_TRADES = 5        # per WF chunk
M5_PER_TRADING_DAY = 288.0

PIP = {
    "GBP_JPY": 0.01, "USD_JPY": 0.01,
    "EUR_JPY": 0.01, "GBP_USD": 0.0001,
}
SP_GATE = {   # IS P90 hardcoded (SOP R5)
    "GBP_JPY": 4.00, "USD_JPY": 2.10,
    "EUR_JPY": 2.50, "GBP_USD": 2.40,
}

# Target configs: (pair, b_pips, rev, n_min, entry_t, xp1, xp2, label)
# exit_t=11 → X3c_1_5 (d=1, k=5); exit_t=13 → X3c_2_5 (d=2, k=5)
TARGETS = [
    ("GBP_JPY", 5, 1, 4, 1, 1, 5, "GBP_JPY_ft"),
    ("GBP_JPY", 5, 1, 4, 1, 2, 5, "GBP_JPY_ft2"),
    ("USD_JPY", 5, 1, 3, 1, 1, 5, "USD_JPY_ft"),
    ("USD_JPY", 5, 1, 3, 1, 2, 5, "USD_JPY_ft2"),
    ("EUR_JPY", 5, 1, 3, 1, 1, 5, "EUR_JPY_ft"),
    ("EUR_JPY", 5, 1, 3, 1, 2, 5, "EUR_JPY_ft2"),
    ("GBP_USD", 2, 3, 8, 1, 1, 5, "GBP_USD_sb"),
    ("GBP_USD", 2, 3, 8, 1, 2, 5, "GBP_USD_sb2"),
]


@nb.njit(inline="always")
def col_sma(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    s = 0.0
    for j in range(count):
        s += hist[(ptr - 1 - j) % MAX_K]
    return s / count


@nb.njit
def run_config(opens, highs, lows, closes, spreads,
               bs_pips, rev, n_min, entry_t, xp1, xp2,
               pip, spread_gate, is_end, fill_at_close):
    """
    Full P&F simulation. fill_at_close=True → trail exit fills at bar close.

    Returns:
        pnl[n], is_flag[n], exit_type[n]
        exit_type: 0=trail, 1=X7
    """
    N  = len(opens)
    bs = bs_pips * pip

    pnf_idx=0; pnf_level=0.0; pnf_dir=0; col_count=0; prev_col=0
    col_hist = np.zeros(MAX_K, np.float64)
    col_hist_ptr = 0; col_hist_n = 0
    pos=0; entry_px=0.0; hw_level=0.0; pending=0

    MAX_T = N // 5 + 100
    pnl_arr  = np.empty(MAX_T, np.float64)
    flag_arr = np.empty(MAX_T, np.int8)
    type_arr = np.empty(MAX_T, np.int8)   # 0=trail, 1=X7
    n_t = 0

    for i in range(N):
        opn = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]
        sp  = spreads[i]
        is_bar = 1 if i < is_end else 0

        bull = (cl >= opn)
        p1 = hi if bull else lo
        p2 = lo if bull else hi

        did_reverse = False; prev_col_at_rev = 0

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
                    if not did_reverse:
                        did_reverse = True; prev_col_at_rev = prev_col

            elif pnf_dir == -1:
                if delta <= -1:
                    pnf_idx += delta; pnf_level = pnf_idx * bs; col_count += (-delta)
                elif delta >= rev:
                    prev_col = col_count
                    col_hist[col_hist_ptr % MAX_K] = prev_col; col_hist_ptr += 1
                    if col_hist_n < MAX_K: col_hist_n += 1
                    pnf_dir = 1; pnf_idx += delta; pnf_level = pnf_idx * bs
                    col_count = delta
                    if not did_reverse:
                        did_reverse = True; prev_col_at_rev = prev_col

        # HWM update
        if pos == 1 and pnf_dir == 1 and pnf_level > hw_level:
            hw_level = pnf_level
        elif pos == -1 and pnf_dir == -1 and pnf_level < hw_level:
            hw_level = pnf_level

        # Exit
        exit_triggered = False; exit_px = 0.0; exit_kind = 0

        if pos != 0:
            d = float(xp1); k = int(xp2)
            # Trail component (X3b / X3c trail)
            if pos == 1:
                trail = hw_level - d * bs
                if lo <= trail:
                    exit_px = cl if fill_at_close else trail
                    exit_triggered = True; exit_kind = 0
            else:
                trail = hw_level + d * bs
                if hi >= trail:
                    exit_px = cl if fill_at_close else trail
                    exit_triggered = True; exit_kind = 0

            # X7 component — always bar-close (unchanged)
            if not exit_triggered and pnf_dir != pos:
                sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                if sma_k > 0.0 and col_count >= sma_k:
                    exit_px = cl
                    exit_triggered = True; exit_kind = 1

        if exit_triggered and n_t < MAX_T:
            pnl_arr[n_t]  = pos * (exit_px - entry_px) / pip - sp
            flag_arr[n_t] = is_bar
            type_arr[n_t] = exit_kind
            n_t += 1
            pos = 0; entry_px = 0.0; hw_level = 0.0

        # Entry
        if pos == 0:
            if sp <= spread_gate:
                if entry_t == 0:
                    if did_reverse and prev_col_at_rev >= n_min:
                        pos = pnf_dir; entry_px = cl; hw_level = pnf_level
                else:
                    if did_reverse and prev_col_at_rev >= n_min:
                        pending = pnf_dir
                    if did_reverse and pending != 0 and pnf_dir != pending:
                        pending = 0
                    if pending != 0 and pnf_dir == pending and col_count > rev:
                        pos = pending; entry_px = cl; hw_level = pnf_level; pending = 0
            else:
                if did_reverse and pending != 0 and pnf_dir != pending:
                    pending = 0

    return pnl_arr[:n_t], flag_arr[:n_t], type_arr[:n_t]


def analyze(pair, label, bs_pips, rev, n_min, entry_t, xp1, xp2,
            opens, highs, lows, closes, spreads, is_end, n_total, pip, sp_gate):
    oos_days = (n_total - is_end) / M5_PER_TRADING_DAY
    is_days  = is_end / M5_PER_TRADING_DAY
    chunk_sz = is_end // N_WF

    rows = []
    for fill_mode, fill_close in [("trail-fill", False), ("bar-close", True)]:
        pnl, flags, etypes = run_config(
            opens, highs, lows, closes, spreads,
            bs_pips, rev, n_min, entry_t, xp1, xp2,
            pip, sp_gate, is_end, fill_close,
        )

        is_mask  = flags == 1
        oos_mask = flags == 0
        is_pnls  = pnl[is_mask]
        oos_pnls = pnl[oos_mask]
        is_types  = etypes[is_mask]
        oos_types = etypes[oos_mask]

        # WF: split IS trades into 3 sequential chunks by bar index
        # Approximate by splitting IS trade sequence evenly
        wf_ok = True
        for k in range(N_WF):
            s = k * (len(is_pnls) // N_WF)
            e = (k+1) * (len(is_pnls) // N_WF) if k < 2 else len(is_pnls)
            chunk_pnl = is_pnls[s:e]
            if len(chunk_pnl) < MIN_IS_TRADES or chunk_pnl.sum() <= 0:
                wf_ok = False; break

        is_wr   = (is_pnls  > 0).mean() if len(is_pnls)  > 0 else 0.0
        oos_wr  = (oos_pnls > 0).mean() if len(oos_pnls) > 0 else 0.0
        is_avgw  = is_pnls[is_pnls > 0].mean()  if (is_pnls > 0).any()  else 0.0
        is_avgl  = is_pnls[is_pnls <= 0].mean() if (is_pnls <= 0).any() else 0.0
        oos_avgw = oos_pnls[oos_pnls > 0].mean()  if (oos_pnls > 0).any()  else 0.0
        oos_avgl = oos_pnls[oos_pnls <= 0].mean() if (oos_pnls <= 0).any() else 0.0

        # Trail exit percentage
        is_trail_pct  = (is_types  == 0).mean() if len(is_types)  > 0 else 0.0
        oos_trail_pct = (oos_types == 0).mean() if len(oos_types) > 0 else 0.0

        # Gap: how much worse is bar-close vs trail fill for trail exits
        # (computed only for bar-close mode vs trail-fill baseline)
        is_pd  = is_pnls.sum()  / is_days  if len(is_pnls)  > 0 else 0.0
        oos_pd = oos_pnls.sum() / oos_days if len(oos_pnls) > 0 else 0.0

        rows.append({
            "pair": pair, "config": label, "fill": fill_mode, "wf": wf_ok,
            "is_pd":   round(is_pd,  1),
            "oos_pd":  round(oos_pd, 1),
            "is_n":    len(is_pnls),
            "oos_n":   len(oos_pnls),
            "is_wr":   round(is_wr,  3),
            "oos_wr":  round(oos_wr, 3),
            "is_avgw": round(is_avgw,  2),
            "is_avgl": round(is_avgl,  2),
            "oos_avgw":round(oos_avgw, 2),
            "oos_avgl":round(oos_avgl, 2),
            "is_trail%":  round(is_trail_pct,  3),
            "oos_trail%": round(oos_trail_pct, 3),
        })

    return rows


def main():
    print("=" * 78)
    print("  FIFO-Trends — Bar-Close Fill Correction (ft vs ft2 configs)")
    print("=" * 78)
    print()

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    _dc = np.cumsum(np.random.randn(5000)) * 0.01 + 150.0
    _dh = _dc + 0.05; _dl = _dc - 0.05
    _ds = np.full(5000, 0.02)
    run_config(_dc, _dh, _dl, _dc, _ds, 5, 1, 3, 1, 1, 5, 0.01, 2.5, 3500, False)
    run_config(_dc, _dh, _dl, _dc, _ds, 5, 1, 3, 1, 1, 5, 0.01, 2.5, 3500, True)
    print("  Done.\n")

    all_rows = []
    loaded   = {}   # cache loaded data per pair

    for (pair, bs, rev, n_min, entry_t, xp1, xp2, label) in TARGETS:
        pip     = PIP[pair]
        sp_gate = SP_GATE[pair]

        if pair not in loaded:
            path = BA_DIR / f"{pair}_M5_BA.parquet"
            df   = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
            op   = df["open"].values.astype(np.float64)
            hi   = df["high"].values.astype(np.float64)
            lo   = df["low"].values.astype(np.float64)
            cl   = df["close"].values.astype(np.float64)
            sp   = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)
            n    = len(op)
            is_e = int(n * IS_FRAC)
            loaded[pair] = (op, hi, lo, cl, sp, n, is_e)
            print(f"  Loaded {pair}: {n:,} bars  IS={is_e:,}  "
                  f"OOS={(n-is_e)/M5_PER_TRADING_DAY:.0f}d  sp_gate={sp_gate:.2f}p")

        op, hi, lo, cl, sp, n, is_e = loaded[pair]
        rows = analyze(pair, label, bs, rev, n_min, entry_t, xp1, xp2,
                       op, hi, lo, cl, sp, is_e, n, pip, sp_gate)
        all_rows.extend(rows)

    # ── Print results ───────────────────────────────────────────────────────────
    print()
    print(f"  {'Config':<18} {'Fill':<11} {'WF':>3} | "
          f"{'IS p/d':>7} {'IS_n':>5} {'IS_WR':>6} {'avg_w':>6} {'avg_l':>6} "
          f"{'trail%':>7} | "
          f"{'OOS p/d':>8} {'OOS_n':>5} {'OOS_WR':>6} {'trail%':>7}")
    print(f"  {'-'*18} {'-'*11} {'-'*3}-+-"
          f"{'-'*7}-{'-'*5}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*7}-+-"
          f"{'-'*8}-{'-'*5}-{'-'*6}-{'-'*7}")

    prev_config = None
    for r in all_rows:
        if r["config"] != prev_config and prev_config is not None:
            print()
        prev_config = r["config"]
        wf_str = "✓" if r["wf"] else "✗"
        print(f"  {r['config']:<18} {r['fill']:<11} {wf_str:>3} | "
              f"{r['is_pd']:>7.1f} {r['is_n']:>5} {r['is_wr']:>6.1%} "
              f"{r['is_avgw']:>6.2f} {r['is_avgl']:>6.2f} {r['is_trail%']:>7.1%} | "
              f"{r['oos_pd']:>8.1f} {r['oos_n']:>5} {r['oos_wr']:>6.1%} "
              f"{r['oos_trail%']:>7.1%}")

    # ── Delta summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  Fill correction impact (bar-close minus trail-fill):")
    print(f"  {'Config':<18} | {'Δ IS p/d':>10} {'Δ OOS p/d':>10} "
          f"{'OOS trail-fill':>15} {'OOS bar-close':>14}")
    print(f"  {'-'*18}-+-{'-'*10}-{'-'*10}-{'-'*15}-{'-'*14}")

    configs = list(dict.fromkeys(r["config"] for r in all_rows))
    deploy_candidates = []
    for cfg in configs:
        sub  = [r for r in all_rows if r["config"] == cfg]
        tf_r = next(r for r in sub if r["fill"] == "trail-fill")
        bc_r = next(r for r in sub if r["fill"] == "bar-close")
        d_is  = bc_r["is_pd"]  - tf_r["is_pd"]
        d_oos = bc_r["oos_pd"] - tf_r["oos_pd"]
        sign  = "✓ DEPLOY?" if bc_r["oos_pd"] > 10 and bc_r["wf"] else "✗"
        print(f"  {cfg:<18} | {d_is:>+10.1f} {d_oos:>+10.1f} "
              f"{tf_r['oos_pd']:>15.1f} {bc_r['oos_pd']:>14.1f}  {sign}")
        if bc_r["oos_pd"] > 10 and bc_r["wf"]:
            deploy_candidates.append((cfg, bc_r))

    if deploy_candidates:
        print(f"\n{'='*78}")
        print("  DEPLOY CANDIDATES (OOS>10 p/d + WF pass with bar-close fills):")
        for cfg, r in deploy_candidates:
            print(f"  🟢 {cfg}: OOS={r['oos_pd']:.1f} p/d  WR={r['oos_wr']:.0%}  "
                  f"avg_win={r['oos_avgw']:.2f}p  avg_loss={r['oos_avgl']:.2f}p  "
                  f"trail%={r['oos_trail%']:.0%}")
    else:
        print("\n  No deploy candidates survive bar-close fill correction.")

    # Save CSV
    df_out = pd.DataFrame(all_rows)
    out_path = OUT / "fifo_fill_correction.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Results → {out_path}")


if __name__ == "__main__":
    main()
