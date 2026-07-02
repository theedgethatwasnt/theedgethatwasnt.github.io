"""
FIFO-Trends P&F — Proper Live Simulation Backtest (v2)
=======================================================

FILL MODEL — explicitly matching live trading reality per exit type:

  ENTRY
    → Market order submitted at M5 bar close (we detect reversal at bar end,
      submit immediately → fill ≈ M5 bar close).
    → Modeled as: entry_px = close[t]

  TRAIL EXIT — two equivalent designs, both valid:
    (a) S5-monitor: Poll OANDA S5 candles every ~5 seconds.  When lo ≤ trail
        (long) or hi ≥ trail (short), submit market order immediately.
        S5 bars are 0.1–0.5 pip wide → fill error < 0.5p.  Effectively
        fills at trail level. **This is what we should build in the live service.**
    (b) OANDA trailing stop: Broker monitors every tick, fills at trigger price.
        Same fill quality as (a).
    Both → modeled as: exit_px = trail   (S5_MONITOR mode)

  X7 EXIT — always manual detection:
    → Condition evaluated at M5 bar close (col_count ≥ col_sma).
    → Market order submitted immediately → fill ≈ M5 bar close.
    → Modeled as: exit_px = close[t]

  REFERENCE MODE (M5_MANUAL):
    → All exits at M5 bar close.  This is what killed FIFO live (5-min detection
      lag means fill is at the bar that may have closed 5–15 pip below trail).
    → Shown for comparison only.  Never deploy with this design.

WHY TRAIL-FILL IS REALISTIC FOR S5 MONITORING:
  Typical S5 bar for GBP/JPY: range ≈ 0.2 pip.
  When trail triggers on an S5 bar, close is within ~0.2 pip of trigger.
  Compare to M5 bar range ≈ 10–15 pip → close can be 5–10 pip below trail.
  S5 monitoring closes the gap by a factor of ~50.

WHAT THIS SCRIPT DOES:
  Sweeps trail distance d ∈ {1, 2, 3, 4} boxes for the four validated pairs,
  using the proper hybrid fill model (trail exits at trail level, X7 at bar close).
  Reports only configs that:
    1. Pass WF (all 3 IS chunks positive, ≥ 5 trades each)
    2. OOS p/d > 10
  Also shows M5_MANUAL result for each config so the margin is visible.

RUN:
  python3 research/experiments/fifo_trends/backtest_fifo_pnf_v2.py
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
OOS_MIN_PD = 10.0        # minimum OOS p/d to report as deploy candidate

PIP = {
    "GBP_JPY": 0.01, "USD_JPY": 0.01,
    "EUR_JPY": 0.01, "GBP_USD": 0.0001,
}
# IS P90 hardcoded (SOP R5 — never recompute from OOS)
SP_GATE = {
    "GBP_JPY": 4.00, "USD_JPY": 2.10,
    "EUR_JPY": 2.50, "GBP_USD": 2.40,
}

# Validated configs per pair: (pair, b_pips, rev, n_min, entry_t, k)
# Sweep d = 1, 2, 3, 4 for each
CONFIGS = [
    ("GBP_JPY", 5, 1, 4, 1, 5),
    ("USD_JPY", 5, 1, 3, 1, 5),
    ("EUR_JPY", 5, 1, 3, 1, 5),
    ("GBP_USD", 2, 3, 8, 1, 5),
]
D_SWEEP = [1, 2, 3, 4]   # trail distance in boxes

# Fill modes
S5_MONITOR = 0   # trail exits at trail level (S5 polling or OANDA stop)
M5_MANUAL  = 1   # all exits at M5 bar close (reference only, broken design)


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
               pip, spread_gate, is_end, fill_mode):
    """
    P&F simulation with explicit fill model per exit type.

    fill_mode = S5_MONITOR (0): trail exits fill at trail level (realistic for
        S5 polling ≤5s detection lag or OANDA trailing stop).
        X7 exits always fill at bar close (manual M5 detection).

    fill_mode = M5_MANUAL (1): all exits fill at bar close (5-min detection lag).
        Reference only — this is the broken design that killed FIFO live.

    Returns:
        pnl[n]       — trade P&L in pips
        is_flag[n]   — 1=IS, 0=OOS
        exit_type[n] — 0=trail, 1=X7
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
    type_arr = np.empty(MAX_T, np.int8)
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

        # HWM: advances only when P&F box advances in trade direction
        # (NOT actual price HWM — P&F coarsening is the noise filter)
        if pos == 1 and pnf_dir == 1 and pnf_level > hw_level:
            hw_level = pnf_level
        elif pos == -1 and pnf_dir == -1 and pnf_level < hw_level:
            hw_level = pnf_level

        # --- Exit logic ---
        exit_triggered = False; exit_px = 0.0; exit_kind = 0

        if pos != 0:
            d = float(xp1); k = int(xp2)

            # Trail component
            if pos == 1:
                trail = hw_level - d * bs
                if lo <= trail:
                    # S5_MONITOR: fill at trail (5-second detection, tiny S5 bar)
                    # M5_MANUAL:  fill at bar close (5-min detection lag, big gap)
                    exit_px = cl if fill_mode == 1 else trail
                    exit_triggered = True; exit_kind = 0
            else:
                trail = hw_level + d * bs
                if hi >= trail:
                    exit_px = cl if fill_mode == 1 else trail
                    exit_triggered = True; exit_kind = 0

            # X7 component — always bar-close (manual detection, market order)
            if not exit_triggered and pnf_dir != pos:
                sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                if sma_k > 0.0 and col_count >= sma_k:
                    exit_px = cl      # bar-close fill in BOTH modes (manual detection)
                    exit_triggered = True; exit_kind = 1

        if exit_triggered and n_t < MAX_T:
            pnl_arr[n_t]  = pos * (exit_px - entry_px) / pip - sp
            flag_arr[n_t] = is_bar
            type_arr[n_t] = exit_kind
            n_t += 1
            pos = 0; entry_px = 0.0; hw_level = 0.0

        # --- Entry logic --- (always bar-close: market order at M5 boundary)
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


def stats(pnls, flags, etypes, is_end, n_total, mode_label):
    """Compute full stats for one fill mode."""
    oos_days = (n_total - is_end) / M5_PER_TRADING_DAY
    is_days  = is_end / M5_PER_TRADING_DAY

    is_mask  = flags == 1
    oos_mask = flags == 0
    is_p  = pnls[is_mask]
    oos_p = pnls[oos_mask]
    is_et  = etypes[is_mask]
    oos_et = etypes[oos_mask]

    # WF: split IS trades sequentially into 3 chunks
    wf_ok = True
    for k in range(N_WF):
        s = k * (len(is_p) // N_WF)
        e = (k+1) * (len(is_p) // N_WF) if k < 2 else len(is_p)
        chunk = is_p[s:e]
        if len(chunk) < MIN_IS_TRADES or chunk.sum() <= 0:
            wf_ok = False; break

    def wr(a):   return (a > 0).mean() if len(a) > 0 else 0.0
    def avgw(a): return a[a > 0].mean() if (a > 0).any() else 0.0
    def avgl(a): return a[a <= 0].mean() if (a <= 0).any() else 0.0
    def tpct(e): return (e == 0).mean() if len(e) > 0 else 0.0
    def pd_(a, days): return a.sum() / days if len(a) > 0 else 0.0

    return {
        "mode":       mode_label,
        "wf":         wf_ok,
        "is_pd":      round(pd_(is_p, is_days), 1),
        "is_n":       len(is_p),
        "is_wr":      round(wr(is_p), 3),
        "is_avgw":    round(avgw(is_p), 2),
        "is_avgl":    round(avgl(is_p), 2),
        "is_trail%":  round(tpct(is_et), 3),
        "oos_pd":     round(pd_(oos_p, oos_days), 1),
        "oos_n":      len(oos_p),
        "oos_wr":     round(wr(oos_p), 3),
        "oos_avgw":   round(avgw(oos_p), 2),
        "oos_avgl":   round(avgl(oos_p), 2),
        "oos_trail%": round(tpct(oos_et), 3),
    }


def main():
    print("=" * 90)
    print("  FIFO-Trends v2 — Proper Live Simulation (S5-monitor fill model)")
    print()
    print("  Fill model:")
    print("    ENTRY     : M5 bar close (market order at M5 boundary)  [both modes]")
    print("    TRAIL EXIT: trail level  (S5/5s detection or OANDA stop) [S5_MONITOR]")
    print("    TRAIL EXIT: M5 bar close (5-min lag, broken design)      [M5_MANUAL]")
    print("    X7 EXIT   : M5 bar close (manual col_count check)        [both modes]")
    print("=" * 90)
    print()

    print("Warming up Numba JIT...")
    _dc = np.cumsum(np.random.randn(5000)) * 0.01 + 150.0
    _dh = _dc + 0.05; _dl = _dc - 0.05; _ds = np.full(5000, 0.02)
    run_config(_dc, _dh, _dl, _dc, _ds, 5, 1, 3, 1, 1, 5, 0.01, 2.5, 3500, S5_MONITOR)
    run_config(_dc, _dh, _dl, _dc, _ds, 5, 1, 3, 1, 1, 5, 0.01, 2.5, 3500, M5_MANUAL)
    print("  Done.\n")

    loaded = {}
    all_results = []

    for (pair, b_pips, rev, n_min, entry_t, k_x7) in CONFIGS:
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
            oos_d = (n - is_e) / M5_PER_TRADING_DAY
            loaded[pair] = (op, hi, lo, cl, sp, n, is_e)
            print(f"  {pair}: {n:,} bars  IS={is_e/M5_PER_TRADING_DAY:.0f}d  "
                  f"OOS={oos_d:.0f}d  sp_gate={sp_gate:.2f}p")

        op, hi, lo, cl, sp, n, is_e = loaded[pair]

        for d in D_SWEEP:
            label = f"{pair}_b{b_pips}r{rev}n{n_min}_d{d}"
            row = {"pair": pair, "b": b_pips, "rev": rev, "n_min": n_min, "d": d}

            for mode, mode_label in [(S5_MONITOR, "S5_monitor"), (M5_MANUAL, "M5_manual")]:
                pnl, flags, etypes = run_config(
                    op, hi, lo, cl, sp,
                    b_pips, rev, n_min, entry_t, d, k_x7,
                    pip, sp_gate, is_e, mode,
                )
                s = stats(pnl, flags, etypes, is_e, n, mode_label)
                row[mode_label] = s

            all_results.append(row)

    # ── Print results ────────────────────────────────────────────────────────────────
    print()
    print(f"  {'Config':<22} {'Mode':<12} {'WF':>3} | "
          f"{'IS p/d':>7} {'IS_n':>5} {'IS_WR':>6} {'avgW':>5} {'avgL':>6} "
          f"{'trail%':>7} | "
          f"{'OOS p/d':>8} {'OOS_n':>5} {'OOS_WR':>6} {'trail%':>7}")
    sep = f"  {'-'*22} {'-'*12} {'-'*3}-+-{'-'*7}-{'-'*5}-{'-'*6}-{'-'*5}-{'-'*6}-{'-'*7}-+-{'-'*8}-{'-'*5}-{'-'*6}-{'-'*7}"

    prev_pair = None
    for row in all_results:
        pair = row["pair"]
        if pair != prev_pair:
            print(); print(sep)
            prev_pair = pair

        for mode_label in ["S5_monitor", "M5_manual"]:
            s = row[mode_label]
            cfg_str = f"{pair}_d{row['d']}" if mode_label == "S5_monitor" else ""
            wf_str  = "✓" if s["wf"] else "✗"
            flag    = " ← DEPLOY?" if (mode_label == "S5_monitor" and
                                        s["wf"] and s["oos_pd"] >= OOS_MIN_PD) else ""
            print(f"  {cfg_str:<22} {mode_label:<12} {wf_str:>3} | "
                  f"{s['is_pd']:>7.1f} {s['is_n']:>5} {s['is_wr']:>6.1%} "
                  f"{s['is_avgw']:>5.2f} {s['is_avgl']:>6.2f} {s['is_trail%']:>7.1%} | "
                  f"{s['oos_pd']:>8.1f} {s['oos_n']:>5} {s['oos_wr']:>6.1%} "
                  f"{s['oos_trail%']:>7.1%}{flag}")

    # ── Deploy candidates ────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  DEPLOY CANDIDATES (S5_monitor: WF pass + OOS ≥ {:.0f} p/d)".format(OOS_MIN_PD))
    print("  Live design required: S5 exit monitoring OR OANDA trailing stop bracket")
    print("-" * 90)

    candidates = [r for r in all_results
                  if r["S5_monitor"]["wf"] and r["S5_monitor"]["oos_pd"] >= OOS_MIN_PD]

    if candidates:
        for row in candidates:
            s   = row["S5_monitor"]
            ref = row["M5_manual"]
            d_oos = s["oos_pd"] - ref["oos_pd"]
            print(f"  🟢 {row['pair']}_b{row['b']}r{row['rev']}n{row['n_min']}_d{row['d']}")
            print(f"     S5_monitor: OOS={s['oos_pd']:>7.1f} p/d  WR={s['oos_wr']:.0%}  "
                  f"avg_win={s['oos_avgw']:.2f}p  avg_loss={s['oos_avgl']:.2f}p  "
                  f"trail%={s['oos_trail%']:.0%}")
            print(f"     M5_manual:  OOS={ref['oos_pd']:>7.1f} p/d  "
                  f"WR={ref['oos_wr']:.0%}  (Δ={d_oos:+.1f} p/d margin)")
            print()
    else:
        print("  No configs pass both WF and OOS threshold under S5_monitor fill model.")

    # ── Save CSV ─────────────────────────────────────────────────────────────────────
    rows_flat = []
    for row in all_results:
        for mode_label in ["S5_monitor", "M5_manual"]:
            s = row[mode_label]
            rows_flat.append({
                "pair":   row["pair"], "b": row["b"], "rev": row["rev"],
                "n_min":  row["n_min"], "d": row["d"],
                **{k: v for k, v in s.items()},
            })
    df_out = pd.DataFrame(rows_flat)
    out_path = OUT / "fifo_pnf_v2_proper_sim.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Results → {out_path}")

    print()
    print("  NOTE: S5_monitor fill model is valid because:")
    print("    • S5 bars are 0.1–0.5 pip wide")
    print("    • Detection lag ≤ 5s → fill within ~0.3p of trail trigger")
    print("    • Live design: poll OANDA S5 candles in position mgmt loop")
    print("    • Alternative: OANDA trailing stop bracket order (same fill quality)")
    print("    • M5_manual is shown only as reference — do NOT deploy with it")


if __name__ == "__main__":
    main()
