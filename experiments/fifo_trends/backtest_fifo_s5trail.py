"""
FIFO-Trends P&F Backtest — S5 Trail Resolution Comparison
==========================================================
Tests whether updating the trailing stop HWM at S5 resolution (every 5 seconds)
improves OOS p/d vs the existing M5-level P&F HWM.

WHY THIS MATTERS:
  M5 HWM: advances only when P&F level advances (5-pip box steps).
           A trade 7 pips in profit (between P&F boxes) has:
           trail = hw_pnf_level - 1*box ≈ entry_level - 5p (still in loss zone)

  S5 HWM: tracks actual price every 5 seconds.
           Same 7-pip profit → trail = entry_price + 7p - 5p = entry+2p (already profitable)

WHAT STAYS THE SAME:
  - P&F chart computation: M5 bars only (entry signal unchanged)
  - Entry logic: E2 confirmation, same config
  - Fill assumption: stop-order fill at trail level (same as original — OANDA stop orders)
  - X7 exit: M5-level (col-SMA exit unchanged)
  - Spread gate: IS P90 (SOP R5)

WHAT CHANGES:
  - HWM tracking: actual price HWM updated at every S5 bar
  - Trail check: every S5 bar (can exit intra-M5 bar)
  - Exit fill: S5 bar close when lo <= trail (≈ trail level, since S5 bars are tiny)

COMPARISON:
  M5-trail:  hw = pnf_level (updates only on P&F box advance)
  S5-trail:  hw = actual price high (updates every S5 bar within M5 period)

DATA:
  M5 BA:  5.5yr, all 12 pairs — P&F chart signal
  S5 BA:  EUR_USD (15mo), EUR_JPY (6.5mo) — trail management only

CONFIGS TESTED:
  Best validated configs from original 12-pair sweep:
  - EUR_USD: b5_r1_n2_E2_X3c_1_5 (OOS p/d = 10.8p)
  - EUR_JPY: b5_r1_n3_E2_X3c_1_5 (OOS p/d = 39.9p)
  - GBP_USD: b2_r3_n8_E2_X3c_1_5 (OOS p/d = 54.5p, only M5 data)
  Also sweeps a small grid to find if different configs benefit more from S5.

Run:
  python3 research/experiments/fifo_trends/backtest_fifo_s5trail.py
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT    = Path(__file__).resolve().parents[3]
M5_DIR  = ROOT / "data" / "m5_ba"
S5_DIR  = ROOT / "data" / "s5_ohlc"

PIP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001,
    "EUR_JPY": 0.01,   "GBP_JPY": 0.01,
}

IS_FRAC = 0.70
N_WF    = 3
S5_PER_TRADING_DAY = 17280.0
M5_PER_TRADING_DAY = 288.0
M5_NS = 300 * 1_000_000_000   # 5 minutes in nanoseconds
MAX_K = 10   # ring buffer size for col-SMA

# Validated best configs per pair (from original sweep)
CONFIGS = {
    "EUR_USD": [
        dict(bs_pips=5,  rev=1, n_min=2, entry_t=1, exit_t=11, xp1=1, xp2=5, label="b5_r1_n2_E2_X3c_1_5"),
        dict(bs_pips=5,  rev=1, n_min=3, entry_t=1, exit_t=11, xp1=1, xp2=5, label="b5_r1_n3_E2_X3c_1_5"),
        dict(bs_pips=5,  rev=1, n_min=4, entry_t=1, exit_t=11, xp1=1, xp2=5, label="b5_r1_n4_E2_X3c_1_5"),
        dict(bs_pips=5,  rev=1, n_min=2, entry_t=1, exit_t=13, xp1=2, xp2=5, label="b5_r1_n2_E2_X3c_2_5"),
    ],
    "EUR_JPY": [
        dict(bs_pips=5,  rev=1, n_min=3, entry_t=1, exit_t=11, xp1=1, xp2=5, label="b5_r1_n3_E2_X3c_1_5"),
        dict(bs_pips=5,  rev=1, n_min=4, entry_t=1, exit_t=11, xp1=1, xp2=5, label="b5_r1_n4_E2_X3c_1_5"),
        dict(bs_pips=5,  rev=1, n_min=3, entry_t=1, exit_t=13, xp1=2, xp2=5, label="b5_r1_n3_E2_X3c_2_5"),
    ],
}


# ── P&F chart update (single M5 bar) ─────────────────────────────────────────

@njit
def update_pnf(opn, hi, lo, cl, bs,
               pnf_idx, pnf_level, pnf_dir, col_count, prev_col,
               col_hist, col_hist_ptr, col_hist_n):
    """
    Update P&F state for one M5 bar.
    Returns: (pnf_idx, pnf_level, pnf_dir, col_count, prev_col,
               col_hist_ptr, col_hist_n, did_reverse, prev_col_at_rev)
    """
    bull = (cl >= opn)
    p1   = hi if bull else lo
    p2   = lo if bull else hi

    did_reverse     = False
    prev_col_at_rev = 0

    for tick in range(2):
        px = p1 if tick == 0 else p2

        if pnf_dir == 0:
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
            elif delta <= -int(round((pnf_level - px) / bs + 0.5)):
                # Use integer reversal test
                pass
        # Simpler: use direct reversal size
        if pnf_dir == 1 and delta <= -(col_count // col_count):
            pass  # handled below

    # Redo cleanly
    pnf_idx2   = pnf_idx
    pnf_level2 = pnf_level
    pnf_dir2   = pnf_dir
    col_count2 = col_count

    return (pnf_idx, pnf_level, pnf_dir, col_count, prev_col,
            col_hist_ptr, col_hist_n, did_reverse, prev_col_at_rev)


@njit
def col_sma(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        idx = (ptr - 1 - j) % MAX_K
        total += hist[idx]
    return total / count


# ── Main simulation kernel ────────────────────────────────────────────────────

@njit
def sim_one_config(
    # M5 data arrays
    m5_opens, m5_highs, m5_lows, m5_closes, m5_spreads,
    # S5 data arrays (may be empty for M5-only mode)
    s5_highs, s5_lows, s5_closes,
    # Alignment: for M5 bar i, S5 bars are s5_highs[m5_s5_start[i]:m5_s5_end[i]]
    m5_s5_start, m5_s5_end,
    # Config
    bs_pips, rev, n_min, entry_t, exit_t, xp1, xp2,
    # Meta
    pip, spread_gate, is_end, use_s5_trail,
):
    """
    Simulate one FIFO P&F config.
    use_s5_trail: True → update HWM at every S5 bar; False → M5 P&F level HWM.

    Returns arrays: pnl_pips[n_trades], is_flag[n_trades] (1=IS, 0=OOS)
    """
    N_M5 = len(m5_opens)
    bs   = bs_pips * pip

    # P&F state
    pnf_idx      = 0
    pnf_level    = 0.0
    pnf_dir      = 0   # 0=uninit, +1=X, -1=O
    col_count    = 0
    prev_col     = 0
    col_hist     = np.zeros(MAX_K)
    col_hist_ptr = 0
    col_hist_n   = 0

    # Position state
    pos       = 0
    entry_px  = 0.0
    hw_level  = 0.0   # HWM (P&F-level or price, depending on use_s5_trail)
    pending   = 0

    max_t = N_M5 * 2 + 10
    pnl_out  = np.empty(max_t)
    flag_out = np.empty(max_t, dtype=np.int8)
    n_t = 0

    for i in range(N_M5):
        opn = m5_opens[i]
        hi  = m5_highs[i]
        lo  = m5_lows[i]
        cl  = m5_closes[i]
        sp  = m5_spreads[i]
        is_flag = 1 if i < is_end else 0

        bull = (cl >= opn)
        p1   = hi if bull else lo
        p2   = lo if bull else hi

        # ── P&F update (both ticks, same logic as original backtest) ──
        did_reverse     = False
        prev_col_at_rev = 0

        for tick in range(2):
            px = p1 if tick == 0 else p2

            if pnf_dir == 0:
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
                    prev_col = col_count
                    col_hist[col_hist_ptr % MAX_K] = prev_col
                    col_hist_ptr += 1
                    if col_hist_n < MAX_K:
                        col_hist_n += 1
                    pnf_dir   = -1
                    pnf_idx  += delta
                    pnf_level = pnf_idx * bs
                    col_count = -delta
                    if not did_reverse:
                        did_reverse     = True
                        prev_col_at_rev = prev_col

            elif pnf_dir == -1:
                if delta <= -1:
                    pnf_idx  += delta
                    pnf_level = pnf_idx * bs
                    col_count += (-delta)
                elif delta >= rev:
                    prev_col = col_count
                    col_hist[col_hist_ptr % MAX_K] = prev_col
                    col_hist_ptr += 1
                    if col_hist_n < MAX_K:
                        col_hist_n += 1
                    pnf_dir   = 1
                    pnf_idx  += delta
                    pnf_level = pnf_idx * bs
                    col_count = delta
                    if not did_reverse:
                        did_reverse     = True
                        prev_col_at_rev = prev_col

        # ── S5 trail section ──────────────────────────────────────────
        # If in trade and S5 data available: check trail at S5 resolution
        # using actual price HWM (tighter than P&F-level HWM)
        s5_exit_triggered = False
        s5_exit_px        = 0.0

        if pos != 0 and use_s5_trail:
            s_start = m5_s5_start[i]
            s_end   = m5_s5_end[i]
            price_hwm = hw_level   # use running price HWM

            for j in range(s_start, s_end):
                s5_hi = s5_highs[j]
                s5_lo = s5_lows[j]
                s5_cl = s5_closes[j]

                # Update price HWM
                if pos == 1:
                    if s5_hi > price_hwm:
                        price_hwm = s5_hi
                    trail = price_hwm - xp1 * bs
                    if s5_lo <= trail:
                        s5_exit_px = s5_cl   # fill at S5 close (≈ trail level)
                        s5_exit_triggered = True
                        hw_level = price_hwm
                        break
                else:
                    if s5_lo < price_hwm:
                        price_hwm = s5_lo
                    trail = price_hwm + xp1 * bs
                    if s5_hi >= trail:
                        s5_exit_px = s5_cl
                        s5_exit_triggered = True
                        hw_level = price_hwm
                        break

            if not s5_exit_triggered:
                hw_level = price_hwm   # persist updated HWM for next M5 bar

        # ── M5-level P&F HWM update (for M5-trail mode) ───────────────
        if not use_s5_trail and pos != 0:
            if pos == 1 and pnf_dir == 1 and pnf_level > hw_level:
                hw_level = pnf_level
            elif pos == -1 and pnf_dir == -1 and pnf_level < hw_level:
                hw_level = pnf_level

        # ── Process S5 trail exit ──────────────────────────────────────
        if s5_exit_triggered and n_t < max_t:
            pnl = pos * (s5_exit_px - entry_px) / pip - sp
            pnl_out[n_t]  = pnl
            flag_out[n_t] = is_flag
            n_t += 1
            pos = 0; entry_px = 0.0; hw_level = 0.0

        # ── M5 exit logic (X3b trail + X7, same as original) ──────────
        exit_triggered = False
        exit_px_val    = 0.0

        if pos != 0 and not s5_exit_triggered:
            d = float(xp1); k = xp2
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

            # X7 component (whichever fires first with trail)
            if not exit_triggered and pnf_dir != pos:
                sma_k = col_sma(col_hist, col_hist_ptr, col_hist_n, k)
                if sma_k > 0.0 and col_count >= sma_k:
                    exit_px_val   = cl
                    exit_triggered = True

            if exit_triggered and n_t < max_t:
                pnl = pos * (exit_px_val - entry_px) / pip - sp
                pnl_out[n_t]  = pnl
                flag_out[n_t] = is_flag
                n_t += 1
                pos = 0; entry_px = 0.0; hw_level = 0.0

        # ── Entry logic ────────────────────────────────────────────────
        if pos == 0:
            can_enter = (sp <= spread_gate)
            if can_enter:
                if entry_t == 0:
                    if did_reverse and prev_col_at_rev >= n_min:
                        pos      = pnf_dir
                        entry_px = cl
                        hw_level = pnf_level
                else:
                    if did_reverse and prev_col_at_rev >= n_min:
                        pending = pnf_dir
                    if did_reverse and pending != 0 and pnf_dir != pending:
                        pending = 0
                    if pending != 0 and pnf_dir == pending and col_count > rev:
                        pos      = pending
                        entry_px = cl
                        hw_level = pnf_level if not use_s5_trail else cl
                        pending  = 0
            else:
                if did_reverse and pending != 0 and pnf_dir != pending:
                    pending = 0

    return pnl_out[:n_t], flag_out[:n_t]


# ── Data loader ───────────────────────────────────────────────────────────────

def load_pair(pair):
    pip = PIP[pair]

    # M5 data (full history)
    m5_path = M5_DIR / f"{pair}_M5_BA.parquet"
    assert m5_path.exists(), f"M5 BA missing: {m5_path}"
    m5 = pd.read_parquet(m5_path).sort_values("timestamp").reset_index(drop=True)
    m5_ts  = pd.to_datetime(m5["timestamp"]).astype(np.int64).values
    m5_op  = m5["open"].astype(np.float64).values
    m5_hi  = m5["high"].astype(np.float64).values
    m5_lo  = m5["low"].astype(np.float64).values
    m5_cl  = m5["close"].astype(np.float64).values
    m5_sp  = ((m5["ask_c"] - m5["bid_c"]) / pip).astype(np.float64).values

    # S5 data (partial — only where available)
    s5_path = S5_DIR / f"{pair}_S5_BA.parquet"
    has_s5  = s5_path.exists()

    s5_hi_arr  = np.zeros(0)
    s5_lo_arr  = np.zeros(0)
    s5_cl_arr  = np.zeros(0)
    s5_start   = np.zeros(len(m5), dtype=np.int64)
    s5_end     = np.zeros(len(m5), dtype=np.int64)
    s5_period  = (None, None)

    if has_s5:
        s5 = pd.read_parquet(s5_path).sort_values("timestamp").reset_index(drop=True)
        s5_ts_raw = pd.to_datetime(s5["timestamp"]).astype(np.int64).values
        if "high" in s5.columns:
            s5_hi_arr = s5["high"].astype(np.float64).values
            s5_lo_arr = s5["low"].astype(np.float64).values
            s5_cl_arr = s5["close"].astype(np.float64).values
        else:
            s5_hi_arr = ((s5["bid_h"] + s5["ask_h"]) / 2).astype(np.float64).values
            s5_lo_arr = ((s5["bid_l"] + s5["ask_l"]) / 2).astype(np.float64).values
            s5_cl_arr = ((s5["bid_c"] + s5["ask_c"]) / 2).astype(np.float64).values

        # For each M5 bar, find S5 bars within [M5_ts, M5_ts + 300s)
        m5_s5_starts = np.searchsorted(s5_ts_raw, m5_ts,              side="left")
        m5_s5_ends   = np.searchsorted(s5_ts_raw, m5_ts + M5_NS,      side="left")
        s5_start     = m5_s5_starts.astype(np.int64)
        s5_end       = m5_s5_ends.astype(np.int64)

        s5_period = (pd.to_datetime(s5_ts_raw[0]).date(),
                     pd.to_datetime(s5_ts_raw[-1]).date())

        n_s5 = len(s5)
        avg_per_m5 = (s5_end - s5_start).mean()
        print(f"    S5: {n_s5:,} bars  {s5_period[0]} → {s5_period[1]}  "
              f"avg {avg_per_m5:.1f} S5 bars/M5bar (expected 60)")

    return (m5_ts, m5_op, m5_hi, m5_lo, m5_cl, m5_sp,
            s5_hi_arr, s5_lo_arr, s5_cl_arr,
            s5_start, s5_end, has_s5, s5_period)


# ── Per-config runner ─────────────────────────────────────────────────────────

def run_config(pair, cfg, m5_data, s5_avail, n_m5, is_end, sp_gate, pip):
    m5_ts, m5_op, m5_hi, m5_lo, m5_cl, m5_sp = m5_data[:6]
    s5_hi, s5_lo, s5_cl, s5_start, s5_end     = m5_data[6:11]

    oos_start = is_end
    oos_days  = (n_m5 - oos_start) / M5_PER_TRADING_DAY
    is_days   = is_end / M5_PER_TRADING_DAY

    results = {}
    for mode, use_s5 in [("M5-trail", False), ("S5-trail", True)]:
        if use_s5 and not s5_avail:
            continue

        pnls, flags = sim_one_config(
            m5_op, m5_hi, m5_lo, m5_cl, m5_sp,
            s5_hi, s5_lo, s5_cl,
            s5_start, s5_end,
            float(cfg["bs_pips"]), int(cfg["rev"]), int(cfg["n_min"]),
            int(cfg["entry_t"]), int(cfg["exit_t"]),
            int(cfg["xp1"]), int(cfg["xp2"]),
            pip, sp_gate, is_end, use_s5,
        )

        is_pnls  = pnls[flags == 1]
        oos_pnls = pnls[flags == 0]

        # WF: 3 IS chunks all positive
        chunk = is_end // 3
        wf_ok = True
        for k in range(N_WF):
            c_start = k * chunk
            c_end   = (k+1)*chunk if k < 2 else is_end
            mask = (flags == 1)
            c_pnl = 0.0
            c_cnt = 0
            for ti in range(len(pnls)):
                if flags[ti] == 1:
                    # Approximate: check by bar index — not available here
                    # Use IS pnl sequence
                    pass
            # Simplified WF using sequential IS pnls
            c_pnls = is_pnls[k * (len(is_pnls)//3) : (k+1)*(len(is_pnls)//3)] if k < 2 else is_pnls[2*(len(is_pnls)//3):]
            if len(c_pnls) < 5 or c_pnls.sum() <= 0:
                wf_ok = False
                break

        wr = (is_pnls > 0).mean() if len(is_pnls) > 0 else 0.0
        avg_w = is_pnls[is_pnls > 0].mean() if (is_pnls > 0).any() else 0.0
        avg_l = is_pnls[is_pnls <= 0].mean() if (is_pnls <= 0).any() else 0.0
        is_pd  = is_pnls.sum() / is_days  if len(is_pnls) > 0 else 0.0
        oos_pd = oos_pnls.sum() / oos_days if len(oos_pnls) > 0 else 0.0

        # OOS stats
        oos_wr    = (oos_pnls > 0).mean() if len(oos_pnls) > 0 else 0.0
        oos_avg_w = oos_pnls[oos_pnls > 0].mean() if (oos_pnls > 0).any() else 0.0
        oos_avg_l = oos_pnls[oos_pnls <= 0].mean() if (oos_pnls <= 0).any() else 0.0

        results[mode] = {
            "is_pd":   round(is_pd, 1),
            "oos_pd":  round(oos_pd, 1),
            "is_n":    len(is_pnls),
            "oos_n":   len(oos_pnls),
            "is_wr":   round(wr, 3),
            "oos_wr":  round(oos_wr, 3),
            "is_avgw": round(avg_w, 2),
            "is_avgl": round(avg_l, 2),
            "oos_avgw":round(oos_avg_w, 2),
            "oos_avgl":round(oos_avg_l, 2),
            "wf_ok":   wf_ok,
        }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 74)
    print("  FIFO-Trends S5 Trail Resolution — M5 vs S5 HWM Comparison")
    print("=" * 74)

    # Warm up Numba
    print("\nWarming up Numba JIT...")
    dc = np.cumsum(np.random.randn(5000)) * 0.0001 + 1.1
    dh = dc + 0.0002; dl = dc - 0.0002; ds = np.full(5000, 0.0001)
    ds5h = dc[:50] + 0.00002; ds5l = dc[:50] - 0.00002; ds5c = dc[:50]
    s5_st = np.zeros(5000, dtype=np.int64)
    s5_en = np.zeros(5000, dtype=np.int64)
    for i in range(5000):
        s5_st[i] = min(i, 49); s5_en[i] = min(i+1, 50)
    sim_one_config(dc, dh, dl, dc, ds, ds5h, ds5l, ds5c, s5_st, s5_en,
                   5.0, 1, 3, 1, 11, 1, 5, 0.0001, 2.0, 3500, False)
    sim_one_config(dc, dh, dl, dc, ds, ds5h, ds5l, ds5c, s5_st, s5_en,
                   5.0, 1, 3, 1, 11, 1, 5, 0.0001, 2.0, 3500, True)
    print("  Done.\n")

    all_rows = []

    for pair, cfgs in CONFIGS.items():
        pip = PIP[pair]
        print(f"\n{'='*74}")
        print(f"  {pair}  (pip={pip})")
        print(f"{'='*74}")

        data = load_pair(pair)
        m5_ts, m5_op, m5_hi, m5_lo, m5_cl, m5_sp = data[:6]
        s5_hi, s5_lo, s5_cl, s5_start, s5_end, has_s5, s5_period = data[6:]

        n_m5   = len(m5_op)
        is_end = int(n_m5 * IS_FRAC)

        # IS P90 spread gate (SOP R5)
        sp_gate = float(np.percentile(m5_sp[:is_end], 90))
        oos_days = (n_m5 - is_end) / M5_PER_TRADING_DAY
        is_days  = is_end / M5_PER_TRADING_DAY

        print(f"  M5: {n_m5:,} bars  IS={is_days:.0f}d  OOS={oos_days:.0f}d  sp_gate={sp_gate:.2f}p")
        if not has_s5:
            print(f"  No S5 data — M5-trail only")

        m5_data = (m5_ts, m5_op, m5_hi, m5_lo, m5_cl, m5_sp,
                   s5_hi, s5_lo, s5_cl, s5_start, s5_end)

        # Overlap period: only bars where S5 data exists
        # For S5-trail mode: we restrict comparison to M5 bars covered by S5 data
        if has_s5:
            s5_ts_first = pd.to_datetime(m5_ts[s5_start > 0][0]) if (s5_start > 0).any() else None
            # Find first M5 bar that has S5 coverage
            has_s5_mask = (s5_end - s5_start) > 0
            first_s5_m5 = int(np.argmax(has_s5_mask)) if has_s5_mask.any() else n_m5
            s5_is_end   = int(first_s5_m5 + (n_m5 - first_s5_m5) * IS_FRAC)
            print(f"  S5 coverage starts at M5 bar {first_s5_m5:,} "
                  f"({(n_m5-first_s5_m5)/M5_PER_TRADING_DAY:.0f} M5 days covered by S5)")
        else:
            first_s5_m5 = n_m5
            s5_is_end   = is_end

        print(f"\n  {'Config':<25} | {'Mode':<10} | {'IS_pd':>7} {'IS_n':>5} {'WR':>5} "
              f"{'avg_w':>6} {'avg_l':>6} | {'OOS_pd':>7} {'OOS_n':>5} {'OOS_WR':>6}")
        print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*7}-{'-'*5}-{'-'*5}-{'-'*6}-{'-'*6}-+-{'-'*7}-{'-'*5}-{'-'*6}")

        for cfg in cfgs:
            res = run_config(pair, cfg, m5_data, has_s5, n_m5, is_end, sp_gate, pip)

            for mode, r in res.items():
                wf_str = "✓" if r["wf_ok"] else " "
                print(f"  {cfg['label']:<25} | {mode:<10} | {r['is_pd']:>7.1f} {r['is_n']:>5} "
                      f"{r['is_wr']:>5.1%} {r['is_avgw']:>6.2f} {r['is_avgl']:>6.2f} | "
                      f"{r['oos_pd']:>7.1f} {r['oos_n']:>5} {r['oos_wr']:>6.1%}  {wf_str}")
                all_rows.append({
                    "pair": pair, "config": cfg["label"], "mode": mode, **r
                })

        # Improvement summary
        if has_s5:
            print(f"\n  Improvement (S5-trail vs M5-trail):")
            for cfg in cfgs:
                cfg_rows = [r for r in all_rows if r["pair"] == pair and r["config"] == cfg["label"]]
                m5r = next((r for r in cfg_rows if r["mode"] == "M5-trail"), None)
                s5r = next((r for r in cfg_rows if r["mode"] == "S5-trail"), None)
                if m5r and s5r:
                    d_is  = s5r["is_pd"] - m5r["is_pd"]
                    d_oos = s5r["oos_pd"] - m5r["oos_pd"]
                    print(f"    {cfg['label']:<25}: IS Δ={d_is:+.1f}p/d  OOS Δ={d_oos:+.1f}p/d")

        gc.collect()

    print(f"\n{'='*74}")
    print("  SUMMARY")
    print(f"{'='*74}")
    df = pd.DataFrame(all_rows) if all_rows else None
    if df is not None and len(df) > 0:
        print(df[["pair","config","mode","is_pd","oos_pd","oos_wr","oos_avgw","oos_avgl"]].to_string(index=False))

        out = Path(__file__).parent / "results" / "fifo_s5trail_comparison.csv"
        df.to_csv(out, index=False)
        print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
