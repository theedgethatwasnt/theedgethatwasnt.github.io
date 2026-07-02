"""
Detailed per-trade stats for b5_r1_n3_E2_X3b_1 (ci=49).
Tracks duration, MFE, MAE, drawdown in addition to PnL.
Pure Python — matches Numba kernel logic exactly.
Run from project root:
  python3 research/experiments/fifo_trends/analyze_best_config.py
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path(__file__).resolve().parents[3]
BA_PATH  = BASE / "data/m5_ba/EUR_JPY_M5_BA.parquet"
PIP      = 0.01
IS_FRAC  = 0.70
SPREAD_GATE_PIPS = 2.5
MAX_K    = 10

# Best config: b5, r1, n_min=3, E2, X3b_1
BS_PIPS  = 5
REV      = 1
N_MIN    = 3
ENTRY_T  = 1      # E2
EXIT_T   = 4      # X3b
D        = 1.0    # trail d=1 box

def col_sma_py(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        idx = (ptr - 1 - j) % MAX_K
        total += hist[idx]
    return total / count

def run():
    df = pd.read_parquet(BA_PATH)
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = ((df["ask_c"] - df["bid_c"]) / PIP).values.astype(np.float64)
    n       = len(df)
    is_end  = int(n * IS_FRAC)

    bs = BS_PIPS * PIP

    # State
    pnf_level = 0.0
    pnf_dir   = 0
    col_count = 0
    prev_col  = 0
    col_hist     = [0.0] * MAX_K
    col_hist_ptr = 0
    col_hist_n   = 0

    pos      = 0
    entry_px = 0.0
    hw_level = 0.0
    pending  = 0
    entry_bar = 0
    entry_sp  = 0.0
    mfe      = 0.0
    mae      = 0.0

    trades = []   # (bar, pnl, duration, mfe, mae, chunk, direction)

    chunk_end0 = is_end // 3
    chunk_end1 = 2 * (is_end // 3)
    def get_chunk(i):
        if i < chunk_end0:   return 0
        if i < chunk_end1:   return 1
        if i < is_end:       return 2
        return 3

    for i in range(n):
        opn = opens[i]
        hi  = highs[i]
        lo  = lows[i]
        cl  = closes[i]
        sp  = spreads[i]
        ck  = get_chunk(i)

        bull = (cl >= opn)
        p1 = hi if bull else lo
        p2 = lo if bull else hi

        did_reverse_p1 = False
        did_reverse_p2 = False
        prev_col_p1    = 0
        prev_col_p2    = 0

        for tick in range(2):
            px = p1 if tick == 0 else p2

            if pnf_dir == 0:
                pnf_level = math.floor(px / bs) * bs
                pnf_dir   = 1
                col_count = 1
                continue

            raw   = (px - pnf_level) / bs
            delta = int(raw)

            if pnf_dir == 1:
                if delta >= 1:
                    pnf_level += delta * bs
                    col_count += delta
                elif delta <= -REV:
                    prev_col = col_count
                    col_hist[col_hist_ptr % MAX_K] = prev_col
                    col_hist_ptr += 1
                    if col_hist_n < MAX_K:
                        col_hist_n += 1
                    pnf_dir   = -1
                    pnf_level += delta * bs
                    col_count  = -delta
                    if tick == 0:
                        did_reverse_p1 = True
                        prev_col_p1    = prev_col
                    else:
                        did_reverse_p2 = True
                        prev_col_p2    = prev_col
            elif pnf_dir == -1:
                if delta <= -1:
                    pnf_level += delta * bs
                    col_count += (-delta)
                elif delta >= REV:
                    prev_col = col_count
                    col_hist[col_hist_ptr % MAX_K] = prev_col
                    col_hist_ptr += 1
                    if col_hist_n < MAX_K:
                        col_hist_n += 1
                    pnf_dir   = 1
                    pnf_level += delta * bs
                    col_count  = delta
                    if tick == 0:
                        did_reverse_p1 = True
                        prev_col_p1    = prev_col
                    else:
                        did_reverse_p2 = True
                        prev_col_p2    = prev_col

        did_reverse = did_reverse_p1 or did_reverse_p2
        prev_col_at_rev = prev_col_p1 if did_reverse_p1 else prev_col_p2

        # Update HW and intra-bar MFE/MAE
        if pos == 1:
            if pnf_dir == 1 and pnf_level > hw_level:
                hw_level = pnf_level
            cur_mfe = (hw_level - entry_px) / PIP
            cur_mae = (entry_px - lo) / PIP
            if cur_mfe > mfe: mfe = cur_mfe
            if cur_mae > mae: mae = cur_mae
        elif pos == -1:
            if pnf_dir == -1 and pnf_level < hw_level:
                hw_level = pnf_level
            cur_mfe = (entry_px - hw_level) / PIP
            cur_mae = (hi - entry_px) / PIP
            if cur_mfe > mfe: mfe = cur_mfe
            if cur_mae > mae: mae = cur_mae

        # EXIT
        exit_triggered = False
        exit_px_val    = 0.0

        if pos != 0:
            d_val = D
            if pos == 1:
                trail = hw_level - d_val * bs
                if lo <= trail:
                    exit_px_val   = trail
                    exit_triggered = True
            else:
                trail = hw_level + d_val * bs
                if hi >= trail:
                    exit_px_val   = trail
                    exit_triggered = True

        if exit_triggered:
            pnl_pips = (exit_px_val - entry_px) * pos / PIP - sp
            dur = i - entry_bar
            trades.append({
                "bar":   i,
                "pnl":   pnl_pips,
                "dur":   dur,
                "mfe":   mfe,
                "mae":   mae,
                "chunk": ck,
                "dir":   pos,
            })
            pos      = 0
            entry_px = 0.0
            hw_level = 0.0
            mfe      = 0.0
            mae      = 0.0

        # ENTRY
        if pos == 0:
            can_enter = (sp <= SPREAD_GATE_PIPS)
            if can_enter:
                if did_reverse and prev_col_at_rev >= N_MIN:
                    pending = pnf_dir
                if did_reverse and pending != 0 and pnf_dir != pending:
                    pending = 0
                if pending != 0 and pnf_dir == pending and col_count > REV:
                    pos       = pending
                    entry_px  = cl
                    hw_level  = pnf_level
                    entry_bar = i
                    entry_sp  = sp
                    mfe       = 0.0
                    mae       = 0.0
                    pending   = 0
            else:
                if did_reverse and pending != 0 and pnf_dir != pending:
                    pending = 0

    td = pd.DataFrame(trades)
    if len(td) == 0:
        print("No trades!")
        return

    is_td  = td[td["chunk"] < 3]
    oos_td = td[td["chunk"] == 3]

    for label, t in [("IS", is_td), ("OOS", oos_td)]:
        print(f"\n─── {label} ───")
        print(f"  Trades:       {len(t)}")
        wins   = t[t["pnl"] > 0]
        losses = t[t["pnl"] <= 0]
        wr     = len(wins)/len(t)*100 if len(t) else 0
        avg_w  = wins["pnl"].mean() if len(wins) else 0
        avg_l  = losses["pnl"].mean() if len(losses) else 0
        pf     = (-wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() < 0 else float('inf')
        print(f"  Win rate:     {wr:.1f}%")
        print(f"  Avg win:      {avg_w:.2f}p")
        print(f"  Avg loss:     {avg_l:.2f}p")
        print(f"  Profit factor:{pf:.2f}")
        print(f"  Mean trade:   {t['pnl'].mean():.2f}p")
        print(f"  Median trade: {t['pnl'].median():.2f}p")
        print(f"  Total pips:   {t['pnl'].sum():.0f}p")

        print(f"  Duration P50: {t['dur'].median():.0f} bars  P90: {t['dur'].quantile(0.9):.0f} bars  P99: {t['dur'].quantile(0.99):.0f} bars  Max: {t['dur'].max()} bars")
        pct_1bar = (t["dur"] <= 1).mean() * 100
        print(f"  % 1-bar:      {pct_1bar:.1f}%")

        print(f"  MFE P50: {t['mfe'].median():.1f}p  P90: {t['mfe'].quantile(0.9):.1f}p  Mean: {t['mfe'].mean():.1f}p")
        print(f"  MAE P50: {t['mae'].median():.1f}p  P90: {t['mae'].quantile(0.9):.1f}p  Mean: {t['mae'].mean():.1f}p")

        # Max drawdown on cumulative equity
        cum = t["pnl"].cumsum().values
        peak = np.maximum.accumulate(cum)
        dd   = (peak - cum)
        max_dd = dd.max()
        max_dd_pct = (max_dd / peak[np.argmax(dd)]) * 100 if peak[np.argmax(dd)] > 0 else 0
        print(f"  Max DD:       {max_dd:.0f}p ({max_dd_pct:.1f}% of peak equity)")

        # PnL / MFE ratio
        ratio = (t["pnl"] / t["mfe"].clip(lower=0.01)).mean()
        print(f"  PnL/MFE ratio:{ratio:.3f}")

    print(f"\n  Duration distribution (OOS):")
    oos_d = oos_td["dur"]
    for v in [0, 1, 2, 3, 4, 5]:
        n = (oos_d == v).sum()
        pct = n / len(oos_d) * 100
        print(f"    {v} bars: {n} ({pct:.1f}%)")
    print(f"    >5 bars: {(oos_d > 5).sum()} ({(oos_d > 5).mean()*100:.1f}%)")

if __name__ == "__main__":
    run()
