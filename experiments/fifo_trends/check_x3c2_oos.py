"""
Quick OOS check: X3c_2_5 (2-box trail + col-SMA5) on GBP_JPY and USD_JPY.
Exact params matching existing ft configs, only trail_d changed 1→2.

GBP_JPY: b=5 r=1 n_min=4 E2 X3c_2_5   (was X3c_1_5 → 71.6 p/d)
USD_JPY: b=5 r=1 n_min=3 E2 X3c_2_5   (was X3c_1_5 → 68.5 p/d)
"""
import sys, time
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC  = 0.70
MAX_K    = 10
MAX_TRADES = 200000  # large enough for single-config full run (prange version uses 20k)

TARGETS = [
    # pair,      pip,    sp_gate, b, r, n_min, entry_t(E2=1), exit_t(X3c_2_5=13), xp1, xp2, label
    ("GBP_JPY", 0.01,   4.0,     5, 1, 4,     1,             13,                  2,   5,  "GBP_JPY_ft2"),
    ("USD_JPY", 0.01,   2.1,     5, 1, 3,     1,             13,                  2,   5,  "USD_JPY_ft2"),
    ("EUR_JPY", 0.01,   2.5,     5, 1, 3,     1,             13,                  2,   5,  "EUR_JPY_ft2"),
]

@nb.njit(inline="always")
def col_sma(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        idx = (ptr - 1 - j) % MAX_K
        total += hist[idx]
    return total / count

@nb.njit
def run_single(opens, highs, lows, closes, spreads, bar_chunks,
               bs_pips, rev, n_min, entry_t, exit_t, xp1, xp2,
               spread_gate, pip, is_end):
    """Run one config; return (pnl_arr, chunk_arr, ntrd) for IS trades only."""
    N = len(opens)
    bs = bs_pips * pip

    pnf_idx=0; pnf_level=0.0; pnf_dir=0; col_count=0; prev_col=0
    col_hist=np.zeros(MAX_K, np.float64)
    col_hist_ptr=0; col_hist_n=0
    pos=0; entry_px=0.0; hw_level=0.0; pending=0

    pnl_buf   = np.zeros(MAX_TRADES, np.float64)
    chunk_buf = np.zeros(MAX_TRADES, np.int8)
    t_cnt = 0

    for i in range(N):
        opn=opens[i]; hi=highs[i]; lo=lows[i]; cl=closes[i]
        sp=spreads[i]; ck=bar_chunks[i]
        bull=(cl>=opn)
        p1=hi if bull else lo
        p2=lo if bull else hi

        did_reverse_p1=False; did_reverse_p2=False
        prev_col_p1=0; prev_col_p2=0

        for tick in range(2):
            px=p1 if tick==0 else p2
            if pnf_dir==0:
                pnf_idx=int(px/bs); pnf_level=pnf_idx*bs
                pnf_dir=1; col_count=1; continue
            delta=int(px/bs)-pnf_idx
            if pnf_dir==1:
                if delta>=1:
                    pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=delta
                elif delta<=-rev:
                    prev_col=col_count
                    col_hist[col_hist_ptr%MAX_K]=prev_col; col_hist_ptr+=1
                    if col_hist_n<MAX_K: col_hist_n+=1
                    pnf_dir=-1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=-delta
                    if tick==0: did_reverse_p1=True; prev_col_p1=prev_col
                    else:       did_reverse_p2=True; prev_col_p2=prev_col
            elif pnf_dir==-1:
                if delta<=-1:
                    pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=(-delta)
                elif delta>=rev:
                    prev_col=col_count
                    col_hist[col_hist_ptr%MAX_K]=prev_col; col_hist_ptr+=1
                    if col_hist_n<MAX_K: col_hist_n+=1
                    pnf_dir=1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=delta
                    if tick==0: did_reverse_p1=True; prev_col_p1=prev_col
                    else:       did_reverse_p2=True; prev_col_p2=prev_col

        did_reverse=did_reverse_p1 or did_reverse_p2
        prev_col_at_rev=prev_col_p1 if did_reverse_p1 else prev_col_p2

        if pos==1:
            if pnf_dir==1 and pnf_level>hw_level: hw_level=pnf_level
        elif pos==-1:
            if pnf_dir==-1 and pnf_level<hw_level: hw_level=pnf_level

        exit_triggered=False; exit_px_val=0.0
        if pos!=0:
            d=float(xp1); k=xp2
            if pos==1:
                trail=hw_level-d*bs
                if lo<=trail: exit_px_val=trail; exit_triggered=True
            else:
                trail=hw_level+d*bs
                if hi>=trail: exit_px_val=trail; exit_triggered=True
            if not exit_triggered and pnf_dir!=pos:
                sma_k=col_sma(col_hist,col_hist_ptr,col_hist_n,k)
                if sma_k>0.0 and col_count>=sma_k:
                    exit_px_val=cl; exit_triggered=True

        if exit_triggered and t_cnt<MAX_TRADES:
            pnl_pips=(exit_px_val-entry_px)*pos/pip-sp
            pnl_buf[t_cnt]=pnl_pips
            chunk_buf[t_cnt]=ck
            t_cnt+=1; pos=0; entry_px=0.0; hw_level=0.0

        if pos==0 and sp<=spread_gate:
            if entry_t==0:
                if did_reverse and prev_col_at_rev>=n_min:
                    pos=pnf_dir; entry_px=cl; hw_level=pnf_level
            else:
                if did_reverse and prev_col_at_rev>=n_min: pending=pnf_dir
                if did_reverse and pending!=0 and pnf_dir!=pending: pending=0
                if pending!=0 and pnf_dir==pending and col_count>rev:
                    pos=pending; entry_px=cl; hw_level=pnf_level; pending=0
        elif pos==0:
            if did_reverse and pending!=0 and pnf_dir!=pending: pending=0

    return pnl_buf[:t_cnt], chunk_buf[:t_cnt], t_cnt

print("X3c_2_5 OOS check — 2-box trail + col-SMA5")
print("="*55)

# warmup
dummy = np.ones(500, np.float64)
dummy_sp = np.zeros(500, np.float64)
dummy_ck = np.zeros(500, np.int8)
print("Warming up Numba JIT...", end=" ", flush=True)
t0 = time.time()
run_single(dummy, dummy, dummy, dummy, dummy_sp, dummy_ck,
           5, 1, 4, 1, 13, 2, 5, 4.0, 0.01, 350)
print(f"done in {time.time()-t0:.1f}s\n")

for pair, pip, sp_gate, b, r, n_min, entry_t, exit_t, xp1, xp2, label in TARGETS:
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    df = pd.read_parquet(ba_path)
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)
    n = len(df)
    is_end = int(n * IS_FRAC)
    oos_days = (n - is_end) / 288.0

    ck0_end = is_end // 3
    ck1_end = 2 * (is_end // 3)
    bar_chunks = np.zeros(n, dtype=np.int8)
    bar_chunks[ck0_end:ck1_end] = 1
    bar_chunks[ck1_end:is_end]  = 2
    bar_chunks[is_end:]         = 3

    pnl_arr, chunk_arr, ntrd = run_single(
        opens, highs, lows, closes, spreads, bar_chunks,
        b, r, n_min, entry_t, exit_t, xp1, xp2,
        sp_gate, pip, is_end
    )

    # IS WF check (3 chunks, all positive, ≥5 trades each)
    wf_pass = True
    for ck in range(3):
        mask = chunk_arr == ck
        ck_pnl = pnl_arr[mask]
        if len(ck_pnl) < 5 or ck_pnl.sum() <= 0:
            wf_pass = False
            break

    # OOS
    oos_mask = chunk_arr == 3
    oos_pnl  = pnl_arr[oos_mask]
    oos_ntrd = int(oos_mask.sum())
    oos_pd   = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
    oos_wr   = float((oos_pnl > 0).sum()) / oos_ntrd * 100 if oos_ntrd > 0 else 0.0

    is_mask   = chunk_arr < 3
    is_pnl    = pnl_arr[is_mask]
    is_pd     = float(is_pnl.sum()) / (is_end / 288.0)

    wf_str = "✅ WF PASS" if wf_pass else "❌ WF FAIL"
    print(f"{label}  ({pair})")
    print(f"  {wf_str}  IS p/d={is_pd:+.1f}  OOS p/d={oos_pd:+.1f}  OOS trades={oos_ntrd}  OOS WR={oos_wr:.0f}%")
    print()
