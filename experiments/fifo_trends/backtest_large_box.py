"""
Large-box P&F sweep: box 10-100p, fractional reversal 1.0-3.5, trail 1-4 boxes.

Key changes vs run_all_pairs.py:
  - Fractional reversal: price-based check (px <= pnf_level - rev*bs) not integer delta
  - Box sizes extended to 100p
  - Explicit trail_d sweep (1-4 boxes) and sma_k sweep (0=disabled, 3, 5, 7)
  - Focus: top 4 pairs only (GBP_JPY, USD_JPY, EUR_JPY, GBP_USD)

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/fifo_trends/backtest_large_box.py
"""

import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from dotenv import load_dotenv
import requests

load_dotenv()
BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg(msg):
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception:
        pass

# ── Parameter grid ──────────────────────────────────────────────────────────
BOX_SIZES   = [10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]  # pips
REVERSALS   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]               # fractional boxes
TRAIL_DISTS = [1, 2, 3, 4]                                   # trail in boxes
SMA_KS      = [0, 3, 5, 7]                                   # col-SMA k (0=off)
MIN_COLS    = [2, 3, 4, 5, 6, 8]                             # n_min entry filter
ENTRIES     = [0, 1]                                          # 0=E1 immediate, 1=E2 confirm

PAIRS = [
    ("GBP_JPY", 0.01),
    ("USD_JPY", 0.01),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
]

IS_FRAC    = 0.70
MAX_K      = 10
MAX_TRADES = 15000   # per prange worker — large boxes = fewer trades than b=5

# ── Config array ─────────────────────────────────────────────────────────────
# columns: [bs_pips, rev, trail_d, sma_k, n_min, entry_t]
def build_configs():
    rows = []
    for bs in BOX_SIZES:
        for rv in REVERSALS:
            for td in TRAIL_DISTS:
                for sk in SMA_KS:
                    for nc in MIN_COLS:
                        for et in ENTRIES:
                            rows.append([bs, rv, float(td), float(sk), float(nc), float(et)])
    return np.array(rows, dtype=np.float64)

CONFIGS = build_configs()
N_CONFIGS = len(CONFIGS)

def config_name(row):
    bs, rv, td, sk, nc, et = row
    entry = "E1" if et == 0 else "E2"
    trail = f"X3c_{int(td)}"
    sma   = f"_{int(sk)}" if sk > 0 else "_noSMA"
    return f"b{int(bs)}_r{rv:.1f}_n{int(nc)}_{entry}_{trail}{sma}"

CONFIG_NAMES = [config_name(CONFIGS[ci]) for ci in range(N_CONFIGS)]

# ── Numba helpers ────────────────────────────────────────────────────────────
@nb.njit(inline="always")
def col_sma(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        total += hist[(ptr - 1 - j) % MAX_K]
    return total / count


@nb.njit(parallel=True)
def run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               configs, spread_gate, pip, is_end,
               trade_pnl, trade_chunk, trade_cnt):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        bs_pips = configs[ci, 0]
        rev     = configs[ci, 1]   # fractional reversal in boxes
        trail_d = int(configs[ci, 2])
        sma_k   = int(configs[ci, 3])
        n_min   = int(configs[ci, 4])
        entry_t = int(configs[ci, 5])

        bs        = bs_pips * pip
        rev_dist  = rev * bs       # price distance to trigger reversal
        trail_px  = trail_d * bs   # price distance for trail stop

        pnf_idx = 0; pnf_level = 0.0; pnf_dir = 0; col_count = 0; prev_col = 0
        col_hist = np.zeros(MAX_K, dtype=np.float64)
        col_hist_ptr = 0; col_hist_n = 0

        pos = 0; entry_px = 0.0; hw_level = 0.0; pending = 0
        t_cnt = 0

        for i in range(N_BARS):
            opn = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]
            sp  = spreads[i]; ck = bar_chunks[i]
            bull = (cl >= opn)
            p1 = hi if bull else lo
            p2 = lo if bull else hi

            did_reverse_p1 = False; did_reverse_p2 = False
            prev_col_p1 = 0;        prev_col_p2 = 0

            for tick in range(2):
                px = p1 if tick == 0 else p2

                if pnf_dir == 0:
                    pnf_idx   = int(px / bs)
                    pnf_level = pnf_idx * bs
                    pnf_dir   = 1; col_count = 1; continue

                new_idx = int(px / bs)
                delta   = new_idx - pnf_idx

                if pnf_dir == 1:
                    if delta >= 1:                          # extension up
                        pnf_idx   = new_idx
                        pnf_level = pnf_idx * bs
                        col_count += delta
                    elif px <= pnf_level - rev_dist:        # fractional reversal down
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1
                        old_idx   = pnf_idx
                        pnf_dir   = -1
                        pnf_idx   = new_idx
                        pnf_level = pnf_idx * bs
                        col_count = max(1, old_idx - new_idx)
                        if tick == 0: did_reverse_p1 = True; prev_col_p1 = prev_col
                        else:         did_reverse_p2 = True; prev_col_p2 = prev_col
                elif pnf_dir == -1:
                    if delta <= -1:                         # extension down
                        pnf_idx   = new_idx
                        pnf_level = pnf_idx * bs
                        col_count += (-delta)
                    elif px >= pnf_level + rev_dist:        # fractional reversal up
                        prev_col = col_count
                        col_hist[col_hist_ptr % MAX_K] = prev_col
                        col_hist_ptr += 1
                        if col_hist_n < MAX_K: col_hist_n += 1
                        old_idx   = pnf_idx
                        pnf_dir   = 1
                        pnf_idx   = new_idx
                        pnf_level = pnf_idx * bs
                        col_count = max(1, new_idx - old_idx)
                        if tick == 0: did_reverse_p1 = True; prev_col_p1 = prev_col
                        else:         did_reverse_p2 = True; prev_col_p2 = prev_col

            did_reverse      = did_reverse_p1 or did_reverse_p2
            prev_col_at_rev  = prev_col_p1 if did_reverse_p1 else prev_col_p2

            # update high-water for open position
            if pos == 1:
                if pnf_dir == 1 and pnf_level > hw_level: hw_level = pnf_level
            elif pos == -1:
                if pnf_dir == -1 and pnf_level < hw_level: hw_level = pnf_level

            # exit logic: trail (always) + optional col-SMA
            exit_triggered = False; exit_px_val = 0.0
            if pos != 0:
                if pos == 1:
                    trail = hw_level - trail_px
                    if lo <= trail: exit_px_val = trail; exit_triggered = True
                else:
                    trail = hw_level + trail_px
                    if hi >= trail: exit_px_val = trail; exit_triggered = True

                if not exit_triggered and sma_k > 0 and pnf_dir != pos:
                    sma_v = col_sma(col_hist, col_hist_ptr, col_hist_n, sma_k)
                    if sma_v > 0.0 and col_count >= sma_v:
                        exit_px_val = cl; exit_triggered = True

            if exit_triggered and t_cnt < MAX_TRADES:
                pnl_pips = (exit_px_val - entry_px) * pos / pip - sp
                trade_pnl[ci, t_cnt]   = np.float32(pnl_pips)
                trade_chunk[ci, t_cnt] = ck
                t_cnt += 1; pos = 0; entry_px = 0.0; hw_level = 0.0

            # entry logic
            if pos == 0:
                if sp <= spread_gate:
                    if entry_t == 0:                        # E1: immediate on reversal
                        if did_reverse and prev_col_at_rev >= n_min:
                            pos = pnf_dir; entry_px = cl; hw_level = pnf_level
                    else:                                   # E2: confirm one box
                        if did_reverse and prev_col_at_rev >= n_min: pending = pnf_dir
                        if did_reverse and pending != 0 and pnf_dir != pending: pending = 0
                        if pending != 0 and pnf_dir == pending and col_count > rev:
                            pos = pending; entry_px = cl; hw_level = pnf_level; pending = 0
                else:
                    if did_reverse and pending != 0 and pnf_dir != pending: pending = 0

        trade_cnt[ci] = t_cnt


# ── Per-pair run ─────────────────────────────────────────────────────────────
def run_pair(pair, pip):
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    if not ba_path.exists():
        print(f"  ⚠ No BA data: {ba_path}"); return

    ba = pd.read_parquet(ba_path)
    n  = len(ba)
    is_end   = int(n * IS_FRAC)
    oos_days = (n - is_end) / 288.0
    is_days  = is_end / 288.0

    opens   = ba["open"].values.astype(np.float64)
    highs   = ba["high"].values.astype(np.float64)
    lows    = ba["low"].values.astype(np.float64)
    closes  = ba["close"].values.astype(np.float64)
    spreads = ((ba["ask_c"] - ba["bid_c"]) / pip).values.astype(np.float64)

    sp_gate = float(np.percentile(spreads[:is_end], 90))

    c0e = is_end // 3; c1e = 2 * (is_end // 3)
    bar_chunks = np.zeros(n, dtype=np.int8)
    bar_chunks[c0e:c1e] = 1; bar_chunks[c1e:is_end] = 2; bar_chunks[is_end:] = 3

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS, dtype=np.int32)

    t0 = time.time()
    run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               CONFIGS, sp_gate, pip, is_end,
               trade_pnl, trade_chunk, trade_cnt)
    elapsed = time.time() - t0
    print(f"  kernel: {elapsed:.1f}s  sp_gate={sp_gate:.1f}p  "
          f"IS={is_days:.0f}d  OOS={oos_days:.0f}d")

    # ── Walk-forward validation: all 3 IS chunks positive, ≥5 trades each ──
    winners = []
    for ci in range(N_CONFIGS):
        tc = trade_cnt[ci]
        if tc == 0: continue
        pnl = trade_pnl[ci, :tc].astype(np.float64)
        ck  = trade_chunk[ci, :tc].astype(np.int32)

        wf_pass = True
        for chunk in range(3):
            cp = pnl[ck == chunk]
            if len(cp) < 5 or cp.sum() <= 0:
                wf_pass = False; break
        if not wf_pass: continue

        oos_mask  = (ck == 3)
        oos_pnl   = pnl[oos_mask]
        oos_ntrd  = int(oos_mask.sum())
        oos_pd    = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
        if oos_pd <= 0: continue

        is_mask   = (ck < 3)
        is_pd     = float(pnl[is_mask].sum()) / is_days
        oos_wr    = float((oos_pnl > 0).sum()) / oos_ntrd * 100 if oos_ntrd > 0 else 0.0
        avg_win   = float(oos_pnl[oos_pnl > 0].mean()) if (oos_pnl > 0).any() else 0.0
        avg_loss  = float(oos_pnl[oos_pnl < 0].mean()) if (oos_pnl < 0).any() else 0.0

        winners.append({
            "ci":       ci,
            "name":     CONFIG_NAMES[ci],
            "is_pd":    round(is_pd, 1),
            "oos_pd":   round(oos_pd, 1),
            "oos_ntrd": oos_ntrd,
            "oos_wr":   round(oos_wr, 1),
            "avg_win":  round(avg_win, 1),
            "avg_loss": round(avg_loss, 1),
        })

    winners.sort(key=lambda x: x["oos_pd"], reverse=True)
    return winners, sp_gate, oos_days


# ── Main ──────────────────────────────────────────────────────────────────────
print("Large-box P&F sweep")
print(f"Configs: {N_CONFIGS:,} per pair  "
      f"(box×{len(BOX_SIZES)} rev×{len(REVERSALS)} trail×{len(TRAIL_DISTS)} "
      f"sma×{len(SMA_KS)} n_min×{len(MIN_COLS)} entry×{len(ENTRIES)})")
print("="*70)

# JIT warmup
dummy = np.ones(500, np.float64)
dummy_sp = np.zeros(500, np.float64)
dummy_ck = np.zeros(500, np.int8)
_tpnl  = np.zeros((1, MAX_TRADES), np.float32)
_tck   = np.zeros((1, MAX_TRADES), np.int8)
_tcnt  = np.zeros(1, np.int32)
_cfg   = np.array([[10.0, 1.0, 1.0, 5.0, 4.0, 1.0]], np.float64)
print("Warming up JIT...", end=" ", flush=True)
t0 = time.time()
run_kernel(dummy, dummy, dummy, dummy, dummy_sp, dummy_ck,
           _cfg, 4.0, 0.01, 350, _tpnl, _tck, _tcnt)
print(f"done in {time.time()-t0:.1f}s\n")

TOP_N = 10

for pair, pip in PAIRS:
    print(f"\n{'─'*70}")
    print(f"  {pair}  (pip={pip})")
    result = run_pair(pair, pip)
    if result is None:
        continue
    winners, sp_gate, oos_days = result

    print(f"  WF winners: {len(winners)} / {N_CONFIGS}")
    if not winners:
        print("  ❌ No OOS winners")
        tg(f"🔴 Large-box {pair}: 0 winners")
        continue

    print(f"\n  {'Rank':<5} {'Config':<40} {'IS p/d':>7} {'OOS p/d':>8} "
          f"{'Trades':>7} {'WR%':>5} {'AvgW':>6} {'AvgL':>7}")
    print(f"  {'─'*95}")
    for rank, w in enumerate(winners[:TOP_N], 1):
        print(f"  {rank:<5} {w['name']:<40} {w['is_pd']:>7.1f} {w['oos_pd']:>8.1f} "
              f"{w['oos_ntrd']:>7} {w['oos_wr']:>5.1f} {w['avg_win']:>6.1f} {w['avg_loss']:>7.1f}")

    best = winners[0]
    msg = (f"🟡 Large-box {pair}\n"
           f"Winners: {len(winners)}/{N_CONFIGS}\n"
           f"Best: {best['name']}\n"
           f"IS={best['is_pd']:+.1f}  OOS={best['oos_pd']:+.1f} p/d\n"
           f"Trades={best['oos_ntrd']}  WR={best['oos_wr']:.0f}%")
    tg(msg)
    print(f"\n  Best: {best['name']}  OOS={best['oos_pd']:+.1f} p/d")

print("\n\nDone.")
