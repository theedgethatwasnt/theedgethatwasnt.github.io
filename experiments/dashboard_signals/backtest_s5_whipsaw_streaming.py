"""S5 whipsaw STREAMING detection within an M1 bar.

At each S5 bar within an M1 bar (M1 buckets = S5 index // 12), we check:

  LONG signal (peak then trough):
    - current s5_mom <= -low_thr          (TROUGH right now)
    - max s5_mom earlier in this M1 bucket > +high_thr   (PEAK was before)
    - peak's S5 idx < current S5 idx        (sequence preserved)

  SHORT signal (trough then peak):
    - current s5_mom >= +high_thr          (PEAK right now)
    - min s5_mom earlier in this M1 bucket < -low_thr    (TROUGH was before)
    - trough's S5 idx < current S5 idx

Enter at current S5 close (~ this S5 bar's mid close).  Hold 60 S5 bars = 5 min.
Exit at the 60th S5 bar's close.  Round-trip spread = 1× average BA at entry.

Cooldown: once a signal fires, suppress further entries until the position closes.
"""
import time, gc
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange
import pyarrow.parquet as pq

PROJECT = Path("/path/to/projects/fx-core")
S5_DIR  = PROJECT / "data" / "s5_ohlc"
OUT     = Path(__file__).resolve().parent / "results"; OUT.mkdir(exist_ok=True)

PAIRS = {
    "USD_JPY": 0.01,   "EUR_JPY": 0.01,   "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,   "CAD_JPY": 0.01,   "NZD_JPY": 0.01,   "CHF_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
    "EUR_GBP": 0.0001,
}

IS_FRAC      = 4/6
S5_PER_M1    = 12

HIGH_THRS    = [30.0, 60.0, 100.0, 150.0, 250.0]
LOW_THRS     = [30.0, 60.0, 100.0, 150.0, 250.0]
TP_PIPS      = [1.0, 2.0, 3.0]            # TP exit (uPnL>=tp → exit)
HOLD_MIN_GRID  = [1, 2, 5, 10, 30]          # time-stop in minutes
BUCKET_M1_GRID = [1, 2, 3]                  # how many M1 bars the whipsaw can span


@njit(cache=True)
def s5_momentum(closes: np.ndarray, pip: float) -> np.ndarray:
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        out[i] = (closes[i] - closes[i-1]) / pip * 12.0
    return out


@njit(cache=True, parallel=True)
def sweep_kernel(s5_mom: np.ndarray, s5_close: np.ndarray, s5_spread_pips: np.ndarray,
                 high_thrs: np.ndarray, low_thrs: np.ndarray, tp_pips: np.ndarray,
                 hold_s5_grid: np.ndarray, bucket_s5_grid: np.ndarray,
                 is_end_s5: int) -> np.ndarray:
    """s5_close is in PIP UNITS. Subtracting two = signed pips directly.
    pnl_net subtracts spread (pips at entry bar)."""
    n_h = len(high_thrs); n_l = len(low_thrs); n_t = len(tp_pips)
    n_hold = len(hold_s5_grid); n_bk = len(bucket_s5_grid)
    n_cfg = n_h * n_l * n_t * n_hold * n_bk
    out = np.zeros((n_cfg, 10), dtype=np.float64)
    n_s5 = len(s5_mom)

    for ci in prange(n_cfg):
        a = n_l * n_t * n_hold * n_bk
        b = n_t * n_hold * n_bk
        c = n_hold * n_bk
        hi = ci // a
        lo = (ci % a) // b
        ti = (ci % b) // c
        hd = (ci % c) // n_bk
        bk = ci % n_bk
        h_thr = high_thrs[hi]; l_thr = low_thrs[lo]; tp = tp_pips[ti]
        hold = hold_s5_grid[hd]; bucket_size = bucket_s5_grid[bk]

        trades = 0; is_n = 0; oos_n = 0
        is_net = 0.0; oos_net = 0.0
        n_long = 0; wins = 0; n_tp = 0
        cum = 0.0; peak_curve = 0.0; oos_dd = 0.0
        in_pos = 0; exit_at = -1; entry_idx = -1; direction = 0; entry_px = 0.0

        cur_bucket = -1
        max_v = -1e18; max_idx = -1
        min_v =  1e18; min_idx = -1

        for i in range(1, n_s5):
            bucket = i // bucket_size
            if bucket != cur_bucket:
                cur_bucket = bucket
                max_v = -1e18; max_idx = -1
                min_v =  1e18; min_idx = -1

            v = s5_mom[i]

            if in_pos == 0:
                # Signal: peak-then-trough → LONG; trough-then-peak → SHORT
                if v <= -l_thr and max_v > h_thr and max_idx < i and max_idx >= 0:
                    direction = 1
                    in_pos = 1
                    entry_idx = i
                    entry_px  = s5_close[i]
                    exit_at = i + hold
                elif v >= h_thr and min_v < -l_thr and min_idx < i and min_idx >= 0:
                    direction = -1
                    in_pos = 1
                    entry_idx = i
                    entry_px  = s5_close[i]
                    exit_at = i + hold
            else:
                # uPnL right now (pips, before spread)
                cur_px = s5_close[i]
                upnl = (cur_px - entry_px) * direction
                exit_now = 0
                exit_reason = 0
                if upnl >= tp:
                    # TP hit at current S5 close
                    exit_now = 1; exit_reason = 1
                elif i >= exit_at or i >= n_s5 - 1:
                    exit_now = 1; exit_reason = 2

                if exit_now:
                    sp_pips = s5_spread_pips[entry_idx]
                    pnl_net = upnl - sp_pips
                    trades += 1
                    cum += pnl_net
                    if cum > peak_curve: peak_curve = cum
                    dd = cum - peak_curve
                    if dd < oos_dd: oos_dd = dd
                    if direction == 1: n_long += 1
                    if exit_reason == 1: n_tp += 1
                    if entry_idx < is_end_s5:
                        is_n += 1; is_net += pnl_net
                    else:
                        oos_n += 1; oos_net += pnl_net
                        if pnl_net > 0: wins += 1
                    in_pos = 0
                    exit_at = -1

            if v > max_v:
                max_v = v; max_idx = i
            if v < min_v:
                min_v = v; min_idx = i

        out[ci, 0] = trades
        out[ci, 1] = is_n;   out[ci, 2] = oos_n
        out[ci, 3] = is_net; out[ci, 4] = oos_net
        out[ci, 5] = oos_dd
        out[ci, 6] = wins
        out[ci, 7] = n_long
        out[ci, 8] = n_tp
    return out


def run_pair(pair: str, pip: float) -> pd.DataFrame:
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists():
        return pd.DataFrame()

    t = time.time()
    df = pq.read_table(path, columns=["timestamp","close","bid_c","ask_c"]).to_pandas()
    df = df.sort_values("timestamp").reset_index(drop=True)
    last = df["timestamp"].iloc[-1]
    if hasattr(last, "tz_localize"):
        cutoff = last - pd.Timedelta(days=365)
    else:
        cutoff = last - np.timedelta64(365, "D")
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    n_s5 = len(df)
    if n_s5 < 100_000:
        return pd.DataFrame()

    # Close in pip units so kernel sees pnl directly in pips
    close_pips  = (df["close"].to_numpy(np.float64) / pip)
    spread_pips = ((df["ask_c"] - df["bid_c"]) / pip).to_numpy(np.float64)
    s5_mom      = s5_momentum(df["close"].to_numpy(np.float64), pip)

    is_end_s5 = int(n_s5 * IS_FRAC)
    high_thrs = np.array(HIGH_THRS, dtype=np.float64)
    low_thrs  = np.array(LOW_THRS,  dtype=np.float64)
    tp_arr    = np.array(TP_PIPS,   dtype=np.float64)
    hold_arr  = np.array([m * 12 for m in HOLD_MIN_GRID], dtype=np.int64)
    bucket_arr= np.array([b * S5_PER_M1 for b in BUCKET_M1_GRID], dtype=np.int64)

    stats = sweep_kernel(s5_mom, close_pips, spread_pips,
                         high_thrs, low_thrs, tp_arr,
                         hold_arr, bucket_arr, is_end_s5)

    rows = []
    n_h = len(high_thrs); n_l = len(low_thrs); n_t = len(tp_arr)
    n_hold = len(hold_arr); n_bk = len(bucket_arr)
    days_oos = (n_s5 - is_end_s5) * 5 / 86400.0
    days_is  = is_end_s5 * 5 / 86400.0
    a = n_l * n_t * n_hold * n_bk
    b = n_t * n_hold * n_bk
    c = n_hold * n_bk
    for ci in range(n_h * n_l * n_t * n_hold * n_bk):
        hi = ci // a
        lo = (ci % a) // b
        ti = (ci % b) // c
        hd = (ci % c) // n_bk
        bk = ci % n_bk
        trades, is_n, oos_n, is_net, oos_net, oos_dd, wins, n_long, n_tp, _ = stats[ci]
        if trades == 0:
            continue
        rows.append({
            "pair":     pair,
            "high_thr": HIGH_THRS[hi],
            "low_thr":  LOW_THRS[lo],
            "tp_pips":  TP_PIPS[ti],
            "hold_min": HOLD_MIN_GRID[hd],
            "bucket_m1":BUCKET_M1_GRID[bk],
            "trades":   int(trades),
            "is_n":     int(is_n),
            "oos_n":    int(oos_n),
            "is_pd":    round(is_net  / max(days_is,  1), 2),
            "oos_pd":   round(oos_net / max(days_oos, 1), 2),
            "is_net":   round(is_net, 1),
            "oos_net":  round(oos_net, 1),
            "oos_dd":   round(oos_dd, 1),
            "oos_wr":   round(wins / max(oos_n, 1) * 100, 1),
            "tp_rate":  round(n_tp / max(trades, 1) * 100, 1),
            "n_long_pct": round(n_long / max(trades, 1) * 100, 1),
        })
    best_is = max((r["is_pd"] for r in rows), default=0)
    print(f"  [{pair}] {n_s5:,} S5 bars ({time.time()-t:.1f}s)  "
          f"is_pd[max]={best_is:+.2f}", flush=True)
    return pd.DataFrame(rows)


def main():
    n_cfg = len(HIGH_THRS) * len(LOW_THRS) * len(TP_PIPS) * len(HOLD_MIN_GRID) * len(BUCKET_M1_GRID)
    print("="*100)
    print("  S5 whipsaw STREAMING — TP+time exit, hold sweep, bucket sweep")
    print(f"  IS_FRAC = {IS_FRAC:.2f}   configs/pair: {n_cfg}")
    print(f"  high_thr: {HIGH_THRS}")
    print(f"  low_thr : {LOW_THRS}")
    print(f"  tp_pips : {TP_PIPS}")
    print(f"  hold_min: {HOLD_MIN_GRID}")
    print(f"  bucket_m1 (whipsaw span): {BUCKET_M1_GRID}")
    print(f"  pairs: {len(PAIRS)}")
    print("="*100, flush=True)

    # JIT warm
    _arr = np.zeros(100); _spr = np.zeros(100)
    _ = s5_momentum(_arr, 0.0001)
    sweep_kernel(_arr, _arr, _spr,
                 np.array([10.0]), np.array([10.0]), np.array([1.0]),
                 np.array([12], dtype=np.int64), np.array([12], dtype=np.int64), 50)

    t0 = time.time()
    all_dfs = []
    for pair, pip in PAIRS.items():
        d = run_pair(pair, pip)
        if not d.empty:
            all_dfs.append(d)
        gc.collect()
    if not all_dfs:
        print("(no rows)"); return
    full = pd.concat(all_dfs, ignore_index=True)
    out_path = OUT / "s5_whipsaw_streaming.csv"
    full.to_csv(out_path, index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s  rows: {len(full)}")
    print(f"  → {out_path}")

    cand = full[(full.is_net > 0) & (full.oos_net > 0) & (full.oos_n >= 20)].copy()
    print(f"\n  IS+OOS+ + oos_n>=20: {len(cand)}/{len(full)}")
    if len(cand):
        print("\n  Top 20 by oos_pd (robust sample):")
        cols = ["pair","high_thr","low_thr","tp_pips","hold_min","bucket_m1",
                "is_pd","oos_pd","oos_dd","oos_wr","oos_n","tp_rate","n_long_pct"]
        print(cand.sort_values("oos_pd", ascending=False).head(20)[cols].to_string(index=False))

        print("\n  Best per pair (IS+OOS+ ∧ oos_n>=20):")
        bp = cand.sort_values(["pair","oos_pd"], ascending=[True,False]).groupby("pair").head(1)
        print(bp[cols].to_string(index=False))
        print(f"\n  Σ OOS pd best-per-pair: {bp.oos_pd.sum():+.2f}")

        print("\n  Per (hold_min, tp_pips, bucket_m1) — count + Σ:")
        g = cand.groupby(["hold_min","tp_pips","bucket_m1"]).agg(
            npairs=("pair","nunique"), n=("pair","count"),
            sum_oos_pd=("oos_pd","sum"), mean_oos_pd=("oos_pd","mean"),
        ).reset_index().sort_values("sum_oos_pd", ascending=False)
        print(g.head(12).to_string(index=False))
    else:
        print("\n  No robust survivors. Showing best IS-pd anyway:")
        top = full[full.oos_n >= 20].sort_values("is_pd", ascending=False).head(15)
        cols = ["pair","high_thr","low_thr","tp_pips","hold_min","bucket_m1",
                "trades","is_pd","oos_pd","oos_wr","oos_n","tp_rate","n_long_pct"]
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
