"""Weighted-sum trigger experiment — FADE direction.

Mirror of backtest_weighted_sum_trigger.py.  The signal-following version
was decisively negative on all 12 pairs (IS pd −49 to −399 p/d) → flipping
the direction should expose the contrarian edge.

Entry rule (fade / contrarian):
  ws  >=  +ws_thr  → SHORT at current S5 close
  ws  <=  -ws_thr  → LONG  at current S5 close

Exit:
  TP at +tp_pips uPnL, OR
  time stop at hold_min minutes, whichever first.

Sweep: ws_thr × tp_pips × hold_min × pair.

Spread cost: 1× BA spread at entry bar (round-trip).
"""
import time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit, prange

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
S5_PER_MIN   = 12

# Window weights (mirror live dashboard's WINDOW_WEIGHTS)
W_S5, W_M1, W_5M, W_15M, W_1H, W_4H, W_24H = 0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25
W_TOTAL = W_S5 + W_M1 + W_5M + W_15M + W_1H + W_4H + W_24H  # = 1.0

# Lookback in S5 bars per window (and minutes per window)
LB_S5  = (1,    1/60.0)        # 5s     /  1/60 min
LB_M1  = (12,   1.0)
LB_5M  = (60,   5.0)
LB_15M = (180,  15.0)
LB_1H  = (720,  60.0)
LB_4H  = (2880, 240.0)
LB_24H = (17280, 1440.0)

# Sweep grids
WS_THRS    = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]   # weighted_sum (pips/min) threshold
TP_PIPS    = [1.0, 2.0, 3.0, 5.0]
HOLD_MINS  = [1, 2, 5, 10, 30]


@njit(cache=True)
def compute_weighted_sum(close_pips: np.ndarray) -> np.ndarray:
    """For each S5 bar, compute the weighted sum signal.  Returns array of
    same length; first 17280 values are NaN (insufficient lookback for 24h)."""
    n = len(close_pips)
    out = np.full(n, np.nan, dtype=np.float64)
    lb24 = 17280
    for t in range(lb24, n):
        # Each component = (close[t] - close[t-lb]) / minutes_in_window
        s5  = (close_pips[t] - close_pips[t-1])     / (1/60.0)
        m1  = (close_pips[t] - close_pips[t-12])    / 1.0
        m5  = (close_pips[t] - close_pips[t-60])    / 5.0
        m15 = (close_pips[t] - close_pips[t-180])   / 15.0
        h1  = (close_pips[t] - close_pips[t-720])   / 60.0
        h4  = (close_pips[t] - close_pips[t-2880])  / 240.0
        h24 = (close_pips[t] - close_pips[t-17280]) / 1440.0
        out[t] = (W_S5*s5 + W_M1*m1 + W_5M*m5 + W_15M*m15 +
                  W_1H*h1 + W_4H*h4 + W_24H*h24)
    return out


@njit(cache=True, parallel=True)
def sweep_kernel(ws: np.ndarray, close_pips: np.ndarray, spread_pips: np.ndarray,
                 ws_thrs: np.ndarray, tp_pips_arr: np.ndarray, hold_s5_arr: np.ndarray,
                 is_end_s5: int) -> np.ndarray:
    n_w = len(ws_thrs); n_t = len(tp_pips_arr); n_h = len(hold_s5_arr)
    n_cfg = n_w * n_t * n_h
    out = np.zeros((n_cfg, 9), dtype=np.float64)
    n_s5 = len(ws)

    for ci in prange(n_cfg):
        wi = ci // (n_t * n_h)
        ti = (ci % (n_t * n_h)) // n_h
        hd = ci % n_h
        w_thr = ws_thrs[wi]; tp = tp_pips_arr[ti]; hold = hold_s5_arr[hd]

        trades = 0; is_n = 0; oos_n = 0
        is_net = 0.0; oos_net = 0.0
        n_long = 0; wins = 0; n_tp = 0
        cum = 0.0; peak_curve = 0.0; oos_dd = 0.0
        in_pos = 0; exit_at = -1; entry_idx = -1; direction = 0; entry_px = 0.0

        for i in range(17280, n_s5):
            v = ws[i]
            if np.isnan(v):
                continue

            if in_pos == 0:
                # FADE: ws >= +thr → SHORT (was LONG);  ws <= -thr → LONG (was SHORT)
                if v >= w_thr:
                    direction = -1
                    in_pos = 1; entry_idx = i; entry_px = close_pips[i]; exit_at = i + hold
                elif v <= -w_thr:
                    direction = 1
                    in_pos = 1; entry_idx = i; entry_px = close_pips[i]; exit_at = i + hold
            else:
                upnl = (close_pips[i] - entry_px) * direction
                exit_now = 0; exit_reason = 0
                if upnl >= tp:
                    exit_now = 1; exit_reason = 1
                elif i >= exit_at or i >= n_s5 - 1:
                    exit_now = 1; exit_reason = 2
                if exit_now:
                    sp_p = spread_pips[entry_idx]
                    pnl_net = upnl - sp_p
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
                    in_pos = 0; exit_at = -1

        out[ci, 0] = trades
        out[ci, 1] = is_n;  out[ci, 2] = oos_n
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
    cutoff = last - pd.Timedelta(days=365)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    n_s5 = len(df)
    if n_s5 < 200_000:
        return pd.DataFrame()

    close_pips  = (df["close"].to_numpy(np.float64) / pip)
    spread_pips = ((df["ask_c"] - df["bid_c"]) / pip).to_numpy(np.float64)
    ws = compute_weighted_sum(close_pips)

    is_end_s5 = int(n_s5 * IS_FRAC)
    ws_arr   = np.array(WS_THRS, dtype=np.float64)
    tp_arr   = np.array(TP_PIPS, dtype=np.float64)
    hold_arr = np.array([m * S5_PER_MIN for m in HOLD_MINS], dtype=np.int64)

    stats = sweep_kernel(ws, close_pips, spread_pips,
                         ws_arr, tp_arr, hold_arr, is_end_s5)

    rows = []
    n_w = len(ws_arr); n_t = len(tp_arr); n_h = len(hold_arr)
    days_oos = (n_s5 - is_end_s5) * 5 / 86400.0
    days_is  = is_end_s5 * 5 / 86400.0
    for ci in range(n_w * n_t * n_h):
        wi = ci // (n_t * n_h)
        ti = (ci % (n_t * n_h)) // n_h
        hd = ci % n_h
        trades, is_n, oos_n, is_net, oos_net, oos_dd, wins, n_long, n_tp = stats[ci]
        if trades == 0:
            continue
        rows.append({
            "pair":     pair,
            "ws_thr":   WS_THRS[wi],
            "tp_pips":  TP_PIPS[ti],
            "hold_min": HOLD_MINS[hd],
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
    n_cfg = len(WS_THRS) * len(TP_PIPS) * len(HOLD_MINS)
    print("="*100)
    print("  Weighted-sum FADE trigger (ws>=+thr → SHORT, <=−thr → LONG)")
    print(f"  IS_FRAC = {IS_FRAC:.2f}   configs/pair: {n_cfg}")
    print(f"  ws_thr: {WS_THRS}    tp_pips: {TP_PIPS}    hold_min: {HOLD_MINS}")
    print(f"  pairs: {len(PAIRS)}    weights: {(W_S5,W_M1,W_5M,W_15M,W_1H,W_4H,W_24H)}")
    print("="*100, flush=True)

    # JIT warm
    _arr = np.zeros(50000)
    _ = compute_weighted_sum(_arr)
    sweep_kernel(_arr, _arr, _arr,
                 np.array([1.0]), np.array([1.0]),
                 np.array([12], dtype=np.int64), 25000)

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
    out_path = OUT / "weighted_sum_fade.csv"
    full.to_csv(out_path, index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s  rows: {len(full)}")
    print(f"  → {out_path}")

    cand = full[(full.is_net > 0) & (full.oos_net > 0) & (full.oos_n >= 20)].copy()
    print(f"\n  IS+OOS+ + oos_n>=20: {len(cand)}/{len(full)}")
    if len(cand):
        print("\n  Top 20 by oos_pd:")
        cols = ["pair","ws_thr","tp_pips","hold_min",
                "is_pd","oos_pd","oos_dd","oos_wr","oos_n","tp_rate","n_long_pct"]
        print(cand.sort_values("oos_pd", ascending=False).head(20)[cols].to_string(index=False))

        print("\n  Best per pair:")
        bp = cand.sort_values(["pair","oos_pd"], ascending=[True,False]).groupby("pair").head(1)
        print(bp[cols].to_string(index=False))
        print(f"\n  Σ OOS pd best-per-pair: {bp.oos_pd.sum():+.2f}")

        print("\n  Per (ws_thr, tp_pips, hold_min) — npairs + Σ:")
        g = cand.groupby(["ws_thr","tp_pips","hold_min"]).agg(
            npairs=("pair","nunique"),
            sum_oos_pd=("oos_pd","sum"),
            mean_oos_pd=("oos_pd","mean"),
        ).reset_index().sort_values("sum_oos_pd", ascending=False)
        print(g.head(12).to_string(index=False))
    else:
        print("\n  No robust survivors. Top 15 by IS pd:")
        top = full[full.oos_n >= 20].sort_values("is_pd", ascending=False).head(15)
        cols = ["pair","ws_thr","tp_pips","hold_min",
                "trades","is_pd","oos_pd","oos_wr","oos_n","tp_rate","n_long_pct"]
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
