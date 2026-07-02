"""S5 whipsaw within an M1 bar → enter at next M1 open, hold 5 min.

Within one M1 bar (12 S5 bars), check whether S5 momentum BOTH:
  (a) peaked above +HIGH_THR pips/min
  (b) troughed below -LOW_THR pips/min

Entry direction by sequence within that M1 bar:
  LONG:  peak comes BEFORE trough   (up-then-down whipsaw)
  SHORT: trough comes BEFORE peak  (down-then-up whipsaw)

Enter at NEXT M1 open (mid). Hold 5 M1 bars (5 min). Exit at market at
M1 bar close. Spread cost deducted once per round trip from per-bar BA.

12 pairs, 1 year window. Sweep over HIGH_THR × LOW_THR.
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
HOLD_M1_BARS = 5            # 5 min hold
S5_PER_M1    = 12

# Threshold grids — S5 momentum is in pips/min
HIGH_THRS = [30.0, 60.0, 100.0, 150.0, 250.0]
LOW_THRS  = [30.0, 60.0, 100.0, 150.0, 250.0]


@njit(cache=True)
def s5_momentum(closes: np.ndarray, pip: float) -> np.ndarray:
    """pips/min between consecutive S5 closes."""
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        out[i] = (closes[i] - closes[i-1]) / pip * 12.0  # 12 = 60s / 5s
    return out


@njit(cache=True, parallel=True)
def sweep_kernel(s5_mom: np.ndarray, m1_open: np.ndarray, m1_close: np.ndarray,
                 m1_spread: np.ndarray,
                 high_thrs: np.ndarray, low_thrs: np.ndarray,
                 hold: int, is_end_m1: int) -> np.ndarray:
    """For each (high_thr, low_thr), scan M1 bars looking for whipsaw.
    Returns (#configs, 8) array of stats:
      [trades, is_n, oos_n, is_net, oos_net, oos_dd, oos_wr, n_long]
    """
    n_h = len(high_thrs); n_l = len(low_thrs)
    out = np.zeros((n_h * n_l, 8), dtype=np.float64)
    n_m1 = len(m1_open)

    for ci in prange(n_h * n_l):
        hi = ci // n_l; lo = ci % n_l
        h_thr = high_thrs[hi]; l_thr = low_thrs[lo]

        trades = 0; is_n = 0; oos_n = 0
        is_net = 0.0; oos_net = 0.0
        n_long = 0
        cum = 0.0; peak = 0.0; oos_dd = 0.0
        wins = 0

        for bi in range(n_m1 - hold - 1):
            # Look inside this M1 bar's S5 mom window
            s5_start = bi * S5_PER_M1
            s5_end   = s5_start + S5_PER_M1
            if s5_end > len(s5_mom):
                break

            # Find argmax + argmin of s5_mom within this window
            argmax = -1; max_v = -1e18
            argmin = -1; min_v =  1e18
            for k in range(s5_start, s5_end):
                v = s5_mom[k]
                if v > max_v:
                    max_v = v; argmax = k
                if v < min_v:
                    min_v = v; argmin = k

            if max_v < h_thr:
                continue
            if min_v > -l_thr:
                continue

            # Whipsaw confirmed; direction by sequence
            direction = 1 if argmax < argmin else -1
            # Enter at NEXT M1 open (bi+1 has already closed though — we use bi+2 open)
            # In live we'd enter at the M1 OPEN after the signal bar closes.
            # Conservatively use bi+1 open as the fill (signal bar bi has just closed).
            entry_bar = bi + 1
            exit_bar  = entry_bar + hold
            if exit_bar >= n_m1:
                break
            entry_px = m1_open[entry_bar]
            exit_px  = m1_close[exit_bar - 1]   # close of last hold bar
            pnl = (exit_px - entry_px) / 1.0 * direction
            # spread cost: round-trip (M1 mid → mid, deduct 1× avg spread)
            sp_pips = m1_spread[entry_bar]
            pnl_net = pnl - sp_pips

            trades += 1
            cum += pnl_net
            if cum > peak: peak = cum
            dd = cum - peak
            if dd < oos_dd: oos_dd = dd
            if direction == 1: n_long += 1

            if entry_bar < is_end_m1:
                is_n  += 1; is_net  += pnl_net
            else:
                oos_n += 1; oos_net += pnl_net
                if pnl_net > 0: wins += 1

        out[ci, 0] = trades
        out[ci, 1] = is_n;  out[ci, 2] = oos_n
        out[ci, 3] = is_net; out[ci, 4] = oos_net
        out[ci, 5] = oos_dd
        out[ci, 6] = wins
        out[ci, 7] = n_long
    return out


def run_pair(pair: str, pip: float) -> pd.DataFrame:
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists():
        print(f"  [skip] {pair} — no parquet"); return pd.DataFrame()

    t = time.time()
    df = pq.read_table(path, columns=["timestamp","open","high","low","close","bid_c","ask_c"]).to_pandas()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Restrict to last 365 days
    last = df["timestamp"].iloc[-1]
    if hasattr(last, "tz_localize"):
        cutoff = last - pd.Timedelta(days=365)
    else:
        cutoff = last - np.timedelta64(365, "D")
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    n_s5 = len(df)
    if n_s5 < 100_000:
        print(f"  [skip] {pair} — only {n_s5} S5 bars"); return pd.DataFrame()

    # Compute per-S5 mid close (already mid in the parquet) and per-S5 spread (pips)
    closes_s5 = df["close"].to_numpy(np.float64)
    spread_s5 = ((df["ask_c"] - df["bid_c"]) / pip).to_numpy(np.float64)
    s5_mom = s5_momentum(closes_s5, pip)

    # Aggregate to M1: open=first, close=last, spread=mean
    trim = (n_s5 // S5_PER_M1) * S5_PER_M1
    closes_s5 = closes_s5[:trim]
    spread_s5 = spread_s5[:trim]
    s5_mom    = s5_mom[:trim]
    opens_s5  = df["open"].to_numpy(np.float64)[:trim]
    m1_open   = opens_s5.reshape(-1, S5_PER_M1)[:, 0]
    m1_close  = closes_s5.reshape(-1, S5_PER_M1)[:, -1]
    m1_spread = spread_s5.reshape(-1, S5_PER_M1).mean(axis=1)
    n_m1      = len(m1_open)
    is_end_m1 = int(n_m1 * IS_FRAC)

    high_thrs = np.array(HIGH_THRS, dtype=np.float64)
    low_thrs  = np.array(LOW_THRS,  dtype=np.float64)
    stats = sweep_kernel(s5_mom, m1_open, m1_close, m1_spread,
                         high_thrs, low_thrs,
                         HOLD_M1_BARS, is_end_m1)

    rows = []
    n_h = len(high_thrs); n_l = len(low_thrs)
    days_oos = (n_m1 - is_end_m1) / 1440.0
    days_is  = is_end_m1 / 1440.0
    for ci in range(n_h * n_l):
        hi = ci // n_l; lo = ci % n_l
        trades, is_n, oos_n, is_net, oos_net, oos_dd, wins, n_long = stats[ci]
        if trades == 0:
            continue
        rows.append({
            "pair": pair,
            "high_thr": HIGH_THRS[hi],
            "low_thr":  LOW_THRS[lo],
            "trades":   int(trades),
            "is_n":     int(is_n),
            "oos_n":    int(oos_n),
            "is_pd":    round(is_net  / max(days_is,  1), 2),
            "oos_pd":   round(oos_net / max(days_oos, 1), 2),
            "is_net":   round(is_net, 1),
            "oos_net":  round(oos_net, 1),
            "oos_dd":   round(oos_dd, 1),
            "oos_wr":   round(wins / max(oos_n, 1) * 100, 1),
            "n_long_pct": round(n_long / max(trades, 1) * 100, 1),
        })
    print(f"  [{pair}] {n_s5:,} S5 bars → {n_m1:,} M1 bars ({time.time()-t:.1f}s) "
          f"is_pd[max]={max((r['is_pd'] for r in rows), default=0):+.2f}")
    return pd.DataFrame(rows)


def main():
    print("="*100)
    print("  S5 whipsaw within M1 bar — long/short by argmax/argmin sequence")
    print(f"  hold = {HOLD_M1_BARS} M1 bars  IS_FRAC = {IS_FRAC:.2f}")
    print(f"  high_thr grid: {HIGH_THRS}")
    print(f"  low_thr  grid: {LOW_THRS}")
    print(f"  pairs: {len(PAIRS)}")
    print("="*100)

    # JIT warm
    _arr = np.zeros(100); _spr = np.zeros(100)
    _ = s5_momentum(_arr, 0.0001)
    sweep_kernel(_arr, _arr[:10], _arr[:10], _arr[:10],
                 np.array([10.0]), np.array([10.0]), 1, 5)

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
    out_path = OUT / "s5_whipsaw_m1.csv"
    full.to_csv(out_path, index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s  rows: {len(full)}")
    print(f"  → {out_path}")

    # Filter: IS+OOS+
    cand = full[(full.is_net > 0) & (full.oos_net > 0)].copy()
    print(f"\n  IS+OOS+ candidates: {len(cand)}/{len(full)}")
    if len(cand):
        print("  Best per pair:")
        bp = cand.sort_values(["pair","oos_pd"], ascending=[True,False]).groupby("pair").head(1)
        cols = ["pair","high_thr","low_thr","is_pd","oos_pd","oos_dd","oos_wr","oos_n","n_long_pct"]
        print(bp[cols].to_string(index=False))
        print(f"\n  Σ OOS pd best-per-pair: {bp.oos_pd.sum():+.2f}")
    # Per (high,low) summary
    print("\n  Per (high_thr,low_thr) summary across pairs (IS+OOS+ count):")
    g = cand.groupby(["high_thr","low_thr"]).agg(
        npairs=("pair","nunique"), sum_oos_pd=("oos_pd","sum")
    ).reset_index().sort_values("sum_oos_pd", ascending=False)
    print(g.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
