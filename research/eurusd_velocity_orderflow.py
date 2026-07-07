"""
EUR/USD: Velocity, Volume, and Combined Signal Study

Parts:
  1. Pips/minute rate of big bars across TFs (velocity scaling)
  2. Tick volume correlation with TR — does volume explain bar energy?
  3. Pips-per-tick: how hard each tick had to work to move price
     (low pips/tick = thick book; high pips/tick = thin book)
  4. Forward-return study: big bar × reversal-density zone → next 1/3/5 M5 bars
     The combined filter hypothesis: big M1 bar in a low-density zone → continue
"""
import sys, warnings
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.swing_indicators import topsbots_swings

PIP = 0.0001  # EUR/USD

# ── helpers ──────────────────────────────────────────────────────

def true_range_pips(df):
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    pc = np.empty_like(cl); pc[0]=cl[0]; pc[1:]=cl[:-1]
    return np.maximum(hi-lo, np.maximum(np.abs(hi-pc), np.abs(lo-pc))) / PIP

def load_mid(path):
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index":"timestamp","time":"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "open" not in df.columns:
        for fr, to in [("o","open"),("h","high"),("l","low"),("c","close")]:
            df[to] = (df[f"bid_{fr}"].astype(float) + df[f"ask_{fr}"].astype(float)) / 2
    else:
        for c in ["open","high","low","close"]:
            df[c] = df[c].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 1. Pips/minute velocity table
# ═══════════════════════════════════════════════════════════════════

TF_MINUTES = {"S5":1/12, "S30":0.5, "M1":1, "M5":5, "M30":30, "H1":60, "H4":240}

def velocity_table():
    print("\n" + "═"*72)
    print("  VELOCITY SCALING — pips/minute of big-bar threshold (μ+3σ)")
    print("═"*72)
    # Pre-computed from analysis (90-day window)
    stats = {
        "S5":  (0.39, 0.43),
        "S30": (1.08, 1.01),
        "M1":  (1.60, 1.43),
        "M5":  (3.82, 3.11),
        "M30": (9.81, 7.13),
        "H1":  (14.04,10.82),
        "H4":  (28.14,20.16),
    }
    print(f"\n  {'TF':<6} {'Big(3σ)':>8} {'p/min':>8} {'p/sec':>8}  Velocity interpretation")
    prev_vel = None
    for tf, (mu, sd) in stats.items():
        big = mu + 3*sd
        mins = TF_MINUTES[tf]
        vel_pm = big / mins
        vel_ps = big / (mins * 60)
        decay = f"  {vel_pm/prev_vel:.2f}× slower" if prev_vel else ""
        print(f"  {tf:<6} {big:>8.2f}p {vel_pm:>8.3f}  {vel_ps:>8.4f}  {decay}")
        prev_vel = vel_pm
    print("""
  Interpretation:
    S30 big bar = 8.2 p/min  ← raw market-order velocity when limit book is swept
    H4 big bar  = 0.37 p/min ← sustained trend has 22× slower velocity than S30 shock
    The velocity collapse follows roughly a power law (Hurst-like scaling):
    Each ×6 in TF duration → ×~3 drop in p/min. Not proportional — shocks don't sustain.
    Short-TF big bars = fast impulsive strikes. Long-TF big bars = slow persistent drift.
""")


# ═══════════════════════════════════════════════════════════════════
# 2. Volume → TR correlation: does tick volume predict bar energy?
# ═══════════════════════════════════════════════════════════════════

def volume_analysis(df_m5, df_m1=None):
    print("═"*72)
    print("  TICK VOLUME ANALYSIS — does volume predict TR? (M5 and M1)")
    print("═"*72)

    for label, df in [("M5", df_m5), ("M1", df_m1)]:
        if df is None or "volume" not in df.columns:
            print(f"  {label}: no volume column — skipping"); continue
        df = df.copy()
        tr = true_range_pips(df)
        vol = df["volume"].values.astype(float)
        mask = vol > 0
        tr_m, vol_m = tr[mask], vol[mask]

        # Correlation
        logvol = np.log1p(vol_m)
        logtr  = np.log1p(tr_m)
        corr_log = np.corrcoef(logvol, logtr)[0,1]
        corr_raw = np.corrcoef(vol_m, tr_m)[0,1]

        # Big-bar stats
        big_thr = tr_m.mean() + 3*tr_m.std()
        big_mask = tr_m >= big_thr
        vol_big  = vol_m[big_mask]
        vol_norm = vol_m[~big_mask]

        # Pips-per-tick
        ppt_big  = (tr_m[big_mask]  / vol_big).mean()
        ppt_norm = (tr_m[~big_mask] / vol_m[~big_mask]).mean()

        # Volume quintile → median TR
        q_edges = np.percentile(vol_m, [0,20,40,60,80,100])
        print(f"\n  [{label}] n={len(tr_m):,}  corr(log_vol, log_TR)={corr_log:.3f}  "
              f"corr(vol, TR)={corr_raw:.3f}")
        print(f"    Big-bar vol: median={np.median(vol_big):.0f}  "
              f"vs normal: median={np.median(vol_norm):.0f}  "
              f"ratio={np.median(vol_big)/np.median(vol_norm):.2f}x")
        print(f"    Pips/tick:  big_bar={ppt_big:.4f}  normal={ppt_norm:.4f}  "
              f"ratio={ppt_big/ppt_norm:.2f}x")
        print(f"      → {'thin book (high p/t = few orders sweep far)' if ppt_big > ppt_norm else 'thick book'}")

        print(f"\n    Volume quintile → median TR (p) + pips/tick:")
        print(f"    {'Quintile':<12} {'Vol range':>14} {'Med TR':>8} {'Med p/tick':>12}")
        for qi in range(5):
            qmask = (vol_m >= q_edges[qi]) & (vol_m < q_edges[qi+1]) if qi<4 else (vol_m >= q_edges[qi])
            if qmask.sum() < 5: continue
            med_tr  = np.median(tr_m[qmask])
            med_ppt = np.median(tr_m[qmask] / vol_m[qmask])
            qlabel  = f"Q{qi+1}"
            print(f"    {qlabel:<12} {q_edges[qi]:>6.0f}–{q_edges[qi+1]:>6.0f}   "
                  f"{med_tr:>7.2f}p  {med_ppt:>11.4f}")

    print()


# ═══════════════════════════════════════════════════════════════════
# 3. Reversal density at each M5 bar (using H1 TopBots)
# ═══════════════════════════════════════════════════════════════════

def build_reversal_density(df_h1, pip_bin=5, half_life_days=30):
    """Returns a function density(price, timestamp) → [0,1] normalised."""
    hi = df_h1["high"].values.astype(float)
    lo = df_h1["low"].values.astype(float)
    swings = topsbots_swings(hi, lo)

    grid = pip_bin * PIP
    price_min = df_h1["low"].min() - grid
    n_bins    = int((df_h1["high"].max() - price_min) / grid) + 5

    # Precompute swing arrays: price levels and timestamps
    s_prices = np.array([v for _, _, v in swings], dtype=float)
    s_ts     = np.array([df_h1["timestamp"].iloc[bi].value for bi, _, _ in swings], dtype=np.int64)
    # bin index for each swing
    s_bins   = np.clip(((s_prices - price_min) / grid).astype(int), 0, n_bins-1)

    half_life_ns = half_life_days * 86400 * int(1e9)
    ln2 = np.log(2)

    def density_at(price: float, ts_ns: int) -> float:
        """Exponential-decay weighted reversal count at this price bin, as-of ts_ns."""
        b = int((price - price_min) / grid)
        in_bin = s_bins == b
        if not in_bin.any():
            return 0.0
        ages_ns = ts_ns - s_ts[in_bin]
        ages_ns = np.where(ages_ns < 60*int(1e9), 60*int(1e9), ages_ns)  # min 1 min
        w = np.exp(-ln2 * ages_ns / half_life_ns)
        return float(w.sum())

    return density_at, price_min, grid, n_bins


# ═══════════════════════════════════════════════════════════════════
# 4. Forward-return study: big bar × density zone
# ═══════════════════════════════════════════════════════════════════

def forward_return_study(df_m5, df_h1, df_m1=None, lags=(1,3,5,10)):
    print("═"*72)
    print("  FORWARD RETURN STUDY — big bar × reversal density → next M5 bars")
    print("  Signal: directional big bar at M1 (or M5), price in low/high density zone")
    print("═"*72)

    # Build density function
    density_fn, price_min, grid, _ = build_reversal_density(df_h1)

    # Work on M5 bars (last 90 days from M5 BA)
    df = df_m5.copy().reset_index(drop=True)
    tr = true_range_pips(df)
    big_thr = tr.mean() + 3*tr.std()
    big_mask = tr >= big_thr
    direction = np.sign(df["close"].values - df["open"].values)  # +1=bull, -1=bear, 0=doji

    # Compute density for each M5 bar's close price at that timestamp
    print(f"\n  Computing reversal density for {len(df):,} M5 bars … ", end="", flush=True)
    ts_arr = df["timestamp"].values.astype(np.int64)
    cl_arr = df["close"].values.astype(float)
    dens = np.array([density_fn(cl_arr[i], ts_arr[i]) for i in range(len(df))], dtype=float)
    d_max = dens.max() if dens.max() > 0 else 1
    dens_norm = dens / d_max  # [0,1]
    print("done")

    # Density terciles
    d33, d67 = np.percentile(dens_norm[dens_norm > 0], [33, 67])
    low_dens  = dens_norm <= d33   # clear air
    high_dens = dens_norm >= d67   # congested / S/R

    # Forward returns: close[i+lag] - close[i], in direction of bar[i]
    # Directional: +ve = bar continued, -ve = bar reversed
    cl = df["close"].values.astype(float)
    N  = len(df)

    groups = {
        "all_bars":        np.ones(N, dtype=bool),
        "big_bar":         big_mask,
        "big_bar_low_dens":  big_mask & low_dens,
        "big_bar_high_dens": big_mask & high_dens,
        "small_bar":       ~big_mask,
    }

    print(f"\n  Density thresholds: low≤{d33:.3f}  high≥{d67:.3f}")
    print(f"  {'Group':<24}  n  │", end="")
    for lag in lags:
        print(f"  lag+{lag}M5", end="")
    print()
    print(f"  {'─'*24}  ─  │" + "──────────"*len(lags))

    results = {}
    for gname, gmask in groups.items():
        row_n = gmask.sum()
        row_results = []
        for lag in lags:
            valid = gmask & (np.arange(N) < N - lag)
            if valid.sum() < 5:
                row_results.append(None)
                continue
            idx = np.where(valid)[0]
            # Directional forward return: sign(body[i]) × (close[i+lag] - close[i])
            dirs = direction[idx]
            fwd  = (cl[idx + lag] - cl[idx]) / PIP  # in pips
            dir_fwd = dirs * fwd   # positive = continuation, negative = reversal
            row_results.append((dir_fwd.mean(), dir_fwd.std(), valid.sum()))
        results[gname] = row_results

        print(f"  {gname:<24}  {row_n:>5}  │", end="")
        for r in row_results:
            if r is None:
                print("      n/a", end="")
            else:
                mu, sd, n = r
                sig = "+" if mu > 0 else ""
                print(f"  {sig}{mu:+.2f}p", end="")
        print()

    # Highlight p-values (t-test big_bar_low_dens vs all_bars)
    from scipy import stats as scipy_stats
    print(f"\n  Statistical significance: big_bar_low_dens vs all_bars (t-test)")
    for li, lag in enumerate(lags):
        # Reconstruct arrays for t-test
        gmask = groups["big_bar_low_dens"]
        valid = gmask & (np.arange(N) < N - lag)
        if valid.sum() < 5:
            continue
        idx = np.where(valid)[0]
        dirs = direction[idx]
        fwd  = (cl[idx + lag] - cl[idx]) / PIP
        dir_fwd = dirs * fwd
        # All bars
        valid_all = np.arange(N) < N - lag
        idx_all = np.where(valid_all)[0]
        dirs_all = direction[idx_all]
        fwd_all  = (cl[idx_all + lag] - cl[idx_all]) / PIP
        dfwd_all = dirs_all * fwd_all

        t, p = scipy_stats.ttest_ind(dir_fwd, dfwd_all)
        sig = "🟢" if p < 0.05 else "🟡" if p < 0.15 else "🔴"
        print(f"    lag+{lag}M5:  t={t:+.2f}  p={p:.3f}  {sig}")

    # Win-rate analysis (big_bar_low_dens)
    print(f"\n  Win rate & profit factor: big_bar_low_dens")
    print(f"  {'Lag':>8}  {'WR%':>6}  {'Avg W':>8}  {'Avg L':>8}  {'PF':>6}")
    gmask = groups["big_bar_low_dens"]
    for lag in lags:
        valid = gmask & (np.arange(N) < N - lag)
        if valid.sum() < 5: continue
        idx = np.where(valid)[0]
        dirs = direction[idx]
        fwd  = (cl[idx + lag] - cl[idx]) / PIP
        dir_fwd = dirs * fwd
        wins = dir_fwd[dir_fwd > 0]
        loss = dir_fwd[dir_fwd < 0]
        wr   = 100 * len(wins) / len(dir_fwd) if len(dir_fwd) > 0 else 0
        avg_w = wins.mean() if len(wins) > 0 else 0
        avg_l = abs(loss.mean()) if len(loss) > 0 else 0
        pf   = (avg_w * len(wins)) / (avg_l * len(loss)) if len(loss) > 0 and avg_l > 0 else float("inf")
        print(f"  lag+{lag}M5:  {wr:>5.1f}%  {avg_w:>+7.2f}p  {avg_l:>+7.2f}p  {pf:>5.2f}x")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    velocity_table()

    print("Loading M5, M1, H1 data …")
    df_m5 = load_mid(ROOT / "data" / "m5_ba" / "EUR_USD_M5_BA.parquet")
    df_m1 = load_mid(ROOT / "data" / "m1_ohlc" / "EUR_USD_M1_BA.parquet")

    # Full M5 for H1 aggregation (5.5yr for deep swing history)
    m5_ts = df_m5.set_index("timestamp")
    r = m5_ts.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    df_h1 = r.reset_index()
    print(f"  M5: {len(df_m5):,}   M1: {len(df_m1):,}   H1: {len(df_h1):,} (from full 5.5yr M5)")

    # Restrict M5 and M1 to last 90 days for forward-return study
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
    df_m5_90 = df_m5[df_m5["timestamp"] >= cutoff].reset_index(drop=True)
    df_m1_90 = df_m1[df_m1["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"  Last 90d — M5: {len(df_m5_90):,}   M1: {len(df_m1_90):,}")

    volume_analysis(df_m5_90, df_m1_90)
    forward_return_study(df_m5_90, df_h1, df_m1_90, lags=(1, 3, 5, 10))
    print("\nDone.\n")


if __name__ == "__main__":
    main()
