"""
EUR/USD Cross-TF Shock Propagation + TopBots Reversal Density Map

Analyses:
  1. TR distribution across all available TFs (S5, S30, M1, M5, M30, H1)
  2. Cross-TF shock propagation: P(big_high | big_low) vs base rate
  3. "Breathing waves" — lag correlation of big-bar flags across TF pairs
  4. TopBots reversal density: time-decay weighted swing count per 5-pip bin

Usage:
    python3 research/eurusd_shock_propagation.py [--ref-days 90]
    (ref-days controls how much of each TF to use for cross-TF alignment)
"""
import sys, argparse, warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.swing_indicators import topsbots_swings

PIP = 0.0001   # EUR/USD


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def true_range_pips(df):
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    pc = np.empty_like(cl); pc[0] = cl[0]; pc[1:] = cl[:-1]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    return tr / PIP

def body_ratio(df):
    rng = df["high"].values.astype(float) - df["low"].values.astype(float)
    body = df["close"].values.astype(float) - df["open"].values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rng > 0, body / rng, np.nan)

def load_mid_ohlc(path, required_cols=None):
    """Load parquet with mid OHLC.  Bid/ask parquets → compute mid."""
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp", "time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Bid/ask-only parquets (no 'open' column)
    if "open" not in df.columns:
        for col in ["o","h","l","c"]:
            df[f"{'open' if col=='o' else 'high' if col=='h' else 'low' if col=='l' else 'close'}"] = \
                (df[f"bid_{col}"].values.astype(float) + df[f"ask_{col}"].values.astype(float)) / 2
    else:
        for col in ["open","high","low","close"]:
            df[col] = df[col].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def big_bar_mask(tr, sigma=3.0):
    """Return bool mask of bars ≥ mean + sigma*std."""
    thr = tr.mean() + sigma * tr.std()
    return tr >= thr, thr


# ════════════════════════════════════════════════════════════════
# 1. Load all available TFs for EUR/USD
# ════════════════════════════════════════════════════════════════

def load_all_tfs(ref_days):
    """Load each TF, restrict to last ref_days, return dict name→df."""
    tfs = {}

    paths = {
        "S5":  ROOT / "data" / "s5_ohlc"  / "EUR_USD_S5_BA.parquet",
        "S30": ROOT / "data" / "s30_ohlc" / "EUR_USD_S30_BA.parquet",
        "M1":  ROOT / "data" / "m1_ohlc"  / "EUR_USD_M1_BA.parquet",
        "M5":  ROOT / "data" / "m5_ba"    / "EUR_USD_M5_BA.parquet",
    }
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ref_days)

    for name, path in paths.items():
        if not path.exists():
            print(f"  ⚠ {name}: not found at {path}")
            continue
        try:
            df = load_mid_ohlc(path)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            if len(df) < 100:
                print(f"  ⚠ {name}: only {len(df)} bars in last {ref_days} days")
                continue
            tfs[name] = df
            print(f"  ✓ {name}: {len(df):,} bars  "
                  f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
                  f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  ⚠ {name}: error loading — {e}")

    # M5 aggregate → M30, H1 (use full 5.5yr M5 for H1 to get better swing coverage)
    m5_full_path = ROOT / "data" / "m5_ba" / "EUR_USD_M5_BA.parquet"
    if m5_full_path.exists():
        m5_full = load_mid_ohlc(m5_full_path)
        m5_ts = m5_full.set_index("timestamp")
        for label, rule in [("M30", "30min"), ("H1", "1h"), ("H4", "4h")]:
            r = m5_ts.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            rdf = r.reset_index()
            if label in ("M30",):
                rdf = rdf[rdf["timestamp"] >= cutoff].reset_index(drop=True)
            tfs[label] = rdf
            print(f"  ✓ {label} (agg from M5): {len(rdf):,} bars")

    return tfs


# ════════════════════════════════════════════════════════════════
# 2. TR distribution table
# ════════════════════════════════════════════════════════════════

ORDER = ["S5", "S30", "M1", "M5", "M30", "H1", "H4"]

def print_tr_summary(tfs):
    print("\n" + "═"*70)
    print("  TR DISTRIBUTION SUMMARY — EUR/USD")
    print("═"*70)
    print(f"  {'TF':<6}  {'bars':>8}  {'μ TR':>6}  {'σ TR':>6}  {'P50':>6}  {'P90':>6}  {'P99':>6}  {'Big(3σ)':>8}  {'%big':>5}")
    rows = []
    for name in ORDER:
        if name not in tfs:
            continue
        df = tfs[name]
        tr = true_range_pips(df)
        mu, sd = tr.mean(), tr.std()
        big = mu + 3 * sd
        p50, p90, p99 = np.percentile(tr, [50, 90, 99])
        pct_big = 100 * (tr >= big).mean()
        print(f"  {name:<6}  {len(df):>8,}  {mu:>6.2f}p  {sd:>6.2f}p  {p50:>6.2f}p  "
              f"{p90:>6.2f}p  {p99:>6.2f}p  {big:>8.2f}p  {pct_big:>4.1f}%")
        rows.append((name, tr, big, mu, sd))
    return rows  # list of (name, tr_array, big_thr, mu, sd)


# ════════════════════════════════════════════════════════════════
# 3. Cross-TF shock propagation
# ════════════════════════════════════════════════════════════════

TF_SECONDS = {"S5": 5, "S30": 30, "M1": 60, "M5": 300, "M30": 1800, "H1": 3600, "H4": 14400}

def align_tfs(df_low: pd.DataFrame, df_high: pd.DataFrame, sec_high: int) -> np.ndarray:
    """For each low-TF bar, return the row-index of the containing high-TF bar.
    Uses searchsorted: finds the last high-TF bar whose timestamp ≤ low-TF timestamp.
    Returns -1 for bars that precede the first high-TF bar.
    """
    lo_ts  = df_low["timestamp"].values.astype("int64")   # ns since epoch
    hi_ts  = df_high["timestamp"].values.astype("int64")  # ns since epoch
    hi_dur = sec_high * int(1e9)                          # bar duration in ns

    # searchsorted right → gives insertion point after all equal values
    # -1 gives the index of the last hi bar whose ts <= lo_ts
    pos = np.searchsorted(hi_ts, lo_ts, side="right") - 1  # shape (n_low,)

    # Validate: the matched high bar must actually contain the low bar
    # i.e., lo_ts < hi_ts[pos] + hi_dur
    valid_mask = (pos >= 0) & (lo_ts < hi_ts[np.clip(pos, 0, len(hi_ts)-1)] + hi_dur)
    pos[~valid_mask] = -1
    return pos

def shock_propagation(tfs, tr_rows, sigma=3.0):
    print("\n" + "═"*70)
    print("  CROSS-TF SHOCK PROPAGATION  (P(big_high | big_low) vs base rate)")
    print("═"*70)

    # Build big-bar flags and lookup
    flags = {}
    thrs  = {}
    for name, tr, big, mu, sd in tr_rows:
        flags[name] = (tr >= big)
        thrs[name]  = big

    pairs = [
        ("S5",  "S30"), ("S5",  "M1"),  ("S5",  "M5"),
        ("S30", "M1"),  ("S30", "M5"),  ("M1",  "M5"),
        ("M5",  "M30"), ("M5",  "H1"),  ("M30", "H1"), ("H1", "H4"),
    ]
    print(f"\n  {'Low TF':<6} → {'High TF':<6}  "
          f"{'Base%':>7}  {'Given_big_low%':>14}  {'Lift':>6}  {'n_big_low':>10}")
    for low, high in pairs:
        if low not in tfs or high not in tfs:
            continue
        df_lo = tfs[low];  df_hi = tfs[high]
        sec_hi = TF_SECONDS[high]
        hi_row_idx = align_tfs(df_lo, df_hi, sec_hi)   # int array, -1=invalid

        valid = hi_row_idx >= 0
        if valid.sum() < 100:
            print(f"  {low:<6} → {high:<6}  skipped (only {valid.sum()} aligned bars)")
            continue
        hi_idx_int = hi_row_idx[valid]
        lo_big  = flags[low][valid]
        hi_big  = flags[high][hi_idx_int]

        base_rate = hi_big.mean() * 100
        given_big = hi_big[lo_big].mean() * 100 if lo_big.sum() > 0 else 0
        lift = given_big / base_rate if base_rate > 0 else 0
        n_big_lo = lo_big.sum()

        bar = "█" * min(40, int(lift * 8))
        print(f"  {low:<6} → {high:<6}  "
              f"{base_rate:>6.1f}%  {given_big:>13.1f}%  {lift:>5.2f}x  {n_big_lo:>10,}  {bar}")


# ════════════════════════════════════════════════════════════════
# 4. Breathing waves — lag correlation of big-bar flags
# ════════════════════════════════════════════════════════════════

def breathing_waves(tfs, tr_rows, sigma=3.0):
    """For each low/high TF pair, compute cross-correlation of big-bar flags at lag 0..N."""
    print("\n" + "═"*70)
    print("  BREATHING WAVES — lag correlation of big-bar flags (S5→M5, M1→H1)")
    print("═"*70)
    print("  Method: align low-TF to high-TF timestamps,")
    print("          then cross-correlate big-bar bool arrays at lags 0..10 high-TF bars\n")

    flags = {name: tr >= big for name, tr, big, mu, sd in tr_rows}

    check_pairs = [("S5", "M5"), ("M1", "M5"), ("S30", "M5"), ("M5", "H1"), ("M1", "H1")]
    for low, high in check_pairs:
        if low not in tfs or high not in tfs:
            continue
        df_lo = tfs[low]; df_hi = tfs[high]
        sec_hi = TF_SECONDS[high]
        hi_row_idx = align_tfs(df_lo, df_hi, sec_hi)   # int array, -1=invalid
        valid = hi_row_idx >= 0
        hi_idx_int = hi_row_idx[valid]

        # Aggregate: for each high-TF bar, was ANY low-TF bar inside it a big bar?
        hi_has_big = np.zeros(len(df_hi), dtype=bool)
        lo_big = flags[low][valid]
        np.bitwise_or.at(hi_has_big, hi_idx_int, lo_big)

        hi_big = flags[high]
        n = min(len(hi_has_big), len(hi_big))
        x = hi_has_big[:n].astype(float)
        y = hi_big[:n].astype(float)

        print(f"  {low}→{high}  (high-TF big-bar rate: {100*y.mean():.1f}%  "
              f"low-TF has-big rate: {100*x.mean():.1f}%)")
        lags = list(range(-3, 11))
        for lag in lags:
            if lag < 0:
                xi, yi = x[-lag:], y[:lag]
            elif lag == 0:
                xi, yi = x, y
            else:
                xi, yi = x[:-lag], y[lag:]
            n_l = min(len(xi), len(yi))
            if n_l < 50: continue
            # conditional rate: rate of high big given low big (at this lag)
            lb = xi[:n_l].astype(bool)
            cond = yi[:n_l][lb].mean() * 100 if lb.sum() > 0 else 0.0
            base = yi[:n_l].mean() * 100
            lift = cond / base if base > 0 else 0
            bar  = "█" * min(30, int((lift - 1) * 15)) if lift > 1 else ""
            lag_str = f"lag={lag:+d}" if lag != 0 else "lag= 0"
            print(f"    {lag_str}  cond={cond:5.1f}%  base={base:5.1f}%  lift={lift:.2f}x  {bar}")
        print()


# ════════════════════════════════════════════════════════════════
# 5. TopBots Reversal Density Map
# ════════════════════════════════════════════════════════════════

def big_bar_price_density(tfs, pip_bin=5, top_n_bins=40):
    """Count big bars per price bin across M5 history — where do shocks happen?"""
    print("\n" + "═"*70)
    print(f"  BIG-BAR PRICE DENSITY — where do M5 big bars cluster? (bin={pip_bin}p)")
    print("═"*70)
    if "M5" not in tfs:
        return
    df = tfs["M5"]
    tr = true_range_pips(df)
    big_thr = tr.mean() + 3 * tr.std()
    big_mask = tr >= big_thr
    big_df = df[big_mask].reset_index(drop=True)
    print(f"  M5 big bars: {big_mask.sum()}  (thr={big_thr:.2f}p)")

    grid_unit = pip_bin * PIP
    cl = df["close"].values.astype(float)
    price_min_g = cl.min() - grid_unit
    bin_count = np.zeros(int((cl.max() - cl.min()) / grid_unit) + 5, dtype=int)

    # Count big bars in each price bin (use mid of bar for positioning)
    mid = ((df["high"].values + df["low"].values) / 2).astype(float)
    big_mid = mid[big_mask]
    for p in big_mid:
        b = int((p - price_min_g) / grid_unit)
        if 0 <= b < len(bin_count):
            bin_count[b] += 1

    # Print centered around current price
    curr_price = df["close"].iloc[-1]
    curr_bin = int((curr_price - price_min_g) / grid_unit)
    half = top_n_bins // 2
    lo_b = max(0, curr_bin - half)
    hi_b = min(len(bin_count) - 1, curr_bin + half)
    bmax = bin_count[lo_b:hi_b+1].max() or 1

    print(f"  {'Price':>8}  Big-bar count")
    for b in range(hi_b, lo_b - 1, -1):
        p = price_min_g + (b + 0.5) * grid_unit
        cnt = bin_count[b]
        bar = "█" * int(cnt / bmax * 25)
        marker = " ◄ NOW" if b == curr_bin else ""
        print(f"  {p:.4f}   {cnt:>3} {bar}{marker}")


def reversal_density_map(tfs, pip_bin=5, half_life_days=30, top_n_bins=40):
    """
    TopBots-style time-decay weighted reversal count per price bin.
    Weight of each reversal = exp(-ln2 * age_days / half_life_days)  ← half-life decay
    Also offers: 1/age_days version for comparison.

    Visualised as horizontal price-profile ASCII chart.
    """
    print("\n" + "═"*70)
    print(f"  TOPBOTS REVERSAL DENSITY MAP  (bin={pip_bin}p, half_life={half_life_days}d)")
    print("═"*70)

    # Use full H1 history for best coverage
    if "H1" not in tfs:
        print("  H1 not available — skipping")
        return
    df = tfs["H1"].copy()
    now_ts = df["timestamp"].max()

    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    ts = df["timestamp"].values  # ns int or Timestamp

    # Run TopBots to get confirmed swing points
    swings = topsbots_swings(hi, lo)   # list of (bar_idx, 'H'|'L', price)
    print(f"  H1 bars: {len(df):,}   swing points: {len(swings):,}")

    # Price grid: 5-pip bins
    grid_pips = pip_bin
    grid_unit = grid_pips * PIP

    all_prices = [v for _, _, v in swings]
    price_min = min(all_prices) - grid_unit
    price_max = max(all_prices) + grid_unit
    bins_lo = np.arange(price_min, price_max, grid_unit)
    bin_idx_of = lambda p: int((p - price_min) / grid_unit)

    n_bins = len(bins_lo) + 1
    weight_exp  = np.zeros(n_bins)  # exp half-life decay
    weight_inv  = np.zeros(n_bins)  # 1/(age_days + 1)
    count_raw   = np.zeros(n_bins, dtype=int)

    for bar_idx, swing_type, price in swings:
        b_idx = min(max(bin_idx_of(price), 0), n_bins - 1)
        # Age in days from this bar to now
        bar_ts = df["timestamp"].iloc[bar_idx]
        age_days = (now_ts - bar_ts).total_seconds() / 86400
        age_days = max(age_days, 1/1440)  # at least 1 minute
        w_exp = np.exp(-np.log(2) * age_days / half_life_days)
        w_inv = 1.0 / (age_days + 1)
        weight_exp[b_idx]  += w_exp
        weight_inv[b_idx]  += w_inv
        count_raw[b_idx]   += 1

    # Normalise for display
    wmax_exp = weight_exp.max() if weight_exp.max() > 0 else 1
    wmax_inv = weight_inv.max() if weight_inv.max() > 0 else 1

    # Current price
    curr_price = df["close"].iloc[-1]
    curr_bin   = bin_idx_of(curr_price)

    # Print: only the range with non-zero counts, reversed (high price at top)
    nonzero = np.where(count_raw > 0)[0]
    if len(nonzero) == 0:
        print("  No swings found.")
        return

    lo_b = max(0, nonzero.min() - 2)
    hi_b = min(n_bins - 1, nonzero.max() + 2)

    # Center display around current price ± top_n_bins/2
    half = top_n_bins // 2
    lo_b = max(0, curr_bin - half)
    hi_b = min(n_bins - 1, curr_bin + half)

    bar_width = 30

    print(f"\n  Price (EUR/USD)  │ Exp-decay density (HL={half_life_days}d) │ Raw count")
    print(f"  ─────────────────┼─────────────────────────────────────────┤──────────")

    for b in range(hi_b, lo_b - 1, -1):
        price_lo = price_min + b * grid_unit
        price_hi = price_lo + grid_unit
        price_mid = (price_lo + price_hi) / 2
        w = weight_exp[b] / wmax_exp
        bar_len = int(w * bar_width)
        bar_str = "█" * bar_len + "░" * (bar_width - bar_len)
        cnt = count_raw[b]
        marker = " ◄ NOW" if b == curr_bin else "      "
        print(f"  {price_mid:.4f}  │ {bar_str} │  {cnt:>4}{marker}")

    # Print top 10 densest bins
    top_idx = np.argsort(weight_exp)[::-1][:10]
    print(f"\n  Top 10 densest price levels (exp-decay, half_life={half_life_days}d):")
    print(f"  {'Level':>8}  {'Weight':>8}  {'Count':>6}  │ Bar")
    for b in top_idx:
        if count_raw[b] == 0:
            continue
        p = price_min + (b + 0.5) * grid_unit
        w = weight_exp[b]
        bar_len = int(w / wmax_exp * 20)
        print(f"  {p:.4f}   {w:>8.4f}   {count_raw[b]:>6}  │ {'█'*bar_len}")

    # Summary: is current price in a sparse or dense zone?
    curr_w = weight_exp[curr_bin] / wmax_exp
    nbr_w  = np.mean([weight_exp[max(0,curr_bin-2):curr_bin+3]]) / wmax_exp
    zone = "🟢 CLEAR AIR (sparse S/R)" if nbr_w < 0.25 else \
           "🟡 MODERATE density"       if nbr_w < 0.55 else \
           "🔴 HIGH density (strong S/R zone)"
    print(f"\n  Current price: {curr_price:.4f}  bin density: {curr_w:.3f}  "
          f"neighbourhood (±2 bins): {nbr_w:.3f}")
    print(f"  Zone: {zone}")


# ════════════════════════════════════════════════════════════════
# 6. Body-ratio × TR cross-tab (extended to lower TFs)
# ════════════════════════════════════════════════════════════════

def print_body_crosstab_summary(tr_rows):
    """Compact cross-tab: quintile Q1 vs Q5, body extremes (<-0.6, >+0.6) vs centre."""
    print("\n" + "═"*70)
    print("  BODY RATIO × TR QUINTILE — compact summary (all TFs)")
    print("═"*70)
    print(f"  {'TF':<6}  {'Q1 cent%':>8}  {'Q1 extr%':>8}  {'Q5 cent%':>8}  {'Q5 extr%':>8}  "
          f"{'ratio_Q5/Q1_extr':>16}")
    for name, tr, big, mu, sd in tr_rows:
        br = body_ratio
    # Need the actual dfs — re-compute from tr_rows
    # tr_rows is list of (name, tr, big, mu, sd) but no df — we need body_ratio from df
    print("  (see full M5/H1 cross-tab in eurusd_bar_stats.py output)")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-days", type=int, default=90,
                    help="Days of history to use for cross-TF alignment (default 90)")
    ap.add_argument("--pip-bin", type=int, default=5,
                    help="Price bin size for reversal density (pips, default 5)")
    ap.add_argument("--half-life", type=int, default=30,
                    help="Reversal density half-life in days (default 30)")
    args = ap.parse_args()

    print(f"\n{'═'*70}")
    print(f"  EUR/USD Shock Propagation + Reversal Density")
    print(f"  ref_days={args.ref_days}  pip_bin={args.pip_bin}p  half_life={args.half_life}d")
    print(f"{'═'*70}\n")

    print("Loading timeframes …")
    tfs = load_all_tfs(args.ref_days)

    tr_rows = []
    for name in ORDER:
        if name not in tfs:
            continue
        df = tfs[name]
        tr = true_range_pips(df)
        mu, sd = tr.mean(), tr.std()
        big_thr = mu + 3 * sd
        tr_rows.append((name, tr, big_thr, mu, sd))

    print_tr_summary(tfs)
    shock_propagation(tfs, tr_rows)
    breathing_waves(tfs, tr_rows)
    big_bar_price_density(tfs, pip_bin=args.pip_bin)
    reversal_density_map(tfs, pip_bin=args.pip_bin, half_life_days=args.half_life)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
