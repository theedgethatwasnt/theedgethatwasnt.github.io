"""EUR/USD candle statistics: TR distribution, body ratio, cross-tab across timeframes."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PIP = 0.0001  # EUR/USD

# ── helpers ─────────────────────────────────────────────────────────────────

def true_range_pips(df):
    """Return TR series in pips (handles gaps via prev_close)."""
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    pc = np.roll(cl, 1); pc[0] = cl[0]  # first bar: no gap
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    return tr / PIP

def body_ratio(df):
    """Signed body ratio: (close-open)/(high-low). ∈ [-1, 1]. NaN on flat bar."""
    rng = df["high"].values - df["low"].values
    body = df["close"].values - df["open"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(rng > 0, body / rng, np.nan)
    return ratio

def resample_ohlc(df, rule):
    """Resample M5 dataframe to coarser OHLC."""
    df2 = df.set_index("timestamp") if "timestamp" in df.columns else df
    r = df2.resample(rule)
    out = pd.DataFrame({
        "open":  r["open"].first(),
        "high":  r["high"].max(),
        "low":   r["low"].min(),
        "close": r["close"].last(),
    }).dropna()
    return out.reset_index()

def print_tr_distribution(name, tr, n_bins=20):
    cap = np.percentile(tr, 99)   # cap at 99th pct to avoid huge tail bins
    bins = np.linspace(0, cap, n_bins + 1)
    counts, edges = np.histogram(tr, bins=bins)
    mu, sd = tr.mean(), tr.std()
    big_bar = mu + 3 * sd
    print(f"\n{'='*56}")
    print(f"  {name}   n={len(tr):,}  μ={mu:.2f}p  σ={sd:.2f}p  big-bar≥{big_bar:.2f}p  P99={cap:.2f}p")
    print(f"{'='*56}")
    print(f"  {'Range (pips)':<20} {'Count':>8}  {'%':>6}  bar")
    total = len(tr)
    for i, (cnt, lo_, hi_) in enumerate(zip(counts, edges[:-1], edges[1:])):
        pct = 100 * cnt / total
        bar = "█" * int(pct / 1.5)
        print(f"  {lo_:5.1f}–{hi_:5.1f}          {cnt:>8,}  {pct:5.1f}%  {bar}")
    n_big = (tr >= big_bar).sum()
    print(f"  big bars (≥{big_bar:.2f}p):  {n_big:,}  ({100*n_big/total:.1f}%)")
    return big_bar, mu, sd

def print_body_distribution(name, br):
    br = br[~np.isnan(br)]
    bin_edges = np.linspace(-1, 1, 11)   # 10 equal bins of 0.2
    labels = [f"{lo_:.1f}→{hi_:.1f}" for lo_, hi_ in zip(bin_edges[:-1], bin_edges[1:])]
    counts, _ = np.histogram(br, bins=bin_edges)
    total = len(br)
    print(f"\n  Body ratio (close-open)/(high-low)  n={total:,}")
    print(f"  {'Bin':<14} {'Count':>8}  {'%':>6}  bar")
    for lbl, cnt in zip(labels, counts):
        pct = 100 * cnt / total
        bar = "█" * int(pct / 1.5)
        print(f"  {lbl:<14} {cnt:>8,}  {pct:5.1f}%  {bar}")
    print(f"  mean body ratio: {br.mean():.4f}")

def print_crosstab(name, tr, br):
    """Cross-tab: 5 TR quintile groups × 10 body-ratio bins."""
    mask = ~np.isnan(br)
    tr_m, br_m = tr[mask], br[mask]
    q_labels = ["Q1(small)", "Q2", "Q3", "Q4", "Q5(big)"]
    q_edges = np.percentile(tr_m, [0, 20, 40, 60, 80, 100])
    body_edges = np.linspace(-1, 1, 11)
    body_labels = [f"{lo_:.1f}/{hi_:.1f}" for lo_, hi_ in zip(body_edges[:-1], body_edges[1:])]

    q_idx = np.digitize(tr_m, q_edges[1:-1])  # 0..4
    b_idx = np.clip(np.digitize(br_m, body_edges[1:-1]), 0, 9)  # 0..9

    print(f"\n  Cross-tab: TR quintile × body ratio (% of row)   [{name}]")
    print(f"  {'TR group':<12}", end="")
    for lbl in body_labels:
        print(f"  {lbl:>7}", end="")
    print()
    for qi, qlbl in enumerate(q_labels):
        row_mask = (q_idx == qi)
        row_n = row_mask.sum()
        if row_n == 0:
            continue
        lo_tr = q_edges[qi]; hi_tr = q_edges[qi + 1]
        print(f"  {qlbl:<12}", end="")
        for bi in range(10):
            cell = ((row_mask) & (b_idx == bi)).sum()
            pct = 100 * cell / row_n
            print(f"  {pct:6.1f}%", end="")
        print(f"  n={row_n:,}  ({lo_tr:.1f}–{hi_tr:.1f}p)")


# ── load data ────────────────────────────────────────────────────────────────

print("Loading M5 BA parquet …")
m5_path = "${HOME}/projects/fx-core/data/m5_ba/EUR_USD_M5_BA.parquet"
df_m5 = pd.read_parquet(m5_path)
# Normalise column names
df_m5.columns = [c.lower() for c in df_m5.columns]
if "timestamp" not in df_m5.columns:
    # index might be timestamp
    df_m5 = df_m5.reset_index()
    df_m5.rename(columns={"index": "timestamp"}, inplace=True)

# Ensure timestamp is datetime
df_m5["timestamp"] = pd.to_datetime(df_m5["timestamp"], utc=True)
df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)

print(f"  M5: {len(df_m5):,} bars  {df_m5['timestamp'].iloc[0].date()} → {df_m5['timestamp'].iloc[-1].date()}")

# Aggregate to M30 and H1
print("Aggregating to M30 and H1 …")
df_m5_ts = df_m5.set_index("timestamp")
df_m30 = df_m5_ts.resample("30min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last"
}).dropna().reset_index()
df_h1 = df_m5_ts.resample("1h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last"
}).dropna().reset_index()
print(f"  M30: {len(df_m30):,} bars   H1: {len(df_h1):,} bars")

# Try dedicated H1 parquet for comparison
h1_path = "${HOME}/projects/csi_factor_study/data/EUR_USD_H1.parquet"
try:
    df_h1b = pd.read_parquet(h1_path)
    df_h1b.columns = [c.lower() for c in df_h1b.columns]
    if "timestamp" not in df_h1b.columns:
        df_h1b = df_h1b.reset_index()
        df_h1b.rename(columns={"index": "timestamp", "time": "timestamp"}, inplace=True)
    df_h1b["timestamp"] = pd.to_datetime(df_h1b["timestamp"], utc=True)
    df_h1b = df_h1b.sort_values("timestamp").reset_index(drop=True)
    print(f"  H1 (dedicated parquet): {len(df_h1b):,} bars")
    use_h1b = True
except Exception as e:
    print(f"  H1 parquet not loaded ({e}) — using aggregated H1")
    use_h1b = False

note = "\n  NOTE: S5, S30, M1 not available locally. Using M5 (native), M30, H1 aggregated from M5."
print(note)

# ── analysis per TF ─────────────────────────────────────────────────────────

frames = [
    ("M5",  df_m5),
    ("M30", df_m30),
    ("H1",  df_h1),
]
if use_h1b:
    frames.append(("H1-direct", df_h1b))

summary_rows = []
for name, df in frames:
    tr = true_range_pips(df)
    br = body_ratio(df)
    big, mu, sd = print_tr_distribution(name, tr)
    print_body_distribution(name, br)
    print_crosstab(name, tr, br)
    summary_rows.append({"TF": name, "n": len(df), "mu_TR": round(mu,2), "sd_TR": round(sd,2), "big_bar_thresh": round(big,2)})

# ── summary table ────────────────────────────────────────────────────────────
print("\n\n" + "="*56)
print("SUMMARY — big-bar thresholds (mean + 3σ)")
print("="*56)
print(f"  {'TF':<12} {'bars':>8}  {'μ TR':>6}  {'σ TR':>6}  {'Big≥':>8}")
for r in summary_rows:
    print(f"  {r['TF']:<12} {r['n']:>8,}  {r['mu_TR']:>6.2f}p  {r['sd_TR']:>6.2f}p  {r['big_bar_thresh']:>8.2f}p")
print()
