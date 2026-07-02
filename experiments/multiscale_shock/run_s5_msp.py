"""
Multiscale Shock Propagation — S5 Resolution Analysis
======================================================
Uses OANDA S5 BA parquets (real live spreads) to study cross-scale
energy propagation in FX price series.

DWT scale ladder at S5 (5-second bars):
  D1 ≈ 10–20s   (2–4 bars)    ← was tick D3
  D2 ≈ 20–40s   (4–8 bars)    ← was tick D4
  D3 ≈ 40–80s   (8–16 bars)   ← was tick D5  [PRIMARY FOCUS]
  D4 ≈ 80–160s  (16–32 bars)  ← was tick D6  [PRIMARY FOCUS]
  D5 ≈ 160–320s (32–64 bars)
  D6 ≈ 320–640s (64–128 bars)
  D7 ≈ 640–1280s (≈10–21 min)
  D8 ≈ 1280–2560s (≈21–43 min)

Spread barrier: use real per-bar spread from ask_c - bid_c.
Round-trip cost = 2 × median spread. Need E[move | shock] >> 3× round-trip.

Outputs (results/ directory):
  s5_msp_{pair}_dwt_bands.png     — per-band energy + shock timeline
  s5_msp_{pair}_xcorr.png         — cross-scale xcorr heatmap D3–D6
  s5_msp_{pair}_lift_fwd.png      — D3/D4 shock → forward move distribution
  s5_msp_{pair}_summary.csv       — key statistics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from numba import njit, prange

# ── Paths ──────────────────────────────────────────────────────────────────
# Check both s5_ohlc (older files) and s5_ba (fetch_s5_ba.py output)
DATA_DIRS = [
    Path(__file__).parents[3] / "data" / "s5_ohlc",
    Path(__file__).parents[3] / "data" / "s5_ba",
]
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Pairs — auto-discovered from available files when script runs
PAIRS_DEFAULT = {
    "EUR_USD": {"pip": 0.0001},
    "EUR_JPY": {"pip": 0.01},
    "GBP_JPY": {"pip": 0.01},
    "USD_JPY": {"pip": 0.01},
    "AUD_JPY": {"pip": 0.01},
}
PIP_MAP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
    "GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01,
    "AUD_JPY": 0.01, "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
}

WAVELET = "db4"
DWT_LEVEL = 8       # 8 levels → D1..D8 + A8
SHOCK_Z_THRESH = 2.5
MAD_WIN = 1024      # rolling window for MAD z-score (in S5 bars, ~85 min)

# Focus bands for cross-scale analysis
FOCUS_BANDS = ["D3", "D4", "D5", "D6"]

# Forward-look lags for move-size study (in S5 bars)
FWD_LAGS = [11, 22, 44, 88, 176]   # ≈ 55s, 110s, 220s, 440s, 880s


# ── Statistics helpers ─────────────────────────────────────────────────────

def mad_zscore_fast(x: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling MAD z-score using pandas (O(n log w) via C-level heap).
    Rolling window is centered; min_periods=w//4 to handle series start.
    """
    s = pd.Series(x.astype(np.float64))
    roll_med = s.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    abs_dev  = (s - roll_med).abs()
    roll_mad = abs_dev.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    z = (s - roll_med) / (1.4826 * roll_mad.clip(lower=1e-12))
    return z.fillna(0.0).values


@njit(cache=True, parallel=True)
def forward_moves_nb(close: np.ndarray, pip: float,
                     shock_mask: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """
    For each shock bar, record pip move at each lag.
    Returns (n_shocks, n_lags) array.
    shock_mask: boolean array, same length as close
    lags: int array of forward-look distances in bars
    """
    shock_idx = np.where(shock_mask)[0]
    n_s = len(shock_idx)
    n_l = len(lags)
    out = np.full((n_s, n_l), np.nan, dtype=np.float64)
    n = len(close)
    for k in prange(n_s):
        i = shock_idx[k]
        for j in range(n_l):
            tgt = i + lags[j]
            if tgt < n:
                out[k, j] = (close[tgt] - close[i]) / pip
    return out


# ── DWT helpers ────────────────────────────────────────────────────────────

def compute_dwt_bands(signal: np.ndarray, wavelet: str = WAVELET,
                      level: int = DWT_LEVEL) -> dict[str, np.ndarray]:
    """
    Decompose signal. Returns dict band_name → array aligned to signal length.
    Uses 'periodization' mode so each band has exactly len(signal) / 2^j length;
    we upsample back to n with np.repeat for temporal alignment.
    """
    n = len(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode="periodization")
    # coeffs[0] = A_level, coeffs[1] = D_level, ..., coeffs[level] = D1
    bands: dict[str, np.ndarray] = {}
    # cA
    ca = coeffs[0]
    factor = (n + len(ca) - 1) // len(ca)   # ceiling division → ensures length ≥ n
    bands[f"A{level}"] = np.repeat(ca, factor)[:n]
    # cD bands: coeffs[k] = D_{level+1-k}
    for k in range(1, level + 1):
        band_name = f"D{level + 1 - k}"
        cd = coeffs[k]
        factor = (n + len(cd) - 1) // len(cd)
        arr = np.repeat(cd, factor)[:n]
        bands[band_name] = arr
    return bands


def band_energy(band: np.ndarray) -> np.ndarray:
    """Log-energy: log(coef² + ε). More Gaussian than raw coef²."""
    return np.log(band ** 2 + 1e-12)


def band_shock_mask(log_energy: np.ndarray, win: int = MAD_WIN,
                    thresh: float = SHOCK_Z_THRESH) -> np.ndarray:
    """Boolean mask: log-energy MAD z-score > thresh (~1-2% rate for 2.5σ)."""
    z = mad_zscore_fast(log_energy, win)
    return z > thresh


# ── Cross-correlation ───────────────────────────────────────────────────────

def xcorr_lags(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean normalized cross-correlation for lag 0..max_lag."""
    # Enforce same length (DWT upsampling may differ by 1)
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.zeros(max_lag + 1), np.arange(max_lag + 1)
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[0] = np.dot(a, b) / (na * nb)
        else:
            out[lag] = np.dot(a[:-lag], b[lag:]) / (na * nb)
    return out, np.arange(max_lag + 1)


# ── Lift ───────────────────────────────────────────────────────────────────

def lift_table(src: np.ndarray, tgt: np.ndarray, lags: list[int]) -> pd.DataFrame:
    """
    P(tgt(t+lag)=1 | src(t)=1) / P(tgt=1)  for each lag.
    src, tgt: boolean arrays.
    """
    p_tgt = tgt.mean()
    rows = []
    for lag in lags:
        if lag == 0:
            cond = tgt[src]
        else:
            mask = src[:-lag]
            cond = tgt[lag:][mask]
        p_cond = cond.mean() if len(cond) > 0 else np.nan
        rows.append({"lag_bars": lag, "lag_sec": lag * 5,
                     "p_tgt": p_tgt, "p_cond": p_cond,
                     "lift": p_cond / p_tgt if p_tgt > 0 else np.nan,
                     "n_events": mask.sum() if lag > 0 else src.sum()})
    return pd.DataFrame(rows)


# ── Per-pair analysis ───────────────────────────────────────────────────────

def find_parquet(pair: str) -> Path | None:
    """Find S5 BA parquet in any known data directory."""
    for d in DATA_DIRS:
        p = d / f"{pair}_S5_BA.parquet"
        if p.exists():
            return p
    return None


def run_pair(pair: str, pip: float) -> dict:
    path = find_parquet(pair)
    if path is None:
        print(f"  {pair}: parquet not found in {[str(d) for d in DATA_DIRS]}, skip")
        return {}

    print(f"\n{'='*60}")
    print(f"  {pair}  pip={pip}")

    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Normalise schema: both "close" (mid pre-computed) and bid_c/ask_c only
    if "close" in df.columns:
        close = df["close"].values.astype(np.float64)
    elif "bid_c" in df.columns and "ask_c" in df.columns:
        close = ((df["bid_c"] + df["ask_c"]) / 2).values.astype(np.float64)
    else:
        print(f"  {pair}: no close or bid_c/ask_c columns, skip")
        return {}
    spread_pips = ((df["ask_c"] - df["bid_c"]) / pip).values
    n = len(close)

    spread_median = np.median(spread_pips)
    spread_p90    = np.percentile(spread_pips, 90)
    rt_cost       = 2 * spread_median         # round-trip
    barrier       = 3 * rt_cost               # 3× round-trip = min viable edge
    print(f"  Bars: {n:,}  Spread median: {spread_median:.2f}p  P90: {spread_p90:.2f}p")
    print(f"  Round-trip cost: {rt_cost:.2f}p  Barrier (3×RT): {barrier:.2f}p")

    # ── DWT ────────────────────────────────────────────────────────────────
    print(f"  Computing DWT ({WAVELET}, level={DWT_LEVEL})...")
    bands = compute_dwt_bands(close, WAVELET, DWT_LEVEL)

    # Energy + shock per band
    # energies[b] = raw coef² (for xcorr / plots)
    # log_energies[b] = log(coef² + ε) (for MAD z-score shock detection)
    energies:     dict[str, np.ndarray] = {}
    log_energies: dict[str, np.ndarray] = {}
    shocks:       dict[str, np.ndarray] = {}
    for bname, bdata in bands.items():
        raw_e = bdata ** 2
        log_e = band_energy(bdata)   # returns log(coef² + ε)
        energies[bname]     = raw_e
        log_energies[bname] = log_e
        shocks[bname]       = band_shock_mask(log_e)

    # Print shock rates for focus bands
    for b in FOCUS_BANDS:
        if b in shocks:
            rate = shocks[b].mean()
            print(f"    {b} shock rate: {rate*100:.2f}%  n={shocks[b].sum():,}")

    # ── Forward move study: D3 and D4 shocks ──────────────────────────────
    print(f"  Computing forward moves...")
    lags_arr = np.array(FWD_LAGS, dtype=np.int64)
    results_fwd = {}
    for src_band in ["D3", "D4"]:
        if src_band not in shocks:
            continue
        mask = shocks[src_band]
        if mask.sum() < 20:
            print(f"    {src_band}: too few shocks ({mask.sum()}), skip")
            continue
        moves = forward_moves_nb(close, pip, mask.astype(np.bool_), lags_arr)
        results_fwd[src_band] = moves

        # Direction: DWT coefficient sign at shock bar → continuation or reversal?
        band_coefs = bands[src_band]
        shock_idx  = np.where(mask)[0]
        coef_signs = np.sign(band_coefs[shock_idx])   # +1=upward, -1=downward shock

        print(f"    {src_band} ({mask.sum():,} shocks) → forward pips:")
        for j, lag in enumerate(FWD_LAGS):
            col   = moves[:, j]
            valid_mask = ~np.isnan(col)
            valid = col[valid_mask]
            if len(valid) == 0:
                continue
            signs_v   = coef_signs[valid_mask]
            # Signed move aligned to shock direction (positive = continuation)
            aligned   = valid * signs_v
            abs_valid = np.abs(valid)
            cont_rate = (aligned > 0).mean()
            abs_p90   = np.percentile(abs_valid, 90)
            print(f"      lag={lag}bars({lag*5}s): "
                  f"mean_abs={abs_valid.mean():.2f}p  P50={np.median(abs_valid):.2f}p  "
                  f"P90={abs_p90:.2f}p  "
                  f"continuation={cont_rate*100:.1f}%  "
                  f"{'✅ CLEARS' if abs_p90 > barrier else '❌ fails'}")

    # ── Cross-correlation: D3→D5, D4→D6, D3→D4 ───────────────────────────
    print(f"  Computing cross-scale xcorr...")
    xcorr_results = {}
    pairs_to_xcorr = [("D3","D5"), ("D4","D6"), ("D3","D4"), ("D1","D4")]
    max_lag = 88  # ≈ 440 seconds
    for (ba, bb) in pairs_to_xcorr:
        if ba not in energies or bb not in energies:
            continue
        corr, lags_xc = xcorr_lags(energies[ba], energies[bb], max_lag)
        peak_idx = np.argmax(np.abs(corr))
        peak_val  = corr[peak_idx]
        peak_lag  = lags_xc[peak_idx]
        xcorr_results[(ba, bb)] = (corr, lags_xc)
        print(f"    xcorr {ba}→{bb}: peak={peak_val:+.3f} @ lag={peak_lag}bars({peak_lag*5}s)")

    # ── Lift table: D3 shock → D5 shock ───────────────────────────────────
    lift_lags_bars = [0, 2, 4, 8, 11, 22, 44]
    lift_results = {}
    for (src, tgt) in [("D3","D5"), ("D4","D6")]:
        if src not in shocks or tgt not in shocks:
            continue
        lt = lift_table(shocks[src], shocks[tgt], lift_lags_bars)
        lift_results[(src, tgt)] = lt
        peak_lift = lt["lift"].max()
        peak_lag_s = lt.loc[lt["lift"].idxmax(), "lag_sec"]
        print(f"    lift {src}→{tgt}: peak lift={peak_lift:.2f}x @ {peak_lag_s}s")

    # ── Plots ───────────────────────────────────────────────────────────────
    _plot_bands(pair, df, bands, shocks, close, pip)
    _plot_xcorr(pair, xcorr_results)
    _plot_fwd_moves(pair, results_fwd, FWD_LAGS, barrier, spread_median)

    # ── Summary CSV ────────────────────────────────────────────────────────
    summary_rows = []
    for src_band, moves in results_fwd.items():
        for j, lag in enumerate(FWD_LAGS):
            col = moves[:, j]
            valid = np.abs(col[~np.isnan(col)])
            if len(valid) == 0:
                continue
            summary_rows.append({
                "pair": pair, "src_band": src_band,
                "lag_bars": lag, "lag_sec": lag * 5,
                "n_shocks": len(valid),
                "mean_abs_pips": valid.mean(),
                "p50_abs_pips":  np.median(valid),
                "p90_abs_pips":  np.percentile(valid, 90),
                "barrier_pips":  barrier,
                "clears_barrier": np.percentile(valid, 90) > barrier,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / f"s5_msp_{pair}_summary.csv", index=False)
    print(f"  Summary saved → results/s5_msp_{pair}_summary.csv")

    return {"spread_median": spread_median, "barrier": barrier,
            "xcorr": xcorr_results, "lift": lift_results,
            "fwd": results_fwd}


# ── Plot helpers ────────────────────────────────────────────────────────────

def _plot_bands(pair, df, bands, shocks, close, pip):
    """Energy + price + shock markers for focus bands."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{pair} — S5 DWT Band Energy + Shocks", fontsize=14)
    n_focus = len(FOCUS_BANDS)
    gs = gridspec.GridSpec(n_focus + 1, 1, hspace=0.6)

    # Price panel
    ax0 = fig.add_subplot(gs[0])
    ts = pd.to_datetime(df["timestamp"].values)
    sample = max(1, len(ts) // 10000)   # downsample for speed
    ax0.plot(ts[::sample], close[::sample], lw=0.5, color="steelblue")
    ax0.set_ylabel("Close (mid)")
    ax0.set_title(f"{pair} Mid Price (S5)")
    ax0.set_xlim(ts[0], ts[-1])

    colors = {"D3": "#ff7043", "D4": "#ffa726", "D5": "#66bb6a", "D6": "#42a5f5"}
    for i, bname in enumerate(FOCUS_BANDS):
        ax = fig.add_subplot(gs[i + 1], sharex=ax0)
        e = bands[bname] ** 2
        ax.plot(ts[::sample], e[::sample], lw=0.4,
                color=colors.get(bname, "gray"), alpha=0.7)
        # Mark shocks
        shock_idx = np.where(shocks[bname])[0]
        if len(shock_idx) > 0:
            ax.scatter(ts[shock_idx[::max(1, len(shock_idx)//1000)]],
                       e[shock_idx[::max(1, len(shock_idx)//1000)]],
                       s=4, color="red", alpha=0.5, zorder=5)
        ax.set_ylabel(f"E({bname})")
        ax.set_title(f"{bname} energy  shock_rate={shocks[bname].mean()*100:.1f}%")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"s5_msp_{pair}_dwt_bands.png", dpi=120)
    plt.close()


def _plot_xcorr(pair, xcorr_results):
    if not xcorr_results:
        return
    n = len(xcorr_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{pair} — Cross-Scale Energy xcorr", fontsize=13)
    for ax, ((ba, bb), (corr, lags_xc)) in zip(axes, xcorr_results.items()):
        ax.plot(lags_xc * 5, corr, color="royalblue")
        peak_i = np.argmax(np.abs(corr))
        ax.axvline(lags_xc[peak_i] * 5, color="red", ls="--", lw=1,
                   label=f"peak={corr[peak_i]:+.3f}@{lags_xc[peak_i]*5}s")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Lag (seconds)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"E({ba}) → E({bb})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"s5_msp_{pair}_xcorr.png", dpi=120)
    plt.close()


def _plot_fwd_moves(pair, results_fwd, fwd_lags, barrier, spread_median):
    if not results_fwd:
        return
    n_src = len(results_fwd)
    n_lag = len(fwd_lags)
    fig, axes = plt.subplots(n_src, n_lag, figsize=(4 * n_lag, 4 * n_src),
                             squeeze=False)
    fig.suptitle(f"{pair} — Forward Pip Move Distribution | barrier={barrier:.1f}p  spread={spread_median:.1f}p",
                 fontsize=12)
    for r, (src_band, moves) in enumerate(results_fwd.items()):
        for c, lag in enumerate(fwd_lags):
            ax = axes[r][c]
            col = np.abs(moves[:, c])
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                ax.set_visible(False)
                continue
            ax.hist(np.clip(valid, 0, barrier * 5), bins=50, color="steelblue",
                    alpha=0.8, edgecolor="none")
            p90 = np.percentile(valid, 90)
            ax.axvline(p90, color="red", ls="--", lw=1.5,
                       label=f"P90={p90:.1f}p")
            ax.axvline(barrier, color="orange", ls="-", lw=1.5,
                       label=f"Barrier={barrier:.1f}p")
            ax.set_title(f"{src_band} shock → +{lag*5}s")
            ax.set_xlabel("|pips|")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"s5_msp_{pair}_lift_fwd.png", dpi=120)
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("S5 Multiscale Shock Propagation Experiment")
    print(f"Data dirs: {[str(d) for d in DATA_DIRS]}")
    print(f"Output: {OUT_DIR}")

    # Auto-discover available parquets
    available = {}
    for d in DATA_DIRS:
        for f in sorted(d.glob("*_S5_BA.parquet")):
            pair = f.stem.replace("_S5_BA", "")
            if pair in PIP_MAP and pair not in available:
                available[pair] = PIP_MAP[pair]
    if not available:
        print("No S5 BA parquets found. Run fetch_s5_ba.py first.")
        sys.exit(1)

    print(f"Found {len(available)} pairs: {list(available.keys())}\n")
    all_summaries = []
    for pair, pip_val in available.items():
        r = run_pair(pair, pip_val)
        if r:
            all_summaries.append({"pair": pair, **{k: v for k, v in r.items()
                                                   if not isinstance(v, (dict, np.ndarray))}})

    # Combined summary across pairs
    csvs = list(OUT_DIR.glob("s5_msp_*_summary.csv"))
    if csvs:
        combined = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        combined.to_csv(OUT_DIR / "s5_msp_combined_summary.csv", index=False)
        print("\n\n=== COMBINED RESULTS ===")
        print(combined[combined["clears_barrier"]].to_string(index=False))
        print(f"\nSaved: results/s5_msp_combined_summary.csv")

    print("\nDone.")
