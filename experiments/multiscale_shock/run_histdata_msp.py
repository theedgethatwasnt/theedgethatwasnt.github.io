"""
MSP Experiment — HistData tick data: AUD/JPY April 2026

Phases implemented:
  1. Load & parse tick data
  2. Resample to time grids → OHLC + TR-rate + shock_z
  3. Tick-count rolling windows (Numba)
  4. DWT wavelet decomposition (PyWavelets db4, 8 levels)
  5. Descriptive: shock rates, lift tables, cross-window xcorr
  6. Wavelet: cross-scale energy xcorr, DWT lift tables
  7. Write results to CSV + plots

Run from fx-core root:
    python3 research/experiments/multiscale_shock/run_histdata_msp.py
"""

from __future__ import annotations
import gc, logging, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("msp")

# ── Config ──────────────────────────────────────────────────────────────────
ZIP_PATH    = Path("data/HISTDATA_COM_ASCII_AUDJPY_T202604.zip")
OUT_DIR     = Path("research/experiments/multiscale_shock/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIP         = 0.01          # AUD/JPY pip size
PAIR        = "AUD_JPY"

TICK_WINDOWS = np.array([16, 64, 256, 1024], dtype=np.int64)
TICK_LABELS  = [f"t{w}" for w in TICK_WINDOWS]

# Time-grid resolutions for OHLC bars
TIME_GRAINS  = ["250ms", "1s", "5s", "30s", "1min", "5min", "15min"]
# Duration in minutes for each grain
GRAIN_MINS   = [250/60_000, 1/60, 5/60, 30/60, 1.0, 5.0, 15.0]

DWT_WAVELET  = "db4"
DWT_LEVEL    = 8            # D1=~2s, D2=~4s … D8=~512s≈8min, A8=trend
DWT_GRAIN    = "1s"         # resample to 1s for DWT input

SHOCK_Z_THR  = 2.5
MAD_LOOK     = 300          # lookback for MAD z-score (bars in each grain)

# ── 1. Load tick data ────────────────────────────────────────────────────────

def load_histdata(zip_path: Path) -> pd.DataFrame:
    log.info("Loading tick data…")
    zf = zipfile.ZipFile(zip_path)
    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
    raw = zf.read(csv_name).decode("utf-8")

    ts_list, bid_list, ask_list = [], [], []
    for line in raw.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        ts_str, bid_s, ask_s = parts[0], parts[1], parts[2]
        ts_str = ts_str.strip()
        # Format: "YYYYMMDD HHMMSSmmm"
        date_p = ts_str[:8]
        time_9 = ts_str[9:18]          # 9 chars: HHMMSS + mmm
        hh, mm, ss, ms = time_9[:2], time_9[2:4], time_9[4:6], time_9[6:9]
        ts_list.append(f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:8]} {hh}:{mm}:{ss}.{ms}")
        bid_list.append(float(bid_s))
        ask_list.append(float(ask_s))

    df = pd.DataFrame({
        "bid": bid_list,
        "ask": ask_list,
    }, index=pd.to_datetime(ts_list))
    df.index.name = "timestamp"
    df = df.sort_index()
    df["mid"]    = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / PIP

    n_ticks = len(df)
    avg_int  = df.index.to_series().diff().dt.total_seconds().median()
    log.info(
        f"  {n_ticks:,} ticks  |  {df.index[0]} → {df.index[-1]}  |  "
        f"median interval={avg_int:.1f}s  |  avg spread={df['spread'].mean():.3f}p"
    )
    return df

# ── 2. MAD z-score (Numba) ───────────────────────────────────────────────────

@nb.njit(cache=True)
def mad_zscore_nb(x: np.ndarray, w: int) -> np.ndarray:
    n   = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg = x[i - w + 1 : i + 1]
        cnt = 0
        for v in seg:
            if not np.isnan(v):
                cnt += 1
        if cnt < w // 2:
            continue
        tmp = np.empty(cnt)
        j = 0
        for v in seg:
            if not np.isnan(v):
                tmp[j] = v; j += 1
        # sort
        tmp = np.sort(tmp)
        med = tmp[cnt // 2] if cnt % 2 == 1 else 0.5 * (tmp[cnt // 2 - 1] + tmp[cnt // 2])
        for k in range(cnt):
            tmp[k] = abs(tmp[k] - med)
        tmp = np.sort(tmp)
        mad = tmp[cnt // 2] if cnt % 2 == 1 else 0.5 * (tmp[cnt // 2 - 1] + tmp[cnt // 2])
        if mad < 1e-12:
            out[i] = 0.0
        else:
            out[i] = 0.6745 * (x[i] - med) / mad
    return out

# ── 3. Time-grid features ────────────────────────────────────────────────────

def time_grid_features(ticks: pd.DataFrame, grain: str, dur_min: float) -> pd.DataFrame:
    """
    Resample ticks to `grain` bars, compute OHLC + TR-rate + shock_z.
    Drops bars with no tick data (weekend gaps etc).
    """
    mid = ticks["mid"]
    bars = mid.resample(grain).ohlc().dropna()
    bars.columns = ["open", "high", "low", "close"]

    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        (bars["high"] - bars["low"]),
        (bars["high"] - prev_close).abs(),
        (bars["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    tr_rate = (tr / PIP) / dur_min
    tr_arr  = tr_rate.values.astype(np.float64)
    look    = min(MAD_LOOK, len(tr_arr) // 2)
    shock_z = mad_zscore_nb(tr_arr, look)

    body_rate = ((bars["close"] - bars["open"]) / PIP) / dur_min

    spread = ticks["spread"].resample(grain).mean().reindex(bars.index)
    return pd.DataFrame({
        "open": bars["open"], "high": bars["high"],
        "low": bars["low"],  "close": bars["close"],
        "tr_rate":   tr_rate,
        "shock_z":   pd.Series(shock_z, index=bars.index),
        "shock":     pd.Series((shock_z > SHOCK_Z_THR).astype(np.float32), index=bars.index),
        "body_rate": body_rate,
        "spread":    spread,
    })

# ── 4. Tick-count window features (Numba) ────────────────────────────────────

@nb.njit(parallel=True, cache=True)
def tick_window_features_nb(
    mid: np.ndarray, ts_ns: np.ndarray, windows: np.ndarray
) -> tuple:
    n   = len(mid)
    nw  = len(windows)
    tr_rate  = np.full((n, nw), np.nan)
    body_rate = np.full((n, nw), np.nan)
    for wi in nb.prange(nw):
        w = windows[wi]
        for i in range(w, n):
            s = i - w
            hi = mid[s]; lo = mid[s]
            for k in range(s + 1, i + 1):
                if mid[k] > hi: hi = mid[k]
                if mid[k] < lo: lo = mid[k]
            op = mid[s]; cl = mid[i]
            prev_cl = mid[s - 1] if s > 0 else op
            tr = max(hi - lo, abs(hi - prev_cl), abs(lo - prev_cl))
            dur = (ts_ns[i] - ts_ns[s]) / 60e9
            if dur > 1e-9:
                tr_rate[i, wi]   = tr / 0.01 / dur
                body_rate[i, wi] = (cl - op) / 0.01 / dur
    return tr_rate, body_rate

def compute_tick_features(ticks: pd.DataFrame) -> pd.DataFrame:
    log.info("  Computing tick-count window features (Numba)…")
    mid   = ticks["mid"].values.astype(np.float64)
    ts_ns = ticks.index.view("int64").astype(np.float64)

    tr_rate, body_rate = tick_window_features_nb(mid, ts_ns, TICK_WINDOWS)

    out = {}
    for wi, lbl in enumerate(TICK_LABELS):
        tr  = tr_rate[:, wi]
        look = min(MAD_LOOK, np.sum(~np.isnan(tr)) // 2)
        if look < 10:
            look = 10
        sz   = mad_zscore_nb(tr, look)
        out[f"{lbl}_tr_rate"]  = tr
        out[f"{lbl}_shock_z"]  = sz
        out[f"{lbl}_shock"]    = (sz > SHOCK_Z_THR).astype(np.float32)
        out[f"{lbl}_body_rate"] = body_rate[:, wi]
    return pd.DataFrame(out, index=ticks.index)

# ── 5. DWT decomposition ─────────────────────────────────────────────────────

BAND_LABELS = [f"D{j+1}" for j in range(DWT_LEVEL)] + [f"A{DWT_LEVEL}"]
# At 1s resolution: D1≈2s, D2≈4s, D3≈8s, D4≈16s, D5≈32s, D6≈64s, D7≈128s, D8≈256s, A8=trend

def dwt_features(bars_1s: pd.DataFrame) -> pd.DataFrame:
    log.info(f"  DWT decomposition ({DWT_WAVELET}, level={DWT_LEVEL}) on {len(bars_1s)} 1s bars…")
    close  = bars_1s["close"].ffill().values.astype(np.float64)
    n      = len(close)

    # Wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    coeffs = pywt.wavedec(close, DWT_WAVELET, level=DWT_LEVEL, mode="periodization")

    out = {}
    for j, c in enumerate(coeffs):
        # j=0 → A8, j=1 → D8, j=2 → D7, ..., j=8 → D1
        if j == 0:
            lbl = f"A{DWT_LEVEL}"
        else:
            lbl = f"D{DWT_LEVEL - j + 1}"

        # Upsample to original length (repeat each coefficient)
        factor  = n // len(c)
        remainder = n - factor * len(c)
        up = np.repeat(c, factor)
        if remainder > 0:
            up = np.concatenate([up, np.repeat(c[-1:], remainder)])
        up = up[:n]

        energy = up ** 2
        look   = min(MAD_LOOK, n // 4)
        sz     = mad_zscore_nb(energy, look)

        out[f"wt_{lbl}_coef"]   = up
        out[f"wt_{lbl}_energy"] = energy
        out[f"wt_{lbl}_shock_z"] = sz
        out[f"wt_{lbl}_shock"]  = (sz > SHOCK_Z_THR).astype(np.float32)

    return pd.DataFrame(out, index=bars_1s.index)

# ── 6. Lift table ────────────────────────────────────────────────────────────

def lift_table(shock_src: np.ndarray, shock_tgt: np.ndarray, lag: int = 1) -> dict:
    """P(shock_j(t+lag)=1 | shock_i(t)=1) / P(shock_j=1)."""
    n       = len(shock_src)
    s       = shock_src[:n - lag]
    t       = shock_tgt[lag:]
    base    = float(np.nanmean(t))
    if base < 1e-9:
        return {"cond_rate": np.nan, "base_rate": base, "lift": np.nan}
    mask    = s == 1
    if mask.sum() < 10:
        return {"cond_rate": np.nan, "base_rate": base, "lift": np.nan}
    cond    = float(np.nanmean(t[mask]))
    return {"cond_rate": round(cond, 6), "base_rate": round(base, 6),
            "lift": round(cond / base, 3)}

# ── 7. Cross-scale energy xcorr ──────────────────────────────────────────────

def xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = 30) -> np.ndarray:
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    std_a = np.nanstd(a); std_b = np.nanstd(b)
    if std_a < 1e-12 or std_b < 1e-12:
        return np.zeros(2 * max_lag + 1)
    corrs = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corrs.append(np.nanmean(a[:-lag] * b[lag:]) / (std_a * std_b))
        elif lag < 0:
            corrs.append(np.nanmean(a[-lag:] * b[:lag]) / (std_a * std_b))
        else:
            corrs.append(np.nanmean(a * b) / (std_a * std_b))
    return np.array(corrs)

# ── 8. Plotting helpers ───────────────────────────────────────────────────────

def save(fig, name: str):
    p = OUT_DIR / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot saved: {p}")

def colorbar_norm(data, vmin=None, vmax=None):
    v = np.nanmax(np.abs(data)) if vmax is None else vmax
    return plt.Normalize(-v, v)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("MSP EXPERIMENT — AUD/JPY April 2026 Tick Data")
    log.info("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    ticks = load_histdata(ZIP_PATH)

    # ── 2. Overview plot ─────────────────────────────────────────
    log.info("Plotting tick data overview…")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ticks_1min = ticks["mid"].resample("1min").last().dropna()
    ax1.plot(ticks_1min.index, ticks_1min.values, lw=0.5, color="#81c784")
    ax1.set_ylabel("AUD/JPY mid"); ax1.set_title("AUD/JPY — April 2026 Tick Data (1-min close)")
    ax1.grid(True, color="#222", lw=0.4)

    spread_1min = ticks["spread"].resample("1min").mean().dropna()
    ax2.plot(spread_1min.index, spread_1min.values, lw=0.5, color="#ffb74d")
    ax2.set_ylabel("Spread (pips)"); ax2.set_xlabel("Date")
    ax2.grid(True, color="#222", lw=0.4)
    fig.patch.set_facecolor("#0d0d0d")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0d0d"); ax.tick_params(colors="#888")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.yaxis.label.set_color("#888"); ax.xaxis.label.set_color("#888")
        ax.title.set_color("#ccc")
    save(fig, "01_overview.png")

    # ── 3. Time-grid features ────────────────────────────────────
    log.info("Computing time-grid features…")
    time_feats: dict[str, pd.DataFrame] = {}
    for grain, dur_min in zip(TIME_GRAINS, GRAIN_MINS):
        log.info(f"  {grain}…")
        time_feats[grain] = time_grid_features(ticks, grain, dur_min)
    log.info("Time-grid features done.")

    # ── 4. Shock rate table ─────────────────────────────────────
    log.info("Shock rates per time grain:")
    shock_rates = {}
    for grain, df in time_feats.items():
        rate = float(df["shock"].mean())
        shock_rates[grain] = rate
        log.info(f"  {grain:>8s}: shock_rate={rate:.4f}  n_bars={len(df):,}")

    sr_df = pd.DataFrame.from_dict(shock_rates, orient="index", columns=["shock_rate"])
    sr_df.to_csv(OUT_DIR / "shock_rates_time.csv")

    # ── 5. Tick-count features ───────────────────────────────────
    log.info("Computing tick-count window features…")
    tick_feats = compute_tick_features(ticks)
    log.info("Tick features done.")

    tick_shock_rates = {}
    for lbl in TICK_LABELS:
        rate = float(np.nanmean(tick_feats[f"{lbl}_shock"].values))
        tick_shock_rates[lbl] = rate
        log.info(f"  {lbl:>6s}: shock_rate={rate:.4f}")

    # ── 6. Lift tables: time-grid windows ───────────────────────
    log.info("Computing lift tables (time-grid)…")
    lift_rows = []
    grain_list = TIME_GRAINS
    shock_arrays = {g: time_feats[g]["shock"].reindex(time_feats[grain_list[-1]].index).ffill().values
                    for g in grain_list}

    # For lift: work on a common 15-min grid (coarsest)
    # Resample each grain's shock to 15min (max within window)
    shock_15 = {}
    for g in grain_list:
        s = time_feats[g]["shock"]
        shock_15[g] = s.resample("15min").max().reindex(
            time_feats["15min"].index
        ).fillna(0).values

    for i, src in enumerate(grain_list):
        for j, tgt in enumerate(grain_list):
            if i >= j:
                continue
            for lag in [0, 1, 2, 3]:
                r = lift_table(shock_15[src], shock_15[tgt], lag=lag)
                if not np.isnan(r["lift"]):
                    lift_rows.append({
                        "src": src, "tgt": tgt, "lag_15min": lag,
                        **r,
                    })

    lift_df = pd.DataFrame(lift_rows).sort_values("lift", ascending=False)
    lift_df.to_csv(OUT_DIR / "lift_table_time.csv", index=False)
    log.info(f"  Top lifts:\n{lift_df.head(10).to_string(index=False)}")

    # ── 7. Cross-window xcorr (TR-rate) ─────────────────────────
    log.info("Cross-window TR-rate xcorr (time grids)…")
    # Align to 1min grid
    tr_aligned = {}
    for g in TIME_GRAINS:
        s = time_feats[g]["tr_rate"].resample("1min").mean()
        tr_aligned[g] = s.reindex(time_feats["1min"].index).ffill().values

    xcorr_grid = np.full((len(TIME_GRAINS), len(TIME_GRAINS)), np.nan)
    for i, gi in enumerate(TIME_GRAINS):
        for j, gj in enumerate(TIME_GRAINS):
            cc = xcorr(tr_aligned[gi], tr_aligned[gj], max_lag=0)
            xcorr_grid[i, j] = cc[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(xcorr_grid, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TIME_GRAINS))); ax.set_yticks(range(len(TIME_GRAINS)))
    ax.set_xticklabels(TIME_GRAINS, rotation=45, ha="right", color="#888")
    ax.set_yticklabels(TIME_GRAINS, color="#888")
    for i in range(len(TIME_GRAINS)):
        for j in range(len(TIME_GRAINS)):
            ax.text(j, i, f"{xcorr_grid[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("TR-rate Correlation Matrix (time windows)", color="#ccc")
    ax.set_facecolor("#0d0d0d"); fig.patch.set_facecolor("#0d0d0d")
    save(fig, "02_xcorr_matrix_time.png")

    # ── 8. Lagged cross-correlation: 1s → 1min, 5min ───────────
    log.info("Lagged xcorr: small → large time windows…")
    lags  = list(range(0, 61))
    pairs = [("1s", "1min"), ("1s", "5min"), ("30s", "5min"), ("30s", "15min"), ("1min", "15min")]

    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 2.8 * len(pairs)), facecolor="#0d0d0d")
    for ax, (src, tgt) in zip(axes, pairs):
        src_arr = tr_aligned[src]; tgt_arr = tr_aligned[tgt]
        cc = [xcorr(src_arr, tgt_arr, max_lag=0)[0]] + \
             [float(np.nanmean(src_arr[:-lag] * tgt_arr[lag:]) /
                    (np.nanstd(src_arr) * np.nanstd(tgt_arr) + 1e-12))
              for lag in lags[1:]]
        ax.plot(lags, cc, color="#4fc3f7", lw=1.2)
        ax.axhline(0, color="#444", lw=0.6, ls="--")
        ax.axvline(0, color="#555", lw=0.4)
        ax.set_title(f"corr(TR_rate[{src}](t),  TR_rate[{tgt}](t+lag))", color="#aaa", fontsize=10)
        ax.set_xlabel("Lag (1-min units)", color="#666", fontsize=8)
        ax.set_ylabel("Corr", color="#666", fontsize=8)
        ax.set_facecolor("#0d0d0d"); ax.tick_params(colors="#555")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.grid(True, color="#1a1a1a", lw=0.4)
    plt.tight_layout()
    save(fig, "03_lagged_xcorr_time.png")

    # ── 9. DWT decomposition ─────────────────────────────────────
    log.info("DWT decomposition…")
    bars_1s  = time_feats["1s"]
    wt_feats = dwt_features(bars_1s)

    log.info("DWT band statistics:")
    for lbl in BAND_LABELS:
        e_mean = float(np.nanmean(wt_feats[f"wt_{lbl}_energy"]))
        s_rate = float(np.nanmean(wt_feats[f"wt_{lbl}_shock"]))
        log.info(f"  {lbl:>4s}: mean_energy={e_mean:.6f}  shock_rate={s_rate:.4f}")

    pd.DataFrame({
        lbl: {
            "mean_energy": float(np.nanmean(wt_feats[f"wt_{lbl}_energy"])),
            "shock_rate":  float(np.nanmean(wt_feats[f"wt_{lbl}_shock"])),
            "coef_std":    float(np.nanstd(wt_feats[f"wt_{lbl}_coef"])),
        }
        for lbl in BAND_LABELS
    }).T.to_csv(OUT_DIR / "dwt_band_stats.csv")

    # ── 10. DWT energy plot (sample day) ─────────────────────────
    log.info("DWT energy plot (first trading day)…")
    day1 = bars_1s.index.date[0]
    mask = wt_feats.index.date == day1
    wt_day = wt_feats[mask]

    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(len(BAND_LABELS) + 1, 1, hspace=0.05)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(BAND_LABELS)))

    # Price
    ax0 = fig.add_subplot(gs[0])
    price_day = bars_1s.loc[mask, "close"]
    ax0.plot(price_day.index, price_day.values, color="#ccc", lw=0.6)
    ax0.set_ylabel("Price", color="#888", fontsize=7); ax0.tick_params(colors="#555", labelsize=6)
    ax0.set_facecolor("#0d0d0d"); ax0.set_xticklabels([])
    for sp in ax0.spines.values(): sp.set_color("#333")
    ax0.set_title(f"AUD/JPY DWT Energy Bands — {day1}", color="#ccc", fontsize=11)

    for i, lbl in enumerate(BAND_LABELS):
        ax = fig.add_subplot(gs[i + 1])
        e = wt_day[f"wt_{lbl}_energy"].values
        ax.fill_between(wt_day.index, 0, e, color=colors[i], alpha=0.7, lw=0)
        ax.set_ylabel(lbl, color=colors[i], fontsize=7, rotation=0, labelpad=24)
        ax.tick_params(colors="#555", labelsize=6)
        ax.set_facecolor("#0d0d0d")
        for sp in ax.spines.values(): sp.set_color("#333")
        if i < len(BAND_LABELS) - 1:
            ax.set_xticklabels([])
    save(fig, "04_dwt_energy_day1.png")

    # ── 11. DWT cross-scale energy xcorr ─────────────────────────
    log.info("DWT cross-scale energy xcorr…")
    MAX_DWT_LAG = 60  # seconds
    detail_bands = [f"D{j+1}" for j in range(DWT_LEVEL)]

    # Build energy arrays aligned to 1s index
    energies = {
        lbl: wt_feats[f"wt_{lbl}_energy"].values
        for lbl in detail_bands
    }

    # Symmetric xcorr matrix at lag=0
    n_bands = len(detail_bands)
    xcorr_dwt = np.full((n_bands, n_bands), np.nan)
    for i, li in enumerate(detail_bands):
        for j, lj in enumerate(detail_bands):
            cc = xcorr(energies[li], energies[lj], max_lag=0)[0]
            xcorr_dwt[i, j] = cc

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(xcorr_dwt, cmap="RdYlBu_r", vmin=-0.5, vmax=1)
    ax.set_xticks(range(n_bands)); ax.set_yticks(range(n_bands))
    ax.set_xticklabels(detail_bands, rotation=45, ha="right", color="#888")
    ax.set_yticklabels(detail_bands, color="#888")
    for i in range(n_bands):
        for j in range(n_bands):
            ax.text(j, i, f"{xcorr_dwt[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("DWT Energy Band Correlation Matrix (lag=0)", color="#ccc")
    ax.set_facecolor("#0d0d0d"); fig.patch.set_facecolor("#0d0d0d")
    save(fig, "05_dwt_xcorr_matrix.png")

    # ── 12. DWT lagged xcorr: D1/D2 → D4/D5/D6 ─────────────────
    log.info("DWT lagged cross-scale xcorr (propagation test)…")
    dwt_pairs = [
        ("D1", "D3"), ("D1", "D4"), ("D1", "D5"),
        ("D2", "D4"), ("D2", "D5"), ("D3", "D5"),
        ("D3", "D6"), ("D4", "D7"),
    ]
    lags_dwt = list(range(0, MAX_DWT_LAG + 1))

    fig, axes = plt.subplots(len(dwt_pairs), 1, figsize=(12, 2.5 * len(dwt_pairs)), facecolor="#0d0d0d")
    dwt_xcorr_rows = []
    for ax, (src, tgt) in zip(axes, dwt_pairs):
        a = energies[src]; b = energies[tgt]
        cc = [xcorr(a, b, max_lag=0)[0]] + [
            float(np.nanmean(a[:-lag] * b[lag:]) / (np.nanstd(a) * np.nanstd(b) + 1e-12))
            for lag in lags_dwt[1:]
        ]
        peak_lag  = int(np.argmax(cc))
        peak_corr = cc[peak_lag]
        dwt_xcorr_rows.append({
            "src": src, "tgt": tgt,
            "corr_lag0": round(cc[0], 4),
            "peak_corr": round(peak_corr, 4),
            "peak_lag_s": peak_lag,
        })
        ax.plot(lags_dwt, cc, color="#ce93d8", lw=1.2)
        ax.axhline(0, color="#444", lw=0.6, ls="--")
        ax.axvline(peak_lag, color="#ffb74d", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(
            f"corr(E_{src}(t), E_{tgt}(t+lag))   peak={peak_corr:.3f} @ lag={peak_lag}s",
            color="#aaa", fontsize=9)
        ax.set_xlabel("Lag (seconds)", color="#666", fontsize=8)
        ax.set_ylabel("Corr", color="#666", fontsize=8)
        ax.set_facecolor("#0d0d0d"); ax.tick_params(colors="#555")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.grid(True, color="#1a1a1a", lw=0.4)
    plt.tight_layout()
    save(fig, "06_dwt_lagged_xcorr.png")

    dwt_xcorr_df = pd.DataFrame(dwt_xcorr_rows)
    dwt_xcorr_df.to_csv(OUT_DIR / "dwt_lagged_xcorr.csv", index=False)
    log.info(f"DWT lagged xcorr:\n{dwt_xcorr_df.to_string(index=False)}")

    # ── 13. DWT lift tables ──────────────────────────────────────
    log.info("DWT lift tables…")
    dwt_lift_rows = []
    shock_arrays_dwt = {lbl: wt_feats[f"wt_{lbl}_shock"].values for lbl in detail_bands}
    for i, src in enumerate(detail_bands):
        for j, tgt in enumerate(detail_bands):
            if i >= j:
                continue
            for lag in [0, 1, 2, 5, 10, 30]:
                r = lift_table(shock_arrays_dwt[src], shock_arrays_dwt[tgt], lag=lag)
                if not np.isnan(r.get("lift", np.nan)):
                    dwt_lift_rows.append({"src": src, "tgt": tgt, "lag_s": lag, **r})

    dwt_lift_df = pd.DataFrame(dwt_lift_rows).sort_values("lift", ascending=False)
    dwt_lift_df.to_csv(OUT_DIR / "dwt_lift_table.csv", index=False)
    log.info(f"Top DWT lifts:\n{dwt_lift_df.head(15).to_string(index=False)}")

    # ── 14. DWT lift heatmap (lag=1s) ───────────────────────────
    lift_lag1 = dwt_lift_df[dwt_lift_df["lag_s"] == 1].set_index(["src", "tgt"])["lift"]
    lift_mat  = np.full((n_bands, n_bands), np.nan)
    for i, si in enumerate(detail_bands):
        for j, tj in enumerate(detail_bands):
            if (si, tj) in lift_lag1.index:
                lift_mat[i, j] = lift_lag1[(si, tj)]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(lift_mat, cmap="RdYlGn", vmin=0.5, vmax=3.0)
    ax.set_xticks(range(n_bands)); ax.set_yticks(range(n_bands))
    ax.set_xticklabels(detail_bands, rotation=45, ha="right", color="#888")
    ax.set_yticklabels(detail_bands, color="#888")
    for i in range(n_bands):
        for j in range(n_bands):
            v = lift_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Lift (lag=1s)")
    ax.set_xlabel("Target band (shock receiver)", color="#888")
    ax.set_ylabel("Source band (shock emitter)", color="#888")
    ax.set_title("DWT Shock Propagation Lift — P(tgt_shock|src_shock,lag=1s)/P(tgt_shock)\n"
                 "Row=source, Col=target; >1 = propagation evidence", color="#ccc")
    ax.set_facecolor("#0d0d0d"); fig.patch.set_facecolor("#0d0d0d")
    save(fig, "07_dwt_lift_heatmap.png")

    # ── 15. Shock cascade sample events ─────────────────────────
    log.info("Locating shock cascade events…")
    # Find moments where D1 shock fires AND D3+ energy spikes in the next 10s
    d1_shock  = wt_feats["wt_D1_shock"].values
    d4_energy = wt_feats["wt_D4_energy"].values
    d4_thresh = np.nanpercentile(d4_energy, 90)

    cascade_idx = []
    for i in range(len(d1_shock) - 15):
        if d1_shock[i] == 1:
            if np.any(d4_energy[i:i+10] > d4_thresh):
                cascade_idx.append(i)

    log.info(f"  Found {len(cascade_idx):,} D1→D4 cascade events "
             f"({len(cascade_idx)/len(d1_shock)*100:.2f}% of all bars)")

    # Plot 3 example events
    if len(cascade_idx) >= 3:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor="#0d0d0d")
        bands_to_show = ["D1", "D2", "D3", "D4", "D5"]
        colors2 = plt.cm.viridis(np.linspace(0.2, 0.9, len(bands_to_show)))
        for k, ax in enumerate(axes):
            idx = cascade_idx[k]
            window = slice(max(0, idx - 30), min(len(wt_feats), idx + 60))
            t = wt_feats.index[window]
            price_w = bars_1s.loc[t, "close"] if t[0] in bars_1s.index else None
            if price_w is not None:
                ax2b = ax.twinx()
                ax2b.plot(t, price_w.reindex(t).ffill().values, color="#444", lw=0.8, alpha=0.5)
                ax2b.set_ylabel("Price", color="#555", fontsize=7)
                ax2b.tick_params(colors="#555")
            for bi, (b, c) in enumerate(zip(bands_to_show, colors2)):
                e = wt_feats[f"wt_{b}_energy"].values[window]
                e_norm = e / (np.nanmax(e) + 1e-12)
                ax.fill_between(range(len(t)), 0, e_norm + bi * 1.1,
                                color=c, alpha=0.6, label=b if k == 0 else "")
            # Mark D1 shock
            t_rel = idx - window.start
            ax.axvline(t_rel, color="#e57373", lw=1.2, ls="--", alpha=0.8)
            ax.set_title(f"Cascade event @ {wt_feats.index[idx]} (D1 shock → D4 spike)",
                         color="#aaa", fontsize=9)
            ax.set_facecolor("#0d0d0d"); ax.tick_params(colors="#555")
            for sp in ax.spines.values(): sp.set_color("#333")
        axes[0].legend(loc="upper left", fontsize=8, facecolor="#111", labelcolor="#888")
        plt.tight_layout()
        save(fig, "08_cascade_events.png")

    # ── 16. Summary ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 60)
    log.info(f"Pair: {PAIR} | April 2026 | {len(ticks):,} ticks")
    log.info(f"Avg tick interval: {ticks.index.to_series().diff().dt.total_seconds().median():.1f}s")
    log.info(f"Avg spread: {ticks['spread'].mean():.3f}p")
    log.info("")
    log.info("Shock rates by time window:")
    for g, r in shock_rates.items():
        log.info(f"  {g:>8s}: {r:.4f} ({r*100:.2f}%)")
    log.info("")

    log.info("DWT cross-scale propagation (key findings):")
    sig = dwt_xcorr_df[dwt_xcorr_df["peak_corr"] > 0.05]
    if len(sig):
        log.info(sig.to_string(index=False))
    else:
        log.info("  No significant positive lagged correlations found.")

    log.info("")
    log.info("Top DWT lift values (lag=1s):")
    top_lift = dwt_lift_df[dwt_lift_df["lag_s"] == 1].head(8)
    log.info(top_lift[["src","tgt","lift","cond_rate","base_rate"]].to_string(index=False))

    log.info("")
    log.info("First hypothesis assessment:")
    top = dwt_xcorr_df.iloc[0] if len(dwt_xcorr_df) else None
    if top is not None and top["peak_corr"] > 0.1:
        log.info(f"  ✅ Cross-scale propagation DETECTED: {top['src']}→{top['tgt']} "
                 f"peak={top['peak_corr']:.3f} @ lag={top['peak_lag_s']}s")
    else:
        log.info("  ⚠️  Weak or no cross-scale propagation in this dataset")

    log.info("")
    log.info("Output files:")
    for f in sorted(OUT_DIR.iterdir()):
        log.info(f"  {f.name}")

    log.info("=" * 60)
    log.info("DONE")

if __name__ == "__main__":
    main()
