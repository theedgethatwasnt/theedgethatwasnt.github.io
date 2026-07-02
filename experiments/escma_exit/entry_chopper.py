"""
entry_chopper.py — Detect high-momentum entry events on S5 BA data and chop
each event into a training sample for the CMA-NN exit-learner.

==========================================================================
WHAT THIS DOES
==========================================================================

1. Load `data/s5_ba/<PAIR>_S5_BA.parquet`.
2. Compute multi-window momentum + σ-scaled momentum at every bar:
     S5 (w=1, 5s), M1 (w=12, 60s), 5m (w=60), 15m (w=180), 1h (w=720)
   Raw  : (close[t] − close[t-w]) / pip / minutes     (pips/min)
   σ-norm: raw / σ_mom_per_minute  (dimensionless, Sharpe-like)
   For S5 (w=1) the σ baseline uses the stddev of per-bar pip-deltas over a
   rolling 60-bar (5 min) lookback (per spec) so it's well-defined.
   For M1/5m/15m/1h: σ_mom_per_min = σ_bar_per_min / sqrt(w). This is the
   CLT correction — the std of an average of w iid bar-deltas (each with
   per-min std σ_bar_per_min) is σ_bar_per_min / sqrt(w). Without this
   correction the σ-norm distribution was off by sqrt(w), so |val|≥2 was
   essentially impossible to hit on 5m/15m/1h.

3. Detect HIGH-MOMENTUM EVENTS using these gates (closed bars only):
     a) ≥2 of {5m, 15m, 1h} σ-normed momenta exceed |thr_sigma| in same sign,
     b) the fastest non-None σ-normed window (S5 or M1) has same sign and
        magnitude ≥ thr_fast_sigma,
     c) previous bar did NOT satisfy the rule (first-bar of spike).

4. Chop each event into:
     pre-context: 720 S5 bars before t_event (= 60 min = 1 hour of run-up),
                 with OHLC + 5 mom + 5 mn + spread per bar.
                 Rationale: NN needs to see the run-up to the shock, not just the
                 last 5 min. 720 matches our longest momentum window (1h).
     post-path : t_event ... t_event + max_post_bars (default 24h ≈ 17280 S5 bars),
                 NOT serialized — meta records (t_event_idx, n_post_bars) and the
                 trainer slices the source parquet directly.

5. Write:
     samples_<PAIR>.parquet   — long-format pre-entry rows (60 per sample)
     meta_<PAIR>.parquet      — per-sample metadata + entry-time mom/mn snapshot

==========================================================================
DEFAULT THRESHOLD CHOICE (documented per spec)
==========================================================================

thr_sigma=2.0, thr_fast_sigma=2.0 — picked because:
- A |t|~2 σ-normed move is the conventional "unusual" threshold from the live
  fx_signals service docstring (see services/fx_signals/main.py:167).
- Requiring 2 of 3 slow windows + a fast confirm keeps events sparse and
  meaningful (avoids single-window false alarms).
On 5.5 years of USD_JPY S5 (~21M bars) this yields a sample count in the
thousands, comfortable for CMA-NN training.

==========================================================================
SOP COMPLIANCE
==========================================================================
R1: All gates use only bars ≤ t (60-bar baseline σ uses [t-60+1, t]).
R3: Signals on close (mid). Entry fill recorded as ask_c[t]/bid_c[t] per dir.
R4: Rolling state via numba running sums (no df.rolling, no full-array ops).
R8: IS/OOS = first 70% / last 30% of events, tagged in meta.

Note on warmup-drop: with pre_bars=720 and the 1h momentum window (w=720),
events at index t need clean features at t-pre_bars (i.e., t-720). The 1h
momentum at bar k requires bars [k-720, k], so the earliest clean event
index is pre_bars + W_1h = 720 + 720 = 1440. Events with t_event_idx<1440
are dropped to avoid serving NaN/partial features in the pre-context.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange


# ── Constants ────────────────────────────────────────────────────────────────

# S5 bar timing
BAR_MIN = 1.0 / 12.0   # 5 seconds = 1/12 minute

# Window stack — (w_bars, label, win_minutes, use_baseline_sigma)
# w=1 (S5) and w=12 (M1) use 60-bar rolling baseline σ per spec.
# Larger windows use within-window σ of bar-deltas.
WINDOWS = [
    (1,   "S5",  1   * BAR_MIN, True),    # 5s / 5min baseline σ
    (12,  "M1",  12  * BAR_MIN, True),    # 60s / 5min baseline σ
    (60,  "5m",  60  * BAR_MIN, False),   # within-window σ
    (180, "15m", 180 * BAR_MIN, False),
    (720, "1h",  720 * BAR_MIN, False),
]
W_S5, W_M1, W_5m, W_15m, W_1h = (w for w, *_ in WINDOWS)
MIN_PER_S5  = WINDOWS[0][2]
MIN_PER_M1  = WINDOWS[1][2]
MIN_PER_5m  = WINDOWS[2][2]
MIN_PER_15m = WINDOWS[3][2]
MIN_PER_1h  = WINDOWS[4][2]

# Rolling baseline σ window for S5/M1 (per spec: 60 S5 bars = 5 min)
SIGMA_BASELINE_W = 60

# Pre/post sample sizing
PRE_BARS_DEFAULT  = 720      # 60 min of run-up context (was 60 = 5 min)
POST_BARS_DEFAULT = 17280    # 24h of S5 bars
IS_FRAC           = 0.70

# Horizons (S5 bars) for the MFE/MAE summary print
SUMMARY_HORIZONS = [
    (12,    "1m"),
    (60,    "5m"),
    (360,   "30m"),
    (720,   "1h"),
    (2880,  "4h"),
]


# ── Pip per pair ─────────────────────────────────────────────────────────────

def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


# ── Feature computation (numba) ──────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _compute_momentum_stack(close, pip):
    """Compute raw momentum + σ-scaled momentum for all 5 windows at every bar.

    σ-scaling rules (per spec):
      • S5 (w=1):   raw = (c[t]-c[t-1])/pip / (1*bar_min)
                    σ_per_min = stddev(pip_delta over last 60 bars) / bar_min
      • M1 (w=12):  raw = (c[t]-c[t-12])/pip / (12*bar_min)
                    σ_per_min = same 60-bar baseline (per explicit spec note)
      • 5m/15m/1h:  raw same; σ_per_min = stddev(bar_deltas over the w bars)/bar_min

    Returns 10 arrays of length n (mom_*, mn_*).
    NaN-equivalent: -1e30 sentinel; trainer should mask via isfinite check via
    np.where(mom == -1e30, np.nan, ...) wrapper outside this kernel. We use
    explicit sentinel to keep the kernel float64-only and Numba-happy.
    """
    n = close.shape[0]

    mom_s5  = np.full(n, np.nan, dtype=np.float64)
    mom_m1  = np.full(n, np.nan, dtype=np.float64)
    mom_5m  = np.full(n, np.nan, dtype=np.float64)
    mom_15m = np.full(n, np.nan, dtype=np.float64)
    mom_1h  = np.full(n, np.nan, dtype=np.float64)

    mn_s5  = np.full(n, np.nan, dtype=np.float64)
    mn_m1  = np.full(n, np.nan, dtype=np.float64)
    mn_5m  = np.full(n, np.nan, dtype=np.float64)
    mn_15m = np.full(n, np.nan, dtype=np.float64)
    mn_1h  = np.full(n, np.nan, dtype=np.float64)

    # 1) Compute bar-to-bar pip deltas
    pip_delta = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        pip_delta[t] = (close[t] - close[t-1]) / pip

    # 2) Rolling 60-bar σ baseline for S5/M1 — Welford-like running sum / sum-of-sq
    #    Maintained on a window of SIGMA_BASELINE_W bars [t-SIGMA_BASELINE_W+1, t]
    base_sum   = 0.0
    base_sumsq = 0.0
    bw = SIGMA_BASELINE_W
    sigma_base_per_min = np.full(n, np.nan, dtype=np.float64)
    for t in range(1, n):
        d = pip_delta[t]
        base_sum   += d
        base_sumsq += d * d
        if t > bw:
            d_old = pip_delta[t - bw]
            base_sum   -= d_old
            base_sumsq -= d_old * d_old
        if t >= bw:
            mean = base_sum / bw
            var  = max(0.0, base_sumsq / bw - mean * mean)
            sigma_bar = np.sqrt(var)
            if sigma_bar > 0.0:
                sigma_base_per_min[t] = sigma_bar / BAR_MIN

    # 3) Raw + σ-scaled momentum per window
    # --- S5 (w=1) ---
    win_min = 1 * BAR_MIN
    for t in range(1, n):
        mom = (close[t] - close[t-1]) / pip / win_min
        mom_s5[t] = mom
        spm = sigma_base_per_min[t]
        if not np.isnan(spm) and spm > 0.0:
            mn_s5[t] = mom / spm

    # --- M1 (w=12) ---
    # Bugfix (2026-06-11): apply CLT correction. σ of an avg of w iid bar
    # deltas is σ_bar/sqrt(w), so std(mom in pips/min) is
    # σ_baseline_per_min/sqrt(w) — not σ_baseline_per_min. Without this
    # the M1 σ-norm distribution sat at ~p99≈0.7 and never crossed |2|σ.
    win_min = 12 * BAR_MIN
    sqrt_w_m1 = np.sqrt(12.0)
    for t in range(12, n):
        mom = (close[t] - close[t-12]) / pip / win_min
        mom_m1[t] = mom
        spm = sigma_base_per_min[t]
        if not np.isnan(spm) and spm > 0.0:
            mn_m1[t] = (mom * sqrt_w_m1) / spm

    # --- 5m (w=60), within-window σ ---
    # Maintain running sum + sumsq over [t-w+1, t] of pip_delta (= bar deltas).
    # NB: pip_delta is defined for indices >=1; we start at t=w when the window is full.
    # Bugfix (2026-06-11): same CLT correction as M1.
    w = W_5m
    win_min = w * BAR_MIN
    sqrt_w = np.sqrt(float(w))
    rsum = 0.0
    rsumsq = 0.0
    for t in range(1, n):
        d = pip_delta[t]
        rsum   += d
        rsumsq += d * d
        if t > w:
            d_old = pip_delta[t - w]
            rsum   -= d_old
            rsumsq -= d_old * d_old
        if t >= w:
            mom = (close[t] - close[t - w]) / pip / win_min
            mom_5m[t] = mom
            mean = rsum / w
            var  = max(0.0, rsumsq / w - mean * mean)
            sigma_bar = np.sqrt(var)
            sigma_per_min = sigma_bar / BAR_MIN
            if sigma_per_min > 0.0:
                mn_5m[t] = (mom * sqrt_w) / sigma_per_min

    # --- 15m (w=180) ---
    w = W_15m
    win_min = w * BAR_MIN
    sqrt_w = np.sqrt(float(w))
    rsum = 0.0
    rsumsq = 0.0
    for t in range(1, n):
        d = pip_delta[t]
        rsum   += d
        rsumsq += d * d
        if t > w:
            d_old = pip_delta[t - w]
            rsum   -= d_old
            rsumsq -= d_old * d_old
        if t >= w:
            mom = (close[t] - close[t - w]) / pip / win_min
            mom_15m[t] = mom
            mean = rsum / w
            var  = max(0.0, rsumsq / w - mean * mean)
            sigma_bar = np.sqrt(var)
            sigma_per_min = sigma_bar / BAR_MIN
            if sigma_per_min > 0.0:
                mn_15m[t] = (mom * sqrt_w) / sigma_per_min

    # --- 1h (w=720) ---
    w = W_1h
    win_min = w * BAR_MIN
    sqrt_w = np.sqrt(float(w))
    rsum = 0.0
    rsumsq = 0.0
    for t in range(1, n):
        d = pip_delta[t]
        rsum   += d
        rsumsq += d * d
        if t > w:
            d_old = pip_delta[t - w]
            rsum   -= d_old
            rsumsq -= d_old * d_old
        if t >= w:
            mom = (close[t] - close[t - w]) / pip / win_min
            mom_1h[t] = mom
            mean = rsum / w
            var  = max(0.0, rsumsq / w - mean * mean)
            sigma_bar = np.sqrt(var)
            sigma_per_min = sigma_bar / BAR_MIN
            if sigma_per_min > 0.0:
                mn_1h[t] = (mom * sqrt_w) / sigma_per_min

    return (mom_s5, mom_m1, mom_5m, mom_15m, mom_1h,
            mn_s5,  mn_m1,  mn_5m,  mn_15m,  mn_1h)


# ── Event detection (numba) ──────────────────────────────────────────────────

@njit(cache=True)
def _detect_events(mn_s5, mn_m1, mn_5m, mn_15m, mn_1h,
                   thr_sigma, thr_fast_sigma, min_gap_bars):
    """First-bar-of-spike detection:

      slow gate: ≥2 of {5m, 15m, 1h} σ-normed momenta exceed |thr_sigma|
                 in the SAME direction (all positive or all negative for the
                 2 that qualify).
      fast gate: prefer M1 if not NaN, else S5. Same sign, |val| ≥ thr_fast_sigma.
      first-bar: rule(t-1) must have been False.
      min_gap : ignore events fired within min_gap_bars of the previous event.
    """
    n = mn_s5.shape[0]
    event_idx = np.empty(n, dtype=np.int64)
    event_dir = np.empty(n, dtype=np.int8)
    n_events = 0

    prev_fired = False
    last_event_t = -10_000_000  # arbitrary far-past sentinel

    for t in range(n):
        v5  = mn_5m[t]
        v15 = mn_15m[t]
        v1h = mn_1h[t]

        # Need all three valid (warmup)
        if np.isnan(v5) or np.isnan(v15) or np.isnan(v1h):
            prev_fired = False
            continue

        # Count slow-window hits, separately positive and negative
        pos_hits = 0
        neg_hits = 0
        if v5  >=  thr_sigma: pos_hits += 1
        if v15 >=  thr_sigma: pos_hits += 1
        if v1h >=  thr_sigma: pos_hits += 1
        if v5  <= -thr_sigma: neg_hits += 1
        if v15 <= -thr_sigma: neg_hits += 1
        if v1h <= -thr_sigma: neg_hits += 1

        direction = 0
        if pos_hits >= 2 and neg_hits == 0:
            direction = 1
        elif neg_hits >= 2 and pos_hits == 0:
            direction = -1
        else:
            prev_fired = False
            continue

        # Fast confirm — prefer M1 (richer signal than S5) but fall back to S5
        vfast = mn_m1[t]
        if np.isnan(vfast):
            vfast = mn_s5[t]
        if np.isnan(vfast):
            prev_fired = False
            continue

        if direction == 1:
            if vfast < thr_fast_sigma:
                prev_fired = False
                continue
        else:  # direction == -1
            if vfast > -thr_fast_sigma:
                prev_fired = False
                continue

        # First-bar rule + min-gap
        if prev_fired:
            # still inside the same sustained spike → not a new event
            continue
        if t - last_event_t < min_gap_bars:
            prev_fired = True   # treat as continuation of the previous spike
            continue

        event_idx[n_events] = t
        event_dir[n_events] = direction
        n_events += 1
        last_event_t = t
        prev_fired = True

    return event_idx[:n_events], event_dir[:n_events]


# ── MFE/MAE summary helper (numba) ───────────────────────────────────────────

@njit(cache=True, parallel=True)
def _mfe_mae_at_horizons(close, bid_c, ask_c, pip,
                         event_idx, event_dir, horizons_arr):
    """For each event, for each horizon h (in bars), compute MFE / MAE in pips
    relative to entry fill (ask_c for long, bid_c for short), using mid close
    for the path. Spread cost is NOT applied here — informational only.

    Returns mfe[n_events, n_horizons], mae[n_events, n_horizons].
    """
    n_e = event_idx.shape[0]
    n_h = horizons_arr.shape[0]
    n   = close.shape[0]
    mfe = np.zeros((n_e, n_h), dtype=np.float64)
    mae = np.zeros((n_e, n_h), dtype=np.float64)
    for i in prange(n_e):
        t = event_idx[i]
        d = event_dir[i]
        e_px = ask_c[t] if d == 1 else bid_c[t]
        for j in range(n_h):
            h = horizons_arr[j]
            end = t + h
            if end > n:
                end = n
            best = 0.0
            worst = 0.0
            for k in range(t + 1, end):
                pnl = d * (close[k] - e_px) / pip
                if pnl > best:
                    best = pnl
                if pnl < worst:
                    worst = pnl
            mfe[i, j] = best
            mae[i, j] = worst
    return mfe, mae


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(pair: str,
        thr_sigma: float = 2.0,
        thr_fast_sigma: float = 2.0,
        min_gap_bars: int = 60,           # ≥ 5 min between events
        pre_bars: int = PRE_BARS_DEFAULT,
        post_bars: int = POST_BARS_DEFAULT,
        out_dir: Path | None = None,
        data_path: Path | None = None) -> dict:

    root = Path("/path/to/projects/fx-core")
    if data_path is None:
        data_path = root / f"data/s5_ba/{pair}_S5_BA.parquet"
    if out_dir is None:
        out_dir = root / "research/experiments/escma_exit"
    out_dir.mkdir(parents=True, exist_ok=True)

    pip = _pip(pair)

    import gc

    t0 = time.time()
    print(f"[load] reading {data_path} ...")
    # Read columns directly from arrow → numpy, skip pandas DF (saves ~1 GB peak)
    tbl = pq.read_table(data_path,
                        columns=["timestamp", "open", "high", "low",
                                 "close", "bid_c", "ask_c"])
    # OHLC + bid/ask kept as float32 (source dtype, zero-copy)
    close = tbl.column("close").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    high  = tbl.column("high" ).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    low   = tbl.column("low"  ).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    open_ = tbl.column("open" ).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    bid_c = tbl.column("bid_c").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    ask_c = tbl.column("ask_c").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    ts    = tbl.column("timestamp").to_numpy(zero_copy_only=False)
    del tbl
    gc.collect()
    print(f"[load] {len(close):,} rows in {time.time()-t0:.1f}s")

    n = close.shape[0]
    print(f"[features] computing 5-window momentum stack on {n:,} bars ...")
    t0 = time.time()
    # Kernel needs float64 close for numerical stability of rolling sums.
    close_f64 = close.astype(np.float64, copy=True)
    (mom_s5, mom_m1, mom_5m, mom_15m, mom_1h,
     mn_s5,  mn_m1,  mn_5m,  mn_15m,  mn_1h) = _compute_momentum_stack(close_f64, pip)
    del close_f64
    # Downcast features to float32 to halve memory (≥ ±2σ precision is plenty)
    mom_s5  = mom_s5.astype(np.float32);  mom_m1  = mom_m1.astype(np.float32)
    mom_5m  = mom_5m.astype(np.float32);  mom_15m = mom_15m.astype(np.float32)
    mom_1h  = mom_1h.astype(np.float32)
    mn_s5   = mn_s5.astype(np.float32);   mn_m1   = mn_m1.astype(np.float32)
    mn_5m   = mn_5m.astype(np.float32);   mn_15m  = mn_15m.astype(np.float32)
    mn_1h   = mn_1h.astype(np.float32)
    gc.collect()
    print(f"[features] done in {time.time()-t0:.1f}s")

    print(f"[detect] thr_sigma={thr_sigma}  thr_fast_sigma={thr_fast_sigma}  "
          f"min_gap={min_gap_bars} bars")
    t0 = time.time()
    event_idx, event_dir = _detect_events(
        mn_s5, mn_m1, mn_5m, mn_15m, mn_1h,
        float(thr_sigma), float(thr_fast_sigma), int(min_gap_bars))
    print(f"[detect] {len(event_idx):,} events in {time.time()-t0:.1f}s")

    if len(event_idx) == 0:
        print("[detect] No events found — try lowering thresholds.")
        return {"n_events": 0}

    # Trim events too close to start/end.
    # Earliest valid event index = pre_bars + W_1h: the 1h momentum needs 720
    # prior bars to be well-defined, and the earliest bar of the pre-context
    # sits at t_event - pre_bars. So we need t_event - pre_bars >= W_1h, i.e.
    # t_event >= pre_bars + W_1h. With defaults (720+720) this drops events in
    # the first 1440 bars of the dataset to avoid NaN/partial features.
    min_event_idx = pre_bars + W_1h
    valid = (event_idx >= min_event_idx) & (event_idx < n - 1)
    event_idx = event_idx[valid]
    event_dir = event_dir[valid]
    print(f"[detect] {len(event_idx):,} events after boundary trim "
          f"(pre>={pre_bars} bars + W_1h={W_1h} warmup, post>=1 bar; "
          f"min event idx = {min_event_idx})")

    n_events = len(event_idx)
    n_long  = int((event_dir ==  1).sum())
    n_short = int((event_dir == -1).sum())

    # Spread at entry (informational)
    spread_at_entry = (ask_c[event_idx] - bid_c[event_idx]) / pip
    med_sp = float(np.median(spread_at_entry))

    # ── IS/OOS split — by event chronology (first 70% IS) ────────────────────
    is_n = int(IS_FRAC * n_events)
    split = np.empty(n_events, dtype=object)
    split[:is_n] = "IS"
    split[is_n:] = "OOS"

    # ── MFE/MAE summary at fixed horizons (informational only) ───────────────
    print("[summary] computing MFE/MAE at horizons ...")
    t0 = time.time()
    horizons_arr = np.array([h for h, _ in SUMMARY_HORIZONS], dtype=np.int64)
    mfe, mae = _mfe_mae_at_horizons(close, bid_c, ask_c, pip,
                                    event_idx, event_dir, horizons_arr)
    print(f"[summary] MFE/MAE in {time.time()-t0:.1f}s")

    # ── Build meta table ─────────────────────────────────────────────────────
    meta = pd.DataFrame({
        "sample_id":     np.arange(n_events, dtype=np.int64),
        "t_event_idx":   event_idx.astype(np.int64),
        "timestamp":     pd.to_datetime(ts[event_idx], utc=True),
        "direction":     event_dir.astype(np.int8),
        "entry_px":      np.where(event_dir == 1, ask_c[event_idx], bid_c[event_idx]),
        "spread_pips":   spread_at_entry.astype(np.float32),
        "n_post_bars":   np.minimum(post_bars, n - event_idx - 1).astype(np.int64),
        "split":         split,
        # Entry-time mom/mn snapshot
        "mom_S5_e":      mom_s5[event_idx].astype(np.float32),
        "mom_M1_e":      mom_m1[event_idx].astype(np.float32),
        "mom_5m_e":      mom_5m[event_idx].astype(np.float32),
        "mom_15m_e":     mom_15m[event_idx].astype(np.float32),
        "mom_1h_e":      mom_1h[event_idx].astype(np.float32),
        "mn_S5_e":       mn_s5[event_idx].astype(np.float32),
        "mn_M1_e":       mn_m1[event_idx].astype(np.float32),
        "mn_5m_e":       mn_5m[event_idx].astype(np.float32),
        "mn_15m_e":      mn_15m[event_idx].astype(np.float32),
        "mn_1h_e":       mn_1h[event_idx].astype(np.float32),
    })

    meta_path = out_dir / f"meta_{pair}.parquet"
    meta.to_parquet(meta_path, index=False, compression="zstd")
    print(f"[write] meta → {meta_path} ({meta_path.stat().st_size/1e6:.2f} MB)")

    # ── Build pre-entry sample rows in chunks; stream to parquet ─────────────
    # Long-format: (sample_id, bar_offset[-pre..-1], OHLC, 5 mom, 5 mn, spread).
    # Chunked to bound peak memory: each chunk = events_per_chunk × pre_bars rows.
    print(f"[samples] building pre-entry context for {n_events:,} events "
          f"({pre_bars} bars each) ...")
    t0 = time.time()

    samples_path = out_dir / f"samples_{pair}.parquet"
    schema = pa.schema([
        ("sample_id",   pa.int64()),
        ("bar_offset",  pa.int32()),
        ("open",        pa.float32()),
        ("high",        pa.float32()),
        ("low",         pa.float32()),
        ("close",       pa.float32()),
        ("mom_S5",      pa.float32()),
        ("mom_M1",      pa.float32()),
        ("mom_5m",      pa.float32()),
        ("mom_15m",     pa.float32()),
        ("mom_1h",      pa.float32()),
        ("mn_S5",       pa.float32()),
        ("mn_M1",       pa.float32()),
        ("mn_5m",       pa.float32()),
        ("mn_15m",      pa.float32()),
        ("mn_1h",       pa.float32()),
        ("spread_pips", pa.float32()),
    ])

    # Choose chunk size so each chunk is <~ 200 MB of feature data.
    # rows_per_event = pre_bars; cols = ~17; ~70 bytes/row → events_per_chunk
    EVENTS_PER_CHUNK = max(1, int(150_000_000 / (pre_bars * 70)))
    print(f"[samples] events_per_chunk = {EVENTS_PER_CHUNK}, "
          f"≈{EVENTS_PER_CHUNK * pre_bars:,} rows / chunk")

    bar_offsets_tile = np.arange(-pre_bars, 0, dtype=np.int32)

    total_rows = 0
    with pq.ParquetWriter(samples_path, schema, compression="zstd") as writer:
        for chunk_start in range(0, n_events, EVENTS_PER_CHUNK):
            chunk_end = min(chunk_start + EVENTS_PER_CHUNK, n_events)
            n_chunk   = chunk_end - chunk_start
            ev_chunk  = event_idx[chunk_start:chunk_end]

            # Vectorised index: each event contributes pre_bars source indices
            sample_id  = np.repeat(
                np.arange(chunk_start, chunk_end, dtype=np.int64), pre_bars)
            bar_offset = np.tile(bar_offsets_tile, n_chunk)
            src_idx    = (ev_chunk.repeat(pre_bars).astype(np.int64)
                          + np.tile(bar_offsets_tile.astype(np.int64), n_chunk))

            tbl_chunk = pa.table({
                "sample_id":   sample_id,
                "bar_offset":  bar_offset,
                "open":        open_[src_idx],
                "high":        high[src_idx],
                "low":         low[src_idx],
                "close":       close[src_idx],
                "mom_S5":      mom_s5[src_idx],
                "mom_M1":      mom_m1[src_idx],
                "mom_5m":      mom_5m[src_idx],
                "mom_15m":     mom_15m[src_idx],
                "mom_1h":      mom_1h[src_idx],
                "mn_S5":       mn_s5[src_idx],
                "mn_M1":       mn_m1[src_idx],
                "mn_5m":       mn_5m[src_idx],
                "mn_15m":      mn_15m[src_idx],
                "mn_1h":       mn_1h[src_idx],
                "spread_pips": ((ask_c[src_idx].astype(np.float32)
                                 - bid_c[src_idx].astype(np.float32))
                                / np.float32(pip)),
            }, schema=schema)
            writer.write_table(tbl_chunk)
            total_rows += len(tbl_chunk)
            del tbl_chunk, sample_id, bar_offset, src_idx
            gc.collect()

    print(f"[samples] {total_rows:,} pre-rows written in {time.time()-t0:.1f}s")
    print(f"[write] samples → {samples_path} "
          f"({samples_path.stat().st_size/1e6:.2f} MB)")

    # ── Summary print ────────────────────────────────────────────────────────
    span_years = (ts[-1] - ts[0]) / np.timedelta64(1, "s") / (365.25 * 86400.0)
    print()
    print("=" * 72)
    print(f"  ENTRY EVENT SUMMARY — {pair}")
    print("=" * 72)
    print(f"  Detected {n_events:,} events in {n:,} S5 bars over {span_years:.2f}y.")
    print(f"  Long:  {n_long:,}  ({100*n_long/n_events:5.1f}%)")
    print(f"  Short: {n_short:,}  ({100*n_short/n_events:5.1f}%)")
    print(f"  IS:    {is_n:,}  (first {100*IS_FRAC:.0f}%)")
    print(f"  OOS:   {n_events - is_n:,}  (last {100*(1-IS_FRAC):.0f}%)")
    print(f"  Median spread at entry: {med_sp:.2f}p")
    print(f"  IS range: {meta['timestamp'].iloc[0]}  →  {meta['timestamp'].iloc[is_n-1]}")
    print(f"  OOS range: {meta['timestamp'].iloc[is_n]}  →  {meta['timestamp'].iloc[-1]}")
    print()
    print("  MFE / MAE distribution by horizon (pips, mid-based, NO spread cost):")
    print(f"  {'horizon':<8} {'mfe p25':>9} {'mfe med':>9} {'mfe p75':>9}  "
          f"{'mae p25':>9} {'mae med':>9} {'mae p75':>9}")
    for j, (h, lbl) in enumerate(SUMMARY_HORIZONS):
        mfe_col = mfe[:, j]
        mae_col = mae[:, j]
        print(f"  {lbl:<8} "
              f"{np.percentile(mfe_col, 25):>9.2f} "
              f"{np.percentile(mfe_col, 50):>9.2f} "
              f"{np.percentile(mfe_col, 75):>9.2f}  "
              f"{np.percentile(mae_col, 25):>9.2f} "
              f"{np.percentile(mae_col, 50):>9.2f} "
              f"{np.percentile(mae_col, 75):>9.2f}")
    print("=" * 72)

    return {
        "n_events":       n_events,
        "n_long":         n_long,
        "n_short":        n_short,
        "is_n":           is_n,
        "oos_n":          n_events - is_n,
        "median_spread":  med_sp,
        "samples_path":   str(samples_path),
        "meta_path":      str(meta_path),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--thr-sigma",      type=float, default=2.0,
                    help="|σ-norm| threshold for 5m/15m/1h gate (≥2 must exceed)")
    ap.add_argument("--thr-fast-sigma", type=float, default=2.0,
                    help="|σ-norm| threshold for the fast (M1/S5) confirm")
    ap.add_argument("--min-gap-bars",   type=int,   default=60,
                    help="Minimum S5 bars between consecutive events")
    ap.add_argument("--pre-bars",       type=int,   default=PRE_BARS_DEFAULT)
    ap.add_argument("--post-bars",      type=int,   default=POST_BARS_DEFAULT)
    ap.add_argument("--out-dir",        type=Path,  default=None)
    ap.add_argument("--data-path",      type=Path,  default=None)
    args = ap.parse_args()

    run(args.pair,
        thr_sigma=args.thr_sigma,
        thr_fast_sigma=args.thr_fast_sigma,
        min_gap_bars=args.min_gap_bars,
        pre_bars=args.pre_bars,
        post_bars=args.post_bars,
        out_dir=args.out_dir,
        data_path=args.data_path)


if __name__ == "__main__":
    main()
