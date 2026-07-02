#!/usr/bin/env python3
"""
escma_exit — CMA-ES + fixed-topology NN that learns to EXIT trades
==================================================================

Problem framing
---------------
A sibling chopper agent emits deterministic entries. We're handed a list of
sampled trades: each has direction d ∈ {+1, -1}, an entry price, a 60-bar
pre-entry CONTEXT window (15 features/bar), and an up-to-17,280-bar post-entry
S5 price path. Our job is to learn an EXIT signal that maximises AMDDP5 reward
PER SAMPLE (a single-trade simulation; the agent re-fires entries elsewhere).

Architecture (273 CMA params)
-----------------------------
Inputs (32):
   8  RFF-projected context (Tancik 2020 random Fourier features of 900-d
       pre-entry window). Fixed projection — NOT in CMA gene.
  15  Live current-bar features (the same 15 columns the chopper agent uses).
   9  MuZero position-state features (the network SEES its own trade — causal,
       computed from the trade's own past trajectory only). Ported from
       muzero_asi_mc/asi_mc_env_v4.py POSITION_DIM=9:
         [0] tanh(u / 20.0)                    bounded unrealized pnl
         [1] float(direction)                  position side -1/0/+1
         [2] hold_frac = t / t_max             trade age normalized by sim t_max
         [3] u / 100.0                          linear unrealized pnl
         [4] (e_px - mid[t]) / mid[t]           entry_price_relative
         [5] max_dd_pips / 50.0                 worst drawdown so far
         [6] acc_dd_pips_so_far / 100.0         running Σ max(0,-u) (mirrors AMDDP)
         [7] bars_since_max_dd / 60.0           bars since the worst point
         [8] risk_flags = 1.0 if max_dd>20 else 0.0   composite risk flag

Hidden: 8 nodes, all sin activation (cma_5in winner).
Output: 1 scalar via tanh → >0 = EXIT, ≤0 = HOLD.

Gene layout (273):
  [  0:256]  W1   (8 hidden × 32 inputs)
  [256:264]  b1   (8 biases)
  [264:272]  W2   (1 output × 8 hidden)
  [272:273]  b2   (1 output bias)

Per-sample simulation
---------------------
For each sample:
  • Walk forward t=0..T_max bar by bar.
  • At each bar compute the 15-d live feature + 9-d position state, build the
    32-d input, forward
    pass → exit_signal.
  • Track running unrealized PnL in pips (mid-based) and cum_dd (Σ |neg_pnl| ×
    bar_minutes) — matches scorer.py units.
  • Exit if:
      (a) exit_signal > 0.5                          → network_exit
      (b) pnl_pips < -sl_cap                          → disaster_sl
      (c) pnl_pips > +tp_cap                          → runaway_tp
      (d) hold == T_max                               → time_cap
  • On close: trade reward = pnl_pips − (λ_pct/100) × cum_dd.

Aggregate fitness = Σ trade reward across the IS batch. Maximise.

CMA-ES
------
  σ0=0.3, popsize=4+3·log(273)≈21, IPOP restart after 10 stagnant gens (gen≥50).
  bounds [-3, +3] on weights.

Data contract
-------------
samples_<PAIR>.parquet  (long table, one row per (sample_id, t_offset))
    sample_id : int
    t_offset  : int  (-60..-1 = context window, 0..T_max-1 = live path)
    mid       : float32
    bid       : float32  (only used at t_offset >= 0)
    ask       : float32  (only used at t_offset >= 0)
    f0..f14   : float32  (the 15 features per bar)

meta_<PAIR>.parquet (one row per sample)
    sample_id : int
    direction : int8   (+1 or -1)
    e_px      : float32  (entry execution price — already includes spread cost)
    pip       : float32
    sl_cap    : float32  (default 100p)
    tp_cap    : float32  (default 200p)
    t_max     : int      (typically 17280)
    split     : str     ("IS" or "OOS")

When --mock is passed, an in-process synthetic generator stands in for these
parquets so the trainer can be smoke-tested end-to-end.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import cma  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))  # for `import entry_chopper` (chopper kernel)

RESULTS_DIR = SCRIPT_DIR / "results"

# ── Architecture constants ────────────────────────────────────────────────
N_CTX_BARS   = 60        # pre-entry context window length
N_FEAT       = 15        # features per bar
N_CTX_RAW    = N_CTX_BARS * N_FEAT  # 900
N_RFF        = 8         # random Fourier features (projected context dim)
N_LIVE       = N_FEAT    # 15 live features at the current bar
N_POS        = 3         # MINIMAL position-state (2026-06-12, per user — keep CMA search
                         # space tight on a fragile edge): signed P/L, hold-frac, acc pain.
                         # (Full 9-feature MuZero set deferred; add only if minimal earns it.)
N_IN         = N_RFF + N_LIVE + N_POS   # 26
N_HID        = 8
N_OUT        = 1

# Offset where the position-feature block starts in the input vector x
POS_OFF = N_RFF + N_LIVE   # 23

W1_END  = N_IN * N_HID                       # 256
B1_END  = W1_END + N_HID                     # 264
W2_END  = B1_END + N_HID * N_OUT             # 272
B2_END  = W2_END + N_OUT                     # 273
N_PARAMS = B2_END                             # 273

# Exit interpretation
EXIT_THRESHOLD = 0.5
# AMDDP5 default (matches scorer.py:AMDDP_K)
AMDDP_K_DEFAULT = 0.05

# S5 cadence: 12 bars per minute → 1/12 minute per bar
S5_BAR_MINUTES = 1.0 / 12.0

# Exit cause codes (kept small for nopython)
CAUSE_NET   = 0
CAUSE_SL    = 1
CAUSE_TP    = 2
CAUSE_TIME  = 3


# ── Random Fourier Features projection (Tancik 2020) ─────────────────────
# Deterministic, NOT in the CMA gene. The same seed gives the same projection
# at training time and serving time — see save() / load() below.
def make_rff(seed: int, in_dim: int = N_CTX_RAW, out_dim: int = N_RFF,
             sigma: float = 1.0):
    """Returns (W_proj, b_proj) for x → cos(W·x + b), W ~ N(0, sigma²),
    b ~ U[0, 2π).  Output dim is `out_dim`."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * sigma
    b = rng.uniform(0.0, 2.0 * math.pi, size=out_dim).astype(np.float32)
    return W, b


@njit(cache=True)
def project_rff(ctx_flat, W_proj, b_proj):
    """ctx_flat : (N_CTX_RAW,) float32 — pre-entry window flattened
       returns  : (N_RFF,) float32 — cos(W·x + b)
    """
    out = np.empty(N_RFF, dtype=np.float32)
    for j in range(N_RFF):
        z = b_proj[j]
        for k in range(N_CTX_RAW):
            z += W_proj[j, k] * ctx_flat[k]
        out[j] = np.cos(z)
    return out


# ── Per-sample trade simulator (Numba JIT) ────────────────────────────────
@njit(cache=True)
def simulate_one(
    rff_ctx,        # (N_RFF,) precomputed
    live_feats,     # (T, N_FEAT) per-bar live features
    mid,            # (T,) mid prices
    bid,            # (T,)
    ask,            # (T,)
    direction,      # +1 or -1 int
    e_px,           # entry execution price (already includes spread)
    pip,
    sl_cap,
    tp_cap,
    t_max,
    weights,
    lambda_k,
    exit_threshold,
    bar_minutes,
):
    """Walk forward, fire NN at each bar, exit on first trigger.
    Returns:
        pnl_pips, cum_dd, hold_bars, exit_cause, exit_t (int)
    """
    T = live_feats.shape[0]
    if t_max > T:
        t_max = T

    cum_dd = 0.0
    pnl_pips = 0.0
    exit_cause = CAUSE_TIME
    exit_t = t_max - 1
    last_mid = mid[t_max - 1] if t_max > 0 else e_px

    # ── MuZero position-state running trackers (causal: trade trajectory only) ──
    peak_pnl = 0.0           # running peak unrealized pnl
    max_dd = 0.0             # worst drawdown so far (peak_pnl - u)
    acc_dd_pips = 0.0        # running Σ max(0, -u) — mirrors AMDDP penalty (pre-scale)
    bars_since_max_dd = 0    # bars since the worst (max_dd) point was set

    # Buffers for forward pass
    x = np.zeros(N_IN, dtype=np.float32)
    h = np.zeros(N_HID, dtype=np.float32)

    # Stuff the RFF half once — it doesn't change
    for j in range(N_RFF):
        x[j] = rff_ctx[j]

    # t_max for hold_frac normalization (avoid div-by-zero)
    t_max_f = float(t_max) if t_max > 0 else 1.0

    for t in range(t_max):
        # ── Running unrealized PnL (mid-based, signed by direction) ──
        u = direction * (mid[t] - e_px) / pip
        if u < 0.0:
            cum_dd += -u * bar_minutes

        # ── MuZero position-state trackers (update each bar after u) ──
        if u > peak_pnl:
            peak_pnl = u
        dd_now = peak_pnl - u            # current drawdown from peak (>= 0)
        if dd_now > max_dd:
            max_dd = dd_now
            bars_since_max_dd = 0
        else:
            bars_since_max_dd += 1
        if u < 0.0:
            acc_dd_pips += -u            # Σ max(0, -u), pre ×bar_minutes

        # ── Disaster SL (broker outer) ──
        if u < -sl_cap:
            # Realized at this bar's worst-side (bid for long, ask for short)
            exit_px = bid[t] if direction == 1 else ask[t]
            pnl_pips = direction * (exit_px - e_px) / pip
            exit_cause = CAUSE_SL
            exit_t = t
            return pnl_pips, cum_dd, t + 1, exit_cause, exit_t

        # ── Runaway TP (broker outer) ──
        if u > tp_cap:
            exit_px = bid[t] if direction == 1 else ask[t]
            pnl_pips = direction * (exit_px - e_px) / pip
            exit_cause = CAUSE_TP
            exit_t = t
            return pnl_pips, cum_dd, t + 1, exit_cause, exit_t

        # ── Build live-feature input slice ──
        for k in range(N_FEAT):
            x[N_RFF + k] = live_feats[t, k]

        # ── MINIMAL position-state (the net SEES its own trade) ──
        # signed u already encodes direction → no separate side feature needed.
        x[POS_OFF + 0] = np.tanh(u / 20.0)                 # signed unrealized P/L (bounded)
        x[POS_OFF + 1] = float(t) / t_max_f                # hold_frac in [0,1)
        x[POS_OFF + 2] = acc_dd_pips / 100.0               # running accumulated pain (mirrors AMDDP)

        # ── Forward pass: 26 → 8 (sin) → 1 (tanh) ──
        for j in range(N_HID):
            z = weights[W1_END + j]   # bias b1[j]
            base = j * N_IN
            for k in range(N_IN):
                z += weights[base + k] * x[k]
            h[j] = np.sin(z)

        out_z = weights[W2_END]   # bias b2[0]
        for j in range(N_HID):
            out_z += weights[B1_END + j] * h[j]
        exit_signal = np.tanh(out_z)

        # ── Network exit ──
        if exit_signal > exit_threshold:
            exit_px = bid[t] if direction == 1 else ask[t]
            pnl_pips = direction * (exit_px - e_px) / pip
            exit_cause = CAUSE_NET
            exit_t = t
            return pnl_pips, cum_dd, t + 1, exit_cause, exit_t

        last_mid = mid[t]

    # ── Time cap: force-close at last bar's worst-side ──
    if t_max > 0:
        exit_px = bid[t_max - 1] if direction == 1 else ask[t_max - 1]
        pnl_pips = direction * (exit_px - e_px) / pip
    return pnl_pips, cum_dd, t_max, exit_cause, exit_t


# ── Trivial baseline simulators (for OOS comparison) ─────────────────────
@njit(cache=True)
def simulate_baseline_tp20(mid, bid, ask, direction, e_px, pip,
                           sl_cap, tp_cap, t_max, bar_minutes):
    """Exit at first bar where unrealized pnl >= +20 pips. If never, exit at
    time cap. SL/TP outer caps still apply."""
    if t_max > mid.shape[0]:
        t_max = mid.shape[0]
    cum_dd = 0.0
    for t in range(t_max):
        u = direction * (mid[t] - e_px) / pip
        if u < 0.0:
            cum_dd += -u * bar_minutes
        if u < -sl_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_SL
        if u > tp_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_TP
        if u >= 20.0:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_NET
    if t_max > 0:
        ex = bid[t_max - 1] if direction == 1 else ask[t_max - 1]
        return direction * (ex - e_px) / pip, cum_dd, t_max, CAUSE_TIME
    return 0.0, 0.0, 0, CAUSE_TIME


@njit(cache=True)
def simulate_baseline_firstneg(mid, bid, ask, direction, e_px, pip,
                               sl_cap, tp_cap, t_max, bar_minutes):
    """Exit at first negative tick."""
    if t_max > mid.shape[0]:
        t_max = mid.shape[0]
    cum_dd = 0.0
    for t in range(t_max):
        u = direction * (mid[t] - e_px) / pip
        if u < 0.0:
            cum_dd += -u * bar_minutes
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_NET
        if u > tp_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_TP
    if t_max > 0:
        ex = bid[t_max - 1] if direction == 1 else ask[t_max - 1]
        return direction * (ex - e_px) / pip, cum_dd, t_max, CAUSE_TIME
    return 0.0, 0.0, 0, CAUSE_TIME


@njit(cache=True)
def simulate_baseline_hold(mid, bid, ask, direction, e_px, pip,
                           sl_cap, tp_cap, t_max, bar_minutes):
    """Hold to time cap (SL/TP outer caps still apply)."""
    if t_max > mid.shape[0]:
        t_max = mid.shape[0]
    cum_dd = 0.0
    for t in range(t_max):
        u = direction * (mid[t] - e_px) / pip
        if u < 0.0:
            cum_dd += -u * bar_minutes
        if u < -sl_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_SL
        if u > tp_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_TP
    if t_max > 0:
        ex = bid[t_max - 1] if direction == 1 else ask[t_max - 1]
        return direction * (ex - e_px) / pip, cum_dd, t_max, CAUSE_TIME
    return 0.0, 0.0, 0, CAUSE_TIME


# ── Sample container & loaders ────────────────────────────────────────────
class SampleBatch:
    """In-memory batch ready for the Numba kernel.
    All arrays are stacked into 2-D Numpy arrays where the leading axis is the
    sample index. We DO NOT need ragged storage because every sample uses the
    same t_max (the path may be shorter than t_max for some samples — handled
    via a per-sample `t_actual` cap)."""

    def __init__(self, sample_ids, rff_ctx, live_feats, mid, bid, ask,
                 direction, e_px, pip, sl_cap, tp_cap, t_actual, split):
        self.sample_ids = sample_ids        # (N,)
        self.rff_ctx = rff_ctx              # (N, N_RFF)
        self.live_feats = live_feats        # (N, T_max, N_FEAT)
        self.mid = mid                      # (N, T_max)
        self.bid = bid
        self.ask = ask
        self.direction = direction          # (N,)
        self.e_px = e_px                    # (N,)
        self.pip = pip                      # (N,)
        self.sl_cap = sl_cap                # (N,)
        self.tp_cap = tp_cap                # (N,)
        self.t_actual = t_actual            # (N,) per-sample valid path length
        self.split = split                  # (N,) str array

    def filter(self, mask):
        return SampleBatch(
            self.sample_ids[mask],
            self.rff_ctx[mask],
            self.live_feats[mask],
            self.mid[mask],
            self.bid[mask],
            self.ask[mask],
            self.direction[mask],
            self.e_px[mask],
            self.pip[mask],
            self.sl_cap[mask],
            self.tp_cap[mask],
            self.t_actual[mask],
            self.split[mask],
        )


def _build_rff_batch(W_proj, b_proj, ctx_windows):
    """ctx_windows: (N, N_CTX_BARS, N_FEAT) → (N, N_RFF)"""
    N = ctx_windows.shape[0]
    out = np.empty((N, N_RFF), dtype=np.float32)
    for i in range(N):
        ctx_flat = ctx_windows[i].reshape(-1).astype(np.float32)
        out[i] = project_rff(ctx_flat, W_proj, b_proj)
    return out


# Column order for the 15-feature vector — MUST match entry_chopper output and
# the chopper kernel's return order (mom 5-stack then mn 5-stack then spread).
# samples_<PAIR>.parquet column order:
PRE_FEAT_COLS = [
    "open", "high", "low", "close",
    "mom_S5", "mom_M1", "mom_5m", "mom_15m", "mom_1h",
    "mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h",
    "spread_pips",
]

# Pre-computed feature parquet cache (avoid re-reading the 21M-row file per call)
_FEATURES_CACHE: dict = {}


def _pip_for(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


# Feature parquet column order for the 15-vector assembly.
# MUST match PRE_FEAT_COLS so the RFF projection + simulate_one see the same
# layout the chopper's pre-context used:
#   [open, high, low, close, mom_S5, mom_M1, mom_5m, mom_15m, mom_1h,
#    mn_S5, mn_M1, mn_5m, mn_15m, mn_1h, spread_pips]
_FEAT15_COLS = PRE_FEAT_COLS  # alias for clarity


def _load_features(pair: str, data_dir: Path):
    """Load features_<PAIR>.parquet ONCE into float32 arrays (~1.3 GB), keyed
    by pair. Pure aligned feature store — NO computation. The index into these
    arrays IS the source bar_idx (rows are dense 0..n-1).

    Returns a dict of column_name -> float32 ndarray plus 'bid_c','ask_c'.
    """
    key = (pair, str(data_dir))
    if key in _FEATURES_CACHE:
        return _FEATURES_CACHE[key]
    import pyarrow.parquet as pq
    feat_path = data_dir / f"features_{pair}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"features parquet missing: {feat_path}\n"
            f"Run: python3 precompute_features.py --pair {pair}")
    cols = (["bar_idx", "bid_c", "ask_c", "spread_pips"]
            + ["mom_S5", "mom_M1", "mom_5m", "mom_15m", "mom_1h",
               "mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h"]
            + ["open", "high", "low", "close"])
    tbl = pq.read_table(feat_path, columns=cols)
    store = {}
    for c in cols:
        arr = tbl.column(c).to_numpy(zero_copy_only=False)
        if c == "bar_idx":
            store[c] = arr.astype(np.int64, copy=False)
        else:
            store[c] = arr.astype(np.float32, copy=False)
    # Sanity: rows must be dense & aligned (bar_idx[i] == i)
    bi = store["bar_idx"]
    if bi[0] != 0 or bi[-1] != len(bi) - 1:
        raise ValueError(
            f"features parquet bar_idx not dense 0..n-1 "
            f"(first={bi[0]} last={bi[-1]} n={len(bi)})")
    _FEATURES_CACHE[key] = store
    return store


def load_real(pair: str, data_dir: Path, W_proj, b_proj,
              t_max_alloc: int = 2880, meta3_name: str = None) -> SampleBatch:
    """Pure index-slicing loader (no feature recomputation anywhere).

    Loads features_<PAIR>.parquet ONCE (aligned, bar_idx == row) and
    meta3_<PAIR>.parquet (three indices per event), then slices feature rows
    by index:
        context    = rows [t_pre   : t_event]   (720 bars) → RFF projection
        trade path = rows [t_event : t_timeout] → per-bar 15-feature live vectors

    The 15-feature column order is PRE_FEAT_COLS (unchanged): open, high, low,
    close, mom_S5..mom_1h, mn_S5..mn_1h, spread_pips. mid=close, bid=bid_c,
    ask=ask_c. NO `_compute_momentum_stack` call — features are read, not built.

    Memory: a dense (N, 17280, 15) live_feats would be ~10 GB. We CAP the dense
    allocation at `t_max_alloc` (default 2880 = 4 h). Any trade still open at
    t_max_alloc is force-closed by the sim's time-cap (t_actual). The broker
    SL/TP outer caps in simulate_one handle disaster cases regardless of the cap.
    """
    meta3_path = data_dir / (meta3_name if meta3_name else f"meta3_{pair}.parquet")
    if not meta3_path.exists():
        raise FileNotFoundError(
            f"meta3 missing: {meta3_path}\n"
            f"Run: python3 rebuild_meta.py --pair {pair}")

    meta3 = pd.read_parquet(meta3_path).reset_index(drop=True)
    N = len(meta3)
    pip_val = _pip_for(pair)

    # ── Aligned feature store (loaded ONCE, pure slicing thereafter) ──
    feat = _load_features(pair, data_dir)
    n_src = feat["close"].shape[0]

    t_pre = meta3["t_pre"].values.astype(np.int64)
    t_event = meta3["t_event"].values.astype(np.int64)
    t_timeout = meta3["t_timeout"].values.astype(np.int64)

    # Global t_max: cap at min(17280, max post-entry length, t_max_alloc)
    post_len = (t_timeout - t_event).astype(np.int64)
    t_max_post = int(min(17280, int(post_len.max())))
    t_max_global = int(min(t_max_post, t_max_alloc))

    # Pre-fetch the 15 feature column arrays in PRE_FEAT_COLS order (slicing only)
    feat15 = [feat[c] for c in _FEAT15_COLS]  # list of 15 aligned f32 arrays
    bid_arr = feat["bid_c"]
    ask_arr = feat["ask_c"]
    close_arr = feat["close"]

    # ── Allocate dense buffers (capped) ──
    rff_ctx_arr = np.zeros((N, N_RFF), dtype=np.float32)
    live_feats = np.zeros((N, t_max_global, N_FEAT), dtype=np.float32)
    mid = np.zeros((N, t_max_global), dtype=np.float32)
    bid = np.zeros((N, t_max_global), dtype=np.float32)
    ask = np.zeros((N, t_max_global), dtype=np.float32)
    t_actual = np.zeros(N, dtype=np.int64)

    # Entry execution price: ask_c at event for long, bid_c for short (R3).
    direction = meta3["direction"].values.astype(np.int64)
    e_px = np.where(direction == 1, ask_arr[t_event], bid_arr[t_event]).astype(np.float32)

    n_hit_cap = 0
    for row_idx in range(N):
        tp = int(t_pre[row_idx])
        te = int(t_event[row_idx])
        tt = int(t_timeout[row_idx])

        # --- pre-entry RFF context: rows [t_pre : t_event] (720 bars), keep
        #     last N_CTX_BARS × 15 feats = 900-d, identical to original RFF feed ---
        ctx_lo = max(tp, te - N_CTX_BARS)
        if ctx_lo < te:
            ctx_window = np.empty((te - ctx_lo, N_FEAT), dtype=np.float32)
            for fi, col in enumerate(feat15):
                ctx_window[:, fi] = col[ctx_lo:te]
            np.nan_to_num(ctx_window, copy=False)
            if ctx_window.shape[0] >= N_CTX_BARS:
                ctx_window = ctx_window[-N_CTX_BARS:]
                rff_ctx_arr[row_idx] = project_rff(
                    ctx_window.reshape(-1), W_proj, b_proj)

        # --- post-entry path: rows [t_event : t_timeout] ---
        npb = tt - te
        if npb > t_max_global:
            npb = t_max_global
            n_hit_cap += 1
        if npb <= 0:
            t_actual[row_idx] = 0
            continue
        hi = te + npb

        lf = live_feats[row_idx]
        for fi, col in enumerate(feat15):
            lf[:npb, fi] = col[te:hi]
        np.nan_to_num(lf[:npb], copy=False)

        mid[row_idx, :npb] = close_arr[te:hi]   # mid = close
        bid[row_idx, :npb] = bid_arr[te:hi]
        ask[row_idx, :npb] = ask_arr[te:hi]
        t_actual[row_idx] = npb

    if n_hit_cap > 0:
        print(f"  [load] {n_hit_cap}/{N} samples hit t_max_alloc={t_max_alloc} "
              f"cap (force-closed at {t_max_alloc} bars ≈ {t_max_alloc*5/3600:.1f}h)")

    # Per-trade caps (broker outer): same as before (100p SL / 200p TP).
    sl_cap = np.full(N, 100.0, dtype=np.float32)
    tp_cap = np.full(N, 200.0, dtype=np.float32)
    pip_arr = np.full(N, pip_val, dtype=np.float32)

    return SampleBatch(
        sample_ids=meta3["sample_id"].values.astype(np.int64),
        rff_ctx=rff_ctx_arr,
        live_feats=live_feats,
        mid=mid,
        bid=bid,
        ask=ask,
        direction=direction,
        e_px=e_px,
        pip=pip_arr,
        sl_cap=sl_cap,
        tp_cap=tp_cap,
        t_actual=t_actual,
        split=meta3["split"].values.astype(str),
    )


def make_mock(W_proj, b_proj, n_samples: int = 50, t_max: int = 1000,
              seed: int = 42, pip: float = 0.01, sl_cap: float = 100.0,
              tp_cap: float = 200.0) -> SampleBatch:
    """Synthetic dataset for smoke testing.

    Two regimes are intentionally embedded so the NN has SOMETHING to learn:
      • Half the samples have a drifting random walk that mostly winners over
        the first 200 bars (the "early exit pays" cohort).
      • Other half are mean-reverting around entry (the "hold for the snap-back"
        cohort).
    The 15-d live features include a noisy regime indicator (f0) so the NN
    can in principle distinguish them.
    """
    rng = np.random.default_rng(seed)
    N = n_samples
    T = t_max

    rff_ctx_arr = np.zeros((N, N_RFF), dtype=np.float32)
    live_feats = np.zeros((N, T, N_FEAT), dtype=np.float32)
    mid = np.zeros((N, T), dtype=np.float32)
    bid = np.zeros((N, T), dtype=np.float32)
    ask = np.zeros((N, T), dtype=np.float32)
    direction = np.zeros(N, dtype=np.int64)
    e_px = np.zeros(N, dtype=np.float32)
    splits = np.empty(N, dtype=object)

    spread_pips = 1.5

    for i in range(N):
        d = 1 if rng.random() < 0.5 else -1
        direction[i] = d
        regime = 0 if i < N // 2 else 1
        # Entry around 150.000 (USDJPY-ish)
        e_mid = 150.000 + rng.normal(0, 0.05)
        e_px[i] = e_mid + d * (spread_pips * pip / 2.0)

        # Context window — just noisy values; the embedded regime info is in
        # the live feed (f0).
        ctx = rng.normal(0, 0.5, size=(N_CTX_BARS, N_FEAT)).astype(np.float32)
        rff_ctx_arr[i] = project_rff(ctx.reshape(-1), W_proj, b_proj)

        # Live path
        if regime == 0:
            # Drift in favour for ~200 bars then random walk
            drift = d * 0.0008 * pip
            path = np.cumsum(rng.normal(drift, 0.0005, T)) + e_mid
            path[200:] = path[200] + np.cumsum(rng.normal(0, 0.0005, T - 200))
        else:
            # Mean revert
            path = np.empty(T, dtype=np.float64)
            path[0] = e_mid + d * 0.003   # immediately goes against you
            for t in range(1, T):
                path[t] = path[t - 1] + (e_mid - path[t - 1]) * 0.005 \
                          + rng.normal(0, 0.0005)

        mid[i] = path.astype(np.float32)
        sp = spread_pips * pip
        bid[i] = (mid[i] - sp / 2.0).astype(np.float32)
        ask[i] = (mid[i] + sp / 2.0).astype(np.float32)

        # Live features — f0 carries regime signal (noisy), rest pure noise
        live_feats[i] = rng.normal(0, 0.5, size=(T, N_FEAT)).astype(np.float32)
        live_feats[i, :, 0] += (regime * 2 - 1) * 0.6   # ±0.6 mean shift

        splits[i] = "IS" if i % 4 != 0 else "OOS"   # ~75/25

    return SampleBatch(
        sample_ids=np.arange(N, dtype=np.int64),
        rff_ctx=rff_ctx_arr,
        live_feats=live_feats,
        mid=mid,
        bid=bid,
        ask=ask,
        direction=direction,
        e_px=e_px,
        pip=np.full(N, pip, dtype=np.float32),
        sl_cap=np.full(N, sl_cap, dtype=np.float32),
        tp_cap=np.full(N, tp_cap, dtype=np.float32),
        t_actual=np.full(N, T, dtype=np.int64),
        split=splits.astype(str),
    )


# ── Batch fitness ────────────────────────────────────────────────────────
def simulate_batch(batch: SampleBatch, weights: np.ndarray,
                   lambda_pct: float, exit_threshold: float,
                   bar_minutes: float):
    """Run simulate_one over every sample in `batch`. Returns a results dict."""
    N = batch.rff_ctx.shape[0]
    pnl = np.empty(N, dtype=np.float64)
    cum_dd = np.empty(N, dtype=np.float64)
    hold = np.empty(N, dtype=np.int64)
    cause = np.empty(N, dtype=np.int64)
    lambda_k = lambda_pct / 100.0
    w64 = weights.astype(np.float64, copy=False)

    for i in range(N):
        p, dd, h, c, _ = simulate_one(
            batch.rff_ctx[i],
            batch.live_feats[i],
            batch.mid[i],
            batch.bid[i],
            batch.ask[i],
            int(batch.direction[i]),
            float(batch.e_px[i]),
            float(batch.pip[i]),
            float(batch.sl_cap[i]),
            float(batch.tp_cap[i]),
            int(batch.t_actual[i]),
            w64,
            lambda_k,
            float(exit_threshold),
            float(bar_minutes),
        )
        pnl[i] = p
        cum_dd[i] = dd
        hold[i] = h
        cause[i] = c

    amddp = pnl - lambda_k * cum_dd
    return {
        "pnl": pnl,
        "cum_dd": cum_dd,
        "hold": hold,
        "cause": cause,
        "amddp": amddp,
        "sum_amddp": float(amddp.sum()),
        "sum_pnl": float(pnl.sum()),
        "mean_hold": float(hold.mean()),
        "wr": float((pnl > 0).mean()),
        "n": int(N),
    }


def fitness_neg(weights, batch, lambda_pct, exit_threshold, bar_minutes):
    """CMA minimises → return negative AMDDP5 sum."""
    res = simulate_batch(batch, weights, lambda_pct, exit_threshold, bar_minutes)
    return -res["sum_amddp"]


# ── Baseline runners ─────────────────────────────────────────────────────
def run_baseline(batch: SampleBatch, kind: str, lambda_pct: float,
                 bar_minutes: float):
    N = batch.rff_ctx.shape[0]
    pnl = np.empty(N, dtype=np.float64)
    cum_dd = np.empty(N, dtype=np.float64)
    hold = np.empty(N, dtype=np.int64)
    cause = np.empty(N, dtype=np.int64)
    fn = {
        "tp20":     simulate_baseline_tp20,
        "firstneg": simulate_baseline_firstneg,
        "hold":     simulate_baseline_hold,
    }[kind]
    for i in range(N):
        p, dd, h, c = fn(
            batch.mid[i], batch.bid[i], batch.ask[i],
            int(batch.direction[i]), float(batch.e_px[i]),
            float(batch.pip[i]), float(batch.sl_cap[i]), float(batch.tp_cap[i]),
            int(batch.t_actual[i]), float(bar_minutes),
        )
        pnl[i] = p
        cum_dd[i] = dd
        hold[i] = h
        cause[i] = c
    k = lambda_pct / 100.0
    amddp = pnl - k * cum_dd
    return {
        "sum_amddp": float(amddp.sum()),
        "sum_pnl": float(pnl.sum()),
        "mean_hold": float(hold.mean()),
        "wr": float((pnl > 0).mean()),
        "cause_net": int((cause == CAUSE_NET).sum()),
        "cause_sl": int((cause == CAUSE_SL).sum()),
        "cause_tp": int((cause == CAUSE_TP).sum()),
        "cause_time": int((cause == CAUSE_TIME).sum()),
    }


# ── Reporting helpers ────────────────────────────────────────────────────
def cause_counts(arr):
    return {
        "network_exit": int((arr == CAUSE_NET).sum()),
        "disaster_sl":  int((arr == CAUSE_SL).sum()),
        "runaway_tp":   int((arr == CAUSE_TP).sum()),
        "time_cap":     int((arr == CAUSE_TIME).sum()),
    }


def fmt_res(res, lambda_pct):
    return (f"sum_amddp{int(lambda_pct)}={res['sum_amddp']:+.2f}p  "
            f"sum_pnl={res['sum_pnl']:+.2f}p  "
            f"mean_hold={res['mean_hold']:.0f}b  "
            f"WR={res['wr']:.1%}  n={res['n']}")


def full_oos_report(batch: SampleBatch, weights: np.ndarray,
                    exit_threshold: float, bar_minutes: float):
    print("\n" + "=" * 70)
    print("  OOS REPORT")
    print("=" * 70)
    res5 = simulate_batch(batch, weights, 5.0, exit_threshold, bar_minutes)
    res1 = simulate_batch(batch, weights, 1.0, exit_threshold, bar_minutes)
    res10 = simulate_batch(batch, weights, 10.0, exit_threshold, bar_minutes)
    print(f"  AMDDP1   sum = {res1['sum_amddp']:+8.2f}p   per-sample mean = "
          f"{res1['sum_amddp']/max(1, res1['n']):+.3f}p")
    print(f"  AMDDP5   sum = {res5['sum_amddp']:+8.2f}p   per-sample mean = "
          f"{res5['sum_amddp']/max(1, res5['n']):+.3f}p")
    print(f"  AMDDP10  sum = {res10['sum_amddp']:+8.2f}p   per-sample mean = "
          f"{res10['sum_amddp']/max(1, res10['n']):+.3f}p")
    print(f"  Raw PnL  sum = {res5['sum_pnl']:+8.2f}p")
    print(f"  Mean hold (S5 bars) = {res5['mean_hold']:.0f}  "
          f"(~{res5['mean_hold']*5/60:.1f} minutes)")
    print(f"  WR = {res5['wr']:.1%}")
    causes = cause_counts(res5["cause"])
    print(f"  Exit causes: {causes}")

    print("\n  Baselines (AMDDP5 sum on same OOS batch):")
    for kind, label in [("tp20", "Fixed +20p TP"),
                        ("firstneg", "First negative tick"),
                        ("hold", "Hold to time cap")]:
        b = run_baseline(batch, kind, 5.0, bar_minutes)
        print(f"    {label:22s}  amddp5={b['sum_amddp']:+8.2f}p  "
              f"pnl={b['sum_pnl']:+7.2f}p  WR={b['wr']:.1%}  "
              f"hold={b['mean_hold']:.0f}b  "
              f"causes net/sl/tp/time = "
              f"{b['cause_net']}/{b['cause_sl']}/{b['cause_tp']}/{b['cause_time']}")
    print("=" * 70)
    return res5


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="escma_exit — CMA-ES + sin-NN exit learner")
    parser.add_argument("--pair", default="USD_JPY")
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rff-seed", type=int, default=1337,
                        help="Seed for the (fixed) RFF projection.")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=0,
                        help="0 → use CMA-ES default 4+3·log(N)")
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--bound", type=float, default=3.0)
    parser.add_argument("--lambda-pct", type=float, default=5.0,
                        help="AMDDP penalty %. 5 → AMDDP5.")
    parser.add_argument("--exit-threshold", type=float, default=EXIT_THRESHOLD)
    parser.add_argument("--bar-minutes", type=float, default=S5_BAR_MINUTES,
                        help="Bar duration in minutes (1/12 for S5).")
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--stagnation-gens", type=int, default=10,
                        help="IPOP restart after this many gens w/o improvement (gen ≥ 50).")
    parser.add_argument("--mode", choices=["continuation", "fade"],
                        default="continuation",
                        help="continuation: trade meta['direction'] as-is "
                             "(long on up-spike). fade: trade -direction.")
    parser.add_argument("--meta3-name", default=None,
                        help="Override meta3 filename (e.g. meta3_USD_JPY_t3.parquet "
                             "for a tighter-gate variant). Default meta3_<PAIR>.parquet.")
    parser.add_argument("--t-max-alloc", type=int, default=2880,
                        help="Cap on dense post-entry bars allocated per sample "
                             "(2880 = 4h). Trades still open at this cap are "
                             "force-closed by the sim time-cap.")
    parser.add_argument("--mock", action="store_true",
                        help="Skip parquet load, use synthetic samples.")
    parser.add_argument("--mock-n", type=int, default=50)
    parser.add_argument("--mock-t", type=int, default=1000)
    parser.add_argument("--label", default="escma_exit")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print(f"  escma_exit — pair={args.pair}  mock={args.mock}")
    print(f"  Topology: {N_IN}→{N_HID}→{N_OUT}  ({N_PARAMS} CMA params)")
    print(f"    [ {N_RFF} RFF context + {N_LIVE} live + {N_POS} position ] "
          f"→ {N_HID} sin → tanh")
    print(f"  AMDDP λ = {args.lambda_pct}%  exit_threshold = "
          f"{args.exit_threshold}")
    print(f"  bar_minutes = {args.bar_minutes:.6f}")
    print(f"  Seed = {args.seed}  RFF seed = {args.rff_seed}")
    print("=" * 70)

    # ── Build RFF projection ───────────────────────────────
    W_proj, b_proj = make_rff(args.rff_seed)
    print(f"\nRFF projection: {W_proj.shape}  (deterministic, NOT in CMA gene)")

    # ── Load data ──────────────────────────────────────────
    if args.mock:
        print(f"\nGenerating MOCK dataset: n={args.mock_n}, T={args.mock_t}")
        batch = make_mock(W_proj, b_proj, n_samples=args.mock_n,
                          t_max=args.mock_t, seed=args.seed)
    else:
        data_dir = Path(args.data_dir)
        print(f"\nLoading real samples from {data_dir}")
        batch = load_real(args.pair, data_dir, W_proj, b_proj,
                          t_max_alloc=args.t_max_alloc,
                          meta3_name=args.meta3_name)

    # ── Direction mode (continuation vs fade) ──────────────
    print(f"\nDirection mode: {args.mode.upper()}", end="")
    if args.mode == "fade":
        batch.direction = -batch.direction
        print("  → trade -meta['direction'] (short up-spikes, long down-spikes)")
    else:
        print("  → trade meta['direction'] as-is (long up-spikes, short down-spikes)")

    is_mask = (batch.split == "IS")
    oos_mask = (batch.split == "OOS")
    is_batch = batch.filter(is_mask)
    oos_batch = batch.filter(oos_mask)
    print(f"\nDataset: total={batch.rff_ctx.shape[0]}  "
          f"IS={is_batch.rff_ctx.shape[0]}  OOS={oos_batch.rff_ctx.shape[0]}")
    if is_batch.rff_ctx.shape[0] == 0:
        print("ERROR: no IS samples")
        sys.exit(1)

    # ── JIT warmup ─────────────────────────────────────────
    print("\nJIT warming up...")
    warm = np.zeros(N_PARAMS, dtype=np.float64)
    _ = simulate_batch(is_batch, warm, args.lambda_pct,
                       args.exit_threshold, args.bar_minutes)
    _ = run_baseline(is_batch, "hold", args.lambda_pct, args.bar_minutes)
    print("  warm.")

    # ── CMA-ES setup ───────────────────────────────────────
    popsize = args.popsize if args.popsize > 0 else (
        4 + int(3 * math.log(N_PARAMS)))   # ~22 for 201 params
    print(f"\nCMA-ES popsize = {popsize}  σ0 = {args.sigma0}  "
          f"bounds = ±{args.bound}")

    x0 = np.random.randn(N_PARAMS) * 0.1
    opts = {
        "popsize": popsize,
        "seed": args.seed,
        "verbose": -9,
        "bounds": [[-args.bound] * N_PARAMS, [args.bound] * N_PARAMS],
        "tolx": 1e-9,
        "tolfun": 1e-3,
        "tolfunhist": 1e-3,
        "tolflatfitness": 50,
        "maxiter": args.gens,
    }
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.label}_{args.pair}_s{args.seed}"
    if args.mock:
        tag += "_mock"

    best_fit = float("inf")
    best_vec = None
    last_improve_gen = 0
    restarts = 0
    t0 = time.time()

    gen = 0
    while not es.stop():
        candidates = es.ask()
        fitnesses = [
            fitness_neg(np.asarray(c), is_batch, args.lambda_pct,
                        args.exit_threshold, args.bar_minutes)
            for c in candidates
        ]
        es.tell(candidates, fitnesses)

        gen_min = min(fitnesses)
        if gen_min < best_fit - 1e-6:
            best_fit = gen_min
            best_vec = np.asarray(candidates[fitnesses.index(gen_min)])
            last_improve_gen = gen

        if gen % args.report_every == 0:
            res = simulate_batch(is_batch, best_vec if best_vec is not None
                                  else np.asarray(candidates[0]),
                                  args.lambda_pct, args.exit_threshold,
                                  args.bar_minutes)
            elapsed = time.time() - t0
            print(f"  Gen {gen:>3}  raw_fit={best_fit:>+9.2f}  "
                  f"IS_amddp{int(args.lambda_pct)}={res['sum_amddp']:>+8.2f}p  "
                  f"hold={res['mean_hold']:>5.0f}b  WR={res['wr']:.1%}  "
                  f"σ={es.sigma:.4f}  t={elapsed:.0f}s")

        # IPOP restart
        if gen >= 50 and gen - last_improve_gen >= args.stagnation_gens:
            restarts += 1
            new_pop = popsize * (2 ** restarts)
            print(f"  [Gen {gen}] STAGNATION — IPOP restart #{restarts}, "
                  f"popsize → {new_pop}")
            x_restart = (best_vec if best_vec is not None else x0) + \
                np.random.randn(N_PARAMS) * args.sigma0
            opts_r = dict(opts)
            opts_r["popsize"] = new_pop
            opts_r["seed"] = args.seed + restarts
            opts_r["maxiter"] = max(1, args.gens - gen)
            es = cma.CMAEvolutionStrategy(x_restart, args.sigma0, opts_r)
            popsize = new_pop
            last_improve_gen = gen

        gen += 1
        if gen >= args.gens:
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete: {gen} gens, {elapsed:.0f}s, "
          f"best_raw_fit={best_fit:.4f}")
    if best_vec is None:
        print("ERROR: CMA-ES produced no candidates.")
        sys.exit(2)

    # ── Final IS + OOS reports ─────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL IS REPORT (best genome)")
    print("=" * 70)
    is_res = simulate_batch(is_batch, best_vec, args.lambda_pct,
                            args.exit_threshold, args.bar_minutes)
    print("  IS  " + fmt_res(is_res, args.lambda_pct))
    print(f"  IS  exit causes: {cause_counts(is_res['cause'])}")

    if oos_batch.rff_ctx.shape[0] > 0:
        full_oos_report(oos_batch, best_vec, args.exit_threshold,
                        args.bar_minutes)
    else:
        print("\n(No OOS samples — skipping OOS report.)")

    # ── Save final genome ──────────────────────────────────
    out_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "weights": best_vec,
            "n_in": N_IN, "n_hid": N_HID, "n_out": N_OUT,
            "n_params": N_PARAMS,
            "rff_W": W_proj, "rff_b": b_proj, "rff_seed": args.rff_seed,
            "n_ctx_bars": N_CTX_BARS, "n_feat": N_FEAT, "n_rff": N_RFF,
            "n_pos": N_POS, "pos_off": POS_OFF,
            "exit_threshold": args.exit_threshold,
            "lambda_pct": args.lambda_pct,
            "bar_minutes": args.bar_minutes,
            "pair": args.pair,
            "seed": args.seed,
            "best_fitness": float(best_fit),
            "is_summary": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                           for k, v in is_res.items() if k != "amddp"},
            "args": vars(args),
        }, f)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
