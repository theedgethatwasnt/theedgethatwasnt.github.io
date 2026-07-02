"""7-input M5 test with regime context + H1 slope + latching entry, CMA-ES optimized.

Inputs (all causal):
  [0] tec5:     signed Kaufman ER over 5 M5 bars → sign(net)×(|net|/path). [-1,+1]
  [1] trending: 1 if |EMA24_slope_on_M5| > 1wk rolling median, else 0
  [2] high_vol: 1 if ATR14_on_M5 > 1wk rolling median, else 0
  [3] h1_slope: arctan((H1_close[t]-H1_close[t-2])/2 / pip / 20) / (π/2) ∈ [-1,+1]
                computed on H1 closes, shifted +1h, ffilled to M5 (strictly causal)
  [4] upnl:     tanh(upnl_pips / 10)
  [5] mae:      tanh(mae_pips / 10)  (running max adverse excursion, init = spread)
  [6] mfe:      tanh(mfe_pips / 10)

Topology (FIXED): 6 → 1 → 3 (BUY / SELL / FLATTEN).
Decision rule:
  - Flat → Long:  BUY > FLATTEN + θ  AND  BUY > SELL + θ
  - Flat → Short: SELL > FLATTEN + θ AND  SELL > BUY + θ
  - In position: argmax exit (no θ) — can close to flat or reverse

Activation bank (evolved via 1 gene, bucketed into 3): {tanh, sin, gauss}
θ evolved as the 15th gene, decoded via sigmoid into [0, 0.5].

Gene layout (15 params total):
  [ 0:6 ]  W1   (6 inputs → 1 hidden)
  [ 6   ]  b1   (hidden bias)
  [ 7:10]  W2   (1 hidden → 3 outputs)
  [10:13]  b2   (output biases)
  [13   ]  act_gene   → activation bucket {tanh, sin, gauss}
  [14   ]  theta_gene → σ(gene) * 0.5 ∈ [0, 0.5]

Fitness = amddp1: score_per_day = (total_pnl − 0.01 × cum_mae) / days
  cum_mae = Σ over every in-position bar of the running MAE at that bar (integrated DD).
Keeps fx-core SOP: pips/day base, bidir asym_penalty, WF-in-fitness bonus branch,
spread charged at entry (MAE init = spread_pips), no MAE-denominator.

Usage:
    python3 test_dclose3_regime_cma.py --pair EUR_JPY --seed 42 --gens 200
"""
import argparse
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
import cma

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)

PAIR_PIP = {"EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
            "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
            "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
            "NZD_USD": 0.0001, "EUR_GBP": 0.0001}
PAIR_SPREAD = {"EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
               "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
               "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
               "NZD_USD": 1.5, "EUR_GBP": 1.4}

N_IN = 7
N_OUT = 3
N_ACTS = 3   # {tanh, sin, gauss}

def n_params(n_hid: int, use_skip: bool) -> int:
    """W1 + b1 + W2 + [Wskip if use_skip] + b2 + act_genes (per hidden) + theta."""
    p = N_IN * n_hid + n_hid + n_hid * N_OUT + N_OUT + n_hid + 1
    if use_skip:
        p += N_IN * N_OUT
    return p

MAX_HOLD = 200        # M5 bars (≈16.7 h)
BARS_PER_DAY = 288.0
MIN_DIR = 0.15
N_CHUNKS = 3
AMDDP_COEF = 0.01


# ══════════════════════════════════════════════════════════════════════
# Causal features
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def compute_tec5(closes, n):
    """Signed Kaufman ER over 5 bars: sign(net) × (|net|/path). Causal, [-1,+1]."""
    out = np.zeros(n)
    for i in range(5, n):
        net = closes[i] - closes[i - 5]
        path = 0.0
        for k in range(i - 4, i + 1):
            path += abs(closes[k] - closes[k - 1])
        if path > 1e-12:
            er = abs(net) / path
            if net > 0:
                out[i] = er
            elif net < 0:
                out[i] = -er
    return out


@njit(cache=True)
def _wilder_atr(h, l, c, period=14):
    """Wilder ATR on arrays (H, L, C) — recursive RMA."""
    n = len(c)
    out = np.zeros(n)
    if n < 2:
        return out
    # True range at bar 0 = high - low (no prev close)
    tr0 = h[0] - l[0]
    tr_sum = tr0
    for i in range(1, period):
        if i >= n:
            break
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr = hl
        if hc > tr: tr = hc
        if lc > tr: tr = lc
        tr_sum += tr
    if period - 1 < n:
        out[period - 1] = tr_sum / period
    # Wilder smoothing thereafter
    for i in range(period, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr = hl
        if hc > tr: tr = hc
        if lc > tr: tr = lc
        out[i] = (out[i - 1] * (period - 1) + tr) / period
    return out


@njit(cache=True)
def _ema_slope(closes, n, span=24, lookback=12):
    """Slope of EMA(span) over `lookback` bars, per bar. Uses only past data."""
    alpha = 2.0 / (span + 1)
    ema = np.zeros(n)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
    slope = np.zeros(n)
    for i in range(lookback, n):
        slope[i] = (ema[i] - ema[i - lookback]) / lookback
    return slope


def build_regime_m5(df_m5: pd.DataFrame, lookback_bars=2016):
    """Pure-M5 regime: ATR14 + EMA24_slope computed directly on M5.
    lookback_bars=2016 M5 bars = 1 week (288 bars/day × 7).
    trending: |EMA24_slope_M5| > 1wk rolling median
    high_vol: ATR14_M5 > 1wk rolling median (percentile > 0.5)

    Causal by construction: every rolling op uses only past+current bars.
    """
    high = df_m5["high"].values.astype(np.float64)
    low = df_m5["low"].values.astype(np.float64)
    close = df_m5["close"].values.astype(np.float64)
    n = len(close)

    atr = _wilder_atr(high, low, close, period=14)
    slope = _ema_slope(close, n, span=24, lookback=12)

    atr_s = pd.Series(atr)
    slope_abs = pd.Series(np.abs(slope))

    atr_rank = atr_s.rolling(lookback_bars, min_periods=1).rank(pct=True).values
    slope_med = slope_abs.rolling(lookback_bars, min_periods=1).median().values

    high_vol = (atr_rank > 0.5).astype(np.float64)
    trending = (np.abs(slope) > slope_med).astype(np.float64)
    return trending, high_vol


def build_h1_slope_m5(df_m5: pd.DataFrame, pip: float) -> np.ndarray:
    """Compute 3-bar H1 linreg slope on H1 close, arctan-normalized, ffilled to M5.

    3-bar linreg slope for y0,y1,y2 = (y2 - y0) / 2 (centered finite diff).
    Normalize by pip then arctan(slope/20)/(π/2) → [-1, +1].
    Shift +1 H1 bar before ffill so each M5 bar sees regime from strictly-closed H1.
    Causal by construction.
    """
    df = df_m5.set_index("timestamp")
    h1_close = df["close"].resample("1h").last().dropna()
    # 3-bar slope = (close[t] - close[t-2]) / 2
    slope_h1 = (h1_close - h1_close.shift(2)) / 2.0
    slope_pips = slope_h1 / pip
    h1_slope_norm = np.arctan(slope_pips / 20.0) / (np.pi / 2)
    # Shift +1h for causality
    h1_slope_norm = h1_slope_norm.shift(1)
    # ffill to M5 index
    m5_slope = h1_slope_norm.reindex(df.index, method="ffill").fillna(0.0).values
    return m5_slope.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════
# Causal validator (pre-training assertion)
# ══════════════════════════════════════════════════════════════════════
def validate_causality(df_m5, pip):
    """Perturb bars [probe+1:] and confirm tec5 + regime at bars ≤ probe unchanged."""
    print("Pre-training causality check on tec5 + M5-native regime...", flush=True)
    sample_n = min(20000, len(df_m5))
    probe = 5000
    df = df_m5.iloc[:sample_n].copy().reset_index(drop=True)

    mid = df["close"].values.astype(np.float64)
    tec_a = compute_tec5(mid, sample_n)
    tr_a, hv_a = build_regime_m5(df)
    h1s_a = build_h1_slope_m5(df, pip)

    rng = np.random.default_rng(42)
    df2 = df.copy()
    perturb = rng.normal(0, 0.1, sample_n - probe - 1)
    for col in ("open", "high", "low", "close"):
        df2.loc[probe + 1:, col] = df2.loc[probe + 1:, col].values + perturb
    mid2 = df2["close"].values.astype(np.float64)
    tec_b = compute_tec5(mid2, sample_n)
    tr_b, hv_b = build_regime_m5(df2)
    h1s_b = build_h1_slope_m5(df2, pip)

    past = slice(0, probe + 1)
    for name, a, b in [("tec5", tec_a, tec_b),
                       ("trending", tr_a, tr_b),
                       ("high_vol", hv_a, hv_b),
                       ("h1_slope", h1s_a, h1s_b)]:
        diff = float(np.max(np.abs(a[past] - b[past])))
        if diff > 1e-10:
            raise RuntimeError(f"{name}: causality violated past_max_diff={diff:.2e}")
        future_diff = float(np.max(np.abs(a[probe + 1:] - b[probe + 1:])))
        if future_diff < 1e-10:
            raise RuntimeError(f"{name}: test broken (future identical under perturbation)")
        print(f"  ✓ {name} causal (future_diff={future_diff:.2e})")


# ══════════════════════════════════════════════════════════════════════
# Simulator
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True, inline="always")
def _activate(z, aid):
    if aid == 0:
        return np.tanh(z)
    elif aid == 1:
        return np.sin(z)
    else:
        return np.exp(-z * z)


@njit(cache=True)
def _decode_theta(gene):
    """σ(gene) × 0.5 → θ ∈ [0, 0.5]."""
    return 0.5 / (1.0 + np.exp(-gene))


@njit(cache=True)
def _decode_act(gene):
    g = gene - np.floor(gene)
    aid = int(g * N_ACTS)
    if aid < 0: aid = 0
    if aid >= N_ACTS: aid = N_ACTS - 1
    return aid


@njit(cache=True)
def simulate(genes, n_hid, use_skip, tec5, trending, high_vol, h1_slope, mid, pip, spread_pips,
             max_hold, chunk_start, chunk_end, amddp_coef):
    """Returns (nt, total_score, total_pnl, cum_mae, nl, ns).

    Gene layout (row-major):
      [ 0                  : N_IN*n_hid           ]  W1
      [ N_IN*n_hid         : N_IN*n_hid+n_hid     ]  b1
      [ after b1           : +n_hid*N_OUT         ]  W2
      if skip:
        [ after W2         : +N_IN*N_OUT          ]  Wskip
      [ after (W2|Wskip)   : +N_OUT               ]  b2
      [ after b2           : +n_hid               ]  act_genes (one per hidden)
      [ last               :                      ]  theta_gene
    """
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    # Decode section offsets
    w1_end = N_IN * n_hid
    b1_end = w1_end + n_hid
    w2_end = b1_end + n_hid * N_OUT
    wskip_end = w2_end + (N_IN * N_OUT if use_skip else 0)
    b2_end = wskip_end + N_OUT
    act_end = b2_end + n_hid
    # theta at act_end (= last index)

    theta = _decode_theta(genes[act_end])

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0

    # Preallocate per-step buffers
    x = np.empty(N_IN)
    h = np.empty(n_hid)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
            cum_mae += mae_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        x[0] = tec5[i]
        x[1] = trending[i]
        x[2] = high_vol[i]
        x[3] = h1_slope[i]
        x[4] = np.tanh(upnl_pips / 10.0)
        x[5] = np.tanh(mae_pips / 10.0)
        x[6] = np.tanh(mfe_pips / 10.0)

        # Hidden layer: h_k = act_k( Σ_j W1[k,j] * x[j] + b1[k] )
        for k in range(n_hid):
            z = genes[b1_end - n_hid + k]  # b1[k]
            w1_row = k * N_IN
            for j in range(N_IN):
                z += genes[w1_row + j] * x[j]
            aid_k = _decode_act(genes[b2_end + k])
            h[k] = _activate(z, aid_k)

        # Output layer: o = b2[o] + Σ_k W2[o,k] * h[k] + [Σ_j Wskip[o,j] * x[j] if skip]
        out = np.empty(N_OUT)
        for o in range(N_OUT):
            val = genes[wskip_end + o]  # b2[o]
            w2_row = b1_end + o * n_hid
            for k in range(n_hid):
                val += genes[w2_row + k] * h[k]
            if use_skip:
                wskip_row = w2_end + o * N_IN
                for j in range(N_IN):
                    val += genes[wskip_row + j] * x[j]
            out[o] = val

        ob = out[0]; os_ = out[1]; of = out[2]

        # Force close if max_hold exceeded
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            total_pnl += pnl
            nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        if position == 0:
            # Latching entry: winner must beat FLAT and OPPOSITE by θ
            if (ob - of) > theta and (ob - os_) > theta:
                position = 1; entry_price = mid[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif (os_ - of) > theta and (os_ - ob) > theta:
                position = -1; entry_price = mid[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            # Simple argmax exit (no θ)
            close_now = False; new_pos = 0
            if of > ob and of > os_:
                close_now = True
            elif position == 1 and os_ > ob and os_ > of:
                close_now = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of:
                close_now = True; new_pos = 1
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl
                nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1: entry_price = mid[i] + spread_pips * pip
                    else: entry_price = mid[i] - spread_pips * pip
                    entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0

    total_score = total_pnl - amddp_coef * cum_mae
    return nt, total_score, total_pnl, cum_mae, nl, ns


# ══════════════════════════════════════════════════════════════════════
# Fitness (3-chunk WF, bidir asym, integrated AMDDP)
# ══════════════════════════════════════════════════════════════════════
_W = {}


def _winit(tec5, trending, high_vol, h1_slope, mid, pip, spread, max_hold, bpd, amddp, n_hid, use_skip):
    _W.update({"tec5": tec5, "trending": trending, "high_vol": high_vol,
               "h1_slope": h1_slope,
               "mid": mid, "pip": pip, "spread": spread, "max_hold": max_hold,
               "bpd": bpd, "amddp": amddp, "n_hid": n_hid, "use_skip": use_skip})


def _eval_one(genes):
    tec5 = _W["tec5"]; trending = _W["trending"]; high_vol = _W["high_vol"]
    h1_slope = _W["h1_slope"]
    mid = _W["mid"]
    n_hid = _W["n_hid"]; use_skip = _W["use_skip"]
    n = len(mid)
    tl = 0; ts = 0; tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0
    chunk_trades_list = []
    for ci in range(N_CHUNKS):
        c_s = int(n * ci / N_CHUNKS)
        c_e = int(n * (ci + 1) / N_CHUNKS)
        nt, score, _pnl, _cm, nl, ns = simulate(
            genes, n_hid, use_skip, tec5, trending, high_vol, h1_slope, mid,
            _W["pip"], _W["spread"], _W["max_hold"],
            c_s, c_e, _W["amddp"])
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / _W["bpd"]
        sps = score / days if days > 0 else 0.0
        chunk_sps.append(sps)
        chunk_trades_list.append(nt)
        if sps < 0: losing += -sps
    total_days = n / _W["bpd"]
    base_sps = tscore / total_days if total_days > 0 else 0.0

    # ── Hard gates (return large penalty with smooth gradient so CMA can climb out) ──
    # Gate 1: min ≈ 1 trade/day per chunk
    chunk_days = (n / N_CHUNKS) / _W["bpd"]
    min_per_chunk = int(chunk_days)
    min_chunk_trades = min(chunk_trades_list) if chunk_trades_list else 0
    trades_short = max(0, min_per_chunk - min_chunk_trades)
    # Gate 2: dir_ratio ≥ MIN_DIR always (not only in bonus branch)
    dir_ratio = (min(tl, ts) / tt) if tt > 0 else 0.0
    dir_short = max(0.0, MIN_DIR - dir_ratio)

    if trades_short > 0 or dir_short > 0:
        # 500 base + gradient: each missing trade = 0.1, each missing dir-pt = 100
        return 500.0 + trades_short * 0.1 + dir_short * 100.0

    # ── Normal fitness (both gates passed) ──
    asym = (1.0 - 2.0 * dir_ratio) * 25.0
    # Harsher activity: reward trading well past the min-per-chunk floor (≥100 total extra)
    activity = max(0.0, 100.0 - tt) * 2.0
    all_prof = all(s > 0 for s in chunk_sps)
    if all_prof:
        score = min(chunk_sps) - asym
    else:
        score = base_sps - asym - activity - losing * 2.0
    return -score  # CMA minimizes


_POOL = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EUR_JPY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--pop", type=int, default=40,
                        help="CMA population (popsize). Default 40.")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Initial CMA step size. Default 0.5.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-hid", type=int, default=1,
                        help="Number of hidden nodes (1, 3, 4 ...). Default 1.")
    parser.add_argument("--skip", action="store_true",
                        help="Add direct input→output skip connections (IronNet V5 style).")
    parser.add_argument("--no-regime", action="store_true",
                        help="Zero out trending + high_vol inputs (isolate TEC_5 + h1_slope + state).")
    parser.add_argument("--no-h1-slope", action="store_true",
                        help="Zero out h1_slope input.")
    args = parser.parse_args()

    NP = n_params(args.n_hid, args.skip)

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    # ── Load M5 OHLC
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n_bars = len(df)
    print(f"Loaded {n_bars:,} M5 bars of {pair}", flush=True)

    # ── Causality checks on full pipeline
    validate_causality(df, pip)
    print()

    # ── Compute features
    t0 = time.time()
    mid = df["close"].values.astype(np.float64)
    tec5_full = compute_tec5(mid, n_bars)
    trending_full, high_vol_full = build_regime_m5(df)
    h1_slope_full = build_h1_slope_m5(df, pip)
    print(f"Features computed in {time.time()-t0:.1f}s", flush=True)
    print(f"  tec5:     range=[{tec5_full[100:].min():+.3f}, {tec5_full[100:].max():+.3f}] std={tec5_full[100:].std():.3f}")
    print(f"  trending: mean={trending_full[100:].mean():.3f}")
    print(f"  high_vol: mean={high_vol_full[100:].mean():.3f}")
    print(f"  h1_slope: range=[{h1_slope_full[100:].min():+.3f}, {h1_slope_full[100:].max():+.3f}] std={h1_slope_full[100:].std():.3f}")

    if args.no_regime:
        print("  ⚠ --no-regime: zeroing trending + high_vol")
        trending_full = np.zeros_like(trending_full)
        high_vol_full = np.zeros_like(high_vol_full)
    if args.no_h1_slope:
        print("  ⚠ --no-h1-slope: zeroing h1_slope")
        h1_slope_full = np.zeros_like(h1_slope_full)

    split = int(n_bars * 0.7)
    tec5_is = tec5_full[:split]; trending_is = trending_full[:split]
    high_vol_is = high_vol_full[:split]; h1_slope_is = h1_slope_full[:split]
    mid_is = mid[:split]
    tec5_oos = tec5_full[split:]; trending_oos = trending_full[split:]
    high_vol_oos = high_vol_full[split:]; h1_slope_oos = h1_slope_full[split:]
    mid_oos = mid[split:]
    print(f"\nIS: {split:,} ({split/BARS_PER_DAY:.0f}d), OOS: {n_bars-split:,} ({(n_bars-split)/BARS_PER_DAY:.0f}d)")

    # ── JIT warm
    dummy_genes = np.zeros(NP)
    simulate(dummy_genes, args.n_hid, args.skip,
             tec5_is[:300], trending_is[:300], high_vol_is[:300], h1_slope_is[:300],
             mid_is[:300], pip, spread, 50, 0, 300, AMDDP_COEF)

    # ── CMA setup
    np.random.seed(args.seed)
    x0 = np.zeros(NP)
    # Seed activation genes (the n_hid slots just before the final theta gene) mid-range
    b2_end = N_IN * args.n_hid + args.n_hid + args.n_hid * N_OUT \
             + (N_IN * N_OUT if args.skip else 0) + N_OUT
    for k in range(args.n_hid):
        x0[b2_end + k] = np.random.uniform(0, 1)
    x0[-1] = 0.0  # θ gene → σ(0)*0.5 = 0.25 initially

    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=args.workers,
        initializer=_winit,
        initargs=(tec5_is, trending_is, high_vol_is, h1_slope_is, mid_is,
                  pip, spread, MAX_HOLD, BARS_PER_DAY, AMDDP_COEF,
                  args.n_hid, args.skip))

    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        "popsize": args.pop,
        "seed": args.seed,
        "verbose": -9,
        "maxiter": args.gens,
        "tolx": 1e-8,
    })

    topo_tag = f"6→{args.n_hid}→3" + ("+skip" if args.skip else "")
    print(f"\nCMA {topo_tag} | {NP} params | pop {args.pop} | gens {args.gens} | σ0 {args.sigma} | amddp_coef {AMDDP_COEF}", flush=True)
    t0 = time.time()
    best_score = float("inf"); best_genes = x0.copy()
    gen = 0
    while not es.stop() and gen < args.gens:
        solutions = es.ask()
        losses = list(_POOL.map(_eval_one, solutions))
        es.tell(solutions, losses)
        gen_best = min(losses); gen_idx = losses.index(gen_best)
        if gen_best < best_score:
            best_score = gen_best
            best_genes = solutions[gen_idx].copy()
        if gen % 10 == 0:
            act_names_list = []
            for k in range(args.n_hid):
                g = best_genes[b2_end + k]
                aid_k = int((g - np.floor(g)) * N_ACTS)
                aid_k = min(max(aid_k, 0), N_ACTS - 1)
                act_names_list.append(["tanh", "sin", "gauss"][aid_k])
            theta_val = 0.5 / (1.0 + np.exp(-best_genes[-1]))
            acts_str = "/".join(act_names_list)
            print(f"  gen {gen:3d} | best fitness {-best_score:+.3f} | acts=[{acts_str}] θ={theta_val:.3f}", flush=True)
        gen += 1
    elapsed = time.time() - t0

    # ── Evaluate winner
    is_nt, _, is_pnl, is_cm, is_nl, is_ns = simulate(
        best_genes, args.n_hid, args.skip,
        tec5_is, trending_is, high_vol_is, h1_slope_is, mid_is,
        pip, spread, MAX_HOLD, 0, len(mid_is), AMDDP_COEF)
    oos_nt, _, oos_pnl, oos_cm, oos_nl, oos_ns = simulate(
        best_genes, args.n_hid, args.skip,
        tec5_oos, trending_oos, high_vol_oos, h1_slope_oos, mid_oos,
        pip, spread, MAX_HOLD, 0, len(mid_oos), AMDDP_COEF)
    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    act_names_list = []
    for k in range(args.n_hid):
        g = best_genes[b2_end + k]
        aid_k = int((g - np.floor(g)) * N_ACTS)
        aid_k = min(max(aid_k, 0), N_ACTS - 1)
        act_names_list.append(["tanh", "sin", "gauss"][aid_k])
    theta_val = 0.5 / (1.0 + np.exp(-best_genes[-1]))
    acts_str = "/".join(act_names_list)

    print(f"\n{'='*72}")
    print(f"  DCLOSE3 + REGIME (CMA {topo_tag}, {NP}p): {pair}")
    print(f"{'='*72}")
    print(f"  Winner: activations=[{acts_str}], θ={theta_val:.3f}, fitness={-best_score:+.3f}")
    print(f"  IS:  {is_nt}T L/S={is_nl}/{is_ns} {is_pnl:+.1f}p ({is_pnl/is_days:+.2f} p/d dir={is_dir:.2f} cumMAE={is_cm:.0f}p)")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns} {oos_pnl:+.1f}p ({oos_pnl/oos_days:+.2f} p/d dir={oos_dir:.2f} cumMAE={oos_cm:.0f}p)")
    print(f"  Elapsed: {elapsed:.0f}s")

    skip_tag = "_skip" if args.skip else ""
    out = OUT_DIR / f"dclose3_regime_cma_h{args.n_hid}{skip_tag}_{pair}_s{args.seed}.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "pair": pair, "seed": args.seed, "genes": best_genes,
            "topology": f"6->{args.n_hid}->3" + ("+skip" if args.skip else "") + " fixed, CMA-ES",
            "n_params": int(NP),
            "inputs": ["tec5", "trending", "high_vol", "h1_slope", "upnl", "mae", "mfe"],
            "activation_bank": ["tanh", "sin", "gauss"],
            "activations_selected": act_names_list, "theta_selected": float(theta_val),
            "amddp_coef": AMDDP_COEF,
            "fitness": float(-best_score),
            "is": {"n_trades": int(is_nt), "total_pnl": float(is_pnl),
                   "pips_per_day": float(is_pnl/is_days),
                   "n_long": int(is_nl), "n_short": int(is_ns),
                   "dir_ratio": float(is_dir), "cum_mae": float(is_cm)},
            "oos": {"n_trades": int(oos_nt), "total_pnl": float(oos_pnl),
                    "pips_per_day": float(oos_pnl/oos_days),
                    "n_long": int(oos_nl), "n_short": int(oos_ns),
                    "dir_ratio": float(oos_dir), "cum_mae": float(oos_cm)},
            "elapsed_s": float(elapsed),
        }, f)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
