#!/usr/bin/env python3
"""
Export D-experiment indicators to parquet.
==========================================
Extends the ASI-MC parquets with additional indicator columns needed
for D1-D6 combo experiments and V3 (ER-enhanced ASI-MC).

New columns added to existing parquets:
  sign_mc_d      — sign(MC_D_A) → {-1, 0, +1}
  sign_mc_dd     — sign(MC_DD_A) → {-1, 0, +1}
  h1_sr_zone     — price vs H1 zigzag S/R → {-1, 0, +1}
  adx_regime     — ADX(14) quantized → {-1, 0, +1} (low/mid/high)
  vol_regime     — ATR% tercile → {-1, 0, +1}
  str_diff_sign  — sign(strongest - weakest currency) → {-1, +1}
  er_norm        — Kaufman ER(60) normalized via arctan → [0, ~0.85]

D-experiment input mappings:
  D1: sign_mc_d, h1_sr_zone, UPnL
  D2: sign_mc_d, adx_regime, UPnL
  D3: sign_mc_d, str_diff_sign, UPnL
  D4: sign_mc_d, vol_regime, UPnL
  D5: h1_sr_zone, str_diff_sign, UPnL
  D6: sign_mc_d, sign_mc_dd, h1_sr_zone, UPnL (4 inputs)
  V3: mc_d_a, mc_dd_a, er_norm, UPnL (4 inputs — ASI-MC + ER filter)

Usage:
  python3 export_d_indicators.py
"""

import sys
import os
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.indicators import ZigzagSR, ATR, KalmanStrength
from lib.pair_config import PAIRS, ALL_PAIR_NAMES

DATA_DIR = PROJECT_ROOT / "data" / "scalper_parquet"
INDICATOR_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"

# Pairs used in Kalman strength decomposition
KALMAN_PAIRS = [
    ("EUR_USD", "EUR", "USD"), ("GBP_USD", "GBP", "USD"),
    ("AUD_USD", "AUD", "USD"), ("NZD_USD", "NZD", "USD"),
    ("USD_JPY", "USD", "JPY"), ("EUR_JPY", "EUR", "JPY"),
    ("GBP_JPY", "GBP", "JPY"), ("AUD_JPY", "AUD", "JPY"),
    ("CAD_JPY", "CAD", "JPY"), ("CHF_JPY", "CHF", "JPY"),
    ("NZD_JPY", "NZD", "JPY"), ("EUR_GBP", "EUR", "GBP"),
]
CURRENCIES = ["EUR", "USD", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"]


@njit(cache=True)
def compute_er_norm(closes, window=60):
    """Kaufman Efficiency Ratio (window bars), arctan-normalized.

    ER = |close[i] - close[i-window]| / sum(|diff(close)| over window)
    er_norm = arctan(ER / 0.3) / (pi/2)
    Maps: ER=0 → 0 (pure chop), ER=0.3 → 0.5, ER=1 → ~0.82 (perfect trend)
    """
    n = len(closes)
    result = np.zeros(n)
    half_pi = np.pi / 2.0
    for i in range(window, n):
        net_change = abs(closes[i] - closes[i - window])
        total_path = 0.0
        for j in range(i - window + 1, i + 1):
            total_path += abs(closes[j] - closes[j - 1])
        if total_path > 0.0:
            er = net_change / total_path
            result[i] = np.arctan(er / 0.3) / half_pi
    return result


@njit(cache=True)
def compute_adx(high, low, close, period=14):
    """Wilder's ADX(14). Returns array of ADX values in [0, 100]."""
    n = len(high)
    adx = np.zeros(n)
    if n < period * 2:
        return adx

    # True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        plus_dm[i] = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
        minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0.0

    # Wilder smoothing (first value = sum of period, then EMA-style)
    atr_s = 0.0
    pdm_s = 0.0
    mdm_s = 0.0
    for i in range(1, period + 1):
        atr_s += tr[i]
        pdm_s += plus_dm[i]
        mdm_s += minus_dm[i]

    dx_vals = np.zeros(n)
    for i in range(period, n):
        if i == period:
            pass  # already summed
        else:
            atr_s = atr_s - atr_s / period + tr[i]
            pdm_s = pdm_s - pdm_s / period + plus_dm[i]
            mdm_s = mdm_s - mdm_s / period + minus_dm[i]

        if atr_s > 0:
            pdi = 100.0 * pdm_s / atr_s
            mdi = 100.0 * mdm_s / atr_s
            di_sum = pdi + mdi
            if di_sum > 0:
                dx_vals[i] = 100.0 * abs(pdi - mdi) / di_sum

    # ADX = Wilder smoothing of DX
    adx_sum = 0.0
    for i in range(period, period * 2):
        adx_sum += dx_vals[i]
    adx[period * 2 - 1] = adx_sum / period

    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx_vals[i]) / period

    return adx


def compute_h1_sr_zone(s5_df, m5_times, n_m5, pip):
    """Compute H1 zigzag S/R zone at M5 cadence.

    Returns array of {-1, 0, +1}:
      -1 = price below support
       0 = price between support and resistance
      +1 = price above resistance
    """
    min_swing = 50 * pip  # 50 pips min swing for zigzag
    zz = ZigzagSR(min_swing=min_swing)

    # Build H1 bars from S5, track S/R at each M5 bar
    sr_zone = np.zeros(n_m5, dtype=np.float64)
    s5_idx = 0
    m5_idx = 0
    current_hour = -1

    timestamps = s5_df.index
    opens = s5_df["o"].values
    highs = s5_df["h"].values
    lows = s5_df["l"].values
    closes = s5_df["c"].values

    h1_open = h1_high = h1_low = h1_close = 0.0

    for i in range(len(s5_df)):
        ts = timestamps[i]
        hour = ts.hour

        if current_hour == -1:
            current_hour = hour
            h1_open = opens[i]
            h1_high = highs[i]
            h1_low = lows[i]
            h1_close = closes[i]
        elif hour != current_hour:
            # Complete H1 bar
            bar = {"open": h1_open, "high": h1_high, "low": h1_low, "close": h1_close}
            zz.update_from_h1_bar(bar)
            current_hour = hour
            h1_open = opens[i]
            h1_high = highs[i]
            h1_low = lows[i]
            h1_close = closes[i]
        else:
            if highs[i] > h1_high:
                h1_high = highs[i]
            if lows[i] < h1_low:
                h1_low = lows[i]
            h1_close = closes[i]

        # Every 60 S5 bars = 1 M5 bar (check if this S5 completes an M5)
        if i > 0 and i % 60 == 59 and m5_idx < n_m5:
            sup = zz.state.support
            res = zz.state.resistance
            price = closes[i]
            if sup > 0 and res > 0 and res > sup:
                if price < sup:
                    sr_zone[m5_idx] = -1.0
                elif price > res:
                    sr_zone[m5_idx] = 1.0
                else:
                    sr_zone[m5_idx] = 0.0
            m5_idx += 1

    # Fill any remaining M5 bars
    if m5_idx > 0:
        for j in range(m5_idx, n_m5):
            sr_zone[j] = sr_zone[m5_idx - 1]

    return sr_zone


def compute_vol_regime(s5_df, m5_times, n_m5):
    """Compute volatility regime at M5 cadence.

    Uses rolling ATR% (ATR/close) with 200-bar lookback on H1.
    Quantized to {-1, 0, +1}: low/mid/high vol tercile.
    """
    # Build H1 ATR from S5
    atr = ATR(period=14)
    h1_atr_vals = []
    h1_close_vals = []
    current_hour = -1
    h1_o = h1_h = h1_l = h1_c = 0.0

    timestamps = s5_df.index
    opens = s5_df["o"].values
    highs = s5_df["h"].values
    lows = s5_df["l"].values
    closes = s5_df["c"].values

    for i in range(len(s5_df)):
        hour = timestamps[i].hour
        if current_hour == -1:
            current_hour = hour
            h1_o, h1_h, h1_l, h1_c = opens[i], highs[i], lows[i], closes[i]
        elif hour != current_hour:
            bar = {"open": h1_o, "high": h1_h, "low": h1_l, "close": h1_c}
            atr_val = atr.update(bar)
            h1_atr_vals.append(atr_val)
            h1_close_vals.append(h1_c)
            current_hour = hour
            h1_o, h1_h, h1_l, h1_c = opens[i], highs[i], lows[i], closes[i]
        else:
            if highs[i] > h1_h: h1_h = highs[i]
            if lows[i] < h1_l: h1_l = lows[i]
            h1_c = closes[i]

    h1_atr = np.array(h1_atr_vals, dtype=np.float64)
    h1_cls = np.array(h1_close_vals, dtype=np.float64)

    if len(h1_cls) == 0:
        return np.zeros(n_m5)

    # ATR% = ATR / close
    atr_pct = np.where(h1_cls > 0, h1_atr / h1_cls * 100.0, 0.0)

    # Rolling tercile over 200-bar window
    vol_regime_h1 = np.zeros(len(atr_pct))
    window = 200
    for i in range(len(atr_pct)):
        start = max(0, i - window)
        chunk = atr_pct[start:i + 1]
        if len(chunk) < 20:
            continue
        p33 = np.percentile(chunk, 33)
        p67 = np.percentile(chunk, 67)
        if atr_pct[i] < p33:
            vol_regime_h1[i] = -1.0  # low vol
        elif atr_pct[i] > p67:
            vol_regime_h1[i] = 1.0   # high vol
        # else 0.0 = mid vol

    # Map H1 to M5 cadence (12 M5 bars per H1)
    vol_regime = np.zeros(n_m5)
    for i in range(n_m5):
        h1_idx = min(i // 12, len(vol_regime_h1) - 1)
        if h1_idx >= 0:
            vol_regime[i] = vol_regime_h1[h1_idx]

    return vol_regime


def compute_adx_regime(s5_df, m5_times, n_m5):
    """Compute ADX regime at M5 cadence.

    ADX(14) on H1, quantized to {-1, 0, +1}: weak/mid/strong trend.
    """
    # Build H1 OHLC from S5
    h1_highs, h1_lows, h1_closes = [], [], []
    current_hour = -1
    h1_h = h1_l = h1_c = 0.0

    timestamps = s5_df.index
    highs = s5_df["h"].values
    lows = s5_df["l"].values
    closes = s5_df["c"].values

    for i in range(len(s5_df)):
        hour = timestamps[i].hour
        if current_hour == -1:
            current_hour = hour
            h1_h, h1_l, h1_c = highs[i], lows[i], closes[i]
        elif hour != current_hour:
            h1_highs.append(h1_h)
            h1_lows.append(h1_l)
            h1_closes.append(h1_c)
            current_hour = hour
            h1_h, h1_l, h1_c = highs[i], lows[i], closes[i]
        else:
            if highs[i] > h1_h: h1_h = highs[i]
            if lows[i] < h1_l: h1_l = lows[i]
            h1_c = closes[i]

    h_arr = np.array(h1_highs, dtype=np.float64)
    l_arr = np.array(h1_lows, dtype=np.float64)
    c_arr = np.array(h1_closes, dtype=np.float64)

    adx = compute_adx(h_arr, l_arr, c_arr, period=14)

    # Quantize: <20 = weak (-1), 20-40 = mid (0), >40 = strong (+1)
    adx_regime_h1 = np.where(adx < 20, -1.0, np.where(adx > 40, 1.0, 0.0))

    # Map H1 to M5 cadence
    adx_regime = np.zeros(n_m5)
    for i in range(n_m5):
        h1_idx = min(i // 12, len(adx_regime_h1) - 1)
        if h1_idx >= 0:
            adx_regime[i] = adx_regime_h1[h1_idx]

    return adx_regime


def compute_strength_diff(all_s5_data, pair, m5_times, n_m5):
    """Compute currency strength difference at M5 cadence.

    Uses Kalman filter to decompose pair returns into currency strengths.
    For a given pair (BASE_QUOTE), returns sign(base_strength - quote_strength).
    """
    kalman = KalmanStrength(CURRENCIES, KALMAN_PAIRS)

    # We need all pairs' H1 closes aligned. Build H1 from S5 for all pairs.
    # Process bar by bar at H1 cadence
    h1_closes_all = {}  # {pair: [close1, close2, ...]}
    for p in ALL_PAIR_NAMES:
        h1_closes_all[p] = []

    # Find common timestamps across all pairs
    # Use the target pair's S5 index as reference
    ref_df = all_s5_data[pair]
    timestamps = ref_df.index
    current_hour = -1
    h1_c_buf = {p: 0.0 for p in ALL_PAIR_NAMES}

    for i in range(len(ref_df)):
        hour = timestamps[i].hour
        if current_hour == -1:
            current_hour = hour
            for p in ALL_PAIR_NAMES:
                if p in all_s5_data and i < len(all_s5_data[p]):
                    h1_c_buf[p] = all_s5_data[p]["c"].values[i]
        elif hour != current_hour:
            # Complete H1 bar — update Kalman with all pair closes
            pair_closes = {}
            for p in ALL_PAIR_NAMES:
                if h1_c_buf[p] > 0:
                    pair_closes[p] = h1_c_buf[p]

            strengths = kalman.update(pair_closes)
            if strengths:
                base, quote = pair.split("_")
                diff = strengths.get(base, 0) - strengths.get(quote, 0)
                for p in ALL_PAIR_NAMES:
                    h1_closes_all[p].append(1.0 if diff > 0 else -1.0)
            else:
                for p in ALL_PAIR_NAMES:
                    h1_closes_all[p].append(0.0)

            current_hour = hour
            for p in ALL_PAIR_NAMES:
                if p in all_s5_data and i < len(all_s5_data[p]):
                    h1_c_buf[p] = all_s5_data[p]["c"].values[i]
        else:
            for p in ALL_PAIR_NAMES:
                if p in all_s5_data and i < len(all_s5_data[p]):
                    h1_c_buf[p] = all_s5_data[p]["c"].values[i]

    str_diff_h1 = np.array(h1_closes_all[pair], dtype=np.float64) if pair in h1_closes_all else np.zeros(0)

    # Map to M5
    str_diff = np.zeros(n_m5)
    for i in range(n_m5):
        h1_idx = min(i // 12, len(str_diff_h1) - 1)
        if h1_idx >= 0:
            str_diff[i] = str_diff_h1[h1_idx]

    return str_diff


def main():
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print(f"Exporting D-experiment indicators for {len(ALL_PAIR_NAMES)} pairs...")
    print(f"Reading from: {DATA_DIR}/")
    print(f"Writing to: {INDICATOR_DIR}/")

    # Pre-load all S5 data for Kalman strength (needs all pairs simultaneously)
    print("\nPre-loading S5 data for all pairs...")
    all_s5 = {}
    for pair in ALL_PAIR_NAMES:
        parquet = DATA_DIR / f"{pair.replace('_', '')}_S5_BA.parquet"
        if not parquet.exists():
            print(f"  {pair}: SKIP (no parquet)")
            continue
        df = pd.read_parquet(parquet, engine="pyarrow")
        ts = pd.to_datetime(df["timestamp"])
        half_spread = (df["ask_c"].values - df["bid_c"].values) / 2.0
        s5_df = pd.DataFrame({
            "o": df["bid_o"].values + half_spread,
            "h": df["bid_h"].values + half_spread,
            "l": df["bid_l"].values + half_spread,
            "c": df["bid_c"].values + half_spread,
        }, index=ts)
        all_s5[pair] = s5_df
        print(f"  {pair}: {len(s5_df):,} S5 bars")
        del df
    gc.collect()

    # Per-pair: compute all D-experiment indicators
    for pair in ALL_PAIR_NAMES:
        if pair not in all_s5:
            continue

        t0 = time.time()
        print(f"\n  {pair}...", flush=True)
        s5_df = all_s5[pair]

        # Load existing parquet to merge
        existing_path = INDICATOR_DIR / f"{pair}_asi_mc.parquet"
        if not existing_path.exists():
            print(f"    SKIP (no existing ASI-MC parquet)")
            continue
        existing = pd.read_parquet(existing_path, engine="pyarrow")

        # Resample to M5 for reference
        m5 = s5_df.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        m5_times = m5.index
        n_m5 = len(m5)
        pip = 0.01 if "JPY" in pair else 0.0001

        # Align with existing parquet length
        if n_m5 != len(existing):
            # Trim to shorter
            n_m5 = min(n_m5, len(existing))
            print(f"    Aligning: {len(existing)} existing, {len(m5)} computed → {n_m5}")

        # 1. sign(MC_D) and sign(MC_dD) from Variant A
        mc_d_a = existing["mc_d_a"].values[:n_m5]
        mc_dd_a = existing["mc_dd_a"].values[:n_m5]
        sign_mc_d = np.sign(mc_d_a).astype(np.float64)
        sign_mc_dd = np.sign(mc_dd_a).astype(np.float64)
        print(f"    sign(MC_D): +1={np.sum(sign_mc_d>0):,} 0={np.sum(sign_mc_d==0):,} -1={np.sum(sign_mc_d<0):,}")

        # 2. H1 SR zone
        h1_sr = compute_h1_sr_zone(s5_df, m5_times, n_m5, pip)
        print(f"    H1_SR: below={np.sum(h1_sr<0):,} range={np.sum(h1_sr==0):,} above={np.sum(h1_sr>0):,}")

        # 3. ADX regime
        adx_reg = compute_adx_regime(s5_df, m5_times, n_m5)
        print(f"    ADX: weak={np.sum(adx_reg<0):,} mid={np.sum(adx_reg==0):,} strong={np.sum(adx_reg>0):,}")

        # 4. Vol regime
        vol_reg = compute_vol_regime(s5_df, m5_times, n_m5)
        print(f"    Vol: low={np.sum(vol_reg<0):,} mid={np.sum(vol_reg==0):,} high={np.sum(vol_reg>0):,}")

        # 5. Strength diff (skip for now if too slow — can be added later)
        str_diff = compute_strength_diff(all_s5, pair, m5_times, n_m5)
        print(f"    StrDiff: +1={np.sum(str_diff>0):,} -1={np.sum(str_diff<0):,}")

        # 6. ER_norm — Kaufman Efficiency Ratio (window=60 M5 bars = 5h lookback)
        m5_closes = m5["c"].values[:n_m5].astype(np.float64)
        er_norm = compute_er_norm(m5_closes, window=60)
        print(f"    ER_norm: mean={er_norm[60:].mean():.3f} std={er_norm[60:].std():.3f} "
              f"(chop<0.3={np.mean(er_norm[60:]<0.3)*100:.0f}% trend>0.6={np.mean(er_norm[60:]>0.6)*100:.0f}%)")

        # Merge into existing parquet
        existing = existing.iloc[:n_m5].copy()
        existing["sign_mc_d"] = sign_mc_d
        existing["sign_mc_dd"] = sign_mc_dd
        existing["h1_sr_zone"] = h1_sr
        existing["adx_regime"] = adx_reg
        existing["vol_regime"] = vol_reg
        existing["str_diff_sign"] = str_diff
        existing["er_norm"] = er_norm

        existing.to_parquet(existing_path, index=False, engine="pyarrow")
        elapsed = time.time() - t0
        print(f"    Saved: {existing_path.name} ({len(existing.columns)} cols, {elapsed:.1f}s)")

    total = time.time() - t_total
    print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")

    # Free memory
    del all_s5
    gc.collect()


if __name__ == "__main__":
    main()
