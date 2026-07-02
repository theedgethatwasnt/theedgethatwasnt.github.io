#!/usr/bin/env python3
"""
Range Bar Builder — fixed and ATR-adaptive.

Converts S5 OHLC data into range bars where each bar represents
exactly N pips of price movement (fixed) or ATR×mult (adaptive).

Usage:
    from range_bars import build_range_bars, build_atr_range_bars
    bars = build_range_bars(closes, pip=0.01, range_pips=10)
    bars = build_atr_range_bars(highs, lows, closes, pip=0.01, atr_period=14, atr_mult=1.0)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RangeBar:
    open: float
    high: float
    low: float
    close: float
    start_idx: int       # S5 bar index when this range bar started
    end_idx: int         # S5 bar index when this range bar completed
    direction: int       # +1 up, -1 down
    range_pips: float    # Actual range in pips
    n_s5_bars: int       # How many S5 bars this range bar spans
    volume: int = 0      # Sum of S5 volumes


def build_range_bars(closes: np.ndarray, pip: float, range_pips: float,
                     highs: np.ndarray = None, lows: np.ndarray = None,
                     volumes: np.ndarray = None) -> List[RangeBar]:
    """
    Build fixed-size range bars.

    A new bar completes when price moves range_pips from the bar's open.
    The bar's close is the price that triggered completion.
    """
    range_price = range_pips * pip
    bars = []

    if len(closes) < 2:
        return bars

    bar_open = closes[0]
    bar_high = closes[0]
    bar_low = closes[0]
    bar_start = 0
    bar_vol = 0

    for i in range(1, len(closes)):
        price = closes[i]
        if highs is not None:
            bar_high = max(bar_high, highs[i])
            bar_low = min(bar_low, lows[i])
        else:
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)

        if volumes is not None:
            bar_vol += int(volumes[i])

        # Check if range exceeded
        move = price - bar_open
        if abs(move) >= range_price:
            direction = 1 if move > 0 else -1
            bars.append(RangeBar(
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=price,
                start_idx=bar_start,
                end_idx=i,
                direction=direction,
                range_pips=abs(move) / pip,
                n_s5_bars=i - bar_start,
                volume=bar_vol,
            ))
            # Start new bar
            bar_open = price
            bar_high = price
            bar_low = price
            bar_start = i
            bar_vol = 0

    return bars


def build_atr_range_bars(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         pip: float, atr_period: int = 14, atr_mult: float = 1.0,
                         min_range_pips: float = 3.0, max_range_pips: float = 50.0,
                         volumes: np.ndarray = None) -> List[RangeBar]:
    """
    Build ATR-adaptive range bars.

    Bar size = ATR(atr_period) × atr_mult, clamped to [min, max].
    ATR updates with each completed range bar (not every S5 bar).
    """
    bars = []

    if len(closes) < atr_period + 1:
        return bars

    # Bootstrap ATR from first atr_period S5 bars
    trs = []
    for i in range(1, min(atr_period + 1, len(closes))):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atr = np.mean(trs) if trs else 10 * pip

    current_range = max(min_range_pips, min(max_range_pips, atr / pip * atr_mult)) * pip

    bar_open = closes[atr_period]
    bar_high = closes[atr_period]
    bar_low = closes[atr_period]
    bar_start = atr_period
    bar_vol = 0

    # Track recent range bar TRs for updating ATR
    recent_bar_ranges = list(trs[-atr_period:])

    for i in range(atr_period + 1, len(closes)):
        price = closes[i]
        bar_high = max(bar_high, highs[i])
        bar_low = min(bar_low, lows[i])
        if volumes is not None:
            bar_vol += int(volumes[i])

        move = price - bar_open
        if abs(move) >= current_range:
            direction = 1 if move > 0 else -1
            bars.append(RangeBar(
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=price,
                start_idx=bar_start,
                end_idx=i,
                direction=direction,
                range_pips=abs(move) / pip,
                n_s5_bars=i - bar_start,
                volume=bar_vol,
            ))

            # Update ATR with this range bar's true range
            bar_tr = bar_high - bar_low
            recent_bar_ranges.append(bar_tr)
            if len(recent_bar_ranges) > atr_period:
                recent_bar_ranges = recent_bar_ranges[-atr_period:]
            atr = np.mean(recent_bar_ranges)
            current_range = max(min_range_pips * pip,
                              min(max_range_pips * pip, atr * atr_mult))

            # Start new bar
            bar_open = price
            bar_high = price
            bar_low = price
            bar_start = i
            bar_vol = 0

    return bars


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    data_dir = "/path/to/projects/fx-core/data/asi_mc_indicators"

    for pair in ["EUR_JPY", "EUR_USD", "GBP_JPY"]:
        df = pd.read_parquet(f"{data_dir}/{pair}_asi_mc.parquet")
        closes = df["mid_close"].values
        pip = 0.01 if "JPY" in pair else 0.0001

        # Fixed range bars
        fixed_bars = build_range_bars(closes, pip, range_pips=10)

        # ATR adaptive
        highs = df.get("mid_high", df["mid_close"]).values
        lows = df.get("mid_low", df["mid_close"]).values
        atr_bars = build_atr_range_bars(highs, lows, closes, pip, atr_mult=1.0)

        s5_count = len(closes)
        m5_count = s5_count  # Already M5 in our parquets
        days = m5_count / 288

        print(f"{pair}:")
        print(f"  M5 bars: {m5_count:,} ({days:.0f} days)")
        print(f"  Fixed 10-pip range bars: {len(fixed_bars):,} ({len(fixed_bars)/days:.1f}/day)")
        print(f"  ATR adaptive range bars: {len(atr_bars):,} ({len(atr_bars)/days:.1f}/day)")
        if fixed_bars:
            avg_duration = np.mean([b.n_s5_bars for b in fixed_bars])
            print(f"  Avg fixed bar duration: {avg_duration:.0f} M5 bars ({avg_duration*5/60:.1f}h)")
        if atr_bars:
            avg_duration = np.mean([b.n_s5_bars for b in atr_bars])
            avg_range = np.mean([b.range_pips for b in atr_bars])
            print(f"  Avg ATR bar duration: {avg_duration:.0f} M5 bars ({avg_duration*5/60:.1f}h)")
            print(f"  Avg ATR bar range: {avg_range:.1f} pips")
        print()
