"""
Indicator engine — IncrementalFeatureBuilder for all trading indicators.

Single source of truth: used by both live curator and backtesting.
Extracted verbatim from scalper/unified/live_neat_pf_unified.py to ensure
bit-identical outputs during shadow validation.

Each indicator class:
  - Declares its warmup requirements
  - Maintains incremental state via circular buffers
  - Has a process_bar() method for live updates
  - Has an initialize_from_history() method for warmup
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── P&F Box Builder ──────────────────────────────────────────────────────

@dataclass
class PnFBox:
    """A single Point & Figure box."""
    level: float
    direction: int       # +1 = X (up), -1 = O (down)
    column_id: int
    timestamp: str
    mid_price: float


@dataclass
class PnFState:
    """Mutable state for one P&F chart (one pair + one config)."""
    current_level: float = 0.0
    direction: int = 0          # 0=uninitialized, +1=X, -1=O
    column_id: int = 0
    column_boxes_count: int = 0
    boxes_since_reversal: int = 0
    box_history: list = field(default_factory=list)
    column_highs: list = field(default_factory=list)
    column_lows: list = field(default_factory=list)

    # Indicator: Momentum Consistency on last N box deltas
    mc_lags: int = 5


class PnFBuilder:
    """Incremental P&F box builder. Extracted from live_neat_pf_unified.py lines 740-887.

    One instance per (pair, config). Maintains chart state and computes MC indicator.

    Warmup: ~500 M1 bars for 5-pip boxes, ~2000 M1 for 15-pip boxes.
    """

    MAX_BOX_HISTORY = 200
    MAX_COLUMNS = 50

    def __init__(self, box_size_pips: int, reversal: int, pip: float):
        self.box_size_pips = box_size_pips
        self.box_size = box_size_pips * pip
        self.reversal = reversal
        self.pip = pip
        self.state = PnFState()
        self.config_name = f"{box_size_pips}pip_rev{reversal}"

    def _snap_to_box(self, price: float) -> float:
        """Round price down to nearest box level."""
        return math.floor(price / self.box_size) * self.box_size

    def process_price(self, mid_price: float, timestamp_str: str = "") -> List[PnFBox]:
        """Process a new price, return list of new boxes created.

        Exact match of live_neat_pf_unified.py _process_price() logic.
        """
        pnf = self.state
        new_boxes = []
        bs = self.box_size
        rev = self.reversal

        # First price — initialize
        if pnf.direction == 0:
            pnf.current_level = self._snap_to_box(mid_price)
            pnf.direction = 1
            pnf.column_id = 0
            pnf.column_boxes_count = 1
            pnf.boxes_since_reversal = 0
            pnf.column_highs = [pnf.current_level]
            pnf.column_lows = [pnf.current_level]
            box = PnFBox(pnf.current_level, 1, 0, timestamp_str, mid_price)
            pnf.box_history.append(box)
            new_boxes.append(box)
            return new_boxes

        delta_boxes = int((mid_price - pnf.current_level) / bs)

        # Same direction continuation (rising)
        if pnf.direction == 1 and delta_boxes >= 1:
            for _ in range(delta_boxes):
                new_level = pnf.current_level + bs
                pnf.current_level = new_level
                pnf.column_boxes_count += 1
                pnf.boxes_since_reversal += 1
                if pnf.column_highs:
                    pnf.column_highs[-1] = max(pnf.column_highs[-1], new_level)
                box = PnFBox(new_level, 1, pnf.column_id, timestamp_str, mid_price)
                pnf.box_history.append(box)
                new_boxes.append(box)

        # Same direction continuation (falling)
        elif pnf.direction == -1 and delta_boxes <= -1:
            for _ in range(abs(delta_boxes)):
                new_level = pnf.current_level - bs
                pnf.current_level = new_level
                pnf.column_boxes_count += 1
                pnf.boxes_since_reversal += 1
                if pnf.column_lows:
                    pnf.column_lows[-1] = min(pnf.column_lows[-1], new_level)
                box = PnFBox(new_level, -1, pnf.column_id, timestamp_str, mid_price)
                pnf.box_history.append(box)
                new_boxes.append(box)

        # Reversal: was rising, now falling
        elif pnf.direction == 1 and delta_boxes <= -rev:
            pnf.direction = -1
            pnf.column_id += 1
            pnf.column_boxes_count = 0
            pnf.boxes_since_reversal = 0
            new_col_level = pnf.current_level - bs
            pnf.column_highs.append(new_col_level)
            pnf.column_lows.append(new_col_level)
            for _ in range(abs(delta_boxes)):
                new_level = pnf.current_level - bs
                pnf.current_level = new_level
                pnf.column_boxes_count += 1
                pnf.boxes_since_reversal += 1
                pnf.column_lows[-1] = min(pnf.column_lows[-1], new_level)
                box = PnFBox(new_level, -1, pnf.column_id, timestamp_str, mid_price)
                pnf.box_history.append(box)
                new_boxes.append(box)

        # Reversal: was falling, now rising
        elif pnf.direction == -1 and delta_boxes >= rev:
            pnf.direction = 1
            pnf.column_id += 1
            pnf.column_boxes_count = 0
            pnf.boxes_since_reversal = 0
            new_col_level = pnf.current_level + bs
            pnf.column_highs.append(new_col_level)
            pnf.column_lows.append(new_col_level)
            for _ in range(delta_boxes):
                new_level = pnf.current_level + bs
                pnf.current_level = new_level
                pnf.column_boxes_count += 1
                pnf.boxes_since_reversal += 1
                pnf.column_highs[-1] = max(pnf.column_highs[-1], new_level)
                box = PnFBox(new_level, 1, pnf.column_id, timestamp_str, mid_price)
                pnf.box_history.append(box)
                new_boxes.append(box)

        # Trim buffers
        if len(pnf.box_history) > self.MAX_BOX_HISTORY:
            pnf.box_history = pnf.box_history[-self.MAX_BOX_HISTORY:]
        if len(pnf.column_highs) > self.MAX_COLUMNS:
            pnf.column_highs = pnf.column_highs[-self.MAX_COLUMNS:]
        if len(pnf.column_lows) > self.MAX_COLUMNS:
            pnf.column_lows = pnf.column_lows[-self.MAX_COLUMNS:]

        return new_boxes

    def compute_mc(self, n_lags: int = 5) -> float:
        """Compute Momentum Consistency from last N box deltas.

        MC = (count of same-sign deltas) / n_lags, mapped to [-1, 1].
        Positive MC = consistent upward momentum, negative = downward.
        """
        boxes = self.state.box_history
        if len(boxes) < n_lags + 1:
            return 0.0

        deltas = []
        for i in range(-n_lags, 0):
            d = boxes[i].level - boxes[i - 1].level
            deltas.append(d)

        if not deltas:
            return 0.0

        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        return (pos - neg) / len(deltas)


# ─── H1 Zigzag S/R ────────────────────────────────────────────────────────

@dataclass
class ZigzagState:
    """State for H1 zigzag support/resistance computation."""
    support: float = 0.0
    resistance: float = 0.0
    zz_direction: int = 0       # 0=undecided, 1=up, -1=down
    running_high: float = 0.0
    running_low: float = 0.0
    h1_bars: list = field(default_factory=list)

    # H1 bar accumulation from S5
    current_hour: int = -1
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_close: float = 0.0


class ZigzagSR:
    """H1 zigzag support/resistance. Extracted from live_neat_pf_unified.py lines 889-999.

    Warmup: 200 H1 bars (~8 trading days).
    """

    def __init__(self, min_swing: float):
        self.min_swing = min_swing
        self.state = ZigzagState()

    def update_from_h1_bar(self, bar: dict):
        """Update zigzag from a completed H1 bar.

        Exact match of live_neat_pf_unified.py _update_zigzag() logic.
        """
        s = self.state
        hi = bar["high"]
        lo = bar["low"]
        ms = self.min_swing

        s.h1_bars.append(bar)
        if len(s.h1_bars) > 600:
            s.h1_bars = s.h1_bars[-500:]

        if s.running_high == 0 and s.running_low == 0:
            s.running_high = hi
            s.running_low = lo
            s.support = lo
            s.resistance = hi
            return

        if hi > s.running_high:
            s.running_high = hi
        if lo < s.running_low:
            s.running_low = lo

        if s.zz_direction == 0:
            if s.running_high - lo >= ms:
                s.resistance = s.running_high
                s.zz_direction = -1
                s.running_low = lo
            elif hi - s.running_low >= ms:
                s.support = s.running_low
                s.zz_direction = 1
                s.running_high = hi
        elif s.zz_direction == 1:
            if s.running_high - lo >= ms:
                s.resistance = s.running_high
                s.zz_direction = -1
                s.running_low = lo
        else:
            if hi - s.running_low >= ms:
                s.support = s.running_low
                s.zz_direction = 1
                s.running_high = hi

    def accumulate_s5(self, s5_bar: dict) -> Optional[dict]:
        """Accumulate S5 bar into H1. Returns completed H1 bar on hour boundary, else None.

        Exact match of live_neat_pf_unified.py _update_h1_from_s5() logic.
        """
        s = self.state
        ts = s5_bar["timestamp"]
        bar_hour = ts.hour if hasattr(ts, 'hour') else int(str(ts)[11:13])

        if s.current_hour == -1:
            s.current_hour = bar_hour
            s.bar_open = s5_bar["open"]
            s.bar_high = s5_bar["high"]
            s.bar_low = s5_bar["low"]
            s.bar_close = s5_bar["close"]
            return None

        if bar_hour != s.current_hour:
            completed = {
                "open": s.bar_open, "high": s.bar_high,
                "low": s.bar_low, "close": s.bar_close,
            }
            self.update_from_h1_bar(completed)

            s.current_hour = bar_hour
            s.bar_open = s5_bar["open"]
            s.bar_high = s5_bar["high"]
            s.bar_low = s5_bar["low"]
            s.bar_close = s5_bar["close"]
            return completed
        else:
            s.bar_high = max(s.bar_high, s5_bar["high"])
            s.bar_low = min(s.bar_low, s5_bar["low"])
            s.bar_close = s5_bar["close"]
            return None


# ─── ATR(14) ───────────────────────────────────────────────────────────────

class ATR:
    """Incremental ATR(14) computation.

    Warmup: 14 bars minimum.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self.prev_close: Optional[float] = None
        self._count: int = 0
        self._sum_tr: float = 0.0

    def update(self, bar: dict) -> float:
        """Update ATR with a new bar. Returns current ATR value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        if self.prev_close is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - self.prev_close), abs(lo - self.prev_close))

        self.prev_close = cl
        self._count += 1

        if self._count <= self.period:
            self._sum_tr += tr
            if self._count == self.period:
                self.value = self._sum_tr / self.period
        else:
            # Wilder smoothing
            self.value = (self.value * (self.period - 1) + tr) / self.period

        return self.value


# ─── ADX (Average Directional Index) ────────────────────────────────────────

class ADX:
    """Incremental ADX(14) computation (Wilder's method).

    Warmup: 28 bars (2×period) minimum.
    Output: self.value (ADX, 0-100), self.regime ({-1, 0, +1}).
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self.regime: float = 0.0  # -1=weak (<20), 0=mid (20-40), +1=strong (>40)
        self.prev_close: Optional[float] = None
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self._count: int = 0
        self._atr_s: float = 0.0
        self._pdm_s: float = 0.0
        self._mdm_s: float = 0.0
        self._adx_sum: float = 0.0
        self._adx_count: int = 0
        self._dx_buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update ADX with a new bar. Returns current ADX value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        if self.prev_close is None:
            self.prev_close = cl
            self.prev_high = hi
            self.prev_low = lo
            return 0.0

        # True Range
        tr = max(hi - lo, abs(hi - self.prev_close), abs(lo - self.prev_close))

        # Directional movement
        h_diff = hi - self.prev_high
        l_diff = self.prev_low - lo
        plus_dm = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
        minus_dm = l_diff if l_diff > h_diff and l_diff > 0 else 0.0

        self.prev_close = cl
        self.prev_high = hi
        self.prev_low = lo
        self._count += 1

        if self._count <= self.period:
            self._atr_s += tr
            self._pdm_s += plus_dm
            self._mdm_s += minus_dm
            if self._count == self.period:
                # First smoothed values
                pass
        else:
            self._atr_s = self._atr_s - self._atr_s / self.period + tr
            self._pdm_s = self._pdm_s - self._pdm_s / self.period + plus_dm
            self._mdm_s = self._mdm_s - self._mdm_s / self.period + minus_dm

        if self._count >= self.period and self._atr_s > 0:
            pdi = 100.0 * self._pdm_s / self._atr_s
            mdi = 100.0 * self._mdm_s / self._atr_s
            di_sum = pdi + mdi
            dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0

            self._dx_buf.append(dx)
            if len(self._dx_buf) == self.period:
                # First ADX = mean of first 'period' DX values
                self.value = sum(self._dx_buf) / self.period
            elif len(self._dx_buf) > self.period:
                # Wilder smoothing
                self.value = (self.value * (self.period - 1) + dx) / self.period

        # Quantize to regime
        if self.value < 20:
            self.regime = -1.0
        elif self.value > 40:
            self.regime = 1.0
        else:
            self.regime = 0.0

        return self.value


class VolRegime:
    """Volatility regime based on ATR% (ATR/close) rolling tercile.

    Warmup: 200 H1 bars.
    Output: self.regime ({-1, 0, +1} = low/mid/high vol).
    """

    def __init__(self, window: int = 200):
        self.window = window
        self.regime: float = 0.0
        self._atr = ATR(period=14)
        self._atr_pct_buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current regime."""
        atr_val = self._atr.update(bar)
        close = bar["close"]
        if close > 0 and atr_val > 0:
            atr_pct = atr_val / close * 100.0
        else:
            atr_pct = 0.0

        self._atr_pct_buf.append(atr_pct)
        if len(self._atr_pct_buf) > self.window:
            self._atr_pct_buf = self._atr_pct_buf[-self.window:]

        if len(self._atr_pct_buf) >= 20:
            import numpy as np
            arr = np.array(self._atr_pct_buf)
            p33 = np.percentile(arr, 33)
            p67 = np.percentile(arr, 67)
            if atr_pct < p33:
                self.regime = -1.0
            elif atr_pct > p67:
                self.regime = 1.0
            else:
                self.regime = 0.0

        return self.regime


# ─── Chandelier Exit ─────────────────────────────────────────────────────

class ChandelierExit:
    """ATR-based dynamic stop levels (Chandelier Exit).

    Tracks highest high / lowest low over `period` bars and offsets by
    multiplier * ATR to produce trailing stop levels for long and short.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 14, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.long_exit: float = 0.0
        self.short_exit: float = 0.0
        self._atr = ATR(period=period)
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns long_exit level."""
        hi = bar["high"]
        lo = bar["low"]

        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.period:
            self._highs = self._highs[-self.period:]
            self._lows = self._lows[-self.period:]

        atr_val = self._atr.update(bar)
        self._count += 1

        if self._count >= self.period and atr_val > 0:
            highest_high = max(self._highs)
            lowest_low = min(self._lows)
            self.long_exit = highest_high - self.multiplier * atr_val
            self.short_exit = lowest_low + self.multiplier * atr_val

        return self.long_exit


# ─── Kaufman Efficiency Ratio ────────────────────────────────────────────

class KaufmanER:
    """Kaufman Efficiency Ratio.

    ER = abs(close - close[period]) / sum(abs(close[i] - close[i-1]))
    Range [0, 1]: 1.0 = perfectly trending, 0.0 = choppy.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 10):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current ER value."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period + 1:
            self._closes = self._closes[-(self.period + 1):]

        if len(self._closes) <= self.period:
            return 0.0

        direction = abs(self._closes[-1] - self._closes[0])
        volatility = sum(
            abs(self._closes[i] - self._closes[i - 1])
            for i in range(1, len(self._closes))
        )

        if volatility > 0:
            self.value = direction / volatility
        else:
            self.value = 0.0

        return self.value


# ─── Z-Score ─────────────────────────────────────────────────────────────

class ZScore:
    """Standard Z-Score: (close - SMA) / StdDev.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current Z-Score."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period:
            self._closes = self._closes[-self.period:]

        if len(self._closes) < self.period:
            return 0.0

        mean = sum(self._closes) / self.period
        variance = sum((x - mean) ** 2 for x in self._closes) / self.period
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std > 0:
            self.value = (cl - mean) / std
        else:
            self.value = 0.0

        return self.value


# ─── Linear Regression Slope ─────────────────────────────────────────────

class LinearRegressionSlope:
    """Least-squares regression slope of close over `period` bars, normalized.

    Output: self.value (slope), self.regime ({-1, 0, +1} based on sign).

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self.regime: float = 0.0
        self._closes: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current slope."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period:
            self._closes = self._closes[-self.period:]

        n = len(self._closes)
        if n < self.period:
            return 0.0

        # Least-squares slope: slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        for i in range(n):
            x = float(i)
            y = self._closes[i]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x

        denom = n * sum_x2 - sum_x * sum_x
        if denom != 0:
            self.value = (n * sum_xy - sum_x * sum_y) / denom
        else:
            self.value = 0.0

        # Regime based on sign
        if self.value > 0:
            self.regime = 1.0
        elif self.value < 0:
            self.regime = -1.0
        else:
            self.regime = 0.0

        return self.value


# ─── Stochastic %K / %D ─────────────────────────────────────────────────

class Stochastic:
    """Stochastic oscillator (%K and %D).

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)

    Warmup: `k_period` bars for %K, `k_period + d_period` for %D.
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.k: float = 0.0
        self.d: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._k_buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current %K."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.k_period:
            self._highs = self._highs[-self.k_period:]
            self._lows = self._lows[-self.k_period:]

        if len(self._highs) < self.k_period:
            return 0.0

        highest = max(self._highs)
        lowest = min(self._lows)
        hl_range = highest - lowest

        if hl_range > 0:
            self.k = 100.0 * (cl - lowest) / hl_range
        else:
            self.k = 50.0  # Midpoint when range is zero

        self._k_buf.append(self.k)
        if len(self._k_buf) > self.d_period:
            self._k_buf = self._k_buf[-self.d_period:]

        if len(self._k_buf) >= self.d_period:
            self.d = sum(self._k_buf) / self.d_period

        return self.k


# ─── MACD ────────────────────────────────────────────────────────────────

class MACD:
    """MACD (Moving Average Convergence Divergence).

    macd_line = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line

    Warmup: `slow` bars for macd_line, `slow + signal` for signal_line.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.macd_line: float = 0.0
        self.signal_line: float = 0.0
        self.histogram: float = 0.0
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._ema_signal: Optional[float] = None
        self._alpha_fast: float = 2.0 / (fast + 1)
        self._alpha_slow: float = 2.0 / (slow + 1)
        self._alpha_signal: float = 2.0 / (signal + 1)
        self._count: int = 0
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0
        self._macd_count: int = 0
        self._sum_signal: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current MACD line value."""
        cl = bar["close"]
        self._count += 1

        # Bootstrap fast EMA
        if self._count <= self.fast:
            self._sum_fast += cl
            if self._count == self.fast:
                self._ema_fast = self._sum_fast / self.fast
        elif self._ema_fast is not None:
            self._ema_fast = self._alpha_fast * cl + (1 - self._alpha_fast) * self._ema_fast

        # Bootstrap slow EMA
        if self._count <= self.slow:
            self._sum_slow += cl
            if self._count == self.slow:
                self._ema_slow = self._sum_slow / self.slow
        elif self._ema_slow is not None:
            self._ema_slow = self._alpha_slow * cl + (1 - self._alpha_slow) * self._ema_slow

        # Compute MACD line once both EMAs are ready
        if self._ema_fast is not None and self._ema_slow is not None:
            self.macd_line = self._ema_fast - self._ema_slow
            self._macd_count += 1

            # Bootstrap signal EMA
            if self._macd_count <= self.signal:
                self._sum_signal += self.macd_line
                if self._macd_count == self.signal:
                    self._ema_signal = self._sum_signal / self.signal
            elif self._ema_signal is not None:
                self._ema_signal = (self._alpha_signal * self.macd_line +
                                    (1 - self._alpha_signal) * self._ema_signal)

            if self._ema_signal is not None:
                self.signal_line = self._ema_signal
                self.histogram = self.macd_line - self.signal_line

        return self.macd_line


# ─── CCI (Commodity Channel Index) ──────────────────────────────────────

class CCI:
    """Commodity Channel Index.

    TP = (high + low + close) / 3
    CCI = (TP - SMA(TP, period)) / (0.015 * mean_deviation)

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._tp_buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current CCI value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        tp = (hi + lo + cl) / 3.0

        self._tp_buf.append(tp)
        if len(self._tp_buf) > self.period:
            self._tp_buf = self._tp_buf[-self.period:]

        if len(self._tp_buf) < self.period:
            return 0.0

        mean_tp = sum(self._tp_buf) / self.period
        mean_dev = sum(abs(x - mean_tp) for x in self._tp_buf) / self.period

        if mean_dev > 0:
            self.value = (tp - mean_tp) / (0.015 * mean_dev)
        else:
            self.value = 0.0

        return self.value


# ─── Aroon ───────────────────────────────────────────────────────────────

class Aroon:
    """Aroon Up/Down indicator.

    aroon_up = 100 * (period - bars_since_highest_high) / period
    aroon_down = 100 * (period - bars_since_lowest_low) / period

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 25):
        self.period = period
        self.up: float = 0.0
        self.down: float = 0.0
        self.oscillator: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current Aroon Up value."""
        hi = bar["high"]
        lo = bar["low"]

        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.period + 1:
            self._highs = self._highs[-(self.period + 1):]
            self._lows = self._lows[-(self.period + 1):]

        n = len(self._highs)
        if n <= self.period:
            return 0.0

        # Find bars since highest high and lowest low within the window
        window_highs = self._highs[-(self.period + 1):]
        window_lows = self._lows[-(self.period + 1):]

        max_val = window_highs[0]
        max_idx = 0
        min_val = window_lows[0]
        min_idx = 0
        for i in range(1, len(window_highs)):
            if window_highs[i] >= max_val:
                max_val = window_highs[i]
                max_idx = i
            if window_lows[i] <= min_val:
                min_val = window_lows[i]
                min_idx = i

        bars_since_high = self.period - max_idx
        bars_since_low = self.period - min_idx

        self.up = 100.0 * (self.period - bars_since_high) / self.period
        self.down = 100.0 * (self.period - bars_since_low) / self.period
        self.oscillator = self.up - self.down

        return self.up


# ─── SuperTrend ─────────────────────────────────────────────────────────

class SuperTrend:
    """ATR-based trend-following overlay (SuperTrend).

    Upper band = (H+L)/2 + multiplier * ATR
    Lower band = (H+L)/2 - multiplier * ATR
    Direction flips when close crosses the active band.

    Warmup: `period` bars (ATR warmup).
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.value: float = 0.0       # current SuperTrend level
        self.direction: int = 1       # +1 = bullish, -1 = bearish
        self._atr = ATR(period=period)
        self._prev_upper: float = 0.0
        self._prev_lower: float = 0.0
        self._prev_close: Optional[float] = None
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current SuperTrend level."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        atr_val = self._atr.update(bar)
        self._count += 1

        if self._count <= self.period:
            self._prev_close = cl
            self.value = cl
            return self.value

        median = (hi + lo) / 2.0
        upper = median + self.multiplier * atr_val
        lower = median - self.multiplier * atr_val

        # Clamp bands to prevent widening against the trend
        if lower > self._prev_lower or (self._prev_close is not None and self._prev_close < self._prev_lower):
            pass  # use new lower
        else:
            lower = self._prev_lower

        if upper < self._prev_upper or (self._prev_close is not None and self._prev_close > self._prev_upper):
            pass  # use new upper
        else:
            upper = self._prev_upper

        # Direction logic
        if self.direction == 1:
            # Bullish: tracking lower band
            if cl < lower:
                self.direction = -1
                self.value = upper
            else:
                self.value = lower
        else:
            # Bearish: tracking upper band
            if cl > upper:
                self.direction = 1
                self.value = lower
            else:
                self.value = upper

        self._prev_upper = upper
        self._prev_lower = lower
        self._prev_close = cl

        return self.value


# ─── Parabolic SAR ──────────────────────────────────────────────────────

class ParabolicSAR:
    """Parabolic Stop and Reverse (Wilder's algorithm).

    Warmup: 2 bars.
    """

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.value: float = 0.0       # current SAR level
        self.direction: int = 1       # +1 = bullish, -1 = bearish
        self._af: float = af_start
        self._ep: float = 0.0         # extreme point
        self._count: int = 0
        self._prev_high: float = 0.0
        self._prev_low: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current SAR level."""
        hi = bar["high"]
        lo = bar["low"]
        self._count += 1

        if self._count == 1:
            self._prev_high = hi
            self._prev_low = lo
            self.value = lo  # start bullish, SAR at first low
            self._ep = hi
            self.direction = 1
            return self.value

        if self._count == 2:
            if hi > self._prev_high:
                self.direction = 1
                self.value = self._prev_low
                self._ep = hi
            else:
                self.direction = -1
                self.value = self._prev_high
                self._ep = lo
            self._af = self.af_start
            self._prev_high = hi
            self._prev_low = lo
            return self.value

        # Compute new SAR
        sar = self.value + self._af * (self._ep - self.value)

        if self.direction == 1:
            # Bullish: SAR must not be above prior two lows
            sar = min(sar, self._prev_low, lo)

            if lo < sar:
                # Reversal to bearish
                self.direction = -1
                sar = self._ep  # SAR = previous extreme point
                self._ep = lo
                self._af = self.af_start
            else:
                if hi > self._ep:
                    self._ep = hi
                    self._af = min(self._af + self.af_step, self.af_max)
        else:
            # Bearish: SAR must not be below prior two highs
            sar = max(sar, self._prev_high, hi)

            if hi > sar:
                # Reversal to bullish
                self.direction = 1
                sar = self._ep  # SAR = previous extreme point
                self._ep = hi
                self._af = self.af_start
            else:
                if lo < self._ep:
                    self._ep = lo
                    self._af = min(self._af + self.af_step, self.af_max)

        self.value = sar
        self._prev_high = hi
        self._prev_low = lo

        return self.value


# ─── Donchian Channel ───────────────────────────────────────────────────

class DonchianChannel:
    """Donchian Channel (highest high / lowest low over period).

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.upper: float = 0.0
        self.lower: float = 0.0
        self.mid: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current mid level."""
        hi = bar["high"]
        lo = bar["low"]

        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.period:
            self._highs = self._highs[-self.period:]
            self._lows = self._lows[-self.period:]

        self.upper = max(self._highs)
        self.lower = min(self._lows)
        self.mid = (self.upper + self.lower) / 2.0

        return self.mid


# ─── Keltner Channel ───────────────────────────────────────────────────

class KeltnerChannel:
    """Keltner Channel: EMA +/- ATR multiple.

    Warmup: max(ema_period, atr_period) bars.
    """

    def __init__(self, ema_period: int = 20, atr_mult: float = 1.5, atr_period: int = 14):
        self.ema_period = ema_period
        self.atr_mult = atr_mult
        self.upper: float = 0.0
        self.lower: float = 0.0
        self.mid: float = 0.0
        self._atr = ATR(period=atr_period)
        self._ema: Optional[float] = None
        self._alpha: float = 2.0 / (ema_period + 1)
        self._count: int = 0
        self._sum: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current mid (EMA) level."""
        cl = bar["close"]
        atr_val = self._atr.update(bar)
        self._count += 1

        # Bootstrap EMA
        if self._count <= self.ema_period:
            self._sum += cl
            if self._count == self.ema_period:
                self._ema = self._sum / self.ema_period
        elif self._ema is not None:
            self._ema = self._alpha * cl + (1.0 - self._alpha) * self._ema

        if self._ema is not None:
            self.mid = self._ema
            self.upper = self._ema + self.atr_mult * atr_val
            self.lower = self._ema - self.atr_mult * atr_val

        return self.mid


# ─── Heiken Ashi ────────────────────────────────────────────────────────

class HeikenAshi:
    """Heiken Ashi modified candlestick calculation.

    HA_Close = (O+H+L+C)/4
    HA_Open  = (prev_HA_Open + prev_HA_Close)/2
    HA_High  = max(H, HA_Open, HA_Close)
    HA_Low   = min(L, HA_Open, HA_Close)

    Warmup: 1 bar.
    """

    def __init__(self):
        self.ha_open: float = 0.0
        self.ha_close: float = 0.0
        self.ha_high: float = 0.0
        self.ha_low: float = 0.0
        self._initialized: bool = False

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current HA close."""
        o = bar["open"]
        h = bar["high"]
        l = bar["low"]
        c = bar["close"]

        ha_close = (o + h + l + c) / 4.0

        if not self._initialized:
            ha_open = (o + c) / 2.0
            self._initialized = True
        else:
            ha_open = (self.ha_open + self.ha_close) / 2.0

        self.ha_open = ha_open
        self.ha_close = ha_close
        self.ha_high = max(h, ha_open, ha_close)
        self.ha_low = min(l, ha_open, ha_close)

        return self.ha_close


# ─── Ichimoku Cloud ─────────────────────────────────────────────────────

class Ichimoku:
    """Ichimoku Cloud (Kinko Hyo).

    Tenkan-sen  = (highest_high + lowest_low) / 2 over tenkan period
    Kijun-sen   = same over kijun period
    Senkou A    = (Tenkan + Kijun) / 2
    Senkou B    = (highest_high + lowest_low) / 2 over senkou_b period

    Note: In live incremental mode, senkou spans are computed without the
    traditional 26-period forward displacement (since we don't have future bars).

    Warmup: `senkou_b` bars (largest lookback).
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.senkou_b_period = senkou_b
        self.tenkan: float = 0.0
        self.kijun: float = 0.0
        self.senkou_a: float = 0.0
        self.senkou_b: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []

    @staticmethod
    def _hl_mid(highs: List[float], lows: List[float]) -> float:
        """(highest_high + lowest_low) / 2."""
        if not highs or not lows:
            return 0.0
        return (max(highs) + min(lows)) / 2.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current Tenkan-sen value."""
        hi = bar["high"]
        lo = bar["low"]

        self._highs.append(hi)
        self._lows.append(lo)

        # Keep enough history for the longest lookback
        max_period = self.senkou_b_period
        if len(self._highs) > max_period:
            self._highs = self._highs[-max_period:]
            self._lows = self._lows[-max_period:]

        n = len(self._highs)

        # Tenkan-sen
        if n >= self.tenkan_period:
            p = self.tenkan_period
            self.tenkan = self._hl_mid(self._highs[-p:], self._lows[-p:])
        # Kijun-sen
        if n >= self.kijun_period:
            p = self.kijun_period
            self.kijun = self._hl_mid(self._highs[-p:], self._lows[-p:])
        # Senkou Span A
        self.senkou_a = (self.tenkan + self.kijun) / 2.0
        # Senkou Span B
        if n >= self.senkou_b_period:
            p = self.senkou_b_period
            self.senkou_b = self._hl_mid(self._highs[-p:], self._lows[-p:])

        return self.tenkan


# ─── MTF-MC (Multi-Timeframe Momentum Consistency) ────────────────────────

class MTFMC:
    """Multi-timeframe EMA consensus indicator.
    Extracted from live_neat_pf_unified.py _compute_mtf_mc_features() lines 1001-1098.

    Maintains a rolling S5 buffer and resamples into 9 timeframes on the fly.
    Computes MC(D) and MC(dD) where D=EMA3-EMA5, weighted by log2(tf/5s).

    Warmup: 5000 S5 bars for H1 timeframe, 100 S5 minimum for basic operation.
    """

    TFS_SEC = [5, 10, 30, 60, 120, 300, 600, 1800, 3600]
    N_LAGS = 5
    MAX_BUFFER = 200000  # No truncation — prevents train/live mismatch

    def __init__(self, pip: float):
        self.pip = pip
        self.s5_buffer: List[float] = []
        self._log_weights = [math.log2(max(s / 5, 1)) + 1 for s in self.TFS_SEC]

    def append_s5(self, mid_price: float):
        """Append an S5 close price to the buffer."""
        self.s5_buffer.append(mid_price / self.pip)
        if len(self.s5_buffer) > self.MAX_BUFFER:
            self.s5_buffer = self.s5_buffer[-self.MAX_BUFFER:]

    def compute(self) -> Tuple[float, float]:
        """Compute MC(D) and MC(dD) across 9 timeframes.

        Returns (mc_d, mc_dd). Both in range [-1, 1].
        Returns (0.0, 0.0) if insufficient data.
        """
        buf = self.s5_buffer
        n = len(buf)

        if n < 100:
            return 0.0, 0.0

        wa_d = 0.0
        wa_dd = 0.0
        tw = 0.0

        for idx, tf_sec in enumerate(self.TFS_SEC):
            w = self._log_weights[idx]
            bars_per = max(tf_sec // 5, 1)
            n_tf = n // bars_per
            if n_tf < self.N_LAGS + 2:
                continue

            n_need = self.N_LAGS + 2
            tf_vals = []
            for i in range(max(0, n_tf - n_need), n_tf):
                idx_s5 = (i + 1) * bars_per - 1
                if idx_s5 < n:
                    tf_vals.append(buf[idx_s5])

            if len(tf_vals) < self.N_LAGS + 2:
                continue

            # EMA(3) and EMA(5)
            e3 = tf_vals[0]
            e5 = tf_vals[0]
            alpha3 = 2.0 / 4.0
            alpha5 = 2.0 / 6.0
            d_vals = []
            for v in tf_vals:
                e3 = alpha3 * v + (1 - alpha3) * e3
                e5 = alpha5 * v + (1 - alpha5) * e5
                d_vals.append(e3 - e5)

            # MC on D
            if len(d_vals) >= self.N_LAGS + 1:
                changes = [d_vals[j] - d_vals[j - 1] for j in range(-self.N_LAGS, 0)]
                pos = sum(1 for c in changes if c > 0)
                neg = sum(1 for c in changes if c < 0)
                mc_d = (pos - neg) / self.N_LAGS
                wa_d += w * mc_d

            # MC on dD (acceleration)
            dd_vals = [d_vals[j] - d_vals[j - 1] for j in range(1, len(d_vals))]
            if len(dd_vals) >= self.N_LAGS + 1:
                changes = [dd_vals[j] - dd_vals[j - 1] for j in range(-self.N_LAGS, 0)]
                pos = sum(1 for c in changes if c > 0)
                neg = sum(1 for c in changes if c < 0)
                mc_dd = (pos - neg) / self.N_LAGS
                wa_dd += w * mc_dd

            tw += w

        if tw > 0:
            return wa_d / tw, wa_dd / tw
        return 0.0, 0.0


# ─── Asian Range ───────────────────────────────────────────────────────────

class AsianRange:
    """Asian session range (00:00-07:00 UTC) from H1 bars.

    Warmup: 24 H1 bars (1 day).
    """

    def __init__(self):
        self.high: float = 0.0
        self.low: float = 0.0
        self.mid: float = 0.0
        self._session_bars: List[dict] = []

    def update_from_h1(self, bar: dict, bar_hour: int):
        """Update Asian range from a completed H1 bar.

        Asian session = 00:00-07:00 UTC. On hour 7 (London open), finalize range.
        """
        if 0 <= bar_hour < 7:
            self._session_bars.append(bar)
        elif bar_hour == 7 and self._session_bars:
            # Finalize Asian range
            self.high = max(b["high"] for b in self._session_bars)
            self.low = min(b["low"] for b in self._session_bars)
            self.mid = (self.high + self.low) / 2
            self._session_bars = []
        elif bar_hour >= 7:
            # Reset for next session at midnight
            if bar_hour == 0:
                self._session_bars = [bar]


# ─── Kalman Currency Strength ─────────────────────────────────────────────

class KalmanStrength:
    """Kalman filter for currency strength decomposition.
    Extracted from live_strength_unified.py and live_neat_pf_unified.py.

    Decomposes 10-pair log returns into 6-8 currency strengths.
    Warmup: 200 H1 bars across all 10+ pairs.
    """

    def __init__(self, currencies: List[str], pairs: List[Tuple[str, str, str]],
                 process_noise: float = 1e-5, measurement_noise: float = 1e-3):
        """
        Args:
            currencies: List of currency codes, e.g., ["EUR", "USD", "GBP", ...]
            pairs: List of (pair_name, base, quote), e.g., [("EUR_USD", "EUR", "USD"), ...]
        """
        import numpy as np

        self.currencies = currencies
        self.pairs = pairs
        n_cur = len(currencies)
        n_pairs = len(pairs)

        # State: strength per currency
        self.x = np.zeros(n_cur)
        self.P = np.eye(n_cur) * 0.01

        # Process/measurement noise
        self.Q = np.eye(n_cur) * process_noise
        self.R = np.eye(n_pairs) * measurement_noise

        # Observation matrix: H maps currencies to pair log returns
        # pair_return = base_strength - quote_strength
        self.H = np.zeros((n_pairs, n_cur))
        cur_idx = {c: i for i, c in enumerate(currencies)}
        for j, (_, base, quote) in enumerate(pairs):
            if base in cur_idx:
                self.H[j, cur_idx[base]] = 1.0
            if quote in cur_idx:
                self.H[j, cur_idx[quote]] = -1.0

        # Track last closes for log return computation
        self.last_closes: Dict[str, float] = {}
        self.warmup_done = False

    def update(self, pair_closes: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Update Kalman filter with new closes for all pairs.

        Args:
            pair_closes: {"EUR_USD": 1.0850, "USD_JPY": 150.20, ...}

        Returns:
            Dict of currency strengths, or None if still warming up.
        """
        import numpy as np

        if not self.last_closes:
            self.last_closes = pair_closes.copy()
            return None  # Need at least 2 updates for log returns

        # Compute log returns
        log_returns = []
        for pair_name, _, _ in self.pairs:
            prev = self.last_closes.get(pair_name, 0)
            curr = pair_closes.get(pair_name, 0)
            if prev > 0 and curr > 0:
                log_returns.append(math.log(curr / prev))
            else:
                log_returns.append(0.0)

        self.last_closes = pair_closes.copy()
        y = np.array(log_returns)

        # Kalman predict + update
        I = np.eye(len(self.currencies))
        x_pred = self.x.copy()
        P_pred = self.P + self.Q
        innovation = y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ innovation
        self.P = (I - K @ self.H) @ P_pred
        self.x -= self.x.mean()  # Zero-mean constraint

        return dict(zip(self.currencies, self.x.tolist()))

    def get_ranks(self) -> Dict[str, int]:
        """Get currency rankings (1=strongest, N=weakest)."""
        sorted_curs = sorted(
            zip(self.currencies, self.x.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return {cur: rank + 1 for rank, (cur, _) in enumerate(sorted_curs)}


# ─── ASI-MC: Accumulative Swing Index → MC(D)/MC(dD) ─────────────────────

class ASIMC:
    """ASI-based momentum consensus indicator.

    Buffers M5 OHLC, computes ASI (Wilder), then runs standard
    MC(D)/MC(dD) on the ASI series (no SMA5 — EMA in MC handles smoothing).

    Matches the JIT computation in lib/asi_indicator.py but as a stateful
    class for the curator. Computes on every M5 bar update.

    Warmup: 100 M5 bars minimum.
    """

    MAX_BUFFER = 200000  # Keep all M5 bars — truncation causes train/live mismatch

    def __init__(self):
        self.m5_o: List[float] = []
        self.m5_h: List[float] = []
        self.m5_l: List[float] = []
        self.m5_c: List[float] = []

    def append_m5(self, o: float, h: float, l: float, c: float):
        """Append an M5 OHLC bar."""
        self.m5_o.append(o)
        self.m5_h.append(h)
        self.m5_l.append(l)
        self.m5_c.append(c)
        if len(self.m5_o) > self.MAX_BUFFER:
            self.m5_o = self.m5_o[-self.MAX_BUFFER:]
            self.m5_h = self.m5_h[-self.MAX_BUFFER:]
            self.m5_l = self.m5_l[-self.MAX_BUFFER:]
            self.m5_c = self.m5_c[-self.MAX_BUFFER:]

    def compute(self) -> Tuple[float, float]:
        """Compute MC(D) and MC(dD) on ASI series.

        Returns (asi_mc_d, asi_mc_dd). Both in range [-1, 1].
        Returns (0.0, 0.0) if insufficient data.
        """
        n = len(self.m5_o)
        if n < 100:
            return 0.0, 0.0

        import numpy as np
        from lib.asi_indicator import compute_asi, compute_mc_on_series, TF_BARS_S5, TF_WEIGHTS, N_TFS

        o = np.array(self.m5_o, dtype=np.float64)
        h = np.array(self.m5_h, dtype=np.float64)
        l = np.array(self.m5_l, dtype=np.float64)
        c = np.array(self.m5_c, dtype=np.float64)

        from lib.asi_indicator import sma_jit
        asi = compute_asi(o, h, l, c, n)
        smooth = sma_jit(asi, 5, n)
        mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)

        return float(mc_d[-1]), float(mc_dd[-1])


# ─── SwingStructure: TopsBots SB-A/SB-P + ERP + HHHL ─────────────────────

class SwingStructure:
    """TopsBots swing structure indicators (curator-side stateful class).

    Buffers M5 OHLC, computes Wilder ASI internally, then runs the identical
    TopsBots 3-stage algorithm from lib/swing_indicators.py on both ASI and
    price series.

    This class is the single source of truth for swing indicators — used by
    both the live curator and the offline export script so training and live
    data are always generated by identical code.

    Returns at each call to compute():
      sb_a      : ASI structural breakout state {-1,-0.5,0,+0.5,+1}
      erp_a     : Extended Range Position on ASI (unclamped continuous)
      hh_asi    : higher_highs binary {0.0, 1.0} on ASI TopsBots swings
      hl_asi    : higher_lows  binary {0.0, 1.0} on ASI TopsBots swings
      erp_p     : Extended Range Position on price (unclamped continuous)
      hh_price  : higher_highs binary {0.0, 1.0} on price TopsBots swings
      hl_price  : higher_lows  binary {0.0, 1.0} on price TopsBots swings

    Encoding (for NEAT inputs):
      SB-A: ordinal {-1,-0.5,0,+0.5,+1} — lossless monotone (preserves ordering)
      ERP:  continuous unclamped — > 1.0 = above HSP, < 0.0 = below LSP
      HH/HL: 2 independent binaries (NOT ordinal — HHLL/LHHL are non-directional)
             hh=1,hl=1 = HHHL bull  | hh=0,hl=0 = LHLL bear
             hh=1,hl=0 = expansion  | hh=0,hl=1 = contraction

    Warmup: ~50-100 M5 bars to establish first confirmed HSP+LSP pair.
    Returns all 0.0 until two confirmed swings of each type exist.
    """

    MAX_BUFFER = 200000  # No truncation — same policy as ASIMC

    def __init__(self):
        self.m5_o: List[float] = []
        self.m5_h: List[float] = []
        self.m5_l: List[float] = []
        self.m5_c: List[float] = []

    def append_m5(self, o: float, h: float, l: float, c: float):
        """Append one M5 OHLC bar."""
        self.m5_o.append(o)
        self.m5_h.append(h)
        self.m5_l.append(l)
        self.m5_c.append(c)
        if len(self.m5_o) > self.MAX_BUFFER:
            self.m5_o = self.m5_o[-self.MAX_BUFFER:]
            self.m5_h = self.m5_h[-self.MAX_BUFFER:]
            self.m5_l = self.m5_l[-self.MAX_BUFFER:]
            self.m5_c = self.m5_c[-self.MAX_BUFFER:]

    # Window for delta (velocity) features: 12 M5 bars = 1 hour
    ERP_DELTA_WINDOW = 12

    def compute(self) -> dict:
        """Compute all swing structure features from buffered M5 bars.

        Returns dict with keys:
          sb_a, erp_a, hh_asi, hl_asi, erp_p, hh_price, hl_price,
          d_erp_p, d_erp_a

        d_erp_p / d_erp_a: 1-hour rate of change in range position.
          Positive = price gaining position within swing range (ascending).
          Negative = price losing position (reversal in progress).
          At 08:09 EUR/JPY reversal: d_erp_p ≈ -0.49 (was >1.0, now 0.56).

        All values are floats. Returns all 0.0 if insufficient data (<20 bars).
        """
        import numpy as np
        from lib.asi_indicator import compute_asi
        from lib.swing_indicators import compute_all_swing_features

        _zero = {"sb_a": 0.0, "erp_a": 0.0, "hh_asi": 0.0, "hl_asi": 0.0,
                 "erp_p": 0.0, "hh_price": 0.0, "hl_price": 0.0,
                 "d_erp_p": 0.0, "d_erp_a": 0.0}

        n = len(self.m5_o)
        if n < 20:
            return _zero

        o = np.array(self.m5_o, dtype=np.float64)
        h = np.array(self.m5_h, dtype=np.float64)
        l = np.array(self.m5_l, dtype=np.float64)
        c = np.array(self.m5_c, dtype=np.float64)

        asi = compute_asi(o, h, l, c, n)
        feats = compute_all_swing_features(o, h, l, c, asi)

        def _last(arr):
            for i in range(len(arr) - 1, -1, -1):
                if not np.isnan(arr[i]):
                    return float(arr[i])
            return 0.0

        def _delta(arr, w):
            """erp[t] - erp[t-w], using last valid values at each position."""
            cur = _last(arr)
            if n <= w:
                return 0.0
            prev = _last(arr[:n - w])
            return cur - prev

        erp_a_arr = feats["erp_a"]
        erp_p_arr = feats["erp_p"]

        return {
            "sb_a":      _last(feats["sb_a"]),
            "erp_a":     _last(erp_a_arr),
            "hh_asi":    _last(feats["hh_asi"]),
            "hl_asi":    _last(feats["hl_asi"]),
            "erp_p":     _last(erp_p_arr),
            "hh_price":  _last(feats["hh_price"]),
            "hl_price":  _last(feats["hl_price"]),
            "d_erp_p":   _delta(erp_p_arr, self.ERP_DELTA_WINDOW),
            "d_erp_a":   _delta(erp_a_arr, self.ERP_DELTA_WINDOW),
        }


# ─── Williams %R ──────────────────────────────────────────────────────────

class WilliamsR:
    """Williams %R oscillator.

    Range: [-100, 0]. Oversold < -80, Overbought > -20.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current %R value."""
        self._highs.append(bar["high"])
        self._lows.append(bar["low"])
        if len(self._highs) > self.period:
            self._highs.pop(0)
            self._lows.pop(0)
        self._count += 1

        if self._count < self.period:
            self.value = 0.0
            return self.value

        hh = max(self._highs)
        ll = min(self._lows)
        rng = hh - ll
        if rng == 0:
            self.value = 0.0
        else:
            self.value = -100.0 * (hh - bar["close"]) / rng
        return self.value


# ─── Money Flow Index ─────────────────────────────────────────────────────

class MFI:
    """Money Flow Index (requires volume).

    Range: [0, 100]. Oversold < 20, Overbought > 80.
    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self._prev_tp: Optional[float] = None
        self._pos_flows: List[float] = []
        self._neg_flows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar (must include 'volume' key). Returns MFI."""
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        vol = bar.get("volume", 0.0)
        mf = tp * vol

        self._count += 1

        if self._prev_tp is None:
            self._prev_tp = tp
            self.value = 0.0
            return self.value

        if tp > self._prev_tp:
            self._pos_flows.append(mf)
            self._neg_flows.append(0.0)
        elif tp < self._prev_tp:
            self._pos_flows.append(0.0)
            self._neg_flows.append(mf)
        else:
            self._pos_flows.append(0.0)
            self._neg_flows.append(0.0)

        if len(self._pos_flows) > self.period:
            self._pos_flows.pop(0)
            self._neg_flows.pop(0)

        self._prev_tp = tp

        if self._count < self.period + 1:
            self.value = 0.0
            return self.value

        pos_sum = sum(self._pos_flows)
        neg_sum = sum(self._neg_flows)
        if neg_sum == 0:
            self.value = 100.0
        else:
            ratio = pos_sum / neg_sum
            self.value = 100.0 - 100.0 / (1.0 + ratio)
        return self.value


# ─── On Balance Volume ────────────────────────────────────────────────────

class OBV:
    """On Balance Volume.

    Cumulative volume indicator. No warmup needed (starts at 0).
    """

    def __init__(self):
        self.value: float = 0.0
        self._prev_close: Optional[float] = None

    def update(self, bar: dict) -> float:
        """Update with a new bar (must include 'volume' key). Returns OBV."""
        cl = bar["close"]
        vol = bar.get("volume", 0.0)

        if self._prev_close is not None:
            if cl > self._prev_close:
                self.value += vol
            elif cl < self._prev_close:
                self.value -= vol
        self._prev_close = cl
        return self.value


# ─── Rate of Change ───────────────────────────────────────────────────────

class ROC:
    """Rate of Change oscillator.

    ROC = 100 * (close - close[n]) / close[n]
    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 12):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current ROC."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period + 1:
            self._closes.pop(0)
        self._count += 1

        if self._count <= self.period:
            self.value = 0.0
            return self.value

        old_close = self._closes[0]
        if old_close == 0:
            self.value = 0.0
        else:
            self.value = 100.0 * (cl - old_close) / old_close
        return self.value


# ─── Price Momentum ───────────────────────────────────────────────────────

class MomentumIndicator:
    """Price Momentum: close - close[n].

    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 10):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current momentum."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period + 1:
            self._closes.pop(0)
        self._count += 1

        if self._count <= self.period:
            self.value = 0.0
            return self.value

        self.value = cl - self._closes[0]
        return self.value


# ─── Deviation from True Oscillator ──────────────────────────────────────

class DTO:
    """Deviation from True Oscillator: (close - SMA) / SMA * 100.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 10):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current DTO value."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period:
            self._closes.pop(0)
        self._count += 1

        if self._count < self.period:
            self.value = 0.0
            return self.value

        sma = sum(self._closes) / self.period
        if sma == 0:
            self.value = 0.0
        else:
            self.value = (cl - sma) / sma * 100.0
        return self.value


# ─── Vortex Indicator ─────────────────────────────────────────────────────

class Vortex:
    """Vortex Indicator.

    VI+ = sum(VM+, period) / sum(TR, period)
    VI- = sum(VM-, period) / sum(TR, period)
    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.vi_plus: float = 0.0
        self.vi_minus: float = 0.0
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._vm_plus: List[float] = []
        self._vm_minus: List[float] = []
        self._trs: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> Tuple[float, float]:
        """Update with a new bar. Returns (VI+, VI-)."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._count += 1

        if self._prev_high is None:
            self._prev_high = hi
            self._prev_low = lo
            self._prev_close = cl
            return 0.0, 0.0

        vm_p = abs(hi - self._prev_low)
        vm_m = abs(lo - self._prev_high)
        tr = max(hi - lo, abs(hi - self._prev_close), abs(lo - self._prev_close))

        self._vm_plus.append(vm_p)
        self._vm_minus.append(vm_m)
        self._trs.append(tr)

        if len(self._vm_plus) > self.period:
            self._vm_plus.pop(0)
            self._vm_minus.pop(0)
            self._trs.pop(0)

        self._prev_high = hi
        self._prev_low = lo
        self._prev_close = cl

        if self._count < self.period + 1:
            return 0.0, 0.0

        sum_tr = sum(self._trs)
        if sum_tr == 0:
            self.vi_plus = 0.0
            self.vi_minus = 0.0
        else:
            self.vi_plus = sum(self._vm_plus) / sum_tr
            self.vi_minus = sum(self._vm_minus) / sum_tr
        return self.vi_plus, self.vi_minus


# ─── Bollinger/Keltner Squeeze ────────────────────────────────────────────

class Squeeze:
    """Bollinger Band / Keltner Channel squeeze detector.

    squeeze_on = True when BB fits inside KC (low volatility, breakout pending).
    momentum = close - BB mid (directional bias).
    Warmup: max(bb_period, kc_period) bars.
    """

    def __init__(self, bb_period: int = 20, bb_mult: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5):
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.squeeze_on: bool = False
        self.momentum: float = 0.0
        self._closes: List[float] = []
        self._atr = ATR(period=kc_period)
        self._count: int = 0

    def update(self, bar: dict) -> Tuple[bool, float]:
        """Update with a new bar. Returns (squeeze_on, momentum)."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.bb_period:
            self._closes.pop(0)
        self._atr.update(bar)
        self._count += 1

        warmup = max(self.bb_period, self.kc_period)
        if self._count < warmup:
            self.squeeze_on = False
            self.momentum = 0.0
            return self.squeeze_on, self.momentum

        # Bollinger Bands
        bb_mid = sum(self._closes) / len(self._closes)
        variance = sum((c - bb_mid) ** 2 for c in self._closes) / len(self._closes)
        bb_std = math.sqrt(variance)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # Keltner Channel
        kc_mid = bb_mid  # Use same SMA for KC midline
        atr_val = self._atr.value
        kc_upper = kc_mid + self.kc_mult * atr_val
        kc_lower = kc_mid - self.kc_mult * atr_val

        self.squeeze_on = (bb_lower > kc_lower) and (bb_upper < kc_upper)
        self.momentum = cl - bb_mid
        return self.squeeze_on, self.momentum


# ─── Bollinger Band Width ─────────────────────────────────────────────────

class BBWidth:
    """Bollinger Band Width: (upper - lower) / mid.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20, mult: float = 2.0):
        self.period = period
        self.mult = mult
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns BB width."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period:
            self._closes.pop(0)
        self._count += 1

        if self._count < self.period:
            self.value = 0.0
            return self.value

        mid = sum(self._closes) / self.period
        if mid == 0:
            self.value = 0.0
            return self.value

        variance = sum((c - mid) ** 2 for c in self._closes) / self.period
        std = math.sqrt(variance)
        upper = mid + self.mult * std
        lower = mid - self.mult * std
        self.value = (upper - lower) / mid
        return self.value


# ─── ATR Ratio (short vs long) ───────────────────────────────────────────

class ATRRatio:
    """Ratio of short ATR to long ATR, mapped via tanh.

    tanh((ATR_short / ATR_long - 1) * 5)
    Range: [-1, 1]. >0 means expanding vol, <0 means contracting.
    Warmup: `long_period` bars.
    """

    def __init__(self, short_period: int = 5, long_period: int = 200):
        self.short_period = short_period
        self.long_period = long_period
        self.value: float = 0.0
        self._atr_short = ATR(period=short_period)
        self._atr_long = ATR(period=long_period)
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns ATR ratio value."""
        self._atr_short.update(bar)
        self._atr_long.update(bar)
        self._count += 1

        if self._count < self.long_period:
            self.value = 0.0
            return self.value

        if self._atr_long.value == 0:
            self.value = 0.0
        else:
            self.value = math.tanh((self._atr_short.value / self._atr_long.value - 1.0) * 5.0)
        return self.value


# ─── Retracement from Last Swing ─────────────────────────────────────────

class Retracement:
    """Retracement from last swing high/low within lookback.

    Value in [0, 1]. 0 = at the extreme, 1 = fully retraced.
    Warmup: `lookback` bars.
    """

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.value: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns retracement value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.lookback:
            self._highs.pop(0)
            self._lows.pop(0)
        self._count += 1

        if self._count < self.lookback:
            self.value = 0.0
            return self.value

        hh = max(self._highs)
        ll = min(self._lows)
        rng = hh - ll
        if rng == 0:
            self.value = 0.0
            return self.value

        # Find which is more recent: peak or trough
        peak_idx = len(self._highs) - 1 - self._highs[::-1].index(hh)
        trough_idx = len(self._lows) - 1 - self._lows[::-1].index(ll)

        if peak_idx >= trough_idx:
            # Peak is more recent — measure pullback from peak
            self.value = (hh - cl) / rng
        else:
            # Trough is more recent — measure bounce from trough
            self.value = (cl - ll) / rng

        # Clamp to [0, 1]
        self.value = max(0.0, min(1.0, self.value))
        return self.value


# ─── Breakout Channel ────────────────────────────────────────────────────

class BreakoutChannel:
    """Breakout detection from Donchian-style channel.

    +1 = breakout above previous period's highest high.
    -1 = breakdown below previous period's lowest low.
    0 = inside range, with position scaled to [-1, 1].
    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns breakout value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._highs.append(hi)
        self._lows.append(lo)
        self._count += 1

        # Need period+1 bars: first `period` form the channel, current bar tests it
        if self._count <= self.period:
            self.value = 0.0
            return self.value

        # Keep period+1 bars so we can separate channel (first period) from current
        if len(self._highs) > self.period + 1:
            self._highs.pop(0)
            self._lows.pop(0)

        # Channel from previous `period` bars (excluding current)
        chan_highs = self._highs[:-1]
        chan_lows = self._lows[:-1]
        chan_hh = max(chan_highs)
        chan_ll = min(chan_lows)
        chan_rng = chan_hh - chan_ll

        if cl > chan_hh:
            self.value = 1.0
        elif cl < chan_ll:
            self.value = -1.0
        elif chan_rng == 0:
            self.value = 0.0
        else:
            self.value = (cl - chan_ll) / chan_rng * 2.0 - 1.0
        return self.value


# ─── Composite / Derived Indicators ─────────────────────────────────────────

class RangePosition:
    """Position within N-bar high/low range, normalized to [-1, 1].

    Output: tanh((pos * 20) - 10) where pos = (close - lowest) / (highest - lowest).
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 240):
        self.period = period
        self.value: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._highs.append(hi)
        self._lows.append(lo)
        self._count += 1
        if len(self._highs) > self.period:
            self._highs.pop(0)
            self._lows.pop(0)
        if self._count < self.period:
            self.value = 0.0
            return self.value
        hh = max(self._highs)
        ll = min(self._lows)
        if hh == ll:
            self.value = 0.0
        else:
            pos = (cl - ll) / (hh - ll)
            self.value = math.tanh(pos * 20.0 - 10.0)
        return self.value


class BBPosition:
    """Position within Bollinger Bands, clamped to [-1, 1].

    pos = (close - SMA) / (4 * std).
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._closes.append(cl)
        self._count += 1
        if len(self._closes) > self.period:
            self._closes.pop(0)
        if self._count < self.period:
            self.value = 0.0
            return self.value
        mean = sum(self._closes) / self.period
        variance = sum((x - mean) ** 2 for x in self._closes) / self.period
        std = math.sqrt(variance)
        if std > 0:
            self.value = max(-1.0, min(1.0, (cl - mean) / (4.0 * std)))
        else:
            self.value = 0.0
        return self.value


class VolExpansion:
    """ATR vs its own SMA — volatility expansion/contraction.

    Output: atan((ATR/SMA(ATR) - 1) * 10) * 2 / pi.
    Warmup: `atr_period + sma_period` bars.
    """

    def __init__(self, atr_period: int = 14, sma_period: int = 20):
        self.atr_period = atr_period
        self.sma_period = sma_period
        self.value: float = 0.0
        self._atr = ATR(period=atr_period)
        self._atr_buf: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        atr_val = self._atr.update(bar)
        self._count += 1
        if self._count >= self.atr_period:
            self._atr_buf.append(atr_val)
            if len(self._atr_buf) > self.sma_period:
                self._atr_buf.pop(0)
        if len(self._atr_buf) < self.sma_period:
            self.value = 0.0
            return self.value
        sma_atr = sum(self._atr_buf) / self.sma_period
        if sma_atr > 0:
            ratio = atr_val / sma_atr
            self.value = math.atan((ratio - 1.0) * 10.0) * 2.0 / math.pi
        else:
            self.value = 0.0
        return self.value


class RSIExtreme:
    """RSI normalized to [-1, 1].

    rsi_norm = clamp((RSI - 50) / 50, -1, 1).
    Uses Wilder EMA of gains/losses internally.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self._prev_close: Optional[float] = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._count: int = 0
        self._gains: List[float] = []
        self._losses: List[float] = []

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        if self._prev_close is None:
            self._prev_close = cl
            return 0.0
        change = cl - self._prev_close
        self._prev_close = cl
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1
        if self._count <= self.period:
            self._gains.append(gain)
            self._losses.append(loss)
            if self._count == self.period:
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                self._gains = []
                self._losses = []
            else:
                self.value = 0.0
                return self.value
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        self.value = max(-1.0, min(1.0, (rsi - 50.0) / 50.0))
        return self.value


class TrendQuality:
    """Trend alignment x ADX strength.

    aligned = 1 if (close > SMA_fast > SMA_slow) or (close < SMA_fast < SMA_slow).
    output = aligned * min(ADX / 50, 1).
    Warmup: `slow_sma` bars + 2 * `adx_period` bars.
    """

    def __init__(self, adx_period: int = 14, fast_sma: int = 50, slow_sma: int = 200):
        self.adx_period = adx_period
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.value: float = 0.0
        self._adx = ADX(period=adx_period)
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._adx.update(bar)
        self._closes.append(cl)
        self._count += 1
        if len(self._closes) > self.slow_sma:
            self._closes.pop(0)
        if self._count < self.slow_sma:
            self.value = 0.0
            return self.value
        sma_fast = sum(self._closes[-self.fast_sma:]) / self.fast_sma
        sma_slow = sum(self._closes) / self.slow_sma
        aligned = 1.0 if (cl > sma_fast > sma_slow) or (cl < sma_fast < sma_slow) else 0.0
        adx_val = self._adx.value
        self.value = aligned * min(adx_val / 50.0, 1.0)
        return self.value


class MomStrength:
    """Directional consistency of momentum.

    momentum[i] = close - close[mom_period ago].
    Count positive momentum bars in last consistency_period bars.
    output = (count/period - 0.5) * 2 * abs(close - close[consistency_period ago]).
    Warmup: `mom_period + consistency_period` bars.
    """

    def __init__(self, mom_period: int = 5, consistency_period: int = 10):
        self.mom_period = mom_period
        self.consistency_period = consistency_period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._closes.append(cl)
        self._count += 1
        max_needed = self.mom_period + self.consistency_period + 1
        if len(self._closes) > max_needed:
            self._closes.pop(0)
        if self._count < self.mom_period + self.consistency_period:
            self.value = 0.0
            return self.value
        # Compute momentum for last consistency_period bars
        n = len(self._closes)
        pos_count = 0
        for j in range(n - self.consistency_period, n):
            mom = self._closes[j] - self._closes[j - self.mom_period]
            if mom > 0:
                pos_count += 1
        ratio = pos_count / self.consistency_period
        price_change = abs(cl - self._closes[n - 1 - self.consistency_period])
        self.value = (ratio - 0.5) * 2.0 * price_change
        return self.value


class PSARDelta:
    """Distance from price to Parabolic SAR.

    output = close - PSAR.
    Warmup: 2 bars (from ParabolicSAR).
    """

    def __init__(self):
        self.value: float = 0.0
        self._psar = ParabolicSAR()

    def update(self, bar: dict) -> float:
        self._psar.update(bar)
        cl = bar["close"]
        self.value = cl - self._psar.value
        return self.value


class DMISignal:
    """Directional Movement Index signal.

    output = (+DI > -DI ? 1 : -1) * ADX / 100.
    Range: [-1, 1].
    Warmup: 2 * `period` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self._adx = ADX(period=period)

    def update(self, bar: dict) -> float:
        self._adx.update(bar)
        adx_obj = self._adx
        if adx_obj._atr_s > 0 and adx_obj._count >= adx_obj.period:
            pdi = 100.0 * adx_obj._pdm_s / adx_obj._atr_s
            mdi = 100.0 * adx_obj._mdm_s / adx_obj._atr_s
            direction = 1.0 if pdi > mdi else -1.0
            self.value = direction * adx_obj.value / 100.0
        else:
            self.value = 0.0
        return self.value


class DeltaPrice:
    """Price change normalized by ATR.

    output = tanh((close - prev_close) / ATR).
    Warmup: `atr_period` bars.
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.value: float = 0.0
        self._atr = ATR(period=atr_period)
        self._prev_close: Optional[float] = None

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        atr_val = self._atr.update(bar)
        if self._prev_close is not None and atr_val > 0:
            self.value = math.tanh((cl - self._prev_close) / atr_val)
        else:
            self.value = 0.0
        self._prev_close = cl
        return self.value


class ASIVelocity:
    """First derivative of Wilder's Accumulative Swing Index.

    output = ASI[current] - ASI[previous].
    Warmup: 2 bars.
    """

    def __init__(self):
        self.value: float = 0.0
        self._asi: float = 0.0
        self._prev_asi: float = 0.0
        self._prev_close: Optional[float] = None
        self._prev_open: Optional[float] = None
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._count: int = 0

    @staticmethod
    def _compute_si(hi: float, lo: float, cl: float, op: float,
                    prev_hi: float, prev_lo: float, prev_cl: float, prev_op: float) -> float:
        """Compute single-bar Swing Index (Wilder)."""
        k = max(abs(hi - prev_cl), abs(lo - prev_cl))
        tr = max(hi - lo, abs(hi - prev_cl), abs(lo - prev_cl))
        if tr == 0:
            return 0.0
        e1 = abs(hi - prev_cl) - 0.5 * abs(lo - prev_cl) + 0.25 * abs(prev_cl - prev_op)
        e2 = abs(lo - prev_cl) - 0.5 * abs(hi - prev_cl) + 0.25 * abs(prev_cl - prev_op)
        e3 = (hi - lo) + 0.25 * abs(prev_cl - prev_op)
        R = max(e1, e2, e3)
        if R == 0:
            return 0.0
        nn = (cl - prev_cl) + 0.5 * (cl - op) + 0.25 * (prev_cl - prev_op)
        return 50.0 * nn / R * k / tr

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        op = bar["open"]
        self._count += 1
        if self._count == 1:
            self._prev_close = cl
            self._prev_open = op
            self._prev_high = hi
            self._prev_low = lo
            self.value = 0.0
            return self.value
        si = self._compute_si(hi, lo, cl, op,
                              self._prev_high, self._prev_low,
                              self._prev_close, self._prev_open)
        self._prev_asi = self._asi
        self._asi += si
        self.value = self._asi - self._prev_asi
        self._prev_close = cl
        self._prev_open = op
        self._prev_high = hi
        self._prev_low = lo
        return self.value


class ASIAcceleration:
    """Second derivative of Wilder's Accumulative Swing Index.

    output = velocity[current] - velocity[previous].
    Warmup: 3 bars.
    """

    def __init__(self):
        self.value: float = 0.0
        self._asi: float = 0.0
        self._prev_asi: float = 0.0
        self._prev2_asi: float = 0.0
        self._prev_close: Optional[float] = None
        self._prev_open: Optional[float] = None
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._count: int = 0

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        op = bar["open"]
        self._count += 1
        if self._count == 1:
            self._prev_close = cl
            self._prev_open = op
            self._prev_high = hi
            self._prev_low = lo
            self.value = 0.0
            return self.value
        si = ASIVelocity._compute_si(hi, lo, cl, op,
                                     self._prev_high, self._prev_low,
                                     self._prev_close, self._prev_open)
        self._prev2_asi = self._prev_asi
        self._prev_asi = self._asi
        self._asi += si
        self._prev_close = cl
        self._prev_open = op
        self._prev_high = hi
        self._prev_low = lo
        if self._count < 3:
            self.value = 0.0
            return self.value
        vel_curr = self._asi - self._prev_asi
        vel_prev = self._prev_asi - self._prev2_asi
        self.value = vel_curr - vel_prev
        return self.value


# ─── Schaff Trend Cycle (STC) ─────────────────────────────────────────

class SchaffTrendCycle:
    """Double-stochastic of MACD — cycle oscillator (0-100).
    Buy: crosses above 25. Sell: crosses below 75.

    Warmup: ~slow_period + 2*tc_len bars.
    """

    def __init__(self, fast: int = 23, slow: int = 50, tc_len: int = 10, factor: float = 0.5):
        self.fast = fast
        self.slow = slow
        self.tc_len = tc_len
        self.factor = factor
        self.value: float = 50.0
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._alpha_fast = 2.0 / (fast + 1)
        self._alpha_slow = 2.0 / (slow + 1)
        self._count: int = 0
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0
        self._macd_buf: List[float] = []
        self._pf: float = 0.0
        self._pf_buf: List[float] = []
        self._stc: float = 50.0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._count += 1

        # EMA fast
        if self._count <= self.fast:
            self._sum_fast += cl
            if self._count == self.fast:
                self._ema_fast = self._sum_fast / self.fast
        elif self._ema_fast is not None:
            self._ema_fast = self._alpha_fast * cl + (1 - self._alpha_fast) * self._ema_fast

        # EMA slow
        if self._count <= self.slow:
            self._sum_slow += cl
            if self._count == self.slow:
                self._ema_slow = self._sum_slow / self.slow
        elif self._ema_slow is not None:
            self._ema_slow = self._alpha_slow * cl + (1 - self._alpha_slow) * self._ema_slow

        if self._ema_fast is None or self._ema_slow is None:
            return self.value

        xmac = self._ema_fast - self._ema_slow
        self._macd_buf.append(xmac)
        if len(self._macd_buf) > self.tc_len:
            self._macd_buf = self._macd_buf[-self.tc_len:]

        if len(self._macd_buf) < self.tc_len:
            return self.value

        lo = min(self._macd_buf)
        hi = max(self._macd_buf)
        rng = hi - lo
        frac1 = ((xmac - lo) / rng * 100.0) if rng > 0 else self._pf
        self._pf = self._pf + self.factor * (frac1 - self._pf)

        self._pf_buf.append(self._pf)
        if len(self._pf_buf) > self.tc_len:
            self._pf_buf = self._pf_buf[-self.tc_len:]

        if len(self._pf_buf) < self.tc_len:
            return self.value

        lo2 = min(self._pf_buf)
        hi2 = max(self._pf_buf)
        rng2 = hi2 - lo2
        frac2 = ((self._pf - lo2) / rng2 * 100.0) if rng2 > 0 else self._stc
        self._stc = self._stc + self.factor * (frac2 - self._stc)
        self.value = max(0.0, min(100.0, self._stc))
        return self.value


# ─── Repulse ──────────────────────────────────────────────────────────

class Repulse:
    """Eric Lefort's Repulse indicator — bull/bear pressure oscillator.
    Positive = bullish, negative = bearish.

    Warmup: n + 5*n bars.
    """

    def __init__(self, period: int = 5):
        self.period = period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._bull_ema: Optional[float] = None
        self._bear_ema: Optional[float] = None
        self._ema_period = 5 * period
        self._alpha = 2.0 / (self._ema_period + 1)
        self._count: int = 0
        self._sum_bull: float = 0.0
        self._sum_bear: float = 0.0

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        op = bar["open"]
        self._closes.append(cl)
        self._opens.append(op)
        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._closes) > self.period:
            self._closes = self._closes[-self.period:]
            self._opens = self._opens[-self.period:]
            self._highs = self._highs[-self.period:]
            self._lows = self._lows[-self.period:]

        if len(self._closes) < self.period:
            return self.value

        lowest = min(self._lows)
        highest = max(self._highs)
        open_n = self._opens[0]

        bull = 100.0 * (3.0 * cl - 2.0 * lowest - open_n) / cl if cl != 0 else 0.0
        bear = 100.0 * (open_n + 2.0 * highest - 3.0 * cl) / cl if cl != 0 else 0.0

        self._count += 1
        if self._count <= self._ema_period:
            self._sum_bull += bull
            self._sum_bear += bear
            if self._count == self._ema_period:
                self._bull_ema = self._sum_bull / self._ema_period
                self._bear_ema = self._sum_bear / self._ema_period
        elif self._bull_ema is not None:
            self._bull_ema = self._alpha * bull + (1 - self._alpha) * self._bull_ema
            self._bear_ema = self._alpha * bear + (1 - self._alpha) * self._bear_ema

        if self._bull_ema is not None and self._bear_ema is not None:
            self.value = self._bull_ema - self._bear_ema
        return self.value


# ─── Stochastic RSI ───────────────────────────────────────────────────

class StochasticRSI:
    """StochRSI = (RSI - lowest RSI) / (highest RSI - lowest RSI).
    Range 0-100. Smoothed with %K and %D SMAs.

    Warmup: rsi_period + stoch_period bars.
    """

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14,
                 k_smooth: int = 3, d_smooth: int = 3):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.value: float = 50.0  # %K
        self.d_line: float = 50.0  # %D
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._prev_close: Optional[float] = None
        self._count: int = 0
        self._gains: List[float] = []
        self._losses: List[float] = []
        self._rsi_buf: List[float] = []
        self._k_buf: List[float] = []
        self._k_smooth = k_smooth
        self._d_smooth = d_smooth

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        if self._prev_close is None:
            self._prev_close = cl
            return self.value

        change = cl - self._prev_close
        self._prev_close = cl
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1

        if self._count <= self.rsi_period:
            self._gains.append(gain)
            self._losses.append(loss)
            if self._count == self.rsi_period:
                self._avg_gain = sum(self._gains) / self.rsi_period
                self._avg_loss = sum(self._losses) / self.rsi_period
            else:
                return self.value
        else:
            self._avg_gain = (self._avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._avg_loss = (self._avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        self._rsi_buf.append(rsi)
        if len(self._rsi_buf) > self.stoch_period:
            self._rsi_buf = self._rsi_buf[-self.stoch_period:]

        if len(self._rsi_buf) < self.stoch_period:
            return self.value

        lo = min(self._rsi_buf)
        hi = max(self._rsi_buf)
        rng = hi - lo
        stoch_rsi = ((rsi - lo) / rng * 100.0) if rng > 0 else 50.0

        self._k_buf.append(stoch_rsi)
        if len(self._k_buf) > self._k_smooth:
            self._k_buf = self._k_buf[-self._k_smooth:]
        self.value = sum(self._k_buf) / len(self._k_buf)  # %K smoothed
        return self.value


# ─── Stochastic Momentum Index (SMI) ─────────────────────────────────

class SMI:
    """Stochastic Momentum Index — close vs midpoint of range.
    Range: -100 to +100. Overbought >40, oversold <-40.

    Warmup: k_len + 2*d_len bars.
    """

    def __init__(self, k_len: int = 10, d_len: int = 3, signal_len: int = 10):
        self.k_len = k_len
        self.d_len = d_len
        self.value: float = 0.0
        self.signal: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        # Double EMA state for relative range
        self._rel_ema1: Optional[float] = None
        self._rel_ema2: Optional[float] = None
        self._rng_ema1: Optional[float] = None
        self._rng_ema2: Optional[float] = None
        self._alpha = 2.0 / (d_len + 1)
        self._sig_alpha = 2.0 / (signal_len + 1)
        self._sig_ema: Optional[float] = None
        self._count: int = 0

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._highs.append(hi)
        self._lows.append(lo)
        self._closes.append(cl)
        if len(self._highs) > self.k_len:
            self._highs = self._highs[-self.k_len:]
            self._lows = self._lows[-self.k_len:]
            self._closes = self._closes[-self.k_len:]

        if len(self._highs) < self.k_len:
            return self.value

        hh = max(self._highs)
        ll = min(self._lows)
        rel = cl - (hh + ll) / 2.0
        hl_range = hh - ll

        self._count += 1
        # Double EMA for rel
        if self._rel_ema1 is None:
            self._rel_ema1 = rel
            self._rel_ema2 = rel
            self._rng_ema1 = hl_range
            self._rng_ema2 = hl_range
        else:
            self._rel_ema1 = self._alpha * rel + (1 - self._alpha) * self._rel_ema1
            self._rel_ema2 = self._alpha * self._rel_ema1 + (1 - self._alpha) * self._rel_ema2
            self._rng_ema1 = self._alpha * hl_range + (1 - self._alpha) * self._rng_ema1
            self._rng_ema2 = self._alpha * self._rng_ema1 + (1 - self._alpha) * self._rng_ema2

        denom = self._rng_ema2 / 2.0 if self._rng_ema2 is not None else 0.0
        if denom != 0:
            self.value = 100.0 * self._rel_ema2 / denom
        else:
            self.value = 0.0
        self.value = max(-100.0, min(100.0, self.value))

        # Signal line
        if self._sig_ema is None:
            self._sig_ema = self.value
        else:
            self._sig_ema = self._sig_alpha * self.value + (1 - self._sig_alpha) * self._sig_ema
        self.signal = self._sig_ema
        return self.value


# ─── Chaikin Money Flow (CMF) ─────────────────────────────────────────

class ChaikinMoneyFlow:
    """CMF = SUM(MFV, n) / SUM(Volume, n). Range [-1, +1].
    Note: uses tick volume for FX.

    Warmup: period bars.
    """

    def __init__(self, period: int = 21):
        self.period = period
        self.value: float = 0.0
        self._mfv_buf: List[float] = []
        self._vol_buf: List[float] = []

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        vol = bar.get("volume", 1.0)

        hl_range = hi - lo
        if hl_range > 0:
            mfm = ((cl - lo) - (hi - cl)) / hl_range
        else:
            mfm = 0.0
        mfv = mfm * vol

        self._mfv_buf.append(mfv)
        self._vol_buf.append(vol)
        if len(self._mfv_buf) > self.period:
            self._mfv_buf = self._mfv_buf[-self.period:]
            self._vol_buf = self._vol_buf[-self.period:]

        vol_sum = sum(self._vol_buf)
        self.value = sum(self._mfv_buf) / vol_sum if vol_sum > 0 else 0.0
        return self.value


# ─── Chaikin Oscillator ───────────────────────────────────────────────

class ChaikinOscillator:
    """Chaikin Oscillator = EMA(ADL, 3) - EMA(ADL, 10).

    Warmup: ~10 bars.
    """

    def __init__(self, fast: int = 3, slow: int = 10):
        self.value: float = 0.0
        self._adl: float = 0.0
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._alpha_fast = 2.0 / (fast + 1)
        self._alpha_slow = 2.0 / (slow + 1)
        self._count: int = 0
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0
        self._fast_period = fast
        self._slow_period = slow

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        vol = bar.get("volume", 1.0)

        hl_range = hi - lo
        if hl_range > 0:
            clv = ((cl - lo) - (hi - cl)) / hl_range
        else:
            clv = 0.0
        self._adl += clv * vol
        self._count += 1

        # Fast EMA of ADL
        if self._count <= self._fast_period:
            self._sum_fast += self._adl
            if self._count == self._fast_period:
                self._fast_ema = self._sum_fast / self._fast_period
        elif self._fast_ema is not None:
            self._fast_ema = self._alpha_fast * self._adl + (1 - self._alpha_fast) * self._fast_ema

        # Slow EMA of ADL
        if self._count <= self._slow_period:
            self._sum_slow += self._adl
            if self._count == self._slow_period:
                self._slow_ema = self._sum_slow / self._slow_period
        elif self._slow_ema is not None:
            self._slow_ema = self._alpha_slow * self._adl + (1 - self._alpha_slow) * self._slow_ema

        if self._fast_ema is not None and self._slow_ema is not None:
            self.value = self._fast_ema - self._slow_ema
        return self.value


# ─── Chaikin Volatility ───────────────────────────────────────────────

class ChaikinVolatility:
    """Chaikin Volatility = % change of EMA(H-L) over n periods.

    Warmup: 2*period bars.
    """

    def __init__(self, period: int = 10):
        self.period = period
        self.value: float = 0.0
        self._ema: Optional[float] = None
        self._alpha = 2.0 / (period + 1)
        self._count: int = 0
        self._sum: float = 0.0
        self._ema_buf: List[float] = []

    def update(self, bar: dict) -> float:
        hl = bar["high"] - bar["low"]
        self._count += 1

        if self._count <= self.period:
            self._sum += hl
            if self._count == self.period:
                self._ema = self._sum / self.period
        elif self._ema is not None:
            self._ema = self._alpha * hl + (1 - self._alpha) * self._ema

        if self._ema is not None:
            self._ema_buf.append(self._ema)
            if len(self._ema_buf) > self.period + 1:
                self._ema_buf = self._ema_buf[-(self.period + 1):]

            if len(self._ema_buf) > self.period:
                prev_ema = self._ema_buf[0]
                if prev_ema != 0:
                    self.value = (self._ema - prev_ema) / prev_ema * 100.0
        return self.value


# ─── Coppock Curve ────────────────────────────────────────────────────

class CoppockCurve:
    """Coppock Curve = WMA(ROC(14) + ROC(11), 10). Long-term momentum.

    Warmup: max(roc1, roc2) + wma_period bars.
    """

    def __init__(self, roc1: int = 14, roc2: int = 11, wma_period: int = 10):
        self.roc1 = roc1
        self.roc2 = roc2
        self.wma_period = wma_period
        self.value: float = 0.0
        self._closes: List[float] = []
        self._roc_sum_buf: List[float] = []

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._closes.append(cl)
        max_lookback = max(self.roc1, self.roc2)

        if len(self._closes) <= max_lookback:
            return self.value

        # Keep only what we need
        if len(self._closes) > max_lookback + 1:
            self._closes = self._closes[-(max_lookback + 1):]

        idx = len(self._closes) - 1
        r1_prev = self._closes[idx - self.roc1]
        r2_prev = self._closes[idx - self.roc2]
        roc_1 = (cl - r1_prev) / r1_prev * 100.0 if r1_prev != 0 else 0.0
        roc_2 = (cl - r2_prev) / r2_prev * 100.0 if r2_prev != 0 else 0.0
        roc_sum = roc_1 + roc_2

        self._roc_sum_buf.append(roc_sum)
        if len(self._roc_sum_buf) > self.wma_period:
            self._roc_sum_buf = self._roc_sum_buf[-self.wma_period:]

        if len(self._roc_sum_buf) >= self.wma_period:
            # WMA: weight = position (1 for oldest, n for newest)
            total_w = 0.0
            total_v = 0.0
            for i, v in enumerate(self._roc_sum_buf):
                w = i + 1
                total_v += v * w
                total_w += w
            self.value = total_v / total_w
        return self.value


# ─── Detrended Price Oscillator (DPO) ────────────────────────────────

class DPO:
    """DPO = Close[n/2+1 ago] - SMA(Close, n). Removes trend.

    Warmup: n + n/2 bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.displacement = period // 2 + 1
        self.value: float = 0.0
        self._closes: List[float] = []

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._closes.append(cl)

        needed = self.period + self.displacement
        if len(self._closes) < needed:
            return self.value

        if len(self._closes) > needed:
            self._closes = self._closes[-needed:]

        # SMA of last n closes ending at current bar
        sma = sum(self._closes[-self.period:]) / self.period
        # Displaced close
        displaced_close = self._closes[-(self.displacement + 1)]
        self.value = displaced_close - sma
        return self.value


# ─── DEMA (Double Exponential Moving Average) ─────────────────────────

class DEMA:
    """DEMA = 2*EMA(n) - EMA(EMA(n)). Reduced lag.

    Warmup: 2*period bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._ema1: Optional[float] = None
        self._ema2: Optional[float] = None
        self._alpha = 2.0 / (period + 1)
        self._count: int = 0
        self._sum1: float = 0.0
        self._count2: int = 0
        self._sum2: float = 0.0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._count += 1

        # EMA1
        if self._count <= self.period:
            self._sum1 += cl
            if self._count == self.period:
                self._ema1 = self._sum1 / self.period
        elif self._ema1 is not None:
            self._ema1 = self._alpha * cl + (1 - self._alpha) * self._ema1

        if self._ema1 is None:
            return self.value

        # EMA2 (EMA of EMA1)
        self._count2 += 1
        if self._count2 <= self.period:
            self._sum2 += self._ema1
            if self._count2 == self.period:
                self._ema2 = self._sum2 / self.period
        elif self._ema2 is not None:
            self._ema2 = self._alpha * self._ema1 + (1 - self._alpha) * self._ema2

        if self._ema2 is not None:
            self.value = 2.0 * self._ema1 - self._ema2
        return self.value


# ─── TEMA (Triple Exponential Moving Average) ─────────────────────────

class TEMA:
    """TEMA = 3*EMA1 - 3*EMA2 + EMA3. Further reduced lag.

    Warmup: 3*period bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._ema1: Optional[float] = None
        self._ema2: Optional[float] = None
        self._ema3: Optional[float] = None
        self._alpha = 2.0 / (period + 1)
        self._c1: int = 0; self._s1: float = 0.0
        self._c2: int = 0; self._s2: float = 0.0
        self._c3: int = 0; self._s3: float = 0.0

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._c1 += 1
        if self._c1 <= self.period:
            self._s1 += cl
            if self._c1 == self.period:
                self._ema1 = self._s1 / self.period
        elif self._ema1 is not None:
            self._ema1 = self._alpha * cl + (1 - self._alpha) * self._ema1

        if self._ema1 is None:
            return self.value

        self._c2 += 1
        if self._c2 <= self.period:
            self._s2 += self._ema1
            if self._c2 == self.period:
                self._ema2 = self._s2 / self.period
        elif self._ema2 is not None:
            self._ema2 = self._alpha * self._ema1 + (1 - self._alpha) * self._ema2

        if self._ema2 is None:
            return self.value

        self._c3 += 1
        if self._c3 <= self.period:
            self._s3 += self._ema2
            if self._c3 == self.period:
                self._ema3 = self._s3 / self.period
        elif self._ema3 is not None:
            self._ema3 = self._alpha * self._ema2 + (1 - self._alpha) * self._ema3

        if self._ema3 is not None:
            self.value = 3.0 * self._ema1 - 3.0 * self._ema2 + self._ema3
        return self.value


# ─── MACD Zero Lag ────────────────────────────────────────────────────

class MACDZeroLag:
    """Zero-lag MACD: uses DEMA instead of EMA for reduced lag.
    ZL_MACD = DEMA(fast) - DEMA(slow), Signal = DEMA(ZL_MACD, signal).

    Warmup: 2*slow_period + 2*signal bars.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.value: float = 0.0
        self.signal: float = 0.0
        self.histogram: float = 0.0
        self._dema_fast = DEMA(fast)
        self._dema_slow = DEMA(slow)
        # Signal line: EMA of ZL_MACD (simpler than full DEMA for signal)
        self._sig_ema: Optional[float] = None
        self._sig_alpha = 2.0 / (signal_period + 1)
        self._sig_count: int = 0
        self._sig_sum: float = 0.0
        self._sig_period = signal_period
        self._ready: bool = False

    def update(self, bar: dict) -> float:
        f = self._dema_fast.update(bar)
        s = self._dema_slow.update(bar)

        if self._dema_fast._ema2 is not None and self._dema_slow._ema2 is not None:
            self._ready = True
            self.value = f - s
        else:
            return self.value

        self._sig_count += 1
        if self._sig_count <= self._sig_period:
            self._sig_sum += self.value
            if self._sig_count == self._sig_period:
                self._sig_ema = self._sig_sum / self._sig_period
        elif self._sig_ema is not None:
            self._sig_ema = self._sig_alpha * self.value + (1 - self._sig_alpha) * self._sig_ema

        if self._sig_ema is not None:
            self.signal = self._sig_ema
        self.histogram = self.value - self.signal
        return self.value


# ─── Dynamic Zone RSI ─────────────────────────────────────────────────

class DynamicZoneRSI:
    """RSI with adaptive Bollinger Bands for overbought/oversold levels.
    .value = RSI, .upper/.lower = dynamic bands.

    Warmup: rsi_period + bb_period bars.
    """

    def __init__(self, rsi_period: int = 14, bb_period: int = 20, coeff: float = 1.3185):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.coeff = coeff
        self.value: float = 50.0
        self.upper: float = 70.0
        self.lower: float = 30.0
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._prev_close: Optional[float] = None
        self._count: int = 0
        self._gains_init: List[float] = []
        self._losses_init: List[float] = []
        self._rsi_buf: List[float] = []

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        if self._prev_close is None:
            self._prev_close = cl
            return self.value

        change = cl - self._prev_close
        self._prev_close = cl
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1

        if self._count <= self.rsi_period:
            self._gains_init.append(gain)
            self._losses_init.append(loss)
            if self._count == self.rsi_period:
                self._avg_gain = sum(self._gains_init) / self.rsi_period
                self._avg_loss = sum(self._losses_init) / self.rsi_period
            else:
                return self.value
        else:
            self._avg_gain = (self._avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._avg_loss = (self._avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        self.value = rsi
        self._rsi_buf.append(rsi)
        if len(self._rsi_buf) > self.bb_period:
            self._rsi_buf = self._rsi_buf[-self.bb_period:]

        if len(self._rsi_buf) >= self.bb_period:
            mean = sum(self._rsi_buf) / len(self._rsi_buf)
            var = sum((x - mean) ** 2 for x in self._rsi_buf) / len(self._rsi_buf)
            std = var ** 0.5
            self.upper = mean + self.coeff * std
            self.lower = mean - self.coeff * std
        return self.value


# ─── Dynamic Zone Stochastic ──────────────────────────────────────────

class DynamicZoneStochastic:
    """Stochastic with adaptive BB for overbought/oversold.
    .value = %K, .upper/.lower = dynamic bands.

    Warmup: stoch_period + bb_period bars.
    """

    def __init__(self, stoch_period: int = 14, k_smooth: int = 3,
                 bb_period: int = 20, coeff: float = 0.8):
        self.stoch_period = stoch_period
        self.bb_period = bb_period
        self.coeff = coeff
        self.value: float = 50.0
        self.upper: float = 80.0
        self.lower: float = 20.0
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        self._k_buf: List[float] = []
        self._k_smooth = k_smooth
        self._stoch_buf: List[float] = []

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        self._highs.append(hi)
        self._lows.append(lo)
        self._closes.append(cl)
        if len(self._highs) > self.stoch_period:
            self._highs = self._highs[-self.stoch_period:]
            self._lows = self._lows[-self.stoch_period:]
            self._closes = self._closes[-self.stoch_period:]

        if len(self._highs) < self.stoch_period:
            return self.value

        hh = max(self._highs)
        ll = min(self._lows)
        rng = hh - ll
        raw_k = ((cl - ll) / rng * 100.0) if rng > 0 else 50.0

        self._k_buf.append(raw_k)
        if len(self._k_buf) > self._k_smooth:
            self._k_buf = self._k_buf[-self._k_smooth:]
        k = sum(self._k_buf) / len(self._k_buf)
        self.value = k

        self._stoch_buf.append(k)
        if len(self._stoch_buf) > self.bb_period:
            self._stoch_buf = self._stoch_buf[-self.bb_period:]

        if len(self._stoch_buf) >= self.bb_period:
            mean = sum(self._stoch_buf) / len(self._stoch_buf)
            var = sum((x - mean) ** 2 for x in self._stoch_buf) / len(self._stoch_buf)
            std = var ** 0.5
            self.upper = mean + self.coeff * std
            self.lower = mean - self.coeff * std
        return self.value


# ─── Envelopes (Moving Average Envelopes) ─────────────────────────────

class Envelopes:
    """Percentage bands around SMA.
    .upper/.lower/.mid values.

    Warmup: period bars.
    """

    def __init__(self, period: int = 20, pct: float = 2.5):
        self.period = period
        self.pct = pct / 100.0
        self.upper: float = 0.0
        self.lower: float = 0.0
        self.mid: float = 0.0
        self._buf: List[float] = []

    def update(self, bar: dict) -> float:
        cl = bar["close"]
        self._buf.append(cl)
        if len(self._buf) > self.period:
            self._buf = self._buf[-self.period:]

        if len(self._buf) >= self.period:
            self.mid = sum(self._buf) / self.period
            self.upper = self.mid * (1.0 + self.pct)
            self.lower = self.mid * (1.0 - self.pct)
        return self.mid


# ─── RSI (Relative Strength Index) ──────────────────────────────────────

class RSI:
    """Standard Wilder RSI.

    Range: [0, 100]. Oversold < 30, Overbought > 70.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 50.0
        self._prev_close: Optional[float] = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._count: int = 0
        self._gains: List[float] = []
        self._losses: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current RSI (0-100)."""
        cl = bar["close"]
        if self._prev_close is None:
            self._prev_close = cl
            return self.value
        change = cl - self._prev_close
        self._prev_close = cl
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1

        if self._count <= self.period:
            self._gains.append(gain)
            self._losses.append(loss)
            if self._count == self.period:
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                self._gains = []
                self._losses = []
            else:
                return self.value
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            self.value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = 100.0 - 100.0 / (1.0 + rs)
        return self.value


# ─── SMA (Simple Moving Average) ────────────────────────────────────────

class SMA:
    """Simple Moving Average of close price.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current SMA."""
        cl = bar["close"]
        self._buf.append(cl)
        if len(self._buf) > self.period:
            self._buf = self._buf[-self.period:]

        if len(self._buf) >= self.period:
            self.value = sum(self._buf) / self.period
        return self.value


# ─── EMA (Exponential Moving Average) ───────────────────────────────────

class EMA:
    """Exponential Moving Average of close price.

    Warmup: `period` bars (SMA seed, then EMA).
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._alpha: float = 2.0 / (period + 1)
        self._count: int = 0
        self._sum: float = 0.0
        self._initialized: bool = False

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current EMA."""
        cl = bar["close"]
        self._count += 1

        if not self._initialized:
            self._sum += cl
            if self._count == self.period:
                self.value = self._sum / self.period
                self._initialized = True
        else:
            self.value = self._alpha * cl + (1 - self._alpha) * self.value

        return self.value


# ─── WMA (Weighted Moving Average) ──────────────────────────────────────

class WMA:
    """Weighted Moving Average (linearly weighted).

    Weight: most recent bar gets weight=period, oldest gets weight=1.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._buf: List[float] = []
        self._denom: float = period * (period + 1) / 2.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current WMA."""
        cl = bar["close"]
        self._buf.append(cl)
        if len(self._buf) > self.period:
            self._buf = self._buf[-self.period:]

        if len(self._buf) >= self.period:
            weighted = sum((i + 1) * self._buf[i] for i in range(self.period))
            self.value = weighted / self._denom
        return self.value


# ─── StdDev (Standard Deviation) ────────────────────────────────────────

class StdDev:
    """Standard Deviation of close price.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current StdDev."""
        cl = bar["close"]
        self._buf.append(cl)
        if len(self._buf) > self.period:
            self._buf = self._buf[-self.period:]

        if len(self._buf) >= self.period:
            mean = sum(self._buf) / self.period
            var = sum((x - mean) ** 2 for x in self._buf) / self.period
            self.value = var ** 0.5
        return self.value


# ─── Bollinger Bands ────────────────────────────────────────────────────

class Bollinger:
    """Bollinger Bands: SMA ± std * multiplier.

    .upper, .mid, .lower values.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20, std: float = 2.0):
        self.period = period
        self.std = std
        self.upper: float = 0.0
        self.mid: float = 0.0
        self.lower: float = 0.0
        self._buf: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns mid (SMA)."""
        cl = bar["close"]
        self._buf.append(cl)
        if len(self._buf) > self.period:
            self._buf = self._buf[-self.period:]

        if len(self._buf) >= self.period:
            self.mid = sum(self._buf) / self.period
            var = sum((x - self.mid) ** 2 for x in self._buf) / self.period
            sd = var ** 0.5
            self.upper = self.mid + self.std * sd
            self.lower = self.mid - self.std * sd
        return self.mid


# ─── Pivot Points (Classic) ─────────────────────────────────────────────

class PivotPoints:
    """Classic Pivot Points from previous day's high/low/close.

    .pivot, .r1, .r2, .s1, .s2 values.
    Warmup: needs at least one completed day.
    """

    def __init__(self):
        self.pivot: float = 0.0
        self.r1: float = 0.0
        self.r2: float = 0.0
        self.s1: float = 0.0
        self.s2: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float('inf')
        self._day_close: float = 0.0
        self._prev_day_set: bool = False
        self._bar_count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Call set_prev_day() at day boundaries,
        or pass bars sequentially — uses H/L/C from accumulated bars.
        Returns pivot value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        self._day_high = max(self._day_high, hi)
        self._day_low = min(self._day_low, lo)
        self._day_close = cl
        self._bar_count += 1
        return self.pivot

    def set_prev_day(self, high: float, low: float, close: float) -> None:
        """Set previous day's H/L/C and compute pivot levels."""
        self.pivot = (high + low + close) / 3.0
        self.r1 = 2.0 * self.pivot - low
        self.s1 = 2.0 * self.pivot - high
        self.r2 = self.pivot + (high - low)
        self.s2 = self.pivot - (high - low)
        self._prev_day_set = True

    def new_day(self) -> None:
        """Call at day boundary to rotate current day into prev day."""
        if self._bar_count > 0:
            self.set_prev_day(self._day_high, self._day_low, self._day_close)
        self._day_high = 0.0
        self._day_low = float('inf')
        self._day_close = 0.0
        self._bar_count = 0


# ─── Woodie Pivot Points ────────────────────────────────────────────────

class WoodiePivots:
    """Woodie Pivot Points: pivot = (H + L + 2*C) / 4.

    .pivot, .r1, .r2, .s1, .s2 values.
    Warmup: needs at least one completed day.
    """

    def __init__(self):
        self.pivot: float = 0.0
        self.r1: float = 0.0
        self.r2: float = 0.0
        self.s1: float = 0.0
        self.s2: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float('inf')
        self._day_close: float = 0.0
        self._bar_count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns pivot value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        self._day_high = max(self._day_high, hi)
        self._day_low = min(self._day_low, lo)
        self._day_close = cl
        self._bar_count += 1
        return self.pivot

    def set_prev_day(self, high: float, low: float, close: float) -> None:
        """Set previous day's H/L/C and compute Woodie pivot levels."""
        self.pivot = (high + low + 2.0 * close) / 4.0
        self.r1 = 2.0 * self.pivot - low
        self.s1 = 2.0 * self.pivot - high
        self.r2 = self.pivot + (high - low)
        self.s2 = self.pivot - (high - low)

    def new_day(self) -> None:
        """Call at day boundary to rotate current day into prev day."""
        if self._bar_count > 0:
            self.set_prev_day(self._day_high, self._day_low, self._day_close)
        self._day_high = 0.0
        self._day_low = float('inf')
        self._day_close = 0.0
        self._bar_count = 0


# ─── Camarilla Pivot Points ─────────────────────────────────────────────

class CamarillaPivots:
    """Camarilla Pivot Points.

    .pivot, .r1, .r2, .s1, .s2 values.
    Warmup: needs at least one completed day.
    """

    def __init__(self):
        self.pivot: float = 0.0
        self.r1: float = 0.0
        self.r2: float = 0.0
        self.s1: float = 0.0
        self.s2: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float('inf')
        self._day_close: float = 0.0
        self._bar_count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns pivot value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        self._day_high = max(self._day_high, hi)
        self._day_low = min(self._day_low, lo)
        self._day_close = cl
        self._bar_count += 1
        return self.pivot

    def set_prev_day(self, high: float, low: float, close: float) -> None:
        """Set previous day's H/L/C and compute Camarilla pivot levels."""
        self.pivot = (high + low + close) / 3.0
        rng = high - low
        self.r1 = close + rng * 1.1 / 12.0
        self.r2 = close + rng * 1.1 / 6.0
        self.s1 = close - rng * 1.1 / 12.0
        self.s2 = close - rng * 1.1 / 6.0

    def new_day(self) -> None:
        """Call at day boundary to rotate current day into prev day."""
        if self._bar_count > 0:
            self.set_prev_day(self._day_high, self._day_low, self._day_close)
        self._day_high = 0.0
        self._day_low = float('inf')
        self._day_close = 0.0
        self._bar_count = 0


# ─── Accumulation/Distribution Line ─────────────────────────────────────

class AccumulationDistribution:
    """Accumulation/Distribution Line (ADL).

    Cumulative indicator. No warmup needed.
    """

    def __init__(self):
        self.value: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar (uses volume). Returns ADL value."""
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]
        vol = bar.get("volume", 1)

        rng = hi - lo
        if rng > 0:
            clv = ((cl - lo) - (hi - cl)) / rng
            self.value += clv * vol
        return self.value


# ─── Adaptive Moving Average (Kaufman AMA) ──────────────────────────────

class AdaptiveMovingAverage:
    """Kaufman Adaptive Moving Average (KAMA).

    Uses Efficiency Ratio to adapt smoothing constant between
    fast (2-bar) and slow (30-bar) EMA.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 10, fast_sc: int = 2, slow_sc: int = 30):
        self.period = period
        self.value: float = 0.0
        self._fast_alpha: float = 2.0 / (fast_sc + 1)
        self._slow_alpha: float = 2.0 / (slow_sc + 1)
        self._closes: List[float] = []
        self._count: int = 0
        self._initialized: bool = False

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns current KAMA value."""
        cl = bar["close"]
        self._closes.append(cl)
        if len(self._closes) > self.period + 1:
            self._closes = self._closes[-(self.period + 1):]
        self._count += 1

        if self._count < self.period + 1:
            self.value = cl
            return self.value

        if not self._initialized:
            self.value = cl
            self._initialized = True

        # Efficiency Ratio
        direction = abs(self._closes[-1] - self._closes[0])
        volatility = sum(
            abs(self._closes[i] - self._closes[i - 1])
            for i in range(1, len(self._closes))
        )
        er = direction / volatility if volatility > 0 else 0.0

        # Smoothing constant
        sc = (er * (self._fast_alpha - self._slow_alpha) + self._slow_alpha) ** 2
        self.value = self.value + sc * (cl - self.value)
        return self.value


# ─── Price Oscillator ───────────────────────────────────────────────────

class PriceOscillator:
    """Price Oscillator: (EMA_fast - EMA_slow) / EMA_slow * 100.

    Warmup: `slow` bars.
    """

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow
        self.value: float = 0.0
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._alpha_fast: float = 2.0 / (fast + 1)
        self._alpha_slow: float = 2.0 / (slow + 1)
        self._count: int = 0
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns Price Oscillator %."""
        cl = bar["close"]
        self._count += 1

        # Bootstrap fast EMA
        if self._count <= self.fast:
            self._sum_fast += cl
            if self._count == self.fast:
                self._ema_fast = self._sum_fast / self.fast
        elif self._ema_fast is not None:
            self._ema_fast = self._alpha_fast * cl + (1 - self._alpha_fast) * self._ema_fast

        # Bootstrap slow EMA
        if self._count <= self.slow:
            self._sum_slow += cl
            if self._count == self.slow:
                self._ema_slow = self._sum_slow / self.slow
        elif self._ema_slow is not None:
            self._ema_slow = self._alpha_slow * cl + (1 - self._alpha_slow) * self._ema_slow

        if self._ema_fast is not None and self._ema_slow is not None and self._ema_slow != 0:
            self.value = (self._ema_fast - self._ema_slow) / self._ema_slow * 100.0
        return self.value


# ─── Volume Oscillator ──────────────────────────────────────────────────

class VolumeOscillator:
    """Volume Oscillator: (vol_EMA_fast - vol_EMA_slow) / vol_EMA_slow * 100.

    Warmup: `slow` bars.
    """

    def __init__(self, fast: int = 5, slow: int = 10):
        self.fast = fast
        self.slow = slow
        self.value: float = 0.0
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._alpha_fast: float = 2.0 / (fast + 1)
        self._alpha_slow: float = 2.0 / (slow + 1)
        self._count: int = 0
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar (uses volume). Returns Volume Oscillator %."""
        vol = bar.get("volume", 1)
        self._count += 1

        # Bootstrap fast EMA
        if self._count <= self.fast:
            self._sum_fast += vol
            if self._count == self.fast:
                self._ema_fast = self._sum_fast / self.fast
        elif self._ema_fast is not None:
            self._ema_fast = self._alpha_fast * vol + (1 - self._alpha_fast) * self._ema_fast

        # Bootstrap slow EMA
        if self._count <= self.slow:
            self._sum_slow += vol
            if self._count == self.slow:
                self._ema_slow = self._sum_slow / self.slow
        elif self._ema_slow is not None:
            self._ema_slow = self._alpha_slow * vol + (1 - self._alpha_slow) * self._ema_slow

        if self._ema_fast is not None and self._ema_slow is not None and self._ema_slow != 0:
            self.value = (self._ema_fast - self._ema_slow) / self._ema_slow * 100.0
        return self.value


# ─── VROC (Volume Rate of Change) ───────────────────────────────────────

class VROC:
    """Volume Rate of Change: 100 * (vol - vol[n]) / vol[n].

    Warmup: `period + 1` bars.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = 0.0
        self._vols: List[float] = []
        self._count: int = 0

    def update(self, bar: dict) -> float:
        """Update with a new bar (uses volume). Returns VROC %."""
        vol = bar.get("volume", 1)
        self._vols.append(vol)
        if len(self._vols) > self.period + 1:
            self._vols = self._vols[-(self.period + 1):]
        self._count += 1

        if self._count <= self.period:
            return self.value

        old_vol = self._vols[0]
        if old_vol == 0:
            self.value = 0.0
        else:
            self.value = 100.0 * (vol - old_vol) / old_vol
        return self.value


# ─── VWAP (Volume Weighted Average Price) ────────────────────────────────

class VWAP:
    """Volume Weighted Average Price (session-based).

    Cumulative within session. Call reset() at session boundary.
    No warmup needed.
    """

    def __init__(self):
        self.value: float = 0.0
        self._cum_tp_vol: float = 0.0
        self._cum_vol: float = 0.0

    def update(self, bar: dict) -> float:
        """Update with a new bar (uses volume). Returns VWAP."""
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        vol = bar.get("volume", 1)
        self._cum_tp_vol += tp * vol
        self._cum_vol += vol
        if self._cum_vol > 0:
            self.value = self._cum_tp_vol / self._cum_vol
        return self.value

    def reset(self) -> None:
        """Reset at session boundary."""
        self._cum_tp_vol = 0.0
        self._cum_vol = 0.0
        self.value = 0.0


# ─── VWMA (Volume Weighted Moving Average) ──────────────────────────────

class VWMA:
    """Volume Weighted Moving Average over a rolling window.

    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.value: float = 0.0
        self._prices: List[float] = []
        self._vols: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar (uses volume). Returns VWMA."""
        cl = bar["close"]
        vol = bar.get("volume", 1)
        self._prices.append(cl)
        self._vols.append(vol)
        if len(self._prices) > self.period:
            self._prices = self._prices[-self.period:]
            self._vols = self._vols[-self.period:]

        if len(self._prices) >= self.period:
            total_vol = sum(self._vols)
            if total_vol > 0:
                self.value = sum(
                    p * v for p, v in zip(self._prices, self._vols)
                ) / total_vol
            else:
                self.value = sum(self._prices) / self.period
        return self.value


# ─── Highs and Lows ─────────────────────────────────────────────────────

class HighsAndLows:
    """Highest high, lowest low, and range over a rolling window.

    .highest, .lowest, .range values.
    Warmup: `period` bars.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.highest: float = 0.0
        self.lowest: float = 0.0
        self.range: float = 0.0
        self._highs: List[float] = []
        self._lows: List[float] = []

    def update(self, bar: dict) -> float:
        """Update with a new bar. Returns range (highest - lowest)."""
        hi = bar["high"]
        lo = bar["low"]
        self._highs.append(hi)
        self._lows.append(lo)
        if len(self._highs) > self.period:
            self._highs = self._highs[-self.period:]
            self._lows = self._lows[-self.period:]

        if len(self._highs) >= self.period:
            self.highest = max(self._highs)
            self.lowest = min(self._lows)
            self.range = self.highest - self.lowest
        return self.range


# ─── MACD Divergence ────────────────────────────────────────────────────

class MACDDivergence:
    """Detects price/MACD divergence.

    .value: -1 = bearish divergence, 0 = none, +1 = bullish divergence.
    Uses swing highs/lows over lookback window to detect divergence.
    Warmup: `slow + signal + lookback` bars.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, lookback: int = 20):
        self.lookback = lookback
        self.value: int = 0
        self._macd = MACD(fast, slow, signal)
        self._closes: List[float] = []
        self._macd_vals: List[float] = []

    def update(self, bar: dict) -> int:
        """Update with a new bar. Returns divergence signal (-1/0/+1)."""
        cl = bar["close"]
        macd_val = self._macd.update(bar)
        self._closes.append(cl)
        self._macd_vals.append(macd_val)
        if len(self._closes) > self.lookback:
            self._closes = self._closes[-self.lookback:]
            self._macd_vals = self._macd_vals[-self.lookback:]

        self.value = 0
        if len(self._closes) < self.lookback:
            return self.value

        # Find recent swing lows (bullish divergence: price lower low, MACD higher low)
        mid = self.lookback // 2
        price_first_half_low = min(self._closes[:mid])
        price_second_half_low = min(self._closes[mid:])
        macd_first_half_low = min(self._macd_vals[:mid])
        macd_second_half_low = min(self._macd_vals[mid:])

        if price_second_half_low < price_first_half_low and macd_second_half_low > macd_first_half_low:
            self.value = 1  # bullish

        # Find recent swing highs (bearish divergence: price higher high, MACD lower high)
        price_first_half_high = max(self._closes[:mid])
        price_second_half_high = max(self._closes[mid:])
        macd_first_half_high = max(self._macd_vals[:mid])
        macd_second_half_high = max(self._macd_vals[mid:])

        if price_second_half_high > price_first_half_high and macd_second_half_high < macd_first_half_high:
            self.value = -1  # bearish

        return self.value


# ─── RSI Divergence ─────────────────────────────────────────────────────

class RSIDivergence:
    """Detects price/RSI divergence.

    .value: -1 = bearish divergence, 0 = none, +1 = bullish divergence.
    Warmup: `period + lookback` bars.
    """

    def __init__(self, period: int = 14, lookback: int = 20):
        self.lookback = lookback
        self.value: int = 0
        self._rsi = RSI(period)
        self._closes: List[float] = []
        self._rsi_vals: List[float] = []

    def update(self, bar: dict) -> int:
        """Update with a new bar. Returns divergence signal (-1/0/+1)."""
        cl = bar["close"]
        rsi_val = self._rsi.update(bar)
        self._closes.append(cl)
        self._rsi_vals.append(rsi_val)
        if len(self._closes) > self.lookback:
            self._closes = self._closes[-self.lookback:]
            self._rsi_vals = self._rsi_vals[-self.lookback:]

        self.value = 0
        if len(self._closes) < self.lookback:
            return self.value

        mid = self.lookback // 2

        # Bullish: price lower low, RSI higher low
        price_lo1 = min(self._closes[:mid])
        price_lo2 = min(self._closes[mid:])
        rsi_lo1 = min(self._rsi_vals[:mid])
        rsi_lo2 = min(self._rsi_vals[mid:])
        if price_lo2 < price_lo1 and rsi_lo2 > rsi_lo1:
            self.value = 1

        # Bearish: price higher high, RSI lower high
        price_hi1 = max(self._closes[:mid])
        price_hi2 = max(self._closes[mid:])
        rsi_hi1 = max(self._rsi_vals[:mid])
        rsi_hi2 = max(self._rsi_vals[mid:])
        if price_hi2 > price_hi1 and rsi_hi2 < rsi_hi1:
            self.value = -1

        return self.value


# ─── CCI Divergence ─────────────────────────────────────────────────────

class CCIDivergence:
    """Detects price/CCI divergence.

    .value: -1 = bearish divergence, 0 = none, +1 = bullish divergence.
    Warmup: `period + lookback` bars.
    """

    def __init__(self, period: int = 20, lookback: int = 20):
        self.lookback = lookback
        self.value: int = 0
        self._cci = CCI(period)
        self._closes: List[float] = []
        self._cci_vals: List[float] = []

    def update(self, bar: dict) -> int:
        """Update with a new bar. Returns divergence signal (-1/0/+1)."""
        cl = bar["close"]
        cci_val = self._cci.update(bar)
        self._closes.append(cl)
        self._cci_vals.append(cci_val)
        if len(self._closes) > self.lookback:
            self._closes = self._closes[-self.lookback:]
            self._cci_vals = self._cci_vals[-self.lookback:]

        self.value = 0
        if len(self._closes) < self.lookback:
            return self.value

        mid = self.lookback // 2

        # Bullish: price lower low, CCI higher low
        price_lo1 = min(self._closes[:mid])
        price_lo2 = min(self._closes[mid:])
        cci_lo1 = min(self._cci_vals[:mid])
        cci_lo2 = min(self._cci_vals[mid:])
        if price_lo2 < price_lo1 and cci_lo2 > cci_lo1:
            self.value = 1

        # Bearish: price higher high, CCI lower high
        price_hi1 = max(self._closes[:mid])
        price_hi2 = max(self._closes[mid:])
        cci_hi1 = max(self._cci_vals[:mid])
        cci_hi2 = max(self._cci_vals[mid:])
        if price_hi2 > price_hi1 and cci_hi2 < cci_hi1:
            self.value = -1

        return self.value


# ─── Time-Based Indicators ────────────────────────────────────────────

class HourOfDay:
    """Sin/cos encoding of hour of day (0-23).
    Circular encoding prevents 23→0 discontinuity.

    .sin_hour, .cos_hour: [-1, +1]
    .hour: raw hour (0-23)
    .session: 0=Asian(0-8), 1=London(8-16), 2=NY(16-24)
    """
    import math as _math

    def __init__(self):
        self.sin_hour: float = 0.0
        self.cos_hour: float = 0.0
        self.hour: int = 0
        self.session: int = 0
        self.value: float = 0.0  # alias for sin_hour

    def update(self, bar: dict) -> float:
        ts = bar.get("timestamp") or bar.get("ts")
        if ts is not None:
            if hasattr(ts, "hour"):
                self.hour = ts.hour
            elif isinstance(ts, str) and len(ts) >= 13:
                try:
                    self.hour = int(ts[11:13])
                except (ValueError, IndexError):
                    pass
        import math
        self.sin_hour = math.sin(2 * math.pi * self.hour / 24.0)
        self.cos_hour = math.cos(2 * math.pi * self.hour / 24.0)
        # Session: Asian=0(0-8 UTC), London=1(8-16), NY=2(16-24)
        if self.hour < 8:
            self.session = 0
        elif self.hour < 16:
            self.session = 1
        else:
            self.session = 2
        self.value = self.sin_hour
        return self.value


class DayOfWeek:
    """Sin/cos encoding of day of week (0=Mon, 4=Fri).
    Circular encoding for smooth transitions.

    .sin_dow, .cos_dow: [-1, +1]
    .dow: raw day (0=Mon, 6=Sun)
    """

    def __init__(self):
        self.sin_dow: float = 0.0
        self.cos_dow: float = 0.0
        self.dow: int = 0
        self.value: float = 0.0

    def update(self, bar: dict) -> float:
        ts = bar.get("timestamp") or bar.get("ts")
        if ts is not None:
            if hasattr(ts, "weekday"):
                self.dow = ts.weekday()
            elif isinstance(ts, str) and len(ts) >= 10:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts[:10])
                    self.dow = dt.weekday()
                except (ValueError, ImportError):
                    pass
        import math
        self.sin_dow = math.sin(2 * math.pi * self.dow / 5.0)  # 5 trading days
        self.cos_dow = math.cos(2 * math.pi * self.dow / 5.0)
        self.value = self.sin_dow
        return self.value


class SessionStrength:
    """Tracks per-session (Asian/London/NY) price action strength.

    .asian_range: range of Asian session (pips)
    .london_break: how far London broke Asian range
    .ny_continuation: NY direction relative to London
    .value: current session momentum (-1 to +1)
    """

    def __init__(self):
        self.value: float = 0.0
        self.asian_range: float = 0.0
        self.london_break: float = 0.0
        self._session_high: float = 0.0
        self._session_low: float = 1e10
        self._asian_high: float = 0.0
        self._asian_low: float = 1e10
        self._london_open: float = 0.0
        self._prev_session: int = -1

    def update(self, bar: dict) -> float:
        hi = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        ts = bar.get("timestamp") or bar.get("ts")
        hour = 0
        if ts is not None:
            if hasattr(ts, "hour"):
                hour = ts.hour
            elif isinstance(ts, str) and len(ts) >= 13:
                try:
                    hour = int(ts[11:13])
                except (ValueError, IndexError):
                    pass

        session = 0 if hour < 8 else (1 if hour < 16 else 2)

        # Session transition
        if session != self._prev_session:
            if session == 1 and self._prev_session == 0:
                # Asian → London: save Asian range
                self.asian_range = self._session_high - self._session_low
                self._asian_high = self._session_high
                self._asian_low = self._session_low
                self._london_open = cl
            elif session == 2 and self._prev_session == 1:
                # London → NY: compute London break
                if self.asian_range > 0:
                    self.london_break = (cl - self._london_open) / self.asian_range
            self._session_high = hi
            self._session_low = lo
            self._prev_session = session
        else:
            if hi > self._session_high:
                self._session_high = hi
            if lo < self._session_low:
                self._session_low = lo

        # Value: session momentum based on position within session range
        rng = self._session_high - self._session_low
        if rng > 0:
            self.value = (cl - self._session_low) / rng * 2.0 - 1.0  # [-1, +1]
        return self.value
