"""
candle_metrics.py
-----------------
Computes raw candle metrics and log-delta features for a pair of consecutive
closed OHLC bars. Designed for ML feature pipelines.

Usage:
    from candle_metrics import CandleMetrics

    prev = {"open": 1.1000, "high": 1.1080, "low": 1.0950, "close": 1.1060}
    curr = {"open": 1.1060, "high": 1.1140, "low": 1.1020, "close": 1.1100}

    cm = CandleMetrics(prev, curr)
    features = cm.feature_vector()   # dict of all raw + delta + categorical
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON = 1e-6          # prevents log(0) for ratio/bounded metrics
MIN_RANGE = 1e-8        # prevents division by zero on doji candles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_log(x: float) -> float:
    """Natural log, guarded against zero/negative inputs."""
    return math.log(max(x, EPSILON))


def log_delta(curr: float, prev: float) -> float:
    """
    Log difference: log(curr) - log(prev) = log(curr / prev).
    Safe for ratio and magnitude metrics (not direction).
    """
    return safe_log(curr) - safe_log(prev)


def ratio_log_delta(curr: float, prev: float) -> float:
    """
    Log difference for metrics bounded in [0, 1].
    Adds epsilon before logging to avoid log(0).
    """
    return math.log(curr + EPSILON) - math.log(prev + EPSILON)


# ---------------------------------------------------------------------------
# Single-bar metrics
# ---------------------------------------------------------------------------
@dataclass
class BarMetrics:
    """All derived metrics for one closed OHLC bar."""

    o: float
    h: float
    l: float
    c: float

    # computed on init
    body_size:       float = field(init=False)
    range_size:      float = field(init=False)
    body_ratio:      float = field(init=False)
    upper_wick:      float = field(init=False)
    lower_wick:      float = field(init=False)
    upper_wick_ratio: float = field(init=False)
    lower_wick_ratio: float = field(init=False)
    wick_ratio:      float = field(init=False)
    close_position:  float = field(init=False)
    direction:       int   = field(init=False)   # +1 / -1 / 0 — categorical

    def __post_init__(self):
        r = max(self.h - self.l, MIN_RANGE)

        self.body_size        = abs(self.c - self.o)
        self.range_size       = r
        self.body_ratio       = self.body_size / r
        self.upper_wick       = self.h - max(self.o, self.c)
        self.lower_wick       = min(self.o, self.c) - self.l
        self.upper_wick_ratio = self.upper_wick / r
        self.lower_wick_ratio = self.lower_wick / r
        self.wick_ratio       = (self.upper_wick + self.lower_wick) / r
        self.close_position   = (self.c - self.l) / r   # 0..1

        if self.c > self.o:
            self.direction = 1
        elif self.c < self.o:
            self.direction = -1
        else:
            self.direction = 0

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        """Return raw metrics as a flat dict (excludes direction)."""
        return {
            f"{prefix}body_size":        self.body_size,
            f"{prefix}range_size":       self.range_size,
            f"{prefix}body_ratio":       self.body_ratio,
            f"{prefix}upper_wick":       self.upper_wick,
            f"{prefix}lower_wick":       self.lower_wick,
            f"{prefix}upper_wick_ratio": self.upper_wick_ratio,
            f"{prefix}lower_wick_ratio": self.lower_wick_ratio,
            f"{prefix}wick_ratio":       self.wick_ratio,
            f"{prefix}close_position":   self.close_position,
        }


# ---------------------------------------------------------------------------
# Two-bar metrics
# ---------------------------------------------------------------------------
@dataclass
class TwoBarMetrics:
    """
    Cross-bar metrics derived from two consecutive closed bars.
    prev = bar[t-1], curr = bar[t]
    """

    prev: BarMetrics
    curr: BarMetrics

    gap:                float = field(init=False)
    gap_type:           int   = field(init=False)   # +1 gap-up / -1 gap-down / 0 none
    range_expansion:    float = field(init=False)
    body_overlap:       float = field(init=False)   # fraction of curr body inside prev body
    prior_close_breach: int   = field(init=False)   # +1 above / -1 below / 0 at
    prior_high_breach:  int   = field(init=False)
    prior_low_breach:   int   = field(init=False)
    is_engulfing:       int   = field(init=False)   # 1 = yes, 0 = no
    is_inside_bar:      int   = field(init=False)
    reversal_score:     int   = field(init=False)   # +1 reversal / -1 continuation
    two_bar_momentum:   float = field(init=False)

    # gap threshold in price units (set externally if needed)
    gap_threshold: float = 1e-5

    def __post_init__(self):
        p, c = self.prev, self.curr

        # --- gap ---
        self.gap = c.o - p.c
        if self.gap > self.gap_threshold:
            self.gap_type = 1
        elif self.gap < -self.gap_threshold:
            self.gap_type = -1
        else:
            self.gap_type = 0

        # --- range expansion ratio ---
        self.range_expansion = c.range_size / max(p.range_size, MIN_RANGE)

        # --- body overlap ---
        curr_body_lo = min(c.o, c.c)
        curr_body_hi = max(c.o, c.c)
        prev_body_lo = min(p.o, p.c)
        prev_body_hi = max(p.o, p.c)
        overlap = max(0.0, min(curr_body_hi, prev_body_hi) - max(curr_body_lo, prev_body_lo))
        curr_body_range = max(curr_body_hi - curr_body_lo, MIN_RANGE)
        self.body_overlap = overlap / curr_body_range

        # --- price acceptance vs prior structure ---
        self.prior_close_breach = (
             1 if c.c > p.c else (-1 if c.c < p.c else 0)
        )
        self.prior_high_breach = 1 if c.h > p.h else 0
        self.prior_low_breach  = 1 if c.l < p.l else 0

        # --- patterns (binary) ---
        self.is_engulfing  = int(c.h > p.h and c.l < p.l)
        self.is_inside_bar = int(c.h < p.h and c.l > p.l)

        # --- reversal vs continuation ---
        # +1 = reversal, -1 = continuation, 0 = one/both are doji
        self.reversal_score = -(p.direction * c.direction)

        # --- two-bar momentum: net move normalised by prev range ---
        self.two_bar_momentum = (c.c - p.o) / max(p.range_size, MIN_RANGE)

    def as_dict(self) -> Dict[str, float]:
        """Raw two-bar metrics (continuous only; categoricals returned separately)."""
        return {
            "gap":               self.gap,
            "range_expansion":   self.range_expansion,
            "body_overlap":      self.body_overlap,
            "two_bar_momentum":  self.two_bar_momentum,
        }

    def categorical_dict(self) -> Dict[str, int]:
        return {
            "gap_type":           self.gap_type,
            "prior_close_breach": self.prior_close_breach,
            "prior_high_breach":  self.prior_high_breach,
            "prior_low_breach":   self.prior_low_breach,
            "is_engulfing":       self.is_engulfing,
            "is_inside_bar":      self.is_inside_bar,
            "reversal_score":     self.reversal_score,
            "curr_direction":     self.curr.direction,
            "prev_direction":     self.prev.direction,
        }


# ---------------------------------------------------------------------------
# Log-delta features
# ---------------------------------------------------------------------------
def compute_log_deltas(prev: BarMetrics, curr: BarMetrics) -> Dict[str, float]:
    """
    Log-delta for every continuous metric between two bars.

    Metrics that are ratios/bounded in [0,1] use ratio_log_delta (epsilon-guarded).
    Metrics that are magnitudes (body_size, range_size, wick sizes) use log_delta.
    Direction is excluded — handled categorically.
    """

    # magnitude metrics: log(curr) - log(prev)
    magnitude_pairs = {
        "dlog_body_size":  (curr.body_size,  prev.body_size),
        "dlog_range_size": (curr.range_size, prev.range_size),
        "dlog_upper_wick": (curr.upper_wick, prev.upper_wick),
        "dlog_lower_wick": (curr.lower_wick, prev.lower_wick),
    }

    # ratio/bounded metrics: log(curr + eps) - log(prev + eps)
    ratio_pairs = {
        "dlog_body_ratio":        (curr.body_ratio,       prev.body_ratio),
        "dlog_upper_wick_ratio":  (curr.upper_wick_ratio, prev.upper_wick_ratio),
        "dlog_lower_wick_ratio":  (curr.lower_wick_ratio, prev.lower_wick_ratio),
        "dlog_wick_ratio":        (curr.wick_ratio,       prev.wick_ratio),
        "dlog_close_position":    (curr.close_position,   prev.close_position),
    }

    deltas = {}

    for name, (c_val, p_val) in magnitude_pairs.items():
        deltas[name] = log_delta(c_val, p_val)

    for name, (c_val, p_val) in ratio_pairs.items():
        deltas[name] = ratio_log_delta(c_val, p_val)

    return deltas


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
class CandleMetrics:
    """
    Compute the full feature vector for bar[t] given bar[t-1].

    Parameters
    ----------
    prev_bar : dict with keys open, high, low, close
    curr_bar : dict with keys open, high, low, close

    Output (feature_vector)
    -----------------------
    Flat dict containing:
      - curr_* : raw metrics for the current bar
      - prev_* : raw metrics for the previous bar  (optional, see include_prev)
      - two_bar_* : two-bar continuous metrics
      - dlog_* : log-delta features
      - cat_* : categorical / directional features (int)
    """

    def __init__(self, prev_bar: dict, curr_bar: dict, gap_threshold: float = 1e-5):
        self.prev_bm = BarMetrics(
            o=prev_bar["open"], h=prev_bar["high"],
            l=prev_bar["low"],  c=prev_bar["close"]
        )
        self.curr_bm = BarMetrics(
            o=curr_bar["open"], h=curr_bar["high"],
            l=curr_bar["low"],  c=curr_bar["close"]
        )
        self.two_bar = TwoBarMetrics(self.prev_bm, self.curr_bm, gap_threshold=gap_threshold)
        self._deltas = compute_log_deltas(self.prev_bm, self.curr_bm)

    def feature_vector(self, include_prev: bool = False) -> Dict[str, float]:
        """
        Returns the full flat feature dict ready for ML ingestion.

        Parameters
        ----------
        include_prev : bool
            If True, also includes raw metrics for the previous bar.
            Useful when you want the model to see both bars' levels directly.
        """
        features: Dict = {}

        # Current bar raw metrics
        features.update(self.curr_bm.as_dict(prefix="curr_"))

        # Previous bar raw metrics (optional)
        if include_prev:
            features.update(self.prev_bm.as_dict(prefix="prev_"))

        # Two-bar continuous metrics
        for k, v in self.two_bar.as_dict().items():
            features[f"two_{k}"] = v

        # Log-delta features
        features.update(self._deltas)

        # Categoricals (kept as int, prefixed cat_)
        for k, v in self.two_bar.categorical_dict().items():
            features[f"cat_{k}"] = v

        return features

    def summary(self) -> None:
        """Pretty-print all features."""
        fv = self.feature_vector(include_prev=True)

        groups = {
            "Current bar (raw)":   {k: v for k, v in fv.items() if k.startswith("curr_")},
            "Previous bar (raw)":  {k: v for k, v in fv.items() if k.startswith("prev_")},
            "Two-bar (raw)":       {k: v for k, v in fv.items() if k.startswith("two_")},
            "Log deltas":          {k: v for k, v in fv.items() if k.startswith("dlog_")},
            "Categoricals":        {k: v for k, v in fv.items() if k.startswith("cat_")},
        }

        for group, metrics in groups.items():
            print(f"\n{'─' * 40}")
            print(f"  {group}")
            print(f"{'─' * 40}")
            for k, v in metrics.items():
                if isinstance(v, int):
                    print(f"  {k:<30} {v:>6d}")
                else:
                    print(f"  {k:<30} {v:>10.6f}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prev = {"open": 1.1000, "high": 1.1080, "low": 1.0950, "close": 1.1060}
    curr = {"open": 1.1065, "high": 1.1145, "low": 1.1020, "close": 1.1040}

    cm = CandleMetrics(prev, curr)
    cm.summary()

    print("\n\n=== Feature vector (dict) ===")
    fv = cm.feature_vector(include_prev=True)
    for k, v in fv.items():
        print(f"  {k}: {v}")
