"""Incremental feature builder — single source of truth for train/live parity.

Pattern adapted from <GITHUB_USER>/new_swt/data/incremental_feature_builder.py.
Core principle: same class, same formula functions, two entry points:
  - initialize_from_history(df)  — walks historical bars chronologically
  - process_new_bar(bar)         — called once per closed bar in live

Training and live produce IDENTICAL features when fed identical chronological bars.

Features computed (see process_new_bar for full dict keys).

Extensible: add new features as new incremental updaters + state buffers.
Each new feature must pass research/experiments/cma_5in/indicator_loop/validate.py
(parity vs reference + causality probe) before being used in experiments.

SERIALIZATION
-------------
FXFeatureBuilder.to_dict() / from_dict(d) serialize / restore the complete internal
state. Combined with lib/feature_state_db.py, this allows a live service to resume
processing on the next bar after a restart without replaying any history.

See also:
  lib/incremental_topsbots.py — O(1) TopsBots swing detector (serializable state)
  lib/feature_state_db.py     — DuckDB persistence layer for any to_dict() state
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import json
import math
import numpy as np
import pandas as pd


# ── Constants matching the training pipeline ──────────────────────────
# ASI parameters (Wilder)
ASI_ATR_PERIOD = 14
ASI_ATR_MULT = 3.0

# MC pipeline: SMA5(ASI) → EMA3-EMA5 → D → 5-lag sign count
SMA_SMOOTHING_PERIOD = 5
EMA_FAST_PERIOD = 3
EMA_SLOW_PERIOD = 5
MC_SIGN_LAGS = 5

# Multi-TF config (matching lib/asi_indicator.TF_BARS_S5 — but curator uses M5 so this scales)
# Note: the original training uses TF_BARS at S5 cadence. For M5-native incremental,
# we re-interpret TF_BARS as multiples of M5 bars. The curator's training parquet
# was built by compute_asi_mc() which uses TF_BARS_S5 on M5 data — treating each
# M5 bar as a single sample. For parity we match that exactly.
TF_BARS = np.array([1, 2, 6, 12, 24, 60, 120, 360, 720], dtype=np.int64)
TF_SEC = [5, 10, 30, 60, 120, 300, 600, 1800, 3600]
TF_WEIGHTS = np.array([math.log2(max(s / 5, 1)) + 1 for s in TF_SEC], dtype=np.float64)
N_TFS = len(TF_BARS)

# ER_norm
ER_WINDOW = 60

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_ATR_PERIOD = 14

# TEC5 (signed Kaufman ER, 5-bar)
TEC5_WINDOW = 5

# Batch 1 windows
ROC_WINDOW = 10
RANGE_POS_WINDOW = 30
RSI_WINDOW = 14
BB_WINDOW = 20
AROON_WINDOW = 25

# Batch 2 windows
EMA8_PERIOD = 8
EMA21_PERIOD = 21
CCI_WINDOW = 20
STOCH_K_WINDOW = 14
STOCH_D_WINDOW = 3
ATR_RATIO_WINDOW = 14
DONCHIAN_WINDOW = 20

# Batch 3 windows (candle + seasonality)
RANGE_EXP_WINDOW = 20

# Batch 4 windows (williams_r, bb_pos, dTEC, regime)
WILLIAMS_WINDOW = 14
BB_POS_WINDOW = 20
DTEC_SHORT = 3
DTEC_LONG = 13
REGIME_SLOPE_SPAN = 24
REGIME_SLOPE_LOOKBACK = 12
REGIME_ROLL_WINDOW = 2016  # 1wk of M5 bars

# Batch 5 windows (tsf_slope, supertrend, adx, hh/hl_price, h1_slope)
TSF_WINDOW = 20
SUPERTREND_ATR_MULT = 3.0
SUPERTREND_ATR_PERIOD = 10
ADX_PERIOD = 14
HHLL_CHANNEL = 200
H1_BARS_PER = 12          # 12 M5 bars = 1h
H1_SLOPE_LOOKBACK = 2     # 3-bar slope

# Batch 6 (hh/hl_asi, squeeze, psar)
ASI_CHANNEL = 200
KELTNER_EMA = 20
KELTNER_ATR = 10
KELTNER_MULT = 1.5
PSAR_AF_START = 0.02
PSAR_AF_MAX = 0.2
PSAR_AF_STEP = 0.02

# Batch 7 (volume-derived: CVD + vol ratio + volume profile)
VOL_SMA_WINDOW = 20
CVD_DIV_WINDOW = 30
VP_WINDOW = 100     # rolling bars in volume profile
VP_BINS = 20        # # price bins per profile


# ── IncrementalZscore ────────────────────────────────────────────────────────

class IncrementalZscore:
    """Sign-preserving z-score via sliding-window variance, O(1) per bar.

    Computes: arctan(r / rolling_std(r, pop)) / (π/2)  ∈ (-1, +1)

    The normalization window contains the PREVIOUS `pop` returns (not the current one).
    The current return r is normalized by the std of the window at the time r arrives,
    then r is appended to the window. This matches the batch algorithm in exp_lgbm.py:

        s1 = sum(r[i-pop..i-1])
        s1q = sum(r[i-pop..i-1]²)
        v1 = s1q/pop - (s1/pop)²
        f1[i] = arctan(r[i] / sqrt(v1)) / (π/2)

    STATE (serializable)
    --------------------
    The sliding window is stored as a bounded deque. Running sum and sum-of-squares
    are maintained for O(1) variance updates via the add-newest / subtract-oldest trick.

    WARMUP
    ------
    Returns 0.0 until the window contains ≥ 2 returns (std cannot be estimated from 0 or 1 sample).
    Full accuracy requires `pop` bars in the window.

    SERIALIZATION
    -------------
    to_dict() stores pop + the full window contents. from_dict() rebuilds sum/sum_sq in O(pop).
    """

    def __init__(self, pop: int = 1000) -> None:
        self.pop = pop
        self._dq: deque = deque(maxlen=pop)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._HP: float = math.pi / 2.0

    def update(self, r: float) -> float:
        """Feed one return value; return the arctan-normalized z-score for r.

        Normalization uses the window BEFORE appending r (strictly causal):
        the std is computed from the past `pop` returns, then r is added to the window.
        """
        n = len(self._dq)
        if n >= 2:
            var = self._sum_sq / n - (self._sum / n) ** 2
            std = math.sqrt(max(var, 1e-20))
        else:
            std = 1e-10

        score = math.atan(r / std) / self._HP if std > 0 else 0.0

        # Slide window: remove oldest if at capacity
        if n == self.pop:
            old = self._dq[0]
            self._sum    -= old
            self._sum_sq -= old * old
        self._dq.append(r)
        self._sum    += r
        self._sum_sq += r * r

        return score

    def to_dict(self) -> dict:
        """Serialize state. Contains pop + all window values (up to `pop` floats)."""
        return {'pop': self.pop, 'vals': list(self._dq)}

    @classmethod
    def from_dict(cls, d: dict) -> 'IncrementalZscore':
        """Restore from serialized dict. Rebuilds running sums in O(pop)."""
        obj = cls.__new__(cls)
        obj.pop = d['pop']
        vals = d['vals']
        obj._dq = deque(vals, maxlen=obj.pop)
        obj._sum    = sum(vals)
        obj._sum_sq = sum(v * v for v in vals)
        obj._HP = math.pi / 2.0
        return obj

    def __repr__(self) -> str:
        return f"IncrementalZscore(pop={self.pop}, n={len(self._dq)})"


# ── Helper: Wilder-style EMA (used for ATR) and standard EMA ──────────
def _ema_update(prev: Optional[float], new_val: float, period: int) -> float:
    """Standard EMA: alpha = 2/(period+1)."""
    if prev is None:
        return new_val
    alpha = 2.0 / (period + 1.0)
    return alpha * new_val + (1.0 - alpha) * prev


def _wilder_ema_update(prev: Optional[float], new_val: float, period: int) -> float:
    """Wilder's smoothing (RMA): alpha = 1/period."""
    if prev is None:
        return new_val
    alpha = 1.0 / period
    return alpha * new_val + (1.0 - alpha) * prev


# ── Per-TF incremental MC(D)/MC(dD) state ─────────────────────────────
@dataclass
class TFState:
    """State for one timeframe of the MC consensus."""
    bp: int                              # bars per TF sample (e.g., 1, 6, 60)
    weight: float                        # weight in consensus
    counter: int = 0                     # bars since last sample
    ema_fast: Optional[float] = None     # EMA(period=3) of input series
    ema_slow: Optional[float] = None     # EMA(period=5) of input series
    d_values: deque = field(default_factory=lambda: deque(maxlen=8))
    # ^ stores last 8 D values: enough for mc_dd which needs d[i-7..i]
    n_samples: int = 0                   # how many times this TF has been updated
    last_mc_d: float = 0.0               # cached output between samples
    last_mc_dd: float = 0.0              # cached output between samples

    def step(self, new_input: float) -> Tuple[float, float]:
        """Process one incoming sample at this TF's rate (called only when counter==bp).
        Returns (mc_d_this_tf, mc_dd_this_tf), both 0.0 until warmed up."""
        self.ema_fast = _ema_update(self.ema_fast, new_input, EMA_FAST_PERIOD)
        self.ema_slow = _ema_update(self.ema_slow, new_input, EMA_SLOW_PERIOD)
        d = self.ema_fast - self.ema_slow
        self.d_values.append(d)
        self.n_samples += 1

        # mc_d: 5-lag sign count on consecutive D differences
        mc_d = 0.0
        mc_dd = 0.0
        if len(self.d_values) >= MC_SIGN_LAGS + 1:
            pos = neg = 0
            dv = list(self.d_values)
            # sign of d[i] - d[i-1] for last 5 indices
            for lag in range(MC_SIGN_LAGS):
                change = dv[-1 - lag] - dv[-2 - lag]
                if change > 0: pos += 1
                elif change < 0: neg += 1
            mc_d = (pos - neg) / MC_SIGN_LAGS

        # mc_dd: sign count on changes in second-difference
        if len(self.d_values) >= MC_SIGN_LAGS + 3:
            pos = neg = 0
            dv = list(self.d_values)
            # For lag in 0..4: j = len-1 - lag (absolute index within dv)
            # dd_now = dv[j] - 2*dv[j-1] + dv[j-2]
            # dd_prev = dv[j-1] - 2*dv[j-2] + dv[j-3]
            n = len(dv)
            for lag in range(MC_SIGN_LAGS):
                j = n - 1 - lag
                if j >= 3:
                    dd_now = dv[j] - 2.0 * dv[j - 1] + dv[j - 2]
                    dd_prev = dv[j - 1] - 2.0 * dv[j - 2] + dv[j - 3]
                    change = dd_now - dd_prev
                    if change > 0: pos += 1
                    elif change < 0: neg += 1
            mc_dd = (pos - neg) / MC_SIGN_LAGS

        return mc_d, mc_dd


# ── Main incremental builder ──────────────────────────────────────────
class FXFeatureBuilder:
    """Incremental feature builder — M5-cadence OHLC in, feature dict out.

    Usage (live):
        builder = FXFeatureBuilder(pair="CHF_JPY")
        builder.initialize_from_history(history_df)   # warmup from REST
        # ... in event loop, on each new closed M5 bar:
        features = builder.process_new_bar(o, h, l, c, timestamp)

    Usage (training):
        builder = FXFeatureBuilder(pair="CHF_JPY")
        snapshots = builder.walk_history(full_df)  # returns full parquet
    """

    def __init__(self, pair: str, smoother: str = "sma5"):
        """
        smoother: upstream ASI smoother. Options:
          - 'sma5': 5-bar SMA (original training pipeline)
          - 'kalman10': Kalman filter q=1.0, r=1.0 (best non-causal AMA sweep winner)
          - 'ema3': 3-period EMA
          - 'rma5': Wilder RMA period=5
        """
        self.pair = pair
        self.smoother = smoother

        # ── OHLC rolling buffers ────────────────────────────────────
        BUFFER = 300
        self.opens = deque(maxlen=BUFFER)
        self.highs = deque(maxlen=BUFFER)
        self.lows = deque(maxlen=BUFFER)
        self.closes = deque(maxlen=BUFFER)
        self.timestamps = deque(maxlen=BUFFER)

        # ── ASI state (cumulative SI) ──────────────────────────────
        self.asi = 0.0
        self.atr = 0.0  # Wilder ATR14 of TR
        self.prev_close: Optional[float] = None
        self.prev_open: Optional[float] = None

        # ── Upstream smoother state ────────────────────────────────
        # For SMA5: deque of last 5 ASI values
        # For kalman10: x (state estimate), p (covariance)
        # For ema3: scalar EMA
        self.asi_window = deque(maxlen=SMA_SMOOTHING_PERIOD)  # SMA fallback
        self.smoother_x: Optional[float] = None    # Kalman state / EMA value
        self.smoother_p: float = 1.0                 # Kalman covariance

        # ── Multi-TF MC state ──────────────────────────────────────
        self.tf_states = [TFState(bp=int(TF_BARS[i]), weight=float(TF_WEIGHTS[i]))
                          for i in range(N_TFS)]

        # ── MACD state ─────────────────────────────────────────────
        self.macd_ema_fast: Optional[float] = None  # EMA12(close)
        self.macd_ema_slow: Optional[float] = None  # EMA26(close)
        self.macd_signal: Optional[float] = None    # EMA9(MACD_line)
        self.macd_atr: Optional[float] = None       # Wilder ATR14 for normalization

        # ── TEC5 state (signed 5-bar Kaufman ER) ───────────────────
        # closes_tec holds last 6 closes; abs_changes holds last 5 |Δclose|
        self.tec_closes: deque = deque(maxlen=TEC5_WINDOW + 1)
        self.tec_abs_changes: deque = deque(maxlen=TEC5_WINDOW)

        # ── Batch 1: roc, range_pos, rsi, bb_width, aroon ──────────
        self.roc_closes: deque = deque(maxlen=ROC_WINDOW + 1)
        self.rp_highs: deque = deque(maxlen=RANGE_POS_WINDOW)
        self.rp_lows: deque = deque(maxlen=RANGE_POS_WINDOW)
        self.rsi_prev_close: Optional[float] = None
        self.rsi_avg_gain: Optional[float] = None
        self.rsi_avg_loss: Optional[float] = None
        self.rsi_samples: int = 0
        self.bb_closes: deque = deque(maxlen=BB_WINDOW)
        self.aroon_highs: deque = deque(maxlen=AROON_WINDOW)
        self.aroon_lows: deque = deque(maxlen=AROON_WINDOW)

        # ── Batch 2: ema8/21_ratio, cci, stoch_d, atr_ratio, donchian ─
        self.ema8: Optional[float] = None
        self.ema21: Optional[float] = None
        self.cci_tp: deque = deque(maxlen=CCI_WINDOW)
        self.stoch_highs: deque = deque(maxlen=STOCH_K_WINDOW)
        self.stoch_lows: deque = deque(maxlen=STOCH_K_WINDOW)
        self.stoch_k_hist: deque = deque(maxlen=STOCH_D_WINDOW)
        self.atr_ratio_atr: Optional[float] = None
        self.atr_ratio_samples: int = 0
        self.donchian_highs: deque = deque(maxlen=DONCHIAN_WINDOW)
        self.donchian_lows: deque = deque(maxlen=DONCHIAN_WINDOW)

        # ── Batch 3: candle + seasonality (needs prev bar OHLC) ────
        self.candle_prev_o: Optional[float] = None
        self.candle_prev_h: Optional[float] = None
        self.candle_prev_l: Optional[float] = None
        self.candle_prev_c: Optional[float] = None
        self.candle_prev2_c: Optional[float] = None
        self.range_exp_ranges: deque = deque(maxlen=RANGE_EXP_WINDOW)

        # ── Batch 4: williams_r, bb_pos, dTEC, trending, high_vol ──
        self.williams_highs: deque = deque(maxlen=WILLIAMS_WINDOW)
        self.williams_lows: deque = deque(maxlen=WILLIAMS_WINDOW)
        self.bb_pos_closes: deque = deque(maxlen=BB_POS_WINDOW)
        self.tec_long_closes: deque = deque(maxlen=DTEC_LONG + 1)
        self.tec_long_changes: deque = deque(maxlen=DTEC_LONG)
        self.tec_short_closes: deque = deque(maxlen=DTEC_SHORT + 1)
        self.tec_short_changes: deque = deque(maxlen=DTEC_SHORT)
        # Regime EMA24_slope tracking
        self.regime_ema24: Optional[float] = None
        self.regime_ema_hist: deque = deque(maxlen=REGIME_SLOPE_LOOKBACK + 1)
        # Rolling 1wk windows for median/rank (abs_slope + atr14)
        self.regime_slope_abs_hist: deque = deque(maxlen=REGIME_ROLL_WINDOW)
        self.regime_atr_hist: deque = deque(maxlen=REGIME_ROLL_WINDOW)
        # Share ATR14 from atr_ratio_atr (already computed)

        # ── Batch 7: volume-derived (CVD + vol ratio + volume profile)
        self.vol_prev_close: Optional[float] = None
        self.cvd_cum: float = 0.0
        self.vol_ratio_hist: deque = deque(maxlen=VOL_SMA_WINDOW)
        # CVD divergence: store last 30 (close, cvd_cum) pairs
        self.cvd_div_closes: deque = deque(maxlen=CVD_DIV_WINDOW)
        self.cvd_div_cvd: deque = deque(maxlen=CVD_DIV_WINDOW)
        # Volume profile: rolling window of (close, volume) pairs — rebuild histogram when needed
        self.vp_closes: deque = deque(maxlen=VP_WINDOW)
        self.vp_volumes: deque = deque(maxlen=VP_WINDOW)

        # ── Batch 6: hh/hl_asi, squeeze, psar ──────────────────────
        self.asi_chan: deque = deque(maxlen=ASI_CHANNEL)
        self.keltner_ema: Optional[float] = None
        self.keltner_atr: Optional[float] = None
        self.keltner_samples: int = 0
        self.psar_af: float = PSAR_AF_START
        self.psar_trend: int = 1
        self.psar_ep: Optional[float] = None   # extreme point
        self.psar_sar: Optional[float] = None

        # ── Batch 5: tsf_slope, supertrend, adx, hh/hl_price, h1_slope ──
        self.tsf_closes: deque = deque(maxlen=TSF_WINDOW)
        # Supertrend: own ATR10 + trend state
        self.st_atr: Optional[float] = None
        self.st_samples: int = 0
        self.st_trend: int = 1      # +1 up, -1 down
        self.st_upper: Optional[float] = None
        self.st_lower: Optional[float] = None
        # ADX: Wilder smoothed +DM / -DM / TR, then smoothed DX -> ADX
        self.adx_pdm: Optional[float] = None
        self.adx_ndm: Optional[float] = None
        self.adx_tr: Optional[float] = None
        self.adx_adx: Optional[float] = None
        self.adx_samples: int = 0
        self.adx_prev_h: Optional[float] = None   # prior bar high for DM computation
        self.adx_prev_l: Optional[float] = None   # prior bar low for DM computation
        # 200-bar channel (hh_price, hl_price)
        self.chan_highs: deque = deque(maxlen=HHLL_CHANNEL)
        self.chan_lows: deque = deque(maxlen=HHLL_CHANNEL)
        # H1 aggregator — M5 bars roll up into a closed H1 candle every 12 bars
        self.h1_bar_count: int = 0
        self.h1_cur_o: Optional[float] = None
        self.h1_cur_h: Optional[float] = None
        self.h1_cur_l: Optional[float] = None
        self.h1_cur_c: Optional[float] = None
        self.h1_closes: deque = deque(maxlen=H1_SLOPE_LOOKBACK + 2)
        self.h1_slope_last: float = 0.0

        # ── Counters ───────────────────────────────────────────────
        self.bar_count = 0
        self.tr_samples = 0  # for ATR warmup

    # ── Core per-bar update (single source of truth) ─────────────
    def _compute_asi_bar(self, o: float, h: float, l: float, c: float) -> None:
        """Update ASI (Wilder) and its inputs. Cumulative — state is self.asi."""
        EPSILON = 1e-10

        if self.prev_close is None:
            # First bar: initialize ATR to range, ASI stays 0
            self.atr = h - l
            self.tr_samples = 1
            self.prev_close = c
            self.prev_open = o
            return

        # True range
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))

        # Wilder ATR (training uses cumulative mean for first 14 then Wilder)
        if self.tr_samples < ASI_ATR_PERIOD:
            self.tr_samples += 1
            self.atr = self.atr + (tr - self.atr) / self.tr_samples
        else:
            self.atr = (self.atr * (ASI_ATR_PERIOD - 1) + tr) / ASI_ATR_PERIOD

        # Wilder SI
        C1 = self.prev_close
        O1 = self.prev_open
        C2, O2, H2, L2 = c, o, h, l

        N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
        t1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
        t2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
        t3 = (H2 - L2) + 0.25 * abs(C1 - O1)
        R = max(t1, t2, t3)
        if R <= 0:
            R = EPSILON
        K = max(abs(H2 - C1), abs(L2 - C1))
        limit = ASI_ATR_MULT * self.atr
        if limit < EPSILON:
            limit = EPSILON
        SI = 50.0 * (N / R) * (K / limit)
        self.asi += SI

        self.prev_close = c
        self.prev_open = o

    def _compute_mc_multi_tf(self, sma5_asi: float) -> Tuple[float, float]:
        """Feed SMA5(ASI) to all TFs, return weighted-avg mc_d, mc_dd.

        Each TF only samples every `bp` bars, but we accumulate the LAST KNOWN
        value continuously between samples (matching training's tf_i = i // bp behavior
        once warmed up). This is causal — no lookahead.
        """
        mc_d_sum = 0.0
        mc_dd_sum = 0.0
        tw = 0.0

        for tfs in self.tf_states:
            tfs.counter += 1
            if tfs.counter >= tfs.bp:
                tfs.counter = 0
                mc_d_tf, mc_dd_tf = tfs.step(sma5_asi)
                tfs.last_mc_d = mc_d_tf
                tfs.last_mc_dd = mc_dd_tf
            # Always accumulate the LAST KNOWN value if this TF has warmed up
            if tfs.n_samples >= MC_SIGN_LAGS + 5:
                mc_d_sum += tfs.weight * tfs.last_mc_d
                mc_dd_sum += tfs.weight * tfs.last_mc_dd
                tw += tfs.weight

        if tw > 0:
            return mc_d_sum / tw, mc_dd_sum / tw
        return 0.0, 0.0

    def _compute_er_norm(self) -> float:
        """Kaufman ER over last 60 closes, arctan-normalized. Returns 0 until warm."""
        if len(self.closes) < ER_WINDOW + 1:
            return 0.0
        closes = np.array(list(self.closes)[-(ER_WINDOW + 1):], dtype=np.float64)
        net = abs(closes[-1] - closes[0])
        path = float(np.sum(np.abs(np.diff(closes))))
        if path <= 0:
            return 0.0
        er = net / path
        return float(np.arctan(er / 0.3) / (np.pi / 2))

    def _compute_roc_10(self, close: float) -> float:
        """10-bar rate of change: (close[t] - close[t-10]) / close[t-10]. 0 until warm."""
        self.roc_closes.append(close)
        if len(self.roc_closes) < ROC_WINDOW + 1:
            return 0.0
        c0 = self.roc_closes[0]
        return (close - c0) / c0 if c0 != 0.0 else 0.0

    def _compute_range_pos_30(self, high: float, low: float, close: float) -> float:
        """30-bar range position: (close - min30_low) / (max30_high - min30_low)."""
        self.rp_highs.append(high)
        self.rp_lows.append(low)
        if len(self.rp_highs) < RANGE_POS_WINDOW:
            return 0.5
        mx = max(self.rp_highs)
        mn = min(self.rp_lows)
        rng = mx - mn
        if rng <= 1e-12:
            return 0.5
        return (close - mn) / rng

    def _compute_rsi_14(self, close: float) -> float:
        """Wilder RSI 14 normalized to [0, 1]. 0.5 until warm."""
        if self.rsi_prev_close is None:
            self.rsi_prev_close = close
            return 0.5
        diff = close - self.rsi_prev_close
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        self.rsi_prev_close = close
        self.rsi_samples += 1
        if self.rsi_samples <= RSI_WINDOW:
            # Seed via cumulative mean
            if self.rsi_avg_gain is None:
                self.rsi_avg_gain = gain
                self.rsi_avg_loss = loss
            else:
                n = self.rsi_samples
                self.rsi_avg_gain = self.rsi_avg_gain + (gain - self.rsi_avg_gain) / n
                self.rsi_avg_loss = self.rsi_avg_loss + (loss - self.rsi_avg_loss) / n
            if self.rsi_samples < RSI_WINDOW:
                return 0.5
        else:
            alpha = 1.0 / RSI_WINDOW
            self.rsi_avg_gain = alpha * gain + (1.0 - alpha) * self.rsi_avg_gain
            self.rsi_avg_loss = alpha * loss + (1.0 - alpha) * self.rsi_avg_loss
        if self.rsi_avg_loss <= 1e-12:
            return 1.0
        rs = self.rsi_avg_gain / self.rsi_avg_loss
        return 1.0 - 1.0 / (1.0 + rs)

    def _compute_bb_width(self, close: float) -> float:
        """(upper_bb - lower_bb) / close — 20-bar, 2σ."""
        self.bb_closes.append(close)
        if len(self.bb_closes) < BB_WINDOW:
            return 0.0
        arr = np.asarray(self.bb_closes, dtype=np.float64)
        m = arr.mean()
        sd = arr.std()
        if close <= 0:
            return 0.0
        return (4.0 * sd) / close  # (upper - lower) = 4σ

    def _compute_aroon_osc(self, high: float, low: float) -> float:
        """25-bar Aroon oscillator normalized to [-1, 1]."""
        self.aroon_highs.append(high)
        self.aroon_lows.append(low)
        n = len(self.aroon_highs)
        if n < AROON_WINDOW:
            return 0.0
        # Bars since highest high / lowest low (0 = most recent bar is the extreme)
        hs = list(self.aroon_highs)
        ls = list(self.aroon_lows)
        # Most recent bar is index n-1
        hh_idx = int(np.argmax(hs))  # oldest-wins on tie
        ll_idx = int(np.argmin(ls))
        periods_since_hh = (n - 1) - hh_idx
        periods_since_ll = (n - 1) - ll_idx
        up = (n - periods_since_hh) / n
        down = (n - periods_since_ll) / n
        return up - down

    def _compute_ema8_ratio(self, close: float) -> float:
        self.ema8 = _ema_update(self.ema8, close, EMA8_PERIOD)
        return close / self.ema8 if self.ema8 and self.ema8 > 0 else 1.0

    def _compute_ema21_ratio(self, close: float) -> float:
        self.ema21 = _ema_update(self.ema21, close, EMA21_PERIOD)
        return close / self.ema21 if self.ema21 and self.ema21 > 0 else 1.0

    def _compute_cci(self, high: float, low: float, close: float) -> float:
        """Commodity Channel Index, 20-bar. Normalized by /200 to soften range."""
        tp = (high + low + close) / 3.0
        self.cci_tp.append(tp)
        if len(self.cci_tp) < CCI_WINDOW:
            return 0.0
        arr = np.asarray(self.cci_tp, dtype=np.float64)
        sma = arr.mean()
        md = np.mean(np.abs(arr - sma))
        if md <= 1e-12:
            return 0.0
        return (tp - sma) / (0.015 * md) / 200.0  # scale: CCI typically ±200

    def _compute_stoch_kd(self, high: float, low: float, close: float) -> tuple:
        """Stochastic (%K 14, %D=3-SMA(%K)). Returns (k, d) both in [0,1]."""
        self.stoch_highs.append(high)
        self.stoch_lows.append(low)
        if len(self.stoch_highs) < STOCH_K_WINDOW:
            k = 0.5
        else:
            hh = max(self.stoch_highs)
            ll = min(self.stoch_lows)
            rng = hh - ll
            k = (close - ll) / rng if rng > 1e-12 else 0.5
        self.stoch_k_hist.append(k)
        d = float(np.mean(self.stoch_k_hist)) if self.stoch_k_hist else 0.5
        return k, d

    def _compute_atr_ratio(self, high: float, low: float, close: float) -> float:
        """ATR14/close (dimensionless)."""
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.atr_ratio_samples += 1
        if self.atr_ratio_samples <= ATR_RATIO_WINDOW:
            if self.atr_ratio_atr is None:
                self.atr_ratio_atr = tr
            else:
                self.atr_ratio_atr = self.atr_ratio_atr + (tr - self.atr_ratio_atr) / self.atr_ratio_samples
        else:
            alpha = 1.0 / ATR_RATIO_WINDOW
            self.atr_ratio_atr = alpha * tr + (1.0 - alpha) * self.atr_ratio_atr
        if close <= 0 or self.atr_ratio_atr is None:
            return 0.0
        return self.atr_ratio_atr / close

    def _compute_donchian_pos(self, high: float, low: float, close: float) -> float:
        """(close - low20) / (high20 - low20), bounded [0, 1]."""
        self.donchian_highs.append(high)
        self.donchian_lows.append(low)
        if len(self.donchian_highs) < DONCHIAN_WINDOW:
            return 0.5
        hh = max(self.donchian_highs)
        ll = min(self.donchian_lows)
        rng = hh - ll
        if rng <= 1e-12:
            return 0.5
        return (close - ll) / rng

    def _compute_batch7_volume(self, close, high, low, volume, atr14):
        """Volume-derived features: CVD (tick-rule), vol ratio, volume profile.
        Returns dict with 7 keys.
        Requires: 'volume' col in input (OANDA tick-count proxy)."""
        # 1. CVD delta (tick rule)
        if self.vol_prev_close is None or close > self.vol_prev_close:
            delta_sign = +1.0
        elif close < self.vol_prev_close:
            delta_sign = -1.0
        else:
            delta_sign = 0.0
        cvd_delta = delta_sign * float(volume)
        self.cvd_cum += cvd_delta
        self.vol_prev_close = close

        # 2. CVD normalized — arctan-scaled cumulative, bounded
        cvd_norm = math.atan(self.cvd_cum / 1e6) / (math.pi / 2)

        # 3. CVD divergence (30-bar): sign(price_slope) - sign(cvd_slope)
        self.cvd_div_closes.append(close)
        self.cvd_div_cvd.append(self.cvd_cum)
        if len(self.cvd_div_closes) < CVD_DIV_WINDOW:
            cvd_div = 0.0
        else:
            p_delta = self.cvd_div_closes[-1] - self.cvd_div_closes[0]
            c_delta = self.cvd_div_cvd[-1] - self.cvd_div_cvd[0]
            sp = 1.0 if p_delta > 0 else -1.0 if p_delta < 0 else 0.0
            sc = 1.0 if c_delta > 0 else -1.0 if c_delta < 0 else 0.0
            cvd_div = sp - sc  # in {-2, -1, 0, +1, +2}
            cvd_div = cvd_div / 2.0  # normalize to [-1, 1]

        # 4. Volume ratio vs 20-SMA
        self.vol_ratio_hist.append(float(volume))
        if len(self.vol_ratio_hist) < VOL_SMA_WINDOW:
            vol_ratio = 1.0
        else:
            avg = sum(self.vol_ratio_hist) / len(self.vol_ratio_hist)
            vol_ratio = float(volume) / avg if avg > 0 else 1.0

        # 5. Capitulation: high volume AND large body (relative to ATR)
        body = abs(close - self.candle_prev_c) if self.candle_prev_c else 0.0
        cap_vol = 1.0 if vol_ratio > 2.5 else 0.0
        cap_body = 1.0 if (atr14 > 0 and body / atr14 > 2.0) else 0.0
        vol_capitulation = cap_vol * cap_body  # only 1 when BOTH

        # 6. Volume profile: rolling 100-bar histogram of volume by price bin
        self.vp_closes.append(close)
        self.vp_volumes.append(float(volume))
        if len(self.vp_closes) < VP_WINDOW // 2:
            dist_to_poc = 0.0
            at_hvn_lvn = 0.0
        else:
            closes_arr = np.asarray(self.vp_closes, dtype=np.float64)
            vols_arr = np.asarray(self.vp_volumes, dtype=np.float64)
            lo = closes_arr.min()
            hi = closes_arr.max()
            rng = hi - lo
            if rng <= 1e-12:
                dist_to_poc = 0.0
                at_hvn_lvn = 0.0
            else:
                bin_w = rng / VP_BINS
                # Histogram: sum volume per price bin
                hist = np.zeros(VP_BINS)
                for k in range(len(closes_arr)):
                    idx = int((closes_arr[k] - lo) / bin_w)
                    if idx >= VP_BINS:
                        idx = VP_BINS - 1
                    hist[idx] += vols_arr[k]
                # POC = bin with max volume
                poc_idx = int(np.argmax(hist))
                poc_price = lo + (poc_idx + 0.5) * bin_w
                # Distance to POC, normalized by range (roughly [-0.5, +0.5] in-range)
                dist_to_poc = (close - poc_price) / rng
                # HVN/LVN: top 20% / bottom 20% of bins by volume
                sorted_vols = np.sort(hist)
                hvn_threshold = sorted_vols[int(0.8 * VP_BINS)]
                lvn_threshold = sorted_vols[int(0.2 * VP_BINS)]
                cur_idx = int((close - lo) / bin_w)
                if cur_idx >= VP_BINS:
                    cur_idx = VP_BINS - 1
                cur_vol = hist[cur_idx]
                if cur_vol >= hvn_threshold:
                    at_hvn_lvn = 1.0
                elif cur_vol <= lvn_threshold:
                    at_hvn_lvn = -1.0
                else:
                    at_hvn_lvn = 0.0

        return {
            "cvd_delta": math.atan(cvd_delta / 1000.0) / (math.pi / 2),
            "cvd_cum": cvd_norm,
            "cvd_divergence_30": cvd_div,
            "vol_ratio_20": math.atan(vol_ratio / 2.0) / (math.pi / 2),  # normalize heavy tail
            "vol_capitulation": vol_capitulation,
            "dist_to_poc_100": dist_to_poc,
            "at_hvn_lvn": at_hvn_lvn,
        }

    def _compute_hh_hl_asi(self):
        """Same pattern as hh_price/hl_price but on ASI series."""
        if len(self.asi_chan) < ASI_CHANNEL:
            return 0.0, 0.0
        mx = max(self.asi_chan)
        mn = min(self.asi_chan)
        hh = 1.0 if self.asi >= mx else 0.0
        hl = 1.0 if self.asi <= mn else 0.0
        return hh, hl

    def _compute_squeeze(self, h, l, c):
        """TTM Squeeze: 1 when BB inside Keltner (low vol compression). Uses own ATR10 + EMA20."""
        # Keltner: EMA20 ± 1.5 × ATR10
        self.keltner_ema = _ema_update(self.keltner_ema, c, KELTNER_EMA)
        if self.prev_close is None:
            tr = h - l
        else:
            # NOTE: self.prev_close already updated by ASI — but we ONLY use as rough TR estimate for Keltner.
            # Acceptable approximation since Keltner is loose.
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.keltner_samples += 1
        if self.keltner_samples <= KELTNER_ATR:
            if self.keltner_atr is None:
                self.keltner_atr = tr
            else:
                self.keltner_atr = self.keltner_atr + (tr - self.keltner_atr) / self.keltner_samples
        else:
            alpha = 1.0 / KELTNER_ATR
            self.keltner_atr = alpha * tr + (1.0 - alpha) * self.keltner_atr
        # Need bb_closes buffered already (from batch 1)
        if (len(self.bb_closes) < BB_WINDOW or self.keltner_ema is None
                or self.keltner_atr is None or self.keltner_samples < KELTNER_ATR):
            return 0.0
        arr = np.asarray(self.bb_closes, dtype=np.float64)
        sma = arr.mean()
        sd = arr.std()
        bb_upper = sma + 2.0 * sd
        bb_lower = sma - 2.0 * sd
        k_upper = self.keltner_ema + KELTNER_MULT * self.keltner_atr
        k_lower = self.keltner_ema - KELTNER_MULT * self.keltner_atr
        return 1.0 if (bb_upper <= k_upper and bb_lower >= k_lower) else 0.0

    def _compute_psar(self, h, l):
        """Parabolic SAR signal: +1 uptrend, -1 downtrend."""
        if self.psar_sar is None:
            # Initialize on first full bar (arbitrary direction + use low as initial SAR)
            if self.psar_trend == 1:
                self.psar_sar = l
                self.psar_ep = h
            else:
                self.psar_sar = h
                self.psar_ep = l
            return float(self.psar_trend)

        # Advance SAR by AF × (EP - SAR)
        new_sar = self.psar_sar + self.psar_af * (self.psar_ep - self.psar_sar)
        # Check reversal
        if self.psar_trend == 1:
            if l < new_sar:
                # Reverse to downtrend
                self.psar_trend = -1
                self.psar_sar = self.psar_ep
                self.psar_ep = l
                self.psar_af = PSAR_AF_START
            else:
                if h > self.psar_ep:
                    self.psar_ep = h
                    self.psar_af = min(self.psar_af + PSAR_AF_STEP, PSAR_AF_MAX)
                self.psar_sar = new_sar
        else:
            if h > new_sar:
                self.psar_trend = 1
                self.psar_sar = self.psar_ep
                self.psar_ep = h
                self.psar_af = PSAR_AF_START
            else:
                if l < self.psar_ep:
                    self.psar_ep = l
                    self.psar_af = min(self.psar_af + PSAR_AF_STEP, PSAR_AF_MAX)
                self.psar_sar = new_sar
        return float(self.psar_trend)

    def _compute_tsf_slope(self, c):
        """Linear-regression slope over last 20 bars, normalized by close."""
        self.tsf_closes.append(c)
        n = len(self.tsf_closes)
        if n < 3:
            return 0.0
        y = np.asarray(self.tsf_closes, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        # slope = cov(x,y) / var(x)
        xm = x.mean(); ym = y.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom <= 1e-12:
            return 0.0
        slope = float(((x - xm) * (y - ym)).sum()) / denom
        if c <= 0:
            return 0.0
        # Normalize to approximate [-1, 1] via arctan(slope/close * 1000)
        return math.atan(slope / c * 1000.0) / (math.pi / 2)

    def _compute_supertrend(self, h, l, c):
        """Supertrend signal: +1 if trend up, -1 if down. Uses ATR10 + 3x mult."""
        # Own ATR10 (different period from Wilder ATR14)
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.st_samples += 1
        if self.st_samples <= SUPERTREND_ATR_PERIOD:
            if self.st_atr is None:
                self.st_atr = tr
            else:
                self.st_atr = self.st_atr + (tr - self.st_atr) / self.st_samples
        else:
            alpha = 1.0 / SUPERTREND_ATR_PERIOD
            self.st_atr = alpha * tr + (1.0 - alpha) * self.st_atr
        if self.st_samples < SUPERTREND_ATR_PERIOD:
            return 0.0
        mid = (h + l) / 2.0
        upper = mid + SUPERTREND_ATR_MULT * self.st_atr
        lower = mid - SUPERTREND_ATR_MULT * self.st_atr
        # Adjust bands (don't relax against trend direction)
        if self.st_upper is None:
            self.st_upper = upper; self.st_lower = lower
        else:
            if upper < self.st_upper or (self.prev_close is not None and self.prev_close > self.st_upper):
                self.st_upper = upper
            if lower > self.st_lower or (self.prev_close is not None and self.prev_close < self.st_lower):
                self.st_lower = lower
        # Determine trend
        if self.st_trend == 1 and c < self.st_lower:
            self.st_trend = -1
        elif self.st_trend == -1 and c > self.st_upper:
            self.st_trend = 1
        return float(self.st_trend)

    def _compute_adx_family(self, h, l, c):
        """Returns (adx, dmi_diff) where adx∈[0,1], dmi_diff∈[-1,+1] (normalized)."""
        if self.prev_close is None:
            return 0.0, 0.0
        # True range
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        # Directional movement
        prev_h = self.candle_prev_h  # set by batch3 before this runs in process_new_bar order? No - before ASI
        # We can't rely on candle_prev_h since it's set by batch3 which runs AFTER ASI.
        # But this is called BEFORE batch3 here (we'll wire it before batch3). Use internal memory.
        return self._adx_update_internal(h, l, c, tr)

    def _adx_update_internal(self, h, l, c, tr):
        """Internal: needs its own prev_high, prev_low."""
        if self.adx_prev_h is None:
            self.adx_prev_h = h
            self.adx_prev_l = l
            return 0.0, 0.0
        up = h - self.adx_prev_h
        dn = self.adx_prev_l - l
        plus_dm = up if (up > dn and up > 0) else 0.0
        minus_dm = dn if (dn > up and dn > 0) else 0.0
        self.adx_prev_h = h; self.adx_prev_l = l

        # Wilder smooth TR, +DM, -DM
        self.adx_samples += 1
        if self.adx_samples <= ADX_PERIOD:
            s = self.adx_samples
            if self.adx_tr is None:
                self.adx_tr = tr
                self.adx_pdm = plus_dm
                self.adx_ndm = minus_dm
            else:
                self.adx_tr = self.adx_tr + (tr - self.adx_tr) / s
                self.adx_pdm = self.adx_pdm + (plus_dm - self.adx_pdm) / s
                self.adx_ndm = self.adx_ndm + (minus_dm - self.adx_ndm) / s
        else:
            alpha = 1.0 / ADX_PERIOD
            self.adx_tr = alpha * tr + (1.0 - alpha) * self.adx_tr
            self.adx_pdm = alpha * plus_dm + (1.0 - alpha) * self.adx_pdm
            self.adx_ndm = alpha * minus_dm + (1.0 - alpha) * self.adx_ndm

        if self.adx_samples < ADX_PERIOD or self.adx_tr <= 1e-12:
            return 0.0, 0.0
        pdi = self.adx_pdm / self.adx_tr
        ndi = self.adx_ndm / self.adx_tr
        dmi_diff = pdi - ndi                      # [-1, +1]
        dx = abs(pdi - ndi) / max(pdi + ndi, 1e-12)  # [0, 1]
        if self.adx_adx is None:
            self.adx_adx = dx
        else:
            alpha = 1.0 / ADX_PERIOD
            self.adx_adx = alpha * dx + (1.0 - alpha) * self.adx_adx
        return self.adx_adx, dmi_diff

    def _compute_hh_hl_price(self, c):
        """hh_price: 1 if close >= rolling 200-bar high, else 0.
        hl_price: 1 if close <= rolling 200-bar low, else 0."""
        # Note: channel_highs already appended this bar by the time we call (after candle memory shift).
        # We actually update channel here to avoid order coupling with batch3.
        if len(self.chan_highs) < HHLL_CHANNEL:
            return 0.0, 0.0
        mx = max(self.chan_highs)
        mn = min(self.chan_lows)
        hh = 1.0 if c >= mx else 0.0
        hl = 1.0 if c <= mn else 0.0
        return hh, hl

    def _compute_h1_slope(self, o, h, l, c):
        """Aggregate 12 M5 bars into an H1 candle, compute 3-bar linreg slope on H1 close,
        arctan-normalize. Returns last-known slope between H1 closes."""
        # Accumulate current H1 candle
        if self.h1_cur_o is None:
            self.h1_cur_o = o
            self.h1_cur_h = h
            self.h1_cur_l = l
        else:
            if h > self.h1_cur_h: self.h1_cur_h = h
            if l < self.h1_cur_l: self.h1_cur_l = l
        self.h1_cur_c = c
        self.h1_bar_count += 1
        # On 12th M5 bar, close H1 candle
        if self.h1_bar_count >= H1_BARS_PER:
            self.h1_closes.append(self.h1_cur_c)
            self.h1_bar_count = 0
            self.h1_cur_o = None
            self.h1_cur_h = None
            self.h1_cur_l = None
            self.h1_cur_c = None
            if len(self.h1_closes) >= H1_SLOPE_LOOKBACK + 1:
                y0 = self.h1_closes[-(H1_SLOPE_LOOKBACK + 1)]
                y2 = self.h1_closes[-1]
                slope_per_bar = (y2 - y0) / H1_SLOPE_LOOKBACK
                # Normalize — use close as scale
                if c > 0:
                    slope_pct = slope_per_bar / c * 10000.0
                    self.h1_slope_last = math.atan(slope_pct / 20.0) / (math.pi / 2)
                else:
                    self.h1_slope_last = 0.0
        return self.h1_slope_last

    def _compute_williams_r(self, h, l, c):
        """Williams %R in [-1, 0]. Essentially -1 * (1 - stoch_k)."""
        self.williams_highs.append(h)
        self.williams_lows.append(l)
        if len(self.williams_highs) < WILLIAMS_WINDOW:
            return -0.5
        hh = max(self.williams_highs)
        ll = min(self.williams_lows)
        rng = hh - ll
        if rng <= 1e-12:
            return -0.5
        return -(hh - c) / rng  # in [-1, 0]

    def _compute_bb_pos(self, c):
        """Close position inside 20-bar bollinger bands: (c - sma) / (2*std). ~[-2, 2]."""
        self.bb_pos_closes.append(c)
        if len(self.bb_pos_closes) < BB_POS_WINDOW:
            return 0.0
        arr = np.asarray(self.bb_pos_closes, dtype=np.float64)
        sma = arr.mean()
        sd = arr.std()
        if sd <= 1e-12:
            return 0.0
        return (c - sma) / (2.0 * sd)

    def _compute_dTEC(self, c):
        """TEC(3) - TEC(13), signed Kaufman ER diff across two lengths."""
        def _update(closes: deque, changes: deque, window: int):
            closes.append(c)
            if len(closes) >= 2:
                prev = closes[-2]
                changes.append(abs(c - prev))
            if len(closes) < window + 1 or len(changes) < window:
                return 0.0
            net = closes[-1] - closes[0]
            path = float(sum(changes))
            if path <= 1e-12:
                return 0.0
            er = abs(net) / path
            return er if net > 0 else -er if net < 0 else 0.0
        tec_s = _update(self.tec_short_closes, self.tec_short_changes, DTEC_SHORT)
        tec_l = _update(self.tec_long_closes, self.tec_long_changes, DTEC_LONG)
        return tec_s - tec_l

    def _compute_regime(self, c):
        """Returns (trending, high_vol) binaries from 1wk rolling medians.
        Uses self.atr_ratio_atr × close as ATR14 magnitude."""
        # EMA24 + 12-bar slope
        alpha24 = 2.0 / (REGIME_SLOPE_SPAN + 1.0)
        if self.regime_ema24 is None:
            self.regime_ema24 = c
        else:
            self.regime_ema24 = alpha24 * c + (1.0 - alpha24) * self.regime_ema24
        self.regime_ema_hist.append(self.regime_ema24)
        slope = 0.0
        if len(self.regime_ema_hist) > REGIME_SLOPE_LOOKBACK:
            slope = (self.regime_ema_hist[-1] - self.regime_ema_hist[0]) / REGIME_SLOPE_LOOKBACK
        abs_slope = abs(slope)
        self.regime_slope_abs_hist.append(abs_slope)
        atr14 = (self.atr_ratio_atr or 0.0)
        self.regime_atr_hist.append(atr14)
        # Medians over rolling 1-week window
        if len(self.regime_slope_abs_hist) < 50:
            return 0.0, 0.0
        slope_med = float(np.median(self.regime_slope_abs_hist))
        atr_med = float(np.median(self.regime_atr_hist))
        trending = 1.0 if abs_slope > slope_med else 0.0
        high_vol = 1.0 if atr14 > atr_med else 0.0
        return trending, high_vol

    def _compute_vol_regime(self, c):
        """Continuous vol regime: atr14 / rolling-median(atr14). ~[0, 2+]."""
        atr14 = (self.atr_ratio_atr or 0.0)
        if len(self.regime_atr_hist) < 50:
            return 1.0
        med = float(np.median(self.regime_atr_hist))
        return atr14 / med if med > 1e-12 else 1.0

    def _compute_batch3_candle(self, o, h, l, c):
        """Compute all batch-3 candle features in one pass. Returns dict."""
        body = c - o
        rng = h - l
        body_abs = abs(body)
        # Pip-free body_pips (return as fraction of close ×1e4)
        body_pips = (body / c) * 1e4 if c > 0 else 0.0
        # candle_range as fraction of close (pip-free)
        candle_range = (rng / c) if c > 0 else 0.0
        body_ratio = (body_abs / rng) if rng > 1e-12 else 0.0

        # Features needing prev bar
        if self.candle_prev_c is None:
            gap_norm = 0.0
            two_bar_mom = 0.0
            dlog = 0.0
            is_engulf = 0.0
            is_inside = 0.0
        else:
            prev_rng = self.candle_prev_h - self.candle_prev_l
            # gap_norm: open - prev_close, normalized by prev range
            gap_norm = ((o - self.candle_prev_c) / prev_rng) if prev_rng > 1e-12 else 0.0
            # Engulfing: current body fully covers prev body (sign matters)
            prev_body = self.candle_prev_c - self.candle_prev_o
            if body > 0 and prev_body < 0 and c >= self.candle_prev_o and o <= self.candle_prev_c:
                is_engulf = 1.0
            elif body < 0 and prev_body > 0 and c <= self.candle_prev_o and o >= self.candle_prev_c:
                is_engulf = -1.0
            else:
                is_engulf = 0.0
            # Inside bar: current H<=prev_H AND current L>=prev_L
            is_inside = 1.0 if (h <= self.candle_prev_h and l >= self.candle_prev_l) else 0.0
            # dlog close
            if self.candle_prev_c > 0 and c > 0:
                dlog = math.log(c / self.candle_prev_c)
            else:
                dlog = 0.0
            # Two-bar momentum: (c - c_prev2) / prev2_range  — need 2 bars back
            if self.candle_prev2_c is not None and prev_rng > 1e-12:
                two_bar_mom = (c - self.candle_prev2_c) / prev_rng
            else:
                two_bar_mom = 0.0

        # Range expansion
        self.range_exp_ranges.append(rng)
        if len(self.range_exp_ranges) < RANGE_EXP_WINDOW:
            range_exp = 1.0
        else:
            avg_rng = float(np.mean(self.range_exp_ranges))
            range_exp = (rng / avg_rng) if avg_rng > 1e-12 else 1.0

        # Update candle memory (shift)
        self.candle_prev2_c = self.candle_prev_c
        self.candle_prev_o = o
        self.candle_prev_h = h
        self.candle_prev_l = l
        self.candle_prev_c = c

        return {
            "body_pips": body_pips,
            "body_ratio": body_ratio,
            "candle_range": candle_range,
            "gap_norm": gap_norm,
            "two_bar_momentum": two_bar_mom,
            "dlog_close_pos": dlog,
            "range_expansion": range_exp,
            "is_engulfing": is_engulf,
            "is_inside_bar": is_inside,
        }

    def _compute_seasonality(self, timestamp):
        """hour_sin/cos, dow_sin/cos from timestamp. Robust to None."""
        if timestamp is None:
            return {"hour_sin": 0.0, "hour_cos": 1.0, "dow_sin": 0.0, "dow_cos": 1.0}
        try:
            ts = pd.Timestamp(timestamp)
            hour = ts.hour + ts.minute / 60.0
            dow = ts.weekday()
        except Exception:
            return {"hour_sin": 0.0, "hour_cos": 1.0, "dow_sin": 0.0, "dow_cos": 1.0}
        return {
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "dow_sin":  math.sin(2 * math.pi * dow / 7.0),
            "dow_cos":  math.cos(2 * math.pi * dow / 7.0),
        }

    def _compute_tec5(self, close: float) -> float:
        """Signed Kaufman ER over last 5 bars on close.
        Causal by construction — uses only past+current closes.
        Returns 0.0 until 5 bars of changes accumulated.
        """
        prev_close = self.tec_closes[-1] if len(self.tec_closes) > 0 else None
        self.tec_closes.append(close)
        if prev_close is not None:
            self.tec_abs_changes.append(abs(close - prev_close))
        if len(self.tec_closes) < TEC5_WINDOW + 1 or len(self.tec_abs_changes) < TEC5_WINDOW:
            return 0.0
        net = self.tec_closes[-1] - self.tec_closes[0]
        path = float(sum(self.tec_abs_changes))
        if path <= 1e-12:
            return 0.0
        er = abs(net) / path
        if net > 0:
            return er
        elif net < 0:
            return -er
        return 0.0

    def _compute_macd_hist(self, close: float, high: float, low: float) -> float:
        """True MACD histogram / Wilder ATR14 — matches training compute_macd_hist."""
        # Update EMAs
        self.macd_ema_fast = _ema_update(self.macd_ema_fast, close, MACD_FAST)
        self.macd_ema_slow = _ema_update(self.macd_ema_slow, close, MACD_SLOW)
        macd_line = self.macd_ema_fast - self.macd_ema_slow
        self.macd_signal = _ema_update(self.macd_signal, macd_line, MACD_SIGNAL)
        hist = macd_line - self.macd_signal

        # Wilder ATR for normalization (EMA with alpha=1/14, separate from ASI's ATR)
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.macd_atr = _wilder_ema_update(self.macd_atr, tr, MACD_ATR_PERIOD)

        if self.macd_atr is None or self.macd_atr <= 0:
            return 0.0
        return hist / self.macd_atr

    # ── Public API ───────────────────────────────────────────────
    def process_new_bar(self, o: float, h: float, l: float, c: float,
                        timestamp=None, volume: float = 0.0) -> Dict[str, float]:
        """Update state with one new closed M5 bar. Returns feature dict.

        CRITICAL: Order of operations matters. Update macd BEFORE asi
        (macd uses prev_close before asi overwrites it).

        volume: OANDA tick-count per bar (0 if not available) — used for batch 7.
        """
        # 1. Compute MACD first (uses prev_close before ASI updates it)
        macd_hist = self._compute_macd_hist(c, h, l)

        # 1b. atr_ratio + supertrend also need old prev_close (same TR calc)
        atr_ratio = self._compute_atr_ratio(h, l, c)
        supertrend = self._compute_supertrend(h, l, c)

        # 2. ASI pipeline
        self._compute_asi_bar(o, h, l, c)

        # 3. Upstream smoother (configurable: sma5 / kalman10 / ema3 / rma5)
        if self.smoother == "sma5":
            self.asi_window.append(self.asi)
            sma5_asi = float(np.mean(list(self.asi_window))) if self.asi_window else 0.0
        elif self.smoother == "kalman10":
            # Kalman filter with q=1.0, r=1.0 (top non-causal AMA sweep result)
            q, r = 1.0, 1.0
            if self.smoother_x is None:
                self.smoother_x = self.asi
            else:
                p_pred = self.smoother_p + q
                k = p_pred / (p_pred + r)
                self.smoother_x = self.smoother_x + k * (self.asi - self.smoother_x)
                self.smoother_p = (1.0 - k) * p_pred
            sma5_asi = self.smoother_x
        elif self.smoother == "ema3":
            if self.smoother_x is None:
                self.smoother_x = self.asi
            else:
                alpha = 2.0 / (3 + 1.0)
                self.smoother_x = alpha * self.asi + (1.0 - alpha) * self.smoother_x
            sma5_asi = self.smoother_x
        elif self.smoother == "rma5":
            if self.smoother_x is None:
                self.smoother_x = self.asi
            else:
                alpha = 1.0 / 5
                self.smoother_x = alpha * self.asi + (1.0 - alpha) * self.smoother_x
            sma5_asi = self.smoother_x
        else:
            raise ValueError(f"Unknown smoother: {self.smoother}")

        # 4. Multi-TF MC
        mc_d, mc_dd = self._compute_mc_multi_tf(sma5_asi)

        # 5. ER_norm (done last, before storing close in deque for this bar)
        # Actually ER needs the current close in the window, so store first:
        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)
        self.timestamps.append(timestamp)
        er_norm = self._compute_er_norm()

        # 6. TEC5 (signed 5-bar Kaufman ER on close)
        tec5 = self._compute_tec5(c)

        # 7. Batch 1 candidates
        roc_10 = self._compute_roc_10(c)
        range_pos_30 = self._compute_range_pos_30(h, l, c)
        rsi_14 = self._compute_rsi_14(c)
        bb_width = self._compute_bb_width(c)
        aroon_osc = self._compute_aroon_osc(h, l)

        # 8. Batch 2 candidates
        ema8_ratio = self._compute_ema8_ratio(c)
        ema21_ratio = self._compute_ema21_ratio(c)
        cci = self._compute_cci(h, l, c)
        stoch_k, stoch_d = self._compute_stoch_kd(h, l, c)
        donchian_pos = self._compute_donchian_pos(h, l, c)

        # 9. Batch 3 candle + seasonality
        b3 = self._compute_batch3_candle(o, h, l, c)
        s = self._compute_seasonality(timestamp)

        # 10. Batch 4 (regime uses atr_ratio_atr which is already set)
        williams_r = self._compute_williams_r(h, l, c)
        bb_pos = self._compute_bb_pos(c)
        dTEC = self._compute_dTEC(c)
        trending, high_vol = self._compute_regime(c)
        vol_regime = self._compute_vol_regime(c)

        # 11. Batch 5 (supertrend computed earlier; hh/hl check PAST channel first)
        tsf_slope = self._compute_tsf_slope(c)
        hh_price, hl_price = self._compute_hh_hl_price(c)
        # Append current AFTER check so past check doesn't include current bar
        self.chan_highs.append(h)
        self.chan_lows.append(l)
        adx, dmi_diff = self._compute_adx_family(h, l, c)
        h1_slope = self._compute_h1_slope(o, h, l, c)

        # 12. Batch 6
        hh_asi, hl_asi = self._compute_hh_hl_asi()
        self.asi_chan.append(self.asi)
        squeeze = self._compute_squeeze(h, l, c)
        psar_signal = self._compute_psar(h, l)

        # 13. Batch 7 (volume-derived — uses existing atr14 and candle_prev_c)
        b7 = self._compute_batch7_volume(c, h, l, volume, self.atr_ratio_atr or (h - l))

        self.bar_count += 1

        out = {
            "mc_d_a": mc_d,
            "mc_dd_a": mc_dd,
            "er_norm": er_norm,
            "macd_hist": macd_hist,
            "tec5": tec5,
            "roc_10": roc_10,
            "range_pos_30": range_pos_30,
            "rsi_14": rsi_14,
            "bb_width": bb_width,
            "aroon_osc": aroon_osc,
            "ema8_ratio": ema8_ratio,
            "ema21_ratio": ema21_ratio,
            "cci": cci,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "atr_ratio": atr_ratio,
            "donchian_pos": donchian_pos,
            "asi": self.asi,
            "sma5_asi": sma5_asi,
            "bar_count": self.bar_count,
        }
        out.update(b3)
        out.update(s)
        out.update({
            "williams_r": williams_r,
            "bb_pos": bb_pos,
            "dTEC": dTEC,
            "trending": trending,
            "high_vol": high_vol,
            "vol_regime": vol_regime,
            "tsf_slope": tsf_slope,
            "supertrend": supertrend,
            "adx": adx,
            "dmi_diff": dmi_diff,
            "hh_price": hh_price,
            "hl_price": hl_price,
            "h1_slope": h1_slope,
            "hh_asi": hh_asi,
            "hl_asi": hl_asi,
            "squeeze": squeeze,
            "psar_signal": psar_signal,
        })
        out.update(b7)
        return out

    def initialize_from_history(self, df) -> None:
        """Prime state by walking historical DataFrame chronologically.

        Expects columns: timestamp, open, high, low, close (order matters).
        """
        for _, row in df.iterrows():
            self.process_new_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                row.get("timestamp"),
                volume=float(row.get("volume", 0)))

    def walk_history(self, df):
        """Walk historical DataFrame and return ALL per-bar features as a DataFrame.

        Use this to pre-compute training parquets that exactly match live output.
        """
        import pandas as pd
        records = []
        for _, row in df.iterrows():
            feats = self.process_new_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                row.get("timestamp"),
                volume=float(row.get("volume", 0)))
            feats["timestamp"] = row.get("timestamp")
            feats["open"] = float(row["open"])
            feats["high"] = float(row["high"])
            feats["low"] = float(row["low"])
            feats["close"] = float(row["close"])
            records.append(feats)
        return pd.DataFrame(records)

    # ── State serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the complete internal state to a JSON-compatible dict.

        This captures every scalar, deque, and sub-state object needed to resume
        processing on the next bar without replaying any history. The dict can be
        stored via lib/feature_state_db.save_state() and restored with from_dict().

        Layout:
          scalars   — floats / ints / bools / None
          deques    — {'__deque__': True, 'maxlen': N, 'data': [...]}
          tf_states — list of dicts (one per TF), matched by position

        Round-trip guarantee:
          b = FXFeatureBuilder.from_dict(a.to_dict())
          b.process_new_bar(...) == a.process_new_bar(...)  # identical output
        """
        def _s(v: Any) -> Any:
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            if isinstance(v, deque):
                return {'__deque__': True, 'maxlen': v.maxlen, 'data': list(v)}
            raise TypeError(f"_s: unsupported type {type(v)}: {v!r}")

        def _tf(t: TFState) -> dict:
            return {
                'bp': t.bp, 'weight': t.weight, 'counter': t.counter,
                'ema_fast': t.ema_fast, 'ema_slow': t.ema_slow,
                'd_values': list(t.d_values),
                'n_samples': t.n_samples,
                'last_mc_d': t.last_mc_d, 'last_mc_dd': t.last_mc_dd,
            }

        return {
            'pair': self.pair,
            'smoother': self.smoother,
            # OHLC buffers
            'opens':      _s(self.opens),
            'highs':      _s(self.highs),
            'lows':       _s(self.lows),
            'closes':     _s(self.closes),
            'timestamps': {'__deque__': True, 'maxlen': self.timestamps.maxlen,
                           'data': [str(t) for t in self.timestamps]},
            # ASI state
            'asi': self.asi, 'atr': self.atr,
            'prev_close': self.prev_close, 'prev_open': self.prev_open,
            'tr_samples': self.tr_samples,
            # Smoother
            'asi_window': _s(self.asi_window),
            'smoother_x': self.smoother_x, 'smoother_p': self.smoother_p,
            # Multi-TF
            'tf_states': [_tf(t) for t in self.tf_states],
            # MACD
            'macd_ema_fast': self.macd_ema_fast, 'macd_ema_slow': self.macd_ema_slow,
            'macd_signal': self.macd_signal, 'macd_atr': self.macd_atr,
            # TEC5
            'tec_closes': _s(self.tec_closes), 'tec_abs_changes': _s(self.tec_abs_changes),
            # Batch 1
            'roc_closes': _s(self.roc_closes),
            'rp_highs': _s(self.rp_highs), 'rp_lows': _s(self.rp_lows),
            'rsi_prev_close': self.rsi_prev_close,
            'rsi_avg_gain': self.rsi_avg_gain, 'rsi_avg_loss': self.rsi_avg_loss,
            'rsi_samples': self.rsi_samples,
            'bb_closes': _s(self.bb_closes),
            'aroon_highs': _s(self.aroon_highs), 'aroon_lows': _s(self.aroon_lows),
            # Batch 2
            'ema8': self.ema8, 'ema21': self.ema21,
            'cci_tp': _s(self.cci_tp),
            'stoch_highs': _s(self.stoch_highs), 'stoch_lows': _s(self.stoch_lows),
            'stoch_k_hist': _s(self.stoch_k_hist),
            'atr_ratio_atr': self.atr_ratio_atr, 'atr_ratio_samples': self.atr_ratio_samples,
            'donchian_highs': _s(self.donchian_highs), 'donchian_lows': _s(self.donchian_lows),
            # Batch 3
            'candle_prev_o': self.candle_prev_o, 'candle_prev_h': self.candle_prev_h,
            'candle_prev_l': self.candle_prev_l, 'candle_prev_c': self.candle_prev_c,
            'candle_prev2_c': self.candle_prev2_c,
            'range_exp_ranges': _s(self.range_exp_ranges),
            # Batch 4
            'williams_highs': _s(self.williams_highs), 'williams_lows': _s(self.williams_lows),
            'bb_pos_closes': _s(self.bb_pos_closes),
            'tec_long_closes': _s(self.tec_long_closes),
            'tec_long_changes': _s(self.tec_long_changes),
            'tec_short_closes': _s(self.tec_short_closes),
            'tec_short_changes': _s(self.tec_short_changes),
            'regime_ema24': self.regime_ema24,
            'regime_ema_hist': _s(self.regime_ema_hist),
            'regime_slope_abs_hist': _s(self.regime_slope_abs_hist),
            'regime_atr_hist': _s(self.regime_atr_hist),
            # Batch 5
            'tsf_closes': _s(self.tsf_closes),
            'st_atr': self.st_atr, 'st_samples': self.st_samples,
            'st_trend': self.st_trend, 'st_upper': self.st_upper, 'st_lower': self.st_lower,
            'adx_pdm': self.adx_pdm, 'adx_ndm': self.adx_ndm, 'adx_tr': self.adx_tr,
            'adx_adx': self.adx_adx, 'adx_samples': self.adx_samples,
            'adx_prev_h': self.adx_prev_h, 'adx_prev_l': self.adx_prev_l,
            'chan_highs': _s(self.chan_highs), 'chan_lows': _s(self.chan_lows),
            'h1_bar_count': self.h1_bar_count,
            'h1_cur_o': self.h1_cur_o, 'h1_cur_h': self.h1_cur_h,
            'h1_cur_l': self.h1_cur_l, 'h1_cur_c': self.h1_cur_c,
            'h1_closes': _s(self.h1_closes), 'h1_slope_last': self.h1_slope_last,
            # Batch 6
            'asi_chan': _s(self.asi_chan),
            'keltner_ema': self.keltner_ema, 'keltner_atr': self.keltner_atr,
            'keltner_samples': self.keltner_samples,
            'psar_af': self.psar_af, 'psar_trend': self.psar_trend,
            'psar_ep': self.psar_ep, 'psar_sar': self.psar_sar,
            # Batch 7
            'vol_prev_close': self.vol_prev_close, 'cvd_cum': self.cvd_cum,
            'vol_ratio_hist': _s(self.vol_ratio_hist),
            'cvd_div_closes': _s(self.cvd_div_closes), 'cvd_div_cvd': _s(self.cvd_div_cvd),
            'vp_closes': _s(self.vp_closes), 'vp_volumes': _s(self.vp_volumes),
            # Counters
            'bar_count': self.bar_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FXFeatureBuilder':
        """Restore from a dict produced by to_dict(). No bar replay required.

        The returned instance is in an identical state to the original at the moment
        to_dict() was called. The next process_new_bar() call will produce the same
        output as if no serialization had occurred.
        """
        def _d(v) -> deque:
            if isinstance(v, dict) and v.get('__deque__'):
                return deque(v['data'], maxlen=v['maxlen'])
            return deque()

        def _tf(t: dict) -> TFState:
            ts = TFState(bp=t['bp'], weight=t['weight'])
            ts.counter   = t['counter']
            ts.ema_fast  = t['ema_fast']
            ts.ema_slow  = t['ema_slow']
            ts.d_values  = deque(t['d_values'], maxlen=8)
            ts.n_samples = t['n_samples']
            ts.last_mc_d  = t['last_mc_d']
            ts.last_mc_dd = t['last_mc_dd']
            return ts

        obj = cls.__new__(cls)
        obj.pair     = d['pair']
        obj.smoother = d['smoother']

        obj.opens      = _d(d['opens'])
        obj.highs      = _d(d['highs'])
        obj.lows       = _d(d['lows'])
        obj.closes     = _d(d['closes'])
        obj.timestamps = _d(d['timestamps'])

        obj.asi       = d['asi'];        obj.atr       = d['atr']
        obj.prev_close = d['prev_close']; obj.prev_open = d['prev_open']
        obj.tr_samples = d['tr_samples']

        obj.asi_window  = _d(d['asi_window'])
        obj.smoother_x  = d['smoother_x']
        obj.smoother_p  = d['smoother_p']

        obj.tf_states = [_tf(t) for t in d['tf_states']]

        obj.macd_ema_fast = d['macd_ema_fast']; obj.macd_ema_slow = d['macd_ema_slow']
        obj.macd_signal   = d['macd_signal'];   obj.macd_atr      = d['macd_atr']

        obj.tec_closes      = _d(d['tec_closes'])
        obj.tec_abs_changes = _d(d['tec_abs_changes'])

        obj.roc_closes  = _d(d['roc_closes'])
        obj.rp_highs    = _d(d['rp_highs']);    obj.rp_lows      = _d(d['rp_lows'])
        obj.rsi_prev_close = d['rsi_prev_close']
        obj.rsi_avg_gain   = d['rsi_avg_gain']; obj.rsi_avg_loss = d['rsi_avg_loss']
        obj.rsi_samples    = d['rsi_samples']
        obj.bb_closes   = _d(d['bb_closes'])
        obj.aroon_highs = _d(d['aroon_highs']); obj.aroon_lows  = _d(d['aroon_lows'])

        obj.ema8  = d['ema8'];           obj.ema21 = d['ema21']
        obj.cci_tp        = _d(d['cci_tp'])
        obj.stoch_highs   = _d(d['stoch_highs']); obj.stoch_lows = _d(d['stoch_lows'])
        obj.stoch_k_hist  = _d(d['stoch_k_hist'])
        obj.atr_ratio_atr     = d['atr_ratio_atr']
        obj.atr_ratio_samples = d['atr_ratio_samples']
        obj.donchian_highs = _d(d['donchian_highs']); obj.donchian_lows = _d(d['donchian_lows'])

        obj.candle_prev_o  = d['candle_prev_o'];  obj.candle_prev_h  = d['candle_prev_h']
        obj.candle_prev_l  = d['candle_prev_l'];  obj.candle_prev_c  = d['candle_prev_c']
        obj.candle_prev2_c = d['candle_prev2_c']
        obj.range_exp_ranges = _d(d['range_exp_ranges'])

        obj.williams_highs = _d(d['williams_highs']); obj.williams_lows = _d(d['williams_lows'])
        obj.bb_pos_closes  = _d(d['bb_pos_closes'])
        obj.tec_long_closes   = _d(d['tec_long_closes'])
        obj.tec_long_changes  = _d(d['tec_long_changes'])
        obj.tec_short_closes  = _d(d['tec_short_closes'])
        obj.tec_short_changes = _d(d['tec_short_changes'])
        obj.regime_ema24          = d['regime_ema24']
        obj.regime_ema_hist       = _d(d['regime_ema_hist'])
        obj.regime_slope_abs_hist = _d(d['regime_slope_abs_hist'])
        obj.regime_atr_hist       = _d(d['regime_atr_hist'])

        obj.tsf_closes  = _d(d['tsf_closes'])
        obj.st_atr      = d['st_atr'];    obj.st_samples = d['st_samples']
        obj.st_trend    = d['st_trend'];  obj.st_upper   = d['st_upper'];  obj.st_lower = d['st_lower']
        obj.adx_pdm     = d['adx_pdm'];  obj.adx_ndm    = d['adx_ndm']
        obj.adx_tr      = d['adx_tr'];   obj.adx_adx    = d['adx_adx']
        obj.adx_samples = d['adx_samples']
        obj.adx_prev_h  = d.get('adx_prev_h'); obj.adx_prev_l = d.get('adx_prev_l')
        obj.chan_highs  = _d(d['chan_highs']); obj.chan_lows = _d(d['chan_lows'])
        obj.h1_bar_count = d['h1_bar_count']
        obj.h1_cur_o = d['h1_cur_o']; obj.h1_cur_h = d['h1_cur_h']
        obj.h1_cur_l = d['h1_cur_l']; obj.h1_cur_c = d['h1_cur_c']
        obj.h1_closes    = _d(d['h1_closes'])
        obj.h1_slope_last = d['h1_slope_last']

        obj.asi_chan        = _d(d['asi_chan'])
        obj.keltner_ema     = d['keltner_ema'];    obj.keltner_atr     = d['keltner_atr']
        obj.keltner_samples = d['keltner_samples']
        obj.psar_af    = d['psar_af'];  obj.psar_trend = d['psar_trend']
        obj.psar_ep    = d['psar_ep'];  obj.psar_sar   = d['psar_sar']

        obj.vol_prev_close = d['vol_prev_close']; obj.cvd_cum = d['cvd_cum']
        obj.vol_ratio_hist = _d(d['vol_ratio_hist'])
        obj.cvd_div_closes = _d(d['cvd_div_closes']); obj.cvd_div_cvd = _d(d['cvd_div_cvd'])
        obj.vp_closes      = _d(d['vp_closes']);       obj.vp_volumes  = _d(d['vp_volumes'])

        obj.bar_count = d['bar_count']
        return obj
