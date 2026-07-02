"""
Indicator Registry — auto-instantiates all available indicator classes per pair.

Used by the curator to compute ALL indicators on every S5 bar and publish
the full snapshot. Strategies and NEAT training can then pick any combination.

Usage:
    registry = IndicatorRegistry()
    for pair in ALL_PAIRS:
        registry.init_pair(pair)

    # On each S5 bar:
    snapshot = registry.update(pair, bar)
    # snapshot = {"rsi_14": 55.2, "ema_20": 1.1050, "bollinger_upper": 1.1080, ...}
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import all indicator classes
from lib.indicators import (
    ATR, ADX, VolRegime, ChandelierExit, KaufmanER, ZScore,
    LinearRegressionSlope, Stochastic, MACD, CCI, Aroon, SuperTrend,
    ParabolicSAR, DonchianChannel, KeltnerChannel, Ichimoku,
    WilliamsR, MFI, OBV, ROC, MomentumIndicator, DTO, Vortex,
    Squeeze, BBWidth, ATRRatio, Retracement, BreakoutChannel,
    RangePosition, BBPosition, VolExpansion, RSIExtreme, TrendQuality,
    MomStrength, PSARDelta, DMISignal, DeltaPrice,
    SchaffTrendCycle, Repulse, StochasticRSI, SMI,
    ChaikinMoneyFlow, ChaikinOscillator, ChaikinVolatility, CoppockCurve,
    DPO, DEMA, TEMA, MACDZeroLag, DynamicZoneRSI,
    DynamicZoneStochastic, Envelopes,
)

# Try importing the new batch (may not exist yet if agent is still adding them)
try:
    from lib.indicators import (
        RSI, SMA, EMA, WMA, StdDev, Bollinger,
        PivotPoints, WoodiePivots, CamarillaPivots,
        AccumulationDistribution, AdaptiveMovingAverage,
        PriceOscillator, VolumeOscillator, VROC, VWAP, VWMA,
        HighsAndLows, MACDDivergence, RSIDivergence, CCIDivergence,
        HourOfDay, DayOfWeek, SessionStrength,
    )
    _HAS_BATCH2 = True
except ImportError:
    _HAS_BATCH2 = False
    logger.warning("Batch 2 indicators not yet available")


# Registry: (name, class, kwargs, extract_fn)
# extract_fn: indicator -> dict of {field_name: value}
INDICATOR_SPECS = [
    # Basic oscillators
    ("rsi_14", RSI if _HAS_BATCH2 else None, {"period": 14}, lambda i: {"rsi_14": i.value}),
    ("stoch_14", Stochastic, {"k_period": 14}, lambda i: {"stoch_k": i.k, "stoch_d": i.d}),
    ("cci_20", CCI, {"period": 20}, lambda i: {"cci_20": i.value}),
    ("williams_r", WilliamsR, {"period": 14}, lambda i: {"williams_r": i.value}),
    ("mfi_14", MFI, {"period": 14}, lambda i: {"mfi_14": i.value}),
    ("stoch_rsi", StochasticRSI, {}, lambda i: {"stoch_rsi": i.value}),
    ("smi", SMI, {}, lambda i: {"smi": i.value, "smi_signal": i.signal}),
    ("dyn_rsi", DynamicZoneRSI, {}, lambda i: {"dyn_rsi": i.value, "dyn_rsi_upper": i.upper, "dyn_rsi_lower": i.lower}),
    ("dyn_stoch", DynamicZoneStochastic, {}, lambda i: {"dyn_stoch": i.value, "dyn_stoch_upper": i.upper, "dyn_stoch_lower": i.lower}),
    ("rsi_extreme", RSIExtreme, {}, lambda i: {"rsi_extreme": i.value}),

    # Trend indicators
    ("atr_14", ATR, {"period": 14}, lambda i: {"atr_14": i.value}),
    ("adx_14", ADX, {"period": 14}, lambda i: {"adx": i.value, "adx_regime": i.regime}),
    ("supertrend", SuperTrend, {}, lambda i: {"supertrend": i.value, "supertrend_dir": i.direction}),
    ("psar", ParabolicSAR, {}, lambda i: {"psar": i.value}),
    ("aroon", Aroon, {}, lambda i: {"aroon_up": i.up, "aroon_down": i.down}),
    ("vortex", Vortex, {}, lambda i: {"vortex_plus": i.plus, "vortex_minus": i.minus}),
    ("dmi_signal", DMISignal, {}, lambda i: {"dmi_signal": i.value}),
    ("trend_quality", TrendQuality, {}, lambda i: {"trend_quality": i.value}),
    ("schaff_tc", SchaffTrendCycle, {}, lambda i: {"schaff_tc": i.value}),
    ("vol_regime", VolRegime, {}, lambda i: {"vol_regime": i.regime}),

    # Moving averages
    ("ema_20", EMA if _HAS_BATCH2 else None, {"period": 20}, lambda i: {"ema_20": i.value}),
    ("sma_50", SMA if _HAS_BATCH2 else None, {"period": 50}, lambda i: {"sma_50": i.value}),
    ("dema_20", DEMA, {"period": 20}, lambda i: {"dema_20": i.value}),
    ("tema_20", TEMA, {"period": 20}, lambda i: {"tema_20": i.value}),
    ("kama", AdaptiveMovingAverage if _HAS_BATCH2 else None, {}, lambda i: {"kama": i.value}),

    # Channels / bands
    ("donchian", DonchianChannel, {}, lambda i: {"donchian_upper": i.upper, "donchian_lower": i.lower, "donchian_mid": i.mid}),
    ("keltner", KeltnerChannel, {}, lambda i: {"keltner_upper": i.upper, "keltner_lower": i.lower, "keltner_mid": i.mid}),
    ("bollinger", Bollinger if _HAS_BATCH2 else None, {}, lambda i: {"bb_upper": i.upper, "bb_lower": i.lower, "bb_mid": i.mid}),
    ("envelopes", Envelopes, {}, lambda i: {"env_upper": i.upper, "env_lower": i.lower, "env_mid": i.mid}),
    ("chandelier", ChandelierExit, {}, lambda i: {"chandelier_long": i.long_stop, "chandelier_short": i.short_stop}),
    ("ichimoku", Ichimoku, {}, lambda i: {"ichi_tenkan": i.tenkan, "ichi_kijun": i.kijun, "ichi_senkou_a": i.senkou_a, "ichi_senkou_b": i.senkou_b}),

    # MACD family
    ("macd", MACD, {}, lambda i: {"macd": i.macd, "macd_signal": i.signal, "macd_hist": i.histogram}),
    ("macd_zl", MACDZeroLag, {}, lambda i: {"macd_zl": i.value, "macd_zl_signal": i.signal, "macd_zl_hist": i.histogram}),

    # Momentum / rate of change
    ("roc_12", ROC, {"period": 12}, lambda i: {"roc_12": i.value}),
    ("momentum_10", MomentumIndicator, {"period": 10}, lambda i: {"momentum_10": i.value}),
    ("coppock", CoppockCurve, {}, lambda i: {"coppock": i.value}),
    ("dpo", DPO, {}, lambda i: {"dpo": i.value}),
    ("mom_strength", MomStrength, {}, lambda i: {"mom_strength": i.value}),
    ("repulse", Repulse, {}, lambda i: {"repulse": i.value}),
    ("delta_price", DeltaPrice, {}, lambda i: {"delta_price": i.value}),

    # Volatility
    ("bb_width", BBWidth, {}, lambda i: {"bb_width": i.value}),
    ("bb_position", BBPosition, {}, lambda i: {"bb_position": i.value}),
    ("vol_expansion", VolExpansion, {}, lambda i: {"vol_expansion": i.value}),
    ("atr_ratio", ATRRatio, {}, lambda i: {"atr_ratio": i.value}),
    ("squeeze", Squeeze, {}, lambda i: {"squeeze_on": i.squeeze_on, "squeeze_mom": i.momentum}),
    ("kaufman_er", KaufmanER, {}, lambda i: {"kaufman_er": i.value}),
    ("chaikin_vol", ChaikinVolatility, {}, lambda i: {"chaikin_vol": i.value}),
    ("std_dev", StdDev if _HAS_BATCH2 else None, {}, lambda i: {"std_dev": i.value}),

    # Volume-based
    ("obv", OBV, {}, lambda i: {"obv": i.value}),
    ("cmf", ChaikinMoneyFlow, {}, lambda i: {"cmf": i.value}),
    ("chaikin_osc", ChaikinOscillator, {}, lambda i: {"chaikin_osc": i.value}),

    # Structure
    ("range_pos", RangePosition, {}, lambda i: {"range_pos": i.value}),
    ("retracement", Retracement, {}, lambda i: {"retracement": i.value}),
    ("breakout_ch", BreakoutChannel, {}, lambda i: {"breakout_ch": i.value}),
    ("psar_delta", PSARDelta, {}, lambda i: {"psar_delta": i.value}),
    ("lr_slope", LinearRegressionSlope, {}, lambda i: {"lr_slope": i.value}),
    ("z_score", ZScore, {}, lambda i: {"z_score": i.value}),
    ("dto", DTO, {}, lambda i: {"dto": i.value}),
    ("highs_lows", HighsAndLows if _HAS_BATCH2 else None, {}, lambda i: {"highest": i.highest, "lowest": i.lowest, "hl_range": i.range}),

    # Time-based
    ("hour_of_day", HourOfDay if _HAS_BATCH2 else None, {}, lambda i: {"sin_hour": i.sin_hour, "cos_hour": i.cos_hour, "hour": float(i.hour), "session": float(i.session)}),
    ("day_of_week", DayOfWeek if _HAS_BATCH2 else None, {}, lambda i: {"sin_dow": i.sin_dow, "cos_dow": i.cos_dow, "dow": float(i.dow)}),
    ("session_strength", SessionStrength if _HAS_BATCH2 else None, {}, lambda i: {"session_strength": i.value, "asian_range": i.asian_range, "london_break": i.london_break}),
]


class IndicatorRegistry:
    """Manages all indicator instances per pair. Update once, read any combo."""

    def __init__(self):
        self._pairs: Dict[str, Dict[str, Any]] = {}  # {pair: {name: indicator_instance}}
        self._extractors: Dict[str, callable] = {}  # {name: extract_fn}
        self._active_specs = [(name, cls, kwargs, extract)
                              for name, cls, kwargs, extract in INDICATOR_SPECS
                              if cls is not None]
        logger.info(f"IndicatorRegistry: {len(self._active_specs)} indicators available")

    def init_pair(self, pair: str):
        """Create all indicator instances for a pair."""
        self._pairs[pair] = {}
        for name, cls, kwargs, extract in self._active_specs:
            try:
                self._pairs[pair][name] = cls(**kwargs)
                self._extractors[name] = extract
            except Exception as e:
                logger.warning(f"Failed to init {name} for {pair}: {e}")

    def update(self, pair: str, bar: dict) -> dict:
        """Update all indicators for a pair with a new bar. Returns full snapshot."""
        if pair not in self._pairs:
            return {}

        snapshot = {}
        for name, indicator in self._pairs[pair].items():
            try:
                indicator.update(bar)
                extract = self._extractors[name]
                snapshot.update(extract(indicator))
            except Exception:
                pass  # Individual indicator failure must not block others

        return snapshot

    def get_value(self, pair: str, field: str) -> float:
        """Get a single indicator field value."""
        if pair not in self._pairs:
            return 0.0
        for name, indicator in self._pairs[pair].items():
            extract = self._extractors.get(name)
            if extract:
                values = extract(indicator)
                if field in values:
                    return values[field]
        return 0.0

    def list_fields(self) -> list:
        """List all available indicator field names."""
        fields = []
        for name, cls, kwargs, extract in self._active_specs:
            # Create a dummy to get field names
            try:
                dummy = cls(**kwargs)
                bar = {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 100}
                dummy.update(bar)
                fields.extend(extract(dummy).keys())
            except Exception:
                pass
        return fields
