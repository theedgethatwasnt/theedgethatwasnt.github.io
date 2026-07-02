#!/usr/bin/env python3
"""
Micro Feature Builder for Live Trading.

Builds the exact 15-feature vector used by Micro MuZero:
- 5 technical indicators
- 4 cyclical time features
- 6 position features

Uses OANDA incremental puller for real-time data.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timezone
import math
import sys
import os

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.oanda_m1_incremental_puller import OandaM1Puller

logger = logging.getLogger(__name__)


class MicroFeatureBuilder:
    """
    Build micro features incrementally for live trading.

    Generates 15-feature vectors matching training:
    - 5 technical indicators (current values only)
    - 4 cyclical time features
    - 6 position features
    """

    def __init__(
        self,
        instrument: str = "GBP_JPY",
        lag_window: int = 32,
        warmup_bars: int = 100
    ):
        """
        Initialize micro feature builder.

        Args:
            instrument: Trading instrument
            lag_window: Number of lag values to maintain (32)
            warmup_bars: Extra bars for indicator calculation
        """
        self.instrument = instrument
        self.lag_window = lag_window
        self.warmup_bars = warmup_bars

        # Need at least 92 bars for 60-bar indicators + 32 lags
        self.min_history = lag_window + 60

        # Initialize OANDA puller
        try:
            self.data_puller = OandaM1Puller()
        except Exception as e:
            logger.error(f"Failed to initialize OANDA puller: {e}")
            self.data_puller = None

        # Price history
        self.price_history = deque(maxlen=self.min_history + 10)
        self.timestamp_history = deque(maxlen=self.min_history + 10)

        # Feature history (32 lags for each feature)
        self.feature_history = {
            'position_in_range_60': deque(maxlen=lag_window),
            'min_max_scaled_momentum_60': deque(maxlen=lag_window),
            'min_max_scaled_rolling_range': deque(maxlen=lag_window),
            'min_max_scaled_momentum_5': deque(maxlen=lag_window),
            'price_change_pips': deque(maxlen=lag_window),
            'dow_cos_final': deque(maxlen=lag_window),
            'dow_sin_final': deque(maxlen=lag_window),
            'hour_cos_final': deque(maxlen=lag_window),
            'hour_sin_final': deque(maxlen=lag_window)
        }

        # Position state (managed externally)
        self.position_state = {
            'position_side': 0.0,
            'position_pips': 0.0,
            'bars_since_entry': 0.0,
            'pips_from_peak': 0.0,
            'max_drawdown_pips': 0.0,
            'accumulated_dd': 0.0
        }

        self.bar_index = 0
        self.last_close = None

        logger.info(f"Initialized MicroFeatureBuilder for {instrument}")
        logger.info(f"  Min history required: {self.min_history} bars")

    def initialize_from_history(self) -> bool:
        """
        Initialize with historical data from OANDA.

        Returns:
            True if successful, False otherwise
        """
        if self.data_puller is None:
            logger.error("OANDA puller not available")
            return False

        try:
            logger.info("Fetching historical data from OANDA...")

            # Get 256 bars of history
            closes = self.data_puller.get_last_256_m1_closes(self.instrument)

            if not closes or len(closes) < self.min_history:
                logger.error(f"Insufficient data: got {len(closes) if closes else 0}, need {self.min_history}")
                return False

            # Use the required number of bars
            closes = closes[-self.min_history:]
            self.price_history.extend(closes)

            # Generate timestamps
            current_time = datetime.now(timezone.utc)
            for i in range(len(closes)):
                ts = current_time.timestamp() - (len(closes) - i) * 60
                self.timestamp_history.append(datetime.fromtimestamp(ts, timezone.utc))

            # Calculate features for historical data
            logger.info("Calculating features for historical data...")

            for i in range(60, len(closes)):  # Need 60 bars for indicators
                # Get price window
                price_window = list(self.price_history)[max(0, i - 60):i + 1]

                if len(price_window) >= 60:
                    # Calculate technical features
                    features = self._calculate_technical_features(price_window)

                    # Calculate cyclical features
                    timestamp = self.timestamp_history[i]
                    cyclical = self._calculate_cyclical_features(timestamp)

                    # Store in history
                    for key, value in features.items():
                        self.feature_history[key].append(value)

                    for key, value in cyclical.items():
                        self.feature_history[key].append(value)

                    self.last_close = closes[i]
                    self.bar_index = i

            logger.info(f"Initialized with {self.bar_index} bars")
            logger.info(f"  Technical features: {len(self.feature_history['position_in_range_60'])} values")
            logger.info(f"  Cyclical features: {len(self.feature_history['dow_cos_final'])} values")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize from history: {e}")
            return False

    def _calculate_technical_features(self, prices: list) -> Dict[str, float]:
        """Calculate technical indicators."""
        features = {}

        if len(prices) < 60:
            return features

        # Convert to numpy for calculations
        prices = np.array(prices)
        current_close = prices[-1]

        # 1. Position in range (60-bar)
        high_60 = np.max(prices[-60:])
        low_60 = np.min(prices[-60:])
        range_60 = high_60 - low_60
        if range_60 > 0:
            features['position_in_range_60'] = np.tanh((current_close - low_60) / range_60 - 0.5)
        else:
            features['position_in_range_60'] = 0.0

        # 2. Momentum (60-bar)
        momentum_60 = current_close - prices[-60]
        features['min_max_scaled_momentum_60'] = np.tanh(momentum_60 / 10.0)

        # 3. Rolling range (60-bar)
        features['min_max_scaled_rolling_range'] = np.tanh(range_60 / 10.0)

        # 4. Momentum (5-bar)
        if len(prices) >= 5:
            momentum_5 = current_close - prices[-5]
            features['min_max_scaled_momentum_5'] = np.tanh(momentum_5 / 10.0)
        else:
            features['min_max_scaled_momentum_5'] = 0.0

        # 5. Price change in pips
        if self.last_close is not None:
            # For GBPJPY, 1 pip = 0.01
            price_change = (current_close - self.last_close) * 100
            features['price_change_pips'] = np.tanh(price_change / 10.0)
        else:
            features['price_change_pips'] = 0.0

        return features

    def _calculate_cyclical_features(self, timestamp: datetime) -> Dict[str, float]:
        """Calculate cyclical time features."""
        features = {}

        # Day of week (0=Monday, 6=Sunday)
        dow = timestamp.weekday()
        dow_angle = 2 * math.pi * dow / 7
        features['dow_cos_final'] = np.tanh(math.cos(dow_angle))
        features['dow_sin_final'] = np.tanh(math.sin(dow_angle))

        # Hour of day (0-23)
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = 2 * math.pi * hour / 24
        features['hour_cos_final'] = np.tanh(math.cos(hour_angle))
        features['hour_sin_final'] = np.tanh(math.sin(hour_angle))

        return features

    def update_position_state(
        self,
        position_side: int,
        position_pips: float,
        bars_since_entry: int,
        pips_from_peak: float,
        max_drawdown_pips: float,
        accumulated_dd: float
    ):
        """Update position state features."""
        self.position_state['position_side'] = np.tanh(float(position_side))
        self.position_state['position_pips'] = np.tanh(position_pips / 100.0)
        self.position_state['bars_since_entry'] = np.tanh(bars_since_entry / 100.0)
        self.position_state['pips_from_peak'] = np.tanh(pips_from_peak / 100.0)
        self.position_state['max_drawdown_pips'] = np.tanh(max_drawdown_pips / 100.0)
        self.position_state['accumulated_dd'] = np.tanh(accumulated_dd / 100.0)

    def add_new_bar(self, close_price: float, timestamp: Optional[datetime] = None) -> bool:
        """
        Add a new bar and calculate features.

        Args:
            close_price: New bar close price
            timestamp: Bar timestamp (optional)

        Returns:
            True if features were calculated, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Add to history
        self.price_history.append(close_price)
        self.timestamp_history.append(timestamp)
        self.bar_index += 1

        # Need enough history for indicators
        if len(self.price_history) < 60:
            logger.warning(f"Insufficient history: {len(self.price_history)} bars")
            return False

        # Calculate features
        price_window = list(self.price_history)[-60:]
        tech_features = self._calculate_technical_features(price_window)
        cycl_features = self._calculate_cyclical_features(timestamp)

        # Update history
        for key, value in tech_features.items():
            self.feature_history[key].append(value)

        for key, value in cycl_features.items():
            self.feature_history[key].append(value)

        self.last_close = close_price

        return True

    def get_feature_vector(self) -> Optional[np.ndarray]:
        """
        Get current feature vector for model input.

        Returns:
            (32, 15) numpy array or None if insufficient data
        """
        # Check if we have enough history
        if len(self.feature_history['position_in_range_60']) < self.lag_window:
            logger.warning(f"Insufficient feature history: {len(self.feature_history['position_in_range_60'])}/{self.lag_window}")
            return None

        # Build feature matrix (32 lags × 15 features)
        feature_matrix = np.zeros((self.lag_window, 15), dtype=np.float32)

        # Technical features (5)
        tech_features = [
            'position_in_range_60',
            'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range',
            'min_max_scaled_momentum_5',
            'price_change_pips'
        ]

        for i, feat in enumerate(tech_features):
            history = list(self.feature_history[feat])[-self.lag_window:]
            feature_matrix[:, i] = history

        # Cyclical features (4)
        cycl_features = [
            'dow_cos_final',
            'dow_sin_final',
            'hour_cos_final',
            'hour_sin_final'
        ]

        for i, feat in enumerate(cycl_features):
            history = list(self.feature_history[feat])[-self.lag_window:]
            feature_matrix[:, 5 + i] = history

        # Position features (6) - same for all lags
        position_features = [
            'position_side',
            'position_pips',
            'bars_since_entry',
            'pips_from_peak',
            'max_drawdown_pips',
            'accumulated_dd'
        ]

        for i, feat in enumerate(position_features):
            feature_matrix[:, 9 + i] = self.position_state[feat]

        return feature_matrix

    def get_current_price(self) -> Optional[float]:
        """Get the current/latest price."""
        if self.data_puller is None:
            return None

        try:
            latest = self.data_puller.get_latest_m1_close(self.instrument)
            if latest and len(latest) > 0:
                return latest[0]
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")

        return None