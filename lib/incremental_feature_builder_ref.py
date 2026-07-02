#!/usr/bin/env python3
"""
Incremental Feature Builder for Live Trading.

Builds feature vectors incrementally using live OANDA M1 data.
Generates the exact same 297-column feature vector as used in training.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timezone
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from oanda_m1_incremental_puller import OandaM1Puller
except ImportError:
    # Fallback for when OANDA module not available
    class OandaM1Puller:
        def get_last_256_m1_closes(self):
            return [199.0 + i * 0.01 for i in range(256)]
        def get_latest_m1_close(self):
            return 199.5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IncrementalFeatureBuilder:
    """
    Build features incrementally for live trading.

    Generates 297-column feature vectors matching training data:
    - 3 metadata columns (timestamp, bar_index, close)
    - 160 technical indicator columns (5 indicators × 32 lags)
    - 128 cyclical time columns (4 features × 32 lags)
    - 6 position features (current only, no lags)
    """

    def __init__(
        self,
        instrument: str = "GBP_JPY",
        lag_window: int = 32,
        warmup_bars: int = 100  # Extra bars for indicator calculation
    ):
        """
        Initialize incremental feature builder.

        Args:
            instrument: Trading instrument (default: GBP_JPY)
            lag_window: Number of lagged values per feature (32)
            warmup_bars: Extra historical bars needed for indicators
        """
        self.instrument = instrument
        self.lag_window = lag_window
        self.warmup_bars = warmup_bars

        # Total bars needed: lag_window + warmup for longest indicator (60-bar range)
        self.min_history = lag_window + 60  # At least 92 bars

        # Initialize OANDA data puller
        self.data_puller = OandaM1Puller()

        # Price history buffer
        self.price_history = deque(maxlen=self.min_history + 10)  # Extra buffer
        self.timestamp_history = deque(maxlen=self.min_history + 10)

        # Feature history buffers (keep last lag_window values)
        self.technical_history = {
            'position_in_range_60': deque(maxlen=lag_window),
            'min_max_scaled_momentum_60': deque(maxlen=lag_window),
            'min_max_scaled_rolling_range': deque(maxlen=lag_window),
            'min_max_scaled_momentum_5': deque(maxlen=lag_window),
            'price_change_pips': deque(maxlen=lag_window)
        }

        self.cyclical_history = {
            'dow_cos_final': deque(maxlen=lag_window),
            'dow_sin_final': deque(maxlen=lag_window),
            'hour_cos_final': deque(maxlen=lag_window),
            'hour_sin_final': deque(maxlen=lag_window)
        }

        # Current position state (managed externally)
        self.position_state = {
            'position_side': 0.0,
            'position_pips': 0.0,
            'bars_since_entry': 0.0,
            'pips_from_peak': 0.0,
            'max_drawdown_pips': 0.0,
            'accumulated_dd': 0.0
        }

        # Bar counter
        self.bar_index = 0

        # Technical indicator parameters
        self.L = 60  # Long-term window
        self.S = 5   # Short-term window

        logger.info(f"Initialized IncrementalFeatureBuilder for {instrument}")
        logger.info(f"  Lag window: {lag_window}")
        logger.info(f"  Min history required: {self.min_history} bars")

    def initialize_from_history(self) -> bool:
        """
        Initialize feature buffers with historical data.

        Returns:
            True if successfully initialized, False otherwise
        """
        logger.info("Fetching initial historical data...")

        # Get enough historical bars
        closes = self.data_puller.get_last_256_m1_closes()

        if closes is None or len(closes) < self.min_history:
            logger.error(f"Insufficient historical data: got {len(closes) if closes else 0} bars, need {self.min_history}")
            return False

        # Use only the required number of bars
        closes = closes[-self.min_history:]

        # Initialize price history
        self.price_history.extend(closes)

        # Generate timestamps (approximate)
        current_time = datetime.now(timezone.utc)
        for i in range(len(closes)):
            # Approximate timestamps going backwards
            ts = current_time.timestamp() - (len(closes) - i) * 60
            self.timestamp_history.append(datetime.fromtimestamp(ts, timezone.utc))

        # Calculate all features for historical data
        logger.info("Calculating features for historical data...")

        for i in range(self.warmup_bars, len(closes)):
            # Get price window for calculations
            price_window = list(self.price_history)[max(0, i - self.L):i + 1]

            # Calculate technical indicators
            tech_features = self._calculate_technical_indicators(price_window, i)

            # Calculate cyclical features
            timestamp = self.timestamp_history[i]
            cycl_features = self._calculate_cyclical_features(timestamp)

            # Store in history buffers
            for name, value in tech_features.items():
                self.technical_history[name].append(value)

            for name, value in cycl_features.items():
                self.cyclical_history[name].append(value)

        # Set initial bar index
        self.bar_index = len(closes)

        logger.info(f"Initialization complete. Bar index: {self.bar_index}")
        logger.info(f"  Technical history: {len(self.technical_history['position_in_range_60'])} values")
        logger.info(f"  Cyclical history: {len(self.cyclical_history['dow_cos_final'])} values")

        return True

    def _calculate_technical_indicators(
        self,
        price_window: List[float],
        current_idx: int
    ) -> Dict[str, float]:
        """
        Calculate technical indicators matching training data.

        Args:
            price_window: Recent price history
            current_idx: Current position in price history

        Returns:
            Dictionary of indicator values
        """
        features = {}

        # Ensure we have enough data
        if len(price_window) < self.L:
            # Return NaN for insufficient data
            return {
                'position_in_range_60': np.nan,
                'min_max_scaled_momentum_60': np.nan,
                'min_max_scaled_rolling_range': np.nan,
                'min_max_scaled_momentum_5': np.nan,
                'price_change_pips': np.nan
            }

        # Current and past prices
        current_close = price_window[-1]

        # 1. Position in 60-bar range [0, 1]
        last_60 = price_window[-self.L:]
        min_60 = min(last_60)
        max_60 = max(last_60)

        if max_60 - min_60 > 0:
            features['position_in_range_60'] = (current_close - min_60) / (max_60 - min_60)
        else:
            features['position_in_range_60'] = 0.5

        # 2. Min-max scaled 60-bar momentum
        if len(price_window) > self.L:
            momentum_60 = current_close - price_window[-self.L - 1]
            # Scale to approximately [-1, 1] range
            features['min_max_scaled_momentum_60'] = np.tanh(momentum_60 / 100)
        else:
            features['min_max_scaled_momentum_60'] = 0.0

        # 3. Min-max scaled rolling range (volatility)
        rolling_range = max_60 - min_60
        # Scale by typical range (assuming ~10 pips is typical)
        features['min_max_scaled_rolling_range'] = np.tanh(rolling_range / 10)

        # 4. Min-max scaled 5-bar momentum
        if len(price_window) > self.S:
            momentum_5 = current_close - price_window[-self.S - 1]
            features['min_max_scaled_momentum_5'] = np.tanh(momentum_5 / 10)
        else:
            features['min_max_scaled_momentum_5'] = 0.0

        # 5. Price change in pips (from previous bar)
        if len(price_window) > 1:
            price_change = (current_close - price_window[-2]) * 100  # Convert to pips
            features['price_change_pips'] = np.tanh(price_change / 10)
        else:
            features['price_change_pips'] = 0.0

        return features

    def _calculate_cyclical_features(self, timestamp: datetime) -> Dict[str, float]:
        """
        Calculate cyclical time features.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary of cyclical features
        """
        # Trading week: Sunday 17:00 ET to Friday 17:00 ET
        # Assuming timestamp is in UTC

        # Day of week encoding
        dow = timestamp.weekday()  # Monday=0, Sunday=6
        dow_radians = 2 * np.pi * dow / 7

        # Hour of day encoding (in trading context)
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_radians = 2 * np.pi * hour / 24

        return {
            'dow_cos_final': np.cos(dow_radians),
            'dow_sin_final': np.sin(dow_radians),
            'hour_cos_final': np.cos(hour_radians),
            'hour_sin_final': np.sin(hour_radians)
        }

    def update_position_state(
        self,
        position_side: float,
        position_pips: float,
        bars_since_entry: int,
        pips_from_peak: float,
        max_drawdown_pips: float,
        accumulated_dd: float
    ):
        """
        Update current position state from trading environment.

        Args:
            position_side: -1 (short), 0 (flat), 1 (long)
            position_pips: Current P&L in pips
            bars_since_entry: Time in position
            pips_from_peak: Distance from best P&L
            max_drawdown_pips: Worst drawdown
            accumulated_dd: Total drawdown area
        """
        self.position_state = {
            'position_side': float(position_side),
            'position_pips': np.tanh(position_pips / 100),
            'bars_since_entry': np.tanh(bars_since_entry / 100),
            'pips_from_peak': np.tanh(pips_from_peak / 100),
            'max_drawdown_pips': np.tanh(max_drawdown_pips / 100),
            'accumulated_dd': np.tanh(accumulated_dd / 100)
        }

    def process_new_bar(
        self,
        close: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[np.ndarray]:
        """
        Process new M1 bar and generate feature vector.

        Args:
            close: Close price of new bar
            timestamp: Bar timestamp (uses current time if None)

        Returns:
            297-element feature vector or None if insufficient history
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Update histories
        self.price_history.append(close)
        self.timestamp_history.append(timestamp)
        self.bar_index += 1

        # Check if we have enough history
        if len(self.price_history) < self.min_history:
            logger.warning(f"Insufficient history: {len(self.price_history)}/{self.min_history} bars")
            return None

        # Calculate new features
        price_window = list(self.price_history)
        tech_features = self._calculate_technical_indicators(price_window, len(price_window) - 1)
        cycl_features = self._calculate_cyclical_features(timestamp)

        # Update feature histories
        for name, value in tech_features.items():
            self.technical_history[name].append(value)

        for name, value in cycl_features.items():
            self.cyclical_history[name].append(value)

        # Build feature vector (297 columns)
        feature_vector = []

        # 1. Metadata (3 columns)
        feature_vector.append(timestamp.timestamp())  # timestamp
        feature_vector.append(float(self.bar_index))  # bar_index
        feature_vector.append(close)  # close

        # 2. Technical indicators with lags (5 × 32 = 160 columns)
        for feature_name in ['position_in_range_60', 'min_max_scaled_momentum_60',
                            'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
                            'price_change_pips']:
            history = list(self.technical_history[feature_name])

            # Add current and lagged values (0 = current, 31 = oldest)
            for lag in range(self.lag_window):
                if lag < len(history):
                    # Get value from end of history (newest first)
                    value = history[-(lag + 1)]
                else:
                    value = np.nan
                feature_vector.append(value)

        # 3. Cyclical features with lags (4 × 32 = 128 columns)
        for feature_name in ['dow_cos_final', 'dow_sin_final',
                            'hour_cos_final', 'hour_sin_final']:
            history = list(self.cyclical_history[feature_name])

            # Add current and lagged values
            for lag in range(self.lag_window):
                if lag < len(history):
                    value = history[-(lag + 1)]
                else:
                    value = np.nan
                feature_vector.append(value)

        # 4. Position features (6 columns, current only)
        for feature_name in ['position_side', 'position_pips', 'bars_since_entry',
                            'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd']:
            feature_vector.append(self.position_state[feature_name])

        # Convert to numpy array
        feature_array = np.array(feature_vector, dtype=np.float32)

        # Verify shape
        if len(feature_array) != 297:
            logger.error(f"Feature vector has wrong shape: {len(feature_array)} != 297")
            return None

        return feature_array

    def get_feature_vector_for_model(self) -> Optional[np.ndarray]:
        """
        Get current feature vector formatted for model input.

        Returns:
            (32, 15) array for micro model or None if insufficient data
        """
        # Check if we have enough history
        if len(self.technical_history['position_in_range_60']) < self.lag_window:
            logger.warning("Insufficient feature history for model input")
            return None

        # Build (32, 15) observation array
        observation = []

        for t in range(self.lag_window):
            row_features = []

            # Technical indicators (5)
            for feature_name in ['position_in_range_60', 'min_max_scaled_momentum_60',
                                'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
                                'price_change_pips']:
                history = list(self.technical_history[feature_name])
                # Get value at time t (0 = oldest, 31 = newest)
                idx = t
                if idx < len(history):
                    value = history[idx]
                else:
                    value = 0.0  # Or np.nan, depending on model training
                row_features.append(value)

            # Cyclical features (4)
            for feature_name in ['dow_cos_final', 'dow_sin_final',
                                'hour_cos_final', 'hour_sin_final']:
                history = list(self.cyclical_history[feature_name])
                idx = t
                if idx < len(history):
                    value = history[idx]
                else:
                    value = 0.0
                row_features.append(value)

            # Position features (6) - same for all timesteps
            for feature_name in ['position_side', 'position_pips', 'bars_since_entry',
                                'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd']:
                row_features.append(self.position_state[feature_name])

            observation.append(row_features)

        return np.array(observation, dtype=np.float32)

    def run_live_updates(self, callback=None):
        """
        Run continuous live updates.

        Args:
            callback: Function to call with each new feature vector
        """
        logger.info("Starting live feature updates...")

        # Initialize with history
        if not self.initialize_from_history():
            logger.error("Failed to initialize with history")
            return

        # Main update loop
        import time
        last_close = None

        while True:
            try:
                # Get latest close
                new_close = self.data_puller.get_latest_m1_close()

                if new_close and new_close != last_close:
                    # New bar detected
                    logger.info(f"New bar: {new_close}")

                    # Generate feature vector
                    features = self.process_new_bar(new_close)

                    if features is not None:
                        # Get model input format
                        model_input = self.get_feature_vector_for_model()

                        if callback and model_input is not None:
                            callback(features, model_input)

                        logger.info(f"Generated feature vector: shape={features.shape}")

                    last_close = new_close

                # Wait before next check
                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                logger.info("Live updates stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in live update: {e}")
                time.sleep(60)  # Wait longer on error


if __name__ == "__main__":
    # Test incremental feature builder
    builder = IncrementalFeatureBuilder()

    # Initialize with history
    if builder.initialize_from_history():
        logger.info("✅ Successfully initialized feature builder")

        # Test generating a feature vector
        test_close = 199.5
        features = builder.process_new_bar(test_close)

        if features is not None:
            logger.info(f"✅ Generated feature vector: shape={features.shape}")
            logger.info(f"  First 5 values: {features[:5]}")
            logger.info(f"  Last 5 values: {features[-5:]}")

            # Get model input
            model_input = builder.get_feature_vector_for_model()
            if model_input is not None:
                logger.info(f"✅ Model input shape: {model_input.shape}")
        else:
            logger.error("Failed to generate feature vector")
    else:
        logger.error("Failed to initialize feature builder")