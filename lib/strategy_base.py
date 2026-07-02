"""
Strategy base class — plug-and-play interface for all trading strategies.

New strategy = subclass Strategy + implement event handlers + add to docker-compose.yml.
The base class handles: ZMQ subscription, event dispatch, position tracking, logging.
Strategy code only contains: entry/exit logic.
"""

import logging
import os
import signal
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from lib.broker_adapter import BrokerAdapter, OrderResult, TradeInfo
from lib.zmq_protocol import (
    Subscriber, MARKET_PUB, ALLOCATION_PUB,
    MSG_CANDLE, MSG_PNF_BOX, MSG_INDICATOR, MSG_KALMAN, MSG_ALLOCATION,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyRequirements:
    """Declares what data a strategy needs from the curator.

    Used for ZMQ topic subscription filtering — the strategy only receives
    events it declared interest in.
    """
    pairs: List[str]
    candle_timeframes: List[str] = field(default_factory=list)  # ["S5", "M5", "H1"]
    pnf_configs: List[str] = field(default_factory=list)        # ["5pip_rev3"]
    needs_kalman: bool = False
    needs_indicators: bool = True
    needs_allocation: bool = True


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Lifecycle:
        1. __init__: set up broker, requirements
        2. on_start: called once after ZMQ connected, before event loop
        3. event loop: on_candle / on_pnf_box / on_indicator / on_kalman / on_allocation
        4. on_stop: called on shutdown (SIGTERM)
    """

    requires: StrategyRequirements

    def __init__(self, broker: BrokerAdapter, strategy_name: str):
        self.broker = broker
        self.strategy_name = strategy_name
        self._shutdown = False
        self._last_allocation: Dict[str, Any] = {}

    # ─── Event Handlers (override in subclass) ─────────────────────────

    def on_start(self):
        """Called once after connection, before event loop. Override for warmup."""
        pass

    def on_stop(self):
        """Called on shutdown. Override for cleanup."""
        pass

    def on_candle(self, pair: str, granularity: str, candle: dict):
        """Called when a new candle completes."""
        pass

    def on_pnf_box(self, pair: str, config: str, boxes: list):
        """Called when new P&F box(es) form."""
        pass

    def on_indicator(self, pair: str, indicators: dict):
        """Called when indicator snapshot updates."""
        pass

    def on_kalman(self, granularity: str, strengths: dict, ranks: dict):
        """Called when Kalman strength updates."""
        pass

    def on_allocation(self, weights: dict):
        """Called when portfolio allocation changes."""
        self._last_allocation = weights

    # ─── Event Loop ────────────────────────────────────────────────────

    def run(self):
        """Main entry point — connects ZMQ, dispatches events until shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Build ZMQ topic subscriptions from requirements
        topics = self._build_topics()
        logger.info(f"[{self.strategy_name}] Subscribing to {len(topics)} topics")

        market_sub = Subscriber(MARKET_PUB, topics=topics)
        alloc_sub = Subscriber(ALLOCATION_PUB) if self.requires.needs_allocation else None

        self.on_start()
        self._touch_alive()
        logger.info(f"[{self.strategy_name}] Entering event loop")

        while not self._shutdown:
            try:
                # Check market events (non-blocking, 100ms timeout)
                result = market_sub.receive(timeout_ms=100)
                if result:
                    topic, msg = result
                    self._dispatch(msg)

                # Check allocation events (non-blocking)
                if alloc_sub:
                    alloc_result = alloc_sub.receive(timeout_ms=0)
                    if alloc_result:
                        _, alloc_msg = alloc_result
                        if alloc_msg.get("type") == MSG_ALLOCATION:
                            self.on_allocation(alloc_msg.get("weights", {}))

                # Touch liveness file periodically
                self._touch_alive()

            except Exception as e:
                logger.error(f"[{self.strategy_name}] Event loop error: {e}", exc_info=True)
                time.sleep(1)

        self.on_stop()
        market_sub.close()
        if alloc_sub:
            alloc_sub.close()
        logger.info(f"[{self.strategy_name}] Shutdown complete")

    def _dispatch(self, msg: dict):
        """Route a ZMQ message to the appropriate handler."""
        msg_type = msg.get("type")
        pair = msg.get("pair")

        if msg_type == MSG_CANDLE:
            self.on_candle(pair, msg.get("granularity"), msg)
        elif msg_type == MSG_PNF_BOX:
            self.on_pnf_box(pair, msg.get("config"), msg.get("boxes", []))
        elif msg_type == MSG_INDICATOR:
            self.on_indicator(pair, msg)
        elif msg_type == MSG_KALMAN:
            self.on_kalman(msg.get("granularity"), msg.get("strengths", {}), msg.get("ranks", {}))

    def _build_topics(self) -> List[str]:
        """Build ZMQ topic filter strings from requirements."""
        topics = []
        for pair in self.requires.pairs:
            for tf in self.requires.candle_timeframes:
                topics.append(f"candle.{pair}.{tf}")
            for cfg in self.requires.pnf_configs:
                topics.append(f"pnf_box.{pair}.{cfg}")
            if self.requires.needs_indicators:
                topics.append(f"indicator.{pair}")
        if self.requires.needs_kalman:
            topics.append("kalman.")
        return topics

    def _touch_alive(self):
        """Touch /tmp/alive file for Docker healthcheck."""
        try:
            with open("/tmp/alive", "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        logger.info(f"[{self.strategy_name}] Signal {signum}, shutting down...")
        self._shutdown = True
