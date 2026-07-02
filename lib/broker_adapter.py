"""
Broker adapter — abstract interface for order execution.

Strategies never import broker-specific libraries (v20, ctrader, mt5).
They use this interface. Swapping brokers = changing one environment variable.

Current implementation: OANDAAdapter (v20 REST API).
Future: CTraderAdapter, MT5Adapter.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of a market order."""
    success: bool
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    error: Optional[str] = None
    cancel_reason: Optional[str] = None


@dataclass
class TradeInfo:
    """Info about an open trade on the broker."""
    trade_id: str
    instrument: str
    units: int              # Signed: positive=long, negative=short
    entry_price: float
    unrealized_pl: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    open_time: Optional[str] = None


@dataclass
class AccountInfo:
    """Account summary from broker."""
    account_id: str
    nav: float
    balance: float
    unrealized_pl: float
    margin_used: float
    margin_available: float
    open_trade_count: int
    currency: str = "USD"


class BrokerAdapter(ABC):
    """Abstract broker interface. All strategies use this."""

    @abstractmethod
    def place_market_order(self, instrument: str, units: int,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None,
                           max_retries: int = 3) -> OrderResult:
        """Place a market order. units: positive=buy, negative=sell."""

    @abstractmethod
    def close_trade(self, trade_id: str) -> OrderResult:
        """Close an open trade at market."""

    @abstractmethod
    def modify_trade_sl_tp(self, trade_id: str,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None) -> bool:
        """Update SL and/or TP on an open trade."""

    @abstractmethod
    def get_account_summary(self) -> Optional[AccountInfo]:
        """Get account NAV, balance, margin info."""

    @abstractmethod
    def get_open_trades(self) -> List[TradeInfo]:
        """Get all open trades on this account."""

    @abstractmethod
    def get_trade_details(self, trade_id: str) -> Optional[TradeInfo]:
        """Get details of a specific trade (check if still open)."""

    @abstractmethod
    def get_candles(self, instrument: str, count: int = 500,
                    granularity: str = "S5", max_retries: int = 5) -> list:
        """Fetch historical candles.

        Returns list of dicts: [{timestamp, open, high, low, close, volume, bid_c, ask_c}]
        """
