"""
OANDA v20 broker adapter — implements BrokerAdapter for OANDA REST API.

Extracted from scalper/unified/live_neat_pf_unified.py OANDAClient class.
This is the ONLY file that imports v20. All other code uses BrokerAdapter.
"""

import os
import time
import logging
import random
from typing import Optional, List

import v20
from dotenv import load_dotenv

from lib.broker_adapter import BrokerAdapter, OrderResult, TradeInfo, AccountInfo
from lib.pair_config import format_price, is_jpy_pair
from lib.sizing import is_weekend_entry_blocked

logger = logging.getLogger(__name__)


class OANDAAdapter(BrokerAdapter):
    """OANDA v20 REST API adapter."""

    def __init__(self, account_id: Optional[str] = None):
        load_dotenv()
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        environment = os.getenv("OANDA_ENVIRONMENT", "live")

        if environment == "live":
            self.rest_host = "api-fxtrade.oanda.com"
            self.stream_host = "stream-fxtrade.oanda.com"
        else:
            self.rest_host = "api-fxpractice.oanda.com"
            self.stream_host = "stream-fxpractice.oanda.com"

        self.ctx = v20.Context(hostname=self.rest_host, port="443", token=self.api_key,
                               poll_timeout=10)   # 10s REST timeout — prevents poll loop stall
        logger.info(f"OANDA adapter: account {self.account_id} ({environment})")

    def place_market_order(self, instrument: str, units: int,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None,
                           max_retries: int = 3) -> OrderResult:
        """Place a market order with optional SL/TP.
        Global weekend guard: rejects all entries during weekend block window."""
        if is_weekend_entry_blocked():
            logger.warning(f"WEEKEND BLOCK: Rejected {instrument} {units}u — market closing/closed")
            return OrderResult(success=False, error="WEEKEND_ENTRY_BLOCKED")
        for attempt in range(max_retries):
            try:
                precision = 3 if is_jpy_pair(instrument) else 5
                kwargs = {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(int(units)),
                }
                if sl_price is not None:
                    kwargs["stopLossOnFill"] = {"price": f"{sl_price:.{precision}f}"}
                if tp_price is not None:
                    kwargs["takeProfitOnFill"] = {"price": f"{tp_price:.{precision}f}"}

                response = self.ctx.order.create(self.account_id, order=kwargs)

                if response.status == 201:
                    fill = response.body.get("orderFillTransaction")
                    if not fill:
                        cancel = response.body.get("orderCancelTransaction")
                        reason = getattr(cancel, "reason", "UNKNOWN") if cancel else "NO_FILL"
                        logger.error(f"Order cancelled by broker: {reason}")
                        return OrderResult(success=False, cancel_reason=str(reason))

                    price = float(fill.price)
                    trade_id = None
                    if hasattr(fill, "tradeOpened") and fill.tradeOpened:
                        trade_id = str(fill.tradeOpened.tradeID)
                    elif hasattr(fill, "tradesOpened") and fill.tradesOpened:
                        trade_id = str(fill.tradesOpened[0].tradeID)

                    logger.info(f"Order filled: {instrument} {units}u @ {price}, id={trade_id}")
                    return OrderResult(success=True, trade_id=trade_id, fill_price=price)
                else:
                    logger.warning(f"Order failed: HTTP {response.status} (attempt {attempt+1})")

            except Exception as e:
                logger.warning(f"Order error: {e} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1 + random.uniform(0, 1))

        return OrderResult(success=False, error="Exhausted retries")

    def close_trade(self, trade_id: str) -> OrderResult:
        """Close a trade at market."""
        try:
            response = self.ctx.trade.close(self.account_id, trade_id)
            if response.status == 200:
                fill = response.body.get("orderFillTransaction", {})
                price = float(getattr(fill, "price", 0))
                pl = float(getattr(fill, "pl", 0))
                logger.info(f"Trade {trade_id} closed @ {price}, P/L=${pl:.4f}")
                return OrderResult(success=True, fill_price=price)
            else:
                return OrderResult(success=False, error=f"HTTP {response.status}")
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def modify_trade_sl_tp(self, trade_id: str,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None) -> bool:
        """Update SL and/or TP on an open trade."""
        try:
            kwargs = {}
            # Determine precision from the trade's instrument
            precision = 5  # default
            try:
                details = self.get_trade_details(trade_id)
                if details and is_jpy_pair(details.instrument):
                    precision = 3
            except Exception:
                pass

            if sl_price is not None:
                kwargs["stopLoss"] = {"price": f"{sl_price:.{precision}f}"}
            if tp_price is not None:
                kwargs["takeProfit"] = {"price": f"{tp_price:.{precision}f}"}

            if not kwargs:
                return True

            response = self.ctx.trade.set_dependent_orders(
                self.account_id, str(trade_id), **kwargs
            )
            return response.status == 200
        except Exception as e:
            logger.error(f"Failed to modify SL/TP on trade {trade_id}: {e}")
            return False

    def modify_trade_trailing_stop(self, trade_id: str, pair: str,
                                    distance_pips: float) -> bool:
        """Replace static SL with a broker-side trailing stop.

        OANDA trails the SL server-side once set — no further modify calls needed.
        `distance_pips` is the trail distance in pips (e.g. 8).
        Removes takeProfit at the same time (trailing stop replaces it).
        """
        try:
            pip = 0.01 if is_jpy_pair(pair) else 0.0001
            distance_price = f"{distance_pips * pip:.{3 if is_jpy_pair(pair) else 5}f}"
            response = self.ctx.trade.set_dependent_orders(
                self.account_id, str(trade_id),
                trailingStopLoss={"distance": distance_price},
            )
            return response.status == 200
        except Exception as e:
            logger.error(f"Failed to set trailing stop on trade {trade_id}: {e}")
            return False

    def get_account_summary(self) -> Optional[AccountInfo]:
        """Get account summary from OANDA."""
        try:
            response = self.ctx.account.summary(self.account_id)
            if response.status == 200:
                acct = response.body.get("account", response.body)
                return AccountInfo(
                    account_id=self.account_id,
                    nav=float(acct.NAV),
                    balance=float(acct.balance),
                    unrealized_pl=float(acct.unrealizedPL),
                    margin_used=float(getattr(acct, 'marginUsed', 0)),
                    margin_available=float(getattr(acct, 'marginAvailable', 0)),
                    open_trade_count=int(getattr(acct, 'openTradeCount', 0)),
                    currency=str(acct.currency),
                )
        except Exception as e:
            logger.warning(f"Account summary failed: {e}")
        return None

    def get_open_trades(self) -> List[TradeInfo]:
        """Get all open trades."""
        try:
            response = self.ctx.trade.list_open(self.account_id)
            if response.status == 200:
                trades = []
                for t in response.body.get("trades", []):
                    sl = getattr(t, 'stopLossOrder', None)
                    tp = getattr(t, 'takeProfitOrder', None)
                    trades.append(TradeInfo(
                        trade_id=str(t.id),
                        instrument=t.instrument,
                        units=int(t.currentUnits),
                        entry_price=float(t.price),
                        unrealized_pl=float(t.unrealizedPL),
                        sl_price=float(sl.price) if sl else None,
                        tp_price=float(tp.price) if tp else None,
                        open_time=str(t.openTime)[:19] if hasattr(t, 'openTime') else None,
                    ))
                return trades
        except Exception as e:
            logger.warning(f"Failed to get open trades: {e}")
        return []

    def get_trade_details(self, trade_id: str) -> Optional[TradeInfo]:
        """Get details of a specific trade (open or closed)."""
        try:
            response = self.ctx.trade.get(self.account_id, trade_id)
            if response.status == 200:
                t = response.body.get("trade", response.body)
                sl = getattr(t, 'stopLossOrder', None)
                tp = getattr(t, 'takeProfitOrder', None)
                info = TradeInfo(
                    trade_id=str(t.id),
                    instrument=t.instrument,
                    units=int(t.currentUnits),
                    entry_price=float(t.price),
                    unrealized_pl=float(getattr(t, 'unrealizedPL', 0)),
                    sl_price=float(sl.price) if sl else None,
                    tp_price=float(tp.price) if tp else None,
                )
                # For closed trades, attach close price and realized P/L
                avg_close = getattr(t, 'averageClosePrice', None)
                if avg_close is not None:
                    info.close_price = float(avg_close)
                realized = getattr(t, 'realizedPL', None)
                if realized is not None:
                    info.realizedPL = float(realized)
                return info
        except Exception as e:
            logger.warning(f"Failed to get trade {trade_id}: {e}")
        return None

    def get_candles(self, instrument: str, count: int = 500,
                    granularity: str = "S5", max_retries: int = 5,
                    price: str = "MBA") -> list:
        """
        Fetch candles with retry + exponential backoff.
        price="MBA" requests mid + bid + ask so bid_c/ask_c are real values.
        """
        for attempt in range(max_retries):
            try:
                response = self.ctx.instrument.candles(
                    instrument=instrument, granularity=granularity,
                    count=count, price=price,
                )
                if response.status != 200:
                    logger.warning(f"Candle fetch {instrument}/{granularity}: "
                                   f"HTTP {response.status} (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + random.uniform(0, 2))
                        continue
                    return []

                candles = []
                for c in response.body.get("candles", []):
                    if c.complete:
                        bid = c.bid if (hasattr(c, 'bid') and c.bid) else c.mid
                        ask = c.ask if (hasattr(c, 'ask') and c.ask) else c.mid
                        candles.append({
                            "timestamp": str(c.time),
                            "open":  float(c.mid.o),
                            "high":  float(c.mid.h),
                            "low":   float(c.mid.l),
                            "close": float(c.mid.c),
                            "bid_c": float(bid.c),
                            "ask_c": float(ask.c),
                            "volume": int(c.volume),
                        })
                return candles
            except Exception as e:
                logger.warning(f"Candle fetch {instrument}/{granularity}: {e} "
                               f"(attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 2))
        return []

    def close_all_trades(self) -> int:
        """Close ALL open trades on this account. Returns count of trades closed."""
        trades = self.get_open_trades()
        closed = 0
        for t in trades:
            result = self.close_trade(t.trade_id)
            if result.success:
                closed += 1
            else:
                logger.error(f"Failed to close trade {t.trade_id} ({t.instrument}): {result.error}")
        if closed:
            logger.info(f"close_all_trades: Closed {closed}/{len(trades)} trades on {self.account_id}")
        return closed

    def create_stream_context(self) -> v20.Context:
        """Create a separate v20 Context for the streaming endpoint."""
        return v20.Context(hostname=self.stream_host, port="443", token=self.api_key,
                           stream_timeout=30)   # streaming context keep-alive timeout
