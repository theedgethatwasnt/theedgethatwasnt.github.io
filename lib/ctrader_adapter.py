"""
cTrader Open API adapter — connects to cTrader brokers (FTMO, FundedNext, etc.)
via REST/WebSocket. No Windows needed.

Setup:
  1. Register app at open.spotware.com → get client_id, client_secret
  2. OAuth2 flow → get access_token, refresh_token
  3. Set env vars:
     CTRADER_CLIENT_ID=...
     CTRADER_CLIENT_SECRET=...
     CTRADER_ACCESS_TOKEN=...
     CTRADER_REFRESH_TOKEN=...
     CTRADER_ACCOUNT_ID=...  (cTrader account number)

cTrader Open API docs: https://connect.spotware.com/docs/open-api
"""

import os
import time
import json
import logging
import requests
from typing import Optional, List
from datetime import datetime, timezone

from lib.broker_adapter import (
    BrokerAdapter, OrderResult, TradeInfo, AccountInfo,
)

logger = logging.getLogger(__name__)

# cTrader Open API base URLs
AUTH_URL = "https://connect.spotware.com/apps/token"
API_BASE = "https://api.spotware.com"

# Granularity mapping: our names → cTrader period names
GRANULARITY_MAP = {
    "S5": "S5", "S10": "S10", "S15": "S15", "S30": "S30",
    "M1": "M1", "M2": "M2", "M5": "M5", "M10": "M10",
    "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4",
    "D1": "D1", "W1": "W1", "MN": "MN",
}

# cTrader uses symbol names without underscore (e.g., "EURUSD" not "EUR_USD")
def _to_ctrader_symbol(instrument: str) -> str:
    return instrument.replace("_", "")

def _from_ctrader_symbol(symbol: str) -> str:
    """Convert EURUSD → EUR_USD."""
    if len(symbol) == 6:
        return f"{symbol[:3]}_{symbol[3:]}"
    return symbol


class CTraderAdapter(BrokerAdapter):
    """cTrader Open API broker adapter.

    Uses REST endpoints for trading operations and historical data.
    WebSocket available for live streaming (future enhancement).
    """

    def __init__(self, account_id: Optional[str] = None):
        self.client_id = os.environ.get("CTRADER_CLIENT_ID", "")
        self.client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "")
        self.access_token = os.environ.get("CTRADER_ACCESS_TOKEN", "")
        self.refresh_token = os.environ.get("CTRADER_REFRESH_TOKEN", "")
        self.account_id = account_id or os.environ.get("CTRADER_ACCOUNT_ID", "")

        if not self.access_token:
            logger.warning("CTRADER_ACCESS_TOKEN not set — adapter will not function")

        # Cache symbol info (pip size, lot size, etc.)
        self._symbol_cache = {}
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        })

        logger.info(f"CTrader adapter initialized for account {self.account_id}")

    def _refresh_auth(self):
        """Refresh OAuth2 access token."""
        if not self.refresh_token or not self.client_id:
            return False
        try:
            resp = requests.post(AUTH_URL, data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token", self.refresh_token)
                self._session.headers["Authorization"] = f"Bearer {self.access_token}"
                logger.info("cTrader token refreshed")
                return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
        return False

    def _api_get(self, path: str, params: dict = None) -> Optional[dict]:
        """GET request to cTrader API with auto-retry on auth failure."""
        url = f"{API_BASE}{path}"
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 401 and attempt == 0:
                    if self._refresh_auth():
                        continue
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"cTrader GET {path}: HTTP {resp.status_code}")
                return None
            except Exception as e:
                logger.error(f"cTrader GET {path}: {e}")
                return None
        return None

    def _api_post(self, path: str, data: dict = None) -> Optional[dict]:
        """POST request to cTrader API."""
        url = f"{API_BASE}{path}"
        for attempt in range(2):
            try:
                resp = self._session.post(url, json=data, timeout=15)
                if resp.status_code == 401 and attempt == 0:
                    if self._refresh_auth():
                        continue
                if resp.status_code in (200, 201):
                    return resp.json()
                logger.warning(f"cTrader POST {path}: HTTP {resp.status_code} — {resp.text[:200]}")
                return None
            except Exception as e:
                logger.error(f"cTrader POST {path}: {e}")
                return None
        return None

    def _get_symbol_info(self, instrument: str) -> dict:
        """Get symbol details (pip size, volume step, etc.). Cached."""
        symbol = _to_ctrader_symbol(instrument)
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        data = self._api_get(f"/v2/webserv/symbols",
                             params={"accountId": self.account_id})
        if data and "data" in data:
            for s in data["data"]:
                name = s.get("symbolName", "")
                self._symbol_cache[name] = {
                    "symbolId": s.get("symbolId"),
                    "pipPosition": s.get("pipPosition", 4),
                    "stepVolume": s.get("stepVolume", 100000),
                    "minVolume": s.get("minVolume", 1000),
                    "digits": s.get("digits", 5),
                }
        return self._symbol_cache.get(symbol, {})

    def _units_to_volume(self, instrument: str, units: int) -> int:
        """Convert our units to cTrader volume (in hundredths of a lot).
        cTrader volume: 100000 = 1 standard lot, 1000 = 0.01 lot (micro).
        """
        # cTrader volumes are in base units (same as our units for FX)
        return abs(units)

    # ─── BrokerAdapter Implementation ─────────────────────────────────

    def place_market_order(self, instrument: str, units: int,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None,
                           max_retries: int = 3) -> OrderResult:
        """Place a market order on cTrader."""
        symbol = _to_ctrader_symbol(instrument)
        volume = self._units_to_volume(instrument, units)
        side = "BUY" if units > 0 else "SELL"

        order_data = {
            "accountId": self.account_id,
            "symbolName": symbol,
            "orderType": "MARKET",
            "tradeSide": side,
            "volume": volume,
        }
        if sl_price is not None:
            order_data["stopLoss"] = sl_price
        if tp_price is not None:
            order_data["takeProfit"] = tp_price

        for attempt in range(max_retries):
            data = self._api_post("/v2/webserv/openTrade", order_data)
            if data:
                trade_id = str(data.get("tradeId") or data.get("orderId") or "")
                fill_price = data.get("executionPrice") or data.get("price")
                if fill_price:
                    fill_price = float(fill_price)
                logger.info(f"cTrader order filled: {instrument} {side} {volume}u "
                           f"@ {fill_price}, id={trade_id}")
                return OrderResult(success=True, trade_id=trade_id, fill_price=fill_price)
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)

        return OrderResult(success=False, error="Exhausted retries")

    def close_trade(self, trade_id: str) -> OrderResult:
        """Close a trade at market."""
        data = self._api_post("/v2/webserv/closeTrade", {
            "accountId": self.account_id,
            "tradeId": trade_id,
        })
        if data:
            fill_price = data.get("executionPrice") or data.get("closingPrice")
            if fill_price:
                fill_price = float(fill_price)
            logger.info(f"cTrader trade {trade_id} closed @ {fill_price}")
            return OrderResult(success=True, fill_price=fill_price)
        return OrderResult(success=False, error="Close failed")

    def modify_trade_sl_tp(self, trade_id: str,
                           sl_price: Optional[float] = None,
                           tp_price: Optional[float] = None) -> bool:
        """Update SL and/or TP on an open trade."""
        mod_data = {
            "accountId": self.account_id,
            "tradeId": trade_id,
        }
        if sl_price is not None:
            mod_data["stopLoss"] = sl_price
        if tp_price is not None:
            mod_data["takeProfit"] = tp_price

        data = self._api_post("/v2/webserv/modifyTrade", mod_data)
        return data is not None

    def get_account_summary(self) -> Optional[AccountInfo]:
        """Get account NAV, balance, margin info."""
        data = self._api_get(f"/v2/webserv/traders/{self.account_id}")
        if not data:
            return None
        try:
            acct = data.get("data", data)
            return AccountInfo(
                account_id=self.account_id,
                nav=float(acct.get("equity", 0)),
                balance=float(acct.get("balance", 0)),
                unrealized_pl=float(acct.get("equity", 0)) - float(acct.get("balance", 0)),
                margin_used=float(acct.get("usedMargin", 0)),
                margin_available=float(acct.get("freeMargin", 0)),
                open_trade_count=int(acct.get("openTradesCount", 0)),
                currency=acct.get("depositCurrency", "USD"),
            )
        except Exception as e:
            logger.error(f"Account summary parse error: {e}")
            return None

    def get_open_trades(self) -> List[TradeInfo]:
        """Get all open trades."""
        data = self._api_get(f"/v2/webserv/openTrades",
                             params={"accountId": self.account_id})
        if not data:
            return []

        trades = []
        for t in data.get("data", []):
            try:
                instrument = _from_ctrader_symbol(t.get("symbolName", ""))
                volume = int(t.get("volume", 0))
                side = t.get("tradeSide", "BUY")
                units = volume if side == "BUY" else -volume
                trades.append(TradeInfo(
                    trade_id=str(t.get("tradeId", "")),
                    instrument=instrument,
                    units=units,
                    entry_price=float(t.get("entryPrice", 0)),
                    unrealized_pl=float(t.get("profit", 0)),
                    sl_price=float(t["stopLoss"]) if t.get("stopLoss") else None,
                    tp_price=float(t["takeProfit"]) if t.get("takeProfit") else None,
                    open_time=t.get("openTimestamp"),
                ))
            except Exception as e:
                logger.warning(f"Trade parse error: {e}")
        return trades

    def get_trade_details(self, trade_id: str) -> Optional[TradeInfo]:
        """Get details of a specific trade."""
        trades = self.get_open_trades()
        for t in trades:
            if t.trade_id == trade_id:
                return t
        return None

    def get_candles(self, instrument: str, count: int = 500,
                    granularity: str = "S5", max_retries: int = 5) -> list:
        """Fetch historical candles from cTrader.

        Returns list of dicts matching OANDA adapter format:
        [{timestamp, open, high, low, close, volume, bid_c, ask_c}]
        """
        symbol = _to_ctrader_symbol(instrument)
        period = GRANULARITY_MAP.get(granularity, granularity)

        for attempt in range(max_retries):
            data = self._api_get(f"/v2/webserv/tradingaccounts/{self.account_id}/symbols/"
                                 f"{symbol}/trendbars/{period}",
                                 params={"count": count})
            if data and "data" in data:
                candles = []
                for bar in data["data"]:
                    try:
                        ts = bar.get("timestamp") or bar.get("time", "")
                        o = float(bar.get("open", 0)) / 100000  # cTrader sends prices * 10^digits
                        h = float(bar.get("high", 0)) / 100000
                        l = float(bar.get("low", 0)) / 100000
                        c = float(bar.get("close", 0)) / 100000

                        # cTrader may return prices already as floats depending on API version
                        # Check if values are sensible
                        if o > 1000:  # Likely needs division
                            info = self._get_symbol_info(instrument)
                            divisor = 10 ** info.get("digits", 5)
                            o /= divisor; h /= divisor; l /= divisor; c /= divisor

                        candles.append({
                            "timestamp": str(ts),
                            "open": o, "high": h, "low": l, "close": c,
                            "bid_c": c,  # cTrader doesn't separate bid/ask in bars
                            "ask_c": c,
                            "volume": int(bar.get("volume", 0)),
                        })
                    except Exception as e:
                        logger.warning(f"Candle parse error: {e}")
                return candles

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        return []
