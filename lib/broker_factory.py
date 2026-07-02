"""
Broker factory — creates the appropriate adapter based on environment config.

Usage:
    from lib.broker_factory import create_broker
    broker = create_broker()  # Reads BROKER_TYPE env var
"""

import os
import logging
from lib.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


def create_broker(broker_type: str = None, account_id: str = None) -> BrokerAdapter:
    """Create a broker adapter based on type.

    Args:
        broker_type: "oanda" (default), "ctrader", or "mt5"
        account_id: Account identifier (format depends on broker)

    Returns:
        BrokerAdapter instance
    """
    if broker_type is None:
        broker_type = os.environ.get("BROKER_TYPE", "oanda").lower()

    if broker_type == "oanda":
        from lib.oanda_adapter import OANDAAdapter
        acct = account_id or os.environ.get("OANDA_ACCOUNT", "")
        api_key = os.environ.get("OANDA_API_KEY", "")
        hostname = os.environ.get("OANDA_HOST", "api-fxtrade.oanda.com")
        return OANDAAdapter(acct, api_key, hostname)

    elif broker_type == "ctrader":
        from lib.ctrader_adapter import CTraderAdapter
        acct = account_id or os.environ.get("CTRADER_ACCOUNT_ID", "")
        return CTraderAdapter(acct)

    elif broker_type == "mt5":
        # Future: MT5 REST bridge adapter
        raise NotImplementedError("MT5 adapter not yet implemented. "
                                  "See ARCHITECTURE.md for the bridge design.")

    else:
        raise ValueError(f"Unknown broker type: {broker_type}. "
                         f"Use 'oanda', 'ctrader', or 'mt5'.")
