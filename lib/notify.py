#!/usr/bin/env python3
"""
Telegram trade notifications for FX trading bots.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your bot, then visit https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       TELEGRAM_CHAT_ID=123456789
"""

import os
import logging
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load once at import — check lib/.env first, then project root .env
_HERE = os.path.dirname(os.path.abspath(__file__))
for _env_path in (os.path.join(_HERE, ".env"), os.path.join(_HERE, "..", ".env")):
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def _send(text: str):
    """Send a Telegram message. Fails silently (trading must never block on notifications)."""
    if not _BOT_TOKEN or not _CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        logger.debug(f"Telegram send failed: {e}")


def _acct_short(account_id: str) -> str:
    """Extract short account number from full OANDA account ID."""
    if not account_id:
        return ""
    # "<OANDA_ACCOUNT_ID>" -> "008"
    parts = account_id.split("-")
    return parts[-1] if parts else account_id


def notify_open(strategy: str, pair: str, direction: str, units, entry_price: float,
                tp: float = None, sl: float = None, extra: str = "",
                account: str = ""):
    """Send trade-open notification. Green circle for LONG, red for SHORT."""
    icon = "\U0001f7e2" if direction.upper() == "LONG" else "\U0001f534"  # green/red
    acct = _acct_short(account)
    acct_tag = f" [{acct}]" if acct else ""
    lines = [
        f"{icon} <b>OPEN | {strategy}{acct_tag}</b>",
        f"<code>{pair} {direction.upper()} {units} units @ {entry_price}</code>",
    ]
    if tp is not None and sl is not None:
        lines.append(f"<code>TP: {tp}  |  SL: {sl}</code>")
    if extra:
        lines.append(f"<code>{extra}</code>")
    _send("\n".join(lines))


def notify_close(strategy: str, pair: str, direction: str, units,
                 entry_price: float, exit_price: float, pnl_pips: float,
                 reason: str = "", extra: str = "",
                 account: str = ""):
    """Send trade-close notification. Green circle for profit, red for loss."""
    icon = "\U0001f7e2" if pnl_pips >= 0 else "\U0001f534"  # green/red by P&L
    sign = "+" if pnl_pips >= 0 else ""
    acct = _acct_short(account)
    acct_tag = f" [{acct}]" if acct else ""
    lines = [
        f"{icon} <b>CLOSE | {strategy}{acct_tag}</b>",
        f"<code>{pair} {direction.upper()} {units} units</code>",
        f"<code>Entry: {entry_price} \u2192 Exit: {exit_price}</code>",
        f"<code>P/L: {sign}{pnl_pips:.1f} pips  |  {reason}</code>",
    ]
    if extra:
        lines.append(f"<code>{extra}</code>")
    _send("\n".join(lines))
