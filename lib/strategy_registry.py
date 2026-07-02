"""
Strategy Registry — single source of truth for all account/strategy configuration.

Usage:
    from lib.strategy_registry import STRATEGIES, get_strategy, enabled_strategies

All services (telegram_bot, dashboard, portfolio_mgr) import from here
instead of maintaining their own hardcoded ACCOUNTS lists.

To add/remove/disable a strategy: edit lib/strategy_registry.json
"""

import json
from pathlib import Path
from typing import Optional


_REGISTRY_PATH = Path(__file__).parent / "strategy_registry.json"


def _load():
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


# Loaded once at import time. Restart service to pick up changes.
STRATEGIES = _load()

# Quick lookups
BY_ACCOUNT = {s["account"]: s for s in STRATEGIES}
BY_STRATEGY_NAME = {s["strategy_name"]: s for s in STRATEGIES}
BY_CONTAINER = {s["container"]: s for s in STRATEGIES}


def get_strategy(account: str) -> Optional[dict]:
    """Get strategy config by account label (e.g. '001')."""
    return BY_ACCOUNT.get(account)


def enabled_strategies() -> list:
    """Return only enabled strategies."""
    return [s for s in STRATEGIES if s.get("enabled", True)]


def disabled_strategies() -> list:
    """Return only disabled strategies."""
    return [s for s in STRATEGIES if not s.get("enabled", True)]


def classic_strategies() -> list:
    """Return classic (non-NEAT) strategies."""
    return [s for s in STRATEGIES if s.get("type") == "classic"]


def neat_strategies() -> list:
    """Return NEAT-based strategies."""
    return [s for s in STRATEGIES if s.get("type") == "neat"]


def ensemble_strategies() -> list:
    """Return ensemble gate strategies (accounts 004/005)."""
    return [s for s in STRATEGIES if "ensemble" in s.get("strategy_name", "")]


ALL_12_PAIRS = ["EUR_JPY", "USD_JPY", "GBP_JPY", "GBP_USD", "EUR_USD", "AUD_USD",
                "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY", "NZD_USD", "EUR_GBP"]
