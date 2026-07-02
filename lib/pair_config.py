"""
Pair configuration — single source of truth for all instrument metadata.

Every service imports pair info from here. No hardcoding pip values anywhere else.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PairConfig:
    """Immutable configuration for a single FX pair."""
    name: str               # e.g., "EUR_JPY"
    pip: float              # Pip size: 0.01 for JPY pairs, 0.0001 for others
    pip_location: int       # OANDA pipLocation: -2 for JPY, -4 for others
    price_precision: int    # Decimal places for price formatting
    base: str               # Base currency (e.g., "EUR")
    quote: str              # Quote currency (e.g., "JPY")
    median_spread_pips: float  # Historical median spread in pips (from S5 data)
    max_entry_spread: float    # 2× median — entry blocked above this


# All 12 traded pairs with verified pip values (from OANDA v20 API pipLocation)
PAIRS: Dict[str, PairConfig] = {
    "EUR_JPY": PairConfig("EUR_JPY", 0.01, -2, 3, "EUR", "JPY", 2.3, 4.6),
    "USD_JPY": PairConfig("USD_JPY", 0.01, -2, 3, "USD", "JPY", 1.7, 3.4),
    "GBP_JPY": PairConfig("GBP_JPY", 0.01, -2, 3, "GBP", "JPY", 3.3, 6.6),
    "AUD_JPY": PairConfig("AUD_JPY", 0.01, -2, 3, "AUD", "JPY", 2.1, 4.2),
    "CAD_JPY": PairConfig("CAD_JPY", 0.01, -2, 3, "CAD", "JPY", 2.3, 4.6),
    "CHF_JPY": PairConfig("CHF_JPY", 0.01, -2, 3, "CHF", "JPY", 3.5, 7.0),
    "NZD_JPY": PairConfig("NZD_JPY", 0.01, -2, 3, "NZD", "JPY", 2.7, 5.4),
    "EUR_USD": PairConfig("EUR_USD", 0.0001, -4, 5, "EUR", "USD", 1.6, 3.2),
    "GBP_USD": PairConfig("GBP_USD", 0.0001, -4, 5, "GBP", "USD", 1.9, 3.8),
    "AUD_USD": PairConfig("AUD_USD", 0.0001, -4, 5, "AUD", "USD", 1.3, 2.6),
    "NZD_USD": PairConfig("NZD_USD", 0.0001, -4, 5, "NZD", "USD", 1.5, 3.0),
    "EUR_GBP": PairConfig("EUR_GBP", 0.0001, -4, 5, "EUR", "GBP", 1.4, 2.8),
}

ALL_PAIR_NAMES = list(PAIRS.keys())

# All currencies that appear in our pairs
CURRENCIES = ["EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]

# P&F box configurations used across strategies
PNF_CONFIGS = [
    {"name": "5pip_rev2", "box_size_pips": 5, "reversal": 2},
    {"name": "5pip_rev3", "box_size_pips": 5, "reversal": 3},
    {"name": "15pip_rev2", "box_size_pips": 15, "reversal": 2},
    {"name": "15pip_rev3", "box_size_pips": 15, "reversal": 3},
]


def get_pair(name: str) -> PairConfig:
    """Get pair config by name. Raises KeyError if unknown."""
    return PAIRS[name]


def is_jpy_pair(name: str) -> bool:
    """Check if a pair is JPY-quoted (pip = 0.01)."""
    return PAIRS[name].pip >= 0.01


def format_price(pair_name: str, price: float) -> str:
    """Format price with correct decimal precision for the pair."""
    precision = PAIRS[pair_name].price_precision
    return f"{price:.{precision}f}"
