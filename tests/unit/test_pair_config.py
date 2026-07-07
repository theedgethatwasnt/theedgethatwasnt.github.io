"""Tests for pair configuration — the foundation of all calculations."""

import pytest
from lib.pair_config import (
    PAIRS, ALL_PAIR_NAMES, CURRENCIES, PNF_CONFIGS,
    get_pair, is_jpy_pair, format_price,
)


class TestPairConfig:
    """Pair metadata must be correct — wrong pip values = wrong trades."""

    def test_all_12_pairs_defined(self):
        assert len(PAIRS) == 12
        assert len(ALL_PAIR_NAMES) == 12

    def test_jpy_pairs_have_correct_pip(self):
        jpy_pairs = ["EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY",
                      "CAD_JPY", "CHF_JPY", "NZD_JPY"]
        for name in jpy_pairs:
            pair = get_pair(name)
            assert pair.pip == 0.01, f"{name} should have pip=0.01"
            assert pair.pip_location == -2
            assert pair.price_precision == 3

    def test_usd_pairs_have_correct_pip(self):
        usd_pairs = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP"]
        for name in usd_pairs:
            pair = get_pair(name)
            assert pair.pip == 0.0001, f"{name} should have pip=0.0001"
            assert pair.pip_location == -4
            assert pair.price_precision == 5

    def test_currencies_extracted_correctly(self):
        # Every base and quote currency should be in CURRENCIES
        for pair in PAIRS.values():
            assert pair.base in CURRENCIES, f"{pair.base} not in CURRENCIES"
            assert pair.quote in CURRENCIES, f"{pair.quote} not in CURRENCIES"

    def test_spread_caps_are_2x_median(self):
        for pair in PAIRS.values():
            assert pair.max_entry_spread == pytest.approx(
                pair.median_spread_pips * 2, abs=0.01
            ), f"{pair.name} max_entry_spread should be 2× median"

    def test_is_jpy_pair(self):
        assert is_jpy_pair("EUR_JPY") is True
        assert is_jpy_pair("EUR_USD") is False

    def test_format_price_jpy(self):
        assert format_price("EUR_JPY", 184.321) == "184.321"
        assert format_price("EUR_JPY", 184.3) == "184.300"

    def test_format_price_usd(self):
        assert format_price("EUR_USD", 1.08765) == "1.08765"

    def test_unknown_pair_raises(self):
        with pytest.raises(KeyError):
            get_pair("INVALID_PAIR")

    def test_pnf_configs_complete(self):
        assert len(PNF_CONFIGS) == 4
        names = {c["name"] for c in PNF_CONFIGS}
        assert names == {"5pip_rev2", "5pip_rev3", "15pip_rev2", "15pip_rev3"}
