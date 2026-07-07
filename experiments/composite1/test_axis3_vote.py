"""test_axis3_vote.py — TDD for axis3_vote.py: the direction-mapping generalization (the
EUR_USD-style sign=+1 case and the USD_JPY-style sign=-1 case), the release-lag-respecting
asof lookup, and the block-preserving-by-currency permutation."""
import numpy as np
import pandas as pd
import pytest

import _paths  # noqa: F401
import cot_signal as sig
from axis3_vote import (
    Z_THRESH,
    build_currency_z_series,
    composite_gate,
    make_zlookup,
    permute_currency_z_series,
    view_direction_for_ccy,
)


def test_pair_to_ccy_sign_covers_all_seven_direct_pairs():
    from axis3_vote import PAIR_TO_CCY_SIGN
    expected_pairs = {"EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD"}
    assert set(PAIR_TO_CCY_SIGN.keys()) == expected_pairs


def test_view_direction_sign_plus_one_eur_usd():
    # fading a HIGH of EUR_USD -> short the pair (direction_pair=-1) -> short EUR (view=-1)
    ccy, view = view_direction_for_ccy(-1, "EUR_USD")
    assert ccy == "EUR"
    assert view == -1
    # fading a LOW -> long the pair (direction_pair=+1) -> long EUR (view=+1)
    ccy, view = view_direction_for_ccy(+1, "EUR_USD")
    assert ccy == "EUR"
    assert view == +1


def test_view_direction_sign_minus_one_usd_jpy():
    # fading a HIGH of USD_JPY -> short the pair (direction_pair=-1) -> LONG JPY (view=+1)
    ccy, view = view_direction_for_ccy(-1, "USD_JPY")
    assert ccy == "JPY"
    assert view == +1
    # fading a LOW of USD_JPY -> long the pair (direction_pair=+1) -> SHORT JPY (view=-1)
    ccy, view = view_direction_for_ccy(+1, "USD_JPY")
    assert ccy == "JPY"
    assert view == -1


def _lookup_from_dict(fixed: dict):
    def lookup(ccy, asof_ts):
        return fixed.get(ccy, float("nan"))
    return lookup


def test_composite_gate_eur_usd_fade_high_requires_z_ge_plus1():
    lookup = _lookup_from_dict({"EUR": 1.5})
    assert composite_gate(-1, "EUR_USD", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is True
    lookup = _lookup_from_dict({"EUR": 0.5})
    assert composite_gate(-1, "EUR_USD", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is False


def test_composite_gate_eur_usd_fade_low_requires_z_le_minus1():
    lookup = _lookup_from_dict({"EUR": -1.5})
    assert composite_gate(+1, "EUR_USD", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is True
    lookup = _lookup_from_dict({"EUR": -0.5})
    assert composite_gate(+1, "EUR_USD", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is False


def test_composite_gate_usd_jpy_fade_high_requires_z_le_minus1_on_jpy():
    # fading a HIGH of USD_JPY -> view_ccy(JPY)=+1 -> requires z(JPY) <= -1.0 (crowded short)
    lookup = _lookup_from_dict({"JPY": -1.5})
    assert composite_gate(-1, "USD_JPY", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is True
    lookup = _lookup_from_dict({"JPY": 1.5})
    assert composite_gate(-1, "USD_JPY", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is False


def test_composite_gate_usd_jpy_fade_low_requires_z_ge_plus1_on_jpy():
    lookup = _lookup_from_dict({"JPY": 1.5})
    assert composite_gate(+1, "USD_JPY", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is True
    lookup = _lookup_from_dict({"JPY": -1.5})
    assert composite_gate(+1, "USD_JPY", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is False


def test_composite_gate_nan_z_never_fires():
    lookup = _lookup_from_dict({})  # no data for any currency
    assert composite_gate(-1, "EUR_USD", pd.Timestamp("2015-01-01", tz="UTC"), lookup) is False


def test_zlookup_respects_release_lag_no_lookahead():
    """Synthetic 2-currency panel with a hand-built z series + release_lag alignment;
    confirms the lookup never returns a z whose action_date is AFTER the query date."""
    cal = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
    report_dates = pd.date_range("2020-01-07", "2020-06-02", freq="7D")  # weekly Tuesdays
    cot_df = pd.concat([
        pd.DataFrame({"currency": "EUR", "report_date": report_dates,
                      "net_noncomm_frac_oi": np.linspace(-0.3, 0.3, len(report_dates))}),
        pd.DataFrame({"currency": "JPY", "report_date": report_dates,
                      "net_noncomm_frac_oi": np.linspace(0.3, -0.3, len(report_dates))}),
    ], ignore_index=True)
    # monkeypatch compute_zscore_panel's window via a small wrapper so this synthetic (short)
    # panel actually produces non-NaN z's -- build_currency_z_series itself is the real,
    # unmodified code path under test, just fed a pre-z-scored frame is not possible since it
    # calls cot_signal.compute_zscore_panel internally with the real Z_WINDOW=156; a synthetic
    # panel that short would be all-NaN under the real window, so this test instead calls the
    # release-lag alignment step directly (also verbatim cot_positioning code) on a
    # hand-computed z frame, which is what build_currency_z_series does internally.
    import release_lag as rl
    z_df = sig.compute_zscore_panel(cot_df, window=4, min_periods=2)
    aligned = rl.align_cot_to_action_dates(z_df.dropna(subset=["z"])[["currency", "report_date", "z"]], cal)
    series = {ccy: g[["action_date", "z"]].sort_values("action_date").reset_index(drop=True)
              for ccy, g in aligned.groupby("currency")}
    lookup = make_zlookup(series)

    # For every released (currency, action_date, z) row, querying exactly at its own
    # action_date must return that z (asof-backward, inclusive of the boundary).
    for ccy, g in series.items():
        for _, row in g.iterrows():
            got = lookup(ccy, row["action_date"])
            assert got == pytest.approx(row["z"])
    # Querying strictly BEFORE the first released action_date must return nan.
    first_date = min(g["action_date"].min() for g in series.values())
    before = first_date - pd.Timedelta(days=1)
    assert np.isnan(lookup("EUR", before))


def test_build_currency_z_series_end_to_end_real_window():
    """Full, unmodified build_currency_z_series (real Z_WINDOW=156) on a synthetic panel
    long enough (200 weeks) to actually emit non-NaN z's -- an integration test of the two
    verbatim-reused cot_positioning calls glued together."""
    cal = pd.bdate_range("2016-01-01", "2020-12-31", tz="UTC")
    report_dates = pd.date_range("2016-01-05", periods=220, freq="7D")  # weekly Tuesdays
    rng = np.random.default_rng(7)
    cot_df = pd.concat([
        pd.DataFrame({"currency": ccy, "report_date": report_dates,
                      "net_noncomm_frac_oi": rng.normal(0, 0.2, len(report_dates))})
        for ccy in ["EUR", "JPY"]
    ], ignore_index=True)
    series = build_currency_z_series(cot_df, cal)
    assert set(series.keys()) == {"EUR", "JPY"}
    for ccy, g in series.items():
        assert len(g) > 0
        assert g["z"].notna().all()
        assert g["action_date"].is_monotonic_increasing


def test_permute_currency_z_series_is_block_preserving():
    series = {
        "EUR": pd.DataFrame({"action_date": pd.date_range("2020-01-01", periods=10, freq="7D", tz="UTC"),
                              "z": np.arange(10.0)}),
        "JPY": pd.DataFrame({"action_date": pd.date_range("2020-01-01", periods=10, freq="7D", tz="UTC"),
                              "z": np.arange(100.0, 110.0)}),
    }
    perm = permute_currency_z_series(series, seed=1)
    for ccy in series:
        # same dates, same multiset of z-values, generally a DIFFERENT order (not asserted
        # strictly, since a random permutation can rarely equal the identity)
        assert list(perm[ccy]["action_date"]) == list(series[ccy]["action_date"])
        assert sorted(perm[ccy]["z"].tolist()) == sorted(series[ccy]["z"].tolist())
    # never mixes currencies: EUR's shuffled values must never include a JPY-range value
    assert perm["EUR"]["z"].max() < 50
    assert perm["JPY"]["z"].min() >= 100


def test_permute_is_seeded_reproducible():
    series = {"EUR": pd.DataFrame({"action_date": pd.date_range("2020-01-01", periods=20, freq="7D", tz="UTC"),
                                    "z": np.arange(20.0)})}
    p1 = permute_currency_z_series(series, seed=42)
    p2 = permute_currency_z_series(series, seed=42)
    assert list(p1["EUR"]["z"]) == list(p2["EUR"]["z"])
