"""is_data.py — COT Contrarian Positioning: HARD IS/OOS BOUNDARY + IS-only loaders.

PREREGISTRATION.md: "IS = first 70% of the joint COT×price window. OOS = final 30%,
SEALED, one shot after user gate."

IS_END_REPORT_DATE was derived ONCE (2026-07-07) as the first 70% BY ROW COUNT of the
fully-resolved joint rebalance schedule (portfolio.build_rebalance_schedule on the
committed cot_weekly.parquet + data/d1_deep_ba/*, hard price ceiling 2026-05-21 already
baked into the D1 fetch): 916 total weekly rebalances, 2008-10-25 -> 2026-05-10 (action
dates). int(916*0.70) = 641 -> the 641st row's report_date is 2021-01-26 (action_date
2021-01-31); the first OOS row is report_date 2021-02-02 (action_date 2021-02-07).
Recompute via: `sched = portfolio.build_rebalance_schedule(...); sched.index[int(len(sched)*0.70)]`.

Row-count split, not calendar-time split, because the weekly cadence only becomes
regular once all 7 currencies have 156 full weeks of history (from 2008-10 on) — a
calendar-time 70% split of the RAW 2005-2026 span would not equal a 70% split of the
actually-tradable joint window, which is the pre-registration's stated intent
("joint COT×price window", not "joint calendar span").

Every script in this battery must route through restrict_sched_to_is() (rebalance
schedule) and is_only_spread_medians() (cost inputs) — no other module may consume the
full schedule or full-history spread medians for anything gate-facing.
"""
import pandas as pd

import d1_data as d1

IS_END_REPORT_DATE = pd.Timestamp("2021-02-02")           # exclusive: report_date < this => IS
IS_PRICE_CUTOFF = pd.Timestamp("2021-02-07", tz="UTC")     # first OOS action_date (entry fill)


def restrict_sched_to_is(sched: pd.DataFrame) -> pd.DataFrame:
    """Hard IS filter on a rebalance schedule (index = report_date). Independent
    re-assertion after the filter, in case a future edit changes the filter and forgets
    this guard (same belt-and-suspenders pattern as multiday_contrarian/is_data.py)."""
    out = sched[sched.index < IS_END_REPORT_DATE].copy()
    if len(out) == 0:
        raise RuntimeError("0 IS rows — check IS_END_REPORT_DATE / sched construction")
    assert out.index.max() < IS_END_REPORT_DATE, "OOS LEAK in restrict_sched_to_is"
    assert out["action_date"].max() < IS_PRICE_CUTOFF, "OOS LEAK: an IS row's action_date crossed IS_PRICE_CUTOFF"
    # The LAST IS week's EXIT is allowed to land exactly AT IS_PRICE_CUTOFF (it closes
    # using the same real, already-public Monday-open price bar that is also the first
    # OOS week's entry fill — using that price to CLOSE a trade is not a COT-data leak,
    # since no COT information past IS_END_REPORT_DATE is ever read to get there).
    assert out["exit_action_date"].max() <= IS_PRICE_CUTOFF, (
        "OOS LEAK: an IS row's exit_action_date crossed past IS_PRICE_CUTOFF"
    )
    assert out["exit_action_date"].max() < IS_PRICE_CUTOFF + pd.Timedelta(days=10), (
        "exit_action_date unexpectedly far past IS_PRICE_CUTOFF — investigate before trusting downstream gates"
    )
    return out


def restrict_sched_to_oos(sched: pd.DataFrame) -> pd.DataFrame:
    """OOS rows — exists for completeness/tripwire-testing ONLY. The IS battery runner
    (run_is_battery.py) must never call this."""
    return sched[sched.index >= IS_END_REPORT_DATE].copy()


def is_only_spread_medians(price_panel: dict, cutoff: pd.Timestamp = IS_PRICE_CUTOFF) -> dict:
    """Per-pair median D1 spread (pips), computed ONLY from price bars strictly before
    `cutoff` — the pre-registration's cost inputs must not peek at OOS-era spread
    compression (analog of the backtest-live SOP's R5 'spread gate is IS-only' rule,
    applied here to a cost input rather than a gate threshold)."""
    out = {}
    for pair, (df, _vol) in price_panel.items():
        is_df = df[df.index < cutoff]
        if len(is_df) == 0:
            raise RuntimeError(f"{pair}: 0 IS-only price rows before cutoff {cutoff}")
        out[pair] = d1.median_spread_pips(is_df)
    return out
