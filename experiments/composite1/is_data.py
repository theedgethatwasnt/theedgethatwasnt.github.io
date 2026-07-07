"""is_data.py — Composite 1: HARD IS/OOS BOUNDARY (inherited seal).

PREREGISTRATION.md "Inherited seal (binding)": "This experiment inherits cot_positioning's
IS/OOS boundary so the held seal is spent exactly once, here: IS = joint window through
2021-01-31; OOS = 2021-02-01 -> 2026-05-21, SEALED, one shot after the user gate. No
component may be evaluated on the OOS segment beforehand."

Two related boundary constants:

  COT_IS_END_REPORT_DATE = 2021-02-02 — IDENTICAL constant to
      cot_positioning/is_data.py's IS_END_REPORT_DATE. Required verbatim for gate 2 (Axis-3
      parity): "same window" means the same report_date cut, not an approximately-similar one.

  AXIS1_IS_ENTRY_CUTOFF = 2021-02-01 (UTC) — an Axis-1/composite trade's ENTRY (the
      next-D1-open fill after the signal bar) must be strictly before this for the trade to
      count as IS. This is the composite's own boundary, stated directly in
      PREREGISTRATION.md's "Inherited seal" section ("OOS = 2021-02-01 -> ..."), and sits
      STRICTLY INSIDE (never past) cot_positioning's own, looser IS_PRICE_CUTOFF
      (2021-02-07) — the more conservative of the two is used throughout this experiment.

  DATA_LOAD_CEILING = AXIS1_IS_ENTRY_CUTOFF + 30 calendar days — every D1 price parquet is
      hard-truncated at this point at LOAD time (a later row is never even read into memory,
      "loaders assert"). This gives an IS-entered trade (entry < AXIS1_IS_ENTRY_CUTOFF) more
      than 2x its own HCAP=10-D1-bar cap (~14 calendar days worst-case across a weekend) of
      runway to resolve its own exit without the loader ever touching a price bar anywhere
      near real OOS territory. This is the same pattern cot_positioning/is_data.py already
      documents and uses for its own last IS week's exit (allowed to land exactly at, never
      past, its own IS_PRICE_CUTOFF) — applied here to entries/exits instead of COT weeks.
"""
import pandas as pd

import _paths  # noqa: F401
import d1_data as d1

COT_IS_END_REPORT_DATE = pd.Timestamp("2021-02-02")
AXIS1_IS_ENTRY_CUTOFF = pd.Timestamp("2021-02-01", tz="UTC")
DATA_LOAD_CEILING = AXIS1_IS_ENTRY_CUTOFF + pd.Timedelta(days=30)


def load_pair_is(pair: str, data_dir: str = None) -> pd.DataFrame:
    """D1 price panel loader, hard-truncated to DATA_LOAD_CEILING at load time — no row at
    or past the ceiling is ever returned. Reuses d1_data.load_pair (verbatim, cot_positioning
    code) for the actual parquet read."""
    df = d1.load_pair(pair, data_dir) if data_dir else d1.load_pair(pair)
    out = df[df.index < DATA_LOAD_CEILING].copy()
    if len(out) == 0:
        raise RuntimeError(f"{pair}: 0 rows before DATA_LOAD_CEILING {DATA_LOAD_CEILING}")
    assert out.index.max() < DATA_LOAD_CEILING, f"{pair}: OOS-buffer LEAK in load_pair_is"
    return out


def restrict_cot_to_is(cot_df: pd.DataFrame) -> pd.DataFrame:
    """Hard IS filter on the raw COT weekly panel (index-free — filters the report_date
    column directly), independently re-asserted after the filter."""
    out = cot_df[cot_df["report_date"] < COT_IS_END_REPORT_DATE].copy()
    if len(out) == 0:
        raise RuntimeError("0 IS COT rows — check COT_IS_END_REPORT_DATE / cot_df")
    assert out["report_date"].max() < COT_IS_END_REPORT_DATE, "OOS LEAK in restrict_cot_to_is"
    return out


def assert_trade_is_is(entry_ts, exit_ts) -> None:
    """Belt-and-suspenders re-check on an assembled trade (mirrors cot_positioning/
    is_data.py's own pattern of an independent post-hoc re-assertion, in case a future edit
    changes a filter upstream and forgets this guard)."""
    entry_ts = pd.Timestamp(entry_ts)
    exit_ts = pd.Timestamp(exit_ts)
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    if exit_ts.tzinfo is None:
        exit_ts = exit_ts.tz_localize("UTC")
    assert entry_ts < AXIS1_IS_ENTRY_CUTOFF, (
        f"OOS LEAK: entry_ts {entry_ts} >= AXIS1_IS_ENTRY_CUTOFF {AXIS1_IS_ENTRY_CUTOFF}"
    )
    assert exit_ts < DATA_LOAD_CEILING, (
        f"OOS-buffer LEAK: exit_ts {exit_ts} >= DATA_LOAD_CEILING {DATA_LOAD_CEILING}"
    )
