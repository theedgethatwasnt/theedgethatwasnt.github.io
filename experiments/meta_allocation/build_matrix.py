#!/usr/bin/env python3
"""
Workstream F (meta-allocation) — build the daily per-strategy P&L matrix from trades.duckdb.

Grouping convention (per plan docs/superpowers/plans/2026-07-06-multiday-unscanned-field-program.md):
  - live trades (is_paper=False): grouped by `strategy` column, pooled across labels/pairs, ANY size.
  - paper trades (is_paper=True): grouped by `strategy` column, pooled across labels/pairs,
    but the group is KEPT only if it has >50 total closed trades.
  - live and paper groups that share a strategy name (e.g. post_shock_retrace has both a live
    retrace_009 label and paper retrace_nofilter/retrace_atr labels) are kept as SEPARATE columns
    (suffix `__paper` on the paper one) — they are different, non-fungible things even though the
    trades table gives them the same `strategy` string.

Daily P&L = sum of pnl_pips over trades whose exit_time falls on that calendar date (UTC, floor to
day). Only CLOSED trades (exit_time + pnl_pips both non-null) are used — open positions carry no
realized P&L yet (R1: only realized/closed information is usable, consistent with the SOP's
"closed bars only" spirit applied to trades instead of bars).

Each strategy-column has an "active window" = [first closed-trade exit date, last closed-trade exit
date]. Inside that window, a day with no trades is 0 pips (the strategy was running but didn't fire).
Outside that window the cell is NaN (the strategy did not exist yet / was already stopped) — this
matters for the equal-weight baseline and the eligible-pool for the random-null (Workstream F step 4).

Outputs (written to /root/work/code_meta/results/ on the compute box):
  daily_pnl_matrix.csv   — date index x strategy columns, NaN outside each strategy's active window
  group_meta.csv         — per-column: n_trades, is_paper, first_date, last_date, total_pips
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/work/trades_2026-07-06.duckdb"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/work/code_meta/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_MIN_TRADES = 50


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT strategy, label, is_paper, pnl_pips, entry_time, exit_time
        FROM trades
        WHERE exit_time IS NOT NULL AND pnl_pips IS NOT NULL
    """).fetchdf()
    con.close()
    n_total = len(df)
    df['exit_date'] = pd.to_datetime(df['exit_time']).dt.floor('D')
    df['is_paper'] = df['is_paper'].astype(bool)

    # group key: live -> strategy; paper -> strategy__paper (kept only if group n>50)
    live = df[~df['is_paper']].copy()
    paper = df[df['is_paper']].copy()

    live['group_key'] = live['strategy']

    paper_counts = paper.groupby('strategy').size()
    paper_keep = set(paper_counts[paper_counts > PAPER_MIN_TRADES].index)
    paper = paper[paper['strategy'].isin(paper_keep)].copy()
    paper['group_key'] = paper['strategy'] + '__paper'

    dropped_paper = sorted(set(paper_counts.index) - paper_keep)
    print(f"Total closed trades in DB: {n_total}")
    print(f"Live groups: {sorted(live['group_key'].unique())}")
    print(f"Paper groups kept (>{PAPER_MIN_TRADES} trades): {sorted(paper['group_key'].unique())}")
    print(f"Paper groups DROPPED (<= {PAPER_MIN_TRADES} trades): {dropped_paper} "
          f"(n={[int(paper_counts[k]) for k in dropped_paper]})")

    kept = pd.concat([live, paper], ignore_index=True)
    print(f"Trades kept after grouping filter: {len(kept)} / {n_total} "
          f"({100*len(kept)/n_total:.1f}%)")

    # daily net pnl per group
    daily = kept.groupby(['exit_date', 'group_key'])['pnl_pips'].sum().unstack('group_key')

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full_idx)
    daily.index.name = 'date'

    # active window per group: NaN outside [first, last] closed-trade date, 0 (not NaN) inside
    meta_rows = []
    for col in daily.columns:
        s = kept[kept['group_key'] == col]
        first, last = s['exit_date'].min(), s['exit_date'].max()
        n = len(s)
        total = s['pnl_pips'].sum()
        is_paper_flag = bool(s['is_paper'].iloc[0])
        meta_rows.append(dict(group_key=col, n_trades=n, is_paper=is_paper_flag,
                               first_date=first, last_date=last, total_pips=total))
        mask_out = (daily.index < first) | (daily.index > last)
        daily.loc[mask_out, col] = np.nan
        daily.loc[~mask_out, col] = daily.loc[~mask_out, col].fillna(0.0)

    meta = pd.DataFrame(meta_rows).sort_values('total_pips')
    daily.to_csv(OUT_DIR / 'daily_pnl_matrix.csv')
    meta.to_csv(OUT_DIR / 'group_meta.csv', index=False)

    print(f"\nMatrix: {daily.shape[0]} days x {daily.shape[1]} strategy-groups "
          f"({daily.index.min().date()} -> {daily.index.max().date()})")
    print(f"\nPer-group summary (sorted by total pips):")
    with pd.option_context('display.max_rows', 200, 'display.width', 160):
        print(meta.to_string(index=False))
    print(f"\nWrote {OUT_DIR/'daily_pnl_matrix.csv'} and {OUT_DIR/'group_meta.csv'}")


if __name__ == '__main__':
    main()
