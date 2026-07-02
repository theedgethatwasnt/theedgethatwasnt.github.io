#!/usr/bin/env python3
"""
S5-BA data loader for the conservative 010 backtest.

Loads per-pair S5 bid/ask parquet, builds base-TF mid OHLC + bid_c/ask_c arrays,
then projects stack-alignment novelty signals + PSAR verbatim from stack010_equity.py.

R3b: asserts 100% finite bid_c/ask_c — hard-fail, no fallback, no fixed-spread proxy.
IS_FRAC = 4/6 (matches all sma_exits experiments).
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

# Import helpers from the sma_exits folder (same as stack010_equity.py)
_SMA = Path(__file__).parent.parent / "sma_exits"
sys.path.insert(0, str(_SMA))

import h17_stack_alignment as H
from _lib import IS_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series

PROJECT = Path(os.environ.get("FX_CORE_ROOT", Path(__file__).resolve().parents[3]))
S5_DIR  = PROJECT / "data" / "s5_ohlc"
MAX_ROWS = 5_000_000

# Verbatim from stack010_equity.py
# pair -> (pip, tf1_min, tf2_min, sma_tuple, tp_pips, use_psar, af_start, act, fence)
CFG = {
    "EUR_JPY": (0.01,   2.0, 10.0, (5, 15, 35), 20.0, False, 0.0,   0.0, 200.0),
    "EUR_USD": (0.0001, 1.0,  5.0, (5, 15, 35), 30.0, True,  0.020, 20.0, 200.0),
    "GBP_USD": (0.0001, 0.5,  1.0, (7, 22, 50),  0.0, True,  0.020, 20.0, 200.0),
    "USD_JPY": (0.01,   1.0,  5.0, (5, 10, 22), 15.0, False, 0.0,   0.0, 200.0),
}


def _fast_tail_read_ba(path: Path, max_rows: int) -> "pd.DataFrame":
    """Like H.fast_tail_read but also reads bid_c and ask_c columns."""
    import pandas as pd
    pf = pq.ParquetFile(str(path))
    n_rg = pf.metadata.num_row_groups
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    take = []; total = 0
    for i in range(n_rg - 1, -1, -1):
        take.append(i); total += rg_rows[i]
        if total >= max_rows: break
    take.sort()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'bid_c', 'ask_c']
    tbl = pf.read_row_groups(take, columns=cols)
    df = tbl.to_pandas()
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df


def load_pair_ba(pair: str) -> dict:
    """Load S5 BA data for one pair, project stack signals verbatim from stack010.

    Returns
    -------
    dict with keys:
      pair, pip, n, is_end,
      m5_o, m5_h, m5_l, m5_c  — mid OHLC per S5 bar (base TF),
      bid_c, ask_c             — close bid/ask per S5 bar (R3b: 100% finite),
      t1l, t1s, t2l, t2s      — novelty alignment on TF1/TF2 projected to base,
      sar                      — PSAR on TF1 projected to base (NaN if not use_psar),
      cfg                      — full CFG tuple for this pair
    """
    pip, t1m, t2m, (ss, sm, sl), tp, use_psar, af, act, fence = CFG[pair]

    # Load S5 BA parquet with bid/ask columns
    df = _fast_tail_read_ba(S5_DIR / f"{pair}_S5_BA.parquet", MAX_ROWS)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Mid OHLC arrays
    o  = df['open'].to_numpy(np.float64)
    h  = df['high'].to_numpy(np.float64)
    l  = df['low'].to_numpy(np.float64)
    c  = df['close'].to_numpy(np.float64)
    ts = df['timestamp'].to_numpy()
    n  = len(df)

    # Bid/ask close arrays
    bid_c = df['bid_c'].to_numpy(np.float64)
    ask_c = df['ask_c'].to_numpy(np.float64)

    # R3b: 100% finite bid/ask — hard-fail, no fallback
    assert np.isfinite(bid_c).all() and np.isfinite(ask_c).all(), (
        f"R3b violation: {pair} has non-finite bid_c/ask_c in S5 BA parquet"
    )

    # Previous-bar timestamps for causal projection (verbatim from stack010)
    prev = np.empty_like(ts)
    prev[0] = ts[0]; prev[1:] = ts[:-1]

    # Resample to TF1/TF2 (verbatim from stack010 — uses H.resample_minutes)
    tf1 = H.resample_minutes(df, t1m, 5 / 60)
    tf2 = H.resample_minutes(df, t2m, 5 / 60)

    def nov(cc, tt):
        """Compute novelty alignment signals on a TF and project to base bars."""
        a = sma(cc, ss); b = sma(cc, sm); d = sma(cc, sl)
        lg = H.novelty(H.tf_signal(cc, a, b, d, 1))
        sh = H.novelty(H.tf_signal(cc, a, b, d, 0))
        return (project_to_m5(prev, tt, lg).astype(np.int8),
                project_to_m5(prev, tt, sh).astype(np.int8))

    t1l, t1s = nov(tf1['close'].to_numpy(float), tf1['timestamp'].to_numpy())
    t2l, t2s = nov(tf2['close'].to_numpy(float), tf2['timestamp'].to_numpy())

    # PSAR on TF1 projected to base bars (verbatim from stack010)
    if use_psar:
        sar = project_to_m5(
            prev, tf1['timestamp'].to_numpy(),
            psar_series(
                tf1['high'].to_numpy(float),
                tf1['low'].to_numpy(float),
                af, 0.10
            )
        )
    else:
        sar = np.full(n, np.nan)

    is_end = int(n * IS_FRAC)

    return {
        'pair':   pair,
        'pip':    pip,
        'n':      n,
        'is_end': is_end,
        # Base-TF timestamps (int64 unix-ns or datetime64, for portfolio sort)
        'ts':     ts,
        # Base-TF mid OHLC
        'm5_o':   o,
        'm5_h':   h,
        'm5_l':   l,
        'm5_c':   c,
        # Bid/ask close (R3b guaranteed finite)
        'bid_c':  bid_c,
        'ask_c':  ask_c,
        # Projected novelty signals
        't1l':    t1l,
        't1s':    t1s,
        't2l':    t2l,
        't2s':    t2s,
        # Projected PSAR (all-NaN if use_psar=False)
        'sar':    sar,
        # Full config tuple (for backtest_pair / sweep use)
        'cfg':    CFG[pair],
    }
