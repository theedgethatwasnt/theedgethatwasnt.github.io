"""Phase 1 of the scratch-exit study.

For the baseline V0 (TP=20p, no SL) on all 10 SMA16 pairs, capture
detailed per-trade telemetry:
  - holding time (M5 bars and hours)
  - exit reason (TP / end)
  - MAE (max adverse excursion, pips)
  - MFE (max favorable excursion, pips)
  - for force-closed trades: did price ever return within X pips of entry
    after some delay? (i.e. would a scratch-exit have caught it?)

Output:
  results/baseline_trades.csv  -- per-trade detail
  prints:
    - holding-time percentiles
    - exit-reason breakdown
    - for non-TP trades: distribution of "time until first revisit to entry +- X pips"

Insight target: find T_activation such that >95% of WINNING trades
have already exited by T_activation, and study what fraction of
non-winning trades revisit entry within a scratch-window after T_activation.
"""
from pathlib import Path
import sys, importlib.util
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

PROJECT = Path('/path/to/projects/fx-core')
DATA = PROJECT / 'data/m5_ohlc'
OUT = PROJECT / 'research/experiments/loss_cap_sweep/results'


def trade_telemetry(pair):
    pip, sp_gate = bv.PAIRS[pair]
    df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
    df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)

    h1 = bv.resample_tf(df, 60)
    m30 = bv.resample_tf(df, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()
    m5_ts = df['timestamp'].to_numpy()
    m5_o = df['open'].to_numpy(); m5_h = df['high'].to_numpy()
    m5_l = df['low'].to_numpy(); m5_c = df['close'].to_numpy()

    TP_PIPS = 20.0
    pos_dir = 0; entry_px = 0.0; entry_bar = -1
    mfe = 0.0; mae = 0.0
    # For each non-TP trade, record the per-bar distance from entry
    # for every bar of the position. Encoded as a list of (bars_since_entry, abs_dist_pips, signed_dist_pips)
    path = []
    trades = []
    for i in range(1, len(df)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            # Update MFE/MAE in pips signed by direction
            high_pip = (m5_h[i] - entry_px) / pip * pos_dir
            low_pip = (m5_l[i] - entry_px) / pip * pos_dir
            mfe = max(mfe, high_pip)
            mae = min(mae, low_pip)
            # signed close distance (positive = in profit, negative = adverse)
            close_pip = (m5_c[i] - entry_px) / pip * pos_dir
            path.append((i - entry_bar, abs(close_pip), close_pip))

            # TP check
            tp_price = entry_px + pos_dir * TP_PIPS * pip
            tp_hit = (pos_dir == 1 and m5_h[i] >= tp_price) or \
                     (pos_dir == -1 and m5_l[i] <= tp_price)
            if tp_hit:
                trades.append({
                    'pair': pair, 'dir': pos_dir,
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'bars': i - entry_bar,
                    'hours': (i - entry_bar) / 12.0,
                    'pips': TP_PIPS, 'reason': 'TP',
                    'mfe': float(mfe), 'mae': float(mae),
                    'path': path.copy(),
                })
                pos_dir = 0; mfe = 0.0; mae = 0.0; path = []
                continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i
            mfe = 0.0; mae = 0.0; path = []

    if pos_dir != 0:
        exit_px = m5_c[-1]
        pnl = (exit_px - entry_px) / pip * pos_dir
        trades.append({
            'pair': pair, 'dir': pos_dir,
            'entry_bar': entry_bar, 'exit_bar': len(df) - 1,
            'bars': len(df) - 1 - entry_bar,
            'hours': (len(df) - 1 - entry_bar) / 12.0,
            'pips': float(pnl), 'reason': 'end',
            'mfe': float(mfe), 'mae': float(mae),
            'path': path.copy(),
        })
    return trades


def main():
    all_trades = []
    for pair in bv.PAIRS:
        print(f'processing {pair}...', flush=True)
        all_trades.extend(trade_telemetry(pair))

    # DataFrame without the heavy 'path' column for csv
    df = pd.DataFrame([{k: v for k, v in t.items() if k != 'path'} for t in all_trades])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'baseline_trades.csv', index=False)
    print(f"\nTotal trades: {len(df)}")
    print(df.groupby('reason').size().to_string())
    print()
    print("=== Holding-time distribution (hours), TP exits only ===")
    tp = df[df['reason'] == 'TP']
    if len(tp):
        print(f"  count: {len(tp)}")
        print(f"  median: {tp['hours'].median():.1f}h")
        print(f"  mean: {tp['hours'].mean():.1f}h")
        for p in [25, 50, 75, 80, 90, 95, 99]:
            print(f"  p{p}: {tp['hours'].quantile(p/100):.1f}h")
        print(f"  max: {tp['hours'].max():.1f}h")
    print()
    print("=== Holding-time distribution (hours), force-closed at end ===")
    nf = df[df['reason'] == 'end']
    if len(nf):
        print(f"  count: {len(nf)}")
        print(f"  hours range: {nf['hours'].min():.1f} - {nf['hours'].max():.1f}")
        print(f"  pips at force-close (mean): {nf['pips'].mean():.1f}")
        print(f"  pips at force-close (worst): {nf['pips'].min():.1f}")
        print(f"  MAE distribution of force-closed:")
        for p in [25, 50, 75, 90, 95]:
            print(f"    p{p}: {nf['mae'].quantile(p/100):.1f}p")

    # === Scratch-exit potential study ===
    # For each NON-TP trade, after T_act hours, was there ever a moment
    # where price came within W pips of entry? If yes, what was the
    # earliest such moment, and what would the exit P&L be?
    print("\n=== Scratch-exit potential on non-TP (force-closed) trades ===")
    scratch_results = []
    for T_act_hours in [2, 4, 6, 8, 12, 24, 48]:
        T_act_bars = int(T_act_hours * 12)
        for W in [2, 5, 10, 15]:
            caught = 0
            improvement_total = 0.0
            actual_end_total = 0.0
            for t in all_trades:
                if t['reason'] != 'end':
                    continue
                # Find earliest bar in path with bars_since_entry >= T_act_bars and |dist| <= W
                exit_at_path_pips = None
                for (b, abs_d, signed_d) in t['path']:
                    if b >= T_act_bars and abs_d <= W:
                        exit_at_path_pips = signed_d
                        break
                if exit_at_path_pips is not None:
                    caught += 1
                    improvement_total += exit_at_path_pips - t['pips']
                    actual_end_total += t['pips']
            n_end = (df['reason'] == 'end').sum()
            if n_end > 0 and caught > 0:
                scratch_results.append({
                    'T_act_h': T_act_hours, 'W_pips': W,
                    'caught': caught, 'of': n_end,
                    'caught_pct': 100 * caught / n_end,
                    'pips_improvement': improvement_total,
                    'avg_caught_improvement': improvement_total / caught,
                })
    sdf = pd.DataFrame(scratch_results)
    if len(sdf):
        print(sdf.to_string(index=False))
        sdf.to_csv(OUT / 'scratch_potential.csv', index=False)
    else:
        print("  no force-closed trades or no scratch opportunities found")


if __name__ == '__main__':
    main()
