"""Focused composite test on the 3 working pairs only.

For each pair, hold the scratch params fixed at the known IS+OOS winning
config, and try adding the early-MFE quality filter (T_q, X) on top.

Question: does adding quality filter to existing scratch overlay
make 3-pair portfolio better, worse, or about the same?
"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

spec2 = importlib.util.spec_from_file_location('comp',
    Path(__file__).parent / 'composite_rule_backtest.py')
comp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(comp)

PROJECT = Path('/path/to/projects/fx-core')
DATA = PROJECT / 'data/m5_ohlc'
OUT = PROJECT / 'research/experiments/loss_cap_sweep/results'

KNOWN_SCRATCH = {
    'USD_JPY': (24.0, 10.0),
    'NZD_USD': (24.0,  5.0),
    'GBP_USD': ( 8.0, 15.0),
}
Q_GRID = [(None, None)] + [(Th, X) for Th in [0.5, 1, 2, 4] for X in [2, 3, 5, 8]]


def main():
    is_bars = int(bv.WINDOW_BARS * 4/6)
    rows = []
    for pair, (Ts, W) in KNOWN_SCRATCH.items():
        df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
        df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)
        T_s_bars = int(Ts * 12)
        for (Tq, X) in Q_GRID:
            T_q_bars = int(Tq * 12) if Tq else None
            r_is = comp.run_composite(pair, df_is, T_q_bars, X, T_s_bars, W) or {}
            r_oos = comp.run_composite(pair, df_oos, T_q_bars, X, T_s_bars, W) or {}
            rows.append({
                'pair': pair, 'T_q_h': Tq, 'X': X, 'T_s_h': Ts, 'W': W,
                'is_net': r_is.get('net', 0), 'is_dd': r_is.get('max_dd', 0),
                'oos_net': r_oos.get('net', 0), 'oos_dd': r_oos.get('max_dd', 0),
                'oos_trades': r_oos.get('trades', 0),
                'oos_tp': r_oos.get('tp', 0),
                'oos_q': r_oos.get('quality', 0),
                'oos_s': r_oos.get('scratch', 0),
                'oos_end': r_oos.get('end', 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'composite_focused.csv', index=False)

    print(f"{'Pair':<8} {'T_q':>5} {'X':>3} {'IS':>9} {'OOS':>9} {'OOS_DD':>9} "
          f"{'Tr':>4} {'TP/Q/S/E':>14}")
    print('-' * 75)
    for pair in KNOWN_SCRATCH:
        sub = df[df['pair']==pair].sort_values('oos_net', ascending=False)
        for _, r in sub.iterrows():
            tag = ''
            if r['T_q_h'] is None or pd.isna(r['T_q_h']):
                tag = '  <-- scratch-alone'
            reasons = f"{int(r['oos_tp'])}/{int(r['oos_q'])}/{int(r['oos_s'])}/{int(r['oos_end'])}"
            tq_str = f"{r['T_q_h']:.1f}" if (r['T_q_h'] is not None and not pd.isna(r['T_q_h'])) else '  -'
            x_str = f"{int(r['X'])}" if (r['X'] is not None and not pd.isna(r['X'])) else '  -'
            print(f"{pair:<8} {tq_str:>5s} {x_str:>3s} {r['is_net']:>+9.0f} {r['oos_net']:>+9.0f} "
                  f"{r['oos_dd']:>+9.0f} {int(r['oos_trades']):>4d} {reasons:>14s}{tag}")
        print()

    print('\n=== Best IS+OOS+ per pair, ranked by OOS ===')
    best = df[(df['is_net']>0) & (df['oos_net']>0)].sort_values('oos_net', ascending=False)
    best_per_pair = best.groupby('pair').head(1)
    for _, r in best_per_pair.iterrows():
        q_label = 'scratch-alone' if (r['T_q_h'] is None or pd.isna(r['T_q_h'])) else f"T_q={r['T_q_h']}h X={int(r['X'])}p"
        print(f"  {r['pair']}: {q_label}, T_s={r['T_s_h']}h W={int(r['W'])}p  "
              f"IS{r['is_net']:+.0f} OOS{r['oos_net']:+.0f} DD{r['oos_dd']:+.0f}")
    tot_is = best_per_pair['is_net'].sum()
    tot_oos = best_per_pair['oos_net'].sum()
    print(f"\n3-pair portfolio: IS={tot_is:+.0f}p OOS={tot_oos:+.0f}p "
          f"worst-pair-DD={best_per_pair['oos_dd'].min():+.0f}p")


if __name__ == '__main__':
    main()
