"""Scratch-exit backtest: add the rule
    "after T_act hours, if price comes within W pips of entry, exit"
to the SMA16 strategy and measure portfolio-net P&L after spread.

Trade-offs:
- Scratch too eagerly: cut winners that would have eventually hit TP
- Scratch too late: stuck-position losses persist
- Scratch window too wide: exits in profit territory unnecessarily
- Scratch window too narrow: never triggers

IS (first 4mo) / OOS (last 2mo) split. Net of spread cost.
"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, requests

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

PROJECT = Path('/path/to/projects/fx-core')
DATA = PROJECT / 'data/m5_ohlc'
OUT = PROJECT / 'research/experiments/loss_cap_sweep/results'

TP_PIPS = 20.0


def run(pair, df_slice, T_act_bars, W_pips):
    pip, sp_gate = bv.PAIRS[pair]
    spread_cost = sp_gate * bv.SPREAD_FRAC

    h1 = bv.resample_tf(df_slice, 60)
    m30 = bv.resample_tf(df_slice, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()
    m5_ts = df_slice['timestamp'].to_numpy()
    m5_o = df_slice['open'].to_numpy(); m5_h = df_slice['high'].to_numpy()
    m5_l = df_slice['low'].to_numpy(); m5_c = df_slice['close'].to_numpy()

    pos_dir = 0; entry_px = 0.0; entry_bar = -1
    trades = []

    for i in range(1, len(df_slice)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - entry_bar
            # 1) TP intrabar
            if pos_dir == 1:
                tp_lvl = entry_px + TP_PIPS * pip
                if m5_h[i] >= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; continue
            else:
                tp_lvl = entry_px - TP_PIPS * pip
                if m5_l[i] <= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; continue
            # 2) Scratch-exit: after T_act_bars, if close is within W_pips of entry, exit at close
            if held >= T_act_bars:
                close_dist_pips = abs(m5_c[i] - entry_px) / pip
                if close_dist_pips <= W_pips:
                    pnl = (m5_c[i] - entry_px) / pip * pos_dir
                    trades.append((pnl, 'scratch', held))
                    pos_dir = 0; continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i
    if pos_dir != 0:
        pnl = (m5_c[-1] - entry_px) / pip * pos_dir
        trades.append((pnl, 'end', len(df_slice) - 1 - entry_bar))
    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=['pips','reason','bars'])
    tdf['pips_net'] = tdf['pips'] - spread_cost
    cum = tdf['pips_net'].cumsum()
    return {
        'trades': len(tdf),
        'gross': float(tdf['pips'].sum()),
        'net': float(tdf['pips_net'].sum()),
        'wr': float((tdf['pips_net'] > 0).mean() * 100),
        'max_dd': float((cum - cum.cummax()).min()),
        'tp_count': int((tdf['reason']=='TP').sum()),
        'scratch_count': int((tdf['reason']=='scratch').sum()),
        'end_count': int((tdf['reason']=='end').sum()),
        'scratch_pips': float(tdf[tdf['reason']=='scratch']['pips_net'].sum()),
        'end_pips': float(tdf[tdf['reason']=='end']['pips_net'].sum()),
    }


def main():
    is_bars = int(bv.WINDOW_BARS * 4/6)
    grid = [(T_h, W) for T_h in [2, 4, 6, 8, 12, 24] for W in [3, 5, 10, 15, 20]]

    rows = []
    for pair in bv.PAIRS:
        df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
        df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)
        for T_h, W in grid:
            T_bars = T_h * 12
            r_is = run(pair, df_is, T_bars, W) or {}
            r_oos = run(pair, df_oos, T_bars, W) or {}
            rows.append({
                'pair': pair, 'T_h': T_h, 'W': W,
                'is_net': r_is.get('net', 0), 'is_dd': r_is.get('max_dd', 0),
                'is_trades': r_is.get('trades', 0),
                'oos_net': r_oos.get('net', 0), 'oos_dd': r_oos.get('max_dd', 0),
                'oos_trades': r_oos.get('trades', 0),
                'oos_scratch': r_oos.get('scratch_count', 0),
                'oos_end': r_oos.get('end_count', 0),
                'oos_scratch_pips': r_oos.get('scratch_pips', 0),
                'oos_end_pips': r_oos.get('end_pips', 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'scratch_exit_per_pair.csv', index=False)

    # Aggregate by (T_h, W) for portfolio across all 10 pairs
    print(f"{'T_h':>4} {'W':>4} {'IS_net':>10} {'OOS_net':>10} {'OOS_DD':>10} "
          f"{'TPs':>5} {'Scr':>5} {'End':>5} {'Scr_p':>9} {'End_p':>10}")
    print('-' * 86)
    portfolio = []
    grid_summary = df.groupby(['T_h','W']).agg(
        is_net=('is_net','sum'), oos_net=('oos_net','sum'),
        worst_dd=('oos_dd','min'), oos_scratch=('oos_scratch','sum'),
        oos_end=('oos_end','sum'),
        oos_scratch_pips=('oos_scratch_pips','sum'),
        oos_end_pips=('oos_end_pips','sum'),
    ).reset_index()
    grid_summary = grid_summary.sort_values('oos_net', ascending=False)
    for _, r in grid_summary.iterrows():
        marker = ''
        if r['oos_net'] > 0 and r['is_net'] > 0:
            marker = '  ★IS+OOS+'
        elif r['oos_net'] > 0:
            marker = '  OOS+'
        print(f"{int(r['T_h']):>4d} {int(r['W']):>4d} "
              f"{r['is_net']:>+10.1f} {r['oos_net']:>+10.1f} "
              f"{r['worst_dd']:>+10.1f} "
              f"{'':>5} {int(r['oos_scratch']):>5d} {int(r['oos_end']):>5d} "
              f"{r['oos_scratch_pips']:>+9.1f} {r['oos_end_pips']:>+10.1f}{marker}")

    grid_summary.to_csv(OUT/'scratch_exit_grid.csv', index=False)

    # Find best IS+OOS combo
    best = grid_summary[(grid_summary['is_net'] > 0) & (grid_summary['oos_net'] > 0)]
    if len(best):
        best = best.sort_values('oos_net', ascending=False).iloc[0]
        msg = (
            f"🎯 SCRATCH-EXIT — IS+OOS positive portfolio!\n"
            f"  Rule: after {int(best['T_h'])}h held, exit at close if "
            f"|price - entry| <= {int(best['W'])} pips\n"
            f"  IS net: {best['is_net']:+.0f}p  OOS net: {best['oos_net']:+.0f}p\n"
            f"  Worst pair OOS DD: {best['worst_dd']:+.0f}p\n"
            f"  Scratch exits in OOS: {int(best['oos_scratch'])} "
            f"({best['oos_scratch_pips']:+.0f}p) vs end-close {int(best['oos_end'])} "
            f"({best['oos_end_pips']:+.0f}p)"
        )
        print('\n' + msg)
        tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        chat = os.environ.get('TELEGRAM_CHAT_ID', '')
        if tok and chat:
            try:
                requests.post(f'https://api.telegram.org/bot{tok}/sendMessage',
                              json={'chat_id': chat, 'text': msg}, timeout=10)
                print('telegram sent ✓')
            except Exception as e:
                print(f'telegram failed: {e}')
    else:
        print('\nNo (T_h, W) combo is IS+OOS positive on portfolio.')


if __name__ == '__main__':
    main()
