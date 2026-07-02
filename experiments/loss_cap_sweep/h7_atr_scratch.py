"""H7: ATR-scaled scratch window.

Rule: instead of fixed W pips, use W = k * ATR(14, H1) at entry.
Hypothesis: pairs with different volatility need different scratch
windows; fixed pips may be wrong for some pairs.

Tests all 10 pairs to find which pairs benefit. IS/OOS split.
"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

PROJECT = Path('/path/to/projects/fx-core')
DATA = PROJECT / 'data/m5_ohlc'
OUT = PROJECT / 'research/experiments/loss_cap_sweep/results'
TP_PIPS = 20.0


def run_atr_scratch(pair, df_slice, T_s_bars, k):
    """W (pips) = k * ATR_at_entry (pips). Recomputed per entry."""
    pip, sp_gate = bv.PAIRS[pair]
    spread_cost = sp_gate * bv.SPREAD_FRAC

    h1 = bv.resample_tf(df_slice, 60); m30 = bv.resample_tf(df_slice, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy(); m30_ts = m30['timestamp'].to_numpy()
    atr = bv.atr_h1(h1)
    m5_ts = df_slice['timestamp'].to_numpy()
    m5_o = df_slice['open'].to_numpy(); m5_h = df_slice['high'].to_numpy()
    m5_l = df_slice['low'].to_numpy(); m5_c = df_slice['close'].to_numpy()

    pos_dir = 0; entry_px = 0.0; entry_bar = -1; W_pips_for_trade = 0.0
    trades = []

    for i in range(1, len(df_slice)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - entry_bar
            if pos_dir == 1:
                tp_lvl = entry_px + TP_PIPS * pip
                if m5_h[i] >= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held, W_pips_for_trade))
                    pos_dir = 0; continue
            else:
                tp_lvl = entry_px - TP_PIPS * pip
                if m5_l[i] <= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held, W_pips_for_trade))
                    pos_dir = 0; continue
            if held >= T_s_bars:
                d = abs(m5_c[i] - entry_px) / pip
                if d <= W_pips_for_trade:
                    pnl = (m5_c[i] - entry_px) / pip * pos_dir
                    trades.append((pnl, 'scratch', held, W_pips_for_trade))
                    pos_dir = 0; continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i
            a = bv.atr_at(m5_ts[i-1], h1_ts, atr)
            if a is None:
                pos_dir = 0; continue   # skip entries without ATR
            atr_pips = a / pip
            W_pips_for_trade = k * atr_pips
    if pos_dir != 0:
        pnl = (m5_c[-1] - entry_px) / pip * pos_dir
        trades.append((pnl, 'end', len(df_slice) - 1 - entry_bar, W_pips_for_trade))
    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=['pips','reason','bars','W'])
    tdf['net'] = tdf['pips'] - spread_cost
    cum = tdf['net'].cumsum()
    return {
        'trades': len(tdf),
        'net': float(tdf['net'].sum()),
        'max_dd': float((cum - cum.cummax()).min()),
        'tp': int((tdf['reason']=='TP').sum()),
        'scratch': int((tdf['reason']=='scratch').sum()),
        'end': int((tdf['reason']=='end').sum()),
        'avg_W': float(tdf['W'].mean()),
        'wr': float((tdf['net']>0).mean()*100),
    }


def main():
    is_bars = int(bv.WINDOW_BARS * 4/6)
    T_s_hours = [6, 8, 12, 24, 48]
    K_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    rows = []
    for pair in bv.PAIRS:
        df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
        df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)
        for Th in T_s_hours:
            for k in K_VALUES:
                T_s = Th * 12
                r_is = run_atr_scratch(pair, df_is, T_s, k) or {}
                r_oos = run_atr_scratch(pair, df_oos, T_s, k) or {}
                rows.append({
                    'pair': pair, 'T_s_h': Th, 'k': k,
                    'is_net': r_is.get('net', 0),
                    'oos_net': r_oos.get('net', 0),
                    'oos_dd': r_oos.get('max_dd', 0),
                    'oos_trades': r_oos.get('trades', 0),
                    'avg_W': r_oos.get('avg_W', 0),
                    'oos_tp': r_oos.get('tp', 0),
                    'oos_scratch': r_oos.get('scratch', 0),
                    'oos_end': r_oos.get('end', 0),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'h7_atr_scratch.csv', index=False)

    print('\n=== H7 (ATR-scaled scratch W) best IS+OOS+ per pair ===')
    best = df[(df['is_net']>0) & (df['oos_net']>0)].sort_values('oos_net', ascending=False)
    bp = best.groupby('pair').head(1).sort_values('oos_net', ascending=False)
    if len(bp):
        for _, r in bp.iterrows():
            print(f"  {r['pair']:<8} T_s={int(r['T_s_h']):>2d}h k={r['k']:.1f} (avg_W={r['avg_W']:.1f}p) "
                  f"IS{r['is_net']:+.0f} OOS{r['oos_net']:+.0f} DD{r['oos_dd']:+.0f}")
        print(f"\n{len(bp)}/10 pairs IS+OOS+")
        print(f"Sum IS: {bp['is_net'].sum():+.0f}p OOS: {bp['oos_net'].sum():+.0f}p")
    else:
        print("  no IS+OOS+ configs found")

    # Compare to fixed-W scratch from earlier
    print('\n=== Compare to fixed-W scratch result (commit ac6f321) ===')
    fixed = {
        'USD_JPY': '24h/10p fixed: OOS+888', 'NZD_USD': '24h/5p fixed: OOS+674',
        'GBP_USD': '8h/15p fixed: OOS+778', 'EUR_USD': 'failed', 'AUD_USD': 'failed',
        'EUR_JPY': 'failed', 'GBP_JPY': 'failed', 'AUD_JPY': 'failed',
        'CAD_JPY': 'failed', 'EUR_GBP': 'failed',
    }
    for pair in bv.PAIRS:
        atr_best = bp[bp['pair']==pair]
        if len(atr_best):
            r = atr_best.iloc[0]
            print(f"  {pair:<8} ATR-W: OOS{r['oos_net']:+.0f}  |  fixed: {fixed[pair]}")
        else:
            print(f"  {pair:<8} ATR-W: no IS+OOS+ config  |  fixed: {fixed[pair]}")


if __name__ == '__main__':
    main()
