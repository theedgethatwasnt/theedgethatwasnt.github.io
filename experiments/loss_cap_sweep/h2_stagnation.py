"""H2: MFE-stagnation exit.

Rule: track bars-since-MFE-was-last-updated. If it exceeds threshold S,
exit at next M5 close. Apply on top of scratch overlay (or alone).

Tests all 10 pairs to find which pairs benefit. IS/OOS split.
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

# Stagnation threshold (bars since MFE last grew)
STAG_BARS = [12, 24, 48, 72, 144, 288]   # 1h, 2h, 4h, 6h, 12h, 24h

# Optionally combine with scratch
SCRATCH_OPTS = [
    (None, None),       # stagnation only
    (24*12, 10.0),      # USD_JPY-like scratch
    (24*12,  5.0),      # NZD_USD-like
    (8*12,  15.0),      # GBP_USD-like
]


def run_stag(pair, df_slice, S_bars, T_s_bars=None, W_pips=None):
    pip, sp_gate = bv.PAIRS[pair]
    spread_cost = sp_gate * bv.SPREAD_FRAC

    h1 = bv.resample_tf(df_slice, 60); m30 = bv.resample_tf(df_slice, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy(); m30_ts = m30['timestamp'].to_numpy()
    m5_ts = df_slice['timestamp'].to_numpy()
    m5_o = df_slice['open'].to_numpy(); m5_h = df_slice['high'].to_numpy()
    m5_l = df_slice['low'].to_numpy(); m5_c = df_slice['close'].to_numpy()

    pos_dir = 0; entry_px = 0.0; entry_bar = -1
    mfe = 0.0; mfe_bar = -1
    trades = []
    use_scratch = (T_s_bars is not None) and (W_pips is not None)

    for i in range(1, len(df_slice)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - entry_bar
            high_pip = (m5_h[i] - entry_px) / pip * pos_dir
            if high_pip > mfe:
                mfe = high_pip; mfe_bar = i
            # TP intrabar
            if pos_dir == 1:
                tp_lvl = entry_px + TP_PIPS * pip
                if m5_h[i] >= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; mfe = 0.0; mfe_bar = -1; continue
            else:
                tp_lvl = entry_px - TP_PIPS * pip
                if m5_l[i] <= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; mfe = 0.0; mfe_bar = -1; continue
            # Stagnation exit: bars since MFE last grew > S, AND MFE > 0
            # (only exit if there WAS some progress; if MFE never grew,
            # leave it to scratch / TP / end)
            if mfe > 0 and (i - mfe_bar) > S_bars:
                pnl = (m5_c[i] - entry_px) / pip * pos_dir
                trades.append((pnl, 'stag', held))
                pos_dir = 0; mfe = 0.0; mfe_bar = -1; continue
            # Scratch
            if use_scratch and held >= T_s_bars:
                d = abs(m5_c[i] - entry_px) / pip
                if d <= W_pips:
                    pnl = (m5_c[i] - entry_px) / pip * pos_dir
                    trades.append((pnl, 'scratch', held))
                    pos_dir = 0; mfe = 0.0; mfe_bar = -1; continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i
            mfe = 0.0; mfe_bar = i
    if pos_dir != 0:
        pnl = (m5_c[-1] - entry_px) / pip * pos_dir
        trades.append((pnl, 'end', len(df_slice) - 1 - entry_bar))
    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=['pips','reason','bars'])
    tdf['net'] = tdf['pips'] - spread_cost
    cum = tdf['net'].cumsum()
    return {
        'trades': len(tdf),
        'net': float(tdf['net'].sum()),
        'max_dd': float((cum - cum.cummax()).min()),
        'tp': int((tdf['reason']=='TP').sum()),
        'stag': int((tdf['reason']=='stag').sum()),
        'scratch': int((tdf['reason']=='scratch').sum()),
        'end': int((tdf['reason']=='end').sum()),
        'wr': float((tdf['net']>0).mean()*100),
    }


def main():
    is_bars = int(bv.WINDOW_BARS * 4/6)
    rows = []
    for pair in bv.PAIRS:
        df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
        df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)
        for S in STAG_BARS:
            for (Ts, W) in SCRATCH_OPTS:
                r_is = run_stag(pair, df_is, S, Ts, W) or {}
                r_oos = run_stag(pair, df_oos, S, Ts, W) or {}
                rows.append({
                    'pair': pair, 'S_bars': S,
                    'T_s_bars': Ts, 'W_pips': W,
                    'is_net': r_is.get('net', 0),
                    'oos_net': r_oos.get('net', 0),
                    'oos_dd': r_oos.get('max_dd', 0),
                    'oos_trades': r_oos.get('trades', 0),
                    'oos_tp': r_oos.get('tp', 0),
                    'oos_stag': r_oos.get('stag', 0),
                    'oos_scratch': r_oos.get('scratch', 0),
                    'oos_end': r_oos.get('end', 0),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'h2_stagnation.csv', index=False)

    # Best IS+OOS+ per pair
    print('\n=== H2 (MFE-stagnation) best IS+OOS+ per pair ===')
    print(f"{'Pair':<8} {'S_h':>4} {'T_s':>5} {'W':>4} {'IS':>9} {'OOS':>9} {'OOS_DD':>9} "
          f"{'Tr':>4} {'TP/St/Sc/E':>14}")
    print('-' * 78)
    best = df[(df['is_net']>0) & (df['oos_net']>0)].sort_values('oos_net', ascending=False)
    if len(best):
        for pair in bv.PAIRS:
            sub = best[best['pair']==pair].head(3)
            for _, r in sub.iterrows():
                ts = f"{r['T_s_bars']/12:.0f}h" if r['T_s_bars'] is not None and not pd.isna(r['T_s_bars']) else '  -'
                w = f"{r['W_pips']:.0f}p" if r['W_pips'] is not None and not pd.isna(r['W_pips']) else '  -'
                reasons = f"{int(r['oos_tp'])}/{int(r['oos_stag'])}/{int(r['oos_scratch'])}/{int(r['oos_end'])}"
                print(f"{pair:<8} {int(r['S_bars']/12):>3d}h {ts:>5s} {w:>4s} "
                      f"{r['is_net']:>+9.0f} {r['oos_net']:>+9.0f} {r['oos_dd']:>+9.0f} "
                      f"{int(r['oos_trades']):>4d} {reasons:>14s}")
    # Per-pair best
    print('\n=== Best IS+OOS+ per pair (ranked by OOS) ===')
    bp = best.groupby('pair').head(1).sort_values('oos_net', ascending=False)
    for _, r in bp.iterrows():
        ts = f"{r['T_s_bars']/12:.0f}h" if r['T_s_bars'] is not None and not pd.isna(r['T_s_bars']) else 'none'
        w = f"{r['W_pips']:.0f}p" if r['W_pips'] is not None and not pd.isna(r['W_pips']) else 'none'
        print(f"  {r['pair']:<8} S={int(r['S_bars']/12)}h scratch=({ts},{w})  "
              f"IS{r['is_net']:+.0f} OOS{r['oos_net']:+.0f} DD{r['oos_dd']:+.0f}")
    print(f"\nTotal IS: {bp['is_net'].sum():+.0f}p  OOS: {bp['oos_net'].sum():+.0f}p")
    print(f"Worst pair DD: {bp['oos_dd'].min():+.0f}p")


if __name__ == '__main__':
    main()
