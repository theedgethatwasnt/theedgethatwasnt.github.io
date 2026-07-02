"""Composite rule backtest: SMA16 + scratch-exit + early-MFE-filter.

The two rules complement each other:
  - early-MFE-filter cuts duds at hour T_q if MFE_so_far < X
  - scratch exits meandering positions at hour T_s if |price - entry| < W

Test 5x4x6x5 = 600 combinations per pair, then per-pair IS/OOS.

Look for configs where the COMPOSITE beats scratch-alone on a per-pair
basis, AND see if any of the 7 currently-bad pairs become positive.
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


def run_composite(pair, df_slice, T_q_bars, X_pips, T_s_bars, W_pips):
    """Run SMA16 + early-MFE-filter + scratch-exit on a data slice.
       T_q_bars   - quality-check time (bars from entry)
       X_pips     - minimum MFE required at quality-check (else exit)
       T_s_bars   - scratch-activation time
       W_pips     - scratch window around entry
       Set T_q_bars=None or X_pips=None to disable quality filter.
       Set T_s_bars=None or W_pips=None to disable scratch."""
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
    mfe = 0.0
    trades = []
    use_q = (T_q_bars is not None) and (X_pips is not None)
    use_s = (T_s_bars is not None) and (W_pips is not None)

    for i in range(1, len(df_slice)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - entry_bar
            high_pip = (m5_h[i] - entry_px) / pip * pos_dir
            if high_pip > mfe:
                mfe = high_pip
            # 1) TP
            if pos_dir == 1:
                tp_lvl = entry_px + TP_PIPS * pip
                if m5_h[i] >= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; mfe = 0.0; continue
            else:
                tp_lvl = entry_px - TP_PIPS * pip
                if m5_l[i] <= tp_lvl:
                    trades.append((TP_PIPS, 'TP', held))
                    pos_dir = 0; mfe = 0.0; continue
            # 2) quality filter: at exactly T_q_bars, if MFE < X, exit at close
            if use_q and held == T_q_bars and mfe < X_pips:
                pnl = (m5_c[i] - entry_px) / pip * pos_dir
                trades.append((pnl, 'quality', held))
                pos_dir = 0; mfe = 0.0; continue
            # 3) scratch
            if use_s and held >= T_s_bars:
                d = abs(m5_c[i] - entry_px) / pip
                if d <= W_pips:
                    pnl = (m5_c[i] - entry_px) / pip * pos_dir
                    trades.append((pnl, 'scratch', held))
                    pos_dir = 0; mfe = 0.0; continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i; mfe = 0.0
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
        'wr': float((tdf['net'] > 0).mean() * 100),
        'max_dd': float((cum - cum.cummax()).min()),
        'tp': int((tdf['reason']=='TP').sum()),
        'quality': int((tdf['reason']=='quality').sum()),
        'scratch': int((tdf['reason']=='scratch').sum()),
        'end': int((tdf['reason']=='end').sum()),
        'tp_pips': float(tdf[tdf['reason']=='TP']['net'].sum()),
        'q_pips': float(tdf[tdf['reason']=='quality']['net'].sum()),
        's_pips': float(tdf[tdf['reason']=='scratch']['net'].sum()),
        'e_pips': float(tdf[tdf['reason']=='end']['net'].sum()),
    }


def main():
    is_bars = int(bv.WINDOW_BARS * 4/6)
    # Coarse grid for the composite rule.
    T_q_hours = [None, 0.5, 1, 2, 4]   # None = no quality filter
    X_pips    = [None, 2, 3, 5, 8]
    T_s_hours = [None, 6, 8, 12, 24]   # None = no scratch
    W_pips    = [None, 5, 10, 15]
    # Generate combinations (skip None mismatches)
    grid = []
    for Th in T_q_hours:
        for X in X_pips:
            if (Th is None) != (X is None): continue
            for Ts in T_s_hours:
                for W in W_pips:
                    if (Ts is None) != (W is None): continue
                    grid.append((Th, X, Ts, W))
    print(f'Grid size: {len(grid)} configs × {len(bv.PAIRS)} pairs')

    rows = []
    for pair in bv.PAIRS:
        df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
        df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)
        for (Th, X, Ts, W) in grid:
            T_q = int(Th * 12) if Th else None
            T_s = int(Ts * 12) if Ts else None
            r_is = run_composite(pair, df_is, T_q, X, T_s, W) or {}
            r_oos = run_composite(pair, df_oos, T_q, X, T_s, W) or {}
            rows.append({
                'pair': pair, 'T_q_h': Th, 'X_pips': X,
                'T_s_h': Ts, 'W_pips': W,
                'is_net': r_is.get('net', 0), 'is_dd': r_is.get('max_dd', 0),
                'oos_net': r_oos.get('net', 0), 'oos_dd': r_oos.get('max_dd', 0),
                'oos_trades': r_oos.get('trades', 0),
                'oos_tp': r_oos.get('tp', 0), 'oos_q': r_oos.get('quality', 0),
                'oos_s': r_oos.get('scratch', 0), 'oos_end': r_oos.get('end', 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'composite_rule_per_pair.csv', index=False)

    # For each pair, find configs where BOTH IS and OOS are positive
    print('\n=== Best IS+OOS+ composite config per pair (require both positive) ===')
    print(f"{'Pair':<8} {'T_q_h':>6} {'X':>3} {'T_s_h':>6} {'W':>3} "
          f"{'IS':>9} {'OOS':>9} {'OOS_DD':>9} {'Tr':>4} {'TP/Q/S/E':>14}")
    best_per_pair = []
    for pair in bv.PAIRS:
        sub = df[(df['pair']==pair) & (df['is_net']>0) & (df['oos_net']>0)]
        if not len(sub):
            print(f"{pair:<8} (no IS+OOS+ composite config)")
            continue
        b = sub.sort_values('oos_net', ascending=False).iloc[0]
        best_per_pair.append(b)
        tq = b['T_q_h']; X = b['X_pips']; ts = b['T_s_h']; W = b['W_pips']
        reasons = f"{int(b['oos_tp'])}/{int(b['oos_q'])}/{int(b['oos_s'])}/{int(b['oos_end'])}"
        print(f"{pair:<8} {str(tq):>6} {str(X):>3} {str(ts):>6} {str(W):>3} "
              f"{b['is_net']:>+9.0f} {b['oos_net']:>+9.0f} {b['oos_dd']:>+9.0f} "
              f"{int(b['oos_trades']):>4d} {reasons:>14s}")

    # Aggregate portfolio if we used best-per-pair config
    if best_per_pair:
        bdf = pd.DataFrame(best_per_pair)
        tot_is = bdf['is_net'].sum()
        tot_oos = bdf['oos_net'].sum()
        wp = (bdf['oos_net'] > 0).sum()
        print(f"\nPortfolio (best-per-pair IS+OOS+): {len(bdf)} pairs, "
              f"IS={tot_is:+.0f}p OOS={tot_oos:+.0f}p worst-DD={bdf['oos_dd'].min():+.0f}p")

        # Compare to scratch-alone baseline for the same 3 pairs we know
        # (USD_JPY 24h/10p, NZD_USD 24h/5p, GBP_USD 8h/15p)
        baseline_three = {
            'USD_JPY': (None, None, 24.0, 10.0),
            'NZD_USD': (None, None, 24.0, 5.0),
            'GBP_USD': (None, None, 8.0, 15.0),
        }
        print('\n=== Compare: scratch-alone vs composite for the 3 winning pairs ===')
        for p, (Th, X, Ts, W) in baseline_three.items():
            scr_row = df[(df['pair']==p) & (df['T_q_h'].isna()) & (df['X_pips'].isna())
                          & (df['T_s_h']==Ts) & (df['W_pips']==W)]
            if len(scr_row):
                scr = scr_row.iloc[0]
            else:
                scr = None
            best_row = bdf[bdf['pair']==p]
            if len(best_row):
                bp = best_row.iloc[0]
                print(f"{p}:")
                if scr is not None:
                    print(f"  scratch-alone : IS={scr['is_net']:+.0f}p OOS={scr['oos_net']:+.0f}p")
                print(f"  composite best: IS={bp['is_net']:+.0f}p OOS={bp['oos_net']:+.0f}p "
                      f"(T_q={bp['T_q_h']}h X={bp['X_pips']}p T_s={bp['T_s_h']}h W={bp['W_pips']}p)")

        # Telegram alert if portfolio significantly improves
        msg_lines = ['🔬 Composite rule (early-MFE + scratch) results:']
        msg_lines.append(f"IS+OOS-positive on {len(bdf)} of 10 pairs")
        msg_lines.append(f"3-pair best-per-pair: IS={tot_is:+.0f}p OOS={tot_oos:+.0f}p "
                         f"worst-DD={bdf['oos_dd'].min():+.0f}p")
        msg_lines.append('')
        msg_lines.append('Per-pair winning configs:')
        for _, r in bdf.iterrows():
            msg_lines.append(
                f"  {r['pair']}: T_q={r['T_q_h']}h X={r['X_pips']}p "
                f"T_s={r['T_s_h']}h W={r['W_pips']}p  IS{r['is_net']:+.0f}/OOS{r['oos_net']:+.0f}"
            )
        msg = '\n'.join(msg_lines)
        try:
            tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
            chat = os.environ.get('TELEGRAM_CHAT_ID', '')
            if tok and chat:
                requests.post(f'https://api.telegram.org/bot{tok}/sendMessage',
                              json={'chat_id': chat, 'text': msg}, timeout=10)
                print('\n[telegram sent]')
        except Exception:
            pass
    else:
        print('\nNo IS+OOS+ composite config found on any pair.')


if __name__ == '__main__':
    main()
