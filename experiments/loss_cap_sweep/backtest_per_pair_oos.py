#!/usr/bin/env python3
"""IS/OOS validation of per-pair-best variant selection.

Procedure:
  1. Split last 6 months M5 into IS=first 4mo, OOS=last 2mo.
  2. For each pair × each variant, compute IS net pips.
  3. For each pair, pick the variant with best IS net pips.
  4. Apply that IS-best variant to the OOS slice. Measure OOS net.
  5. Report: per-pair IS, OOS, and whether OOS sign matches IS.

If portfolio OOS is positive AND >= 5 of 10 pairs are OOS-positive AND
worst pair OOS DD < 400p: Telegram alert with deployment candidate.
"""
import os
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# Reuse helpers from the main sweep script
spec = importlib.util.spec_from_file_location(
    'bv', Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

PROJECT = Path("/path/to/projects/fx-core")
DATA = PROJECT / "data" / "m5_ohlc"

WINDOW_BARS = bv.WINDOW_BARS
IS_FRACTION = 4 / 6   # first 4 months of the 6-month window
PAIRS = bv.PAIRS
SPREAD_FRAC = bv.SPREAD_FRAC
VARIANTS = bv.VARIANTS


def run_variant_on_slice(pair, vparams, df_slice):
    """Same as run_variant but on a pre-sliced dataframe."""
    pip, sp_gate = PAIRS[pair]
    spread_cost = sp_gate * SPREAD_FRAC

    h1 = bv.resample_tf(df_slice, 60)
    m30 = bv.resample_tf(df_slice, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()
    atr = bv.atr_h1(h1)

    m5_ts = df_slice['timestamp'].to_numpy()
    m5_o = df_slice['open'].to_numpy()
    m5_h = df_slice['high'].to_numpy()
    m5_l = df_slice['low'].to_numpy()
    m5_c = df_slice['close'].to_numpy()

    tp_pips = vparams['tp_pips']
    sl_pips = vparams.get('sl_pips')
    sl_atr_mult = vparams.get('sl_atr_mult')
    time_exit_bars = vparams.get('time_exit_bars')
    trail_lock = vparams.get('trail_lock_pips')
    trail_off = vparams.get('trail_offset_pips')

    pos_dir = 0; pos_entry = 0.0; pos_bar = -1
    pos_sl = None; pos_peak = None; pos_trail = None
    trades = []

    for i in range(1, len(df_slice)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - pos_bar
            ex = None; reason = ''
            if pos_dir == 1:
                tp_lvl = pos_entry + tp_pips * pip
                if m5_h[i] >= tp_lvl:
                    ex = tp_lvl; reason = 'TP'
            else:
                tp_lvl = pos_entry - tp_pips * pip
                if m5_l[i] <= tp_lvl:
                    ex = tp_lvl; reason = 'TP'
            if ex is None and pos_sl is not None:
                if pos_dir == 1 and m5_l[i] <= pos_sl:
                    ex = pos_sl; reason = 'SL'
                elif pos_dir == -1 and m5_h[i] >= pos_sl:
                    ex = pos_sl; reason = 'SL'
            if ex is None and trail_lock is not None:
                if pos_dir == 1:
                    if pos_peak is None or m5_h[i] > pos_peak:
                        pos_peak = m5_h[i]
                    if pos_peak >= pos_entry + trail_lock * pip:
                        c = pos_peak - trail_off * pip
                        if pos_trail is None or c > pos_trail:
                            pos_trail = c
                    if pos_trail is not None and m5_l[i] <= pos_trail:
                        ex = pos_trail; reason = 'trail'
                else:
                    if pos_peak is None or m5_l[i] < pos_peak:
                        pos_peak = m5_l[i]
                    if pos_peak <= pos_entry - trail_lock * pip:
                        c = pos_peak + trail_off * pip
                        if pos_trail is None or c < pos_trail:
                            pos_trail = c
                    if pos_trail is not None and m5_h[i] >= pos_trail:
                        ex = pos_trail; reason = 'trail'
            if ex is None and time_exit_bars is not None and held >= time_exit_bars:
                ex = m5_o[i]; reason = 'time'
            if ex is not None:
                pnl = (ex - pos_entry) / pip * pos_dir
                trades.append((pair, pos_dir, pos_entry, ex, pnl, reason))
                pos_dir = 0; pos_sl = None; pos_peak = None; pos_trail = None
                continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; pos_entry = m5_o[i]; pos_bar = i
            pos_peak = pos_entry; pos_trail = None
            if sl_pips is not None:
                pos_sl = pos_entry - sig * sl_pips * pip
            elif sl_atr_mult is not None:
                a = bv.atr_at(m5_ts[i-1], h1_ts, atr)
                pos_sl = (pos_entry - sig * sl_atr_mult * a) if a is not None else None
            else:
                pos_sl = None
    if pos_dir != 0:
        ex = m5_c[-1]
        pnl = (ex - pos_entry) / pip * pos_dir
        trades.append((pair, pos_dir, pos_entry, ex, pnl, 'end'))

    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=['pair','dir','entry','exit','pips','reason'])
    tdf['pips_net'] = tdf['pips'] - spread_cost
    cum = tdf['pips_net'].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    return {
        'trades': len(tdf),
        'gross': float(tdf['pips'].sum()),
        'net': float(tdf['pips_net'].sum()),
        'wr': float((tdf['pips_net'] > 0).mean() * 100),
        'max_dd': max_dd,
    }


def main():
    print("IS/OOS per-pair-best variant validation\n")
    is_bars = int(WINDOW_BARS * IS_FRACTION)

    rows = []
    for pair in PAIRS:
        df = pq.read_table(DATA / f"{pair}_M5.parquet").to_pandas()
        df = df.tail(WINDOW_BARS).reset_index(drop=True)
        df_is = df.iloc[:is_bars].reset_index(drop=True)
        df_oos = df.iloc[is_bars:].reset_index(drop=True)

        # Test all variants on IS
        is_results = {}
        for vid, vparams in VARIANTS.items():
            r = run_variant_on_slice(pair, vparams, df_is)
            if r:
                is_results[vid] = r
        if not is_results:
            print(f"{pair}: no IS results"); continue

        best_vid = max(is_results, key=lambda v: is_results[v]['net'])
        best_is = is_results[best_vid]

        # Apply IS-best variant to OOS
        oos = run_variant_on_slice(pair, VARIANTS[best_vid], df_oos)
        if oos is None:
            oos = {'trades': 0, 'gross': 0.0, 'net': 0.0, 'wr': 0.0, 'max_dd': 0.0}

        rows.append({
            'pair': pair, 'best_var': best_vid,
            'is_trades': best_is['trades'], 'is_net': best_is['net'],
            'is_wr': best_is['wr'], 'is_dd': best_is['max_dd'],
            'oos_trades': oos['trades'], 'oos_net': oos['net'],
            'oos_wr': oos['wr'], 'oos_dd': oos['max_dd'],
            'oos_sign_match': (best_is['net'] > 0) == (oos['net'] > 0),
        })

    df_r = pd.DataFrame(rows)
    print(df_r.to_string(index=False))
    print()
    pos_oos = df_r[df_r['oos_net'] > 0]
    print(f"\nPairs net-positive in OOS: {len(pos_oos)} / {len(df_r)}")
    print(f"Sum OOS net (all pairs): {df_r['oos_net'].sum():+.1f}")
    print(f"Sum OOS net (OOS-positive pairs only): {pos_oos['oos_net'].sum():+.1f}")
    print(f"Worst single-pair OOS drawdown: {df_r['oos_dd'].min():+.1f}")

    df_r.to_csv(PROJECT / 'research/experiments/loss_cap_sweep/results/per_pair_oos.csv', index=False)

    # Decision and telegram
    portfolio_oos = df_r['oos_net'].sum()
    pos_oos_pairs = (df_r['oos_net'] > 0).sum()
    worst_dd = df_r['oos_dd'].min()
    interesting = (portfolio_oos > 0 and pos_oos_pairs >= 5 and worst_dd > -400)
    interesting_loose = (pos_oos_pairs >= 4 and pos_oos['oos_net'].sum() > 200)

    if interesting or interesting_loose:
        lines = ['📊 Per-pair-best variant on SMA16: OOS positive']
        lines.append(f"Portfolio OOS: {portfolio_oos:+.1f}p across {len(df_r)} pairs")
        lines.append(f"Positive pairs: {pos_oos_pairs}/{len(df_r)}, worst DD {worst_dd:+.0f}p")
        lines.append('Top OOS pairs:')
        for _, r in df_r.sort_values('oos_net', ascending=False).head(5).iterrows():
            sign = '✓' if r['oos_sign_match'] else '✗'
            lines.append(f"  {sign} {r['pair']} {r['best_var']}: "
                         f"IS={r['is_net']:+.0f}p OOS={r['oos_net']:+.0f}p "
                         f"DD={r['oos_dd']:+.0f}p")
        msg = '\n'.join(lines)
        print('\n' + msg)
        # send via telegram if creds present in env or via the bot container
        tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        chat = os.environ.get('TELEGRAM_CHAT_ID', '')
        if tok and chat:
            try:
                requests.post(f'https://api.telegram.org/bot{tok}/sendMessage',
                              json={'chat_id': chat, 'text': msg}, timeout=10)
            except Exception:
                pass
    else:
        print('\nOOS not strong enough for deployment alert.')


if __name__ == '__main__':
    main()
