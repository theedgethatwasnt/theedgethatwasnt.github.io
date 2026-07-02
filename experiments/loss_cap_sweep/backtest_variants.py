#!/usr/bin/env python3
"""Sweep exit-rule variants on the SMA16 H1+M30 momentum strategy to
find configurations that cap losses while remaining net-positive
after spread cost.

The base strategy: SMA(16) lags=(8,10,15) 6-of-6 momentum agreement on
H1+M30. Entry on first M5 bar after signal fires; only one open
position per pair (no hedging - matches OANDA FIFO).

Variants tested (all otherwise identical):
  V0  TP=20 no SL                              (current live)
  V1  TP=20  SL=50p
  V2  TP=20  SL=100p
  V3  TP=20  time exit 48h
  V4  TP=20  time exit 48h  SL=100p
  V5  TP=20  time exit 24h
  V6  TP=20  trail: lock once +10p; trail offset 5p
  V7  TP=20  trail: lock once +5p;  trail offset 3p
  V8  TP=20  SL = 2x ATR(14) on H1
  V9  TP=30  SL=10p (asymmetric small)
  V10 TP=20  SL=20p (symmetric)
  V11 TP=20  time exit 12h
  V12 TP=20  SL=200p (catastrophe-only)

For each variant, report: trades, gross pips, net pips (after spread),
WR, profit-factor, max-DD, sharpe-of-trade-series.

Send Telegram alert if ANY variant produces net-positive pips/day
AND a bounded max-drawdown (< 200p per pair, < 400p portfolio).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import json

PROJECT = Path("/path/to/projects/fx-core")
DATA = PROJECT / "data" / "m5_ohlc"
OUT_DIR = PROJECT / "research" / "experiments" / "loss_cap_sweep" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMA_N = 16
LAGS = (8, 10, 15)
WINDOW_BARS = 6 * 30 * 24 * 12   # ~6 months M5
PAIRS = {
    "USD_JPY": (0.01,   2.10), "EUR_JPY": (0.01,   2.50),
    "GBP_JPY": (0.01,   4.00), "AUD_JPY": (0.01,   2.30),
    "EUR_USD": (0.0001, 1.70), "GBP_USD": (0.0001, 2.40),
    "CAD_JPY": (0.01,   2.60), "AUD_USD": (0.0001, 1.60),
    "EUR_GBP": (0.0001, 2.00), "NZD_USD": (0.0001, 2.00),
}
SPREAD_FRAC = 0.6                # typical entered spread / IS-P90 gate
BARS_PER_HOUR_M5 = 12


def sma(arr, n):
    out = np.full(len(arr), np.nan)
    if len(arr) < n: return out
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out[n-1:] = (cs[n:] - cs[:-n]) / n
    return out


def momentum_sig(closes, lags):
    s = sma(closes, SMA_N)
    sig = np.zeros(len(closes), dtype=np.int8)
    for i in range(max(lags) + SMA_N, len(closes)):
        if np.isnan(s[i]):
            continue
        ups = sum(1 for lg in lags if not np.isnan(s[i-lg]) and s[i] > s[i-lg])
        dns = sum(1 for lg in lags if not np.isnan(s[i-lg]) and s[i] < s[i-lg])
        if ups == len(lags): sig[i] = 1
        elif dns == len(lags): sig[i] = -1
    return sig


def resample_tf(df_m5, minutes):
    d = df_m5.set_index('timestamp')
    return d.resample(f'{minutes}min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()


def signal_at(t, h1_ts, h1_sig, m30_ts, m30_sig):
    h = np.searchsorted(h1_ts, t, side='right') - 1
    m = np.searchsorted(m30_ts, t, side='right') - 1
    if h < 0 or m < 0: return 0
    a, b = h1_sig[h], m30_sig[m]
    if a == 1 and b == 1: return 1
    if a == -1 and b == -1: return -1
    return 0


def atr_h1(h1_df, period=14):
    h = h1_df['high'].to_numpy()
    l = h1_df['low'].to_numpy()
    c = h1_df['close'].to_numpy()
    tr = np.zeros(len(h))
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.full(len(h), np.nan)
    if len(h) > period:
        a[period] = tr[1:period+1].mean()
        for i in range(period+1, len(h)):
            a[i] = (a[i-1]*(period-1) + tr[i])/period
    return a


def atr_at(t, h1_ts, atr_arr):
    idx = np.searchsorted(h1_ts, t, side='right') - 1
    if idx < 0 or idx >= len(atr_arr) or np.isnan(atr_arr[idx]):
        return None
    return atr_arr[idx]


# ── Variants ────────────────────────────────────────────────────────────────
def run_variant(pair, variant_id, variant_params):
    """Execute one variant on one pair. Returns metrics dict."""
    pip, sp_gate = PAIRS[pair]
    spread_cost = sp_gate * SPREAD_FRAC

    df = pq.read_table(DATA / f"{pair}_M5.parquet").to_pandas()
    df = df.tail(WINDOW_BARS).reset_index(drop=True)

    h1 = resample_tf(df, 60)
    m30 = resample_tf(df, 30)
    h1_sig = momentum_sig(h1['close'].to_numpy(), LAGS)
    m30_sig = momentum_sig(m30['close'].to_numpy(), LAGS)
    h1_ts = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()
    atr = atr_h1(h1)

    m5_ts = df['timestamp'].to_numpy()
    m5_o = df['open'].to_numpy()
    m5_h = df['high'].to_numpy()
    m5_l = df['low'].to_numpy()
    m5_c = df['close'].to_numpy()

    tp_pips = variant_params['tp_pips']
    sl_pips = variant_params.get('sl_pips')   # None => no SL
    sl_atr_mult = variant_params.get('sl_atr_mult')  # dynamic SL
    time_exit_bars = variant_params.get('time_exit_bars')
    trail_lock_pips = variant_params.get('trail_lock_pips')
    trail_offset_pips = variant_params.get('trail_offset_pips')

    pos_dir = 0
    pos_entry = 0.0
    pos_entry_bar = -1
    pos_sl = None        # absolute price
    pos_peak = None      # for trailing stop tracking
    pos_trail = None     # current trailing stop price
    trades = []

    for i in range(1, len(df)):
        sig = signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)

        if pos_dir != 0:
            bars_held = i - pos_entry_bar
            exit_px = None
            exit_reason = ''

            # 1) TP intrabar
            if pos_dir == 1:
                tp_level = pos_entry + tp_pips * pip
                if m5_h[i] >= tp_level:
                    exit_px = tp_level; exit_reason = 'TP'
            else:
                tp_level = pos_entry - tp_pips * pip
                if m5_l[i] <= tp_level:
                    exit_px = tp_level; exit_reason = 'TP'

            # 2) SL intrabar
            if exit_px is None and pos_sl is not None:
                if pos_dir == 1:
                    if m5_l[i] <= pos_sl:
                        exit_px = pos_sl; exit_reason = 'SL'
                else:
                    if m5_h[i] >= pos_sl:
                        exit_px = pos_sl; exit_reason = 'SL'

            # 3) trailing stop
            if exit_px is None and trail_lock_pips is not None:
                # Track peak favorable price
                if pos_dir == 1:
                    high_now = m5_h[i]
                    if pos_peak is None or high_now > pos_peak:
                        pos_peak = high_now
                    # Activate trail once peak >= entry + lock_pips
                    if pos_peak >= pos_entry + trail_lock_pips * pip:
                        candidate_trail = pos_peak - trail_offset_pips * pip
                        if pos_trail is None or candidate_trail > pos_trail:
                            pos_trail = candidate_trail
                    if pos_trail is not None and m5_l[i] <= pos_trail:
                        exit_px = pos_trail; exit_reason = 'trail'
                else:
                    low_now = m5_l[i]
                    if pos_peak is None or low_now < pos_peak:
                        pos_peak = low_now
                    if pos_peak <= pos_entry - trail_lock_pips * pip:
                        candidate_trail = pos_peak + trail_offset_pips * pip
                        if pos_trail is None or candidate_trail < pos_trail:
                            pos_trail = candidate_trail
                    if pos_trail is not None and m5_h[i] >= pos_trail:
                        exit_px = pos_trail; exit_reason = 'trail'

            # 4) time exit
            if exit_px is None and time_exit_bars is not None and bars_held >= time_exit_bars:
                exit_px = m5_o[i]; exit_reason = 'time'

            if exit_px is not None:
                pnl_pips = (exit_px - pos_entry) / pip * pos_dir
                trades.append((pair, pos_dir, pos_entry, exit_px,
                               pnl_pips, exit_reason, bars_held))
                pos_dir = 0; pos_entry = 0.0
                pos_sl = None; pos_peak = None; pos_trail = None
                continue

        if pos_dir == 0 and sig != 0:
            pos_dir = sig
            pos_entry = m5_o[i]
            pos_entry_bar = i
            pos_peak = pos_entry
            pos_trail = None
            # Set SL
            if sl_pips is not None:
                pos_sl = pos_entry - sig * sl_pips * pip
            elif sl_atr_mult is not None:
                a = atr_at(m5_ts[i-1], h1_ts, atr)
                if a is not None:
                    pos_sl = pos_entry - sig * sl_atr_mult * a
                else:
                    pos_sl = None
            else:
                pos_sl = None

    # Force-close at end
    if pos_dir != 0:
        exit_px = m5_c[-1]
        pnl_pips = (exit_px - pos_entry) / pip * pos_dir
        trades.append((pair, pos_dir, pos_entry, exit_px,
                       pnl_pips, 'end', len(df) - pos_entry_bar))

    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=['pair','dir','entry','exit','pips','reason','bars'])
    tdf['pips_net'] = tdf['pips'] - spread_cost
    cum = tdf['pips_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()  # most negative
    wins = tdf['pips_net'][tdf['pips_net'] > 0].sum()
    losses = -tdf['pips_net'][tdf['pips_net'] < 0].sum()
    pf = wins / losses if losses > 0 else float('inf')
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()/86400
    return {
        'pair': pair, 'variant': variant_id,
        'trades': len(tdf),
        'gross': float(tdf['pips'].sum()),
        'net': float(tdf['pips_net'].sum()),
        'pd_net': float(tdf['pips_net'].sum() / days),
        'wr_net': float((tdf['pips_net'] > 0).mean() * 100),
        'avg_win': float(tdf[tdf['pips_net']>0]['pips_net'].mean() or 0),
        'avg_loss': float(tdf[tdf['pips_net']<0]['pips_net'].mean() or 0),
        'pf': float(pf) if pf != float('inf') else 999.0,
        'max_dd': float(max_dd),
        'days': days,
        'reasons': tdf['reason'].value_counts().to_dict(),
    }


VARIANTS = {
    'V0_baseline':              {'tp_pips': 20.0},
    'V1_sl50':                  {'tp_pips': 20.0, 'sl_pips': 50.0},
    'V2_sl100':                 {'tp_pips': 20.0, 'sl_pips': 100.0},
    'V3_time48h':               {'tp_pips': 20.0, 'time_exit_bars': 48*BARS_PER_HOUR_M5},
    'V4_time48h_sl100':         {'tp_pips': 20.0, 'time_exit_bars': 48*BARS_PER_HOUR_M5, 'sl_pips': 100.0},
    'V5_time24h':               {'tp_pips': 20.0, 'time_exit_bars': 24*BARS_PER_HOUR_M5},
    'V6_trail_lock10':          {'tp_pips': 20.0, 'trail_lock_pips': 10.0, 'trail_offset_pips': 5.0},
    'V7_trail_lock5':           {'tp_pips': 20.0, 'trail_lock_pips': 5.0,  'trail_offset_pips': 3.0},
    'V8_sl_2xATR':              {'tp_pips': 20.0, 'sl_atr_mult': 2.0},
    'V9_asym_tp30_sl10':        {'tp_pips': 30.0, 'sl_pips': 10.0},
    'V10_sym_tp20_sl20':        {'tp_pips': 20.0, 'sl_pips': 20.0},
    'V11_time12h':              {'tp_pips': 20.0, 'time_exit_bars': 12*BARS_PER_HOUR_M5},
    'V12_sl200_catastrophe':    {'tp_pips': 20.0, 'sl_pips': 200.0},
    'V13_time12h_sl100':        {'tp_pips': 20.0, 'time_exit_bars': 12*BARS_PER_HOUR_M5, 'sl_pips': 100.0},
    'V14_time6h':               {'tp_pips': 20.0, 'time_exit_bars': 6*BARS_PER_HOUR_M5},
    'V15_time6h_sl50':          {'tp_pips': 20.0, 'time_exit_bars': 6*BARS_PER_HOUR_M5, 'sl_pips': 50.0},
    'V16_trail_lock15_off8':    {'tp_pips': 20.0, 'trail_lock_pips': 15.0, 'trail_offset_pips': 8.0},
    'V17_trail_lock5_off2':     {'tp_pips': 20.0, 'trail_lock_pips': 5.0,  'trail_offset_pips': 2.0},
    'V18_sl_3xATR':             {'tp_pips': 20.0, 'sl_atr_mult': 3.0},
}


def telegram_alert(msg: str):
    tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not (tok and chat):
        print(f"[no telegram creds] would send: {msg[:200]}")
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={'chat_id': chat, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"telegram error: {e}")


def main():
    print(f"Sweeping {len(VARIANTS)} variants × {len(PAIRS)} pairs "
          f"on {WINDOW_BARS} M5 bars (~6 months)\n")

    results = []
    portfolio = {}
    for vid, vparams in VARIANTS.items():
        pair_results = []
        for pair in PAIRS:
            r = run_variant(pair, vid, vparams)
            if r:
                pair_results.append(r)
                results.append(r)
        if pair_results:
            net_total = sum(r['net'] for r in pair_results)
            gross_total = sum(r['gross'] for r in pair_results)
            trades = sum(r['trades'] for r in pair_results)
            wr_overall = sum(r['wr_net'] * r['trades'] for r in pair_results) / max(trades, 1)
            max_pair_dd = min(r['max_dd'] for r in pair_results)
            portfolio_dd_proxy = sum(r['max_dd'] for r in pair_results)
            days = pair_results[0]['days']
            pos_pairs = sum(1 for r in pair_results if r['net'] > 0)
            portfolio[vid] = {
                'gross': gross_total, 'net': net_total,
                'pd_net': net_total / days, 'trades': trades,
                'wr': wr_overall, 'max_pair_dd': max_pair_dd,
                'portfolio_dd_proxy': portfolio_dd_proxy,
                'pos_pairs': pos_pairs,
            }

    # Print portfolio summary, sorted by net pips/day descending
    print(f"\n{'Variant':<25s}{'Trades':>8s}{'Gross':>10s}{'Net':>10s}"
          f"{'p/d':>9s}{'WR%':>6s}{'PosPrs':>7s}{'WorstDD':>10s}{'SumDD':>10s}")
    print('-' * 95)
    sorted_v = sorted(portfolio.items(), key=lambda kv: -kv[1]['pd_net'])
    interesting = []
    for vid, p in sorted_v:
        marker = ''
        if p['net'] > 0 and p['pd_net'] > 0:
            marker = '  ★ NET-POSITIVE'
            interesting.append((vid, p))
        if p['max_pair_dd'] > -150 and p['net'] > -500:
            marker += '  [BOUNDED-DD]'
        print(f"{vid:<25s}{p['trades']:>8d}{p['gross']:>+10.1f}{p['net']:>+10.1f}"
              f"{p['pd_net']:>+9.2f}{p['wr']:>5.1f}%{p['pos_pairs']:>7d}/10"
              f"{p['max_pair_dd']:>+10.1f}{p['portfolio_dd_proxy']:>+10.1f}{marker}")

    # Save detailed results
    out_csv = OUT_DIR / 'all_results.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nDetailed per-pair results: {out_csv}")

    # Telegram alert if anything looks interesting
    if interesting:
        lines = ['🔬 Loss-cap sweep found net-positive variants:']
        for vid, p in interesting[:5]:
            lines.append(
                f"  {vid}: net={p['net']:+.0f}p ({p['pd_net']:+.2f} p/d) "
                f"WR={p['wr']:.0f}% {p['pos_pairs']}/10 pairs+ "
                f"worst pair DD={p['max_pair_dd']:+.0f}p"
            )
        telegram_alert('\n'.join(lines))
    else:
        print("\nNo net-positive variant found. Telegram skipped.")


if __name__ == '__main__':
    main()
