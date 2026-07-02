#!/usr/bin/env python3
"""Backtest comparison: current fx-sma-live (no FIFO handling) vs
FIFO-aware variant that closes opposing position before opening new.

Strategy params (matched to live config):
  SMA(16) on H1 + M30, lags = (8, 10, 15)
  LONG  when all 6 momentum readings > 0 (3 H1 + 3 M30)
  SHORT when all 6 < 0
  TP    = 20 pips per trade
  No stop-loss

Two variants:
  A. CURRENT  -- position lives until TP hits; opposing signals are
                 ignored (matches OANDA-FIFO-rejected real behavior).
  B. FIFOAWARE -- when signal flips while position is open, close at
                  current price and open the opposite.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path("/path/to/projects/fx-core")
DATA_DIR = PROJECT / "data" / "m5_ohlc"

SMA_N = 16
LAGS = (8, 10, 15)
TP_PIPS = 20.0
# pip_size, IS-P90 spread gate, typical-entered spread (0.6 of gate)
PAIRS = {
    "USD_JPY": (0.01,   2.10), "EUR_JPY": (0.01,   2.50),
    "GBP_JPY": (0.01,   4.00), "AUD_JPY": (0.01,   2.30),
    "EUR_USD": (0.0001, 1.70), "GBP_USD": (0.0001, 2.40),
    "CAD_JPY": (0.01,   2.60), "AUD_USD": (0.0001, 1.60),
    "EUR_GBP": (0.0001, 2.00), "NZD_USD": (0.0001, 2.00),
}
WINDOW_BARS = 6 * 30 * 24 * 12  # ~6 months at M5
SPREAD_FRAC_OF_GATE = 0.6        # typical entered spread ≈ 60% of P90 gate


def resample_to_tf(df_m5, tf_minutes):
    df = df_m5.copy().set_index('timestamp')
    rule = f'{tf_minutes}min'
    out = df.resample(rule, label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    }).dropna()
    return out.reset_index()


def sma(arr, n):
    out = np.full(len(arr), np.nan)
    if len(arr) < n:
        return out
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out[n-1:] = (cs[n:] - cs[:-n]) / n
    return out


def compute_momentum_signal(closes, lags):
    s = sma(closes, SMA_N)
    sig = np.zeros(len(closes), dtype=np.int8)
    for i in range(max(lags) + SMA_N, len(closes)):
        cur = s[i]
        if np.isnan(cur):
            continue
        ups = sum(1 for lg in lags if not np.isnan(s[i-lg]) and cur > s[i-lg])
        dns = sum(1 for lg in lags if not np.isnan(s[i-lg]) and cur < s[i-lg])
        if ups == len(lags): sig[i] = 1
        elif dns == len(lags): sig[i] = -1
    return sig


def signal_at(m5_ts, h1_ts, h1_sig, m30_ts, m30_sig):
    h_idx = np.searchsorted(h1_ts, m5_ts, side='right') - 1
    m_idx = np.searchsorted(m30_ts, m5_ts, side='right') - 1
    if h_idx < 0 or m_idx < 0:
        return 0
    h_s = h1_sig[h_idx]; m_s = m30_sig[m_idx]
    if h_s == 1 and m_s == 1: return 1
    if h_s == -1 and m_s == -1: return -1
    return 0


def backtest_pair(pair, mode):
    pip, sp_gate = PAIRS[pair]
    tp_price = TP_PIPS * pip
    spread_cost_pips = sp_gate * SPREAD_FRAC_OF_GATE
    df = pq.read_table(DATA_DIR / f"{pair}_M5.parquet").to_pandas()
    df = df.tail(WINDOW_BARS).reset_index(drop=True)
    h1 = resample_to_tf(df, 60)
    m30 = resample_to_tf(df, 30)
    h1_sig = compute_momentum_signal(h1['close'].to_numpy(), LAGS)
    m30_sig = compute_momentum_signal(m30['close'].to_numpy(), LAGS)
    h1_ts = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()
    m5_ts = df['timestamp'].to_numpy()
    m5_o = df['open'].to_numpy()
    m5_h = df['high'].to_numpy()
    m5_l = df['low'].to_numpy()
    m5_c = df['close'].to_numpy()
    pos_dir = 0; pos_entry = 0.0
    trades = []
    for i in range(1, len(df)):
        sig = signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            if pos_dir == 1:
                tp_level = pos_entry + tp_price
                if m5_h[i] >= tp_level:
                    trades.append((pair, +1, pos_entry, tp_level, TP_PIPS, 'TP'))
                    pos_dir = 0; pos_entry = 0.0
                    continue
            else:
                tp_level = pos_entry - tp_price
                if m5_l[i] <= tp_level:
                    trades.append((pair, -1, pos_entry, tp_level, TP_PIPS, 'TP'))
                    pos_dir = 0; pos_entry = 0.0
                    continue
            if mode == 'fifoaware' and sig != 0 and sig != pos_dir:
                exit_px = m5_o[i]
                pnl_pips = (exit_px - pos_entry) / pip * pos_dir
                trades.append((pair, pos_dir, pos_entry, exit_px, pnl_pips, 'flip'))
                pos_dir = sig; pos_entry = exit_px
                continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; pos_entry = m5_o[i]
    if pos_dir != 0:
        exit_px = m5_c[-1]
        pnl_pips = (exit_px - pos_entry) / pip * pos_dir
        trades.append((pair, pos_dir, pos_entry, exit_px, pnl_pips, 'end'))
    tdf = pd.DataFrame(trades, columns=['pair','dir','entry','exit','pips','reason'])
    # Apply spread cost once per round-trip (entry + exit through bid-ask)
    if len(tdf) > 0:
        tdf['pips_net'] = tdf['pips'] - spread_cost_pips
    if len(tdf) == 0:
        return {'pair': pair, 'mode': mode, 'trades': 0,
                'pips_gross': 0.0, 'pips_net': 0.0,
                'wr_gross': 0.0, 'wr_net': 0.0,
                'tp_count': 0, 'flip_count': 0, 'end_count': 0,
                'days': 0, 'pd_gross': 0.0, 'pd_net': 0.0,
                'spread_cost': spread_cost_pips, 'tdf': tdf}
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()/86400
    return {
        'pair': pair, 'mode': mode, 'trades': len(tdf),
        'pips_gross': tdf['pips'].sum(),
        'pips_net': tdf['pips_net'].sum(),
        'wr_gross': (tdf['pips'] > 0).mean() * 100,
        'wr_net': (tdf['pips_net'] > 0).mean() * 100,
        'tp_count': (tdf['reason'] == 'TP').sum(),
        'flip_count': (tdf['reason'] == 'flip').sum(),
        'end_count': (tdf['reason'] == 'end').sum(),
        'days': days,
        'pd_gross': tdf['pips'].sum() / max(days, 1),
        'pd_net': tdf['pips_net'].sum() / max(days, 1),
        'spread_cost': spread_cost_pips,
        'tdf': tdf,
    }


def main():
    print("Backtesting fx-sma-live: CURRENT (FIFO-rejected = ignore opposing) "
          "vs FIFO-AWARE (close-and-flip)\n")
    print(f"Window: last {WINDOW_BARS} M5 bars (~6 months)")
    print(f"Params: SMA={SMA_N}, lags={LAGS}, TP={TP_PIPS}p, no SL\n")
    print(f"{'Pair':<10s} {'Mode':<10s} {'Days':>4s} {'Trades':>6s} "
          f"{'Gross':>8s} {'Net':>8s} {'WRn%':>5s} {'TPs':>4s} {'Flps':>5s} "
          f"{'p/d_g':>7s} {'p/d_n':>7s} {'sp':>4s}")
    print('-' * 88)
    rows_a, rows_b = [], []
    for pair in PAIRS:
        ra = backtest_pair(pair, 'current')
        rb = backtest_pair(pair, 'fifoaware')
        rows_a.append(ra); rows_b.append(rb)
        for r in (ra, rb):
            print(f"{r['pair']:<10s} {r['mode']:<10s} {r['days']:4.0f} "
                  f"{r['trades']:6d} {r['pips_gross']:8.1f} {r['pips_net']:8.1f} "
                  f"{r['wr_net']:5.1f} {r['tp_count']:4d} {r['flip_count']:5d} "
                  f"{r['pd_gross']:7.2f} {r['pd_net']:7.2f} {r['spread_cost']:4.1f}")
        print()
    print('=' * 88)
    print("PORTFOLIO TOTALS (Net = after spread cost ≈ 0.6 × IS-P90 gate per round-trip)")
    print('=' * 88)
    for label, rows in [('CURRENT  (no FIFO)', rows_a),
                        ('FIFO-AWARE        ', rows_b)]:
        total_trades = sum(r['trades'] for r in rows)
        total_gross = sum(r['pips_gross'] for r in rows)
        total_net = sum(r['pips_net'] for r in rows)
        if total_trades > 0:
            all_net = pd.concat([r['tdf']['pips_net'] for r in rows if r['trades'] > 0])
            wr = (all_net > 0).mean() * 100
        else:
            wr = 0.0
        days = rows[0]['days']
        print(f"  {label}:  trades={total_trades:5d}  "
              f"gross={total_gross:+9.1f}  net={total_net:+9.1f}  "
              f"WR_net={wr:5.1f}%  TPs={sum(r['tp_count'] for r in rows)}  "
              f"flips={sum(r['flip_count'] for r in rows)}  "
              f"p/d_net={total_net/days:+.2f}")
    print()
    a_net = sum(r['pips_net'] for r in rows_a)
    b_net = sum(r['pips_net'] for r in rows_b)
    print(f"DELTA (FIFO-aware - current, NET): {b_net - a_net:+.1f} pips total, "
          f"{(b_net - a_net) / rows_a[0]['days']:+.2f} p/d")


if __name__ == '__main__':
    main()
