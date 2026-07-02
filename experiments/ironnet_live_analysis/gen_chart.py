#!/usr/bin/env python3
"""Generate 5-panel EUR/JPY IronNet live analysis chart.

Panels:
  1. EUR/JPY M5 price + trade markers + duration spans
  2. Cumulative pips
  3. SBA (swing state on 10-pip range bars)
  4. mc_d (cyan) + mc_dd (purple) from FXFeatureBuilder(kalman10)
  5. Per-trade pip bars (L/S labeled)
"""
import sys, json, re
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# Project imports
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.incremental_features import FXFeatureBuilder
from lib.swing_indicators import compute_swing_features

# ── Config ────────────────────────────────────────────────────────────────────
CANDLE_CACHE = Path(__file__).parent / "candles_eurjpy.json"
TRADE_CACHE  = Path(__file__).parent / "trades_013.json"
OUT_PATH     = Path(__file__).parent / "eurjpy_ironnet_v2.png"

CUTOFF      = datetime(2026, 4, 30, 17, 10, 0, tzinfo=timezone.utc)
PAIR        = "EUR_JPY"
PIP         = 0.01
RANGE_PIPS  = 10.0
RANGE_PRICE = RANGE_PIPS * PIP
SWING_BUF   = 500


# ── Fetch / cache helpers ─────────────────────────────────────────────────────

def fetch_candles():
    """Pull 1200 M5 bars from OANDA via docker exec and cache locally."""
    import subprocess, textwrap
    script = textwrap.dedent("""
    import os, v20, json
    ctx = v20.Context(hostname='api-fxtrade.oanda.com', port='443',
                      token=os.environ.get('OANDA_API_KEY'))
    resp = ctx.instrument.candles('EUR_JPY', count=1200, granularity='M5',
                                  price='M', alignmentTimezone='UTC')
    candles = resp.body['candles']
    out = []
    for c in candles:
        if not c.complete: continue
        m = c.mid
        out.append({'t': c.time, 'o': float(m.o), 'h': float(m.h),
                    'l': float(m.l), 'c': float(m.c)})
    print(f'M5 candles: {len(out)}')
    print(json.dumps(out))
    """)
    result = subprocess.run(
        ["docker", "exec", "fx-core-fx-data-curator-1",
         "python3", "-c", script],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker exec failed: {result.stderr[:500]}")
    lines = result.stdout.strip().split('\n')
    data = json.loads(lines[-1])
    CANDLE_CACHE.write_text(json.dumps(data))
    print(f"Fetched {len(data)} M5 candles")
    return data


def fetch_trades():
    """Pull all closed trades from account 013 via docker exec and cache."""
    import subprocess, textwrap
    script = textwrap.dedent("""
    import os, v20, json
    ctx = v20.Context(hostname='api-fxtrade.oanda.com', port='443',
                      token=os.environ.get('OANDA_API_KEY'))
    acct = os.environ.get('OANDA_ACCOUNT_ID_013', '')
    resp = ctx.trade.list_closed(acct, count=1000)
    trades = resp.body['trades']
    out = []
    for t in trades:
        out.append({
            'id':          str(t.id),
            'instrument':  t.instrument,
            'openTime':    t.openTime,
            'closeTime':   t.closeTime,
            'initialUnits': float(t.initialUnits),
            'realizedPL':  float(t.realizedPL),
            'price':       float(t.price),
            'closePrice':  float(t.closePrice),
            'financing':   float(t.financing),
        })
    print(f'Trades: {len(out)}')
    print(json.dumps(out))
    """)
    result = subprocess.run(
        ["docker", "exec", "fx-core-fx-data-curator-1",
         "python3", "-c", script],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker exec failed: {result.stderr[:500]}")
    lines = result.stdout.strip().split('\n')
    data = json.loads(lines[-1])
    TRADE_CACHE.write_text(json.dumps(data))
    print(f"Fetched {len(data)} trades")
    return data


def load_or_fetch_candles():
    if CANDLE_CACHE.exists():
        data = json.loads(CANDLE_CACHE.read_text())
        print(f"Loaded {len(data)} M5 candles from cache")
        return data
    return fetch_candles()


def load_or_fetch_trades():
    if TRADE_CACHE.exists():
        data = json.loads(TRADE_CACHE.read_text())
        print(f"Loaded {len(data)} trades from cache")
        return data
    return fetch_trades()


# ── Range bar builder ─────────────────────────────────────────────────────────

class RangeBarBuilder:
    def __init__(self, range_price: float):
        self.range_price = range_price
        self.bar_open = None

    def feed(self, close: float):
        if self.bar_open is None:
            self.bar_open = close
            return None
        if abs(close - self.bar_open) >= self.range_price:
            self.bar_open = close
            return close
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    raw_candles = load_or_fetch_candles()
    raw_trades  = load_or_fetch_trades()

    # ── Parse candles ─────────────────────────────────────────────────────────
    candles = []
    for c in raw_candles:
        ts = pd.Timestamp(c['t']).tz_convert('UTC')
        candles.append({'ts': ts, 'o': c['o'], 'h': c['h'], 'l': c['l'], 'c': c['c']})
    df_c = pd.DataFrame(candles).set_index('ts').sort_index()

    # ── Filter EUR/JPY trades opened after cutoff ─────────────────────────────
    ej_trades = []
    for t in raw_trades:
        if t['instrument'] != 'EUR_JPY':
            continue
        open_ts = pd.Timestamp(t['openTime']).tz_convert('UTC')
        if open_ts <= CUTOFF:
            continue
        close_ts = pd.Timestamp(t['closeTime']).tz_convert('UTC')
        units    = float(t['initialUnits'])
        pnl_ccy  = float(t['realizedPL'])
        pnl_pips = (float(t['closePrice']) - float(t['price'])) / PIP
        if units < 0:
            pnl_pips = -pnl_pips
        ej_trades.append({
            'id':         t['id'],
            'open_ts':    open_ts,
            'close_ts':   close_ts,
            'entry':      float(t['price']),
            'exit':       float(t['closePrice']),
            'units':      units,
            'pnl_pips':   pnl_pips,
            'direction':  'L' if units > 0 else 'S',
        })
    ej_trades.sort(key=lambda x: x['open_ts'])
    print(f"EUR/JPY post-cutoff trades: {len(ej_trades)}")

    # ── Build features: FXFeatureBuilder + range bars ─────────────────────────
    builder  = FXFeatureBuilder('EUR_JPY', smoother='kalman10')
    rb       = RangeBarBuilder(RANGE_PRICE)
    rb_buf   = deque(maxlen=SWING_BUF)

    ts_list   = []
    price_mid = []
    sba_arr   = []
    mcd_arr   = []
    mcdd_arr  = []

    cur_sba  = 0.0
    cur_mcd  = 0.0
    cur_mcdd = 0.0

    for ts, row in df_c.iterrows():
        o, h, l, c = row['o'], row['h'], row['l'], row['c']
        mid = (h + l) / 2.0

        # FXFeatureBuilder → mc_d, mc_dd
        feats = builder.process_new_bar(o, h, l, c, timestamp=ts)
        cur_mcd  = feats.get('mc_d',  0.0) or 0.0
        cur_mcdd = feats.get('mc_dd', 0.0) or 0.0

        # Range bar → SBA update
        completed = rb.feed(mid)
        if completed is not None:
            rb_buf.append(completed)
            if len(rb_buf) >= 3:
                closes = np.array(rb_buf, dtype=np.float64)
                state, _, _, _, _ = compute_swing_features(closes, closes, closes)
                cur_sba = float(state[-1]) / 2.0

        ts_list.append(ts)
        price_mid.append(mid)
        sba_arr.append(cur_sba)
        mcd_arr.append(cur_mcd)
        mcdd_arr.append(cur_mcdd)

    ts_arr    = np.array(ts_list)
    price_arr = np.array(price_mid)
    sba_arr   = np.array(sba_arr)
    mcd_arr   = np.array(mcd_arr)
    mcdd_arr  = np.array(mcdd_arr)

    # ── Cumulative pips ───────────────────────────────────────────────────────
    cum_pips = 0.0
    cum_list = []
    for tr in ej_trades:
        cum_pips += tr['pnl_pips']
        cum_list.append(cum_pips)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(18, 20),
                             gridspec_kw={'height_ratios': [4, 2, 1.5, 1.5, 2]},
                             sharex=True)
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # ─ Panel 1: Price ─────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(ts_arr, price_arr, color='#58a6ff', linewidth=0.8, alpha=0.9, label='EUR/JPY')
    ax1.set_ylabel('EUR/JPY', color='#c9d1d9')
    ax1.set_title('EUR/JPY IronNet Live — Post Apr30 17:10 UTC', color='#e6edf3',
                  fontsize=13, fontweight='bold', pad=10)

    # Trade spans and markers
    for tr in ej_trades:
        color = '#3fb950' if tr['pnl_pips'] >= 0 else '#f85149'
        alpha = 0.15
        # shade duration
        ax1.axvspan(tr['open_ts'], tr['close_ts'], color=color, alpha=alpha)
        # entry marker
        marker = '^' if tr['direction'] == 'L' else 'v'
        ax1.scatter([tr['open_ts']], [tr['entry']], marker=marker,
                    color='#3fb950' if tr['direction'] == 'L' else '#f85149',
                    s=60, zorder=5)
        # exit marker (X)
        ax1.scatter([tr['close_ts']], [tr['exit']], marker='x',
                    color=color, s=40, linewidths=1.5, zorder=5)

    ax1.legend(loc='upper left', facecolor='#21262d', labelcolor='#c9d1d9',
               edgecolor='#30363d', fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # ─ Panel 2: Cumulative pips ───────────────────────────────────────────────
    ax2 = axes[1]
    open_times = [tr['open_ts'] for tr in ej_trades]
    pnls       = [tr['pnl_pips'] for tr in ej_trades]
    colors_p   = ['#3fb950' if p >= 0 else '#f85149' for p in pnls]

    if ej_trades:
        ax2.step(open_times + [open_times[-1]], [0] + cum_list,
                 where='post', color='#e3b341', linewidth=1.5)
        ax2.scatter(open_times, cum_list, c=colors_p, s=40, zorder=5)
        ax2.axhline(0, color='#30363d', linewidth=0.8)
    ax2.set_ylabel('Cum pips', color='#c9d1d9')
    total = cum_list[-1] if cum_list else 0.0
    ax2.text(0.99, 0.05, f'Total: {total:+.1f}p', transform=ax2.transAxes,
             ha='right', va='bottom', color='#e3b341', fontsize=10)

    # ─ Panel 3: SBA ──────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.step(ts_arr, sba_arr, where='post', color='#d29922', linewidth=1.2)
    ax3.fill_between(ts_arr, sba_arr, step='post', alpha=0.25, color='#d29922')
    ax3.axhline(0, color='#30363d', linewidth=0.8)
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_ylabel('SBA', color='#c9d1d9')

    # ─ Panel 4: mc_d / mc_dd ─────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(ts_arr, mcd_arr,  color='#58d1eb', linewidth=0.9, label='mc_d')
    ax4.plot(ts_arr, mcdd_arr, color='#a371f7', linewidth=0.9, label='mc_dd', alpha=0.85)
    ax4.axhline(0, color='#30363d', linewidth=0.8)
    ax4.set_ylabel('mc_d / mc_dd', color='#c9d1d9')
    ax4.legend(loc='upper left', facecolor='#21262d', labelcolor='#c9d1d9',
               edgecolor='#30363d', fontsize=8, ncol=2)

    # ─ Panel 5: Per-trade bars ────────────────────────────────────────────────
    ax5 = axes[4]
    if ej_trades:
        for i, tr in enumerate(ej_trades):
            c = '#3fb950' if tr['pnl_pips'] >= 0 else '#f85149'
            ax5.bar(tr['open_ts'], tr['pnl_pips'], width=pd.Timedelta(minutes=4),
                    color=c, alpha=0.85)
            ax5.text(tr['open_ts'], tr['pnl_pips'] + (0.3 if tr['pnl_pips'] >= 0 else -1.0),
                     tr['direction'], ha='center', va='bottom' if tr['pnl_pips'] >= 0 else 'top',
                     color='#c9d1d9', fontsize=6)
    ax5.axhline(0, color='#30363d', linewidth=0.8)
    ax5.set_ylabel('Trade pips', color='#c9d1d9')

    # ─ Shared x-axis formatting ───────────────────────────────────────────────
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right', color='#8b949e')

    fig.tight_layout(h_pad=0.5)
    fig.savefig(str(OUT_PATH), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
