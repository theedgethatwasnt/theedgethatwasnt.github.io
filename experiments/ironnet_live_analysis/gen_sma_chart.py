#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

CANDLE_CACHE = Path(__file__).parent / "candles_eurjpy.json"
OUT_PATH     = Path(__file__).parent / "eurjpy_sma.png"

raw = json.loads(CANDLE_CACHE.read_text())
df = pd.DataFrame([{
    'ts': pd.Timestamp(c['t']).tz_convert('UTC'),
    'price': (c['h'] + c['l']) / 2.0
} for c in raw]).set_index('ts').sort_index()

df['sma5']  = df['price'].rolling(5).mean()
df['sma50'] = df['price'].rolling(50).mean()
df['p_s5']  = df['price'] - df['sma5']
df['p_s50'] = df['price'] - df['sma50']
df['s5_s50']= df['sma5']  - df['sma50']

fig, axes = plt.subplots(4, 1, figsize=(18, 14),
                         gridspec_kw={'height_ratios': [4, 1.5, 1.5, 1.5]},
                         sharex=True)
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.axhline(0, color='#30363d', linewidth=0.7)

ax1, ax2, ax3, ax4 = axes

# Panel 1: price + SMA5 + SMA50
ax1.plot(df.index, df['price'], color='#58a6ff', linewidth=0.7, label='price')
ax1.plot(df.index, df['sma5'],  color='#e3b341', linewidth=1.1, label='SMA5')
ax1.plot(df.index, df['sma50'], color='#f85149', linewidth=1.2, label='SMA50')
ax1.set_ylabel('EUR/JPY')
ax1.set_title('EUR/JPY  —  Price + SMA5 + SMA50  (M5)', color='#e6edf3',
              fontsize=12, fontweight='bold', pad=8)
ax1.legend(loc='upper left', facecolor='#21262d', labelcolor='#c9d1d9',
           edgecolor='#30363d', fontsize=9)
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
pad = (df['price'].max() - df['price'].min()) * 0.05
ax1.set_ylim(df['price'].min() - pad, df['price'].max() + pad)

def _fill(ax, x, y, label, color):
    ax.plot(x, y, color=color, linewidth=0.9)
    ax.fill_between(x, y, where=y >= 0, color='#3fb950', alpha=0.25)
    ax.fill_between(x, y, where=y <  0, color='#f85149', alpha=0.25)
    ax.set_ylabel(label)

_fill(ax2, df.index, df['p_s5'],   'price − SMA5',  '#e3b341')
_fill(ax3, df.index, df['p_s50'],  'price − SMA50', '#f85149')
_fill(ax4, df.index, df['s5_s50'], 'SMA5 − SMA50',  '#a371f7')

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', color='#8b949e')

fig.tight_layout(h_pad=0.4)
fig.savefig(str(OUT_PATH), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PATH}")
