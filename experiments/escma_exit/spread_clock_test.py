"""
Does spread follow volatility, lead it, or just follow the CLOCK?
USD_JPY M1.  Build hour-of-day (UTC) profiles of spread and realized sigma,
both z-scored, and overlay.  If spread is a clock, its diurnal cycle is huge
and only loosely shares shape with sigma's.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pair, pip = 'USD_JPY', 0.01
df = pd.read_parquet(f'data/s5_ba/{pair}_S5_BA.parquet',
                     columns=['timestamp', 'open', 'close', 'bid_c', 'ask_c']).set_index('timestamp')
df['spread'] = (df['ask_c'] - df['bid_c']) / pip
m1 = pd.DataFrame({
    'open': df['open'].resample('1min').first(),
    'close': df['close'].resample('1min').last(),
    'spread': df['spread'].resample('1min').mean(),
}).dropna()
m1['move'] = (m1['close'] - m1['open']) / pip
m1['sigma'] = m1['move'].rolling(240).std()
m1 = m1.dropna()
m1['hour'] = m1.index.hour
m1['dow'] = m1.index.dayofweek

prof = m1.groupby('hour').agg(spread=('spread', 'mean'),
                              sigma=('sigma', 'mean'),
                              absmove=('move', lambda x: x.abs().mean()))
zs = lambda s: (s - s.mean()) / s.std()

# how much of spread variance is explained by hour-of-week vs by sigma?
m1['how'] = m1['dow'] * 24 + m1['hour']
how_mean = m1.groupby('how')['spread'].transform('mean')
r2_clock = 1 - ((m1['spread'] - how_mean) ** 2).sum() / ((m1['spread'] - m1['spread'].mean()) ** 2).sum()
# r2 of spread on sigma (linear)
c = np.corrcoef(m1['spread'], m1['sigma'])[0, 1]
print(f'spread variance explained by hour-of-week dummies : {r2_clock*100:5.1f}%')
print(f'spread variance explained by realized sigma (r^2) : {c**2*100:5.1f}%')
print(f'corr(hourly spread profile, hourly sigma profile) : {np.corrcoef(prof["spread"],prof["sigma"])[0,1]:+.3f}')

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(prof.index, zs(prof['spread']), 'o-', color='#c0392b', label='spread (z)')
ax[0].plot(prof.index, zs(prof['sigma']), 's-', color='#2c3e50', label='realized $\\sigma$ (z)')
ax[0].plot(prof.index, zs(prof['absmove']), '^--', color='#7f8c8d', alpha=.6, label='|move| (z)')
ax[0].axhline(0, color='k', lw=.5)
ax[0].set_xlabel('hour of day (UTC)'); ax[0].set_ylabel('z-score of hourly mean')
ax[0].set_title('Spread vs volatility: diurnal profiles\n(spread is a clock; $\\sigma$ peaks at London/NY)')
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

# distribution overlay of the two normalizations
A = (m1['move'] / m1['spread']).values
B = (m1['move'] / m1['sigma']).values
bins = np.linspace(-4, 4, 121)
ax[1].hist(A, bins=bins, density=True, alpha=.5, color='#c0392b', label='move / spread  (kurt 48)')
ax[1].hist(B, bins=bins, density=True, alpha=.5, color='#2c3e50', label='move / $\\sigma$  (kurt 9)')
ax[1].set_yscale('log')
ax[1].set_xlabel('normalized M1 net move'); ax[1].set_ylabel('density (log)')
ax[1].set_title('Same numerator, different ruler\nspread-scale keeps the fat tail; $\\sigma$-scale whitens it')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
plt.tight_layout()
out = 'research/experiments/escma_exit/spread_vs_sigma_m1.png'
plt.savefig(out, dpi=110)
print('saved', out)
