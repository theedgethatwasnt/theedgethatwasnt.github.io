"""
M1 net-move under two normalizations:
  A) move / spread   (cost-normalized -- "how many spreads did it travel")
  B) move / sigma    (vol-normalized  -- the mn_ style)

Question: on the same timeframe, do these two series tell the same story or
different ones?  Where they agree, spread tracks volatility.  Where they
diverge, the cost-of-trading has decoupled from the size-of-move (news,
illiquid sessions, rollover) -- and THAT divergence is itself the signal.

Also reports the timeframe at which mean true range ~ one spread.
"""
import numpy as np
import pandas as pd
import sys

PAIRS = ['USD_JPY', 'GBP_JPY']
ROLL = 240          # ~4h of M1 bars for sigma
np.set_printoptions(suppress=True)


def load_m1(pair):
    pip = 0.01 if 'JPY' in pair else 0.0001
    df = pd.read_parquet(f'data/s5_ba/{pair}_S5_BA.parquet',
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'bid_c', 'ask_c'])
    df = df.set_index('timestamp')
    df['spread'] = (df['ask_c'] - df['bid_c']) / pip
    m1 = pd.DataFrame({
        'open':  df['open'].resample('1min').first(),
        'high':  df['high'].resample('1min').max(),
        'low':   df['low'].resample('1min').min(),
        'close': df['close'].resample('1min').last(),
        'spread': df['spread'].resample('1min').mean(),
    }).dropna()
    m1['move'] = (m1['close'] - m1['open']) / pip           # net move, pips
    m1['tr']   = (m1['high'] - m1['low']) / pip             # bar true range, pips
    return m1, pip


def tf_for_one_spread(df_s5, pip):
    """At what resampling does mean true range ~ one spread?"""
    sp = ((df_s5['ask_c'] - df_s5['bid_c']) / pip).mean()
    out = []
    for rule, label in [('5s', 'S5'), ('1min', 'M1'), ('5min', 'M5'),
                        ('15min', 'M15'), ('1h', 'H1')]:
        if rule == '5s':
            tr = ((df_s5['high'] - df_s5['low']) / pip).mean()
        else:
            hi = df_s5['high'].resample(rule).max()
            lo = df_s5['low'].resample(rule).min()
            tr = ((hi - lo) / pip).dropna().mean()
        out.append((label, tr, tr / sp))
    return sp, out


for pair in PAIRS:
    pip = 0.01 if 'JPY' in pair else 0.0001
    raw = pd.read_parquet(f'data/s5_ba/{pair}_S5_BA.parquet',
                          columns=['timestamp', 'high', 'low', 'bid_c', 'ask_c']).set_index('timestamp')
    sp, tf = tf_for_one_spread(raw, pip)
    del raw

    m1, _ = load_m1(pair)
    move, spread, tr = m1['move'].values, m1['spread'].values, m1['tr'].values
    sigma = m1['move'].rolling(ROLL).std().values

    valid = np.isfinite(sigma) & (sigma > 0) & (spread > 0)
    A = move[valid] / spread[valid]          # move / spread
    B = move[valid] / sigma[valid]           # move / sigma  (mn_)
    sp_v, sg_v = spread[valid], sigma[valid]

    corr = np.corrcoef(A, B)[0, 1]
    # the scale factor that maps B -> A is (sigma/spread) per bar
    ratio = sg_v / sp_v                      # sigma-in-spreads: cost-vs-vol regime

    print('=' * 70)
    print(f'{pair}   (M1, {len(A):,} bars, sigma roll={ROLL})')
    print(f'  mean spread = {sp:.2f}p')
    print('  mean true range by timeframe (TR / spread):')
    for lbl, t, r in tf:
        flag = '  <-- ~1 spread' if 0.7 < r < 1.6 else ''
        print(f'    {lbl:4s}  TR={t:6.2f}p   {r:5.2f}x{flag}')
    print(f'  --- normalized move series ---')
    print(f'  A=move/spread : std={A.std():.3f}  med|.|={np.median(np.abs(A)):.3f}  '
          f'kurt={pd.Series(A).kurt():.1f}  P(|A|>1)={(np.abs(A)>1).mean()*100:.1f}%')
    print(f'  B=move/sigma  : std={B.std():.3f}  med|.|={np.median(np.abs(B)):.3f}  '
          f'kurt={pd.Series(B).kurt():.1f}  P(|B|>1)={(np.abs(B)>1).mean()*100:.1f}%')
    print(f'  corr(A,B)     = {corr:.4f}')
    print(f'  sigma/spread  : mean={ratio.mean():.2f}  std={ratio.std():.2f}  '
          f'P10={np.percentile(ratio,10):.2f}  P90={np.percentile(ratio,90):.2f}')
    # cointegration-style: do log spread and log sigma move together?
    ls, lg = np.log(sp_v), np.log(sg_v)
    print(f'  corr(log spread, log sigma) = {np.corrcoef(ls, lg)[0,1]:.4f}')

    # ---- does spread LEAD or FOLLOW the move? ----
    # cross-correlate spread(t) with |move|(t+k). k>0 => spread leads move.
    sp_s = pd.Series(sp_v); sp_s = (sp_s - sp_s.mean()) / sp_s.std()
    am_s = pd.Series(np.abs(move[valid])); am_s = (am_s - am_s.mean()) / am_s.std()
    print('  lead/lag  corr(spread_t, |move|_{t+k}):  (k>0 = spread leads)')
    row = []
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        c = sp_s.corr(am_s.shift(-k))
        row.append(f'k={k:+d}:{c:+.3f}')
    print('     ' + '  '.join(row))
    # does spread predict NEXT-bar realized vol beyond current sigma?
    fwd = pd.Series(np.abs(move[valid])).rolling(5).mean().shift(-5).values  # next-5-bar mean |move|
    m = np.isfinite(fwd)
    import numpy.linalg as la
    X = np.column_stack([np.ones(m.sum()), sg_v[m], sp_v[m]])
    beta, *_ = la.lstsq(X, fwd[m], rcond=None)
    # partial corr of spread with fwd vol controlling for sigma
    def resid(y, ctrl):
        Xc = np.column_stack([np.ones(len(ctrl)), ctrl]); b, *_ = la.lstsq(Xc, y, rcond=None)
        return y - Xc @ b
    pr = np.corrcoef(resid(sp_v[m], sg_v[m]), resid(fwd[m], sg_v[m]))[0, 1]
    print(f'  fwd5 |move| ~ a + b1*sigma + b2*spread : b2(spread)={beta[2]:+.4f}')
    print(f'  partial corr(spread, fwd5 vol | sigma) = {pr:+.4f}'
          f'   ({"spread carries info beyond realized vol" if pr>0.02 else "no extra info"})')
