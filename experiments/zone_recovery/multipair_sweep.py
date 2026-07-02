"""
Multi-pair boundary zone recovery sweep — Phase 9 (break-even sizing).
Objective: maximise $/hr per base-unit (= net_pnl_pips / trading_hours).
For each pair: wide grid zw × E/Z, 5-gate filter, report winner + full table.

Engine: BoundaryZoneEngine with break-even sizing (Roni cBot logic):
  - Only add a leg when net P&L at that target is adverse (<0)
  - Volume = ceil(-net_pips / target_pips × profit_factor=1.19)

Analytical prior: zw_prior = 0.30 × median_daily_range (known to overestimate ~60%)
Grid covers 0.10×–2.20× prior to capture both narrow and wide zones.
E/Z range 0.5–4.0 (small targets may exit faster with fewer legs).

All pip values are in pair-native pips (0.01 for JPY, 0.0001 for others).
P&L is normalised pips × pip_value_usd to get common dollar units.
"""

import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from boundary_engine import BoundaryZoneEngine
from backtest import run_backtest, compute_metrics, run_5gate_validation

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')

PAIRS = [
    'EUR_USD','GBP_USD','AUD_USD','NZD_USD','EUR_GBP',
    'USD_JPY','EUR_JPY','GBP_JPY','AUD_JPY','CAD_JPY','CHF_JPY','NZD_JPY',
]

# Pip sizes and approximate USD pip value per 1 OANDA unit
PAIR_META = {p: {'pip': 0.01   if 'JPY' in p else 0.0001,
                 'pip_usd': 0.000091 if 'JPY' in p else 0.0001}
             for p in PAIRS}
# EUR_GBP pip value approx (GBP/USD ~1.27)
PAIR_META['EUR_GBP']['pip_usd'] = 0.000127

BARS_PER_TRADING_DAY = 288   # M5, 24h


def load_pair(pair: str):
    path = os.path.join(DATA_DIR, f'{pair}_M5.parquet')
    df = pd.read_parquet(path)
    df = df.sort_index()
    # Standardise column names
    df.columns = [c.lower() for c in df.columns]
    for col in ('open','high','low','close'):
        if col not in df.columns:
            raise ValueError(f'{pair}: missing {col}')
    return df


def make_features(df):
    n = len(df)
    return {
        'open':  df['open'].values.astype(np.float64),
        'high':  df['high'].values.astype(np.float64),
        'low':   df['low'].values.astype(np.float64),
        'close': df['close'].values.astype(np.float64),
        'atr_short': np.ones(n),
        'atr_long':  np.ones(n),
    }


def analytical_prior(df, pip_size: float):
    """ATR-based zone width prior: 0.30 × median daily range in pips."""
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    # True range daily (approximate from M5: max over 288-bar window)
    daily_range = []
    for i in range(288, len(close), 288):
        daily_range.append((high[i-288:i].max() - low[i-288:i].min()) / pip_size)
    median_range = np.median(daily_range) if daily_range else 50.0
    zw_prior = round(median_range * 0.30)
    return zw_prior, median_range


def run_pair(pair: str):
    meta     = PAIR_META[pair]
    pip      = meta['pip']
    pip_usd  = meta['pip_usd']

    df = load_pair(pair)
    features = make_features(df)
    n = len(features['close'])
    n_train = int(n * 0.70)
    train_f = {k: v[:n_train] for k, v in features.items()}
    test_f  = {k: v[n_train:] for k, v in features.items()}
    oos_hrs  = len(test_f['close']) / BARS_PER_TRADING_DAY * 24

    zw_prior, median_range = analytical_prior(df, pip)

    # Wide grid: 0.10–2.20× prior, E/Z 0.5–4.0
    # Prior known to overestimate ~60%; wide range catches true optimal for break-even sizing
    zw_values = sorted(set(max(5, round(zw_prior * m))
                           for m in [0.10, 0.15, 0.20, 0.30, 0.40, 0.55,
                                     0.70, 0.90, 1.10, 1.40, 1.80, 2.20]))
    ez_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    rows = []
    for zw in zw_values:
        for ez in ez_values:
            tgt = round(zw * ez)
            if tgt < 3:
                continue

            eng = BoundaryZoneEngine(
                zone_width_pips=zw, target_beyond_pips=tgt,
                max_legs=10, profit_factor=1.19,
                spread_pips=1.4, pip_size=pip,
            )

            try:
                r = run_5gate_validation(eng, features, train_f, test_f, seed=42)
            except Exception as e:
                continue

            m  = r['oos_metrics']
            g  = r['gates_passed']
            v  = r['verdict']

            # $/hr at 1 OANDA unit — uses pair-specific pip USD value
            pnl_usd   = m['net_pnl_pips'] * pip_usd
            dd_usd    = m['max_drawdown_pips'] * pip_usd
            pnl_hr    = pnl_usd / oos_hrs
            calmar    = pnl_hr / abs(dd_usd / oos_hrs * BARS_PER_TRADING_DAY * 24) \
                        if dd_usd != 0 else 0   # $/hr / ($/day drawdown rate)

            rows.append({
                'pair': pair, 'zw': zw, 'tgt': tgt, 'ez': ez,
                'gates': g, 'verdict': v,
                'n': m['n_cycles'],
                'avg_hrs': m['avg_duration_bars'] * 5 / 60,
                'esc_pct': m['exit_reasons'].get('target', 0) / max(1, m['n_cycles']) * 100,
                'pnl_usd': pnl_usd,
                'pnl_hr_1u': pnl_hr,
                'dd_usd': dd_usd,
                'calmar': pnl_hr / abs(pnl_hr - dd_usd / oos_hrs) if dd_usd != 0 else 0,
                'sharpe': m['sharpe'],
                'sqn': m['sqn'],
                'wr': m['win_rate'] * 100,
                'zw_prior': zw_prior,
                'median_range_pips': round(median_range, 1),
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────
all_rows   = []
pair_winners = []

print(f"{'Pair':<10} {'zw':>4} {'tgt':>4} {'ez':>5} {'g':>2} "
      f"{'n':>5} {'avgH':>5} {'$/hr@1u':>10} {'$/day@1ku':>10} "
      f"{'maxDD@1ku':>10} {'Sharpe':>7} {'SQN':>6} {'prior':>6}")
print("─"*110)

for pair in PAIRS:
    t0 = time.time()
    try:
        df_pair = run_pair(pair)
    except Exception as e:
        print(f"  {pair}: ERROR — {e}")
        continue

    all_rows.append(df_pair)

    passing = df_pair[df_pair['verdict'] == 'PASS'].copy()
    if passing.empty:
        print(f"  {pair}: NO PASSING CONFIGS")
        continue

    # Winner = max $/hr among 5/5 gate passing
    winner = passing.loc[passing['pnl_hr_1u'].idxmax()]
    pair_winners.append(winner.to_dict())

    # Print all passing configs for this pair
    for _, row in passing.sort_values('pnl_hr_1u', ascending=False).iterrows():
        tag = " ◄ BEST" if (row['zw'] == winner['zw'] and row['tgt'] == winner['tgt']) else ""
        print(f"  {row['pair']:<10} {row['zw']:>4} {row['tgt']:>4} {row['ez']:>5.1f} "
              f"{row['gates']:>2} {row['n']:>5} {row['avg_hrs']:>5.1f} "
              f"${row['pnl_hr_1u']:>9.5f} "
              f"${row['pnl_hr_1u']*1000*24:>9.2f} "
              f"${row['dd_usd']*1000:>9.2f} "
              f"{row['sharpe']:>7.4f} {row['sqn']:>6.2f} "
              f"{row['zw_prior']:>5}p{tag}")

    elapsed = time.time() - t0
    print(f"  {'':10} {'—'*95} [{elapsed:.0f}s]\n")

# ── Summary table ─────────────────────────────────────────────────────────
if pair_winners:
    summary = pd.DataFrame(pair_winners).sort_values('pnl_hr_1u', ascending=False)
    print("\n" + "═"*110)
    print("WINNER PER PAIR (max $/hr @ 1u, 5/5 gates)")
    print("═"*110)
    print(f"{'Pair':<10} {'zw':>4} {'tgt':>4} {'ez':>5} {'n':>5} {'avgH':>5} "
          f"{'$/hr@1u':>10} {'$/day@1ku':>10} {'maxDD@1ku':>10} "
          f"{'Sharpe':>7} {'SQN':>6} {'prior':>6}")
    print("─"*110)
    for _, w in summary.iterrows():
        print(f"  {w['pair']:<10} {w['zw']:>4.0f} {w['tgt']:>4.0f} {w['ez']:>5.1f} "
              f"{w['n']:>5.0f} {w['avg_hrs']:>5.1f} "
              f"${w['pnl_hr_1u']:>9.5f} "
              f"${w['pnl_hr_1u']*1000*24:>9.2f} "
              f"${w['dd_usd']*1000:>9.2f} "
              f"{w['sharpe']:>7.4f} {w['sqn']:>6.2f} "
              f"{w['zw_prior']:>5.0f}p")

    # Analytical prior accuracy
    print(f"\nAnalytical prior check (zw_prior vs winner zw):")
    for _, w in summary.iterrows():
        err = (w['zw'] - w['zw_prior']) / w['zw_prior'] * 100
        print(f"  {w['pair']:<10}  prior={w['zw_prior']:.0f}p  winner={w['zw']:.0f}p  "
              f"error={err:+.0f}%  median_range={w['median_range_pips']:.0f}p")

# ── Save ──────────────────────────────────────────────────────────────────
if all_rows:
    full = pd.concat(all_rows, ignore_index=True)
    out  = os.path.join(os.path.dirname(__file__), 'results', 'phase9_multipair_breakeven.csv')
    full.to_csv(out, index=False)
    print(f"\nSaved {len(full)} rows → {out}")
