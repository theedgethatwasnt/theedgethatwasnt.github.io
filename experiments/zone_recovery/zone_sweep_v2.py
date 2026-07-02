"""
Zone Recovery Phase 5: Large-zone sweep + session filter + directional bias
"""
import numpy as np
import pandas as pd
from data_utils import load_m5, prepare_features
from engine import ZoneRecoveryEngine
from backtest import run_backtest, compute_metrics, run_5gate_validation

PAIR = 'EUR_USD'
SPREAD = 1.4

df = load_m5(PAIR)
feats = prepare_features(df)
n = len(feats['close'])
train_f = {k: v[:int(n*0.7)] for k, v in feats.items()}
test_f  = {k: v[int(n*0.7):] for k, v in feats.items()}
ts = feats['timestamps']  # numpy datetime64

# ── Session mask ──────────────────────────────────────────────────────────────
hours = ts.astype('datetime64[h]').astype(int) % 24
# London 07-12 UTC (morning momentum) + NY open 13-18 UTC
london_ny = ((hours >= 7) & (hours < 12)) | ((hours >= 13) & (hours < 18))
mask_full = london_ny

# ── Direction signals ─────────────────────────────────────────────────────────
close = feats['close']
# Momentum: last 20-bar SMA slope. If close > SMA20 → Long (+1) first.
sma20 = np.full(n, np.nan)
for i in range(20, n):
    sma20[i] = close[i-20:i].mean()
momentum_sig = np.where(close > sma20, 1, -1).astype(np.int8)
# Mean-rev: opposite
meanrev_sig  = -momentum_sig

# ── Helper ────────────────────────────────────────────────────────────────────
def make_engine(hz, tgt, ml=10):
    return ZoneRecoveryEngine(
        mode='classic', half_zone_pips=hz, target_beyond_pips=tgt,
        init_target_pips=9999.0, base_unit=1000, max_legs=ml,
        sizing_mode='convex', convex_exponent=1.5, spread_pips=SPREAD,
    )

def row(label, hz, tgt, ml, mask, sig):
    eng = make_engine(hz, tgt, ml)
    vr  = run_5gate_validation(eng, feats, train_f, test_f,
                                entry_mask=mask, direction_signal=sig)
    m   = vr['oos_metrics']
    res = run_backtest(eng, test_f, seed=42,
                       entry_mask=mask[int(n*0.7):] if mask is not None else None,
                       direction_signal=sig[int(n*0.7):] if sig is not None else None)
    legs = np.array([len(r.legs) for r in res]) if res else np.array([0])
    esc  = sum(1 for r in res if r.exit_reason=='target')
    nr   = len(res) or 1
    return {
        'label': label, 'hz': hz, 'tgt': tgt, 'ml': ml,
        'gates': vr['gates_passed'], 'verdict': vr['verdict'],
        'n': nr, 'esc_pct': esc/nr*100, 'ml_pct': (nr-esc)/nr*100,
        'avg_legs': legs.mean(),
        'pnl_$': m['net_pnl_pips']*0.10,
        'sharpe': m['sharpe'], 'sqn': m['sqn'],
        'wr': m['win_rate']*100,
    }

rows = []

# ── Experiment A: Large zones, E/Z≈0.5, random direction ─────────────────────
print('A: Large zone sweep E/Z≈0.5 ...')
for hz in [25, 35, 45, 55, 65, 75]:
    for tgt in [hz//2, int(hz*0.38), hz]:   # E/Z 0.25, 0.38, 0.50
        ez = tgt / (2*hz)
        for ml in [7, 10]:
            r = row(f'rand_hz{hz}_tgt{tgt}_ml{ml}', hz, tgt, ml, None, None)
            r['exp'] = 'A_random'; r['ez'] = round(ez,2)
            rows.append(r)
            g = r['gates']; v = r['verdict']
            print(f"  hz={hz:2d} tgt={tgt:2d} E/Z={ez:.2f} ml={ml} | {g}/5 {v} | "
                  f"avgL={r['avg_legs']:.1f} esc={r['esc_pct']:.0f}% "
                  f"pnl=${r['pnl_$']:+,.0f} sh={r['sharpe']:+.4f}")

# ── Experiment B: Best zones + London/NY session filter ───────────────────────
print('\nB: Session filter (London 07-12 + NY 13-18 UTC) ...')
for hz, tgt, ml in [(15,15,10),(20,15,10),(25,15,10),(35,18,10),(45,22,10)]:
    for ez_label, tgt_use in [('E/Z=0.50', tgt), ('E/Z=0.38', int(hz*0.38))]:
        r = row(f'sess_hz{hz}_tgt{tgt_use}_ml{ml}', hz, tgt_use, ml, mask_full, None)
        r['exp'] = 'B_session'; r['ez'] = round(tgt_use/(2*hz), 2)
        rows.append(r)
        g = r['gates']; v = r['verdict']
        print(f"  hz={hz:2d} tgt={tgt_use:2d} E/Z={tgt_use/(2*hz):.2f} | {g}/5 {v} | "
              f"n={r['n']:4d} avgL={r['avg_legs']:.1f} esc={r['esc_pct']:.0f}% "
              f"pnl=${r['pnl_$']:+,.0f} sh={r['sharpe']:+.4f}")

# ── Experiment C: Directional bias (momentum vs mean-rev) ────────────────────
print('\nC: Directional bias ...')
for hz, tgt, ml in [(15,15,10),(20,15,10),(25,15,10)]:
    for bias, sig in [('momentum', momentum_sig), ('mean_rev', meanrev_sig)]:
        r = row(f'{bias}_hz{hz}_tgt{tgt}', hz, tgt, ml, None, sig)
        r['exp'] = f'C_{bias}'; r['ez'] = round(tgt/(2*hz), 2)
        rows.append(r)
        g = r['gates']; v = r['verdict']
        print(f"  {bias:10s} hz={hz:2d} tgt={tgt:2d} | {g}/5 {v} | "
              f"n={r['n']:4d} avgL={r['avg_legs']:.1f} esc={r['esc_pct']:.0f}% "
              f"pnl=${r['pnl_$']:+,.0f} sh={r['sharpe']:+.4f}")

# ── Experiment D: Session + best bias combo ───────────────────────────────────
print('\nD: Session + best directional bias combos ...')
for hz, tgt in [(15,15),(20,15),(25,15)]:
    for bias, sig in [('momentum', momentum_sig), ('mean_rev', meanrev_sig)]:
        r = row(f'sess+{bias}_hz{hz}_tgt{tgt}', hz, tgt, 10, mask_full, sig)
        r['exp'] = f'D_sess+{bias}'; r['ez'] = round(tgt/(2*hz), 2)
        rows.append(r)
        g = r['gates']; v = r['verdict']
        print(f"  sess+{bias:10s} hz={hz:2d} tgt={tgt:2d} | {g}/5 {v} | "
              f"n={r['n']:4d} avgL={r['avg_legs']:.1f} esc={r['esc_pct']:.0f}% "
              f"pnl=${r['pnl_$']:+,.0f} sh={r['sharpe']:+.4f}")

df_res = pd.DataFrame(rows).sort_values(['gates','sharpe'], ascending=False)
df_res.to_csv('results/phase5_sweep.csv', index=False)
print('\n=== TOP RESULTS ===')
top = df_res[df_res['gates'] >= 4]
print(top[['label','exp','ez','gates','verdict','avg_legs','esc_pct','pnl_$','sharpe','sqn']].to_string(index=False))
