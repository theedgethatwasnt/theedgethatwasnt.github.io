"""
Boundary-entry vs center-entry zone recovery comparison.
Sweeps zone_width=20-55p, target_beyond=20-90p (E/Z=0.5-2.5) on EUR_USD M5.
Runs 5-gate validation on both engine variants side-by-side.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from data_utils import load_m5, prepare_features
from engine import ZoneRecoveryEngine
from boundary_engine import BoundaryZoneEngine
from backtest import run_5gate_validation, compute_metrics, run_backtest

PAIR = 'EUR_USD'
raw = load_m5(PAIR)
features = prepare_features(raw)
n = len(features['close'])
n_train = int(n * 0.7)
train_f = {k: v[:n_train] for k, v in features.items()}
test_f  = {k: v[n_train:] for k, v in features.items()}

rows = []

# Parameter grid
zone_widths    = [20, 25, 30, 35, 40, 45, 50, 55]
ez_ratios      = [0.50, 0.70, 0.85, 1.00, 1.25, 1.50, 2.00, 2.50]

print(f"{'label':<38} {'variant':<10} {'gates':>5} {'n':>5} "
      f"{'avgL':>5} {'esc%':>5} {'pnl$':>9} {'sh':>7} {'sqn':>6}")
print("-"*95)

for zw in zone_widths:
    for ez in ez_ratios:
        tgt = round(zw * ez)
        if tgt < 5:
            continue

        label = f"zw={zw:2d} tgt={tgt:3d} ez={ez:.2f}"

        for variant, eng in [
            ("center",   ZoneRecoveryEngine(
                            mode='classic',
                            half_zone_pips=zw/2,  # half_zone so full zone = zw
                            target_beyond_pips=tgt,
                            init_target_pips=9999,
                            max_legs=10,
                            sizing_mode='convex', convex_exponent=1.5,
                            spread_pips=1.4)),
            ("boundary", BoundaryZoneEngine(
                            zone_width_pips=zw,
                            target_beyond_pips=tgt,
                            max_legs=10,
                            convex_exponent=1.5,
                            spread_pips=1.4)),
        ]:
            r = run_5gate_validation(eng, features, train_f, test_f, seed=42)
            m = r['oos_metrics']
            g = r['gates_passed']
            v = r['verdict']
            pnl = m['net_pnl_pips'] * 0.01
            esc = m['exit_reasons'].get('target', 0) / max(1, m['n_cycles']) * 100
            print(f"  {label:<36} {variant:<10} {g:>5} {m['n_cycles']:>5} "
                  f"{m['avg_legs']:>5.1f} {esc:>5.1f} {pnl:>+9,.0f} "
                  f"{m['sharpe']:>+7.4f} {m['sqn']:>6.2f}")
            rows.append({
                'zone_width': zw, 'tgt': tgt, 'ez': ez,
                'variant': variant, 'gates': g, 'verdict': v,
                'n': m['n_cycles'], 'avg_legs': m['avg_legs'],
                'esc_pct': esc, 'pnl_$': pnl,
                'sharpe': m['sharpe'], 'sqn': m['sqn'],
                'wr': m['win_rate'] * 100,
                'max_dd': m['max_drawdown_pips'] * 0.01,
            })

df = pd.DataFrame(rows)
df.to_csv('results/phase7_boundary_vs_center.csv', index=False)

print("\n=== TOP 15 overall (5/5 gates, by Sharpe) ===")
top = df[df['verdict'] == 'PASS'].sort_values('sharpe', ascending=False).head(15)
print(top[['zone_width','tgt','ez','variant','gates','n','avg_legs','esc_pct',
           'pnl_$','sharpe','sqn']].to_string(index=False))

print("\n=== BOUNDARY vs CENTER: avg Sharpe across 5/5 PASS configs ===")
for v in ['center', 'boundary']:
    sub = df[(df['verdict']=='PASS') & (df['variant']==v)]
    print(f"  {v:10s}: {len(sub):3d} passing configs  "
          f"mean_sharpe={sub['sharpe'].mean():.4f}  "
          f"mean_sqn={sub['sqn'].mean():.2f}  "
          f"best_sharpe={sub['sharpe'].max():.4f}")
