"""E/Z ratio sweep: hz=30,35,40, E/Z=0.30-0.90, ml=10, convex, random direction."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from data_utils import load_m5, prepare_features
from engine import ZoneRecoveryEngine
from backtest import run_backtest, compute_metrics, run_5gate_validation

PAIR = 'EUR_USD'
raw = load_m5(PAIR)
features = prepare_features(raw)
n = len(features["close"])
n_train = int(n * 0.7)
train_f = {k: v[:n_train] for k, v in features.items()}
test_f  = {k: v[n_train:] for k, v in features.items()}

rows = []
ez_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
hz_values = [30, 35, 40]

for hz in hz_values:
    for ez in ez_values:
        tgt = round(hz * 2 * ez)
        if tgt < 5:
            continue
        label = f"hz={hz} tgt={tgt} E/Z={ez:.2f}"
        eng = ZoneRecoveryEngine(
            mode='classic',
            half_zone_pips=hz,
            target_beyond_pips=tgt,
            init_target_pips=9999,
            max_legs=10,
            sizing_mode='convex',
            convex_exponent=1.5,
            spread_pips=1.4,
        )
        r = run_5gate_validation(eng, features, train_f, test_f, seed=42)
        m = r['oos_metrics']
        g = r['gates_passed']
        v = r['verdict']
        pnl_dollars = m['net_pnl_pips'] * 0.01  # 1 base_unit = 1000u = $0.01/pip
        print(f"  {label} | {g}/5 {'PASS' if v=='PASS' else 'FAIL'} | "
              f"avgL={m['avg_legs']:.1f} esc={100*m['max_drawdown_pips']/max(1,m['net_pnl_pips']):.0f}% "
              f"esc_rate={m['exit_reasons'].get('target',0)/max(1,m['n_cycles'])*100:.1f}% "
              f"pnl=${pnl_dollars:+,.0f} sh={m['sharpe']:+.4f} sqn={m['sqn']:.2f}")
        rows.append({
            'hz': hz, 'tgt': tgt, 'ez': ez, 'gates': g, 'verdict': v,
            'n': m['n_cycles'], 'avg_legs': m['avg_legs'],
            'esc_pct': m['exit_reasons'].get('target',0)/max(1,m['n_cycles'])*100,
            'max_legs_pct': m['exit_reasons'].get('max_legs',0)/max(1,m['n_cycles'])*100,
            'pnl_$': pnl_dollars, 'sharpe': m['sharpe'], 'sqn': m['sqn'],
            'wr': m['win_rate']*100, 'max_dd': m['max_drawdown_pips']*0.01,
        })

df = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
print("\n=== TOP RESULTS (by Sharpe, 5/5 gates) ===")
print(df[df['verdict']=='PASS'].to_string(index=False))
print("\n=== ALL RESULTS ===")
print(df.to_string(index=False))
df.to_csv('results/phase6_ez_sweep.csv', index=False)
print("\nSaved: results/phase6_ez_sweep.csv")
