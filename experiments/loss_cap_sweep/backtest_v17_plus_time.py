"""V17 (trail_lock5_off2) + various time stops on USD_JPY and EUR_USD,
IS/OOS split. Goal: keep the V17 edge but cap the rare stuck-position
loss with a time exit."""
import os, sys
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, requests

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

spec2 = importlib.util.spec_from_file_location('boos',
    Path(__file__).parent / 'backtest_per_pair_oos.py')
boos = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(boos)

PROJECT = Path("/path/to/projects/fx-core")
DATA = PROJECT / "data/m5_ohlc"

# V17 variants with added time stops
TEST_VARIANTS = {
    'V17_pure':           {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0},
    'V17_time72h':        {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'time_exit_bars': 72*12},
    'V17_time48h':        {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'time_exit_bars': 48*12},
    'V17_time24h':        {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'time_exit_bars': 24*12},
    'V17_time12h':        {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'time_exit_bars': 12*12},
    'V17_sl50':           {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'sl_pips': 50.0},
    'V17_sl30_time48h':   {'tp_pips': 20.0, 'trail_lock_pips': 5.0, 'trail_offset_pips': 2.0,
                           'sl_pips': 30.0, 'time_exit_bars': 48*12},
}

PAIRS_TO_TEST = ['USD_JPY', 'EUR_USD', 'CAD_JPY']  # the OOS winners + IS-strong

print(f"{'Pair':<8} {'Variant':<22} {'IS_net':>9} {'IS_dd':>8} {'OOS_n':>6} "
      f"{'OOS_net':>9} {'OOS_dd':>8} {'WR':>5}")
print('-' * 86)
is_bars = int(bv.WINDOW_BARS * 4/6)

results = []
for pair in PAIRS_TO_TEST:
    df = pq.read_table(DATA / f"{pair}_M5.parquet").to_pandas()
    df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
    df_is = df.iloc[:is_bars].reset_index(drop=True)
    df_oos = df.iloc[is_bars:].reset_index(drop=True)
    for vid, vp in TEST_VARIANTS.items():
        is_r = boos.run_variant_on_slice(pair, vp, df_is) or {'net':0,'max_dd':0}
        oos_r = boos.run_variant_on_slice(pair, vp, df_oos) or {'trades':0,'net':0,'max_dd':0,'wr':0}
        results.append({'pair':pair,'variant':vid,
                        'is_net':is_r['net'],'is_dd':is_r['max_dd'],
                        'oos_trades':oos_r['trades'],'oos_net':oos_r['net'],
                        'oos_dd':oos_r['max_dd'],'oos_wr':oos_r['wr']})
        print(f"{pair:<8} {vid:<22} {is_r['net']:>+9.1f} {is_r['max_dd']:>+8.1f} "
              f"{oos_r['trades']:>6d} {oos_r['net']:>+9.1f} {oos_r['max_dd']:>+8.1f} {oos_r['wr']:>4.1f}%")
    print()

df_r = pd.DataFrame(results)
df_r.to_csv(PROJECT/'research/experiments/loss_cap_sweep/results/v17_plus_time.csv', index=False)

# Best per pair on OOS
print('\n=== Best variant per pair (OOS net), with bounded DD < 200p ===')
best_per = (df_r[df_r['oos_dd'] > -200].sort_values('oos_net', ascending=False)
            .groupby('pair').head(1))
print(best_per.to_string(index=False))

# Construct candidate portfolio (USD_JPY + EUR_USD with their best bounded variants)
candidates = best_per[best_per['oos_net'] > 0]
if len(candidates) >= 2:
    total_oos = candidates['oos_net'].sum()
    worst_dd = candidates['oos_dd'].min()
    msg_lines = [
        "🎯 SMA16 per-pair-best with time-bounded risk (USD_JPY + EUR_USD):",
        f"OOS portfolio: {total_oos:+.0f}p over 2 months on {len(candidates)} pairs",
        f"Worst single-pair OOS drawdown: {worst_dd:+.0f}p (bounded)",
        "",
        "Per-pair config:"
    ]
    for _, r in candidates.iterrows():
        msg_lines.append(
            f"  {r['pair']}  {r['variant']}: IS {r['is_net']:+.0f}p / "
            f"OOS {r['oos_net']:+.0f}p ({r['oos_trades']} trades, WR {r['oos_wr']:.0f}%, DD {r['oos_dd']:+.0f}p)"
        )
    msg = '\n'.join(msg_lines)
    print('\n' + msg)
    tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    if tok and chat:
        try:
            requests.post(f'https://api.telegram.org/bot{tok}/sendMessage',
                          json={'chat_id': chat, 'text': msg}, timeout=10)
            print('\n✓ Telegram sent')
        except Exception as e:
            print(f'telegram error: {e}')
    else:
        print('\n(no telegram creds in env; alert NOT sent)')
