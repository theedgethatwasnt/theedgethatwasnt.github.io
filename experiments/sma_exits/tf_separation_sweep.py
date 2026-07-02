#!/usr/bin/env python3
"""
TF-separation robustness sweep for GBP_USD and GBP_JPY (the two SMA-stack pairs that
run a 2:1 TF ratio). Question: is a wider, less-redundant TF separation more robust
than 2:1? Reuses h17_stack_alignment's exact entry (stack-align + monotone + novelty
on BOTH TFs) and exit kernel. Holds SMA at the live per-pair values and a single fixed
TP=20 + M_exit=0 as a CONSISTENT testbed across ratios (note: live GBP_USD/GBP_JPY use
PSAR-only exits — this isolates the TF effect, it is not a live-P&L replica).

Robustness = both IS and OOS positive, OOS/IS consistency, and stability across nearby
ratios. Run: python3 tf_separation_sweep.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H

H.SMA_COMBOS = [(7, 22, 50), (5, 15, 35)]     # GBP_USD live / GBP_JPY live
H.M_EXIT     = [0]                            # M_exit=0 dominates (CLAUDE)
H.TP_GRID    = [20.0]                         # fixed exit testbed
LIVE_SMA = {'GBP_USD': '7/22/50', 'GBP_JPY': '5/15/35'}
S5, M5 = H.S5_DIR, H.M5_DIR
GU, GJ = {"GBP_USD"}, {"GBP_JPY"}

# (label, base_dir, suffix, base_min, tf1_min, tf2_min, max_rows, filter, ratio)
PAIRINGS = [
    ("S5/S30/M1",  S5, "_S5_BA.parquet", 5/60, 0.5,  1, 2_000_000, GU,  2),   # current
    ("S5/S30/M2",  S5, "_S5_BA.parquet", 5/60, 0.5,  2, 2_000_000, GU,  4),
    ("S5/M1/M3",   S5, "_S5_BA.parquet", 5/60,   1,  3, 2_000_000, GU,  3),
    ("S5/M1/M5",   S5, "_S5_BA.parquet", 5/60,   1,  5, 2_000_000, GU,  5),
    ("S5/M1/M10",  S5, "_S5_BA.parquet", 5/60,   1, 10, 2_000_000, GU, 10),
    ("S5/M2/M10",  S5, "_S5_BA.parquet", 5/60,   2, 10, 2_000_000, GU,  5),
    ("M5/M30/H1",  M5, "_M5.parquet",    5.0,   30, 60,   300_000, GJ,  2),   # current
    ("M5/M30/H2",  M5, "_M5.parquet",    5.0,   30,120,   300_000, GJ,  4),
    ("M5/M30/H4",  M5, "_M5.parquet",    5.0,   30,240,   300_000, GJ,  8),
    ("M5/H1/H4",   M5, "_M5.parquet",    5.0,   60,240,   300_000, GJ,  4),
    ("M5/M15/H1",  M5, "_M5.parquet",    5.0,   15, 60,   300_000, GJ,  4),
]
RATIO = {p[0]: p[8] for p in PAIRINGS}
CURRENT = {"GBP_USD": "S5/S30/M1", "GBP_JPY": "M5/M30/H1"}

_c = np.zeros(100); _s = np.zeros(100, np.int8)
H.tf_signal(_c, _c, _c, _c, 1)
H.kernel(_c, _c, _c, _c, _s, _s, _s, _s, _c, _c, _c, _c, _c, _c, _c, _c, 0.0001, 1, 20.0, 1)

rows = []
for cfg in PAIRINGS:
    rows += H.run_tf_combo(*cfg[:8])
df = pd.DataFrame(rows)
df = df[df.apply(lambda r: r['sma'] == LIVE_SMA.get(r['pair']), axis=1)].copy()
df['ratio'] = df['tf_label'].map(RATIO)
df['IS+OOS+'] = (df['is_net'] > 0) & (df['oos_net'] > 0)
df['oos/is'] = (df['oos_pd'] / df['is_pd'].replace(0, np.nan)).round(2)

for pair in ('GBP_USD', 'GBP_JPY'):
    d = df[df.pair == pair].sort_values('ratio')
    print(f"\n===== {pair}  (live SMA {LIVE_SMA[pair]}, exit=TP20/M_exit0 testbed) =====")
    print(f"  {'TF stack':<12}{'ratio':>6}{'trades':>8}{'IS_pd':>8}{'OOS_pd':>8}{'oos/is':>8}{'OOS_WR':>8}{'OOS_DD':>9}{'IS+OOS+':>9}")
    for _, r in d.iterrows():
        cur = " <- current" if r['tf_label'] == CURRENT[pair] else ""
        print(f"  {r['tf_label']:<12}{r['ratio']:>6}{r['trades']:>8}{r['is_pd']:>8.2f}{r['oos_pd']:>8.2f}"
              f"{(r['oos/is'] if pd.notna(r['oos/is']) else 0):>8.2f}{r['oos_wr']:>7.0f}%{r['oos_dd']:>9.1f}{str(r['IS+OOS+']):>9}{cur}")
out = H.OUT / "tf_separation_sweep.csv"; df.to_csv(out, index=False)
print(f"\n  saved -> {out.name}")
