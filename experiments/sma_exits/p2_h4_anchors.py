"""P2 — Wider H4 anchors on S5 base (full-history rerun).

Tests H4 (240-min) as the upper TF using full S5 parquets (286-478d each).

Combos (4), all on S5 base:
  S5/M5/H4    S5/M15/H4    S5/M30/H4    S5/H1/H4

5 pairs (S5 BA available).
"""
import sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import SMA_COMBOS, M_EXIT, TP_GRID, S5_DIR, S5_PAIRS, tf_signal, kernel
from h17d_full_history import run_one

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

TF_COMBOS = [
    ("S5/M5/H4",   5, 240),
    ("S5/M15/H4", 15, 240),
    ("S5/M30/H4", 30, 240),
    ("S5/H1/H4",  60, 240),
]


def main():
    print("="*100)
    print(f"  P2 — H4 anchor on S5 base (FULL HISTORY)")
    print(f"  SMA combos: {len(SMA_COMBOS)}   M_exit: {M_EXIT}   TP: {TP_GRID}")
    print(f"  TF combos: {len(TF_COMBOS)}   pairs: {sorted(S5_PAIRS)}")
    print("="*100, flush=True)

    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c, _c, _c, _c, _s, _s, _s, _s,
           _c, _c, _c, _c, _c, _c, _c, _c,
           0.0001, 1, 20.0, 1)

    all_rows = []; t0 = time.time()
    for label, tf1_min, tf2_min in TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in sorted(S5_PAIRS):
            rows, days = run_one(pair, S5_DIR, "_S5_BA.parquet", 12.0,
                                 label, tf1_min, tf2_min, 5/60, None)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0) & (df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
                      f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  N={int(r['oos_n']):>4}  "
                      f"WR={r['oos_wr']:>5.1f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    out_path = OUT / 'p2_h4_anchors.csv'
    df_all.to_csv(out_path, index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s   rows: {len(df_all)}")
    print(f"  → {out_path}")

    if df_all.empty or 'is_net' not in df_all.columns:
        print("\n  (no rows)"); return

    cand_all = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp_all = cand_all.groupby('pair').head(1)
    print()
    print("="*100)
    print("  BEST IS+OOS+ ACROSS P2 H4-ANCHORED COMBOS, per pair")
    print("="*100)
    print(f"  {'Pair':<9} {'TF':<14} {'SMA':<10} {'M':>2} {'TP':>4} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  {'days':>6}")
    for _, r in bp_all.iterrows():
        print(f"  {r['pair']:<9} {r['tf_label']:<14} {r['sma']:<10} "
              f"{int(r['M_exit']):>2d} {r['tp_pips']:>4.0f} "
              f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
              f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  {r['days']:>6.1f}")
    print(f"\n  Total: {len(bp_all)} pairs IS+OOS+  Σ OOS pd: {bp_all['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
