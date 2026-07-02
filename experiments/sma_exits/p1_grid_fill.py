"""P1 — Fill 3-TF S5-base grid gaps (full-history rerun).

ORIGINAL run was on max_rows=200000 (~11.6d), producing tiny OOS samples
(median 4 trades) that don't survive MC. Rewritten to use h17d_full_history's
run_one with full file (286-478d per pair), matching the realistic stats
shown in results/h17d_full_history.csv.

Combos added (none tested by h17/h17b/h17d/h17e/h17f/h17g):
  S5/S30/M5    S5/S30/M15   S5/S30/M30   S5/S30/H1
  S5/M1/M15    S5/M1/M30    S5/M1/H1
  S5/M2/M15    S5/M2/M30
  S5/M5/M30    S5/M5/H1
  S5/M10/M30   S5/M10/H1

5 pairs (S5 BA available).
"""
import sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as h17
from h17_stack_alignment import SMA_COMBOS, M_EXIT, TP_GRID, S5_DIR, S5_PAIRS, tf_signal, kernel
from h17d_full_history import run_one, fast_full_read, bin_resample, project_via_index

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

# 13 new 3-TF combos.  base = S5, n_base_per_min = 12, base_min_per_bar = 5/60
TF_COMBOS = [
    ("S5/S30/M5",  0.5,   5),
    ("S5/S30/M15", 0.5,  15),
    ("S5/S30/M30", 0.5,  30),
    ("S5/S30/H1",  0.5,  60),
    ("S5/M1/M15",    1,  15),
    ("S5/M1/M30",    1,  30),
    ("S5/M1/H1",     1,  60),
    ("S5/M2/M15",    2,  15),
    ("S5/M2/M30",    2,  30),
    ("S5/M5/M30",    5,  30),
    ("S5/M5/H1",     5,  60),
    ("S5/M10/M30",  10,  30),
    ("S5/M10/H1",   10,  60),
]


def main():
    print("="*100)
    print(f"  P1 — Fill 3-TF S5-base grid gaps (FULL HISTORY)")
    print(f"  SMA combos ({len(SMA_COMBOS)}): {SMA_COMBOS}")
    print(f"  M_exit: {M_EXIT}   TP_pips: {TP_GRID}")
    print(f"  TF combos: {len(TF_COMBOS)}   pairs: {sorted(S5_PAIRS)}")
    rows_per_combo = len(S5_PAIRS) * len(SMA_COMBOS) * len(M_EXIT) * len(TP_GRID)
    print(f"  Expected rows: {len(TF_COMBOS) * rows_per_combo}")
    print("="*100, flush=True)

    # JIT warmup
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
            n_pairs = df['pair'].nunique()
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{n_pairs} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
                      f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  N={int(r['oos_n']):>4}  "
                      f"WR={r['oos_wr']:>5.1f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    out_path = OUT / 'p1_grid_fill.csv'
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
    print("  BEST IS+OOS+ ACROSS P1 TF COMBOS, per pair")
    print("="*100)
    print(f"  {'Pair':<9} {'TF':<14} {'SMA':<10} {'M':>2} {'TP':>4} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  {'days':>6}")
    for _, r in bp_all.iterrows():
        print(f"  {r['pair']:<9} {r['tf_label']:<14} {r['sma']:<10} "
              f"{int(r['M_exit']):>2d} {r['tp_pips']:>4.0f} "
              f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
              f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  {r['days']:>6.1f}")
    print(f"\n  Total: {len(bp_all)} pairs IS+OOS+  Σ OOS pd: {bp_all['oos_pd'].sum():+.2f}")

    print()
    print("  ── Per-TF combo summary (best per pair, summed) ──")
    for label, _, _ in TF_COMBOS:
        sub = df_all[df_all.tf_label == label]
        c = sub[(sub.is_net>0)&(sub.oos_net>0)]
        bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
        print(f"    {label:<14}  {len(bp) if len(bp) else 0:>2}/{sub['pair'].nunique():>2} pairs "
              f"Σ OOS pd = {bp['oos_pd'].sum() if len(bp) else 0:+.2f}")


if __name__ == '__main__':
    main()
