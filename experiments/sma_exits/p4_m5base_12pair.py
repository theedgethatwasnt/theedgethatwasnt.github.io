"""P4 — 12-pair M5-base sweep with H1/H4 anchors.

Tests whether using M5 as the base TF (slower than S5 but available for all
12 pairs) yields enough OOS edge per pair to compensate for the cadence loss,
and whether wider H4 anchors add edge on top of H1.

Combos (4):
  M5/M30/H1   M5/H1/H4    M5/M30/H4   M5/M15/H4

12 pairs (full M5 parquet coverage). Uses an extended PAIRS dict that includes
CHF_JPY and NZD_JPY (not in default _lib.PAIRS).
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as h17  # mutate PAIRS at module level for full 12-pair run
from h17_stack_alignment import SMA_COMBOS, M_EXIT, TP_GRID, tf_signal, kernel

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")
M5_DIR = PROJECT / "data" / "m5_ohlc"

# Extend PAIRS to all 12 — add CHF_JPY + NZD_JPY with sensible spread gates
EXT_PAIRS = dict(h17.PAIRS)
EXT_PAIRS.setdefault("CHF_JPY", (0.01,   2.80))
EXT_PAIRS.setdefault("NZD_JPY", (0.01,   2.50))

# Monkey-patch PAIRS for the duration of this run so run_tf_combo iterates
# all 12 pairs.
h17.PAIRS = EXT_PAIRS

TF_PAIRINGS = [
    ("M5/M30/H1",  M5_DIR, "_M5.parquet",  5.0,  30,  60, 51840, None),
    ("M5/H1/H4",   M5_DIR, "_M5.parquet",  5.0,  60, 240, 51840, None),
    ("M5/M30/H4",  M5_DIR, "_M5.parquet",  5.0,  30, 240, 51840, None),
    ("M5/M15/H4",  M5_DIR, "_M5.parquet",  5.0,  15, 240, 51840, None),
]


def main():
    print("="*100)
    print(f"  P4 — 12-pair M5-base sweep (H1/H4 anchors)")
    print(f"  SMA combos ({len(SMA_COMBOS)})   M_exit: {M_EXIT}   TP: {TP_GRID}")
    print(f"  TF combos: {len(TF_PAIRINGS)}   pairs: {len(EXT_PAIRS)}")
    rows_per_combo = len(EXT_PAIRS) * len(SMA_COMBOS) * len(M_EXIT) * len(TP_GRID)
    print(f"  Expected rows: {len(TF_PAIRINGS) * rows_per_combo}")
    print("="*100, flush=True)

    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c, _c, _c, _c, _s, _s, _s, _s,
           _c, _c, _c, _c, _c, _c, _c, _c,
           0.0001, 1, 20.0, 1)

    all_rows = []; t0 = time.time()
    for cfg in TF_PAIRINGS:
        label = cfg[0]
        t1 = time.time()
        rows = h17.run_tf_combo(*cfg)
        all_rows.extend(rows)
        rdf = pd.DataFrame(rows) if rows else pd.DataFrame()
        cand = rdf[(rdf.is_net>0)&(rdf.oos_net>0)] if not rdf.empty else rdf
        bp = (cand.sort_values(['pair','oos_pd'], ascending=[True,False])
                  .groupby('pair').head(1)) if not cand.empty else pd.DataFrame()
        dur = time.time() - t1
        n_pairs = rdf['pair'].nunique() if not rdf.empty else 0
        print(f"  {label:<14}  {dur:5.1f}s  {len(rows):5d} rows  "
              f"{len(bp) if len(bp) else 0}/{n_pairs} pairs IS+OOS+  "
              f"Σ OOS pd = {bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    out_path = OUT / 'p4_m5base_12pair.csv'
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
    print("  BEST IS+OOS+ ACROSS P4 M5-BASE COMBOS, per pair")
    print("="*100)
    print(f"  {'Pair':<9} {'TF':<14} {'SMA':<10} {'M':>2} {'TP':>4} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  {'days':>6}")
    for _, r in bp_all.iterrows():
        print(f"  {r['pair']:<9} {r['tf_label']:<14} {r['sma']:<10} "
              f"{int(r['M_exit']):>2d} {r['tp_pips']:>4.0f} "
              f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
              f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  {r['days']:>6.1f}")
    print(f"\n  Total: {len(bp_all)} pairs IS+OOS+  "
          f"Σ OOS pd: {bp_all['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
