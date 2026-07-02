"""H17b-extra — additional pure-S5 TF pairings requested mid-flight.

Adds S5/M2/M10 to the lower-TF set, plus a couple of related fast-TF
combos that fit the same architecture (both upper TFs derive from S5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Re-use everything from h17b
import h17b_lower_tf as base
from h17b_lower_tf import S5_DIR, S5_PAIRS, run_one_tf
import numpy as np, pandas as pd, time

# Replace TF_PAIRINGS list with the new combos only
base.TF_PAIRINGS = [
    ("S5/M2/M10", S5_DIR, "_S5_BA.parquet",  5/60,   2, 10, 100000, S5_PAIRS),
    ("S5/M1/M10", S5_DIR, "_S5_BA.parquet",  5/60,   1, 10, 100000, S5_PAIRS),
    ("S5/M2/M5",  S5_DIR, "_S5_BA.parquet",  5/60,   2,  5, 100000, S5_PAIRS),
]


def main():
    print(f"H17b-extra — additional pure-S5 TF pairings ({len(base.TF_PAIRINGS)} combos)")
    # warm JIT (reuse via base's main)
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    from h17_stack_alignment import tf_signal, kernel
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c,_c,_c,_c,_s,_s,_s,_s,_c,_c,_c,_c,_c,_c,_c,_c,0.0001,1,20.0,1)

    all_rows = []; t0 = time.time()
    for cfg in base.TF_PAIRINGS:
        label = cfg[0]
        print(f"\n=== TF {label} ===", flush=True)
        t1 = time.time()
        rows = run_one_tf(*cfg)
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        if len(df):
            c = df[(df.is_net>0)&(df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t1:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])} "
                      f"OOSpd={r['oos_pd']:+.1f} DD={r['oos_dd']:+.0f} N={int(r['oos_n'])} "
                      f"WR={r['oos_wr']:.0f}%")
    df_all = pd.DataFrame(all_rows)
    out = Path(__file__).parent / "results" / "h17b_extra.csv"
    df_all.to_csv(out, index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s   rows: {len(df_all)}")


if __name__ == '__main__':
    main()
