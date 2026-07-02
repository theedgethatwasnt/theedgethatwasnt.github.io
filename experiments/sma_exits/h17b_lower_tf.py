"""H17b — Lower-TF complement to H17 v3.

Runs ONLY S5/M1-base TF pairings (skipping M5 which v3 covered).
Smaller SMA grid + smaller data window so it actually finishes:

  SMA combos: 6 representative configs spanning user's typical ranges
  Window:     S5 → last 100K bars (~5 trading days)
              M1 → last 50K bars  (~35 trading days)
  TP_GRID, M_EXIT, novelty/monotone/slope rules identical to v3.

TF pairings:
  S5/S30/M1   (5 pairs)
  S5/M1/M5    (5 pairs)
  M1/M2/M5    (EUR_USD only)
  M1/M5/M15   (EUR_USD only)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import (tf_signal, kernel, novelty,
                                   resample_minutes, fast_tail_read)
from _lib import PAIRS, IS_FRAC, sma, project_to_m5, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

# Smaller SMA grid — representative coverage of user's ranges
SMA_COMBOS = [
    (5, 10, 22),
    (5, 15, 35),
    (5, 22, 50),
    (7, 15, 35),
    (7, 22, 50),
    (10, 22, 50),
]
M_EXIT  = [0, 1]
TP_GRID = [15.0, 20.0, 30.0]

S5_DIR = PROJECT / "data/s5_ohlc"
M1_DIR = PROJECT / "data/m1_ohlc"
S5_PAIRS = {"USD_JPY","GBP_USD","EUR_USD","GBP_JPY","EUR_JPY"}
M1_PAIRS = {"EUR_USD"}

TF_PAIRINGS = [
    ("S5/S30/M1",  S5_DIR, "_S5_BA.parquet",  5/60,  0.5,  1, 100000, S5_PAIRS),
    ("S5/M1/M5",   S5_DIR, "_S5_BA.parquet",  5/60,    1,  5, 100000, S5_PAIRS),
    ("M1/M2/M5",   M1_DIR, "_M1_BA.parquet",  1.0,     2,  5,  50000, M1_PAIRS),
    ("M1/M5/M15",  M1_DIR, "_M1_BA.parquet",  1.0,     5, 15,  50000, M1_PAIRS),
]


def run_one_tf(label, base_dir, suffix, base_min, tf1_min, tf2_min,
               max_rows, pairs_filter):
    rows = []
    pair_list = [p for p in PAIRS if pairs_filter is None or p in pairs_filter]
    pip_of  = lambda p: PAIRS[p][0]
    sp_cost = lambda p: PAIRS[p][1] * SPREAD_FRAC

    for pair in pair_list:
        path = base_dir / f"{pair}{suffix}"
        if not path.exists():
            continue
        df = fast_tail_read(path, max_rows).sort_values('timestamp').reset_index(drop=True)
        if len(df) < 3000:
            print(f"  [skip] {pair}: only {len(df)} bars")
            continue

        opens  = df['open'].to_numpy(np.float64)
        highs  = df['high'].to_numpy(np.float64)
        lows   = df['low'].to_numpy(np.float64)
        closes = df['close'].to_numpy(np.float64)
        ts     = df['timestamp'].to_numpy()
        n = len(df); is_end = int(n * IS_FRAC)
        days = n * base_min / 1440

        t_resamp = time.time()
        tf1 = resample_minutes(df, tf1_min, base_min)
        tf2 = resample_minutes(df, tf2_min, base_min)
        print(f"  {pair} n={n:,} tf1={len(tf1)} tf2={len(tf2)} (resample {time.time()-t_resamp:.1f}s)",
              flush=True)
        tf1_c = tf1['close'].to_numpy(np.float64); tf1_ts = tf1['timestamp'].to_numpy()
        tf2_c = tf2['close'].to_numpy(np.float64); tf2_ts = tf2['timestamp'].to_numpy()
        prev_ts = np.empty_like(ts); prev_ts[0]=ts[0]; prev_ts[1:]=ts[:-1]
        sp = sp_cost(pair); pip = pip_of(pair)

        for (n_sm, n_md, n_lg) in SMA_COMBOS:
            t1_sm = sma(tf1_c, n_sm); t1_md = sma(tf1_c, n_md); t1_lg = sma(tf1_c, n_lg)
            t2_sm = sma(tf2_c, n_sm); t2_md = sma(tf2_c, n_md); t2_lg = sma(tf2_c, n_lg)
            t1_long = tf_signal(tf1_c, t1_sm, t1_md, t1_lg, 1)
            t1_shrt = tf_signal(tf1_c, t1_sm, t1_md, t1_lg, 0)
            t2_long = tf_signal(tf2_c, t2_sm, t2_md, t2_lg, 1)
            t2_shrt = tf_signal(tf2_c, t2_sm, t2_md, t2_lg, 0)
            t1_long_nov = novelty(t1_long); t1_shrt_nov = novelty(t1_shrt)
            t2_long_nov = novelty(t2_long); t2_shrt_nov = novelty(t2_shrt)
            t1_long_nov_b = project_to_m5(prev_ts, tf1_ts, t1_long_nov).astype(np.int8)
            t1_shrt_nov_b = project_to_m5(prev_ts, tf1_ts, t1_shrt_nov).astype(np.int8)
            t2_long_nov_b = project_to_m5(prev_ts, tf2_ts, t2_long_nov).astype(np.int8)
            t2_shrt_nov_b = project_to_m5(prev_ts, tf2_ts, t2_shrt_nov).astype(np.int8)
            t1_sm_b = project_to_m5(prev_ts, tf1_ts, t1_sm)
            t1_md_b = project_to_m5(prev_ts, tf1_ts, t1_md)
            t1_lg_b = project_to_m5(prev_ts, tf1_ts, t1_lg)
            t1_c_b  = project_to_m5(prev_ts, tf1_ts, tf1_c)
            t2_sm_b = project_to_m5(prev_ts, tf2_ts, t2_sm)
            t2_md_b = project_to_m5(prev_ts, tf2_ts, t2_md)
            t2_lg_b = project_to_m5(prev_ts, tf2_ts, t2_lg)
            t2_c_b  = project_to_m5(prev_ts, tf2_ts, tf2_c)
            for M_min in M_EXIT:
                for tp_p in TP_GRID:
                    p, e = kernel(opens, highs, lows, closes,
                                  t1_long_nov_b, t1_shrt_nov_b,
                                  t2_long_nov_b, t2_shrt_nov_b,
                                  t1_sm_b, t1_md_b, t1_lg_b, t1_c_b,
                                  t2_sm_b, t2_md_b, t2_lg_b, t2_c_b,
                                  pip, M_min, tp_p, 1)
                    if len(p) == 0:
                        rows.append({'tf_label':label,'pair':pair,
                                     'sma':f"{n_sm}/{n_md}/{n_lg}",
                                     'M_exit':M_min,'tp_pips':tp_p,
                                     'trades':0,'is_n':0,'oos_n':0,
                                     'is_pd':0,'oos_pd':0,'oos_dd':0,'oos_wr':0,
                                     'is_net':0,'oos_net':0,'days':round(days,1)})
                        continue
                    net = p - sp
                    is_mask = e < is_end; oos_mask = ~is_mask
                    is_days = (is_end / n) * days; oos_days = days - is_days
                    is_net  = float(net[is_mask].sum())
                    oos_net = float(net[oos_mask].sum())
                    if oos_mask.sum() > 0:
                        cum = net[oos_mask].cumsum()
                        oos_dd = float((cum - np.maximum.accumulate(cum)).min())
                        oos_wr = float((net[oos_mask] > 0).mean() * 100)
                    else:
                        oos_dd = 0.0; oos_wr = 0.0
                    rows.append({'tf_label':label,'pair':pair,
                                 'sma':f"{n_sm}/{n_md}/{n_lg}",
                                 'M_exit':M_min,'tp_pips':tp_p,
                                 'trades':int(len(p)),
                                 'is_n':int(is_mask.sum()),'oos_n':int(oos_mask.sum()),
                                 'is_net':round(is_net,1),'oos_net':round(oos_net,1),
                                 'is_pd':round(is_net/max(is_days,1),2),
                                 'oos_pd':round(oos_net/max(oos_days,1),2),
                                 'oos_dd':round(oos_dd,1),'oos_wr':round(oos_wr,1),
                                 'days':round(days,1)})
    return rows


def main():
    print(f"H17b — lower-TF (S5/M1) focused sweep")
    print(f"  {len(SMA_COMBOS)} SMA combos × {len(M_EXIT)} M × {len(TP_GRID)} TP × {len(TF_PAIRINGS)} TF")
    # warm JIT
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c,_c,_c,_c,_s,_s,_s,_s,_c,_c,_c,_c,_c,_c,_c,_c,0.0001,1,20.0,1)

    all_rows = []; t0 = time.time()
    for cfg in TF_PAIRINGS:
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
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}",
                  flush=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT/'h17b_lower_tf.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}")

    if df_all.empty:
        print("(no rows)"); return

    cand = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\nBest IS+OOS+ per pair, lower-TF:")
    for _, r in bp.iterrows():
        print(f"  {r['pair']:<9}{r['tf_label']:<13}{r['sma']:<10} "
              f"M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
              f"OOSpd={r['oos_pd']:+.1f}  DD={r['oos_dd']:+.0f}  "
              f"N={int(r['oos_n'])}  WR={r['oos_wr']:.0f}%")


if __name__ == '__main__':
    main()
