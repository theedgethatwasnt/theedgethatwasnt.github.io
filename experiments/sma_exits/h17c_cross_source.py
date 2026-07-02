"""H17c — Lower-TF base + clock-aligned higher-TF signals from m5_ohlc.

WHY THIS EXISTS.  H17d uses numpy-bin resample which drifts from clock
alignment (e.g., an "H1 bar" might span 09:33→10:33 instead of 10:00→11:00).
For research conclusions to translate to live, upper TFs should match the
live curator's data path: M5 → M15/M30/H1 resampled to clock boundaries.

DESIGN.
  - Base TF (S5 or M1): provides per-bar OHLC for trade simulation
  - Upper TFs (M5/M15/M30/H1): resampled from m5_ohlc with pandas (clock-aligned)
  - Overlap range: intersection of [base.first, base.last] and [m5.first, m5.last]
  - Project upper-TF signal arrays onto base timeline via timestamp searchsorted
  - Same SMA-momentum stack-alignment rule as H17 v3

TF pairings:
  S5 base × (M5+M15) (M5+M30) (M5+H1) (M15+M30) (M15+H1) (M30+H1)    5 pairs
  M1 base × (M5+M15) (M15+M30) (M15+H1) (M30+H1)                      1 pair
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import (tf_signal, kernel, novelty, fast_tail_read,
                                   resample_minutes)
from h17d_full_history import fast_full_read
from _lib import PAIRS, IS_FRAC, sma, project_to_m5, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

SMA_COMBOS = [
    (5, 10, 22), (5, 15, 35), (5, 22, 50),
    (7, 15, 35), (7, 22, 50),
    (10, 22, 50),
]
M_EXIT  = [0, 1]
TP_GRID = [15.0, 20.0, 30.0]

S5_DIR = PROJECT / "data/s5_ohlc"
M1_DIR = PROJECT / "data/m1_ohlc"
M5_DIR = PROJECT / "data/m5_ohlc"
S5_PAIRS = ["EUR_JPY","EUR_USD","GBP_JPY","GBP_USD","USD_JPY"]
M1_PAIRS = ["EUR_USD"]

# (label, tf1_minutes, tf2_minutes)
S5_TF_COMBOS = [
    ("S5/M5/M15",   5,  15),
    ("S5/M5/M30",   5,  30),
    ("S5/M5/H1",    5,  60),
    ("S5/M15/M30", 15,  30),
    ("S5/M15/H1",  15,  60),
    ("S5/M30/H1",  30,  60),
]
M1_TF_COMBOS = [
    ("M1/M5/M15",   5,  15),
    ("M1/M15/M30", 15,  30),
    ("M1/M15/H1",  15,  60),
    ("M1/M30/H1",  30,  60),
]


def load_m5_for_pair(pair, t_start, t_end, history_pad_days=14):
    """Load m5_ohlc tail covering [t_start - pad, t_end].
    pad ensures enough history for SMA(50) + lookback on the upper TF."""
    path = M5_DIR / f"{pair}_M5.parquet"
    if not path.exists():
        return None
    # m5_ohlc has 5+ years; we want a slice around the base data range.
    # Read all and filter — m5_ohlc is small enough.
    df = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                         max_rows=None)
    pad = pd.Timedelta(days=history_pad_days)
    mask = (df['timestamp'] >= (pd.Timestamp(t_start) - pad)) & (df['timestamp'] <= pd.Timestamp(t_end))
    return df[mask].reset_index(drop=True)


def run_one(pair, base_dir, suffix, label, tf1_min, tf2_min,
            base_min_per_bar, max_rows):
    """Cross-source: load base from base_dir, upper TFs resampled from m5_ohlc."""
    path = base_dir / f"{pair}{suffix}"
    if not path.exists():
        return [], 0
    t_load = time.time()
    df_base = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                              max_rows=max_rows)
    if len(df_base) < 5000:
        return [], 0
    base_ts = df_base['timestamp'].to_numpy()
    t_start, t_end = base_ts.min(), base_ts.max()
    # Load m5_ohlc for the same date range
    df_m5 = load_m5_for_pair(pair, t_start, t_end, history_pad_days=14)
    if df_m5 is None or len(df_m5) < 2000:
        return [], 0
    print(f"  {pair}  base={len(df_base):,}  m5={len(df_m5):,}  "
          f"range={pd.Timestamp(t_start).date()}→{pd.Timestamp(t_end).date()}  "
          f"({time.time()-t_load:.1f}s)", flush=True)

    opens  = df_base['open'].to_numpy(np.float64)
    highs  = df_base['high'].to_numpy(np.float64)
    lows   = df_base['low'].to_numpy(np.float64)
    closes = df_base['close'].to_numpy(np.float64)
    n_base = len(df_base)
    is_end = int(n_base * IS_FRAC)
    days = n_base * base_min_per_bar / 1440

    # Resample m5_ohlc → TF1 and TF2 (pandas, on small ~52K row frame — fast)
    tf1 = resample_minutes(df_m5, tf1_min, 5)
    tf2 = resample_minutes(df_m5, tf2_min, 5)
    tf1_c = tf1['close'].to_numpy(np.float64); tf1_ts = tf1['timestamp'].to_numpy()
    tf2_c = tf2['close'].to_numpy(np.float64); tf2_ts = tf2['timestamp'].to_numpy()

    # prev_ts = base bar's "as-of" timestamp for projection (use previous bar's ts to avoid peek)
    prev_ts = np.empty_like(base_ts)
    prev_ts[0] = base_ts[0]
    prev_ts[1:] = base_ts[:-1]

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

    rows = []
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
                net = p - sp_cost
                is_mask = e < is_end; oos_mask = ~is_mask
                is_days = (is_end / n_base) * days; oos_days = days - is_days
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
    return rows, days


def main():
    print(f"H17c — cross-source: lower-TF base + clock-aligned m5_ohlc upper TFs")
    print(f"  S5 pairs: {S5_PAIRS}   M1 pairs: {M1_PAIRS}")
    print(f"  S5 TF combos: {len(S5_TF_COMBOS)}   M1 TF combos: {len(M1_TF_COMBOS)}")
    # JIT warmup
    import numpy as np
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c,_c,_c,_c,_s,_s,_s,_s,_c,_c,_c,_c,_c,_c,_c,_c,0.0001,1,20.0,1)

    all_rows = []; t0 = time.time()

    for label, tf1_min, tf2_min in S5_TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in S5_PAIRS:
            rows, days = run_one(pair, S5_DIR, "_S5_BA.parquet",
                                  label, tf1_min, tf2_min, 5/60, None)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0)&(df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
                      f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  N={int(r['oos_n']):>4}  "
                      f"WR={r['oos_wr']:>5.1f}", flush=True)

    for label, tf1_min, tf2_min in M1_TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in M1_PAIRS:
            rows, days = run_one(pair, M1_DIR, "_M1_BA.parquet",
                                  label, tf1_min, tf2_min, 1.0, 500000)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0)&(df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
                      f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  N={int(r['oos_n']):>4}  "
                      f"WR={r['oos_wr']:>5.1f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT/'h17c_cross_source.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}", flush=True)

    cand = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n=== Cross-TF best IS+OOS+ per pair ===")
    for _, r in bp.iterrows():
        print(f"  {r['pair']:<9}{r['tf_label']:<13}{r['sma']:<10}"
              f"M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
              f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  "
              f"N={int(r['oos_n']):>4}  WR={r['oos_wr']:>5.1f}  days={r['days']:.0f}")
    print(f"\n  Σ OOS pd: {bp['oos_pd'].sum():+.2f}  on {len(bp)} pairs")


if __name__ == '__main__':
    main()
