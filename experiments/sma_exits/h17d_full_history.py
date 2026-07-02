"""H17d — Full-history S5/M1 sweep, all TF combos via fast numpy-bin resample.

Replaces pandas-resample with a vectorised numpy bin aggregator. For each
base bar at index i, upper-TF bar = i // N where N is the integer
bar-multiplier (e.g., S5→M1 N=12, S5→H1 N=720).

This collapses what was going to be split H17c/H17d into one sweep: the
S5 data has 15-month coverage, so every higher TF (up to H1) can be
derived from S5 directly with negligible cost (no pandas).

Window:
  S5 pairs:  full file (5-8M bars, 449-807 trading days)
  M1 pair:   last 500K bars (~350 trading days, EUR_USD only)

TF pairings: pure-source (all upper TFs binned from base).
  S5 base, TF1×TF2 in {(S30,M1), (M1,M5), (M2,M10), (M5,M15),
                        (M15,M30), (M15,H1), (M30,H1)}
  M1 base, TF1×TF2 in {(M5,M15), (M15,M30), (M15,H1), (M30,H1)}

Reuses h17_stack_alignment.tf_signal + kernel for the core logic.
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import (tf_signal, kernel, novelty,
                                   fast_tail_read)
from _lib import PAIRS, IS_FRAC, sma, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

# Smaller SMA grid (representative of user's typical ranges)
SMA_COMBOS = [
    (5, 10, 22), (5, 15, 35), (5, 22, 50),
    (7, 15, 35), (7, 22, 50),
    (10, 22, 50),
]
M_EXIT  = [0, 1]
TP_GRID = [15.0, 20.0, 30.0]

S5_DIR = PROJECT / "data/s5_ohlc"
M1_DIR = PROJECT / "data/m1_ohlc"
S5_PAIRS = ["EUR_JPY","EUR_USD","GBP_JPY","GBP_USD","USD_JPY"]
M1_PAIRS = ["EUR_USD"]

# (label, tf1_minutes_from_base, tf2_minutes_from_base) — base depends on bundle
S5_TF_COMBOS = [
    ("S5/S30/M1",  0.5,   1),
    ("S5/M1/M5",     1,   5),
    ("S5/M2/M10",    2,  10),
    ("S5/M5/M15",    5,  15),
    ("S5/M15/M30",  15,  30),
    ("S5/M15/H1",   15,  60),
    ("S5/M30/H1",   30,  60),
]
M1_TF_COMBOS = [
    ("M1/M5/M15",   5,   15),
    ("M1/M15/M30", 15,   30),
    ("M1/M15/H1",  15,   60),
    ("M1/M30/H1",  30,   60),
]


def bin_resample(opens, highs, lows, closes, n_per_bar):
    """Bin every n_per_bar consecutive base bars into one upper-TF bar.
    Returns 4 arrays (o, h, l, c) of length len(base) // n_per_bar.
    open=first, high=max, low=min, close=last (within each bin)."""
    n = len(closes)
    n_out = n // n_per_bar
    if n_out < 10:
        return None
    trim = n_out * n_per_bar
    o = opens[:trim].reshape(n_out, n_per_bar)[:, 0]
    c = closes[:trim].reshape(n_out, n_per_bar)[:, -1]
    h = highs[:trim].reshape(n_out, n_per_bar).max(axis=1)
    l = lows[:trim].reshape(n_out, n_per_bar).min(axis=1)
    return o, h, l, c


def fast_full_read(path, columns=('timestamp','open','high','low','close'), max_rows=None):
    """Read full parquet (or last max_rows if specified) without pandas resample overhead."""
    pf = pq.ParquetFile(str(path))
    n_rg = pf.metadata.num_row_groups
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    if max_rows is None:
        take = list(range(n_rg))
    else:
        take = []; total = 0
        for i in range(n_rg - 1, -1, -1):
            take.append(i); total += rg_rows[i]
            if total >= max_rows: break
        take.sort()
    tbl = pf.read_row_groups(take, columns=list(columns))
    df = tbl.to_pandas()
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df.sort_values('timestamp').reset_index(drop=True)


def project_via_index(base_idx_into_upper, upper_arr):
    """Forward-fill upper-TF array onto base-bar indices.
    base_idx_into_upper[i] = which upper-TF bar's *previous-completed* bar
    we use at base bar i. We use i//n_per_bar - 1 (i.e., the most recently
    completed upper-TF bar before base bar i)."""
    out = np.full(len(base_idx_into_upper), np.nan, dtype=upper_arr.dtype if upper_arr.dtype.kind=='f' else np.float64)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    out_typed = upper_arr[idx_clipped].astype(np.float64).copy()
    if upper_arr.dtype.kind == 'f':
        out_typed[~valid] = np.nan
    else:
        out_typed[~valid] = 0
    return out_typed


def run_one(pair, base_dir, suffix, n_base_per_min, label,
            tf1_min, tf2_min, base_min_per_bar, max_rows):
    """n_base_per_min: how many base bars per minute (e.g., S5 → 12, M1 → 1)
    base_min_per_bar: minutes per base bar (S5=5/60, M1=1.0) — used for p/d normalization
    """
    path = base_dir / f"{pair}{suffix}"
    if not path.exists():
        return [], 0
    t_load = time.time()
    df = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                         max_rows=max_rows)
    if len(df) < 5000:
        return [], 0
    opens  = df['open'].to_numpy(np.float64)
    highs  = df['high'].to_numpy(np.float64)
    lows   = df['low'].to_numpy(np.float64)
    closes = df['close'].to_numpy(np.float64)
    n_base = len(df)
    is_end = int(n_base * IS_FRAC)
    days = n_base * base_min_per_bar / 1440
    print(f"  {pair} loaded {n_base:,} bars ({days:.0f}d, {time.time()-t_load:.1f}s)", flush=True)

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

    # Pre-compute bin sizes
    n1 = int(round(tf1_min * n_base_per_min))   # base bars per TF1 bar
    n2 = int(round(tf2_min * n_base_per_min))
    if n1 < 2 or n2 < 2:
        return [], days

    # Resample base → TF1 and TF2 via numpy bins
    r1 = bin_resample(opens, highs, lows, closes, n1)
    r2 = bin_resample(opens, highs, lows, closes, n2)
    if r1 is None or r2 is None:
        return [], days
    t1_o, t1_h, t1_l, t1_c = r1
    t2_o, t2_h, t2_l, t2_c = r2

    # base_idx → "which completed upper bar to use at base bar i"
    # = i // n - 1   (previous completed upper bar, no peek)
    base_to_t1 = np.arange(n_base) // n1 - 1
    base_to_t2 = np.arange(n_base) // n2 - 1

    rows = []
    for (n_sm, n_md, n_lg) in SMA_COMBOS:
        t1_sm = sma(t1_c, n_sm); t1_md = sma(t1_c, n_md); t1_lg = sma(t1_c, n_lg)
        t2_sm = sma(t2_c, n_sm); t2_md = sma(t2_c, n_md); t2_lg = sma(t2_c, n_lg)
        t1_long = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 1)
        t1_shrt = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 0)
        t2_long = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 1)
        t2_shrt = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 0)
        t1_long_nov = novelty(t1_long); t1_shrt_nov = novelty(t1_shrt)
        t2_long_nov = novelty(t2_long); t2_shrt_nov = novelty(t2_shrt)
        # Project to base
        t1_long_nov_b = project_via_index(base_to_t1, t1_long_nov).astype(np.int8)
        t1_shrt_nov_b = project_via_index(base_to_t1, t1_shrt_nov).astype(np.int8)
        t2_long_nov_b = project_via_index(base_to_t2, t2_long_nov).astype(np.int8)
        t2_shrt_nov_b = project_via_index(base_to_t2, t2_shrt_nov).astype(np.int8)
        t1_sm_b = project_via_index(base_to_t1, t1_sm)
        t1_md_b = project_via_index(base_to_t1, t1_md)
        t1_lg_b = project_via_index(base_to_t1, t1_lg)
        t1_c_b  = project_via_index(base_to_t1, t1_c)
        t2_sm_b = project_via_index(base_to_t2, t2_sm)
        t2_md_b = project_via_index(base_to_t2, t2_md)
        t2_lg_b = project_via_index(base_to_t2, t2_lg)
        t2_c_b  = project_via_index(base_to_t2, t2_c)
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
    print(f"H17d — full-history S5/M1 sweep, numpy-bin resample")
    print(f"  SMA combos: {len(SMA_COMBOS)}   M_exit: {M_EXIT}   TP: {TP_GRID}")
    print(f"  S5 pairs: {S5_PAIRS}   M1 pairs: {M1_PAIRS}")
    print(f"  S5 TF combos: {len(S5_TF_COMBOS)}   M1 TF combos: {len(M1_TF_COMBOS)}")
    # JIT warmup
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c,_c,_c,_c,_s,_s,_s,_s,_c,_c,_c,_c,_c,_c,_c,_c,0.0001,1,20.0,1)

    all_rows = []; t0 = time.time()

    # S5 base — full file
    for label, tf1_min, tf2_min in S5_TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in S5_PAIRS:
            rows, days = run_one(pair, S5_DIR, "_S5_BA.parquet", 12.0,
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

    # M1 base
    for label, tf1_min, tf2_min in M1_TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in M1_PAIRS:
            rows, days = run_one(pair, M1_DIR, "_M1_BA.parquet", 1.0,
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
    df_all.to_csv(OUT/'h17d_full_history.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}", flush=True)

    cand = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n=== Cross-TF best IS+OOS+ per pair ===")
    print(f"  {'Pair':<9}{'TF':<13}{'SMA':<10}{'M':>2}{'TP':>4}"
          f"{'OOSpd':>7}{'DD':>7}{'N':>5}{'WR%':>5}  {'days':>6}")
    for _, r in bp.iterrows():
        print(f"  {r['pair']:<9}{r['tf_label']:<13}{r['sma']:<10}"
              f"{int(r['M_exit']):>2}{int(r['tp_pips']):>4}"
              f"{r['oos_pd']:>+7.2f}{r['oos_dd']:>+7.0f}"
              f"{int(r['oos_n']):>5}{r['oos_wr']:>5.0f}  {r['days']:>6.0f}")
    print(f"\n  Σ OOS pd (cross-TF best): {bp['oos_pd'].sum():+.2f}  on {len(bp)} pairs")

    # Per-TF summary
    print(f"\n  ── Per-TF aggregate (best per pair, summed) ──")
    for tf in df_all['tf_label'].unique():
        sub = df_all[df_all.tf_label == tf]
        c = sub[(sub.is_net>0)&(sub.oos_net>0)]
        bp_tf = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
        print(f"    {tf:<14}  {len(bp_tf)}/{sub['pair'].nunique()}  ΣOOS={bp_tf['oos_pd'].sum() if len(bp_tf) else 0:+.2f}")


if __name__ == '__main__':
    main()
