"""H17 v3 — Stack-alignment + monotonic-closes + novelty entry, multi-TF + multi-TP sweep.

ENTRY (long).  All conditions must hold on BOTH TF1 AND TF2 (using last
completed bars), AND the alignment must be NEWLY FORMED:

  (a) Monotone closes:   c[-1] > c[-2] > c[-3]
  (b) Stacked order:     c[-1] > SMA_sm > SMA_md > SMA_lg
  (c) Slopes rising:     SMA_sm/md/lg each rose over last LB_SLOPE steps
  novelty = aligned_t AND NOT aligned_{t-1}

EXIT.  At each base-TF close, count alignment aspects holding per upper-TF.
       Exit when fewer than M_exit_min aspects hold on BOTH TFs.
       Optional fixed TP at TP_PIPS in parallel.

GRID.
  SMA combos     — user range: sm 5-10, md 7-22, lg 15-50, with s<m<l
  M_exit         ∈ {0, 1}
  TP_PIPS        ∈ {15, 20, 30}
  TF pairings:
    M5-base:  M5/M30/H1,  M5/M15/M30,  M5/M15/H1     (10 pairs each)
    S5-base:  S5/S30/M1,  S5/M1/M5                    ( 5 pairs each — subset)
    M1-base:  M1/M2/M5,   M1/M5/M15                   ( 1 pair  each — EUR_USD only)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, sma, project_to_m5, SPREAD_FRAC)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

# User's typical ranges: sm 5-10, md 7-22, lg 15-50. Enforce s<m, m≤l for sanity.
SMA_SM_GRID = [5, 7, 10]
SMA_MD_GRID = [10, 15, 22]
SMA_LG_GRID = [22, 35, 50]
SMA_COMBOS  = [(s, m, l) for s in SMA_SM_GRID for m in SMA_MD_GRID
                          for l in SMA_LG_GRID if s < m and m <= l]
M_EXIT      = [0, 1]
TP_GRID     = [15.0, 20.0, 30.0]
LB_CLOSE    = 3
LB_SLOPE    = 2

M5_DIR = PROJECT / "data/m5_ohlc"
M1_DIR = PROJECT / "data/m1_ohlc"
S5_DIR = PROJECT / "data/s5_ohlc"

S5_PAIRS = {"USD_JPY","GBP_USD","EUR_USD","GBP_JPY","EUR_JPY"}
M1_PAIRS = {"EUR_USD"}    # only file available

# (label, base_dir, base_suffix, base_minutes_per_bar, tf1_min, tf2_min, max_rows, pairs_filter)
TF_PAIRINGS = [
    ("M5/M30/H1",  M5_DIR, "_M5.parquet",     5.0,    30,  60, 51840, None),
    ("M5/M15/M30", M5_DIR, "_M5.parquet",     5.0,    15,  30, 51840, None),
    ("M5/M15/H1",  M5_DIR, "_M5.parquet",     5.0,    15,  60, 51840, None),
    ("S5/S30/M1",  S5_DIR, "_S5_BA.parquet",  5/60,  0.5,   1, 200000, S5_PAIRS),
    ("S5/M1/M5",   S5_DIR, "_S5_BA.parquet",  5/60,    1,   5, 200000, S5_PAIRS),
    ("M1/M2/M5",   M1_DIR, "_M1_BA.parquet",  1.0,     2,   5, 100000, M1_PAIRS),
    ("M1/M5/M15",  M1_DIR, "_M1_BA.parquet",  1.0,     5,  15, 100000, M1_PAIRS),
]


def fast_tail_read(path: Path, max_rows: int) -> pd.DataFrame:
    """Read the last max_rows of a (possibly huge) parquet file efficiently
    by reading row groups from the end."""
    pf = pq.ParquetFile(str(path))
    n_rg = pf.metadata.num_row_groups
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    # Walk row groups from the end until we have ≥ max_rows
    take = []; total = 0
    for i in range(n_rg - 1, -1, -1):
        take.append(i); total += rg_rows[i]
        if total >= max_rows: break
    take.sort()
    tbl = pf.read_row_groups(take, columns=['timestamp','open','high','low','close'])
    df = tbl.to_pandas()
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df


@nb.njit(cache=True)
def tf_signal(closes, sma_sm, sma_md, sma_lg, dir_long):
    n = len(closes)
    aligned = np.zeros(n, np.int8)
    start = max(LB_CLOSE, LB_SLOPE + 2)
    for i in range(start, n):
        if np.isnan(sma_sm[i]) or np.isnan(sma_md[i]) or np.isnan(sma_lg[i]):
            continue
        if dir_long == 1:
            mono = True
            for k in range(1, LB_CLOSE):
                if not (closes[i-k+1] > closes[i-k]):
                    mono = False; break
            if not mono: continue
            if not (closes[i] > sma_sm[i] > sma_md[i] > sma_lg[i]):
                continue
            slopes_ok = True
            for sma_idx in range(3):
                if sma_idx == 0: arr = sma_sm
                elif sma_idx == 1: arr = sma_md
                else: arr = sma_lg
                for k in range(LB_SLOPE):
                    a, b = arr[i-k], arr[i-k-1]
                    if np.isnan(a) or np.isnan(b) or not (a > b):
                        slopes_ok = False; break
                if not slopes_ok: break
            if slopes_ok: aligned[i] = 1
        else:
            mono = True
            for k in range(1, LB_CLOSE):
                if not (closes[i-k+1] < closes[i-k]):
                    mono = False; break
            if not mono: continue
            if not (closes[i] < sma_sm[i] < sma_md[i] < sma_lg[i]):
                continue
            slopes_ok = True
            for sma_idx in range(3):
                if sma_idx == 0: arr = sma_sm
                elif sma_idx == 1: arr = sma_md
                else: arr = sma_lg
                for k in range(LB_SLOPE):
                    a, b = arr[i-k], arr[i-k-1]
                    if np.isnan(a) or np.isnan(b) or not (a < b):
                        slopes_ok = False; break
                if not slopes_ok: break
            if slopes_ok: aligned[i] = 1
    return aligned


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes,
           h1_long_nov, h1_shrt_nov, m30_long_nov, m30_shrt_nov,
           h1_sm, h1_md, h1_lg, h1_c_prev,
           m30_sm, m30_md, m30_lg, m30_c_prev,
           pip, M_exit_min, tp_pips, with_tp):
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if h1_long_nov[i] == 1 and m30_long_nov[i] == 1:
                pos = 1; entry_px = opens[i]; entry_bar = i; continue
            if h1_shrt_nov[i] == 1 and m30_shrt_nov[i] == 1:
                pos = -1; entry_px = opens[i]; entry_bar = i; continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            if with_tp == 1:
                tp_lvl = entry_px + pos * tp_pips * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0 and M_exit_min > 0:
                cl = closes[i]
                hsm = h1_sm[i]; hmd = h1_md[i]; hlg = h1_lg[i]; hpc = h1_c_prev[i]
                msm = m30_sm[i]; mmd = m30_md[i]; mlg = m30_lg[i]; mpc = m30_c_prev[i]
                h_count = 0
                if not (np.isnan(hsm) or np.isnan(hmd) or np.isnan(hlg) or np.isnan(hpc)):
                    if pos == 1:
                        if cl > hpc: h_count += 1
                        if cl > hsm and hsm > hmd and hmd > hlg: h_count += 1
                        if hsm > hmd and hmd > hlg: h_count += 1
                    else:
                        if cl < hpc: h_count += 1
                        if cl < hsm and hsm < hmd and hmd < hlg: h_count += 1
                        if hsm < hmd and hmd < hlg: h_count += 1
                m_count = 0
                if not (np.isnan(msm) or np.isnan(mmd) or np.isnan(mlg) or np.isnan(mpc)):
                    if pos == 1:
                        if cl > mpc: m_count += 1
                        if cl > msm and msm > mmd and mmd > mlg: m_count += 1
                        if msm > mmd and mmd > mlg: m_count += 1
                    else:
                        if cl < mpc: m_count += 1
                        if cl < msm and msm < mmd and mmd < mlg: m_count += 1
                        if msm < mmd and mmd < mlg: m_count += 1
                if h_count < M_exit_min and m_count < M_exit_min:
                    exit_px = cl; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def novelty(arr):
    nv = np.zeros(len(arr), dtype=np.int8)
    nv[1:] = ((arr[1:] == 1) & (arr[:-1] == 0)).astype(np.int8)
    return nv


def resample_minutes(df_base, minutes_per_target_bar, base_minutes_per_bar):
    """Resample base TF up to a target TF.  Uses pandas .resample.
    minutes_per_target_bar: e.g., 30 for M30; can be fractional (0.5 for S30).
    base_minutes_per_bar: e.g., 5.0 for M5, 0.0833 for S5 (5/60)."""
    if minutes_per_target_bar < 1.0:
        # Sub-minute (e.g., S30): use seconds-based resample rule
        secs = int(minutes_per_target_bar * 60)
        rule = f'{secs}s'
    else:
        rule = f'{int(minutes_per_target_bar)}min'
    d = df_base.set_index('timestamp')
    return d.resample(rule, label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()


def run_tf_combo(label, base_dir, suffix, base_min, tf1_min, tf2_min,
                 max_rows, pairs_filter):
    rows = []
    pair_list = list(PAIRS.keys())
    if pairs_filter is not None:
        pair_list = [p for p in pair_list if p in pairs_filter]
    pip_of  = lambda p: PAIRS[p][0]
    sp_cost = lambda p: PAIRS[p][1] * SPREAD_FRAC

    for pair in pair_list:
        path = base_dir / f"{pair}{suffix}"
        if not path.exists():
            continue
        try:
            df = fast_tail_read(path, max_rows)
        except Exception as e:
            print(f"  [skip] {pair} {label} — load failed: {e}")
            continue
        df = df.sort_values('timestamp').reset_index(drop=True)
        if len(df) < 3000:
            continue

        opens  = df['open'].to_numpy(np.float64)
        highs  = df['high'].to_numpy(np.float64)
        lows   = df['low'].to_numpy(np.float64)
        closes = df['close'].to_numpy(np.float64)
        ts     = df['timestamp'].to_numpy()
        n = len(df); is_end = int(n * IS_FRAC)

        tf1 = resample_minutes(df, tf1_min, base_min)
        tf2 = resample_minutes(df, tf2_min, base_min)
        tf1_c = tf1['close'].to_numpy(np.float64); tf1_ts = tf1['timestamp'].to_numpy()
        tf2_c = tf2['close'].to_numpy(np.float64); tf2_ts = tf2['timestamp'].to_numpy()

        prev_ts = np.empty_like(ts); prev_ts[0]=ts[0]; prev_ts[1:]=ts[:-1]
        sp = sp_cost(pair); pip = pip_of(pair)
        # Approximate "days" of base series (for p/d normalization)
        days = n * base_min / (60 * 24)

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
                    is_days  = (is_end / n) * days
                    oos_days = days - is_days
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
    print("="*100)
    print(f"  H17 v3 — Stack-alignment, multi-TF + multi-TP")
    print(f"  SMA combos ({len(SMA_COMBOS)}): {SMA_COMBOS}")
    print(f"  M_exit: {M_EXIT}   TP_pips: {TP_GRID}")
    print(f"  TF pairings: {len(TF_PAIRINGS)}")
    print("="*100)
    # JIT warmup
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel(_c, _c, _c, _c, _s, _s, _s, _s, _c, _c, _c, _c, _c, _c, _c, _c, 0.0001, 1, 20.0, 1)

    all_rows = []; t0 = time.time()
    for cfg in TF_PAIRINGS:
        label = cfg[0]
        t1 = time.time()
        rows = run_tf_combo(*cfg)
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
    df_all.to_csv(OUT/'h17_stack_alignment.csv', index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s   rows: {len(df_all)}")

    # Cross-TF best per pair
    if df_all.empty or 'is_net' not in df_all.columns:
        print("\n  (no rows)"); return
    cand_all = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp_all = cand_all.groupby('pair').head(1)
    print()
    print("="*100)
    print("  BEST IS+OOS+ ACROSS ALL TF PAIRINGS, per pair")
    print("="*100)
    print(f"  {'Pair':<9} {'TF':<14} {'SMA':<10} {'M':>2} {'TP':>4} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  {'days':>6}")
    for _, r in bp_all.iterrows():
        print(f"  {r['pair']:<9} {r['tf_label']:<14} {r['sma']:<10} "
              f"{int(r['M_exit']):>2d} {r['tp_pips']:>4.0f} "
              f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
              f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  {r['days']:>6.1f}")
    print(f"\n  Total: {len(bp_all)} pairs IS+OOS+  Σ OOS pd: {bp_all['oos_pd'].sum():+.2f}")

    # Per-TF summary
    print()
    print("  ── Per-TF pairing summary (best per pair, summed) ──")
    for cfg in TF_PAIRINGS:
        lab = cfg[0]
        sub = df_all[df_all.tf_label == lab]
        c = sub[(sub.is_net>0)&(sub.oos_net>0)]
        bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
        print(f"    {lab:<14}  {len(bp) if len(bp) else 0:>2}/{sub['pair'].nunique():>2} pairs "
              f"Σ OOS pd = {bp['oos_pd'].sum() if len(bp) else 0:+.2f}")


if __name__ == '__main__':
    main()
