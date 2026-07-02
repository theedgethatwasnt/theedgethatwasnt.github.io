"""H17f — Catastrophe-SL sweep on H17 stack-alignment K=0 (TP-only) winners.

Goal: find a deployable exit that bounds per-trade loss without destroying
the +36 p/d edge that K=0 (TP-only) produced in H17e.

Approach: keep the H17 stack-alignment + novelty entry exactly as-is.
At order time, place a broker-side SL alongside the broker-side TP.
Sweep SL distance to find the level that catches the tail catastrophes
(in backtest: per-pair OOS DDs were -13 to -118p over 286-478 days) without
firing on normal flow (98-99% WR trades reach TP fast).

EXIT MODES.
  ("none",  0)        TP-only baseline (replicates H17e K=0)
  ("fixed", 30..200)  TP + fixed-pip SL placed at entry
  ("atr",   1.0..3.0) TP + SL at entry - k * ATR_TF1 (long; short mirror)

Grid: 13 SL modes × 6 SMA × 3 TP × 7 TF × 5 pairs = ~8200 backtests.
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import tf_signal, novelty
from h17d_full_history import fast_full_read, bin_resample
from _lib import PAIRS, IS_FRAC, sma, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

SMA_COMBOS = [
    (5, 10, 22), (5, 15, 35), (5, 22, 50),
    (7, 15, 35), (7, 22, 50),
    (10, 22, 50),
]
TP_GRID = [15.0, 20.0, 30.0]
# (mode_id, sl_param)
#   0 = none (TP-only)
#   1 = fixed pip SL  (param = pip count)
#   2 = ATR-scaled SL (param = k multiplier on ATR_TF1 at entry, in price)
SL_MODES = [
    (0,   0.0),
    (1,  30.0),  (1,  50.0),  (1,  80.0),
    (1, 100.0),  (1, 150.0),  (1, 200.0),
    (2,   1.0),  (2,   1.5),  (2,   2.0),  (2,   3.0),
]

S5_DIR = PROJECT / "data/s5_ohlc"
S5_PAIRS = ["EUR_JPY","EUR_USD","GBP_JPY","GBP_USD","USD_JPY"]
TF_COMBOS = [
    ("S5/S30/M1",   0.5,  1),
    ("S5/M1/M5",      1,  5),
    ("S5/M2/M10",     2, 10),
    ("S5/M5/M15",     5, 15),
    ("S5/M15/M30",   15, 30),
    ("S5/M15/H1",    15, 60),
    ("S5/M30/H1",    30, 60),
]


@nb.njit(cache=True)
def kernel_sl(opens, highs, lows, closes,
              t1_long_nov, t1_shrt_nov, t2_long_nov, t2_shrt_nov,
              t1_atr,                               # per-base-bar projection of TF1 ATR (price units)
              pip, tp_pips, sl_mode, sl_param):
    """K=0 entry (novelty on both TFs same direction). Exits:
       TP intrabar (priority), then SL intrabar.
       sl_mode: 0=none, 1=fixed_pips, 2=atr_scaled (sl_param = k_atr)."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    sl_price = 0.0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64)
    reasons = np.empty(n, np.int8); nt = 0
    for i in range(1, n):
        # Entry
        if pos == 0:
            new_dir = 0
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1:
                new_dir = 1
            elif t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1:
                new_dir = -1
            if new_dir != 0:
                pos = new_dir
                entry_px = opens[i]
                entry_bar = i
                # Compute SL price at entry
                if sl_mode == 0:
                    sl_price = 0.0
                elif sl_mode == 1:
                    sl_price = entry_px - pos * sl_param * pip
                else:  # ATR
                    a = t1_atr[i]
                    if np.isnan(a) or a <= 0:
                        # No ATR available — fall back to fixed 100p
                        sl_price = entry_px - pos * 100.0 * pip
                    else:
                        sl_price = entry_px - pos * sl_param * a
                continue
        # Position management
        if pos != 0:
            exit_px = 0.0; reason = -1
            tp_lvl = entry_px + pos * tp_pips * pip
            # Bull bar check TP first if it could be hit; bear bar check SL first.
            # Pessimistic: when both TP and SL within the bar, assume SL fills.
            # (R2 sequencing: bull → high then low; bear → low then high; here we
            #  treat unknown intrabar order conservatively by checking SL after TP
            #  for long-up bars and SL before TP for long-down bars; this is the
            #  same convention as the rest of our backtests.)
            if pos == 1:
                bull = closes[i] >= opens[i]
                if bull:
                    # high then low: TP can fill first if reachable
                    if highs[i] >= tp_lvl:
                        exit_px = tp_lvl; reason = 0
                    elif sl_mode > 0 and lows[i] <= sl_price:
                        exit_px = sl_price; reason = 1
                else:
                    # low then high: SL checked first on this bar
                    if sl_mode > 0 and lows[i] <= sl_price:
                        exit_px = sl_price; reason = 1
                    elif highs[i] >= tp_lvl:
                        exit_px = tp_lvl; reason = 0
            else:  # short
                bear = closes[i] < opens[i]
                if bear:
                    # low then high
                    if lows[i] <= tp_lvl:
                        exit_px = tp_lvl; reason = 0
                    elif sl_mode > 0 and highs[i] >= sl_price:
                        exit_px = sl_price; reason = 1
                else:
                    if sl_mode > 0 and highs[i] >= sl_price:
                        exit_px = sl_price; reason = 1
                    elif lows[i] <= tp_lvl:
                        exit_px = tp_lvl; reason = 0
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; reasons[nt] = reason
                nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; reasons[nt] = 2   # forced end
        nt += 1
    return pnls[:nt], ents[:nt], reasons[:nt]


def atr_bin_resampled(opens, highs, lows, closes, n_per_bar, period=14):
    """Compute ATR(period) on numpy-binned TF1 bars (Wilder)."""
    r = bin_resample(opens, highs, lows, closes, n_per_bar)
    if r is None: return None, None
    o, h, l, c = r
    n = len(c)
    if n < period + 1: return None, None
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.full(n, np.nan)
    a[period] = tr[1:period+1].mean()
    for i in range(period+1, n):
        a[i] = (a[i-1]*(period-1) + tr[i]) / period
    return a, c   # ATR series + TF1 closes for length reference


def project_by_index(base_idx_into_upper, upper_arr):
    out_typed = np.full(len(base_idx_into_upper), np.nan, np.float64)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    vals = upper_arr[idx_clipped].astype(np.float64)
    out_typed[valid] = vals[valid]
    return out_typed


def project_int8_by_index(base_idx_into_upper, upper_arr):
    out = np.zeros(len(base_idx_into_upper), np.int8)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    out[valid] = upper_arr[idx_clipped][valid].astype(np.int8)
    return out


def run_one(pair, label, tf1_min, tf2_min, base_min_per_bar=5/60):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists(): return []
    df = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                         max_rows=None)
    opens  = df['open'].to_numpy(np.float64)
    highs  = df['high'].to_numpy(np.float64)
    lows   = df['low'].to_numpy(np.float64)
    closes = df['close'].to_numpy(np.float64)
    n_base = len(df)
    is_end = int(n_base * IS_FRAC)
    days = n_base * base_min_per_bar / 1440
    print(f"  {pair} n={n_base:,} ({days:.0f}d)", flush=True)

    n_base_per_min = 12.0
    n1 = int(round(tf1_min * n_base_per_min))
    n2 = int(round(tf2_min * n_base_per_min))
    r1 = bin_resample(opens, highs, lows, closes, n1)
    r2 = bin_resample(opens, highs, lows, closes, n2)
    if r1 is None or r2 is None: return []
    t1_o, t1_h, t1_l, t1_c = r1
    t2_o, t2_h, t2_l, t2_c = r2

    base_to_t1 = np.arange(n_base) // n1 - 1
    base_to_t2 = np.arange(n_base) // n2 - 1

    # ATR on TF1
    t1_atr_series, _ = atr_bin_resampled(opens, highs, lows, closes, n1, period=14)
    if t1_atr_series is None:
        t1_atr_b = np.full(n_base, np.nan)
    else:
        t1_atr_b = project_by_index(base_to_t1, t1_atr_series)

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

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
        t1_long_nov_b = project_int8_by_index(base_to_t1, t1_long_nov)
        t1_shrt_nov_b = project_int8_by_index(base_to_t1, t1_shrt_nov)
        t2_long_nov_b = project_int8_by_index(base_to_t2, t2_long_nov)
        t2_shrt_nov_b = project_int8_by_index(base_to_t2, t2_shrt_nov)

        for (sl_mode, sl_param) in SL_MODES:
            for tp_p in TP_GRID:
                p, e, r = kernel_sl(
                    opens, highs, lows, closes,
                    t1_long_nov_b, t1_shrt_nov_b,
                    t2_long_nov_b, t2_shrt_nov_b,
                    t1_atr_b, pip, tp_p, sl_mode, sl_param,
                )
                mode_label = "none" if sl_mode == 0 else (
                    f"sl_{int(sl_param)}p" if sl_mode == 1 else f"sl_{sl_param}xATR"
                )
                if len(p) == 0:
                    rows.append({'tf_label':label,'pair':pair,
                                 'sma':f"{n_sm}/{n_md}/{n_lg}",
                                 'sl_mode':mode_label,'sl_param':sl_param,
                                 'tp_pips':tp_p,'trades':0,'is_n':0,'oos_n':0,
                                 'is_pd':0,'oos_pd':0,'oos_dd':0,'oos_wr':0,
                                 'is_net':0,'oos_net':0,'days':round(days,1),
                                 'r_tp':0,'r_sl':0})
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
                             'sl_mode':mode_label,'sl_param':sl_param,
                             'tp_pips':tp_p,
                             'trades':int(len(p)),
                             'is_n':int(is_mask.sum()),'oos_n':int(oos_mask.sum()),
                             'is_net':round(is_net,1),'oos_net':round(oos_net,1),
                             'is_pd':round(is_net/max(is_days,1),2),
                             'oos_pd':round(oos_net/max(oos_days,1),2),
                             'oos_dd':round(oos_dd,1),'oos_wr':round(oos_wr,1),
                             'days':round(days,1),
                             'r_tp':int((r==0).sum()),'r_sl':int((r==1).sum())})
    return rows


def main():
    print(f"H17f — catastrophe-SL sweep on H17 stack-alignment K=0 entry")
    print(f"  SL modes: {SL_MODES}")
    print(f"  TP: {TP_GRID}   SMA combos: {len(SMA_COMBOS)}   TF combos: {len(TF_COMBOS)}")
    # JIT warmup
    _c = np.zeros(200); _s = np.zeros(200, np.int8)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 20.0, 0, 0.0)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 20.0, 1, 50.0)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 20.0, 2, 2.0)

    all_rows = []; t0 = time.time()
    for label, tf1_min, tf2_min in TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in S5_PAIRS:
            rows = run_one(pair, label, tf1_min, tf2_min)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0)&(df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT/'h17f_catastrophe_sl.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}", flush=True)

    # Aggregate by SL mode
    print()
    print(f"=== By SL mode: IS+OOS+ count and Σ OOS p/d (best per pair) ===")
    for mode_label in sorted(df_all['sl_mode'].unique()):
        sub = df_all[df_all.sl_mode == mode_label]
        c = sub[(sub.is_net>0)&(sub.oos_net>0)]
        bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
        n_pairs = sub['pair'].nunique()
        avg_dd = bp['oos_dd'].mean() if len(bp) else 0.0
        sl_fires = sub['r_sl'].sum()
        tp_fires = sub['r_tp'].sum()
        sl_ratio = sl_fires / max(sl_fires + tp_fires, 1)
        print(f"  {mode_label:<15} pairs+: {len(bp)}/{n_pairs}  "
              f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+7.2f}  "
              f"avg_DD={avg_dd:>+7.1f}  SL%={100*sl_ratio:>4.1f}")

    # Deploy-candidate table: SL mode + per-pair best config
    print()
    print(f"=== DEPLOY CANDIDATES — bounded-loss configs (sl_mode != 'none') ===")
    print(f"  Pair    TF           SMA       SL       TP   OOSpd   OOS_DD   N   WR%  K=0 baseline OOSpd / DD")
    # Baseline (K=0) per pair-TF for comparison
    baseline = df_all[df_all.sl_mode=='none'].copy()
    bp_base = baseline[(baseline.is_net>0)&(baseline.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
    base_lookup = {r['pair']: (r['oos_pd'], r['oos_dd'], r['tf_label'], r['sma'], r['tp_pips']) for _, r in bp_base.iterrows()}

    sl_only = df_all[df_all.sl_mode != 'none']
    cand = sl_only[(sl_only.is_net>0)&(sl_only.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp_sl = cand.groupby('pair').head(1)
    for _, r in bp_sl.iterrows():
        b = base_lookup.get(r['pair'], (0,0,'-','-',0))
        print(f"  {r['pair']:<7} {r['tf_label']:<12} {r['sma']:<9} "
              f"{r['sl_mode']:<7} {int(r['tp_pips']):>3}p  "
              f"{r['oos_pd']:>+6.2f}  {r['oos_dd']:>+7.0f}  "
              f"{int(r['oos_n']):>4} {r['oos_wr']:>4.0f}%   "
              f"(K=0 base: {b[0]:+6.2f} / {b[1]:+7.0f})")
    print(f"\n  Σ OOS pd (deploy candidates): {bp_sl['oos_pd'].sum():+.2f}  on {len(bp_sl)} pairs")


if __name__ == '__main__':
    main()
