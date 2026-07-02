#!/usr/bin/env python3
"""
Confirm the TF-sweep winner for GBP_JPY (M5/H1/H4, 4:1) using the ACTUAL live exit
(PSAR trail: arm at 20p MFE, af 0.010→0.10, + 200p broker fence) instead of the
TP20 testbed. Faithful port of update_psar from strategy_sma_stack_live. Compares
the current live M5/M30/H1 (2:1) vs the candidate M5/H1/H4 (4:1), IS/OOS 4/6 + 3-chunk
WF, net of spread. Backtest only — no live change. Run: python3 gbpjpy_h1h4_psar.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC, sma, project_to_m5

PAIR = "GBP_JPY"; PIP = 0.01
SMA_SM, SMA_MD, SMA_LG = 5, 15, 35           # live GBP_JPY
PSAR_AF_START, PSAR_AF_MAX, PSAR_ACT = 0.010, 0.10, 20.0
FENCE = 200.0
M5 = H.M5_DIR


@nb.njit(cache=True)
def psar_series(highs, lows, af_start, af_max):
    n = len(highs); sar = np.full(n, np.nan); init = False
    d = 0; s = 0.0; ep = 0.0; af = 0.0; ph1 = 0.0; pl1 = 0.0; ph2 = 0.0; pl2 = 0.0
    for i in range(n):
        h = highs[i]; l = lows[i]
        if not init:
            if ph1 == 0.0 and pl1 == 0.0:
                ph1 = h; pl1 = l; continue
            if h > ph1: d = 1; ep = h; s = pl1
            else:       d = -1; ep = l; s = ph1
            af = af_start; ph2 = ph1; pl2 = pl1; ph1 = h; pl1 = l; init = True; sar[i] = s; continue
        new = s + af * (ep - s)
        if d == 1:
            if new > pl1: new = pl1
            if new > pl2: new = pl2
            if l < new:
                d = -1; s = ep; ep = l; af = af_start
            else:
                s = new
                if h > ep: ep = h; af = min(af + af_start, af_max)
        else:
            if new < ph1: new = ph1
            if new < ph2: new = ph2
            if h > new:
                d = 1; s = ep; ep = h; af = af_start
            else:
                s = new
                if l < ep: ep = l; af = min(af + af_start, af_max)
        ph2 = ph1; pl2 = pl1; ph1 = h; pl1 = l; sar[i] = s
    return sar


@nb.njit(cache=True)
def psar_kernel(opens, highs, lows, closes, t1_long_nov, t1_shrt_nov,
                t2_long_nov, t2_shrt_nov, sar_b, pip, act_pips, fence_pips):
    n = len(opens); pos = 0; entry = 0.0; ebar = -1; mfe = 0.0; armed = False
    pnls = np.empty(n); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1:
                pos = 1; entry = opens[i]; ebar = i; mfe = 0.0; armed = False; continue
            if t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1:
                pos = -1; entry = opens[i]; ebar = i; mfe = 0.0; armed = False; continue
        if pos != 0:
            fav = (highs[i] - entry) / pip if pos == 1 else (entry - lows[i]) / pip
            if fav > mfe: mfe = fav
            if (not armed) and mfe >= act_pips: armed = True
            exit_px = 0.0; reason = -1
            if pos == 1 and lows[i] <= entry - fence_pips * pip:
                exit_px = entry - fence_pips * pip; reason = 2
            elif pos == -1 and highs[i] >= entry + fence_pips * pip:
                exit_px = entry + fence_pips * pip; reason = 2
            if reason < 0 and armed and not np.isnan(sar_b[i]):
                c = closes[i]
                if pos == 1 and c < sar_b[i]: exit_px = c; reason = 1
                elif pos == -1 and c > sar_b[i]: exit_px = c; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry) / pip * pos; ents[nt] = ebar; nt += 1; pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry) / pip * pos; ents[nt] = ebar; nt += 1
    return pnls[:nt], ents[:nt]


@nb.njit(cache=True)
def tp_kernel(opens, highs, lows, closes, t1_long_nov, t1_shrt_nov,
              t2_long_nov, t2_shrt_nov, pip, tp_pips, fence_pips):
    """Entry = both TFs aligned+novel. Exit = TP (limit fill at level) OR 200p fence."""
    n = len(opens); pos = 0; entry = 0.0; ebar = -1
    pnls = np.empty(n); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1:
                pos = 1; entry = opens[i]; ebar = i; continue
            if t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1:
                pos = -1; entry = opens[i]; ebar = i; continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            tp_lvl = entry + pos * tp_pips * pip
            fc_lvl = entry - pos * fence_pips * pip
            # fence checked first (conservative)
            if pos == 1:
                if lows[i] <= fc_lvl: exit_px = fc_lvl; reason = 2
                elif highs[i] >= tp_lvl: exit_px = tp_lvl; reason = 0
            else:
                if highs[i] >= fc_lvl: exit_px = fc_lvl; reason = 2
                elif lows[i] <= tp_lvl: exit_px = tp_lvl; reason = 0
            if reason >= 0:
                pnls[nt] = (exit_px - entry) / pip * pos; ents[nt] = ebar; nt += 1; pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry) / pip * pos; ents[nt] = ebar; nt += 1
    return pnls[:nt], ents[:nt]


def run(label, tf1_min, tf2_min, max_rows=300_000, mode='psar'):
    df = H.fast_tail_read(M5 / f"{PAIR}_M5.parquet", max_rows).sort_values('timestamp').reset_index(drop=True)
    opens = df['open'].to_numpy(float); highs = df['high'].to_numpy(float)
    lows = df['low'].to_numpy(float); closes = df['close'].to_numpy(float)
    ts = df['timestamp'].to_numpy(); n = len(df); is_end = int(n * IS_FRAC)
    prev_ts = np.empty_like(ts); prev_ts[0] = ts[0]; prev_ts[1:] = ts[:-1]
    days = n * 5 / (60 * 24); sp = PAIRS[PAIR][1] * SPREAD_FRAC

    tf1 = H.resample_minutes(df, tf1_min, 5.0); tf2 = H.resample_minutes(df, tf2_min, 5.0)
    t1c = tf1['close'].to_numpy(float); t1ts = tf1['timestamp'].to_numpy()
    t2c = tf2['close'].to_numpy(float); t2ts = tf2['timestamp'].to_numpy()
    # entry signals on each TF (reuse h17 alignment)
    def novsig(c, ts_):
        s = sma(c, SMA_SM); m = sma(c, SMA_MD); l = sma(c, SMA_LG)
        lo = H.novelty(H.tf_signal(c, s, m, l, 1)); sh = H.novelty(H.tf_signal(c, s, m, l, 0))
        return (project_to_m5(prev_ts, ts_, lo).astype(np.int8),
                project_to_m5(prev_ts, ts_, sh).astype(np.int8))
    t1_lo, t1_sh = novsig(t1c, t1ts); t2_lo, t2_sh = novsig(t2c, t2ts)
    if mode == 'psar':
        sar1 = psar_series(tf1['high'].to_numpy(float), tf1['low'].to_numpy(float), PSAR_AF_START, PSAR_AF_MAX)
        sar_b = project_to_m5(prev_ts, t1ts, sar1)
        p, e = psar_kernel(opens, highs, lows, closes, t1_lo, t1_sh, t2_lo, t2_sh, sar_b, PIP, PSAR_ACT, FENCE)
    else:  # 'tp20'
        p, e = tp_kernel(opens, highs, lows, closes, t1_lo, t1_sh, t2_lo, t2_sh, PIP, 20.0, FENCE)
    net = p - sp
    is_m = e < is_end; oos_m = ~is_m
    is_days = is_end / n * days; oos_days = days - is_days
    def pd_(x, d): return x.sum() / max(d, 1)
    wf = "n/a"
    if is_m.sum() >= 9:
        ch = np.array_split(net[is_m], 3); wf = "+".join("Y" if c.sum() > 0 else "n" for c in ch)
    dd = 0.0
    if oos_m.sum() > 0:
        cum = net[oos_m].cumsum(); dd = float((cum - np.maximum.accumulate(cum)).min())
    print(f"  {label:<12}{len(p):>7}{is_m.sum():>6}{oos_m.sum():>6}"
          f"{net[is_m].sum():>9.0f}{net[oos_m].sum():>9.0f}"
          f"{pd_(net[is_m],is_days):>8.2f}{pd_(net[oos_m],oos_days):>8.2f}"
          f"{(net[oos_m]>0).mean()*100 if oos_m.sum() else 0:>6.0f}%{dd:>9.0f}   WF={wf}")


def main():
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    H.tf_signal(_c, _c, _c, _c, 1); psar_series(_c+1, _c+1, 0.01, 0.1)
    psar_kernel(_c, _c, _c, _c, _s, _s, _s, _s, _c, 0.01, 20.0, 200.0)
    tp_kernel(_c, _c, _c, _c, _s, _s, _s, _s, 0.01, 20.0, 200.0)
    print(f"GBP_JPY stack, ~2.85yr IS/OOS 4/6 + 3-chunk WF, net of spread, 200p fence.")
    print(f"  {'config':<12}{'trades':>7}{'is_n':>6}{'oos_n':>6}{'IS_net':>9}{'OOS_net':>9}{'IS_pd':>8}{'OOS_pd':>8}{'OOS_WR':>7}{'OOS_DD':>9}")
    print("--- LIVE exit: PSAR trail (arm 20p, af .01→.10) ---")
    run("M30/H1 (2:1)", 30, 60, mode='psar')
    run("H1/H4 (4:1)", 60, 240, mode='psar')
    print("--- ALT exit: hard TP=20p (limit fill) ---")
    run("M30/H1 (2:1)", 30, 60, mode='tp20')
    run("H1/H4 (4:1)", 60, 240, mode='tp20')
    run("M30/H4 (8:1)", 30, 240, mode='tp20')


if __name__ == "__main__":
    main()
