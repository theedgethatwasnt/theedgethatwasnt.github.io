#!/usr/bin/env python3
"""
Is GBP_JPY genuinely dead, or just mis-configured to the slowest TFs (M30/H1)?
Test FASTER configs (the M1/M5 / M2/M10 speeds the working JPY pairs use) against
BOTH exits: fixed TP=20p + SL=200p (EUR_JPY/USD_JPY style) AND trailing PSAR.
S5 base (so M1 is resamplable). SMA 5/15/35 (GBP_JPY live). ~5.8mo IS/OOS + 3-chunk WF.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import h17_stack_alignment as H
from _lib import PAIRS, IS_FRAC, SPREAD_FRAC, sma, project_to_m5
from gbpjpy_h1h4_psar import psar_series, psar_kernel, tp_kernel

PAIR = "GBP_JPY"; PIP = 0.01
SMA_SM, SMA_MD, SMA_LG = 5, 15, 35
PSAR_AF_START, PSAR_AF_MAX, PSAR_ACT = 0.010, 0.10, 20.0
FENCE = 200.0; TP = 20.0
S5 = H.S5_DIR; MAX_ROWS = 3_000_000


def run(label, tf1_min, tf2_min, mode):
    df = H.fast_tail_read(S5 / f"{PAIR}_S5_BA.parquet", MAX_ROWS).sort_values('timestamp').reset_index(drop=True)
    opens = df['open'].to_numpy(float); highs = df['high'].to_numpy(float)
    lows = df['low'].to_numpy(float); closes = df['close'].to_numpy(float)
    ts = df['timestamp'].to_numpy(); n = len(df); is_end = int(n * IS_FRAC)
    prev_ts = np.empty_like(ts); prev_ts[0] = ts[0]; prev_ts[1:] = ts[:-1]
    days = n * (5/60) / (60*24); sp = PAIRS[PAIR][1] * SPREAD_FRAC

    tf1 = H.resample_minutes(df, tf1_min, 5/60); tf2 = H.resample_minutes(df, tf2_min, 5/60)
    t1c = tf1['close'].to_numpy(float); t1ts = tf1['timestamp'].to_numpy()
    t2c = tf2['close'].to_numpy(float); t2ts = tf2['timestamp'].to_numpy()
    def novsig(c, ts_):
        s = sma(c, SMA_SM); m = sma(c, SMA_MD); l = sma(c, SMA_LG)
        lo = H.novelty(H.tf_signal(c, s, m, l, 1)); sh = H.novelty(H.tf_signal(c, s, m, l, 0))
        return (project_to_m5(prev_ts, ts_, lo).astype(np.int8), project_to_m5(prev_ts, ts_, sh).astype(np.int8))
    t1_lo, t1_sh = novsig(t1c, t1ts); t2_lo, t2_sh = novsig(t2c, t2ts)

    if mode == 'psar':
        sar1 = psar_series(tf1['high'].to_numpy(float), tf1['low'].to_numpy(float), PSAR_AF_START, PSAR_AF_MAX)
        sar_b = project_to_m5(prev_ts, t1ts, sar1)
        p, e = psar_kernel(opens, highs, lows, closes, t1_lo, t1_sh, t2_lo, t2_sh, sar_b, PIP, PSAR_ACT, FENCE)
    else:
        p, e = tp_kernel(opens, highs, lows, closes, t1_lo, t1_sh, t2_lo, t2_sh, PIP, TP, FENCE)

    net = p - sp; is_m = e < is_end; oos_m = ~is_m
    is_days = is_end/n*days; oos_days = days - is_days
    wf = "n/a"
    if is_m.sum() >= 9:
        ch = np.array_split(net[is_m], 3); wf = "+".join("Y" if c.sum() > 0 else "n" for c in ch)
    dd = 0.0
    if oos_m.sum() > 0:
        cum = net[oos_m].cumsum(); dd = float((cum - np.maximum.accumulate(cum)).min())
    print(f"  {label:<14}{mode:>6}{len(p):>7}{is_m.sum():>6}{oos_m.sum():>6}{net[is_m].sum():>9.0f}{net[oos_m].sum():>9.0f}"
          f"{net[is_m].sum()/max(is_days,1):>8.2f}{net[oos_m].sum()/max(oos_days,1):>8.2f}"
          f"{(net[oos_m]>0).mean()*100 if oos_m.sum() else 0:>6.0f}%{dd:>9.0f}  WF={wf}")


def main():
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    H.tf_signal(_c, _c, _c, _c, 1); psar_series(_c+1, _c+1, 0.01, 0.1)
    psar_kernel(_c, _c, _c, _c, _s, _s, _s, _s, _c, 0.01, 20.0, 200.0)
    tp_kernel(_c, _c, _c, _c, _s, _s, _s, _s, 0.01, 20.0, 200.0)
    print(f"GBP_JPY faster configs, BOTH exits. SMA 5/15/35, ~5.8mo S5, IS/OOS + 3-chunk WF, net spread.")
    print(f"  {'config':<14}{'exit':>6}{'trades':>7}{'is_n':>6}{'oos_n':>6}{'IS_net':>9}{'OOS_net':>9}{'IS_pd':>8}{'OOS_pd':>8}{'OOS_WR':>7}{'OOS_DD':>9}")
    for (lab, t1, t2) in [("M30/H1 (2:1)", 30, 60), ("M5/M15 (3:1)", 5, 15),
                          ("M2/M10 (5:1)", 2, 10), ("M1/M5 (5:1)", 1, 5)]:
        for mode in ('psar', 'tp20'):
            run(lab, t1, t2, mode)


if __name__ == "__main__":
    main()
