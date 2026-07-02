#!/usr/bin/env python3
"""Exit-strategy sweep on RANDOM entries — can any exit combo beat the current one?

Entry has no edge (proven: random_entry_test.py + random_market_entry.py). So this asks
the exit-side question directly: with random entry time + random direction (same calibrated
rate as the live config), which EXIT combination gives the best net? If a tight-TP/wide-SL
(mean-reversion harvest) or a trailing combo turns POSITIVE on random entries, that is a
genuine market exit-side asymmetry independent of any entry signal. Prior work predicts
none clears the OANDA spread — this tests it head-on and finds the least-bad.

Method: reuse the random-market-entry kernel (_kern_rme) but call it with many exit
specs (fence / tp / psar+act). p_enter fixed per pair (calibrated to the live config) so
every combo faces the SAME entry process — tighter exits churn more and pay more cost,
which is the realistic tradeoff. Portfolio mean net over N seeds ranks the combos.

'CURRENT' = each pair's live per-pair exit (from CFG) — the thing to beat.
MEMORY-SAFE: one pair resident at a time; del+gc between.
"""
import sys, gc
import numpy as np

PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba, CFG
from engine import backtest_pair
from random_market_entry import _kern_rme   # reuse the exact random-entry kernel

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
N_SEEDS = 100
SLIP = 2.0
NO_FLIP = True
BIG = 1.0e5   # "no effective stop" fence

# Exit combos: (name, fence_pips, tp_pips, use_psar, act_pips). fence=BIG => no SL.
COMBOS = [
    ("SL50",              50,   0.0, False, 0.0),
    ("SL100",            100,   0.0, False, 0.0),
    ("SL150",            150,   0.0, False, 0.0),
    ("SL200",            200,   0.0, False, 0.0),
    ("SL300",            300,   0.0, False, 0.0),
    ("SL500",            500,   0.0, False, 0.0),
    ("TP20+SL_BIG",      BIG,  20.0, False, 0.0),   # pure fade (mean-rev)
    ("TP50+SL_BIG",      BIG,  50.0, False, 0.0),
    ("TP100+SL_BIG",     BIG, 100.0, False, 0.0),
    ("TP10+SL500",       500,  10.0, False, 0.0),   # tight fade, wide backstop
    ("TP20+SL500",       500,  20.0, False, 0.0),
    ("TP20+SL100",       100,  20.0, False, 0.0),
    ("TP50+SL100",       100,  50.0, False, 0.0),
    ("TP50+SL200",       200,  50.0, False, 0.0),
    ("TP100+SL200",      200, 100.0, False, 0.0),
    ("TP100+SL50",        50, 100.0, False, 0.0),   # momentum (wide TP, tight SL)
    ("TP200+SL50",        50, 200.0, False, 0.0),
    ("PSAR_act10+SL500", 500,   0.0, True,  10.0),  # trailing families
    ("PSAR_act20+SL500", 500,   0.0, True,  20.0),
    ("PSAR_act40+SL500", 500,   0.0, True,  40.0),
    ("TP50+PSAR20+SL300",300,  50.0, True,  20.0),
    ("TP20+PSAR20+SL200",200,  20.0, True,  20.0),
]


def run(d, pip, fence, tp, psar, act, p_enter, seed):
    return _kern_rme(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'], d['bid_c'], d['ask_c'],
                     d['sar'], pip, tp, psar, act, fence, SLIP, p_enter, seed).sum()


# net[combo_idx, pair, seed] ; plus CURRENT
n_combo = len(COMBOS)
net = np.zeros((n_combo + 1, len(PAIRS), N_SEEDS))
ntr = np.zeros((n_combo + 1, len(PAIRS)))
for pi, p in enumerate(PAIRS):
    d = load_pair_ba(p)
    cfg = CFG[p]
    pip, _t1m, _t2m, _sma, tp_cur, psar_cur, _af, act_cur, fence_cur = cfg
    # calibrate p_enter to the live config's trade count (same for all combos)
    prod = backtest_pair(d, cfg, slippage_pips=SLIP, no_flip=NO_FLIP)
    n_real = len(prod)
    avg_hold = float(np.mean(prod['exit_bar'] - prod['entry_bar'])) if n_real else 1.0
    N = len(d['m5_o'])
    p_enter = min(n_real / max(N - n_real * avg_hold, 1.0), 1.0)
    # CURRENT (live per-pair exit) as the last row
    for s in range(N_SEEDS):
        rp = _kern_rme(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'], d['bid_c'], d['ask_c'],
                       d['sar'], pip, tp_cur, psar_cur, act_cur, fence_cur, SLIP, p_enter,
                       pi * 100003 + s)
        net[n_combo, pi, s] = rp.sum(); ntr[n_combo, pi] += len(rp)
    # swept combos
    for ci, (_nm, fence, tp, psar, act) in enumerate(COMBOS):
        for s in range(N_SEEDS):
            k = _kern_rme(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'], d['bid_c'], d['ask_c'],
                          d['sar'], pip, tp, psar, act, fence, SLIP, p_enter, pi * 100003 + s)
            net[ci, pi, s] = k.sum(); ntr[ci, pi] += len(k)
    print(f"  {p}: p_enter={p_enter:.2e} done ({n_combo+1} combos x {N_SEEDS} seeds)", flush=True)
    del d, prod; gc.collect()

ntr /= N_SEEDS
port = net.sum(axis=1)            # [combo, seed] portfolio net
mean = port.mean(axis=1)
std = port.std(axis=1)
tot_ntr = ntr.sum(axis=1)
cur_mean = mean[n_combo]

rows = [("CURRENT (live)", cur_mean, std[n_combo], tot_ntr[n_combo])]
for ci, (nm, *_ ) in enumerate(COMBOS):
    rows.append((nm, mean[ci], std[ci], tot_ntr[ci]))
rows.sort(key=lambda r: -r[1])   # best (highest net) first

print("\n=========  EXIT SWEEP ON RANDOM ENTRIES  (portfolio net, 4 pairs, 100 seeds)  =========")
print(f"{'exit combo':<20} {'mean_net':>9} {'std':>7} {'~trades':>8}  {'vs CURRENT':>11}")
for nm, m, sd, nt in rows:
    tag = "  <== CURRENT" if nm.startswith("CURRENT") else ""
    print(f"{nm:<20} {m:>8.0f}p {sd:>6.0f}p {nt:>8.0f}  {m-cur_mean:>+10.0f}p{tag}")
best = rows[0]
print(f"\nCURRENT (live exit) random-entry mean: {cur_mean:.0f}p")
print(f"BEST exit combo: {best[0]}  {best[1]:.0f}p  ({best[1]-cur_mean:+.0f}p vs current)")
n_beat = sum(1 for r in rows if not r[0].startswith('CURRENT') and r[1] > cur_mean)
n_pos = sum(1 for r in rows if r[1] > 0)
print(f"{n_beat}/{n_combo} combos beat CURRENT; {n_pos} combos POSITIVE net.")
print("Note: on random entries, POSITIVE net = a real market exit-side asymmetry (fade/trend),")
print("      not entry skill. Least-negative usually = lowest-churn (least spread paid).")
