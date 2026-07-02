#!/usr/bin/env python3
"""Risk-reality check on the 'positive' no-stop exit combos from random_exit_sweep.

TP-with-no-stop looked +2,833p on random entries. Suspicion: it only banks the TP
because losers are held indefinitely (unbounded drawdown) -> the pip P&L ignores that a
finite-margin account would be liquidated first. Measure the WORST per-trade adverse
excursion (how far underwater a single position went before it finally hit TP) and the
count of positions still open at series end. If max MAE >> the ~25% ruin drawdown, the
'edge' is the finite-margin trap, not real.
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np
import numba as nb
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba, CFG
from engine import backtest_pair

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
N_SEEDS = 50; SLIP = 2.0; BIG = 1.0e5


@nb.njit(cache=True)
def _kern_mae(o, h, l, c, bid_c, ask_c, sar_b, pip, tp_pips, use_psar, act,
              fence, slip, p_enter, seed):
    """Random entry + given exit; returns (net, worst_trade_MAE_pips, n_closed, ended_open)."""
    np.random.seed(seed)
    n = len(o); pos = 0; entry_fill = 0.0; mfe = 0.0; armed = False
    net = 0.0; worst_mae = 0.0; n_closed = 0
    for i in range(1, n):
        if pos == 0:
            if np.random.random() < p_enter:
                pos = 1 if np.random.random() < 0.5 else -1
                entry_fill = ask_c[i] if pos == 1 else bid_c[i]; mfe = 0.0; armed = False
                continue
        if pos != 0:
            fav = (h[i]-entry_fill)/pip if pos == 1 else (entry_fill-l[i])/pip
            adv = (l[i]-entry_fill)/pip if pos == 1 else (entry_fill-h[i])/pip  # <=0 worst
            if adv < worst_mae: worst_mae = adv
            if fav > mfe: mfe = fav
            if use_psar and (not armed) and mfe >= act: armed = True
            ex_fill = 0.0; rsn = -1
            fc = entry_fill - pos*fence*pip
            if pos == 1 and l[i] <= fc: ex_fill = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: ex_fill = fc + slip*pip; rsn = 0
            if rsn < 0 and tp_pips > 0.0:
                tp = entry_fill + pos*tp_pips*pip
                if pos == 1 and h[i] >= tp: ex_fill = tp; rsn = 2
                elif pos == -1 and l[i] <= tp: ex_fill = tp; rsn = 2
            if rsn < 0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos == 1 and c[i] < sar_b[i]: ex_fill = bid_c[i]-slip*pip; rsn = 1
                elif pos == -1 and c[i] > sar_b[i]: ex_fill = ask_c[i]+slip*pip; rsn = 1
            if rsn >= 0:
                net += (ex_fill-entry_fill)/pip if pos == 1 else (entry_fill-ex_fill)/pip
                n_closed += 1; pos = 0
    return net, worst_mae, n_closed, (1 if pos != 0 else 0)


TESTS = [("TP50+SL_BIG", BIG, 50.0, False, 0.0),
         ("TP50+SL200",  200, 50.0, False, 0.0)]
for nm, fence, tp, psar, act in TESTS:
    print(f"\n### {nm}")
    for p in PAIRS:
        d = load_pair_ba(p); cfg = CFG[p]
        pip = cfg[0]
        prod = backtest_pair(d, cfg, slippage_pips=SLIP, no_flip=True)
        nr = len(prod); ah = float(np.mean(prod['exit_bar']-prod['entry_bar'])) if nr else 1.0
        N = len(d['m5_o']); p_enter = min(nr/max(N-nr*ah, 1.0), 1.0)
        maes = []; nets = []
        for s in range(N_SEEDS):
            net, wmae, ncl, op = _kern_mae(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'],
                                           d['bid_c'], d['ask_c'], d['sar'], pip, tp, psar,
                                           act, fence, SLIP, p_enter, s)
            maes.append(wmae); nets.append(net)
        print(f"  {p}: net~{np.mean(nets):.0f}p  WORST 1-trade drawdown: "
              f"median {np.median(maes):.0f}p  p95 {np.percentile(maes,5):.0f}p  "
              f"worst {np.min(maes):.0f}p")
        del d, prod; gc.collect()
