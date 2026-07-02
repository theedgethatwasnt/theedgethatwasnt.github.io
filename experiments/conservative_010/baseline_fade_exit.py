#!/usr/bin/env python3
"""Baseline: REAL 010 entry + fixed-TP + hard-bounded-SL fade exit (no PSAR).

Replaces the PSAR trail with a fade exit (take profit at fixed TP, cut at a HARD bounded
SL) on the actual SMA-stack entry. Purpose: a fixed, risk-capped exit harness to hold
constant while iterating on entry logic. HARD REQUIREMENT: per-trade loss must be bounded
(no -2000p tail) — the fence IS the cap (fills at SL level + slip), verified in output.

Small TP x SL grid on the real entry (no_flip=True, matches headline). Picks the best net,
then runs full validation (IS/OOS, 6-fold WF, MC, equity maxDD, margin sim) on it.
MEMORY-SAFE: one pair resident at a time.
"""
import sys, gc
import numpy as np
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba, CFG
from engine import backtest_pair
from validate import split_is_oos, walk_forward, monte_carlo, equity_drawdown
from margin_sim import simulate_account

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SLIP = 2.0
TPS = [50.0, 100.0, 150.0]
SLS = [150.0, 200.0]      # hard bounded stop (the risk cap)
CONFIGS = [(tp, sl) for sl in SLS for tp in TPS]


def fade_cfg(cfg, tp, sl):
    pip, t1m, t2m, sma, _tp, _psar, af, act, _fence = cfg
    return (pip, t1m, t2m, sma, tp, False, af, act, sl)   # use_psar=False, fence=sl


# collect trades per config, one pair at a time
trades = {c: [] for c in CONFIGS}          # list of {exit_time,pnl_pips,pair}
perpair = {c: {} for c in CONFIGS}
maxloss = {c: 0.0 for c in CONFIGS}
for p in PAIRS:
    d = load_pair_ba(p); ts = np.asarray(d['ts'])
    for c in CONFIGS:
        tp, sl = c
        tr = backtest_pair(d, fade_cfg(CFG[p], tp, sl), slippage_pips=SLIP, no_flip=True)
        en = ts[tr['exit_bar']].astype('datetime64[ns]').astype(np.int64)
        pnl = tr['pnl_pips_net'].astype(float)
        for i in range(len(tr)):
            trades[c].append({'exit_time': int(en[i]), 'pnl_pips': float(pnl[i]), 'pair': p})
        perpair[c][p] = float(pnl.sum())
        if len(pnl): maxloss[c] = min(maxloss[c], float(pnl.min()))
    del d, ts; gc.collect()

print("=== REAL entry + fade exit (TP + hard SL), grid — portfolio, 4 pairs ===")
print(f"{'TP/SL':>10} {'net':>8} {'ntr':>6} {'worst_trade':>12} {'eqMaxDD':>9}  per-pair (EJ/EU/GU/UJ)")
rank = []
for c in CONFIGS:
    tl = sorted(trades[c], key=lambda t: t['exit_time'])
    net = np.array([t['pnl_pips'] for t in tl]); N = len(net)
    dd = equity_drawdown(np.cumsum(net))['max_dd'] if N else 0.0
    pp = perpair[c]
    rank.append((c, net.sum(), N, maxloss[c], dd, tl, net))
    print(f"{int(c[0])}/{int(c[1])!s:>4} {net.sum():>8.0f} {N:>6d} {maxloss[c]:>11.0f}p {dd:>8.0f}p  "
          f"{pp['EUR_JPY']:.0f}/{pp['EUR_USD']:.0f}/{pp['GBP_USD']:.0f}/{pp['USD_JPY']:.0f}")

rank.sort(key=lambda r: -r[1])
best_c, best_net, best_N, best_ml, best_dd, best_tl, best_arr = rank[0]
print(f"\n=== BEST fade baseline: TP{int(best_c[0])}/SL{int(best_c[1])}  net={best_net:.0f}p "
      f"n={best_N}  worst_trade={best_ml:.0f}p  eqMaxDD={best_dd:.0f}p ===")
print(f"RISK CAP CHECK: worst single trade {best_ml:.0f}p (bound = -{int(best_c[1])}-slip) "
      f"-> {'OK bounded' if best_ml > -(best_c[1]+5) else 'CAP BREACHED!'}")
# full validation on the winner
eb = np.arange(best_N); is_end = int(best_N*4/6)
io = split_is_oos(eb, best_arr, is_end)
wf = walk_forward(eb, best_arr, best_N, n_folds=6)
mc = monte_carlo(best_arr, n=300)
margin = [simulate_account(best_tl, b) for b in (100, 500, 1000)]
print(f"IS net {io['is_net']:.0f}p | OOS net {io['oos_net']:.0f}p (OOS WR {io['oos_wr']:.0f}%)")
print(f"Walk-forward {sum(1 for f in wf if f['net']>0)}/6 folds positive")
print(f"Monte-Carlo p_net={mc['p_net']:.3f} p_maxdd={mc['p_maxdd']:.3f}")
print("Margin sim (finite acct): " +
      "; ".join(f"${m['start_balance']}->{'HALT '+m['halted_reason'] if m['halted'] else 'ok'} "
                f"end=${m['final_balance']:.2f}" for m in margin))
print(f"\nvs current headline (PSAR/fence, real entry): -3,616p net, maxDD -4,510p")
print("This fade exit is the fixed risk-capped harness for entry experiments.")
