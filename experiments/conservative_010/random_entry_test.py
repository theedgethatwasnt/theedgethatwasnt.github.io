#!/usr/bin/env python3
"""Random-DIRECTION control test for conservative 010.

Question: does the ENTRY DIRECTION carry any edge? Keep EVERYTHING identical to the
headline no-flip conservative backtest — same entry TRIGGER timing (a trade opens iff
the SMA-stack novelty fires while flat), same exits (200p fence / TP / PSAR, all
direction-aware), same real per-bar spread + worse-side fills + 2p stop slippage —
but at order-send, pick the direction by a COIN FLIP instead of following the signal.

If the real signal's net sits inside the random-direction distribution, the entry
direction adds nothing. If real << random, the signal is anti-predictive. If real >>
random, the entry has genuine directional edge.

Kernel `_kern_rd` is a verbatim copy of engine._kern with ONE change: the entry
direction. With rand_dir=False it MUST reproduce the production kernel exactly — we
assert that per pair (SOP R7 consistency gate) before trusting the rand_dir=True runs.

MEMORY-SAFE: one pair resident at a time; run all N seeds for that pair, then del+gc.
"""
import sys, gc
import numpy as np
import numba as nb

PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from data import load_pair_ba, CFG
from engine import backtest_pair

PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
N_SEEDS = 500
SLIP = 2.0
NO_FLIP = True   # match the headline verdict (live-frequency-matched)


@nb.njit(cache=True)
def _kern_rd(o, h, l, c, bid_c, ask_c, t1l, t1s, t2l, t2s, sar_b,
             pip, tp_pips, use_psar, act, fence, slip, no_flip, rand_dir, seed):
    """engine._kern verbatim EXCEPT the entry-direction block. rand_dir=False =>
    identical to production; rand_dir=True => coin-flip direction on each signal."""
    if rand_dir:
        np.random.seed(seed)
    n = len(o)
    pos = 0; entry_fill = 0.0; ebar = -1; mfe = 0.0; armed = False
    _pnl = np.empty(n, np.float64)
    _rsn = np.empty(n, np.int64)
    nt = 0
    for i in range(1, n):
        # ── ENTRY (same TRIGGER; direction flipped when rand_dir) ──
        if pos == 0:
            long_sig = (t1l[i] == 1 and t2l[i] == 1)
            short_sig = (t1s[i] == 1 and t2s[i] == 1)
            if long_sig or short_sig:
                if rand_dir:
                    d = 1 if np.random.random() < 0.5 else -1
                else:
                    d = 1 if long_sig else -1     # production: long precedence
                pos = d
                entry_fill = ask_c[i] if d == 1 else bid_c[i]
                ebar = i; mfe = 0.0; armed = False
                continue
        # ── IN TRADE (verbatim exits, all direction-aware via pos) ──
        if pos != 0:
            fav = (h[i] - entry_fill) / pip if pos == 1 else (entry_fill - l[i]) / pip
            if fav > mfe: mfe = fav
            if use_psar and (not armed) and mfe >= act: armed = True
            ex_fill = 0.0; rsn = -1
            fc = entry_fill - pos * fence * pip
            if pos == 1 and l[i] <= fc:
                ex_fill = fc - slip * pip; rsn = 0
            elif pos == -1 and h[i] >= fc:
                ex_fill = fc + slip * pip; rsn = 0
            if rsn < 0 and tp_pips > 0.0:
                tp = entry_fill + pos * tp_pips * pip
                if pos == 1 and h[i] >= tp:
                    ex_fill = tp; rsn = 2
                elif pos == -1 and l[i] <= tp:
                    ex_fill = tp; rsn = 2
            if rsn < 0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos == 1 and c[i] < sar_b[i]:
                    ex_fill = bid_c[i] - slip * pip; rsn = 1
                elif pos == -1 and c[i] > sar_b[i]:
                    ex_fill = ask_c[i] + slip * pip; rsn = 1
            if rsn < 0 and not no_flip:
                if pos == 1 and t1s[i] == 1 and t2s[i] == 1:
                    ex_fill = bid_c[i]; rsn = 3
                elif pos == -1 and t1l[i] == 1 and t2l[i] == 1:
                    ex_fill = ask_c[i]; rsn = 3
            if rsn >= 0:
                pnl = ((ex_fill - entry_fill) / pip if pos == 1
                       else (entry_fill - ex_fill) / pip)
                _pnl[nt] = pnl; _rsn[nt] = rsn
                nt += 1; pos = 0
    return _pnl[:nt]


def run_kern(d, cfg, rand_dir, seed):
    pip, _t1m, _t2m, _sma, tp_pips, use_psar, _af, act, fence = cfg
    return _kern_rd(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'], d['bid_c'], d['ask_c'],
                    d['t1l'], d['t1s'], d['t2l'], d['t2s'], d['sar'],
                    pip, tp_pips, use_psar, act, fence, SLIP, NO_FLIP, rand_dir, seed)


real_net = {}
rand_net = np.zeros((len(PAIRS), N_SEEDS))   # [pair, seed]
rand_ntr = np.zeros(len(PAIRS))
for pi, p in enumerate(PAIRS):
    d = load_pair_ba(p)
    # --- R7 consistency gate: rand_dir=False copy == production kernel ---
    copy_pnl = run_kern(d, CFG[p], rand_dir=False, seed=0)
    prod = backtest_pair(d, CFG[p], slippage_pips=SLIP, no_flip=NO_FLIP)
    assert len(copy_pnl) == len(prod), f"{p}: trade count {len(copy_pnl)} != {len(prod)}"
    assert np.allclose(copy_pnl, prod['pnl_pips_net'], atol=1e-6), f"{p}: pnl diverged"
    real_net[p] = float(copy_pnl.sum())
    # --- N random-direction realizations ---
    ntr = 0
    for s in range(N_SEEDS):
        rp = run_kern(d, CFG[p], rand_dir=True, seed=pi * 100003 + s)
        rand_net[pi, s] = rp.sum()
        ntr += len(rp)
    rand_ntr[pi] = ntr / N_SEEDS
    print(f"  {p}: real_net={real_net[p]:.0f}p (n={len(prod)})  |  "
          f"random mean={rand_net[pi].mean():.0f}p  avg_ntr={rand_ntr[pi]:.0f}  [R7 gate OK]",
          flush=True)
    del d, prod, copy_pnl; gc.collect()

real_port = sum(real_net.values())
rand_port = rand_net.sum(axis=0)          # portfolio net per seed
p_ge = float((rand_port >= real_port).mean())
print("\n================  RANDOM-DIRECTION CONTROL  ================")
print(f"REAL signal portfolio net (no_flip):  {real_port:.0f} pips")
print(f"RANDOM-direction net over {N_SEEDS} draws:")
print(f"   mean   {rand_port.mean():.0f}p   std {rand_port.std():.0f}p")
print(f"   5th/50th/95th pct  {np.percentile(rand_port,5):.0f} / "
      f"{np.percentile(rand_port,50):.0f} / {np.percentile(rand_port,95):.0f} p")
print(f"   min/max  {rand_port.min():.0f} / {rand_port.max():.0f} p")
print(f"P(random >= real) = {p_ge:.3f}   "
      f"z(real vs random) = {(real_port-rand_port.mean())/rand_port.std():+.2f}")
print("\nInterpretation:")
if p_ge > 0.05 and p_ge < 0.95:
    print("  Real net sits INSIDE the random-direction cloud -> entry direction adds NO edge.")
elif p_ge >= 0.95:
    print("  Real net is WORSE than almost all random -> signal is ANTI-predictive.")
else:
    print("  Real net BEATS almost all random -> entry direction has genuine edge.")
