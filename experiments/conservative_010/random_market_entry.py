#!/usr/bin/env python3
"""Random-MARKET-ENTRY control for conservative 010.

Test 1 (random_entry_test.py) kept the SMA-stack TRIGGER timing and coin-flipped only
direction -> direction has no edge. This test throws the signal out ENTIRELY: when flat,
enter at a RANDOM bar (per-bar Bernoulli p) in a RANDOM direction, exits UNCHANGED
(200p fence / TP / PSAR, no_flip), one trade at a time.

Entry rate p is calibrated per pair so the expected trade count ~= the real run, so the
transaction-cost drag is comparable. If this net ~= Test-1 random (~-1,900p), the signal's
TIMING adds nothing either -> neither WHEN nor WHICH-WAY the SMA-stack enters has value.

Exit block is copied verbatim from engine._kern (same exits as production/Test 1).
MEMORY-SAFE: one pair resident at a time; N seeds; del+gc between.
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
NO_FLIP = True
REAL_PORT = -3616      # headline no-flip net (for reference)
T1_RANDOM_MEAN = -1903 # Test-1 random-direction mean (for reference)


@nb.njit(cache=True)
def _kern_rme(o, h, l, c, bid_c, ask_c, sar_b,
              pip, tp_pips, use_psar, act, fence, slip, p_enter, seed):
    """Random market entry: enter random bar (Bernoulli p_enter) + random direction.
    Exit block is VERBATIM engine._kern (fence/TP/PSAR). One position at a time."""
    np.random.seed(seed)
    n = len(o)
    pos = 0; entry_fill = 0.0; mfe = 0.0; armed = False
    _pnl = np.empty(n, np.float64)
    nt = 0
    for i in range(1, n):
        # ── RANDOM ENTRY (no signal; random time + random direction) ──
        if pos == 0:
            if np.random.random() < p_enter:
                d = 1 if np.random.random() < 0.5 else -1
                pos = d
                entry_fill = ask_c[i] if d == 1 else bid_c[i]
                mfe = 0.0; armed = False
                continue
        # ── IN TRADE (exits verbatim from engine._kern) ──
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
            if rsn >= 0:
                pnl = ((ex_fill - entry_fill) / pip if pos == 1
                       else (entry_fill - ex_fill) / pip)
                _pnl[nt] = pnl
                nt += 1; pos = 0
    return _pnl[:nt]


if __name__ == "__main__":
    rand_net = np.zeros((len(PAIRS), N_SEEDS))
    rand_ntr = np.zeros(len(PAIRS))
    p_used = {}
    for pi, p in enumerate(PAIRS):
        d = load_pair_ba(p)
        cfg = CFG[p]
        pip, _t1m, _t2m, _sma, tp_pips, use_psar, _af, act, fence = cfg
        # --- calibrate p_enter to match real trade count ---
        prod = backtest_pair(d, cfg, slippage_pips=SLIP, no_flip=NO_FLIP)
        n_real = len(prod)
        avg_hold = float(np.mean(prod['exit_bar'] - prod['entry_bar'])) if n_real else 1.0
        N = len(d['m5_o'])
        flat_bars = max(N - n_real * avg_hold, 1.0)
        p_enter = min(n_real / flat_bars, 1.0)
        p_used[p] = p_enter
        ntr = 0
        for s in range(N_SEEDS):
            rp = _kern_rme(d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'], d['bid_c'], d['ask_c'],
                           d['sar'], pip, tp_pips, use_psar, act, fence, SLIP, p_enter,
                           pi * 100003 + s)
            rand_net[pi, s] = rp.sum()
            ntr += len(rp)
        rand_ntr[pi] = ntr / N_SEEDS
        print(f"  {p}: real n={n_real} (avg_hold {avg_hold:.0f} bars)  p_enter={p_enter:.2e}  "
              f"-> random avg_ntr={rand_ntr[pi]:.0f}  random_mean_net={rand_net[pi].mean():.0f}p",
              flush=True)
        del d, prod; gc.collect()

    rand_port = rand_net.sum(axis=0)
    p_ge = float((rand_port >= REAL_PORT).mean())
    print("\n============  RANDOM MARKET ENTRY (random time + random dir)  ============")
    print(f"REAL signal portfolio net (ref):       {REAL_PORT} pips")
    print(f"Test-1 random-DIRECTION mean (ref):    {T1_RANDOM_MEAN} pips")
    print(f"RANDOM market-entry net over {N_SEEDS} draws:")
    print(f"   mean   {rand_port.mean():.0f}p   std {rand_port.std():.0f}p")
    print(f"   5th/50th/95th pct  {np.percentile(rand_port,5):.0f} / "
          f"{np.percentile(rand_port,50):.0f} / {np.percentile(rand_port,95):.0f} p")
    print(f"   min/max  {rand_port.min():.0f} / {rand_port.max():.0f} p")
    print(f"P(random-entry >= real signal) = {p_ge:.3f}   "
          f"z(real vs this random) = {(REAL_PORT-rand_port.mean())/rand_port.std():+.2f}")
    print("\nInterpretation:")
    print(f"  random market-entry mean {rand_port.mean():.0f}p vs Test-1 (real timing) {T1_RANDOM_MEAN}p:")
    print("    if similar -> the signal's ENTRY TIMING adds nothing either (neither when nor which-way).")
