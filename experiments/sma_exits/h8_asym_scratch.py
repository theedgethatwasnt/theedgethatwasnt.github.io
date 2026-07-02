"""H8 — Asymmetric scratch: only fire when trade is at break-even or slightly green.

Numba kernel, identical entry to sma_live (SMA16 lags=(8,10,15) H1+M30).
Sweeps both symmetric (H7 baseline) and asymmetric scratch.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, WINDOW_BARS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

T_ACT_HOURS = [6, 8, 12, 24, 48]

# Symmetric (H7 baseline) needs wider windows to catch meander; asym
# only makes sense at small windows (the whole point is "slightly above b/e").
SYM_X_FIXED  = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
SYM_K_ATR    = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ASYM_X_FIXED = [0.5, 1.0, 2.0, 3.0, 5.0]
ASYM_K_ATR   = [0.1, 0.2, 0.3, 0.5, 0.8]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5, h1_atr_m5,
           pip, T_bars, mode, x_or_k, sym):
    """mode: 0=fixed (x_or_k is pip count), 1=atr (x_or_k is ATR coefficient)
    sym: 1=symmetric (|d|≤W), 0=asymmetric (0≤d×dir≤W)."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1; W_pips = 0.0
    pnls = np.empty(n, np.float64)
    ents = np.empty(n, np.int64)
    nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            held = i - entry_bar
            exit_px = 0.0; reason = -1
            tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
            if pos == 1 and highs[i] >= tp_lvl:
                exit_px = tp_lvl; reason = 0
            elif pos == -1 and lows[i] <= tp_lvl:
                exit_px = tp_lvl; reason = 0
            if reason < 0 and held >= T_bars:
                d_signed = (closes[i] - entry_px) / pip * pos
                if sym == 1:
                    hit = abs(d_signed) <= W_pips
                else:
                    hit = (0.0 <= d_signed) and (d_signed <= W_pips)
                if hit:
                    exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar
                nt += 1
                pos = 0; continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i
            if mode == 0:
                W_pips = x_or_k
            else:
                a = h1_atr_m5[i-1]
                if np.isnan(a):
                    pos = 0; continue
                W_pips = x_or_k * (a / pip)
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar
        nt += 1
    return pnls[:nt], ents[:nt]


def main():
    print("="*92)
    print("  H8 — ASYMMETRIC SCRATCH vs SYMMETRIC (H7) BASELINE")
    print(f"  T={T_ACT_HOURS} h   SYM X={SYM_X_FIXED} p K={SYM_K_ATR}   "
          f"ASYM X={ASYM_X_FIXED} p K={ASYM_K_ATR}")
    print("="*92)

    # Warmup JIT
    _o = np.zeros(100); _s = np.zeros(100, np.int8); _a = np.full(100, 1.0)
    kernel(_o, _o, _o, _o, _s, _a, 0.0001, 12, 0, 1.0, 1)

    rows = []
    t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair)
        sp = b['spread_cost']; pip = b['pip']
        print(f"  {pair}  cost={sp:.2f}p")
        for T_h in T_ACT_HOURS:
            T_bars = T_h * BARS_PER_H1
            # Symmetric (H7 baseline) — wider windows
            for X in SYM_X_FIXED:
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_atr_m5'], pip, T_bars, 0, X, 1)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'mode':'sym_fixed','T_h':T_h,'param':X,**s})
            for k in SYM_K_ATR:
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_atr_m5'], pip, T_bars, 1, k, 1)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'mode':'sym_atr','T_h':T_h,'param':k,**s})
            # Asymmetric — only small windows make sense
            for X in ASYM_X_FIXED:
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_atr_m5'], pip, T_bars, 0, X, 0)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'mode':'asym_fixed','T_h':T_h,'param':X,**s})
            for k in ASYM_K_ATR:
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_atr_m5'], pip, T_bars, 1, k, 0)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'mode':'asym_atr','T_h':T_h,'param':k,**s})

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT/'h8_asym_scratch.csv', index=False)
    print(f"\n  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")

    # Compare sym vs asym head-to-head per pair
    print("\n=== BEST IS+OOS+ PER PAIR — SYM (H7) vs ASYM (H8) ===")
    sym_total = 0.0; asym_total = 0.0; sym_pairs = 0; asym_pairs = 0
    print(f"  {'Pair':<9} | {'SYM (H7-style)':<48} | {'ASYM (H8)':<48}")
    print("  " + "-"*9 + "-+-" + "-"*48 + "-+-" + "-"*48)
    for pair in PAIRS:
        sub = rdf[rdf.pair == pair]
        sym_sub  = sub[sub['mode'].str.startswith('sym')]
        asym_sub = sub[sub['mode'].str.startswith('asym')]
        sym_pass = sym_sub[(sym_sub.is_net>0)&(sym_sub.oos_net>0)].sort_values('oos_pd', ascending=False)
        asym_pass = asym_sub[(asym_sub.is_net>0)&(asym_sub.oos_net>0)].sort_values('oos_pd', ascending=False)
        if len(sym_pass):
            r = sym_pass.iloc[0]
            left = (f"{r['mode']:<10} T={int(r['T_h']):>2}h p={r['param']:>5.2f} "
                    f"IS={r['is_pd']:+6.2f} OOS={r['oos_pd']:+6.2f} DD={r['oos_dd']:+6.0f}")
            sym_total += r['oos_pd']; sym_pairs += 1
        else:
            left = "  -- no IS+OOS+ config --"
        if len(asym_pass):
            r = asym_pass.iloc[0]
            right = (f"{r['mode']:<10} T={int(r['T_h']):>2}h p={r['param']:>5.2f} "
                     f"IS={r['is_pd']:+6.2f} OOS={r['oos_pd']:+6.2f} DD={r['oos_dd']:+6.0f}")
            asym_total += r['oos_pd']; asym_pairs += 1
        else:
            right = "  -- no IS+OOS+ config --"
        print(f"  {pair:<9} | {left:<48} | {right:<48}")
    print()
    print(f"  SYM  : {sym_pairs:>2}/10 pairs   Σ OOS_pd = {sym_total:+7.2f}")
    print(f"  ASYM : {asym_pairs:>2}/10 pairs   Σ OOS_pd = {asym_total:+7.2f}")
    print(f"  Δ    : asym − sym = {asym_total - sym_total:+.2f}  p/d aggregate")


if __name__ == '__main__':
    main()
