#!/usr/bin/env python3
"""
BB-excursion accumulation entry (user idea, 2026-07-01) — backtest, MA x sigma sweep.

Signal (H1, Bollinger = MA(period) +/- k*sigma):
  MA in {SMA, EMA}, period in {50,100,150,200}, k(sigma) in {1.0,1.5,2.0,2.5}.
  - Per H1 bar, signed fraction of the bar's RANGE outside the bands:
      frac = clip((high - max(upper,low))/(high-low),0,1)     # portion above upper (+)
           - clip((min(lower,high) - low)/(high-low),0,1)     # portion below lower (-)
  - Accumulate a rolling sum over N bars -> accum in [-N, +N].
  - Entry, two directions:
      momentum   (dir=+1): long if accum > +T_enter ; short if accum < -T_enter
      contrarian (dir=-1): fade -> short if accum > +T_enter ; long if accum < -T_enter
  - Exit: signal hysteresis (|accum| back inside T_exit) OR hard SL.
  - Real per-bar spread (H1 close bid/ask), worse-side fills, stop slippage (SOP R3/R3a).
    sigma = rolling std(period) for both SMA and EMA centres (clean SMA-vs-EMA compare).

Reuses the frozen harness (data.load_pair_ba S5-BA R3b; validate.*). Resamples S5->H1 and
DISCARDS S5 per pair (one pair loaded at a time; H1 small -> memory-safe).
"""
import sys, gc
import numpy as np
import pandas as pd
import numba as nb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import load_pair_ba
from validate import split_is_oos, walk_forward, monte_carlo, equity_drawdown

PAIRS   = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
MA_GRID = [(t, p) for t in ('sma', 'ema') for p in (50, 100, 150, 200)]
K_GRID  = [1.0, 1.5, 2.0, 2.5]
N_GRID  = [5, 10, 20]
TEN_FR  = [0.20, 0.30, 0.40, 0.50]
TEX_FR  = [0.00, 0.10, 0.20]
SL_PIPS = 150.0
SLIP    = 2.0
IS_FRAC = 4.0 / 6.0


def resample_h1(d):
    df = pd.DataFrame({
        'ts': pd.to_datetime(d['ts']),
        'o': d['m5_o'], 'h': d['m5_h'], 'l': d['m5_l'], 'c': d['m5_c'],
        'bid_c': d['bid_c'], 'ask_c': d['ask_c'],
    }).set_index('ts')
    return df.resample('1h').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'),
                                 c=('c', 'last'), bid_c=('bid_c', 'last'),
                                 ask_c=('ask_c', 'last')).dropna()


def make_frac(c, h, l, ma_type, period, k):
    cs = pd.Series(c)
    center = (cs.ewm(span=period, adjust=False).mean() if ma_type == 'ema'
              else cs.rolling(period).mean()).to_numpy()
    sd = cs.rolling(period).std(ddof=0).to_numpy()
    upper = center + k * sd
    lower = center - k * sd
    rng = h - l
    rng = np.where(rng > 0, rng, np.nan)
    above = np.clip((h - np.maximum(upper, l)) / rng, 0.0, 1.0)
    below = np.clip((np.minimum(lower, h) - l) / rng, 0.0, 1.0)
    frac = np.nan_to_num(above) - np.nan_to_num(below)
    frac[:period] = 0.0
    return frac.astype(np.float64)


@nb.njit(cache=True)
def _kern(h, l, bid_c, ask_c, accum, start, pip, direction,
          t_enter, t_exit, sl_pips, slip):
    n = len(h)
    pos = 0; entry_fill = 0.0; ebar = -1
    eb = np.empty(n, np.int64); pn = np.empty(n, np.float64); nt = 0
    for i in range(start, n):
        a = accum[i]
        if pos == 0:
            go_long = False; go_short = False
            if direction == 1:
                if a > t_enter:  go_long = True
                elif a < -t_enter: go_short = True
            else:
                if a > t_enter:  go_short = True
                elif a < -t_enter: go_long = True
            if go_long:
                pos = 1; entry_fill = ask_c[i]; ebar = i
            elif go_short:
                pos = -1; entry_fill = bid_c[i]; ebar = i
        else:
            ex_fill = 0.0; hit = False
            slv = entry_fill - pos * sl_pips * pip
            if pos == 1 and l[i] <= slv:
                ex_fill = slv - slip * pip; hit = True
            elif pos == -1 and h[i] >= slv:
                ex_fill = slv + slip * pip; hit = True
            if not hit:
                if pos == 1:
                    if direction == 1 and a < t_exit:    ex_fill = bid_c[i]; hit = True
                    elif direction == -1 and a > -t_exit: ex_fill = bid_c[i]; hit = True
                else:
                    if direction == 1 and a > -t_exit:   ex_fill = ask_c[i]; hit = True
                    elif direction == -1 and a < t_exit:  ex_fill = ask_c[i]; hit = True
            if hit:
                pnl = ((ex_fill - entry_fill) / pip if pos == 1
                       else (entry_fill - ex_fill) / pip)
                eb[nt] = ebar; pn[nt] = pnl; nt += 1; pos = 0
    return eb[:nt], pn[:nt]


def load_h1(pair):
    d = load_pair_ba(pair); pip = d['pip']
    r = resample_h1(d); del d; gc.collect()
    out = {'h': r['h'].to_numpy(np.float64), 'l': r['l'].to_numpy(np.float64),
           'c': r['c'].to_numpy(np.float64),
           'bid_c': r['bid_c'].to_numpy(np.float64),
           'ask_c': r['ask_c'].to_numpy(np.float64),
           'pip': pip, 'n': len(r), 'is_end': int(len(r) * IS_FRAC)}
    del r; gc.collect()
    return out


def run():
    print(f"BB-excursion accumulation | MA{{sma,ema}} x period{{50,100,150,200}} x "
          f"sigma{K_GRID} H1 | SL={SL_PIPS}p", flush=True)
    H1 = {}
    for p in PAIRS:
        H1[p] = load_h1(p)
        print(f"  {p}: {H1[p]['n']} H1 bars, is_end={H1[p]['is_end']}", flush=True)
    FRAC = {}
    for p in PAIRS:
        s = H1[p]
        for (t, per) in MA_GRID:
            for k in K_GRID:
                FRAC[(p, t, per, k)] = make_frac(s['c'], s['h'], s['l'], t, per, k)
    print("  frac precomputed for all MA x sigma configs\n", flush=True)

    results = []
    for (mt, per) in MA_GRID:
        for k in K_GRID:
            for direction, dname in [(1, 'mom'), (-1, 'contra')]:
                for N in N_GRID:
                    for ten in TEN_FR:
                        for tex in TEX_FR:
                            Te = ten * N; Tx = tex * N
                            if Tx >= Te:
                                continue
                            pis = pos_ = 0.0; oosp = 0; ntot = 0; per_pair = {}
                            for p in PAIRS:
                                s = H1[p]
                                accum = np.nan_to_num(pd.Series(FRAC[(p, mt, per, k)]).rolling(N).sum().to_numpy())
                                eb, pn = _kern(s['h'], s['l'], s['bid_c'], s['ask_c'], accum,
                                               per + N, s['pip'], direction, Te, Tx, SL_PIPS, SLIP)
                                io = split_is_oos(eb, pn, s['is_end'])
                                pis += io['is_net']; pos_ += io['oos_net']; ntot += len(pn)
                                if io['oos_net'] > 0: oosp += 1
                                per_pair[p] = (len(pn), io['is_net'], io['oos_net'])
                            results.append({'band': f"{mt}{per}/{k}", 'mt': mt, 'per': per, 'k': k,
                                            'dir': dname, 'N': N, 'Te': round(Te, 2), 'Tx': round(Tx, 2),
                                            'is': pis, 'oos': pos_, 'oosp': oosp, 'ntot': ntot,
                                            'pp': per_pair})

    good = [r for r in results if r['is'] > 0 and r['oos'] > 0]
    good.sort(key=lambda r: r['oos'], reverse=True)
    print(f"=== {len(results)} configs | {len(good)} with BOTH IS>0 and OOS>0 ===", flush=True)

    # pass-count matrix: rows = MA config, cols = sigma
    print("\n-- pass count (IS>0 & OOS>0) : rows=MA, cols=sigma --", flush=True)
    print(f"{'MA':<9}" + "".join(f"k={k:<5}" for k in K_GRID), flush=True)
    for (mt, per) in MA_GRID:
        row = f"{mt}{per:<6}"
        for k in K_GRID:
            c = sum(1 for r in good if r['mt'] == mt and r['per'] == per and r['k'] == k)
            row += f"{c:<7}"
        print(row, flush=True)

    print(f"\n-- top {min(30,len(good))} configs by OOS net --", flush=True)
    print(f"{'band':<12}{'dir':<7}{'N':>3}{'Te':>6}{'Tx':>6}{'IS':>9}{'OOS':>9}{'oosP':>5}{'n':>6}", flush=True)
    for r in good[:30]:
        print(f"{r['band']:<12}{r['dir']:<7}{r['N']:>3}{r['Te']:>6.1f}{r['Tx']:>6.1f}"
              f"{r['is']:>9.0f}{r['oos']:>9.0f}{r['oosp']:>5}{r['ntot']:>6}", flush=True)

    if not good:
        print("\nNo config IS+OOS positive net of spread — no edge in any MA/sigma/direction.", flush=True)
        return

    best = good[0]
    print(f"\n=== BEST: {best['band']} {best['dir']} N={best['N']} Te={best['Te']} Tx={best['Tx']} "
          f"| IS={best['is']:.0f} OOS={best['oos']:.0f} oosP={best['oosp']}/4 ===", flush=True)
    for p in PAIRS:
        nt, isn, oosn = best['pp'][p]
        print(f"  {p}: n={nt} IS={isn:.0f} OOS={oosn:.0f}", flush=True)
    direction = 1 if best['dir'] == 'mom' else -1
    pooled = []
    for p in PAIRS:
        s = H1[p]
        accum = np.nan_to_num(pd.Series(FRAC[(p, best['mt'], best['per'], best['k'])]).rolling(best['N']).sum().to_numpy())
        eb, pn = _kern(s['h'], s['l'], s['bid_c'], s['ask_c'], accum, best['per'] + best['N'],
                       s['pip'], direction, best['Te'], best['Tx'], SL_PIPS, SLIP)
        pooled.append(pn)
    pooled = np.concatenate(pooled) if pooled else np.array([])
    mc = monte_carlo(pooled, n=300); dd = equity_drawdown(np.cumsum(pooled))
    print(f"\n  pooled n={len(pooled)} net={pooled.sum():.0f}p  MC p_net={mc['p_net']:.3f} "
          f"p_maxdd={mc['p_maxdd']:.3f} maxDD={dd['max_dd']:.0f}p", flush=True)
    print(f"  GATE: IS+OOS>0 [{'PASS' if best['is']>0 and best['oos']>0 else 'FAIL'}]  "
          f"oosP>=3 [{'PASS' if best['oosp']>=3 else 'FAIL'}]  "
          f"MC p_net<0.05 [{'PASS' if mc['p_net']<0.05 else 'FAIL'}]", flush=True)


if __name__ == "__main__":
    run()
