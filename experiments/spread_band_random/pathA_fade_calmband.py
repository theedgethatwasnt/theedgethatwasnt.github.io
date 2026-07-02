"""
Path A: replace random entry with a contrarian FADE (and FOLLOW control) of a fast
extension, gated to the CALM spread band. Same exit toolkit: trail=5 + fixed TP,
60-min stop, spread charged up front, MID PnL, SOP R2 within-bar order. One trade
at a time; after a trade closes, resume scanning for the next signal.

Signal at bar i (causal): move = (close[i]-close[i-M])/pip over the last M bars.
 - FADE:   ran up >= N  -> SHORT ;  ran down <= -N -> LONG
 - FOLLOW: ran up >= N  -> LONG  ;  ran down <= -N -> SHORT
Gate: trade only if trailing-12-bar avg spread <= CALM_THR pips.

Usage: python3 pathA_fade_calmband.py [PAIR]
"""
import numpy as np
import duckdb
from numba import njit

PAIR = "EUR_USD"
K_SPREAD = 12
HOLD_BARS = 720
GAP_SECS = 60
TRAIL = 5.0
TP = 2.0
CALM_THR = 1.50            # lowest spread quintile edge (pips)
PIP = 0.01 if PAIR.endswith("JPY") else 0.0001
PATH = f"data/s5_ohlc/{PAIR}_S5_BA.parquet"

Ms   = [12, 24, 60, 120]          # lookback bars: 1, 2, 5, 10 min
Ns   = [3, 5, 7, 10]              # extension threshold (pips)
MODES = [(-1, "fade"), (1, "follow")]


@njit(cache=True)
def run(opn, high, low, close, spread, regime, ts, M, N, mode, calm_thr, trail_pips, tp_pips, use_calm):
    n = high.shape[0]
    trail = trail_pips * PIP
    mt = n // 4 + 8
    out_pnl = np.empty(mt, np.float64)
    out_reg = np.empty(mt, np.float64)
    t = 0
    i = M + 1
    while i < n - 1:
        reg = regime[i] / PIP
        if use_calm and reg > calm_thr:
            i += 1; continue
        move = (close[i] - close[i - M]) / PIP
        d = 0.0
        if mode < 0:                      # fade
            if move >= N: d = -1.0
            elif move <= -N: d = 1.0
        else:                             # follow
            if move >= N: d = 1.0
            elif move <= -N: d = -1.0
        if d == 0.0:
            i += 1; continue
        entry = close[i]
        sp = spread[i] / PIP
        tp_off = (tp_pips + sp) * PIP
        last = i + HOLD_BARS
        if last > n - 1:
            last = n - 1
        hwm = entry; lwm = entry
        exit_i = -1; pnl = 0.0
        j = i + 1
        while j <= last:
            if ts[j] - ts[j - 1] > GAP_SECS:
                jj = j - 1
                pnl = d * (close[jj] - entry) / PIP - sp; exit_i = jj; break
            bull = close[j] >= opn[j]; h = high[j]; l = low[j]
            if d > 0.0:
                tp_lvl = entry + tp_off
                if bull:
                    if h >= tp_lvl: pnl = tp_pips; exit_i = j; break
                    if h > hwm: hwm = h
                    if l <= hwm - trail: pnl = (hwm - trail - entry) / PIP - sp; exit_i = j; break
                else:
                    if l <= hwm - trail: pnl = (hwm - trail - entry) / PIP - sp; exit_i = j; break
                    if h >= tp_lvl: pnl = tp_pips; exit_i = j; break
                    if h > hwm: hwm = h
            else:
                tp_lvl = entry - tp_off
                if bull:
                    if h >= lwm + trail: pnl = (entry - (lwm + trail)) / PIP - sp; exit_i = j; break
                    if l <= tp_lvl: pnl = tp_pips; exit_i = j; break
                    if l < lwm: lwm = l
                else:
                    if l <= tp_lvl: pnl = tp_pips; exit_i = j; break
                    if l < lwm: lwm = l
                    if h >= lwm + trail: pnl = (entry - (lwm + trail)) / PIP - sp; exit_i = j; break
            j += 1
        if exit_i < 0:
            exit_i = last
            pnl = d * (close[last] - entry) / PIP - sp
        out_pnl[t] = pnl; out_reg[t] = reg; t += 1
        if exit_i <= i: exit_i = i + 1
        i = exit_i
    return out_pnl[:t], out_reg[:t]


def load():
    df = duckdb.sql(
        f"SELECT epoch(timestamp)::BIGINT ts, open, high, low, close, (ask_c-bid_c) sp "
        f"FROM '{PATH}' WHERE ask_c > bid_c ORDER BY timestamp").df()
    a = lambda c, t: df[c].to_numpy(t)
    ts = a("ts", np.int64)
    opn, high, low, close, sp = (a("open", np.float64), a("high", np.float64),
                                 a("low", np.float64), a("close", np.float64), a("sp", np.float64))
    csum = np.cumsum(sp); regime = np.empty_like(sp)
    regime[:K_SPREAD] = csum[:K_SPREAD] / (np.arange(K_SPREAD) + 1)
    regime[K_SPREAD:] = (csum[K_SPREAD:] - csum[:-K_SPREAD]) / K_SPREAD
    return ts, opn, high, low, close, sp, regime


def main():
    print(f"Loading {PATH} ...")
    ts, opn, high, low, close, sp, regime = load()
    print(f"{len(sp):,} bars.  Exit: trail={TRAIL:.0f}p + TP={TP:.0f}p, 60-min stop. "
          f"Calm gate: trailing avg spread <= {CALM_THR} pips.\n")
    print(f"{'mode':>7} | {'M(min)':>6} | {'N(p)':>4} | {'trades':>7} | {'mean':>7} | {'win%':>5} | "
          f"{'net':>9} | {'P5':>6} | {'worst':>7}")
    print("-" * 78)
    best = None
    for mode, mlbl in MODES:
        for M in Ms:
            for N in Ns:
                pnl, reg = run(opn, high, low, close, sp, regime, ts, M, float(N), float(mode),
                               CALM_THR, TRAIL, TP, True)
                if len(pnl) < 200:
                    continue
                row = (mlbl, M * 5 // 60, N, len(pnl), pnl.mean(), 100 * (pnl > 0).mean(),
                       pnl.sum(), np.percentile(pnl, 5), pnl.min())
                print(f"{row[0]:>7} | {row[1]:>6} | {row[2]:>4} | {row[3]:>7,} | {row[4]:>7.3f} | "
                      f"{row[5]:>5.1f} | {row[6]:>9,.0f} | {row[7]:>6.2f} | {row[8]:>7.1f}")
                if best is None or row[4] > best[4]:
                    best = row
        print()
    print(f"BEST cell: {best[0]} M={best[1]}min N={best[2]}p -> mean {best[4]:.3f} p/trade "
          f"({best[3]:,} trades, win {best[5]:.1f}%)")
    print("Random-entry calm-band baseline (Part 3, trail5/TP2): ~ -1.46 p/trade.")


if __name__ == "__main__":
    main()
