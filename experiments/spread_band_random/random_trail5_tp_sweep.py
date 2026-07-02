"""
Part 3: best trail (=5p, best mean/trade in Part 2) + a FIXED TP on top, sweep TP.

Exit = first of:  fixed TP (+X net)  |  5p trailing stop  |  60-min time stop  |  session gap.
Within-bar order per SOP R2: bull bar -> HIGH then LOW; bear bar -> LOW then HIGH.
Random entry, always-in-market, one-at-a-time, spread charged up front, PnL on MID.
Regime = trailing-12-bar avg spread band (same as Parts 1-2).

Usage: python3 random_trail5_tp_sweep.py [PAIR] [SEED] [TRAIL]
"""
import sys
import numpy as np
import duckdb
from numba import njit

PAIR     = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
SEED     = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
TRAIL    = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0   # best trail from Part 2
K_SPREAD = 12
HOLD_BARS = 720
GAP_SECS = 60
TPS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
PIP = 0.01 if PAIR.endswith("JPY") else 0.0001
PATH = f"data/s5_ohlc/{PAIR}_S5_BA.parquet"


@njit(cache=True)
def simulate(opn, high, low, close, spread, regime, ts, start, seed, trail_pips, tp_pips):
    n = high.shape[0]
    np.random.seed(seed)
    trail = trail_pips * PIP
    mt = n // 2 + 8
    out_pnl = np.empty(mt, np.float64)
    out_reg = np.empty(mt, np.float64)
    out_rsn = np.empty(mt, np.int64)   # 0=TP 1=trail 2=time 3=gap
    t = 0
    i = start
    while i < n - 1:
        d = 1.0 if np.random.random() < 0.5 else -1.0
        entry = close[i]
        sp = spread[i] / PIP
        reg = regime[i] / PIP
        tp_off = (tp_pips + sp) * PIP        # mid move for +tp net
        last = i + HOLD_BARS
        if last > n - 1:
            last = n - 1
        hwm = entry
        lwm = entry
        exit_i = -1; pnl = 0.0; rsn = 2
        j = i + 1
        while j <= last:
            if ts[j] - ts[j - 1] > GAP_SECS:
                jj = j - 1
                pnl = d * (close[jj] - entry) / PIP - sp; exit_i = jj; rsn = 3; break
            bull = close[j] >= opn[j]
            h = high[j]; l = low[j]
            if d > 0.0:                                  # LONG: TP above, trail below
                tp_lvl = entry + tp_off
                if bull:                                 # HIGH then LOW
                    if h >= tp_lvl:
                        pnl = tp_pips; exit_i = j; rsn = 0; break
                    if h > hwm: hwm = h
                    if l <= hwm - trail:
                        pnl = (hwm - trail - entry) / PIP - sp; exit_i = j; rsn = 1; break
                else:                                    # LOW then HIGH
                    if l <= hwm - trail:
                        pnl = (hwm - trail - entry) / PIP - sp; exit_i = j; rsn = 1; break
                    if h >= tp_lvl:
                        pnl = tp_pips; exit_i = j; rsn = 0; break
                    if h > hwm: hwm = h
            else:                                        # SHORT: TP below, trail above
                tp_lvl = entry - tp_off
                if bull:                                 # HIGH then LOW
                    if h >= lwm + trail:
                        pnl = (entry - (lwm + trail)) / PIP - sp; exit_i = j; rsn = 1; break
                    if l <= tp_lvl:
                        pnl = tp_pips; exit_i = j; rsn = 0; break
                    if l < lwm: lwm = l
                else:                                    # LOW then HIGH
                    if l <= tp_lvl:
                        pnl = tp_pips; exit_i = j; rsn = 0; break
                    if l < lwm: lwm = l
                    if h >= lwm + trail:
                        pnl = (entry - (lwm + trail)) / PIP - sp; exit_i = j; rsn = 1; break
            j += 1
        if exit_i < 0:
            exit_i = last
            pnl = d * (close[last] - entry) / PIP - sp; rsn = 2
        out_pnl[t] = pnl; out_reg[t] = reg; out_rsn[t] = rsn
        t += 1
        if exit_i <= i:
            exit_i = i + 1
        i = exit_i
    return out_pnl[:t], out_reg[:t], out_rsn[:t]


def load():
    df = duckdb.sql(
        f"SELECT epoch(timestamp)::BIGINT ts, open, high, low, close, (ask_c-bid_c) sp "
        f"FROM '{PATH}' WHERE ask_c > bid_c ORDER BY timestamp"
    ).df()
    ts    = df["ts"].to_numpy(np.int64)
    opn   = df["open"].to_numpy(np.float64)
    high  = df["high"].to_numpy(np.float64)
    low   = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    sp    = df["sp"].to_numpy(np.float64)
    csum = np.cumsum(sp)
    regime = np.empty_like(sp)
    regime[:K_SPREAD] = csum[:K_SPREAD] / (np.arange(K_SPREAD) + 1)
    regime[K_SPREAD:] = (csum[K_SPREAD:] - csum[:-K_SPREAD]) / K_SPREAD
    return ts, opn, high, low, close, sp, regime


def main():
    print(f"Loading {PATH} ...")
    ts, opn, high, low, close, sp, regime = load()
    print(f"{len(sp):,} bars   TRAIL={TRAIL:.0f}p (fixed, best from Part 2)\n")
    edges = np.unique(np.round(np.quantile(regime / PIP, np.linspace(0, 1, 6)), 4))
    band_lbl = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]

    print(f"Random entry, exit = TP | {TRAIL:.0f}p-trail | 60-min | gap.  "
          f"Regime = trailing {K_SPREAD}-bar avg spread.\n")
    print(f"{'TP':>4} | {'trades':>7} | {'net':>9} | {'mean':>7} | {'win%':>5} | "
          f"{'TP%':>5} | {'trail%':>6} | {'time%':>5} | {'P5':>6} | {'worst':>7}")
    print("-" * 84)
    band_means = {}
    for tp in TPS:
        pnl, reg, rsn = simulate(opn, high, low, close, sp, regime, ts, K_SPREAD, SEED, TRAIL, float(tp))
        print(f"{tp:>4} | {len(pnl):>7,} | {pnl.sum():>9,.0f} | {pnl.mean():>7.3f} | "
              f"{100*(pnl>0).mean():>5.1f} | {100*(rsn==0).mean():>5.1f} | {100*(rsn==1).mean():>6.1f} | "
              f"{100*(rsn==2).mean():>5.1f} | {np.percentile(pnl,5):>6.2f} | {pnl.min():>7.1f}")
        bm = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (reg >= a) & (reg < b) if b != edges[-1] else (reg >= a) & (reg <= b)
            bm.append(pnl[m].mean() if m.sum() else np.nan)
        band_means[tp] = bm

    print(f"\nMean pips/trade by spread band x TP (trail={TRAIL:.0f}p) — 'does the band still help?'\n")
    print(f"{'TP':>4} | " + " | ".join(f"{l:>11}" for l in band_lbl) + " | best")
    print("-" * (7 + 14 * len(band_lbl) + 12))
    for tp in TPS:
        bm = band_means[tp]
        best = int(np.nanargmax(bm))
        print(f"{tp:>4} | " + " | ".join(f"{v:>11.3f}" for v in bm) +
              f" | {band_lbl[best]}")
    print("\n(Lowest band = calmest spread. All means net of spread; random entry has no edge.)")


if __name__ == "__main__":
    main()
