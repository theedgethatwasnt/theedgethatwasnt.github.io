"""
Same as random_tp1_spreadband.py but the exit is a TRAILING STOP (no fixed TP).
Sweep trail distance 5,7,9,...,21 pips. Question: does conditioning on the
trailing-spread band help under a trailing-stop exit?

Exit logic (per trade):
- Long:  hwm starts at entry; stop = hwm - trail; exit when low <= stop.
- Short: lwm starts at entry; stop = lwm + trail; exit when high >= stop.
- Within-bar order per SOP R2: bull bar (close>=open) -> HIGH then LOW; bear -> LOW then HIGH.
- Still: random entry, always in market, one-at-a-time, spread charged up front,
  60-min time stop, session-gap close. PnL on MID; net of entry spread.

Usage: python3 random_trail_sweep.py [PAIR] [SEED]
"""
import sys
import numpy as np
import duckdb
from numba import njit

PAIR    = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
SEED    = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
K_SPREAD = 12
HOLD_BARS = 720
GAP_SECS = 60
TRAILS = [5, 7, 9, 11, 13, 15, 17, 19, 21]
PIP = 0.01 if PAIR.endswith("JPY") else 0.0001
PATH = f"data/s5_ohlc/{PAIR}_S5_BA.parquet"


@njit(cache=True)
def simulate(opn, high, low, close, spread, regime, ts, start, seed, trail_pips):
    n = high.shape[0]
    np.random.seed(seed)
    trail = trail_pips * PIP
    max_trades = n // 2 + 8
    out_pnl = np.empty(max_trades, np.float64)
    out_reg = np.empty(max_trades, np.float64)
    out_dur = np.empty(max_trades, np.int64)
    out_won = np.empty(max_trades, np.int64)
    t = 0
    i = start
    while i < n - 1:
        d = 1.0 if np.random.random() < 0.5 else -1.0
        entry = close[i]
        sp = spread[i] / PIP
        reg = regime[i] / PIP
        last = i + HOLD_BARS
        if last > n - 1:
            last = n - 1
        hwm = entry          # long high-water mark
        lwm = entry          # short low-water mark
        exit_i = -1
        pnl = 0.0
        j = i + 1
        while j <= last:
            if ts[j] - ts[j - 1] > GAP_SECS:               # session gap
                jj = j - 1
                pnl = d * (close[jj] - entry) / PIP - sp
                exit_i = jj
                break
            bull = close[j] >= opn[j]
            if d > 0.0:                                     # LONG: stop = hwm - trail
                if bull:
                    if high[j] > hwm: hwm = high[j]
                    stop = hwm - trail
                    if low[j] <= stop:
                        pnl = (stop - entry) / PIP - sp; exit_i = j; break
                else:
                    stop = hwm - trail
                    if low[j] <= stop:
                        pnl = (stop - entry) / PIP - sp; exit_i = j; break
                    if high[j] > hwm: hwm = high[j]
            else:                                           # SHORT: stop = lwm + trail
                if bull:
                    stop = lwm + trail
                    if high[j] >= stop:
                        pnl = (entry - stop) / PIP - sp; exit_i = j; break
                    if low[j] < lwm: lwm = low[j]
                else:
                    if low[j] < lwm: lwm = low[j]
                    stop = lwm + trail
                    if high[j] >= stop:
                        pnl = (entry - stop) / PIP - sp; exit_i = j; break
            j += 1
        if exit_i < 0:                                      # time stop
            exit_i = last
            pnl = d * (close[last] - entry) / PIP - sp
        out_pnl[t] = pnl
        out_reg[t] = reg
        out_dur[t] = exit_i - i
        out_won[t] = 1 if pnl > 0.0 else 0
        t += 1
        if exit_i <= i:
            exit_i = i + 1
        i = exit_i
    return out_pnl[:t], out_reg[:t], out_dur[:t], out_won[:t]


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
    print(f"{len(sp):,} bars\n")

    # fixed spread-band edges from the regime distribution (same bands for all trails)
    edges = np.unique(np.round(np.quantile(regime / PIP, np.linspace(0, 1, 6)), 4))  # quintiles
    band_lbl = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]

    print(f"Random entry, TRAILING-STOP exit, no fixed TP. {HOLD_BARS*5//60}-min time stop. "
          f"Regime = trailing {K_SPREAD}-bar avg spread.\n")
    print(f"{'trail':>5} | {'trades':>7} | {'net':>9} | {'mean':>7} | {'win%':>5} | "
          f"{'dur':>5} | {'P5':>6} | {'P1':>6} | {'worst':>7} | {'<-20%':>6}")
    print("-" * 86)

    band_means = {}   # trail -> list of per-band mean
    for tr in TRAILS:
        pnl, reg, dur, won = simulate(opn, high, low, close, sp, regime, ts, K_SPREAD, SEED, float(tr))
        print(f"{tr:>5} | {len(pnl):>7,} | {pnl.sum():>9,.0f} | {pnl.mean():>7.3f} | "
              f"{100*won.mean():>5.1f} | {dur.mean()*5/60:>4.1f}m | "
              f"{np.percentile(pnl,5):>6.2f} | {np.percentile(pnl,1):>6.2f} | "
              f"{pnl.min():>7.1f} | {100*(pnl<-20).mean():>6.2f}")
        bm = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (reg >= a) & (reg < b) if b != edges[-1] else (reg >= a) & (reg <= b)
            bm.append(pnl[m].mean() if m.sum() else np.nan)
        band_means[tr] = bm

    # ---- does the band help? mean P&L per spread band, per trail ----
    print(f"\nMean pips/trade by spread band (columns) x trail (rows) — 'does the band help?'\n")
    print(f"{'trail':>5} | " + " | ".join(f"{l:>11}" for l in band_lbl) + " | best band")
    print("-" * (8 + 14 * len(band_lbl) + 12))
    for tr in TRAILS:
        bm = band_means[tr]
        best = int(np.nanargmax(bm))
        print(f"{tr:>5} | " + " | ".join(f"{v:>11.3f}" for v in bm) +
              f" | {band_lbl[best]} ({bm[best]:.2f})")
    print("\n(Band edges are quintiles of trailing avg spread, pips. "
          "Lowest band = calmest/tightest spread.)")


if __name__ == "__main__":
    main()
