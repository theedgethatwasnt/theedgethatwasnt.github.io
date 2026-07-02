"""
Random-entry, always-in-market, TP=+1 pip net of spread, no SL, 60-min time stop.
Regime variable = trailing average spread (last K S5 bars).

Question: is there a band of (recent) spread in which deep losers are rarer?

Model (SOP-faithful):
- Signals/PnL on MID OHLC (open/high/low/close in the parquet are mid).
- Spread charged up front: net_pnl_pips = dir*(exit_mid - entry_mid)/pip - spread_entry.
- TP fires when dir*(mid_move)/pip >= 1 + spread_entry  (i.e. +1 pip AFTER spread).
- No stop. Close at 60 min (=720 S5 bars) OR at a session gap (so no weekend holds).
- Always in market, one trade at a time: next trade opens at the bar the prior closed.
- Regime = mean spread over trailing K S5 bars at entry (K default 12 = ~1 min).

Usage: python3 random_tp1_spreadband.py [PAIR] [SEED]
"""
import sys
import numpy as np
import duckdb
from numba import njit

PAIR    = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
SEED    = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
K_SPREAD = 12          # trailing bars for the spread regime (~1 min)
HOLD_BARS = 720        # 60 min / 5 s
TP_NET   = 1.0         # target pips net of spread
GAP_SECS = 60          # >60s between bars => session boundary, close the trade
PIP = 0.01 if PAIR.endswith("JPY") else 0.0001
PATH = f"data/s5_ohlc/{PAIR}_S5_BA.parquet"


@njit(cache=True)
def simulate(high, low, close, spread, regime, ts, start, seed):
    """Single forward pass; returns (pnl, regime_at_entry, duration_bars, won) per trade."""
    n = high.shape[0]
    np.random.seed(seed)
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
        sp = spread[i] / PIP                 # entry spread in pips (round-trip cost)
        thr = (TP_NET + sp) * PIP            # mid move (price) needed for +1 net
        reg = regime[i] / PIP                # trailing-avg spread in pips
        last = i + HOLD_BARS
        if last > n - 1:
            last = n - 1
        exit_i = -1
        won = 0
        pnl = 0.0
        j = i + 1
        while j <= last:
            if ts[j] - ts[j - 1] > GAP_SECS:        # session gap -> close at prior bar
                jj = j - 1
                pnl = d * (close[jj] - entry) / PIP - sp
                exit_i = jj
                break
            if d > 0.0:
                if high[j] - entry >= thr:           # long TP
                    pnl = TP_NET; won = 1; exit_i = j; break
            else:
                if entry - low[j] >= thr:             # short TP
                    pnl = TP_NET; won = 1; exit_i = j; break
            j += 1
        if exit_i < 0:                                # timed out at 60 min
            exit_i = last
            pnl = d * (close[last] - entry) / PIP - sp
        out_pnl[t] = pnl
        out_reg[t] = reg
        out_dur[t] = exit_i - i
        out_won[t] = won
        t += 1
        if exit_i <= i:                               # safety: never stall
            exit_i = i + 1
        i = exit_i                                    # next trade opens where this closed
    return out_pnl[:t], out_reg[:t], out_dur[:t], out_won[:t]


def main():
    print(f"Loading {PATH} ...")
    df = duckdb.sql(
        f"SELECT epoch(timestamp)::BIGINT ts, high, low, close, (ask_c-bid_c) sp "
        f"FROM '{PATH}' WHERE ask_c > bid_c ORDER BY timestamp"
    ).df()
    ts    = df["ts"].to_numpy(np.int64)
    high  = df["high"].to_numpy(np.float64)
    low   = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    sp    = df["sp"].to_numpy(np.float64)
    # trailing mean spread over K bars (causal: includes current bar)
    csum = np.cumsum(sp)
    regime = np.empty_like(sp)
    regime[:K_SPREAD] = csum[:K_SPREAD] / (np.arange(K_SPREAD) + 1)
    regime[K_SPREAD:] = (csum[K_SPREAD:] - csum[:-K_SPREAD]) / K_SPREAD
    print(f"{len(sp):,} bars  {df['ts'].iloc[0]}..{df['ts'].iloc[-1]}")

    pnl, reg, dur, won = simulate(high, low, close, sp, regime, ts, K_SPREAD, SEED)
    print(f"\nTrades: {len(pnl):,}   net pips: {pnl.sum():,.0f}   "
          f"mean/trade: {pnl.mean():.4f}   win%: {100*won.mean():.1f}   "
          f"avg dur: {dur.mean()*5/60:.1f} min")

    # ---- bucket by trailing-spread regime (quantile bands) ----
    qs = np.quantile(reg, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    edges = np.unique(np.round(qs, 4))
    print(f"\nRegime = trailing {K_SPREAD}-bar avg spread (pips). "
          f"TP=+{TP_NET:.0f} net, no SL, {HOLD_BARS*5//60}-min stop.\n")
    hdr = (f"{'spread band (pips)':>20} | {'n':>7} | {'win%':>5} | {'mean':>7} | "
           f"{'med':>6} | {'P5':>7} | {'P1':>7} | {'worst':>7} | "
           f"{'<-5%':>5} | {'<-10%':>6} | {'<-20%':>6}")
    print(hdr); print("-" * len(hdr))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (reg >= a) & (reg < b) if b != edges[-1] else (reg >= a) & (reg <= b)
        if m.sum() == 0:
            continue
        p = pnl[m]
        print(f"{a:8.2f}–{b:<8.2f} | {m.sum():>7,} | {100*won[m].mean():>5.1f} | "
              f"{p.mean():>7.3f} | {np.median(p):>6.2f} | {np.percentile(p,5):>7.2f} | "
              f"{np.percentile(p,1):>7.2f} | {p.min():>7.1f} | "
              f"{100*(p<-5).mean():>5.1f} | {100*(p<-10).mean():>6.1f} | {100*(p<-20).mean():>6.1f}")

    # deep-loser summary: correlation of regime with tail severity
    print("\nOverall loss tail:  P5 = %.2f   P1 = %.2f   worst = %.1f pips"
          % (np.percentile(pnl, 5), np.percentile(pnl, 1), pnl.min()))


if __name__ == "__main__":
    main()
