#!/usr/bin/env python3
"""
Oracle optimal-trade decomposition on raw S5 price (perfect-foresight CEILING).

Q (user): if we could enter anywhere and exit anywhere, which non-overlapping
segments give the highest reward — and what are their traits (length, amplitude,
efficiency)? Caveats: (1) no fragmenting a good run into sub-trades ("don't go
left") — handled by charging spread per trade so fragmentation is never optimal;
(2) one action per bar, NO same-bar enter+exit → trades span >=2 bars.

Method: max-total-net-pips DP with per-trade spread cost, LONG and SHORT, one
position at a time (LeetCode-714 state machine extended to shorts). O(n), numba.
Fill at bar-close mid; spread paid up front at entry (SOP R3). Backward pass
reconstructs the optimal trade set.

NOT tradeable (uses future) — it's the opportunity ceiling + trait distribution +
labels for what a causal entry/exit should aim to catch.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
PAIR = sys.argv[1] if len(sys.argv) > 1 else "GBP_JPY"
PIP = 0.01 if "JPY" in PAIR else 0.0001
NEG = -1e18


@njit(cache=True)
def oracle_dp(mid, sp_px):
    """Forward DP. Returns F,L,S value arrays + entry-bar trackers le,se."""
    n = len(mid)
    F = np.empty(n); L = np.empty(n); S = np.empty(n)
    le = np.full(n, -1, np.int64); se = np.full(n, -1, np.int64)
    F[0] = 0.0
    L[0] = -mid[0] - sp_px[0]; le[0] = 0          # enter long at bar 0
    S[0] = mid[0] - sp_px[0];  se[0] = 0          # enter short at bar 0
    for i in range(1, n):
        # FLAT: stay, or close a long (sell +mid), or close a short (buy -mid)
        f_stay = F[i-1]; f_cl = L[i-1] + mid[i]; f_cs = S[i-1] - mid[i]
        F[i] = f_stay
        if f_cl > F[i]: F[i] = f_cl
        if f_cs > F[i]: F[i] = f_cs
        # LONG: keep, or enter from flat (buy: -mid-spread)
        l_keep = L[i-1]; l_new = F[i-1] - mid[i] - sp_px[i]
        if l_keep >= l_new:
            L[i] = l_keep; le[i] = le[i-1]
        else:
            L[i] = l_new; le[i] = i
        # SHORT: keep, or enter from flat (sell: +mid-spread)
        s_keep = S[i-1]; s_new = F[i-1] + mid[i] - sp_px[i]
        if s_keep >= s_new:
            S[i] = s_keep; se[i] = se[i-1]
        else:
            S[i] = s_new; se[i] = i
    return F, L, S, le, se


def reconstruct(mid, sp_px, F, L, S, le, se):
    """Backward walk from FLAT at n-1 → list of (entry_i, exit_j, dir)."""
    n = len(mid); trades = []
    state = 0  # 0 flat, 1 long, 2 short
    i = n - 1
    cur_exit = -1
    while i >= 0:
        if state == 0:
            # how was F[i] achieved?
            if i == 0:
                break
            f_cl = L[i-1] + mid[i]; f_cs = S[i-1] - mid[i]
            if abs(F[i] - (L[i-1] + mid[i])) < 1e-9 and f_cl >= F[i-1] - 1e-12 and f_cl >= f_cs:
                state = 1; cur_exit = i; i -= 1            # we closed a long at i
            elif abs(F[i] - (S[i-1] - mid[i])) < 1e-9 and f_cs >= F[i-1] - 1e-12:
                state = 2; cur_exit = i; i -= 1            # we closed a short at i
            else:
                i -= 1                                      # stayed flat
        elif state == 1:
            # long held at i; entered at le[i]; was it just entered here?
            l_new = F[i-1] - mid[i] - sp_px[i] if i > 0 else -1e18
            if i == le[i] or i == 0:
                trades.append((i, cur_exit, 1)); state = 0  # entry bar
                i -= 1
            else:
                i -= 1
        else:  # state == 2 short
            if i == se[i] or i == 0:
                trades.append((i, cur_exit, -1)); state = 0
                i -= 1
            else:
                i -= 1
    trades.reverse()
    return trades


def main():
    f = PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet"
    df = pd.read_parquet(f, columns=["timestamp", "close", "bid_c", "ask_c"]).sort_values("timestamp")
    mid = df["close"].values.astype(np.float64)
    sp_px = (df["ask_c"].values - df["bid_c"].values).astype(np.float64)
    n = len(mid); days = n / 17280
    print(f"{PAIR} S5: {n:,} bars, {days:.0f} trading days, median spread {np.median(sp_px)/PIP:.2f}p")

    F, L, S, le, se = oracle_dp(mid, sp_px)
    print(f"ORACLE ceiling (max total net pips, all non-overlapping trades): {F[-1]/PIP:,.0f}p "
          f"({F[-1]/PIP/days:,.0f} p/day)")
    trades = reconstruct(mid, sp_px, F, L, S, le, se)
    t = np.array(trades, dtype=np.int64)
    ei, xj, d = t[:, 0], t[:, 1], t[:, 2]
    net = (d * (mid[xj] - mid[ei]) / PIP) - (sp_px[ei] / PIP)
    hold = xj - ei
    # efficiency + underwater for each optimal leg
    eff = np.empty(len(t)); cum_dd = np.empty(len(t))
    for k in range(len(t)):
        seg = mid[ei[k]:xj[k]+1]
        path = np.abs(np.diff(seg)).sum()
        eff[k] = abs(seg[-1]-seg[0]) / path if path > 0 else 0.0
        upnl = d[k]*(seg - seg[0])/PIP
        cum_dd[k] = np.clip(-upnl, 0, None).sum()       # underwater pip-bars
    amddp5 = net - 0.05*cum_dd
    inmkt = hold.sum() / n * 100

    print(f"\n=== OPTIMAL TRADE SET — traits ===")
    print(f"  trades: {len(t):,}  ({len(t)/days:.1f}/day)   long {np.mean(d>0)*100:.0f}% / short {np.mean(d<0)*100:.0f}%")
    print(f"  time in market: {inmkt:.0f}% of bars   (flat {100-inmkt:.0f}%)")
    print(f"  hold bars:   median {np.median(hold):.0f}  p90 {np.percentile(hold,90):.0f}  "
          f"max {hold.max():.0f}   (×5s = median {np.median(hold)*5:.0f}s / p90 {np.percentile(hold,90)*5/60:.1f}min)")
    print(f"  net pips/trade: median {np.median(net):.1f}  p90 {np.percentile(net,90):.1f}  "
          f"max {net.max():.1f}  mean {net.mean():.2f}")
    print(f"  efficiency |net|/path: median {np.median(eff):.2f}  (1.0=perfectly clean leg)")
    print(f"  amddp5/trade: median {np.median(amddp5):.1f}  mean {amddp5.mean():.2f}  "
          f"(≈net ⇒ optimal legs are clean)")
    # reward concentration — do a few big legs dominate?
    order = np.argsort(-net)
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = int(len(net)*frac)
        print(f"  top {frac*100:.0f}% of trades hold {net[order[:k]].sum()/net.sum()*100:.0f}% of total reward")
    print(f"\n  10 single highest-reward segments (the 'best traits'):")
    print(f"  {'rank':>4}{'dir':>4}{'hold_bars':>10}{'hold_min':>9}{'net_p':>8}{'eff':>6}")
    for r, k in enumerate(order[:10], 1):
        print(f"  {r:>4}{'L' if d[k]>0 else 'S':>4}{hold[k]:>10}{hold[k]*5/60:>9.1f}{net[k]:>8.1f}{eff[k]:>6.2f}")
    # save labels (entry/exit bars + traits) for downstream supervised use
    out = pd.DataFrame({"entry_idx": ei, "exit_idx": xj, "dir": d, "net_pips": net,
                        "hold_bars": hold, "eff": eff, "amddp5": amddp5})
    outdir = Path(__file__).parent; outdir.mkdir(exist_ok=True)
    out.to_parquet(outdir / f"{PAIR}_oracle_trades.parquet")
    print(f"\n  labels saved → {PAIR}_oracle_trades.parquet ({len(out):,} optimal trades)")


if __name__ == "__main__":
    main()
