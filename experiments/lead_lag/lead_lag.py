#!/usr/bin/env python3
"""
Majors -> minors (crosses) LEAD-LAG test, triangular form. S5, ~14mo common window.

A cross is the PRODUCT of two USD majors (identity, not a discovered correlation):
  r_EURJPY ≈ r_EURUSD + r_USDJPY,   r_EURGBP ≈ r_EURUSD - r_GBPUSD, ...
The only exploitable thing is LAG: does the cross trail its majors, leaving a window to
trade the catch-up? We measure, per triplet:
  (a) contemporaneous R²  — sanity, the identity should hold ~1 at 5s.
  (b) MAJORS-LEAD IC      — corr(implied major return at t-1, cross return at t).
  (c) CROSS-LEADS IC      — corr(cross return at t-1, implied major return at t).
       Asymmetry between (b) and (c) answers "is it one-way?"
  (d) net-of-spread expectancy of the best majors-lead signal on the cross.

Causal: predict r_cross(t) from info dated <= t-1 only. Mid returns; cross spread
deducted explicitly (R3). IS/OOS split.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/path/to/projects/fx-core")
S5 = PROJECT / "data" / "s5_ohlc"
WIN_START = "2025-03-05"
WIN_END   = "2026-05-12"
IS_FRAC = 0.6

# cross : [(leg, sign), (leg, sign)]  in log-return space ; plus cross spread (pips), pip
TRIPLETS = {
    "EUR_JPY": ([("EUR_USD", +1), ("USD_JPY", +1)], 2.5, 0.01),
    "GBP_JPY": ([("GBP_USD", +1), ("USD_JPY", +1)], 4.0, 0.01),
    "AUD_JPY": ([("AUD_USD", +1), ("USD_JPY", +1)], 2.3, 0.01),
    "NZD_JPY": ([("NZD_USD", +1), ("USD_JPY", +1)], 3.0, 0.01),
    "EUR_GBP": ([("EUR_USD", +1), ("GBP_USD", -1)], 2.0, 0.0001),
}
ALL = sorted({p for v in TRIPLETS.values() for p, _ in v[0]} | set(TRIPLETS))


def load_grid():
    """Aligned 5s mid panel across all needed pairs (inner join on 5s bucket)."""
    con = duckdb.connect()
    dfs = {}
    for p in ALL:
        f = S5 / f"{p}_S5_BA.parquet"
        q = f"""SELECT CAST(epoch(timestamp)/5 AS BIGINT) AS b,
                       arg_max(close, timestamp) AS mid
                FROM '{f}'
                WHERE timestamp >= TIMESTAMP '{WIN_START}' AND timestamp < TIMESTAMP '{WIN_END}'
                GROUP BY b"""
        d = con.execute(q).df().set_index("b")
        d.columns = [p]
        dfs[p] = d
        print(f"  loaded {p}: {len(d):,} 5s buckets")
    con.close()
    panel = pd.concat(dfs.values(), axis=1, join="inner").sort_index()
    print(f"  aligned panel: {panel.shape[0]:,} common 5s buckets")
    return panel


def ic(a, b):
    if len(a) < 100:
        return np.nan, np.nan
    r = np.corrcoef(a, b)[0, 1]
    t = r * np.sqrt(len(a) - 2) / np.sqrt(max(1 - r * r, 1e-12))
    return r, t


def main():
    panel = load_grid()
    lr = np.log(panel.values[1:] / panel.values[:-1]).astype(np.float64)
    cols = {p: i for i, p in enumerate(panel.columns)}
    T = lr.shape[0]
    is_end = int(T * IS_FRAC)
    print(f"\nReturns T={T:,}  IS<{is_end:,}  OOS>= {is_end:,}\n")

    print(f"{'cross':<9}{'contemp R2':>11}{'majLEAD IC(t)':>15}{'crossLEAD IC(t)':>17}{'asym':>8}")
    results = {}
    for cross, (legs, sp_pips, pip) in TRIPLETS.items():
        rc = lr[:, cols[cross]]
        # implied major return = signed sum of leg returns
        imp = np.zeros(T)
        for leg, sgn in legs:
            imp += sgn * lr[:, cols[leg]]
        # (a) contemporaneous identity check
        b_co = np.polyfit(imp, rc, 1)
        r2 = np.corrcoef(imp, rc)[0, 1] ** 2
        # residual after contemporaneous fit (what the identity does NOT explain at t)
        resid = rc - (b_co[0] * imp + b_co[1])
        # (b) majors lead: does implied(t-1) predict the cross residual(t)? (cross catching up)
        ml_r, ml_t = ic(imp[:-1], resid[1:])
        # (c) cross leads majors: does cross resid(t-1) predict implied(t)?
        cl_r, cl_t = ic(resid[:-1], imp[1:])
        asym = abs(ml_r) - abs(cl_r)
        print(f"{cross:<9}{r2:>11.4f}{ml_r:>+11.4f}({ml_t:>+4.0f}){cl_r:>+13.4f}({cl_t:>+4.0f}){asym:>+8.4f}")
        results[cross] = dict(rc=rc, imp=imp, resid=resid, sp_pips=sp_pips, pip=pip,
                              ml_r=ml_r, is_end=is_end)

    # (d) exploitation: trade the catch-up. Signal at t-1 = sign of implied move over last
    # k bars; bet cross moves that way next bar. Net of cross spread (round trip). IS/OOS.
    print(f"\n=== majors-lead catch-up trade, net of cross spread (round-trip), IS/OOS ===")
    print(f"{'cross':<9}{'k':>3}{'IS_exp_p':>10}{'OOS_exp_p':>11}{'IS_WR%':>8}{'OOS_WR%':>9}{'n/day':>8}")
    for cross, R in results.items():
        rc, imp, sp_pips, pip = R["rc"], R["imp"], R["sp_pips"], R["pip"]
        is_end = R["is_end"]
        # cross next-bar return in pips (mid)
        for k in (1, 3, 6, 12):
            # cumulative implied move over last k bars, ending at t-1 (causal)
            sig = pd.Series(imp).rolling(k).sum().shift(1).values  # known at t
            d = np.sign(sig)
            # pnl pips on the cross next bar in the signalled direction, minus spread
            # cross return at t -> pips: rc[t] is log-return; pip move ≈ rc * price / pip.
            # use price level for conversion
            px = panel[cross].values[1:]  # aligned to lr length T (lr[i] uses panel[i+1]/panel[i]); approx level
            pnl = d * rc * px / pip - sp_pips  # only when we trade (d!=0)
            m = ~np.isnan(d) & (d != 0)
            ism = m.copy(); ism[is_end:] = False
            oosm = m.copy(); oosm[:is_end] = False
            days = len(rc) * 5 / 86400 * (5/7)  # rough trading-day count
            if ism.sum() < 100 or oosm.sum() < 100:
                continue
            print(f"{cross:<9}{k:>3}{pnl[ism].mean():>+10.3f}{pnl[oosm].mean():>+11.3f}"
                  f"{(pnl[ism]>0).mean()*100:>7.1f}%{(pnl[oosm]>0).mean()*100:>8.1f}%{m.sum()/max(days,1):>8.0f}")
    print("\n  exp_p > 0 in BOTH IS and OOS net of spread => exploitable catch-up lead.")
    print("  (At 5s resolution the HFT lead-lag is mostly within-bar; expect contemp R2~1, leads~0.)")


if __name__ == "__main__":
    main()
