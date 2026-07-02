#!/usr/bin/env python3
"""
SMA Momentum — MC validation of winning config
===============================================
Config: SMA16, lags (8,10,15), H1+M30, strict 6/6, TP=20p
Runs 2000 sign-shuffles per pair + combined portfolio shuffle.
Pass criteria: mc_p < 0.05 AND IS 3/3 AND OOS p/d > 0.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD",
    "AUD_JPY", "EUR_USD", "EUR_GBP", "AUD_USD",
    "NZD_JPY", "CHF_JPY", "NZD_USD", "CAD_JPY",
]
JPY_PAIRS = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

SMA_N  = 16
LAGS   = (8, 10, 15)
TP     = 20.0
IS_FRAC = 0.70
MC_N    = 2000


def pip_size(pair):
    return 0.01 if pair in JPY_PAIRS else 0.0001


def build_signal(df, sma_n=SMA_N, lags=LAGS):
    moms = []
    for tf in ["1h", "30min"]:
        rs  = df["close"].resample(tf).last().dropna()
        sma = rs.rolling(sma_n, min_periods=sma_n).mean().shift(1)
        sma = sma.reindex(df.index, method="ffill")
        for k in lags:
            moms.append(sma - sma.shift(k))
    all_moms = pd.concat(moms, axis=1)
    n_ind    = len(moms)                      # 6
    score    = (all_moms > 0).sum(axis=1)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score >= n_ind] = np.int8(1)
    sig[score <= 0]     = np.int8(-1)
    return sig


def simulate_tp(df, sig, pip, tp_pips, sp_gate):
    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = (ask - bid) / pip
    s   = sig.values
    n   = len(df)
    pnls = []
    in_trade = False; dir_ = 0; entry_px = 0.0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - entry_px) / pip * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - entry_px) / pip * dir_ - sp[i])
                in_trade = False
        else:
            nd = s[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                entry_px = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_trade = True
    return np.array(pnls, dtype=np.float64)


@njit(parallel=True)
def mc_sign_shuffle(pnls, observed_ppd, oos_days, n_shuffles):
    n = len(pnls)
    beats = 0
    for _ in prange(n_shuffles):
        total = 0.0
        for j in range(n):
            sign = 1.0 if np.random.random() > 0.5 else -1.0
            total += pnls[j] * sign
        if (total / oos_days) >= observed_ppd:
            beats += 1
    return beats / n_shuffles


def run_pair(pair):
    path = DATA / f"{pair}_M5_BA.parquet"
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_size(pair)

    n_is    = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    oos_df  = df.iloc[n_is:]
    oos_days = len(oos_df) / 288

    sig = build_signal(df)

    # IS 3-fold
    fold_size = n_is // 3
    is_ppds = []
    for f in range(3):
        s = f * fold_size
        e = s + fold_size if f < 2 else n_is
        days_f = (e - s) / 288
        t = simulate_tp(df.iloc[s:e], sig.iloc[s:e], pip, TP, sp_gate)
        is_ppds.append(t.sum() / days_f if len(t) else 0.0)
    is_pass = sum(1 for x in is_ppds if x > 0)

    # OOS
    t_oos = simulate_tp(oos_df, sig.iloc[n_is:], pip, TP, sp_gate)
    oos_ppd = t_oos.sum() / oos_days if len(t_oos) > 0 else 0.0
    oos_wr  = float((t_oos > 0).mean() * 100) if len(t_oos) > 0 else 0.0
    tpd     = len(t_oos) / oos_days

    # MC
    mc_p = None
    if len(t_oos) > 0:
        mc_p = mc_sign_shuffle(t_oos, oos_ppd, oos_days, MC_N)

    return dict(
        pair=pair, sp_gate=round(sp_gate, 2),
        is_pass=is_pass,
        is_f1=round(is_ppds[0], 2), is_f2=round(is_ppds[1], 2), is_f3=round(is_ppds[2], 2),
        oos_ppd=round(oos_ppd, 2), oos_nt=len(t_oos),
        oos_wr=round(oos_wr, 1), tpd=round(tpd, 3),
        oos_days=round(oos_days, 0),
        mc_p=round(float(mc_p), 4) if mc_p is not None else None,
        pnls=t_oos,
    )


def main():
    # JIT warm-up
    mc_sign_shuffle(np.array([1.0, -1.0]), 0.0, 1.0, 10)

    print(f"MC Validation — SMA{SMA_N} lags={LAGS} TP={TP}p  ({MC_N} shuffles/pair)")
    print("=" * 80)

    results = []
    all_pnls_list = []
    total_days = 0.0

    for pair in PAIRS:
        print(f"  {pair} …", end=" ", flush=True)
        r = run_pair(pair)
        pnls = r.pop("pnls")

        is_gate = "✅" if r["is_pass"] == 3 else ("🟡" if r["is_pass"] == 2 else "❌")
        mc_gate  = ("✅" if r["mc_p"] is not None and r["mc_p"] < 0.05 else "❌") \
                   if r["mc_p"] is not None else "—"
        print(f"IS {r['is_pass']}/3 {is_gate}  "
              f"OOS {r['oos_ppd']:+.1f}p/d  WR {r['oos_wr']}%  "
              f"n={r['oos_nt']}  mc_p={r['mc_p']:.4f} {mc_gate}")

        results.append(r)
        all_pnls_list.append(pnls)
        total_days = max(total_days, r["oos_days"])

    # Portfolio-level MC: pool all OOS trades, shuffle together
    all_pnls   = np.concatenate(all_pnls_list)
    port_ppd   = sum(r["oos_ppd"] for r in results)
    port_mc_p  = mc_sign_shuffle(all_pnls, port_ppd, total_days, MC_N)
    port_nt    = sum(r["oos_nt"] for r in results)
    port_wr    = float((all_pnls > 0).mean() * 100) if len(all_pnls) > 0 else 0.0

    print()
    print("=" * 80)
    print(f"PORTFOLIO SUMMARY — SMA{SMA_N} lags={LAGS} TP={TP}p")
    print("=" * 80)
    print(f"  Total OOS trades : {port_nt}")
    print(f"  Portfolio p/day  : {port_ppd:+.1f}p")
    print(f"  Portfolio WR     : {port_wr:.1f}%")
    print(f"  Portfolio MC p   : {port_mc_p:.4f}  "
          f"({'SIGNIFICANT ✅' if port_mc_p < 0.05 else 'NOT SIGNIFICANT ❌'})")
    print()

    # Gate summary
    n_is3  = sum(1 for r in results if r["is_pass"] == 3)
    n_mc05 = sum(1 for r in results if r["mc_p"] is not None and r["mc_p"] < 0.05)
    n_both = sum(1 for r in results
                 if r["is_pass"] == 3 and r["mc_p"] is not None and r["mc_p"] < 0.05)

    print(f"  IS 3/3           : {n_is3}/{len(PAIRS)} pairs")
    print(f"  MC p<0.05        : {n_mc05}/{len(PAIRS)} pairs")
    print(f"  IS3 AND MC<0.05  : {n_both}/{len(PAIRS)} pairs  ← deploy candidates")
    print()

    # Deploy candidates
    deploy = [r for r in results
              if r["is_pass"] == 3 and r["mc_p"] is not None and r["mc_p"] < 0.05]
    print(f"DEPLOY CANDIDATES ({len(deploy)} pairs):")
    print(f"  {'Pair':>10} {'IS':>5} {'p/d':>8} {'WR':>6} {'t/d':>6} "
          f"{'mc_p':>8} {'sp_gate':>8}")
    print("  " + "-" * 60)
    for r in sorted(deploy, key=lambda x: -x["oos_ppd"]):
        print(f"  {r['pair']:>10} {r['is_pass']}/3  {r['oos_ppd']:>+7.1f}  "
              f"{r['oos_wr']:>5.1f}%  {r['tpd']:>5.3f}  "
              f"{r['mc_p']:>8.4f}  {r['sp_gate']:>6.2f}p")

    # Save
    df_out = pd.DataFrame(results)
    df_out.to_csv(RESULTS / "sma_mc_validation.csv", index=False)
    print(f"\nSaved → {RESULTS / 'sma_mc_validation.csv'}")


if __name__ == "__main__":
    main()
