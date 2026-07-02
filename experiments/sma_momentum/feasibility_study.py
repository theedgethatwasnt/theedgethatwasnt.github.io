#!/usr/bin/env python3
"""
SMA22 Momentum — Feasibility ceiling study
==========================================
For each pair, sweep TP from 5p to 100p (strict 6/6 entry).
Report portfolio-level trades/day and p/d at each TP level.
Goal: find the natural frontier of what this signal can deliver.
"""

import numpy as np
import pandas as pd
from pathlib import Path
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

SMA_N  = 22
LAGS   = [1, 5, 10]
TP_LEVELS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
IS_FRAC = 0.70
VOTES   = 6   # strict — best for most pairs


def pip_size(pair):
    return 0.01 if pair in JPY_PAIRS else 0.0001


def compute_tf_mom(m5_close, tf):
    rs  = m5_close.resample(tf).last().dropna()
    sma = rs.rolling(SMA_N, min_periods=SMA_N).mean()
    moms = pd.DataFrame(index=rs.index)
    for k in LAGS:
        moms[f"mom{k}"] = sma - sma.shift(k)
    moms = moms.shift(1)
    return moms.reindex(m5_close.index, method="ffill")


def build_signal(df):
    h1  = compute_tf_mom(df["close"], "1h")
    m30 = compute_tf_mom(df["close"], "30min")
    all_moms = pd.concat([h1, m30], axis=1)
    score = (all_moms > 0).sum(axis=1)
    sig = pd.Series(0, index=df.index)
    sig[score >= VOTES]       =  1
    sig[score <= 6 - VOTES]   = -1
    return sig


def simulate_tp(df, sig, pip, tp_pips, sp_gate):
    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = (ask - bid) / pip
    s   = sig.values
    n   = len(df)

    pnls = []
    in_trade = False
    dir_ = 0
    entry_px = 0.0

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
                dir_ = nd
                in_trade = True

    return np.array(pnls)


def run_pair(pair):
    path = DATA / f"{pair}_M5_BA.parquet"
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_size(pair)

    n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    oos_df  = df.iloc[n_is:]
    oos_days = len(oos_df) / 288

    sig     = build_signal(df)
    oos_sig = sig.iloc[n_is:]

    # IS 3-fold check at TP=20p (representative)
    is_df  = df.iloc[:n_is]
    is_sig = sig.iloc[:n_is]
    fold_size = n_is // 3
    is_pass = 0
    for f in range(3):
        s = f * fold_size
        e = s + fold_size if f < 2 else n_is
        days_f = (e - s) / 288
        t = simulate_tp(is_df.iloc[s:e], is_sig.iloc[s:e], pip, 20.0, sp_gate)
        if len(t) > 0 and t.sum() / days_f > 0:
            is_pass += 1

    rows = []
    for tp in TP_LEVELS:
        t = simulate_tp(oos_df, oos_sig, pip, tp, sp_gate)
        nt = len(t)
        ppd  = t.sum() / oos_days if nt > 0 else 0.0
        tpd  = nt / oos_days
        wr   = float((t > 0).mean() * 100) if nt > 0 else 0.0
        rows.append(dict(pair=pair, tp=tp, is_pass=is_pass,
                         ppd=round(ppd,2), tpd=round(tpd,3),
                         nt=nt, wr=round(wr,1), oos_days=round(oos_days,0)))
    return rows


def main():
    all_rows = []
    for pair in PAIRS:
        print(f"  {pair} …", end=" ", flush=True)
        rows = run_pair(pair)
        all_rows.extend(rows)
        # quick summary at TP=20
        r20 = next(r for r in rows if r["tp"] == 20)
        gate = "✅" if r20["is_pass"] == 3 else ("🟡" if r20["is_pass"] == 2 else "❌")
        print(f"IS {r20['is_pass']}/3 {gate}  "
              f"TP20→ {r20['ppd']:+.1f}p/d  {r20['tpd']:.2f}t/d")

    df_all = pd.DataFrame(all_rows)

    # Portfolio view: sum across ALL 12 pairs (use each at its best—for now sum all)
    print("\n" + "=" * 72)
    print(f"PORTFOLIO FRONTIER — all 12 pairs, strict 6/6, SMA{SMA_N}")
    print("=" * 72)
    print(f"{'TP':>6} {'t/day':>7} {'p/day':>8} {'avg p/trade':>12} "
          f"{'pairs>0':>8} {'IS3/3 pairs':>11}")
    print("-" * 72)

    frontier_rows = []
    for tp in TP_LEVELS:
        sub = df_all[df_all["tp"] == tp]
        total_tpd  = sub["tpd"].sum()
        total_ppd  = sub["ppd"].sum()
        n_pos      = (sub["ppd"] > 0).sum()
        n_is3      = (sub["is_pass"] == 3).sum()
        avg_p_trade = total_ppd / total_tpd if total_tpd > 0 else 0

        print(f"{tp:>5}p  {total_tpd:>6.2f}  {total_ppd:>+7.1f}  "
              f"{avg_p_trade:>11.2f}  {n_pos:>7}/{len(sub)}  {n_is3:>9}/{len(sub)}")
        frontier_rows.append(dict(tp=tp, tpd=round(total_tpd,2),
                                   ppd=round(total_ppd,2),
                                   avg_p_trade=round(avg_p_trade,2),
                                   n_positive=n_pos, n_is3=n_is3))

    print()

    # Best IS-3/3 pairs only
    is3_pairs = df_all[df_all["is_pass"] == 3]["pair"].unique()
    print(f"IS 3/3 pairs only ({len(is3_pairs)}): {', '.join(sorted(is3_pairs))}")
    print()
    print(f"{'TP':>6} {'t/day':>7} {'p/day':>8} {'avg p/trade':>12}")
    print("-" * 40)
    for tp in TP_LEVELS:
        sub = df_all[(df_all["tp"] == tp) & (df_all["pair"].isin(is3_pairs))]
        total_tpd = sub["tpd"].sum()
        total_ppd = sub["ppd"].sum()
        avg_p = total_ppd / total_tpd if total_tpd > 0 else 0
        print(f"{tp:>5}p  {total_tpd:>6.2f}  {total_ppd:>+7.1f}  {avg_p:>11.2f}")

    # Per-pair detail at key TP levels
    print("\n\nPER-PAIR DETAIL at TP = 10 / 20 / 50p")
    print("-" * 72)
    print(f"{'Pair':>10} {'IS':>5}  "
          f"{'TP10 tpd':>9} {'ppd':>7}  "
          f"{'TP20 tpd':>9} {'ppd':>7}  "
          f"{'TP50 tpd':>9} {'ppd':>7}")
    print("-" * 72)
    for pair in PAIRS:
        sub = df_all[df_all["pair"] == pair]
        ip  = sub.iloc[0]["is_pass"]
        r10 = sub[sub["tp"]==10].iloc[0]
        r20 = sub[sub["tp"]==20].iloc[0]
        r50 = sub[sub["tp"]==50].iloc[0]
        g   = "✅" if ip==3 else ("🟡" if ip==2 else "❌")
        print(f"{pair:>10} {g}{ip}/3  "
              f"{r10['tpd']:>8.2f} {r10['ppd']:>+7.1f}  "
              f"{r20['tpd']:>8.2f} {r20['ppd']:>+7.1f}  "
              f"{r50['tpd']:>8.2f} {r50['ppd']:>+7.1f}")

    csv = RESULTS / "sma22_feasibility.csv"
    df_all.to_csv(csv, index=False)
    pd.DataFrame(frontier_rows).to_csv(
        RESULTS / "sma22_frontier.csv", index=False)
    print(f"\nSaved → {csv}")


if __name__ == "__main__":
    main()
