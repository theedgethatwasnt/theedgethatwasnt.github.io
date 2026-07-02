#!/usr/bin/env python3
"""
SMA22 Dual-TF Momentum — 12-pair sweep + MC validation
=======================================================
Signal: SMA(22) on H1 + M30, momentum at lags 1/5/10.
Entry: LONG when N of 6 momentum values > 0; SHORT when N < 0.
Exit: TP=20p (best from single-pair run).
Sweep: all 12 pairs × votes_needed in {6, 5, 4, 3}.
MC: 2000 sign-shuffles on GBP_JPY 6/6 winner.
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

SMA_N   = 22
LAGS    = [1, 5, 10]       # momentum lags (in TF bars)
TP_PIPS = 20.0             # fixed take-profit
MC_N    = 2000             # Monte Carlo shuffles
IS_FRAC = 0.70


# ── feature computation ───────────────────────────────────────────────────────

def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


def compute_tf_mom(m5_close: pd.Series, tf: str) -> pd.DataFrame:
    """SMA(22) momentum at lags 1/5/10 on resampled TF, aligned back to M5."""
    rs  = m5_close.resample(tf).last().dropna()
    sma = rs.rolling(SMA_N, min_periods=SMA_N).mean()
    moms = pd.DataFrame(index=rs.index)
    for k in LAGS:
        moms[f"mom{k}"] = sma - sma.shift(k)
    moms = moms.shift(1)                              # causality: use completed bar
    return moms.reindex(m5_close.index, method="ffill")


def build_vote_signal(df: pd.DataFrame, votes_needed: int) -> pd.Series:
    """
    Score = number of the 6 momentum values that are positive.
    LONG  signal when score >= votes_needed.
    SHORT signal when (6 - score) >= votes_needed  (i.e. score <= 6 - votes_needed).
    """
    h1  = compute_tf_mom(df["close"], "1h")
    m30 = compute_tf_mom(df["close"], "30min")
    all_moms = pd.concat([h1, m30], axis=1)           # 6 columns
    score = (all_moms > 0).sum(axis=1)                # 0..6 positives

    sig = pd.Series(0, index=df.index)
    sig[score >= votes_needed]        =  1
    sig[score <= (6 - votes_needed)]  = -1
    return sig


# ── trade simulation (pure Python — fast enough for single sim pass) ──────────

def simulate_tp(df: pd.DataFrame, sig: pd.Series,
                pip: float, tp_pips: float, sp_gate: float) -> np.ndarray:
    """Returns array of per-trade pnl values."""
    bid  = df["bid_c"].values.astype(np.float64)
    ask  = df["ask_c"].values.astype(np.float64)
    mid  = df["close"].values.astype(np.float64)
    sp   = (ask - bid) / pip
    s    = sig.values
    n    = len(df)

    pnls     = []
    in_trade = False
    dir_     = 0
    entry_px = 0.0

    for i in range(1, n):
        if in_trade:
            pnl_now = (mid[i] - entry_px) / pip * dir_
            if pnl_now >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - entry_px) / pip * dir_ - sp[i])
                in_trade = False
        else:
            nd = s[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                entry_px = ask[i] if nd == 1 else bid[i]
                dir_     = nd
                in_trade = True

    return np.array(pnls, dtype=np.float64)


# ── Monte Carlo sign-shuffle ──────────────────────────────────────────────────

@njit(parallel=True)
def mc_sign_shuffle(pnls: np.ndarray, observed_ppd: float,
                    oos_days: float, n_shuffles: int) -> float:
    """
    Randomly flip each trade's P&L sign. Count fraction of shuffles
    where shuffled p/d >= observed p/d. Lower = more significant.
    """
    n = len(pnls)
    beats = 0
    for _ in prange(n_shuffles):
        total = 0.0
        for j in range(n):
            sign = 1.0 if (np.random.random() > 0.5) else -1.0
            total += pnls[j] * sign
        if (total / oos_days) >= observed_ppd:
            beats += 1
    return beats / n_shuffles


# ── walk-forward validation ───────────────────────────────────────────────────

def wf_validate(df, sig, pip, sp_gate, run_mc=False):
    n      = len(df)
    is_end = int(n * IS_FRAC)

    fold_size = is_end // 3
    is_pds = []
    for f in range(3):
        s = f * fold_size
        e = s + fold_size if f < 2 else is_end
        days = (e - s) / 288
        t = simulate_tp(df.iloc[s:e], sig.iloc[s:e], pip, TP_PIPS,
                        sp_gate)
        is_pds.append(t.sum() / days if len(t) else 0.0)

    oos_df  = df.iloc[is_end:]
    oos_sig = sig.iloc[is_end:]
    oos_days = len(oos_df) / 288
    t_oos = simulate_tp(oos_df, oos_sig, pip, TP_PIPS, sp_gate)

    oos_pd = t_oos.sum() / oos_days if len(t_oos) else 0.0
    oos_wr = float((t_oos > 0).mean() * 100) if len(t_oos) else 0.0
    is_pass = sum(1 for x in is_pds if x > 0)

    mc_p = None
    if run_mc and len(t_oos) > 0:
        mc_p = mc_sign_shuffle(t_oos, oos_pd, oos_days, MC_N)

    return {
        "is_pass":  is_pass,
        "is_f1":    round(is_pds[0], 2),
        "is_f2":    round(is_pds[1], 2),
        "is_f3":    round(is_pds[2], 2),
        "oos_pd":   round(oos_pd, 2),
        "oos_nt":   len(t_oos),
        "oos_wr":   round(oos_wr, 1),
        "tpd":      round(len(t_oos) / oos_days, 2),
        "oos_days": round(oos_days, 0),
        "mc_p":     round(mc_p, 4) if mc_p is not None else None,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rows = []
    mc_result = None

    for pair in PAIRS:
        path = DATA / f"{pair}_M5_BA.parquet"
        if not path.exists():
            print(f"  {pair}: no data, skipping")
            continue

        df = pd.read_parquet(path)
        df = df.set_index("timestamp").sort_index()
        df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})

        pip    = pip_size(pair)
        n_is   = int(len(df) * IS_FRAC)
        sp_ser = (df["ask_c"] - df["bid_c"]) / pip
        sp_gate = float(np.percentile(sp_ser.iloc[:n_is].dropna(), 90))

        print(f"\n{pair}  bars={len(df):,}  sp_gate={sp_gate:.2f}p")

        for votes in [6, 5, 4, 3]:
            sig = build_vote_signal(df, votes)
            n_long  = (sig ==  1).sum()
            n_short = (sig == -1).sum()

            run_mc = (pair == "GBP_JPY" and votes == 6)
            res = wf_validate(df, sig, pip, sp_gate, run_mc=run_mc)

            if run_mc:
                mc_result = res

            flag = "✅" if res["is_pass"] == 3 else ("🟡" if res["is_pass"] == 2 else "❌")
            mc_str = f"  mc_p={res['mc_p']:.4f}" if res["mc_p"] is not None else ""
            print(f"  votes={votes}/6  IS {res['is_pass']}/3 {flag}  "
                  f"OOS {res['oos_pd']:+.1f}p/d  WR {res['oos_wr']}%  "
                  f"n={res['oos_nt']}  tpd={res['tpd']:.2f}{mc_str}")

            rows.append({
                "pair": pair, "votes": votes,
                "sp_gate": sp_gate,
                "signal_pct": round((n_long + n_short) / len(df) * 100, 1),
                **res,
            })

    out = pd.DataFrame(rows)
    csv = RESULTS / "sma22_sweep_12pairs.csv"
    out.to_csv(csv, index=False)

    print("\n" + "=" * 80)
    print(f"SUMMARY — TP={TP_PIPS}p | SMA={SMA_N} | Lags={LAGS}")
    print("=" * 80)

    # Show 3/3 IS passes only, sorted by tpd then oos_pd
    winners = out[out["is_pass"] == 3].copy()
    print(f"\n✅ IS 3/3 passes: {len(winners)} / {len(out)} configs\n")
    cols = ["pair","votes","oos_pd","oos_wr","oos_nt","tpd","is_f1","is_f2","is_f3","mc_p"]
    print(winners[cols].sort_values(["tpd","oos_pd"], ascending=False).to_string(index=False))

    if mc_result:
        print(f"\n{'='*60}")
        print(f"MC VALIDATION — GBP_JPY 6/6 TP=20p ({MC_N} sign-shuffles)")
        print(f"  OOS p/d observed : {mc_result['oos_pd']:+.2f}")
        print(f"  MC p-value       : {mc_result['mc_p']:.4f}  "
              f"({'SIGNIFICANT ✅' if mc_result['mc_p'] < 0.05 else 'not significant ❌'})")
        print(f"  Trades (OOS)     : {mc_result['oos_nt']}")
        print(f"  Win rate         : {mc_result['oos_wr']}%")

    print(f"\nResults → {csv}")


if __name__ == "__main__":
    main()
