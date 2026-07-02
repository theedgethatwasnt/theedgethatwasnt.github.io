#!/usr/bin/env python3
"""
Weekend overreaction & reversal in spot FX (Dao, McGroarty & Urquhart 2016, JMFM).
Hypothesis: after a large Friday-close -> weekend-open gap, the pair REVERSES over the
following days. Fade the gap at the post-weekend open; hold H days; exit at mid.

Causal & cost-honest (SOP): signal uses only the prior Friday close and the post-weekend
open (both known at entry). Mid OHLC for P&L, full spread deducted once (round-trip proxy)
from the post-weekend-open bar. IS/OOS 70/30 by time; thresholds frozen on IS. Multi-pair
(7 JPY crosses) — never single-pair. One trade per weekend per pair.

usage: backtest_weekend_reversal.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
PAIRS = ["AUD_JPY", "CAD_JPY", "CHF_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "USD_JPY"]
PIP = 0.01
HOLDS = [1, 2, 3, 5]                 # calendar days held
SIG_THR = [0.0, 0.5, 1.0, 1.5]       # gap threshold in units of trailing gap-sigma (per pair)
ROLL = 26                            # weekends (~6mo) for the trailing gap sigma


def weekend_trades(pair):
    df = pd.read_parquet(PROJECT / "data" / "m5_ba" / f"{pair}_M5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mid_o = df["open"].values; mid_c = df["close"].values
    sp = (df["ask_c"].values - df["bid_c"].values) / PIP
    dt_h = ts.diff().dt.total_seconds().values / 3600.0
    # weekend = a time gap > 12h between consecutive M5 bars; entry bar = first after gap
    post = np.where(dt_h > 12.0)[0]                 # indices of first post-weekend bars
    rows = []
    for i in post:
        if i < 1:
            continue
        close_fri = mid_c[i - 1]                    # last bar before the gap
        open_wkd = mid_o[i]                         # post-weekend open
        gap = (open_wkd - close_fri) / PIP
        entry_t = ts.iloc[i]
        entry_mid = mid_c[i]                        # tradeable: first post-weekend bar close
        entry_sp = sp[i]
        rows.append((entry_t, gap, entry_mid, entry_sp, i))
    W = pd.DataFrame(rows, columns=["t", "gap", "entry_mid", "spread", "idx"])
    # exits: nearest bar >= entry_t + H days
    tvals = ts.values
    for H in HOLDS:
        ex = []
        for _, r in W.iterrows():
            target = r["t"] + pd.Timedelta(days=H)
            j = np.searchsorted(tvals, np.datetime64(target))
            ex.append(mid_c[min(j, len(mid_c) - 1)])
        W[f"exit{H}"] = ex
    # trailing gap sigma (causal: shift so it uses only past weekends)
    W["gsig"] = W["gap"].abs().rolling(ROLL, min_periods=8).std().shift(1)
    return W.dropna(subset=["gsig"]).reset_index(drop=True)


def main():
    allW = {p: weekend_trades(p) for p in PAIRS}
    n_each = {p: len(w) for p, w in allW.items()}
    print(f"Weekend-reversal (Dao 2016) on {len(PAIRS)} JPY crosses. weekends/pair: "
          + ", ".join(f"{p.split('_')[0]}={n}" for p, n in n_each.items()))
    print("Fade the gap at post-weekend open; full spread deducted once. IS/OOS 70/30.\n")

    print(f"  {'thr(sig)':>9}{'hold':>6}{'pairs+':>8}{'IS p/t':>9}{'OOS p/t':>9}{'OOS WR':>8}{'OOS n':>7}  verdict")
    best = None
    for thr in SIG_THR:
        for H in HOLDS:
            is_pt_all, oos_pt_all, oos_wr_all, oos_n_all = [], [], [], []
            pairs_oos_pos = 0
            for p in PAIRS:
                W = allW[p]
                take = W["gap"].abs() >= thr * W["gsig"]
                w = W[take]
                if len(w) < 20:
                    continue
                cut = int(len(w) * 0.70)
                d = -np.sign(w["gap"].values)                       # fade the gap
                net = d * (w[f"exit{H}"].values - w["entry_mid"].values) / PIP - w["spread"].values
                is_net, oos_net = net[:cut], net[cut:]
                if len(oos_net) < 5:
                    continue
                is_pt_all.append(is_net.mean()); oos_pt_all.append(oos_net.mean())
                oos_wr_all.append((oos_net > 0).mean() * 100); oos_n_all.append(len(oos_net))
                if oos_net.mean() > 0:
                    pairs_oos_pos += 1
            if not oos_pt_all:
                continue
            is_pt = np.mean(is_pt_all); oos_pt = np.mean(oos_pt_all)
            oos_wr = np.mean(oos_wr_all); oos_n = int(np.sum(oos_n_all))
            v = "OOS+ multi" if (oos_pt > 0 and pairs_oos_pos >= 4) else ("oos+" if oos_pt > 0 else "")
            print(f"  {thr:>9.1f}{H:>6}{pairs_oos_pos:>5}/{len(PAIRS)}{is_pt:>9.1f}{oos_pt:>9.1f}{oos_wr:>7.0f}%{oos_n:>7}  {v}")
            score = oos_pt * (pairs_oos_pos >= 4)
            if best is None or score > best[0]:
                best = (score, thr, H, oos_pt, pairs_oos_pos)
    if best and best[0] > 0:
        print(f"\n  best multi-pair config: thr={best[1]}sig hold={best[2]}d -> OOS {best[3]:.1f} pips/trade, "
              f"{best[4]}/{len(PAIRS)} pairs OOS+")
        # per-pair detail at the best config
        thr, H = best[1], best[2]
        print(f"\n  per-pair @ thr={thr}sig hold={H}d:")
        for p in PAIRS:
            W = allW[p]; w = W[W["gap"].abs() >= thr * W["gsig"]]
            if len(w) < 20: continue
            cut = int(len(w)*0.70); d = -np.sign(w["gap"].values)
            net = d*(w[f"exit{H}"].values - w["entry_mid"].values)/PIP - w["spread"].values
            oos = net[cut:]
            print(f"    {p:<8} IS {net[:cut].mean():+6.1f}  OOS {oos.mean():+6.1f} p/t  WR {(oos>0).mean()*100:3.0f}%  n_oos {len(oos)}")
    else:
        print("\n  NO multi-pair OOS-positive config — weekend reversal does not survive here.")


if __name__ == "__main__":
    main()
