#!/usr/bin/env python3
"""
M/W/D/H/M5 alignment score → forward-return study.

For each M5 bar, capture the sign of the last completed bar (current close vs
prior completed bar's close) on:
   M  (monthly)
   W  (weekly)
   D  (daily)
   H  (hourly)
   M5 (5-minute)

   alignment = sign(M) + sign(W) + sign(D) + sign(H) + sign(M5)   ∈ {−5, …, +5}

Question: does this alignment score predict forward returns at 1h / 4h / 24h
horizons?  Display:
  bin (score)   n   mean fwd_1h (pips)  WR%   mean fwd_4h   mean fwd_24h
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70

# Forward windows (in M5 bars)
FWD_WINDOWS = {"1h": 12, "4h": 48, "24h": 288}


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def per_tf_sign(df_m5, freq):
    """At each M5 bar, what's the sign of the last completed TF bar's close
    delta vs the bar before that? R1-clean: shift TF bars by 1 before
    reindexing so the current in-progress bar isn't used.
    """
    closes = df_m5["close"].resample(freq).last().dropna()
    diffs  = closes.diff()
    # Sign of last COMPLETED bar = shift by 1 (current TF bar may still be open)
    sign_shifted = np.sign(diffs).shift(1)
    aligned = sign_shifted.reindex(df_m5.index, method="ffill").fillna(0)
    return aligned.values.astype(np.int8)


def run_pair(pair, fwd_rows, score_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    pip = pip_sz(pair)

    # Sign per TF
    m_sign  = per_tf_sign(df_m5, "1MS")    # month-start
    w_sign  = per_tf_sign(df_m5, "1W")
    d_sign  = per_tf_sign(df_m5, "1D")
    h_sign  = per_tf_sign(df_m5, "1h")
    m5_sign = per_tf_sign(df_m5, "5min")

    score = (m_sign + w_sign + d_sign + h_sign + m5_sign).astype(np.int8)

    close = df_m5["close"].values.astype(np.float64)
    n = len(close)
    n_is = int(n * IS_FRAC)

    # Forward returns at each window (in pips), evaluated at every M5 bar in OOS
    # Then bin by score.
    score_oos = score[n_is:]
    close_oos = close[n_is:]

    for lbl, W in FWD_WINDOWS.items():
        if len(close_oos) <= W:
            continue
        fwd = (close_oos[W:] - close_oos[:-W]) / pip   # signed forward return in pips
        scr = score_oos[:-W]
        n_pts = len(fwd)
        # Bin by score
        for s in range(-5, 6):
            mask = (scr == s)
            cnt = int(mask.sum())
            if cnt == 0:
                fwd_rows.append(dict(pair=pair, fwd_window=lbl, score=s,
                                     n=0, mean_pips=0.0, wr=0.0, std_pips=0.0))
                continue
            fwd_rows.append(dict(
                pair=pair, fwd_window=lbl, score=int(s),
                n=cnt,
                mean_pips=round(float(fwd[mask].mean()), 3),
                wr=round(float((fwd[mask] > 0).mean()) * 100, 1),
                std_pips=round(float(fwd[mask].std()), 1),
            ))

    # Also: signed correlation of score and forward returns
    # (pair-level summary)
    for lbl, W in FWD_WINDOWS.items():
        if len(close_oos) <= W:
            continue
        fwd = (close_oos[W:] - close_oos[:-W]) / pip
        scr = score_oos[:-W]
        # Convert to numpy for correlation
        corr = np.corrcoef(scr.astype(np.float64), fwd)[0, 1]
        score_rows.append(dict(pair=pair, fwd_window=lbl,
                               corr=round(float(corr), 4),
                               n=int(len(fwd))))

    del df_m5
    gc.collect()


def main():
    print("M/W/D/H/M5 alignment → forward-return study")
    print(f"  pairs={len(ALL_PAIRS)}  forward windows={list(FWD_WINDOWS.keys())}")
    fwd_rows = []; score_rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, fwd_rows, score_rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    fwd_df = pd.DataFrame(fwd_rows)
    s_df   = pd.DataFrame(score_rows)
    fwd_df.to_csv(OUT / "mwdh_nesting_bins.csv", index=False)
    s_df.to_csv(OUT / "mwdh_nesting_corr.csv", index=False)
    print(f"\n→ results/mwdh_nesting_bins.csv  ({len(fwd_df)} rows)")
    print(f"→ results/mwdh_nesting_corr.csv  ({len(s_df)} rows)")
    print(f"({time.time()-t0:.1f}s total)")

    # ── Aggregate across all 12 pairs ─────────────────────────────────────
    print("\n" + "="*100)
    print("  AGGREGATE (sum across 12 pairs) — forward return by alignment score")
    print("="*100)
    for lbl in FWD_WINDOWS:
        sub = fwd_df[fwd_df.fwd_window == lbl]
        g = (sub.groupby("score")
              .agg(n=("n","sum"),
                   mean_pips=("mean_pips", lambda s: np.average(s,
                       weights=sub.loc[s.index, "n"]) if sub.loc[s.index, "n"].sum() > 0 else 0),
                   wr=("wr", lambda s: np.average(s,
                       weights=sub.loc[s.index, "n"]) if sub.loc[s.index, "n"].sum() > 0 else 0))
              .reset_index())
        print(f"\n  --- forward window: {lbl} ---")
        print(f"  {'score':>6}  {'n':>9}  {'mean fwd (pips)':>18}  {'WR%':>6}")
        for _, r in g.iterrows():
            if r.n == 0: continue
            sign_arrow = ('▲' if r.score > 0 else ('▼' if r.score < 0 else '─'))
            print(f"  {int(r.score):>+6} {sign_arrow}  {int(r.n):>9,}  {r.mean_pips:>+18.3f}  {r.wr:>6.1f}")

    # ── Correlation summary ──────────────────────────────────────────────
    print("\n" + "="*100)
    print("  Pair-level Pearson correlation: alignment_score vs forward return")
    print("="*100)
    print(f"  {'pair':<10}  " + "  ".join(f"{lbl:>10}" for lbl in FWD_WINDOWS))
    for pair in ALL_PAIRS:
        row = []
        for lbl in FWD_WINDOWS:
            r = s_df[(s_df.pair == pair) & (s_df.fwd_window == lbl)]
            row.append(float(r["corr"].iloc[0]) if len(r) else 0.0)
        s_str = "  ".join(f"{c:>+10.4f}" for c in row)
        print(f"  {pair:<10}  {s_str}")

    # Mean corr across pairs
    print("\n  ─── mean across 12 pairs ───")
    print(f"  {'mean':<10}  " + "  ".join(
        f"{s_df[s_df.fwd_window == lbl]['corr'].mean():>+10.4f}" for lbl in FWD_WINDOWS))


if __name__ == "__main__":
    main()
