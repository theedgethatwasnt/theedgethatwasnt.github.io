#!/usr/bin/env python3
"""
Alignment-score filter on the live momentum book.

Hypothesis (from chart of 012/AUD_JPY entered at 114.75 near peak after
5-month uptrend): the SMA16 H1+M30 strategy walks into momentum-exhaustion
zones because it's blind to multi-TF alignment. The bin study showed
forward returns are NEGATIVE when alignment_score is extreme positive
(+5 → -0.67p/24h; +4 → -1.17p/24h; +3 → +0.31p/24h).

This experiment: replay live SMA16 entries OOS, compute alignment_score
at each entry, compare baseline vs filtered.

Filter rule:
   skip LONG  if alignment_score >=  +score_thr
   skip SHORT if alignment_score <=  -score_thr

Sweep score_thr ∈ {3, 4, 5}.

Live strategy (012):
   SMA_N=16, LAGS=(8,10,15), TP=20p broker-side, no SL.
   compute_signal() — LONG if all 6 SMA-momentums positive on H1 + M30.

For honest accounting: trades open at signal-formation bar, exit at +20p
(broker TP) or +60p MAE (unrealistic floor — no live SL, but caps the
backtest's worst case) or end of data.
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}

SMA_N    = 16
LAGS     = (8, 10, 15)
TP_PIPS  = 20.0
IS_FRAC  = 0.70
SCORE_THRS = [3, 4, 5]
# Cap MAE to prevent absurd backtest trade hold times when TP never fires
MAX_HOLD_M5 = 2880   # 10 days (288 bars/day × 10)


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def per_tf_sign(df_m5, freq):
    """Sign of last completed TF bar's close-delta, M5-aligned, R1-clean."""
    closes = df_m5["close"].resample(freq).last().dropna()
    diffs  = closes.diff()
    sign_shifted = np.sign(diffs).shift(1)
    return sign_shifted.reindex(df_m5.index, method="ffill").fillna(0).values.astype(np.int8)


def build_sma_signal(df_m5):
    """Replicate live SMA16 H1+M30 compute_signal at M5 cadence.

    Returns sig array (length n_m5) with +1/-1/0. Signal changes only on
    H1 / M30 bar boundaries; in-between bars carry forward the prior signal
    (the strategy doesn't fire mid-bar but it stays armed)."""
    # Compute on each TF then reindex to M5
    sig_combined = np.zeros(len(df_m5), dtype=np.int8)

    for tf_label, tf_freq in [("H1", "1h"), ("M30", "30min")]:
        closes_tf = df_m5["close"].resample(tf_freq).last().dropna()
        sma_tf = closes_tf.rolling(SMA_N).mean()
        # For each lag, compute sma_now - sma_lag
        # All 3 lags positive AND sma_now > sma_at_lag for all lags → long signal
        # Use shift to align: signal[i] = check (sma[i] - sma[i-k]) for k in LAGS
        # Pad and reindex to M5
        moms_pos = np.ones(len(closes_tf), dtype=bool)
        moms_neg = np.ones(len(closes_tf), dtype=bool)
        for k in LAGS:
            delta = sma_tf - sma_tf.shift(k)
            moms_pos = moms_pos & (delta > 0).values
            moms_neg = moms_neg & (delta < 0).values
        sig_tf = np.zeros(len(closes_tf), dtype=np.int8)
        sig_tf[moms_pos] = 1
        sig_tf[moms_neg] = -1
        # R1 shift by 1 then reindex to M5 (only completed TF bars)
        sig_tf_series = pd.Series(sig_tf, index=closes_tf.index).shift(1)
        sig_tf_aligned = sig_tf_series.reindex(df_m5.index, method="ffill").fillna(0).values.astype(np.int8)
        # Combine: BOTH TFs must agree (this is how the live code works —
        # the all-6-positive includes 3 from H1 + 3 from M30 simultaneously)
        if tf_label == "H1":
            sig_combined = sig_tf_aligned.copy()
        else:
            # Both TFs must be +1 for long, -1 for short
            both_long  = (sig_combined == 1)  & (sig_tf_aligned == 1)
            both_short = (sig_combined == -1) & (sig_tf_aligned == -1)
            sig_combined[:] = 0
            sig_combined[both_long]  = 1
            sig_combined[both_short] = -1

    return sig_combined


@njit(cache=True)
def replay_entries(close, bid, ask, sp, sig, score, pip, tp_pips,
                   sp_gate, max_hold_m5, n_is):
    """Replay live SMA16 entries with TP-only exit.

    Entry rule (live): first M5 bar where prev_sig != 0 → cur_sig != prev_sig,
    or more practically: when signal crosses from 0/opposite to new direction.
    We treat: first bar where signal changes value to a non-zero state with
    sp <= sp_gate → enter. Re-entry only after position closes.

    Returns per-trade: pnl, direction, alignment_score_at_entry.
    """
    n = len(close)
    pnl_arr   = np.empty(n, dtype=np.float64)
    dir_arr   = np.empty(n, dtype=np.int8)
    score_arr = np.empty(n, dtype=np.int8)
    hold_arr  = np.empty(n, dtype=np.int32)
    type_arr  = np.empty(n, dtype=np.int8)   # 0=TP, 1=maxhold, 2=eod
    count = 0
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0
    entry_score = 0
    prev_sig = 0

    for i in range(n_is, n):
        s = sig[i]
        if in_trade:
            # Exit at TP if net pnl >= tp_pips
            if direction == 1:
                net_pnl = (bid[i] - entry_price) / pip
            else:
                net_pnl = (entry_price - ask[i]) / pip
            if net_pnl >= tp_pips:
                pnl_arr[count]   = tp_pips
                hold_arr[count]  = i - entry_bar
                type_arr[count]  = 0
                dir_arr[count]   = direction
                score_arr[count] = entry_score
                count += 1; in_trade = False
            elif i - entry_bar >= max_hold_m5:
                pnl_arr[count]   = net_pnl
                hold_arr[count]  = i - entry_bar
                type_arr[count]  = 1
                dir_arr[count]   = direction
                score_arr[count] = entry_score
                count += 1; in_trade = False
        else:
            # Detect signal crossing: previously not in this direction,
            # now signal == new direction, and spread OK
            if s != 0 and s != prev_sig and sp[i] <= sp_gate:
                # Open new position
                if s == 1:
                    entry_price = ask[i]
                else:
                    entry_price = bid[i]
                entry_bar = i
                direction = s
                entry_score = score[i]
                in_trade = True
        prev_sig = s

    # Close any open position at EOD with current market
    if in_trade:
        i = n - 1
        if direction == 1:
            pnl_arr[count] = (bid[i] - entry_price) / pip
        else:
            pnl_arr[count] = (entry_price - ask[i]) / pip
        hold_arr[count]  = i - entry_bar
        type_arr[count]  = 2
        dir_arr[count]   = direction
        score_arr[count] = entry_score
        count += 1

    return (pnl_arr[:count], dir_arr[:count], score_arr[:count],
            hold_arr[:count], type_arr[:count])


def warmup_jit():
    n = 1000
    c = np.linspace(1.0, 1.01, n)
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n); sig = np.zeros(n, dtype=np.int8); score = np.zeros(n, dtype=np.int8)
    sig[100] = 1; sig[500] = -1
    score[100] = -2; score[500] = 3
    replay_entries(c, b, a, sp, sig, score, 0.0001, 20.0, 2.0, 2880, 0)


def run_pair(pair, all_rows):
    df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
            .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]

    close = df["close"].values.astype(np.float64)
    bid   = df["bid_c"].values.astype(np.float64)
    ask   = df["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 288.0

    sig = build_sma_signal(df)

    # Alignment score
    m_sign  = per_tf_sign(df, "1MS")
    w_sign  = per_tf_sign(df, "1W")
    d_sign  = per_tf_sign(df, "1D")
    h_sign  = per_tf_sign(df, "1h")
    m5_sign = per_tf_sign(df, "5min")
    score = (m_sign + w_sign + d_sign + h_sign + m5_sign).astype(np.int8)

    p, d, s_score, h, t = replay_entries(
        close, bid, ask, sp, sig, score, pip, TP_PIPS, sg, MAX_HOLD_M5, n_is)
    n = len(p)
    if n == 0:
        return

    # Baseline (all trades)
    base_pnl = float(p.sum())
    base_ppd = base_pnl / oos_days
    base_wr  = (p > 0).sum() / n * 100
    base_n   = n

    all_rows.append(dict(
        pair=pair, filter="none", score_thr=0,
        n=n, sum_pips=round(base_pnl, 1),
        ppd=round(base_ppd, 2),
        wr=round(base_wr, 1),
        n_long=int((d == 1).sum()),
        n_short=int((d == -1).sum()),
        days=round(oos_days, 1),
    ))

    # Filtered for each threshold
    for thr in SCORE_THRS:
        # Skip LONG when score_arr >= +thr, SHORT when score_arr <= -thr
        keep = np.ones(n, dtype=np.bool_)
        for k in range(n):
            if d[k] == 1 and s_score[k] >= thr:
                keep[k] = False
            elif d[k] == -1 and s_score[k] <= -thr:
                keep[k] = False
        p_k = p[keep]; d_k = d[keep]; s_k = s_score[keep]
        n_k = len(p_k)
        if n_k == 0:
            all_rows.append(dict(
                pair=pair, filter=f"score>={thr}", score_thr=thr,
                n=0, sum_pips=0.0, ppd=0.0, wr=0.0,
                n_long=0, n_short=0, days=round(oos_days, 1),
            ))
            continue
        ppd_k = p_k.sum() / oos_days
        wr_k  = (p_k > 0).sum() / n_k * 100
        all_rows.append(dict(
            pair=pair, filter=f"score>={thr}", score_thr=thr,
            n=n_k, sum_pips=round(float(p_k.sum()), 1),
            ppd=round(ppd_k, 2), wr=round(wr_k, 1),
            n_long=int((d_k == 1).sum()),
            n_short=int((d_k == -1).sum()),
            days=round(oos_days, 1),
        ))

    # Also: bin the trades by alignment score for diagnostics
    for s_val in range(-5, 6):
        mask = (s_score == s_val)
        if not mask.any(): continue
        sub_p = p[mask]
        sub_d = d[mask]
        all_rows.append(dict(
            pair=pair, filter=f"_bin_{s_val:+d}", score_thr=s_val,
            n=int(mask.sum()),
            sum_pips=round(float(sub_p.sum()), 1),
            ppd=round(float(sub_p.sum()) / oos_days, 2),
            wr=round(float((sub_p > 0).mean()) * 100, 1),
            n_long=int((sub_d == 1).sum()),
            n_short=int((sub_d == -1).sum()),
            days=round(oos_days, 1),
        ))

    del df
    gc.collect()


def main():
    warmup_jit()
    print("Alignment-score filter on SMA16 H1+M30 (account 012)")
    print(f"  SMA_N={SMA_N}, LAGS={LAGS}, TP={TP_PIPS}p")
    print(f"  filter sweep: score_thr ∈ {SCORE_THRS}")
    print(f"  max_hold_M5={MAX_HOLD_M5} (10 days — cap on TP-only no-SL strategy)")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        n_base = next((r["n"] for r in rows if r["pair"] == pair and r["filter"] == "none"), 0)
        print(f"  {pair}: {time.time()-ts:.1f}s  baseline_n={n_base}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "alignment_filter_momentum.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # Per-pair: baseline vs each filter
    print("\n" + "="*120)
    print("  Per-pair: baseline vs filtered  (* = pair where filter HELPS)")
    print("="*120)
    print(f"  {'pair':<10}  "
          f"{'base_n':>7}{'base_ppd':>10}{'base_wr':>9}  | "
          f"{'thr3_n':>7}{'thr3_ppd':>10}{'thr3_wr':>9}{'Δppd':>8}  | "
          f"{'thr4_n':>7}{'thr4_ppd':>10}{'thr4_wr':>9}{'Δppd':>8}  | "
          f"{'thr5_n':>7}{'thr5_ppd':>10}{'thr5_wr':>9}{'Δppd':>8}")
    for pair in ALL_PAIRS:
        b = df[(df.pair == pair) & (df["filter"] == "none")].iloc[0]
        out = f"  {pair:<10}  {int(b['n']):>7d}{b['ppd']:>+10.2f}{b['wr']:>9.1f}  |"
        for thr in SCORE_THRS:
            f = df[(df.pair == pair) & (df["filter"] == f"score>={thr}")]
            if len(f):
                r = f.iloc[0]
                d_ppd = r['ppd'] - b['ppd']
                marker = "*" if d_ppd > 0 else " "
                out += f" {int(r['n']):>7d}{r['ppd']:>+10.2f}{r['wr']:>9.1f}{d_ppd:>+7.2f}{marker} |"
            else:
                out += f" {'-':>7s}{'-':>10s}{'-':>9s}{'-':>8s}  |"
        print(out)

    # Portfolio totals
    print("\n" + "="*120)
    print("  Portfolio totals (12 pairs)")
    print("="*120)
    for filter_label, thr in [("baseline", 0), ("score>=+3", 3), ("score>=+4", 4), ("score>=+5", 5)]:
        if thr == 0:
            sub = df[df["filter"] == "none"]
        else:
            sub = df[df["filter"] == f"score>={thr}"]
        total = sub.sum_pips.sum()
        avg_days = sub.days.mean()
        ppd = total / avg_days
        n_total = sub.n.sum()
        n_pos = int((sub.ppd > 0).sum())
        wr_mean = sub.wr.mean() if n_total > 0 else 0
        print(f"  {filter_label:<12}: Σ pips={total:>+9.1f}  ppd={ppd:>+7.2f}  "
              f"trades={int(n_total):>5}  pairs+={n_pos}/12  mean WR={wr_mean:>5.1f}%")

    # Bin-level analysis: forward returns by alignment score
    print("\n" + "="*120)
    print("  Trades binned by alignment_score at entry (aggregate across 12 pairs)")
    print("="*120)
    bin_rows = df[df["filter"].str.startswith("_bin_", na=False)]
    print(f"  {'score':>6}  {'n':>6}  {'Σ pips':>10}  {'mean':>9}  {'WR%':>5}  "
          f"{'n_long':>7}  {'n_short':>8}")
    for s in range(-5, 6):
        sub = bin_rows[bin_rows.score_thr == s]
        if not len(sub): continue
        n_b = int(sub.n.sum())
        if n_b == 0: continue
        sp_b = sub.sum_pips.sum()
        mean_b = sp_b / n_b
        n_long_b = int(sub.n_long.sum())
        n_short_b = int(sub.n_short.sum())
        # WR weighted
        wr_b = (sub.wr * sub.n).sum() / n_b
        print(f"  {s:>+6d}  {n_b:>6d}  {sp_b:>+10.1f}  {mean_b:>+9.2f}  "
              f"{wr_b:>5.1f}  {n_long_b:>7d}  {n_short_b:>8d}")


if __name__ == "__main__":
    main()
