"""
Reversal Accumulator Signal (RACS) — IC Study
==============================================

Hypothesis: price levels with dense recent reversal history form S/R zones.
The Z-score of the current price bin's accumulated weight (vs all bins) should
have predictive IC for the next N M5 bars.

Construction
------------
1. Reversal detection: 3-bar Stage-1 local extremum on H1 or H4 bars.
   Causal 1-bar delay: when HTF bar i closes, bar i-1 is confirmed top if
   h[i-1] > h[i-2] AND h[i-1] > h[i].  Bottom: l[i-1] < l[i-2] AND l[i-1] < l[i].
   Known at last M5 bar of confirming HTF bar i → rev_bar_m5 = (i+1)*M5_PER_HTF - 1.

2. Price bins: integer-rounded to P&F box size.
   bin_idx = round(price / (box_pips * pip))
   Top events → bin(high[i-1]), bottoms → bin(low[i-1]).

3. Accumulator: at M5 bar m, for each past reversal j in lookback window:
   hist[bin_j] += sign_j / (m - j)        # age in M5 bars; scale-invariant for Z
   sign_j = +1 (top → resistance), -1 (bottom → support)
   Lookback window: m - j <= MAX_AGE_HTF * M5_PER_HTF

4. Signal at bar m: raw = hist[bin(close_m)]
   Z-score (sign-preserving, across all occupied bins):
   z(m) = (raw - mean(hist.values)) / std(hist.values)

   Positive z = current price has more top history → expect reversal downward
   Negative z = more bottom history → expect reversal upward
   Expected IC sign: negative (counter-trend)

5. Test: Spearman IC of z(m) vs next-N-bar M5 mid return, bootstrap p-value.
   IS/OOS 70/30. Gate: |t-stat| > 2.

Sweep: HTF ∈ {H1, H4}  ×  12 pairs  ×  8 horizons

SOP: R1 closed bars, R4 incremental (reversals precomputed, age tracked),
     R5 IS-only spread gate not applicable here (no entry signal).

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/reversal_acc/backtest_racs.py
"""

import time, gc
import numpy as np
import pandas as pd
import numba as nb
from scipy.stats import spearmanr, t as t_dist
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC     = 0.70
M5_PER_DAY  = 288

PAIRS = [
    ("GBP_JPY", 0.01), ("USD_JPY", 0.01), ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001), ("AUD_JPY", 0.01),
    ("CHF_JPY", 0.01), ("NZD_JPY", 0.01), ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001), ("NZD_USD", 0.0001), ("EUR_GBP", 0.0001),
]

HTF_MODES = [
    ("H1", 12),    # 12 M5 bars per H1
    ("H4", 48),    # 48 M5 bars per H4
]

HORIZONS    = [1, 6, 12, 24, 48, 72, 144, 288]  # M5 bars forward
BOX_PIPS    = 5                                   # price bin width in pips
MAX_AGE_HTF = 500                                 # lookback in HTF bars


# ── Numba kernel: compute accumulator + Z-score for all M5 bars ───────────────

@nb.njit(cache=True)
def _racs_kernel(close_bins,   # int64[n]  — bin index for each M5 bar's close
                 rev_bars,     # int64[R]  — M5 bar index when reversal became known
                 rev_bins,     # int64[R]  — price bin (absolute)
                 rev_signs,    # float64[R]— +1 top / -1 bottom
                 max_age_m5,   # int        — max lookback in M5 bars
                 bin_offset,   # int        — min(rev_bins) → maps to dense index 0
                 n_bins):      # int        — dense histogram size
    """
    Dense-array histogram: O(1) bin lookup, O(n_occ) reset per bar.
    Age weight = 1 / (m - j) where m-j is age in M5 bars.
    Scale-invariant for Z-score regardless of M5_PER_HTF.
    """
    n      = len(close_bins)
    n_rev  = len(rev_bars)
    raw    = np.zeros(n,  dtype=np.float64)
    zscore = np.full(n, np.nan, dtype=np.float64)

    hist     = np.zeros(n_bins, dtype=np.float64)
    occupied = np.zeros(n_bins, dtype=nb.boolean)
    occ_list = np.empty(1200, dtype=np.int64)
    n_occ    = 0

    for i in range(n):
        cb_abs = close_bins[i]
        cb     = cb_abs - bin_offset
        n_occ  = 0

        # Binary search: first reversal with bar_idx >= i - max_age_m5
        lo = 0; hi = n_rev; tgt = i - max_age_m5
        while lo < hi:
            m = (lo + hi) >> 1
            if rev_bars[m] < tgt: lo = m + 1
            else:                  hi = m

        # Fill dense histogram for window [lo, i) — only reversals known before bar i
        for k in range(lo, n_rev):
            j = rev_bars[k]
            if j >= i: break           # causal: only past reversals
            age = i - j
            rb  = rev_bins[k] - bin_offset
            if rb < 0 or rb >= n_bins: continue
            if not occupied[rb]:
                occupied[rb] = True
                if n_occ < 1200:
                    occ_list[n_occ] = rb
                    n_occ += 1
            hist[rb] += rev_signs[k] / age

        # Signal at current price bin
        sig = hist[cb] if 0 <= cb < n_bins else 0.0
        raw[i] = sig

        # Z-score across all occupied bins
        if n_occ >= 3:
            mv = 0.0
            for h in range(n_occ): mv += hist[occ_list[h]]
            mv /= n_occ
            vv = 0.0
            for h in range(n_occ): vv += (hist[occ_list[h]] - mv) ** 2
            vv /= n_occ
            if vv > 1e-15:
                zscore[i] = (sig - mv) / vv ** 0.5

        # Reset occupied bins only
        for h in range(n_occ):
            hist[occ_list[h]] = 0.0
            occupied[occ_list[h]] = False

    return raw, zscore


# ── HTF candle builder ────────────────────────────────────────────────────────

def make_htf_bars(highs, lows, M5_PER_HTF):
    """Group M5 OHLC into HTF bars by reshape. Returns full HTF arrays."""
    n_m5  = len(highs)
    n_htf = n_m5 // M5_PER_HTF
    used  = n_htf * M5_PER_HTF
    htf_h = highs[:used].reshape(n_htf, M5_PER_HTF).max(axis=1)
    htf_l = lows[:used].reshape(n_htf, M5_PER_HTF).min(axis=1)
    return htf_h, htf_l


# ── Reversal detection on HTF bars, returns M5 bar indices ───────────────────

def detect_reversals_htf(htf_highs, htf_lows, pip, box_pips, M5_PER_HTF, n_m5):
    """
    3-bar Stage-1 local extremum on HTF bars.
    Reversal is known when HTF confirming bar i closes:
      rev_bar_m5 = (i+1)*M5_PER_HTF - 1  (last M5 bar of HTF bar i)
    Returns (rev_bars, rev_bins, rev_signs) in M5 bar index space.
    """
    n  = len(htf_highs)
    bs = box_pips * pip
    rb, rbi, rs = [], [], []

    for i in range(2, n):
        confirm_m5 = min((i + 1) * M5_PER_HTF - 1, n_m5 - 1)
        # Top: bar i-1 is local high
        if htf_highs[i-1] > htf_highs[i-2] and htf_highs[i-1] > htf_highs[i]:
            rb.append(confirm_m5)
            rbi.append(int(round(htf_highs[i-1] / bs)))
            rs.append(1.0)
        # Bottom: bar i-1 is local low
        if htf_lows[i-1] < htf_lows[i-2] and htf_lows[i-1] < htf_lows[i]:
            rb.append(confirm_m5)
            rbi.append(int(round(htf_lows[i-1] / bs)))
            rs.append(-1.0)

    idx = np.argsort(rb, kind='stable')
    return (np.array(rb,  dtype=np.int64)[idx],
            np.array(rbi, dtype=np.int64)[idx],
            np.array(rs,  dtype=np.float64)[idx])


# ── IC computation with analytical p-value ────────────────────────────────────

def ic_test(signal, returns):
    """
    Spearman IC + two-sided p-value via t-distribution.
    For n > 100 the t-approximation is equivalent to the bootstrap;
    this avoids 500 × spearmanr calls per horizon (was ~200s per pair).
    """
    mask = ~(np.isnan(signal) | np.isnan(returns))
    s = signal[mask]; r = returns[mask]
    n = len(s)
    if n < 30:
        return np.nan, np.nan, np.nan, n
    ic, _ = spearmanr(s, r)
    if np.isnan(ic) or abs(ic) >= 1.0:
        return ic, np.nan, np.nan, n
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-12))
    p = float(2 * t_dist.sf(abs(t_stat), df=n - 2))
    return ic, p, round(t_stat, 3), n


# ── Per-pair analysis ─────────────────────────────────────────────────────────

def run_pair(pair, pip, htf_label, M5_PER_HTF):
    df = pd.read_parquet(BA_DIR / f"{pair}_M5_BA.parquet")
    n  = len(df)

    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    is_end = int(n * IS_FRAC)
    bs     = BOX_PIPS * pip
    close_bins = np.round(closes / bs).astype(np.int64)

    # Build HTF bars and detect reversals
    htf_h, htf_l = make_htf_bars(highs, lows, M5_PER_HTF)
    rev_bars, rev_bins, rev_signs = detect_reversals_htf(
        htf_h, htf_l, pip, BOX_PIPS, M5_PER_HTF, n)

    n_rev     = len(rev_bars)
    pct_rev   = 100.0 * n_rev / (n // M5_PER_HTF)   # % of HTF bars with a reversal

    # Dense histogram bounds
    if n_rev == 0:
        return pd.DataFrame(), [np.nan] * 10
    bin_offset = int(rev_bins.min()) - 2
    n_bins     = int(rev_bins.max()) - bin_offset + 3

    max_age_m5 = MAX_AGE_HTF * M5_PER_HTF

    # Compute RACS signal
    raw, zscore = _racs_kernel(close_bins, rev_bars, rev_bins, rev_signs,
                               max_age_m5, bin_offset, n_bins)

    close_p  = closes / pip
    oos_start = is_end

    results = []
    for h in HORIZONS:
        ret = (close_p[h:] - close_p[:-h])
        sig = zscore[:-h]

        s_oos = sig[oos_start : oos_start + len(ret) - oos_start]
        r_oos = ret[oos_start : oos_start + len(ret) - oos_start]
        ic_oos, p_oos, t_stat, n_oos = ic_test(s_oos, r_oos)

        s_is = sig[:is_end]
        r_is = ret[:is_end]
        ic_is, p_is, _, n_is = ic_test(s_is, r_is)

        results.append({
            "htf":     htf_label,
            "pair":    pair,
            "horizon": h,
            "ic_is":   round(ic_is,  4) if not np.isnan(ic_is)  else np.nan,
            "ic_oos":  round(ic_oos, 4) if not np.isnan(ic_oos) else np.nan,
            "p_oos":   round(p_oos,  3) if not np.isnan(p_oos)  else np.nan,
            "t_stat":  round(t_stat, 2) if not np.isnan(t_stat) else np.nan,
            "n_oos":   n_oos,
            "n_rev":   n_rev,
            "pct_rev": round(pct_rev, 1),
        })

    # Decile analysis on OOS for H=12
    h12  = 12
    ret12 = (close_p[h12:] - close_p[:-h12])
    sig12 = zscore[:-h12]
    s_oos = sig12[oos_start:]
    r_oos = ret12[oos_start:]
    mask  = ~(np.isnan(s_oos) | np.isnan(r_oos))
    if mask.sum() > 100:
        deciles   = np.nanpercentile(s_oos[mask], np.linspace(0, 100, 11))
        bin_means = []
        for d in range(10):
            in_bin = (s_oos >= deciles[d]) & (s_oos < deciles[d+1]) & mask
            bin_means.append(r_oos[in_bin].mean() if in_bin.sum() > 0 else np.nan)
        decile_ret = bin_means
    else:
        decile_ret = [np.nan] * 10

    return pd.DataFrame(results), decile_ret


# ── Print results for one HTF mode ────────────────────────────────────────────

def print_htf_results(full, all_dec, htf_label):
    df = full[full["htf"] == htf_label]
    if df.empty:
        return

    header = f"{'Pair':<10}" + "".join(f" H={h:>3}" for h in HORIZONS)

    print()
    print("=" * 100)
    print(f"[{htf_label} reversals]  OOS Spearman IC  (** p<0.01  * p<0.05  ~ p<0.10)")
    print("=" * 100)
    print(header)
    print("-" * len(header))

    n_sig_05 = 0; n_sig_01 = 0; n_total = 0

    for pair, _ in PAIRS:
        row = df[df["pair"] == pair]
        if row.empty: continue
        line = f"{pair:<10}"
        for h in HORIZONS:
            r   = row[row["horizon"] == h]
            if r.empty:
                line += "     nan "; continue
            r = r.iloc[0]
            ic  = r["ic_oos"]; p = r["p_oos"]
            sig = "**" if (not np.isnan(p) and p < 0.01) else \
                  "*"  if (not np.isnan(p) and p < 0.05) else \
                  "~"  if (not np.isnan(p) and p < 0.10) else "  "
            val = f"{ic:+.3f}{sig}" if not np.isnan(ic) else "  nan  "
            line += f" {val:>7}"
            n_total += 1
            if not np.isnan(p):
                if p < 0.05: n_sig_05 += 1
                if p < 0.01: n_sig_01 += 1
        print(line)

    print("-" * len(header))
    pct = 100 * n_sig_05 / n_total if n_total > 0 else 0
    print(f"Significant (p<0.05): {n_sig_05}/{n_total}  |  (p<0.01): {n_sig_01}/{n_total}"
          f"  |  Observed: {pct:.1f}%  (H0 baseline: 5.0%)")

    # t-stat heatmap
    print()
    print(f"[{htf_label} reversals]  OOS t-stat  (|t|>2 = gate pass)")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for pair, _ in PAIRS:
        row = df[df["pair"] == pair]
        if row.empty: continue
        line = f"{pair:<10}"
        for h in HORIZONS:
            r = row[row["horizon"] == h]
            if r.empty: line += "     nan "; continue
            t = r.iloc[0]["t_stat"]
            flag = "!" if (not np.isnan(t) and abs(t) >= 2.0) else " "
            line += f" {t:+6.2f}{flag}" if not np.isnan(t) else "     nan "
        print(line)

    # Decile table
    print()
    print(f"[{htf_label} reversals]  OOS Decile mean return at H=12 (D1=lowest z, D10=highest z)")
    print("Counter-trend → D1 positive, D10 negative (monotone decrease L→R)")
    print("=" * 100)
    print(f"{'Pair':<10}" + "".join(f" D{i+1:>4}" for i in range(10)))
    print("-" * 60)
    for pair, _ in PAIRS:
        dec = all_dec.get((htf_label, pair), [np.nan]*10)
        line = f"{pair:<10}"
        for v in dec:
            line += f" {v:>5.1f}" if not np.isnan(v) else "   nan"
        print(line)

    # IS vs OOS sanity at H=12
    print()
    print(f"[{htf_label} reversals]  IS vs OOS IC at H=12")
    print(f"{'Pair':<10} {'IC_IS':>8} {'IC_OOS':>8} {'t_OOS':>8} {'p_OOS':>8} {'n_rev':>7}")
    print("-" * 50)
    for pair, _ in PAIRS:
        r = df[(df["pair"] == pair) & (df["horizon"] == 12)]
        if r.empty: continue
        r = r.iloc[0]
        print(f"{r['pair']:<10} {r['ic_is']:>8.4f} {r['ic_oos']:>8.4f} "
              f"{r['t_stat']:>8.2f} {r['p_oos']:>8.3f} {r['n_rev']:>7}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Reversal Accumulator Signal (RACS) — IC Study")
    print(f"box_pips={BOX_PIPS}  max_age={MAX_AGE_HTF} HTF bars  IS={IS_FRAC:.0%}  OOS={1-IS_FRAC:.0%}")
    print(f"HTF modes: {[m[0] for m in HTF_MODES]}  (H1=12M5, H4=48M5)")
    print(f"max_age in M5: H1={MAX_AGE_HTF*12}  H4={MAX_AGE_HTF*48}")
    print(f"Horizons (M5 bars): {HORIZONS}")
    print()

    print("JIT compile...", end=" ", flush=True)
    _cb  = np.array([10, 11, 10, 12, 11, 10], dtype=np.int64)
    _rb  = np.array([0, 2, 4], dtype=np.int64)
    _rbi = np.array([10, 11, 12], dtype=np.int64)
    _rs  = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    _racs_kernel(_cb, _rb, _rbi, _rs, 20, 9, 6)
    print("done.\n")

    all_dfs = []
    all_dec = {}

    for htf_label, M5_PER_HTF in HTF_MODES:
        print(f"── {htf_label} reversals (M5_PER_HTF={M5_PER_HTF}) ──")
        for pair, pip in PAIRS:
            t1 = time.time()
            print(f"  {pair}...", end=" ", flush=True)
            df_res, deciles = run_pair(pair, pip, htf_label, M5_PER_HTF)
            if not df_res.empty:
                all_dfs.append(df_res)
                all_dec[(htf_label, pair)] = deciles
                nr = df_res["n_rev"].iloc[0]
                pr = df_res["pct_rev"].iloc[0]
                print(f"n_rev={nr} ({pr:.1f}% of HTF bars) — {time.time()-t1:.1f}s")
            else:
                print("no reversals")
            gc.collect()
        print()

    full = pd.concat(all_dfs, ignore_index=True)

    # Print results per HTF mode
    for htf_label, _ in HTF_MODES:
        print_htf_results(full, all_dec, htf_label)

    # Summary: count passes across all HTF × pair × horizon
    print()
    print("=" * 100)
    print("SUMMARY — gate passes (|t|>2) across all HTF modes")
    print("=" * 100)
    for htf_label, _ in HTF_MODES:
        df = full[full["htf"] == htf_label]
        passes = df[df["t_stat"].abs() >= 2.0].shape[0]
        total  = df.shape[0]
        print(f"  {htf_label}: {passes}/{total} cell passes  ({100*passes/total:.1f}%)")

    out = Path(__file__).parent / "results_racs.csv"
    full.to_csv(out, index=False)
    print(f"\nFull results → {out}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
