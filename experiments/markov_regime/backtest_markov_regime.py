#!/usr/bin/env python3
"""
Markov Regime Transition Matrix — Phase 1: IC Study
====================================================
Question: Does the current D1 regime (Bull/Sideways/Bear) give predictive
power for tomorrow's direction in FX?

Method:
  1. Resample M5 closes to D1 log-returns
  2. Label each day: Bull (N-day cumret > thr), Bear (< -thr), Sideways
  3. Build causal empirical transition matrix T[state_t → state_{t+1}]
     — updated daily using only data up to day t (no lookahead, R1)
  4. Signal = P(Bull tomorrow | state_t) − P(Bear tomorrow | state_t)
  5. IC = Spearman correlation of signal with next-day log-return
  6. Report IS IC, OOS IC, t-stat per (pair, window, threshold) config

Sweep: window ∈ {5,10,20} D1 bars × thr ∈ {0.1,0.2,0.5,1.0%} × 4 JPY pairs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAIRS   = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
IS_FRAC = 0.70

WINDOWS    = [5, 10, 20]
THRESHOLDS = [0.001, 0.002, 0.005, 0.010]  # 0.1% to 1.0%
MIN_PRIME  = 30   # minimum transitions to trust matrix before generating signal

BULL, SIDE, BEAR = 0, 1, 2
SNAMES = ["Bull", "Side", "Bear"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_d1(pair: str) -> pd.Series:
    """M5 BA → D1 log-returns via last-close-of-day resampling."""
    df  = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
           .set_index("timestamp").sort_index())
    d1  = df["close"].resample("1D").last().dropna()
    lr  = np.log(d1 / d1.shift(1)).dropna()
    return lr


# ── State labeling ────────────────────────────────────────────────────────────

def label_states(lr: pd.Series, window: int, thr: float) -> pd.Series:
    """
    Rolling N-bar cumulative return → state label.
    Causal: rolling(window).sum() uses only past bars (R1).
    """
    roll = lr.rolling(window).sum()
    return roll.apply(
        lambda r: BULL if r > thr else (BEAR if r < -thr else SIDE)
        if not np.isnan(r) else np.nan
    ).dropna().astype(int)


# ── Causal transition matrix ──────────────────────────────────────────────────

def compute_signals(states: pd.Series, lr: pd.Series, is_cutoff_idx: int
                    ) -> pd.DataFrame:
    """
    Walk forward through states:
      - Maintain running T[3×3] count matrix
      - At each day t: signal = P(Bull|s) - P(Bear|s) from T built on [0,t-1]
      - Target: next-day log-return

    Returns DataFrame with columns: date, signal, ret_next, is_oos
    """
    T       = np.zeros((3, 3), dtype=np.float64)
    records = []

    for i in range(len(states) - 1):
        s     = states.iloc[i]
        s_nxt = states.iloc[i + 1]
        day   = states.index[i]

        # Generate signal BEFORE updating T with today's transition (causal)
        row_sum = T[s].sum()
        if row_sum >= MIN_PRIME:
            p_bull = T[s, BULL] / row_sum
            p_bear = T[s, BEAR] / row_sum
            sig    = p_bull - p_bear

            # Target: next-day log-return
            nxt_day = states.index[i + 1]
            if nxt_day in lr.index:
                records.append({
                    "date":    nxt_day,
                    "signal":  sig,
                    "p_bull":  p_bull,
                    "p_bear":  p_bear,
                    "state":   s,
                    "ret":     lr.loc[nxt_day],
                    "is_oos":  "IS" if i < is_cutoff_idx else "OOS",
                })

        # Update T with today's transition
        T[s, s_nxt] += 1.0

    return pd.DataFrame(records)


def ic_stats(sig: np.ndarray, ret: np.ndarray):
    """Spearman IC + t-stat."""
    if len(sig) < 10:
        return np.nan, np.nan, np.nan
    rho, pval = stats.spearmanr(sig, ret)
    n = len(sig)
    t = rho * np.sqrt(n - 2) / np.sqrt(max(1 - rho**2, 1e-12))
    return rho, t, pval


def stickiness(T_hist: np.ndarray) -> float:
    """Mean diagonal of row-normalised transition matrix."""
    denom = T_hist.sum(axis=1, keepdims=True).clip(1e-9)
    return float(np.mean(np.diag(T_hist / denom)))


# ── Direction accuracy ────────────────────────────────────────────────────────

def dir_accuracy(sig: np.ndarray, ret: np.ndarray) -> float:
    """% of trades where sign(signal) == sign(return) (excluding zeros)."""
    mask = sig != 0
    if mask.sum() < 5:
        return np.nan
    s, r = sig[mask], ret[mask]
    return float((np.sign(s) == np.sign(r)).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading D1 returns …")
lr_cache = {}
for pair in PAIRS:
    lr_cache[pair] = load_d1(pair)
    print(f"  {pair}: {len(lr_cache[pair])} D1 bars  "
          f"({len(lr_cache[pair])/252:.1f} years)")

all_rows = []
best_T_cache = {}   # store final T for best config per pair

for pair in PAIRS:
    lr    = lr_cache[pair]
    n_all = len(lr)
    print(f"\n{'─'*60}")
    print(f"{pair}")

    for window in WINDOWS:
        for thr in THRESHOLDS:
            states = label_states(lr, window, thr)
            if len(states) < 60:
                continue

            is_n   = int(len(states) * IS_FRAC)
            df_sig = compute_signals(states, lr, is_n)
            if df_sig.empty:
                continue

            # IS
            is_df  = df_sig[df_sig["is_oos"] == "IS"]
            oos_df = df_sig[df_sig["is_oos"] == "OOS"]

            ic_is,  t_is,  _ = ic_stats(is_df["signal"].values,  is_df["ret"].values)
            ic_oos, t_oos, _ = ic_stats(oos_df["signal"].values, oos_df["ret"].values)
            acc_is  = dir_accuracy(is_df["signal"].values,  is_df["ret"].values)
            acc_oos = dir_accuracy(oos_df["signal"].values, oos_df["ret"].values)

            # Final (IS+OOS) transition matrix for stickiness score
            T_final = np.zeros((3, 3))
            for i in range(len(states) - 1):
                T_final[states.iloc[i], states.iloc[i + 1]] += 1
            sticky = stickiness(T_final)

            # State distribution
            state_counts = np.bincount(states.values, minlength=3) / len(states)

            row = dict(
                pair=pair, window=window, thr=thr,
                n_is=len(is_df), n_oos=len(oos_df),
                ic_is=ic_is, t_is=t_is,
                ic_oos=ic_oos, t_oos=t_oos,
                acc_is=acc_is, acc_oos=acc_oos,
                stickiness=sticky,
                bull_pct=state_counts[BULL]*100,
                side_pct=state_counts[SIDE]*100,
                bear_pct=state_counts[BEAR]*100,
            )
            all_rows.append(row)

            marker = " ◄" if abs(t_oos) > 2 else ""
            print(f"  win={window:2d} thr={thr:.3f}  "
                  f"IS: IC={ic_is:+.3f} t={t_is:+.1f}  "
                  f"OOS: IC={ic_oos:+.3f} t={t_oos:+.1f}  "
                  f"acc={acc_oos:.2f}  sticky={sticky:.2f}{marker}")

            # Cache best T (by |t_oos|) per pair
            if (pair not in best_T_cache or
                    abs(t_oos) > abs(best_T_cache[pair]["t_oos"])):
                best_T_cache[pair] = {
                    "t_oos": t_oos, "ic_oos": ic_oos,
                    "window": window, "thr": thr,
                    "T": T_final, "df": df_sig,
                    "state_counts": state_counts,
                }

df_res = pd.DataFrame(all_rows)
df_res.to_csv(RESULTS / "markov_ic_study.csv", index=False)

# ── Summary tables ────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("OOS SURVIVORS  |t-stat| > 2")
print(f"{'='*70}")
surv = df_res[df_res["t_oos"].abs() > 2].sort_values(
    "t_oos", key=abs, ascending=False)
if surv.empty:
    print("  None")
else:
    cols = ["pair","window","thr","ic_is","t_is","ic_oos","t_oos","acc_oos","stickiness"]
    print(surv[cols].to_string(index=False, float_format="{:+.3f}".format))

print(f"\n{'='*70}")
print("STICKINESS SCORES — top 15 by stickiness")
print(f"{'='*70}")
top_s = df_res.nlargest(15, "stickiness")[
    ["pair","window","thr","stickiness","bull_pct","side_pct","bear_pct"]]
print(top_s.to_string(index=False, float_format="{:.3f}".format))

print(f"\n{'='*70}")
print("BEST TRANSITION MATRIX PER PAIR")
print(f"{'='*70}")
for pair, cache in best_T_cache.items():
    T   = cache["T"]
    # row-normalise
    row_sums = T.sum(axis=1, keepdims=True).clip(1e-9)
    Tn  = T / row_sums
    print(f"\n  {pair}  win={cache['window']} thr={cache['thr']:.3f}  "
          f"IC_OOS={cache['ic_oos']:+.3f}  t={cache['t_oos']:+.1f}")
    print(f"  State dist:  "
          f"Bull={cache['state_counts'][BULL]*100:.0f}%  "
          f"Side={cache['state_counts'][SIDE]*100:.0f}%  "
          f"Bear={cache['state_counts'][BEAR]*100:.0f}%")
    print(f"  {'':12s}  → Bull   → Side   → Bear")
    for s in range(3):
        print(f"  {SNAMES[s]:5s}      |  {Tn[s,BULL]:.3f}   {Tn[s,SIDE]:.3f}   {Tn[s,BEAR]:.3f}")

print(f"\n{'='*70}")
print("SIGNAL STRENGTH BREAKDOWN (OOS, best config per pair)")
print(f"('strong' = |signal| > 0.25)")
print(f"{'='*70}")
for pair, cache in best_T_cache.items():
    oos = cache["df"][cache["df"]["is_oos"] == "OOS"]
    if oos.empty:
        continue
    strong = oos[oos["signal"].abs() > 0.25]
    weak   = oos[oos["signal"].abs() <= 0.25]
    if len(strong) > 5:
        ic_s, t_s, _ = ic_stats(strong["signal"].values, strong["ret"].values)
    else:
        ic_s = t_s = np.nan
    if len(weak) > 5:
        ic_w, t_w, _ = ic_stats(weak["signal"].values, weak["ret"].values)
    else:
        ic_w = t_w = np.nan
    print(f"  {pair}  "
          f"strong n={len(strong):3d} IC={ic_s:+.3f} t={t_s:+.1f}  "
          f"weak n={len(weak):3d} IC={ic_w:+.3f} t={t_w:+.1f}")

print(f"\nResults saved → {RESULTS / 'markov_ic_study.csv'}")
