#!/usr/bin/env python3
"""
Portfolio Ranking — MC Validation + Correlation + Weighting
============================================================
Loads results from the three sweep scripts, runs MC validation on
WF-passing configs, computes pairwise signal correlation to find
low-correlation combinations, and outputs a ranked portfolio.

Steps:
  1. Load tp_frontier.csv + variant_sweep.csv + exhaust_cont.csv
  2. Identify candidate configs (n_is3 >= 10 and oos_pd > 0 at portfolio level)
  3. Re-simulate OOS for each candidate → MC sign-shuffle (2000 shuffles)
  4. Gate: mc_p < 0.05 per pair, portfolio mc_p < 0.05
  5. Compute signal correlation (fraction of overlapping active bars)
  6. Greedy portfolio selection: pick N lowest-correlation MC-passing configs
  7. Portfolio MC on selected combination
  8. Output: results/portfolio_candidates.csv + results/portfolio_final.csv

Usage: python portfolio_rank.py [--max-variants N]  (default 12)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")
import argparse, sys

PROJECT  = Path(__file__).resolve().parents[3]
DATA     = PROJECT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC  = 0.70
MC_N     = 2000
SMA_N_EXHAUST = 14

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


@njit(cache=True)
def sim_tp(mid, bid, ask, sp, sig, tp_pips, sp_gate):
    n = len(mid); pnl_sum = 0.0; n_trades = 0; n_wins = 0
    in_trade = False; dir_ = 0; ep = 0.0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) * dir_ >= tp_pips:
                ex = bid[i] if dir_ == 1 else ask[i]
                p  = (ex - ep) * dir_ - sp[i]
                pnl_sum += p; n_trades += 1
                if p > 0.0: n_wins += 1
                in_trade = False
        else:
            nd = sig[i-1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_trade = True
    return pnl_sum, n_trades, n_wins


@njit(parallel=True, cache=True)
def mc_sign_shuffle(pnls, observed_ppd, oos_days, n_shuffles):
    beats = 0
    for _ in prange(n_shuffles):
        total = 0.0
        for j in range(len(pnls)):
            sign = 1.0 if np.random.random() > 0.5 else -1.0
            total += pnls[j] * sign
        if oos_days > 0 and (total / oos_days) >= observed_ppd:
            beats += 1
    return beats / n_shuffles


@njit(cache=True)
def compute_sma(close, period):
    n = len(close); out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(close[i - period + 1 : i + 1])
    return out


@njit(cache=True)
def build_exhaust_sig_nb(close_p, open_p, sma, n_consec, dist_mult, sp_gate):
    n = len(close_p); sig = np.zeros(n, dtype=np.int8)
    for i in range(n_consec - 1, n - 1):
        if np.isnan(sma[i]): continue
        all_bull = True; all_bear = True
        for j in range(i - n_consec + 1, i + 1):
            if close_p[j] <= open_p[j]: all_bull = False
            if close_p[j] >= open_p[j]: all_bear = False
        dist = close_p[i] - sma[i]
        if all_bull and dist >= dist_mult * sp_gate: sig[i] = 1
        elif all_bear and (-dist) >= dist_mult * sp_gate: sig[i] = -1
    return sig


def build_momentum_sig(df, tf1, tf2, sma_n, lags):
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        if sma_n > 0:
            rs = rs.rolling(sma_n, min_periods=sma_n).mean()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


TF_LABEL_MAP = {
    "M15+M5":  ("15min", "5min"),
    "M30+M15": ("30min", "15min"),
    "H1+M30":  ("1h",    "30min"),
    "H4+H1":   ("4h",    "1h"),
}

SMA_LABEL_MAP = {
    "pmom":  0,
    "sma8":  8, "sma12": 12, "sma16": 16, "sma22": 22,
}


def get_sig_for_row(row, df, pip, sp_gate):
    """Reconstruct signal array for a candidate row."""
    sig_type = row["signal"]

    if sig_type == "exhaust_cont":
        close_p = (df["close"].values / pip).astype(np.float64)
        open_p  = (df["open"].values  / pip).astype(np.float64) if "open" in df.columns \
                  else close_p.copy()
        sma = compute_sma(close_p, SMA_N_EXHAUST)
        sig = build_exhaust_sig_nb(close_p, open_p, sma,
                                   int(row["n_consec"]),
                                   float(row["dist_mult"]), sp_gate)
        return sig

    # momentum variant
    if sig_type in ("pmom_m15m5", "sma16_h1m30"):
        # frontier configs
        if sig_type == "pmom_m15m5":
            tf1, tf2, sma_n, lags = "15min", "5min", 0, (1,3,8)
        else:
            tf1, tf2, sma_n, lags = "1h", "30min", 16, (8,10,15)
    else:
        # variant_sweep configs
        tfs = row.get("tfs", row.get("tf", "M15+M5"))
        tf1, tf2 = TF_LABEL_MAP.get(tfs, ("15min","5min"))
        sma_n = SMA_LABEL_MAP.get(sig_type, 0)
        lags = tuple(int(x) for x in row["lags"].strip("()").split(","))

    tp = float(row["tp"])
    sig_ser = build_momentum_sig(df, tf1, tf2, sma_n, lags)
    return sig_ser.values.astype(np.int8), tp


def validate_candidate(row, pair_cache):
    """Run MC on a candidate config across all pairs. Returns mc_p per pair + portfolio."""
    sig_type = row["signal"]
    all_pnls = []; pair_results = []

    for pair in PAIRS:
        c = pair_cache[pair]
        df = c["df"]; pip = c["pip"]; sg = c["sp_gate"]; n_is = c["n_is"]

        if sig_type == "exhaust_cont":
            close_p = (df["close"].values / pip).astype(np.float64)
            open_p  = (df["open"].values  / pip).astype(np.float64) if "open" in df.columns \
                      else close_p.copy()
            sma = compute_sma(close_p, SMA_N_EXHAUST)
            sig = build_exhaust_sig_nb(close_p, open_p, sma,
                                       int(row["n_consec"]), float(row["dist_mult"]), sg)
            tp = float(row["tp"])
        else:
            if sig_type in ("pmom_m15m5", "sma16_h1m30"):
                if sig_type == "pmom_m15m5":
                    tf1, tf2, sma_n, lags = "15min", "5min", 0, (1,3,8)
                else:
                    tf1, tf2, sma_n, lags = "1h", "30min", 16, (8,10,15)
            else:
                tfs = row.get("tfs", row.get("tf", "M15+M5"))
                tf1, tf2 = TF_LABEL_MAP.get(tfs, ("15min","5min"))
                sma_n = SMA_LABEL_MAP.get(sig_type, 0)
                lags  = tuple(int(x) for x in row["lags"].strip("()").split(","))
            sig_ser = build_momentum_sig(df, tf1, tf2, sma_n, lags)
            sig = sig_ser.values.astype(np.int8)
            tp  = float(row["tp"])

        mid = (df["close"].values / pip).astype(np.float64)
        bid = (df["bid_c"].values / pip).astype(np.float64)
        ask = (df["ask_c"].values / pip).astype(np.float64)
        sp  = (ask - bid)

        p_sum, n_t, n_w = sim_tp(mid[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
                                   sig[n_is:], tp, sg)
        oos_days = c["oos_days"]
        oos_pd = p_sum / oos_days if oos_days > 0 else 0.0

        if n_t > 0:
            pnls_arr = np.array([oos_pd / n_t] * n_t, dtype=np.float64)
            mc_p = float(mc_sign_shuffle(pnls_arr * n_t, p_sum / oos_days, oos_days, MC_N))
        else:
            mc_p = 1.0

        all_pnls.extend([p_sum / n_t if n_t > 0 else 0.0] * n_t)
        pair_results.append(dict(pair=pair, oos_pd=oos_pd, n_t=n_t, mc_p=mc_p))

    # Portfolio MC
    if all_pnls:
        all_arr = np.array(all_pnls, dtype=np.float64)
        total_days = sum(c["oos_days"] for c in pair_cache.values()) / len(PAIRS)
        port_pd = sum(r["oos_pd"] for r in pair_results)
        port_mc_p = float(mc_sign_shuffle(all_arr, port_pd / len(PAIRS), total_days, MC_N))
    else:
        port_mc_p = 1.0

    return pair_results, port_mc_p


def signal_overlap(sig1, sig2):
    """Fraction of bars where both signals are active in same direction."""
    active1 = (sig1 != 0); active2 = (sig2 != 0)
    both = active1 & active2
    if both.sum() == 0: return 0.0
    agree = (sig1[both] == sig2[both]).mean()
    return float(agree)


# ── Load sweep results ────────────────────────────────────────────────────────
print("Loading sweep results …")
dfs = []
for fname in ["tp_frontier.csv", "variant_sweep.csv", "exhaust_cont.csv"]:
    path = RESULTS / fname
    if path.exists():
        dfs.append(pd.read_csv(path))
        print(f"  Loaded {fname}: {len(dfs[-1])} rows")
    else:
        print(f"  MISSING: {fname} — skipping")
if not dfs:
    print("No results found. Run the sweep scripts first."); sys.exit(1)

all_results = pd.concat(dfs, ignore_index=True)

# ── Identify portfolio-level candidates from variant_sweep ───────────────────
# For variant_sweep: aggregate is at portfolio level already
# For tp_frontier and exhaust_cont: aggregate per config across pairs
print("\nAggregating to portfolio level …")

def get_config_key(row):
    sig = row.get("signal","")
    tp  = row.get("tp", 0)
    if sig == "exhaust_cont":
        return (sig, row.get("n_consec",""), row.get("dist_mult",""), tp)
    tfs = row.get("tfs", row.get("tf",""))
    lags = row.get("lags","")
    sma_n = row.get("sma_n", 0)
    return (sig, tfs, sma_n, lags, tp)

# For variant_sweep: candidates where n_is3 >= 10 and portfolio_pd > 0
vs_candidates = []
if (RESULTS / "variant_sweep.csv").exists():
    vs = pd.read_csv(RESULTS / "variant_sweep.csv")
    vs_candidates = vs[(vs["n_is3"] >= 10) & (vs["portfolio_pd"] > 0)].copy()
    print(f"  variant_sweep candidates (n_is3>=10, port_pd>0): {len(vs_candidates)}")

# For tp_frontier: aggregate by signal+tp
tf_candidates = []
if (RESULTS / "tp_frontier.csv").exists():
    tf = pd.read_csv(RESULTS / "tp_frontier.csv")
    tf_agg = tf.groupby(["signal","tp"]).agg(
        n_is3=("wf_pass","sum"), portfolio_pd=("oos_pd","sum")
    ).reset_index()
    tf_candidates = tf_agg[(tf_agg["n_is3"] >= 10) & (tf_agg["portfolio_pd"] > 0)].copy()
    print(f"  tp_frontier candidates: {len(tf_candidates)}")

# For exhaust_cont: aggregate by n_consec+dist_mult+tp
ec_candidates = []
if (RESULTS / "exhaust_cont.csv").exists():
    ec = pd.read_csv(RESULTS / "exhaust_cont.csv")
    ec_agg = ec.groupby(["n_consec","dist_mult","tp"]).agg(
        n_is3=("wf_pass","sum"), portfolio_pd=("oos_pd","sum")
    ).reset_index()
    ec_agg["signal"] = "exhaust_cont"
    ec_candidates = ec_agg[(ec_agg["n_is3"] >= 10) & (ec_agg["portfolio_pd"] > 0)].copy()
    print(f"  exhaust_cont candidates: {len(ec_candidates)}")

# ── Load pair data for MC and signal computation ──────────────────────────────
print("\nLoading pair data for MC validation …")
pair_cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    pair_cache[pair] = dict(df=df, pip=pip, sp_gate=sp_gate, n_is=n_is,
                             oos_days=len(df.iloc[n_is:]) / 288.0)

# ── MC validation on all candidates ──────────────────────────────────────────
print("\nRunning MC validation on candidates …")

def mc_validate_df(candidates_df, extra_cols=None):
    """Run MC on rows of candidates_df. Returns rows with mc results."""
    results = []
    for i, row in candidates_df.iterrows():
        pair_res, port_mc = validate_candidate(row, pair_cache)
        mc_pass_count = sum(1 for r in pair_res if r["mc_p"] < 0.05)
        mean_mc_p = float(np.mean([r["mc_p"] for r in pair_res]))
        row_dict = dict(row)
        row_dict.update(dict(mc_pairs_pass=mc_pass_count, mean_mc_p=round(mean_mc_p,4),
                              port_mc_p=round(port_mc,4),
                              mc_pass=int(port_mc < 0.05 and mc_pass_count >= 10)))
        results.append(row_dict)
        print(f"    {row.get('signal','')} tp={row.get('tp','')}  "
              f"port_pd={row.get('portfolio_pd',0):+.1f}  "
              f"mc_pairs={mc_pass_count}/12  port_mc_p={port_mc:.3f}", flush=True)
    return pd.DataFrame(results)

mc_rows = []

if len(vs_candidates) > 0:
    top_vs = vs_candidates.sort_values("portfolio_pd", ascending=False).head(50)
    print(f"  Validating top 50 variant_sweep candidates (of {len(vs_candidates)}) …")
    mc_rows.append(mc_validate_df(top_vs))

if len(tf_candidates) > 0:
    print(f"  Validating {len(tf_candidates)} tp_frontier candidates …")
    mc_rows.append(mc_validate_df(tf_candidates))

if len(ec_candidates) > 0:
    print(f"  Validating {len(ec_candidates)} exhaust_cont candidates …")
    mc_rows.append(mc_validate_df(ec_candidates))

if not mc_rows:
    print("No candidates to validate. Exiting."); sys.exit(0)

mc_all = pd.concat(mc_rows, ignore_index=True)
mc_all.to_csv(RESULTS / "portfolio_candidates.csv", index=False)
mc_passed = mc_all[mc_all["mc_pass"] == 1].sort_values("portfolio_pd", ascending=False)
print(f"\nMC-passing configs: {len(mc_passed)}")
_show_cols = [c for c in ["signal","tfs","sma_n","lags","n_consec","dist_mult","tp",
                            "n_is3","portfolio_pd","mc_pairs_pass","port_mc_p"]
              if c in mc_passed.columns]
print(mc_passed[_show_cols].head(20).to_string(index=False))

# ── Greedy decorrelated selection ─────────────────────────────────────────────
parser = argparse.ArgumentParser(); parser.add_argument("--max-variants", type=int, default=12)
args, _ = parser.parse_known_args()

if len(mc_passed) == 0:
    print("\nNo MC-passing configs. Portfolio selection skipped."); sys.exit(0)

print(f"\nBuilding portfolio of up to {args.max_variants} decorrelated variants …")

# Compute per-pair OOS signal arrays for correlation measurement
def get_oos_signal(row, pair, pair_cache):
    c = pair_cache[pair]; df = c["df"]; pip = c["pip"]; sg = c["sp_gate"]; n_is = c["n_is"]
    sig_type = row["signal"]
    if sig_type == "exhaust_cont":
        close_p = (df["close"].values / pip).astype(np.float64)
        open_p  = (df["open"].values  / pip).astype(np.float64) if "open" in df.columns \
                  else close_p.copy()
        sma = compute_sma(close_p, SMA_N_EXHAUST)
        sig = build_exhaust_sig_nb(close_p, open_p, sma,
                                   int(row["n_consec"]), float(row["dist_mult"]), sg)
        return sig[n_is:]
    if sig_type in ("pmom_m15m5", "sma16_h1m30"):
        tf1, tf2 = ("15min","5min") if "pmom" in sig_type else ("1h","30min")
        sma_n    = 0 if "pmom" in sig_type else 16
        lags     = (1,3,8) if "pmom" in sig_type else (8,10,15)
    else:
        tfs = row.get("tfs", row.get("tf","M15+M5"))
        tf1, tf2 = TF_LABEL_MAP.get(tfs, ("15min","5min"))
        sma_n = SMA_LABEL_MAP.get(sig_type, 0)
        lags = tuple(int(x) for x in row["lags"].strip("()").split(","))
    sig_ser = build_momentum_sig(df, tf1, tf2, sma_n, lags)
    return sig_ser.values[n_is:].astype(np.int8)

# Compute aggregate signal vectors (concatenate over all pairs)
print("  Computing signal arrays for correlation …")
sig_vectors = {}
for idx, row in mc_passed.iterrows():
    key = idx
    vec = np.concatenate([get_oos_signal(row, p, pair_cache) for p in PAIRS])
    sig_vectors[key] = vec

# Greedy selection: pick highest p/d first, then add next highest that is < 0.6 corr with all selected
selected_idxs = []
remaining = list(mc_passed.index)

while len(selected_idxs) < args.max_variants and remaining:
    best = remaining[0]
    if not selected_idxs:
        selected_idxs.append(best); remaining.remove(best); continue
    # Check correlation with all selected
    max_corr = max(abs(np.corrcoef(sig_vectors[best], sig_vectors[s])[0,1])
                   for s in selected_idxs)
    if max_corr < 0.70:
        selected_idxs.append(best); remaining.remove(best)
    else:
        remaining.remove(best)

portfolio = mc_passed.loc[selected_idxs].copy()
print(f"\nSelected {len(portfolio)} variants (max pairwise corr < 0.70):")
_show_cols2 = [c for c in ["signal","tfs","sma_n","lags","n_consec","dist_mult","tp",
                             "n_is3","portfolio_pd","port_mc_p"]
               if c in portfolio.columns]
print(portfolio[_show_cols2].to_string(index=False))

# ── Portfolio-level MC ────────────────────────────────────────────────────────
print("\nPortfolio-level MC (combined trades from all variants) …")
all_combined_pnls = []
total_port_pd = 0.0
for _, row in portfolio.iterrows():
    for pair in PAIRS:
        c = pair_cache[pair]; df = c["df"]; pip = c["pip"]; sg = c["sp_gate"]; n_is = c["n_is"]
        sig_type = row["signal"]
        if sig_type == "exhaust_cont":
            close_p = (df["close"].values / pip).astype(np.float64)
            open_p  = (df["open"].values  / pip).astype(np.float64) if "open" in df.columns \
                      else close_p.copy()
            sma = compute_sma(close_p, SMA_N_EXHAUST)
            sig = build_exhaust_sig_nb(close_p, open_p, sma,
                                       int(row["n_consec"]), float(row["dist_mult"]), sg)
            tp = float(row["tp"])
        else:
            if sig_type in ("pmom_m15m5","sma16_h1m30"):
                tf1, tf2 = ("15min","5min") if "pmom" in sig_type else ("1h","30min")
                sma_n = 0 if "pmom" in sig_type else 16
                lags  = (1,3,8) if "pmom" in sig_type else (8,10,15)
            else:
                tfs = row.get("tfs","M15+M5"); tf1, tf2 = TF_LABEL_MAP.get(tfs,("15min","5min"))
                sma_n = SMA_LABEL_MAP.get(sig_type, 0)
                lags  = tuple(int(x) for x in row["lags"].strip("()").split(","))
            sig_ser = build_momentum_sig(df, tf1, tf2, sma_n, lags)
            sig = sig_ser.values.astype(np.int8); tp = float(row["tp"])

        mid = (df["close"].values / pip).astype(np.float64)
        bid = (df["bid_c"].values / pip).astype(np.float64)
        ask = (df["ask_c"].values / pip).astype(np.float64)
        sp  = (ask - bid)
        p_sum, n_t, _ = sim_tp(mid[n_is:], bid[n_is:], ask[n_is:], sp[n_is:], sig[n_is:], tp, sg)
        total_port_pd += p_sum / c["oos_days"] if c["oos_days"] > 0 else 0.0
        if n_t > 0:
            avg_pnl = p_sum / n_t
            all_combined_pnls.extend([avg_pnl] * n_t)

avg_oos_days = np.mean([c["oos_days"] for c in pair_cache.values()])
if all_combined_pnls:
    arr = np.array(all_combined_pnls, dtype=np.float64)
    combo_mc_p = float(mc_sign_shuffle(arr, total_port_pd / len(PAIRS), avg_oos_days, MC_N))
else:
    combo_mc_p = 1.0

print(f"\nPortfolio combined p/d: {total_port_pd:+.1f}")
print(f"Portfolio MC p-value:   {combo_mc_p:.4f}  {'PASS ✓' if combo_mc_p < 0.05 else 'FAIL ✗'}")

portfolio["portfolio_rank"] = range(1, len(portfolio)+1)
portfolio.to_csv(RESULTS / "portfolio_final.csv", index=False)
print(f"\nSaved final portfolio → results/portfolio_final.csv")

print("\n" + "="*70)
print("FINAL PORTFOLIO SUMMARY")
print("="*70)
print(f"  Variants selected : {len(portfolio)}")
print(f"  Combined OOS p/d  : {total_port_pd:+.1f} pips/day")
print(f"  Portfolio MC p    : {combo_mc_p:.4f}")
print()
for rank, (_, row) in enumerate(portfolio.iterrows(), 1):
    sig = row.get("signal",""); tfs = row.get("tfs",""); lags = row.get("lags","")
    tp = row.get("tp",""); ppd = row.get("portfolio_pd", 0)
    print(f"  {rank:2d}. {sig:<20} {tfs:<10} lags={lags:<18} TP={tp}p  portf_pd={ppd:+.1f}")
