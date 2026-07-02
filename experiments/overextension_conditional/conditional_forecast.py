#!/usr/bin/env python3
"""
CONDITIONAL FORECAST after an over-extension event.

Question (predictability/confidence, NOT an edge hunt):
  Given a bar that closes FULLY OUTSIDE the SMA9 +/- K*sigma band (over-extension
  event), observed NOW, plus its early reaction over the next 1-2 bars, how much
  does that narrow -- or shift -- the distribution of where price settles 6 bars
  ahead, vs the UNCONDITIONAL 6-bar baseline?

Definitions
  signed_TR (per bar) = TrueRange * sign(close - open)
      down-bar with 5p range = -5p ; up-bar with 5p range = +5p.
  SMA9 of close, rolling std9 of close (population). Band = SMA9 +/- K*std9.
  EVENT at bar i: close_i fully outside band, i.e.
      up-protrusion:   close_i > SMA9_i + K_lo*std9_i  (and low_i > upper?  NO --
                       "fully outside" = the whole BAR (low for up / high for down)
                       is outside the band, not just the close).
      We require the entire bar body+wick on the protrusion side to be outside:
        up:   low_i  > SMA9_i + K*std9_i
        down: high_i < SMA9_i - K*std9_i
      K is the sigma-level the bar REACHES; we bucket by how far close_i protrudes:
        z_i = (close_i - SMA9_i) / std9_i ; |z| in {1.0-1.5,1.5-2,2-3,3+}.
      Trigger floor: must be fully-outside at K>=1.0.

  EARLY REACTION over bars i+1..i+2, signed relative to protrusion direction:
      r = sum(signed_TR[i+1], signed_TR[i+2]) projected onto protrusion sign.
      proj = sign(protrusion) * (signed_TR_{i+1} + signed_TR_{i+2})
      CONTINUATION : proj > +flat_thr   (price keeps extending in protrusion dir)
      REVERSION    : proj < -flat_thr   (price pulls back)
      FLAT         : |proj| <= flat_thr
      flat_thr is set per (pair,TF) as a small fraction of typical signed_TR
      magnitude (0.25 * median |signed_TR|) so "flat" is genuinely small.

  TARGET: 6-bar-ahead return = (close_{i+6} - close_i) in pips, SIGNED in price terms
      (positive = price up). We ALSO report it projected onto protrusion direction
      (settle_proj = sign(protrusion)*ret6) so "reversion" reads as negative settle.

Measured per cell [sigma-bucket x direction x early-reaction], pooled across 12
pairs (pips are pair-comparable), per TF in {M5,M15,H1}:
  1. conditional mean of ret6 (and projected) -- directional tilt vs ~spread
  2. conditional std + IQR of ret6, and RATIO to UNCONDITIONAL 6-bar std
     (unconditional = std of ALL 6-bar returns on that pair/TF, pooled)
  3. R^2 / variance-explained of ret6 by [event sigma-bucket + direction + reaction]
  4. gates (Newey-West t on projected settle, walk-forward sign consistency over
     3 temporal thirds, MC sign-shuffle p) on each cell.

Spread: real per-bar spread = (ask_c - bid_c)/pip on the close bar. We report
median spread per pair/TF so "mean vs spread" is concrete.

Output: prints table, writes .out (redirect), writes JSON of the cell table.
"""

import sys, os, json, time, gc, argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "s5_ohlc"
OUT_DIR = Path(__file__).resolve().parent

PAIRS = ["AUD_JPY","AUD_USD","CAD_JPY","CHF_JPY","EUR_GBP","EUR_JPY",
         "EUR_USD","GBP_JPY","GBP_USD","NZD_JPY","NZD_USD","USD_JPY"]

TF_RULE = {"M5": "5min", "M15": "15min", "H1": "1h"}

SMA_N = 9
HORIZON = 6          # bars ahead for the settle target
REACT_BARS = 2       # early-reaction window i+1..i+2
SIGMA_EDGES = [1.0, 1.5, 2.0, 3.0, np.inf]
SIGMA_LABELS = ["1.0-1.5", "1.5-2", "2-3", "3+"]


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def load_tf(pair: str, tf: str) -> pd.DataFrame:
    """Load S5 BA parquet, resample to TF. mid OHLC + close-bar spread in pips."""
    f = DATA_DIR / f"{pair}_S5_BA.parquet"
    df = pd.read_parquet(f, engine="pyarrow",
                         columns=["timestamp","open","high","low","close","bid_c","ask_c"])
    ts = pd.to_datetime(df["timestamp"])
    spread_pip = (df["ask_c"].values - df["bid_c"].values) / pip_size(pair)
    s5 = pd.DataFrame({
        "open": df["open"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low":  df["low"].values.astype(np.float64),
        "close":df["close"].values.astype(np.float64),
        "spread": spread_pip.astype(np.float64),
    }, index=ts)
    del df; gc.collect()
    rule = TF_RULE[tf]
    agg = s5.resample(rule).agg({
        "open":"first","high":"max","low":"min","close":"last","spread":"last"
    }).dropna()
    return agg


def compute_events(df: pd.DataFrame, pair: str):
    """Return per-bar arrays + event mask for one pair/TF.

    Returns dict with:
      ret6        : (close_{i+6}-close_i)/pip   signed price return, NaN at tail
      uncond_std  : std of ret6 over all valid bars (scalar)
      signed_tr   : signed TrueRange per bar in pips
      z           : (close - sma9)/std9
      fully_up    : bool, whole bar above upper band
      fully_dn    : bool, whole bar below lower band
      proj_react  : projected early reaction (pips) over i+1..i+2
      spread      : per-bar spread pips
      idx_time    : timestamps (for WF thirds)
      flat_thr    : per-pair/TF flat threshold
    """
    pip = pip_size(pair)
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    n = len(c)

    # True Range in pips
    prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]) / pip
    sign_bar = np.where(c >= o, 1.0, -1.0)
    signed_tr = tr * sign_bar

    # SMA9 + rolling std9 (population) of close, causal (uses bars up to i)
    cser = pd.Series(c)
    sma9 = cser.rolling(SMA_N).mean().values
    std9 = cser.rolling(SMA_N).std(ddof=0).values

    upper = sma9 + std9  # K>=1 floor band edge (1 sigma)
    lower = sma9 - std9
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (c - sma9) / std9

    # fully outside (entire bar on protrusion side, K>=1)
    fully_up = (l > upper) & np.isfinite(upper)
    fully_dn = (h < lower) & np.isfinite(lower)

    # early reaction over i+1..i+2 (signed_tr sum), aligned to bar i
    # react_sum[i] = signed_tr[i+1] + signed_tr[i+2]
    react_sum = np.full(n, np.nan)
    if n > REACT_BARS:
        rs = np.zeros(n - REACT_BARS)
        for k in range(1, REACT_BARS + 1):
            rs += signed_tr[k: k + (n - REACT_BARS)]
        react_sum[:n - REACT_BARS] = rs

    # 6-bar settle return in pips (close_{i+6} - close_i)
    ret6 = np.full(n, np.nan)
    if n > HORIZON:
        ret6[:n - HORIZON] = (c[HORIZON:] - c[:n - HORIZON]) / pip

    # CLEAN forward: from END of reaction window to settle: close_{i+6}-close_{i+2}
    # No overlap between the early-reaction predictor (i+1..i+2) and this target,
    # so it is a genuine out-of-window forecast, not mechanical persistence.
    fwd_clean = np.full(n, np.nan)
    if n > HORIZON:
        fwd_clean[:n - HORIZON] = (c[HORIZON:] - c[REACT_BARS:REACT_BARS + (n - HORIZON)]) / pip

    flat_thr = 0.25 * np.nanmedian(np.abs(signed_tr))

    return {
        "ret6": ret6,
        "fwd_clean": fwd_clean,
        "signed_tr": signed_tr,
        "z": z,
        "fully_up": fully_up,
        "fully_dn": fully_dn,
        "react_sum": react_sum,
        "spread": df["spread"].values,
        "time": df.index.values,
        "flat_thr": flat_thr,
        "n": n,
    }


def newey_west_tstat(x, lag):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return 0.0
    xbar = x.mean()
    gamma0 = np.mean((x - xbar) ** 2)
    var = gamma0
    for j in range(1, lag + 1):
        if j >= n:
            break
        w = 1.0 - j / (lag + 1.0)
        gj = np.mean((x[j:] - xbar) * (x[:-j] - xbar))
        var += 2.0 * w * gj
    var = max(var / n, 1e-20)
    return xbar / np.sqrt(var)


def mc_sign_pvalue(x, n_shuffle=2000, seed=0):
    """MC p-value: prob that random sign-flips of |x| produce a mean as extreme."""
    x = np.asarray(x, float)
    if len(x) < 10:
        return 1.0
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    a = np.abs(x)
    cnt = 0
    for _ in range(n_shuffle):
        s = rng.choice([-1.0, 1.0], size=len(a))
        if abs((a * s).mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (n_shuffle + 1)


def wf_sign_consistency(ret6_proj, times):
    """Split into 3 temporal thirds; return list of mean-sign per third + agree."""
    order = np.argsort(times)
    rp = ret6_proj[order]
    thirds = np.array_split(rp, 3)
    means = [float(t.mean()) if len(t) else 0.0 for t in thirds]
    signs = [np.sign(m) for m in means]
    agree = (abs(sum(signs)) == 3)  # all same nonzero sign
    return means, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="M5,M15,H1")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--json", default=str(OUT_DIR / "conditional_cells.json"))
    args = ap.parse_args()

    tfs = args.tfs.split(",")
    pairs = args.pairs.split(",")
    all_cells = {}

    for tf in tfs:
        print("\n" + "=" * 100)
        print(f"TIMEFRAME {tf}")
        print("=" * 100)

        # Pool across pairs. Collect per-event records.
        recs = []  # dict per event
        # unconditional 6-bar return dispersion via running moments (memory-lean)
        uc_n = 0; uc_sum = 0.0; uc_sumsq = 0.0
        spread_by_pair = {}
        t0 = time.time()
        for pair in pairs:
            df = load_tf(pair, tf)
            E = compute_events(df, pair)
            spread_by_pair[pair] = float(np.nanmedian(E["spread"]))
            ret6 = E["ret6"]; z = E["z"]; fwdc = E["fwd_clean"]
            fu = E["fully_up"]; fdn = E["fully_dn"]
            rs = E["react_sum"]; flat = E["flat_thr"]
            tarr = E["time"]
            n = E["n"]

            vu = ret6[np.isfinite(ret6)]
            uc_n += vu.size; uc_sum += float(vu.sum()); uc_sumsq += float(np.dot(vu, vu))
            del vu

            # event indices: fully outside AND ret6,react finite
            ev = (fu | fdn) & np.isfinite(ret6) & np.isfinite(rs) & np.isfinite(z) & np.isfinite(fwdc)
            idx = np.where(ev)[0]
            for i in idx:
                direction = 1 if fu[i] else -1
                proj_react = direction * rs[i]
                if proj_react > flat:
                    react = "continuation"
                elif proj_react < -flat:
                    react = "reversion"
                else:
                    react = "flat"
                az = abs(z[i])
                sb = None
                for b in range(len(SIGMA_LABELS)):
                    if SIGMA_EDGES[b] <= az < SIGMA_EDGES[b + 1]:
                        sb = SIGMA_LABELS[b]; break
                if sb is None:
                    continue
                recs.append({
                    "pair": pair,
                    "dir": "up" if direction == 1 else "down",
                    "sigma": sb,
                    "react": react,
                    "ret6": float(ret6[i]),
                    "ret6_proj": float(direction * ret6[i]),
                    "fwd_clean_proj": float(direction * fwdc[i]),
                    "spread": float(E["spread"][i]),
                    "t": tarr[i],
                })
            del df, E; gc.collect()

        uc_mean = uc_sum / uc_n
        uncond_std = float(np.sqrt(max(uc_sumsq / uc_n - uc_mean ** 2, 0.0)))
        print(f"loaded {len(pairs)} pairs in {time.time()-t0:.0f}s | "
              f"events={len(recs)} | uncond 6-bar std={uncond_std:.2f}p "
              f"(n={uc_n:,})")
        print(f"median spread/pair (pips): " +
              ", ".join(f"{p}={spread_by_pair[p]:.1f}" for p in pairs))

        rdf = pd.DataFrame(recs)
        if rdf.empty:
            print("NO EVENTS"); continue

        # ---- Unconditional baseline projected (by direction): mean of ret6_proj
        base_proj_mean = float(rdf["ret6_proj"].mean())
        print(f"event-pool projected ret6 mean (all events) = {base_proj_mean:.2f}p, "
              f"std={rdf['ret6_proj'].std():.2f}p")

        # ---- R^2: variance of ret6 explained by [sigma x dir x react] group means
        grp = rdf.groupby(["sigma", "dir", "react"])["ret6"]
        gmean = grp.transform("mean")
        ss_tot = np.sum((rdf["ret6"] - rdf["ret6"].mean()) ** 2)
        ss_res = np.sum((rdf["ret6"] - gmean) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"R^2 (ret6 explained by sigma x dir x react cells) = {r2:.4f}")
        # also direction-projected R^2
        gmp = rdf.groupby(["sigma", "dir", "react"])["ret6_proj"].transform("mean")
        sst = np.sum((rdf["ret6_proj"] - rdf["ret6_proj"].mean()) ** 2)
        ssr = np.sum((rdf["ret6_proj"] - gmp) ** 2)
        r2p = 1.0 - ssr / sst if sst > 0 else 0.0
        print(f"R^2 (ret6_proj explained by cells)            = {r2p:.4f}")

        # ---- per-cell table
        print("\n" + "-" * 118)
        hdr = (f"{'sigma':<8}{'dir':<6}{'react':<13}{'n':>7}"
               f"{'mean_proj':>10}{'std':>9}{'std/uncond':>11}"
               f"{'IQR':>9}{'NWt':>8}{'WFagree':>9}{'MCp':>8}{'med_spd':>9}"
               f"{'cleanFwd':>10}{'cleanNWt':>10}")
        print("NOTE: mean_proj/std/NWt = full 6-bar settle from event close (predictor"
              " bars i+1..i+2 OVERLAP the target -> persistence partly mechanical).")
        print("      cleanFwd = projected close_{i+6}-close_{i+2}: NO overlap with the"
              " early-reaction predictor (genuine forecast).")
        print(hdr); print("-" * 118)

        cell_records = []
        lag = HORIZON  # overlapping 6-bar returns -> NW lag = horizon
        for (sb, dr, rc), sub in rdf.groupby(["sigma", "dir", "react"]):
            nn = len(sub)
            rp = sub["ret6_proj"].values
            cfp = sub["fwd_clean_proj"].values
            std = float(rp.std())
            iqr = float(np.percentile(rp, 75) - np.percentile(rp, 25))
            cf_mean = float(cfp.mean())
            if nn < 30:
                print(f"{sb:<8}{dr:<6}{rc:<13}{nn:>7}{rp.mean():>10.2f}"
                      f"{std:>9.2f}{std/uncond_std:>11.2f}{iqr:>9.2f}"
                      f"{'-':>8}{'-':>9}{'-':>8}{sub['spread'].median():>9.1f}"
                      f"{cf_mean:>10.2f}{'-':>10}")
                cell_records.append({
                    "sigma": sb, "dir": dr, "react": rc, "n": nn,
                    "mean_proj": float(rp.mean()), "std": std,
                    "std_ratio": std / uncond_std, "iqr": iqr,
                    "nw_t": None, "wf_agree": None, "mc_p": None,
                    "clean_fwd_mean": cf_mean, "clean_fwd_nwt": None,
                    "med_spread": float(sub["spread"].median()),
                })
                continue
            nwt = newey_west_tstat(rp, lag)
            cf_nwt = newey_west_tstat(cfp, HORIZON - REACT_BARS)
            _, agree = wf_sign_consistency(rp, sub["t"].values)
            mcp = mc_sign_pvalue(rp, n_shuffle=args.mc, seed=hash((sb, dr, rc)) % 2**31)
            print(f"{sb:<8}{dr:<6}{rc:<13}{nn:>7}{rp.mean():>10.2f}"
                  f"{std:>9.2f}{std/uncond_std:>11.2f}{iqr:>9.2f}"
                  f"{nwt:>8.2f}{str(agree):>9}{mcp:>8.4f}{sub['spread'].median():>9.1f}"
                  f"{cf_mean:>10.2f}{cf_nwt:>10.2f}")
            cell_records.append({
                "sigma": sb, "dir": dr, "react": rc, "n": nn,
                "mean_proj": float(rp.mean()), "std": std,
                "std_ratio": std / uncond_std, "iqr": iqr,
                "nw_t": float(nwt), "wf_agree": bool(agree), "mc_p": float(mcp),
                "clean_fwd_mean": cf_mean, "clean_fwd_nwt": float(cf_nwt),
                "med_spread": float(sub["spread"].median()),
            })
        print("-" * 118)

        # ---- aggregate by reaction only (marginal): is the carve real?
        print("\nMarginal by early-reaction (pooled over sigma+dir):")
        for rc, sub in rdf.groupby("react"):
            rp = sub["ret6_proj"].values
            cfp = sub["fwd_clean_proj"].values
            print(f"  {rc:<13} n={len(sub):>7} mean_proj={rp.mean():>7.2f}p "
                  f"std={rp.std():>6.2f}p std/uncond={rp.std()/uncond_std:>5.2f} "
                  f"NWt={newey_west_tstat(rp,lag):>6.2f} | "
                  f"cleanFwd={cfp.mean():>7.2f}p cleanNWt={newey_west_tstat(cfp,HORIZON-REACT_BARS):>6.2f}")
        print("\nMarginal by sigma-bucket:")
        for sb, sub in rdf.groupby("sigma"):
            rp = sub["ret6_proj"].values
            print(f"  {sb:<8} n={len(sub):>7} mean_proj={rp.mean():>7.2f}p "
                  f"std={rp.std():>6.2f}p std/uncond={rp.std()/uncond_std:>5.2f}")

        all_cells[tf] = {
            "uncond_std": uncond_std,
            "n_events": len(rdf),
            "r2_ret6": r2, "r2_proj": r2p,
            "base_proj_mean": base_proj_mean,
            "spread_by_pair": spread_by_pair,
            "cells": cell_records,
        }
        del rdf; gc.collect()
        # incremental write: never lose a completed TF on a late crash
        with open(args.json, "w") as f:
            json.dump(all_cells, f, indent=2, default=str)
        print(f"  [wrote {tf} to {args.json}]")

    print(f"\nDONE. wrote {args.json}")


if __name__ == "__main__":
    main()
