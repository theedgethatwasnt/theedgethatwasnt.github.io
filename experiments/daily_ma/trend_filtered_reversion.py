#!/usr/bin/env python3
"""
trend_filtered_reversion.py — CLEAN bar-based reversion fade with a TREND FILTER.

Question: does filtering reversion fades to "WITH the prevailing trend" produce an edge that
GENERALIZES across diverse pairs and survives trend reversals — or is it single-pair drift?

Mirrors the FIXED lib/bb_fade.py re-entry logic but uses a REAL SYMMETRIC explicit stop
(stop_pips from entry), NOT a protrusion-peak. Both LONG and SHORT enabled, reported separately.

Signal (SOP R1-R6, causal, mid OHLC, real per-bar spread deducted explicitly):
  - basis = SMA(close, SMA_N); bands = basis +/- K*std(close, SMA_N).
  - close_beyond trigger: prev bar fully outside the band; this bar re-touches it.
  - Entry fills at NEXT bar open.
  - REAL SYMMETRIC explicit stop: long -> entry - stop_pips; short -> entry + stop_pips.
  - Target = opposite band. Time cap = tcap bars.
  - Entry-validity gate: skip if fill already past stop (long open<=stop; short open>=stop).
  - SELF-CHECK: every exit price within the bar's [low,high]. Abort loudly on violation.
  - Spread: REAL per-bar (ask_c-bid_c)/pip, deducted explicitly at the ENTRY bar.

Trend filter (causal): price vs SMA(win), and sign of win-bar return. "WITH trend" = fade
direction agrees. uptrend->LONG fades only; downtrend->SHORT fades only.
Books: 0=unfiltered, 1=with-trend, 2=against-trend.
WF >=4 folds spanning full multi-year history (contains reversals). 12 pairs + portfolio.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from numba import njit

PAIRS = ["EUR_USD","GBP_USD","AUD_USD","NZD_USD","EUR_GBP","USD_JPY",
         "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY"]
DATA = "data/s5_ohlc"
SMA_N = 9
K = 1.0
MEAT = 1.0
N_FOLDS = 5

def pip_of(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001

def resample(df, rule):
    df = df.set_index("timestamp")
    agg = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low":  df["low"].resample(rule).min(),
        "close":df["close"].resample(rule).last(),
        "bid":  df["bid_c"].resample(rule).last(),
        "ask":  df["ask_c"].resample(rule).last(),
    }).dropna()
    return agg

def compute_bands(c, n=SMA_N, k=K):
    s = pd.Series(c)
    basis = s.rolling(n).mean().values
    sd = s.rolling(n).std(ddof=0).values
    return basis, basis + k*sd, basis - k*sd

def trend_signal(c, kind, win):
    c = np.asarray(c, float); n = len(c); t = np.zeros(n, np.int64)
    if kind == "sma":
        sm = pd.Series(c).rolling(win).mean().values
        m = ~np.isnan(sm)
        t[m] = np.sign(c[m] - sm[m]).astype(np.int64)
    elif kind == "ret":
        d = np.full(n, np.nan); d[win:] = c[win:] - c[:-win]
        m = ~np.isnan(d); t[m] = np.sign(d[m]).astype(np.int64)
    return t


@njit(cache=False)
def _bt(o,h,l,c, up,lo,basis, trend, spread_arr, pip, meat, stop_pips, tcap, book):
    n = len(c)
    e_bar = np.empty(n, np.int64); e_dir = np.empty(n, np.int64)
    e_pnl = np.empty(n, np.float64); e_xbar = np.empty(n, np.int64)
    nt = 0; violations = 0; skipped = 0
    pos = 0; ei = 0; ent = 0.0; stp = 0.0; ent_spread = 0.0
    for i in range(SMA_N, n-1):
        if np.isnan(basis[i]) or np.isnan(basis[i-1]):
            continue
        if pos != 0:
            ex = np.nan
            # Gap handling: if the bar OPENS already beyond a trigger level, the realistic fill is the
            # bar's OPEN (price gapped through, you fill at open, not at the un-reachable level). This is
            # the worst-case honest fill and keeps every exit price inside [low,high] of the exit bar.
            op = o[i]
            if pos == -1:
                # short: stop above (loss), target = lower band (profit). Stop has priority on a wick.
                if h[i] >= stp:
                    ex = stp if op <= stp else op   # gap-up through stop -> fill at open
                elif l[i] <= lo[i]:
                    ex = lo[i] if op >= lo[i] else op   # gap-down through target -> fill at open
            else:
                # long: stop below (loss), target = upper band (profit).
                if l[i] <= stp:
                    ex = stp if op >= stp else op
                elif h[i] >= up[i]:
                    ex = up[i] if op <= up[i] else op
            if np.isnan(ex) and (i - ei) >= tcap:
                ex = c[i]
            if not np.isnan(ex):
                tol = 1e-6
                if ex < l[i]-tol or ex > h[i]+tol:
                    violations += 1
                pnl = pos*(ex-ent)/pip - ent_spread
                e_bar[nt]=ei; e_dir[nt]=pos; e_pnl[nt]=pnl; e_xbar[nt]=i; nt+=1
                pos = 0
        if pos == 0:
            e = o[i+1]
            sig_dir = 0
            if l[i-1] > up[i-1] and l[i] <= up[i] and 0.5*(c[i]-basis[i])/pip - spread_arr[i] >= meat:
                sig_dir = -1
            elif h[i-1] < lo[i-1] and h[i] >= lo[i] and 0.5*(basis[i]-c[i])/pip - spread_arr[i] >= meat:
                sig_dir = 1
            if sig_dir != 0:
                tr = trend[i]; take = True
                if book == 1: take = (sig_dir == tr) and (tr != 0)
                elif book == 2: take = (sig_dir == -tr) and (tr != 0)
                if take:
                    if sig_dir == -1:
                        stop_lvl = e + stop_pips*pip
                        if e >= stop_lvl: skipped += 1
                        else:
                            pos=-1; ent=e; ei=i+1; stp=stop_lvl; ent_spread=spread_arr[i+1]
                    else:
                        stop_lvl = e - stop_pips*pip
                        if e <= stop_lvl: skipped += 1
                        else:
                            pos=1; ent=e; ei=i+1; stp=stop_lvl; ent_spread=spread_arr[i+1]
    return e_bar[:nt], e_dir[:nt], e_pnl[:nt], e_xbar[:nt], violations, skipped


def load_pair(pair, tf_rule):
    df = pd.read_parquet(f"{DATA}/{pair}_S5_BA.parquet",
                         columns=["timestamp","open","high","low","close","bid_c","ask_c"])
    bars = resample(df, tf_rule)
    pip = pip_of(pair)
    o=bars["open"].values.astype(float); h=bars["high"].values.astype(float)
    l=bars["low"].values.astype(float); c=bars["close"].values.astype(float)
    spread = ((bars["ask"].values - bars["bid"].values)/pip).astype(float)
    spread = np.clip(spread, 0.0, None)
    basis, up, lo = compute_bands(c)
    return dict(pair=pair, pip=pip, o=o,h=h,l=l,c=c, spread=spread,
                basis=basis, up=up, lo=lo, ts=bars.index.values, n=len(c))


def stats_from(pnl, days):
    """summary stats from a pnl array + elapsed days for p/d."""
    if len(pnl)==0:
        return dict(n=0, pips=0.0, ppd=0.0, wr=0.0, mean=0.0, maxdd=0.0)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return dict(n=int(len(pnl)), pips=float(eq[-1]),
                ppd=float(eq[-1]/days) if days>0 else 0.0,
                wr=float((pnl>0).mean()), mean=float(pnl.mean()),
                maxdd=float(dd.min()))


def main():
    # config sweep
    tf = os.environ.get("TF","H1")
    tf_rule = {"M15":"15min","H1":"1h","H4":"4h"}[tf]
    # trend definitions to sweep (kind, win) — win in bars of the signal TF
    trend_defs = [("sma",100),("sma",200),("ret",100)]
    stop_pips = float(os.environ.get("STOP", "40"))
    tcap_map = {"M15": 96, "H1": 48, "H4": 24}   # ~1 day, 2 days, 4 days
    tcap = int(os.environ.get("TCAP", tcap_map[tf]))

    print(f"=== TREND-FILTERED REVERSION ===  TF={tf} ({tf_rule})  stop={stop_pips}p  tcap={tcap}  MEAT={MEAT}  SMA={SMA_N} K={K}")
    print(f"Trend defs: {trend_defs}")

    # load all pairs once
    t0=time.time()
    pdata = {}
    for p in PAIRS:
        pdata[p] = load_pair(p, tf_rule)
    print(f"loaded 12 pairs in {time.time()-t0:.0f}s; bar counts:",
          {p: pdata[p]['n'] for p in PAIRS})

    total_viol = 0; total_skip = 0
    results = {}  # results[(kind,win)][book] -> per-pair + portfolio + WF

    for (kind, win) in trend_defs:
        for p in PAIRS:
            pdata[p][f"trend_{kind}_{win}"] = trend_signal(pdata[p]["c"], kind, win)

        cfgkey = f"{kind}{win}"
        results[cfgkey] = {}
        for book, bname in [(0,"unfiltered"),(1,"with-trend"),(2,"against-trend")]:
            book_per_pair = {}
            # portfolio fold accumulation: align trades by global time, sum pnl per fold
            # Each pair has its own bar timeline; we define folds per pair by bar index quantiles.
            port_fold_pnl = [[] for _ in range(N_FOLDS)]
            for p in PAIRS:
                pd_ = pdata[p]
                tr = pd_[f"trend_{kind}_{win}"]
                eb,ed,ep,ex,viol,skip = _bt(
                    pd_["o"],pd_["h"],pd_["l"],pd_["c"],
                    pd_["up"], pd_["lo"], pd_["basis"],
                    tr, pd_["spread"], pd_["pip"], MEAT, stop_pips, tcap, book)
                total_viol += viol; total_skip += skip
                n = pd_["n"]
                # elapsed days from timestamps
                ts = pd_["ts"]
                days = (pd_["ts"][-1]-pd_["ts"][0]) / np.timedelta64(1,'D')
                # WF folds by ENTRY bar index, contiguous equal slices over the full timeline
                fold_edges = np.linspace(0, n, N_FOLDS+1).astype(int)
                fold_stats = []
                for f in range(N_FOLDS):
                    lo_e, hi_e = fold_edges[f], fold_edges[f+1]
                    mask = (eb >= lo_e) & (eb < hi_e)
                    fp = ep[mask]
                    fdays = max((ts[min(hi_e,n-1)]-ts[lo_e])/np.timedelta64(1,'D'), 1e-9)
                    fold_stats.append(stats_from(fp, fdays))
                    # portfolio: collect (entry_time, pnl) tuples
                    for j in np.where(mask)[0]:
                        port_fold_pnl[f].append((ts[eb[j]], ep[j]))
                book_per_pair[p] = dict(overall=stats_from(ep, days), folds=fold_stats,
                                        long=stats_from(ep[ed==1], days),
                                        short=stats_from(ep[ed==-1], days))
            # portfolio per-fold (equal weight = just pool all pairs' trades in the fold window)
            port_folds = []
            for f in range(N_FOLDS):
                lst = port_fold_pnl[f]
                if lst:
                    lst.sort(key=lambda x: x[0])
                    ppnl = np.array([x[1] for x in lst])
                    tspan = (lst[-1][0]-lst[0][0])/np.timedelta64(1,'D')
                    port_folds.append(stats_from(ppnl, max(tspan,1e-9)))
                else:
                    port_folds.append(stats_from(np.array([]),1))
            # portfolio overall = sum across all folds
            all_port = []
            for f in range(N_FOLDS): all_port.extend(port_fold_pnl[f])
            all_port.sort(key=lambda x: x[0])
            allpnl = np.array([x[1] for x in all_port]) if all_port else np.array([])
            tspan = (all_port[-1][0]-all_port[0][0])/np.timedelta64(1,'D') if all_port else 1
            port_overall = stats_from(allpnl, max(tspan,1e-9))
            results[cfgkey][bname] = dict(per_pair=book_per_pair,
                                          port_overall=port_overall,
                                          port_folds=port_folds)

    print(f"\n### SELF-CHECK: range violations = {total_viol}  |  entry-validity skips = {total_skip}")
    if total_viol > 0:
        print("!!! ABORT-WORTHY: exit prices outside bar range detected. Numbers NOT trustworthy. !!!")

    # ===== REPORT =====
    print("\n" + "="*100)
    print("PORTFOLIO SUMMARY (equal-weight pool, all 12 pairs)  — pips total / per-day / WR / maxDD")
    print("="*100)
    for cfgkey in results:
        print(f"\n--- TREND DEF: {cfgkey} ---")
        for bname in ["unfiltered","with-trend","against-trend"]:
            po = results[cfgkey][bname]["port_overall"]
            print(f"  {bname:>14}: n={po['n']:6d}  pips={po['pips']:9.1f}  ppd={po['ppd']:7.2f}  "
                  f"WR={po['wr']*100:5.1f}%  mean={po['mean']:6.2f}p  maxDD={po['maxdd']:9.1f}")
            pf = results[cfgkey][bname]["port_folds"]
            folds_str = "  ".join([f"F{i+1}:{x['pips']:+8.1f}({x['n']})" for i,x in enumerate(pf)])
            print(f"  {'':>14}  WF folds: {folds_str}")

    # per-pair table for the headline config (with-trend, first trend def)
    print("\n" + "="*100)
    print("PER-PAIR (with-trend book)  — for each trend def")
    print("="*100)
    for cfgkey in results:
        print(f"\n--- {cfgkey}  with-trend ---")
        print(f"  {'pair':<8} {'n':>6} {'pips':>9} {'ppd':>7} {'WR%':>6} {'mean':>6} {'maxDD':>9} | folds(pips)")
        bp = results[cfgkey]["with-trend"]["per_pair"]
        npos=0
        for p in PAIRS:
            ov = bp[p]["overall"]
            fs = " ".join([f"{x['pips']:+6.0f}" for x in bp[p]["folds"]])
            if ov["pips"]>0: npos+=1
            print(f"  {p:<8} {ov['n']:>6} {ov['pips']:>9.1f} {ov['ppd']:>7.2f} {ov['wr']*100:>6.1f} "
                  f"{ov['mean']:>6.2f} {ov['maxdd']:>9.1f} | {fs}")
        print(f"  --> pairs positive overall: {npos}/12")

    # asymmetry check
    print("\n" + "="*100)
    print("ASYMMETRY: against-trend per-pair (should be robustly NEGATIVE if filter logic holds)")
    print("="*100)
    for cfgkey in results:
        bp = results[cfgkey]["against-trend"]["per_pair"]
        nneg = sum(1 for p in PAIRS if bp[p]["overall"]["pips"]<0)
        po = results[cfgkey]["against-trend"]["port_overall"]
        print(f"  {cfgkey}: portfolio {po['pips']:+.1f}p ({po['ppd']:+.2f} p/d)  pairs negative: {nneg}/12")

    # dump JSON for downstream
    out = {}
    for cfgkey in results:
        out[cfgkey] = {}
        for bname in results[cfgkey]:
            r = results[cfgkey][bname]
            out[cfgkey][bname] = dict(
                port_overall=r["port_overall"],
                port_folds=r["port_folds"],
                per_pair={p: r["per_pair"][p]["overall"] for p in PAIRS},
                per_pair_folds={p: r["per_pair"][p]["folds"] for p in PAIRS},
            )
    with open(f"research/experiments/daily_ma/tfr_results_{tf}.json","w") as f:
        json.dump(dict(tf=tf, stop=stop_pips, tcap=tcap, n_folds=N_FOLDS,
                       violations=total_viol, skips=total_skip, results=out), f, indent=1)
    print(f"\nJSON -> research/experiments/daily_ma/tfr_results_{tf}.json")
    print("DONE", tf)


if __name__ == "__main__":
    main()
