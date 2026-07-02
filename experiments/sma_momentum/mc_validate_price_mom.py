#!/usr/bin/env python3
"""
Price Momentum Confluence — MC Validation
==========================================
Validates two top configs from price_mom_sweep.py:
  A) H1+M30  lags=(8,10,20) TP=15p  — highest p/d (+33.7)
  B) M15+M5  lags=(1,3,8)   TP=10p  — best IS3 12/12, shorter holds

2000 sign-shuffles per pair + portfolio MC.
Pass criteria: IS 3/3 AND mc_p < 0.05 AND OOS p/d > 0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70
MC_N    = 2000

CONFIGS = [
    dict(label="H1+M30 lags=(8,10,20) TP=15p", tf1="1h",   tf2="30min", lags=(8,10,20), tp=15.0),
    dict(label="M15+M5 lags=(1,3,8)   TP=10p", tf1="15min",tf2="5min",  lags=(1,3,8),  tp=10.0),
]

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001


def build_signal(df, lags, tf1, tf2):
    moms = []
    for tf in [tf1, tf2]:
        rs = df["close"].resample(tf).last().dropna()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


def simulate_tp(df, sig, pip, tp_pips, sp_gate):
    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = (ask - bid) / pip
    s   = sig.values; n = len(df)
    pnls = []; holds = []
    in_trade = False; dir_ = 0; ep = 0.0; ei = 0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) / pip * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnls.append((exit_px - ep) / pip * dir_ - sp[i])
                holds.append(i - ei)
                in_trade = False
        else:
            nd = s[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True
    return np.array(pnls, dtype=np.float64), np.array(holds, dtype=np.int32)


@njit(parallel=True)
def mc_sign_shuffle(pnls, observed_ppd, oos_days, n_shuffles):
    beats = 0
    for _ in prange(n_shuffles):
        total = 0.0
        for j in range(len(pnls)):
            sign = 1.0 if np.random.random() > 0.5 else -1.0
            total += pnls[j] * sign
        if (total / oos_days) >= observed_ppd:
            beats += 1
    return beats / n_shuffles


def run_config(cfg):
    tf1, tf2, lags, tp = cfg["tf1"], cfg["tf2"], cfg["lags"], cfg["tp"]
    results = []; all_pnls = []; total_days = 0.0

    for pair in PAIRS:
        df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
        df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
        pip = pip_sz(pair)
        n_is = int(len(df) * IS_FRAC)
        sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
        oos_df  = df.iloc[n_is:]
        oos_days = len(oos_df) / 288

        sig = build_signal(df, lags, tf1, tf2)

        # IS 3-fold
        fold = n_is // 3; is_ppds = []
        for f in range(3):
            s = f * fold; e = s + fold if f < 2 else n_is
            days_f = (e - s) / 288
            p, _ = simulate_tp(df.iloc[s:e], sig.iloc[s:e], pip, tp, sp_gate)
            is_ppds.append(p.sum() / days_f if len(p) else 0.0)
        is_pass = sum(1 for x in is_ppds if x > 0)

        # OOS
        p_oos, h_oos = simulate_tp(oos_df, sig.iloc[n_is:], pip, tp, sp_gate)
        oos_ppd = p_oos.sum() / oos_days if len(p_oos) else 0.0
        oos_wr  = float((p_oos > 0).mean() * 100) if len(p_oos) else 0.0
        tpd     = len(p_oos) / oos_days
        mean_h  = float(h_oos.mean() * 5 / 60) if len(h_oos) else 0.0
        p50_h   = float(np.percentile(h_oos, 50) * 5 / 60) if len(h_oos) else 0.0
        p90_h   = float(np.percentile(h_oos, 90) * 5 / 60) if len(h_oos) else 0.0

        mc_p = float(mc_sign_shuffle(p_oos, oos_ppd, oos_days, MC_N)) if len(p_oos) else None

        results.append(dict(
            pair=pair, sp_gate=round(sp_gate, 2),
            is_pass=is_pass,
            is_f1=round(is_ppds[0],2), is_f2=round(is_ppds[1],2), is_f3=round(is_ppds[2],2),
            oos_ppd=round(oos_ppd,2), oos_nt=len(p_oos),
            oos_wr=round(oos_wr,1), tpd=round(tpd,3),
            mean_h=round(mean_h,1), p50_h=round(p50_h,1), p90_h=round(p90_h,1),
            oos_days=round(oos_days,0),
            mc_p=round(mc_p,4) if mc_p is not None else None,
        ))
        all_pnls.append(p_oos)
        total_days = max(total_days, oos_days)

        is_g = "✅" if is_pass==3 else ("🟡" if is_pass==2 else "❌")
        mc_g = ("✅" if mc_p is not None and mc_p < 0.05 else "❌") if mc_p is not None else "—"
        print(f"    {pair:<10} IS {is_pass}/3 {is_g}  "
              f"OOS {oos_ppd:+.1f}p/d  WR {oos_wr:.0f}%  "
              f"n={len(p_oos)}  mc_p={mc_p:.4f} {mc_g}  "
              f"hold P50={p50_h:.1f}h P90={p90_h:.1f}h")

    # Portfolio MC
    port_pnls = np.concatenate(all_pnls)
    port_ppd  = sum(r["oos_ppd"] for r in results)
    port_mc_p = float(mc_sign_shuffle(port_pnls, port_ppd, total_days, MC_N))
    port_nt   = sum(r["oos_nt"] for r in results)
    port_wr   = float((port_pnls > 0).mean() * 100) if len(port_pnls) else 0.0

    n_is3  = sum(1 for r in results if r["is_pass"] == 3)
    n_mc05 = sum(1 for r in results if r["mc_p"] is not None and r["mc_p"] < 0.05)
    n_both = sum(1 for r in results
                 if r["is_pass"] == 3 and r["mc_p"] is not None and r["mc_p"] < 0.05)

    print(f"\n  {'─'*60}")
    print(f"  Portfolio p/day   : {port_ppd:+.1f}p")
    print(f"  Portfolio WR      : {port_wr:.1f}%")
    print(f"  Portfolio MC p    : {port_mc_p:.4f}  "
          f"({'SIGNIFICANT ✅' if port_mc_p < 0.05 else 'NOT SIGNIFICANT ❌'})")
    print(f"  IS 3/3            : {n_is3}/{len(PAIRS)} pairs")
    print(f"  MC p<0.05         : {n_mc05}/{len(PAIRS)} pairs")
    print(f"  IS3 AND MC<0.05   : {n_both}/{len(PAIRS)} pairs  ← deploy candidates")

    deploy = [r for r in results
              if r["is_pass"] == 3 and r["mc_p"] is not None and r["mc_p"] < 0.05]
    print(f"\n  Deploy candidates ({len(deploy)} pairs):")
    print(f"  {'Pair':>10} {'IS':>5} {'p/d':>8} {'WR':>6} {'t/d':>6} "
          f"{'mc_p':>8} {'sp_gate':>8} {'P50h':>6} {'P90h':>6}")
    print("  " + "─" * 68)
    for r in sorted(deploy, key=lambda x: -x["oos_ppd"]):
        print(f"  {r['pair']:>10} {r['is_pass']}/3  {r['oos_ppd']:>+7.1f}  "
              f"{r['oos_wr']:>5.1f}%  {r['tpd']:>5.3f}  "
              f"{r['mc_p']:>8.4f}  {r['sp_gate']:>6.2f}p  "
              f"{r['p50_h']:>5.1f}h  {r['p90_h']:>5.1f}h")

    pd.DataFrame(results).to_csv(
        RESULTS / f"price_mc_{cfg['label'][:6].replace(' ','_')}.csv", index=False)

    return dict(port_ppd=port_ppd, port_mc_p=port_mc_p, n_is3=n_is3,
                n_mc05=n_mc05, n_both=n_both, deploy=deploy)


# ── JIT warm-up ──────────────────────────────────────────────────────────────
mc_sign_shuffle(np.array([1.0,-1.0]), 0.0, 1.0, 10)

# ── Run both configs ─────────────────────────────────────────────────────────
summaries = []
for cfg in CONFIGS:
    print(f"\n{'='*72}")
    print(f"CONFIG: {cfg['label']}")
    print(f"{'='*72}")
    s = run_config(cfg)
    summaries.append((cfg["label"], s))

# ── Final comparison ──────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("COMPARISON SUMMARY")
print(f"{'='*72}")
print(f"{'Config':<36} {'p/d':>7} {'port_mc':>9} {'IS3':>6} {'MC<.05':>7} {'Deploy':>7}")
print("─" * 72)
for label, s in summaries:
    print(f"{label:<36} {s['port_ppd']:>+6.1f}p  "
          f"{s['port_mc_p']:>8.4f}  "
          f"{s['n_is3']:>4}/12  "
          f"{s['n_mc05']:>5}/12  "
          f"{s['n_both']:>5}/12")
print()
print("SMA16 baseline: +29.8p/d  mc_p=0.0000  IS3=10/12  MC=10/12  Deploy=10/12")
