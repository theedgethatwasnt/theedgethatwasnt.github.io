#!/usr/bin/env python3
"""
Two studies in one:
A) MAE stop-loss sweep on current H1+M30 SMA16 strategy
B) TF × TP sweep with hold-time tracking — find 2-4h avg hold configs
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
DEPLOY = ["USD_JPY","EUR_JPY","GBP_JPY","AUD_JPY","EUR_USD","GBP_USD",
          "CAD_JPY","AUD_USD","EUR_GBP","NZD_USD"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

def pip(p): return 0.01 if p in JPY else 0.0001

IS_FRAC = 0.70
SMA_N   = 16
LAGS    = (8, 10, 15)

def build_signal(df, sma_n, lags, tf1, tf2):
    moms = []
    for tf in [tf1, tf2]:
        rs  = df["close"].resample(tf).last().dropna()
        sma = rs.rolling(sma_n, min_periods=sma_n).mean().shift(1)
        sma = sma.reindex(df.index, method="ffill")
        for k in lags:
            moms.append(sma - sma.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    sig   = pd.Series(np.int8(0), index=df.index)
    sig[score == len(moms)] = np.int8(1)
    sig[score == 0]         = np.int8(-1)
    return sig

def simulate_full(df, sig, pip_sz, tp_pips, sp_gate, mae_stop=None):
    """Returns pnls, hold_bars, mae_at_exit arrays."""
    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = (ask - bid) / pip_sz
    s   = sig.values; n = len(df)
    pnls=[]; holds=[]; maes=[]; reasons=[]
    in_trade=False; dir_=0; ep=0.0; ei=0; worst=0.0
    for i in range(1, n):
        if in_trade:
            cur = (mid[i] - ep) / pip_sz * dir_
            if cur < worst: worst = cur
            if mae_stop and worst <= -mae_stop:
                exit_px = bid[i] if dir_==1 else ask[i]
                pnls.append((exit_px-ep)/pip_sz*dir_ - sp[i])
                holds.append(i-ei); maes.append(worst); reasons.append("mae")
                in_trade=False; worst=0.0
            elif cur >= tp_pips:
                exit_px = bid[i] if dir_==1 else ask[i]
                pnls.append((exit_px-ep)/pip_sz*dir_ - sp[i])
                holds.append(i-ei); maes.append(worst); reasons.append("tp")
                in_trade=False; worst=0.0
        else:
            nd = s[i-1]
            if nd!=0 and sp[i]<=sp_gate:
                ep=(ask[i] if nd==1 else bid[i]); dir_=nd; ei=i
                in_trade=True; worst=0.0
    return (np.array(pnls, dtype=np.float64),
            np.array(holds, dtype=np.int32),
            np.array(maes,  dtype=np.float64),
            reasons)

def is_folds(df, sig, pip_sz, sp_gate, n_is, tp_pips, mae_stop=None):
    fold=n_is//3; passes=0
    for f in range(3):
        s=f*fold; e=s+fold if f<2 else n_is; days=(e-s)/288
        p,_,_,_ = simulate_full(df.iloc[s:e], sig.iloc[s:e],
                                 pip_sz, tp_pips, sp_gate, mae_stop)
        if len(p)>0 and p.sum()/days>0: passes+=1
    return passes

# ── Pre-load ──────────────────────────────────────────────────────────────────
print("Loading data …")
cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    p  = pip(pair)
    n_is = int(len(df)*IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"]-df["bid_c"]).iloc[:n_is]/p, 90))
    cache[pair] = dict(df=df, pip=p, sp_gate=sp_gate, n_is=n_is,
                       oos_days=len(df.iloc[n_is:])/288)

# ══════════════════════════════════════════════════════════════════════════════
# STUDY A — MAE stop-loss sweep on current H1+M30 SMA16 strategy
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STUDY A — MAE Stop-Loss Sweep  (H1+M30, SMA16, lags 8/10/15, TP=20p)")
print("="*72)

# First: build MAE distribution of all TP winners (no stop)
print("\nMAE distribution of TP winners (no stop):")
all_winner_mae = []
for pair in DEPLOY:
    c = cache[pair]
    sig = build_signal(c["df"], SMA_N, LAGS, "1h", "30min")
    p,h,m,r = simulate_full(c["df"].iloc[c["n_is"]:], sig.iloc[c["n_is"]:],
                             c["pip"], 20.0, c["sp_gate"], mae_stop=None)
    winner_mae = [-m[i] for i,reason in enumerate(r) if reason=="tp"]
    all_winner_mae.extend(winner_mae)

mae_arr = np.array(all_winner_mae)
for pct in [50,75,90,95,99]:
    print(f"  P{pct} MAE of TP winners: {np.percentile(mae_arr,pct):.1f}p")

MAE_STOPS = [3, 5, 8, 10, 12, 15, 20, 25, 30]
print(f"\n{'Stop':>6} {'p/day':>8} {'WR':>6} {'trades':>7} {'mae%':>7} {'IS3≥8':>7} {'Δp/d':>8}")
print("-"*55)

base_ppd = None
mae_rows = []
for stop in [None] + MAE_STOPS:
    ppd_sum=0; all_p=[]; n_is3=0; time_exit_n=0; total_n=0
    for pair in DEPLOY:
        c = cache[pair]
        sig = build_signal(c["df"], SMA_N, LAGS, "1h", "30min")
        p,h,m,r = simulate_full(c["df"].iloc[c["n_is"]:], sig.iloc[c["n_is"]:],
                                 c["pip"], 20.0, c["sp_gate"], mae_stop=stop)
        ppd = p.sum()/c["oos_days"] if len(p) else 0
        ppd_sum += ppd; all_p.append(p)
        time_exit_n += r.count("mae"); total_n += len(r)
        is_p = is_folds(c["df"], sig, c["pip"], c["sp_gate"],
                        c["n_is"], 20.0, mae_stop=stop)
        if is_p==3: n_is3+=1

    all_pnl = np.concatenate(all_p)
    wr  = (all_pnl>0).mean()*100 if len(all_pnl) else 0
    mae_pct = time_exit_n/total_n*100 if total_n else 0
    delta = ppd_sum - base_ppd if base_ppd else 0
    if stop is None:
        base_ppd = ppd_sum
        lbl = "  none"
    else:
        lbl = f"{stop:>5}p"
    print(f"{lbl}  {ppd_sum:>+7.1f}  {wr:>5.1f}%  {len(all_pnl):>6}  {mae_pct:>6.1f}%  {n_is3:>4}/10  {delta:>+7.1f}")
    mae_rows.append(dict(stop=stop or 0, ppd=round(ppd_sum,2), wr=round(wr,1),
                         n=len(all_pnl), mae_pct=round(mae_pct,1), n_is3=n_is3))

pd.DataFrame(mae_rows).to_csv(RESULTS/"mae_stop_sweep.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# STUDY B — TF × TP hold-time study — find 2-4h average hold configs
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*72)
print("STUDY B — Hold-Time Analysis: find configs with 2-4h average hold")
print("Target: mean hold 24-48 bars (2-4h on M5), P90 < 288 bars (24h)")
print("="*72)

TF_COMBOS = [
    ("1h",  "30min", "H1+M30",  PAIRS),
    ("30min","15min","M30+M15", PAIRS),
    ("15min","5min", "M15+M5",  PAIRS),
]
TP_SWEEP = [5, 8, 10, 12, 15, 20]
SMA_SWEEP = [8, 16]
LAG_OPTS  = [(3,5,8),(5,8,10),(8,10,15)]

hold_rows = []
print(f"\n{'TFs':>10} {'SMA':>4} {'Lags':>10} {'TP':>4} "
      f"{'p/d':>7} {'t/d':>6} {'P50h':>6} {'P90h':>6} "
      f"{'mean_h':>7} {'IS3≥8':>7}")
print("-"*75)

for tf1, tf2, label, pairs_use in TF_COMBOS:
    for sma_n in SMA_SWEEP:
        for lags in LAG_OPTS:
            for tp in TP_SWEEP:
                ppd_sum=0; tpd_sum=0; all_h=[]; n_is3=0
                for pair in pairs_use:
                    c = cache[pair]
                    sig = build_signal(c["df"], sma_n, lags, tf1, tf2)
                    p,h,m,r = simulate_full(c["df"].iloc[c["n_is"]:],
                                            sig.iloc[c["n_is"]:],
                                            c["pip"], tp, c["sp_gate"])
                    if len(p)>0:
                        ppd_sum += p.sum()/c["oos_days"]
                        tpd_sum += len(p)/c["oos_days"]
                        all_h.extend(h.tolist())
                    is_p = is_folds(c["df"], sig, c["pip"], c["sp_gate"],
                                    c["n_is"], tp)
                    if is_p==3: n_is3+=1

                if not all_h: continue
                h_arr = np.array(all_h)*5/60  # convert bars→hours
                mean_h = h_arr.mean()
                p50_h  = np.percentile(h_arr, 50)
                p90_h  = np.percentile(h_arr, 90)

                # flag configs with target hold profile
                target = (1.5 <= mean_h <= 5.0) and p90_h < 24.0

                hold_rows.append(dict(
                    tfs=label, sma=sma_n, lags=str(lags), tp=tp,
                    ppd=round(ppd_sum,2), tpd=round(tpd_sum,3),
                    mean_h=round(mean_h,1), p50_h=round(p50_h,1),
                    p90_h=round(p90_h,1), n_is3=n_is3,
                    target=target,
                ))

                if target:
                    star = " ◄"
                    print(f"{label:>10} {sma_n:>4} {str(lags):>10} {tp:>3}p "
                          f"{ppd_sum:>+6.1f} {tpd_sum:>5.2f}  "
                          f"{p50_h:>5.1f}h {p90_h:>5.1f}h {mean_h:>6.1f}h "
                          f"  {n_is3:>3}/12{star}")

df_hold = pd.DataFrame(hold_rows)
df_hold.to_csv(RESULTS/"hold_time_study.csv", index=False)

# Summary: best target configs by p/d
print("\n\nBest configs in 2-4h hold target (mean 1.5-5h, P90<24h), sorted by p/d:")
targets = df_hold[df_hold["target"]].sort_values("ppd", ascending=False)
print(f"{'TFs':>10} {'SMA':>4} {'Lags':>10} {'TP':>4} "
      f"{'p/d':>7} {'t/d':>5} {'mean_h':>7} {'P90_h':>6} {'IS3':>6}")
print("-"*68)
for _, r in targets.head(20).iterrows():
    print(f"{r.tfs:>10} {int(r.sma):>4} {r.lags:>10} {int(r.tp):>3}p "
          f"{r.ppd:>+6.1f} {r.tpd:>5.2f}  {r.mean_h:>6.1f}h {r.p90_h:>5.1f}h "
          f"  {int(r.n_is3):>2}/12")

print(f"\nSaved → {RESULTS/'mae_stop_sweep.csv'}, {RESULTS/'hold_time_study.csv'}")
