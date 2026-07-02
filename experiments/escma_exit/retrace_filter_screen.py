"""
retrace_filter_screen.py — find complementary ENTRY filters for the retrace (2026-06-12).

The exit isn't the lever (proven). The Markov filter turns raw -386 p/d into +9 by selecting
which shocks retrace. Can a COMPLEMENTARY entry-time feature reject more dead-on-arrival shocks
while keeping the TP winners? Build a per-entry dataset (entry-time features + realized baseline
pnl), screen which features separate winners from losers (LightGBM + univariate IC + deciles),
then test simple threshold filters for OOS p/d improvement (multi-pair, IS/OOS).

All features causal — computed at the shock bar t (entry is at t+45). 3 live pairs.
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
from backtest_retrace_exits import compute_shock_z, sim_exits, PEAK_BARS, HORIZON, Z_WINDOW, THR

PROJECT = Path(__file__).resolve().parents[3]
S5_DIR = PROJECT / "data" / "s5_ba"
PAIRS = ["GBP_JPY", "USD_JPY", "AUD_JPY"]
IS_FRAC = 0.70


def build_entries(pair):
    df = pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
    close = df["close"].values.astype(np.float64)
    bid = df["bid_c"].values.astype(np.float64); ask = df["ask_c"].values.astype(np.float64)
    ts = pd.to_datetime(df["timestamp"]).dt.hour.values if "timestamp" in df.columns else np.zeros(len(close), int)
    z, vel = compute_shock_z(close, 0.01)
    sf = (np.abs(z) > THR).astype(np.int8)
    pnl = sim_exits(bid, ask, close, sf, vel, 0.01, 0, 0, 0, 0, 0)   # baseline TP20/SL30/horizon
    n = len(close)
    # reconstruct entry shock bars (same loop/cooldown as sim)
    shock_t = []; cd = 0
    for t in range(Z_WINDOW, n - PEAK_BARS - HORIZON - 2):
        if cd > 0: cd -= 1; continue
        if sf[t] != 1: continue
        shock_t.append(t); cd = (PEAK_BARS + HORIZON)//2
    shock_t = np.array(shock_t[:len(pnl)])
    # per-entry features at shock bar t
    rows = []
    for k, t in enumerate(shock_t):
        d_shock = 1 if vel[t] > 0 else -1
        fade = -d_shock
        # spike extension during [t, t+44] in shock direction (pips)
        if d_shock == 1:
            ext = (ask[t:t+PEAK_BARS+1].max() - ask[t]) / 0.01
        else:
            ext = (bid[t] - bid[t:t+PEAK_BARS+1].min()) / 0.01
        m1h = (close[t] - close[t-720]) / 0.01 if t >= 720 else 0.0
        m4h = (close[t] - close[t-2880]) / 0.01 if t >= 2880 else 0.0
        rows.append((abs(z[t]), abs(vel[t]), ext,
                     fade*np.sign(m1h), fade*np.sign(m4h),   # +1 = fade aligns with HTF trend
                     abs(m1h), abs(m4h),
                     (ask[t]-bid[t])/0.01, int(ts[t]), t))
    cols = ["z","abs_vel","spike_ext","fade_align_1h","fade_align_4h",
            "abs_m1h","abs_m4h","spread","hour","bar_t"]
    fdf = pd.DataFrame(rows, columns=cols)
    fdf["pnl"] = pnl[:len(fdf)]
    fdf["pair"] = pair
    fdf["is_oos"] = np.where(fdf["bar_t"] < int(n*IS_FRAC), "IS", "OOS")
    fdf["oos_days"] = (n - int(n*IS_FRAC)) / 17280
    fdf["n_bar"] = n
    del df, close, bid, ask, z, vel; gc.collect()
    return fdf


def main():
    import lightgbm as lgb
    parts = [build_entries(p) for p in PAIRS]
    D = pd.concat(parts, ignore_index=True)
    feats = ["z","abs_vel","spike_ext","fade_align_1h","fade_align_4h","abs_m1h","abs_m4h","spread","hour"]
    print(f"{'='*72}\nRETRACE ENTRY-FILTER SCREEN  (3 pairs, {len(D)} entries)\nbaseline mean pnl/entry = {D['pnl'].mean():+.3f}p   TP-rate(pnl>=18)={100*(D['pnl']>=18).mean():.1f}%\n{'='*72}")

    is_m = D["is_oos"]=="IS"; oos = D[~is_m]
    # univariate: mean pnl by feature sign / bucket
    print("\nUnivariate (OOS mean pnl by feature condition):")
    for f, cond, desc in [
        ("fade_align_4h", D["fade_align_4h"]>0, "fade WITH 4h trend"),
        ("fade_align_4h", D["fade_align_4h"]<0, "fade AGAINST 4h trend"),
        ("fade_align_1h", D["fade_align_1h"]>0, "fade WITH 1h trend"),
        ("z", D["z"]>4, "big shock z>4"),
        ("z", D["z"]<=3, "small shock z<=3"),
        ("spike_ext", D["spike_ext"]>15, "spike_ext>15p"),
        ("spike_ext", D["spike_ext"]<=8, "spike_ext<=8p"),
    ]:
        m = (~is_m) & cond
        print(f"  {desc:26s} n={int(m.sum()):>6} mean_pnl={D.loc[m,'pnl'].mean():+.3f}p")

    # LightGBM importance + OOS IC
    model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=300, verbosity=-1)
    model.fit(D.loc[is_m, feats], D.loc[is_m, "pnl"])
    pred = model.predict(D.loc[~is_m, feats]); ytrue = D.loc[~is_m, "pnl"].values
    ic = np.corrcoef(pred, ytrue)[0,1]
    print(f"\nLightGBM importances: " + ", ".join(f"{f}={i}" for f,i in sorted(zip(feats, model.feature_importances_), key=lambda x:-x[1])))
    print(f"LightGBM OOS IC(pred,pnl) = {ic:+.4f}")
    order = np.argsort(pred); d = len(order)//5
    print(f"  top quintile mean_pnl={ytrue[order[-d:]].mean():+.3f}p   bottom quintile={ytrue[order[:d]].mean():+.3f}p")

    # filter test: apply best simple filters, OOS p/d vs baseline (per pair then summed)
    print(f"\n{'='*72}\nFILTER TEST — OOS p/d (summed 3 pairs), baseline = unfiltered\n{'='*72}")
    def oos_pd(mask):
        s = 0.0
        for p in PAIRS:
            sub = D[(D["pair"]==p) & (~is_m) & mask]
            if len(sub):
                s += sub["pnl"].sum() / sub["oos_days"].iloc[0]
        return s, int(((~is_m)&mask).sum())
    base_pd, base_n = oos_pd(pd.Series(True, index=D.index))
    print(f"  {'unfiltered baseline':40s} OOS_pd={base_pd:+8.1f}  n={base_n}")
    for desc, mask in [
        ("fade WITH 4h trend", D["fade_align_4h"]>0),
        ("fade WITH 1h AND 4h trend", (D["fade_align_4h"]>0)&(D["fade_align_1h"]>0)),
        ("spike_ext>15p", D["spike_ext"]>15),
        ("spike_ext>15 AND fade_align_4h>0", (D["spike_ext"]>15)&(D["fade_align_4h"]>0)),
        ("z>4 AND fade_align_4h>0", (D["z"]>4)&(D["fade_align_4h"]>0)),
        ("LGBM top-quintile pred", pd.Series(False, index=D.index)),  # placeholder filled below
    ]:
        if "LGBM" in desc:
            thr = np.quantile(pred, 0.8)
            full = pd.Series(False, index=D.index); full.loc[D.index[~is_m]] = pred >= thr
            o, nn = oos_pd(full)
        else:
            o, nn = oos_pd(mask)
        flag = "🟢" if o > base_pd else "🔴"
        print(f"  {desc:40s} OOS_pd={o:+8.1f}  n={nn}  {flag} (Δ{o-base_pd:+.1f})")
    print("="*72)


if __name__ == "__main__":
    main()
