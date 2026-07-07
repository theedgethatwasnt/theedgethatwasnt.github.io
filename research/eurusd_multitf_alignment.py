"""
EUR/USD Multi-TF Alignment Study
=================================
For each M5 bar, check:
  - Did any S30 sub-bar qualify as a big bar?
  - Did any M1 sub-bar qualify as a big bar?
  - Is the M5 bar itself big?
  - Do all firing TFs agree on direction (all bull / all bear)?

All 8 alignment states (2^3), directional variants, plus:
  - Density filter layered on top
  - S5 included when available
  - Profit factor, win rate, t-test vs baseline

Forward returns: close[i+lag] - close[i], signed by direction of the M5 bar.
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.swing_indicators import topsbots_swings

PIP = 0.0001

# ── helpers ──────────────────────────────────────────────────────────────────

def true_range_pips(df):
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    pc = np.empty_like(cl); pc[0]=cl[0]; pc[1:]=cl[:-1]
    return np.maximum(hi-lo, np.maximum(np.abs(hi-pc), np.abs(lo-pc))) / PIP

def load_mid(path):
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index":"timestamp","time":"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "open" not in df.columns:
        for fr,to in [("o","open"),("h","high"),("l","low"),("c","close")]:
            df[to] = (df[f"bid_{fr}"].astype(float) + df[f"ask_{fr}"].astype(float)) / 2
    else:
        for c in ["open","high","low","close"]:
            df[c] = df[c].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def align_low_to_high(df_low, df_high, sec_high):
    """Return int array len(df_low): index into df_high, -1 if out of range."""
    lo_ts = df_low["timestamp"].values.astype(np.int64)
    hi_ts = df_high["timestamp"].values.astype(np.int64)
    hi_dur = sec_high * int(1e9)
    pos = np.searchsorted(hi_ts, lo_ts, side="right") - 1
    valid = (pos >= 0) & (lo_ts < hi_ts[np.clip(pos, 0, len(hi_ts)-1)] + hi_dur)
    pos[~valid] = -1
    return pos

def big_flag_and_dir(df, sigma=3.0):
    """Returns (big_bool_array, direction_array +1/-1/0)."""
    tr  = true_range_pips(df)
    thr = tr.mean() + sigma * tr.std()
    big = tr >= thr
    dr  = np.sign(df["close"].values.astype(float) - df["open"].values.astype(float))
    return big, dr, thr

def build_reversal_density(df_h1, pip_bin=5, half_life_days=30):
    hi = df_h1["high"].values.astype(float)
    lo = df_h1["low"].values.astype(float)
    swings = topsbots_swings(hi, lo)
    grid = pip_bin * PIP
    price_min = df_h1["low"].min() - grid
    n_bins = int((df_h1["high"].max() - price_min) / grid) + 5
    s_prices = np.array([v for _,_,v in swings], dtype=float)
    s_ts     = np.array([df_h1["timestamp"].iloc[bi].value for bi,_,_ in swings], dtype=np.int64)
    s_bins   = np.clip(((s_prices - price_min) / grid).astype(int), 0, n_bins-1)
    half_ns  = half_life_days * 86400 * int(1e9)
    ln2 = np.log(2)
    def density_at(price, ts_ns):
        b = int((price - price_min) / grid)
        in_bin = s_bins == b
        if not in_bin.any():
            return 0.0
        ages = np.maximum(ts_ns - s_ts[in_bin], 60*int(1e9))
        return float(np.exp(-ln2 * ages / half_ns).sum())
    return density_at

def fwd_stats(dir_fwd, label, lag):
    """Returns dict with mean, wr, avgW, avgL, pf."""
    if len(dir_fwd) < 5:
        return None
    wins = dir_fwd[dir_fwd > 0]
    loss = dir_fwd[dir_fwd < 0]
    wr   = 100*len(wins)/len(dir_fwd)
    avgW = wins.mean()  if len(wins)  > 0 else 0
    avgL = abs(loss.mean()) if len(loss) > 0 else 0
    pf   = (avgW*len(wins)) / (avgL*len(loss)) if (len(loss)>0 and avgL>0) else np.inf
    return dict(n=len(dir_fwd), mean=dir_fwd.mean(), wr=wr, avgW=avgW, avgL=avgL, pf=pf)

def print_row(label, n, lag_stats, baseline_mean):
    """Print one results row."""
    print(f"  {label:<32} {n:>5} │", end="")
    for s in lag_stats:
        if s is None:
            print("      n/a", end="")
        else:
            arrow = "↑" if s["mean"] > baseline_mean+0.02 else ("↓" if s["mean"] < baseline_mean-0.02 else "─")
            print(f"  {s['mean']:+.2f}p{arrow}", end="")
    print()

def ttest_vs_base(signal_vals, base_vals, label, lags):
    """Print t-test results for each lag."""
    for li, (sv, lag) in enumerate(zip(signal_vals, lags)):
        if sv is None or len(sv) < 5:
            continue
        t, p = scipy_stats.ttest_ind(sv, base_vals[li], equal_var=False)
        sig = "🟢" if p < 0.05 else ("🟡" if p < 0.15 else "🔴")
        print(f"    lag+{lag}M5:  n={len(sv):>4}  mean={np.mean(sv):+.3f}p  "
              f"t={t:+.2f}  p={p:.3f}  {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    LAGS = (1, 3, 5, 10)
    SIGMA = 3.0
    REF_DAYS = 90

    print(f"\n{'═'*72}")
    print(f"  EUR/USD Multi-TF Big-Bar Alignment Study")
    print(f"  sigma={SIGMA}  ref_days={REF_DAYS}  lags={LAGS}")
    print(f"{'═'*72}\n")

    # ── load data ────────────────────────────────────────────────────────────
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=REF_DAYS)

    print("Loading timeframes …")
    df_m5_full = load_mid(ROOT / "data" / "m5_ba" / "EUR_USD_M5_BA.parquet")
    df_m5 = df_m5_full[df_m5_full["timestamp"] >= cutoff].reset_index(drop=True)

    df_m1_raw = load_mid(ROOT / "data" / "m1_ohlc" / "EUR_USD_M1_BA.parquet")
    df_m1 = df_m1_raw[df_m1_raw["timestamp"] >= cutoff].reset_index(drop=True)

    df_s30_raw = load_mid(ROOT / "data" / "s30_ohlc" / "EUR_USD_S30_BA.parquet")
    df_s30 = df_s30_raw[df_s30_raw["timestamp"] >= cutoff].reset_index(drop=True)

    s5_path = ROOT / "data" / "s5_ohlc" / "EUR_USD_S5_BA.parquet"
    has_s5 = s5_path.exists()
    if has_s5:
        df_s5_raw = load_mid(s5_path)
        df_s5 = df_s5_raw[df_s5_raw["timestamp"] >= cutoff].reset_index(drop=True)
        print(f"  S5: {len(df_s5):,}  S30: {len(df_s30):,}  M1: {len(df_m1):,}  M5: {len(df_m5):,}")
    else:
        df_s5 = None
        print(f"  S30: {len(df_s30):,}  M1: {len(df_m1):,}  M5: {len(df_m5):,}  (no S5)")

    # H1 from full M5 for reversal density
    h1 = df_m5_full.set_index("timestamp").resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna().reset_index()
    print(f"  H1 (density): {len(h1):,} bars (5.5yr)")

    # ── big-bar flags and directions ─────────────────────────────────────────
    big_m5, dir_m5, thr_m5 = big_flag_and_dir(df_m5, SIGMA)
    big_m1, dir_m1, thr_m1 = big_flag_and_dir(df_m1, SIGMA)
    big_s30, dir_s30, thr_s30 = big_flag_and_dir(df_s30, SIGMA)
    if has_s5:
        big_s5, dir_s5, thr_s5 = big_flag_and_dir(df_s5, SIGMA)

    print(f"\n  Big-bar thresholds (3σ):")
    print(f"    M5:  {thr_m5:.2f}p  ({100*big_m5.mean():.1f}%  n={big_m5.sum()})")
    print(f"    M1:  {thr_m1:.2f}p  ({100*big_m1.mean():.1f}%  n={big_m1.sum()})")
    print(f"    S30: {thr_s30:.2f}p  ({100*big_s30.mean():.1f}%  n={big_s30.sum()})")
    if has_s5:
        print(f"    S5:  {thr_s5:.2f}p  ({100*big_s5.mean():.1f}%  n={big_s5.sum()})")

    # ── align lower TFs to M5 ────────────────────────────────────────────────
    print("\n  Aligning lower TFs to M5 bars …", flush=True)
    N5 = len(df_m5)

    # M1 → M5
    m1_to_m5  = align_low_to_high(df_m1, df_m5, 300)
    m5_has_big_m1  = np.zeros(N5, dtype=bool)
    m5_dir_m1_bull = np.zeros(N5, dtype=int)  # count of bull big M1 bars
    m5_dir_m1_bear = np.zeros(N5, dtype=int)
    valid1 = m1_to_m5 >= 0
    np.bitwise_or.at(m5_has_big_m1, m1_to_m5[valid1], big_m1[valid1])
    bull_big_m1 = valid1 & big_m1 & (dir_m1 > 0)
    bear_big_m1 = valid1 & big_m1 & (dir_m1 < 0)
    np.add.at(m5_dir_m1_bull, m1_to_m5[bull_big_m1], 1)
    np.add.at(m5_dir_m1_bear, m1_to_m5[bear_big_m1], 1)

    # S30 → M5
    s30_to_m5 = align_low_to_high(df_s30, df_m5, 300)
    m5_has_big_s30  = np.zeros(N5, dtype=bool)
    m5_dir_s30_bull = np.zeros(N5, dtype=int)
    m5_dir_s30_bear = np.zeros(N5, dtype=int)
    valid30 = s30_to_m5 >= 0
    np.bitwise_or.at(m5_has_big_s30, s30_to_m5[valid30], big_s30[valid30])
    bull_big_s30 = valid30 & big_s30 & (dir_s30 > 0)
    bear_big_s30 = valid30 & big_s30 & (dir_s30 < 0)
    np.add.at(m5_dir_s30_bull, s30_to_m5[bull_big_s30], 1)
    np.add.at(m5_dir_s30_bear, s30_to_m5[bear_big_s30], 1)

    # S5 → M5 (if available)
    if has_s5:
        s5_to_m5 = align_low_to_high(df_s5, df_m5, 300)
        m5_has_big_s5  = np.zeros(N5, dtype=bool)
        m5_dir_s5_bull = np.zeros(N5, dtype=int)
        m5_dir_s5_bear = np.zeros(N5, dtype=int)
        valid5 = s5_to_m5 >= 0
        np.bitwise_or.at(m5_has_big_s5, s5_to_m5[valid5], big_s5[valid5])
        bull_big_s5 = valid5 & big_s5 & (dir_s5 > 0)
        bear_big_s5 = valid5 & big_s5 & (dir_s5 < 0)
        np.add.at(m5_dir_s5_bull, s5_to_m5[bull_big_s5], 1)
        np.add.at(m5_dir_s5_bear, s5_to_m5[bear_big_s5], 1)

    print("  Done.")

    # ── reversal density for each M5 bar ────────────────────────────────────
    print("  Computing reversal density …", flush=True)
    density_fn = build_reversal_density(h1)
    cl5 = df_m5["close"].values.astype(float)
    ts5 = df_m5["timestamp"].values.astype(np.int64)
    dens = np.array([density_fn(cl5[i], ts5[i]) for i in range(N5)], dtype=float)
    dens_norm = dens / (dens.max() or 1)
    d33, d67 = np.percentile(dens_norm[dens_norm > 0], [33, 67])
    low_dens  = dens_norm <= d33
    high_dens = dens_norm >= d67
    print(f"  Density: low≤{d33:.3f}  high≥{d67:.3f}")

    # ── forward return arrays ────────────────────────────────────────────────
    def get_fwd(mask, lag):
        """Directional forward return: sign(M5 body) × (close[i+lag]-close[i])."""
        valid = mask & (np.arange(N5) < N5 - lag)
        idx = np.where(valid)[0]
        fwd = (cl5[idx + lag] - cl5[idx]) / PIP
        return dir_m5[idx] * fwd   # +ve = continuation, -ve = reversal

    # ── define all alignment groups ──────────────────────────────────────────
    # Directional agreement: M5 direction matches all firing lower TFs
    def dir_agree_bull(m5_i):
        return (dir_m5 > 0) & m5_i
    def dir_agree(mask_m5, bull_s30, bear_s30, bull_m1, bear_m1):
        """Direction agreed: M5 is bull and all lower big bars are bull, or all bear."""
        bull_agreed = (dir_m5 > 0) & mask_m5 & (bull_s30 > 0) & (bull_m1 > 0) & (bear_s30 == 0) & (bear_m1 == 0)
        bear_agreed = (dir_m5 < 0) & mask_m5 & (bear_s30 > 0) & (bear_m1 > 0) & (bull_s30 == 0) & (bull_m1 == 0)
        return bull_agreed | bear_agreed

    # All 8 alignment states
    s30_ = m5_has_big_s30
    m1_  = m5_has_big_m1
    m5_  = big_m5

    groups_base = {
        "none (0/3)":         (~s30_) & (~m1_) & (~m5_),
        "S30 only (1/3)":      s30_   & (~m1_) & (~m5_),
        "M1 only (1/3)":      (~s30_) &   m1_  & (~m5_),
        "M5 only (1/3)":      (~s30_) & (~m1_) &   m5_,
        "S30+M1 (2/3)":        s30_   &   m1_  & (~m5_),
        "S30+M5 (2/3)":        s30_   & (~m1_) &   m5_,
        "M1+M5 (2/3)":        (~s30_) &   m1_  &   m5_,
        "S30+M1+M5 (3/3)":     s30_   &   m1_  &   m5_,
    }
    # Directional agreement subsets
    dir_s30m1m5 = dir_agree(s30_ & m1_ & m5_, m5_dir_s30_bull, m5_dir_s30_bear,
                             m5_dir_m1_bull, m5_dir_m1_bear)

    groups_dir = {
        "S30+M1+M5 dir_agree":       dir_s30m1m5,
        "S30+M1+M5 dir_agree+lowD":  dir_s30m1m5 & low_dens,
        "S30+M1+M5 dir_agree+highD": dir_s30m1m5 & high_dens,
    }
    if has_s5:
        s5_ = m5_has_big_s5
        groups_base["S5+S30+M1+M5 (4/4)"] = s5_ & s30_ & m1_ & m5_
        dir_s5s30m1m5 = (
            dir_s30m1m5 &
            ((dir_m5 > 0) & (m5_dir_s5_bull > 0) & (m5_dir_s5_bear == 0) |
             (dir_m5 < 0) & (m5_dir_s5_bear > 0) & (m5_dir_s5_bull == 0))
        )
        groups_dir["S5+S30+M1+M5 dir_agree"] = dir_s5s30m1m5
        groups_dir["S5+S30+M1+M5 dir+lowD"]  = dir_s5s30m1m5 & low_dens

    # ── print results ────────────────────────────────────────────────────────
    baseline = np.ones(N5, dtype=bool)
    base_fwds = [get_fwd(baseline, lag) for lag in LAGS]
    base_means = [f.mean() if len(f) else 0 for f in base_fwds]

    def section(title, groups):
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"{'═'*72}")
        print(f"  {'Group':<32}  n   │" + "".join(f"  lag+{l}M5" for l in LAGS))
        print(f"  {'─'*32}  ─   │" + "─────────"*len(LAGS))

        all_signal_fwds = []
        for gname, gmask in groups.items():
            lag_stats = []
            lag_fwds  = []
            for lag in LAGS:
                fv = get_fwd(gmask, lag)
                lag_fwds.append(fv)
                lag_stats.append(fwd_stats(fv, gname, lag))
            all_signal_fwds.append((gname, gmask, lag_stats, lag_fwds))
            print_row(gname, gmask.sum(), lag_stats, base_means[0])

        # Baseline
        lag_stats_base = [fwd_stats(base_fwds[li], "baseline", LAGS[li]) for li in range(len(LAGS))]
        print_row("── baseline (all bars) ──", N5, lag_stats_base, base_means[0])

        # Win rate + PF details for interesting groups
        print(f"\n  Win rate & PF details:")
        print(f"  {'Group':<32} {'Lag':>5}  {'WR%':>6}  {'AvgW':>8}  {'AvgL':>8}  {'PF':>6}  {'n':>5}")
        for gname, gmask, lag_stats, lag_fwds in all_signal_fwds:
            n = gmask.sum()
            if n < 10:
                continue
            for li, (lag, fv) in enumerate(zip(LAGS, lag_fwds)):
                s = lag_stats[li]
                if s is None: continue
                wins = fv[fv > 0]; loss = fv[fv < 0]
                wr   = 100*len(wins)/len(fv) if len(fv) else 0
                avgW = wins.mean()  if len(wins) else 0
                avgL = abs(loss.mean()) if len(loss) else 0
                pf   = (avgW*len(wins))/(avgL*len(loss)) if len(loss)>0 and avgL>0 else np.inf
                bar  = "🟢" if pf>1.1 and wr>52 else ("🟡" if pf>0.95 else "🔴")
                print(f"  {gname:<32} {f'+{lag}':>5}  {wr:>5.1f}%  {avgW:>+7.2f}p  "
                      f"{avgL:>7.2f}p  {pf:>5.2f}x  {n:>5}  {bar}")

        # T-test for standout groups
        print(f"\n  Significance vs baseline (Welch t-test):")
        for gname, gmask, lag_stats, lag_fwds in all_signal_fwds:
            n = gmask.sum()
            if n < 10: continue
            print(f"  [{gname}]")
            for li, (lag, fv) in enumerate(zip(LAGS, lag_fwds)):
                if len(fv) < 5: continue
                t, p = scipy_stats.ttest_ind(fv, base_fwds[li], equal_var=False)
                sig = "🟢" if p < 0.05 else ("🟡" if p < 0.15 else "🔴")
                print(f"    lag+{lag}M5: t={t:+.2f}  p={p:.3f}  {sig}  "
                      f"mean={fv.mean():+.3f}p  n={len(fv)}")

    section("ALIGNMENT STATE — all 8 combinations (S30 / M1 / M5)", groups_base)
    section("DIRECTIONAL AGREEMENT — all TFs same direction", groups_dir)

    # ── summary interpretation ────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  SUMMARY")
    print(f"{'═'*72}")
    mask_3tf = s30_ & m1_ & m5_
    mask_dir = dir_s30m1m5
    for label, mask in [("S30+M1+M5 (any dir)", mask_3tf),
                         ("S30+M1+M5 dir_agree", mask_dir),
                         ("S30+M1+M5 dir+lowD",  mask_dir & low_dens)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: n={n} — too few bars")
            continue
        fv1 = get_fwd(mask, 1)
        fv5 = get_fwd(mask, 5)
        t1, p1 = scipy_stats.ttest_ind(fv1, base_fwds[0], equal_var=False)
        t5, p5 = scipy_stats.ttest_ind(fv5, base_fwds[2], equal_var=False)
        wins1 = fv1[fv1>0]; loss1 = fv1[fv1<0]
        wr1 = 100*len(wins1)/len(fv1) if len(fv1) else 0
        pf1 = (wins1.mean()*len(wins1))/(abs(loss1.mean())*len(loss1)) if len(loss1)>0 else np.inf
        print(f"\n  {label}  n={n}")
        print(f"    lag+1: mean={fv1.mean():+.3f}p  WR={wr1:.1f}%  PF={pf1:.2f}x  "
              f"p={p1:.3f} {'🟢' if p1<0.05 else '🔴'}")
        print(f"    lag+5: mean={fv5.mean():+.3f}p  p={p5:.3f} {'🟢' if p5<0.05 else '🔴'}")
    print()


if __name__ == "__main__":
    main()
