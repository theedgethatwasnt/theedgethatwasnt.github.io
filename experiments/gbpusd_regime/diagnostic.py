#!/usr/bin/env python3
"""
GBP_USD regime/transition diagnostic. User's observation: price wanders around a level,
then a sharp EFFICIENT news-driven shift to a new level, then wanders. Make-or-break
question: is a transition CAUSALLY detectable and rideable net of spread, or only
describable in hindsight?

Three reads (M5 mid, real spread):
 (1) CONCENTRATION — how much of each day's net movement lives in a few efficient windows.
 (2) DETECT→CONTINUE — define a shift candidate causally at bar t (trailing W-bar net >= K
     pips AND efficiency |net|/path >= E). Then measure the NEXT H bars: does price continue
     the same direction, and what is the oriented move NET OF SPREAD? Compare vs all bars and
     vs same-size LOW-efficiency moves (the false pokes). If continue-net ~ -spread => the
     efficient move is already spent at detection = hindsight-only, not tradeable.
 (3) HOUR clustering of shift candidates (session/news proxy; no calendar needed).
"""
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[3]
PAIR = "GBP_USD"; PIP = 0.0001


def main():
    df = pd.read_parquet(PROJECT/"data"/"m5_ba"/f"{PAIR}_M5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mid = df["close"].values.astype(float)
    sp = ((df["ask_c"].values - df["bid_c"].values)/PIP)
    n = len(mid); med_sp = np.median(sp)
    print(f"{PAIR} M5: {n:,} bars, {ts.iloc[0].date()}..{ts.iloc[-1].date()}, median spread {med_sp:.1f}p\n")

    # rolling net + efficiency over W bars (causal, known at bar t)
    W = 12  # 1 hour
    diff = np.abs(np.diff(mid, prepend=mid[0]))/PIP
    path = pd.Series(diff).rolling(W).sum().values
    net = np.full(n, np.nan); net[W:] = (mid[W:] - mid[:-W])/PIP
    eff = np.abs(net)/np.where(path > 0, path, np.nan)

    # (1) CONCENTRATION: per UTC-day, share of total |path| that is net directional,
    #     and share of the day's summed |15min net| concentrated in the top-3 windows.
    day = ts.dt.floor("1D").values
    d = pd.DataFrame({"day": day, "mid": mid})
    # 15-min window net moves within each day
    w3 = 3  # 3 M5 = 15 min
    net15 = np.full(n, np.nan); net15[w3:] = np.abs(mid[w3:]-mid[:-w3])/PIP
    dd = pd.DataFrame({"day": day, "net15": net15, "abs_step": diff}).dropna()
    g = dd.groupby("day")
    day_path = g["abs_step"].sum()
    # top-3 fifteen-min windows' share of the day's summed 15-min |net|
    def top_share(x):
        x = np.sort(x.values)[::-1]; tot = x.sum()
        return (x[:3].sum()/tot*100) if tot > 0 and len(x) >= 3 else np.nan
    top3 = g["net15"].apply(top_share)
    print("=== (1) CONCENTRATION (per day) ===")
    print(f"  median top-3 fifteen-min windows' share of the day's total 15m movement: {np.nanmedian(top3):.0f}%")
    print(f"  (a 'pure step' day concentrates ~all movement in a few windows; a 'random walk' day spreads it evenly)")
    # efficiency-of-the-day: |daily net| / daily path
    print()

    # (2) DETECT -> CONTINUE  (the make-or-break)
    print("=== (2) DETECT a high-efficiency breakout causally, then measure the NEXT move ===")
    print(f"  {'K(net)':>7}{'E(eff)':>7}{'H(fwd)':>7}{'n_cand':>8}{'P(cont)':>9}{'cont_net/sp':>13}{'lowEff_net':>12}")
    for K in (15, 25, 40):
        for E in (0.5, 0.7):
            for H in (6, 12, 24):
                cand = (np.abs(net) >= K) & (eff >= E)
                cand &= np.arange(n) < (n - H)
                idx = np.where(cand)[0]
                if len(idx) < 50:
                    continue
                fwd = (mid[idx+H] - mid[idx])/PIP
                oriented = np.sign(net[idx]) * fwd
                cont = (oriented > 0).mean()*100
                cont_net = oriented.mean() - med_sp
                # false-poke control: same size move but LOW efficiency
                low = (np.abs(net) >= K) & (eff < 0.35) & (np.arange(n) < (n-H))
                li = np.where(low)[0]
                low_net = (np.sign(net[li])*((mid[li+H]-mid[li])/PIP)).mean() - med_sp if len(li) > 30 else np.nan
                print(f"  {K:>7}{E:>7.1f}{H:>7}{len(idx):>8}{cont:>8.1f}%{cont_net:>13.2f}{(low_net if np.isfinite(low_net) else 0):>12.2f}")
    print("  cont_net/sp = oriented next-H move minus one spread (>0 = the breakout still has juice to ride;")
    print("               ~<=0 = the efficient move is already spent at detection => hindsight-only).")

    # (3) HOUR clustering of shift candidates
    print("\n=== (3) HOUR-of-day clustering of shift candidates (K=25,E=0.6) ===")
    cand = (np.abs(net) >= 25) & (eff >= 0.6)
    hours = ts.dt.hour.values[cand]
    base = ts.dt.hour.values
    hc = pd.Series(hours).value_counts().sort_index()
    bc = pd.Series(base).value_counts().sort_index()
    lift = (hc/hc.sum()) / (bc/bc.sum())   # over/under-representation vs baseline hour mix
    top = lift.sort_values(ascending=False).head(6)
    print("  hours where shifts are MOST over-represented (UTC hour: lift vs baseline):")
    for h, l in top.items():
        tag = " <- London open" if h in (7,8) else (" <- NY open / US data" if h in (12,13,14) else "")
        print(f"    {int(h):02d}:00  lift {l:.2f}x{tag}")


if __name__ == "__main__":
    main()
