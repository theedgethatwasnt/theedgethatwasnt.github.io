"""
screen_shock_continuation.py — shock → consolidation → continuation breakout (2026-06-14).

User idea (from the Jun-11 USD/JPY S5 zoom): after a shock leg + a brief consolidation/shelf,
price often BREAKS the shelf and CONTINUES. Prior research tested the immediate FADE (won) and
the symmetric straddle (lost) but never the consolidation-conditioned continuation at +10–15 min.

For each shock (|z|>2.5):
  leg extreme in [t, t+44]; leg_mag = |close[t]-extreme|.
  consolidation window [t_ext, t_ext+W]: shelf_low/high, bounce → retrace_depth = bounce/leg_mag.
  BREAKOUT: price breaks the shelf in the SHOCK direction within BO bars after the window → enter
  (short after down-shock, long after up-shock) at bid/ask. Continuation pnl = forward H bars,
  R3-correct (entry bid/short, exit ask; mirror long). Net of real spread.
Bucket by retrace_depth (shallow/mid/deep). Hypothesis: shallow-retrace continuation > spread.
Also reports the unconditional (no-breakout) continuation as a baseline. Multi-pair, IS/OOS 70/30.
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/path/to/projects/fx-core")
S5_DIR = ROOT / "data/s5_ohlc"
PAIRS = ["USD_JPY", "GBP_JPY", "EUR_JPY", "AUD_JPY", "EUR_USD", "GBP_USD"]
THR, PB, ZW, MADW = 2.5, 44, 6, 2048
W, BO, MIN_LEG = 120, 30, 8.0        # consol window (10min), breakout window, min leg pips
HORIZONS = [120, 180]                # +10, +15 min
IS_FRAC = 0.70


def shock_z(close, pip):
    n = len(close); vel = np.empty(n); vel[:ZW] = 0.0
    vel[ZW:] = (close[ZW:] - close[:n-ZW]) / pip
    vs = pd.Series(vel)
    rm = vs.rolling(MADW, min_periods=50).median()
    rmad = (vs - rm).abs().rolling(MADW, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel


def process(pair, H):
    pip = 0.01 if "JPY" in pair else 0.0001
    t = pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet", columns=["close", "bid_c", "ask_c"])
    close = t["close"].to_numpy().astype(np.float64)
    bid = t["bid_c"].to_numpy().astype(np.float64); ask = t["ask_c"].to_numpy().astype(np.float64)
    n = len(close)
    z, vel = shock_z(close, pip)
    is_end = int(n * IS_FRAC)
    cd = 0; recs = []
    for ti in range(MADW + 100, n - PB - W - BO - H - 2):
        if cd > 0:
            cd -= 1; continue
        if abs(z[ti]) <= THR:
            continue
        cd = (PB + W) // 2
        d = 1 if vel[ti] > 0 else -1               # shock direction (+1 up, -1 down)
        seg = close[ti:ti+PB+1]
        if d == -1:
            ext = seg.min(); t_ext = ti + int(seg.argmin())
        else:
            ext = seg.max(); t_ext = ti + int(seg.argmax())
        leg = abs(close[ti] - ext) / pip
        if leg < MIN_LEG:
            continue
        cw = close[t_ext:t_ext+W+1]
        if d == -1:
            shelf = cw.min(); bounce = cw.max(); depth = (bounce - ext) / (leg*pip)
        else:
            shelf = cw.max(); bounce = cw.min(); depth = (ext - bounce) / (leg*pip)
        dec = t_ext + W
        # breakout in shock direction within BO bars
        ent = None
        for j in range(dec, dec+BO+1):
            if d == -1 and close[j] < shelf: ent = j; break
            if d == 1 and close[j] > shelf: ent = j; break
        if ent is None or ent + H >= n:
            continue
        if d == -1:                                # SHORT continuation: sell bid, buy back ask
            pnl = (bid[ent] - ask[ent+H]) / pip
        else:                                      # LONG: buy ask, sell bid
            pnl = (bid[ent+H] - ask[ent]) / pip
        recs.append((depth, pnl, ent < is_end))
    del t, close, bid, ask, z, vel; gc.collect()
    return recs


def main():
    for H in HORIZONS:
        print(f"\n{'='*84}\nSHOCK→SHELF→BREAKOUT CONTINUATION  H=+{H*5//60}min  W={W*5//60}min consol  "
              f"({len(PAIRS)} pairs, net spread)\n{'='*84}")
        allrec = []
        for p in PAIRS:
            allrec += [(p,)+r for r in process(p, H)]
        D = pd.DataFrame(allrec, columns=["pair", "depth", "pnl", "is_"])
        oos = D[~D["is_"]]
        buckets = [("shallow <0.33", oos["depth"] < 0.33),
                   ("mid 0.33-0.66", (oos["depth"] >= 0.33) & (oos["depth"] < 0.66)),
                   ("deep >0.66", oos["depth"] >= 0.66),
                   ("ALL", oos["depth"] >= -99)]
        print(f"  {'retrace bucket':16s} {'n_oos':>6s} {'mean pnl':>9s} {'WR':>5s} {'IS mean':>8s}  net-of-spread")
        for name, m in buckets:
            sub = oos[m]; isub = D[D["is_"] & (D["depth"].between(*( (-99,0.33) if 'shallow' in name else (0.33,0.66) if 'mid' in name else (0.66,99) if 'deep' in name else (-99,99))))]
            if len(sub) < 20:
                print(f"  {name:16s} n={len(sub)} (too few)"); continue
            flag = "🟢" if sub["pnl"].mean() > 0 else "🔴"
            print(f"  {name:16s} {len(sub):>6d} {sub['pnl'].mean():>+9.3f} {100*(sub['pnl']>0).mean():>4.0f}% "
                  f"{isub['pnl'].mean():>+8.3f}  {flag}")
        # per-pair shallow bucket (is it universal or one-pair?)
        print("  -- shallow<0.33 bucket, per pair (OOS mean pnl) --")
        for p in PAIRS:
            sp = oos[(oos["pair"]==p) & (oos["depth"]<0.33)]
            if len(sp) >= 15:
                print(f"     {p:8s} n={len(sp):>4d} mean={sp['pnl'].mean():+.3f} WR={100*(sp['pnl']>0).mean():.0f}%")
    print("="*84)


if __name__ == "__main__":
    main()
