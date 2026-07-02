#!/usr/bin/env python3
"""
Stage 1 — Momentum × Efficiency characterization (no trading).
==============================================================
User's features (causal, rolling window W minutes on M5):
  momentum  = (close[t] - close[t-W]) / pip / W_minutes      # signed pips/min (net velocity)
  path_len  = sum_{window} |Δclose| / pip                    # total travel (pips)
  efficiency= |net move pips| / path_len  ∈ [0,1]            # clean vs wiggle

Question: across the (|momentum| × efficiency) plane, does the next move CONTINUE
the momentum direction or REVERSE it — and does efficiency sharpen that?

Cell metric = mean( sign(momentum) * forward_return_pips ) - spread
  > 0  → momentum CONTINUES here (trend), net of spread
  < 0  → momentum REVERSES here (fade pays)

Per-pair 5×5 grid (|momentum| quintile rows × efficiency quintile cols), then
averaged across the 12 pairs (equal weight) + count of pairs with cell>0.
Also: corr(momentum, fwd) within each efficiency quintile (signed IC).

Forward horizon = window-matched (next W minutes) and 2×W. Read-only on data/m5_ba.
NOTE: quintile edges use full-series quantiles (characterization only; Stage 2
would use IS-only thresholds). Features are strictly causal; only the TARGET sees
the future. Overlapping windows inflate t-stats — read the PATTERN, not |t|.
"""
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)
PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
def pip_sz(p): return 0.01 if p in JPY else 0.0001
WINDOWS_MIN = [15, 30, 60, 120]   # rolling window length, minutes
NQ = 5                            # quintiles

def load_m5(pair):
    df = pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    return df.astype({c:"float64" for c in df.select_dtypes("float32").columns})

def features(df, pip, W):
    bw = W // 5                                   # M5 bars in window
    c = df["close"]
    net_pips = (c - c.shift(bw)) / pip            # signed net move (pips)
    mom      = net_pips / W                       # signed pips/min
    path     = (c.diff().abs() / pip).rolling(bw).sum()   # travel (pips)
    eff      = net_pips.abs() / path.replace(0, np.nan)   # [0,1]
    spread   = (df["ask_c"] - df["bid_c"]) / pip
    return mom, eff, spread, net_pips

def per_pair_grid(pair, W, Hbars):
    df = load_m5(pair); pip = pip_sz(pair)
    mom, eff, spread, net = features(df, pip, W)
    c = df["close"]
    fwd = (c.shift(-Hbars) - c) / pip             # forward return (pips), the TARGET
    cont = np.sign(mom) * fwd - spread            # continuation pnl net of spread
    d = pd.DataFrame({"absmom": mom.abs(), "eff": eff, "mom": mom, "fwd": fwd, "cont": cont}).dropna()
    d = d[d["absmom"] > 0]
    if len(d) < 2000: return None, None, None
    d["mq"] = pd.qcut(d["absmom"], NQ, labels=False, duplicates="drop")
    d["eq"] = pd.qcut(d["eff"],    NQ, labels=False, duplicates="drop")
    grid = d.groupby(["mq","eq"])["cont"].mean().unstack()
    grid = grid.reindex(index=range(NQ), columns=range(NQ))
    # signed IC of momentum vs forward return within each efficiency quintile
    ic = []
    for q in range(NQ):
        s = d[d["eq"]==q]
        ic.append(np.corrcoef(s["mom"], s["fwd"])[0,1] if len(s)>50 and s["mom"].std()>0 else np.nan)
    return grid.values, np.array(ic), d.groupby(["mq","eq"]).size().unstack().reindex(index=range(NQ),columns=range(NQ)).values

print("Stage 1 — momentum × efficiency characterization\n")
summary = []
for W in WINDOWS_MIN:
    bw = W // 5
    for Hlabel, Hbars in [("1xW", bw), ("2xW", 2*bw)]:
        grids=[]; ics=[]; ns=[]
        for pair in PAIRS:
            g, ic, n = per_pair_grid(pair, W, Hbars)
            if g is None: continue
            grids.append(g); ics.append(ic); ns.append(n)
        G = np.nanmean(np.stack(grids), axis=0)            # avg per-pair continuation (pips net spread)
        POS = np.nansum(np.stack([gg>0 for gg in grids]), axis=0)  # # pairs with cell>0
        IC = np.nanmean(np.stack(ics), axis=0)
        print("="*84)
        print(f"W={W}min  forward H={Hlabel} ({Hbars*5}min)   cell = mean(sign(mom)*fwd) - spread  [pips]")
        print(f"  rows = |momentum| quintile (Q0 slowest .. Q4 fastest net move)")
        print(f"  cols = efficiency quintile  (Q0 wiggly .. Q4 clean)")
        hdr = "        " + "".join(f"  eff_Q{q}" for q in range(NQ))
        print(hdr)
        for mq in range(NQ):
            row = "  |m|Q%d "%mq + "".join(f"{G[mq,q]:+7.2f}" for q in range(NQ))
            posr = " ".join(f"{int(POS[mq,q])}" for q in range(NQ))
            print(row + f"   pairs+: {posr}")
        print(f"  momentum→fwd IC by efficiency quintile: " +
              "  ".join(f"effQ{q}={IC[q]:+.3f}" for q in range(NQ)))
        # best & user's target cell (mid |mom|, top eff)
        bi = np.unravel_index(np.nanargmax(G), G.shape)
        wi = np.unravel_index(np.nanargmin(G), G.shape)
        tgt = G[2, NQ-1]   # moderate momentum, highest efficiency (user's hypothesis)
        print(f"  best cell  = |m|Q{bi[0]},effQ{bi[1]}  {G[bi]:+.2f}p ({int(POS[bi])}/12 pairs+)")
        print(f"  worst cell = |m|Q{wi[0]},effQ{wi[1]}  {G[wi]:+.2f}p ({int(POS[wi])}/12 pairs+)")
        print(f"  user-target (mod |m| Q2, clean effQ4) = {tgt:+.2f}p ({int(POS[2,NQ-1])}/12 pairs+)")
        summary.append(dict(W=W, H=Hlabel, best_cell=f"m{bi[0]}e{bi[1]}", best=round(float(G[bi]),2),
                            best_pairs=int(POS[bi]), worst=round(float(G[wi]),2),
                            target_modmom_cleaneff=round(float(tgt),2), target_pairs=int(POS[2,NQ-1]),
                            ic_lowEff=round(float(IC[0]),3), ic_hiEff=round(float(IC[NQ-1]),3)))
        print()

sm = pd.DataFrame(summary); sm.to_csv(RESULTS/"stage1_mom_eff.csv", index=False)
print("="*84); print("SUMMARY  → results/stage1_mom_eff.csv"); print("="*84)
print(sm.to_string(index=False))
print("""
READ:
 - If cells are mostly NEGATIVE and grow more negative with |momentum| → fast net
   moves REVERSE (fade pays); efficiency columns show whether clean moves reverse less.
 - If cells turn POSITIVE in moderate-|momentum| + high-efficiency → continuation edge
   (user's hypothesis). 'pairs+' >= 9/12 = robust cell.
 - IC sign by efficiency quintile: does momentum predict (+) or anti-predict (-) the
   next move, and does high efficiency flip/strengthen it?
""")
