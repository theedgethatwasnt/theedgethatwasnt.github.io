#!/usr/bin/env python3
"""
v3 extension — extended trail multiplier sweep (tighter trails)

Fixed entry  : sma_n=7, n_small=1, thld_mult=2.0×ATR  (the v3 winning config)
Swept exit   : activate ∈ {0.25, 0.5, 1.0, 1.5, 2.0}
               trail    ∈ {0.10, 0.20, 0.25, 0.30, 0.40, 0.50}
12 pairs OOS; also reports 7-winner subset.

Question: does tighter trail rescue the GBP_JPY (-1.94, -772p worst) and
USD_JPY (-0.86, -432p) tails the v3 default trail=0.5×ATR couldn't bound?
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

# Import sim + helpers from v3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_h1_xbreak_atrail import (
    _sim_atrail, resample_h1, wilder_atr, max_dd, warmup_jit,
    ALL_PAIRS, JPY, SP_GATES, pip_sz, IS_FRAC, MAX_HOLD_H1, ATR_PERIOD,
)

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

WINNERS_7 = {"AUD_JPY","NZD_JPY","CHF_JPY","EUR_GBP","EUR_JPY","EUR_USD","GBP_USD"}

# Fixed entry (v3 winner)
SMA_N      = 7
N_SMALL    = 1
THLD_MULT  = 2.0

# Extended exit grid (tighter trail focus)
ACTIVATE_MULTS = [0.25, 0.5, 1.0, 1.5, 2.0]
TRAIL_MULTS    = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    h1 = resample_h1(df_m5)
    del df_m5
    pip = pip_sz(pair); sg = SP_GATES[pair]

    high  = h1["high"].values.astype(np.float64)
    low   = h1["low"].values.astype(np.float64)
    close = h1["close"].values.astype(np.float64)
    bid   = h1["bid_c"].values.astype(np.float64)
    ask   = h1["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close)
    n_is    = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 24.0
    atr_price = wilder_atr(high, low, close, ATR_PERIOD)

    sma = pd.Series(close).rolling(SMA_N).mean().values.astype(np.float64)
    thld_arr = THLD_MULT * atr_price

    for act in ACTIVATE_MULTS:
        for trail in TRAIL_MULTS:
            p, h, t = _sim_atrail(
                close[n_is:], sma[n_is:],
                high[n_is:], low[n_is:],
                bid[n_is:], ask[n_is:], sp[n_is:], atr_price[n_is:],
                pip, thld_arr[n_is:],
                int(N_SMALL), float(act), float(trail), int(MAX_HOLD_H1), float(sg))
            n = len(p)
            if n == 0:
                all_rows.append(dict(pair=pair, activate=act, trail=trail,
                                     n=0, ppd=0.0, wr=0.0, mdd=0.0, calmar=0.0,
                                     mean_hold_h1=0.0, max_loss=0.0, max_win=0.0,
                                     days=round(oos_days,1)))
                continue
            wr  = (p>0).sum() / n * 100
            ppd = p.sum() / oos_days
            mdd = max_dd(p)
            cal = ppd / mdd if mdd > 0 else 0.0
            all_rows.append(dict(pair=pair, activate=act, trail=trail,
                                 n=n, ppd=round(ppd,2), wr=round(wr,1),
                                 mdd=round(mdd,1), calmar=round(cal,2),
                                 mean_hold_h1=round(float(h.mean()),1),
                                 max_loss=round(float(p.min()),1),
                                 max_win=round(float(p.max()),1),
                                 days=round(oos_days,1)))
    del close, bid, ask, sp, h1, atr_price, high, low, sma, thld_arr
    gc.collect()


def main():
    warmup_jit()
    print("v3 extension — tighter trail sweep")
    print(f"  fixed entry: sma_n={SMA_N} n_small={N_SMALL} thld_mult={THLD_MULT}×ATR")
    print(f"  activate ∈ {ACTIVATE_MULTS}")
    print(f"  trail    ∈ {TRAIL_MULTS}")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "h1_tight_trail.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Summary: 12-pair portfolio top configs ─────────────────────────────
    print("\n" + "="*90)
    print("  Top configs ALL 12 pairs (sorted by Σ p/d)")
    print("="*90)
    g = (df.groupby(["activate","trail"])
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"),
                 worst_max_loss=("max_loss","min"))
            .reset_index().sort_values("sum_ppd", ascending=False))
    print(f"  {'act':>4}  {'trail':>5}  {'Σ ppd':>7}  {'pos':>3}/12 {'Σn':>5}  {'WR%':>5}  {'worst trade':>12}")
    for _, r in g.iterrows():
        print(f"  {r['activate']:>4.2f}  {r['trail']:>5.2f}  {r['sum_ppd']:>+7.2f}  "
              f"{int(r['n_pos']):>3d}/12 {int(r['total_n']):>5d}  "
              f"{r['mean_wr']:>5.1f}  {r['worst_max_loss']:>+12.1f}")

    # ── 7-winner subset rankings ────────────────────────────────────────────
    df_sub = df[df.pair.isin(WINNERS_7)]
    g_sub = (df_sub.groupby(["activate","trail"])
                .agg(sum_ppd=("ppd","sum"),
                     n_pos=("ppd", lambda x: int((x>0).sum())),
                     total_n=("n","sum"),
                     mean_wr=("wr","mean"),
                     worst_max_loss=("max_loss","min"))
                .reset_index().sort_values("sum_ppd", ascending=False))
    print("\n" + "="*90)
    print(f"  Top configs on 7-WINNER subset ({sorted(WINNERS_7)})")
    print("="*90)
    print(f"  {'act':>4}  {'trail':>5}  {'Σ ppd':>7}  {'pos':>3}/7 {'Σn':>5}  {'WR%':>5}  {'worst trade':>12}")
    for _, r in g_sub.head(15).iterrows():
        print(f"  {r['activate']:>4.2f}  {r['trail']:>5.2f}  {r['sum_ppd']:>+7.2f}  "
              f"{int(r['n_pos']):>3d}/7  {int(r['total_n']):>5d}  "
              f"{r['mean_wr']:>5.1f}  {r['worst_max_loss']:>+12.1f}")

    # ── Per-pair detail at top 12-pair config ────────────────────────────
    top = g.iloc[0]
    print("\n" + "="*90)
    print(f"  Per-pair at top 12-config: act={top['activate']} trail={top['trail']}")
    print("="*90)
    sub = df[(df.activate==top['activate'])&(df.trail==top['trail'])]
    sub = sub.sort_values("ppd", ascending=False)
    print(f"  {'pair':<10}{'n':>4}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'hold':>6}{'max_loss':>10}{'max_win':>9}")
    for _, r in sub.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>4}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['mdd']:>7.1f}{r['mean_hold_h1']:>6.1f}"
              f"{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")

    top_sub = g_sub.iloc[0]
    print("\n" + "="*90)
    print(f"  Per-pair at top 7-subset config: act={top_sub['activate']} trail={top_sub['trail']}")
    print("="*90)
    sub2 = df_sub[(df_sub.activate==top_sub['activate'])&(df_sub.trail==top_sub['trail'])]
    sub2 = sub2.sort_values("ppd", ascending=False)
    print(f"  {'pair':<10}{'n':>4}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'hold':>6}{'max_loss':>10}{'max_win':>9}")
    for _, r in sub2.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>4}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
              f"{r['mdd']:>7.1f}{r['mean_hold_h1']:>6.1f}"
              f"{r['max_loss']:>+10.1f}{r['max_win']:>+9.1f}")


if __name__ == "__main__":
    main()
