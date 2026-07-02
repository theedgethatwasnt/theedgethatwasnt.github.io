"""
Multi-Timeframe Donchian Backtest v2
=====================================
Two strategy modes tested in same sweep:

MODE 0 — TREND: Enter on LTF Donchian breakout (close > N-bar high),
  filtered by HTF Donchian midline direction.
  Entry gated to first M5 bar of each LTF period (one attempt per LTF bar).

MODE 1 — COUNTER-TREND: Enter when M5 close touches/penetrates the LTF
  Donchian LOWER band while HTF is bullish (or UPPER band while HTF bearish).
  Targets Donchian midline as TP.

TF combos:
  M5 → H1   (bpg=1 / 12)
  M5 → H4   (bpg=1 / 48)
  M30 → H4  (bpg=6 / 48)
  H1 → H4   (bpg=12 / 48)

SOP R1/R3/R4/R5/R8 — see CLAUDE.md.

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/donchian_mtf/backtest_donchian_mtf.py
"""

import time, gc
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE    = Path(__file__).resolve().parents[3]
BA_DIR  = BASE / "data/m5_ba"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

IS_FRAC             = 0.70
M5_PER_TRADING_DAY  = 288

PAIRS = [
    ("GBP_JPY", 0.01), ("USD_JPY", 0.01), ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001), ("AUD_JPY", 0.01),
    ("CHF_JPY", 0.01), ("NZD_JPY", 0.01), ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001), ("NZD_USD", 0.0001), ("EUR_GBP", 0.0001),
]

# (ltf_label, htf_label, ltf_m5_bars, htf_m5_bars)
TF_COMBOS = [
    ("M5",  "H1",  1,  12),
    ("M5",  "H4",  1,  48),
    ("M30", "H4",  6,  48),
    ("H1",  "H4",  12, 48),
]

N_LTF_VALS    = np.array([5, 10, 20],           dtype=np.int32)
N_HTF_VALS    = np.array([10, 20, 40],          dtype=np.int32)
TRAIL_VALS    = np.array([1.0, 1.5, 2.0, 2.5],  dtype=np.float64)
MAX_HOLD_VALS = np.array([24, 48, 96],          dtype=np.int32)   # in LTF bars
MODE_VALS     = np.array([0, 1],                dtype=np.int32)   # 0=trend, 1=counter

MIN_TRADES_IS  = 20
MIN_TRADES_OOS = 8


# ─── ATR (Wilder, in pip units) ───────────────────────────────────────────────
def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    n = len(high)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = high[0] - low[0]
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * (period - 1) + tr) / period
    return atr


# ─── Feature computation ──────────────────────────────────────────────────────
def make_ltf_features(df_m5: pd.DataFrame, bpg: int, n_ltf: int, pip: float
                      ) -> tuple:
    """Causal LTF Donchian hi/lo/mid in PIP units at M5 resolution. Shift(1)."""
    n = len(df_m5)
    if bpg == 1:
        dhi = df_m5['high'].rolling(n_ltf, min_periods=n_ltf).max().shift(1).values / pip
        dlo = df_m5['low'].rolling(n_ltf,  min_periods=n_ltf).min().shift(1).values / pip
    else:
        df_m5 = df_m5.copy()
        df_m5['_g'] = np.arange(n) // bpg
        ltf = df_m5.groupby('_g').agg(hi=('high', 'max'), lo=('low', 'min'))
        dhi_g = ltf['hi'].rolling(n_ltf, min_periods=n_ltf).max().shift(1).values / pip
        dlo_g = ltf['lo'].rolling(n_ltf, min_periods=n_ltf).min().shift(1).values / pip
        dhi = np.full(n, np.nan); dlo = np.full(n, np.nan)
        for g in range(len(ltf)):
            s = g * bpg; e = min(s + bpg, n)
            dhi[s:e] = dhi_g[g]; dlo[s:e] = dlo_g[g]

    dmid = np.where(np.isnan(dhi), np.nan, (dhi + dlo) / 2.0)
    return dhi, dlo, dmid


def make_htf_mid(df_m5: pd.DataFrame, bpg: int, n_htf: int, pip: float
                 ) -> np.ndarray:
    """Causal HTF Donchian midline in PIP units at M5 resolution. Shift(1)."""
    n = len(df_m5)
    df_m5 = df_m5.copy()
    df_m5['_g'] = np.arange(n) // bpg
    htf = df_m5.groupby('_g').agg(hi=('high', 'max'), lo=('low', 'min'))
    dhi = htf['hi'].rolling(n_htf, min_periods=n_htf).max().shift(1).values
    dlo = htf['lo'].rolling(n_htf, min_periods=n_htf).min().shift(1).values
    mid_g = np.where(np.isnan(dhi), np.nan, (dhi + dlo) / 2.0) / pip
    mid = np.full(n, np.nan)
    for g in range(len(htf)):
        s = g * bpg; e = min(s + bpg, n)
        mid[s:e] = mid_g[g]
    return mid


# ─── Numba kernel ─────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _run(close_p, spread_p, atr_p,
         ltf_hi_p, ltf_lo_p, ltf_mid_p, htf_mid_p,
         trail, max_hold_ltf, ltf_bpg, mode):
    """
    All arrays: pip units.
    mode 0 = trend breakout; mode 1 = counter-trend band-touch.
    Entry gated: for bpg>1, only first M5 bar of each LTF period checks entry.
    max_hold_ltf is in LTF bars; converted internally to M5 bars.
    Returns (n_trades, total_pips, max_drawdown, n_days).
    """
    n          = len(close_p)
    max_hold_m5 = max_hold_ltf * ltf_bpg
    pos        = 0; entry_p = 0.0; peak_p = 0.0; stop_p = 0.0
    held_m5    = 0; total = 0.0; eq = 0.0; peak_eq = 0.0
    max_dd     = 0.0; nt = 0

    for i in range(1, n):
        cl = close_p[i]; sp = spread_p[i]; at = atr_p[i]
        lh = ltf_hi_p[i]; ll = ltf_lo_p[i]; lm = ltf_mid_p[i]
        hm = htf_mid_p[i]

        # Skip bars with invalid data
        if at <= 0.0 or sp < 0.0 or sp > 50.0:
            continue
        # NaN guard (stored as sentinel: lh=1e9, ll=-1e9, lm=0, hm=0)
        if lh > 1e8 or ll < -1e8:
            continue
        if hm == 0.0:  # NaN HTF mid sentinel
            continue

        # ── Manage open position ──────────────────────────────────────────────
        if pos != 0:
            held_m5 += 1
            if mode == 0:  # trend: ATR trail
                if pos == 1:
                    if cl > peak_p: peak_p = cl; stop_p = peak_p - trail * at
                    exit_now = cl <= stop_p or held_m5 >= max_hold_m5
                else:
                    if cl < peak_p: peak_p = cl; stop_p = peak_p + trail * at
                    exit_now = cl >= stop_p or held_m5 >= max_hold_m5
            else:  # counter: exit at LTF midline
                if pos == 1:
                    exit_now = cl >= lm or cl <= (entry_p - trail * at) or held_m5 >= max_hold_m5
                else:
                    exit_now = cl <= lm or cl >= (entry_p + trail * at) or held_m5 >= max_hold_m5

            if exit_now:
                pnl = (cl - sp * 0.5 - entry_p) * pos
                total += pnl; eq += pnl; nt += 1
                if eq > peak_eq: peak_eq = eq
                dd = peak_eq - eq
                if dd > max_dd: max_dd = dd
                pos = 0; held_m5 = 0
                continue

        # ── Entry check ───────────────────────────────────────────────────────
        # Gate: only first M5 bar of each LTF period for multi-bar LTF
        is_ltf_open = (ltf_bpg == 1) or (i % ltf_bpg == 0)

        if pos == 0 and is_ltf_open:
            if mode == 0:  # trend breakout
                if cl > lh and cl > hm:
                    pos = 1; entry_p = cl + sp * 0.5
                    peak_p = cl; stop_p = cl - trail * at; held_m5 = 0
                elif cl < ll and cl < hm:
                    pos = -1; entry_p = cl - sp * 0.5
                    peak_p = cl; stop_p = cl + trail * at; held_m5 = 0
            else:  # counter-trend: touch outer band, trade toward midline
                if cl <= ll and cl > hm:   # at/below lower band + HTF bullish → LONG
                    pos = 1; entry_p = cl + sp * 0.5
                    peak_p = cl; stop_p = cl - trail * at; held_m5 = 0
                elif cl >= lh and cl < hm: # at/above upper band + HTF bearish → SHORT
                    pos = -1; entry_p = cl - sp * 0.5
                    peak_p = cl; stop_p = cl + trail * at; held_m5 = 0

    # Close open at end
    if pos != 0 and held_m5 > 0:
        pnl = (close_p[-1] - spread_p[-1] * 0.5 - entry_p) * pos
        total += pnl; nt += 1

    n_days = n / float(M5_PER_TRADING_DAY)
    return nt, total, max_dd, n_days


@nb.njit(parallel=True, cache=True)
def _sweep(close_p, spread_p, atr_p,
           lhi_all, llo_all, lmid_all, hm_all,
           trails, holds, modes,
           is_end, ltf_bpg,
           n_a, n_b, n_c, n_d, n_e):
    total = n_a * n_b * n_c * n_d * n_e
    out   = np.empty((total, 9), dtype=np.float64)
    for k in prange(total):
        a = k // (n_b * n_c * n_d * n_e)
        b = (k % (n_b * n_c * n_d * n_e)) // (n_c * n_d * n_e)
        c = (k % (n_c * n_d * n_e)) // (n_d * n_e)
        d = (k % (n_d * n_e)) // n_e
        e = k % n_e

        nt, tp, md, nd = _run(
            close_p[:is_end], spread_p[:is_end], atr_p[:is_end],
            lhi_all[a, :is_end], llo_all[a, :is_end], lmid_all[a, :is_end],
            hm_all[b, :is_end],
            trails[c], holds[d], ltf_bpg, modes[e]
        )
        pd_ = tp / nd if nd > 0.0 else 0.0
        out[k, 0] = a; out[k, 1] = b; out[k, 2] = c
        out[k, 3] = d; out[k, 4] = e
        out[k, 5] = nt; out[k, 6] = tp; out[k, 7] = pd_; out[k, 8] = md
    return out


# ─── Per-pair ─────────────────────────────────────────────────────────────────
def run_pair(pair: str, pip: float) -> list:
    fpath = BA_DIR / f"{pair}_M5_BA.parquet"
    if not fpath.exists():
        print(f"  SKIP {pair}: no file"); return []

    df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
    n  = len(df); is_end = int(n * IS_FRAC)

    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    cl  = df['close'].values.astype(np.float64)
    sp  = (df['ask_c'] - df['bid_c']).values.astype(np.float64)

    cl_p  = cl / pip
    sp_p  = sp / pip
    atr_p = wilder_atr(hi, lo, cl, 14) / pip

    sp_gate   = float(np.percentile(sp_p[:is_end], 90))
    sp_filt   = np.where(sp_p > sp_gate, 999.0, sp_p)

    results = []

    for ltf_lbl, htf_lbl, ltf_bpg, htf_bpg in TF_COMBOS:
        t0 = time.time()

        n_nl = len(N_LTF_VALS); n_nh = len(N_HTF_VALS)

        # Precompute LTF arrays (pip units)
        lhi_all  = np.full((n_nl, n), 1e9)
        llo_all  = np.full((n_nl, n), -1e9)
        lmid_all = np.zeros((n_nl, n))
        for ai, nl in enumerate(N_LTF_VALS):
            dhi, dlo, dmid = make_ltf_features(df, ltf_bpg, int(nl), pip)
            lhi_all[ai]  = np.where(np.isnan(dhi),  1e9,  dhi)
            llo_all[ai]  = np.where(np.isnan(dlo),  -1e9, dlo)
            lmid_all[ai] = np.where(np.isnan(dmid), 0.0,  dmid)

        # Precompute HTF midline (pip units)
        hm_all = np.zeros((n_nh, n))
        for bi, nh in enumerate(N_HTF_VALS):
            mid = make_htf_mid(df, htf_bpg, int(nh), pip)
            hm_all[bi] = np.where(np.isnan(mid), 0.0, mid)

        raw = _sweep(
            cl_p, sp_filt, atr_p,
            lhi_all, llo_all, lmid_all, hm_all,
            TRAIL_VALS, MAX_HOLD_VALS, MODE_VALS,
            is_end, ltf_bpg,
            n_nl, n_nh, len(TRAIL_VALS), len(MAX_HOLD_VALS), len(MODE_VALS)
        )

        n_surv = 0
        for row in raw:
            ai, bi, ci, di, ei = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
            nt_is = int(row[5]); pd_is = row[7]; md_is = row[8]
            if nt_is < MIN_TRADES_IS or pd_is <= 0.0:
                continue

            trail = TRAIL_VALS[ci]; hold = int(MAX_HOLD_VALS[di]); mode = int(MODE_VALS[ei])

            nt_oos, tp_oos, md_oos, nd_oos = _run(
                cl_p[is_end:], sp_filt[is_end:], atr_p[is_end:],
                lhi_all[ai, is_end:], llo_all[ai, is_end:], lmid_all[ai, is_end:],
                hm_all[bi, is_end:],
                trail, int(hold), ltf_bpg, mode
            )

            if nt_oos < MIN_TRADES_OOS or tp_oos <= 0.0:
                continue

            nd_oos_days = nd_oos
            pd_oos  = tp_oos / nd_oos_days if nd_oos_days > 0 else 0.0
            tpd_oos = nt_oos / nd_oos_days if nd_oos_days > 0 else 0.0
            calmar  = pd_oos / md_oos if md_oos > 0 else 0.0

            results.append({
                "pair": pair, "ltf": ltf_lbl, "htf": htf_lbl,
                "strat": "trend" if mode == 0 else "counter",
                "n_ltf": int(N_LTF_VALS[ai]), "n_htf": int(N_HTF_VALS[bi]),
                "trail": trail, "max_hold": int(hold), "sp_gate": round(sp_gate, 2),
                "nt_is": nt_is,  "pd_is": round(pd_is, 1),
                "nt_oos": nt_oos, "tp_oos": round(tp_oos, 1),
                "pd_oos": round(pd_oos, 1), "tpd_oos": round(tpd_oos, 2),
                "md_oos": round(md_oos, 1), "calmar": round(calmar, 2),
            })
            n_surv += 1

        elapsed = time.time() - t0
        print(f"  {pair} {ltf_lbl}→{htf_lbl}: {n_surv} survivors ({elapsed:.1f}s)")
        del lhi_all, llo_all, lmid_all, hm_all; gc.collect()

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_rows = []
    for pair, pip in PAIRS:
        print(f"\n{'='*60}\n  {pair}")
        rows = run_pair(pair, pip)
        all_rows.extend(rows)
        gc.collect()

    if not all_rows:
        print("\nNo OOS survivors."); return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "all_survivors.csv", index=False)
    df['score'] = df['pd_oos'] * np.sqrt(df['calmar'].clip(0))

    print("\n" + "="*85)
    print("BEST PER PAIR × TF × MODE (by OOS p/d)")
    print("="*85)
    hdr = (f"{'Pair':10} {'LTF':4} {'HTF':4} {'Strat':8} {'NL':3} {'NH':3} "
           f"{'Tr':4} {'MH':3} {'p/d_IS':7} {'p/d_OOS':8} {'t/d':5} {'MaxDD':6} {'Calmar':6}")
    print(hdr); print("-"*len(hdr))
    for _, grp in df.groupby(["pair", "ltf", "htf", "strat"]):
        r = grp.nlargest(1, "pd_oos").iloc[0]
        print(f"{r.pair:10} {r.ltf:4} {r.htf:4} {r.strat:8} {r.n_ltf:3} {r.n_htf:3} "
              f"{r.trail:4.1f} {r.max_hold:3} {r.pd_is:7.1f} {r.pd_oos:8.1f} "
              f"{r.tpd_oos:5.2f} {r.md_oos:6.1f} {r.calmar:6.2f}")

    print("\n" + "="*85)
    print("TOP 25 OVERALL (pd_oos × √calmar)")
    print("="*85)
    for _, r in df.nlargest(25, 'score').iterrows():
        print(f"{r.pair:10} {r.ltf}→{r.htf} {r.strat:8} N={r.n_ltf}/{r.n_htf} "
              f"tr={r.trail:.1f} mh={r.max_hold}  "
              f"p/d={r.pd_oos:+.1f}  t/d={r.tpd_oos:.2f}  "
              f"MaxDD={r.md_oos:.1f}  Calmar={r.calmar:.2f}")

    print("\n" + "="*85)
    print("PORTFOLIO — best per pair, aggregate t/d (target > 2/day)")
    print("="*85)
    best = (df.sort_values('pd_oos', ascending=False)
              .groupby('pair').first().reset_index()
              .sort_values('pd_oos', ascending=False))
    total_tpd = best['tpd_oos'].sum()
    total_ppd = best['pd_oos'].sum()
    print(f"{'Pair':10} {'Config':32} {'p/d':7} {'t/d':5} {'MaxDD':7}")
    print("-"*70)
    for _, r in best.iterrows():
        cfg = f"{r.ltf}→{r.htf} {r.strat} N={r.n_ltf}/{r.n_htf} tr={r.trail:.1f}"
        print(f"{r.pair:10} {cfg:32} {r.pd_oos:7.1f} {r.tpd_oos:5.2f} {r.md_oos:7.1f}")
    print("-"*70)
    print(f"{'TOTAL':43} {total_ppd:7.1f} {total_tpd:5.2f}")
    status = "✅ MET" if total_tpd >= 2.0 else "❌ NOT MET"
    print(f"\n  > 2 trades/day target: {status} ({total_tpd:.2f} t/d)")
    df.to_csv(OUT_DIR / "all_survivors.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
