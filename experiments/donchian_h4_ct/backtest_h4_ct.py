"""
H4 Donchian Counter-Trend: Fixed Pip SL + Vol Filter
=====================================================
Entry: H4 bar closes beyond outer Donchian band AND bar_range/ATR >= vol_thresh
  (0.0 = no filter).  Fade the spike back toward channel midline.

TP modes:
  0 — midline at entry time (channel midpoint; R:R > 1 since entry is outside
      the band while target is inside)
  1 — fixed pip target

SL: always fixed pips — the key improvement over Session 055 counter-trend
  (ATR trail inflated by spike volatility → SL >> TP → Calmar destroyed).

Within-bar sequencing: SOP R2 (bull bar → favourable exit first).
Spread gate: IS P90, bars above get sentinel 999.  SOP R5.

Survivors: OOS p/d > 0, OOS trades >= 15, Calmar > 0.30

Usage:
  cd /path/to/projects/fx-core
  python3 research/experiments/donchian_h4_ct/backtest_h4_ct.py
"""

import time, gc
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC    = 0.70
ATR_PER    = 14
H4_BPG     = 48       # M5 bars per H4 bar
H4_PER_DAY = 6.0      # calendar (24h / 4h)

PAIRS = [
    ("GBP_JPY", 0.01), ("USD_JPY", 0.01), ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001), ("AUD_JPY", 0.01),
    ("CHF_JPY", 0.01), ("NZD_JPY", 0.01), ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001), ("NZD_USD", 0.0001), ("EUR_GBP", 0.0001),
]

N_VALS    = [5, 10, 20, 40]
VOL_VALS  = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
SL_VALS   = [10.0, 15.0, 20.0, 30.0, 50.0]
TP_VALS   = [5.0, 10.0, 15.0, 20.0]   # fixed pip TP (mode 1 only)
MODE_VALS = [0, 1]                      # 0=midline, 1=fixed pip
HOLD_VALS = [4, 8, 24]                  # max hold in H4 bars

MIN_OOS_TRADES = 15
MIN_CALMAR     = 0.30


# ─────────────────────────────────────────────────────────────────────────────
def load_h4(pair: str, pip: float) -> pd.DataFrame:
    """Load M5 BA parquet, group into H4 bars (48 M5 bars each).
    Returns DataFrame with OHLC + spread in pips. Index is H4-group integer."""
    df = pd.read_parquet(BA_DIR / f"{pair}_M5_BA.parquet")
    n = len(df)
    df["_g"] = np.arange(n) // H4_BPG
    h4 = df.groupby("_g").agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low=("low",    "min"),
        close=("close", "last"),
        bid_c=("bid_c", "last"),
        ask_c=("ask_c", "last"),
    )
    h4["spread"] = (h4["ask_c"] - h4["bid_c"]) / pip
    return h4


# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _wilder_atr(high, low, close, period):
    n = len(close)
    atr = np.empty(n)
    atr[:] = np.nan
    seed = 0.0
    for j in range(1, period + 1):
        seed += max(high[j] - low[j],
                    abs(high[j] - close[j-1]),
                    abs(low[j]  - close[j-1]))
    atr[period] = seed / period
    for i in range(period + 1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * (period - 1) + tr) / period
    return atr


@nb.njit(cache=True)
def _donchian(high, low, n_period):
    """Causal: upper[i] = max(high[i-n:i]) — excludes bar i (SOP R1)."""
    bars = len(high)
    upper = np.full(bars, np.nan)
    lower = np.full(bars, np.nan)
    mid   = np.full(bars, np.nan)
    for i in range(n_period, bars):
        u = np.max(high[i - n_period : i])
        l = np.min(low[i  - n_period : i])
        upper[i] = u
        lower[i] = l
        mid[i]   = (u + l) * 0.5
    return upper, lower, mid


# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _run(open_p, high_p, low_p, close_p, spread_p,
         upper_p, lower_p, mid_p, atr_p,
         vol_thresh, sl, tp, tp_mode, max_hold, is_end):
    """
    Single config simulation.  All arrays in pip units.

    Entry: close beyond Donchian outer band + vol filter.
    TP (set once at entry):
      mode 0 — midline at entry bar  (always profitable direction because
                entry is outside band while midline is inside)
      mode 1 — fixed pip from entry
    SL: fixed pip (not ATR-trail — avoids inflated SL on spike entry).

    Within-bar sequencing (SOP R2):
      bull bar → check favourable exit before adverse
      bear bar → check adverse before favourable
      (favourable = TP for LONG / SL for SHORT; adverse = opposite)

    Spread deducted per trade at exit bar.  Bars with spread sentinel 999
    skip entries.
    """
    n      = len(close_p)
    warmup = ATR_PER + max(40, max_hold) + 5

    pos  = 0;    ep = 0.0;   ebar = 0
    tgt  = 0.0;  stp = 0.0
    eq   = 0.0;  peak = 0.0; max_dd = 0.0
    oos_p = 0.0; oos_t = 0;  is_t = 0

    for i in range(warmup, n):
        u   = upper_p[i]
        lo  = lower_p[i]
        m   = mid_p[i]
        atr = atr_p[i]
        sp  = spread_p[i]
        if u != u or lo != lo or atr != atr or atr <= 0.0:
            continue

        bar_range = high_p[i] - low_p[i]
        bull      = close_p[i] >= open_p[i]

        # ── Exit ─────────────────────────────────────────────────────────────
        if pos != 0:
            held   = i - ebar
            pnl    = 0.0
            exited = False

            if pos == 1:    # LONG: TP above, SL below
                if bull:    # bull bar → TP (high) first, SL (low) second
                    if high_p[i] >= tgt:
                        pnl = tgt - ep - sp;  exited = True
                    elif low_p[i] <= stp:
                        pnl = stp - ep - sp;  exited = True
                else:       # bear bar → SL (low) first, TP (high) second
                    if low_p[i] <= stp:
                        pnl = stp - ep - sp;  exited = True
                    elif high_p[i] >= tgt:
                        pnl = tgt - ep - sp;  exited = True
                if not exited and held >= max_hold:
                    pnl = close_p[i] - ep - sp;  exited = True

            else:           # SHORT: TP below, SL above
                if bull:    # bull bar → SL (high) first, TP (low) second
                    if high_p[i] >= stp:
                        pnl = ep - stp - sp;  exited = True
                    elif low_p[i] <= tgt:
                        pnl = ep - tgt - sp;  exited = True
                else:       # bear bar → TP (low) first, SL (high) second
                    if low_p[i] <= tgt:
                        pnl = ep - tgt - sp;  exited = True
                    elif high_p[i] >= stp:
                        pnl = ep - stp - sp;  exited = True
                if not exited and held >= max_hold:
                    pnl = ep - close_p[i] - sp;  exited = True

            if exited:
                pos = 0
                eq += pnl
                if eq > peak: peak = eq
                dd = peak - eq
                if dd > max_dd: max_dd = dd
                if i >= is_end:
                    oos_p += pnl;  oos_t += 1
                else:
                    is_t += 1

        # ── Entry (flat + not spread-gated) ──────────────────────────────────
        if pos == 0 and sp < 900.0:
            vol_ok = (vol_thresh <= 0.0) or (bar_range / atr >= vol_thresh)
            if vol_ok:
                c = close_p[i]
                if c < lo:     # LONG: close broke below lower band
                    pos  = 1
                    ep   = c;       ebar = i
                    stp  = ep - sl
                    tgt  = m if tp_mode == 0 else ep + tp
                elif c > u:    # SHORT: close broke above upper band
                    pos  = -1
                    ep   = c;       ebar = i
                    stp  = ep + sl
                    tgt  = m if tp_mode == 0 else ep - tp

    oos_days = (n - is_end) / H4_PER_DAY
    return oos_p, float(oos_t), max_dd, oos_days, float(is_t)


# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, cache=True)
def _sweep(open_p, high_p, low_p, close_p, spread_p,
           upper_p, lower_p, mid_p, atr_p,
           c_vol, c_sl, c_tp, c_mode, c_hold,
           is_end):
    """
    Parallel sweep for ONE N value.  Avoids 2D array indexing inside prange
    (which caused segfaults with Numba).  Called once per N from Python loop.
    """
    nc       = len(c_vol)
    oos_pips = np.zeros(nc)
    oos_t    = np.zeros(nc)
    max_dds  = np.zeros(nc)
    oos_days = np.zeros(nc)

    for k in prange(nc):
        r = _run(open_p, high_p, low_p, close_p, spread_p,
                 upper_p, lower_p, mid_p, atr_p,
                 c_vol[k], c_sl[k], c_tp[k], c_mode[k], c_hold[k],
                 is_end)
        oos_pips[k] = r[0]
        oos_t[k]    = r[1]
        max_dds[k]  = r[2]
        oos_days[k] = r[3]

    return oos_pips, oos_t, max_dds, oos_days


# ─────────────────────────────────────────────────────────────────────────────
def build_configs() -> dict:
    """Expand parameter grid.  Returns arrays keyed by (vol, sl, tp, mode, hold)."""
    vols  = []; sls = []; tps = []; modes = []; holds = []
    for vol in VOL_VALS:
        for sl in SL_VALS:
            for tp in TP_VALS:
                for mode in MODE_VALS:
                    for hold in HOLD_VALS:
                        vols.append(vol); sls.append(sl);   tps.append(tp)
                        modes.append(mode); holds.append(hold)
    return {
        "vol":  np.array(vols,  dtype=np.float64),
        "sl":   np.array(sls,   dtype=np.float64),
        "tp":   np.array(tps,   dtype=np.float64),
        "mode": np.array(modes, dtype=np.int64),
        "hold": np.array(holds, dtype=np.int64),
    }


def run_pair(pair: str, pip: float, configs: dict) -> pd.DataFrame:
    h4     = load_h4(pair, pip)
    n      = len(h4)
    is_end = int(n * IS_FRAC)

    op = (h4["open"].values  / pip).astype(np.float64)
    hi = (h4["high"].values  / pip).astype(np.float64)
    lo = (h4["low"].values   / pip).astype(np.float64)
    cl = (h4["close"].values / pip).astype(np.float64)

    sp_raw  = h4["spread"].values.astype(np.float64)
    sp_gate = np.percentile(sp_raw[:is_end], 90)
    sp      = np.where(sp_raw > sp_gate, 999.0, sp_raw)
    atr     = _wilder_atr(hi, lo, cl, ATR_PER)

    oos_days = (n - is_end) / H4_PER_DAY

    rows = []

    for nv in N_VALS:
        upper_p, lower_p, mid_p = _donchian(hi, lo, nv)

        op_p, oo_t, om_dd, od = _sweep(
            op, hi, lo, cl, sp,
            upper_p, lower_p, mid_p, atr,
            configs["vol"], configs["sl"], configs["tp"],
            configs["mode"], configs["hold"],
            is_end,
        )

        nc = len(configs["vol"])
        for k in range(nc):
            ot = oo_t[k]
            if ot < MIN_OOS_TRADES:
                continue
            pd_val = op_p[k] / oos_days if oos_days > 0 else 0.0
            if pd_val <= 0.0:
                continue
            calmar = (pd_val * 252) / om_dd[k] if om_dd[k] > 0 else 0.0
            if calmar < MIN_CALMAR:
                continue
            mode = configs["mode"][k]
            tp   = configs["tp"][k]
            tp_l = "mid" if mode == 0 else f"{int(tp)}p"
            rows.append({
                "pair":   pair,
                "N":      nv,
                "vol":    configs["vol"][k],
                "sl":     configs["sl"][k],
                "tp":     tp,
                "mode":   mode,
                "hold":   configs["hold"][k],
                "config": (f"N{nv}_v{configs['vol'][k]:.2f}_sl{int(configs['sl'][k])}"
                           f"_{tp_l}_h{configs['hold'][k]}"),
                "p_d":    round(pd_val, 2),
                "t_d":    round(ot / oos_days, 3),
                "MaxDD":  round(om_dd[k], 1),
                "Calmar": round(calmar, 2),
                "OOS_t":  int(ot),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    configs = build_configs()
    nc_per_N = len(configs["vol"])
    total_configs = len(N_VALS) * nc_per_N

    print("H4 Donchian Counter-Trend — Fixed SL + Vol Filter Sweep")
    print(f"Pairs: {len(PAIRS)}  |  N values: {N_VALS}")
    print(f"Configs per N: {nc_per_N}  |  Total per pair: {total_configs}")
    print(f"IS={IS_FRAC:.0%}  OOS={1-IS_FRAC:.0%}  "
          f"Min OOS trades={MIN_OOS_TRADES}  Min Calmar={MIN_CALMAR}\n")

    print("Compiling Numba kernels...", end=" ", flush=True)
    _dummy = np.ones(100, dtype=np.float64)
    _donchian(_dummy, _dummy, 5)
    _wilder_atr(_dummy, _dummy, _dummy, 14)
    print("done.\n")

    all_survivors  = []
    best_per_pair  = []

    for pair, pip in PAIRS:
        t1 = time.time()
        print(f"  {pair}...", end=" ", flush=True)
        try:
            df_surv = run_pair(pair, pip, configs)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        if df_surv.empty:
            print("0 survivors")
            continue

        all_survivors.append(df_surv)
        best = df_surv.sort_values("Calmar", ascending=False).iloc[0]
        best_per_pair.append(best)
        print(f"{len(df_surv):4d} survivors | best Calmar={best['Calmar']:.2f} "
              f"p/d={best['p_d']:.1f} t/d={best['t_d']:.3f} "
              f"MaxDD={best['MaxDD']:.0f}  [{best['config']}]  "
              f"({time.time()-t1:.1f}s)")
        gc.collect()

    print()
    print("─" * 95)
    print(f"PORTFOLIO — best per pair (Calmar-ranked), aggregate (target: t/d > 2)")
    print(f"{'Pair':<12} {'Config':<44} {'p/d':>6} {'t/d':>6} {'MaxDD':>7} {'Calmar':>7}")
    print("─" * 95)

    total_pd = total_td = 0.0
    for r in sorted(best_per_pair, key=lambda x: x["Calmar"], reverse=True):
        print(f"{r['pair']:<12} {r['config']:<44} {r['p_d']:>6.1f} "
              f"{r['t_d']:>6.3f} {r['MaxDD']:>7.1f} {r['Calmar']:>7.2f}")
        total_pd += r["p_d"]
        total_td += r["t_d"]

    print("─" * 95)
    ok = "✅" if total_td >= 2.0 else "❌"
    print(f"{'TOTAL':<12} {'':44} {total_pd:>6.1f} {total_td:>6.3f}")
    print(f"> 2 t/d target: {ok} ({total_td:.3f} t/d)\n")
    print(f"Total runtime: {time.time()-t0:.1f}s")

    if all_survivors:
        out = Path(__file__).parent / "results_h4_ct.csv"
        pd.concat(all_survivors, ignore_index=True).to_csv(out, index=False)
        print(f"Survivors → {out}")


if __name__ == "__main__":
    main()
