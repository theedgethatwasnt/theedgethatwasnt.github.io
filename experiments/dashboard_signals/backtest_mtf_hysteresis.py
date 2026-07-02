#!/usr/bin/env python3
"""
(a) MTF state-agreement WITH HYSTERESIS.

Same M5+H1 rule as backtest_mtf_state_agreement.py but the position only
flips after N consecutive M5 bars of disagreement with the prior state.
Sweep N ∈ {3, 6, 12, 24}. Goal: cut the 358k-trade noise sieve down to a
manageable trade rate.

For each bar i:
  raw_state = +1 if c[t]>c[t-1] AND c[t]>c[t-12]
              -1 if both negative
               0 otherwise (disagreement)
  if raw_state != held_state:
      mismatch_counter += 1
      if mismatch_counter >= N:
          held_state = raw_state; mismatch_counter = 0
  else:
      mismatch_counter = 0
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}
IS_FRAC = 0.70
M5_LAG = 1
H1_LAG = 12
HYSTERESIS_NS = [3, 6, 12, 24]

def pip_sz(p): return 0.01 if p in JPY else 0.0001


@njit(cache=True)
def sim_mtf_hyst(close, bid, ask, sp, pip, sp_gate, m5_lag, h1_lag, hysteresis_n):
    n = len(close)
    pnl_arr = np.empty(n, dtype=np.float64)
    hold_arr= np.empty(n, dtype=np.int32)
    dir_arr = np.empty(n, dtype=np.int8)
    count = 0
    held_state = 0
    mismatch_count = 0
    entry_px = 0.0; entry_i = 0

    start = max(h1_lag, m5_lag) + 1
    for i in range(start, n):
        m5_diff = close[i] - close[i - m5_lag]
        h1_diff = close[i] - close[i - h1_lag]
        raw = 0
        if   m5_diff > 0.0 and h1_diff > 0.0: raw =  1
        elif m5_diff < 0.0 and h1_diff < 0.0: raw = -1

        # Hysteresis: only update held_state after N bars of mismatch
        if raw != held_state:
            mismatch_count += 1
            if mismatch_count >= hysteresis_n:
                # Close existing position if any
                if held_state != 0:
                    exit_px = bid[i] if held_state == 1 else ask[i]
                    pnl_arr[count]  = (exit_px - entry_px) / pip * held_state - sp[i]
                    hold_arr[count] = i - entry_i
                    dir_arr[count]  = held_state
                    count += 1
                # Open new position if non-flat and spread acceptable
                new_state = raw
                if new_state != 0:
                    if sp[i] <= sp_gate:
                        entry_px = ask[i] if new_state == 1 else bid[i]
                        entry_i  = i
                    else:
                        new_state = 0
                held_state = new_state
                mismatch_count = 0
        else:
            mismatch_count = 0

    # Close final
    if held_state != 0:
        i = n - 1
        exit_px = bid[i] if held_state == 1 else ask[i]
        pnl_arr[count]  = (exit_px - entry_px) / pip * held_state - sp[i]
        hold_arr[count] = i - entry_i
        dir_arr[count]  = held_state
        count += 1

    return pnl_arr[:count], hold_arr[:count], dir_arr[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 200; c = np.linspace(1.0, 1.05, n)
    b = c - 0.0002; a = c + 0.0002; sp = np.ones(n)
    sim_mtf_hyst(c, b, a, sp, 0.0001, 2.0, 1, 12, 3)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]
    close = df_m5["close"].values.astype(np.float64)
    bid   = df_m5["bid_c"].values.astype(np.float64)
    ask   = df_m5["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 288.0

    for H in HYSTERESIS_NS:
        p, h, d = sim_mtf_hyst(
            close[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
            pip, float(sg), M5_LAG, H1_LAG, int(H))
        n = len(p)
        if n == 0:
            all_rows.append(dict(pair=pair, hyst=H, n=0, ppd=0.0, wr=0.0,
                                 mdd=0.0, mean_hold_m5=0.0, mean_pnl=0.0,
                                 max_loss=0.0, max_win=0.0, sum_pips=0.0,
                                 days=round(oos_days,1)))
            continue
        wr = (p>0).sum() / n * 100
        ppd = p.sum() / oos_days
        mdd = max_dd(p)
        all_rows.append(dict(pair=pair, hyst=H, n=n,
                             ppd=round(ppd, 2),
                             wr=round(wr, 1),
                             mdd=round(mdd, 1),
                             mean_hold_m5=round(float(h.mean()), 1),
                             mean_pnl=round(float(p.mean()), 2),
                             max_loss=round(float(p.min()), 1),
                             max_win=round(float(p.max()), 1),
                             sum_pips=round(float(p.sum()), 1),
                             days=round(oos_days, 1)))
    del df_m5, close, bid, ask, sp; gc.collect()


def main():
    warmup_jit()
    print("(a) MTF M5+H1 agreement WITH HYSTERESIS")
    print(f"  Hysteresis N ∈ {HYSTERESIS_NS} M5-bars of disagreement before flip")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "mtf_state_hysteresis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Portfolio totals per hysteresis N ─────────────────────────────────
    print("\n" + "="*100)
    print("  Portfolio totals (12 pairs) by hysteresis window")
    print("="*100)
    for H in HYSTERESIS_NS:
        sub = df[df.hyst == H]
        total = sub.sum_pips.sum()
        avg_days = sub.days.mean()
        ppd = total / avg_days
        n_total = sub.n.sum()
        wr_mean = sub.wr.mean()
        n_pos = int((sub.ppd > 0).sum())
        mean_hold = sub.mean_hold_m5.mean()
        print(f"  hyst={H:>2}: Σ pips={total:>+9.1f}  ppd={ppd:>+7.2f}  "
              f"Σ trades={int(n_total):>6}  mean WR={wr_mean:>5.1f}%  "
              f"pairs+={n_pos}/12  hold={mean_hold:.0f} M5 bars")

    # Per-pair detail at the best hysteresis
    best_h = max(HYSTERESIS_NS, key=lambda H: df[df.hyst == H].sum_pips.sum())
    print("\n" + "="*100)
    print(f"  Per-pair detail at best hysteresis N={best_h}")
    print("="*100)
    sub = df[df.hyst == best_h].sort_values("ppd", ascending=False)
    print(f"  {'pair':<10}{'n':>6}{'Σ pips':>10}{'ppd':>8}{'WR%':>6}{'hold(M5)':>10}{'max_loss':>10}{'max_win':>10}")
    for _, r in sub.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>6}{r['sum_pips']:>+10.1f}{r['ppd']:>+8.2f}"
              f"{r['wr']:>6.1f}{r['mean_hold_m5']:>10.1f}{r['max_loss']:>+10.1f}{r['max_win']:>+10.1f}")


if __name__ == "__main__":
    main()
