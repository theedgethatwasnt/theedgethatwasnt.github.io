#!/usr/bin/env python3
"""
(b) MTF state-agreement at H1 + H4 (slower TFs).

Same logic but on H1 bars with:
  h1_mom = close[t]   - close[t-1]    (1 H1 bar back)
  h4_mom = close[t]   - close[t-4]    (4 H1 bars back = H4-equivalent)

Reduces noise drastically: ~24 bars/day vs 288 at M5. Should kill the
spread-cost problem of the M5 version.

Same rule, no fixed TP/SL, no time stop. Pay spread on each H1 transition.
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
H1_LAG  = 1   # 1 H1 bar
H4_LAG  = 4   # 4 H1 bars = H4-equivalent

def pip_sz(p): return 0.01 if p in JPY else 0.0001

def resample_h1(df_m5):
    o = df_m5["open"].resample("1h").first()
    h = df_m5["high"].resample("1h").max()
    l = df_m5["low"].resample("1h").min()
    c = df_m5["close"].resample("1h").last()
    bc = df_m5["bid_c"].resample("1h").last()
    ac = df_m5["ask_c"].resample("1h").last()
    out = pd.concat([o,h,l,c,bc,ac], axis=1)
    out.columns = ["open","high","low","close","bid_c","ask_c"]
    return out.dropna()


@njit(cache=True)
def sim_mtf_state(close, bid, ask, sp, pip, sp_gate, lag_short, lag_long):
    n = len(close)
    pnl_arr = np.empty(n, dtype=np.float64)
    hold_arr= np.empty(n, dtype=np.int32)
    dir_arr = np.empty(n, dtype=np.int8)
    count = 0
    state_prev = 0
    entry_px = 0.0; entry_i = 0
    start = max(lag_short, lag_long) + 1
    for i in range(start, n):
        s_diff = close[i] - close[i - lag_short]
        l_diff = close[i] - close[i - lag_long]
        s = 0
        if   s_diff > 0.0 and l_diff > 0.0: s =  1
        elif s_diff < 0.0 and l_diff < 0.0: s = -1

        if s != state_prev:
            if state_prev != 0:
                exit_px = bid[i] if state_prev == 1 else ask[i]
                pnl_arr[count]  = (exit_px - entry_px) / pip * state_prev - sp[i]
                hold_arr[count] = i - entry_i
                dir_arr[count]  = state_prev
                count += 1
            if s != 0 and sp[i] <= sp_gate:
                entry_px = ask[i] if s == 1 else bid[i]
                entry_i  = i
            else:
                s = 0
            state_prev = s

    if state_prev != 0:
        i = n - 1
        exit_px = bid[i] if state_prev == 1 else ask[i]
        pnl_arr[count]  = (exit_px - entry_px) / pip * state_prev - sp[i]
        hold_arr[count] = i - entry_i
        dir_arr[count]  = state_prev
        count += 1

    return pnl_arr[:count], hold_arr[:count], dir_arr[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 200; c = np.linspace(1.0, 1.05, n)
    b = c - 0.0002; a = c + 0.0002; sp = np.ones(n)
    sim_mtf_state(c, b, a, sp, 0.0001, 2.0, 1, 4)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    h1 = resample_h1(df_m5); del df_m5
    pip = pip_sz(pair); sg = SP_GATES[pair]
    close = h1["close"].values.astype(np.float64)
    bid   = h1["bid_c"].values.astype(np.float64)
    ask   = h1["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 24.0   # H1 bars/day

    p, h, d = sim_mtf_state(
        close[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
        pip, float(sg), H1_LAG, H4_LAG)
    n = len(p)
    if n == 0:
        all_rows.append(dict(pair=pair, n=0, ppd=0.0, wr=0.0,
                             mdd=0.0, mean_hold_h1=0.0, mean_pnl=0.0,
                             max_loss=0.0, max_win=0.0, sum_pips=0.0,
                             days=round(oos_days,1)))
        return
    wr  = (p>0).sum() / n * 100
    ppd = p.sum() / oos_days
    mdd = max_dd(p)
    all_rows.append(dict(
        pair=pair, n=n,
        ppd=round(ppd, 2), wr=round(wr, 1),
        mdd=round(mdd, 1),
        mean_hold_h1=round(float(h.mean()), 1),
        mean_pnl=round(float(p.mean()), 2),
        max_loss=round(float(p.min()), 1),
        max_win=round(float(p.max()), 1),
        sum_pips=round(float(p.sum()), 1),
        days=round(oos_days, 1),
    ))


def main():
    warmup_jit()
    print("(b) MTF H1+H4 state-agreement")
    print(f"  rule: long while c[t]>c[t-1] AND c[t]>c[t-4]  (H1 bars)")
    print(f"        short while opposite; flat otherwise")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        n = rows[-1]['n']
        print(f"  {pair}: {time.time()-ts:.1f}s  trades={n}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "mtf_state_h1h4.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Per-pair detail ──────────────────────────────────────────────────
    df = df.sort_values("ppd", ascending=False)
    print("\n" + "="*100)
    print("  Per-pair OOS (sorted by p/d):")
    print("="*100)
    print(f"  {'pair':<10}{'n':>6}{'Σ pips':>10}{'ppd':>8}{'WR%':>6}"
          f"{'hold(H1)':>10}{'MDD':>8}{'max_loss':>10}{'max_win':>10}")
    for _, r in df.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>6}{r['sum_pips']:>+10.1f}{r['ppd']:>+8.2f}"
              f"{r['wr']:>6.1f}{r['mean_hold_h1']:>10.1f}{r['mdd']:>8.1f}"
              f"{r['max_loss']:>+10.1f}{r['max_win']:>+10.1f}")

    print("\n" + "="*100)
    total_pips = df.sum_pips.sum()
    avg_days = df.days.mean()
    portfolio_ppd = total_pips / avg_days
    print(f"  Portfolio: Σ pips={total_pips:+.1f}  ppd={portfolio_ppd:+.2f}  "
          f"Σ trades={int(df.n.sum())}  pairs+={int((df.ppd>0).sum())}/12  "
          f"mean WR={df.wr.mean():.1f}%")


if __name__ == "__main__":
    main()
