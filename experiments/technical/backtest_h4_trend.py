"""
H4 Trend-Following Strategy Study — Session 053.

Two classic strategies on GBP_JPY, USD_JPY, EUR_JPY, GBP_USD:
  A. Dual SMA cross + ATR(14) trailing stop
     Entry:  fast SMA crosses slow SMA → enter at next bar open
     Exit:   trail from peak by atr_trail × ATR, OR opposite cross
     Params: fast ∈ {5,10,20}  slow ∈ {20,50,100}  atr_sl ∈ {1,2,3}  atr_trail ∈ {1,2,3}

  B. Donchian N-bar breakout + ATR(14) trailing stop
     Entry:  close > N-bar highest high → LONG; close < N-bar lowest low → SHORT
     Exit:   trail from peak by atr_trail × ATR, OR opposite N-bar signal
     Params: N ∈ {10,20,50}  atr_sl ∈ {1,2,3}  atr_trail ∈ {1,2,3}

P/L in pips. Spread deducted at entry only (round-trip = 2× spread).
Validation: IS=3/3 + OOS=3/3 + bootstrap P5>0 + P(+)>0.95 (n_boot=2000).
H4 bars derived from M5 via timestamp resample (handles weekend/holiday gaps).
"""

import math, time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIRS = [
    ("GBP_JPY", 0.01),
    ("USD_JPY", 0.01),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
]

OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000
MAX_HOLD   = 120   # H4 bars max hold (~20 trading days)
ATR_PERIOD = 14

SMA_FAST   = [5, 10, 20]
SMA_SLOW   = [20, 50, 100]
ATR_SL     = [1.0, 2.0, 3.0]
ATR_TRAIL  = [1.0, 2.0, 3.0]
DONCH_N    = [10, 20, 50]

OUT_CSV = Path(__file__).parent / "h4_trend_results.csv"


# ── Indicators ────────────────────────────────────────────────────────────────

def wilder_atr(hi, lo, cl, period=ATR_PERIOD):
    tr = np.maximum(hi - lo,
         np.maximum(np.abs(hi - np.concatenate([[cl[0]], cl[:-1]])),
                    np.abs(lo - np.concatenate([[cl[0]], cl[:-1]]))))
    atr = np.zeros(len(tr))
    atr[:period] = np.nan
    atr[period-1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def sma(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w).mean().values


def donchian(hi, lo, n):
    hh = pd.Series(hi).shift(1).rolling(n, min_periods=n).max().values
    ll = pd.Series(lo).shift(1).rolling(n, min_periods=n).min().values
    return hh, ll


# ── Trade simulator ───────────────────────────────────────────────────────────

def simulate(op, hi, lo, cl, sp, signal_long, signal_short,
             atr, pip, atr_sl, atr_trail, max_hold):
    """
    Generic simulation: signal_long/short are bar-index arrays of entry signals.
    Returns array of pnl in pips.
    """
    n = len(cl)
    trades = []
    pos = 0          # 0=flat, 1=long, -1=short
    entry_px = 0.0
    trail_stop = 0.0
    peak = 0.0
    entry_bar = 0

    for i in range(1, n):
        if pos == 0:
            if signal_long[i]:
                pos = 1
                entry_px = op[i] + sp[i] * pip   # fill at open + spread
                trail_stop = entry_px - atr_sl * atr[i]
                peak = op[i]
                entry_bar = i
            elif signal_short[i]:
                pos = -1
                entry_px = op[i] - sp[i] * pip   # fill at open - spread
                trail_stop = entry_px + atr_sl * atr[i]
                peak = op[i]
                entry_bar = i
        else:
            # Update trail stop from new peak
            if pos == 1:
                if hi[i] > peak:
                    peak = hi[i]
                    new_ts = peak - atr_trail * atr[i]
                    if new_ts > trail_stop:
                        trail_stop = new_ts
            else:
                if lo[i] < peak:
                    peak = lo[i]
                    new_ts = peak + atr_trail * atr[i]
                    if new_ts < trail_stop:
                        trail_stop = new_ts

            # Check stop hit or opposite signal
            stop_hit = (pos == 1 and lo[i] <= trail_stop) or \
                       (pos == -1 and hi[i] >= trail_stop)
            opp_sig  = (pos == 1 and signal_short[i]) or \
                       (pos == -1 and signal_long[i])
            max_hit  = (i - entry_bar) >= max_hold

            if stop_hit or opp_sig or max_hit:
                if stop_hit:
                    exit_px = trail_stop - sp[i] * pip if pos == 1 else trail_stop + sp[i] * pip
                else:
                    exit_px = op[i] - sp[i] * pip if pos == 1 else op[i] + sp[i] * pip
                pnl = pos * (exit_px - entry_px) / pip
                trades.append(pnl)
                pos = 0

                # Immediately reverse if opposite signal
                if opp_sig and not max_hit:
                    new_dir = -pos if pos != 0 else 0
                    if new_dir == 0:
                        new_dir = -1 if signal_short[i] else 1
                    pos = new_dir
                    entry_px = op[i] + (sp[i] * pip if pos == 1 else -sp[i] * pip)
                    trail_stop = entry_px - (atr_sl * atr[i] * pos)
                    peak = op[i]
                    entry_bar = i

    return np.array(trades)


# ── Validation helpers ────────────────────────────────────────────────────────

def boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


def run_validation(op, hi, lo, cl, sp, atr, signal_long, signal_short, pip,
                   atr_sl, atr_trail, nb, is_end, is_csz, oos_csz, oos_days, rng):
    def run_slice(s, e):
        return simulate(op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
                        signal_long[s:e], signal_short[s:e],
                        atr[s:e], pip, atr_sl, atr_trail, MAX_HOLD)

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
        p = run_slice(s_, e_)
        if len(p) > 0 and p.sum() > 0:
            is_wf += 1

    pnl_oos = run_slice(is_end, nb)
    ppd = pnl_oos.sum() / oos_days if len(pnl_oos) > 0 else 0.0
    nc_oos = len(pnl_oos)

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
        p = run_slice(s_, e_)
        if len(p) > 0 and p.sum() > 0:
            oos_wf += 1

    p5 = p_pos = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and nc_oos > 0:
        p5, p_pos = boot_stats(pnl_oos, oos_days, N_BOOT, rng)

    passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0
              and not math.isnan(p_pos) and p_pos > 0.95)

    win_pct = float(np.mean(pnl_oos > 0)) * 100 if nc_oos > 0 else float('nan')
    avg_pnl = float(np.mean(pnl_oos)) if nc_oos > 0 else float('nan')

    return dict(ppd=ppd, nc=nc_oos, is_wf=is_wf, oos_wf=oos_wf,
                p5=p5, p_pos=p_pos, passed=passed,
                win_pct=win_pct, avg_pnl=avg_pnl)


# ── Per-pair sweep ────────────────────────────────────────────────────────────

def sweep_pair(pair, pip, rng):
    print(f"\n  {pair}  pip={pip}", flush=True)

    mid = (pd.read_parquet(DATA_DIR_MID / f'{pair}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{pair}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    df_m5 = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner').reset_index(drop=True)

    # Resample to H4
    df_h4 = (df_m5.set_index('timestamp').resample('4h').agg({
        'open':  'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'bid_c': 'last',  'ask_c': 'last'
    }).dropna().reset_index())

    op = df_h4['open'].values.astype(np.float64)
    hi = df_h4['high'].values.astype(np.float64)
    lo = df_h4['low'].values.astype(np.float64)
    cl = df_h4['close'].values.astype(np.float64)
    sp = ((df_h4['ask_c'] - df_h4['bid_c']) / pip).clip(lower=0.1).values.astype(np.float64)

    nb      = len(cl)
    is_end  = int(nb * (1 - OOS_FRAC))
    is_csz  = is_end // IS_CHUNKS
    oos_len = nb - is_end
    oos_csz = oos_len // OOS_CHUNKS
    oos_days = oos_len / 6.0   # ~6 H4 bars per trading day

    gate = float(np.percentile(sp[:is_end], 90))
    sp_g = np.where(sp > gate, gate, sp)

    atr_arr = wilder_atr(hi, lo, cl, ATR_PERIOD)
    print(f"    H4 bars={nb:,}  is={is_end:,}  oos={oos_len:,}  oos_days={oos_days:.0f}  gate={gate:.2f}p")

    results = []
    n_pass = 0

    # ── Strategy A: SMA cross ──────────────────────────────────────────────
    combos_a = list(product(SMA_FAST, SMA_SLOW, ATR_SL, ATR_TRAIL))
    print(f"    SMA cross: {len(combos_a)} combos...", end=' ', flush=True)
    for fast, slow, asl, atr in combos_a:
        if fast >= slow:
            continue
        fsma = sma(cl, fast)
        ssma = sma(cl, slow)
        prev_f = np.concatenate([[np.nan], fsma[:-1]])
        prev_s = np.concatenate([[np.nan], ssma[:-1]])

        sig_l = np.zeros(nb, bool)
        sig_s = np.zeros(nb, bool)
        valid = ~np.isnan(prev_f) & ~np.isnan(prev_s) & ~np.isnan(fsma) & ~np.isnan(ssma) & (sp_g <= gate)
        sig_l[valid] = (fsma[valid] > ssma[valid]) & (prev_f[valid] <= prev_s[valid])
        sig_s[valid] = (fsma[valid] < ssma[valid]) & (prev_f[valid] >= prev_s[valid])

        r = run_validation(op, hi, lo, cl, sp_g, atr_arr, sig_l, sig_s, pip,
                           asl, atr, nb, is_end, is_csz, oos_csz, oos_days, rng)
        r.update(pair=pair, strategy='sma_cross', fast=fast, slow=slow,
                 atr_sl=asl, atr_trail=atr, N=0)
        results.append(r)
        if r['passed']:
            n_pass += 1

    print(f"done. pass={n_pass}", flush=True)

    # ── Strategy B: Donchian ───────────────────────────────────────────────
    combos_b = list(product(DONCH_N, ATR_SL, ATR_TRAIL))
    print(f"    Donchian:  {len(combos_b)} combos...", end=' ', flush=True)
    n_pass_d = 0
    for N, asl, atr in combos_b:
        hh, ll = donchian(hi, lo, N)
        prev_cl = np.concatenate([[np.nan], cl[:-1]])
        sig_l = np.zeros(nb, bool)
        sig_s = np.zeros(nb, bool)
        valid = ~np.isnan(hh) & ~np.isnan(ll) & (sp_g <= gate)
        sig_l[valid] = cl[valid] > hh[valid]
        sig_s[valid] = cl[valid] < ll[valid]

        r = run_validation(op, hi, lo, cl, sp_g, atr_arr, sig_l, sig_s, pip,
                           asl, atr, nb, is_end, is_csz, oos_csz, oos_days, rng)
        r.update(pair=pair, strategy='donchian', fast=0, slow=0,
                 atr_sl=asl, atr_trail=atr, N=N)
        results.append(r)
        if r['passed']:
            n_pass_d += 1

    print(f"done. pass={n_pass_d}", flush=True)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    all_results = []

    print("H4 Trend-Following Study")
    print(f"Pairs: {[p for p,_ in PAIRS]}  OOS={OOS_FRAC:.0%}  IS/OOS WF={IS_CHUNKS}/{OOS_CHUNKS}  boot={N_BOOT}")

    for pair, pip in PAIRS:
        results = sweep_pair(pair, pip, rng)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Summary
    passing = df[df.passed].copy()
    passing = passing.sort_values('p5', ascending=False).reset_index(drop=True)

    sep = "─" * 110
    print(f"\n{'═'*110}")
    print(f"  H4 Trend-Following — Results  ({len(df)} combos tested, {len(passing)} passed all gates)")
    print(f"{'═'*110}")
    print(sep)
    hdr = (f"  {'pair':<10} {'strategy':<12} {'params':<25} | "
           f"{'p/d':>8} {'n':>5} {'win%':>5} | "
           f"{'IS':>2} {'OS':>2} | {'P5':>8} {'P(+)':>6}")
    print(hdr); print(sep)

    if passing.empty:
        print("  🔴 NO CONFIGS PASSED ALL GATES")
    else:
        for _, r in passing.head(20).iterrows():
            if r.strategy == 'sma_cross':
                params = f"f{int(r.fast)}/s{int(r.slow)} sl{r.atr_sl}×ATR tr{r.atr_trail}×ATR"
            else:
                params = f"N={int(r.N)} sl{r.atr_sl}×ATR tr{r.atr_trail}×ATR"
            p5_s   = f"{r.p5:8.1f}" if not math.isnan(r.p5) else "     nan"
            ppos_s = f"{r.p_pos:6.3f}" if not math.isnan(r.p_pos) else "   nan"
            win_s  = f"{r.win_pct:5.1f}" if not math.isnan(r.win_pct) else "  nan"
            print(f"  {r.pair:<10} {r.strategy:<12} {params:<25} | "
                  f"  {r.ppd:>8.1f} {r.nc:>5} {win_s} | "
                  f"{r.is_wf:>2}/{IS_CHUNKS} {r.oos_wf:>2}/{OOS_CHUNKS} | "
                  f"{p5_s} {ppos_s}")

    print(sep)

    # Per-pair summary
    print(f"\n  Per-pair pass count:")
    for pair, _ in PAIRS:
        p_df = passing[passing.pair == pair]
        all_df = df[df.pair == pair]
        pa = p_df[p_df.strategy=='sma_cross']
        pb = p_df[p_df.strategy=='donchian']
        print(f"    {pair:<10}  SMA cross: {len(pa):>2}/{len(all_df[all_df.strategy=='sma_cross']):>2} pass  "
              f"Donchian: {len(pb):>2}/{len(all_df[all_df.strategy=='donchian']):>2} pass  "
              f"total={len(p_df)}")

    # Best per pair
    if not passing.empty:
        print(f"\n  BEST CONFIG PER PAIR (by P5):")
        for pair, _ in PAIRS:
            best = passing[passing.pair == pair].head(1)
            if best.empty:
                print(f"    {pair:<10}  🔴 no passing config")
            else:
                r = best.iloc[0]
                if r.strategy == 'sma_cross':
                    params = f"SMA f{int(r.fast)}/s{int(r.slow)} sl={r.atr_sl}×ATR trail={r.atr_trail}×ATR"
                else:
                    params = f"Donchian N={int(r.N)} sl={r.atr_sl}×ATR trail={r.atr_trail}×ATR"
                print(f"    {pair:<10}  {params}  p/d={r.ppd:.1f}  P5={r.p5:.1f}  P(+)={r.p_pos:.3f}  "
                      f"win={r.win_pct:.1f}%  n={r.nc}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Results saved → {OUT_CSV}")
    print(f"  Done in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
