"""
H4 Donchian — ATR-ratio entry filter sweep (Session 054)
=========================================================
Baseline: Donchian(10) + ATR(14) trail=1× sl=2×  (best validated config)
Filter:   only enter when atrr_20 >= threshold
          atrr_20 = Wilder_ATR(14)[i] / SMA20(ATR(14))[i]

Thresholds swept: 0.0 (no filter), 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0

Hypothesis (from 8A-D screening): entering in elevated-volatility regime
(atrr > 1.0) produces bigger moves → higher p/d and/or better Calmar.

Metrics reported per threshold × pair:
  OOS p/d, MaxDD (pips), Calmar = p/d / MaxDD × trading-days,
  win%, avg_win, avg_loss, n_trades, IS_wf, OOS_wf, P5, P(+)

Validation: IS=3/3 chunks + OOS=3/3 chunks + bootstrap P5>0 + P(+)>0.95
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIRS = [
    ("GBP_JPY", 0.01),
    ("USD_JPY", 0.01),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
]

DON_N      = 10
ATR_SL     = 2.0
ATR_TRAIL  = 1.0
ATR_PERIOD = 14
ATRR_SMA   = 20      # window for baseline SMA of ATR

THRESHOLDS = [0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000
MAX_HOLD   = 120     # H4 bars


# ── Indicators ────────────────────────────────────────────────────────────────

def wilder_atr(hi, lo, cl, period=ATR_PERIOD):
    tr = np.maximum(hi - lo,
         np.maximum(np.abs(hi - np.concatenate([[cl[0]], cl[:-1]])),
                    np.abs(lo - np.concatenate([[cl[0]], cl[:-1]]))))
    atr = np.zeros(len(tr))
    atr[:period] = np.nan
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def atrr(atr_arr, sma_w=ATRR_SMA):
    """ATR ratio: current ATR / SMA(ATR, sma_w). >1 = elevated volatility."""
    sma = pd.Series(atr_arr).rolling(sma_w, min_periods=sma_w).mean().values
    return np.where(sma > 0, atr_arr / sma, np.nan)


def donchian(hi, lo, n=DON_N):
    hh = pd.Series(hi).shift(1).rolling(n, min_periods=n).max().values
    ll = pd.Series(lo).shift(1).rolling(n, min_periods=n).min().values
    return hh, ll


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate(op, hi, lo, cl, sp, sig_l, sig_s, atr_a, pip):
    n      = len(cl)
    trades = []
    eq     = [0.0]       # running P&L for MaxDD
    pos = 0; entry_px = 0.0; trail = 0.0; peak = 0.0; entry_bar = 0

    for i in range(1, n):
        if pos == 0:
            if sig_l[i]:
                pos = 1
                entry_px = op[i] + sp[i] * pip
                trail    = entry_px - ATR_SL * atr_a[i]
                peak     = op[i]; entry_bar = i
            elif sig_s[i]:
                pos = -1
                entry_px = op[i] - sp[i] * pip
                trail    = entry_px + ATR_SL * atr_a[i]
                peak     = op[i]; entry_bar = i
        else:
            if pos == 1:
                if hi[i] > peak:
                    peak = hi[i]
                    trail = max(trail, peak - ATR_TRAIL * atr_a[i])
            else:
                if lo[i] < peak:
                    peak = lo[i]
                    trail = min(trail, peak + ATR_TRAIL * atr_a[i])

            stop_hit = (pos == 1 and lo[i] <= trail) or (pos == -1 and hi[i] >= trail)
            opp_sig  = (pos == 1 and sig_s[i]) or (pos == -1 and sig_l[i])
            max_hit  = (i - entry_bar) >= MAX_HOLD

            if stop_hit or opp_sig or max_hit:
                if stop_hit:
                    exit_px = (trail - sp[i]*pip) if pos==1 else (trail + sp[i]*pip)
                else:
                    exit_px = (op[i] - sp[i]*pip) if pos==1 else (op[i] + sp[i]*pip)
                pnl = pos * (exit_px - entry_px) / pip
                trades.append(pnl)
                eq.append(eq[-1] + pnl)
                pos = 0

                if opp_sig and not max_hit:
                    pos      = 1 if sig_l[i] else -1
                    entry_px = op[i] + (sp[i]*pip if pos==1 else -sp[i]*pip)
                    trail    = entry_px - pos * ATR_SL * atr_a[i]
                    peak     = op[i]; entry_bar = i

    pnl_arr = np.array(trades) if trades else np.array([0.0])
    eq_arr  = np.array(eq)
    # MaxDD in pips
    peak_eq = np.maximum.accumulate(eq_arr)
    dd      = peak_eq - eq_arr
    maxdd   = float(dd.max()) if len(dd) > 1 else 0.0
    return pnl_arr, maxdd


def boot_stats(pnl, oos_days, n_boot, rng):
    if len(pnl) == 0:
        return float('nan'), float('nan')
    boots = np.array([rng.choice(pnl, len(pnl), replace=True).sum() / oos_days
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 5)), float(np.mean(boots > 0))


# ── Per-pair, per-threshold ───────────────────────────────────────────────────

def run_pair(pair, pip, rng):
    # Load and merge
    try:
        mid = pd.read_parquet(DATA_DIR_MID / f'{pair}_M5.parquet')
        mid.columns = [c.lower() for c in mid.columns]
        if 'timestamp' not in mid.columns:
            mid = mid.reset_index().rename(columns={'index': 'timestamp'})
        mid['timestamp'] = pd.to_datetime(mid['timestamp'])
        mid = mid.sort_values('timestamp').reset_index(drop=True)

        ba = pd.read_parquet(DATA_DIR_BA / f'{pair}_M5_BA.parquet')
        ba.columns = [c.lower() for c in ba.columns]
        if 'timestamp' not in ba.columns:
            ba = ba.reset_index().rename(columns={'index': 'timestamp'})
        ba['timestamp'] = pd.to_datetime(ba['timestamp'])
        ba = ba.sort_values('timestamp').reset_index(drop=True)

        mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
        ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
        df = mid.merge(ba[['ts_key', 'bid_c', 'ask_c']], on='ts_key', how='inner')
    except Exception as e:
        print(f"  {pair}: load error {e}")
        return []

    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, utc=True)
    h4 = df.resample('4h').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),   close=('close', 'last'),
        bid_c=('bid_c', 'last'), ask_c=('ask_c', 'last'),
    ).dropna().reset_index()

    op = h4['open'].values.astype(np.float64)
    hi = h4['high'].values.astype(np.float64)
    lo = h4['low'].values.astype(np.float64)
    cl = h4['close'].values.astype(np.float64)
    sp = ((h4['ask_c'] - h4['bid_c']) / pip).clip(lower=0.1).values.astype(np.float64)

    nb       = len(cl)
    is_end   = int(nb * (1 - OOS_FRAC))
    is_csz   = is_end // IS_CHUNKS
    oos_len  = nb - is_end
    oos_csz  = oos_len // OOS_CHUNKS
    oos_days = oos_len / 6.0

    gate = float(np.percentile(sp[:is_end], 90))
    sp_g = np.where(sp > gate, gate, sp)

    atr_a  = wilder_atr(hi, lo, cl)
    atrr_a = atrr(atr_a)
    hh, ll = donchian(hi, lo)

    print(f"  {pair}: H4={nb}  is={is_end}  oos={oos_len}  oos_days={oos_days:.0f}  "
          f"sp_gate={gate:.2f}p", flush=True)

    rows = []
    for thr in THRESHOLDS:
        # Build entry signals with atrr gate
        valid = (~np.isnan(hh) & ~np.isnan(ll) & ~np.isnan(atr_a)
                 & (sp_g <= gate))
        if thr > 0.0:
            valid &= (~np.isnan(atrr_a) & (atrr_a >= thr))

        sig_l = np.zeros(nb, bool)
        sig_s = np.zeros(nb, bool)
        sig_l[valid] = cl[valid] > hh[valid]
        sig_s[valid] = cl[valid] < ll[valid]

        def run_slice(s, e):
            p, dd = simulate(op[s:e], hi[s:e], lo[s:e], cl[s:e], sp_g[s:e],
                             sig_l[s:e], sig_s[s:e], atr_a[s:e], pip)
            return p, dd

        # IS walk-forward
        is_wf = 0
        for ch in range(IS_CHUNKS):
            s_ = ch * is_csz
            e_ = (ch + 1) * is_csz if ch < IS_CHUNKS - 1 else is_end
            p, _ = run_slice(s_, e_)
            if len(p) > 0 and p.sum() > 0:
                is_wf += 1

        # OOS
        pnl_oos, maxdd_oos = run_slice(is_end, nb)
        ppd = pnl_oos.sum() / oos_days if len(pnl_oos) > 0 else 0.0

        oos_wf = 0
        for ch in range(OOS_CHUNKS):
            s_ = is_end + ch * oos_csz
            e_ = is_end + (ch + 1) * oos_csz if ch < OOS_CHUNKS - 1 else nb
            p, _ = run_slice(s_, e_)
            if len(p) > 0 and p.sum() > 0:
                oos_wf += 1

        p5 = p_pos = float('nan')
        if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS and len(pnl_oos) > 0:
            p5, p_pos = boot_stats(pnl_oos, oos_days, N_BOOT, rng)

        passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                  and not math.isnan(p5) and p5 > 0
                  and not math.isnan(p_pos) and p_pos > 0.95)

        n_trades  = len(pnl_oos)
        win_pct   = float(np.mean(pnl_oos > 0)) * 100 if n_trades > 0 else float('nan')
        avg_win   = float(np.mean(pnl_oos[pnl_oos > 0])) if (pnl_oos > 0).any() else float('nan')
        avg_loss  = float(np.mean(pnl_oos[pnl_oos <= 0])) if (pnl_oos <= 0).any() else float('nan')
        calmar    = ppd * oos_days / maxdd_oos if maxdd_oos > 0 else float('nan')

        rows.append(dict(
            pair=pair, thr=thr, ppd=round(ppd, 1), maxdd=round(maxdd_oos, 1),
            calmar=round(calmar, 1) if not math.isnan(calmar) else float('nan'),
            win_pct=round(win_pct, 1), avg_win=round(avg_win, 1),
            avg_loss=round(avg_loss, 1), n_trades=n_trades,
            is_wf=is_wf, oos_wf=oos_wf,
            p5=round(p5, 1) if not math.isnan(p5) else float('nan'),
            p_pos=round(p_pos, 3) if not math.isnan(p_pos) else float('nan'),
            passed=passed,
        ))
        mark = "✅" if passed else "  "
        print(f"    thr={thr:.1f}  p/d={ppd:>7.1f}  maxdd={maxdd_oos:>7.1f}  "
              f"calmar={calmar:>7.1f}  wr={win_pct:>5.1f}%  n={n_trades:>4}  "
              f"IS={is_wf}/3 OOS={oos_wf}/3  P5={p5:>7.1f}  P(+)={p_pos:.3f}  {mark}")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)
    all_rows = []
    print("H4 Donchian ATR-ratio filter sweep")
    print(f"Config: N={DON_N} ATR_SL={ATR_SL}× ATR_TRAIL={ATR_TRAIL}× atrr_sma={ATRR_SMA}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    for pair, pip in PAIRS:
        rows = run_pair(pair, pip, rng)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent / "h4_atrr_filter_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY — OOS p/d and Calmar by pair × threshold  (✅ = passed all gates)")
    print("=" * 90)

    for pair, _ in PAIRS:
        sub = df[df['pair'] == pair].set_index('thr')
        print(f"\n  {pair}")
        print(f"  {'thr':>5}  {'p/d':>7}  {'maxdd':>7}  {'calmar':>7}  "
              f"{'wr%':>5}  {'n':>4}  {'P5':>7}  {'P(+)':>6}  pass?")
        base = sub.loc[0.0] if 0.0 in sub.index else None
        for thr, row in sub.iterrows():
            chg = ""
            if base is not None and thr > 0.0:
                delta = row['ppd'] - base['ppd']
                chg   = f" ({delta:+.1f})"
            mark = "✅" if row['passed'] else "  "
            print(f"  {thr:>5.1f}  {row['ppd']:>7.1f}{chg:<10}  {row['maxdd']:>7.1f}  "
                  f"{row['calmar']:>7.1f}  {row['win_pct']:>5.1f}  {row['n_trades']:>4}  "
                  f"{row['p5']:>7.1f}  {row['p_pos']:>6.3f}  {mark}")

    # ── Cross-pair winners ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("CROSS-PAIR: thresholds that pass on ALL 4 pairs")
    print("=" * 90)
    for thr in THRESHOLDS:
        sub = df[df['thr'] == thr]
        n_pass = sub['passed'].sum()
        avg_ppd = sub['ppd'].mean()
        avg_cal = sub[sub['calmar'].notna()]['calmar'].mean()
        mark = "✅✅✅✅" if n_pass == 4 else ("✅" * int(n_pass) + "  " * (4 - int(n_pass)))
        print(f"  thr={thr:.1f}  pass={int(n_pass)}/4  {mark}  "
              f"avg_p/d={avg_ppd:.1f}  avg_calmar={avg_cal:.1f}")


if __name__ == "__main__":
    main()
