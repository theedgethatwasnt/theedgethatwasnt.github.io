#!/usr/bin/env python3
"""
SMA22 Dual-TF Momentum Alignment — GBP_JPY
===========================================
Signal:
  SMA(22) on H1 close and M30 close (causal from M5 data).
  mom_k = sma[t] - sma[t-k]  for k in {1, 5, 10}

  LONG  when all 6 momentum values > 0  (H1 + M30 lags 1,5,10)
  SHORT when all 6 momentum values < 0

Entry: ask_c (long) or bid_c (short) at bar after signal fires.
Exit sweep: TP, trailing stop, signal-reversal, max-hold.
WF: 70% IS (3-fold gate), 30% OOS.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba" / "GBP_JPY_M5_BA.parquet"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIR       = "GBP_JPY"
PIP        = 0.01          # GBP/JPY pip size
SMA_N      = 22
LAGS       = [1, 5, 10]


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_tf_mom(m5_close: pd.Series, tf: str) -> pd.DataFrame:
    """
    Resample M5 close to H1 or M30, compute SMA(22), compute momentum
    at lags 1/5/10, then forward-fill back to M5 index (shift 1 for causality).
    Returns DataFrame aligned to m5_close.index with columns mom1/mom5/mom10.
    """
    resampled = m5_close.resample(tf).last().dropna()
    sma = resampled.rolling(SMA_N, min_periods=SMA_N).mean()
    moms = pd.DataFrame(index=resampled.index)
    for k in LAGS:
        moms[f"mom{k}"] = sma - sma.shift(k)
    # shift(1) so we only use completed bars, then ffill to M5 grid
    moms = moms.shift(1)
    moms = moms.reindex(m5_close.index, method="ffill")
    return moms


def build_signals(df: pd.DataFrame) -> pd.Series:
    """
    +1 = long signal, -1 = short signal, 0 = flat
    Both H1 and M30 must agree on direction at all three lags.
    """
    h1  = compute_tf_mom(df["close"], "1h")
    m30 = compute_tf_mom(df["close"], "30min")

    long_cond  = ((h1 > 0).all(axis=1)  & (m30 > 0).all(axis=1))
    short_cond = ((h1 < 0).all(axis=1)  & (m30 < 0).all(axis=1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig, h1, m30


def simulate(df: pd.DataFrame, sig: pd.Series,
             exit_type: str, tp_p: float, trail_act: float, trail_dist: float,
             max_hold: int, sp_gate: float) -> pd.DataFrame:
    """
    Single-pass trade simulation.
    exit_type: 'tp' | 'trail' | 'signal' | 'signal_trail' | 'hold'
    tp_p       : fixed take-profit pips
    trail_act  : pips MFE to activate trailing stop
    trail_dist : pips trailing distance
    max_hold   : bars before forced exit
    sp_gate    : IS-p90 spread — any bar with spread > gate is skipped
    """
    bid  = df["bid_c"].values
    ask  = df["ask_c"].values
    mid  = df["close"].values
    sp   = (ask - bid) / PIP
    s    = sig.values
    n    = len(df)

    trades = []
    in_trade = False
    dir_ = 0
    entry_px = 0.0
    entry_i  = 0
    mfe      = 0.0
    trail_px = np.nan

    for i in range(1, n):
        if in_trade:
            # current mid P&L in pips
            pnl_now = (mid[i] - entry_px) / PIP * dir_

            # track MFE
            if pnl_now > mfe:
                mfe = pnl_now
                if exit_type in ("trail", "signal_trail") and mfe >= trail_act:
                    trail_px = entry_px + dir_ * (mfe - trail_dist) * PIP

            # check exits
            exit_now = False
            exit_px  = bid[i] if dir_ == 1 else ask[i]

            if exit_type == "tp":
                if pnl_now >= tp_p:
                    exit_now = True

            elif exit_type == "trail":
                if not np.isnan(trail_px):
                    if dir_ == 1 and mid[i] <= trail_px:
                        exit_now = True
                    elif dir_ == -1 and mid[i] >= trail_px:
                        exit_now = True

            elif exit_type == "signal":
                # exit when signal flips or goes flat
                if s[i] != dir_:
                    exit_now = True

            elif exit_type == "signal_trail":
                if not np.isnan(trail_px):
                    if dir_ == 1 and mid[i] <= trail_px:
                        exit_now = True
                    elif dir_ == -1 and mid[i] >= trail_px:
                        exit_now = True
                if s[i] != dir_:
                    exit_now = True

            elif exit_type == "hold":
                if (i - entry_i) >= max_hold:
                    exit_now = True

            if exit_now:
                pnl_final = (exit_px - entry_px) / PIP * dir_ - sp[i]
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "dir": dir_, "pnl": pnl_final,
                    "hold": i - entry_i, "mfe": mfe,
                })
                in_trade = False
                dir_ = 0
                mfe  = 0.0
                trail_px = np.nan

        else:
            # check for new entry signal on bar i (signal fired on i-1)
            new_dir = s[i - 1]
            if new_dir != 0 and sp[i] <= sp_gate:
                ep = ask[i] if new_dir == 1 else bid[i]
                entry_px = ep
                entry_i  = i
                dir_     = new_dir
                in_trade = True
                mfe      = 0.0
                trail_px = np.nan

    return pd.DataFrame(trades)


def wf_validate(df, sig, sp_gate, cfg):
    n = len(df)
    is_end = int(n * 0.70)
    oos_df  = df.iloc[is_end:]
    oos_sig = sig.iloc[is_end:]

    # IS: 3 folds
    fold_size = is_end // 3
    is_pds = []
    for f in range(3):
        s = f * fold_size
        e = s + fold_size if f < 2 else is_end
        t = simulate(df.iloc[s:e], sig.iloc[s:e], sp_gate=sp_gate, **cfg)
        if len(t) == 0:
            is_pds.append(np.nan)
            continue
        days = (e - s) / 288
        is_pds.append(t["pnl"].sum() / days)

    # OOS
    t_oos = simulate(oos_df, oos_sig, sp_gate=sp_gate, **cfg)
    oos_days = len(oos_df) / 288
    if len(t_oos) == 0:
        oos_pd = 0.0
        wr = 0.0
    else:
        oos_pd = t_oos["pnl"].sum() / oos_days
        wr = (t_oos["pnl"] > 0).mean() * 100

    is_pass = sum(1 for x in is_pds if x is not None and x > 0)

    return {
        "is_fold1": round(is_pds[0], 2) if is_pds[0] is not None else None,
        "is_fold2": round(is_pds[1], 2) if is_pds[1] is not None else None,
        "is_fold3": round(is_pds[2], 2) if is_pds[2] is not None else None,
        "is_pass":  is_pass,
        "oos_pd":   round(oos_pd, 2),
        "oos_nt":   len(t_oos),
        "oos_wr":   round(wr, 1),
        "oos_days": round(oos_days, 0),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {PAIR} M5 data …")
    df = pd.read_parquet(DATA)
    df = df.set_index("timestamp").sort_index()
    df = df.astype({c: "float32" for c in df.select_dtypes("float64").columns})

    # IS spread gate (p90 of first 70%)
    n_is = int(len(df) * 0.70)
    sp_series = (df["ask_c"] - df["bid_c"]) / PIP
    sp_gate = float(np.percentile(sp_series.iloc[:n_is].dropna(), 90))
    print(f"Bars: {len(df):,}  |  IS spread gate (p90): {sp_gate:.2f}p")

    print("Computing signals …")
    sig, h1_mom, m30_mom = build_signals(df)
    n_long  = (sig ==  1).sum()
    n_short = (sig == -1).sum()
    print(f"Signal bars → LONG: {n_long:,}  SHORT: {n_short:,}  "
          f"({n_long/len(sig)*100:.1f}% / {n_short/len(sig)*100:.1f}%)")

    # Exit configurations to sweep
    exit_cfgs = [
        ("tp_10p",         dict(exit_type="tp",           tp_p=10,  trail_act=0,  trail_dist=0,  max_hold=0)),
        ("tp_15p",         dict(exit_type="tp",           tp_p=15,  trail_act=0,  trail_dist=0,  max_hold=0)),
        ("tp_20p",         dict(exit_type="tp",           tp_p=20,  trail_act=0,  trail_dist=0,  max_hold=0)),
        ("tp_30p",         dict(exit_type="tp",           tp_p=30,  trail_act=0,  trail_dist=0,  max_hold=0)),
        ("tp_50p",         dict(exit_type="tp",           tp_p=50,  trail_act=0,  trail_dist=0,  max_hold=0)),
        ("trail_5_5",      dict(exit_type="trail",        tp_p=0,   trail_act=5,  trail_dist=5,  max_hold=0)),
        ("trail_10_5",     dict(exit_type="trail",        tp_p=0,   trail_act=10, trail_dist=5,  max_hold=0)),
        ("trail_10_10",    dict(exit_type="trail",        tp_p=0,   trail_act=10, trail_dist=10, max_hold=0)),
        ("trail_20_10",    dict(exit_type="trail",        tp_p=0,   trail_act=20, trail_dist=10, max_hold=0)),
        ("signal_rev",     dict(exit_type="signal",       tp_p=0,   trail_act=0,  trail_dist=0,  max_hold=0)),
        ("sig_trail_10_5", dict(exit_type="signal_trail", tp_p=0,   trail_act=10, trail_dist=5,  max_hold=0)),
        ("hold_144",       dict(exit_type="hold",         tp_p=0,   trail_act=0,  trail_dist=0,  max_hold=144)),   # 12h
        ("hold_288",       dict(exit_type="hold",         tp_p=0,   trail_act=0,  trail_dist=0,  max_hold=288)),   # 24h
        ("hold_576",       dict(exit_type="hold",         tp_p=0,   trail_act=0,  trail_dist=0,  max_hold=576)),   # 48h
    ]

    rows = []
    for label, cfg in exit_cfgs:
        print(f"  {label} …", end="  ", flush=True)
        res = wf_validate(df, sig, sp_gate, cfg)
        res["exit_config"] = label
        rows.append(res)
        wf = "✅" if res["is_pass"] >= 3 else ("🟡" if res["is_pass"] >= 2 else "❌")
        print(f"IS {res['is_pass']}/3 {wf}  OOS {res['oos_pd']:+.1f} p/d  "
              f"WR {res['oos_wr']}%  n={res['oos_nt']}")

    out = pd.DataFrame(rows)[["exit_config","is_pass","is_fold1","is_fold2","is_fold3",
                               "oos_pd","oos_nt","oos_wr","oos_days"]]
    out = out.sort_values("oos_pd", ascending=False)
    csv_path = RESULTS / f"sma22_momentum_{PAIR}.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"PAIR: {PAIR}  |  SMA={SMA_N}  |  Lags: {LAGS}")
    print(f"{'='*70}")
    print(out.to_string(index=False))
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
