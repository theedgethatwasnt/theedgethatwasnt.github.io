"""Phase 4: early-momentum-quality study.

For every baseline V0 trade, capture MFE at fixed early checkpoints
(0.5h, 1h, 2h, 4h, 8h, 12h, 24h) and the FINAL outcome.

Question: does early MFE predict final outcome?
  - If trades that end at TP show meaningfully larger early MFE than
    trades that end at force-close, an early-MFE threshold could
    serve as a quality filter ("exit if MFE < X by hour T").
"""
from pathlib import Path
import sys, importlib.util
import numpy as np, pandas as pd, pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('bv',
    Path(__file__).parent / 'backtest_variants.py')
bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

PROJECT = Path('/path/to/projects/fx-core')
DATA = PROJECT / 'data/m5_ohlc'
OUT = PROJECT / 'research/experiments/loss_cap_sweep/results'
CHECKPOINTS_H = [0.5, 1, 2, 4, 8, 12, 24]
TP_PIPS = 20.0


def trade_paths(pair):
    """Return list of trades with MFE at each checkpoint + final outcome."""
    pip, sp_gate = bv.PAIRS[pair]
    df = pq.read_table(DATA / f'{pair}_M5.parquet').to_pandas()
    df = df.tail(bv.WINDOW_BARS).reset_index(drop=True)
    h1 = bv.resample_tf(df, 60); m30 = bv.resample_tf(df, 30)
    h1_sig = bv.momentum_sig(h1['close'].to_numpy(), bv.LAGS)
    m30_sig = bv.momentum_sig(m30['close'].to_numpy(), bv.LAGS)
    h1_ts = h1['timestamp'].to_numpy(); m30_ts = m30['timestamp'].to_numpy()
    m5_ts = df['timestamp'].to_numpy()
    m5_o = df['open'].to_numpy(); m5_h = df['high'].to_numpy()
    m5_l = df['low'].to_numpy(); m5_c = df['close'].to_numpy()

    pos_dir = 0; entry_px = 0.0; entry_bar = -1
    mfe = 0.0; mae = 0.0
    cp_bars = [int(h * 12) for h in CHECKPOINTS_H]
    mfe_at = {}
    out = []
    for i in range(1, len(df)):
        sig = bv.signal_at(m5_ts[i-1], h1_ts, h1_sig, m30_ts, m30_sig)
        if pos_dir != 0:
            held = i - entry_bar
            high_pip = (m5_h[i] - entry_px) / pip * pos_dir
            low_pip = (m5_l[i] - entry_px) / pip * pos_dir
            if high_pip > mfe: mfe = high_pip
            if low_pip < mae: mae = low_pip
            for cp_b in cp_bars:
                if held == cp_b:
                    mfe_at[cp_b] = mfe
            tp_lvl = entry_px + pos_dir * TP_PIPS * pip
            tp_hit = (pos_dir == 1 and m5_h[i] >= tp_lvl) or \
                     (pos_dir == -1 and m5_l[i] <= tp_lvl)
            if tp_hit:
                out.append({'pair': pair, 'reason': 'TP', 'bars': held,
                            'pips': TP_PIPS, 'mfe': mfe, 'mae': mae,
                            **{f'mfe_{int(b/12*60)}m': mfe_at.get(b, np.nan) for b in cp_bars}})
                pos_dir = 0; mfe = 0.0; mae = 0.0; mfe_at = {}
                continue
        if pos_dir == 0 and sig != 0:
            pos_dir = sig; entry_px = m5_o[i]; entry_bar = i
            mfe = 0.0; mae = 0.0; mfe_at = {}
    if pos_dir != 0:
        exit_px = m5_c[-1]
        pnl = (exit_px - entry_px) / pip * pos_dir
        held = len(df) - 1 - entry_bar
        out.append({'pair': pair, 'reason': 'end', 'bars': held,
                    'pips': pnl, 'mfe': mfe, 'mae': mae,
                    **{f'mfe_{int(b/12*60)}m': mfe_at.get(b, np.nan) for b in cp_bars}})
    return out


def main():
    all_t = []
    for p in bv.PAIRS:
        print(f'  {p}...', flush=True)
        all_t.extend(trade_paths(p))
    df = pd.DataFrame(all_t)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'early_mfe_per_trade.csv', index=False)

    print(f"\nTotal trades: {len(df)}  TP={int((df['reason']=='TP').sum())}  "
          f"end={int((df['reason']=='end').sum())}\n")
    print("=== Early MFE distribution by outcome (median pips) ===")
    cps = [c for c in df.columns if c.startswith('mfe_')]
    g = df.groupby('reason')[cps].median()
    print(g.to_string())
    print()
    print("=== Early MFE by outcome (p25 / p75) ===")
    for reason in df['reason'].unique():
        sub = df[df['reason'] == reason]
        print(f"\n  {reason} (n={len(sub)}):")
        for c in cps:
            p25 = sub[c].quantile(0.25)
            p50 = sub[c].quantile(0.50)
            p75 = sub[c].quantile(0.75)
            print(f"    {c}: p25={p25:+.1f}p  p50={p50:+.1f}p  p75={p75:+.1f}p")

    # Critical question: at each checkpoint, what's the survival profile?
    print("\n=== If we exit when MFE < threshold at checkpoint, what fraction of TPs survive? ===")
    print(f"{'Checkpoint':<12s}{'Threshold (p)':<15s}{'Survive_TP%':>13s}{'Survive_End%':>14s}{'Filter cuts':>13s}")
    for cp in cps:
        cp_label = cp
        # Only trades that REACHED this checkpoint (held >= cp time)
        cp_minutes = int(cp.split('_')[1].replace('m',''))
        cp_bars_eq = (cp_minutes/60)*12
        sub = df[df['bars'] >= cp_bars_eq].copy()
        if len(sub) == 0: continue
        for thr in [0, 2, 3, 5, 8, 10]:
            keep = sub[sub[cp] >= thr]
            cut = sub[sub[cp] < thr]
            tp_kept = ((keep['reason'] == 'TP')).sum()
            tp_cut = ((cut['reason'] == 'TP')).sum()
            end_kept = ((keep['reason'] == 'end')).sum()
            end_cut = ((cut['reason'] == 'end')).sum()
            total_tp = (sub['reason'] == 'TP').sum()
            total_end = (sub['reason'] == 'end').sum()
            if total_tp == 0: continue
            tp_pct = 100 * tp_kept / total_tp
            end_pct = 100 * end_kept / max(total_end, 1)
            print(f"  {cp_label:<10s} >= {thr:>3d}p     {tp_pct:>12.1f}%  {end_pct:>12.1f}%  "
                  f"cuts {len(cut):>3d} ({tp_cut} TPs, {end_cut} ends)")


if __name__ == '__main__':
    main()
