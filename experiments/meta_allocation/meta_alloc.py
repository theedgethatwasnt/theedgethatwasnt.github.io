#!/usr/bin/env python3
"""
Workstream F — meta-allocation ensemble across the project's own strategies.

Generalizes the WF-validated equity-MA overlay (conservative_010/bb_equity_switch*.py) from a
single strategy's own trade stream to a PORTFOLIO of strategy-columns built by build_matrix.py.

Question: does regime-switched ALLOCATION across strategies (turn a strategy off when its own
trailing equity is below its own trailing MA, monthly rebalance) beat (a) equal-weight always-on,
(b) the best single strategy alone, and (c) a random-allocation null with identical rebalance
dates and identical slot counts?

Causality: every decision for month m uses ONLY data with exit_date < month_start(m) (R1-style
"closed bars only", applied to trades/months instead of price bars). The first month has no prior
history so every eligible strategy defaults to "on" (mirrors bb_equity_switch.py's
`state_live = True` default before i >= W).

No OOS seal exists for this exploratory workstream; the month-by-month causal decision process
itself IS the honesty gate (a strategy's inclusion can never see its own future).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/work/code_meta/results")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/work/code_meta/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W_SWEEP = (10, 20, 40)     # trailing-equity-MA windows (calendar days), robustness sweep
N_SEEDS = 100
SEED0 = 20260706


def maxdd(equity):
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def sharpe_ish(daily):
    d = np.asarray(daily, dtype=float)
    if d.std(ddof=1) < 1e-12 or len(d) < 2:
        return 0.0
    return float(d.mean() / d.std(ddof=1) * np.sqrt(365))


def month_bounds(idx):
    months = sorted(set((d.year, d.month) for d in idx))
    out = []
    for y, m in months:
        start = pd.Timestamp(y, m, 1)
        end = (start + pd.offsets.MonthEnd(1))
        out.append((start, min(end, idx.max())))
    return out


def eligible_pool(meta, mstart, mend):
    m = meta[(meta['first_date'] <= mend) & (meta['last_date'] >= mstart)]
    return list(m['group_key'])


def own_equity_before(matrix, col, before_date):
    """Cumulative equity of this strategy's own daily pnl, using only its active days
    strictly before `before_date`. Returns a pd.Series indexed by date (may be empty)."""
    s = matrix[col]
    s = s[s.index < before_date].dropna()
    if s.empty:
        return s
    return s.cumsum()


def decide_on_set(matrix, meta, mstart, pool, W):
    """For month starting mstart, decide which pool members are 'on' using data < mstart."""
    on = []
    for col in pool:
        eq = own_equity_before(matrix, col, mstart)
        if len(eq) < W:
            on.append(col)  # not enough own history yet -> default ON (benefit of the doubt)
            continue
        ma = eq.rolling(W).mean()
        last_eq, last_ma = eq.iloc[-1], ma.iloc[-1]
        if np.isnan(last_ma):
            on.append(col)
            continue
        if last_eq > last_ma:
            on.append(col)
    return on


def portfolio_daily(matrix, mstart, mend, members):
    """Equal-weight daily pnl across `members`, using only cells non-NaN (active) that day."""
    if not members:
        idx = matrix.loc[mstart:mend].index
        return pd.Series(0.0, index=idx)
    sub = matrix.loc[mstart:mend, members]
    return sub.mean(axis=1, skipna=True).fillna(0.0)


def run_switch(matrix, meta, months, W):
    curve = []
    log = []
    for mstart, mend in months:
        pool = eligible_pool(meta, mstart, mend)
        on = decide_on_set(matrix, meta, mstart, pool, W)
        d = portfolio_daily(matrix, mstart, mend, on)
        curve.append(d)
        log.append(dict(month=mstart.strftime('%Y-%m'), pool_n=len(pool), on_n=len(on),
                         on_set=','.join(sorted(on))))
    daily = pd.concat(curve).sort_index()
    return daily, pd.DataFrame(log)


def run_equal_weight_always_on(matrix):
    return matrix.mean(axis=1, skipna=True).fillna(0.0)


def run_random_null(matrix, meta, months, k_by_month, n_seeds=N_SEEDS, seed0=SEED0):
    nets, dds, shs = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed0 + seed)
        curve = []
        for (mstart, mend), k in zip(months, k_by_month):
            pool = eligible_pool(meta, mstart, mend)
            k = min(k, len(pool))
            chosen = list(rng.choice(pool, size=k, replace=False)) if k > 0 else []
            curve.append(portfolio_daily(matrix, mstart, mend, chosen))
        daily = pd.concat(curve).sort_index()
        eq = daily.cumsum().to_numpy()
        nets.append(eq[-1] if len(eq) else 0.0)
        dds.append(maxdd(eq))
        shs.append(sharpe_ish(daily.to_numpy()))
    return np.array(nets), np.array(dds), np.array(shs)


def summarize(name, daily):
    eq = daily.cumsum().to_numpy()
    return dict(name=name, net=eq[-1] if len(eq) else 0.0, maxdd=maxdd(eq),
                sharpe=sharpe_ish(daily.to_numpy()), n_days=len(daily))


def main():
    matrix = pd.read_csv(IN_DIR / 'daily_pnl_matrix.csv', index_col=0, parse_dates=True)
    meta = pd.read_csv(IN_DIR / 'group_meta.csv', parse_dates=['first_date', 'last_date'])
    months = month_bounds(matrix.index)
    print(f"Matrix: {matrix.shape[0]} days x {matrix.shape[1]} groups, "
          f"{matrix.index.min().date()} -> {matrix.index.max().date()}, {len(months)} calendar months")
    print(f"Months: {[m[0].strftime('%Y-%m') for m in months]}")

    # ---- baselines ----
    results = []
    ew = run_equal_weight_always_on(matrix)
    results.append(summarize('equal_weight_always_on', ew))

    solo_rows = []
    for col in matrix.columns:
        s = maxdd_col = matrix[col].dropna()
        eq = s.cumsum().to_numpy()
        solo_rows.append(dict(name=col, net=eq[-1] if len(eq) else 0.0, maxdd=maxdd(eq),
                               sharpe=sharpe_ish(s.to_numpy()), n_days=len(s)))
    solo_df = pd.DataFrame(solo_rows).sort_values('net', ascending=False)
    solo_df.to_csv(OUT_DIR / 'solo_strategy_summary.csv', index=False)

    # ---- treatment: monthly-rebalanced equity-MA switch, sweep W ----
    switch_logs = {}
    for W in W_SWEEP:
        daily, log = run_switch(matrix, meta, months, W)
        results.append(summarize(f'switch_W{W}', daily))
        switch_logs[W] = log
        log.to_csv(OUT_DIR / f'switch_log_W{W}.csv', index=False)

    # ---- R10 random null, matched to the W=20 switch's monthly slot counts ----
    W_REF = 20
    ref_log = switch_logs[W_REF]
    k_by_month = list(ref_log['on_n'])
    rnd_net, rnd_dd, rnd_sh = run_random_null(matrix, meta, months, k_by_month)
    results.append(dict(name=f'random_null_mean(k=switchW{W_REF})', net=float(rnd_net.mean()),
                         maxdd=float(rnd_dd.mean()), sharpe=float(rnd_sh.mean()), n_days=matrix.shape[0]))

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / 'baseline_vs_treatment.csv', index=False)

    switch20 = res_df[res_df.name == f'switch_W{W_REF}'].iloc[0]
    # "beats" = switch is better than that fraction of random draws (net: higher is better;
    # maxdd: less-negative i.e. numerically greater is better).
    pct_net = float((switch20['net'] > rnd_net).mean() * 100)
    pct_dd = float((switch20['maxdd'] > rnd_dd).mean() * 100)

    print("\n=== Baselines vs treatment ===")
    print(res_df.to_string(index=False))
    print(f"\nSwitch W={W_REF} vs random null (100 seeds): "
          f"net beats {pct_net:.0f}% of random draws, maxDD beats {pct_dd:.0f}% of random draws")
    print(f"Random null net: mean={rnd_net.mean():.0f} std={rnd_net.std():.0f} "
          f"[{np.percentile(rnd_net,5):.0f}, {np.percentile(rnd_net,95):.0f}] (5-95pct)")
    print(f"Random null maxDD: mean={rnd_dd.mean():.0f} std={rnd_dd.std():.0f} "
          f"[{np.percentile(rnd_dd,95):.0f}, {np.percentile(rnd_dd,5):.0f}] (95-5pct, less-neg to more-neg)")

    np.savez(OUT_DIR / 'random_null_dist.npz', net=rnd_net, dd=rnd_dd, sharpe=rnd_sh)

    print("\n=== Top 10 solo strategies by net ===")
    print(solo_df.head(10).to_string(index=False))
    print("\n=== Bottom 10 solo strategies by net ===")
    print(solo_df.tail(10).to_string(index=False))

    print(f"\n=== Switch W={W_REF} monthly log ===")
    print(ref_log.to_string(index=False))

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == '__main__':
    main()
