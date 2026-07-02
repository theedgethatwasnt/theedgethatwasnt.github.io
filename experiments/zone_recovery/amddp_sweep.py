"""
AMDDP-per-stem sweep for Zone Recovery.

Metric: pain_ratio = accumulated_drawdown_pips / net_profit_pips per cycle
        (lower = faster recovery, shallower hole relative to reward)

Also tracks AUDDC (area under drawdown curve) = same thing, pip-bars.

Sweeps: ZW × TGT × PF (profit factor multiplier on break-even vol).

Scoring per config:
  - total_pips (OOS)
  - pain_ratio (avg AMDDP / avg net profit per cycle)
  - auddc_per_pip (pip-bars of drawdown per pip earned)
  - cycles_per_1000bars
  - avg_legs
  - worst_drawdown (max single-cycle accumulated DD)

Analytical insight (Lagrange, book §5.3.6):
  For fixed (ZW, TGT, N), minimum-exposure sizing satisfying break-even is convex:
    V_k ∝ k^0.5 (incremental), cumulative ∝ k^1.5
  Our break-even engine already achieves this implicitly when PF=1.0 — it IS
  the Lagrange-optimal solution. PF>1 adds profit margin without changing the
  shape (still convex, NOT martingale). We cap PF at 1.25 — no aggressive vol.

  AMDDP_ratio ≈ ZW × ΣV_j / (TGT × PF)
  → Analytical prediction: pain_ratio ↓ when TGT/ZW ↑ (and PF ↑).
  → Empirical sweep tests whether hit-rate erosion at large TGT outweighs this.
"""

import os, math
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR     = 'GBP_JPY'
PIP      = 0.01
PIP_USD  = 0.000091
UNITS    = 1_000
MAX_LEGS = 10
SPREAD   = 1.4


df = pd.read_parquet(f'{DATA_DIR}/{PAIR}_M5.parquet').sort_index()
df.columns = [c.lower() for c in df.columns]
n = len(df)
n_train = int(n * 0.70)
df_oos = df.iloc[n_train:].reset_index(drop=True)
open_a  = df_oos['open'].values.astype(np.float64)
high_a  = df_oos['high'].values.astype(np.float64)
low_a   = df_oos['low'].values.astype(np.float64)
close_a = df_oos['close'].values.astype(np.float64)
n_oos   = len(close_a)


def net_basket(legs, price):
    gross = sum(l['vol'] * l['dir'] * (price - l['price']) / PIP for l in legs)
    cost  = sum(l['vol'] for l in legs) * SPREAD
    return gross - cost


def simulate(zw: int, tgt: int, pf: float = 1.19) -> list:
    zone_w = zw  * PIP
    tgt_b  = tgt * PIP
    rng    = np.random.RandomState(42)
    cycles = []
    i      = 0

    def breakeven_vol(legs, target):
        net = net_basket(legs, target)
        if net >= 0:
            return 0.0
        return max(1.0, math.ceil(-net / tgt * pf))

    while i < n_oos:
        entry     = close_a[i]
        direction = int(rng.choice([-1, 1]))

        if direction == 1:
            upper_zone   = entry
            lower_zone   = entry - zone_w
            upper_target = entry + tgt_b
            lower_target = lower_zone - tgt_b
        else:
            lower_zone   = entry
            upper_zone   = entry + zone_w
            lower_target = entry - tgt_b
            upper_target = upper_zone + tgt_b

        legs         = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        entry_bar    = i
        last_crossed = last_crossed_bar = None
        closed       = False
        exit_reason  = 'eod'
        exit_price   = entry
        exit_bar     = i

        # AMDDP tracking
        accumulated_dd  = 0.0   # Σ |UPnL| on bars where UPnL < 0
        auddc           = 0.0   # Area under drawdown curve (pip-bars)
        max_dd_depth    = 0.0   # Worst instantaneous UPnL

        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]
            lo = low_a[i]
            cl = close_a[i]
            bullish = cl >= open_a[i]

            # Track drawdown BEFORE exit check (using bar close)
            upnl_now = net_basket(legs, cl)
            if upnl_now < 0:
                accumulated_dd += abs(upnl_now)
                auddc          += abs(upnl_now)  # each bar = 1 unit time
            if upnl_now < max_dd_depth:
                max_dd_depth = upnl_now

            seq = [(hi, True), (lo, False)] if bullish else [(lo, False), (hi, True)]

            for extreme, is_high in seq:
                if closed: break

                if is_high and hi >= upper_target:
                    exit_price, exit_reason, exit_bar = upper_target, 'target', i
                    closed = True; break
                if not is_high and lo <= lower_target:
                    exit_price, exit_reason, exit_bar = lower_target, 'target', i
                    closed = True; break

                if is_high and hi >= upper_zone:
                    if not (last_crossed == 'upper' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'upper', i
                        vol = breakeven_vol(legs, upper_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': 1, 'price': upper_zone, 'vol': vol})

                if not is_high and lo <= lower_zone:
                    if not (last_crossed == 'lower' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'lower', i
                        vol = breakeven_vol(legs, lower_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': -1, 'price': lower_zone, 'vol': vol})

            if not closed:
                i += 1

        net = net_basket(legs, exit_price)
        cycles.append({
            'net_pips':       net,
            'exit_reason':    exit_reason,
            'n_legs':         len(legs),
            'duration_bars':  exit_bar - entry_bar,
            'accumulated_dd': accumulated_dd,   # sum of |UPnL| on negative bars
            'auddc':          auddc,             # same — pip-bars
            'max_dd_depth':   abs(max_dd_depth), # worst single-bar UPnL depth
        })

        if not closed:
            break

    return cycles


def score(cycles: list, zw: int, tgt: int, pf: float) -> dict:
    df = pd.DataFrame(cycles)
    n_cyc       = len(df)
    total_pips  = df['net_pips'].sum()
    avg_pips    = df['net_pips'].mean()
    avg_legs    = df['n_legs'].mean()
    n_target    = (df['exit_reason'] == 'target').sum()
    n_ml        = (df['exit_reason'] == 'max_legs').sum()

    # Pain-per-dollar: how many pips of drawdown per pip of profit earned
    # Only for winning cycles (positive net) to avoid divide-by-tiny
    winners = df[df['net_pips'] > 0]
    if len(winners) > 0:
        pain_ratio    = (winners['accumulated_dd'] / winners['net_pips']).mean()
        auddc_per_pip = (winners['auddc'] / winners['net_pips']).mean()
    else:
        pain_ratio    = float('inf')
        auddc_per_pip = float('inf')

    worst_dd   = df['accumulated_dd'].max()
    cycles_per_1k = n_cyc / n_oos * 1000

    return {
        'zw': zw, 'tgt': tgt, 'pf': pf,
        'ratio': tgt / zw,
        'total_pips': total_pips,
        'total_usd':  total_pips * PIP_USD * UNITS,
        'avg_pips':   avg_pips,
        'n_cyc':      n_cyc,
        'cyc_per_1k': cycles_per_1k,
        'pct_target': n_target / n_cyc * 100,
        'pct_ml':     n_ml / n_cyc * 100,
        'avg_legs':   avg_legs,
        'pain_ratio': pain_ratio,
        'auddc_per_pip': auddc_per_pip,
        'worst_dd':   worst_dd,
    }


# ── Sweep ─────────────────────────────────────────────────────────────────
# PF constraint: break-even formula is already Lagrange-optimal (convex, not martingale).
# PF > 1.0 is the profit lever — we keep it CONSERVATIVE (no aggressive multipliers).
# PF=1.0 = pure break-even (zero profit target), PF=1.25 = upper bound.
ZW_VALS  = [28, 42, 56, 70, 84]
TGT_VALS = [14, 21, 28, 35, 42, 56]
PF_VALS  = [1.0, 1.05, 1.10, 1.15, 1.19, 1.25]

print("="*120)
print(f"  AMDDP SWEEP — {PAIR}  @{UNITS:,}u  OOS={n_oos:,} bars")
print(f"  Analytical prediction: pain_ratio ∝ ZW/TGT, so maximize TGT/ZW ratio")
print("="*120)
print(f"  {'ZW':>4} {'TGT':>4} {'PF':>5} {'Ratio':>6} | {'TotalPips':>10} {'TotalUSD':>10} | "
      f"{'Cyc':>5} {'Cyc/1k':>7} {'%Tgt':>6} {'%ML':>5} {'AvgLegs':>8} | "
      f"{'PainRatio':>10} {'AUDDC/p':>9} {'WorstDD':>9}")
print("─"*120)

results = []
for zw in ZW_VALS:
    for tgt in TGT_VALS:
        if tgt > zw * 2:   # impractical — target too far beyond zone
            continue
        if tgt < zw // 4:  # target too tiny
            continue
        for pf in PF_VALS:
            cyc = simulate(zw=zw, tgt=tgt, pf=pf)
            r   = score(cyc, zw, tgt, pf)
            results.append(r)
            print(f"  {r['zw']:>4} {r['tgt']:>4} {r['pf']:>5.2f} {r['ratio']:>6.2f} | "
                  f"{r['total_pips']:>+10,.0f} {r['total_usd']:>+10,.0f} | "
                  f"{r['n_cyc']:>5} {r['cyc_per_1k']:>7.1f} {r['pct_target']:>6.1f} {r['pct_ml']:>5.1f} "
                  f"{r['avg_legs']:>8.2f} | "
                  f"{r['pain_ratio']:>10.2f} {r['auddc_per_pip']:>9.2f} {r['worst_dd']:>9.0f}")
    print()

# ── Analysis ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_pos = df_res[df_res['total_pips'] > 0].copy()

print("="*120)
print("  PARETO FRONTIER: Best pain_ratio among profitable configs (total_pips > 0)")
print("─"*120)
pareto = df_pos.nsmallest(15, 'pain_ratio')
for _, r in pareto.iterrows():
    print(f"  zw={r['zw']:>3}  tgt={r['tgt']:>3}  pf={r['pf']:.2f}  ratio={r['ratio']:.2f} | "
          f"pain={r['pain_ratio']:.2f}  total={r['total_pips']:>+,.0f}p  cyc={r['n_cyc']:>4}  legs={r['avg_legs']:.1f}")

print()
print("  TOP CONFIGS BY TOTAL PIPS (profitable only)")
print("─"*120)
top = df_pos.nlargest(10, 'total_pips')
for _, r in top.iterrows():
    print(f"  zw={r['zw']:>3}  tgt={r['tgt']:>3}  pf={r['pf']:.2f}  ratio={r['ratio']:.2f} | "
          f"pain={r['pain_ratio']:.2f}  total={r['total_pips']:>+,.0f}p  cyc={r['n_cyc']:>4}  legs={r['avg_legs']:.1f}")

print()
print("  ANALYTICAL VALIDATION: correlation of TGT/ZW ratio with pain_ratio")
corr = df_pos[['ratio', 'pain_ratio']].corr().iloc[0, 1]
print(f"  Pearson r(ratio, pain_ratio) = {corr:+.3f}  (predicted negative: larger ratio → less pain)")

print()
print("  BEST BALANCED CONFIG: maximize total_pips / pain_ratio (profit per unit of pain)")
df_pos['efficiency'] = df_pos['total_pips'] / df_pos['pain_ratio'].clip(lower=0.01)
best = df_pos.loc[df_pos['efficiency'].idxmax()]
print(f"  zw={best['zw']:.0f}  tgt={best['tgt']:.0f}  pf={best['pf']:.2f}")
print(f"  total_pips={best['total_pips']:>+,.0f}  pain_ratio={best['pain_ratio']:.2f}  "
      f"efficiency={best['efficiency']:,.0f}")
print("="*120)
