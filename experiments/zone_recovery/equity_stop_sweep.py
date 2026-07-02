"""
Equity stop sweep for zone recovery.

Tests dollar-denominated equity stops: close cycle when basket UPnL (USD) < -stop_usd.

Stop is expressed as stop_usd_per_base_unit so results are portable across any BASE_UNITS:
  actual_stop_usd = stop_per_unit × BASE_UNITS_LIVE

Reports: total pips, % of cycles stopped early, capital needed, and ROI estimate.

Key question: what stop level preserves OOS edge while bounding max float loss?
"""

import os, math
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR     = 'GBP_JPY'
PIP      = 0.01
PIP_USD  = 0.000091   # $/pip/unit for GBP_JPY
UNITS    = 1_000      # display scale (does not affect sim logic)
ZW       = 56
TGT      = 28
MAX_LEGS = 10
PF       = 1.25
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


def simulate(equity_stop_pips=None) -> list:
    """
    equity_stop_pips: close when net_basket < -equity_stop_pips.
    None = no stop (baseline).
    Pips here are vol-weighted (same as net_basket units).
    Convert from dollar stop: equity_stop_pips = stop_usd / (PIP_USD * BASE_UNITS_LIVE)
    """
    zone_w = ZW  * PIP
    tgt_b  = TGT * PIP
    rng    = np.random.RandomState(42)
    cycles = []
    i      = 0

    def bvol(legs, target):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / TGT * PF))

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
        max_legs_reached = 0

        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]
            lo = low_a[i]
            cl = close_a[i]
            bullish = cl >= open_a[i]

            # Dollar equity stop — checked at bar close
            if equity_stop_pips is not None:
                net_now = net_basket(legs, cl)
                if net_now <= -equity_stop_pips:
                    exit_price, exit_reason, exit_bar = cl, 'equity_stop', i
                    closed = True
                    break

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
                        vol = bvol(legs, upper_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': 1, 'price': upper_zone, 'vol': vol})
                            max_legs_reached = max(max_legs_reached, len(legs))

                if not is_high and lo <= lower_zone:
                    if not (last_crossed == 'lower' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'lower', i
                        vol = bvol(legs, lower_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': -1, 'price': lower_zone, 'vol': vol})
                            max_legs_reached = max(max_legs_reached, len(legs))

            if not closed:
                i += 1

        net = net_basket(legs, exit_price)
        cycles.append({
            'net_pips':    net,
            'exit_reason': exit_reason,
            'n_legs':      len(legs),
            'max_legs':    max_legs_reached or len(legs),
            'duration_bars': exit_bar - entry_bar,
        })

        if not closed:
            break

    return cycles


def report(label, cycles, stop_pips=None):
    df = pd.DataFrame(cycles)
    n_cyc       = len(df)
    total_pips  = df['net_pips'].sum()
    total_usd   = total_pips * PIP_USD * UNITS
    n_target    = (df['exit_reason'] == 'target').sum()
    n_estop     = (df['exit_reason'] == 'equity_stop').sum()
    n_ml        = (df['exit_reason'] == 'max_legs').sum()
    avg_legs    = df['n_legs'].mean()
    worst_pips  = df['net_pips'].min()
    worst_usd   = worst_pips * PIP_USD * UNITS

    # Capital needed per BASE_UNIT = stop_usd / BASE_UNITS_LIVE
    # We express it as stop_pips × PIP_USD (in dollars at 1u base, scales linearly)
    cap_per_unit = stop_pips * PIP_USD if stop_pips else 0
    # At 88u for $500/yr: total_usd × 88/1000 = annual $ at 88u
    annual_88u   = total_usd * 88 / 1000 / 5      # 5yr OOS

    return {
        'label': label,
        'n_cyc': n_cyc, 'total_pips': total_pips, 'total_usd': total_usd,
        'n_target': n_target, 'n_estop': n_estop, 'n_ml': n_ml,
        'pct_target': n_target/n_cyc*100, 'pct_estop': n_estop/n_cyc*100,
        'avg_legs': avg_legs,
        'worst_pips': worst_pips, 'worst_usd': worst_usd,
        'cap_per_unit': cap_per_unit,
        'annual_88u': annual_88u,
        'stop_pips': stop_pips,
    }


# ── Baseline ─────────────────────────────────────────────────────────────
print("="*110)
print(f"  EQUITY STOP SWEEP — {PAIR}  zw={ZW}  tgt={TGT}  PF={PF}  @{UNITS:,}u display")
print(f"  Stop = close when basket UPnL < -stop_pips  (vol-weighted, 1u base)")
print(f"  Capital per unit = stop_pips × PIP_USD = stop_pips × {PIP_USD}")
print("="*110)

results = []

base_c = simulate(equity_stop_pips=None)
r0 = report("BASELINE (no stop)", base_c, stop_pips=None)
results.append(r0)

# Convert dollar-per-unit stops to basket pips
# stop_usd_per_unit → stop_basket_pips = stop_usd_per_unit / PIP_USD
# e.g. $0.10/unit → 0.10/0.000091 = 1099 basket pips

stop_usd_per_unit_list = [
    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50,
    1.00, 2.00, 5.00, 10.00,
]

for s_usd in stop_usd_per_unit_list:
    s_pips = s_usd / PIP_USD
    c = simulate(equity_stop_pips=s_pips)
    label = f"stop=${s_usd:.3f}/unit ({s_pips:,.0f}p)"
    r = report(label, c, stop_pips=s_pips)
    results.append(r)

# ── Print table ───────────────────────────────────────────────────────────
print(f"\n  {'Config':<32} | {'TotalPips':>10} {'TotalUSD@1ku':>13} | {'%Tgt':>6} {'%Stop':>6} {'%ML':>4} | {'AvgLeg':>7} | {'Worst$@1ku':>11} | {'Cap/unit':>9} | {'$/yr@88u':>10}")
print("─"*130)

for r in results:
    cap_str = f"${r['cap_per_unit']:.4f}" if r['cap_per_unit'] else "   —"
    annual  = f"${r['annual_88u']:>+,.0f}" if r['annual_88u'] else "   —"
    print(f"  {r['label']:<32} | {r['total_pips']:>+10,.0f} {r['total_usd']:>+13,.0f} | "
          f"{r['pct_target']:>6.1f} {r['pct_estop']:>6.1f} {r['n_ml']:>4} | "
          f"{r['avg_legs']:>7.2f} | {r['worst_usd']:>+11.2f} | "
          f"{cap_str:>9} | {annual:>10}")

# ── Find optimal ──────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)
df_pos = df_r[df_r['total_pips'] > 0].copy()

print()
print("="*110)
print("  OPTIMAL STOP: maximize (total_pips) subject to (total_pips > 0)")
print("  ROI = annual_$/yr @ 88u / (cap_per_unit × 2 accounts × 88u)")
if len(df_pos) > 1:
    df_pos2 = df_pos[df_pos['stop_pips'].notna()].copy()
    df_pos2['roi'] = df_pos2['annual_88u'] / (df_pos2['cap_per_unit'] * 2 * 88) * 100
    df_pos2_sorted = df_pos2.sort_values('roi', ascending=False)
    print()
    print(f"  {'Config':<32} | {'Total$@1ku':>12} | {'$/yr@88u':>10} | {'Cap/unit':>9} | {'Total cap':>10} | {'ROI%/yr':>9}")
    print("─"*100)
    for _, r in df_pos2_sorted.head(8).iterrows():
        total_cap = r['cap_per_unit'] * 2 * 88
        roi = r['annual_88u'] / total_cap * 100
        print(f"  {r['label']:<32} | {r['total_usd']:>+12,.0f} | {r['annual_88u']:>+10,.0f} | "
              f"${r['cap_per_unit']:>8.4f} | ${total_cap:>9.2f} | {roi:>9.0f}%")

print()
print(f"  Baseline (no stop): ${df_r.iloc[0]['annual_88u']:>+,.0f}/yr @ 88u, uncapped capital")
print("="*110)
