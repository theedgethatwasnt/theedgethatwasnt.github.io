"""
Stop-optimization experiment for GBP_JPY zw=56 tgt=28.
Tests three stop mechanisms at 1,000u:
  A. Max-leg stop:   force close when N legs reached (vs current max=10)
  B. Time stop:      force close if cycle age > N bars (M5)
  C. Equity stop:    force close if basket unrealized P&L < -X pips

For each mechanism + threshold, compute vs baseline (no stop, max_legs=10):
  - Total net P&L (pips + dollars @ 1000u)
  - Stop-out count + avg loss per stop + worst single stop loss
  - Net cycle P&L delta vs baseline
"""

import sys, os, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from boundary_engine import BoundaryZoneEngine, _net_at_target
from engine import Leg, get_pl_pips_at_target, PIP

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR      = 'GBP_JPY'
PIP_SIZE  = 0.01
PIP_USD   = 0.000091          # $/pip/unit for GBP_JPY
UNITS     = 1_000             # simulation scale
ZW        = 56
TGT       = 28
MAX_LEGS  = 10
PF        = 1.19
SPREAD    = 1.4

# ── Load OOS data ─────────────────────────────────────────────────────────
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


def _net_basket(legs: list, price: float) -> float:
    gross = sum(l['vol'] * l['dir'] * (price - l['price']) / PIP_SIZE for l in legs)
    cost  = sum(l['vol'] for l in legs) * SPREAD
    return gross - cost


def simulate(max_legs_stop: int = 10,
             time_stop_bars: Optional[int] = None,
             equity_stop_pips: Optional[float] = None):
    """
    Run OOS simulation with given stop settings.
    Returns list of cycle dicts with full P&L detail.
    """
    pip    = PIP_SIZE
    zone_w = ZW * pip
    tgt_b  = TGT * pip
    tgt_p  = float(TGT)

    def breakeven_vol(legs, target):
        net = _net_basket(legs, target)
        if net >= 0:
            return 0.0
        return max(1.0, math.ceil(-net / tgt_p * PF))

    cycles = []
    i = 0
    rng = np.random.RandomState(42)

    while i < n_oos:
        entry = close_a[i]
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

        legs = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        entry_bar = i
        last_crossed = last_crossed_bar = None
        closed = False
        exit_reason = 'eod'
        exit_price  = entry
        exit_bar    = i

        i += 1
        while i < n_oos and not closed:
            hi = high_a[i]
            lo = low_a[i]
            cl = close_a[i]
            bullish = cl >= open_a[i]

            # Time stop
            if time_stop_bars and (i - entry_bar) >= time_stop_bars:
                exit_reason, exit_price, exit_bar = 'time_stop', cl, i
                closed = True; break

            # Equity stop (check vs bar close)
            if equity_stop_pips is not None:
                net_now = _net_basket(legs, cl)
                if net_now <= -equity_stop_pips:
                    exit_reason, exit_price, exit_bar = 'equity_stop', cl, i
                    closed = True; break

            seq = [(hi, True), (lo, False)] if bullish else [(lo, False), (hi, True)]

            for extreme, is_high in seq:
                if closed: break

                if is_high and hi >= upper_target:
                    exit_reason, exit_price, exit_bar = 'target', upper_target, i
                    closed = True; break
                if not is_high and lo <= lower_target:
                    exit_reason, exit_price, exit_bar = 'target', lower_target, i
                    closed = True; break

                if is_high and hi >= upper_zone:
                    if not (last_crossed == 'upper' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'upper', i
                        vol = breakeven_vol(legs, upper_target)
                        if vol > 0:
                            if len(legs) >= max_legs_stop:
                                exit_reason, exit_price, exit_bar = 'max_legs', cl, i
                                closed = True; break
                            legs.append({'dir': 1, 'price': upper_zone, 'vol': vol})

                if not is_high and lo <= lower_zone:
                    if not (last_crossed == 'lower' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'lower', i
                        vol = breakeven_vol(legs, lower_target)
                        if vol > 0:
                            if len(legs) >= max_legs_stop:
                                exit_reason, exit_price, exit_bar = 'max_legs', cl, i
                                closed = True; break
                            legs.append({'dir': -1, 'price': lower_zone, 'vol': vol})

            if not closed:
                i += 1

        net = _net_basket(legs, exit_price)
        cycles.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'n_legs': len(legs),
            'net_pips': net,
            'exit_reason': exit_reason,
            'duration_bars': exit_bar - entry_bar,
        })

        if not closed:
            break

    return cycles


def report(label: str, cycles: list):
    df = pd.DataFrame(cycles)
    total_pips = df['net_pips'].sum()
    total_usd  = total_pips * PIP_USD * UNITS

    stops = df[df['exit_reason'] != 'target']
    targets = df[df['exit_reason'] == 'target']

    stop_count = len(stops)
    stop_pips  = stops['net_pips'].sum() if stop_count else 0.0
    worst_stop = stops['net_pips'].min() if stop_count else 0.0
    avg_stop   = stops['net_pips'].mean() if stop_count else 0.0

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Cycles:       {len(df):>6}  |  Target exits: {len(targets):>5}  |  Stop exits: {stop_count:>4}")
    print(f"  Total pips:   {total_pips:>+10,.0f}  |  Total $@1ku:  ${total_usd:>+10,.2f}")
    print(f"  Avg pips/cyc: {df['net_pips'].mean():>+10.1f}  |  Avg legs:     {df['n_legs'].mean():>6.2f}")
    if stop_count:
        print(f"  Stop losses:  {stop_count:>6}  |  Avg loss:     {avg_stop:>+10.1f}p  (${avg_stop*PIP_USD*UNITS:>+8.2f})")
        print(f"  Worst stop:   {worst_stop:>+10.1f}p  (${worst_stop*PIP_USD*UNITS:>+8.2f})")
        print(f"  Stop P&L tot: {stop_pips:>+10.1f}p  (${stop_pips*PIP_USD*UNITS:>+8.2f})")
    return total_pips, total_usd


# ── Baseline ──────────────────────────────────────────────────────────────
print("="*70)
print(f"  STOP OPTIMIZATION — {PAIR}  zw={ZW}  tgt={TGT}  @{UNITS:,}u")
print("="*70)

base_cycles = simulate(max_legs_stop=10)
base_pips, base_usd = report("BASELINE (max_legs=10, no other stops)", base_cycles)

# ── A: Max-leg stops ───────────────────────────────────────────────────────
print("\n\n  ═══ A. MAX-LEG STOP (reduce ceiling) ═══")
for ml in [5, 6, 7, 8]:
    c = simulate(max_legs_stop=ml)
    report(f"Max-leg stop = {ml}", c)

# ── B: Time stops ─────────────────────────────────────────────────────────
print("\n\n  ═══ B. TIME STOP (bars = M5 periods) ═══")
for hrs in [12, 24, 48, 96]:
    bars = hrs * 12  # M5 bars per hour × hours
    c = simulate(time_stop_bars=bars)
    report(f"Time stop = {hrs}h ({bars} bars)", c)

# ── C: Equity stops ───────────────────────────────────────────────────────
print("\n\n  ═══ C. EQUITY STOP (basket unrealized pips) ═══")
for ep in [50, 100, 150, 200, 300, 500]:
    c = simulate(equity_stop_pips=ep)
    report(f"Equity stop = -{ep}p (${ep*PIP_USD*UNITS:,.2f}@1ku)", c)

# ── D: Combos ─────────────────────────────────────────────────────────────
print("\n\n  ═══ D. COMBINATIONS ═══")
combos = [
    dict(max_legs_stop=7, equity_stop_pips=200),
    dict(max_legs_stop=6, equity_stop_pips=150),
    dict(max_legs_stop=7, time_stop_bars=48*12),
    dict(max_legs_stop=6, equity_stop_pips=200, time_stop_bars=48*12),
]
for kw in combos:
    label = " + ".join(f"{k}={v}" for k,v in kw.items())
    c = simulate(**kw)
    report(label, c)

print(f"\n{'═'*70}")
print(f"  Baseline reference: {base_pips:>+,.0f}p  /  ${base_usd:>+,.2f}  (no stops except ml=10)")
print(f"{'═'*70}\n")
