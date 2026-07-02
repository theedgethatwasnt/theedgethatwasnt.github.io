"""
Trailing-first hybrid experiment — GBP_JPY zw=56 tgt=28.

Baseline: leg 1 exits at fixed +28p target, then zone recovery if needed.

Hybrid: leg 1 uses a trailing stop.
  - Trail activates once MFE >= activation_pips
  - Trail distance = trail_pips behind MFE (never moves against direction)
  - When trail_mode=True (single-leg): fixed target exit is SUPPRESSED —
    trail is the only exit mechanism for leg 1.
  - If trail fires BEFORE zone boundary crossed: 1-leg exit at (MFE - trail_pips)
  - If zone boundary crossed BEFORE trail fires: zone recovery takes over
    (leg 1 still open, recovery legs added as normal with fixed 28p targets)
  - Zone recovery exit (target hit): closes ALL legs including leg 1

PART A: Broad sweep (activation × distance) — reference
PART B: activation=28 sweep — "start trail AFTER passing initial target"
  Fixed target exit is suppressed; trail activates exactly at the 28p target.
  This is the key fix vs prior run where target exit competed with trail.

All results at 1,000u. GBP_JPY pip_usd = 0.000091.
"""

import sys, os, math
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR     = 'GBP_JPY'
PIP      = 0.01
PIP_USD  = 0.000091
UNITS    = 1_000
ZW       = 56
TGT      = 28
MAX_LEGS = 10
PF       = 1.19
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


def breakeven_vol(legs, target):
    net = net_basket(legs, target)
    if net >= 0:
        return 0.0
    return max(1.0, math.ceil(-net / TGT * PF))


def simulate(activation_pips=None, trail_pips=None):
    """
    activation_pips=None → fixed target (baseline).
    Otherwise leg 1 trails; zone recovery legs use fixed target.
    """
    zone_w = ZW  * PIP
    tgt_b  = TGT * PIP
    rng    = np.random.RandomState(42)
    cycles = []
    i      = 0

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
        entry_bar    = i
        last_crossed = last_crossed_bar = None
        closed       = False
        exit_reason  = 'eod'
        exit_price   = entry
        exit_bar     = i

        # Trailing state for leg 1
        mfe_pips    = 0.0      # max favourable excursion of leg 1 (pips)
        trail_stop  = None     # price level of the trailing stop (None = not yet active)
        trail_mode  = False    # True once trail has activated
        zone_crossed = False   # True once zone recovery has started (≥2 legs)

        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]
            lo = low_a[i]
            cl = close_a[i]
            bullish = cl >= open_a[i]

            # Update MFE and trail stop for leg 1 (only before zone recovery starts)
            if activation_pips is not None and not zone_crossed:
                # MFE of leg 1
                if direction == 1:
                    bar_mfe = (hi - entry) / PIP
                else:
                    bar_mfe = (entry - lo) / PIP
                if bar_mfe > mfe_pips:
                    mfe_pips = bar_mfe

                # Activate trail
                if not trail_mode and mfe_pips >= activation_pips:
                    trail_mode = True

                if trail_mode:
                    # Update trail stop level
                    if direction == 1:
                        new_stop = entry + (mfe_pips - trail_pips) * PIP
                        if trail_stop is None or new_stop > trail_stop:
                            trail_stop = new_stop
                    else:
                        new_stop = entry - (mfe_pips - trail_pips) * PIP
                        if trail_stop is None or new_stop < trail_stop:
                            trail_stop = new_stop

                    # Check if trail stop hit this bar
                    trail_hit = (direction == 1 and lo <= trail_stop) or \
                                (direction == -1 and hi >= trail_stop)
                    if trail_hit:
                        exit_price  = trail_stop
                        exit_reason = 'trail_stop'
                        exit_bar    = i
                        closed      = True
                        break

            seq = [(hi, True), (lo, False)] if bullish else [(lo, False), (hi, True)]

            for extreme, is_high in seq:
                if closed: break

                # Fixed target exits (whole basket).
                # Suppressed when leg-1 trail is active (single-leg mode) —
                # trail is now the sole exit for leg 1; zone recovery legs
                # restore fixed-target exits once zone_crossed=True.
                if not (trail_mode and not zone_crossed):
                    if is_high and hi >= upper_target:
                        exit_price, exit_reason, exit_bar = upper_target, 'target', i
                        closed = True; break
                    if not is_high and lo <= lower_target:
                        exit_price, exit_reason, exit_bar = lower_target, 'target', i
                        closed = True; break

                # Zone crossings → zone recovery
                if is_high and hi >= upper_zone:
                    if not (last_crossed == 'upper' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'upper', i
                        vol = breakeven_vol(legs, upper_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': 1, 'price': upper_zone, 'vol': vol})
                            zone_crossed = True

                if not is_high and lo <= lower_zone:
                    if not (last_crossed == 'lower' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'lower', i
                        vol = breakeven_vol(legs, lower_target)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': -1, 'price': lower_zone, 'vol': vol})
                            zone_crossed = True

            if not closed:
                i += 1

        net  = net_basket(legs, exit_price)
        cycles.append({
            'n_legs':       len(legs),
            'net_pips':     net,
            'exit_reason':  exit_reason,
            'mfe_pips':     mfe_pips,
            'trail_mode':   trail_mode,
            'zone_crossed': zone_crossed,
            'duration_bars': exit_bar - entry_bar,
        })

        if not closed:
            break

    return cycles


def report(label, cycles):
    df = pd.DataFrame(cycles)
    total_pips  = df['net_pips'].sum()
    total_usd   = total_pips * PIP_USD * UNITS
    avg_pips    = df['net_pips'].mean()
    n_target    = (df['exit_reason'] == 'target').sum()
    n_trail     = (df['exit_reason'] == 'trail_stop').sum()
    n_ml        = (df['exit_reason'] == 'max_legs').sum()
    avg_legs    = df['n_legs'].mean()
    best        = df['net_pips'].max()
    worst       = df['net_pips'].min()
    print(f"  {label:<45} | ${total_usd:>+10,.0f} | {avg_pips:>+7.0f}p/cyc | "
          f"tgt={n_target:>4} trail={n_trail:>4} ml={n_ml} | "
          f"legs={avg_legs:.2f} | best={best:+.0f}p worst={worst:+.0f}p")
    return total_usd


# ── Baseline ──────────────────────────────────────────────────────────────
print("="*130)
print(f"  TRAILING-FIRST HYBRID — {PAIR}  zw={ZW}  tgt={TGT}  @{UNITS:,}u")
print("="*130)
print(f"  {'Config':<45} | {'Total $@1ku':>12} | {'$/cyc':>9} | {'exits':>22} | "
      f"{'avgLegs':>7} | extremes")
print("─"*130)

base_cycles = simulate(activation_pips=None, trail_pips=None)
base_usd    = report("BASELINE (fixed 28p target)", base_cycles)

# ── PART A: Broad sweep (activation × distance) ───────────────────────────
print()
print("  PART A — broad sweep (with target-suppression fix)")
print("─"*130)
activations = [14, 28, 56, 84]      # pips MFE before trail activates
distances   = [7, 14, 21, 28]       # pips trail distance behind MFE

for act in activations:
    for dist in distances:
        if dist >= act:
            continue
        c = simulate(activation_pips=act, trail_pips=dist)
        usd = report(f"trail: activate={act}p  distance={dist}p", c)

# ── PART B: activation=TGT — "start trail AFTER passing initial target" ───
print()
print("  PART B — activation=28p (trail replaces fixed target exit)")
print("─"*130)
distances_b = [7, 14, 21, 28, 35, 42]

for dist in distances_b:
    c = simulate(activation_pips=TGT, trail_pips=dist)
    usd = report(f"trail: activate={TGT}p (=TGT)  distance={dist}p", c)

print()
print("─"*130)
print(f"  Baseline: ${base_usd:>+,.0f}")
print("="*130)
