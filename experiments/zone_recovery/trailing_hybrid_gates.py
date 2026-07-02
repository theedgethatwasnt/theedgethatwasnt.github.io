"""
5-gate validation for trailing hybrid winner: activate=14p, trail=7p
on GBP_JPY zw=56 tgt=28.

Gate 1: OOS total positive
Gate 2: Walk-forward — all 3 OOS chunks positive
Gate 3: Permutation p < 0.05 (shuffle direction assignment)
Gate 4: Seed CV — seeds 42-46 all positive
Gate 5: SQN > 1.0  (mean/std × sqrt(N))
"""

import sys, os, math
import numpy as np
import pandas as pd

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

# Winner config
ACT_PIPS   = 14
TRAIL_PIPS = 7

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


def simulate(activation_pips=None, trail_pips=None, seed=42,
             idx_start=0, idx_end=None):
    if idx_end is None:
        idx_end = n_oos
    zone_w = ZW  * PIP
    tgt_b  = TGT * PIP
    rng    = np.random.RandomState(seed)
    cycles = []
    i      = idx_start

    while i < idx_end:
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

        mfe_pips    = 0.0
        trail_stop  = None
        trail_mode  = False
        zone_crossed = False

        i += 1

        while i < idx_end and not closed:
            hi = high_a[i]
            lo = low_a[i]
            cl = close_a[i]
            bullish = cl >= open_a[i]

            if activation_pips is not None and not zone_crossed:
                if direction == 1:
                    bar_mfe = (hi - entry) / PIP
                else:
                    bar_mfe = (entry - lo) / PIP
                if bar_mfe > mfe_pips:
                    mfe_pips = bar_mfe

                if not trail_mode and mfe_pips >= activation_pips:
                    trail_mode = True

                if trail_mode:
                    if direction == 1:
                        new_stop = entry + (mfe_pips - trail_pips) * PIP
                        if trail_stop is None or new_stop > trail_stop:
                            trail_stop = new_stop
                    else:
                        new_stop = entry - (mfe_pips - trail_pips) * PIP
                        if trail_stop is None or new_stop < trail_stop:
                            trail_stop = new_stop

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

                if not (trail_mode and not zone_crossed):
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
        cycles.append({'net_pips': net, 'exit_reason': exit_reason,
                       'n_legs': len(legs), 'duration_bars': exit_bar - entry_bar})
        if not closed:
            break

    return cycles


def pips(cycles):
    return pd.DataFrame(cycles)['net_pips'].sum()


def usd(p):
    return p * PIP_USD * UNITS


print("="*70)
print(f"  5-GATE VALIDATION — Trailing Hybrid activate={ACT_PIPS}p trail={TRAIL_PIPS}p")
print(f"  {PAIR}  zw={ZW}  tgt={TGT}  @{UNITS:,}u")
print("="*70)

# ── Gate 1: OOS total positive ────────────────────────────────────────────
base_c  = simulate(activation_pips=None,      trail_pips=None,       seed=42)
trail_c = simulate(activation_pips=ACT_PIPS,  trail_pips=TRAIL_PIPS, seed=42)
base_p  = pips(base_c)
trail_p = pips(trail_c)
g1 = trail_p > 0
print(f"\n  Gate 1 — OOS positive")
print(f"    Baseline : {base_p:>+,.0f}p  (${usd(base_p):>+,.0f})")
print(f"    Hybrid   : {trail_p:>+,.0f}p  (${usd(trail_p):>+,.0f})")
print(f"    {'🟢 PASS' if g1 else '🔴 FAIL'}")

# ── Gate 2: Walk-forward (3 OOS chunks) ───────────────────────────────────
chunk = n_oos // 3
starts = [0, chunk, chunk*2]
ends   = [chunk, chunk*2, n_oos]
print(f"\n  Gate 2 — Walk-forward (3 OOS chunks of {chunk:,} bars each)")
chunk_pips = []
for k, (s, e) in enumerate(zip(starts, ends)):
    c = simulate(activation_pips=ACT_PIPS, trail_pips=TRAIL_PIPS,
                 seed=42, idx_start=s, idx_end=e)
    p = pips(c)
    chunk_pips.append(p)
    print(f"    Chunk {k+1} [{s:>6}–{e:>6}]: {p:>+,.0f}p  (${usd(p):>+,.0f})")
g2 = all(p > 0 for p in chunk_pips)
print(f"    {'🟢 PASS' if g2 else '🔴 FAIL'} ({sum(p>0 for p in chunk_pips)}/3 positive)")

# ── Gate 3: Permutation test ──────────────────────────────────────────────
N_PERM = 500
print(f"\n  Gate 3 — Permutation test ({N_PERM} shuffles of direction seed)")
perm_pips = []
for s in range(N_PERM):
    c = simulate(activation_pips=ACT_PIPS, trail_pips=TRAIL_PIPS, seed=1000+s)
    perm_pips.append(pips(c))
p_val = np.mean(np.array(perm_pips) >= trail_p)
g3 = p_val < 0.05
print(f"    Observed : {trail_p:>+,.0f}p")
print(f"    Perm mean: {np.mean(perm_pips):>+,.0f}p  ± {np.std(perm_pips):>,.0f}p")
print(f"    p-value  : {p_val:.4f}")
print(f"    {'🟢 PASS' if g3 else '🔴 FAIL'} (p < 0.05 required)")

# ── Gate 4: Seed CV (seeds 42-46 all positive) ───────────────────────────
print(f"\n  Gate 4 — Seed robustness (seeds 42–46)")
seed_pips = []
for s in range(42, 47):
    c = simulate(activation_pips=ACT_PIPS, trail_pips=TRAIL_PIPS, seed=s)
    p = pips(c)
    seed_pips.append(p)
    flag = '🟢' if p > 0 else '🔴'
    print(f"    Seed {s}: {p:>+,.0f}p  (${usd(p):>+,.0f})  {flag}")
g4 = all(p > 0 for p in seed_pips)
print(f"    {'🟢 PASS' if g4 else '🔴 FAIL'} ({sum(p>0 for p in seed_pips)}/5 positive)")

# ── Gate 5: SQN > 1.0 ────────────────────────────────────────────────────
df_t = pd.DataFrame(trail_c)
sqn  = (df_t['net_pips'].mean() / df_t['net_pips'].std()) * math.sqrt(len(df_t))
g5   = sqn > 1.0
print(f"\n  Gate 5 — SQN")
print(f"    Cycles   : {len(df_t):,}  |  mean: {df_t['net_pips'].mean():>+.1f}p  |  std: {df_t['net_pips'].std():>.1f}p")
print(f"    SQN      : {sqn:.3f}")
print(f"    {'🟢 PASS' if g5 else '🔴 FAIL'} (SQN > 1.0 required)")

# ── Summary ───────────────────────────────────────────────────────────────
gates  = [g1, g2, g3, g4, g5]
passed = sum(gates)
labels = ['OOS+', 'WF', 'Perm', 'SeedCV', 'SQN']
print(f"\n{'='*70}")
print(f"  RESULT: {passed}/5 gates passed")
for label, g in zip(labels, gates):
    print(f"    {'🟢' if g else '🔴'} {label}")
verdict = '🟢 DEPLOY CANDIDATE' if passed >= 4 else ('🟡 MARGINAL' if passed == 3 else '🔴 FAIL')
print(f"\n  {verdict}")
print(f"{'='*70}\n")
