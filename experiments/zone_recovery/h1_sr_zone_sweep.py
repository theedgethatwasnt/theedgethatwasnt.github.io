"""
H1 TopsBots S/R as Zone Boundaries — Zone Recovery Sweep.

Idea: instead of fixed-pip zone width, use the H1 confirmed swing H/L
as the zone's upper (resistance) and lower (support) boundaries.
Zone width self-calibrates to actual market structure.

Entry:
  - DIRECTIONAL: at support → LONG, at resistance → SHORT
  - Entry when M5 close <= act_l (support) or >= act_h (resistance)
  - Zone frozen at cycle start; new H1 swings don't move active cycle's boundaries.

Sweep:
  - TGT_BEYOND as fraction of ZW: [0.25, 0.50, 0.75, 1.0]
  - ZW cap: [None, 80, 60] — cap the zone width to avoid extreme ranges
  - Compare: fixed ZW=56 tgt=28 (our production baseline)

Lookahead invariant:
  H1 bar i-1 is a confirmed Stage-1 swing ONLY when H1 bar i closes.
  We propagate H1 act_h/act_l forward across all M5 bars within each H1 period.
  No M5 bar ever sees a swing confirmed in its own H1 bar.
"""

import os, math
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR  = 'GBP_JPY'
PIP   = 0.01
PIP_USD = 0.000091
UNITS   = 1_000
MAX_LEGS = 10
PF       = 1.25
SPREAD   = 1.4
M5_PER_H1 = 12   # 12 × 5min = 1h


# ── Load M5 OOS data ──────────────────────────────────────────────────────────
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


# ── Build H1 bars from M5 ────────────────────────────────────────────────────
h1_hi, h1_lo, h1_end_m5 = [], [], []
for start in range(0, n_oos, M5_PER_H1):
    end = min(start + M5_PER_H1, n_oos)
    h1_hi.append(float(np.max(high_a[start:end])))
    h1_lo.append(float(np.min(low_a[start:end])))
    h1_end_m5.append(end - 1)

h1_hi = np.array(h1_hi)
h1_lo = np.array(h1_lo)
n_h1  = len(h1_hi)


# ── Causal TopsBots on H1 ────────────────────────────────────────────────────
def build_h1_topsbots_m5_array():
    """
    Run causal TopsBots on H1 H/L.
    Bar i-1 confirmed as Stage-1 extreme when bar i closes.
    Full Stage 1+2+3 applied incrementally (same algorithm as lib/swing_indicators.py).
    Returns act_h_m5, act_l_m5: arrays of length n_oos, NaN until first confirmed pair.
    """
    # Incremental Stage 3 state
    confirmed = []  # list of [idx, type, val]
    lh = ll = float('nan')
    glh = glh2 = False   # exceeding-extremes gate flags

    act_h_h1 = np.full(n_h1, float('nan'))
    act_l_h1 = np.full(n_h1, float('nan'))

    cur_act_h = cur_act_l = float('nan')

    for i in range(1, n_h1):
        # When H1 bar i closes, confirm whether bar i-1 is a Stage-1 extreme
        if i >= 2:
            # Local high: h[i-1] > h[i-2] AND h[i-1] > h[i]
            if h1_hi[i-1] > h1_hi[i-2] and h1_hi[i-1] > h1_hi[i]:
                _try_add(confirmed, (i-1, 'H', h1_hi[i-1]),
                         lh, ll, glh, glh2)
                # Re-read state after add
            # Local low: l[i-1] < l[i-2] AND l[i-1] < l[i]
            if h1_lo[i-1] < h1_lo[i-2] and h1_lo[i-1] < h1_lo[i]:
                _try_add(confirmed, (i-1, 'L', h1_lo[i-1]),
                         lh, ll, glh, glh2)

            # Recompute Stage 3 state from scratch (simplest correct approach for incremental)
            lh, ll, glh, glh2, confirmed = _rerun_stage3(confirmed)

        # Current act_h/act_l = most recent confirmed H and L
        cur_act_h = cur_act_l = float('nan')
        for _, t, v in reversed(confirmed):
            if t == 'H' and math.isnan(cur_act_h): cur_act_h = v
            if t == 'L' and math.isnan(cur_act_l): cur_act_l = v
            if not math.isnan(cur_act_h) and not math.isnan(cur_act_l): break
        act_h_h1[i] = cur_act_h
        act_l_h1[i] = cur_act_l

    # Propagate H1 values to M5 bars
    act_h_m5 = np.full(n_oos, float('nan'))
    act_l_m5 = np.full(n_oos, float('nan'))
    for h1_i, end_m5 in enumerate(h1_end_m5):
        next_end = h1_end_m5[h1_i + 1] if h1_i + 1 < n_h1 else n_oos
        act_h_m5[end_m5:next_end] = act_h_h1[h1_i]
        act_l_m5[end_m5:next_end] = act_l_h1[h1_i]

    # Forward-fill within each H1 period (in case of gaps at period start)
    for i in range(1, n_oos):
        if math.isnan(act_h_m5[i]): act_h_m5[i] = act_h_m5[i-1]
        if math.isnan(act_l_m5[i]): act_l_m5[i] = act_l_m5[i-1]

    return act_h_m5, act_l_m5


def _try_add(confirmed, item, lh, ll, glh, glh2):
    """Stage 2: same-type run → keep most extreme. (Stage 3 handled by _rerun_stage3)."""
    idx, t, v = item
    if confirmed and confirmed[-1][1] == t:
        # Same type — update in place if more extreme
        if t == 'H' and v > confirmed[-1][2]:
            confirmed[-1] = [idx, t, v]
        elif t == 'L' and v < confirmed[-1][2]:
            confirmed[-1] = [idx, t, v]
    else:
        confirmed.append([idx, t, v])


def _rerun_stage3(raw_confirmed):
    """Apply Stage 3 exceeding-extremes gate to the Stage-2 output. Returns new state."""
    sig = []
    lh = ll = float('nan')
    glh = glh2 = False
    for (idx, t, v) in raw_confirmed:
        if t == 'H':
            if math.isnan(lh) or v > lh or glh:
                sig.append([idx, t, v])
                lh = v; glh = False; glh2 = True
        else:
            if math.isnan(ll) or v < ll or glh2:
                sig.append([idx, t, v])
                ll = v; glh2 = False; glh = True
    return lh, ll, glh, glh2, sig


print("Building H1 TopsBots S/R levels...", flush=True)
act_h_m5, act_l_m5 = build_h1_topsbots_m5_array()

# How many M5 bars have valid S/R
valid_sr = np.sum(~np.isnan(act_h_m5) & ~np.isnan(act_l_m5))
zw_vals = (act_h_m5 - act_l_m5) / PIP
print(f"Valid M5 bars with H1 S/R: {valid_sr:,} / {n_oos:,} "
      f"({valid_sr/n_oos*100:.1f}%)")
vz = zw_vals[~np.isnan(zw_vals)]
print(f"H1 zone widths (pips): mean={vz.mean():.1f} median={np.median(vz):.1f} "
      f"P10={np.percentile(vz,10):.1f} P90={np.percentile(vz,90):.1f}")


# ── Zone Recovery Simulation ──────────────────────────────────────────────────

def net_basket(legs, price):
    gross = sum(l['vol'] * l['dir'] * (price - l['price']) / PIP for l in legs)
    cost  = sum(l['vol'] for l in legs) * SPREAD
    return gross - cost


def simulate(zw_cap=None, tgt_frac=0.5, entry_dir='directional'):
    """
    Parameters
    ----------
    zw_cap      : max zone width in pips (None = uncapped)
    tgt_frac    : TGT_BEYOND = tgt_frac × ZW (how far beyond the zone)
    entry_dir   : 'directional' (long at support, short at resistance)
                  'random' (random direction, for comparison)
    """
    rng    = np.random.RandomState(42)
    cycles = []
    i      = 0
    skip_until = 0  # don't enter a new cycle until this bar

    def bvol(legs, target, tgt_pips):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / tgt_pips * PF))

    while i < n_oos:
        if i < skip_until:
            i += 1; continue

        # Need valid H1 S/R at this bar
        upper_zone = act_h_m5[i]
        lower_zone = act_l_m5[i]
        if math.isnan(upper_zone) or math.isnan(lower_zone):
            i += 1; continue
        if upper_zone <= lower_zone:
            i += 1; continue

        zw_raw = (upper_zone - lower_zone) / PIP
        if zw_cap and zw_raw > zw_cap:
            i += 1; continue
        zw_pips = min(zw_raw, zw_cap) if zw_cap else zw_raw
        tgt_pips  = zw_pips * tgt_frac

        # Entry: price closes at or beyond a zone boundary
        entry = close_a[i]
        if entry_dir == 'directional':
            if entry <= lower_zone:
                direction = 1   # LONG at support
            elif entry >= upper_zone:
                direction = -1  # SHORT at resistance
            else:
                i += 1; continue
        else:
            if not (entry <= lower_zone or entry >= upper_zone):
                i += 1; continue
            direction = int(rng.choice([-1, 1]))

        # Set cycle targets
        tgt_b = tgt_pips * PIP
        if direction == 1:
            upper_target = upper_zone + tgt_b
            lower_target = lower_zone - tgt_b
        else:
            upper_target = upper_zone + tgt_b
            lower_target = lower_zone - tgt_b

        legs         = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        entry_bar    = i
        last_crossed = last_crossed_bar = None
        closed       = False
        exit_reason  = 'eod'
        exit_price   = entry
        exit_bar     = i
        max_legs_seen = 1

        i += 1
        while i < n_oos and not closed:
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bullish = cl >= open_a[i]

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
                        vol = bvol(legs, upper_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': 1, 'price': upper_zone, 'vol': vol})
                            max_legs_seen = max(max_legs_seen, len(legs))

                if not is_high and lo <= lower_zone:
                    if not (last_crossed == 'lower' and last_crossed_bar == i):
                        last_crossed, last_crossed_bar = 'lower', i
                        vol = bvol(legs, lower_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                exit_price, exit_reason, exit_bar = cl, 'max_legs', i
                                closed = True; break
                            legs.append({'dir': -1, 'price': lower_zone, 'vol': vol})
                            max_legs_seen = max(max_legs_seen, len(legs))

            if not closed: i += 1

        net = net_basket(legs, exit_price)
        cycles.append({
            'net_pips':    net,
            'exit_reason': exit_reason,
            'n_legs':      len(legs),
            'zw_pips':     zw_pips,
            'tgt_pips':    tgt_pips,
            'duration':    exit_bar - entry_bar,
        })
        if not closed: break

    return cycles


# ── Baseline: fixed ZW=56 random direction ────────────────────────────────────
def simulate_fixed_baseline():
    """Reproduce current production: ZW=56, TGT=28, random, PF=1.25."""
    ZW = 56; TGT = 28
    rng = np.random.RandomState(42)
    cycles = []; i = 0

    def bvol(legs, target):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / TGT * PF))

    while i < n_oos:
        entry = close_a[i]; direction = int(rng.choice([-1, 1]))
        if direction == 1:
            uz=entry; lz=entry-ZW*PIP; ut=entry+TGT*PIP; lt=lz-TGT*PIP
        else:
            lz=entry; uz=entry+ZW*PIP; lt=entry-TGT*PIP; ut=uz+TGT*PIP
        legs=[{'dir': direction, 'price': entry, 'vol': 1.0}]
        entry_bar=i; lc=lcb=None; closed=False; er='eod'; ep=entry; eb=i
        i += 1
        while i < n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for ex,ih in seq:
                if closed: break
                if ih and hi>=ut: ep,er,eb=ut,'target',i; closed=True; break
                if not ih and lo<=lt: ep,er,eb=lt,'target',i; closed=True; break
                if ih and hi>=uz:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol(legs,ut)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er,eb=cl,'max_legs',i; closed=True; break
                            legs.append({'dir':1,'price':uz,'vol':vol})
                if not ih and lo<=lz:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol(legs,lt)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er,eb=cl,'max_legs',i; closed=True; break
                            legs.append({'dir':-1,'price':lz,'vol':vol})
            if not closed: i += 1
        net=net_basket(legs,ep); cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs)})
        if not closed: break
    return cycles


# ── Run sweep ─────────────────────────────────────────────────────────────────
print()
print("="*110)
print(f"  H1 TopsBots S/R Zone Recovery Sweep — {PAIR}  PF={PF}")
print(f"  Entry: directional (LONG at support, SHORT at resistance)")
print(f"  Zone: [act_l, act_h] from H1 confirmed swings (causal, 1-bar lag)")
print(f"  TGT_BEYOND = tgt_frac × ZW pips beyond zone boundary")
print("="*110)
print(f"  {'Config':<38} | {'TotalPips':>10} {'Total$@1ku':>11} | "
      f"{'Cycles':>6} {'%Tgt':>6} {'%ML':>4} | {'AvgLegs':>7} {'AvgZW':>7} | {'vs_base':>8}")
print("─"*110)

# Baseline (fixed ZW=56, random)
base = simulate_fixed_baseline()
df_b = pd.DataFrame(base)
bt = df_b['net_pips'].sum(); bu = bt * PIP_USD * UNITS
print(f"  {'BASELINE fixed ZW=56 random':<38} | {bt:>+10,.0f} {bu:>+11,.0f} | "
      f"{len(df_b):>6} {(df_b['exit_reason']=='target').sum()/len(df_b)*100:>6.1f} "
      f"{(df_b['exit_reason']=='max_legs').sum():>4} | "
      f"{df_b['n_legs'].mean():>7.2f} {'56.0':>7} | {'ref':>8}")

# H1 directional sweeps
for zw_cap in [None, 80, 60]:
    for tgt_frac in [0.25, 0.50, 0.75, 1.0]:
        cyc = simulate(zw_cap=zw_cap, tgt_frac=tgt_frac, entry_dir='directional')
        if not cyc: continue
        df_c = pd.DataFrame(cyc)
        tot = df_c['net_pips'].sum(); usd = tot * PIP_USD * UNITS
        n_cyc = len(df_c)
        pct_tgt = (df_c['exit_reason']=='target').sum()/n_cyc*100
        n_ml    = (df_c['exit_reason']=='max_legs').sum()
        avg_legs = df_c['n_legs'].mean()
        avg_zw   = df_c['zw_pips'].mean()
        vs_base  = (tot - bt) / abs(bt) * 100
        cap_str  = f'cap={zw_cap}p' if zw_cap else 'uncapped'
        label    = f'H1-SR dir {cap_str} tgt={tgt_frac:.2f}×ZW'
        print(f"  {label:<38} | {tot:>+10,.0f} {usd:>+11,.0f} | "
              f"{n_cyc:>6} {pct_tgt:>6.1f} {n_ml:>4} | "
              f"{avg_legs:>7.2f} {avg_zw:>7.1f} | {vs_base:>+7.1f}%")
    print()

# Also test random direction (vs fixed baseline)
print("  --- Random direction (same as baseline logic) ---")
for tgt_frac in [0.25, 0.50, 0.75]:
    cyc = simulate(zw_cap=80, tgt_frac=tgt_frac, entry_dir='random')
    if not cyc: continue
    df_c = pd.DataFrame(cyc)
    tot = df_c['net_pips'].sum(); usd = tot * PIP_USD * UNITS
    n_cyc = len(df_c)
    pct_tgt = (df_c['exit_reason']=='target').sum()/n_cyc*100
    n_ml    = (df_c['exit_reason']=='max_legs').sum()
    avg_zw  = df_c['zw_pips'].mean()
    vs_base = (tot - bt) / abs(bt) * 100
    label   = f'H1-SR random cap=80p tgt={tgt_frac:.2f}×ZW'
    print(f"  {label:<38} | {tot:>+10,.0f} {usd:>+11,.0f} | "
          f"{n_cyc:>6} {pct_tgt:>6.1f} {n_ml:>4} | "
          f"{df_c['n_legs'].mean():>7.2f} {avg_zw:>7.1f} | {vs_base:>+7.1f}%")

print()
print("="*110)
