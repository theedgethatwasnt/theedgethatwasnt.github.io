"""
Multi-TF and ASI swing S/R sweep for zone recovery.

Tests:
  1. Single TF directional: H1 (12×M5), H2 (24), H4 (48)
  2. Two-TF consensus: H1+H4 — enter only when BOTH TFs agree (strict: price at support on both)
  3. ASI swing S/R — TopsBots on running ASI values; S/R level = close price at ASI swing bar

All directional (LONG at support, SHORT at resistance), tgt=0.25×ZW, uncapped.
Compared against: baseline fixed ZW=56 random (+$28,400) and best H1 single-TF (+$72,360).

Lookahead invariants:
  - OHLC TopsBots: bar i-1 confirmed when bar i closes (same as h1_sr_zone_sweep.py)
  - ASI: computed bar-by-bar; SI[i] uses current OHLC + prev close only. Causal.
  - ASI TopsBots: same 1-bar lag — asi[i-1] confirmed as Stage-1 swing when asi[i] is computed.
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
PF       = 1.25
SPREAD   = 1.4


# ── Load M5 OOS ───────────────────────────────────────────────────────────────
df = pd.read_parquet(f'{DATA_DIR}/{PAIR}_M5.parquet').sort_index()
df.columns = [c.lower() for c in df.columns]
n = len(df)
df_oos  = df.iloc[int(n * 0.70):].reset_index(drop=True)
open_a  = df_oos['open'].values.astype(np.float64)
high_a  = df_oos['high'].values.astype(np.float64)
low_a   = df_oos['low'].values.astype(np.float64)
close_a = df_oos['close'].values.astype(np.float64)
n_oos   = len(close_a)


# ── TopsBots Stage 2+3 helpers ────────────────────────────────────────────────
def _try_add(confirmed, item):
    idx, t, v = item
    if confirmed and confirmed[-1][1] == t:
        if t == 'H' and v > confirmed[-1][2]: confirmed[-1] = [idx, t, v]
        elif t == 'L' and v < confirmed[-1][2]: confirmed[-1] = [idx, t, v]
    else:
        confirmed.append([idx, t, v])

def _rerun_stage3(raw):
    sig = []; lh = ll = float('nan'); glh = glh2 = False
    for (idx, t, v) in raw:
        if t == 'H':
            if math.isnan(lh) or v > lh or glh:
                sig.append([idx, t, v]); lh = v; glh = False; glh2 = True
        else:
            if math.isnan(ll) or v < ll or glh2:
                sig.append([idx, t, v]); ll = v; glh2 = False; glh = True
    return lh, ll, glh, glh2, sig

def _get_act(confirmed):
    cur_h = cur_l = float('nan')
    for _, t, v in reversed(confirmed):
        if t == 'H' and math.isnan(cur_h): cur_h = v
        if t == 'L' and math.isnan(cur_l): cur_l = v
        if not math.isnan(cur_h) and not math.isnan(cur_l): break
    return cur_h, cur_l


# ── Build OHLC-based TopsBots S/R for any TF ─────────────────────────────────
def build_ohlc_sr(m5_per_tf):
    """
    Aggregate M5 → TF bars, run causal TopsBots on TF H/L.
    Returns act_h_m5, act_l_m5 arrays of length n_oos.
    """
    tf_hi, tf_lo, tf_end_m5 = [], [], []
    for start in range(0, n_oos, m5_per_tf):
        end = min(start + m5_per_tf, n_oos)
        tf_hi.append(float(np.max(high_a[start:end])))
        tf_lo.append(float(np.min(low_a[start:end])))
        tf_end_m5.append(end - 1)
    tf_hi = np.array(tf_hi); tf_lo = np.array(tf_lo); n_tf = len(tf_hi)

    confirmed = []
    act_h_tf = np.full(n_tf, float('nan'))
    act_l_tf = np.full(n_tf, float('nan'))
    cur_h = cur_l = float('nan')

    for i in range(1, n_tf):
        if i >= 2:
            if tf_hi[i-1] > tf_hi[i-2] and tf_hi[i-1] > tf_hi[i]:
                _try_add(confirmed, (i-1, 'H', tf_hi[i-1]))
            if tf_lo[i-1] < tf_lo[i-2] and tf_lo[i-1] < tf_lo[i]:
                _try_add(confirmed, (i-1, 'L', tf_lo[i-1]))
            _, _, _, _, confirmed = _rerun_stage3(confirmed)
        cur_h, cur_l = _get_act(confirmed)
        act_h_tf[i] = cur_h; act_l_tf[i] = cur_l

    # Propagate to M5
    act_h_m5 = np.full(n_oos, float('nan'))
    act_l_m5 = np.full(n_oos, float('nan'))
    for ti, end_m5 in enumerate(tf_end_m5):
        nxt = tf_end_m5[ti + 1] if ti + 1 < n_tf else n_oos
        act_h_m5[end_m5:nxt] = act_h_tf[ti]
        act_l_m5[end_m5:nxt] = act_l_tf[ti]
    for i in range(1, n_oos):
        if math.isnan(act_h_m5[i]): act_h_m5[i] = act_h_m5[i-1]
        if math.isnan(act_l_m5[i]): act_l_m5[i] = act_l_m5[i-1]

    valid = np.sum(~np.isnan(act_h_m5) & ~np.isnan(act_l_m5))
    zw = (act_h_m5 - act_l_m5)[~np.isnan(act_h_m5)] / PIP
    return act_h_m5, act_l_m5, valid, zw


# ── Build ASI swing S/R ───────────────────────────────────────────────────────
def build_asi_sr(atr_period=14, atr_mult=3.0):
    """
    Vectorized Wilder ASI, then causal TopsBots on ASI scalar series.
    S/R price = close at the bar where ASI made its confirmed swing.
    """
    EPSILON = 1e-10

    # Vectorized Wilder ATR
    tr_arr = np.maximum(high_a[1:] - low_a[1:],
              np.maximum(np.abs(high_a[1:] - close_a[:-1]),
                         np.abs(low_a[1:]  - close_a[:-1])))
    atr = np.zeros(n_oos); atr[0] = high_a[0] - low_a[0]
    for i in range(1, n_oos):
        if i < atr_period: atr[i] = atr[i-1] + (tr_arr[i-1] - atr[i-1]) / (i + 1)
        else: atr[i] = (atr[i-1] * (atr_period - 1) + tr_arr[i-1]) / atr_period

    # Vectorized SI
    C2,O2,H2,L2 = close_a[1:],open_a[1:],high_a[1:],low_a[1:]
    C1,O1 = close_a[:-1],open_a[:-1]
    N  = (C2-C1) + 0.5*(C2-O2) + 0.25*(C1-O1)
    t1 = np.abs(H2-C1) - 0.5*np.abs(L2-C1) + 0.25*np.abs(C1-O1)
    t2 = np.abs(L2-C1) - 0.5*np.abs(H2-C1) + 0.25*np.abs(C1-O1)
    t3 = (H2-L2) + 0.25*np.abs(C1-O1)
    R  = np.maximum(np.maximum(t1,t2),np.maximum(t3,EPSILON))
    K  = np.maximum(np.abs(H2-C1), np.abs(L2-C1))
    SI = 50.0*(N/R)*(K/np.maximum(atr_mult*atr[1:],EPSILON))
    asi_vals = np.zeros(n_oos); asi_vals[1:] = np.cumsum(SI)

    # Causal TopsBots on ASI (fast: _rerun_stage3 only when list grows)
    confirmed = []
    act_h_m5  = np.full(n_oos, float('nan'))
    act_l_m5  = np.full(n_oos, float('nan'))
    prev_len = 0

    for i in range(2, n_oos):
        if asi_vals[i-1] > asi_vals[i-2] and asi_vals[i-1] > asi_vals[i]:
            _try_add(confirmed, (i-1, 'H', close_a[i-1]))
        if asi_vals[i-1] < asi_vals[i-2] and asi_vals[i-1] < asi_vals[i]:
            _try_add(confirmed, (i-1, 'L', close_a[i-1]))
        if len(confirmed) != prev_len:
            _, _, _, _, confirmed = _rerun_stage3(confirmed)
            prev_len = len(confirmed)
        act_h_m5[i], act_l_m5[i] = _get_act(confirmed)

    valid = np.sum(~np.isnan(act_h_m5) & ~np.isnan(act_l_m5) & (act_h_m5 > act_l_m5))
    zw = (act_h_m5 - act_l_m5)[~np.isnan(act_h_m5) & (act_h_m5 > act_l_m5)] / PIP
    return act_h_m5, act_l_m5, valid, zw


# ── Zone Recovery simulator ───────────────────────────────────────────────────
def net_basket(legs, price):
    gross = sum(l['vol'] * l['dir'] * (price - l['price']) / PIP for l in legs)
    cost  = sum(l['vol'] for l in legs) * SPREAD
    return gross - cost

def simulate(act_h, act_l, tgt_frac=0.25, zw_cap=None):
    """
    Directional zone recovery using pre-built S/R arrays.
    LONG at support (close <= act_l), SHORT at resistance (close >= act_h).
    """
    def bvol(legs, target, tgt_pips):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / tgt_pips * PF))

    cycles = []; i = 0
    while i < n_oos:
        uh = act_h[i]; ul = act_l[i]
        if math.isnan(uh) or math.isnan(ul) or uh <= ul:
            i += 1; continue
        zw_raw  = (uh - ul) / PIP
        if zw_cap and zw_raw > zw_cap: i += 1; continue
        zw_pips  = zw_raw
        tgt_pips = zw_pips * tgt_frac
        tgt_b    = tgt_pips * PIP

        entry = close_a[i]
        if   entry <= ul: direction = 1
        elif entry >= uh: direction = -1
        else:             i += 1; continue

        upper_target = uh + tgt_b; lower_target = ul - tgt_b
        legs = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        lc = lcb = None; closed = False; er = 'eod'; ep = entry; eb = i
        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bull = cl >= open_a[i]
            seq  = [(hi, True), (lo, False)] if bull else [(lo, False), (hi, True)]
            for extreme, is_high in seq:
                if closed: break
                if is_high and hi >= upper_target:
                    ep, er, eb = upper_target, 'target', i; closed = True; break
                if not is_high and lo <= lower_target:
                    ep, er, eb = lower_target, 'target', i; closed = True; break
                if is_high and hi >= uh:
                    if not (lc == 'upper' and lcb == i):
                        lc, lcb = 'upper', i
                        vol = bvol(legs, upper_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                ep, er, eb = cl, 'max_legs', i; closed = True; break
                            legs.append({'dir': 1, 'price': uh, 'vol': vol})
                if not is_high and lo <= ul:
                    if not (lc == 'lower' and lcb == i):
                        lc, lcb = 'lower', i
                        vol = bvol(legs, lower_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                ep, er, eb = cl, 'max_legs', i; closed = True; break
                            legs.append({'dir': -1, 'price': ul, 'vol': vol})
            if not closed: i += 1

        net = net_basket(legs, ep)
        cycles.append({'net_pips': net, 'exit_reason': er,
                       'n_legs': len(legs), 'zw_pips': zw_pips})
        if not closed: break
    return cycles


def simulate_consensus(act_h1, act_l1, act_h2, act_l2, tgt_frac=0.25, use_wider=False):
    """
    Two-TF consensus: LONG when price <= BOTH supports; SHORT when price >= BOTH resistances.
    Zone: use TF1 boundaries (smaller = more precise) OR TF2 (wider, larger target).
    """
    def bvol(legs, target, tgt_pips):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / tgt_pips * PF))

    cycles = []; i = 0
    while i < n_oos:
        uh1 = act_h1[i]; ul1 = act_l1[i]
        uh2 = act_h2[i]; ul2 = act_l2[i]
        if any(math.isnan(x) for x in [uh1, ul1, uh2, ul2]):
            i += 1; continue
        if uh1 <= ul1 or uh2 <= ul2:
            i += 1; continue

        entry = close_a[i]

        # Strict consensus: price at support on BOTH TFs
        if entry <= ul1 and entry <= ul2:
            direction = 1
        elif entry >= uh1 and entry >= uh2:
            direction = -1
        else:
            i += 1; continue

        # Zone boundaries: use wider TF (TF2) for larger targets if requested
        uh = uh2 if use_wider else uh1
        ul = ul2 if use_wider else ul1
        if uh <= ul: i += 1; continue

        zw_pips  = (uh - ul) / PIP
        tgt_pips = zw_pips * tgt_frac
        tgt_b    = tgt_pips * PIP
        upper_target = uh + tgt_b; lower_target = ul - tgt_b

        legs = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        lc = lcb = None; closed = False; er = 'eod'; ep = entry; eb = i
        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bull = cl >= open_a[i]
            seq  = [(hi, True), (lo, False)] if bull else [(lo, False), (hi, True)]
            for extreme, is_high in seq:
                if closed: break
                if is_high and hi >= upper_target:
                    ep, er, eb = upper_target, 'target', i; closed = True; break
                if not is_high and lo <= lower_target:
                    ep, er, eb = lower_target, 'target', i; closed = True; break
                if is_high and hi >= uh:
                    if not (lc == 'upper' and lcb == i):
                        lc, lcb = 'upper', i
                        vol = bvol(legs, upper_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                ep, er, eb = cl, 'max_legs', i; closed = True; break
                            legs.append({'dir': 1, 'price': uh, 'vol': vol})
                if not is_high and lo <= ul:
                    if not (lc == 'lower' and lcb == i):
                        lc, lcb = 'lower', i
                        vol = bvol(legs, lower_target, tgt_pips)
                        if vol > 0:
                            if len(legs) >= MAX_LEGS:
                                ep, er, eb = cl, 'max_legs', i; closed = True; break
                            legs.append({'dir': -1, 'price': ul, 'vol': vol})
            if not closed: i += 1

        net = net_basket(legs, ep)
        cycles.append({'net_pips': net, 'exit_reason': er,
                       'n_legs': len(legs), 'zw_pips': zw_pips})
        if not closed: break
    return cycles


def simulate_fixed_baseline():
    """Fixed ZW=56 random (production baseline)."""
    ZW = 56; TGT = 28
    rng = np.random.RandomState(42); cycles = []; i = 0

    def bvol_b(legs, target):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / TGT * PF))

    while i < n_oos:
        entry = close_a[i]; direction = int(rng.choice([-1, 1]))
        if direction == 1:
            uz=entry; lz=entry-ZW*PIP; ut=entry+TGT*PIP; lt=lz-TGT*PIP
        else:
            lz=entry; uz=entry+ZW*PIP; lt=entry-TGT*PIP; ut=uz+TGT*PIP
        legs=[{'dir':direction,'price':entry,'vol':1.0}]
        lc=lcb=None; closed=False; er='eod'; ep=entry
        i += 1
        while i < n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for ex,ih in seq:
                if closed: break
                if ih and hi>=ut: ep,er=ut,'target'; closed=True; break
                if not ih and lo<=lt: ep,er=lt,'target'; closed=True; break
                if ih and hi>=uz:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol_b(legs,ut)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':1,'price':uz,'vol':vol})
                if not ih and lo<=lz:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol_b(legs,lt)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':-1,'price':lz,'vol':vol})
            if not closed: i += 1
        net=net_basket(legs,ep); cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs)})
        if not closed: break
    return cycles


def summarise(cycles, pip_usd=PIP_USD, units=UNITS):
    df_c = pd.DataFrame(cycles)
    tp   = df_c['net_pips'].sum()
    return {
        'total_pips': tp,
        'total_usd':  tp * pip_usd * units,
        'n_cyc':      len(df_c),
        'pct_tgt':    (df_c['exit_reason']=='target').sum() / len(df_c) * 100,
        'n_ml':       (df_c['exit_reason']=='max_legs').sum(),
        'avg_legs':   df_c['n_legs'].mean(),
        'avg_zw':     df_c['zw_pips'].mean() if 'zw_pips' in df_c else 56.0,
    }


# ── Build all S/R arrays ──────────────────────────────────────────────────────
print(f"Building S/R arrays for {PAIR}...")
h1_h, h1_l, h1_valid, h1_zw = build_ohlc_sr(12)
h2_h, h2_l, h2_valid, h2_zw = build_ohlc_sr(24)
h4_h, h4_l, h4_valid, h4_zw = build_ohlc_sr(48)
asi_h, asi_l, asi_valid, asi_zw = build_asi_sr()

print(f"  H1  valid={h1_valid:,}  zw: median={np.median(h1_zw):.1f}p  P10={np.percentile(h1_zw,10):.1f}  P90={np.percentile(h1_zw,90):.1f}")
print(f"  H2  valid={h2_valid:,}  zw: median={np.median(h2_zw):.1f}p  P10={np.percentile(h2_zw,10):.1f}  P90={np.percentile(h2_zw,90):.1f}")
print(f"  H4  valid={h4_valid:,}  zw: median={np.median(h4_zw):.1f}p  P10={np.percentile(h4_zw,10):.1f}  P90={np.percentile(h4_zw,90):.1f}")
print(f"  ASI valid={asi_valid:,}  zw: median={np.median(asi_zw):.1f}p  P10={np.percentile(asi_zw,10):.1f}  P90={np.percentile(asi_zw,90):.1f}")

# ── Run sims ──────────────────────────────────────────────────────────────────
print("\nRunning simulations...")
base_cyc   = simulate_fixed_baseline()
h1_25_cyc  = simulate(h1_h, h1_l, tgt_frac=0.25)
h1_50_cyc  = simulate(h1_h, h1_l, tgt_frac=0.50)
h2_25_cyc  = simulate(h2_h, h2_l, tgt_frac=0.25)
h2_50_cyc  = simulate(h2_h, h2_l, tgt_frac=0.50)
h4_25_cyc  = simulate(h4_h, h4_l, tgt_frac=0.25)
h4_50_cyc  = simulate(h4_h, h4_l, tgt_frac=0.50)
c_h1h4_narrow = simulate_consensus(h1_h, h1_l, h4_h, h4_l, tgt_frac=0.25, use_wider=False)
c_h1h4_wide   = simulate_consensus(h1_h, h1_l, h4_h, h4_l, tgt_frac=0.25, use_wider=True)
c_h1h2_narrow = simulate_consensus(h1_h, h1_l, h2_h, h2_l, tgt_frac=0.25, use_wider=False)
asi_25_cyc = simulate(asi_h, asi_l, tgt_frac=0.25)
asi_50_cyc = simulate(asi_h, asi_l, tgt_frac=0.50)

# ── Report ────────────────────────────────────────────────────────────────────
base_usd = summarise(base_cyc)['total_usd']
h1_25_usd = summarise(h1_25_cyc)['total_usd']  # reference: +$72,360

configs = [
    ("BASELINE fixed ZW=56 random",    base_cyc,         '—'),
    ("H1 dir tgt=0.25×ZW [ref]",       h1_25_cyc,       'ref'),
    ("H1 dir tgt=0.50×ZW",             h1_50_cyc,       'h1ref'),
    ("H2 dir tgt=0.25×ZW",             h2_25_cyc,       'ref'),
    ("H2 dir tgt=0.50×ZW",             h2_50_cyc,       'h1ref'),
    ("H4 dir tgt=0.25×ZW",             h4_25_cyc,       'ref'),
    ("H4 dir tgt=0.50×ZW",             h4_50_cyc,       'h1ref'),
    ("H1+H4 consensus narrow tgt=0.25",c_h1h4_narrow,   'ref'),
    ("H1+H4 consensus wide  tgt=0.25", c_h1h4_wide,     'ref'),
    ("H1+H2 consensus narrow tgt=0.25",c_h1h2_narrow,   'ref'),
    ("ASI-swing dir tgt=0.25×ZW",      asi_25_cyc,      'ref'),
    ("ASI-swing dir tgt=0.50×ZW",      asi_50_cyc,      'h1ref'),
]

print()
print("="*125)
print(f"  Multi-TF + ASI Swing S/R Sweep — {PAIR}  PF={PF}  tgt=0.25×ZW or 0.50×ZW")
print("="*125)
print(f"  {'Config':<36} | {'TotalPips':>10} {'Total$@1ku':>11} | {'Cycles':>7} {'%Tgt':>6} {'%ML':>4} | {'AvgLegs':>7} {'AvgZW':>7} | {'vs Base':>8} | {'vs H1-25':>9}")
print("─"*125)

for label, cyc, _ in configs:
    if not cyc:
        print(f"  {label:<36} | {'NO CYCLES':>21}")
        continue
    s  = summarise(cyc)
    vs_base = (s['total_usd'] - base_usd) / abs(base_usd) * 100 if base_usd != 0 else 0
    vs_h1   = (s['total_usd'] - h1_25_usd) / abs(h1_25_usd) * 100 if h1_25_usd != 0 else 0
    flag = '🟢' if s['total_usd'] > h1_25_usd * 0.9 else ('🟡' if s['total_usd'] > base_usd else '🔴')
    print(f"  {label:<36} | {s['total_pips']:>+10,.0f} {s['total_usd']:>+11,.0f} | "
          f"{s['n_cyc']:>7,} {s['pct_tgt']:>6.1f} {s['n_ml']:>4} | "
          f"{s['avg_legs']:>7.2f} {s['avg_zw']:>7.1f} | "
          f"{vs_base:>+7.1f}% | {vs_h1:>+8.1f}% {flag}")

print()
print("="*125)
