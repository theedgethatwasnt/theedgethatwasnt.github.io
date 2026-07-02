"""
H1 TopsBots S/R — 12-pair sweep.

Runs the best config from single-pair test:
  directional, uncapped, tgt_frac=0.25 and 0.50

Also runs fixed ZW=56 random baseline per pair for direct comparison.

Reports per-pair then aggregate totals.
"""

import os, math
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')

PAIRS = [
    'AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY',
    'EUR_GBP','EUR_JPY','EUR_USD','GBP_JPY',
    'GBP_USD','NZD_JPY','NZD_USD','USD_JPY',
]

PIP_USD_MAP = {
    'AUD_JPY': 0.000067, 'AUD_USD': 0.000100, 'CAD_JPY': 0.000069,
    'CHF_JPY': 0.000107, 'EUR_GBP': 0.000126, 'EUR_JPY': 0.000064,
    'EUR_USD': 0.000100, 'GBP_JPY': 0.000091, 'GBP_USD': 0.000100,
    'NZD_JPY': 0.000061, 'NZD_USD': 0.000100, 'USD_JPY': 0.000064,
}
PIP_MAP = {p: (0.01 if 'JPY' in p or 'CHF_JPY' in p or 'CAD_JPY' in p or 'AUD_JPY' in p
               else 0.0001) for p in PAIRS}
# All pairs with JPY use pip=0.01; others use 0.0001
PIP_MAP = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}

UNITS    = 1_000
MAX_LEGS = 10
PF       = 1.25
SPREAD   = 1.4      # pips — conservative for all pairs
M5_PER_H1 = 12


def run_pair(pair):
    pip     = PIP_MAP[pair]
    pip_usd = PIP_USD_MAP[pair]

    path = f'{DATA_DIR}/{pair}_M5.parquet'
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path).sort_index()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    n_train = int(n * 0.70)
    df_oos  = df.iloc[n_train:].reset_index(drop=True)
    open_a  = df_oos['open'].values.astype(np.float64)
    high_a  = df_oos['high'].values.astype(np.float64)
    low_a   = df_oos['low'].values.astype(np.float64)
    close_a = df_oos['close'].values.astype(np.float64)
    n_oos   = len(close_a)

    # ── H1 bars ──────────────────────────────────────────────────────────────
    h1_hi, h1_lo, h1_end_m5 = [], [], []
    for start in range(0, n_oos, M5_PER_H1):
        end = min(start + M5_PER_H1, n_oos)
        h1_hi.append(float(np.max(high_a[start:end])))
        h1_lo.append(float(np.min(low_a[start:end])))
        h1_end_m5.append(end - 1)
    h1_hi = np.array(h1_hi); h1_lo = np.array(h1_lo); n_h1 = len(h1_hi)

    # ── Causal TopsBots ───────────────────────────────────────────────────────
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

    confirmed = []; lh = ll = float('nan'); glh = glh2 = False
    act_h_h1  = np.full(n_h1, float('nan'))
    act_l_h1  = np.full(n_h1, float('nan'))
    cur_h = cur_l = float('nan')

    for i in range(1, n_h1):
        if i >= 2:
            if h1_hi[i-1] > h1_hi[i-2] and h1_hi[i-1] > h1_hi[i]:
                _try_add(confirmed, (i-1, 'H', h1_hi[i-1]))
            if h1_lo[i-1] < h1_lo[i-2] and h1_lo[i-1] < h1_lo[i]:
                _try_add(confirmed, (i-1, 'L', h1_lo[i-1]))
            lh, ll, glh, glh2, confirmed = _rerun_stage3(confirmed)
        cur_h = cur_l = float('nan')
        for _, t, v in reversed(confirmed):
            if t == 'H' and math.isnan(cur_h): cur_h = v
            if t == 'L' and math.isnan(cur_l): cur_l = v
            if not math.isnan(cur_h) and not math.isnan(cur_l): break
        act_h_h1[i] = cur_h; act_l_h1[i] = cur_l

    act_h_m5 = np.full(n_oos, float('nan'))
    act_l_m5 = np.full(n_oos, float('nan'))
    for h1_i, end_m5 in enumerate(h1_end_m5):
        nxt = h1_end_m5[h1_i + 1] if h1_i + 1 < n_h1 else n_oos
        act_h_m5[end_m5:nxt] = act_h_h1[h1_i]
        act_l_m5[end_m5:nxt] = act_l_h1[h1_i]
    for i in range(1, n_oos):
        if math.isnan(act_h_m5[i]): act_h_m5[i] = act_h_m5[i-1]
        if math.isnan(act_l_m5[i]): act_l_m5[i] = act_l_m5[i-1]

    # ── Simulation helpers ────────────────────────────────────────────────────
    def net_basket(legs, price):
        gross = sum(l['vol'] * l['dir'] * (price - l['price']) / pip for l in legs)
        cost  = sum(l['vol'] for l in legs) * SPREAD
        return gross - cost

    def bvol(legs, target, tgt_pips):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net / tgt_pips * PF))

    # ── H1-SR directional sim ─────────────────────────────────────────────────
    def simulate_h1sr(tgt_frac):
        rng = np.random.RandomState(42); cycles = []; i = 0
        while i < n_oos:
            uh = act_h_m5[i]; ul = act_l_m5[i]
            if math.isnan(uh) or math.isnan(ul) or uh <= ul:
                i += 1; continue
            zw_pips = (uh - ul) / pip
            tgt_pips = zw_pips * tgt_frac; tgt_b = tgt_pips * pip
            entry = close_a[i]
            if entry <= ul:   direction = 1
            elif entry >= uh: direction = -1
            else:             i += 1; continue

            upper_target = uh + tgt_b; lower_target = ul - tgt_b
            legs = [{'dir': direction, 'price': entry, 'vol': 1.0}]
            entry_bar = i; lc = lcb = None; closed = False
            er = 'eod'; ep = entry; eb = i

            i += 1
            while i < n_oos and not closed:
                hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
                bull = cl >= open_a[i]
                seq = [(hi, True), (lo, False)] if bull else [(lo, False), (hi, True)]
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

    # ── Fixed ZW=56 baseline sim ──────────────────────────────────────────────
    def simulate_baseline():
        ZW = 56; TGT = 28
        rng = np.random.RandomState(42); cycles = []; i = 0

        def bvol_b(legs, target):
            net = net_basket(legs, target)
            if net >= 0: return 0.0
            return max(1.0, math.ceil(-net / TGT * PF))

        while i < n_oos:
            entry = close_a[i]; direction = int(rng.choice([-1, 1]))
            if direction == 1:
                uz=entry; lz=entry-ZW*pip; ut=entry+TGT*pip; lt=lz-TGT*pip
            else:
                lz=entry; uz=entry+ZW*pip; lt=entry-TGT*pip; ut=uz+TGT*pip
            legs=[{'dir':direction,'price':entry,'vol':1.0}]
            lc=lcb=None; closed=False; er='eod'; ep=entry; eb=i
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
                            lc,lcb='upper',i; vol=bvol_b(legs,ut)
                            if vol>0:
                                if len(legs)>=MAX_LEGS: ep,er,eb=cl,'max_legs',i; closed=True; break
                                legs.append({'dir':1,'price':uz,'vol':vol})
                    if not ih and lo<=lz:
                        if not(lc=='lower' and lcb==i):
                            lc,lcb='lower',i; vol=bvol_b(legs,lt)
                            if vol>0:
                                if len(legs)>=MAX_LEGS: ep,er,eb=cl,'max_legs',i; closed=True; break
                                legs.append({'dir':-1,'price':lz,'vol':vol})
                if not closed: i += 1
            net=net_basket(legs,ep); cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs)})
            if not closed: break
        return cycles

    base   = simulate_baseline()
    h1_25  = simulate_h1sr(tgt_frac=0.25)
    h1_50  = simulate_h1sr(tgt_frac=0.50)

    def summarise(cycles):
        df_c = pd.DataFrame(cycles)
        tp   = df_c['net_pips'].sum()
        return {
            'total_pips': tp,
            'total_usd':  tp * pip_usd * UNITS,
            'n_cyc':      len(df_c),
            'pct_tgt':    (df_c['exit_reason']=='target').sum() / len(df_c) * 100,
            'n_ml':       (df_c['exit_reason']=='max_legs').sum(),
            'avg_legs':   df_c['n_legs'].mean(),
            'avg_zw':     df_c['zw_pips'].mean() if 'zw_pips' in df_c else 56.0,
        }

    return {
        'pair':   pair,
        'base':   summarise(base),
        'h1_25':  summarise(h1_25),
        'h1_50':  summarise(h1_50),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print("="*130)
print("  H1 TopsBots S/R Zone Recovery — 12-Pair Sweep")
print("  Config A: directional uncapped tgt=0.25×ZW  |  Config B: directional uncapped tgt=0.50×ZW")
print("  Baseline: fixed ZW=56 random PF=1.25")
print("="*130)
hdr = f"  {'Pair':<10} | {'Base $@1ku':>11} | {'H1-25 $@1ku':>12} {'vs%':>6} | {'H1-50 $@1ku':>12} {'vs%':>6} | {'Cyc-A':>7} {'AvgLeg-A':>9} | {'Cyc-B':>7}"
print(hdr)
print("─"*130)

results = []
agg = {'base_usd':0,'h25_usd':0,'h50_usd':0,'base_tp':0,'h25_tp':0,'h50_tp':0}

for pair in PAIRS:
    print(f"  running {pair}...", end='\r', flush=True)
    r = run_pair(pair)
    if r is None:
        print(f"  {pair:<10} | {'NO DATA':>11}")
        continue
    results.append(r)

    b  = r['base'];  h25 = r['h1_25']; h50 = r['h1_50']
    vs25 = (h25['total_usd'] - b['total_usd']) / max(abs(b['total_usd']), 1) * 100
    vs50 = (h50['total_usd'] - b['total_usd']) / max(abs(b['total_usd']), 1) * 100
    flag25 = '🟢' if vs25 > 20 else ('🟡' if vs25 > -20 else '🔴')
    flag50 = '🟢' if vs50 > 20 else ('🟡' if vs50 > -20 else '🔴')
    print(f"  {pair:<10} | {b['total_usd']:>+11,.0f} | "
          f"{h25['total_usd']:>+12,.0f} {vs25:>+5.0f}% {flag25} | "
          f"{h50['total_usd']:>+12,.0f} {vs50:>+5.0f}% {flag50} | "
          f"{h25['n_cyc']:>7,} {h25['avg_legs']:>9.2f} | "
          f"{h50['n_cyc']:>7,}")
    agg['base_usd'] += b['total_usd'];  agg['base_tp'] += b['total_pips']
    agg['h25_usd']  += h25['total_usd']; agg['h25_tp'] += h25['total_pips']
    agg['h50_usd']  += h50['total_usd']; agg['h50_tp'] += h50['total_pips']

print("─"*130)
vs25_agg = (agg['h25_usd'] - agg['base_usd']) / max(abs(agg['base_usd']),1) * 100
vs50_agg = (agg['h50_usd'] - agg['base_usd']) / max(abs(agg['base_usd']),1) * 100
print(f"  {'AGGREGATE':<10} | {agg['base_usd']:>+11,.0f} | "
      f"{agg['h25_usd']:>+12,.0f} {vs25_agg:>+5.0f}%   | "
      f"{agg['h50_usd']:>+12,.0f} {vs50_agg:>+5.0f}%")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
n_pos_25 = sum(1 for r in results if r['h1_25']['total_usd'] > r['base']['total_usd'])
n_pos_50 = sum(1 for r in results if r['h1_50']['total_usd'] > r['base']['total_usd'])
print(f"  H1-25 beats baseline: {n_pos_25}/{len(results)} pairs")
print(f"  H1-50 beats baseline: {n_pos_50}/{len(results)} pairs")
print()
print("="*130)
