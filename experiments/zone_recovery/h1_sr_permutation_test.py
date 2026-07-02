"""
Permutation test for H1 TopsBots directional signal.

Null hypothesis: direction at H1 S/R boundary has no predictive value.

Method:
  For each of N_PERM shuffles, assign RANDOM direction at every entry opportunity
  (keeping entry bars identical to the real run — only direction is randomized).
  p-value = fraction of shuffles with total_usd >= actual result.

Real result (H1-25, GBP_JPY OOS): +$72,360
p < 0.05 required for deployment.
"""

import os, math
import numpy as np
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR      = 'GBP_JPY'
PIP       = 0.01
PIP_USD   = 0.000091
UNITS     = 1_000
MAX_LEGS  = 10
PF        = 1.25
SPREAD    = 1.4
N_PERM    = 2_000   # shuffle count
TGT_FRAC  = 0.25
M5_PER_H1 = 12

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(f'{DATA_DIR}/{PAIR}_M5.parquet').sort_index()
df.columns = [c.lower() for c in df.columns]
n = len(df)
df_oos  = df.iloc[int(n * 0.70):].reset_index(drop=True)
open_a  = df_oos['open'].values.astype(np.float64)
high_a  = df_oos['high'].values.astype(np.float64)
low_a   = df_oos['low'].values.astype(np.float64)
close_a = df_oos['close'].values.astype(np.float64)
n_oos   = len(close_a)

# ── Build H1 TopsBots S/R (same as h1_sr_zone_sweep.py) ─────────────────────
def _try_add(confirmed, item):
    idx, t, v = item
    if confirmed and confirmed[-1][1] == t:
        if t == 'H' and v > confirmed[-1][2]: confirmed[-1] = [idx, t, v]
        elif t == 'L' and v < confirmed[-1][2]: confirmed[-1] = [idx, t, v]
    else:
        confirmed.append([idx, t, v])

def _rerun_stage3(raw):
    sig=[]; lh=ll=float('nan'); glh=glh2=False
    for idx,t,v in raw:
        if t=='H':
            if math.isnan(lh) or v>lh or glh: sig.append([idx,t,v]); lh=v; glh=False; glh2=True
        else:
            if math.isnan(ll) or v<ll or glh2: sig.append([idx,t,v]); ll=v; glh2=False; glh=True
    return lh,ll,glh,glh2,sig

h1_hi,h1_lo,h1_end = [],[],[]
for start in range(0, n_oos, M5_PER_H1):
    end = min(start + M5_PER_H1, n_oos)
    h1_hi.append(float(np.max(high_a[start:end])))
    h1_lo.append(float(np.min(low_a[start:end])))
    h1_end.append(end - 1)
h1_hi=np.array(h1_hi); h1_lo=np.array(h1_lo); n_h1=len(h1_hi)

confirmed=[]; cur_h=cur_l=float('nan')
act_h_h1=np.full(n_h1,float('nan')); act_l_h1=np.full(n_h1,float('nan'))
for i in range(1,n_h1):
    if i>=2:
        if h1_hi[i-1]>h1_hi[i-2] and h1_hi[i-1]>h1_hi[i]: _try_add(confirmed,(i-1,'H',h1_hi[i-1]))
        if h1_lo[i-1]<h1_lo[i-2] and h1_lo[i-1]<h1_lo[i]: _try_add(confirmed,(i-1,'L',h1_lo[i-1]))
        _,_,_,_,confirmed=_rerun_stage3(confirmed)
    cur_h=cur_l=float('nan')
    for _,t,v in reversed(confirmed):
        if t=='H' and math.isnan(cur_h): cur_h=v
        if t=='L' and math.isnan(cur_l): cur_l=v
        if not math.isnan(cur_h) and not math.isnan(cur_l): break
    act_h_h1[i]=cur_h; act_l_h1[i]=cur_l

act_h_m5=np.full(n_oos,float('nan')); act_l_m5=np.full(n_oos,float('nan'))
for ti,em in enumerate(h1_end):
    nxt=h1_end[ti+1] if ti+1<n_h1 else n_oos
    act_h_m5[em:nxt]=act_h_h1[ti]; act_l_m5[em:nxt]=act_l_h1[ti]
for i in range(1,n_oos):
    if math.isnan(act_h_m5[i]): act_h_m5[i]=act_h_m5[i-1]
    if math.isnan(act_l_m5[i]): act_l_m5[i]=act_l_m5[i-1]

# ── Pre-collect all entry bars and their valid S/R ────────────────────────────
# An "entry opportunity" is any bar where price touches H1 support or resistance.
# We record (bar_idx, 'support'|'resistance', act_h, act_l) for each opportunity.
print("Collecting entry opportunities...", flush=True)
entry_ops = []  # (i, signal_dir, uh, ul)
i = 0
while i < n_oos:
    uh = act_h_m5[i]; ul = act_l_m5[i]
    if not math.isnan(uh) and not math.isnan(ul) and uh > ul:
        entry = close_a[i]
        if entry <= ul:
            entry_ops.append((i, 1, uh, ul))   # directional: LONG
            i += 1; continue
        elif entry >= uh:
            entry_ops.append((i, -1, uh, ul))  # directional: SHORT
            i += 1; continue
    i += 1
print(f"  {len(entry_ops):,} entry opportunities found", flush=True)


# ── Core simulate function (takes direction array) ────────────────────────────
def simulate_with_directions(directions):
    """
    directions: array of +1/-1, one per entry_ops entry.
    Sequential: after entering cycle at entry_ops[k].bar, skip forward
    until cycle closes, then pick up from next available entry_ops entry.
    Returns total_pips.
    """
    def net_basket(legs, price):
        gross = sum(l['vol']*l['dir']*(price-l['price'])/PIP for l in legs)
        return gross - sum(l['vol'] for l in legs)*SPREAD

    def bvol(legs, target, tgt_pips):
        net = net_basket(legs, target)
        if net >= 0: return 0.0
        return max(1.0, math.ceil(-net/tgt_pips*PF))

    total_pips = 0.0
    op_idx = 0
    n_ops  = len(entry_ops)
    cur_bar = 0  # earliest bar we can start a new cycle

    while op_idx < n_ops:
        bar_i, _, uh, ul = entry_ops[op_idx]
        if bar_i < cur_bar:
            op_idx += 1; continue

        direction = directions[op_idx]
        zw_pips   = (uh - ul) / PIP
        tgt_pips  = zw_pips * TGT_FRAC
        tgt_b     = tgt_pips * PIP
        entry     = close_a[bar_i]
        upper_target = uh + tgt_b; lower_target = ul - tgt_b

        legs = [{'dir': direction, 'price': entry, 'vol': 1.0}]
        lc = lcb = None; closed = False; ep = entry
        i = bar_i + 1

        while i < n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for extreme,is_high in seq:
                if closed: break
                if is_high and hi>=upper_target: ep='tgt'; closed=True; break
                if not is_high and lo<=lower_target: ep='tgt'; closed=True; break
                if is_high and hi>=uh:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol(legs,upper_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep=cl; closed=True; break
                            legs.append({'dir':1,'price':uh,'vol':vol})
                if not is_high and lo<=ul:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol(legs,lower_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep=cl; closed=True; break
                            legs.append({'dir':-1,'price':ul,'vol':vol})
            if not closed: i += 1

        exit_price = upper_target if ep=='tgt' and direction==1 else \
                     lower_target if ep=='tgt' and direction==-1 else \
                     ep if ep!='tgt' else close_a[min(i, n_oos-1)]

        net = sum(l['vol']*l['dir']*(exit_price-l['price'])/PIP for l in legs) \
              - sum(l['vol'] for l in legs)*SPREAD
        total_pips += net
        cur_bar = i + 1
        op_idx += 1

    return total_pips


# ── Real result ───────────────────────────────────────────────────────────────
real_dirs   = np.array([d for _, d, _, _ in entry_ops])
print("Computing real result...", flush=True)
real_pips   = simulate_with_directions(real_dirs)
real_usd    = real_pips * PIP_USD * UNITS
print(f"  Real result: {real_pips:+,.0f} pips  /  ${real_usd:+,.0f}", flush=True)

# ── Permutation ───────────────────────────────────────────────────────────────
print(f"\nRunning {N_PERM:,} permutations...", flush=True)
rng      = np.random.RandomState(99)
null_usd = []

for p in range(N_PERM):
    shuf_dirs = rng.choice([-1, 1], size=len(entry_ops))
    pips      = simulate_with_directions(shuf_dirs)
    null_usd.append(pips * PIP_USD * UNITS)
    if (p + 1) % 200 == 0:
        print(f"  {p+1}/{N_PERM}  null mean=${np.mean(null_usd):+,.0f}", flush=True)

null_usd   = np.array(null_usd)
pval       = (null_usd >= real_usd).sum() / N_PERM
null_mean  = null_usd.mean()
null_p5    = np.percentile(null_usd, 5)
null_p95   = np.percentile(null_usd, 95)

print()
print("="*70)
print(f"  PERMUTATION TEST — H1 TopsBots Directional Signal  ({PAIR} OOS)")
print("="*70)
print(f"  Real result:      ${real_usd:>+12,.0f}")
print(f"  Null mean:        ${null_mean:>+12,.0f}")
print(f"  Null P5/P95:      ${null_p5:>+,.0f}  /  ${null_p95:>+,.0f}")
print(f"  p-value:          {pval:.4f}  ({'PASS ✓' if pval < 0.05 else 'FAIL ✗'}  threshold=0.05)")
print(f"  Shuffles above real: {int(null_usd >= real_usd).sum() if hasattr(null_usd,'sum') else (null_usd >= real_usd).sum()}/{N_PERM}")
print("="*70)
if pval < 0.001:
    print("  🟢 p < 0.001 — extremely significant. Directional signal is real.")
elif pval < 0.01:
    print("  🟢 p < 0.01 — highly significant.")
elif pval < 0.05:
    print("  🟢 p < 0.05 — significant. Passes gate.")
else:
    print("  🔴 p ≥ 0.05 — not significant. Signal may be noise.")
print("="*70)
