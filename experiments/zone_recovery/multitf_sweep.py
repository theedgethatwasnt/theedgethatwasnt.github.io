"""
Multi-TF S/R zone recovery sweep (OHLC TopsBots only, fast).

Tests H1, H2, H4 single-TF directional entry and H1+H4 consensus.
ASI swing S/R is a separate script (asi_sr_sweep.py).

Uses numba JIT for the inner simulation loop.
"""

import os, math
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
PAIR     = 'GBP_JPY'
PIP      = 0.01
PIP_USD  = 0.000091
UNITS    = 1_000
MAX_LEGS = 10
PF       = 1.25
SPREAD   = 1.4


# ── Load OOS ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(f'{DATA_DIR}/{PAIR}_M5.parquet').sort_index()
df.columns = [c.lower() for c in df.columns]
n = len(df)
df_oos  = df.iloc[int(n * 0.70):].reset_index(drop=True)
open_a  = df_oos['open'].values.astype(np.float64)
high_a  = df_oos['high'].values.astype(np.float64)
low_a   = df_oos['low'].values.astype(np.float64)
close_a = df_oos['close'].values.astype(np.float64)
n_oos   = len(close_a)


# ── Build OHLC TopsBots S/R for any TF (pure Python, fast for <10K TF bars) ──
def build_ohlc_sr(m5_per_tf):
    tf_hi, tf_lo = [], []
    for start in range(0, n_oos, m5_per_tf):
        end = min(start + m5_per_tf, n_oos)
        tf_hi.append(float(np.max(high_a[start:end])))
        tf_lo.append(float(np.min(low_a[start:end])))
    tf_hi = np.array(tf_hi); tf_lo = np.array(tf_lo); n_tf = len(tf_hi)

    def _try_add(conf, idx, t, v):
        if conf and conf[-1][1] == t:
            if t == 'H' and v > conf[-1][2]: conf[-1] = [idx, t, v]
            elif t == 'L' and v < conf[-1][2]: conf[-1] = [idx, t, v]
        else: conf.append([idx, t, v])

    def _stage3(raw):
        sig = []; lh = ll = float('nan'); glh = glh2 = False
        for idx, t, v in raw:
            if t == 'H':
                if math.isnan(lh) or v > lh or glh: sig.append([idx,t,v]); lh=v; glh=False; glh2=True
            else:
                if math.isnan(ll) or v < ll or glh2: sig.append([idx,t,v]); ll=v; glh2=False; glh=True
        return sig

    conf = []
    act_h_tf = np.full(n_tf, np.nan); act_l_tf = np.full(n_tf, np.nan)
    for i in range(1, n_tf):
        if i >= 2:
            if tf_hi[i-1] > tf_hi[i-2] and tf_hi[i-1] > tf_hi[i]: _try_add(conf, i-1, 'H', tf_hi[i-1])
            if tf_lo[i-1] < tf_lo[i-2] and tf_lo[i-1] < tf_lo[i]: _try_add(conf, i-1, 'L', tf_lo[i-1])
            conf = _stage3(conf)
        cur_h = cur_l = np.nan
        for _, t, v in reversed(conf):
            if t == 'H' and math.isnan(cur_h): cur_h = v
            if t == 'L' and math.isnan(cur_l): cur_l = v
            if not math.isnan(cur_h) and not math.isnan(cur_l): break
        act_h_tf[i] = cur_h; act_l_tf[i] = cur_l

    # Propagate to M5
    ends = list(range(m5_per_tf - 1, n_oos, m5_per_tf))
    if len(ends) < n_tf: ends.append(n_oos - 1)
    act_h_m5 = np.full(n_oos, np.nan); act_l_m5 = np.full(n_oos, np.nan)
    for ti in range(n_tf):
        em = ends[ti]; nxt = ends[ti+1] if ti+1 < n_tf else n_oos
        act_h_m5[em:nxt] = act_h_tf[ti]; act_l_m5[em:nxt] = act_l_tf[ti]
    for i in range(1, n_oos):
        if math.isnan(act_h_m5[i]): act_h_m5[i] = act_h_m5[i-1]
        if math.isnan(act_l_m5[i]): act_l_m5[i] = act_l_m5[i-1]
    return act_h_m5, act_l_m5


# ── Numba JIT simulate ────────────────────────────────────────────────────────
@njit(cache=True)
def _simulate_jit(close_a, open_a, high_a, low_a, act_h, act_l,
                  tgt_frac, pip, spread, pf, max_legs):
    """
    Directional zone recovery simulation.
    Returns (total_pips, n_cycles, n_target, n_ml, sum_legs, sum_zw).
    Legs array: pre-allocated stack (vol[k], dir[k], price[k]) up to max_legs.
    """
    total_pips = 0.0; n_cyc = 0; n_tgt = 0; n_ml = 0
    sum_legs = 0.0; sum_zw = 0.0
    n_oos = len(close_a)
    leg_vol  = np.zeros(max_legs); leg_dir = np.zeros(max_legs); leg_px = np.zeros(max_legs)

    def net_bask(n_legs, price):
        gross = 0.0; cost = 0.0
        for k in range(n_legs):
            gross += leg_vol[k] * leg_dir[k] * (price - leg_px[k]) / pip
            cost  += leg_vol[k]
        return gross - cost * spread

    def bvol_fn(n_legs, target, tgt_pips):
        net = net_bask(n_legs, target)
        if net >= 0.0: return 0.0
        v = math.ceil(-net / tgt_pips * pf)
        return max(1.0, v)

    i = 0
    while i < n_oos:
        uh = act_h[i]; ul = act_l[i]
        if uh != uh or ul != ul or uh <= ul:  # nan check
            i += 1; continue
        zw_pips = (uh - ul) / pip
        tgt_pips = zw_pips * tgt_frac; tgt_b = tgt_pips * pip

        entry = close_a[i]
        if entry <= ul:   direction = 1.0
        elif entry >= uh: direction = -1.0
        else:             i += 1; continue

        ut = uh + tgt_b; lt = ul - tgt_b
        leg_vol[0]=1.0; leg_dir[0]=direction; leg_px[0]=entry; n_legs=1
        lc_upper = lc_lower = -1  # bar of last crossing
        closed = False; ep = entry; is_ml = False; is_tgt = False
        i += 1

        while i < n_oos and not closed:
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bull = cl >= open_a[i]
            # Process high then low (or low then high) depending on candle direction
            for pass_ in range(2):
                if closed: break
                if (bull and pass_ == 0) or (not bull and pass_ == 1):
                    # check high side
                    if hi >= ut: ep=ut; closed=True; is_tgt=True; break
                    if hi >= uh and lc_upper != i:
                        lc_upper = i
                        vol = bvol_fn(n_legs, ut, tgt_pips)
                        if vol > 0:
                            if n_legs >= max_legs: ep=cl; closed=True; is_ml=True; break
                            leg_vol[n_legs]=vol; leg_dir[n_legs]=1.0; leg_px[n_legs]=uh; n_legs+=1
                else:
                    # check low side
                    if lo <= lt: ep=lt; closed=True; is_tgt=True; break
                    if lo <= ul and lc_lower != i:
                        lc_lower = i
                        vol = bvol_fn(n_legs, lt, tgt_pips)
                        if vol > 0:
                            if n_legs >= max_legs: ep=cl; closed=True; is_ml=True; break
                            leg_vol[n_legs]=vol; leg_dir[n_legs]=-1.0; leg_px[n_legs]=ul; n_legs+=1
            if not closed: i += 1

        net = net_bask(n_legs, ep)
        total_pips += net; n_cyc += 1; sum_legs += n_legs; sum_zw += zw_pips
        if is_tgt: n_tgt += 1
        if is_ml:  n_ml  += 1
        if not closed: break

    return total_pips, n_cyc, n_tgt, n_ml, sum_legs, sum_zw


@njit(cache=True)
def _simulate_consensus_jit(close_a, open_a, high_a, low_a,
                             act_h1, act_l1, act_h2, act_l2,
                             tgt_frac, use_wider, pip, spread, pf, max_legs):
    """
    Two-TF consensus: LONG when close <= both supports; SHORT when close >= both resistances.
    Zone: TF1 (narrower) or TF2 (wider).
    """
    total_pips=0.0; n_cyc=0; n_tgt=0; n_ml=0; sum_legs=0.0; sum_zw=0.0
    n_oos=len(close_a)
    leg_vol=np.zeros(max_legs); leg_dir=np.zeros(max_legs); leg_px=np.zeros(max_legs)

    def net_bask(n_legs, price):
        gross=0.0; cost=0.0
        for k in range(n_legs):
            gross += leg_vol[k]*leg_dir[k]*(price-leg_px[k])/pip; cost+=leg_vol[k]
        return gross-cost*spread

    def bvol_fn(n_legs, target, tgt_pips):
        net=net_bask(n_legs,target)
        if net>=0.0: return 0.0
        return max(1.0, math.ceil(-net/tgt_pips*pf))

    i=0
    while i<n_oos:
        uh1=act_h1[i]; ul1=act_l1[i]; uh2=act_h2[i]; ul2=act_l2[i]
        if uh1!=uh1 or ul1!=ul1 or uh2!=uh2 or ul2!=ul2: i+=1; continue
        if uh1<=ul1 or uh2<=ul2: i+=1; continue
        entry=close_a[i]
        if entry<=ul1 and entry<=ul2: direction=1.0
        elif entry>=uh1 and entry>=uh2: direction=-1.0
        else: i+=1; continue
        uh=uh2 if use_wider else uh1; ul=ul2 if use_wider else ul1
        if uh<=ul: i+=1; continue
        zw_pips=(uh-ul)/pip; tgt_pips=zw_pips*tgt_frac; tgt_b=tgt_pips*pip
        ut=uh+tgt_b; lt=ul-tgt_b
        leg_vol[0]=1.0; leg_dir[0]=direction; leg_px[0]=entry; n_legs=1
        lc_upper=lc_lower=-1; closed=False; ep=entry; is_ml=False; is_tgt=False
        i+=1
        while i<n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for pass_ in range(2):
                if closed: break
                if (bull and pass_==0) or (not bull and pass_==1):
                    if hi>=ut: ep=ut; closed=True; is_tgt=True; break
                    if hi>=uh and lc_upper!=i:
                        lc_upper=i; vol=bvol_fn(n_legs,ut,tgt_pips)
                        if vol>0:
                            if n_legs>=max_legs: ep=cl; closed=True; is_ml=True; break
                            leg_vol[n_legs]=vol; leg_dir[n_legs]=1.0; leg_px[n_legs]=uh; n_legs+=1
                else:
                    if lo<=lt: ep=lt; closed=True; is_tgt=True; break
                    if lo<=ul and lc_lower!=i:
                        lc_lower=i; vol=bvol_fn(n_legs,lt,tgt_pips)
                        if vol>0:
                            if n_legs>=max_legs: ep=cl; closed=True; is_ml=True; break
                            leg_vol[n_legs]=vol; leg_dir[n_legs]=-1.0; leg_px[n_legs]=ul; n_legs+=1
            if not closed: i+=1
        net=net_bask(n_legs,ep); total_pips+=net; n_cyc+=1; sum_legs+=n_legs; sum_zw+=zw_pips
        if is_tgt: n_tgt+=1
        if is_ml:  n_ml+=1
        if not closed: break
    return total_pips,n_cyc,n_tgt,n_ml,sum_legs,sum_zw


def simulate(act_h, act_l, tgt_frac):
    tp, nc, nt, nm, sl, sz = _simulate_jit(
        close_a, open_a, high_a, low_a, act_h, act_l,
        tgt_frac, PIP, SPREAD, PF, MAX_LEGS)
    usd = tp * PIP_USD * UNITS
    return dict(total_pips=tp, total_usd=usd, n_cyc=int(nc),
                pct_tgt=nt/nc*100 if nc else 0, n_ml=int(nm),
                avg_legs=sl/nc if nc else 0, avg_zw=sz/nc if nc else 0)

def simulate_consensus(act_h1, act_l1, act_h2, act_l2, tgt_frac, use_wider):
    tp,nc,nt,nm,sl,sz = _simulate_consensus_jit(
        close_a, open_a, high_a, low_a, act_h1, act_l1, act_h2, act_l2,
        tgt_frac, use_wider, PIP, SPREAD, PF, MAX_LEGS)
    usd = tp * PIP_USD * UNITS
    return dict(total_pips=tp, total_usd=usd, n_cyc=int(nc),
                pct_tgt=nt/nc*100 if nc else 0, n_ml=int(nm),
                avg_legs=sl/nc if nc else 0, avg_zw=sz/nc if nc else 0)

def simulate_baseline():
    """Fixed ZW=56 TGT=28 random direction (production baseline)."""
    ZW=56; TGT=28
    rng=np.random.RandomState(42)
    act_h=np.empty(n_oos); act_l=np.empty(n_oos)
    # For baseline: build fake uniform S/R so all bars trigger
    # Actually reproduce original: use random direction at each entry
    # This needs pure Python since direction is random
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    leg_v=[None]*MAX_LEGS; leg_d=[None]*MAX_LEGS; leg_p=[None]*MAX_LEGS

    def nb(n_legs, price):
        g=sum(leg_v[k]*leg_d[k]*(price-leg_p[k])/PIP for k in range(n_legs))
        return g-sum(leg_v[k] for k in range(n_legs))*SPREAD

    def bv(n_legs, target):
        net=nb(n_legs,target)
        if net>=0: return 0.0
        return max(1.0,math.ceil(-net/TGT*PF))

    i=0
    while i<n_oos:
        entry=close_a[i]; d=int(rng.choice([-1,1]))
        if d==1: uz=entry; lz=entry-ZW*PIP; ut=entry+TGT*PIP; lt=lz-TGT*PIP
        else:    lz=entry; uz=entry+ZW*PIP; lt=entry-TGT*PIP; ut=uz+TGT*PIP
        n_legs=1; leg_v[0]=1.0; leg_d[0]=d; leg_p[0]=entry
        lc_u=lc_l=-1; closed=False; ep=entry; is_tgt=False; is_ml=False
        i+=1
        while i<n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for pass_ in range(2):
                if closed: break
                if (bull and pass_==0) or (not bull and pass_==1):
                    if hi>=ut: ep=ut; closed=True; is_tgt=True; break
                    if hi>=uz and lc_u!=i:
                        lc_u=i; v=bv(n_legs,ut)
                        if v>0:
                            if n_legs>=MAX_LEGS: ep=cl; closed=True; is_ml=True; break
                            leg_v[n_legs]=v; leg_d[n_legs]=1.0; leg_p[n_legs]=uz; n_legs+=1
                else:
                    if lo<=lt: ep=lt; closed=True; is_tgt=True; break
                    if lo<=lz and lc_l!=i:
                        lc_l=i; v=bv(n_legs,lt)
                        if v>0:
                            if n_legs>=MAX_LEGS: ep=cl; closed=True; is_ml=True; break
                            leg_v[n_legs]=v; leg_d[n_legs]=-1.0; leg_p[n_legs]=lz; n_legs+=1
            if not closed: i+=1
        tp+=nb(n_legs,ep); nc+=1; sl+=n_legs
        if is_tgt: nt+=1
        if is_ml: nm+=1
        if not closed: break
    usd=tp*PIP_USD*UNITS
    return dict(total_pips=tp,total_usd=usd,n_cyc=nc,
                pct_tgt=nt/nc*100 if nc else 0,n_ml=nm,
                avg_legs=sl/nc if nc else 0,avg_zw=56.0)


# ── Build S/R arrays ──────────────────────────────────────────────────────────
print(f"Building OHLC TopsBots S/R arrays...", flush=True)
h1_h, h1_l = build_ohlc_sr(12)   # H1
h2_h, h2_l = build_ohlc_sr(24)   # H2
h4_h, h4_l = build_ohlc_sr(48)   # H4

for label, ah, al in [('H1',h1_h,h1_l),('H2',h2_h,h2_l),('H4',h4_h,h4_l)]:
    v = np.sum(~np.isnan(ah) & ~np.isnan(al) & (ah>al))
    zw = (ah-al)[~np.isnan(ah) & (ah>al)] / PIP
    print(f"  {label}: valid={v:,}  zw median={np.median(zw):.1f}p  P10={np.percentile(zw,10):.1f}p  P90={np.percentile(zw,90):.1f}p", flush=True)

print("JIT compiling...", flush=True)
# Warmup JIT with tiny arrays
_simulate_jit(close_a[:100], open_a[:100], high_a[:100], low_a[:100],
              np.full(100,100.0), np.full(100,99.0), 0.25, PIP, SPREAD, PF, MAX_LEGS)
_simulate_consensus_jit(close_a[:100], open_a[:100], high_a[:100], low_a[:100],
                        np.full(100,100.0), np.full(100,99.0),
                        np.full(100,100.5), np.full(100,98.5),
                        0.25, False, PIP, SPREAD, PF, MAX_LEGS)
print("Running simulations...", flush=True)

base = simulate_baseline()
h1_25 = simulate(h1_h, h1_l, 0.25)
h1_50 = simulate(h1_h, h1_l, 0.50)
h2_25 = simulate(h2_h, h2_l, 0.25)
h2_50 = simulate(h2_h, h2_l, 0.50)
h4_25 = simulate(h4_h, h4_l, 0.25)
h4_50 = simulate(h4_h, h4_l, 0.50)
# Consensus: H1+H4, narrow zone (H1), tgt=0.25×ZW
con_h1h4_n25 = simulate_consensus(h1_h, h1_l, h4_h, h4_l, 0.25, False)
# Consensus: H1+H4, wide zone (H4), tgt=0.25×ZW
con_h1h4_w25 = simulate_consensus(h1_h, h1_l, h4_h, h4_l, 0.25, True)
# Consensus: H1+H2, narrow, tgt=0.25×ZW
con_h1h2_n25 = simulate_consensus(h1_h, h1_l, h2_h, h2_l, 0.25, False)
# Consensus: H1+H4, narrow, tgt=0.50×ZW
con_h1h4_n50 = simulate_consensus(h1_h, h1_l, h4_h, h4_l, 0.50, False)

# ── Report ────────────────────────────────────────────────────────────────────
base_usd  = base['total_usd']
h1_25_usd = h1_25['total_usd']

configs = [
    ("BASELINE fixed ZW=56 random",     base),
    ("H1 dir tgt=0.25×ZW [best ref]",   h1_25),
    ("H1 dir tgt=0.50×ZW",              h1_50),
    ("H2 dir tgt=0.25×ZW",              h2_25),
    ("H2 dir tgt=0.50×ZW",              h2_50),
    ("H4 dir tgt=0.25×ZW",              h4_25),
    ("H4 dir tgt=0.50×ZW",              h4_50),
    ("H1+H4 consensus narrow tgt=0.25", con_h1h4_n25),
    ("H1+H4 consensus wide  tgt=0.25",  con_h1h4_w25),
    ("H1+H2 consensus narrow tgt=0.25", con_h1h2_n25),
    ("H1+H4 consensus narrow tgt=0.50", con_h1h4_n50),
]

print()
print("="*118)
print(f"  Multi-TF S/R Sweep — {PAIR}  PF={PF}  directional entry at S/R boundary")
print("="*118)
print(f"  {'Config':<36} | {'TotalPips':>10} {'Total$@1ku':>11} | {'Cyc':>6} {'%Tgt':>6} {'%ML':>4} | {'AvgLeg':>7} {'AvgZW':>7} | {'vs Base':>8} | {'vs H1-25':>9}")
print("─"*118)

for label, s in configs:
    vs_base = (s['total_usd'] - base_usd) / abs(base_usd) * 100 if base_usd else 0
    vs_h1   = (s['total_usd'] - h1_25_usd) / abs(h1_25_usd) * 100 if h1_25_usd else 0
    flag = '🟢' if vs_h1 > 10 else ('🟡' if vs_h1 > -30 else '🔴')
    print(f"  {label:<36} | {s['total_pips']:>+10,.0f} {s['total_usd']:>+11,.0f} | "
          f"{s['n_cyc']:>6,} {s['pct_tgt']:>6.1f} {s['n_ml']:>4} | "
          f"{s['avg_legs']:>7.2f} {s['avg_zw']:>7.1f} | "
          f"{vs_base:>+7.1f}% | {vs_h1:>+8.1f}% {flag}")

print()
print("="*118)
print(f"\nBaseline: ${base_usd:+,.0f}  |  Best H1-25: ${h1_25_usd:+,.0f}")


# ── 12-pair summary (quick version for H4 best config) ────────────────────────
if __name__ == '__main__':
    pass
