"""
ASI Swing S/R — zone recovery sweep across all 12 pairs.

Wilder Accumulative Swing Index, causal TopsBots on ASI values.
S/R price = close at the bar where ASI made its confirmed swing.
Numba JIT for the inner simulation loop.

Compares: baseline, H4-25 (winner), ASI-25, ASI-50
"""

import os, math
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')

PIP_MAP = {
    'GBP_JPY': 0.01, 'EUR_JPY': 0.01, 'AUD_JPY': 0.01, 'CHF_JPY': 0.01,
    'NZD_JPY': 0.01, 'CAD_JPY': 0.01, 'USD_JPY': 0.01,
    'GBP_USD': 0.0001, 'EUR_USD': 0.0001, 'AUD_USD': 0.0001,
    'NZD_USD': 0.0001, 'EUR_GBP': 0.0001,
}
PIP_USD_MAP = {
    'GBP_JPY': 0.000091, 'EUR_JPY': 0.000091, 'AUD_JPY': 0.000091,
    'CHF_JPY': 0.000091, 'NZD_JPY': 0.000091, 'CAD_JPY': 0.000091,
    'USD_JPY': 0.000091,
    'GBP_USD': 0.0001, 'EUR_USD': 0.0001, 'AUD_USD': 0.0001,
    'NZD_USD': 0.0001, 'EUR_GBP': 0.0001,
}
PAIRS = list(PIP_MAP.keys())
UNITS    = 1_000
MAX_LEGS = 10
PF       = 1.25
SPREAD_USD  = 1.4   # JPY pairs — overridden per-pair below

SPREAD_MAP = {
    'GBP_JPY': 1.4, 'EUR_JPY': 1.0, 'AUD_JPY': 1.2, 'CHF_JPY': 1.2,
    'NZD_JPY': 1.5, 'CAD_JPY': 1.5, 'USD_JPY': 0.8,
    'GBP_USD': 1.2, 'EUR_USD': 0.8, 'AUD_USD': 1.0,
    'NZD_USD': 1.5, 'EUR_GBP': 1.2,
}


# ── Numba JIT simulation ──────────────────────────────────────────────────────
@njit(cache=True)
def _sim_jit(close_a, open_a, high_a, low_a, act_h, act_l,
             tgt_frac, pip, spread, pf, max_legs):
    total_pips=0.0; n_cyc=0; n_tgt=0; n_ml=0; sum_legs=0.0; sum_zw=0.0
    n=len(close_a)
    lv=np.zeros(max_legs); ld=np.zeros(max_legs); lp=np.zeros(max_legs)

    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g+=lv[k]*ld[k]*(price-lp[k])/pip; c+=lv[k]
        return g-c*spread

    def bv(nl, tgt, tpips):
        net=nb(nl,tgt)
        if net>=0.0: return 0.0
        return max(1.0, math.ceil(-net/tpips*pf))

    i=0
    while i<n:
        uh=act_h[i]; ul=act_l[i]
        if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
        zw=(uh-ul)/pip; tp=zw*tgt_frac; tb=tp*pip
        entry=close_a[i]
        if entry<=ul: d=1.0
        elif entry>=uh: d=-1.0
        else: i+=1; continue
        ut=uh+tb; lt=ul-tb
        lv[0]=1.0; ld[0]=d; lp[0]=entry; nl=1
        lcu=lcl=-1; closed=False; ep=entry; is_tgt=False; is_ml=False
        i+=1
        while i<n and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p in range(2):
                if closed: break
                if (bull and p==0) or (not bull and p==1):
                    if hi>=ut: ep=ut; closed=True; is_tgt=True; break
                    if hi>=uh and lcu!=i:
                        lcu=i; v=bv(nl,ut,tp)
                        if v>0:
                            if nl>=max_legs: ep=cl; closed=True; is_ml=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uh; nl+=1
                else:
                    if lo<=lt: ep=lt; closed=True; is_tgt=True; break
                    if lo<=ul and lcl!=i:
                        lcl=i; v=bv(nl,lt,tp)
                        if v>0:
                            if nl>=max_legs: ep=cl; closed=True; is_ml=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=ul; nl+=1
            if not closed: i+=1
        total_pips+=nb(nl,ep); n_cyc+=1; sum_legs+=nl; sum_zw+=zw
        if is_tgt: n_tgt+=1
        if is_ml:  n_ml+=1
        if not closed: break
    return total_pips, n_cyc, n_tgt, n_ml, sum_legs, sum_zw


@njit(cache=True)
def _sim_baseline(close_a, open_a, high_a, low_a, pip, spread, pf, max_legs, seed):
    ZW=56; TGT=28
    rng_state=np.uint64(seed)
    def _rand():
        nonlocal rng_state
        rng_state^=rng_state<<13; rng_state^=rng_state>>7; rng_state^=rng_state<<17
        return rng_state
    total_pips=0.0; n_cyc=0; n_tgt=0; n_ml=0; sum_legs=0.0
    n=len(close_a)
    lv=np.zeros(max_legs); ld=np.zeros(max_legs); lp=np.zeros(max_legs)

    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g+=lv[k]*ld[k]*(price-lp[k])/pip; c+=lv[k]
        return g-c*spread

    def bv(nl, tgt_p):
        net=nb(nl,tgt_p)
        if net>=0.0: return 0.0
        return max(1.0, math.ceil(-net/TGT*pf))

    i=0
    while i<n:
        entry=close_a[i]; d=1.0 if _rand()%2==0 else -1.0
        uz=entry; lz=entry-ZW*pip; ut=entry+TGT*pip; lt=lz-TGT*pip
        if d==-1.0: lz=entry; uz=entry+ZW*pip; lt=entry-TGT*pip; ut=uz+TGT*pip
        lv[0]=1.0; ld[0]=d; lp[0]=entry; nl=1
        lcu=lcl=-1; closed=False; ep=entry; is_tgt=False; is_ml=False
        i+=1
        while i<n and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p in range(2):
                if closed: break
                if (bull and p==0) or (not bull and p==1):
                    if hi>=ut: ep=ut; closed=True; is_tgt=True; break
                    if hi>=uz and lcu!=i:
                        lcu=i; v=bv(nl,ut)
                        if v>0:
                            if nl>=max_legs: ep=cl; closed=True; is_ml=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                else:
                    if lo<=lt: ep=lt; closed=True; is_tgt=True; break
                    if lo<=lz and lcl!=i:
                        lcl=i; v=bv(nl,lt)
                        if v>0:
                            if nl>=max_legs: ep=cl; closed=True; is_ml=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            if not closed: i+=1
        total_pips+=nb(nl,ep); n_cyc+=1; sum_legs+=nl
        if is_tgt: n_tgt+=1
        if is_ml:  n_ml+=1
        if not closed: break
    return total_pips, n_cyc, n_tgt, n_ml, sum_legs


# ── ASI S/R builder ───────────────────────────────────────────────────────────
def _try_add(conf, item):
    idx, t, v = item
    if conf and conf[-1][1] == t:
        if t == 'H' and v > conf[-1][2]: conf[-1] = [idx, t, v]
        elif t == 'L' and v < conf[-1][2]: conf[-1] = [idx, t, v]
    else:
        conf.append([idx, t, v])

def _stage3(raw):
    sig=[]; lh=ll=float('nan'); glh=glh2=False
    for idx,t,v in raw:
        if t=='H':
            if math.isnan(lh) or v>lh or glh: sig.append([idx,t,v]); lh=v; glh=False; glh2=True
        else:
            if math.isnan(ll) or v<ll or glh2: sig.append([idx,t,v]); ll=v; glh2=False; glh=True
    return sig

def _get_act(conf):
    ch=cl=float('nan')
    for _,t,v in reversed(conf):
        if t=='H' and math.isnan(ch): ch=v
        if t=='L' and math.isnan(cl): cl=v
        if not math.isnan(ch) and not math.isnan(cl): break
    return ch, cl

def build_asi_sr(close_a, open_a, high_a, low_a, n_oos):
    """
    Vectorized Wilder ASI → TopsBots → S/R close prices.

    Fast path: numpy vectorized Stage-1, then Python Stage-2/3 on the small
    event list (~10K items), then O(n_oos) forward-fill. Avoids the O(n²)
    per-bar Stage-3 rebuild that made the incremental loop take >5 min/pair.
    """
    EPSILON = 1e-10
    C2,O2,H2,L2 = close_a[1:],open_a[1:],high_a[1:],low_a[1:]
    C1,O1 = close_a[:-1],open_a[:-1]
    N  = (C2-C1) + 0.5*(C2-O2) + 0.25*(C1-O1)
    t1 = np.abs(H2-C1) - 0.5*np.abs(L2-C1) + 0.25*np.abs(C1-O1)
    t2 = np.abs(L2-C1) - 0.5*np.abs(H2-C1) + 0.25*np.abs(C1-O1)
    t3 = (H2-L2) + 0.25*np.abs(C1-O1)
    R  = np.maximum(np.maximum(t1,t2), np.maximum(t3, EPSILON))
    K  = np.maximum(np.abs(H2-C1), np.abs(L2-C1))
    SI = 50.0*(N/R)*(K/np.maximum(K, EPSILON))
    asi = np.zeros(n_oos); asi[1:] = np.cumsum(SI)

    # Vectorized Stage-1: local extremes (confirmed 1-bar lag → avail at j+1)
    jh = np.where((asi[1:-1] > asi[:-2]) & (asi[1:-1] > asi[2:]))[0] + 1
    jl = np.where((asi[1:-1] < asi[:-2]) & (asi[1:-1] < asi[2:]))[0] + 1
    evts = ([(j, 'H', close_a[j], j+1) for j in jh] +
            [(j, 'L', close_a[j], j+1) for j in jl])
    evts.sort(key=lambda x: x[0])

    # Stage-2: alternation + most-extreme of consecutive same-type
    s2 = []
    for j, t, v, avail in evts:
        if s2 and s2[-1][1] == t:
            if t == 'H' and v > s2[-1][2]: s2[-1] = (j, t, v, avail)
            elif t == 'L' and v < s2[-1][2]: s2[-1] = (j, t, v, avail)
        else:
            s2.append((j, t, v, avail))

    # Stage-3: exceeding-extremes gate
    s3 = []
    lh = ll = float('nan'); glh = glh2 = False
    for j, t, v, avail in s2:
        if t == 'H':
            if math.isnan(lh) or v > lh or glh:
                s3.append((j, t, v, avail)); lh = v; glh = False; glh2 = True
        else:
            if math.isnan(ll) or v < ll or glh2:
                s3.append((j, t, v, avail)); ll = v; glh2 = False; glh = True

    # Separate H and L events for forward-fill
    h_ev = [(avail, v) for j, t, v, avail in s3 if t == 'H']
    l_ev = [(avail, v) for j, t, v, avail in s3 if t == 'L']

    # Forward-fill into M5 arrays (O(n_oos) pass)
    act_h = np.full(n_oos, np.nan); act_l = np.full(n_oos, np.nan)
    cur_h = cur_l = float('nan')
    hi = li = 0
    for i in range(n_oos):
        while hi < len(h_ev) and h_ev[hi][0] <= i: cur_h = h_ev[hi][1]; hi += 1
        while li < len(l_ev) and l_ev[li][0] <= i: cur_l = l_ev[li][1]; li += 1
        act_h[i] = cur_h; act_l[i] = cur_l
    return act_h, act_l

def build_h4_sr(close_a, open_a, high_a, low_a, n_oos):
    """H4 OHLC TopsBots S/R (reference from multitf_sweep)."""
    m5_per_tf=48
    tf_hi=[]; tf_lo=[]; ends=[]
    for start in range(0,n_oos,m5_per_tf):
        end=min(start+m5_per_tf,n_oos)
        tf_hi.append(float(np.max(high_a[start:end])))
        tf_lo.append(float(np.min(low_a[start:end])))
        ends.append(end-1)
    tf_hi=np.array(tf_hi); tf_lo=np.array(tf_lo); n_tf=len(tf_hi)
    conf=[]
    act_h_tf=np.full(n_tf,np.nan); act_l_tf=np.full(n_tf,np.nan)
    for i in range(1,n_tf):
        if i>=2:
            if tf_hi[i-1]>tf_hi[i-2] and tf_hi[i-1]>tf_hi[i]: _try_add(conf,(i-1,'H',tf_hi[i-1]))
            if tf_lo[i-1]<tf_lo[i-2] and tf_lo[i-1]<tf_lo[i]: _try_add(conf,(i-1,'L',tf_lo[i-1]))
            conf=_stage3(conf)
        ch,cl=_get_act(conf); act_h_tf[i]=ch; act_l_tf[i]=cl
    act_h=np.full(n_oos,np.nan); act_l=np.full(n_oos,np.nan)
    for ti,em in enumerate(ends):
        nxt=ends[ti+1] if ti+1<n_tf else n_oos
        act_h[em:nxt]=act_h_tf[ti]; act_l[em:nxt]=act_l_tf[ti]
    for i in range(1,n_oos):
        if np.isnan(act_h[i]): act_h[i]=act_h[i-1]
        if np.isnan(act_l[i]): act_l[i]=act_l[i-1]
    return act_h, act_l


def run_pair(pair):
    path = f'{DATA_DIR}/{pair}_M5.parquet'
    if not os.path.exists(path):
        return None
    pip     = PIP_MAP[pair]
    pip_usd = PIP_USD_MAP[pair]
    spread  = SPREAD_MAP[pair]

    df = pd.read_parquet(path).sort_index()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    df_oos = df.iloc[int(n*0.70):].reset_index(drop=True)
    o = df_oos['open'].values.astype(np.float64)
    h = df_oos['high'].values.astype(np.float64)
    l = df_oos['low'].values.astype(np.float64)
    c = df_oos['close'].values.astype(np.float64)
    noos = len(c)

    def sim(ah, al, tgt_frac):
        tp,nc,nt,nm,sl,sz = _sim_jit(c,o,h,l,ah,al,tgt_frac,pip,spread,PF,MAX_LEGS)
        return tp*pip_usd*UNITS, int(nc), sz/nc if nc else 0, sz/nc if nc else 0

    def base():
        tp,nc,nt,nm,sl = _sim_baseline(c,o,h,l,pip,spread,PF,MAX_LEGS,np.uint64(42))
        return tp*pip_usd*UNITS, int(nc)

    # Build S/R
    ah4,al4 = build_h4_sr(c,o,h,l,noos)
    aah,aal = build_asi_sr(c,o,h,l,noos)

    b_usd, b_cyc  = base()
    h4_usd, h4_cyc,_,_ = sim(ah4,al4,0.25)
    a25_usd, a25_cyc,_,_ = sim(aah,aal,0.25)
    a50_usd, a50_cyc,_,_ = sim(aah,aal,0.50)

    # ASI zone stats
    valid = (~np.isnan(aah)) & (~np.isnan(aal)) & (aah>aal)
    zw_asi = (aah[valid]-aal[valid])/pip
    zw_med = float(np.median(zw_asi)) if len(zw_asi) else 0

    return pair, b_usd, b_cyc, h4_usd, h4_cyc, a25_usd, a25_cyc, a50_usd, a50_cyc, zw_med


# ── JIT warmup ────────────────────────────────────────────────────────────────
print("JIT warmup...", flush=True)
_ca=np.array([100.0]*100,dtype=np.float64)
_sim_jit(_ca,_ca,_ca+0.1,_ca-0.1,np.full(100,100.5),np.full(100,99.5),
         0.25,0.01,1.4,PF,MAX_LEGS)
_sim_baseline(_ca,_ca,_ca+0.1,_ca-0.1,0.01,1.4,PF,MAX_LEGS,np.uint64(42))
print("Running ASI sweep across 12 pairs...\n", flush=True)

results = []
for pair in PAIRS:
    print(f"  {pair}...", flush=True, end=' ')
    r = run_pair(pair)
    if r: results.append(r); print("done", flush=True)
    else: print("MISSING", flush=True)

print()
print("="*110)
print(f"  ASI Swing S/R — 12-pair OOS  PF={PF}  SPREAD=per-pair  tgt_frac=0.25/0.50")
print("="*110)
print(f"  {'Pair':<10} | {'Base$':>9} | {'H4-25$':>9} vs% | {'ASI-25$':>9} vs% | {'ASI-50$':>9} vs% | ZWmed(ASI)")
print("─"*110)

tot_b=tot_h4=tot_a25=tot_a50=0.0
for r in results:
    pair,b,bc,h4,h4c,a25,a25c,a50,a50c,zwm = r
    def pct(v,ref): return (v-ref)/abs(ref)*100 if ref else 0
    flag25='🟢' if a25>h4 else ('🟡' if a25>b else '🔴')
    flag50='🟢' if a50>h4 else ('🟡' if a50>b else '🔴')
    print(f"  {pair:<10} | {b:>+9,.0f} | {h4:>+9,.0f} {pct(h4,b):>+5.0f}% | "
          f"{a25:>+9,.0f} {pct(a25,b):>+5.0f}% {flag25} | "
          f"{a50:>+9,.0f} {pct(a50,b):>+5.0f}% {flag50} | {zwm:>6.1f}p")
    tot_b+=b; tot_h4+=h4; tot_a25+=a25; tot_a50+=a50

print("─"*110)
def pct(v,ref): return (v-ref)/abs(ref)*100 if ref else 0
print(f"  {'AGGREGATE':<10} | {tot_b:>+9,.0f} | {tot_h4:>+9,.0f} {pct(tot_h4,tot_b):>+5.0f}% | "
      f"{tot_a25:>+9,.0f} {pct(tot_a25,tot_b):>+5.0f}%   | "
      f"{tot_a50:>+9,.0f} {pct(tot_a50,tot_b):>+5.0f}%   |")
print()
print(f"  H4-25 aggregate: ${tot_h4:+,.0f}")
print(f"  ASI-25 aggregate: ${tot_a25:+,.0f}  ({'BETTER' if tot_a25>tot_h4 else 'WORSE'} than H4-25 by ${abs(tot_a25-tot_h4):,.0f})")
print(f"  ASI-50 aggregate: ${tot_a50:+,.0f}  ({'BETTER' if tot_a50>tot_h4 else 'WORSE'} than H4-25 by ${abs(tot_a50-tot_h4):,.0f})")
