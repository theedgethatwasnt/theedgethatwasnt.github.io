"""
Backtest: H4-directional entry + fixed ZW=56p boundary geometry.

This is the exact mirror of the deployed live code after 2026-04-30 fixes:
  - H4 TopsBots S/R provides entry TRIGGER and DIRECTION only
  - Zone geometry: LONG→upper=entry/lower=entry-ZW, SHORT→lower=entry/upper=entry+ZW
  - Fixed ZW=56p, tgt=ZW*0.25=14p, PF=1.25, spread=1.4

Compared against:
  A. Baseline:       random direction + fixed ZW=56  (original pre-directional baseline)
  B. Old live code:  H4 direction    + dynamic ZW    (had hedge-disabled bug + no ZW cap)
  C. New live code:  H4 direction    + fixed ZW=56   (current deployed)
"""

import os, math
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.expanduser('~/projects/fx-core/data/m5_ohlc')
PAIRS = ['AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY','EUR_GBP','EUR_JPY',
         'EUR_USD','GBP_JPY','GBP_USD','NZD_JPY','NZD_USD','USD_JPY']
PIP_USD_MAP = {'AUD_JPY':0.000067,'AUD_USD':0.000100,'CAD_JPY':0.000069,
               'CHF_JPY':0.000107,'EUR_GBP':0.000126,'EUR_JPY':0.000064,
               'EUR_USD':0.000100,'GBP_JPY':0.000091,'GBP_USD':0.000100,
               'NZD_JPY':0.000061,'NZD_USD':0.000100,'USD_JPY':0.000064}
PIP_MAP = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}
UNITS = 1_000; MAX_LEGS = 10; PF = 1.25; SPREAD = 1.4
ZW_PIPS = 56.0; TGT_PIPS = ZW_PIPS * 0.25   # 14p


# ── A. Baseline: random direction + fixed ZW=56 ───────────────────────────────
@njit(cache=True)
def _sim_baseline(close_a, open_a, high_a, low_a, rng_dirs, pip, spread, pf, ml):
    ZW = ZW_PIPS; TGT = TGT_PIPS
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a); lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g += lv[k]*ld[k]*(price-lp[k])/pip; c += lv[k]
        return g - c*spread
    def bv(nl, tgt):
        net = nb(nl, tgt)
        if net >= 0.0: return 0.0
        return max(1.0, math.ceil(-net / TGT * pf))
    i=0; ri=0
    while i < n:
        entry = close_a[i]; dr = rng_dirs[ri % len(rng_dirs)]; ri += 1
        if dr == 1.0: uz=entry; lz=entry-ZW*pip; ut=entry+TGT*pip; lt=lz-TGT*pip
        else:         lz=entry; uz=entry+ZW*pip; lt=entry-TGT*pip; ut=uz+TGT*pip
        lv[0]=1.0; ld[0]=dr; lp[0]=entry; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i += 1
        while i < n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uz and lu!=i:
                        lu=i; v=bv(nl,ut)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=lz and ll!=i:
                        ll=i; v=bv(nl,lt)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            if not cl2: i += 1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp, nc, nt, nm, sl


# ── B. Old live: H4 direction + dynamic ZW (bug present but rarely fires in backtest) ──
@njit(cache=True)
def _sim_old_live(close_a, open_a, high_a, low_a, act_h, act_l, tgt_frac, pip, spread, pf, ml):
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a); lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g += lv[k]*ld[k]*(price-lp[k])/pip; c += lv[k]
        return g - c*spread
    def bv(nl, tgt, tp2):
        net = nb(nl, tgt)
        if net >= 0.0: return 0.0
        return max(1.0, math.ceil(-net / tp2 * pf))
    i = 0
    while i < n:
        uh=act_h[i]; ul=act_l[i]
        if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
        zw=(uh-ul)/pip; tp2=zw*tgt_frac; tb=tp2*pip
        entry=close_a[i]
        if entry<=ul: dr=1.0
        elif entry>=uh: dr=-1.0
        else: i+=1; continue
        # Old geometry: zone = H4 S/R boundaries (dynamic ZW)
        ut=uh+tb; lt=ul-tb
        # Bug: lp[0]=entry (can be below lower_zone-tgt, disabling hedges)
        lv[0]=1.0; ld[0]=dr; lp[0]=entry; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i += 1
        while i < n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uh and lu!=i:
                        lu=i; v=bv(nl,ut,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uh; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=ul and ll!=i:
                        ll=i; v=bv(nl,lt,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=ul; nl+=1
            if not cl2: i += 1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp, nc, nt, nm, sl


# ── C. New live: H4 direction + fixed ZW=56 boundary geometry ─────────────────
@njit(cache=True)
def _sim_new_live(close_a, open_a, high_a, low_a, act_h, act_l, pip, spread, pf, ml):
    ZW = ZW_PIPS; TGT = TGT_PIPS
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a); lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g += lv[k]*ld[k]*(price-lp[k])/pip; c += lv[k]
        return g - c*spread
    def bv(nl, tgt):
        net = nb(nl, tgt)
        if net >= 0.0: return 0.0
        return max(1.0, math.ceil(-net / TGT * pf))
    i = 0
    while i < n:
        uh=act_h[i]; ul=act_l[i]
        if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
        entry=close_a[i]
        if entry<=ul: dr=1.0
        elif entry>=uh: dr=-1.0
        else: i+=1; continue
        # Fixed ZW boundary geometry — exact mirror of live _start_cycle
        if dr == 1.0: uz=entry; lz=entry-ZW*pip; ut=entry+TGT*pip; lt=lz-TGT*pip
        else:         lz=entry; uz=entry+ZW*pip; lt=entry-TGT*pip; ut=uz+TGT*pip
        # lp[0]=entry = upper_zone(LONG)/lower_zone(SHORT) → get_pl_at(lower_tgt)=-(ZW+TGT) always negative
        lv[0]=1.0; ld[0]=dr; lp[0]=entry; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i += 1
        while i < n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uz and lu!=i:
                        lu=i; v=bv(nl,ut)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=lz and ll!=i:
                        ll=i; v=bv(nl,lt)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            if not cl2: i += 1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp, nc, nt, nm, sl


# ── H4 TopsBots S/R builder (same as live _topsbots_last) ─────────────────────
def build_h4_sr(high_a, low_a, m5_per_h4=48):
    n = len(high_a)
    tf_hi, tf_lo = [], []
    for s in range(0, n, m5_per_h4):
        e = min(s + m5_per_h4, n)
        tf_hi.append(float(np.max(high_a[s:e])))
        tf_lo.append(float(np.min(low_a[s:e])))
    tf_hi = np.array(tf_hi); tf_lo = np.array(tf_lo)
    n_tf = len(tf_hi)

    def _try_add(conf, idx, t, v):
        if conf and conf[-1][1] == t:
            if t == 'H' and v > conf[-1][2]: conf[-1] = [idx, t, v]
            elif t == 'L' and v < conf[-1][2]: conf[-1] = [idx, t, v]
        else:
            conf.append([idx, t, v])

    def _stage3(raw):
        sig=[]; lh=ll=float('nan'); glh=glh2=False
        for idx, t, v in raw:
            if t == 'H':
                if math.isnan(lh) or v > lh or glh:
                    sig.append([idx, t, v]); lh=v; glh=False; glh2=True
            else:
                if math.isnan(ll) or v < ll or glh2:
                    sig.append([idx, t, v]); ll=v; glh2=False; glh=True
        return sig

    conf = []
    ah = np.full(n_tf, np.nan); al = np.full(n_tf, np.nan)
    for i in range(1, n_tf):
        if i >= 2:
            if tf_hi[i-1] > tf_hi[i-2] and tf_hi[i-1] > tf_hi[i]:
                _try_add(conf, i-1, 'H', tf_hi[i-1])
            if tf_lo[i-1] < tf_lo[i-2] and tf_lo[i-1] < tf_lo[i]:
                _try_add(conf, i-1, 'L', tf_lo[i-1])
            conf = _stage3(conf)
        ch = cl = float('nan')
        for _, t, v in reversed(conf):
            if t == 'H' and math.isnan(ch): ch = v
            if t == 'L' and math.isnan(cl): cl = v
            if not math.isnan(ch) and not math.isnan(cl): break
        ah[i] = ch; al[i] = cl

    # Map TF values back to M5 bars (forward-fill, update at end of each H4 bar)
    ends = list(range(m5_per_h4 - 1, n, m5_per_h4))
    ahm = np.full(n, np.nan); alm = np.full(n, np.nan)
    for ti in range(n_tf):
        em = ends[ti] if ti < len(ends) else n - 1
        nxt = ends[ti+1] if ti+1 < len(ends) else n
        ahm[em:nxt] = ah[ti]; alm[em:nxt] = al[ti]
    for i in range(1, n):
        if math.isnan(ahm[i]): ahm[i] = ahm[i-1]
        if math.isnan(alm[i]): alm[i] = alm[i-1]
    return ahm, alm


# ── JIT warmup ────────────────────────────────────────────────────────────────
_d = np.ones(200); _h = _d * 1.01; _l = _d * 0.99
_ah = _d * 1.005; _al = _d * 0.995
_dirs = np.ones(200)
_sim_baseline(_d, _d, _h, _l, _dirs, 0.01, SPREAD, PF, MAX_LEGS)
_sim_old_live(_d, _d, _h, _l, _ah, _al, 0.25, 0.01, SPREAD, PF, MAX_LEGS)
_sim_new_live(_d, _d, _h, _l, _ah, _al, 0.01, SPREAD, PF, MAX_LEGS)
print("JIT compiled\n", flush=True)

# ── Run 12 pairs ──────────────────────────────────────────────────────────────
agg = {'base': 0.0, 'old': 0.0, 'new': 0.0}
print(f"{'Pair':<10} | {'Base $':>9} | {'OldLive $':>10} {'vs%':>5} | {'NewLive $':>10} {'vs%':>5} | {'nc':>5} {'tgt%':>6} {'legs':>5}")
print("─" * 80)

for pair in PAIRS:
    pip = PIP_MAP[pair]; pu = PIP_USD_MAP[pair]
    df = pd.read_parquet(f'{DATA_DIR}/{pair}_M5.parquet').sort_index()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    oos = df.iloc[int(n * 0.70):].reset_index(drop=True)
    oa = oos['open'].values.astype(np.float64)
    ha = oos['high'].values.astype(np.float64)
    la = oos['low'].values.astype(np.float64)
    ca = oos['close'].values.astype(np.float64)

    h4h, h4l = build_h4_sr(ha, la)

    rng = np.random.RandomState(42)
    dirs = rng.choice(np.array([-1.0, 1.0]), len(ca))

    btp, bnc, bnt, bnm, bsl = _sim_baseline(ca, oa, ha, la, dirs, pip, SPREAD, PF, MAX_LEGS)
    otp, onc, ont, onm, osl = _sim_old_live(ca, oa, ha, la, h4h, h4l, 0.25, pip, SPREAD, PF, MAX_LEGS)
    ntp, nnc, nnt, nnm, nsl = _sim_new_live(ca, oa, ha, la, h4h, h4l, pip, SPREAD, PF, MAX_LEGS)

    busd = btp * pu * UNITS
    ousd = otp * pu * UNITS
    nusd = ntp * pu * UNITS

    vs_old = (ousd - busd) / max(abs(busd), 1) * 100
    vs_new = (nusd - busd) / max(abs(busd), 1) * 100
    tgt_pct = (nnt / nnc * 100) if nnc > 0 else 0
    avg_legs = (nsl / nnc) if nnc > 0 else 0

    fo = '🟢' if vs_old > 30 else ('🟡' if vs_old > 0 else '🔴')
    fn = '🟢' if vs_new > 30 else ('🟡' if vs_new > 0 else '🔴')

    print(f"{pair:<10} | {busd:>+9,.0f} | {ousd:>+10,.0f} {vs_old:>+4.0f}%{fo} | "
          f"{nusd:>+10,.0f} {vs_new:>+4.0f}%{fn} | {nnc:>5} {tgt_pct:>5.1f}% {avg_legs:>5.1f}",
          flush=True)

    agg['base'] += busd; agg['old'] += ousd; agg['new'] += nusd

print("─" * 80)
vs_old_a = (agg['old'] - agg['base']) / max(abs(agg['base']), 1) * 100
vs_new_a = (agg['new'] - agg['base']) / max(abs(agg['base']), 1) * 100
print(f"{'AGGREGATE':<10} | {agg['base']:>+9,.0f} | {agg['old']:>+10,.0f} {vs_old_a:>+4.0f}%  | "
      f"{agg['new']:>+10,.0f} {vs_new_a:>+4.0f}%")
print(f"\nConfig: ZW={ZW_PIPS:.0f}p  tgt={TGT_PIPS:.0f}p  PF={PF}  spread={SPREAD}p  "
      f"max_legs={MAX_LEGS}  OOS=30%  units={UNITS}")
print(f"Columns: Base=random-dir+fixedZW | OldLive=H4-dir+dynamicZW | NewLive=H4-dir+fixedZW(current)")


# ── D. Dynamic ZW + hedge bug fixed (grid_price=lower_zone) + MAX_ZW cap ──────
@njit(cache=True)
def _sim_fixed_hedge(close_a, open_a, high_a, low_a, act_h, act_l,
                     tgt_frac, max_zw, pip, spread, pf, ml):
    """Dynamic ZW with: (a) hedge bug fixed (grid_price=zone_boundary), (b) MAX_ZW cap."""
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a); lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl, price):
        g=0.0; c=0.0
        for k in range(nl): g += lv[k]*ld[k]*(price-lp[k])/pip; c += lv[k]
        return g - c*spread
    def bv(nl, tgt, tp2):
        net = nb(nl, tgt)
        if net >= 0.0: return 0.0
        return max(1.0, math.ceil(-net / tp2 * pf))
    i = 0
    while i < n:
        uh=act_h[i]; ul=act_l[i]
        if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
        zw=(uh-ul)/pip
        if zw > max_zw: i+=1; continue      # ZW cap
        tp2=zw*tgt_frac; tb=tp2*pip
        entry=close_a[i]
        if entry<=ul: dr=1.0
        elif entry>=uh: dr=-1.0
        else: i+=1; continue
        ut=uh+tb; lt=ul-tb
        # Bug fixed: grid_price = zone boundary (not fill price)
        grid_p = ul if dr==1.0 else uh
        lv[0]=1.0; ld[0]=dr; lp[0]=grid_p; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i += 1
        while i < n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uh and lu!=i:
                        lu=i; v=bv(nl,ut,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uh; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=ul and ll!=i:
                        ll=i; v=bv(nl,lt,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=ul; nl+=1
            if not cl2: i += 1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp, nc, nt, nm, sl


# JIT warmup for D
_sim_fixed_hedge(_d, _d, _h, _l, _ah, _al, 0.25, 999.0, 0.01, SPREAD, PF, MAX_LEGS)
print("\n── D. Dynamic ZW + hedge fix + MAX_ZW=150 vs baseline ──\n")
print(f"{'Pair':<10} | {'Base $':>9} | {'DynZW+fix $':>12} {'vs%':>5} | {'nc':>5} {'tgt%':>6} {'avgZW':>7} {'legs':>5}")
print("─" * 70)

agg_d = 0.0
for pair in PAIRS:
    pip = PIP_MAP[pair]; pu = PIP_USD_MAP[pair]
    df = pd.read_parquet(f'{DATA_DIR}/{pair}_M5.parquet').sort_index()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    oos = df.iloc[int(n * 0.70):].reset_index(drop=True)
    oa = oos['open'].values.astype(np.float64)
    ha = oos['high'].values.astype(np.float64)
    la = oos['low'].values.astype(np.float64)
    ca = oos['close'].values.astype(np.float64)
    h4h, h4l = build_h4_sr(ha, la)

    rng = np.random.RandomState(42)
    dirs = rng.choice(np.array([-1.0, 1.0]), len(ca))
    btp, *_ = _sim_baseline(ca, oa, ha, la, dirs, pip, SPREAD, PF, MAX_LEGS)
    busd = btp * pu * UNITS

    dtp, dnc, dnt, dnm, dsl = _sim_fixed_hedge(
        ca, oa, ha, la, h4h, h4l, 0.25, 150.0, pip, SPREAD, PF, MAX_LEGS)
    dusd = dtp * pu * UNITS
    vs_d = (dusd - busd) / max(abs(busd), 1) * 100
    tgt_pct = (dnt / dnc * 100) if dnc > 0 else 0
    avg_legs = (dsl / dnc) if dnc > 0 else 0
    # Avg ZW for entered cycles: approx from h4 array at entry bars
    valid_zw = (h4h - h4l) / pip
    valid_zw = valid_zw[(h4h > 0) & (h4l > 0) & ((h4h - h4l) / pip <= 150)]
    avg_zw = float(np.nanmean(valid_zw)) if len(valid_zw) > 0 else 0

    fd = '🟢' if vs_d > 30 else ('🟡' if vs_d > 0 else '🔴')
    print(f"{pair:<10} | {busd:>+9,.0f} | {dusd:>+12,.0f} {vs_d:>+4.0f}%{fd} | "
          f"{dnc:>5} {tgt_pct:>5.1f}% {avg_zw:>7.1f}p {avg_legs:>5.1f}", flush=True)
    agg_d += dusd

vs_da = (agg_d - agg['base']) / max(abs(agg['base']), 1) * 100
print("─" * 70)
print(f"{'AGGREGATE':<10} | {agg['base']:>+9,.0f} | {agg_d:>+12,.0f} {vs_da:>+4.0f}%")
print(f"\nD = H4-directional + dynamic ZW + hedge-bug-fixed + MAX_ZW=150p cap")
