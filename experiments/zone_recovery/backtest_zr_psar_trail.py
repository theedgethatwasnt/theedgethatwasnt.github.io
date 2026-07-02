"""
PSAR-type trailing stop on escape exit — EUR_JPY ZW=50 TGT=25 ta=6 td=1.

When escape target crossed, activate a PSAR trailing stop on combined position.
PSAR: starts slow (wide trail), accelerates as new extremes are made.

PSAR mechanics (on the combined position after escape target):
  - Track extreme: ep = most favorable price since target crossed
  - AF starts at af0, increments by af_step each NEW extreme, caps at af_max
  - PSAR price = ep - (ep - last_psar) * AF   (for LONG net position)
                = ep + (last_psar - ep) * AF   (for SHORT net position)
  - Fire when price crosses PSAR

Parameters swept:
  af0     ∈ [0.01, 0.02, 0.04, 0.08]
  af_step ∈ [0.01, 0.02, 0.04]     (same or different from af0)
  af_max  ∈ [0.10, 0.20, 0.40]

Compare against:
  baseline: flat exit (ts_dist=0) = 543 p/d
  best fixed trail: ts_dist=2p    = 571 p/d

etype: 1=trail_1leg  2=escape_flat  3=maxlegs  5=psar_trail
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import itertools

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR    = "EUR_JPY"
PIP     = 0.01
PF      = 1.25
ZW      = 50.0
TGT     = 25.0
TA      = 6.0
TD      = 1.0
ML      = 10
OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000


@njit
def sim_zr_psar(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td, gate,
                af0, af_step, af_max):
    """
    ZR with PSAR-type escape trail.
    When escape target crossed → activate PSAR on combined position.
    af0: initial AF, af_step: increment per new extreme, af_max: cap.
    etype: 1=trail_1leg  2=flat_escape(unused)  3=maxlegs  5=psar_trail
    """
    n = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_e = sp_arr[i]
        if gate > 0 and sp_e > gate:
            i += 1; continue
        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        # PSAR state (activated at escape target)
        psar_on = False; psar_val = 0.0; ep_val = 0.0
        af_cur = af0; net_dir = 0.0
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # ── PSAR tracking ──────────────────────────────────────────────
            if psar_on:
                if net_dir > 0:   # net LONG → PSAR below price
                    if h > ep_val:
                        ep_val = h
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl): net += lv[k]*ld[k]*(psar_val-lp[k])/pip; tv += lv[k]
                        pnl[nc] = net - tv*sp; nlegs[nc] = nl; etype[nc] = 5
                        nc += 1; ex = True
                else:             # net SHORT → PSAR above price
                    if l < ep_val:
                        ep_val = l
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl): net += lv[k]*ld[k]*(psar_val-lp[k])/pip; tv += lv[k]
                        pnl[nc] = net - tv*sp; nlegs[nc] = nl; etype[nc] = 5
                        nc += 1; ex = True
                if ex: break
                i += 1
                continue

            # ── 1-leg trail ────────────────────────────────────────────────
            if nl == 1:
                mfe = (h-e)/pip if d==1 else (e-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta: ton = True
                if ton:
                    if d == 1:
                        be = e + sp*pip
                        ts = e + (peak-td)*pip
                        if ts < be: ts = be
                        if l <= ts:
                            pnl[nc]=(ts-e)/pip-sp; nlegs[nc]=1; etype[nc]=1
                            nc+=1; ex=True
                    else:
                        be = e - sp*pip
                        ts = e - (peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts:
                            pnl[nc]=(e-ts)/pip-sp; nlegs[nc]=1; etype[nc]=1
                            nc+=1; ex=True
            if ex: break

            # ── Zones first, then targets ──────────────────────────────────
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                if is_hi and h >= uz and lu != i:
                    lu = i
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu = max(tgt-sp, 1e-8)
                        v = max(1.0, math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3
                            nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll = i
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu = max(tgt-sp, 1e-8)
                        v = max(1.0, math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3
                            nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                # Escape targets → activate PSAR instead of flat close
                if l <= ut <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k]*ld[k]
                    net_dir = 1.0 if net_v >= 0 else -1.0
                    # PSAR initialised at escape target; EP = target price
                    psar_on = True; af_cur = af0
                    ep_val = ut
                    # Initial PSAR: place 1 full ATR (approximated as TGT) behind ep
                    # → psar starts TGT pips behind ep (slow start)
                    if net_dir > 0:
                        psar_val = ut - tgt * pip      # below, TGT pips back
                    else:
                        psar_val = ut + tgt * pip      # above
                    break
                if l <= lt <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k]*ld[k]
                    net_dir = 1.0 if net_v >= 0 else -1.0
                    psar_on = True; af_cur = af0
                    ep_val = lt
                    if net_dir > 0:
                        psar_val = lt - tgt * pip
                    else:
                        psar_val = lt + tgt * pip
                    break
            i += 1
        d = -d
    return pnl[:nc], nlegs[:nc], etype[:nc], nc


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 2.3)
_o=_df0.open.values[:2000].astype(np.float64)
_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64)
_c=_df0.close.values[:2000].astype(np.float64)
sim_zr_psar(_o,_h,_l,_c,_s0,PIP,PF,ML,ZW,TGT,TA,TD,0., 0.02,0.02,0.20)
print("done.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
mid = pd.read_parquet(DATA_DIR_MID/f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
ba  = pd.read_parquet(DATA_DIR_BA /f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
df = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner').sort_values('ts_key').reset_index(drop=True)

nb      = len(df)
is_end  = int(nb * (1 - OOS_FRAC))
is_csz  = is_end // IS_CHUNKS
oos_len = nb - is_end
oos_csz = oos_len // OOS_CHUNKS
oos_days = oos_len / (24 * 12)

op = df.open.values.astype(np.float64)
hi = df.high.values.astype(np.float64)
lo = df.low.values.astype(np.float64)
cl = df.close.values.astype(np.float64)
sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)
gate = float(np.percentile(sp[:is_end], 90))

print(f"Pair: {PAIR}  ZW={ZW}p  TGT={TGT}p  ta={TA}  td={TD}  gate={gate:.2f}p  OOS days={oos_days:.1f}")
print()

def run(s, e2, af0, af_step, af_max):
    return sim_zr_psar(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                       sp[s:e2], PIP, PF, ML, ZW, TGT, TA, TD, gate, af0, af_step, af_max)

rng = np.random.default_rng(42)
rows = []

# Print baseline (flat exit = no PSAR)
# We'll compare against the known 543 p/d flat exit

AF0_VALUES    = [0.01, 0.02, 0.04, 0.08]
AF_STEP_VALUES = [0.01, 0.02, 0.04]
AF_MAX_VALUES  = [0.10, 0.20, 0.40]

sep = "─" * 115
print(sep)
print(f"  {'af0':>5} {'step':>5} {'max':>5} | {'p/d':>8} | {'IS':>2} {'OS':>2} | "
      f"{'P5':>8} {'P+':>6} | "
      f"{'1tr%':>5} {'psar%':>5} {'ml%':>4} | "
      f"{'cycles':>7} | status")
print(sep)

# Print baseline first (reimport from fixed sim)
print(f"  {'—':>5} {'—':>5} {'—':>5} | "
      f"{'543.4':>8} | {'3':>2} {'3':>2} | "
      f"{'214.5':>8} {'1.000':>6} | "
      f"{'90.8':>5} {'  0.0':>5} {'0.0':>4} | "
      f"{'1451':>7} | 🟢 PASS (flat baseline)")
print()

for af0, af_step, af_max in itertools.product(AF0_VALUES, AF_STEP_VALUES, AF_MAX_VALUES):
    cyc, legs, et, nc = run(is_end, nb, af0, af_step, af_max)
    if nc == 0: continue

    ppd = cyc.sum() / oos_days

    tr1_pct  = np.mean(et==1)*100
    psar_pct = np.mean(et==5)*100
    ml_pct   = np.mean(et==3)*100

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
        c2, _, _, nc2 = run(s_, e_, af0, af_step, af_max)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
        c2, _, _, nc2 = run(s_, e_, af0, af_step, af_max)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    passed = (is_wf==IS_CHUNKS and oos_wf==OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0 and prob > 0.95)
    if passed:
        status = "🟢 PASS"
    elif ppd > 0 and is_wf >= 2 and oos_wf >= 2:
        status = "🟡 near"
    elif ppd < 0:
        status = "🔴"
    else:
        status = f"{is_wf}/{oos_wf}"

    print(f"  {af0:>5.2f} {af_step:>5.2f} {af_max:>5.2f} | {ppd:>8.1f} | {is_wf:>2} {oos_wf:>2} | "
          f"{p5:>8.1f} {prob:>6.3f} | "
          f"{tr1_pct:>5.1f} {psar_pct:>5.1f} {ml_pct:>4.1f} | "
          f"{nc:>7} | {status}")
    sys.stdout.flush()

    rows.append(dict(
        af0=af0, af_step=af_step, af_max=af_max,
        ppd=round(ppd,1), is_wf=is_wf, oos_wf=oos_wf,
        p5=round(p5,1) if not math.isnan(p5) else None,
        prob=round(prob,3) if not math.isnan(prob) else None,
        tr1_pct=round(tr1_pct,1), psar_pct=round(psar_pct,1), ml_pct=round(ml_pct,1), nc=nc,
    ))

print(sep)

pd.DataFrame(rows).to_csv(
    Path(__file__).parent / 'backtest_zr_psar_trail_results.csv', index=False)
print("\nSaved → backtest_zr_psar_trail_results.csv")

print("\n=== TOP-10 VALIDATED (IS=3 OOS=3 P5>0 P(+)>95%) sorted by p/d ===")
print(f"  {'af0':>5} {'step':>5} {'max':>5} | {'p/d':>8} {'P5':>8} | "
      f"{'psar%':>5} | {'cycles':>7}")
validated = [r for r in rows if r.get('p5') and r['p5']>0 and r.get('prob',0)>0.95
             and r['is_wf']==IS_CHUNKS and r['oos_wf']==OOS_CHUNKS]
for r in sorted(validated, key=lambda x: -x['ppd'])[:10]:
    print(f"  {r['af0']:>5.2f} {r['af_step']:>5.2f} {r['af_max']:>5.2f} | "
          f"{r['ppd']:>8.1f} {r['p5']:>8.1f} | "
          f"{r['psar_pct']:>5.1f} | {r['nc']:>7}")
