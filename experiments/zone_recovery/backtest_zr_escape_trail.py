"""
Escape Trail experiment — EUR_JPY ZW=50 TGT=25 ta=6 td=1.

Base: flat-close all legs when escape target is hit (P&L ≥ 0).
Test: when escape target crossed, activate trailing stop on combined position.
      → Close all legs when trail fires. Trail direction = sign(net_vol).

Parameters swept:
  ts_dist ∈ [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50] pips
  ts_dist=0 → flat-exit (baseline)

Output: p/d, P5, P(+), IS/OOS WF, exit breakdown per ts_dist.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

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
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000

TS_DIST_VALUES = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]


@njit
def sim_zr_et(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td, gate, ts_dist):
    """
    ZR sim with optional escape trail.
    ts_dist=0 → standard flat exit at escape target.
    ts_dist>0 → when escape target is crossed, activate trailing stop
                 (trail distance = ts_dist pips, direction = sign(net_vol)).
                 Close all when trail fires.

    etype: 1=trail_1leg  2=escape_flat  3=maxlegs  4=escape_trail
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
        # escape trail state
        et_active = False; et_peak = 0.0; et_dir = 0.0
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # ── Escape trail tracking (when activated) ──────────────────────────
            if et_active:
                if et_dir > 0:
                    if h > et_peak: et_peak = h
                    ts_p = et_peak - ts_dist * pip
                    if l <= ts_p:
                        # close all at ts_p
                        net = 0.0; tv = 0.0
                        for k in range(nl): net += lv[k]*ld[k]*(ts_p-lp[k])/pip; tv += lv[k]
                        pnl[nc] = net - tv*sp; nlegs[nc] = nl; etype[nc] = 4
                        nc += 1; ex = True
                else:
                    if l < et_peak: et_peak = l
                    ts_p = et_peak + ts_dist * pip
                    if h >= ts_p:
                        net = 0.0; tv = 0.0
                        for k in range(nl): net += lv[k]*ld[k]*(ts_p-lp[k])/pip; tv += lv[k]
                        pnl[nc] = net - tv*sp; nlegs[nc] = nl; etype[nc] = 4
                        nc += 1; ex = True
                if ex: break
                i += 1
                continue

            # ── 1-leg trail ──────────────────────────────────────────────────────
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

            # ── Zones first, then targets ─────────────────────────────────────────
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
                # Escape targets
                if l <= ut <= h:
                    if ts_dist <= 0:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                        nc+=1; ex=True; break
                    else:
                        # activate escape trail
                        net_v = 0.0
                        for k in range(nl): net_v += lv[k]*ld[k]
                        et_dir = 1.0 if net_v >= 0 else -1.0
                        et_peak = ut
                        et_active = True
                        break
                if l <= lt <= h:
                    if ts_dist <= 0:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                        nc+=1; ex=True; break
                    else:
                        net_v = 0.0
                        for k in range(nl): net_v += lv[k]*ld[k]
                        et_dir = 1.0 if net_v >= 0 else -1.0
                        et_peak = lt
                        et_active = True
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
sim_zr_et(_o,_h,_l,_c,_s0,PIP,PF,ML,ZW,TGT,TA,TD,0.,0.)
sim_zr_et(_o,_h,_l,_c,_s0,PIP,PF,ML,ZW,TGT,TA,TD,0.,5.)
print("done.\n")

# ── Load full data ────────────────────────────────────────────────────────────
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

print(f"Pair: {PAIR}  ZW={ZW}p  TGT={TGT}p  ta={TA}  td={TD}")
print(f"gate={gate:.2f}p  OOS days={oos_days:.1f}")
print()

sep = "─" * 110
hdr = (f"  {'ts_dist':>7} | {'p/d':>8} | {'IS':>2} {'OS':>2} | "
       f"{'P5':>8} {'P+':>6} | "
       f"{'1tr%':>5} {'flat%':>5} {'etrl%':>5} {'ml%':>4} | "
       f"{'cycles':>7} | note")
print(sep); print(hdr); print(sep)

rng = np.random.default_rng(42)

def run(s, e2, ts_d):
    return sim_zr_et(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                     sp[s:e2], PIP, PF, ML, ZW, TGT, TA, TD, gate, ts_d)

for ts_d in TS_DIST_VALUES:
    cyc, legs, et, nc = run(is_end, nb, ts_d)
    if nc == 0: continue

    ppd = cyc.sum() / oos_days

    tr1_pct  = np.mean(et==1)*100
    flat_pct = np.mean(et==2)*100
    etrl_pct = np.mean(et==4)*100
    ml_pct   = np.mean(et==3)*100

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
        c2, _, _, nc2 = run(s_, e_, ts_d)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
        c2, _, _, nc2 = run(s_, e_, ts_d)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    passed = (is_wf==IS_CHUNKS and oos_wf==OOS_CHUNKS
              and not math.isnan(p5) and p5>0 and prob>0.95)
    note = "🟢 PASS" if passed else ("🟡 near" if (ppd>0 and is_wf>=2 and oos_wf>=2) else "🔴")
    tag  = "(baseline)" if ts_d == 0 else ""

    print(f"  {ts_d:>7.0f} | {ppd:>8.1f} | {is_wf:>2} {oos_wf:>2} | "
          f"{p5:>8.1f} {prob:>6.3f} | "
          f"{tr1_pct:>5.1f} {flat_pct:>5.1f} {etrl_pct:>5.1f} {ml_pct:>4.1f} | "
          f"{nc:>7} | {note} {tag}")
    sys.stdout.flush()

print(sep)
print()
print("Legend: ts_dist=0 → flat exit at escape target (baseline)")
print("        ts_dist>0 → trail activates at escape target, fires when price")
print("                    retreats ts_dist pips from peak after target crossing")
print("        1tr%=1-leg trail exits, flat%=flat escape exits, etrl%=escape trail exits")
