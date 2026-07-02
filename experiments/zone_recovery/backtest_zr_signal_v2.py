"""
Signal-enhanced ZR entry — EUR_JPY ZW=50 TGT=25 ta=6 td=1 (fixed sim).

Baseline: pure random alternating (543 p/d, confirmed).

Enhancement: replace random direction with directional signal.
  filter mode: keep alternating; SKIP entry when signal contradicts direction
  only   mode: always enter in signal direction (breaks alternation)

Signals (all causal):
  h4_dir       H4 bar close > open  (+1 bull, -1 bear)
  h4_smaP      H4 price > SMA-P    (P = 5, 10, 20)
  h1_smaP      H1 price > SMA-P
  m5_smaP      M5 price > SMA-P

Output per signal×mode: p/d, IS/OOS WF, P5, P(+), entry%, trail%, esc%, avg_legs.
Full IS/OOS WF (3+3 chunks) + bootstrap MC (2000 samples).
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
OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000


# ── Signal computation (vectorised, causal) ────────────────────────────────

def sma(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).mean().values

def m5_to_h1_close(cl_m5):
    """Last M5 close of each H1 group."""
    n = len(cl_m5)
    h1_n = n // 12
    return np.array([cl_m5[i*12+11] for i in range(h1_n)])

def m5_to_h4_close_open(cl_m5, op_m5):
    """H4 OHLC via 48-bar M5 grouping: first open, last close."""
    n = len(cl_m5)
    h4_n = n // 48
    h4_op = np.array([op_m5[i*48]    for i in range(h4_n)])
    h4_cl = np.array([cl_m5[i*48+47] for i in range(h4_n)])
    return h4_op, h4_cl

def build_signals(op_m5, cl_m5):
    """Return dict of signal arrays (int8, +1/-1/0), aligned to M5 bars."""
    n = len(cl_m5)
    sigs = {}

    # ── H4 direction: last H4 bar close > open ──────────────────────────────
    h4_op, h4_cl = m5_to_h4_close_open(cl_m5, op_m5)
    h4_n = len(h4_cl)
    h4_dir_bar = np.where(h4_cl > h4_op, np.int8(1), np.int8(-1))
    # upsample: each H4 spans 48 M5 bars; shift 1 H4 bar to avoid lookahead
    sig = np.zeros(n, dtype=np.int8)
    for i in range(1, h4_n):
        sig[i*48:(i+1)*48] = h4_dir_bar[i-1]
    sigs['h4_dir'] = sig

    # ── H4 SMA-P ────────────────────────────────────────────────────────────
    for period in [5, 10, 20]:
        h4_sma = sma(h4_cl, period)
        # signal: H4 close above SMA → +1. Shift 1 H4 bar for causality.
        h4_sig = np.where(np.isnan(h4_sma), np.int8(0),
                          np.where(h4_cl > h4_sma, np.int8(1), np.int8(-1)))
        sig = np.zeros(n, dtype=np.int8)
        for i in range(1, h4_n):
            sig[i*48:(i+1)*48] = h4_sig[i-1]
        sigs[f'h4_sma{period}'] = sig

    # ── H1 SMA-P ────────────────────────────────────────────────────────────
    h1_cl = m5_to_h1_close(cl_m5)
    h1_n  = len(h1_cl)
    for period in [5, 10, 20]:
        h1_sma = sma(h1_cl, period)
        h1_sig = np.where(np.isnan(h1_sma), np.int8(0),
                          np.where(h1_cl > h1_sma, np.int8(1), np.int8(-1)))
        sig = np.zeros(n, dtype=np.int8)
        for i in range(1, h1_n):
            sig[i*12:(i+1)*12] = h1_sig[i-1]
        sigs[f'h1_sma{period}'] = sig

    # ── M5 SMA-P ────────────────────────────────────────────────────────────
    for period in [20, 50, 100]:
        m5_sma_val = sma(cl_m5, period)
        sig = np.where(np.isnan(m5_sma_val), np.int8(0),
                       np.where(cl_m5 > m5_sma_val, np.int8(1), np.int8(-1))).astype(np.int8)
        # shift 1 bar for causality
        sigs[f'm5_sma{period}'] = np.roll(sig, 1)
        sigs[f'm5_sma{period}'][0] = 0

    return sigs


# ── Numba ZR sim with signal gate ─────────────────────────────────────────

@njit
def sim_zr_sig(op, hi, lo, cl, sp_arr, sig_arr, pip, pf, ml, zw, tgt, ta, td, gate, sig_mode):
    """
    Fixed ZR sim + signal gate.
    sig_mode 0 = filter: alternating d, skip if sig contradicts
    sig_mode 1 = only:   enter in sig direction (breaks alternation)
    sig_arr[i]: +1 long, -1 short, 0 neutral
    Returns: pnl[], nlegs[], etype[], n_skip_sig, n_skip_spread
    etype: 1=trail_1leg  2=escape_flat  3=maxlegs
    """
    n = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc = 0; n_skip_sig = 0; n_skip_sp = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_e = sp_arr[i]
        if gate > 0 and sp_e > gate:
            n_skip_sp += 1; i += 1; continue
        s = sig_arr[i]
        if sig_mode == 0:   # filter
            if s != 0 and s != d:
                n_skip_sig += 1; d = -d; i += 1; continue
        else:               # only
            if s == 0:
                n_skip_sig += 1; i += 1; continue
            d = int(s)
        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])
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
                if l <= ut <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                    nc+=1; ex=True; break
                if l <= lt <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                    nc+=1; ex=True; break
            i += 1
        if sig_mode == 0: d = -d
    return pnl[:nc], nlegs[:nc], etype[:nc], n_skip_sig, n_skip_sp


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 2.3); _sig0 = np.ones(2000, dtype=np.int8)
_o=_df0.open.values[:2000].astype(np.float64)
_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64)
_c=_df0.close.values[:2000].astype(np.float64)
sim_zr_sig(_o,_h,_l,_c,_s0,_sig0,PIP,PF,ML,ZW,TGT,TA,TD,0.,0)
sim_zr_sig(_o,_h,_l,_c,_s0,_sig0,PIP,PF,ML,ZW,TGT,TA,TD,0.,1)
print("done.\n")

# ── Load data ──────────────────────────────────────────────────────────────
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

print(f"Pair: {PAIR}  ZW={ZW}p  TGT={TGT}p  ta={TA}  td={TD}  gate={gate:.2f}p")
print(f"IS bars: {is_end}  OOS days: {oos_days:.1f}")
print()

# ── Precompute signals on full dataset ────────────────────────────────────
print("Computing signals...", end=' ', flush=True)
all_sigs = build_signals(op, cl)
# Also build the "no_signal" baseline: alternating random (all bars = no signal gate)
all_sigs['random'] = np.zeros(nb, dtype=np.int8)
print(f"done. Signals: {list(all_sigs.keys())}")
print()

# ── Run experiment ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# header
sep = "─" * 120
print(sep)
print(f"  {'signal':>15} {'mode':>6} | {'p/d':>8} | {'IS':>2} {'OS':>2} | "
      f"{'P5':>8} {'P+':>6} | "
      f"{'ent%':>5} {'1tr%':>5} {'esc%':>5} {'ml%':>4} | "
      f"{'cycles':>7} | status")
print(sep)

rows = []

def run(s, e2, sig_arr, sig_mode):
    return sim_zr_sig(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                      sp[s:e2], sig_arr[s:e2], PIP, PF, ML, ZW, TGT, TA, TD, gate, sig_mode)

# Print baseline first (no signal gate)
for sig_name, sig_arr in all_sigs.items():
    modes = [('filter', 0), ('only', 1)] if sig_name != 'random' else [('random', 0)]
    for mode_label, sig_mode in modes:
        if sig_name == 'random' and sig_mode == 0:
            # pure random: use all-zero signal array in filter mode → no skips
            sig_mode_eff = 0
        else:
            sig_mode_eff = sig_mode

        cyc, legs, et, n_skip_sig, _ = run(is_end, nb, sig_arr, sig_mode_eff)
        nc = len(cyc)
        if nc == 0: continue

        total_bars = oos_len
        ent_pct = nc / (total_bars / 1) * 100  # rough: cycles per bar
        ppd = cyc.sum() / oos_days

        tr1_pct = np.mean(et==1)*100
        esc_pct = np.mean(et==2)*100
        ml_pct  = np.mean(et==3)*100

        is_wf = 0
        for ch in range(IS_CHUNKS):
            s_ = ch * is_csz
            e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
            c2, _, _, _, _ = run(s_, e_, sig_arr, sig_mode_eff)
            if len(c2) > 0 and c2.sum() > 0: is_wf += 1

        oos_wf = 0
        for ch in range(OOS_CHUNKS):
            s_ = is_end + ch * oos_csz
            e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
            c2, _, _, _, _ = run(s_, e_, sig_arr, sig_mode_eff)
            if len(c2) > 0 and c2.sum() > 0: oos_wf += 1

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

        vs_base = f"(baseline)" if sig_name == 'random' else ""

        print(f"  {sig_name:>15} {mode_label:>6} | {ppd:>8.1f} | {is_wf:>2} {oos_wf:>2} | "
              f"{p5:>8.1f} {prob:>6.3f} | "
              f"{ent_pct:>5.2f} {tr1_pct:>5.1f} {esc_pct:>5.1f} {ml_pct:>4.1f} | "
              f"{nc:>7} | {status} {vs_base}")
        sys.stdout.flush()

        rows.append(dict(
            signal=sig_name, mode=mode_label,
            ppd=round(ppd,1), is_wf=is_wf, oos_wf=oos_wf,
            p5=round(p5,1) if not math.isnan(p5) else None,
            prob=round(prob,3) if not math.isnan(prob) else None,
            ent_pct=round(ent_pct,3), tr1_pct=round(tr1_pct,1),
            esc_pct=round(esc_pct,1), ml_pct=round(ml_pct,1), nc=nc,
        ))

print(sep)

pd.DataFrame(rows).to_csv(
    Path(__file__).parent / 'backtest_zr_signal_v2_results.csv', index=False)
print("\nSaved results.")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== VALIDATED (IS=3 OOS=3 P5>0 P(+)>95%) sorted by p/d ===")
print(f"  {'signal':>15} {'mode':>6} | {'p/d':>8} {'P5':>8} | "
      f"{'1tr%':>5} {'esc%':>5} | {'cycles':>7}")
validated = [r for r in rows if r.get('p5') and r['p5']>0 and r.get('prob',0)>0.95
             and r['is_wf']==IS_CHUNKS and r['oos_wf']==OOS_CHUNKS]
for r in sorted(validated, key=lambda x: -x['ppd']):
    print(f"  {r['signal']:>15} {r['mode']:>6} | "
          f"{r['ppd']:>8.1f} {r['p5']:>8.1f} | "
          f"{r['tr1_pct']:>5.1f} {r['esc_pct']:>5.1f} | {r['nc']:>7}")
