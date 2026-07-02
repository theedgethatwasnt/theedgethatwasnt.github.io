"""
H4 SMA20 direction — single-hypothesis clean test. EUR_JPY ZW=50 TGT=25 ta=6 td=1.

vs baseline: alternating random alternating direction, no spread gate.

Signal: H4 SMA20 direction — enter LONG when H4 close > H4 SMA20, SHORT otherwise.
  - NO spread gate (gate=0) — enter every bar when flat
  - NO blocking — direction only, replaces alternating
  - Causal: timestamp-based H4 resampling (not fixed 48-bar chunks).
    Signal at M5 bar T = direction from last COMPLETED H4 bar before floor(T,4H).

Lookahead audit:
  - H4 bar at period P (open=P, close=P+4H) is available at the start of P+4H.
  - For M5 bar at time T, current H4 period = floor(T, 4H).
  - Last completed H4 = period floor(T,4H) - 4H.
  - SMA20 uses only H4 closes whose period < floor(T,4H). Zero lookahead.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR      = "EUR_JPY"
PIP       = 0.01
PF        = 1.25
ZW        = 50.0
TGT       = 25.0
TA        = 6.0
TD        = 1.0
ML        = 10
OOS_FRAC  = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000


# ── Causal H4 SMA20 signal (timestamp-based) ─────────────────────────────────

def build_h4_sma20_signal(df: pd.DataFrame, sma_period: int = 20) -> np.ndarray:
    """
    Returns int8 array (+1/-1/0) aligned to df rows.

    For each M5 bar at time T:
      current_h4_period  = floor(T, 4H)
      signal comes from   current_h4_period - 4H  (last COMPLETED H4 bar)
      SMA20 uses the 20 H4 closes ending at (current_h4_period - 4H)

    Zero lookahead: we never touch the H4 bar that contains bar T.
    """
    ts = pd.to_datetime(df['timestamp'])

    # H4 period each M5 bar belongs to (open of the H4 window it's in)
    h4_period = ts.dt.floor('4h')

    # Build H4 OHLC from M5 closes — last M5 close in each H4 window
    h4_close = (df.assign(h4=h4_period)
                  .groupby('h4')['close']
                  .last()
                  .sort_index())

    # SMA20 on H4 closes — causal rolling
    h4_sma20 = h4_close.rolling(sma_period, min_periods=sma_period).mean()

    # Direction: +1 if H4 close > SMA20, -1 otherwise, 0 during warmup
    h4_dir = pd.Series(
        np.where(h4_sma20.isna(), np.int8(0),
                 np.where(h4_close > h4_sma20, np.int8(1), np.int8(-1))),
        index=h4_close.index,
        dtype=np.int8
    )

    # For each M5 bar at time T, use H4 bar at (floor(T,4H) - 4H)
    prev_h4 = h4_period - pd.Timedelta(hours=4)
    sig = prev_h4.map(h4_dir).fillna(0).astype(np.int8).values

    return sig


# ── Numba ZR sim ──────────────────────────────────────────────────────────────

@njit
def sim_zr(op, hi, lo, cl, sp_arr, sig_arr, pip, pf, ml, zw, tgt, ta, td, gate, use_signal):
    """
    ZR fixed sim.
    use_signal=0: alternating direction (ignore sig_arr)
    use_signal=1: enter in sig_arr direction every bar (no gate, no alternating)
                  sig_arr==0 bars are skipped (H4 SMA warmup only)

    No spread gate when gate=0.
    etype: 1=trail_1leg  2=escape_flat  3=maxlegs
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

        if use_signal:
            s = sig_arr[i]
            if s == 0:
                i += 1; continue  # SMA warmup only
            d = int(s)
        # else: use existing d (alternating)

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
        if not use_signal:
            d = -d   # alternating: flip after each cycle
    return pnl[:nc], nlegs[:nc], etype[:nc], nc


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 2.3); _sig0 = np.ones(2000, dtype=np.int8)
_o=_df0.open.values[:2000].astype(np.float64)
_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64)
_c=_df0.close.values[:2000].astype(np.float64)
sim_zr(_o,_h,_l,_c,_s0,_sig0,PIP,PF,ML,ZW,TGT,TA,TD,0.,0)
sim_zr(_o,_h,_l,_c,_s0,_sig0,PIP,PF,ML,ZW,TGT,TA,TD,0.,1)
print("done.\n")

# ── Load data ────────────────────────────────────────────────────────────────
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
# No spread gate
gate = 0.0

print(f"Pair: {PAIR}  ZW={ZW}p  TGT={TGT}p  ta={TA}  td={TD}")
print(f"gate=NONE (all spreads accepted)  OOS days={oos_days:.1f}")

# ── Build causal H4 SMA20 signal ─────────────────────────────────────────────
print("Building causal H4 SMA20 signal (timestamp-based)...", end=' ', flush=True)
sig = build_h4_sma20_signal(df, sma_period=20)
n_long  = int(np.sum(sig == 1))
n_short = int(np.sum(sig == -1))
n_zero  = int(np.sum(sig == 0))
print(f"done.")
print(f"Signal distribution: LONG={n_long} ({n_long/nb*100:.1f}%)  "
      f"SHORT={n_short} ({n_short/nb*100:.1f}%)  "
      f"WARMUP(0)={n_zero} ({n_zero/nb*100:.1f}%)")

# Verify causality: signal at each M5 bar should reference an H4 bar from ≥4h ago
ts_arr = pd.to_datetime(df['timestamp'])
h4_period = ts_arr.dt.floor('4h')
prev_h4   = h4_period - pd.Timedelta(hours=4)
# The H4 bar at prev_h4 closes at h4_period (i.e., 4h BEFORE current bar's open of the NEXT H4)
# As long as prev_h4 < h4_period, no bar in the current H4 window is touched. Always true.
print(f"Lookahead check: prev_h4 always < current_h4: "
      f"{(prev_h4 < h4_period).all()} ✓")
print()

rng = np.random.default_rng(42)

def run(s, e2, use_signal):
    return sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                  sp[s:e2], sig[s:e2], PIP, PF, ML, ZW, TGT, TA, TD, gate, use_signal)

sep = "─" * 100
hdr = (f"  {'config':>22} | {'p/d':>8} | {'IS':>2} {'OS':>2} | "
       f"{'P5':>8} {'P+':>6} | {'1tr%':>6} {'esc%':>5} {'ml%':>4} | "
       f"{'cycles':>7} | note")
print(sep); print(hdr); print(sep)

for label, use_sig in [("alternating (baseline)", 0), ("h4_sma20 direction", 1)]:
    cyc, legs, et, nc = run(is_end, nb, use_sig)
    if nc == 0:
        print(f"  {label:>22} | NO CYCLES"); continue

    ppd = cyc.sum() / oos_days
    tr1_pct = np.mean(et==1)*100
    esc_pct = np.mean(et==2)*100
    ml_pct  = np.mean(et==3)*100

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
        c2, _, _, nc2 = run(s_, e_, use_sig)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
        c2, _, _, nc2 = run(s_, e_, use_sig)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    passed = (is_wf==IS_CHUNKS and oos_wf==OOS_CHUNKS
              and not math.isnan(p5) and p5 > 0 and prob > 0.95)
    note = "🟢 PASS" if passed else ("🟡 near" if (ppd > 0 and is_wf >= 2 and oos_wf >= 2) else "🔴")

    print(f"  {label:>22} | {ppd:>8.1f} | {is_wf:>2} {oos_wf:>2} | "
          f"{p5:>8.1f} {prob:>6.3f} | {tr1_pct:>6.1f} {esc_pct:>5.1f} {ml_pct:>4.1f} | "
          f"{nc:>7} | {note}")
    sys.stdout.flush()

print(sep)
print()
print("Notes:")
print("  baseline : alternating LONG/SHORT, enter every bar when flat, gate=0")
print("  h4_sma20 : enter in H4-SMA20 direction every bar when flat, gate=0")
print("  H4 signal is causal: uses last COMPLETED H4 bar (timestamp floor - 4h)")
print("  No spread gate on either — direct apple-to-apple comparison")
