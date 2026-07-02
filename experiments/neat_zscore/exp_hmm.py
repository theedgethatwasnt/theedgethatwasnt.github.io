#!/usr/bin/env python3
"""Experiment 2: HMM regime filter on f1+f10 signal."""
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from hmmlearn.hmm import GaussianHMM
from scipy import stats

S5_DIR = Path(__file__).resolve().parents[3] / "data" / "s5_ohlc"
PAIR, PIP, SPREAD = "EUR_JPY", 0.01, 2.3
IS_FRAC = 0.70

@njit(cache=True)
def compute_features(close, pop=1000, lb1=1, lb2=10):
    n = len(close)
    r1=np.zeros(n); r10=np.zeros(n)
    for i in range(lb1,n): r1[i]=(close[i]-close[i-lb1])/PIP
    for i in range(lb2,n): r10[i]=(close[i]-close[i-lb2])/(lb2*PIP)
    f1=np.zeros(n); f10=np.zeros(n); hp=np.pi/2.; start=pop+lb2
    s1=0.;s1q=0.;s10=0.;s10q=0.
    for k in range(start-pop,start):
        s1+=r1[k];s1q+=r1[k]*r1[k];s10+=r10[k];s10q+=r10[k]*r10[k]
    for i in range(start,n):
        if i>start:
            a1=r1[i-1];e1=r1[i-1-pop];s1+=a1-e1;s1q+=a1*a1-e1*e1
            a10=r10[i-1];e10=r10[i-1-pop];s10+=a10-e10;s10q+=a10*a10-e10*e10
        v1=s1q/pop-(s1/pop)**2;v10=s10q/pop-(s10/pop)**2
        std1=v1**0.5 if v1>1e-20 else 1e-10;std10=v10**0.5 if v10>1e-20 else 1e-10
        f1[i]=np.arctan(r1[i]/std1)/hp;f10[i]=np.arctan(r10[i]/std10)/hp
    return f1,f10,start

@njit(cache=True)
def simulate_filtered(signal, close, states, active_state, max_hold, thresh):
    pos=0;ep=0.;eb=0;pnl=0.;score_sum=0.;nt=0
    n=len(close)
    for i in range(n):
        s=signal[i]
        if pos!=0:
            pnl=(close[i]-ep)/PIP if pos==1 else (ep-close[i])/PIP
            fc=(i-eb)>=max_hold
            sc=(pos==1 and s>thresh) or (pos==-1 and s<-thresh) or \
               (pos==1 and s<-thresh*0.5) or (pos==-1 and s>thresh*0.5)
            exit_regime = (active_state>=0 and states[i]!=active_state)
            if fc or sc or exit_regime:
                score_sum+=pnl;nt+=1;pos=0;pnl=0.
        if pos==0 and (active_state<0 or states[i]==active_state):
            if s<-thresh:
                ep=close[i]+SPREAD*0.5*PIP;eb=i;pos=1;pnl=-SPREAD
            elif s>thresh:
                ep=close[i]-SPREAD*0.5*PIP;eb=i;pos=-1;pnl=-SPREAD
    return nt,score_sum/nt if nt>0 else 0.,score_sum

print("Loading + computing features...")
path = S5_DIR / "EUR_JPY_S5_BA.parquet"
df = pd.read_parquet(path); df.columns=[c.lower() for c in df.columns]
closes = df["bid_c"].values.astype(np.float64)[::12]
n=len(closes); n_is=int(n*IS_FRAC)
f1,f10,start = compute_features(closes)
print(f"  M5={n:,}  IS={n_is:,}  OOS={n-n_is:,}")

# JIT warmup
dummy_states = np.zeros(200, dtype=np.int32)
simulate_filtered(f1[:200], closes[:200], dummy_states, -1, 48, 0.5)

# Best threshold params from exp_threshold (or use defaults)
T, N, W1, W2 = 0.7, 10, 0.5, 1.0
signal = W1*f1 + W2*f10

# Baseline (no filter)
nt_b, mp_b, pnl_b = simulate_filtered(signal[:n_is], closes[:n_is],
    np.zeros(n_is,dtype=np.int32), -1, N, T)
nt_b_oos, mp_b_oos, pnl_b_oos = simulate_filtered(signal[n_is:], closes[n_is:],
    np.zeros(n-n_is,dtype=np.int32), -1, N, T)
print(f"\nBaseline (no regime filter): IS={pnl_b:+.1f} ({nt_b}t) | OOS={pnl_b_oos:+.1f} ({nt_b_oos}t, {mp_b_oos:+.3f}p/t)")

# Forward return for IC-per-state analysis
fwd10 = np.empty(n); fwd10[:]=np.nan
fwd10[:n-10] = (closes[10:] - closes[:n-10]) / PIP

print(f"\n{'='*70}")
for n_states in [2, 3, 4]:
    X_is = np.column_stack([f1[:n_is], f10[:n_is]])[start:]
    hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                      n_iter=200, random_state=42)
    hmm.fit(X_is)
    states_is_raw = hmm.predict(X_is)

    # Map raw states to full IS array (pad start with state 0)
    states_is = np.zeros(n_is, dtype=np.int32)
    states_is[start:] = states_is_raw.astype(np.int32)

    # Predict OOS states
    X_oos = np.column_stack([f1[n_is:], f10[n_is:]])
    states_oos_raw = hmm.predict(X_oos)
    states_oos = states_oos_raw.astype(np.int32)

    print(f"\nHMM n_states={n_states}:")
    # IC per state on IS
    state_ics = []
    for st in range(n_states):
        mask = (states_is == st) & np.isfinite(fwd10[:n_is])
        f10_s = f10[:n_is][mask]; fwd_s = fwd10[:n_is][mask]
        if len(f10_s) > 100:
            r, p = stats.spearmanr(f10_s, fwd_s)
            n_bars = mask.sum()
            t = r * np.sqrt(n_bars-2) / np.sqrt(1-r**2+1e-12)
            frac = n_bars / n_is
            sig = "🟢" if abs(t)>3 else ("🟡" if abs(t)>2 else "🔴")
            print(f"  {sig} State {st}: IC(f10)={r:+.4f}  t={t:+.2f}  n={n_bars:,} ({frac:.1%} of IS)")
            state_ics.append((abs(r), st, r))

    # Trade in each state and compare
    best_state = max(state_ics, key=lambda x: x[0])[1] if state_ics else 0
    for st in range(n_states):
        nt_f, mp_f, pnl_f = simulate_filtered(signal[:n_is], closes[:n_is], states_is, st, N, T)
        nt_f_oos, mp_f_oos, pnl_f_oos = simulate_filtered(signal[n_is:], closes[n_is:], states_oos, st, N, T)
        flag = "← strongest IC" if st==best_state else ""
        oos_flag = "🟢" if pnl_f_oos>0 else "🔴"
        print(f"  Trade state {st} only: IS={pnl_f:+.1f}({nt_f}t) | "
              f"{oos_flag}OOS={pnl_f_oos:+.1f}({nt_f_oos}t, {mp_f_oos:+.3f}p/t) {flag}")
