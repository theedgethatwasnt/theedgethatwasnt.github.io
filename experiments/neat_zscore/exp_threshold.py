#!/usr/bin/env python3
"""Experiment 1: Threshold grid on combined f1+f10 signal."""
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from itertools import product

S5_DIR = Path(__file__).resolve().parents[3] / "data" / "s5_ohlc"
PAIR, PIP, SPREAD = "EUR_JPY", 0.01, 2.3
IS_FRAC = 0.70

@njit(cache=True)
def compute_features(close, pop=1000, lb1=1, lb2=10):
    n = len(close)
    r1 = np.zeros(n); r10 = np.zeros(n)
    for i in range(lb1, n): r1[i]  = (close[i]-close[i-lb1]) / PIP
    for i in range(lb2, n): r10[i] = (close[i]-close[i-lb2]) / (lb2*PIP)
    f1 = np.zeros(n); f10 = np.zeros(n)
    hp = np.pi/2.0; start = pop+lb2
    s1=0.;s1q=0.;s10=0.;s10q=0.
    for k in range(start-pop, start):
        s1+=r1[k]; s1q+=r1[k]*r1[k]; s10+=r10[k]; s10q+=r10[k]*r10[k]
    for i in range(start, n):
        if i>start:
            a1=r1[i-1]; e1=r1[i-1-pop]; s1+=a1-e1; s1q+=a1*a1-e1*e1
            a10=r10[i-1]; e10=r10[i-1-pop]; s10+=a10-e10; s10q+=a10*a10-e10*e10
        v1=s1q/pop-(s1/pop)**2; v10=s10q/pop-(s10/pop)**2
        std1=v1**0.5 if v1>1e-20 else 1e-10; std10=v10**0.5 if v10>1e-20 else 1e-10
        f1[i]=np.arctan(r1[i]/std1)/hp; f10[i]=np.arctan(r10[i]/std10)/hp
    return f1, f10, start

@njit(cache=True)
def simulate(signal, close, max_hold, thresh):
    """signal > thresh → SELL (mean-rev); signal < -thresh → BUY."""
    pos=0; ep=0.; eb=0; pnl=0.; score_sum=0.; nt=0
    n = len(close)
    for i in range(n):
        s = signal[i]
        if pos != 0:
            pnl = (close[i]-ep)/PIP if pos==1 else (ep-close[i])/PIP
            fc = (i-eb) >= max_hold
            sc = (pos==1 and s > thresh) or (pos==-1 and s < -thresh) or \
                 (pos==1 and s < -thresh*0.5) or (pos==-1 and s > thresh*0.5)
            if fc or sc:
                score_sum += pnl; nt += 1; pos=0; pnl=0.
        if pos==0:
            if s < -thresh:   # BUY (mean-rev: big down → bounce)
                ep = close[i] + SPREAD*0.5*PIP; eb=i; pos=1; pnl=-SPREAD
            elif s > thresh:  # SELL
                ep = close[i] - SPREAD*0.5*PIP; eb=i; pos=-1; pnl=-SPREAD
    return nt, score_sum/nt if nt>0 else 0., score_sum

print("Loading + computing features...")
path = S5_DIR / "EUR_JPY_S5_BA.parquet"
df = pd.read_parquet(path); df.columns=[c.lower() for c in df.columns]
closes = df["bid_c"].values.astype(np.float64)[::12]
n = len(closes); n_is = int(n*IS_FRAC)
f1, f10, start = compute_features(closes)
print(f"  M5={n:,}  IS={n_is:,}  OOS={n-n_is:,}")

# JIT warmup
simulate(f1[:200]+f10[:200], closes[:200], 48, 0.5)

combos = []
for w1, w2, T, N in product([0.5,1.0], [0.5,1.0], [0.3,0.5,0.7,0.9,1.1], [5,10,20,48]):
    sig_is  = w1*f1[:n_is]  + w2*f10[:n_is]
    sig_oos = w1*f1[n_is:]  + w2*f10[n_is:]
    cl_is   = closes[:n_is]; cl_oos = closes[n_is:]
    nt_is,  mp_is,  pnl_is  = simulate(sig_is,  cl_is,  N, T)
    nt_oos, mp_oos, pnl_oos = simulate(sig_oos, cl_oos, N, T)
    if nt_is > 20:
        combos.append((pnl_is, pnl_oos, nt_is, nt_oos, mp_is, mp_oos, w1, w2, T, N))

combos.sort(reverse=True)

print(f"\n{'='*80}")
print(f"Top IS results (sorted by IS pnl):")
print(f"{'w1':>4} {'w2':>4} {'T':>5} {'N':>4} | {'IS_pnl':>8} {'IS_nt':>6} {'IS_mean':>8} | {'OOS_pnl':>8} {'OOS_nt':>6} {'OOS_mean':>8}")
print("-"*80)
for row in combos[:20]:
    pnl_is,pnl_oos,nt_is,nt_oos,mp_is,mp_oos,w1,w2,T,N = row
    oos_flag = "🟢" if pnl_oos>0 else "🔴"
    print(f"{w1:>4.1f} {w2:>4.1f} {T:>5.1f} {N:>4d} | {pnl_is:>8.1f} {nt_is:>6d} {mp_is:>8.3f}p | "
          f"{oos_flag}{pnl_oos:>7.1f} {nt_oos:>6d} {mp_oos:>8.3f}p")

# Best OOS
combos_oos = sorted(combos, key=lambda x: x[1], reverse=True)
print(f"\nTop OOS results:")
print("-"*80)
for row in combos_oos[:10]:
    pnl_is,pnl_oos,nt_is,nt_oos,mp_is,mp_oos,w1,w2,T,N = row
    print(f"w1={w1} w2={w2} T={T} N={N} | IS={pnl_is:+.1f} ({nt_is}t) | OOS={pnl_oos:+.1f} ({nt_oos}t, {mp_oos:+.3f}p/t)")
