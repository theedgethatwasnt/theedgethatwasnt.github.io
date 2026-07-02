#!/usr/bin/env python3
"""
Post-Shock Retrace — ATR entry-gate + variable (ATR-scaled) TP validation.

Motivation (009 live diagnosis 2026-06-17): the fixed 20-pip TP failed when
M5 volatility compressed (entry ATR wk23 4.36 → wk24 3.38, p90 MFE 17.4→9.8,
0% reached 15p). Two fixes to test rigorously on the full multi-year data:
  1. ATR ENTRY GATE — only take shocks when M5-ATR(14) >= gate (skip low-vol
     regimes that can't produce the retrace).
  2. VARIABLE TP — TP = k × M5-ATR(14) at entry instead of a fixed pip value,
     so the target adapts to the regime.

Held at the validated base: thr=2.5, peak=44b, sd=3 (the deployed 009 config).
Same pipeline as backtest_post_shock_retrace.py: full OOS sweep → Walk-Forward
(3 OOS sub-chunks) → Monte-Carlo sign-shuffle. Baseline (gate=0, tp=20 fixed)
is included so the prior +56-70 p/d reproduces for reference.

GATE: do the gate/variable-TP configs BEAT the baseline OOS AND survive WF+MC?
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
S5_DIR  = PROJECT / "data" / "s5_ba"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP   = 0.01   # all JPY

# validated base (deployed 009)
THR, PEAK, SD = 2.5, 44, 3
# new dimensions
ATR_GATES = [0.0, 3.0, 4.0, 5.0]                          # 0 = off
TP_SPECS  = [(0, 10.0), (0, 15.0), (0, 20.0),             # fixed pips
             (1, 1.0), (1, 1.5), (1, 2.0)]                # k × ATR
HORIZON, Z_WINDOW, MAD_WIN = 600, 6, 2048
IS_FRAC, WF_CHUNKS, N_MC = 0.70, 3, 1000
WF_PASS_THRESH = 10   # of 12 (4 pairs × 3 chunks)
BARS_PER_DAY = 17280


def compute_shock_z(close, pip, w=6, mad_win=2048):
    n = len(close)
    vel = np.empty(n); vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n - w]) / pip
    vs = pd.Series(vel)
    rm = vs.rolling(mad_win, min_periods=50).median()
    rmad = (vs - rm).abs().rolling(mad_win, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


def compute_atr_m5_on_s5(high, low, close, pip, m5=60, period=14):
    """Causal M5-ATR(14) in pips broadcast to each S5 bar (uses the last
    COMPLETED M5 block, so no lookahead)."""
    n = len(close); nblk = n // m5
    H = high[:nblk * m5].reshape(nblk, m5).max(1)
    L = low[:nblk * m5].reshape(nblk, m5).min(1)
    C = close[:nblk * m5].reshape(nblk, m5)[:, -1]
    pc = np.roll(C, 1); pc[0] = C[0]
    tr = np.maximum.reduce([H - L, np.abs(H - pc), np.abs(L - pc)])
    atr_blk = pd.Series(tr).rolling(period).mean().values / pip
    use = np.concatenate([[np.nan], atr_blk[:-1]])          # block b → atr_blk[b-1]
    s5 = np.repeat(use, m5)
    out = np.empty(n); out[:len(s5)] = s5
    out[len(s5):] = use[-1] if len(use) else np.nan
    return np.nan_to_num(out, nan=0.0)


@njit
def sim_retrace(bid, ask, close, shock_flag, vel, atr, pip,
                peak_bars, stop_pips, atr_gate, tp_mode, tp_param, horizon):
    n = len(close); pb = int(peak_bars); max_ev = n // 10
    pnl_out = np.zeros(max_ev); tp_out = np.zeros(max_ev, dtype=np.int8)
    fl_out = np.zeros(max_ev, dtype=np.int8)
    ev = 0; cooldown = 0
    for t in range(Z_WINDOW, n - pb - int(horizon) - 2):
        if cooldown > 0:
            cooldown -= 1; continue
        if shock_flag[t] != 1:
            continue
        if atr_gate > 0.0 and atr[t] < atr_gate:
            continue
        tp_pips = tp_param if tp_mode == 0 else tp_param * atr[t]
        if tp_pips <= 0.0:
            continue
        d = np.int8(1) if vel[t] > 0 else np.int8(-1)
        peak_ask = ask[t]; peak_bid = bid[t]
        for k in range(1, pb + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]
        sp = (ask[t] - bid[t]) / pip
        ws = t + pb + 1; we = t + pb + int(horizon)
        if ws >= n or we >= n:
            continue
        fld = 0; tp = 0; fill = 0.0; pnl = 0.0
        if stop_pips == 0.0:
            fld = 1
            fill = bid[ws] if d == 1 else ask[ws]
            tp_level = fill - tp_pips * pip * d
            if d == 1 and bid[ws] <= tp_level: tp = 1; pnl = tp_pips - sp
            elif d == -1 and ask[ws] >= tp_level: tp = 1; pnl = tp_pips - sp
            ls = ws + 1
        else:
            entry = (peak_ask - stop_pips * pip) if d == 1 else (peak_bid + stop_pips * pip)
            tp_level = entry - tp_pips * pip * d
            ls = ws
        for j in range(ls, min(we + 1, n - 1)):
            lo = bid[j]; hi = ask[j]
            if stop_pips > 0.0 and fld == 0:
                if d == 1 and lo <= entry:
                    fld = 1; fill = entry
                    if lo <= tp_level: tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= entry:
                    fld = 1; fill = entry
                    if hi >= tp_level: tp = 1; pnl = tp_pips - sp
            if fld == 1 and tp == 0:
                if d == 1 and lo <= tp_level: tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= tp_level: tp = 1; pnl = tp_pips - sp
            if fld == 1 and tp == 1:
                break
        if fld == 1 and tp == 0:
            ej = min(we, n - 1)
            pnl = ((fill - bid[ej]) / pip - sp) if d == 1 else ((ask[ej] - fill) / pip - sp)
        elif fld == 0:
            pnl = 0.0
        if ev < max_ev:
            pnl_out[ev] = pnl; tp_out[ev] = tp; fl_out[ev] = fld; ev += 1
        cooldown = (pb + int(horizon)) // 2
    return pnl_out[:ev], tp_out[:ev], fl_out[:ev]


def mc_pvalue(arrs, days, actual, n_mc=1000):
    beat = 0
    for _ in range(n_mc):
        s = sum((a * np.where(np.random.random(len(a)) > 0.5, 1.0, -1.0)).sum() / d
                for a, d in zip(arrs, days))
        if s >= actual: beat += 1
    return beat / n_mc


def load_oos(pair):
    df = pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
    df = df.iloc[int(len(df) * IS_FRAC):].reset_index(drop=True)
    close = df["close"].values.astype(np.float64)
    bid = df["bid_c"].values.astype(np.float64); ask = df["ask_c"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64); low = df["low"].values.astype(np.float64)
    z, vel = compute_shock_z(close, PIP)
    atr = compute_atr_m5_on_s5(high, low, close, PIP)
    return close, bid, ask, vel, atr, (np.abs(z) > THR).astype(np.int8), len(df) / BARS_PER_DAY


# warmup
_b = np.ones(5000) * 150.0; _a = _b + 0.02; _c = _b.copy()
_v = np.zeros(5000); _v[100] = 1.0; _sf = np.zeros(5000, np.int8); _sf[100] = 1
_atr = np.ones(5000) * 5.0
sim_retrace(_b, _a, _c, _sf, _v, _atr, 0.01, 44, 3.0, 4.0, 0, 20.0, 600)
sim_retrace(_b, _a, _c, _sf, _v, _atr, 0.01, 44, 3.0, 0.0, 1, 1.5, 600)

CONFIGS = [(g, m, p) for g in ATR_GATES for (m, p) in TP_SPECS]


def tp_label(m, p):
    return f"fix{p:.0f}p" if m == 0 else f"ATR×{p:.1f}"


print("=" * 78)
print("PHASE 1 — Full OOS sweep (base thr=2.5 peak=44b sd=3) × ATR-gate × TP-mode")
print("=" * 78)
oos = {}
rows = {}
for pair in PAIRS:
    print(f"  loading {pair} …")
    oos[pair] = load_oos(pair)
    close, bid, ask, vel, atr, sf, days = oos[pair]
    for (g, m, p) in CONFIGS:
        pnl, tph, fl = sim_retrace(bid, ask, close, sf, vel, atr, PIP,
                                   float(PEAK), float(SD), g, m, p, HORIZON)
        rows.setdefault((g, m, p), {})[pair] = (pnl.sum() / days, len(pnl),
                                                fl.sum(), tph.sum())
print()
agg = []
for (g, m, p), d in rows.items():
    ppd = sum(v[0] for v in d.values())
    ev = sum(v[1] for v in d.values())
    fr = sum(v[2] for v in d.values()) / max(ev, 1) * 100
    tr = sum(v[3] for v in d.values()) / max(sum(v[2] for v in d.values()), 1) * 100
    agg.append(dict(gate=g, tp_mode=m, tp_param=p, ppd=ppd, ev=ev, fill=fr, tp_rate=tr))
A = pd.DataFrame(agg).sort_values("ppd", ascending=False)
print(f"{'gate':>5} {'TP':>9} {'ppd':>9} {'ev':>7} {'fill%':>6} {'tp%':>6}")
for _, r in A.iterrows():
    base = "  ← BASELINE" if (r.gate == 0 and r.tp_mode == 0 and r.tp_param == 20) else ""
    print(f"{r.gate:>5.0f} {tp_label(r.tp_mode, r.tp_param):>9} {r.ppd:>+9.1f} "
          f"{int(r.ev):>7} {r.fill:>6.1f} {r.tp_rate:>6.1f}{base}")

# ── WF + MC on positive configs ────────────────────────────────────────────
pos = A[A.ppd > 0]
print(f"\n{'='*78}\nPHASE 2/3 — WF (3 OOS sub-chunks) + MC on {len(pos)} positive configs\n{'='*78}")
final = []
for _, cfg in pos.iterrows():
    g, m, p = cfg.gate, int(cfg.tp_mode), cfg.tp_param
    wf_pos = 0; wf_tot = 0; arrs = []; daysl = []
    for pair in PAIRS:
        close, bid, ask, vel, atr, sf, days = oos[pair]
        n = len(close); cs = n // WF_CHUNKS
        for ch in range(WF_CHUNKS):
            s = ch * cs; e = (ch + 1) * cs if ch < WF_CHUNKS - 1 else n
            chd = (e - s) / BARS_PER_DAY
            pnl, _, _ = sim_retrace(bid[s:e], ask[s:e], close[s:e], sf[s:e],
                                    vel[s:e], atr[s:e], PIP, float(PEAK), float(SD),
                                    g, m, p, HORIZON)
            wf_tot += 1
            if pnl.sum() / chd > 0: wf_pos += 1
        pnl, _, _ = sim_retrace(bid, ask, close, sf, vel, atr, PIP, float(PEAK),
                                float(SD), g, m, p, HORIZON)
        arrs.append(pnl); daysl.append(days)
    actual = sum(a.sum() / d for a, d in zip(arrs, daysl))
    mc_p = mc_pvalue(arrs, daysl, actual, N_MC) if wf_pos >= WF_PASS_THRESH else 1.0
    final.append(dict(gate=g, tp=tp_label(m, p), ppd=cfg.ppd, wf=f"{wf_pos}/{wf_tot}",
                      wf_pass=wf_pos >= WF_PASS_THRESH, mc_p=mc_p))

F = pd.DataFrame(final).sort_values("ppd", ascending=False)
F.to_csv(RESULTS / "retrace_atr_validation.csv", index=False)
print(f"\n{'gate':>5} {'TP':>9} {'ppd':>9} {'WF':>7} {'mc_p':>7} {'DEPLOYABLE':>11}")
for _, r in F.iterrows():
    dep = "🟢 YES" if (r.wf_pass and r.mc_p < 0.05) else ""
    print(f"{r.gate:>5.0f} {r.tp:>9} {r.ppd:>+9.1f} {r.wf:>7} {r.mc_p:>7.4f} {dep:>11}")

base = A[(A.gate == 0) & (A.tp_mode == 0) & (A.tp_param == 20)]
base_ppd = float(base.ppd.iloc[0]) if len(base) else float("nan")
dep = F[(F.wf_pass) & (F.mc_p < 0.05)]
print(f"\nBaseline (gate0/fix20) OOS ppd = {base_ppd:+.1f}")
if len(dep):
    best = dep.iloc[0]
    print(f"Best deployable: gate={best.gate:.0f} {best.tp} → {best.ppd:+.1f} p/d "
          f"(WF {best.wf}, mc_p={best.mc_p:.4f})  Δ vs baseline = {best.ppd-base_ppd:+.1f}")
else:
    print("No gate/variable-TP config is deployable (WF+MC) — fixes do NOT validate.")
print(f"\nResults → {RESULTS}/retrace_atr_validation.csv")
