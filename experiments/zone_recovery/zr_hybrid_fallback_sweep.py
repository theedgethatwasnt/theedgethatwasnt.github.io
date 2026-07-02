"""
Hybrid ZR: random fallback when flat + P&F signal override.

Hypothesis: Waiting for P&F reversals leaves the system idle for long stretches
(b10r3 = ~5h avg between entries). A random-fallback entry when flat captures
that dead time. When a genuine P&F signal arrives during a random cycle:
  - Same direction → confirm (continue the cycle, upgrade from random)
  - Opposite direction → override if single-leg AND unrealized >= threshold

This tests whether filling idle time with random entries improves OOS p/d
versus pure P&F timing, at various fallback intervals (n_fallback bars) and
override thresholds (in pips unrealized at time of signal).

Modes tested per pair × config:
  Pure P&F ALT  (baseline — current live logic)
  Hybrid ALT n_fallback={1,5,12,24,48} × override not applicable (ALT dirs never conflict)
  Pure P&F COL
  Hybrid COL n_fallback={1,5,12,24,48} × override_thresh={-5,-2,0} pips

IS=70%, OOS=30%, WF=3 chunks. Pairs: CHF_JPY, NZD_JPY, AUD_JPY.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
OUT_PATH  = Path('/path/to/projects/fx-core/research/experiments/zone_recovery/zr_hybrid_results.csv')

SPREAD = 1.4; MAX_LEGS = 10; PF = 1.25
OOS_FRAC  = 0.30
WF_CHUNKS = 3

PIP_MAP = {
    "CHF_JPY": 0.01, "NZD_JPY": 0.01, "AUD_JPY": 0.01, "EUR_JPY": 0.01,
}

# Best P&F configs per pair (from sweep)
PAIR_CFG = {
    "CHF_JPY": dict(zw=40.0, tgt=20.0, ta=5.0, td=3.0, box_pips=10, rev=3.0),
    "NZD_JPY": dict(zw=40.0, tgt=20.0, ta=5.0, td=3.0, box_pips=5,  rev=4.0),
    "AUD_JPY": dict(zw=50.0, tgt=25.0, ta=5.0, td=3.0, box_pips=15, rev=2.0),
    "EUR_JPY": dict(zw=50.0, tgt=25.0, ta=5.0, td=3.0, box_pips=5,  rev=1.5),
}

N_FALLBACKS     = [1, 5, 12, 24, 48]   # M5 bars: 5min, 25min, 1h, 2h, 4h
OVERRIDE_THRESH = [-5.0, -2.0, 0.0]    # pips — only for COL mode conflicts


# ─── Hybrid ZR simulator ─────────────────────────────────────────────────────

@njit
def sim_zr_hybrid(op, hi, lo, cl,
                  box_size, rev_thresh_mult,
                  dir_mode,          # 0=alt, 1=col
                  pip, spread, pf, ml, zw, tgt, ta, td,
                  n_fallback,        # bars before random fallback (999999=pure P&F)
                  override_thresh):  # pips unrealized to allow kill (ignored for ALT)
    """
    Bar-by-bar ZR simulation with inline P&F state + random fallback.

    When flat:
      - If P&F just reversed → enter immediately (signal entry, is_random=False)
      - Elif bars_flat >= n_fallback → enter in next_dir (random fallback, is_random=True)

    When in random cycle and P&F reversal fires:
      ALT mode: direction is always next_d → P&F only resets the alternation sequence.
                No conflict possible (both use same alternation). Upgrade to confirmed.
      COL mode: sig_dir = pnf column direction. If sig_dir != cycle_dir AND nl==1 AND
                unrealized >= override_thresh (pips) → close + reopen in sig_dir.

    Returns (total_pips, n_cycles, n_trail, n_signal_entries, n_random_entries, n_overrides)
    """
    n = len(cl)
    total = 0.0; nc = 0; n_trail = 0; n_sig = 0; n_rnd = 0; n_ovr = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)

    # Direction alternator (flipped on each cycle EXIT)
    d = np.int8(1)

    # Cycle state
    in_cycle     = False
    is_random    = False
    nl           = 0
    direction    = 0.0
    e            = 0.0       # entry price (1st leg)
    uz=0.0; lz=0.0; ut=0.0; lt=0.0
    lu=-1; ll=-1
    peak_mfe=0.0; trail_on=False
    bars_flat    = 0

    # Inline P&F state
    rev_thresh = rev_thresh_mult * box_size
    pnf_col    = np.int8(1)
    pnf_ext    = (hi[0] + lo[0]) * 0.5

    for i in range(1, n):
        h = hi[i]; l = lo[i]; c = cl[i]; bull = c >= op[i]

        # ── Update P&F ────────────────────────────────────────────────────
        just_reversed = False
        new_col       = np.int8(0)
        if pnf_col == 1:
            if h >= pnf_ext + box_size:
                pnf_ext += math.floor((h - pnf_ext) / box_size) * box_size
            elif l <= pnf_ext - rev_thresh:
                pnf_col = np.int8(-1); pnf_ext -= box_size
                just_reversed = True; new_col = np.int8(-1)
        else:
            if l <= pnf_ext - box_size:
                pnf_ext -= math.floor((pnf_ext - l) / box_size) * box_size
            elif h >= pnf_ext + rev_thresh:
                pnf_col = np.int8(1); pnf_ext += box_size
                just_reversed = True; new_col = np.int8(1)

        # ── No active cycle ───────────────────────────────────────────────
        if not in_cycle:
            bars_flat += 1
            enter = False
            entry_dir = float(d)
            entry_random = True

            if just_reversed:
                entry_dir    = float(d) if dir_mode == 0 else float(new_col)
                enter        = True
                entry_random = False
            elif bars_flat >= n_fallback:
                entry_dir    = float(d)   # random fallback always alternates
                enter        = True
                entry_random = True

            if enter:
                in_cycle  = True
                is_random = entry_random
                direction = entry_dir
                e = c
                if direction == 1.0:
                    uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
                else:
                    lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
                for k in range(ml): lv[k]=0.0; ld[k]=0.0; lp[k]=0.0
                lv[0]=1.0; ld[0]=direction; lp[0]=e
                nl=1; lu=-1; ll=-1; peak_mfe=0.0; trail_on=False; bars_flat=0
                if entry_random: n_rnd += 1
                else:            n_sig += 1

        # ── Active cycle ──────────────────────────────────────────────────
        else:
            # Signal override check (COL mode only; ALT never conflicts)
            if just_reversed and is_random:
                sig_dir = float(d) if dir_mode == 0 else float(new_col)

                if dir_mode == 0:
                    # ALT: signal direction = next_d = d (not yet flipped this cycle).
                    # Since we entered in d (pre-flip) and d is now -d, sig_dir = float(d) = -entry_dir.
                    # So it's the "opposite" — but for ALT this just means P&F wants next entry.
                    # We simply upgrade to confirmed (no kill — ALT conflicts are artificial).
                    is_random = False
                else:
                    # COL: real directional signal
                    if sig_dir == direction:
                        is_random = False   # confirm and continue
                    elif nl == 1:
                        # opposite direction, single leg — evaluate override
                        cur_upl = (c - e) / pip if direction == 1.0 else (e - c) / pip
                        if cur_upl >= override_thresh:
                            # Close current, realise P&L
                            net = ld[0] * (c - lp[0]) / pip
                            total += net - spread; nc += 1; n_ovr += 1
                            # Flip d (exit)
                            d = np.int8(-d)
                            # Reopen in signal direction
                            direction = sig_dir
                            e = c
                            if direction == 1.0:
                                uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
                            else:
                                lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
                            for k in range(ml): lv[k]=0.0; ld[k]=0.0; lp[k]=0.0
                            lv[0]=1.0; ld[0]=direction; lp[0]=e
                            nl=1; lu=-1; ll=-1; peak_mfe=0.0; trail_on=False
                            is_random = False; n_sig += 1
                            continue   # skip rest of bar (just opened fresh)

            # ── Trail (1st leg only) ──────────────────────────────────────
            ex = False
            if nl == 1:
                cur_mfe = (h - e) / pip if direction == 1.0 else (e - l) / pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if direction == 1.0:
                        ts = e + (peak_mfe - td) * pip
                        if l <= ts:
                            total += (ts - e) / pip - spread; nc += 1; n_trail += 1; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts:
                            total += (e - ts) / pip - spread; nc += 1; n_trail += 1; ex = True

            # ── ZR target exits + leg additions ──────────────────────────
            if not ex:
                for pn in range(2):
                    if ex: break
                    dh = (bull and pn == 0) or (not bull and pn == 1)
                    if l <= ut <= h:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                        total+=net-tv*spread; nc+=1; ex=True; break
                    if l <= lt <= h:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                        total+=net-tv*spread; nc+=1; ex=True; break
                    if dh and h >= uz and lu != i:
                        lu=i; nt2=0.0; tv=0.0
                        for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                        nt2 -= tv*spread
                        if nt2 >= 0:
                            if c >= ut: total+=nt2; nc+=1; ex=True; break
                        else:
                            v = max(1.0, math.ceil(-nt2/tgt*pf))
                            if nl >= ml:
                                net=0.0; tv2=0.0
                                for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                                total+=net-tv2*spread; nc+=1; ex=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                    if not dh and l <= lz and ll != i:
                        ll=i; nt2=0.0; tv=0.0
                        for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                        nt2 -= tv*spread
                        if nt2 >= 0:
                            if c <= lt: total+=nt2; nc+=1; ex=True; break
                        else:
                            v = max(1.0, math.ceil(-nt2/tgt*pf))
                            if nl >= ml:
                                net=0.0; tv2=0.0
                                for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                                total+=net-tv2*spread; nc+=1; ex=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1

            if ex:
                in_cycle = False; is_random = False
                d = np.int8(-d)   # flip on exit

    return total, nc, n_trail, n_sig, n_rnd, n_ovr


# ─── JIT warm-up ─────────────────────────────────────────────────────────────

print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:3000].astype(np.float64); _h=_df0.high.values[:3000].astype(np.float64)
_l=_df0.low.values[:3000].astype(np.float64);  _c=_df0.close.values[:3000].astype(np.float64)
sim_zr_hybrid(_o,_h,_l,_c,0.0005,3.0,0,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.,1,0.0)
sim_zr_hybrid(_o,_h,_l,_c,0.0005,3.0,1,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.,1,-2.0)
print("done.\n")


# ─── Main sweep ──────────────────────────────────────────────────────────────

rows = []

for pair, cfg in PAIR_CFG.items():
    pip      = PIP_MAP[pair]
    zw=cfg['zw']; tgt=cfg['tgt']; ta=cfg['ta']; td=cfg['td']
    box_size = cfg['box_pips'] * pip
    rev_mult = cfg['rev']

    df = pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(np.float64); hi=df.high.values.astype(np.float64)
    lo=df.low.values.astype(np.float64);  cl=df.close.values.astype(np.float64)
    nb = len(cl)

    is_end   = int(nb*(1-OOS_FRAC))
    is_op=op[:is_end]; is_hi=hi[:is_end]; is_lo=lo[:is_end]; is_cl=cl[:is_end]
    oos_op=op[is_end:]; oos_hi=hi[is_end:]; oos_lo=lo[is_end:]; oos_cl=cl[is_end:]
    oos_bars = len(oos_cl); oos_td = oos_bars/(24*12)
    is_bars  = is_end;     is_td  = is_bars/(24*12)
    is_chunk = is_bars // WF_CHUNKS

    print(f"\n{'='*72}")
    print(f"{pair}  b={cfg['box_pips']} r={rev_mult} zw={zw} tgt={tgt} ta={ta} td={td}")
    print(f"  {'mode':22} {'fb':>4} {'ovr':>5} | {'is_ppd':>8} {'oos_ppd':>8} {'wf':>3} "
          f"{'n_cyc':>6} {'sig%':>6} {'rnd%':>6} {'ovr#':>5}")
    print(f"  {'-'*80}")

    for dir_mode, dir_name in [(0,'ALT'), (1,'COL')]:
        # ── Pure P&F baseline (n_fallback=999999) ─────────────────────────
        tot_is, nc_is, _, _, _, _ = sim_zr_hybrid(
            is_op,is_hi,is_lo,is_cl,box_size,rev_mult,dir_mode,
            pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td,999999,0.0)
        is_ppd = tot_is / is_td

        wf = 0
        cs = 0
        for ch in range(WF_CHUNKS):
            ce = (ch+1)*is_chunk if ch < WF_CHUNKS-1 else is_bars
            ct,_,_,_,_,_ = sim_zr_hybrid(
                is_op[cs:ce],is_hi[cs:ce],is_lo[cs:ce],is_cl[cs:ce],
                box_size,rev_mult,dir_mode,pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td,999999,0.0)
            wf += (ct > 0); cs = ce

        tot_oos, nc_oos, nt, ns, nr, no = sim_zr_hybrid(
            oos_op,oos_hi,oos_lo,oos_cl,box_size,rev_mult,dir_mode,
            pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td,999999,0.0)
        oos_ppd = tot_oos / oos_td
        sig_pct = 100*ns/max(nc_oos,1); rnd_pct = 100*nr/max(nc_oos,1)
        print(f"  Pure P&F {dir_name:13} {'∞':>4} {'—':>5} | {is_ppd:>8.1f} {oos_ppd:>8.1f} "
              f"{wf:>3} {nc_oos:>6} {sig_pct:>5.1f}% {rnd_pct:>5.1f}% {no:>5}")
        rows.append(dict(pair=pair,dir=dir_name,mode='pure_pnf',n_fallback=999999,
                         override_thresh=0,is_ppd=round(is_ppd,1),oos_ppd=round(oos_ppd,1),
                         wf=wf,n_cycles=nc_oos,sig_pct=round(sig_pct,1),
                         rnd_pct=round(rnd_pct,1),n_overrides=no))

        # ── Hybrid variants ────────────────────────────────────────────────
        ovr_list = [0.0] if dir_mode == 0 else OVERRIDE_THRESH
        for nfb in N_FALLBACKS:
            for ovr in ovr_list:
                tot_is2,_,_,_,_,_ = sim_zr_hybrid(
                    is_op,is_hi,is_lo,is_cl,box_size,rev_mult,dir_mode,
                    pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td,nfb,ovr)
                is_ppd2 = tot_is2 / is_td

                wf2 = 0; cs = 0
                for ch in range(WF_CHUNKS):
                    ce = (ch+1)*is_chunk if ch < WF_CHUNKS-1 else is_bars
                    ct,_,_,_,_,_ = sim_zr_hybrid(
                        is_op[cs:ce],is_hi[cs:ce],is_lo[cs:ce],is_cl[cs:ce],
                        box_size,rev_mult,dir_mode,pip,SPREAD,PF,MAX_LEGS,
                        zw,tgt,ta,td,nfb,ovr)
                    wf2 += (ct > 0); cs = ce

                tot_oos2,nc_oos2,nt2,ns2,nr2,no2 = sim_zr_hybrid(
                    oos_op,oos_hi,oos_lo,oos_cl,box_size,rev_mult,dir_mode,
                    pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td,nfb,ovr)
                oos_ppd2 = tot_oos2 / oos_td
                sp = 100*ns2/max(nc_oos2,1); rp = 100*nr2/max(nc_oos2,1)
                ovr_str = f"{int(ovr):+d}p" if dir_mode == 1 else "—"
                fb_str  = f"{nfb*5}m"
                print(f"  Hybrid {dir_name:15} {fb_str:>4} {ovr_str:>5} | "
                      f"{is_ppd2:>8.1f} {oos_ppd2:>8.1f} {wf2:>3} {nc_oos2:>6} "
                      f"{sp:>5.1f}% {rp:>5.1f}% {no2:>5}")
                rows.append(dict(pair=pair,dir=dir_name,mode='hybrid',n_fallback=nfb,
                                 override_thresh=int(ovr),is_ppd=round(is_ppd2,1),
                                 oos_ppd=round(oos_ppd2,1),wf=wf2,n_cycles=nc_oos2,
                                 sig_pct=round(sp,1),rnd_pct=round(rp,1),n_overrides=no2))
        print()

    sys.stdout.flush()

# ─── Save + summary ───────────────────────────────────────────────────────────

df_res = pd.DataFrame(rows)
df_res.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved {len(df_res)} rows → {OUT_PATH}")

print("\n=== SUMMARY: best hybrid vs pure P&F per pair×dir (WF≥2) ===")
print(f"{'Pair+Dir':20} {'Mode':22} {'fb':>5} {'ovr':>5} | {'oos_ppd':>8} {'wf':>3} | vs_pure")
print("─"*80)
for (pair,dirn), grp in df_res.groupby(['pair','dir']):
    pure = grp[grp['mode']=='pure_pnf']['oos_ppd'].values[0]
    best = grp[grp['wf']>=2].sort_values('oos_ppd', ascending=False).iloc[0]
    delta = best['oos_ppd'] - pure
    flag  = "🟢" if delta > 0 else "🔴"
    print(f"{pair+' '+dirn:20} {best['mode']:22} {best['n_fallback']:>5} "
          f"{best['override_thresh']:>5} | {best['oos_ppd']:>8.1f} {int(best['wf']):>3} | "
          f"{delta:>+8.1f} {flag}")
