"""
bb_reversion_proper.py — PROPER redo of the intraday mean-reversion (BB-fade) experiment.

Goal: find a config that is simultaneously
  (1) FREQUENT   (>= a few trades/day/pair),
  (2) MEATY      (median favorable reversion move >> spread), and
  (3) DEFINITIVE (positive expectancy NET of REAL per-bar spread, robust across 12 pairs + WF).

FIXED FILL MODEL (reuses lib/bb_fade.py's corrected logic):
  - Entry fills at the NEXT bar's open (causal). Meat gate uses the CURRENT close (known at signal).
  - Entry-validity gate: only enter if the fill is on the PROTECTIVE side of the stop
      short: fill < extension_peak ;  long: fill > extension_peak.
    (Past the peak => fade pre-invalidated => SKIP. This is the phantom-fill bug fix.)
  - A stop only books when genuinely adverse and the bar actually reaches it.
  - HARNESS SELF-CHECK: no booked exit price may lie outside the bar's actual [low, high].

REAL SPREAD: per-bar (ask_c - bid_c)/pip, carried through resampling (spread at the FILL bar).

SWEEP:
  MA in {5,9,14,20,50};  K (sigma mult) in {1.0,1.5,2.0,2.5,3.0}
  trigger in {reenter (bar fully outside then re-enters), close_beyond (close past Kσ), two_outside}
  exit: opposite band (opp_band), stop=extension peak, time cap.
  Multi-TF agreement: fast TF triggers entry on re-entry; slow TF must agree on over-extension.
    agree defs: 'both' (slow bar also fully outside its own Kσ band, same dir)
                'basis' (slow close beyond its basis in the fade-favorable direction)
    Single-TF baselines included (slow_tf = '').

Usage:  python3 bb_reversion_proper.py 2>&1 | tee bb_reversion_proper.out
"""
import os, sys, time, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit, prange

DATA_DIR = "data/s5_ohlc"
PAIRS = ["EUR_USD","GBP_USD","AUD_USD","NZD_USD","EUR_GBP","USD_JPY",
         "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY"]
def pip_of(p): return 0.01 if p.endswith("JPY") else 0.0001

# S5 multipliers per TF
TF_MULT = {"10s":2,"30s":6,"1m":12,"5m":60,"10m":120,"30m":360,"1h":720}
# bars/day per TF (24h FX): 86400s/day / (5s*mult)
def bars_per_day(tf): return 86400.0/(5.0*TF_MULT[tf])

MAS    = [5,9,14,20,50]
KS     = [1.0,1.5,2.0,2.5,3.0]
TRIGS  = {"reenter":0,"close_beyond":1,"two_outside":2}
# Multi-TF combos: (fast, slow). slow='' => single-TF baseline.
COMBOS = [
    ("10s","5m"),("30s","1h"),("1m","1h"),("5m","1h"),("10m","1h"),("5m","30m"),
    ("10s",""),("30s",""),("1m",""),("5m",""),("10m",""),("30m",""),("1h",""),
]
AGREE  = {"both":0,"basis":1}   # only used when slow_tf != ''
MEAT   = 0.0    # meat gate handled as a reported distribution + a soft floor; we report median meat.
                # We do NOT pre-filter on meat in the coarse pass (we want the distribution);
                # the entry-validity + opp-band/stop economics already encode "enough gap".
TCAP_DAYS = 1.0 # time cap = ~1 trading day worth of bars on the FAST tf

# ---------------------------------------------------------------------------
# Resampling: from S5 arrays build TF OHLC (mid) + per-bar spread (pips, mean over the bar).
# ---------------------------------------------------------------------------
def resample_tf(o,h,l,c,sp_pips, mult):
    n = len(c); nb = n//mult
    o=o[:nb*mult]; h=h[:nb*mult]; l=l[:nb*mult]; c=c[:nb*mult]; sp=sp_pips[:nb*mult]
    O=o.reshape(nb,mult)[:,0]
    H=h.reshape(nb,mult).max(1)
    L=l.reshape(nb,mult).min(1)
    C=c.reshape(nb,mult)[:,-1]
    SP=sp.reshape(nb,mult).mean(1)   # mean spread over the bar (pips)
    return O,H,L,C,SP

# ---------------------------------------------------------------------------
# Slow-TF over-extension state, propagated causally onto fast-bar indices.
# For each fast bar i (its close timestamp = (i+1)*fast_mult S5 bars), we need the
# slow-TF band state from the LAST FULLY CLOSED slow bar at that moment (causal, no leak).
# We compute, per fast bar, two flags: slow_up_over (slow over-extended UP), slow_dn_over.
# 'both'  : slow bar fully outside its Kσ band (low>up => up_over ; high<lo => dn_over)
# 'basis' : slow close beyond basis (close>up_basis? no — beyond BASIS toward the band):
#           up_over if slow close > slow basis ; dn_over if slow close < slow basis.
# ---------------------------------------------------------------------------
@njit(cache=False)
def slow_flags_on_fast(C_slow, basis_s, up_s, lo_s, H_slow, L_slow,
                       fast_mult, slow_mult, n_fast, ma_slow, agree):
    """Return up_over[n_fast], dn_over[n_fast] using only CLOSED slow bars (causal)."""
    up_over = np.zeros(n_fast, dtype=np.int8)
    dn_over = np.zeros(n_fast, dtype=np.int8)
    ns = len(C_slow)
    # per slow bar j, its closing S5 index = (j+1)*slow_mult - 1.
    # A fast bar i closes at S5 index (i+1)*fast_mult - 1. The latest CLOSED slow bar at that
    # moment is the largest j with (j+1)*slow_mult-1 <= (i+1)*fast_mult-1  =>
    #   j <= ((i+1)*fast_mult)/slow_mult - 1
    ratio = fast_mult/float(slow_mult)
    for i in range(n_fast):
        j = int(np.floor((i+1)*ratio)) - 1
        if j < ma_slow-1 or j >= ns: continue
        if np.isnan(basis_s[j]): continue
        if agree == 0:   # both: slow bar fully outside its band
            if L_slow[j] > up_s[j]: up_over[i] = 1
            elif H_slow[j] < lo_s[j]: dn_over[i] = 1
        else:            # basis: slow close beyond basis
            if C_slow[j] > basis_s[j]: up_over[i] = 1
            elif C_slow[j] < basis_s[j]: dn_over[i] = 1
    return up_over, dn_over

# ---------------------------------------------------------------------------
# Core backtest (numba). Fixed-fill model from bb_fade.py.
# trig: 0 reenter, 1 close_beyond, 2 two_outside.
# use_slow: 0/1 ; slow_up/slow_dn int8 arrays aligned to fast bars.
# Returns parallel arrays: entry_bar, exit_bar, dir, pnl_pips_net, meat_pips, fav_move_pips
# ---------------------------------------------------------------------------
@njit(cache=False)
def backtest_core(o,h,l,c,sp, basis,sd,up,lo, trig, tcap, ma,
                  use_slow, slow_up, slow_dn,
                  out_ei, out_xi, out_dir, out_pnl, out_meat, out_fav):
    n=len(c); k=0
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0; ext_dist=0.0
    prev_up_out=False; prev_dn_out=False
    prev2_up_out=False; prev2_dn_out=False
    for i in range(ma, n-1):
        if np.isnan(basis[i]):
            prev2_up_out=prev_up_out; prev2_dn_out=prev_dn_out
            prev_up_out=False; prev_dn_out=False
            continue
        up_out = l[i] > up[i]      # bar fully above upper band
        dn_out = h[i] < lo[i]      # bar fully below lower band
        # extension peak tracking (the protrusion high/low)
        if up_out:
            if ext != 1: peak=h[i]; ext=1
            else: peak=max(peak,h[i])
        elif dn_out:
            if ext != -1: peak=l[i]; ext=-1
            else: peak=min(peak,l[i])
        # manage open position
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i] > stp:                  # adverse: extension peak above -> stop
                    # limit/stop fill only valid if level is inside the bar; if the bar gapped
                    # entirely above the stop, the realizable fill is this bar's OPEN (worse).
                    ex = stp if stp >= l[i] else o[i]
                elif l[i] <= lo[i]:             # target: opposite (lower) band
                    # if the bar traded entirely BELOW the band (gapped through), you don't get
                    # the band price as a short cover — book the OPEN (the price you actually saw).
                    ex = lo[i] if lo[i] <= h[i] else o[i]
            else:
                if l[i] < stp:
                    ex = stp if stp <= h[i] else o[i]
                elif h[i] >= up[i]:             # target: upper band
                    ex = up[i] if up[i] >= l[i] else o[i]
            if np.isnan(ex) and (i-ei) >= tcap: ex=c[i]
            if not np.isnan(ex):
                # SELF-CHECK: a booked exit price must lie within this bar's [low, high].
                # (Time-cap exits use c[i], inherently in range. Stop/target exits use stp/band.)
                # If ex is outside the bar range we flag dir=-9999 so the caller aborts loudly.
                bad = (ex > h[i]+1e-9) or (ex < l[i]-1e-9)
                # store exit/entry PRICES (cols reused); pip conversion + spread done by the caller
                out_ei[k]=ei; out_xi[k]=i; out_dir[k]= -9999 if bad else pos
                out_pnl[k]=ex; out_meat[k]=ent
                out_fav[k]=ext_dist
                k+=1; pos=0
        # entry trigger
        if pos==0:
            # determine trigger fired this bar (short side first)
            short_sig=False; long_sig=False
            if trig==0:   # reenter: prev fully outside, this bar touches band
                if prev_up_out and (l[i] <= up[i]): short_sig=True
                elif prev_dn_out and (h[i] >= lo[i]): long_sig=True
            elif trig==1: # close_beyond: this close beyond Kσ band (and re-entry not required)
                if c[i] > up[i]: short_sig=True
                elif c[i] < lo[i]: long_sig=True
            else:         # two_outside: prev2 & prev both fully outside, this bar touches
                if prev2_up_out and prev_up_out and (l[i] <= up[i]): short_sig=True
                elif prev2_dn_out and prev_dn_out and (h[i] >= lo[i]): long_sig=True
            if use_slow==1:
                if short_sig and slow_up[i]==0: short_sig=False
                if long_sig and slow_dn[i]==0: long_sig=False
            ent_px=o[i+1]
            if short_sig:
                # protective side: fill must be below the extension peak (stop)
                if ent_px < peak:
                    pos=-1; ent=ent_px; ei=i+1; stp=peak
                    ext_dist=(peak-up[i])  # protrusion size beyond band (price units)
            elif long_sig:
                if ent_px > peak:
                    pos=1; ent=ent_px; ei=i+1; stp=peak
                    ext_dist=(lo[i]-peak)*-1.0
        prev2_up_out=prev_up_out; prev2_dn_out=prev_dn_out
        prev_up_out=up_out; prev_dn_out=dn_out
    return k

# ---------------------------------------------------------------------------
# Load all pairs once, resample to all needed TFs, cache.
# ---------------------------------------------------------------------------
def load_pair_tfs(pair, needed_tfs):
    t=pq.read_table(f"{DATA_DIR}/{pair}_S5_BA.parquet",
                    columns=["open","high","low","close","bid_c","ask_c"]).to_pandas()
    pip=pip_of(pair)
    o=t["open"].to_numpy(np.float64); h=t["high"].to_numpy(np.float64)
    l=t["low"].to_numpy(np.float64); c=t["close"].to_numpy(np.float64)
    sp=((t["ask_c"]-t["bid_c"]).to_numpy(np.float64))/pip
    sp=np.clip(sp,0.0,1e6)
    out={}
    for tf in needed_tfs:
        out[tf]=resample_tf(o,h,l,c,sp,TF_MULT[tf])
    return out, pip

def bands_for(C, ma, K):
    s=pd.Series(C)
    basis=s.rolling(ma).mean().to_numpy()
    sd=s.rolling(ma).std(ddof=0).to_numpy()
    return basis, sd, basis+K*sd, basis-K*sd

# preallocated buffers (max trades ~ generous)
_MAXTR=2_000_000
_ei=np.empty(_MAXTR,np.int64); _xi=np.empty(_MAXTR,np.int64)
_dr=np.empty(_MAXTR,np.int64); _pn=np.empty(_MAXTR,np.float64)
_mt=np.empty(_MAXTR,np.float64); _fv=np.empty(_MAXTR,np.float64)

def run_config(pdata, pip, fast_tf, slow_tf, ma, K, trig, agree):
    """Run one config on one pair. Returns dict of trade arrays + meta, or None."""
    O,H,L,C,SP = pdata[fast_tf]
    n=len(C)
    if n < ma+10: return None
    basis,sd,up,lo = bands_for(C, ma, K)
    use_slow=0; slow_up=np.zeros(n,np.int8); slow_dn=np.zeros(n,np.int8)
    if slow_tf:
        Os,Hs,Ls,Cs,SPs = pdata[slow_tf]
        # slow MA: use same ma length on slow TF
        bs,sds,ups,los = bands_for(Cs, ma, K)
        slow_up,slow_dn = slow_flags_on_fast(Cs,bs,ups,los,Hs,Ls,
                                              TF_MULT[fast_tf],TF_MULT[slow_tf],n,ma,agree)
        use_slow=1
    tcap=int(round(bars_per_day(fast_tf)*TCAP_DAYS))
    k=backtest_core(O,H,L,C,SP,basis,sd,up,lo,trig,tcap,ma,
                    use_slow,slow_up,slow_dn,_ei,_xi,_dr,_pn,_mt,_fv)
    if k==0: return None
    ei=_ei[:k].copy(); xi=_xi[:k].copy(); dr=_dr[:k].copy()
    exitpx=_pn[:k].copy(); entpx=_mt[:k].copy(); extd=_fv[:k].copy()
    # self-check: any -9999 dir = exit outside bar range -> abort loudly
    if np.any(dr==-9999):
        raise RuntimeError(f"PHANTOM FILL detected {fast_tf}/{slow_tf} ma{ma} K{K} trig{trig}")
    # convert to pip pnl net spread (spread at the FILL bar = ei)
    spread_at_fill = SP[ei]
    pnl = dr*(exitpx-entpx)/pip - spread_at_fill
    # favorable reversion move available (meat): for shorts = entry - opposite_band reached or basis;
    # report the realized favorable excursion proxy = max(0, dir*(entry - exit_if_target))...
    # simpler/robust meat proxy: distance entry->opposite band at entry time is unknown post-hoc;
    # use the actual favorable move captured on winners + the protrusion size.
    fav = dr*(exitpx-entpx)/pip  # gross favorable/adverse move in pips (before spread)
    return dict(ei=ei, xi=xi, dr=dr, pnl=pnl, fav=fav, spread=spread_at_fill, n_bars=n)

# ---------------------------------------------------------------------------
# Aggregation / scoring
# ---------------------------------------------------------------------------
def wf_folds(ei_all, pnl_all, n_bars, nfolds=3):
    """ei_all: trade entry bar index (within the pair's fast series). Split by bar index into nfolds
       equal time chunks; report per-fold mean pnl. Returns list of fold means (only folds with >=15 tr)."""
    edges=np.linspace(0,n_bars,nfolds+1).astype(int)
    res=[]
    for f in range(nfolds):
        m=(ei_all>=edges[f])&(ei_all<edges[f+1])
        if m.sum()>=15: res.append(pnl_all[m].mean())
        else: res.append(np.nan)
    return res

def main():
    t0=time.time()
    needed=set()
    for f,s in COMBOS:
        needed.add(f)
        if s: needed.add(s)
    needed=sorted(needed, key=lambda x:TF_MULT[x])
    print(f"Loading {len(PAIRS)} pairs, TFs={needed} ...", flush=True)
    PD={}; PIP={}
    for p in PAIRS:
        PD[p],PIP[p]=load_pair_tfs(p,needed)
        # report span once
    O,H,L,C,SP=PD[PAIRS[0]][needed[0]]
    print(f"Loaded. Example {PAIRS[0]} {needed[0]}: {len(C)} bars.", flush=True)

    # ---- COARSE IS PASS (single combined backtest, then WF on full series) ----
    # We run every (combo,ma,K,trig,agree) across all 12 pairs, aggregate.
    print("\n=== COARSE PASS: full-history multi-pair, real spread, fixed fills ===", flush=True)
    results=[]
    ncfg=0
    for (fast_tf, slow_tf) in COMBOS:
        agrees = AGREE.items() if slow_tf else [("",-1)]
        for ag_name, ag in agrees:
            for trig_name,trig in TRIGS.items():
                for ma in MAS:
                    for K in KS:
                        ncfg+=1
                        # collect across pairs
                        all_pnl=[]; all_fav=[]; all_sp=[]
                        per_pair_pos=0; per_pair_n=0
                        wf_pos=np.zeros(3,dtype=int); wf_tot=np.zeros(3,dtype=int)
                        wf_sum=np.zeros(3); wf_cnt=np.zeros(3,dtype=int)
                        total_trades=0; total_days=0.0
                        for p in PAIRS:
                            r=run_config(PD[p],PIP[p],fast_tf,slow_tf,ma,K,trig,ag)
                            if r is None: continue
                            pnl=r["pnl"];
                            if len(pnl)<10:
                                # still count days for freq
                                pass
                            all_pnl.append(pnl); all_fav.append(r["fav"]); all_sp.append(r["spread"])
                            per_pair_n+=1
                            if len(pnl)>0 and pnl.mean()>0: per_pair_pos+=1
                            total_trades+=len(pnl)
                            total_days+= r["n_bars"]/bars_per_day(fast_tf)
                            # WF
                            fm=wf_folds(r["ei"],pnl,r["n_bars"],3)
                            for fi,v in enumerate(fm):
                                if not np.isnan(v):
                                    wf_cnt[fi]+=1; wf_sum[fi]+=v
                                    wf_tot[fi]+=1
                                    if v>0: wf_pos[fi]+=1
                        if not all_pnl: continue
                        pnl=np.concatenate(all_pnl); fav=np.concatenate(all_fav); spv=np.concatenate(all_sp)
                        if len(pnl)<60: continue
                        expc=pnl.mean(); wr=(pnl>0).mean()
                        # total_days = SUM of per-pair date spans (pair-days). The correct
                        # PER-PAIR-PER-DAY rate divides portfolio totals by total pair-days:
                        #   trades/day/pair = total_trades / total_days
                        #   pips/day/pair   = pnl.sum()  / total_days
                        # (Old code multiplied by per_pair_n, inflating tdpp ~12x. Fixed 2026-06-25.)
                        tdpp = total_trades/total_days if total_days else 0.0
                        pdpp = pnl.sum()/total_days if total_days else 0.0
                        # WF folds positive (fraction of pairs positive per fold -> fold "passes" if >50% pairs +)
                        wf_folds_pos=int(sum(1 for fi in range(3) if wf_cnt[fi]>0 and (wf_pos[fi]/wf_cnt[fi])>0.5))
                        # meat: median favorable move on WINNERS vs median spread
                        win_fav=fav[fav>0]
                        med_meat=np.median(win_fav) if len(win_fav) else 0.0
                        med_sp=np.median(spv)
                        results.append(dict(
                            fast=fast_tf, slow=slow_tf, ag=ag_name, trig=trig_name, ma=ma, K=K,
                            exp=expc, wr=wr, n=len(pnl), tdpp=tdpp, pdpp=pdpp,
                            pairs_pos=per_pair_pos, pairs_n=per_pair_n,
                            wf_pos=wf_folds_pos, med_meat=med_meat, med_sp=med_sp))
        print(f"  ...combo {fast_tf}/{slow_tf or '-'} done ({ncfg} cfgs, {time.time()-t0:.0f}s)", flush=True)

    df=pd.DataFrame(results)
    df.to_csv("research/experiments/daily_ma/bb_reversion_coarse.csv",index=False)
    print(f"\nTotal configs evaluated: {len(df)}", flush=True)

    # ----- Filter for the THREE bars -----
    # FREQUENT: tdpp >= 2 (a few trades/day/pair)
    # MEATY:    med_meat >= 2 * med_sp  (favorable move at least 2x spread)
    # DEFINITIVE: exp>0 AND pairs_pos/pairs_n >= 0.6 AND wf_pos==3
    df["meaty"]=df["med_meat"]>=2*df["med_sp"]
    df["frequent"]=df["tdpp"]>=2.0
    df["definitive"]=(df["exp"]>0)&(df["pairs_pos"]/df["pairs_n"]>=0.6)&(df["wf_pos"]==3)
    df["all3"]=df["meaty"]&df["frequent"]&df["definitive"]
    # combined score: rank positives by exp * sqrt(n) but only meaningful for exp>0
    df["score"]=np.where(df["exp"]>0, df["exp"]*np.sqrt(df["n"])*(df["pairs_pos"]/df["pairs_n"]),
                         df["exp"])

    def show(sub,title,by="score",n=25):
        print(f"\n=== {title} ===")
        if len(sub)==0: print("  (none)"); return
        s=sub.sort_values(by,ascending=False).head(n)
        hdr=f"{'fast':>5}{'slow':>5}{'agr':>6}{'trig':>13}{'MA':>4}{'K':>5} |{'td/d/p':>8}{'exp':>8}{'WR':>6}{'p/d/p':>8}{'pr+':>6}{'WF+':>5}{'meat':>7}{'sp':>6}"
        print(hdr); print("-"*len(hdr))
        for _,r in s.iterrows():
            print(f"{r['fast']:>5}{r['slow'] or '-':>5}{r['ag'] or '-':>6}{r['trig']:>13}{int(r['ma']):>4}{r['K']:>5.1f} |"
                  f"{r['tdpp']:>8.1f}{r['exp']:>+8.2f}{100*r['wr']:>5.0f}%{r['pdpp']:>+8.1f}"
                  f"{int(r['pairs_pos'])}/{int(r['pairs_n']):<2}{r['wf_pos']:>4}{r['med_meat']:>7.1f}{r['med_sp']:>6.2f}")

    show(df[df["all3"]], "CONFIGS PASSING ALL THREE BARS (frequent + meaty + definitive)")
    show(df[df["definitive"]], "DEFINITIVE (exp>0, >=60% pairs+, WF 3/3) — regardless of freq/meat")
    show(df[df["exp"]>0].sort_values("exp",ascending=False), "TOP BY EXPECTANCY (net spread, exp>0)", by="exp")
    show(df.sort_values("pdpp",ascending=False), "TOP BY PIPS/DAY/PAIR", by="pdpp")

    # Multi-TF vs single-TF comparison: best exp per fast TF, single vs paired
    print("\n=== MULTI-TF AGREEMENT vs SINGLE-TF (best exp per fast TF) ===")
    print(f"{'fast':>5} {'single best exp':>16} {'paired best exp':>16}  (slow/agr that won paired)")
    for ft in sorted(set(df["fast"]),key=lambda x:TF_MULT[x]):
        sub=df[df["fast"]==ft]
        single=sub[sub["slow"]==""]
        paired=sub[sub["slow"]!=""]
        se=single["exp"].max() if len(single) else float("nan")
        if len(paired):
            pe_row=paired.loc[paired["exp"].idxmax()]
            pe=pe_row["exp"]; tag=f"{pe_row['slow']}/{pe_row['ag']}"
        else:
            pe=float("nan"); tag="-"
        print(f"{ft:>5} {se:>+16.2f} {pe:>+16.2f}  {tag}")

    print(f"\nDone in {time.time()-t0:.0f}s. Coarse CSV -> bb_reversion_coarse.csv", flush=True)

if __name__=="__main__":
    main()
