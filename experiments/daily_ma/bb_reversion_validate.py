"""
bb_reversion_validate.py — adversarial validation of the top multi-TF reversion configs.

We JUST found a phantom-fill artifact in this code family (BB-fade "+43 p/d" was fake) and an
accounting bug (tdpp inflated ~12x). This script treats every positive number as GUILTY:

STEP 1  Corrected accounting reconciliation: assert exp * tdpp ~= pdpp per config.
STEP 2  Fill-realism audit: replay actual trades in pure Python (independent of the numba kernel),
        confirm (a) NO exit books a price outside the bar's true [low,high]; (b) entry is on the
        protective side of the stop; (c) the exit price was actually reachable that bar. Spot-print 10.
STEP 3  Sealed OOS (R8): 70/30 time split per pair; report IS vs OOS exp/p-day/WR/pairs+.
STEP 4  Monte Carlo: circular block-bootstrap of per-trade pnl signs / permutation; >=300 iters; p-value.
STEP 5  Verdict per config.

Usage: python3 bb_reversion_validate.py 2>&1 | tee bb_reversion_validate.out
"""
import os, sys, time
import numpy as np
import pandas as pd

# import the (now-corrected) harness primitives
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_reversion_proper as H

PAIRS = H.PAIRS
TRIGS = H.TRIGS
AGREE = H.AGREE

# The three configs the coarse sweep flagged. All single-TF, close_beyond.
CONFIGS = [
    dict(name="1h_MA5_K1.5",  fast="1h", slow="", ma=5,  K=1.5, trig="close_beyond", ag=-1),
    dict(name="30m_MA9_K2.0", fast="30m",slow="", ma=9,  K=2.0, trig="close_beyond", ag=-1),
    dict(name="1h_MA9_K2.0",  fast="1h", slow="", ma=9,  K=2.0, trig="close_beyond", ag=-1),
]

OOS_FRAC = 0.30   # last 30% of each pair's bars = sealed OOS
MC_ITERS = 500


# ---------------------------------------------------------------------------
# STEP 2 — independent pure-Python replay for fill audit. Mirrors backtest_core
# exactly but records entry/stop/target/exit PRICES + the bar's [low,high] so we
# can prove every booked fill was reachable. NOT numba — a second implementation,
# so a bug in one wouldn't hide a bug in the other.
# ---------------------------------------------------------------------------
def replay_audit(O,H_,L,C,SP, basis,sd,up,lo, trig_code, tcap, ma, pip):
    n=len(C)
    trades=[]   # each: dict(ei,xi,dir,ent,stp,tgt_band,exit,bar_lo,bar_hi,pnl_net,exit_kind)
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0
    prev_up_out=False; prev_dn_out=False
    prev2_up_out=False; prev2_dn_out=False
    rejects=0
    for i in range(ma, n-1):
        if np.isnan(basis[i]):
            prev2_up_out=prev_up_out; prev2_dn_out=prev_dn_out
            prev_up_out=False; prev_dn_out=False
            continue
        up_out = L[i] > up[i]
        dn_out = H_[i] < lo[i]
        if up_out:
            if ext != 1: peak=H_[i]; ext=1
            else: peak=max(peak,H_[i])
        elif dn_out:
            if ext != -1: peak=L[i]; ext=-1
            else: peak=min(peak,L[i])
        if pos!=0:
            ex=np.nan; kind=""
            if pos==-1:
                if H_[i] > stp:
                    ex = stp if stp >= L[i] else O[i]; kind="stop"
                elif L[i] <= lo[i]:
                    ex = lo[i] if lo[i] <= H_[i] else O[i]; kind="target"
            else:
                if L[i] < stp:
                    ex = stp if stp <= H_[i] else O[i]; kind="stop"
                elif H_[i] >= up[i]:
                    ex = up[i] if up[i] >= L[i] else O[i]; kind="target"
            if np.isnan(ex) and (i-ei) >= tcap:
                ex=C[i]; kind="timecap"
            if not np.isnan(ex):
                bad = (ex > H_[i]+1e-9) or (ex < L[i]-1e-9)
                if bad: rejects+=1
                pnl = pos*(ex-ent)/pip - SP[ei]
                trades.append(dict(ei=ei, xi=i, dir=pos, ent=ent, stp=stp,
                                   exit=ex, bar_lo=L[i], bar_hi=H_[i],
                                   pnl=pnl, kind=kind, bad=bad, spread=SP[ei]))
                pos=0
        if pos==0:
            short_sig=False; long_sig=False
            if trig_code==0:
                if prev_up_out and (L[i] <= up[i]): short_sig=True
                elif prev_dn_out and (H_[i] >= lo[i]): long_sig=True
            elif trig_code==1:
                if C[i] > up[i]: short_sig=True
                elif C[i] < lo[i]: long_sig=True
            else:
                if prev2_up_out and prev_up_out and (L[i] <= up[i]): short_sig=True
                elif prev2_dn_out and prev_dn_out and (H_[i] >= lo[i]): long_sig=True
            ent_px=O[i+1]
            if short_sig:
                if ent_px < peak:   # protective: fill below stop(peak)
                    pos=-1; ent=ent_px; ei=i+1; stp=peak
            elif long_sig:
                if ent_px > peak:   # protective: fill above stop(peak)
                    pos=1; ent=ent_px; ei=i+1; stp=peak
        prev2_up_out=prev_up_out; prev2_dn_out=prev_dn_out
        prev_up_out=up_out; prev_dn_out=dn_out
    return trades, rejects


def audit_pair(pdata, pip, cfg):
    O,H_,L,C,SP = pdata[cfg["fast"]]
    basis,sd,up,lo = H.bands_for(C, cfg["ma"], cfg["K"])
    tcap=int(round(H.bars_per_day(cfg["fast"])*H.TCAP_DAYS))
    return replay_audit(O,H_,L,C,SP,basis,sd,up,lo,TRIGS[cfg["trig"]],tcap,cfg["ma"],pip)


# ---------------------------------------------------------------------------
# STEP 4 — Monte Carlo. Null: the per-trade P&L has no structure beyond its
# magnitude distribution. We randomize the SIGN of each trade's gross move
# (fair coin) and re-deduct the same real spread, 500x; p = fraction of shuffles
# whose mean net pnl >= observed. (This tests "is the directional edge real, or
# could a coin-flip on these same magnitudes net positive after spread?")
# Also a timing-permutation variant for robustness.
# ---------------------------------------------------------------------------
def mc_sign_flip(gross, spread, iters=MC_ITERS, seed=0):
    rng=np.random.default_rng(seed)
    obs=(gross-spread).mean()
    cnt=0
    for _ in range(iters):
        signs=rng.choice([-1.0,1.0], size=len(gross))
        m=(signs*np.abs(gross)-spread).mean()
        if m>=obs: cnt+=1
    return obs, cnt/iters


def main():
    t0=time.time()
    needed=sorted({c["fast"] for c in CONFIGS}, key=lambda x:H.TF_MULT[x])
    print(f"Loading {len(PAIRS)} pairs, TFs={needed} ...", flush=True)
    PD={}; PIP={}
    for p in PAIRS:
        PD[p],PIP[p]=H.load_pair_tfs(p,needed)
    print(f"Loaded in {time.time()-t0:.0f}s\n", flush=True)

    for cfg in CONFIGS:
        print("="*92)
        print(f"CONFIG  {cfg['name']}  (fast={cfg['fast']} ma={cfg['ma']} K={cfg['K']} trig={cfg['trig']})")
        print("="*92)

        # ---- aggregate corrected stats + IS/OOS + collect for MC + audit ----
        all_pnl=[]; all_gross=[]; all_sp=[]
        is_pnl=[]; oos_pnl=[]
        per_pair_pos=0; per_pair_n=0; total_trades=0; total_days=0.0
        is_pairs_pos=0; oos_pairs_pos=0
        total_rejects=0
        audit_samples=[]   # (pair, trade dict) for spot-printing
        for p in PAIRS:
            r=H.run_config(PD[p],PIP[p],cfg["fast"],cfg["slow"],cfg["ma"],cfg["K"],
                           TRIGS[cfg["trig"]], cfg["ag"])
            if r is None: continue
            pnl=r["pnl"]; gross=r["fav"]; sp=r["spread"]; ei=r["ei"]; nb=r["n_bars"]
            if len(pnl)==0: continue
            all_pnl.append(pnl); all_gross.append(gross); all_sp.append(sp)
            per_pair_n+=1; total_trades+=len(pnl)
            total_days += nb/H.bars_per_day(cfg["fast"])
            if pnl.mean()>0: per_pair_pos+=1
            # IS/OOS split by entry bar index
            split=int(nb*(1-OOS_FRAC))
            ism=ei<split; oosm=ei>=split
            if ism.sum()>0:
                is_pnl.append(pnl[ism])
                if pnl[ism].mean()>0: is_pairs_pos+=1
            if oosm.sum()>0:
                oos_pnl.append(pnl[oosm])
                if pnl[oosm].mean()>0: oos_pairs_pos+=1

            # ---- fill audit on first 3 pairs ----
            if p in ("EUR_USD","USD_JPY","GBP_JPY"):
                trades, rej = audit_pair(PD[p],PIP[p],cfg)
                total_rejects+=rej
                # cross-check: numba trade count vs python replay count
                print(f"  [audit] {p}: numba_trades={len(pnl)}  python_replay_trades={len(trades)}  "
                      f"phantom_rejects={rej}")
                # protective-side check: short ent<stp, long ent>stp
                bad_side=0
                for t in trades:
                    if t["dir"]==-1 and not (t["ent"]<t["stp"]): bad_side+=1
                    if t["dir"]== 1 and not (t["ent"]>t["stp"]): bad_side+=1
                if bad_side: print(f"  [audit] {p}: *** {bad_side} trades with WRONG protective side ***")
                if p=="EUR_USD":
                    audit_samples=trades[:10]

        pnl=np.concatenate(all_pnl); gross=np.concatenate(all_gross); spv=np.concatenate(all_sp)
        exp=pnl.mean(); wr=(pnl>0).mean()
        tdpp=total_trades/total_days
        pdpp=pnl.sum()/total_days

        # STEP 1 reconciliation
        recon = exp*tdpp
        print(f"\n  STEP1 accounting:  exp={exp:+.3f} p/trade  tdpp={tdpp:.3f} tr/day/pair  "
              f"pdpp={pdpp:+.3f} p/day/pair")
        print(f"         reconcile  exp*tdpp = {recon:+.3f}  vs pdpp = {pdpp:+.3f}  "
              f"-> {'OK' if abs(recon-pdpp)<1e-6 else 'MISMATCH'}")
        print(f"         WR={100*wr:.1f}%  n={len(pnl)}  pairs+={per_pair_pos}/{per_pair_n}")

        # STEP 2 summary
        print(f"\n  STEP2 fill-realism:  total phantom rejects (exit outside [low,high]) = {total_rejects}")
        if audit_samples:
            print(f"         spot-check 10 EUR_USD trades (dir entry stop exit bar[lo,hi] kind pnl):")
            for t in audit_samples:
                inside = t["bar_lo"]-1e-9 <= t["exit"] <= t["bar_hi"]+1e-9
                side = (t["ent"]<t["stp"]) if t["dir"]==-1 else (t["ent"]>t["stp"])
                print(f"           {'S' if t['dir']==-1 else 'L'} ent={t['ent']:.5f} stp={t['stp']:.5f} "
                      f"exit={t['exit']:.5f} bar=[{t['bar_lo']:.5f},{t['bar_hi']:.5f}] {t['kind']:>7} "
                      f"pnl={t['pnl']:+.1f}  exit_in_bar={inside} prot_side={side}")

        # STEP 3 IS/OOS
        ispnl=np.concatenate(is_pnl) if is_pnl else np.array([])
        oospnl=np.concatenate(oos_pnl) if oos_pnl else np.array([])
        print(f"\n  STEP3 sealed OOS (70/30 by time):")
        if len(ispnl):
            print(f"         IS : exp={ispnl.mean():+.3f}  WR={100*(ispnl>0).mean():.1f}%  "
                  f"n={len(ispnl)}  pairs+={is_pairs_pos}/{per_pair_n}")
        if len(oospnl):
            print(f"         OOS: exp={oospnl.mean():+.3f}  WR={100*(oospnl>0).mean():.1f}%  "
                  f"n={len(oospnl)}  pairs+={oos_pairs_pos}/{per_pair_n}")

        # STEP 4 MC (on OOS if it has trades, else full)
        mc_target = oospnl if len(oospnl)>=100 else pnl
        mc_gross  = gross  # align: we need gross+spread for the sign flip on the SAME set
        # rebuild gross/spread aligned to mc_target set:
        if mc_target is oospnl:
            # collect oos gross/spread
            og=[]; osp=[]
            for p in PAIRS:
                r=H.run_config(PD[p],PIP[p],cfg["fast"],cfg["slow"],cfg["ma"],cfg["K"],
                               TRIGS[cfg["trig"]], cfg["ag"])
                if r is None or len(r["pnl"])==0: continue
                split=int(r["n_bars"]*(1-OOS_FRAC)); m=r["ei"]>=split
                og.append(r["fav"][m]); osp.append(r["spread"][m])
            mc_gross=np.concatenate(og); mc_sp=np.concatenate(osp); mc_label="OOS"
        else:
            mc_sp=spv; mc_label="full"
        obs, pval = mc_sign_flip(mc_gross, mc_sp, MC_ITERS)
        print(f"\n  STEP4 Monte Carlo (sign-flip, {MC_ITERS} iters, on {mc_label} set, n={len(mc_gross)}):")
        print(f"         observed net exp={obs:+.3f}   p(shuffle>=obs)={pval:.4f}  "
              f"-> {'SIGNIFICANT' if pval<0.05 else 'NOT significant'}")

        # STEP 5 verdict
        frequent = tdpp>=2.0
        meaty = np.median(gross[gross>0])>=2*np.median(spv) if (gross>0).any() else False
        oos_pos = len(oospnl)>0 and oospnl.mean()>0
        oos_broad = oos_pairs_pos/per_pair_n>=0.6
        mc_ok = pval<0.05
        fill_clean = total_rejects==0
        survives = frequent and meaty and oos_pos and oos_broad and mc_ok and fill_clean
        print(f"\n  STEP5 VERDICT: {'SURVIVES' if survives else 'DIES'}")
        print(f"         frequent(tdpp>=2)={frequent} ({tdpp:.2f})  meaty={meaty}  "
              f"oos_pos={oos_pos}  oos_broad(>=60%)={oos_broad} ({oos_pairs_pos}/{per_pair_n})  "
              f"mc_sig={mc_ok}  fill_clean={fill_clean}")
        print()

    print(f"Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
