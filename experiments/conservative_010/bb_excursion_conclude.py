#!/usr/bin/env python3
"""
BB-excursion — the CONCLUSION run (needs the 5-7yr S5 for all 12 pairs on the volume).

Phase A — ENTRY GATE (anti-overfit).  FIXED best config from the 4-pair sweep
  (sma200 / 1 sigma / contrarian / N=20 / T_enter=6 / T_exit=0 / sig-revert exit / SL150)
  on ALL 12 pairs, SEEN (the 4 chosen on) vs UNSEEN (other 8). No re-sweep -> the 8 unseen
  pairs are a clean OOS test (the gate that refuted the H1 breakout).

Phase B — EXIT BANK + full metrics (only if A passes).  Same fixed ENTRY, every exit in the
  bank, all 12 pairs. Per exit: net, IS/OOS, pairs+, trades/day, pips/day, MFE, MAE, duration,
  max drawdown, AMDDP5 (= mean per-trade pnl - 0.05*ddsum; ddsum = accumulated worsening
  intra-trade DD path, per lib/fast_eval.py), and MFE-capture (median realized / own MFE)
  vs the oracle ceiling (exit at MFE peak).

Memory-safe: one pair at a time (S5 read -> H1 resample -> discard). Real per-bar spread,
worse-side fills, 2p stop slippage (SOP R3/R3a).
"""
import sys, gc, os
import numpy as np
import pandas as pd
import numba as nb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate import split_is_oos, monte_carlo, equity_drawdown

ROOT   = Path(os.environ.get("FX_CORE_ROOT", Path(__file__).resolve().parents[3]))
S5_DIR = ROOT / "data" / "s5_ohlc"
SEEN   = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
ALL12  = ["AUD_JPY","AUD_USD","CAD_JPY","CHF_JPY","EUR_GBP","EUR_JPY",
          "EUR_USD","GBP_JPY","GBP_USD","NZD_JPY","NZD_USD","USD_JPY"]
UNSEEN = [p for p in ALL12 if p not in SEEN]

MA_PER, K            = 200, 1.0
N, T_ENTER, T_EXIT   = 20, 6.0, 0.0
DIRECTION            = -1
SL_PIPS, SLIP        = 150.0, 2.0
IS_FRAC              = 4.0 / 6.0
ATR_P                = 14
AMDDP1_BETA          = 0.01      # AMDDP1 — gentle DD penalty (fair for LONG-horizon holds)
AMDDP5_BETA          = 0.05      # AMDDP5 — heavier DD penalty


def pip_of(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def load_h1(pair):
    df = pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet",
                         columns=['timestamp','open','high','low','close','bid_c','ask_c'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    r = df.set_index('timestamp').resample('1h').agg(
        o=('open','first'), h=('high','max'), l=('low','min'), c=('close','last'),
        bid_c=('bid_c','last'), ask_c=('ask_c','last')).dropna()
    del df; gc.collect()
    c=r['c'].to_numpy(np.float64); h=r['h'].to_numpy(np.float64); l=r['l'].to_numpy(np.float64)
    center=pd.Series(c).rolling(MA_PER).mean().to_numpy(); sd=pd.Series(c).rolling(MA_PER).std(ddof=0).to_numpy()
    up=center+K*sd; lo=center-K*sd; rng=np.where((h-l)>0,h-l,np.nan)
    frac=np.nan_to_num(np.clip((h-np.maximum(up,l))/rng,0,1))-np.nan_to_num(np.clip((np.minimum(lo,h)-l)/rng,0,1))
    accum=np.nan_to_num(pd.Series(frac).rolling(N).sum().to_numpy())
    tr=np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr=np.full(len(c),np.nan)
    if len(tr)>=ATR_P:
        atr[ATR_P]=tr[:ATR_P].mean()
        for i in range(ATR_P+1,len(c)): atr[i]=(atr[i-1]*(ATR_P-1)+tr[i-1])/ATR_P
    atr=np.nan_to_num(atr)
    span=max(1.0,(r.index[-1]-r.index[0]).total_seconds()/86400.0)
    out={'pair':pair,'pip':pip_of(pair),'n':len(r),'is_end':int(len(r)*IS_FRAC),'span':span,
         'h':h,'l':l,'c':c,'bid_c':r['bid_c'].to_numpy(np.float64),
         'ask_c':r['ask_c'].to_numpy(np.float64),'accum':accum,'atr':atr}
    del r; gc.collect(); return out


@nb.njit(cache=True)
def _sim(h,l,c,bid_c,ask_c,accum,atr,start,pip,direction,t_enter,t_exit,
         mode,tp_pips,sl_pips,arm_pips,lock_frac,st_mult,nbar,slip):
    n=len(h); pos=0; ef=0.0; ebar=0; mfe=0.0; mae=0.0; ddsum=0.0; prev_dd=0.0; st=0.0
    eb=np.empty(n,np.int64); rn=np.empty(n,np.float64); fe=np.empty(n,np.float64)
    ae=np.empty(n,np.float64); hd=np.empty(n,np.int64); ds=np.empty(n,np.float64); nt=0
    for i in range(start,n):
        a=accum[i]
        if pos==0:
            gl=gs=False
            if direction==-1:
                if a>t_enter: gs=True
                elif a<-t_enter: gl=True
            else:
                if a>t_enter: gl=True
                elif a<-t_enter: gs=True
            if gl: pos=1; ef=ask_c[i]
            elif gs: pos=-1; ef=bid_c[i]
            if pos!=0:
                ebar=i; mfe=0.0; mae=0.0; ddsum=0.0; prev_dd=0.0
                if mode==6: st=(h[i]+l[i])/2.0-pos*st_mult*atr[i]
            continue
        fav=(h[i]-ef)/pip if pos==1 else (ef-l[i])/pip
        adv=(ef-l[i])/pip if pos==1 else (h[i]-ef)/pip
        if fav>mfe: mfe=fav
        if adv>mae: mae=adv
        cur_dd=mfe-fav
        if cur_dd<0.0: cur_dd=0.0
        if cur_dd>prev_dd: ddsum+=cur_dd-prev_dd
        prev_dd=cur_dd
        exf=0.0; hit=False
        slv=ef-pos*sl_pips*pip
        if pos==1 and l[i]<=slv: exf=slv-slip*pip; hit=True
        elif pos==-1 and h[i]>=slv: exf=slv+slip*pip; hit=True
        if not hit:
            if mode==0 or mode==2:
                if pos==1 and ((direction==-1 and a>-t_exit) or (direction==1 and a<t_exit)): exf=bid_c[i]; hit=True
                elif pos==-1 and ((direction==-1 and a<t_exit) or (direction==1 and a>-t_exit)): exf=ask_c[i]; hit=True
            elif mode==1:
                tp=ef+pos*tp_pips*pip
                if pos==1 and h[i]>=tp: exf=tp; hit=True
                elif pos==-1 and l[i]<=tp: exf=tp; hit=True
            elif mode==3:
                if mfe>=arm_pips and fav<=mfe*lock_frac: exf=(bid_c[i] if pos==1 else ask_c[i]); hit=True
            elif mode==4:
                if mfe>=arm_pips:
                    if pos==1 and l[i]<=ef: exf=ef; hit=True
                    elif pos==-1 and h[i]>=ef: exf=ef; hit=True
            elif mode==5:
                if (i-ebar)>=nbar: exf=(bid_c[i] if pos==1 else ask_c[i]); hit=True
            elif mode==6:
                nb_=(h[i]+l[i])/2.0-pos*st_mult*atr[i]
                if pos==1: st=nb_ if i==ebar else max(st,nb_)
                else:      st=nb_ if i==ebar else min(st,nb_)
                if pos==1 and c[i]<st: exf=bid_c[i]-slip*pip; hit=True
                elif pos==-1 and c[i]>st: exf=ask_c[i]+slip*pip; hit=True
        if hit:
            pnl=(exf-ef)/pip if pos==1 else (ef-exf)/pip
            eb[nt]=ebar; rn[nt]=pnl; fe[nt]=mfe; ae[nt]=mae; hd[nt]=i-ebar; ds[nt]=ddsum; nt+=1; pos=0
    return eb[:nt],rn[:nt],fe[:nt],ae[:nt],hd[:nt],ds[:nt]


EXITS=[  # label, mode, tp, sl, arm, lock, st_mult, nbar
    ("sig-revert(Tx0)",0,0.0,SL_PIPS,0.0,0.0,0.0,0),
    ("TP50/SL150",1,50.0,150.0,0.0,0.0,0.0,0),("TP100/SL150",1,100.0,150.0,0.0,0.0,0.0,0),
    ("TP150/SL150",1,150.0,150.0,0.0,0.0,0.0,0),("TP200/SL150",1,200.0,150.0,0.0,0.0,0.0,0),
    ("giveback50@40",3,0.0,150.0,40.0,0.50,0.0,0),("giveback70@40",3,0.0,150.0,40.0,0.70,0.0,0),
    ("breakeven@40",4,0.0,150.0,40.0,0.0,0.0,0),("nbar24",5,0.0,150.0,0.0,0.0,0.0,24),
    ("nbar72",5,0.0,150.0,0.0,0.0,0.0,72),("supertrend m2",6,0.0,150.0,0.0,0.0,2.0,0),
    ("supertrend m3",6,0.0,150.0,0.0,0.0,3.0,0),
]


def run_exit(H1, pairs, spec):
    _,mode,tp,sl,arm,lock,stm,nbar=spec
    R=[];F=[];A=[];Hd=[];Ds=[];ISN=OOSN=0.0;oosp=0;span=0.0
    for p in pairs:
        s=H1[p]; span=max(span,s['span'])
        eb,rn,fe,ae,hd,ds=_sim(s['h'],s['l'],s['c'],s['bid_c'],s['ask_c'],s['accum'],s['atr'],
                               MA_PER+N,s['pip'],DIRECTION,T_ENTER,T_EXIT,mode,tp,sl,arm,lock,stm,nbar,SLIP)
        io=split_is_oos(eb,rn,s['is_end']); ISN+=io['is_net']; OOSN+=io['oos_net']
        if io['oos_net']>0: oosp+=1
        R.append(rn);F.append(fe);A.append(ae);Hd.append(hd);Ds.append(ds)
    cat=lambda L:(np.concatenate(L) if L and sum(len(x) for x in L) else np.array([]))
    R=cat(R);F=cat(F);A=cat(A);Hd=cat(Hd);Ds=cat(Ds)
    n=len(R)
    m={'net':float(R.sum()),'is':ISN,'oos':OOSN,'oosp':oosp,'n':n,'span':span,'R':R,'F':F,'A':A,'Hd':Hd,'Ds':Ds}
    if n:
        m['tpd']=n/span; m['ppd']=R.sum()/span
        m['mMFE']=float(np.median(F)); m['mMAE']=float(np.median(A)); m['mHold']=float(np.median(Hd))
        m['maxDD']=equity_drawdown(np.cumsum(R))['max_dd']
        m['amddp1']=float(np.mean(R-AMDDP1_BETA*Ds))
        m['amddp5']=float(np.mean(R-AMDDP5_BETA*Ds))
        cap=R/np.where(F>1e-9,F,np.nan); m['cap']=float(np.nanmedian(cap))*100
        Af=np.maximum(A,1.0)                      # floor MAE at 1 pip (avoid div blow-up)
        m['pnl_mae']=float(np.median(R/Af))       # reward per unit heat (robust)
        m['eratio']=float(np.median(F/Af))        # E-ratio: MFE/MAE signal quality
    return m


def main():
    miss=[p for p in ALL12 if not (S5_DIR/f"{p}_S5_BA.parquet").exists()]
    if miss: print(f"MISSING parquets: {miss} — await the fetch."); return
    print("Loading 12 pairs (one at a time)...",flush=True)
    H1={}
    for p in ALL12:
        H1[p]=load_h1(p); print(f"  {p}: {H1[p]['n']} H1 bars ({H1[p]['span']/365:.1f}yr)",flush=True)

    print("\n========= PHASE A — ENTRY GATE (frozen config, seen vs unseen) =========",flush=True)
    seen=run_exit(H1,SEEN,EXITS[0]); uns=run_exit(H1,UNSEEN,EXITS[0])
    mc=monte_carlo(uns['R'],n=300) if uns['n'] else {'p_net':1.0}
    print(f"SEEN  (4): net={seen['net']:.0f}p IS={seen['is']:.0f} OOS={seen['oos']:.0f} oosP={seen['oosp']}/4 n={seen['n']}",flush=True)
    print(f"UNSEEN(8): net={uns['net']:.0f}p IS={uns['is']:.0f} OOS={uns['oos']:.0f} oosP={uns['oosp']}/8 n={uns['n']} MC p_net={mc['p_net']:.3f}",flush=True)
    passed=uns['oos']>0 and uns['oosp']>=5 and mc['p_net']<0.05
    print(f"GATE (unseen OOS>0 + >=5/8 pairs + MC<0.05): {'PASS - real edge' if passed else 'FAIL - overfit (like the breakout)'}",flush=True)

    print("\n========= PHASE B — EXIT BANK + FULL METRICS (all 12 pairs) =========",flush=True)
    if not passed:
        print("(entry FAILED the gate — exit sweep shown for completeness, but there is no edge to harvest)",flush=True)
    print(f"{'exit':<17}{'net':>8}{'OOS':>7}{'P+':>4}{'td/d':>6}{'pip/d':>7}{'MFE':>6}{'MAE':>6}{'hold_h':>7}{'maxDD':>8}{'AMDDP1':>8}{'AMDDP5':>8}{'pnl/MAE':>8}{'eR':>5}{'cap%':>6}",flush=True)
    rows=[]
    for spec in EXITS:
        r=run_exit(H1,ALL12,spec)
        if r['n']==0: continue
        rows.append((spec[0],r))
        print(f"{spec[0]:<17}{r['net']:>8.0f}{r['oos']:>7.0f}{r['oosp']:>4}{r['tpd']:>6.2f}{r['ppd']:>7.2f}"
              f"{r['mMFE']:>6.0f}{r['mMAE']:>6.0f}{r['mHold']:>7.0f}{r['maxDD']:>8.0f}{r['amddp1']:>8.2f}{r['amddp5']:>8.2f}{r['pnl_mae']:>8.2f}{r['eratio']:>5.2f}{r['cap']:>6.0f}",flush=True)
    ref=rows[0][1] if rows else None
    if ref is not None:
        orc=float(ref['F'].sum())
        print(f"\nOracle ceiling (exit each trade at its MFE peak): {orc:.0f}p / {ref['n']} trades "
              f"({orc/ref['span']:.1f} pip/day) — the harvestable max",flush=True)
        best=max(rows,key=lambda x:x[1]['oos'])
        b=best[1]
        print(f"Best exit by OOS: {best[0]}  net={b['net']:.0f}p OOS={b['oos']:.0f}p cap={b['cap']:.0f}% "
              f"AMDDP1={b['amddp1']:.2f} AMDDP5={b['amddp5']:.2f} maxDD={b['maxDD']:.0f}p hold={b['mHold']:.0f}h",flush=True)
    print("\ncolumns: td/d=trades/day  pip/d=pips/day (portfolio)  MFE/MAE=median pips  "
          "hold_h=median duration (H1 bars=hours)  maxDD=equity max drawdown  "
          "AMDDP1=mean(pnl-0.01*ddsum) AMDDP5=mean(pnl-0.05*ddsum)  pnl/MAE=median reward/heat (MAE>=1p) "
          "eR=median MFE/MAE (signal quality)  cap%=median realized/own-MFE",flush=True)


if __name__=="__main__":
    main()
