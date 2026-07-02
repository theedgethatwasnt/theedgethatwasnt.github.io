"""
sequential_validate.py — GO-LIVE validation of the BB re-entry fade, M15, opposite-band exit.
SEQUENTIAL engine (one position at a time per pair = realistic; reconciles the bake-off's inflated
independent forward-sim). Entry = re-entry fade + half-distance meat>=6 (real-spread-aware gate).
Exit compared: basis (old) vs opp_band (new winner). Stop = extension peak; time cap.
Reports PIPS/DAY (portfolio = 12 pairs concurrent, calendar-day normalized) + every gate:
IS/OOS (sealed), 4-fold walk-forward (all must be +), Monte-Carlo sign-flip p, SQN, t-stat, breadth.
Section A: 5.3y M15 mid, per-pair MEDIAN real spread.  Section B: 1.5y M15 real PER-BAR BA spread.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0; MEAT=6.0; TCAP=96
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}

def backtest(o,h,l,c,ts,basis,sd,sp_arr,fixed_sp,exit_mode,pip):
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0; tr=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        sp = fixed_sp if fixed_sp is not None else sp_arr[i]
        if np.isnan(sp): continue
        uo=l[i]>up[i]; do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stp: ex=stp
                elif exit_mode=="basis" and l[i]<=basis[i]: ex=basis[i]
                elif exit_mode=="opp_band" and l[i]<=lo[i]: ex=lo[i]
            else:
                if l[i]<stp: ex=stp
                elif exit_mode=="basis" and h[i]>=basis[i]: ex=basis[i]
                elif exit_mode=="opp_band" and h[i]>=up[i]: ex=up[i]
            if np.isnan(ex) and (i-ei)>=TCAP: ex=c[i]
            if not np.isnan(ex):
                tr.append((ts[ei], pos*(ex-ent)/pip-sp, (ts[i]-ts[ei])/np.timedelta64(1,'D'))); pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(e-basis[i])/pip-sp>=MEAT: pos=-1; ent=e; ei=i+1; stp=peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-e)/pip-sp>=MEAT: pos=1; ent=e; ei=i+1; stp=peak
    return tr  # (entry_ts, net_pnl, hold_days)

def gates(tr, span_days, rng, label):
    if len(tr)<30: print(f"  {label}: too few"); return
    t=sorted(tr,key=lambda x:x[0]); pnl=np.array([p for _,p,_ in t]); ets=np.array([x for x,_,_ in t])
    hold=np.array([hd for _,_,hd in t])
    ppd=pnl.sum()/span_days
    is_cut=ets[int(len(ets)*0.6)]; oos=pnl[ets>=is_cut]; isn=pnl[ets<is_cut]
    oos_days=span_days*0.4
    folds=np.array_split(pnl,4); fppd=[f.sum()/(span_days/4) for f in folds]
    null=np.array([(pnl*rng.choice([-1.,1.],len(pnl))).mean() for _ in range(5000)])
    mcp=(np.abs(null)>=abs(pnl.mean())).mean()
    sqn=np.sqrt(len(pnl))*pnl.mean()/pnl.std()
    tstat=pnl.mean()/(pnl.std()/np.sqrt(len(pnl)))
    print(f"  {label}")
    print(f"    PIPS/DAY: all {ppd:+.1f}  | IS {isn.sum()/(span_days*0.6):+.1f}  | OOS {oos.sum()/oos_days:+.1f}   (per-trade {pnl.mean():+.2f}p, WR {100*(pnl>0).mean():.0f}%)")
    print(f"    trades {len(pnl)} ({len(pnl)/span_days:.2f}/day), avg hold {hold.mean():.1f}d")
    print(f"    GATES: MC p={mcp:.4f} {'PASS' if mcp<0.05 else 'FAIL'} | SQN={sqn:.2f} {'PASS' if sqn>1 else 'FAIL'} | t-stat={tstat:.1f} {'PASS' if abs(tstat)>2 else 'FAIL'}")
    print(f"    WALK-FWD (4 folds, pips/day): [{' '.join(f'{x:+.1f}' for x in fppd)}]  {'ALL+ PASS' if all(x>0 for x in fppd) else 'FAIL (a fold negative)'}")
    return ppd

def run_section(con, use_real_bar, rng):
    med_sp={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df()
        med_sp[p]=float(np.nanmedian(s.s.values)/pip)
    out={}
    for mode in ["basis","opp_band"]:
        port=[]; perpair={}; spans=[]
        for p,pip in PAIRS.items():
            if use_real_bar:
                d=con.execute(f"SELECT timestamp,open,high,low,close,bid_c,ask_c FROM 'data/s5_ohlc/{p}_S5_BA.parquet' ORDER BY timestamp").df()
                d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d=d.set_index("timestamp")
                d=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","bid_c":"last","ask_c":"last"}).dropna()
                sp_arr=((d.ask_c.values-d.bid_c.values)/pip); fixed=None
            else:
                d=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
                d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d=d.set_index("timestamp")
                d=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
                sp_arr=None; fixed=med_sp[p]
            o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
            basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
            tr=backtest(o,h,l,c,ts,basis,sd,sp_arr,fixed,mode,pip)
            span=(ts[-1]-ts[0])/np.timedelta64(1,'D'); spans.append(span)
            port+=tr; perpair[p]=sum(x[1] for x in tr)/span if tr else 0.0
            gc.collect()
        span=np.mean(spans)
        npos=sum(1 for v in perpair.values() if v>0)
        print(f"\n  === exit = {mode} ===")
        ppd=gates(port,span,rng,f"PORTFOLIO (12 pairs, ~{span:.0f} days)")
        print(f"    BREADTH: {npos}/12 pairs positive | per-pair pips/day: {({k:round(v,2) for k,v in perpair.items()})}")
        out[mode]=ppd
    return out

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    print("="*96); print("SECTION A — 5.3y M15, per-pair MEDIAN real spread  (reconciliation + full gates)"); print("="*96)
    a=run_section(con,False,rng)
    print("\n"+"="*96); print("SECTION B — 1.5y M15, REAL PER-BAR BA spread  (strict real-cost confirmation)"); print("="*96)
    b=run_section(con,True,rng)
    print("\n"+"="*96)
    print(f"RECONCILE: basis vs opp_band portfolio pips/day — A(5.3y med): {a.get('basis',0):.1f} -> {a.get('opp_band',0):.1f}"
          f" | B(1.5y real): {b.get('basis',0):.1f} -> {b.get('opp_band',0):.1f}")

if __name__=="__main__": main()
