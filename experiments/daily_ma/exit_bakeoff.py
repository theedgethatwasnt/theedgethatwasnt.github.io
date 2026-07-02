"""
exit_bakeoff.py — improve the TAKE on the BB re-entry fade (M15, the validated config).
Fixed entry = re-entry fade + half-distance meat>=6 (12 pairs). For each entry, simulate forward
under competing exits and compare per-trade net P&L (same entries => fair). Plus an MFE diagnostic:
how far does the reversion run past the basis (is there uncaptured overshoot?).
Exits: basis(current) | opp_band | trail_basis | scaleout(half basis/half opp) | supertrend |
       timedecay | amddp(profit-protection give-back trail — penalize drawdown from peak MFE).
12 pairs, per-pair MEDIAN real spread (from S5 BA), 5.3y M15 mid, IS/OOS 60/40 + 3-fold WF.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9; K=1.0; MEAT=6.0; TCAP=96; GIVE=0.5   # amddp: exit after giving back 50% of peak MFE
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
MODES=["basis","opp_band","trail_basis","scaleout","supertrend","timedecay","amddp"]

def atr_w(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values

def supertrend(h,l,c,period=10,mult=3.0):
    n=len(c); a=atr_w(h,l,c,period); hl2=(h+l)/2
    ub=hl2+mult*a; lb=hl2-mult*a; st=np.full(n,np.nan); d=np.ones(n)
    for i in range(1,n):
        if np.isnan(a[i]): continue
        ub[i]=ub[i] if (ub[i]<ub[i-1] or c[i-1]>ub[i-1]) else ub[i-1]
        lb[i]=lb[i] if (lb[i]>lb[i-1] or c[i-1]<lb[i-1]) else lb[i-1]
        d[i]=1 if c[i]>ub[i-1] else (-1 if c[i]<lb[i-1] else d[i-1])
        st[i]=lb[i] if d[i]==1 else ub[i]
    return st

def sim(mode,i0,dr,ent,peak,basis,up,lo,st,h,l,c,pip):
    n=len(c); jend=min(i0+TCAP,n-1); reached=False; half1=None; peak_pnl=0.0
    j=i0
    while j<jend:
        j+=1
        cur=((ent-c[j]) if dr==-1 else (c[j]-ent))/pip
        peak_pnl=max(peak_pnl,((ent-l[j]) if dr==-1 else (h[j]-ent))/pip)
        # shared hard stop (extension peak)
        if dr==-1 and h[j]>peak:
            r=(ent-peak)/pip; return 0.5*half1+0.5*r if (mode=="scaleout" and half1 is not None) else r
        if dr==1 and l[j]<peak:
            r=(peak-ent)/pip; return 0.5*half1+0.5*r if (mode=="scaleout" and half1 is not None) else r
        hit_basis=(l[j]<=basis[j]) if dr==-1 else (h[j]>=basis[j])
        basis_pnl=((ent-basis[j]) if dr==-1 else (basis[j]-ent))/pip
        hit_opp=(l[j]<=lo[j]) if dr==-1 else (h[j]>=up[j])
        opp_pnl=((ent-lo[j]) if dr==-1 else (up[j]-ent))/pip
        if mode=="basis" and hit_basis: return basis_pnl
        elif mode=="opp_band" and hit_opp: return opp_pnl
        elif mode=="supertrend" and ((dr==-1 and c[j]>st[j]) or (dr==1 and c[j]<st[j])): return cur
        elif mode=="timedecay":
            if hit_basis: return basis_pnl
            if (j-i0)>=8: return cur
        elif mode=="trail_basis":
            if hit_basis: reached=True
            if reached and ((dr==-1 and c[j]>basis[j]) or (dr==1 and c[j]<basis[j])): return cur
        elif mode=="scaleout":
            if half1 is None and hit_basis: half1=basis_pnl
            if hit_opp: return 0.5*(half1 if half1 is not None else opp_pnl)+0.5*opp_pnl
        elif mode=="amddp":
            if peak_pnl>=MEAT and (peak_pnl-cur)>=GIVE*peak_pnl: return cur   # gave back half of a real profit
            if hit_basis and peak_pnl<MEAT: return basis_pnl                  # weak move -> take the mean
    px=c[jend]; base=((ent-px) if dr==-1 else (px-ent))/pip
    return 0.5*half1+0.5*base if (mode=="scaleout" and half1 is not None) else base

def entries(o,h,l,c,basis,sd):
    n=len(c); up=basis+K*sd; lo=basis-K*sd; pos=0; ei=0; ext=0; peak=0.0; stp=0.0; out=[]
    for i in range(SMA+2,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]): continue
        uo=l[i]>up[i]; do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            done=(pos==-1 and (h[i]>stp or l[i]<=basis[i])) or (pos==1 and (l[i]<stp or h[i]>=basis[i]))
            if done or (i-ei)>=TCAP: pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(e-basis[i])/PIP-MED>=MEAT:
                pos=-1; ei=i+1; stp=peak; out.append((i+1,-1,e,peak))
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-e)/PIP-MED>=MEAT:
                pos=1; ei=i+1; stp=peak; out.append((i+1,1,e,peak))
    return out

def main():
    global PIP, MED
    con=duckdb.connect(); rng=np.random.default_rng(0)
    med_sp={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df()
        med_sp[p]=float(np.nanmedian(s.s.values)/pip)
    res={m:[] for m in MODES}; mfe_all=[]; bd_all=[]
    for p,pip in PAIRS.items():
        PIP=pip; MED=med_sp[p]
        m=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/m5_ohlc/{p}_M5.parquet' ORDER BY timestamp").df()
        m["timestamp"]=pd.to_datetime(m["timestamp"],utc=True); m=m.set_index("timestamp")
        d=m.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; ts=d.index.values.astype("datetime64[ns]")
        basis=pd.Series(c).rolling(SMA).mean().values; sd=pd.Series(c).rolling(SMA).std().values
        up=basis+K*sd; lo=basis-K*sd; st=supertrend(h,l,c)
        for (i0,dr,ent,peak) in entries(o,h,l,c,basis,sd):
            jmax=min(i0+TCAP,len(c)-1)
            mfe=((ent-np.min(l[i0:jmax])) if dr==-1 else (np.max(h[i0:jmax])-ent))/pip
            bd=((ent-basis[i0]) if dr==-1 else (basis[i0]-ent))/pip
            mfe_all.append(mfe); bd_all.append(bd)
            for mode in MODES:
                res[mode].append((ts[i0],sim(mode,i0,dr,ent,peak,basis,up,lo,st,h,l,c,pip)-MED))
        del m; gc.collect()
    mfe_all=np.array(mfe_all); bd_all=np.array(bd_all)
    print(f"MFE DIAGNOSTIC (M15 fades, full {TCAP}-bar window): n={len(mfe_all)}")
    print(f"  reversion MFE median {np.nanmedian(mfe_all):.1f}p | basis-distance (current target) median {np.nanmedian(bd_all):.1f}p")
    print(f"  % MFE exceeds basis-distance (overshoots the mean): {100*np.nanmean(mfe_all>bd_all):.0f}%"
          f" | median overshoot when it does: {np.nanmedian((mfe_all-bd_all)[mfe_all>bd_all]):.1f}p\n")
    allts=np.sort([t for t,_ in res['basis']]); is_cut=allts[int(len(allts)*0.6)]
    print(f"EXIT BAKE-OFF — M15, 12 pairs, real median spread, {len(res['basis'])} trades (entries fixed)")
    print("="*94)
    print(f"  {'exit':>12} {'p/trade':>8} {'WR':>5} {'OOS p/t':>8} {'OOS WR':>7} {'p/yr':>7} {'OOS MCp':>8} {'WF folds (p/t)':>22}")
    for mode in MODES:
        t=sorted(res[mode],key=lambda x:x[0]); pnl=np.array([p for _,p in t]); ts_=np.array([x for x,_ in t])
        oos=pnl[ts_>=is_cut]; folds=[f.mean() for f in np.array_split(pnl,3)]
        null=np.array([(oos*rng.choice([-1.,1.],len(oos))).mean() for _ in range(2000)])
        print(f"  {mode:>12} {pnl.mean():>+8.2f} {100*(pnl>0).mean():>4.0f}% {oos.mean():>+8.2f} {100*(oos>0).mean():>6.0f}% "
              f"{pnl.sum()/5.3:>+7.0f} {(np.abs(null)>=abs(oos.mean())).mean():>8.3f}  [{' '.join(f'{x:+.1f}' for x in folds)}]")

if __name__=="__main__": main()
