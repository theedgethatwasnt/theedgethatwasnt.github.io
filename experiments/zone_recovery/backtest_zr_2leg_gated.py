"""
Two-leg Zone Recovery (initial bet + ONE cover), bounded loss, with a
volatility / efficiency ENTRY GATE.

Motivation:
  - Classic ZR edge is monotonic in leg depth (ML=4 = -112 vp/d, ML=10 = +7158).
    Capping at 2 legs strips the deep-recovery edge -> needs an entry edge.
  - User idea: only open in HIGH volatility (clusters -> zone-width clears,
    cycle resolves instead of stalling in calm chop).
  - Counter-idea tested alongside: EFFICIENCY gate (trendiness), since what ZR
    actually needs is net displacement, not raw magnitude.

Structure per cycle: leg1 at close. If price breaches the far zone boundary and
the basket is underwater, add ONE cover (leg2, martingale-sized). If a SECOND
breach happens (nl == ml == 2), close at market = bounded loss. Target hit any
time = win. Real per-bar M5 spread charged on every leg (R3).

IS/OOS 70/30, thresholds IS-frozen. Reports p/d IS & OOS, win/coverloss split.
"""
import numpy as np, pandas as pd, math, sys
from numba import njit
from pathlib import Path

PAIRS = ['USD_JPY', 'EUR_JPY', 'GBP_USD', 'EUR_USD']
ML = 2
PF = 1.25
OOS_FRAC = 0.30
LOOKBACK = 48          # M5 bars (~4h) for regime features


@njit
def sim_2leg(op, hi, lo, cl, sp, pip, pf, ml, gate, zw_pips, tgt_pips):
    n = len(cl)
    total = 0.0; nc = 0; nwin = 0; ncover_loss = 0; sl = 0.0
    cover_loss_sum = 0.0; win_sum = 0.0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; direction = 1
    while i < n:
        if not gate[i]:
            i += 1; continue
        entry = cl[i]
        if direction == 1:
            uz = entry; lz = entry - zw_pips*pip
            ut = entry + tgt_pips*pip; lt = lz - tgt_pips*pip
        else:
            lz = entry; uz = entry + zw_pips*pip
            lt = entry - tgt_pips*pip; ut = uz + tgt_pips*pip
        lv[0]=1.0; ld[0]=float(direction); lp[0]=entry
        nl=1; lu=ll=-1; exited=False
        i += 1
        while i < n and not exited:
            h=hi[i]; l=lo[i]; c=cl[i]; s=sp[i]; bull = c>=op[i]
            for pass_n in range(2):
                if exited: break
                do_hi = (bull and pass_n==0) or (not bull and pass_n==1)
                if l <= ut <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    pr=net-tv*s; total+=pr; nc+=1; nwin+=1; win_sum+=pr; sl+=nl; exited=True; break
                if l <= lt <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    pr=net-tv*s; total+=pr; nc+=1; nwin+=1; win_sum+=pr; sl+=nl; exited=True; break
                if do_hi and h >= uz and lu != i:
                    lu=i
                    net_t=0.0; tv=0.0
                    for k in range(nl): net_t+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net_t-=tv*s
                    if net_t>=0:
                        if c>=ut: total+=net_t; nc+=1; nwin+=1; win_sum+=net_t; sl+=nl; exited=True; break
                    else:
                        vol=max(1.0, math.ceil(-net_t/tgt_pips*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            pr=net-tv*s; total+=pr; nc+=1; ncover_loss+=1; cover_loss_sum+=pr; sl+=nl; exited=True; break
                        lv[nl]=vol; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not do_hi and l <= lz and ll != i:
                    ll=i
                    net_t=0.0; tv=0.0
                    for k in range(nl): net_t+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net_t-=tv*s
                    if net_t>=0:
                        if c<=lt: total+=net_t; nc+=1; nwin+=1; win_sum+=net_t; sl+=nl; exited=True; break
                    else:
                        vol=max(1.0, math.ceil(-net_t/tgt_pips*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            pr=net-tv*s; total+=pr; nc+=1; ncover_loss+=1; cover_loss_sum+=pr; sl+=nl; exited=True; break
                        lv[nl]=vol; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1
        direction = -direction
    avg_legs = sl/max(nc,1)
    avg_win = win_sum/max(nwin,1)
    avg_cl  = cover_loss_sum/max(ncover_loss,1)
    return total, nc, nwin, ncover_loss, avg_legs, avg_win, avg_cl


def build_gates(cl, ret, is_end, pip):
    """Return dict name->bool gate array, thresholds frozen on IS only."""
    n = len(cl)
    s = pd.Series(ret)
    vol = s.rolling(LOOKBACK).std().to_numpy()                       # realized vol
    absmove = s.abs().rolling(LOOKBACK).sum().to_numpy()
    netmove = np.abs(pd.Series(cl).diff().rolling(LOOKBACK).sum().to_numpy()/pip)
    eff = netmove / np.where(absmove>0, absmove, np.nan)             # efficiency ratio 0..1
    gates = {}
    ones = np.ones(n, dtype=np.bool_)
    ones[:LOOKBACK] = False
    gates['ungated'] = ones
    for nm, arr in [('vol', vol), ('eff', eff)]:
        a = arr[:is_end]; a = a[np.isfinite(a)]
        for pct, lbl in [(50, 'P50'), (75, 'P75')]:
            thr = np.percentile(a, pct)
            g = np.isfinite(arr) & (arr >= thr)
            g[:LOOKBACK] = False
            gates[f'{nm}>{lbl}'] = g
    return gates


for pair in PAIRS:
    pip = 0.01 if 'JPY' in pair else 0.0001
    df = pd.read_parquet(f'/path/to/projects/fx-core/data/m5_ohlc/{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    ba = pd.read_parquet(f'/path/to/projects/fx-core/data/m5_ba/{pair}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
    # align spread to df length (M5 BA has bid_c/ask_c)
    sp_full = ((ba['ask_c'] - ba['bid_c']) / pip).to_numpy(float)
    m = min(len(df), len(sp_full))
    op=df.open.to_numpy(float)[:m]; hi=df.high.to_numpy(float)[:m]
    lo=df.low.to_numpy(float)[:m]; cl=df.close.to_numpy(float)[:m]
    sp=sp_full[:m]
    sp = np.where(np.isfinite(sp) & (sp>0), sp, np.nanmedian(sp))
    ret = np.diff(cl, prepend=cl[0]) / pip
    n=m; is_end=int(n*(1-OOS_FRAC))
    span = n*5/(60*24*5/7); is_days=is_end*5/(60*24*5/7); oos_days=span-is_days

    gates = build_gates(cl, ret, is_end, pip)
    # warmup
    _=sim_2leg(op[:3000],hi[:3000],lo[:3000],cl[:3000],sp[:3000],pip,PF,ML,gates['ungated'][:3000],30.0,15.0)

    # pick a representative config (zw, tgt_f). small grid, report best-OOS per gate.
    print('='*78)
    print(f'{pair}  median spread={np.median(sp):.2f}p  IS={is_days:.0f}d OOS={oos_days:.0f}d  (2-leg, ml={ML})')
    print(f'  {"gate":10s} {"zw":>3} {"tf":>4} | {"IS p/d":>8} {"OOS p/d":>8} {"c/d":>6} {"win%":>5} {"avgW":>6} {"avgCoverL":>9}')
    for gname, g in gates.items():
        best=None
        for zw in [20.0,30.0,40.0]:
            for tf in [0.5,1.0]:
                tgt=zw*tf
                # IS
                gi=g.copy(); gi[is_end:]=False
                ti,nci,nwi,ncli,al,aw,acl = sim_2leg(op,hi,lo,cl,sp,pip,PF,ML,gi,zw,tgt)
                # OOS
                go=g.copy(); go[:is_end]=False
                to,nco,nwo,nclo,alo,awo,aclo = sim_2leg(op,hi,lo,cl,sp,pip,PF,ML,go,zw,tgt)
                isppd=ti/is_days; oosppd=to/oos_days
                cand=(isppd,oosppd,nco/oos_days, nwo/max(nco,1)*100, awo, aclo, zw, tf)
                # choose config by IS p/d (no OOS peeking)
                if best is None or isppd>best[0]: best=cand
        ip,op_,cd,wr,aw,acl,zw,tf=best
        flag=' 🟢' if op_>0 else ' 🔴'
        print(f'  {gname:10s} {zw:>3.0f} {tf:>4.2f} | {ip:>8.1f} {op_:>8.1f} {cd:>6.2f} {wr:>4.0f}% {aw:>6.1f} {acl:>9.1f}{flag}')
