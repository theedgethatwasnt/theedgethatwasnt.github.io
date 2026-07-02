"""
Directional 2-leg ZR driven by the POST-SHOCK RETRACE signal (the one signal
that passed the full gate on acct 009).  Quantifies: is "two shots at the
direction" (bet + one cover) better than "one shot" (bet + stop), and by how
much, on the SAME entries?

  leg-1 dir = fade(-sign vel) after a |z|>2.5 shock, market entry at t+peak+1.
  1-shot (ml=1): TP=tgt favorable; if price breaches zone (W against) -> cut.
  2-shot (ml=2): on adverse breach, add ONE martingale cover (opposite dir);
                 basket then wins on EITHER target; cut only on 2nd breach.

Same kernel, ml=1 vs ml=2 -> the delta is purely "the second shot."
S5 BA, real per-bar spread.  IS/OOS 70/30 + MC sign-shuffle on 2-shot OOS.
Entries are shock events (vol-gated by construction); cooldown=hold => non-overlap.
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path

PROJ = Path('/path/to/projects/fx-core')
PAIRS = ['GBP_JPY', 'USD_JPY', 'EUR_JPY', 'AUD_JPY']
PIP = 0.01
THR = 2.5
PEAK = 44
PF = 1.25
OOS_FRAC = 0.30
HOLD = 600          # S5 bars max hold / cooldown (non-overlap)
Z_WINDOW = 6
MAD_WIN = 2048


def compute_shock_z(close, pip, w=6, mad_win=2048):
    n = len(close)
    vel = np.empty(n); vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip
    vs = pd.Series(vel)
    rm = vs.rolling(mad_win, min_periods=50).median()
    ad = (vs - rm).abs()
    rmad = ad.rolling(mad_win, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


@njit
def sim_dir_zr(op, hi, lo, cl, sp, entry_dir, pip, pf, ml,
               zw_pips, tgt_pips, hold):
    """Cycle starts at bar i where entry_dir[i]!=0, in that direction.
       Returns per-cycle pnl array + leg counts + outcome codes.
       outcome: 1=leg1 win, 2=cover win, -1=cut(maxleg), 0=horizon expire."""
    n = len(cl)
    max_ev = n // 20 + 8
    pnl_out = np.zeros(max_ev); out_code = np.zeros(max_ev, dtype=np.int8)
    legn = np.zeros(max_ev, dtype=np.int8); t_idx = np.zeros(max_ev, dtype=np.int64)
    ev = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0
    while i < n:
        d = entry_dir[i]
        if d == 0:
            i += 1; continue
        entry = cl[i]
        if d == 1:   # long: zone below, target above
            uz = entry; lz = entry            # not used symmetric; define adverse=down
            adv = entry - zw_pips*pip         # adverse breach (down)
            t_fav = entry + tgt_pips*pip       # leg1 favorable target (up)
            t_cont = adv - tgt_pips*pip        # cover (short) continuation target (down)
        else:        # short: adverse up, favorable down
            adv = entry + zw_pips*pip
            t_fav = entry - tgt_pips*pip
            t_cont = adv + tgt_pips*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=entry
        nl=1; exited=False; start=i; code=0; pr=0.0
        i += 1
        end = min(start+hold, n)
        while i < end and not exited:
            h=hi[i]; l=lo[i]; c=cl[i]; s=sp[i]
            # favorable target (leg1 dir)
            if d==1 and h>=t_fav:
                net=0.0; tv=0.0
                for k in range(nl): net+=lv[k]*ld[k]*(t_fav-lp[k])/pip; tv+=lv[k]
                pr=net-tv*s; code=1 if nl==1 else 2; exited=True; break
            if d==-1 and l<=t_fav:
                net=0.0; tv=0.0
                for k in range(nl): net+=lv[k]*ld[k]*(t_fav-lp[k])/pip; tv+=lv[k]
                pr=net-tv*s; code=1 if nl==1 else 2; exited=True; break
            # cover continuation target (only meaningful after cover)
            if nl>=2:
                if d==1 and l<=t_cont:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(t_cont-lp[k])/pip; tv+=lv[k]
                    pr=net-tv*s; code=2; exited=True; break
                if d==-1 and h>=t_cont:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(t_cont-lp[k])/pip; tv+=lv[k]
                    pr=net-tv*s; code=2; exited=True; break
            # adverse breach
            adv_hit = (d==1 and l<=adv) or (d==-1 and h>=adv)
            if adv_hit:
                if nl>=ml:
                    # cut at market (close at adverse level approx = adv)
                    px = adv
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(px-lp[k])/pip; tv+=lv[k]
                    pr=net-tv*s; code=-1; exited=True; break
                else:
                    # place cover, martingale-sized
                    net_t=0.0; tv=0.0
                    for k in range(nl): net_t+=lv[k]*ld[k]*(adv-lp[k])/pip; tv+=lv[k]
                    net_t-=tv*s
                    vol=max(1.0, math.ceil(-net_t/tgt_pips*pf))
                    lv[nl]=vol; ld[nl]=float(-d); lp[nl]=adv; nl+=1
            i += 1
        if not exited:
            # horizon expire: close at last close
            px=cl[min(i,n-1)]; s=sp[min(i,n-1)]
            net=0.0; tv=0.0
            for k in range(nl): net+=lv[k]*ld[k]*(px-lp[k])/pip; tv+=lv[k]
            pr=net-tv*s; code=0
        pnl_out[ev]=pr; out_code[ev]=code; legn[ev]=nl; t_idx[ev]=start; ev+=1
        i = start + hold   # cooldown = hold (non-overlap)
    return pnl_out[:ev], out_code[:ev], legn[:ev], t_idx[:ev]


def mc_signflip(pnls, n=2000):
    if len(pnls) < 5: return 1.0
    actual = pnls.sum()
    rng = np.random.default_rng(42)
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], size=len(pnls))
        if (pnls*s).sum() >= actual: cnt += 1
    return cnt / n


print(f"2-leg retrace ZR  thr={THR} peak={PEAK} hold={HOLD}  (ml=1 one-shot vs ml=2 two-shot)\n")
ZW, TGT = 20.0, 15.0   # zone width / target pips (retrace tp~20, zone~adverse)
for ZW, TGT in [(15.0,15.0),(20.0,15.0),(25.0,20.0)]:
    print(f"################  ZW={ZW:.0f}  TGT={TGT:.0f}  ################")
    print(f"  {'pair':8s} {'ml':>2} | {'IS p/d':>7} {'OOS p/d':>7} {'trades':>6} {'leg1W%':>6} {'covW%':>6} {'cut%':>5} {'avgW':>6} {'avgL':>7} {'mcOOS':>6}")
    for pair in PAIRS:
        df = pd.read_parquet(PROJ/'data'/'s5_ba'/f'{pair}_S5_BA.parquet',
                             columns=['timestamp','open','high','low','close','bid_c','ask_c'])
        op=df.open.to_numpy(float); hi=df.high.to_numpy(float)
        lo=df.low.to_numpy(float); cl=df.close.to_numpy(float)
        sp=((df.ask_c-df.bid_c)/PIP).to_numpy(float)
        sp=np.where(np.isfinite(sp)&(sp>0), sp, np.nanmedian(sp))
        n=len(cl); is_end=int(n*(1-OOS_FRAC))
        span=n*5/(60*60*24*5/7); is_days=is_end*5/(60*60*24*5/7); oos_days=span-is_days

        z, vel = compute_shock_z(cl, PIP)
        shock = np.abs(z) > THR
        entry_dir = np.zeros(n, dtype=np.float64)
        # place fade entries at t+peak+1, cooldown handled in kernel via hold
        last = -10**9
        for t in range(Z_WINDOW, n-PEAK-HOLD-2):
            if not shock[t]: continue
            if t - last < HOLD: continue
            ws = t+PEAK+1
            if ws>=n: continue
            entry_dir[ws] = -1.0 if vel[t]>0 else 1.0   # fade
            last = t
        # warmup
        _=sim_dir_zr(op[:5000],hi[:5000],lo[:5000],cl[:5000],sp[:5000],entry_dir[:5000],PIP,PF,2,ZW,TGT,HOLD)
        for ml in (1,2):
            pn, code, legn, tix = sim_dir_zr(op,hi,lo,cl,sp,entry_dir,PIP,PF,ml,ZW,TGT,HOLD)
            ism = tix < is_end; oosm = ~ism
            isppd = pn[ism].sum()/is_days; oosppd = pn[oosm].sum()/oos_days
            ntr=len(pn)
            l1w=(code==1).mean()*100; cvw=(code==2).mean()*100; cut=(code==-1).mean()*100
            wins=pn[pn>0]; loss=pn[pn<0]
            aw=wins.mean() if len(wins) else 0.0; al=loss.mean() if len(loss) else 0.0
            mcp = mc_signflip(pn[oosm]) if ml==2 else np.nan
            flag=' 🟢' if oosppd>0 else ' 🔴'
            mcs = f'{mcp:.3f}' if ml==2 else '  -  '
            print(f"  {pair:8s} {ml:>2} | {isppd:>7.2f} {oosppd:>7.2f} {ntr:>6} {l1w:>5.0f}% {cvw:>5.0f}% {cut:>4.0f}% {aw:>6.1f} {al:>7.1f} {mcs:>6}{flag}")
    print()
