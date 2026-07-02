"""
ASI trendline-swing (third-touch) strategy on H1.

Spec (user):
  - swings on the ASI line; uptrend = higher highs AND higher lows
  - draw trendline through the previous two swing lows
  - enter LONG when ASI pulls back to touch that ascending line (third touch)
  - protective stop below the prior swing low (price)
  - exit when the trendline is broken; mirror for short

Causal: ASI pivots confirmed W bars after the pivot (no lookahead). Trendline,
entry, stop, exit all use only confirmed-by-t information. Real H1 spread.
All 12 pairs, IS/OOS 70/30 + 3-chunk WF on IS.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, '/path/to/projects/fx-core')
from lib.asi_indicator import compute_asi

PROJ = Path('/path/to/projects/fx-core')
PAIRS = ['AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY','EUR_GBP','EUR_JPY',
         'EUR_USD','GBP_JPY','GBP_USD','NZD_JPY','NZD_USD','USD_JPY']
W = 3                 # pivot half-width (H1 bars)
OOS_FRAC = 0.30


def pivots(x, W):
    """Confirmed swing lows/highs: x[i] strict min/max over [i-W, i+W].
       Returns sorted arrays of indices for lows and highs."""
    n = len(x); los = []; his = []
    for i in range(W, n - W):
        seg = x[i-W:i+W+1]
        if x[i] == seg.min() and (seg == x[i]).sum() == 1: los.append(i)
        if x[i] == seg.max() and (seg == x[i]).sum() == 1: his.append(i)
    return np.array(los, dtype=np.int64), np.array(his, dtype=np.int64)


def backtest(o, h, l, c, sp, pip, asi, tol_k, brk_k):
    """ASI = swing FILTER (pivot detection only). Trendline, touch, stop, break
       all measured on PRICE. Returns per-trade pnl (pips) and entry-bar index."""
    n = len(c)
    lo_idx, hi_idx = pivots(asi, W)        # clean swing structure from ASI
    lo_conf = lo_idx + W; hi_conf = hi_idx + W
    pnls = []; eidx = []; brk_list = []
    pos = 0
    entry_px = stop_px = slope = anchor_v = 0.0; anchor_t = 0; chan0 = 1.0
    last_setup = -1
    li = hi = 0
    for t in range(W+2, n):
        while li < len(lo_conf) and lo_conf[li] <= t: li += 1
        while hi < len(hi_conf) and hi_conf[hi] <= t: hi += 1
        if pos == 0:
            if li >= 2 and hi >= 2:
                B, C = lo_idx[li-2], lo_idx[li-1]      # swing-low bars (from ASI)
                Ha, Hb = hi_idx[hi-2], hi_idx[hi-1]    # swing-high bars (from ASI)
                # PRICE values at those swing bars
                lB, lC = l[B], l[C]; hA, hB = h[Ha], h[Hb]
                # LONG: higher lows + higher highs in PRICE; ascending price line
                if lC > lB and hB > hA and C > B:
                    s = (lC - lB) / (C - B)            # price slope
                    chan = hB - lC                     # price channel height
                    if s > 0 and chan > 0:
                        Lval = lC + s * (t - C)         # projected support, price
                        tol = tol_k * chan; brk = brk_k * chan
                        # price pulled back to touch the line, closed above the break
                        if l[t] <= Lval + tol and c[t] >= Lval - brk and C != last_setup:
                            pos = 1; entry_px = c[t] + sp[t]*pip/2
                            stop_px = l[C]              # below prior swing low (price)
                            slope = s; anchor_v = lC; anchor_t = C; chan0 = chan; last_setup = C
                            # ASI broke into NEW TERRITORY above prior ASI swing high?
                            broke = 1 if (t > Hb and asi[Hb+1:t+1].max() > asi[Hb]) else 0
                            eidx.append(t); brk_list.append(broke)
                # SHORT: lower highs + lower lows in PRICE; descending price line
                elif hB < hA and lC < lB and Hb > Ha:
                    sH = (hB - hA) / (Hb - Ha)
                    chanH = hA - lC if (hA - lC) > 0 else (hB - l[C])
                    if sH < 0 and chanH > 0:
                        LvalH = hB + sH * (t - Hb)
                        tol = tol_k * chanH; brk = brk_k * chanH
                        if h[t] >= LvalH - tol and c[t] <= LvalH + brk and Hb != last_setup:
                            pos = -1; entry_px = c[t] - sp[t]*pip/2
                            stop_px = h[Hb]
                            slope = sH; anchor_v = hB; anchor_t = Hb; chan0 = chanH; last_setup = Hb
                            # ASI broke into NEW TERRITORY below prior ASI swing low?
                            broke = 1 if (t > C and asi[C+1:t+1].min() < asi[C]) else 0
                            eidx.append(t); brk_list.append(broke)
        else:
            Lnow = anchor_v + slope * (t - anchor_t)
            brk = brk_k * chan0
            exit_now = False
            if pos == 1:
                if l[t] <= stop_px:
                    xpx = stop_px; exit_now = True
                elif c[t] < Lnow - brk:               # price broke trendline
                    xpx = c[t]; exit_now = True
                if exit_now:
                    pnls.append((xpx - sp[t]*pip/2 - entry_px)/pip); pos = 0
            else:
                if h[t] >= stop_px:
                    xpx = stop_px; exit_now = True
                elif c[t] > Lnow + brk:
                    xpx = c[t]; exit_now = True
                if exit_now:
                    pnls.append((entry_px - (xpx + sp[t]*pip/2))/pip); pos = 0
    if pos == 1:
        pnls.append((c[-1] - sp[-1]*pip/2 - entry_px)/pip)
    elif pos == -1:
        pnls.append((entry_px - (c[-1] + sp[-1]*pip/2))/pip)
    return np.array(pnls), np.array(eidx, dtype=np.int64), np.array(brk_list, dtype=np.int64)


print("ASI trendline-swing H1  (12 pairs, IS/OOS + WF)\n")
TOL_K, BRK_K = 0.25, 0.20
print(f"  params: W={W} tol_k={TOL_K} brk_k={BRK_K}  (TP=none, ride trend)")
print(f"  sub-categories: BREAKOUT = ASI broke into new territory past prior swing; NOBREAK = did not\n")
print(f"  {'pair':8s} {'subcat':8s} | {'trades':>6} {'IS p/d':>7} {'OOS p/d':>8} {'WF+/3':>6} {'WR%':>5} {'avgW':>6} {'avgL':>7}")
agg = {}  # subcat -> [oos_pips_total]
agg_oos = {'ALL':0.0, 'BREAKOUT':0.0, 'NOBREAK':0.0}
for pair in PAIRS:
    pip = 0.01 if 'JPY' in pair else 0.0001
    m5 = pd.read_parquet(PROJ/'data'/'m5_ohlc'/f'{pair}_M5.parquet').sort_values('timestamp').set_index('timestamp')
    ba = pd.read_parquet(PROJ/'data'/'m5_ba'/f'{pair}_M5_BA.parquet').sort_values('timestamp').set_index('timestamp')
    m5['spread'] = (ba['ask_c'] - ba['bid_c']).reindex(m5.index).values / pip
    h1 = pd.DataFrame({
        'open': m5['open'].resample('1h').first(), 'high': m5['high'].resample('1h').max(),
        'low': m5['low'].resample('1h').min(), 'close': m5['close'].resample('1h').last(),
        'spread': m5['spread'].resample('1h').mean()}).dropna()
    o=h1.open.values; h=h1.high.values; l=h1.low.values; c=h1.close.values
    spv=np.where(np.isfinite(h1.spread.values)&(h1.spread.values>0), h1.spread.values, np.nanmedian(h1.spread.values))
    n=len(c); asi=compute_asi(o,h,l,c,n)
    days = n/24*5/7
    pnls,eidx,brkf = backtest(o,h,l,c,spv,pip,asi,TOL_K,BRK_K)
    if len(pnls)==0:
        print(f"  {pair:8s} {'-':8s} | {'0':>6}   no trades"); continue
    is_end=int(n*(1-OOS_FRAC)); is_days=is_end/24*5/7; oos_days=days-is_days
    for sub, mask in [('ALL', np.ones(len(pnls),bool)),
                      ('BREAKOUT', brkf==1), ('NOBREAK', brkf==0)]:
        p=pnls[mask]; e=eidx[mask]
        if len(p)==0: continue
        ism=e<is_end; oosm=~ism
        isppd=p[ism].sum()/is_days; oosppd=p[oosm].sum()/oos_days
        wf=0
        for ch in range(3):
            a=int(is_end*ch/3); b=int(is_end*(ch+1)/3)
            cm=(e>=a)&(e<b)
            if cm.sum()>0 and p[cm].sum()>0: wf+=1
        wr=(p>0).mean()*100; aw=p[p>0].mean() if (p>0).any() else 0; al=p[p<0].mean() if (p<0).any() else 0
        flag=' 🟢' if oosppd>0 else ' 🔴'
        agg_oos[sub]+=p[oosm].sum()/oos_days
        print(f"  {pair:8s} {sub:8s} | {len(p):>6} {isppd:>7.2f} {oosppd:>8.2f} {wf:>5}/3 {wr:>4.0f}% {aw:>6.1f} {al:>7.1f}{flag}")
    print()
print(f"  portfolio OOS p/d (sum across pairs):  ALL={agg_oos['ALL']:+.1f}   BREAKOUT={agg_oos['BREAKOUT']:+.1f}   NOBREAK={agg_oos['NOBREAK']:+.1f}")
