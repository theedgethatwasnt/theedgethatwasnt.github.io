"""
bb_fade.py — ONE CODE PATH for the BB re-entry fade (SOP R6): the exact same signal+exit logic
runs in backtest (batch loop) and live (incremental, one bar per poll). The R7 consistency test
(tests/r7_bb_fade.py) replays history through both and asserts identical trades.

Rule: basis=SMA(close,9), bands=basis +/- K*std(close,9). ENTRY = previous bar fully outside the
band AND current bar touches it (re-entry), gated by projected half-distance meat (>= MEAT net
spread). EXIT (opp_band) = price reaches the opposite band; STOP = extension peak; TIME CAP.
Entry fills at the NEXT bar open (causal). Mid OHLC for signal; spread deducted explicitly (R3).
"""
from collections import deque
import numpy as np

SMA=9; K=1.0

def bands(closes):
    """basis, std, upper, lower from the trailing window (incl. current close). None if < SMA."""
    if len(closes)<SMA: return None
    w=np.array(list(closes)[-SMA:]); b=w.mean(); s=w.std()
    return b,s,b+K*s,b-K*s

class BBFadeProcessor:
    """Incremental, causal. Call update(o,h,l,c) once per CLOSED bar; returns an action dict or None.
    Tracks position internally; entry fills at the next bar's open (handled on the following call)."""
    def __init__(self, pip, spread, meat, tcap):
        self.pip=pip; self.spread=spread; self.meat=meat; self.tcap=tcap
        self.closes=deque(maxlen=SMA)
        self.prev=None            # (up,lo,low,high) of the previous bar
        self.ext=0; self.peak=0.0
        self.pos=0; self.entry=0.0; self.ei=0; self.stop=0.0
        self.bar=-1; self.pending=None   # (dir, stop) to fill at next open
        self.trades=[]
        self.cur_basis=None; self.cur_up=None; self.cur_lo=None  # latest band (for S5 exit monitoring)

    def update(self, o,h,l,c):
        self.bar+=1; act=None
        # (1) fill a pending entry at THIS bar's open
        if self.pending is not None:
            d,stp=self.pending; self.pending=None
            # entry-validity gate (R-fix 2026-06-25): only enter if the fill is on the PROTECTIVE side of
            # the stop (short: open<peak; long: open>peak). If the fill has already passed the protrusion
            # peak, the fade is pre-invalidated — skip it. Entering past the stop was the phantom-fill bug
            # (the exit then booked a profit at a stop level below entry that price never reached).
            if (d==-1 and o<stp) or (d==1 and o>stp):
                self.pos=d; self.entry=o; self.ei=self.bar; self.stop=stp
                act={"type":"ENTER","dir":d,"price":o,"bar":self.ei}
        # (2) rolling bands incl. this close
        self.closes.append(c); bd=bands(self.closes)
        if bd is None: self.prev=None; return act
        basis,sd,up,lo=bd
        self.cur_basis=basis; self.cur_up=up; self.cur_lo=lo
        # (3) extension tracking (uses this bar's high/low vs its bands)
        uo=l>up; do=h<lo
        if uo: self.peak=h if self.ext!=1 else max(self.peak,h); self.ext=1
        elif do: self.peak=l if self.ext!=-1 else min(self.peak,l); self.ext=-1
        # (4) manage open position -> exit? (exit CAN fire on the fill bar: after the open fill,
        #     the same bar's range can hit target/stop — matches the batch path)
        if self.pos!=0:
            ex=None
            if self.pos==-1:
                if h>self.stop: ex=self.stop
                elif l<=lo: ex=lo
            else:
                if l<self.stop: ex=self.stop
                elif h>=up: ex=up
            if ex is None and (self.bar-self.ei)>=self.tcap: ex=c
            if ex is not None:
                pnl=self.pos*(ex-self.entry)/self.pip-self.spread
                self.trades.append((self.ei,self.bar,self.pos,pnl));
                act={"type":"EXIT","price":ex,"pnl":pnl,"entry_bar":self.ei,"bar":self.bar}; self.pos=0
        # (5) entry signal (prev bar fully outside + this bar touches), if flat & nothing pending
        if self.pos==0 and self.pending is None and self.prev is not None:
            pup,plo,plow,phigh=self.prev
            # gate on the CURRENT CLOSE (causal — known at signal time), entry fills next open
            if plow>pup and l<=up and 0.5*(c-basis)/self.pip-self.spread>=self.meat:
                self.pending=(-1,self.peak)
            elif phigh<plo and h>=lo and 0.5*(basis-c)/self.pip-self.spread>=self.meat:
                self.pending=(1,self.peak)
        self.prev=(up,lo,l,h)
        return act

def backtest(o,h,l,c,pip,spread,meat,tcap):
    """Batch reference (the validated path). Returns trades as (entry_bar, exit_bar, dir, pnl)."""
    n=len(c); cl=np.array(c)
    basis=np.full(n,np.nan); sd=np.full(n,np.nan)
    for i in range(SMA-1,n):
        w=cl[i-SMA+1:i+1]; basis[i]=w.mean(); sd[i]=w.std()
    up=basis+K*sd; lo=basis-K*sd
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0; tr=[]
    for i in range(SMA,n-1):
        if np.isnan(basis[i]): continue
        uo=l[i]>up[i]; do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stp: ex=stp
                elif l[i]<=lo[i]: ex=lo[i]
            else:
                if l[i]<stp: ex=stp
                elif h[i]>=up[i]: ex=up[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex): tr.append((ei,i,pos,pos*(ex-ent)/pip-spread)); pos=0
        if pos==0:
            e=o[i+1]   # entry fills at next open; meat gate uses the CLOSE (causal, no lookahead)
            # entry-validity gate (R-fix 2026-06-25): only enter if the fill is on the PROTECTIVE side of
            # the stop (short: e<peak; long: e>peak); else the fade is pre-invalidated. Entering past the
            # stop was the phantom-fill bug (exit booked a profit at a stop below entry price never reached).
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-spread>=meat and e<peak: pos=-1; ent=e; ei=i+1; stp=peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-spread>=meat and e>peak: pos=1; ent=e; ei=i+1; stp=peak
    return tr
