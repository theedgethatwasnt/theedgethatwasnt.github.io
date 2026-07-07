#!/usr/bin/env python3
"""
Equity-MA switch — PAPER-TRACKED live risk overlay (prototype; does NOT gate real orders).

For each live account it reads the realized-P&L equity curve from trades.duckdb, puts a moving
average (W trades) on it, and determines a causal LIVE / WIND-DOWN state (LIVE while equity is
above its MA, WIND-DOWN when it rolls under). It tracks what the account's equity WOULD be if the
switch were active (paper-switched), logs the comparison, and Telegram-alerts on a state FLIP.
This validates the overlay on the live tape before it is ever allowed to gate orders.

Run periodically (hourly cron). Env: EQSW_W (default 10), EQSW_ALWAYS=1 to always send the summary.
"""
import os, json, shutil, sys
import numpy as np
import pandas as pd

DB = '/data/db/trades.duckdb'
STATE = '/data/db/equity_switch_state.json'
W = int(os.environ.get('EQSW_W', '10'))
ACCTS = [('010 SMA-Stack', 'sma_stack%'), ('001 SMA-fade', 'sma_fade%')]


def maxdd(e):
    if len(e) == 0:
        return 0.0
    return float((e - np.maximum.accumulate(e)).min())


def connect():
    import duckdb
    try:
        return duckdb.connect(DB, read_only=True)
    except Exception:                        # writer holds a lock → read a snapshot copy
        tmp = '/tmp/_eqsw_snap.duckdb'
        shutil.copy(DB, tmp)
        return duckdb.connect(tmp, read_only=True)


def main():
    con = connect()
    try:
        prev = json.load(open(STATE))
    except Exception:
        prev = {}
    newstate = {}; lines = []; flips = []
    for name, lab in ACCTS:
        rows = con.execute(
            "SELECT exit_time, pnl_pips FROM trades WHERE is_paper=false AND label LIKE ? "
            "AND pnl_pips IS NOT NULL ORDER BY exit_time", [lab]).fetchall()
        pnl = np.array([r[1] for r in rows], float); n = len(pnl)
        if n < W + 2:
            lines.append(f"{name}: {n} closed trades — need >{W+2} for MA({W}); tracking, no signal yet")
            newstate[name] = 'n/a'; continue
        eq = np.cumsum(pnl)
        ma = pd.Series(eq).rolling(W).mean().to_numpy()
        live = np.ones(n, bool); st = True
        for i in range(n):
            if i >= W and not np.isnan(ma[i - 1]):
                st = eq[i - 1] > ma[i - 1]
            live[i] = st
        sw = np.cumsum(np.where(live, pnl, 0.0))
        cur = 'LIVE' if eq[-1] > ma[-1] else 'WIND-DOWN'
        newstate[name] = cur
        lines.append(f"{name}: {n}t  state={cur}  | raw net={eq[-1]:.0f}p DD={maxdd(eq):.0f} "
                     f"| would-be switched net={sw[-1]:.0f}p DD={maxdd(sw):.0f}  %live={100*live.mean():.0f}")
        if prev.get(name) not in (None, 'n/a', cur):
            flips.append(f"⚠️ {name}: {prev.get(name)} → {cur}")
    try:
        json.dump(newstate, open(STATE, 'w'))
    except Exception:
        pass
    header = f"📊 Equity-MA switch (paper-tracked, W={W}) — signals only, NOT gating orders"
    msg = header + "\n" + "\n".join((flips + lines) if flips else lines)
    print(msg, flush=True)
    tok = os.environ.get('TELEGRAM_BOT_TOKEN'); chat = os.environ.get('TELEGRAM_CHAT_ID')
    if tok and chat and (flips or os.environ.get('EQSW_ALWAYS')):
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({'chat_id': chat, 'text': msg}).encode()
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=10)
        except Exception as e:
            print(f"telegram send failed: {e}", flush=True)


if __name__ == '__main__':
    main()
