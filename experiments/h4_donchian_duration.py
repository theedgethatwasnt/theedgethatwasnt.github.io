"""
H4 Donchian Strategy — Win/Loss Duration Analysis
Pairs: GBP_JPY, USD_JPY, EUR_JPY, GBP_USD
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/path/to/projects/fx-core/data/m5_ba")

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "GBP_USD"]

PIP = {"GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01, "GBP_USD": 0.0001}
SP_GATE = {"GBP_JPY": 3.80, "USD_JPY": 2.00, "EUR_JPY": 2.40, "GBP_USD": 2.30}

DON_N     = 10
ATR_PERIOD = 14
ATR_TRAIL  = 1.0
WARMUP     = DON_N + ATR_PERIOD   # 24 H4 bars

# ── Duration buckets ─────────────────────────────────────────────────────────
BUCKET_EDGES  = [0, 1, 2, 6, 12, 30, 60, 120]   # in H4 bars
BUCKET_LABELS = [
    "<4h (1b)",
    "4-8h (2b)",
    "8-24h (2-6b)",
    "24-48h (6-12b)",
    "2-5d (12-30b)",
    "5-10d (30-60b)",
    "10-20d (60-120b)",
    ">20d (>120b)",
]


def resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    h4 = df.resample("4h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        bid_c=("bid_c", "last"),
        ask_c=("ask_c", "last"),
    ).dropna()
    return h4.reset_index()


def wilder_atr(high, low, close, period=14):
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    atr = np.empty(n)
    atr[0] = tr[0]
    # seed with simple average up to period
    for i in range(1, min(period, n)):
        atr[i] = (atr[i-1] * i + tr[i]) / (i + 1)
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def donchian_causal(high, low, n):
    """hh[i] = max(high[i-n:i])  (shift-1 rolling n max, causal)"""
    hh = np.full(len(high), np.nan)
    ll = np.full(len(low),  np.nan)
    for i in range(n, len(high)):
        hh[i] = np.max(high[i-n:i])
        ll[i] = np.min(low[i-n:i])
    return hh, ll


def backtest_pair(pair: str) -> list[dict]:
    pip  = PIP[pair]
    gate = SP_GATE[pair]

    raw = pd.read_parquet(DATA_DIR / f"{pair}_M5_BA.parquet")
    h4  = resample_h4(raw)

    o = h4["open"].values.astype(float)
    h = h4["high"].values.astype(float)
    l = h4["low"].values.astype(float)
    c = h4["close"].values.astype(float)
    bid_c = h4["bid_c"].values.astype(float)
    ask_c = h4["ask_c"].values.astype(float)

    sp  = (ask_c - bid_c) / pip
    atr = wilder_atr(h, l, c, ATR_PERIOD)
    hh, ll = donchian_causal(h, l, DON_N)

    trades = []

    pos      = 0          # 0, 1 (long), -1 (short)
    entry_px = 0.0
    entry_i  = 0
    peak_px  = 0.0
    trail_stop = 0.0

    def open_trade(direction, bar_i):
        nonlocal pos, entry_px, entry_i, peak_px, trail_stop
        pos = direction
        entry_px = o[bar_i]
        entry_i  = bar_i
        if direction == 1:
            peak_px    = h[bar_i]
            trail_stop = peak_px - ATR_TRAIL * atr[bar_i]
        else:
            peak_px    = l[bar_i]
            trail_stop = peak_px + ATR_TRAIL * atr[bar_i]

    def close_trade(exit_px, bar_i, reason):
        nonlocal pos
        if pos == 1:
            pnl = (exit_px - entry_px) / pip - sp[bar_i]
        else:
            pnl = (entry_px - exit_px) / pip - sp[bar_i]
        dur = bar_i - entry_i
        trades.append(dict(
            pair=pair, pnl_pips=pnl,
            duration_bars=dur, duration_hours=dur * 4,
            win=(pnl > 0), exit_reason=reason,
            entry_i=entry_i, exit_i=bar_i,
        ))
        pos = 0

    for i in range(WARMUP, len(o)):
        if np.isnan(hh[i]) or np.isnan(ll[i]) or atr[i] <= 0:
            continue

        spread_ok = sp[i] <= gate

        if pos == 0:
            if not spread_ok:
                continue
            if c[i] > hh[i]:
                open_trade(1, i)
            elif c[i] < ll[i]:
                open_trade(-1, i)
            continue

        # ── Update trail FIRST ────────────────────────────────────────────
        if pos == 1:
            if h[i] > peak_px:
                peak_px    = h[i]
                trail_stop = peak_px - ATR_TRAIL * atr[i]
        else:
            if l[i] < peak_px:
                peak_px    = l[i]
                trail_stop = peak_px + ATR_TRAIL * atr[i]

        # ── Exit checks ───────────────────────────────────────────────────
        exited = False
        reverse_dir = 0

        if pos == 1:
            if l[i] <= trail_stop:
                close_trade(min(o[i], trail_stop), i, "trail")
                exited = True
            elif c[i] < ll[i]:
                close_trade(o[i], i, "reverse")
                exited = True
                reverse_dir = -1
        else:
            if h[i] >= trail_stop:
                close_trade(max(o[i], trail_stop), i, "trail")
                exited = True
            elif c[i] > hh[i]:
                close_trade(o[i], i, "reverse")
                exited = True
                reverse_dir = 1

        # ── Immediate re-entry on reverse ─────────────────────────────────
        if exited and reverse_dir != 0 and spread_ok:
            open_trade(reverse_dir, i)

    return trades


# ──────────────────────────────────────────────────────────────────────────────
def bucket_distribution(durations: pd.Series) -> pd.Series:
    counts = pd.cut(
        durations,
        bins=BUCKET_EDGES + [np.inf],
        labels=BUCKET_LABELS,
        right=True,
    ).value_counts().reindex(BUCKET_LABELS, fill_value=0)
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "pct%": pct})


# ──────────────────────────────────────────────────────────────────────────────
all_trades = []
for pair in PAIRS:
    print(f"  backtesting {pair}...", flush=True)
    all_trades.extend(backtest_pair(pair))

df = pd.DataFrame(all_trades)
print(f"\nTotal trades across 4 pairs: {len(df)}\n")

# ── 1. Overall stats ──────────────────────────────────────────────────────────
wins   = df[df.win]
losses = df[~df.win]
total  = len(df)
wr     = wins.shape[0] / total * 100
print("=" * 60)
print("1. OVERALL STATS")
print("=" * 60)
print(f"  Total trades : {total}")
print(f"  Win rate     : {wr:.1f}%")
print(f"  Avg win      : {wins.pnl_pips.mean():.2f} pips")
print(f"  Avg loss     : {losses.pnl_pips.mean():.2f} pips")
print(f"  Expectancy   : {df.pnl_pips.mean():.3f} pips/trade")
print(f"  Total P/L    : {df.pnl_pips.sum():.1f} pips")

# ── 2. Duration distributions ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DURATION DISTRIBUTION — WINNERS")
print("=" * 60)
win_dist = bucket_distribution(wins.duration_bars)
print(win_dist.to_string())

print("\n" + "=" * 60)
print("2. DURATION DISTRIBUTION — LOSERS")
print("=" * 60)
loss_dist = bucket_distribution(losses.duration_bars)
print(loss_dist.to_string())

# ── 3. Per-pair breakdown ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PER-PAIR BREAKDOWN")
print("=" * 60)
hdr = f"{'Pair':<12} {'Trades':>7} {'WR%':>6} {'TotalP/L':>10} " \
      f"{'MedWinH':>9} {'MedLossH':>10} {'AvgWin':>8} {'AvgLoss':>9}"
print(hdr)
print("-" * len(hdr))
for pair in PAIRS:
    sub  = df[df.pair == pair]
    w    = sub[sub.win]
    l_   = sub[~sub.win]
    wr_p = w.shape[0] / len(sub) * 100 if len(sub) > 0 else 0
    med_w = w.duration_hours.median() if len(w) > 0 else 0
    med_l = l_.duration_hours.median() if len(l_) > 0 else 0
    print(f"{pair:<12} {len(sub):>7} {wr_p:>5.1f}% {sub.pnl_pips.sum():>10.1f} "
          f"{med_w:>9.1f}h {med_l:>9.1f}h "
          f"{w.pnl_pips.mean() if len(w) else 0:>8.2f} "
          f"{l_.pnl_pips.mean() if len(l_) else 0:>9.2f}")

# ── 4. Key insight ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. KEY INSIGHT — WIN vs LOSS DURATION")
print("=" * 60)
print(f"  Median WIN  duration : {wins.duration_bars.median():.1f} bars "
      f"= {wins.duration_hours.median():.1f} h")
print(f"  Median LOSS duration : {losses.duration_bars.median():.1f} bars "
      f"= {losses.duration_hours.median():.1f} h")
print(f"  Mean WIN  duration   : {wins.duration_bars.mean():.1f} bars "
      f"= {wins.duration_hours.mean():.1f} h")
print(f"  Mean LOSS duration   : {losses.duration_bars.mean():.1f} bars "
      f"= {losses.duration_hours.mean():.1f} h")

ratio_med = wins.duration_bars.median() / max(losses.duration_bars.median(), 0.5)
ratio_mean = wins.duration_bars.mean()  / max(losses.duration_bars.mean(),  0.5)
if ratio_med >= 1.2:
    direction = f"Winners hold {ratio_med:.1f}× LONGER (median) — trend-riding confirmed."
elif ratio_med <= 0.85:
    direction = f"Winners exit {1/ratio_med:.1f}× FASTER (median) — mean-reversion pattern."
else:
    direction = "Win/loss durations similar — no strong asymmetry."
print(f"\n  >> {direction}")

# ── 5. Exit reason breakdown ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. EXIT REASON BREAKDOWN")
print("=" * 60)
for reason in ["trail", "reverse"]:
    sub = df[df.exit_reason == reason]
    if len(sub) == 0:
        continue
    w2 = sub[sub.win]
    print(f"\n  {reason.upper()} exits : {len(sub)} trades "
          f"(WR={len(w2)/len(sub)*100:.1f}%, "
          f"avg={sub.pnl_pips.mean():.2f}p, "
          f"med_dur={sub.duration_hours.median():.0f}h)")
