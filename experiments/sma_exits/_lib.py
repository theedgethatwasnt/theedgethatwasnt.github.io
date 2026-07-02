"""Shared infrastructure for SMA-exit experiments (H8, H10, H11, H12, H13, H14).

Key optimisation: per-bar M5 confluence signal pre-computed once per pair,
then every exit-rule sweep runs in Numba over indexed numpy arrays.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path("/path/to/projects/fx-core")
DATA    = PROJECT / "data" / "m5_ohlc"

SMA_N         = 16
LAGS          = (8, 10, 15)
WINDOW_BARS   = 6 * 30 * 24 * 12
IS_FRAC       = 4/6
SPREAD_FRAC   = 0.6
BARS_PER_H1   = 12
M5_PER_DAY    = 288
TP_PIPS_BASE  = 20.0

PAIRS = {
    "USD_JPY": (0.01,   2.10), "EUR_JPY": (0.01,   2.50),
    "GBP_JPY": (0.01,   4.00), "AUD_JPY": (0.01,   2.30),
    "EUR_USD": (0.0001, 1.70), "GBP_USD": (0.0001, 2.40),
    "CAD_JPY": (0.01,   2.60), "AUD_USD": (0.0001, 1.60),
    "EUR_GBP": (0.0001, 2.00), "NZD_USD": (0.0001, 2.00),
}


def sma(arr, n):
    out = np.full(len(arr), np.nan)
    if len(arr) < n: return out
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out[n-1:] = (cs[n:] - cs[:-n]) / n
    return out


def momentum_sig_vec(closes, lags=LAGS, sma_n=SMA_N):
    """Vectorised SMA-momentum confluence signal: +1/-1/0 per bar."""
    s = sma(closes, sma_n)
    sig = np.zeros(len(closes), dtype=np.int8)
    valid = np.zeros(len(closes), dtype=bool)
    for lg in lags:
        valid |= np.isnan(s)
    # Compute pairwise direction for each lag in vectorised form
    n = len(closes)
    up_count = np.zeros(n, dtype=np.int8)
    dn_count = np.zeros(n, dtype=np.int8)
    for lg in lags:
        shifted = np.full(n, np.nan)
        shifted[lg:] = s[:n-lg]
        diff = s - shifted
        up_count += (diff > 0).astype(np.int8)
        dn_count += (diff < 0).astype(np.int8)
    sig[up_count == len(lags)] = 1
    sig[dn_count == len(lags)] = -1
    # Zero out bars without enough SMA history
    sig[:max(lags) + sma_n] = 0
    return sig


def resample_tf(df_m5, minutes):
    d = df_m5.set_index('timestamp')
    return d.resample(f'{minutes}min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last'}
    ).dropna().reset_index()


def atr_h1_series(h1_df, period=14):
    h = h1_df['high'].to_numpy(); l = h1_df['low'].to_numpy(); c = h1_df['close'].to_numpy()
    tr = np.zeros(len(h)); tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.full(len(h), np.nan)
    if len(h) > period:
        a[period] = tr[1:period+1].mean()
        for i in range(period+1, len(h)):
            a[i] = (a[i-1]*(period-1) + tr[i])/period
    return a


def project_to_m5(m5_ts, tf_ts, tf_arr):
    """Forward-fill a higher-TF array onto M5 timeline using last-completed lookup.
    For int arrays, leaves the original value (idx clipped to 0); use a separate
    'valid' mask if you need to know whether a real bar existed yet."""
    idx = np.searchsorted(tf_ts, m5_ts, side='right') - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(tf_arr) - 1)
    out = tf_arr[idx].copy()
    if out.dtype.kind == 'f':
        out[~valid] = np.nan
    else:
        out[~valid] = 0
    return out


def load_pair(pair):
    """Load one pair, build all per-M5 arrays needed for sweeps."""
    pip, sp_gate = PAIRS[pair]
    spread_cost = sp_gate * SPREAD_FRAC

    df = pq.read_table(DATA / f"{pair}_M5.parquet").to_pandas()
    df = df.tail(WINDOW_BARS).reset_index(drop=True)

    h1  = resample_tf(df, 60)
    m30 = resample_tf(df, 30)

    h1_sig  = momentum_sig_vec(h1['close'].to_numpy())
    m30_sig = momentum_sig_vec(m30['close'].to_numpy())
    atr_h1  = atr_h1_series(h1)

    m5_ts = df['timestamp'].to_numpy()
    h1_ts  = h1['timestamp'].to_numpy()
    m30_ts = m30['timestamp'].to_numpy()

    # Project H1, M30 signal + H1 ATR + H1 close onto M5 timeline (forward-fill, no peek)
    # signal at M5 bar i uses TF bar with ts ≤ m5_ts[i-1] (last completed)
    prev_ts = np.empty_like(m5_ts); prev_ts[0] = m5_ts[0]; prev_ts[1:] = m5_ts[:-1]
    h1_sig_m5  = project_to_m5(prev_ts, h1_ts,  h1_sig.astype(np.int64)).astype(np.int8)
    m30_sig_m5 = project_to_m5(prev_ts, m30_ts, m30_sig.astype(np.int64)).astype(np.int8)
    h1_atr_m5  = project_to_m5(prev_ts, h1_ts,  atr_h1)
    h1_c_m5    = project_to_m5(prev_ts, h1_ts,  h1['close'].to_numpy())

    # Confluence: +1 if both H1 and M30 = +1, -1 if both -1, else 0
    sig_m5 = np.zeros(len(m5_ts), dtype=np.int8)
    sig_m5[(h1_sig_m5 == 1) & (m30_sig_m5 == 1)] = 1
    sig_m5[(h1_sig_m5 == -1) & (m30_sig_m5 == -1)] = -1

    return {
        'pair': pair, 'pip': pip, 'spread_cost': spread_cost,
        'n': len(m5_ts), 'is_end': int(len(m5_ts) * IS_FRAC),
        'm5_ts': m5_ts,
        'opens': df['open'].to_numpy(np.float64),
        'highs': df['high'].to_numpy(np.float64),
        'lows':  df['low'].to_numpy(np.float64),
        'closes':df['close'].to_numpy(np.float64),
        'sig_m5':  sig_m5,
        'h1_atr_m5': h1_atr_m5,
        'h1_c_m5':   h1_c_m5,
        # Raw H1 arrays (not projected) for indicators that need to be
        # computed on the H1 timescale and then forward-filled to M5
        'h1_ts': h1_ts,
        'h1_o': h1['open'].to_numpy(),
        'h1_h': h1['high'].to_numpy(),
        'h1_l': h1['low'].to_numpy(),
        'h1_c': h1['close'].to_numpy(),
    }


def trade_stats_from_arrays(pnls, entry_bars, is_end, n_total, spread_cost):
    """Stats from raw numpy arrays produced by Numba kernel."""
    if len(pnls) == 0:
        return {'trades':0,'is_n':0,'oos_n':0,'is_pd':0,'oos_pd':0,
                'is_net':0,'oos_net':0,'oos_dd':0,'oos_wr':0,'wf_ok':False}
    net = pnls - spread_cost
    is_mask  = entry_bars < is_end
    oos_mask = ~is_mask
    is_n  = int(is_mask.sum()); oos_n = int(oos_mask.sum())
    is_net  = float(net[is_mask].sum())
    oos_net = float(net[oos_mask].sum())
    is_days  = is_end / M5_PER_DAY
    oos_days = (n_total - is_end) / M5_PER_DAY
    is_pd  = is_net  / max(is_days, 1)
    oos_pd = oos_net / max(oos_days, 1)

    if oos_n > 0:
        cum = net[oos_mask].cumsum()
        oos_dd = float((cum - np.maximum.accumulate(cum)).min())
        oos_wr = float((net[oos_mask] > 0).mean() * 100)
    else:
        oos_dd = 0.0; oos_wr = 0.0

    wf_ok = False
    if is_n >= 15:
        is_pnls = net[is_mask]
        chunks = np.array_split(is_pnls, 3)
        wf_ok = all(c.sum() > 0 and len(c) >= 5 for c in chunks)

    return {
        'trades':   len(pnls),
        'is_n':     is_n,   'oos_n':  oos_n,
        'is_net':   round(is_net,1),   'oos_net': round(oos_net,1),
        'is_pd':    round(is_pd,2),    'oos_pd':  round(oos_pd,2),
        'oos_dd':   round(oos_dd,1),   'oos_wr':  round(oos_wr,1),
        'wf_ok':    bool(wf_ok),
    }


def telegram(msg):
    """Send telegram via bot env vars (best-effort)."""
    tok = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not (tok and chat):
        print(f"[no telegram] {msg[:200]}")
        return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={'chat_id': chat, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"telegram err: {e}")
