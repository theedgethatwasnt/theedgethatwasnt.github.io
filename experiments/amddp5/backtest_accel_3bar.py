"""
AMDDP5 sweep — 3-bar momentum acceleration rule family
=======================================================

Entry rule (deterministic, R1/R3/R5/R6 compliant):
    At bar close t:
      d1 = c[t]   − c[t-1]
      d2 = c[t-1] − c[t-2]
      d3 = c[t-2] − c[t-3]
    Trigger if:
      • sign(d1) == sign(d2) == sign(d3)       (all 3 deltas same direction)
      • |d1| > |d2| > |d3|                     (strictly accelerating)
    Direction = sign(d1 + d2 + d3).
    Trade is placed at bar t+1 close (R1) — fills at ask_c (long)/bid_c (short).

Exit grid:
    tp_pips  ∈ {10, 15, 20, 30}
    sl_pips  ∈ {0, 10, 20, 50}   (0 ≡ no SL)
    max_hold ∈ {12, 60, 288}     (1h / 5h / 24h)

Validation (per pair):
    IS/OOS = 70/30
    Gate 1 (WF): 3 IS chunks each must have sum_amddp5 > 0
                 and trades_per_chunk >= MIN_TRADES_PER_CHUNK
    Gate 2 (OOS AMDDP5): oos_pd_amddp5 > 0
    Gate 3 (MC): sign-shuffle p-value on per-trade AMDDP5 array, p < 0.05

    Also reported (no gate, control only): oos_pd_raw

We compare AMDDP5-optimal vs raw-pnl-optimal configs per pair. If they diverge,
AMDDP5 is providing information that raw P&L misses (which is the entire point
of the metric: rewarding clean trades).

SOP compliance:
    R1: All decisions at bar close. sig[t] consumed at t+1.
    R3: Mid prices for the running unrealized P&L. Spread deducted at exit
        from spread[exit_t] = (ask_c − bid_c) (one full crossing).
    R5: sp_gate = np.percentile(spread[:is_end] / pip, 90), IS-only.
    R6: Same kernel as live would use.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from scorer import score_signal, amddp5_from_arrays, mc_pvalue_amddp, AMDDP_K

DATA_DIR = ROOT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
    "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "NZD_JPY", "CHF_JPY", "EUR_GBP",
]
PIP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
    "USD_JPY": 0.01,   "EUR_JPY": 0.01,   "GBP_JPY": 0.01,   "AUD_JPY": 0.01,
    "CAD_JPY": 0.01,   "NZD_JPY": 0.01,   "CHF_JPY": 0.01,   "EUR_GBP": 0.0001,
}

IS_FRAC              = 0.70
N_WF                 = 3
N_MC                 = 300
MIN_TRADES_PER_CHUNK = 5
BARS_PER_DAY         = 288.0

TP_PIPS_GRID  = [10.0, 15.0, 20.0, 30.0]
SL_PIPS_GRID  = [0.0, 10.0, 20.0, 50.0]    # 0 = no SL
MAX_HOLD_GRID = [12, 60, 288]              # 1h / 5h / 24h


def build_signal(close: np.ndarray) -> np.ndarray:
    """3-bar accelerating-momentum signal.
    Returns int8 array; sig[t] valid at close of bar t (consumed at t+1).
    """
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    if n < 4:
        return sig

    d1 = close[3:]   - close[2:-1]
    d2 = close[2:-1] - close[1:-2]
    d3 = close[1:-2] - close[:-3]

    s1 = np.sign(d1)
    s2 = np.sign(d2)
    s3 = np.sign(d3)

    a1 = np.abs(d1)
    a2 = np.abs(d2)
    a3 = np.abs(d3)

    same_dir = (s1 == s2) & (s2 == s3) & (s1 != 0)
    accelerating = (a1 > a2) & (a2 > a3)
    trigger = same_dir & accelerating

    net = d1 + d2 + d3
    direction = np.where(net > 0, 1, -1).astype(np.int8)

    sig[3:] = np.where(trigger, direction, 0).astype(np.int8)
    return sig


def load_pair(pair: str):
    df = pd.read_parquet(DATA_DIR / f"{pair}_M5_BA.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return (
        df["close"].to_numpy(np.float64),
        df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64),
        df["bid_c"].to_numpy(np.float64),
        df["ask_c"].to_numpy(np.float64),
        (df["ask_c"] - df["bid_c"]).to_numpy(np.float64),
    )


def evaluate_config(close, hi, lo, bid_c, ask_c, sp, pip,
                    sig, tp, sl, mh, sp_gate, is_end, n):
    """Run a single config through WF + OOS + MC and return a row dict (or None)."""
    chunk_sz = is_end // N_WF
    chunk_bounds = [0]
    for k in range(N_WF):
        s = chunk_bounds[-1]
        e = s + chunk_sz if k < N_WF - 1 else is_end
        chunk_bounds.append(e)

    is_amddp_chunks = []
    is_pnl_chunks   = []
    is_trades_chunks = []

    wf_ok = True
    for k in range(N_WF):
        s, e = chunk_bounds[k], chunk_bounds[k + 1]
        df_c, agg = score_signal(close, hi, lo, bid_c, ask_c, sp, pip, sig,
                                 tp_pips=tp, sl_pips=sl,
                                 max_hold=int(mh),
                                 sp_gate=sp_gate, start=s, end=e)
        is_amddp_chunks.append(agg["sum_amddp5"])
        is_pnl_chunks.append(agg["sum_pnl"])
        is_trades_chunks.append(agg["n_trades"])
        if agg["n_trades"] < MIN_TRADES_PER_CHUNK or agg["sum_amddp5"] <= 0:
            wf_ok = False

    is_total_trades = sum(is_trades_chunks)

    # OOS
    oos_df, oos_agg = score_signal(close, hi, lo, bid_c, ask_c, sp, pip, sig,
                                   tp_pips=tp, sl_pips=sl,
                                   max_hold=int(mh),
                                   sp_gate=sp_gate, start=is_end, end=n)
    oos_days = (n - is_end) / BARS_PER_DAY
    oos_trd  = oos_agg["n_trades"]
    if oos_trd == 0:
        oos_pd_amddp5 = 0.0
        oos_pd_raw    = 0.0
        oos_wr        = 0.0
        oos_max_dd    = 0.0
        mc_p          = 1.0
    else:
        amddp_arr = oos_df["amddp5"].to_numpy(np.float64)
        oos_pd_amddp5 = oos_agg["sum_amddp5"] / oos_days
        oos_pd_raw    = oos_agg["sum_pnl"]    / oos_days
        oos_wr        = oos_agg["n_wins"] / oos_trd
        oos_max_dd    = float(oos_df["max_dd_pips"].max())
        mc_p          = float(mc_pvalue_amddp(amddp_arr, N_MC))

    wf_pass = bool(wf_ok and oos_pd_amddp5 > 0 and mc_p < 0.05)

    return {
        "tp":             tp,
        "sl":             sl,
        "max_hold":       int(mh),
        "is_pd_amddp5":   round(sum(is_amddp_chunks) / (is_end / BARS_PER_DAY), 3),
        "oos_pd_amddp5":  round(oos_pd_amddp5, 3),
        "oos_pd_raw":     round(oos_pd_raw, 3),
        "is_trd":         is_total_trades,
        "oos_trd":        oos_trd,
        "oos_wr":         round(oos_wr, 4),
        "oos_max_dd":     round(oos_max_dd, 2),
        "mc_p":           round(mc_p, 4),
        "wf_pass":        wf_pass,
        # for diagnostics
        "is_chunk_amddp": [round(x, 1) for x in is_amddp_chunks],
        "is_chunk_pnl":   [round(x, 1) for x in is_pnl_chunks],
        "is_chunk_trd":   is_trades_chunks,
        "wf_pos_chunks":  int(sum(1 for x in is_amddp_chunks if x > 0)),
    }


def run_pair(pair: str):
    pip = PIP[pair]
    close, hi, lo, bid_c, ask_c, sp = load_pair(pair)
    n = len(close)
    is_end = int(n * IS_FRAC)
    oos_days = (n - is_end) / BARS_PER_DAY

    # R5: gate from IS only, hardcoded scalar
    sp_gate = float(np.percentile(sp[:is_end] / pip, 90))

    sig = build_signal(close)
    n_signals = int((sig != 0).sum())
    print(f"  {pair}: n={n:,}  IS_end={is_end:,}  OOS_days={oos_days:.0f}  "
          f"sp_gate={sp_gate:.2f}p  signals={n_signals:,}")

    rows = []
    for tp in TP_PIPS_GRID:
        for sl in SL_PIPS_GRID:
            for mh in MAX_HOLD_GRID:
                row = evaluate_config(close, hi, lo, bid_c, ask_c, sp, pip,
                                      sig, tp, sl, mh, sp_gate, is_end, n)
                row["pair"]    = pair
                row["sp_gate"] = round(sp_gate, 2)
                rows.append(row)
    return rows


def main():
    print("=" * 72)
    print("  AMDDP5 Sweep — 3-bar Accelerating Momentum")
    print("=" * 72)
    print(f"TP grid       : {TP_PIPS_GRID}")
    print(f"SL grid       : {SL_PIPS_GRID}  (0 = no SL)")
    print(f"max_hold grid : {MAX_HOLD_GRID} M5 bars  (1h / 5h / 24h)")
    print(f"Pairs         : {len(PAIRS)}")
    print(f"Configs/pair  : {len(TP_PIPS_GRID) * len(SL_PIPS_GRID) * len(MAX_HOLD_GRID)}")
    print(f"WF            : IS={IS_FRAC:.0%}, {N_WF} chunks, min_trd/chunk={MIN_TRADES_PER_CHUNK}")
    print(f"MC            : {N_MC} sign-shuffles on AMDDP5")
    print()

    # Warm Numba JIT
    print("Warming Numba JIT...")
    dummy_n = 1500
    rng = np.random.default_rng(0)
    px = 1.10 + np.cumsum(rng.normal(0, 1e-5, dummy_n))
    dummy_sig = np.zeros(dummy_n, dtype=np.int8)
    dummy_sig[::40] = 1
    score_signal(px, px + 1e-5, px - 1e-5, px - 5e-6, px + 5e-6,
                 np.full(dummy_n, 1e-5), 0.0001, dummy_sig,
                 tp_pips=10.0, sl_pips=10.0, max_hold=12, sp_gate=5.0)
    print("  Done.\n")

    all_rows = []
    for pair in PAIRS:
        if not (DATA_DIR / f"{pair}_M5_BA.parquet").exists():
            print(f"  {pair}: data missing, skipping")
            continue
        all_rows.extend(run_pair(pair))
        gc.collect()

    df = pd.DataFrame(all_rows)

    # Persistent column order
    cols = [
        "pair", "tp", "sl", "max_hold",
        "is_pd_amddp5", "oos_pd_amddp5", "oos_pd_raw",
        "is_trd", "oos_trd", "oos_wr", "oos_max_dd",
        "mc_p", "wf_pass", "wf_pos_chunks",
        "sp_gate", "is_chunk_amddp", "is_chunk_pnl", "is_chunk_trd",
    ]
    df = df[cols]

    out_path = RESULTS / "accel_3bar.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path} ({len(df)} rows)")

    # ── Verdict ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    n_total = len(df)
    n_amddp_pass = int(df["wf_pass"].sum())

    # "Raw-pnl WF-pass" control: same WF positivity but on raw pnl chunks.
    # We reconstruct from is_chunk_pnl + oos>0 + min trades.
    def raw_wf_pass(row):
        if row["oos_pd_raw"] <= 0:
            return False
        for v, t in zip(row["is_chunk_pnl"], row["is_chunk_trd"]):
            if t < MIN_TRADES_PER_CHUNK or v <= 0:
                return False
        return True
    df["raw_wf_pass"] = df.apply(raw_wf_pass, axis=1)
    n_raw_pass = int(df["raw_wf_pass"].sum())

    print(f"  AMDDP5 WF+MC-pass configs : {n_amddp_pass}/{n_total}")
    print(f"  Raw P&L WF-pass configs   : {n_raw_pass}/{n_total}  (control, no MC)")

    # Best per pair — by AMDDP5 and by raw pnl
    print()
    print("  Best AMDDP5 config per pair (only WF-pass shown):")
    print(f"    {'pair':<9} {'tp':>4} {'sl':>5} {'hold':>5} "
          f"{'amddp/d':>9} {'pnl/d':>8} {'oos_trd':>8} {'WR':>6} "
          f"{'maxDD':>7} {'mc_p':>7}")
    for pair in PAIRS:
        sub = df[(df["pair"] == pair) & (df["wf_pass"])]
        if sub.empty:
            print(f"    {pair:<9}  (no AMDDP5 WF+MC survivors)")
            continue
        r = sub.sort_values("oos_pd_amddp5", ascending=False).iloc[0]
        sl_str = "—" if r["sl"] == 0 else f"{int(r['sl'])}p"
        print(f"    {pair:<9} {int(r['tp']):>3}p {sl_str:>5} "
              f"{int(r['max_hold']):>5} "
              f"{r['oos_pd_amddp5']:>+8.2f} {r['oos_pd_raw']:>+8.2f} "
              f"{int(r['oos_trd']):>8} {r['oos_wr']*100:>5.1f}% "
              f"{r['oos_max_dd']:>6.1f} {r['mc_p']:>7.4f}")

    # Best raw-pnl config per pair (control, OOS+ + WF+)
    print()
    print("  Best RAW PNL config per pair (raw_wf_pass only):")
    print(f"    {'pair':<9} {'tp':>4} {'sl':>5} {'hold':>5} "
          f"{'amddp/d':>9} {'pnl/d':>8} {'oos_trd':>8} {'WR':>6}")
    for pair in PAIRS:
        sub = df[(df["pair"] == pair) & (df["raw_wf_pass"])]
        if sub.empty:
            print(f"    {pair:<9}  (no raw-pnl WF survivors)")
            continue
        r = sub.sort_values("oos_pd_raw", ascending=False).iloc[0]
        sl_str = "—" if r["sl"] == 0 else f"{int(r['sl'])}p"
        print(f"    {pair:<9} {int(r['tp']):>3}p {sl_str:>5} "
              f"{int(r['max_hold']):>5} "
              f"{r['oos_pd_amddp5']:>+8.2f} {r['oos_pd_raw']:>+8.2f} "
              f"{int(r['oos_trd']):>8} {r['oos_wr']*100:>5.1f}%")

    # Discriminator check
    print()
    print("  Same config wins under both metrics per pair?")
    n_same = n_diff = 0
    for pair in PAIRS:
        a = df[(df["pair"] == pair) & (df["wf_pass"])]
        b = df[(df["pair"] == pair) & (df["raw_wf_pass"])]
        if a.empty or b.empty:
            continue
        ra = a.sort_values("oos_pd_amddp5", ascending=False).iloc[0]
        rb = b.sort_values("oos_pd_raw",    ascending=False).iloc[0]
        same = (ra["tp"] == rb["tp"] and ra["sl"] == rb["sl"]
                and ra["max_hold"] == rb["max_hold"])
        if same:
            n_same += 1
        else:
            n_diff += 1
    print(f"    same config: {n_same}   different: {n_diff}")
    if n_diff > 0:
        print("    → AMDDP5 is discriminating between candidates that look")
        print("      similar by raw P&L. That is the metric doing useful work.")
    elif n_amddp_pass > 0:
        print("    → For this rule family, AMDDP5 and raw P&L pick the same")
        print("      winners. AMDDP5 added no discrimination here.")

    if n_amddp_pass == 0:
        print()
        print("  → No AMDDP5 deployable candidates. The 3-bar accel rule does")
        print("    not survive the spread-net + WF + MC gates on this data.")
        print("    Move to a different rule family.")

    print()


if __name__ == "__main__":
    main()
