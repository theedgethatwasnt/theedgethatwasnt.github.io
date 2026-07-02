"""
Wavelet multi-scale co-activation gate for the USD_JPY shock-fade experiment.

Hypothesis
----------
A genuine SHOCK shows up as simultaneous energy across MULTIPLE wavelet scales
(a true multiscale event), whereas noise activates only one scale. Filtering the
sigma-gate event list (baseline meta3_USD_JPY.parquet) down to events that are
ALSO confirmed by wavelet co-activation should isolate real shocks and yield a
stronger reversion edge.

CAUSALITY (non-negotiable)
--------------------------
`pywt.wavedec` over a full array is NON-CAUSAL at the boundaries (periodization
wraps, multi-level transform mixes future samples). To avoid look-ahead, for each
candidate event bar t we recompute the DWT on a TRAILING window
`close[t-W+1 : t+1]` and read the co-activation at the LAST sample only. No future
bar ever enters the window. The MAD z-score is also computed causally (trailing
median + trailing MAD inside the same window, last-sample value). A small R7-style
check (append future bars, flag must not change) gates the whole pipeline.

Outputs
-------
For each (z_thr, k_bands) config:
  meta3_USD_JPY_wav_z{z}_k{k}.parquet  — same schema as baseline meta3, filtered to
  wavelet-confirmed events, with IS/OOS 70/30 chronological split recomputed on the
  filtered subset.

Reuses the wavelet primitives from
research/experiments/multiscale_shock/build_wavelet_db.py (compute_dwt_bands,
WAVELET, DWT_LEVEL) but replaces the non-causal centered MAD with a causal
trailing-window last-sample MAD z-score.
"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MS_DIR = PROJECT_ROOT / "research" / "experiments" / "multiscale_shock"
sys.path.insert(0, str(MS_DIR))

# Reuse the exact wavelet config from the existing infrastructure.
from build_wavelet_db import WAVELET, DWT_LEVEL, MAD_WIN  # noqa: E402


def s5_ba_path(pair: str) -> Path:
    return PROJECT_ROOT / "data" / "s5_ba" / f"{pair}_S5_BA.parquet"


def baseline_meta3_path(pair: str) -> Path:
    return SCRIPT_DIR / f"meta3_{pair}.parquet"

# Trailing window length for the causal DWT (S5 bars). Must be >= MAD_WIN so the
# MAD z-score at the last sample has a full reference distribution, and large
# enough for an 8-level db4 decomposition.
W = 2048

# Set of detail bands used to measure multi-scale co-activation.
# D2..D5 = the fast→medium scales where a real shock should light up jointly
# (matches the joint_timing_direction.py hypothesis: D2-D3 = fine-scale event,
#  D4-D5 elevation = true multiscale shock).
CO_BANDS = ["D2", "D3", "D4", "D5"]

IS_FRAC = 0.70  # chronological 70/30 split on the filtered subset


# ─────────────────────────────────────────────────────────────────────────────
# Causal last-sample wavelet co-activation
# ─────────────────────────────────────────────────────────────────────────────
def last_sample_band_z(window_close: np.ndarray) -> dict:
    """
    Causal per-band MAD z-score of log-energy at the LAST sample of `window_close`.

    DWT is run on the trailing window only (no future samples present). For each
    detail band we take the coefficient aligned to the last sample (the last
    upsampled value, == last detail coefficient of that level), form its
    log-energy, and z-score it against the trailing log-energy distribution of
    that same band over the last MAD_WIN bars of the window. Everything is read
    at the final index → strictly causal.
    """
    n = len(window_close)
    coeffs = pywt.wavedec(window_close, WAVELET, level=DWT_LEVEL, mode="periodization")
    out = {}
    for k in range(1, DWT_LEVEL + 1):
        name = f"D{DWT_LEVEL + 1 - k}"
        cd = coeffs[k]
        factor = (n + len(cd) - 1) // len(cd)
        band = np.repeat(cd, factor)[:n]                 # upsample to bar resolution
        log_e = np.log(band ** 2 + 1e-12)
        # causal trailing MAD z at the last sample
        ref = log_e[-MAD_WIN:] if n >= MAD_WIN else log_e
        med = np.median(ref)
        mad = np.median(np.abs(ref - med))
        z = (log_e[-1] - med) / (1.4826 * max(mad, 1e-12))
        out[name] = float(z)
    return out


def coactivation_count(window_close: np.ndarray, z_thr: float) -> int:
    """Number of CO_BANDS whose |last-sample shock_z| >= z_thr."""
    zs = last_sample_band_z(window_close)
    return int(sum(1 for b in CO_BANDS if abs(zs[b]) >= z_thr))


# ─────────────────────────────────────────────────────────────────────────────
# Causality check (REQUIRED gate)
# ─────────────────────────────────────────────────────────────────────────────
def causality_check(close: np.ndarray, t_events: np.ndarray, z_thr: float,
                    k_bands: int, n_check: int = 3, future_pad: int = 5000) -> bool:
    """
    For n_check random events, compute the co-activation flag at t_event from
    [t_event-W : t_event] and again from a window that ALSO contains future bars.
    The two windows share the same trailing W samples ending at t_event, so the
    last-sample flag MUST be identical. If it differs, the wavelet leaks future
    data → FAIL.
    """
    rng = np.random.default_rng(12345)
    # only pick events with enough history AND room for future_pad
    eligible = t_events[(t_events >= W) & (t_events + future_pad < len(close))]
    if len(eligible) < n_check:
        eligible = t_events[t_events >= W]
    picks = rng.choice(eligible, size=min(n_check, len(eligible)), replace=False)

    all_pass = True
    print(f"\n  Causality check (z_thr={z_thr}, k_bands={k_bands}, W={W}):")
    for te in picks:
        te = int(te)
        # trailing-only window ending exactly at te (inclusive)
        w_now = close[te - W + 1: te + 1]
        c_now = coactivation_count(w_now, z_thr)
        flag_now = int(c_now >= k_bands)

        # window with extra FUTURE bars appended; the last sample we evaluate
        # is still te, so we slice the future-padded array but read index te.
        ext = close[te - W + 1: te + 1 + future_pad]
        # to read the flag "at te" from the extended array, take the same trailing
        # W samples ending at te (i.e. the first W samples of `ext`).
        w_future = ext[:W]
        c_future = coactivation_count(w_future, z_thr)
        flag_future = int(c_future >= k_bands)

        ok = (flag_now == flag_future) and (c_now == c_future)
        all_pass = all_pass and ok
        print(f"    t_event={te:>10d}  count_now={c_now} flag_now={flag_now}  "
              f"count_future={c_future} flag_future={flag_future}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  Causality check: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Filtering + meta3 emission
# ─────────────────────────────────────────────────────────────────────────────
def recompute_split(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological 70/30 IS/OOS split on the filtered (sorted) subset."""
    df = df.sort_values("t_event").reset_index(drop=True)
    n = len(df)
    is_end = int(round(n * IS_FRAC))
    df["split"] = ["IS"] * is_end + ["OOS"] * (n - is_end)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY",
                    help="instrument; reads meta3_<PAIR>.parquet + "
                         "data/s5_ba/<PAIR>_S5_BA.parquet, writes "
                         "meta3_<PAIR>_wav_z{z}_k{k}.parquet")
    ap.add_argument("--z-thrs", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--k-bands", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--check-only", action="store_true",
                    help="run causality check then exit (no parquet writes)")
    args = ap.parse_args()

    pair = args.pair
    s5_path = s5_ba_path(pair)
    baseline_meta3 = baseline_meta3_path(pair)

    print("=" * 70)
    print(f"  Wavelet multi-scale co-activation gate — {pair} shock-fade")
    print("=" * 70)
    print(f"  WAVELET={WAVELET}  DWT_LEVEL={DWT_LEVEL}  MAD_WIN={MAD_WIN}  W={W}")
    print(f"  co-activation bands: {CO_BANDS}")
    print(f"  grid: z_thr={args.z_thrs}  k_bands={args.k_bands}")

    # Load baseline sigma-gate events
    meta3 = pd.read_parquet(baseline_meta3).reset_index(drop=True)
    n_sigma = len(meta3)
    print(f"\n  Baseline sigma-gate events: {n_sigma:,}")

    # Load close (mid) from s5_ba. meta3 t_event indexes rows of this array.
    print(f"  Loading close from {s5_path.name} ...")
    s5 = pd.read_parquet(s5_path, columns=["close"])
    close = s5["close"].values.astype(np.float64)
    del s5
    gc.collect()
    print(f"  close len = {len(close):,}")

    t_events = meta3["t_event"].values.astype(np.int64)
    if t_events.max() >= len(close):
        raise SystemExit(
            f"meta3 t_event max {t_events.max()} >= close len {len(close)} — "
            f"index mismatch, aborting.")

    # ── Causality gate (run once; the flag logic is identical across configs) ──
    chk = causality_check(close, t_events, z_thr=args.z_thrs[0],
                          k_bands=args.k_bands[0])
    if not chk:
        raise SystemExit(
            "\nCAUSALITY CHECK FAILED — wavelet computation leaks future data. "
            "Stopping; do NOT train.")

    if args.check_only:
        print("\n--check-only: causality PASS, exiting before parquet writes.")
        return

    # ── Compute co-activation count once per event (max z_thr granularity) ──
    # We compute the per-band |z| once per event, then threshold for each config.
    print(f"\n  Computing causal wavelet co-activation for {n_sigma:,} events "
          f"(trailing window W={W}) ...")
    # store |z| for each CO_BAND per event
    zabs = np.full((n_sigma, len(CO_BANDS)), np.nan, dtype=np.float64)
    valid = t_events >= W  # need a full trailing window
    n_skipped = int((~valid).sum())
    t0 = pd.Timestamp.now()
    for i, te in enumerate(t_events):
        te = int(te)
        if te < W:
            continue
        w = close[te - W + 1: te + 1]
        zs = last_sample_band_z(w)
        for j, b in enumerate(CO_BANDS):
            zabs[i, j] = abs(zs[b])
        if (i + 1) % 1000 == 0:
            el = (pd.Timestamp.now() - t0).total_seconds()
            rate = (i + 1) / el
            eta = (n_sigma - i - 1) / rate
            print(f"    {i + 1:>6,}/{n_sigma:,}  {rate:.0f} ev/s  ETA {eta:.0f}s")
    print(f"    done. skipped {n_skipped} events with < W history "
          f"(t_event < {W}).")

    # ── Emit filtered meta3 per config ──
    summary = []
    base_cols = ["sample_id", "t_pre", "t_event", "t_timeout", "direction", "split"]
    for z_thr in args.z_thrs:
        for k in args.k_bands:
            count = np.nansum(zabs >= z_thr, axis=1)        # bands lit per event
            keep = valid & (count >= k)
            sub = meta3.loc[keep, base_cols].copy()
            sub = recompute_split(sub)
            n_keep = len(sub)
            frac = n_keep / n_sigma if n_sigma else 0.0
            out = SCRIPT_DIR / f"meta3_{pair}_wav_z{z_thr:g}_k{k}.parquet"
            sub.to_parquet(out, index=False)
            n_is = int((sub["split"] == "IS").sum())
            n_oos = int((sub["split"] == "OOS").sum())
            print(f"\n  z_thr={z_thr:g} k_bands={k}: kept {n_keep:,}/{n_sigma:,} "
                  f"({frac:.1%})  IS={n_is} OOS={n_oos}  -> {out.name}")
            summary.append(dict(z_thr=z_thr, k_bands=k, n_keep=n_keep,
                                frac=frac, n_is=n_is, n_oos=n_oos,
                                file=out.name))

    print("\n" + "=" * 70)
    print("  Survival summary")
    print("=" * 70)
    print(f"  sigma-gate baseline events: {n_sigma:,}")
    for s in summary:
        print(f"  z={s['z_thr']:g} k={s['k_bands']}: "
              f"{s['n_keep']:>6,} ({s['frac']:.1%})  "
              f"IS={s['n_is']} OOS={s['n_oos']}")

    # persist summary for the results markdown
    survival_csv = SCRIPT_DIR / f"wavelet_gate_survival_{pair}.csv"
    pd.DataFrame(summary).to_csv(survival_csv, index=False)
    print(f"\n  Wrote survival summary -> {survival_csv.name}")


if __name__ == "__main__":
    main()
