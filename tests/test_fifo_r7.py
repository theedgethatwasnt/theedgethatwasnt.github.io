"""
R7 Consistency Test: Numba backtest kernel == lib.pnf_engine (canonical Python).

Because strategy_fifo_paper and strategy_fifo_live both import from lib.pnf_engine,
a single engine-vs-Numba pass covers all three code paths.

Three-way check:
  1. Numba run_single() kernel on N bars
  2. lib.pnf_engine.process_bar() direct loop on same N bars
  3. FIFOPaperSim.process_bar() on same N bars (service wiring check)

Diffs are labeled:
  eng:fieldname  — Numba vs engine divergence (core logic bug)
  sim:fieldname  — engine vs PaperSim divergence (service wiring bug)

Run:
    cd ${HOME}/projects/fx-core
    python3 tests/test_fifo_r7.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "services" / "strategy_fifo_paper"))

from main import FIFOConfig, FIFOPaperSim, CONFIGS
from lib.pnf_engine import (
    PnFState, PnFConfig, process_bar as _eng_process_bar, MAX_COL_HIST,
)

BA_DIR  = BASE / "data/m5_ba"
N_BARS  = 1000   # replay last N bars

# ── Numba reference kernel (single-config, no prange) ─────────────────────────
import numba as nb

MAX_K_R7 = MAX_COL_HIST

@nb.njit(inline="always")
def _col_sma_r7(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0: return 0.0
    total = 0.0
    for j in range(count):
        idx = (ptr - 1 - j) % MAX_K_R7
        total += hist[idx]
    return total / count


@nb.njit
def run_single(opens, highs, lows, closes, spreads,
               bs_pips, rev, n_min, trail_d, x7_k, pip, spread_gate):
    """
    Run one FIFO-Trends config on bar array.
    Returns final P&F state tuple:
      (pnf_level, pnf_dir, col_count, pos, entry_px, hw_level, pending,
       col_hist_ptr, col_hist_n, col_hist)
    """
    bs = bs_pips * pip

    pnf_idx = 0; pnf_level = 0.0; pnf_dir = 0; col_count = 0; prev_col_nb = 0
    col_hist     = np.zeros(MAX_K_R7, dtype=np.float64)
    col_hist_ptr = 0; col_hist_n = 0
    pos = 0; entry_px = 0.0; hw_level = 0.0; pending = 0

    for i in range(len(opens)):
        opn=opens[i]; hi=highs[i]; lo=lows[i]; cl=closes[i]; sp=spreads[i]
        bull=(cl>=opn)
        p1=hi if bull else lo
        p2=lo if bull else hi

        did_reverse_p1=False; did_reverse_p2=False
        prev_col_p1=0; prev_col_p2=0

        for tick in range(2):
            px = p1 if tick==0 else p2
            if pnf_dir==0:
                pnf_idx=int(px/bs); pnf_level=pnf_idx*bs
                pnf_dir=1; col_count=1; continue
            delta=int(px/bs)-pnf_idx
            if pnf_dir==1:
                if delta>=1:
                    pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=delta
                elif delta<=-rev:
                    prev_col_nb=col_count
                    col_hist[col_hist_ptr%MAX_K_R7]=prev_col_nb; col_hist_ptr+=1
                    if col_hist_n<MAX_K_R7: col_hist_n+=1
                    pnf_dir=-1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=-delta
                    if tick==0: did_reverse_p1=True; prev_col_p1=prev_col_nb
                    else:       did_reverse_p2=True; prev_col_p2=prev_col_nb
            elif pnf_dir==-1:
                if delta<=-1:
                    pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=(-delta)
                elif delta>=rev:
                    prev_col_nb=col_count
                    col_hist[col_hist_ptr%MAX_K_R7]=prev_col_nb; col_hist_ptr+=1
                    if col_hist_n<MAX_K_R7: col_hist_n+=1
                    pnf_dir=1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=delta
                    if tick==0: did_reverse_p1=True; prev_col_p1=prev_col_nb
                    else:       did_reverse_p2=True; prev_col_p2=prev_col_nb

        did_reverse=did_reverse_p1 or did_reverse_p2
        prev_col_at_rev=prev_col_p1 if did_reverse_p1 else prev_col_p2

        if pos==1:
            if pnf_dir==1 and pnf_level>hw_level: hw_level=pnf_level
        elif pos==-1:
            if pnf_dir==-1 and pnf_level<hw_level: hw_level=pnf_level

        exit_triggered=False
        if pos!=0:
            d=float(trail_d)
            if pos==1:
                trail=hw_level-d*bs
                if lo<=trail: exit_triggered=True
            else:
                trail=hw_level+d*bs
                if hi>=trail: exit_triggered=True
            if not exit_triggered and x7_k>0 and pnf_dir!=pos:
                sma_k=_col_sma_r7(col_hist,col_hist_ptr,col_hist_n,x7_k)
                if sma_k>0.0 and col_count>=sma_k:
                    exit_triggered=True

        if exit_triggered:
            pos=0; entry_px=0.0; hw_level=0.0

        if pos==0 and sp<=spread_gate:
            if did_reverse and prev_col_at_rev>=n_min: pending=pnf_dir
            if did_reverse and pending!=0 and pnf_dir!=pending: pending=0
            if pending!=0 and pnf_dir==pending and col_count>rev:
                pos=pending; entry_px=cl; hw_level=pnf_level; pending=0
        elif pos==0:
            if did_reverse and pending!=0 and pnf_dir!=pending: pending=0

    return (pnf_level, pnf_dir, col_count, pos, entry_px, hw_level, pending,
            col_hist_ptr, col_hist_n, col_hist.copy())


def run_r7(cfg: FIFOConfig, n_bars: int = N_BARS) -> dict:
    ba_path = BA_DIR / f"{cfg.pair}_M5_BA.parquet"
    assert ba_path.exists(), f"BA file missing: {ba_path}"

    ba   = pd.read_parquet(ba_path)
    bars = ba.tail(n_bars)

    opens   = bars["open"].values.astype(np.float64)
    highs   = bars["high"].values.astype(np.float64)
    lows    = bars["low"].values.astype(np.float64)
    closes  = bars["close"].values.astype(np.float64)
    spreads = ((bars["ask_c"] - bars["bid_c"]) / cfg.pip).values.astype(np.float64)

    # 1. Numba reference
    (nb_pnf_lvl, nb_pnf_dir, nb_col_cnt, nb_pos, nb_entry_px, nb_hw,
     nb_pending, nb_hist_ptr, nb_hist_n, nb_hist) = run_single(
        opens, highs, lows, closes, spreads,
        cfg.box_pips, cfg.rev, cfg.n_min, cfg.trail_d, cfg.x7_k,
        cfg.pip, cfg.sp_gate,
    )

    # 2. lib.pnf_engine direct loop
    eng_cfg = PnFConfig(
        pip=cfg.pip, box_pips=cfg.box_pips, rev=cfg.rev, n_min=cfg.n_min,
        trail_d=cfg.trail_d, x7_k=cfg.x7_k, sp_gate=cfg.sp_gate,
    )
    eng_st = PnFState()
    for j in range(len(bars)):
        _eng_process_bar(
            eng_st, eng_cfg,
            float(opens[j]), float(highs[j]), float(lows[j]), float(closes[j]),
            float(spreads[j]),
        )

    # 3. FIFOPaperSim wiring check
    sim = FIFOPaperSim(cfg)
    for j in range(len(bars)):
        bar_dict = {
            "open":  float(opens[j]),
            "high":  float(highs[j]),
            "low":   float(lows[j]),
            "close": float(closes[j]),
        }
        sim.process_bar(bar_dict, float(spreads[j]), str(j))
    sim_st = sim.st

    # ── Compare states ─────────────────────────────────────────────────────────
    tol = 0.001
    pip = cfg.pip
    diffs = {}

    def check_eng(name, nb_val, eng_val, scale=1.0):
        diff = abs(nb_val - eng_val) * scale
        if diff > tol:
            diffs[f"eng:{name}"] = {"numba": nb_val, "engine": eng_val, "diff": diff}

    def check_sim(name, eng_val, sim_val, scale=1.0):
        diff = abs(eng_val - sim_val) * scale
        if diff > tol:
            diffs[f"sim:{name}"] = {"engine": eng_val, "sim": sim_val, "diff": diff}

    check_eng("pnf_level",  nb_pnf_lvl,       eng_st.pnf_level,    1.0/pip)
    check_eng("pnf_dir",    float(nb_pnf_dir), float(eng_st.pnf_dir), 1.0)
    check_eng("col_count",  float(nb_col_cnt), float(eng_st.col_count), 1.0)
    check_eng("pos",        float(nb_pos),     float(eng_st.pos),     1.0)
    check_eng("entry_px",   nb_entry_px,       eng_st.entry_px,     1.0/pip)
    check_eng("hw_level",   nb_hw,             eng_st.hw_level,     1.0/pip)
    check_eng("pending",    float(nb_pending), float(eng_st.pending), 1.0)
    check_eng("hist_ptr",   float(nb_hist_ptr),float(eng_st.col_hist_ptr), 1.0)
    check_eng("hist_n",     float(nb_hist_n),  float(eng_st.col_hist_n),   1.0)
    for j in range(MAX_K_R7):
        if abs(nb_hist[j] - eng_st.col_hist[j]) > tol:
            diffs[f"eng:col_hist[{j}]"] = {"numba": nb_hist[j], "engine": eng_st.col_hist[j]}

    check_sim("pnf_level",  eng_st.pnf_level,    sim_st.pnf_level,    1.0/pip)
    check_sim("pnf_dir",    float(eng_st.pnf_dir), float(sim_st.pnf_dir), 1.0)
    check_sim("col_count",  float(eng_st.col_count), float(sim_st.col_count), 1.0)
    check_sim("pos",        float(eng_st.pos),     float(sim_st.pos),     1.0)
    check_sim("entry_px",   eng_st.entry_px,       sim_st.entry_px,     1.0/pip)
    check_sim("hw_level",   eng_st.hw_level,       sim_st.hw_level,     1.0/pip)
    check_sim("pending",    float(eng_st.pending), float(sim_st.pending), 1.0)
    check_sim("hist_ptr",   float(eng_st.col_hist_ptr), float(sim_st.col_hist_ptr), 1.0)
    check_sim("hist_n",     float(eng_st.col_hist_n),   float(sim_st.col_hist_n),   1.0)
    for j in range(MAX_K_R7):
        if abs(eng_st.col_hist[j] - sim_st.col_hist[j]) > tol:
            diffs[f"sim:col_hist[{j}]"] = {"engine": eng_st.col_hist[j], "sim": sim_st.col_hist[j]}

    return {
        "pair":   cfg.pair,
        "label":  cfg.label,
        "n_bars": n_bars,
        "pass":   len(diffs) == 0,
        "diffs":  diffs,
        "state_summary": {
            "pnf_level": round(nb_pnf_lvl, 6),
            "pnf_dir":   int(nb_pnf_dir),
            "col_count": int(nb_col_cnt),
            "pos":       int(nb_pos),
            "entry_px":  round(nb_entry_px, 6),
            "hw_level":  round(nb_hw, 6),
        },
    }


if __name__ == "__main__":
    print("R7 Consistency Test: Numba kernel == lib.pnf_engine == FIFOPaperSim")
    print(f"N_BARS={N_BARS} ({N_BARS/288:.1f} trading days), MAX_COL_HIST={MAX_K_R7}")
    print()

    print("Compiling Numba kernel...")
    _d = np.array([1.0] * 10, dtype=np.float64)
    run_single(_d, _d, _d, _d, _d, 5, 1, 3, 1, 5, 0.01, 2.5)
    print("  OK")
    print()

    all_pass = True
    for cfg in CONFIGS:
        result = run_r7(cfg)
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        print(f"  {status}  {cfg.label:14s}  ({cfg.pair})")
        if result["diffs"]:
            for k, v in result["diffs"].items():
                prefix = "eng" if k.startswith("eng:") else "sim"
                vals = "  ".join(f"{fk}={fv}" for fk, fv in v.items())
                print(f"    ❌ [{prefix}] {k[4:]}: {vals}")
            all_pass = False
        else:
            st = result["state_summary"]
            pos_str = f"pos={st['pos']}"
            if st["pos"] != 0:
                pos_str += f" entry={st['entry_px']:.5f} hw={st['hw_level']:.5f}"
            print(f"         pnf={st['pnf_level']:.4f} dir={st['pnf_dir']:+d} "
                  f"col={st['col_count']}  {pos_str}")

    print()
    if all_pass:
        print("✅ ALL CONFIGS PASS — Numba kernel, lib.pnf_engine, and FIFOPaperSim are identical")
        print("   R7 satisfied — safe to deploy")
    else:
        print("❌ SOME CONFIGS FAIL — DO NOT DEPLOY until divergences are resolved")
        sys.exit(1)
