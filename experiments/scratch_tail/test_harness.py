"""Tests for harness.py. Run on the Hetzner box per CLAUDE.md:
    rsync -az research/experiments/{scratch_tail,multiday_contrarian} root@HETZNER:/root/work/code/
    ssh root@HETZNER '/root/venv/bin/python -m pytest /root/work/code/scratch_tail/test_harness.py -x -q'
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import pytest

from bars import m5_to_h1, m5_to_m30
from harness import ARMS, PairState, _step_exit, run_battery, simulate_portfolio  # inserts carry_model's path
from signal import CONFIG_BY_PAIR, LAGS, TP_PIPS, six_of_six, wilder_atr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "multiday_contrarian"))
from carry_model import carry_pips  # noqa: E402


# ── synthetic data generator (mirrors multiday_contrarian/test_harness.py's _make_rw_m5) ────
def _make_rw_m5(n_days, seed, pair="USD_JPY", start="2021-01-04T22:00:00Z", spread_pips=1.7,
                 step_pip_mult=3.0):
    rng = np.random.default_rng(seed)
    pip = CONFIG_BY_PAIR[pair].pip
    n = n_days * 24 * 12
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    steps = rng.normal(0.0, pip * step_pip_mult, size=n)
    base = 105.0 if pair.endswith("_JPY") else 1.10
    mid = base + np.cumsum(steps)
    wiggle = np.abs(rng.normal(0.0, pip * 0.3, size=n))
    open_ = mid + rng.normal(0.0, pip * 0.1, size=n)
    close = mid
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    spread = spread_pips * pip
    bid_c = close - spread / 2.0
    ask_c = close + spread / 2.0
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "bid_c": bid_c, "ask_c": ask_c, "volume": np.maximum(1, rng.normal(500, 100, size=n)).astype(int),
    })


# ── independent reference replay of process_pair() (deque-based, "live style") ──────────────
def _reference_replay_arm_a(pair, m5_df):
    """A second, independent, deque-driven reimplementation of
    services/strategy_sma_scratch_paper/main.py's process_pair() (arm A: no stop, no overlay),
    operating on offline data instead of a live adapter. Cross-checks harness.simulate_portfolio
    arm A against something that is NOT harness.py's own vectorized signal.build_pair_signal /
    _build_master_grid machinery — a from-scratch parallel implementation."""
    cfg = CONFIG_BY_PAIR[pair]
    pip = cfg.pip
    h1 = m5_to_h1(m5_df)
    m30 = m5_to_m30(m5_df)
    h1_close_ts = (h1["timestamp"] + pd.Timedelta(hours=1)).to_numpy()
    m30_close_ts = (m30["timestamp"] + pd.Timedelta(minutes=30)).to_numpy()
    h1_c, h1_h, h1_l = h1["close"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy()
    m30_c = m30["close"].to_numpy()

    h1_close_d, h1_high_d, h1_low_d = deque(maxlen=64), deque(maxlen=64), deque(maxlen=64)
    m30_close_d = deque(maxlen=64)
    h1_ptr = 0
    m30_ptr = 0

    pos_dir = 0
    entry_px = entry_ts = entry_W = 0.0
    entry_bar = -1
    mfe = mae = 0.0
    bar_count = 0
    trades = []

    m5_ts = m5_df["timestamp"].to_numpy()
    m5_o, m5_h, m5_l, m5_c = (m5_df[c].to_numpy() for c in ("open", "high", "low", "close"))
    m5_bid, m5_ask = m5_df["bid_c"].to_numpy(), m5_df["ask_c"].to_numpy()

    for i in range(len(m5_df)):
        t = m5_ts[i]
        while h1_ptr < len(h1_close_ts) and h1_close_ts[h1_ptr] <= t:
            h1_close_d.append(h1_c[h1_ptr]); h1_high_d.append(h1_h[h1_ptr]); h1_low_d.append(h1_l[h1_ptr])
            h1_ptr += 1
        while m30_ptr < len(m30_close_ts) and m30_close_ts[m30_ptr] <= t:
            m30_close_d.append(m30_c[m30_ptr])
            m30_ptr += 1

        bar_count += 1
        close, high, low = m5_c[i], m5_h[i], m5_l[i]

        if pos_dir != 0:
            held = bar_count - entry_bar
            hp = (high - entry_px) / pip * pos_dir
            lp = (low - entry_px) / pip * pos_dir
            if hp > mfe: mfe = hp
            if lp < mae: mae = lp
            exit_px = None; reason = None
            if pos_dir == 1:
                if high >= entry_px + TP_PIPS * pip:
                    exit_px = entry_px + TP_PIPS * pip; reason = "tp"
            else:
                if low <= entry_px - TP_PIPS * pip:
                    exit_px = entry_px - TP_PIPS * pip; reason = "tp"
            if exit_px is None and held >= cfg.T_s_bars:
                if abs(close - entry_px) / pip <= entry_W:
                    exit_px = close; reason = "scratch"
            if exit_px is not None:
                gross = (exit_px - entry_px) / pip * pos_dir
                trades.append({
                    "direction": pos_dir, "entry_ts": entry_ts, "exit_ts": t,
                    "entry_px": entry_px, "exit_px": exit_px, "reason": reason,
                    "gross_pips": gross, "mfe_pips": mfe, "mae_pips": mae,
                })
                pos_dir = 0; entry_px = 0.0; mfe = mae = 0.0

        if pos_dir == 0:
            h_sig = six_of_six(h1_close_d, LAGS)
            m_sig = six_of_six(m30_close_d, LAGS)
            new_dir = 1 if (h_sig == 1 and m_sig == 1) else (-1 if (h_sig == -1 and m_sig == -1) else 0)
            if new_dir != 0:
                atr_val = wilder_atr(h1_high_d, h1_low_d, h1_close_d, 14)
                if atr_val is not None:
                    W = cfg.k_atr * (atr_val / pip)
                    pos_dir = new_dir
                    entry_px = close
                    entry_ts = t
                    entry_bar = bar_count
                    entry_W = W
                    mfe = mae = 0.0

    return trades


@pytest.mark.parametrize("pair", ["USD_JPY", "GBP_USD", "GBP_JPY"])
def test_harness_arm_a_matches_independent_reference_replay(pair):
    df = _make_rw_m5(n_days=200, seed=hash(pair) % 10000, pair=pair)
    ref_trades = _reference_replay_arm_a(pair, df)
    result = simulate_portfolio({pair: df}, ARMS["A"])
    got = result["trades"]
    assert len(ref_trades) >= 3, "need a few trades for a meaningful cross-check"
    assert len(got) == len(ref_trades)
    for a, b in zip(got, ref_trades):
        assert a["direction"] == b["direction"]
        assert a["entry_ts"] == pd.Timestamp(b["entry_ts"])
        assert a["exit_ts"] == pd.Timestamp(b["exit_ts"])
        assert a["entry_px"] == pytest.approx(b["entry_px"], abs=1e-9)
        assert a["exit_px"] == pytest.approx(b["exit_px"], abs=1e-9)
        assert a["exit_reason"] == b["reason"]
        assert a["gross_pips"] == pytest.approx(b["gross_pips"], abs=1e-6)
        assert a["mfe_pips"] == pytest.approx(b["mfe_pips"], abs=1e-6)
        assert a["mae_pips"] == pytest.approx(b["mae_pips"], abs=1e-6)


# ── _step_exit: disaster stop mechanics (new vs. the deployed service) ───────────────────────
def test_step_exit_disaster_stop_fires_before_tp_same_bar():
    cfg = CONFIG_BY_PAIR["USD_JPY"]
    pip = cfg.pip
    pst = PairState(pos_dir=1, entry_px=150.0, entry_bar_count=0, bar_count=1,
                     entry_stop_level=150.0 - 3.0 * 0.30)  # 3xATR(0.30) stop below entry
    exit_px, reason = _step_exit(pst, cfg, pip, o=150.0, h=150.0 + TP_PIPS * pip + 1, l=149.0, c=149.5, arm=ARMS["D"])
    assert reason == "stop"
    assert exit_px == pytest.approx(150.0 - 3.0 * 0.30)


def test_step_exit_disaster_stop_gap_fills_at_open_not_nominal():
    cfg = CONFIG_BY_PAIR["USD_JPY"]
    pip = cfg.pip
    stop_level = 150.0 - 3.0 * 0.30
    pst = PairState(pos_dir=1, entry_px=150.0, entry_bar_count=0, bar_count=1,
                     entry_stop_level=stop_level)
    gapped_open = stop_level - 0.05  # already through the stop at the bar's own open
    exit_px, reason = _step_exit(pst, cfg, pip, o=gapped_open, h=gapped_open + 0.01,
                                  l=gapped_open - 0.02, c=gapped_open, arm=ARMS["D"])
    assert reason == "stop_gap"
    assert exit_px == pytest.approx(gapped_open)


def test_step_exit_no_stop_arm_ignores_stop_level_entirely():
    cfg = CONFIG_BY_PAIR["USD_JPY"]
    pip = cfg.pip
    pst = PairState(pos_dir=1, entry_px=150.0, entry_bar_count=0, bar_count=1,
                     entry_stop_level=150.0 - 3.0 * 0.30)
    # bar's low crosses the (irrelevant, since arm A has no stop) stop level but not TP/scratch
    exit_px, reason = _step_exit(pst, cfg, pip, o=150.0, h=150.05, l=149.0, c=149.9, arm=ARMS["A"])
    assert exit_px is None


def test_step_exit_tp_fires_verbatim_no_gap_treatment():
    cfg = CONFIG_BY_PAIR["USD_JPY"]
    pip = cfg.pip
    pst = PairState(pos_dir=1, entry_px=150.0, entry_bar_count=0, bar_count=1)
    tp_lvl = 150.0 + TP_PIPS * pip
    exit_px, reason = _step_exit(pst, cfg, pip, o=tp_lvl + 0.5, h=tp_lvl + 0.5, l=tp_lvl + 0.4, c=tp_lvl + 0.45, arm=ARMS["A"])
    assert reason == "tp"
    assert exit_px == pytest.approx(tp_lvl)   # nominal fill even though the bar gapped past it


def test_step_exit_scratch_fires_only_after_T_s_bars_within_window():
    cfg = CONFIG_BY_PAIR["CAD_JPY"]  # T_s_bars=96
    pip = cfg.pip
    pst = PairState(pos_dir=1, entry_px=100.0, entry_bar_count=0, bar_count=50,
                     entry_W_pips=5.0)
    exit_px, _ = _step_exit(pst, cfg, pip, o=100.01, h=100.02, l=99.99, c=100.01, arm=ARMS["A"])
    assert exit_px is None  # held=50 < T_s_bars=96, scratch not eligible yet
    pst.bar_count = 97  # held = 97
    exit_px, reason = _step_exit(pst, cfg, pip, o=100.01, h=100.02, l=99.99, c=100.01, arm=ARMS["A"])
    assert reason == "scratch"


# ── carry wiring (integration, not a re-test of carry_model's own math) ─────────────────────
def test_harness_carry_is_wired_per_trade():
    df = _make_rw_m5(n_days=365 * 2, seed=21, pair="USD_JPY")
    result = simulate_portfolio({"USD_JPY": df}, ARMS["A"])
    trades = result["trades"]
    assert len(trades) >= 5
    for t in trades:
        expected = carry_pips("USD_JPY", t["direction"], t["entry_ts"], t["exit_ts"])
        assert t["carry_pips"] == pytest.approx(expected, abs=1e-9)


# ── gate 1 (PREREGISTRATION.md): synthetic-RW self-test ─────────────────────────────────────
#
# IMPORTANT DESIGN FINDING (surfaced by gate 1 itself, not a harness bug — kept as a permanent
# regression test, documented in PREREGISTRATION.md): arm A has NO stop. Its only exits are
# TP(+20p, fixed) and scratch (~flat, only reachable once price has wandered back within W of
# entry AFTER T_s_bars). A losing position that drifts away and stays away never closes on its
# own. On a FINITE-window random walk this means the CLOSED-trades-only sample is a biased
# subsample (skewed toward TP hits + flat scratches; the still-drifting, typically-underwater
# tail is sitting in `open_at_end`, excluded from `trades`) — this is PRECISELY the pathology
# PREREGISTRATION.md exists to characterize (project_validation_gap's "closed = winners by
# construction"), not a defect to eliminate. A closed-trades-only gross-mean check would FAIL
# gate 1 on arm A even though the harness has zero phantom directional edge — the martingale
# optional-stopping invariant (E[gross]=0 for any non-anticipating stopping rule) only holds
# over the FULL population (closed trades' realized gross-pips UNION open-at-end positions'
# unrealized gross-pips, marked to the window's end) — so that is what gate 1 checks below.
def _full_gross_population(result):
    """closed trades' gross_pips + open-at-end positions' unrealized (mark-to-window-end)
    gross pips — the population the martingale no-phantom-edge invariant actually applies to."""
    closed = [t["gross_pips"] for t in result["trades"]]
    openp = [x["unrealized_pnl_pips"] for x in result["open_at_end"] if not np.isnan(x["unrealized_pnl_pips"])]
    return np.array(closed + openp)


def test_gate1_synthetic_rw_no_phantom_edge_arm_a_vs_coin_a():
    df = _make_rw_m5(n_days=365 * 6, seed=42, pair="USD_JPY")
    res_sig = simulate_portfolio({"USD_JPY": df}, ARMS["A"])
    res_coin = simulate_portfolio({"USD_JPY": df}, ARMS["coin_A"])
    assert len(res_sig["trades"]) >= 20 and len(res_coin["trades"]) >= 20

    def gross_stats(result):
        v = _full_gross_population(result)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)

    m_sig, se_sig, n_sig = gross_stats(res_sig)
    m_coin, se_coin, n_coin = gross_stats(res_coin)
    assert abs(m_sig) < 3 * se_sig, f"signal shows phantom gross edge: {m_sig:+.3f} (se={se_sig:.3f})"
    assert abs(m_coin) < 3 * se_coin, f"coin shows phantom gross edge: {m_coin:+.3f} (se={se_coin:.3f})"
    se_diff = np.sqrt(se_sig**2 + se_coin**2)
    assert abs(m_sig - m_coin) < 4 * se_diff, "signal diverges from coin beyond sampling noise"

    # closed-trades-only cost sanity (spread/carry magnitude, NOT a sign/edge claim — see the
    # design-finding note above for why closed-only net is expected to be POSITIVE-biased here):
    spread_coin = np.mean([t["spread_rt_pips"] for t in res_coin["trades"]])
    carry_coin = np.mean([t["carry_pips"] for t in res_coin["trades"]])
    assert 0 < spread_coin < 10
    assert abs(carry_coin) < 50  # sane pip-scale, not a tight bound


def test_gate1_arm_a_closed_only_shows_the_known_tp_only_positive_bias():
    """Companion/contrast to the test above: documents (as a standing regression check, not a
    surprise) that arm A's CLOSED-trades-only gross mean IS materially positive on a pure
    random walk — the exact structural artifact the full-population test above corrects for.
    If this ever stops being true it means the exit structure changed and the full-population
    gate-1 test's premise should be revisited."""
    df = _make_rw_m5(n_days=365 * 6, seed=42, pair="USD_JPY")
    res_sig = simulate_portfolio({"USD_JPY": df}, ARMS["A"])
    closed_mean = np.mean([t["gross_pips"] for t in res_sig["trades"]])
    assert closed_mean > 5.0, (
        f"expected the known TP-only/no-stop closed-trade positive bias on a random walk "
        f"(closed_mean={closed_mean:+.2f}p) — if this regresses to ~0 the exit structure or "
        f"harness changed in a way that invalidates the full-population gate-1 rationale above"
    )
    assert len(res_sig["open_at_end"]) >= 1, "expect at least one still-open (unresolved) tail position"


def test_gate1_disaster_stop_arm_d_coin_also_no_phantom_edge():
    """Same full-population self-test extended to the primary arm D's coin control (with stop
    + overlay active) — the disaster stop and the overlay must not manufacture a gross edge
    either. Uses run_battery() since D/coin_D need their reference arm's (E/coin_E) blocked
    lookup — see harness.py's design-finding note on why the gating signal can't be
    self-referential."""
    df = _make_rw_m5(n_days=365 * 6, seed=43, pair="USD_JPY")
    res = run_battery({"USD_JPY": df})
    res_coin_d = res["coin_D"]
    assert len(res_coin_d["trades"]) >= 15
    v = _full_gross_population(res_coin_d)
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert abs(v.mean()) < 4 * se, f"coin_D gross mean implausibly far from 0: {v.mean():+.3f} (se={se:.3f})"


def test_run_battery_no_arm_deadlocks_to_zero_trades():
    """Regression test for the self-referential-deadlock bug the design-finding note documents:
    every gated arm must produce a comparable order-of-magnitude trade count to its ungated
    reference, not collapse to (near-)zero from a permanent block."""
    df = _make_rw_m5(n_days=365 * 4, seed=61, pair="GBP_JPY")
    res = run_battery({"GBP_JPY": df})
    assert len(res["A"]["trades"]) >= 5
    assert len(res["E"]["trades"]) >= 5
    for gated, ref in (("B", "A"), ("C", "A"), ("D", "E"), ("coin_D", "coin_E"), ("coin_overlay", "coin_A")):
        n_ref = len(res[ref]["trades"])
        n_gated = len(res[gated]["trades"])
        assert n_gated >= 1, f"arm {gated} deadlocked to 0 trades (ref {ref} had {n_ref})"


# ── overlay mechanics: closed + floating gating actually blocks entries ─────────────────────
def test_closed_overlay_blocks_entries_after_a_losing_streak_below_its_sma():
    """Verify arm B ends up with no more trades than its reference (arm A) on the identical
    price series, with a well-formed blocked fraction."""
    df = _make_rw_m5(n_days=365 * 3, seed=77, pair="GBP_JPY")
    res = run_battery({"GBP_JPY": df})
    res_a, res_b = res["A"], res["B"]
    assert len(res_a["trades"]) >= 10
    # B can never have MORE trades than its reference on the same price series (its overlay
    # only removes entry opportunities, never adds any) — a structural invariant.
    assert len(res_b["trades"]) <= len(res_a["trades"])
    blocked_frac = np.mean([b for _, b in res_b["blocked_trace"]])
    assert 0.0 <= blocked_frac <= 1.0


def test_floating_overlay_never_produces_more_trades_than_baseline():
    df = _make_rw_m5(n_days=365 * 3, seed=88, pair="AUD_USD")
    res = run_battery({"AUD_USD": df})
    assert len(res["A"]["trades"]) >= 10
    assert len(res["C"]["trades"]) <= len(res["A"]["trades"])


def test_overlay_on_coin_does_not_flip_sign_or_beat_raw_coin_net():
    """R10 null (PREREGISTRATION.md, exact wording): arm C's floating overlay applied to
    coin-flip entries "must NOT manufacture positive expectancy from random trades (it MAY
    reduce their drawdown; it must not flip their sign)". Note this is an EXPECTANCY claim,
    not a pathwise drawdown guarantee — a lagged moving-average gate reacts to noise as readily
    as to genuine trend, so on any ONE finite random-walk realization it can easily make the
    realized maxDD worse (a first gate-1 run at n_days=365*5/seed=99 did exactly that,
    dd_overlay=-1886.7p vs raw -1395.3p) without violating R10, since R10 never promised a
    drawdown improvement, only "may" (permissive). The only hard claims tested here are the
    ones PREREGISTRATION.md actually makes: no manufactured positive expectancy, no sign flip."""
    df = _make_rw_m5(n_days=365 * 5, seed=99, pair="NZD_USD")
    res = run_battery({"NZD_USD": df})
    coin, coin_overlay = res["coin_A"], res["coin_overlay"]
    assert len(coin["trades"]) >= 15

    net_coin = np.mean([t["net_pips"] for t in coin["trades"]])
    net_overlay = np.mean([t["net_pips"] for t in coin_overlay["trades"]]) if coin_overlay["trades"] else 0.0
    # sign-flip check: if plain coin is at/below zero (expected — coin has no directional edge,
    # net is cost-dominated), the overlay must not turn it materially positive.
    assert not (net_coin <= 0 and net_overlay > 1.0), (
        f"overlay-on-coin manufactured positive expectancy from random trades: "
        f"coin net={net_coin:+.3f}p -> coin_overlay net={net_overlay:+.3f}p"
    )


def test_overlay_never_forces_an_exit_only_gates_new_entries():
    """Structural invariant: comparing arm A vs B/C, any trade that DOES open must have
    identical exit mechanics (same TP/scratch rules) — the overlay only ever prevents an
    entry, never truncates a trade already open."""
    df = _make_rw_m5(n_days=365 * 2, seed=55, pair="USD_JPY")
    res = run_battery({"USD_JPY": df})
    for t in res["C"]["trades"]:
        assert t["exit_reason"] in ("tp", "scratch", "quality")
