#!/usr/bin/env python3
"""
Numba-JIT compiled NEAT genome evaluator.
=========================================
Compiles the ENTIRE evaluation loop (network forward pass + trading env step
+ trade tracking) into a single @njit function. Zero Python overhead per bar.

Typical speedup: 10-50× over pure Python env.step() loop.
"""

import numpy as np
from numba import njit
from numba.types import float64, int64, boolean
import neat
import math

# Register sin/cos + wavelet (ricker/morlet) activations with NEAT-Python.
# gauss is a neat-python built-in. ricker = mexican-hat = (1-x^2)e^{-x^2/2};
# morlet (real) = cos(5x)e^{-x^2/2}. Additive — does not alter existing IDs.
def _sin_activation(x):
    return math.sin(x)
def _cos_activation(x):
    return math.cos(x)
def _ricker_activation(x):
    return (1.0 - x * x) * math.exp(-x * x / 2.0)
def _morlet_activation(x):
    return math.cos(5.0 * x) * math.exp(-x * x / 2.0)
def _dog_activation(x):                      # derivative-of-Gaussian (odd wavelet)
    return -x * math.exp(-x * x / 2.0)
def _sech_activation(x):                     # hyperbolic-secant bump
    return 1.0 / math.cosh(x)
def _sinc_activation(x):                     # normalized sinc (oscillatory-decay)
    return 1.0 if abs(x) < 1e-9 else math.sin(x) / x
try:
    neat.genome.DefaultGenome.add_activation('sin', _sin_activation)
    neat.genome.DefaultGenome.add_activation('cos', _cos_activation)
    neat.genome.DefaultGenome.add_activation('ricker', _ricker_activation)
    neat.genome.DefaultGenome.add_activation('morlet', _morlet_activation)
    neat.genome.DefaultGenome.add_activation('dog', _dog_activation)
    neat.genome.DefaultGenome.add_activation('sech', _sech_activation)
    neat.genome.DefaultGenome.add_activation('sinc', _sinc_activation)
except Exception:
    pass  # Already registered


# ── Network extraction: NEAT genome → numpy arrays ─────────────────────────

# Activation function IDs for numba
ACT_TANH = 0
ACT_SIGMOID = 1
ACT_RELU = 2
ACT_IDENTITY = 3
ACT_SIN = 4
ACT_COS = 5
ACT_GAUSS = 6
ACT_RICKER = 7
ACT_MORLET = 8
ACT_DOG = 9
ACT_SECH = 10
ACT_SINC = 11

_ACT_MAP = {
    'tanh_activation': ACT_TANH,
    'sigmoid_activation': ACT_SIGMOID,
    'relu_activation': ACT_RELU,
    'identity_activation': ACT_IDENTITY,
    'sin_activation': ACT_SIN,
    'cos_activation': ACT_COS,
    'gauss_activation': ACT_GAUSS,
    'tanh': ACT_TANH,
    'sigmoid': ACT_SIGMOID,
    'relu': ACT_RELU,
    'identity': ACT_IDENTITY,
    'sin': ACT_SIN,
    'cos': ACT_COS,
    'gauss': ACT_GAUSS,
    'ricker_activation': ACT_RICKER,
    'morlet_activation': ACT_MORLET,
    'ricker': ACT_RICKER,
    'morlet': ACT_MORLET,
    'dog_activation': ACT_DOG,
    'dog': ACT_DOG,
    'sech_activation': ACT_SECH,
    'sech': ACT_SECH,
    'sinc_activation': ACT_SINC,
    'sinc': ACT_SINC,
}


def extract_network(genome, config):
    """Convert NEAT genome to numpy arrays for numba evaluation.

    Returns:
        n_inputs: int
        n_outputs: int
        n_nodes: int (total eval nodes: outputs + hidden)
        node_bias: float64[n_nodes]
        node_response: float64[n_nodes]
        node_act: int64[n_nodes]  (activation function ID)
        conn_from: int64[n_connections]  (source index into values array)
        conn_to: int64[n_connections]    (target node index in node arrays)
        conn_weight: float64[n_connections]
        output_indices: int64[n_outputs]  (indices into values array for outputs)
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Build node ID → index mapping
    # Values array: [input_0, input_1, ..., input_N, eval_node_0, eval_node_1, ...]
    n_inputs = len(net.input_nodes)
    n_outputs = len(net.output_nodes)

    # Map all node IDs to sequential indices
    id_to_idx = {}
    for i, nid in enumerate(net.input_nodes):
        id_to_idx[nid] = i  # inputs: 0..n_inputs-1

    n_eval = len(net.node_evals)
    for i, (node, _, _, _, _, _) in enumerate(net.node_evals):
        id_to_idx[node] = n_inputs + i  # eval nodes: n_inputs..n_inputs+n_eval-1

    total_values = n_inputs + n_eval

    # Extract node properties (in topological order)
    node_bias = np.zeros(n_eval, dtype=np.float64)
    node_response = np.zeros(n_eval, dtype=np.float64)
    node_act = np.zeros(n_eval, dtype=np.int64)

    # Collect all connections
    all_conn_from = []
    all_conn_to = []
    all_conn_weight = []

    for i, (node, act_func, agg_func, bias, response, links) in enumerate(net.node_evals):
        node_bias[i] = bias
        node_response[i] = response
        act_name = act_func.__name__ if hasattr(act_func, '__name__') else str(act_func)
        node_act[i] = _ACT_MAP.get(act_name, ACT_TANH)

        for src_id, weight in links:
            all_conn_from.append(id_to_idx[src_id])
            all_conn_to.append(i)  # target node INDEX in eval array
            all_conn_weight.append(weight)

    conn_from = np.array(all_conn_from, dtype=np.int64)
    conn_to = np.array(all_conn_to, dtype=np.int64)
    conn_weight = np.array(all_conn_weight, dtype=np.float64)

    # Output node indices in the values array
    output_indices = np.array([id_to_idx[nid] for nid in net.output_nodes], dtype=np.int64)

    return (n_inputs, n_outputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight,
            output_indices)


# ── Numba JIT: network forward pass ────────────────────────────────────────

@njit(cache=True)
def _activate(values, n_inputs, n_eval,
              node_bias, node_response, node_act,
              conn_from, conn_to, conn_weight):
    """Forward pass through extracted NEAT network. Modifies values in-place."""
    # Reset eval node values
    for i in range(n_eval):
        values[n_inputs + i] = 0.0

    # Accumulate weighted inputs for each eval node
    # (aggregation = sum, which is NEAT's default)
    node_sums = np.zeros(n_eval)
    for c in range(len(conn_from)):
        src_val = values[conn_from[c]]
        node_sums[conn_to[c]] += src_val * conn_weight[c]

    # Apply activation functions in topological order
    for i in range(n_eval):
        x = node_bias[i] + node_response[i] * node_sums[i]
        act = node_act[i]
        if act == 0:  # tanh
            values[n_inputs + i] = np.tanh(x)
        elif act == 1:  # sigmoid
            values[n_inputs + i] = 1.0 / (1.0 + np.exp(-x))
        elif act == 2:  # relu
            values[n_inputs + i] = max(0.0, x)
        elif act == 4:  # sin
            values[n_inputs + i] = np.sin(x)
        elif act == 5:  # cos
            values[n_inputs + i] = np.cos(x)
        elif act == 6:  # gauss
            values[n_inputs + i] = np.exp(-x * x)
        elif act == 7:  # ricker (mexican-hat wavelet)
            values[n_inputs + i] = (1.0 - x * x) * np.exp(-x * x / 2.0)
        elif act == 8:  # morlet wavelet (real)
            values[n_inputs + i] = np.cos(5.0 * x) * np.exp(-x * x / 2.0)
        elif act == 9:  # derivative-of-Gaussian (odd wavelet)
            values[n_inputs + i] = -x * np.exp(-x * x / 2.0)
        elif act == 10:  # sech bump
            values[n_inputs + i] = 1.0 / np.cosh(x)
        elif act == 11:  # normalized sinc
            values[n_inputs + i] = 1.0 if abs(x) < 1e-9 else np.sin(x) / x
        else:  # identity
            values[n_inputs + i] = x


# ── Numba JIT: full evaluation loop ────────────────────────────────────────

@njit(cache=True)
def evaluate_mtf_jit(
    # M5 precomputed indicators
    m5_slope3, m5_swing_rp, m5_tec5, m5_dtec13_3,
    # H1 mapping
    h1_struct_slope, m5_to_h1,
    # P&F mapping
    pnf_tec8, m5_to_pnf_count,
    # M5 price data for trading
    m5_close, m5_atr,
    # Trading params
    pip, spread_pips, max_bars, max_hold,
    # Reward params
    r_gamma,  # AMDDP beta
    # Network arrays
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices,
    # Window offset (0 = start from beginning)
    start_offset,
):
    """Run full genome evaluation. Returns (n_trades, total_pnl, sharpe,
    win_rate, mean_pnl, avg_mfe, avg_mae, avg_capture, max_dd, avg_bars)."""

    values = np.zeros(total_values)
    data_len = len(m5_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 20, 20)  # warmup

    # Trade tracking
    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    # Position state
    position = 0  # -1, 0, +1
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0
    mae = 0.0
    hwm_pnl = 0.0
    dd_sum = 0.0
    prev_max_dd = 0.0

    for i in range(start_bar, end_bar):
        # ── Build observation ──
        slope3_val = m5_slope3[i]
        srp_val = m5_swing_rp[i]
        tec5_val = m5_tec5[i]
        dtec_val = m5_dtec13_3[i]

        h1_idx = m5_to_h1[i]
        h1_sts = h1_struct_slope[h1_idx]

        pnf_count = m5_to_pnf_count[i]
        pnf_t8 = pnf_tec8[pnf_count - 1] if pnf_count > 0 else 0.0

        # Unrealized PnL
        if position != 0:
            pnl_raw = (m5_close[i] - entry_price) * position / pip - spread_pips
            atr_pips = m5_atr[i] / pip if m5_atr[i] > 0 else 1.0
            unrealized = pnl_raw / atr_pips
        else:
            pnl_raw = 0.0
            unrealized = 0.0

        # Scale observations
        values[0] = np.tanh(slope3_val)
        values[1] = srp_val  # native [-1,+1]
        values[2] = tec5_val  # native [-1,+1]
        values[3] = (2.0 / 3.14159265) * np.arctan(dtec_val * 5.0)
        values[4] = (2.0 / 3.14159265) * np.arctan(h1_sts * 1000.0)
        values[5] = pnf_t8  # native [-1,+1]
        values[6] = np.tanh(unrealized)

        # ── Network forward pass ──
        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        # Get action (argmax of output nodes)
        action = 0
        best_val = values[output_indices[0]]
        for k in range(1, 4):
            v = values[output_indices[k]]
            if v > best_val:
                best_val = v
                action = k

        # ── Update MFE/MAE/AMDDP ──
        if position != 0:
            pnl_pips = (m5_close[i] - entry_price) * position / pip - spread_pips
            if pnl_pips > mfe:
                mfe = pnl_pips
            if -pnl_pips > mae:
                mae = -pnl_pips
            if pnl_pips > hwm_pnl:
                hwm_pnl = pnl_pips
            dd_open = max(0.0, -pnl_pips)
            dd_hwm = max(0.0, hwm_pnl - pnl_pips)
            cur_dd = max(dd_open, dd_hwm)
            if cur_dd > prev_max_dd:
                dd_sum += cur_dd - prev_max_dd
                prev_max_dd = cur_dd

        # ── Force close at max hold ──
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (m5_close[i] - entry_price) * position / pip - spread_pips
            bars_held = i - entry_bar
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_mfes[n_trades] = mfe
                trade_maes[n_trades] = mae
                trade_bars[n_trades] = bars_held
                trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                trade_ddsums[n_trades] = dd_sum
                n_trades += 1
            position = 0
            mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0

        # ── Execute action ──
        if action == 1:  # BUY
            if position == -1:  # close short first
                pnl = (m5_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum
                    n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
            if position == 0:
                position = 1
                entry_price = m5_close[i]
                entry_bar = i

        elif action == 2:  # SELL
            if position == 1:  # close long first
                pnl = (m5_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum
                    n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
            if position == 0:
                position = -1
                entry_price = m5_close[i]
                entry_bar = i

        elif action == 3:  # CLOSE
            if position != 0:
                pnl = (m5_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum
                    n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0

    # Close any remaining position
    if position != 0 and end_bar > start_bar:
        pnl = (m5_close[end_bar - 1] - entry_price) * position / pip - spread_pips
        bars_held = (end_bar - 1) - entry_bar
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_mfes[n_trades] = mfe
            trade_maes[n_trades] = mae
            trade_bars[n_trades] = bars_held
            trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
            trade_ddsums[n_trades] = dd_sum
            n_trades += 1

    # ── Compute stats ──
    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pnls = trade_pnls[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades):
        total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades

    # Std
    var = 0.0
    for j in range(n_trades):
        var += (pnls[j] - mean_pnl) ** 2
    std = (var / n_trades) ** 0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades ** 0.5) if std > 0 else 0.0

    # Win rate
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0:
            wins += 1
    win_rate = 100.0 * wins / n_trades

    # Avg MFE, MAE, capture, bars
    avg_mfe = 0.0; avg_mae = 0.0; avg_cap = 0.0; avg_bars = 0.0
    for j in range(n_trades):
        avg_mfe += trade_mfes[j]
        avg_mae += trade_maes[j]
        avg_cap += trade_captures[j]
        avg_bars += trade_bars[j]
    avg_mfe /= n_trades
    avg_mae /= n_trades
    avg_cap /= n_trades
    avg_bars /= n_trades

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for j in range(n_trades):
        cum += pnls[j]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Avg PnL/MAE (risk-adjusted return per trade)
    avg_pnl_mae = 0.0
    for j in range(n_trades):
        mae_j = trade_maes[j]
        if mae_j > 0.1:
            avg_pnl_mae += trade_pnls[j] / mae_j
        else:
            avg_pnl_mae += trade_pnls[j] / 0.1 if trade_pnls[j] > 0 else 0.0
    avg_pnl_mae /= n_trades

    # Avg AMDDP/MAE: (PnL - β×dd_sum) / MAE per trade
    # β=0.10 (10%) for meaningful path penalty
    avg_amddp_mae = 0.0
    beta_amddp = 0.10
    for j in range(n_trades):
        mae_j = trade_maes[j]
        amddp_val = trade_pnls[j] - beta_amddp * trade_ddsums[j]
        if mae_j > 0.1:
            avg_amddp_mae += amddp_val / mae_j
        else:
            avg_amddp_mae += amddp_val / 0.1 if amddp_val > 0 else 0.0
    avg_amddp_mae /= n_trades

    # PnL / (MAE × sqrt(bars)) — time-decay: fast clean trades score highest
    avg_pnl_mae_speed = 0.0
    for j in range(n_trades):
        mae_j = trade_maes[j]
        bars_j = trade_bars[j]
        sqrt_bars = (max(bars_j, 1.0)) ** 0.5
        if mae_j > 0.1:
            avg_pnl_mae_speed += trade_pnls[j] / (mae_j * sqrt_bars)
        else:
            avg_pnl_mae_speed += trade_pnls[j] / (0.1 * sqrt_bars) if trade_pnls[j] > 0 else 0.0
    avg_pnl_mae_speed /= n_trades

    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, avg_amddp_mae, avg_pnl_mae_speed)


def run_genome_fast(genome, config, mtf_data, pip, spread_pips,
                    max_bars=12000, max_hold=100, r_gamma=0.01,
                    start_offset=0):
    """Fast genome evaluation using numba JIT. Drop-in replacement for _run_genome_on_data."""

    # Extract network
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    # Run JIT evaluation
    result = evaluate_mtf_jit(
        mtf_data["_m5_slope3"], mtf_data["_m5_swing_rp"],
        mtf_data["_m5_tec5"], mtf_data["_m5_dtec13_3"],
        mtf_data["h1_struct_slope"], mtf_data["m5_to_h1"],
        mtf_data["_pnf_tec8"], mtf_data["m5_to_pnf_count"],
        mtf_data["m5_close"], mtf_data["m5_atr"],
        pip, spread_pips, max_bars, max_hold,
        r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices,
        start_offset,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, avg_amddp_mae) = result

    return {
        "n_trades": int(n_trades),
        "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4),
        "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2),
        "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1),
        "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1),
        "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "avg_amddp_mae": round(float(avg_amddp_mae), 3),
    }


@njit(cache=True)
def _map_m5_to_tick(m5_indicators, tick_to_m5, n_tick):
    """Map M5-resolution indicator array to tick-resolution via index lookup."""
    out = np.zeros(n_tick)
    for i in range(n_tick):
        out[i] = m5_indicators[tick_to_m5[i]]
    return out


def precompute_mtf_indicators(mtf_data, spread_pips):
    """Precompute all MTF indicators and store in data dict.
    Indicators are computed on M5 data, then mapped to tick resolution.
    Call once after loading/slicing data. Adds _m5_slope3, _m5_tec5, etc."""
    from trading_env_mtf import _precompute_m5_indicators, _precompute_pnf_tec8

    pip = mtf_data["pip"]
    min_swing = 3.0 * spread_pips * pip
    n_tick = mtf_data["n_m5"]  # n_m5 is actually n_tick in new naming

    # Check if we have separate M5 indicator data (M1 cadence) or same (M5 cadence)
    if "ind_m5_close" in mtf_data:
        # M1 cadence: compute on M5 data, map to tick
        slope3, tec5, dtec13_3, swing_rp = _precompute_m5_indicators(
            mtf_data["ind_m5_close"], mtf_data["ind_m5_high"],
            mtf_data["ind_m5_low"], mtf_data["ind_m5_atr"], min_swing)

        tick_to_m5 = mtf_data["tick_to_m5"]
        mtf_data["_m5_slope3"] = _map_m5_to_tick(slope3, tick_to_m5, n_tick)
        mtf_data["_m5_swing_rp"] = _map_m5_to_tick(swing_rp, tick_to_m5, n_tick)
        mtf_data["_m5_tec5"] = _map_m5_to_tick(tec5, tick_to_m5, n_tick)
        mtf_data["_m5_dtec13_3"] = _map_m5_to_tick(dtec13_3, tick_to_m5, n_tick)
    else:
        # M5 cadence: compute directly on tick data (which IS M5)
        slope3, tec5, dtec13_3, swing_rp = _precompute_m5_indicators(
            mtf_data["m5_close"], mtf_data["m5_high"],
            mtf_data["m5_low"], mtf_data["m5_atr"], min_swing)

        mtf_data["_m5_slope3"] = slope3
        mtf_data["_m5_swing_rp"] = swing_rp
        mtf_data["_m5_tec5"] = tec5
        mtf_data["_m5_dtec13_3"] = dtec13_3

    mtf_data["_pnf_tec8"] = _precompute_pnf_tec8(
        mtf_data["pnf_close"], mtf_data["n_pnf"])

    return mtf_data


# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL 3-INPUT EVALUATOR
# Inputs: MomentumConsistency, DistToH1SR, UnrealizedPnL
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _precompute_h1_sr_levels(h1_high, h1_low, h1_close, min_swing):
    """Precompute nearest S/R level for each H1 bar from zigzag swing points.
    Returns (support, resistance) arrays — last confirmed swing low/high prices."""
    n = len(h1_close)
    support = np.zeros(n)
    resistance = np.zeros(n)

    last_sh = h1_high[0]
    last_sl = h1_low[0]
    direction = 0
    running_high = h1_high[0]
    running_low = h1_low[0]

    for i in range(1, n):
        if h1_high[i] > running_high:
            running_high = h1_high[i]
        if h1_low[i] < running_low:
            running_low = h1_low[i]

        if direction == 0:
            if running_high - h1_low[i] >= min_swing:
                last_sh = running_high
                direction = -1
                running_low = h1_low[i]
            elif h1_high[i] - running_low >= min_swing:
                last_sl = running_low
                direction = 1
                running_high = h1_high[i]
        elif direction == 1:
            if running_high - h1_low[i] >= min_swing:
                last_sh = running_high
                direction = -1
                running_low = h1_low[i]
        else:
            if h1_high[i] - running_low >= min_swing:
                last_sl = running_low
                direction = 1
                running_high = h1_high[i]

        support[i] = last_sl
        resistance[i] = last_sh

    return support, resistance


@njit(cache=True)
def _precompute_momentum_consistency(close, atr, n_lags=5):
    """MC = sign(Σ ΔC[t-4..t]) × (count_same_sign / n_lags). Range [-1, +1]."""
    n = len(close)
    mc = np.zeros(n)

    for i in range(n_lags, n):
        deltas = np.zeros(n_lags)
        for k in range(n_lags):
            deltas[k] = close[i - n_lags + 1 + k] - close[i - n_lags + k]

        total = 0.0
        for k in range(n_lags):
            total += deltas[k]
        sign_total = 1.0 if total > 0 else (-1.0 if total < 0 else 0.0)

        # Count how many deltas match the dominant direction
        count_same = 0
        for k in range(n_lags):
            if sign_total > 0 and deltas[k] > 0:
                count_same += 1
            elif sign_total < 0 and deltas[k] < 0:
                count_same += 1

        mc[i] = sign_total * count_same / n_lags

    return mc


@njit(cache=True)
def evaluate_minimal_jit(
    # Precomputed indicators (tick-resolution)
    momentum_consistency,
    # H1 S/R levels + mapping
    h1_support, h1_resistance, tick_to_h1,
    # Tick price data
    tick_close, tick_atr,
    # Trading params
    pip, spread_pips, max_bars, max_hold,
    r_gamma,
    # Network
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices,
    start_offset,
):
    """3-input minimal evaluator: MC + dist_to_SR + UPnL."""
    values = np.zeros(total_values)
    data_len = len(tick_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 20, 20)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0
    mae = 0.0
    hwm_pnl = 0.0
    dd_sum = 0.0
    prev_max_dd = 0.0

    for i in range(start_bar, end_bar):
        # ── Observation: 4 inputs ──
        mc = momentum_consistency[i]

        # Distance to nearest H1 S/R
        h1_idx = tick_to_h1[i]
        sup = h1_support[h1_idx]
        res = h1_resistance[h1_idx]
        sr_range = res - sup
        if sr_range > 0:
            sr_pos = 2.0 * (tick_close[i] - sup) / sr_range - 1.0
        else:
            sr_pos = 0.0

        # Unrealized PnL + Pips from Peak
        if position != 0:
            pnl_raw = (tick_close[i] - entry_price) * position / pip - spread_pips
            atr_pips = tick_atr[i] / pip if tick_atr[i] > 0 else 1.0
            unrealized = pnl_raw / atr_pips
            # Pips from peak: how much profit we've given back
            pfp = (mfe - pnl_raw) / atr_pips if mfe > 0 else 0.0
        else:
            pnl_raw = 0.0
            unrealized = 0.0
            pfp = 0.0

        values[0] = mc  # [-1, +1] direction + quality
        values[1] = np.tanh(sr_pos)  # [-1, +1] structure position
        values[2] = np.tanh(unrealized)  # [-1, +1] current P/L
        values[3] = np.tanh(pfp)  # [0, +1] giveback alert (0=at peak, +1=giving back)

        # ── Network forward pass ──
        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        action = 0
        best_val = values[output_indices[0]]
        for k in range(1, 4):
            v = values[output_indices[k]]
            if v > best_val:
                best_val = v
                action = k

        # ── Update MFE/MAE/AMDDP ──
        if position != 0:
            pnl_pips = (tick_close[i] - entry_price) * position / pip - spread_pips
            if pnl_pips > mfe:
                mfe = pnl_pips
            if -pnl_pips > mae:
                mae = -pnl_pips
            if pnl_pips > hwm_pnl:
                hwm_pnl = pnl_pips
            dd_open = max(0.0, -pnl_pips)
            dd_hwm = max(0.0, hwm_pnl - pnl_pips)
            cur_dd = max(dd_open, dd_hwm)
            if cur_dd > prev_max_dd:
                dd_sum += cur_dd - prev_max_dd
                prev_max_dd = cur_dd

        # ── Force close at max hold ──
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (tick_close[i] - entry_price) * position / pip - spread_pips
            bars_held = i - entry_bar
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_mfes[n_trades] = mfe
                trade_maes[n_trades] = mae
                trade_bars[n_trades] = bars_held
                trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                trade_ddsums[n_trades] = dd_sum
                n_trades += 1
            position = 0
            mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0

        # ── Execute action ──
        if action == 1:  # BUY
            if position == -1:
                pnl = (tick_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl; trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae; trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum; n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
            if position == 0:
                position = 1; entry_price = tick_close[i]; entry_bar = i

        elif action == 2:  # SELL
            if position == 1:
                pnl = (tick_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl; trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae; trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum; n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
            if position == 0:
                position = -1; entry_price = tick_close[i]; entry_bar = i

        elif action == 3:  # CLOSE
            if position != 0:
                pnl = (tick_close[i] - entry_price) * position / pip - spread_pips
                bars_held = i - entry_bar
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl; trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae; trade_bars[n_trades] = bars_held
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    trade_ddsums[n_trades] = dd_sum; n_trades += 1
                position = 0
                mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0

    # Close remaining
    if position != 0 and end_bar > start_bar:
        pnl = (tick_close[end_bar - 1] - entry_price) * position / pip - spread_pips
        bars_held = (end_bar - 1) - entry_bar
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl; trade_mfes[n_trades] = mfe
            trade_maes[n_trades] = mae; trade_bars[n_trades] = bars_held
            trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
            trade_ddsums[n_trades] = dd_sum; n_trades += 1

    # Stats (same as MTF evaluator)
    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pnls = trade_pnls[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades): total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades

    var = 0.0
    for j in range(n_trades): var += (pnls[j] - mean_pnl) ** 2
    std = (var / n_trades) ** 0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades ** 0.5) if std > 0 else 0.0

    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0: wins += 1
    win_rate = 100.0 * wins / n_trades

    avg_mfe = 0.0; avg_mae = 0.0; avg_cap = 0.0; avg_bars = 0.0
    for j in range(n_trades):
        avg_mfe += trade_mfes[j]; avg_mae += trade_maes[j]
        avg_cap += trade_captures[j]; avg_bars += trade_bars[j]
    avg_mfe /= n_trades; avg_mae /= n_trades
    avg_cap /= n_trades; avg_bars /= n_trades

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for j in range(n_trades):
        cum += pnls[j]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    avg_pnl_mae = 0.0
    for j in range(n_trades):
        m = trade_maes[j]
        if m > 0.1: avg_pnl_mae += pnls[j] / m
        else: avg_pnl_mae += pnls[j] / 0.1 if pnls[j] > 0 else 0.0
    avg_pnl_mae /= n_trades

    avg_amddp_mae = 0.0
    for j in range(n_trades):
        m = trade_maes[j]
        av = pnls[j] - 0.10 * trade_ddsums[j]
        if m > 0.1: avg_amddp_mae += av / m
        else: avg_amddp_mae += av / 0.1 if av > 0 else 0.0
    avg_amddp_mae /= n_trades

    avg_pnl_mae_speed = 0.0
    for j in range(n_trades):
        m = trade_maes[j]
        sb = (max(trade_bars[j], 1.0)) ** 0.5
        if m > 0.1: avg_pnl_mae_speed += pnls[j] / (m * sb)
        else: avg_pnl_mae_speed += pnls[j] / (0.1 * sb) if pnls[j] > 0 else 0.0
    avg_pnl_mae_speed /= n_trades

    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, avg_amddp_mae, avg_pnl_mae_speed)


def run_genome_minimal(genome, config, mtf_data, pip, spread_pips,
                       max_bars=12000, max_hold=100, r_gamma=0.01,
                       start_offset=0):
    """Fast 3-input minimal genome evaluation."""
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    result = evaluate_minimal_jit(
        mtf_data["_momentum_consistency"],
        mtf_data["_h1_support"], mtf_data["_h1_resistance"],
        mtf_data["m5_to_h1"],
        mtf_data["m5_close"], mtf_data["m5_atr"],
        pip, spread_pips, max_bars, max_hold, r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices, start_offset,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, avg_amddp_mae, avg_pnl_mae_speed) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "avg_amddp_mae": round(float(avg_amddp_mae), 3),
        "avg_pnl_mae_speed": round(float(avg_pnl_mae_speed), 3),
    }


def precompute_minimal_indicators(mtf_data, spread_pips):
    """Precompute 3-input minimal indicators: MC + H1 S/R."""
    from trading_env_mtf import _precompute_hsp_lsp_slopes

    pip = mtf_data["pip"]
    min_swing = 3.0 * spread_pips * pip

    # Momentum consistency on tick data
    mc = _precompute_momentum_consistency(mtf_data["m5_close"], mtf_data["m5_atr"], 5)
    mtf_data["_momentum_consistency"] = mc

    # H1 S/R levels — need H1 high/low data
    # These are stored in build_mtf_data but we need the raw H1 arrays
    # For now, use the H1 struct slope data which already has swing detection
    # Actually we need to pass H1 OHLC through build_mtf_data

    # Check if we have H1 OHLC in the data dict
    if "ind_h1_high" in mtf_data:
        h1_high = mtf_data["ind_h1_high"]
        h1_low = mtf_data["ind_h1_low"]
        h1_close = mtf_data["ind_h1_close"]
    else:
        # Fallback: reconstruct from existing data
        # We need to add H1 OHLC to build_mtf_data
        raise KeyError("H1 OHLC not in mtf_data — need to add ind_h1_high/low/close to build_mtf_data")

    sup, res = _precompute_h1_sr_levels(h1_high, h1_low, h1_close, min_swing)
    mtf_data["_h1_support"] = sup
    mtf_data["_h1_resistance"] = res

    return mtf_data


# ═══════════════════════════════════════════════════════════════════════════
# P&F CADENCE EVALUATOR — decisions on box fills, no look-ahead
# Same 3 inputs (MC + H1_SR + UPnL) but stepping through P&F boxes
# H1 S/R uses only COMPLETED H1 bars at time of each box (causal)
# Entry/exit at actual S5 prices, not box levels
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _precompute_pnf_mc(pnf_close, n_lags=5):
    """Momentum Consistency on P&F box price changes."""
    n = len(pnf_close)
    mc = np.zeros(n)
    for i in range(n_lags, n):
        deltas = np.zeros(n_lags)
        for k in range(n_lags):
            deltas[k] = pnf_close[i - n_lags + 1 + k] - pnf_close[i - n_lags + k]
        total = 0.0
        for k in range(n_lags):
            total += deltas[k]
        sign_total = 1.0 if total > 0 else (-1.0 if total < 0 else 0.0)
        count_same = 0
        for k in range(n_lags):
            if sign_total > 0 and deltas[k] > 0:
                count_same += 1
            elif sign_total < 0 and deltas[k] < 0:
                count_same += 1
        mc[i] = sign_total * count_same / n_lags
    return mc


@njit(cache=True)
def evaluate_pnf_cadence_jit(
    pnf_close, pnf_actual_price, pnf_mc,
    h1_support, h1_resistance, pnf_to_h1,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
    action_map=np.array([0, 1, 2, 3], dtype=np.int64),
):
    """P&F cadence: inputs (MC + H1_SR + UPnL), decisions on box fills.
    action_map: maps argmax index → action code (0=HOLD,1=BUY,2=SELL,3=CLOSE).
    For 4-output: [0,1,2,3] (identity). For 3-output long: [0,1,3]. Short: [0,2,3]."""
    values = np.zeros(total_values)
    data_len = len(pnf_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
    n_long = 0; n_short = 0

    for i in range(start_bar, end_bar):
        mc = pnf_mc[i]

        # H1 S/R — causal: pnf_to_h1[i] points to last COMPLETED H1 bar
        h1_idx = pnf_to_h1[i]
        sup = h1_support[h1_idx]
        res = h1_resistance[h1_idx]
        price_now = pnf_close[i] * pip
        sr_range = res - sup
        sr_pos = 2.0 * (price_now - sup) / sr_range - 1.0 if sr_range > 0 else 0.0

        actual_now = pnf_actual_price[i]
        if position != 0:
            pnl_pips = (actual_now - entry_price) * position - spread_pips
        else:
            pnl_pips = 0.0

        values[0] = mc
        values[1] = np.tanh(sr_pos)
        values[2] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        n_outputs = len(action_map)
        raw_action = 0
        best_val = values[output_indices[0]]
        for k in range(1, n_outputs):
            v = values[output_indices[k]]
            if v > best_val: best_val = v; raw_action = k
        action = action_map[raw_action]

        # Update MFE/MAE/AMDDP
        if position != 0:
            pnl_now = (actual_now - entry_price) * position - spread_pips
            if pnl_now > mfe: mfe = pnl_now
            if -pnl_now > mae: mae = -pnl_now
            if pnl_now > hwm_pnl: hwm_pnl = pnl_now
            cur_dd = max(max(0.0, -pnl_now), max(0.0, hwm_pnl - pnl_now))
            if cur_dd > prev_max_dd: dd_sum += cur_dd - prev_max_dd; prev_max_dd = cur_dd

        # Force close
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (actual_now - entry_price) * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                trade_bars[n_trades]=i-entry_bar
                trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                trade_ddsums[n_trades]=dd_sum; n_trades+=1
            position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

        if action == 1:  # BUY
            if position == -1:
                pnl=(actual_now-entry_price)*position-spread_pips
                if n_trades<max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0
            if position == 0: position=1; entry_price=actual_now; entry_bar=i; n_long+=1

        elif action == 2:  # SELL
            if position == 1:
                pnl=(actual_now-entry_price)*position-spread_pips
                if n_trades<max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0
            if position == 0: position=-1; entry_price=actual_now; entry_bar=i; n_short+=1

        elif action == 3:  # CLOSE
            if position != 0:
                pnl=(actual_now-entry_price)*position-spread_pips
                if n_trades<max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

    if position != 0 and end_bar > start_bar:
        pnl=(pnf_actual_price[end_bar-1]-entry_price)*position-spread_pips
        if n_trades<max_trades:
            trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
            trade_bars[n_trades]=(end_bar-1)-entry_bar
            trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
            trade_ddsums[n_trades]=dd_sum; n_trades+=1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pnls=trade_pnls[:n_trades]; total_pnl=0.0
    for j in range(n_trades): total_pnl+=pnls[j]
    mean_pnl=total_pnl/n_trades
    var=0.0
    for j in range(n_trades): var+=(pnls[j]-mean_pnl)**2
    std=(var/n_trades)**0.5 if n_trades>1 else 1.0
    sharpe=mean_pnl/std*(n_trades**0.5) if std>0 else 0.0
    wins=0
    for j in range(n_trades):
        if pnls[j]>0: wins+=1
    win_rate=100.0*wins/n_trades
    avg_mfe=0.0; avg_mae=0.0; avg_cap=0.0; avg_bars=0.0
    for j in range(n_trades):
        avg_mfe+=trade_mfes[j]; avg_mae+=trade_maes[j]; avg_cap+=trade_captures[j]; avg_bars+=trade_bars[j]
    avg_mfe/=n_trades; avg_mae/=n_trades; avg_cap/=n_trades; avg_bars/=n_trades
    cum=0.0; peak_eq=0.0; max_dd=0.0
    for j in range(n_trades):
        cum+=pnls[j]
        if cum>peak_eq: peak_eq=cum
        dd=peak_eq-cum
        if dd>max_dd: max_dd=dd
    avg_pnl_mae=0.0
    for j in range(n_trades):
        m=trade_maes[j]
        if m>0.1: avg_pnl_mae+=pnls[j]/m
        else: avg_pnl_mae+=pnls[j]/0.1 if pnls[j]>0 else 0.0
    avg_pnl_mae/=n_trades
    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, float(n_long), float(n_short))


def run_genome_pnf(genome, config, data, pip, spread_pips,
                    max_bars=5000, max_hold=50, r_gamma=0.01, start_offset=0,
                    action_map=None):
    """P&F cadence genome evaluation."""
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    if action_map is None:
        action_map = np.array([0, 1, 2, 3], dtype=np.int64)

    result = evaluate_pnf_cadence_jit(
        data["pnf_close"], data["pnf_actual_price"], data["_pnf_mc"],
        data["_h1_support"], data["_h1_resistance"], data["_pnf_to_h1"],
        pip, spread_pips, max_bars, max_hold, r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices, start_offset,
        action_map,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, n_long, n_short) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "n_long": int(n_long), "n_short": int(n_short),
    }


def precompute_pnf_cadence_indicators(data, spread_pips):
    """Precompute P&F cadence indicators. No look-ahead — H1 S/R causal via pnf_to_h1."""
    pip = data["pip"]
    min_swing = 3.0 * spread_pips * pip

    data["_pnf_mc"] = _precompute_pnf_mc(data["pnf_close"], 5)

    # H1 S/R (causal — computed on H1 data only, no future info)
    sup, res = _precompute_h1_sr_levels(
        data["ind_h1_high"], data["ind_h1_low"], data["ind_h1_close"], min_swing)
    data["_h1_support"] = sup
    data["_h1_resistance"] = res

    # Map P&F boxes → H1 bars causally via source timestamps
    # pnf_source_idx[i] = which S5 bar created box i
    # We map proportionally: S5 position → H1 position
    n_pnf = data["n_pnf"]
    n_h1 = data["n_h1"]
    pnf_source = data.get("pnf_source_idx", None)

    if pnf_source is not None:
        # Use actual S5 indices for causal mapping
        # Estimate total S5 bars from max source index
        max_s5 = max(float(pnf_source[-1]), 1.0) if n_pnf > 0 else 1.0
        pnf_to_h1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            frac = pnf_source[i] / max_s5
            h1_idx = int(frac * (n_h1 - 1))
            # Use previous H1 bar (last COMPLETED, not current)
            pnf_to_h1[i] = max(0, h1_idx - 1)
        data["_pnf_to_h1"] = pnf_to_h1
    else:
        pnf_to_h1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            pnf_to_h1[i] = max(0, int(i / max(n_pnf, 1) * (n_h1 - 1)) - 1)
        data["_pnf_to_h1"] = pnf_to_h1

    # H1 ADX(14) — for pnf_2out_adx representation (4th input)
    from indicator_detectability import compute_adx as _compute_adx
    adx_14, _, _ = _compute_adx(
        data["ind_h1_high"], data["ind_h1_low"], data["ind_h1_close"], 14)
    data["_h1_adx"] = adx_14  # 0-100 range, normalized in JIT loop

    # Ensure actual prices exist
    if "pnf_actual_price" not in data:
        data["pnf_actual_price"] = data["pnf_close"].copy()

    return data


# ═══════════════════════════════════════════════════════════════════════════
# 2-OUTPUT P&F CADENCE EVALUATOR (LONG-ONLY or SHORT-ONLY)
# Inputs: 3 (MC + H1_SR + UPnL) — same as v6 cadence
# Outputs: 2 (ENTER vs CLOSE) — clean 2-node, no HOLD output
# Direction: is_long=True → BUY/CLOSE, is_long=False → SELL/CLOSE
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_pnf_2out_jit(
    pnf_close, pnf_actual_price, pnf_mc,
    h1_support, h1_resistance, pnf_to_h1,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
    is_long,
):
    """P&F cadence with 2 outputs: output[0]=ENTER, output[1]=CLOSE.
    When flat: output[0] > output[1] → enter (long or short per is_long).
    When in position: output[1] > output[0] → close."""
    values = np.zeros(total_values)
    data_len = len(pnf_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
    n_long = 0; n_short = 0
    direction = 1 if is_long else -1

    for i in range(start_bar, end_bar):
        mc = pnf_mc[i]

        # H1 S/R — causal
        h1_idx = pnf_to_h1[i]
        sup = h1_support[h1_idx]
        res = h1_resistance[h1_idx]
        price_now = pnf_close[i] * pip
        sr_range = res - sup
        sr_pos = 2.0 * (price_now - sup) / sr_range - 1.0 if sr_range > 0 else 0.0

        actual_now = pnf_actual_price[i]
        if position != 0:
            pnl_pips = (actual_now - entry_price) * position - spread_pips
        else:
            pnl_pips = 0.0

        # 3 inputs: MC, H1_SR, UPnL
        values[0] = mc
        values[1] = np.tanh(sr_pos)
        values[2] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_enter = values[output_indices[0]]
        out_close = values[output_indices[1]]

        # Update MFE/MAE/AMDDP
        if position != 0:
            pnl_now = (actual_now - entry_price) * position - spread_pips
            if pnl_now > mfe: mfe = pnl_now
            if -pnl_now > mae: mae = -pnl_now
            if pnl_now > hwm_pnl: hwm_pnl = pnl_now
            cur_dd = max(max(0.0, -pnl_now), max(0.0, hwm_pnl - pnl_now))
            if cur_dd > prev_max_dd: dd_sum += cur_dd - prev_max_dd; prev_max_dd = cur_dd

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (actual_now - entry_price) * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                trade_bars[n_trades]=i-entry_bar
                trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                trade_ddsums[n_trades]=dd_sum; n_trades+=1
            position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

        # Decision logic
        if position == 0:
            # Flat: enter if ENTER output dominates
            if out_enter > out_close:
                position = direction
                entry_price = actual_now
                entry_bar = i
                if is_long:
                    n_long += 1
                else:
                    n_short += 1
        else:
            # In position: close if CLOSE output dominates
            if out_close > out_enter:
                pnl = (actual_now - entry_price) * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

    # Close any open position at end
    if position != 0 and end_bar > start_bar:
        pnl=(pnf_actual_price[end_bar-1]-entry_price)*position-spread_pips
        if n_trades<max_trades:
            trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
            trade_bars[n_trades]=(end_bar-1)-entry_bar
            trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
            trade_ddsums[n_trades]=dd_sum; n_trades+=1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pnls=trade_pnls[:n_trades]; total_pnl=0.0
    for j in range(n_trades): total_pnl+=pnls[j]
    mean_pnl=total_pnl/n_trades
    var=0.0
    for j in range(n_trades): var+=(pnls[j]-mean_pnl)**2
    std=(var/n_trades)**0.5 if n_trades>1 else 1.0
    sharpe=mean_pnl/std*(n_trades**0.5) if std>0 else 0.0
    wins=0
    for j in range(n_trades):
        if pnls[j]>0: wins+=1
    win_rate=100.0*wins/n_trades
    avg_mfe=0.0; avg_mae=0.0; avg_cap=0.0; avg_bars=0.0
    for j in range(n_trades):
        avg_mfe+=trade_mfes[j]; avg_mae+=trade_maes[j]; avg_cap+=trade_captures[j]; avg_bars+=trade_bars[j]
    avg_mfe/=n_trades; avg_mae/=n_trades; avg_cap/=n_trades; avg_bars/=n_trades
    cum=0.0; peak_eq=0.0; max_dd=0.0
    for j in range(n_trades):
        cum+=pnls[j]
        if cum>peak_eq: peak_eq=cum
        dd=peak_eq-cum
        if dd>max_dd: max_dd=dd
    avg_pnl_mae=0.0
    for j in range(n_trades):
        m=trade_maes[j]
        if m>0.1: avg_pnl_mae+=pnls[j]/m
        else: avg_pnl_mae+=pnls[j]/0.1 if pnls[j]>0 else 0.0
    avg_pnl_mae/=n_trades
    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, float(n_long), float(n_short))


def run_genome_pnf_2out(genome, config, data, pip, spread_pips,
                        max_bars=5000, max_hold=50, r_gamma=0.01, start_offset=0,
                        is_long=True):
    """P&F cadence 2-output genome evaluation (long-only or short-only)."""
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    result = evaluate_pnf_2out_jit(
        data["pnf_close"], data["pnf_actual_price"], data["_pnf_mc"],
        data["_h1_support"], data["_h1_resistance"], data["_pnf_to_h1"],
        pip, spread_pips, max_bars, max_hold, r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices, start_offset,
        is_long,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, n_long, n_short) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "n_long": int(n_long), "n_short": int(n_short),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2-OUTPUT P&F CADENCE + ADX EVALUATOR (4 inputs)
# Inputs: 4 (MC + H1_SR + UPnL + H1_ADX) — same as pnf_2out + ADX
# Outputs: 2 (ENTER vs CLOSE)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_pnf_2out_adx_jit(
    pnf_close, pnf_actual_price, pnf_mc,
    h1_support, h1_resistance, h1_adx, pnf_to_h1,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
    is_long,
):
    """P&F cadence with 4 inputs (MC, H1_SR, UPnL, ADX) and 2 outputs."""
    values = np.zeros(total_values)
    data_len = len(pnf_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
    n_long = 0; n_short = 0
    direction = 1 if is_long else -1

    for i in range(start_bar, end_bar):
        mc = pnf_mc[i]

        # H1 S/R — causal
        h1_idx = pnf_to_h1[i]
        sup = h1_support[h1_idx]
        res = h1_resistance[h1_idx]
        price_now = pnf_close[i] * pip
        sr_range = res - sup
        sr_pos = 2.0 * (price_now - sup) / sr_range - 1.0 if sr_range > 0 else 0.0

        actual_now = pnf_actual_price[i]
        if position != 0:
            pnl_pips = (actual_now - entry_price) * position - spread_pips
        else:
            pnl_pips = 0.0

        # 4 inputs: MC, H1_SR, UPnL, ADX
        values[0] = mc
        values[1] = np.tanh(sr_pos)
        values[2] = np.tanh(pnl_pips / 20.0)
        values[3] = np.tanh(h1_adx[h1_idx] / 50.0 - 0.5)  # center around ADX=25

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_enter = values[output_indices[0]]
        out_close = values[output_indices[1]]

        # Update MFE/MAE/AMDDP
        if position != 0:
            pnl_now = (actual_now - entry_price) * position - spread_pips
            if pnl_now > mfe: mfe = pnl_now
            if -pnl_now > mae: mae = -pnl_now
            if pnl_now > hwm_pnl: hwm_pnl = pnl_now
            cur_dd = max(max(0.0, -pnl_now), max(0.0, hwm_pnl - pnl_now))
            if cur_dd > prev_max_dd: dd_sum += cur_dd - prev_max_dd; prev_max_dd = cur_dd

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (actual_now - entry_price) * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                trade_bars[n_trades]=i-entry_bar
                trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                trade_ddsums[n_trades]=dd_sum; n_trades+=1
            position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

        # Decision logic
        if position == 0:
            if out_enter > out_close:
                position = direction
                entry_price = actual_now
                entry_bar = i
                if is_long:
                    n_long += 1
                else:
                    n_short += 1
        else:
            if out_close > out_enter:
                pnl = (actual_now - entry_price) * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

    # Close any open position
    if position != 0 and end_bar > 0:
        actual_now = pnf_actual_price[min(end_bar, data_len - 1)]
        pnl = (actual_now - entry_price) * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
            trade_bars[n_trades]=end_bar-entry_bar
            trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
            trade_ddsums[n_trades]=dd_sum; n_trades+=1

    if n_trades == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = trade_pnls[:n_trades]
    total_pnl = np.sum(pnls)
    mean_pnl = total_pnl / n_trades
    std_pnl = np.std(pnls) if n_trades > 1 else 1.0
    sharpe = mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0
    sharpe = sharpe * np.sqrt(n_trades)
    win_rate = np.sum(pnls > 0) / n_trades * 100.0
    avg_mfe = np.mean(trade_mfes[:n_trades])
    avg_mae = np.mean(trade_maes[:n_trades])
    avg_cap = np.mean(trade_captures[:n_trades])
    avg_bars = np.mean(trade_bars[:n_trades])
    avg_pnl_mae = mean_pnl / (avg_mae + 0.01)

    eq = np.cumsum(pnls)
    peak = eq[0]
    max_dd = 0.0
    for j in range(len(eq)):
        if eq[j] > peak: peak = eq[j]
        dd = peak - eq[j]
        if dd > max_dd: max_dd = dd

    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, float(n_long), float(n_short))


def run_genome_pnf_2out_adx(genome, config, data, pip, spread_pips,
                             max_bars=5000, max_hold=50, r_gamma=0.01, start_offset=0,
                             is_long=True):
    """P&F cadence 4-input (MC+SR+UPnL+ADX) 2-output genome evaluation."""
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    result = evaluate_pnf_2out_adx_jit(
        data["pnf_close"], data["pnf_actual_price"], data["_pnf_mc"],
        data["_h1_support"], data["_h1_resistance"], data["_h1_adx"],
        data["_pnf_to_h1"],
        pip, spread_pips, max_bars, max_hold, r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices, start_offset,
        is_long,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, n_long, n_short) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "n_long": int(n_long), "n_short": int(n_short),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7-INPUT, 2-OUTPUT P&F SCALPER EVALUATOR (SHORT-ONLY)
# Inputs: MC, H1_SR, VolRegime, ColLen, M1_CC1, M5_CC1, H1_CC1
# Outputs: 2 (SELL vs HOLD/CLOSE)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_pnf_scalper_jit(
    pnf_close, pnf_actual_price, pnf_mc, pnf_col_len,
    h1_support, h1_resistance, h1_atr, median_h1_atr,
    m1_cc1, m5_cc1, h1_cc1,
    pnf_to_h1, pnf_to_m1, pnf_to_m5,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """P&F scalper: 7 inputs, 2 outputs (short-only).
    When flat: output[0] > output[1] → SELL (enter short), else HOLD.
    When in position: output[1] > output[0] → CLOSE, else HOLD."""
    values = np.zeros(total_values)
    data_len = len(pnf_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    trade_ddsums = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0; prev_max_dd = 0.0
    n_long = 0; n_short = 0

    for i in range(start_bar, end_bar):
        mc = pnf_mc[i]

        # H1 S/R — causal: pnf_to_h1[i] points to last COMPLETED H1 bar
        h1_idx = pnf_to_h1[i]
        sup = h1_support[h1_idx]
        res = h1_resistance[h1_idx]
        price_now = pnf_close[i] * pip
        sr_range = res - sup
        sr_pos = 2.0 * (price_now - sup) / sr_range - 1.0 if sr_range > 0 else 0.0

        # Volatility regime
        cur_h1_atr = h1_atr[h1_idx]
        vol_regime = cur_h1_atr / median_h1_atr - 1.0 if median_h1_atr > 0.0 else 0.0

        # Column length
        col_len = pnf_col_len[i]

        # Close-to-close changes
        m1_idx = pnf_to_m1[i]
        m5_idx = pnf_to_m5[i]
        m1_cc = m1_cc1[m1_idx]
        m5_cc = m5_cc1[m5_idx]
        h1_cc = h1_cc1[h1_idx]

        # Set 7 inputs
        values[0] = mc
        values[1] = np.tanh(sr_pos)
        values[2] = np.tanh(vol_regime)
        values[3] = np.tanh(col_len / 10.0)
        values[4] = np.tanh(m1_cc / (2.0 * pip))
        values[5] = np.tanh(m5_cc / (5.0 * pip))
        values[6] = np.tanh(h1_cc / (20.0 * pip))

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        # 2-output decision: output[0] = SELL signal, output[1] = HOLD/CLOSE signal
        out0 = values[output_indices[0]]
        out1 = values[output_indices[1]]

        actual_now = pnf_actual_price[i]

        # Update MFE/MAE/AMDDP
        if position != 0:
            pnl_now = (actual_now - entry_price) * position - spread_pips
            if pnl_now > mfe: mfe = pnl_now
            if -pnl_now > mae: mae = -pnl_now
            if pnl_now > hwm_pnl: hwm_pnl = pnl_now
            cur_dd = max(max(0.0, -pnl_now), max(0.0, hwm_pnl - pnl_now))
            if cur_dd > prev_max_dd: dd_sum += cur_dd - prev_max_dd; prev_max_dd = cur_dd

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (actual_now - entry_price) * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                trade_bars[n_trades]=i-entry_bar
                trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                trade_ddsums[n_trades]=dd_sum; n_trades+=1
            position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

        if position == 0:
            # Flat: SELL if output[0] > output[1]
            if out0 > out1:
                position = -1; entry_price = actual_now; entry_bar = i; n_short += 1
        else:
            # In position: CLOSE if output[1] > output[0]
            if out1 > out0:
                pnl = (actual_now - entry_price) * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
                    trade_bars[n_trades]=i-entry_bar
                    trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
                    trade_ddsums[n_trades]=dd_sum; n_trades+=1
                position=0; mfe=0.0; mae=0.0; hwm_pnl=0.0; dd_sum=0.0; prev_max_dd=0.0

    # Close any open position at end
    if position != 0 and end_bar > start_bar:
        pnl=(pnf_actual_price[end_bar-1]-entry_price)*position-spread_pips
        if n_trades<max_trades:
            trade_pnls[n_trades]=pnl; trade_mfes[n_trades]=mfe; trade_maes[n_trades]=mae
            trade_bars[n_trades]=(end_bar-1)-entry_bar
            trade_captures[n_trades]=pnl/(mfe+0.1) if mfe>0.1 else 0.0
            trade_ddsums[n_trades]=dd_sum; n_trades+=1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pnls=trade_pnls[:n_trades]; total_pnl=0.0
    for j in range(n_trades): total_pnl+=pnls[j]
    mean_pnl=total_pnl/n_trades
    var=0.0
    for j in range(n_trades): var+=(pnls[j]-mean_pnl)**2
    std=(var/n_trades)**0.5 if n_trades>1 else 1.0
    sharpe=mean_pnl/std*(n_trades**0.5) if std>0 else 0.0
    wins=0
    for j in range(n_trades):
        if pnls[j]>0: wins+=1
    win_rate=100.0*wins/n_trades
    avg_mfe=0.0; avg_mae=0.0; avg_cap=0.0; avg_bars=0.0
    for j in range(n_trades):
        avg_mfe+=trade_mfes[j]; avg_mae+=trade_maes[j]; avg_cap+=trade_captures[j]; avg_bars+=trade_bars[j]
    avg_mfe/=n_trades; avg_mae/=n_trades; avg_cap/=n_trades; avg_bars/=n_trades
    cum=0.0; peak_eq=0.0; max_dd=0.0
    for j in range(n_trades):
        cum+=pnls[j]
        if cum>peak_eq: peak_eq=cum
        dd=peak_eq-cum
        if dd>max_dd: max_dd=dd
    avg_pnl_mae=0.0
    for j in range(n_trades):
        m=trade_maes[j]
        if m>0.1: avg_pnl_mae+=pnls[j]/m
        else: avg_pnl_mae+=pnls[j]/0.1 if pnls[j]>0 else 0.0
    avg_pnl_mae/=n_trades
    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
            avg_pnl_mae, float(n_long), float(n_short))


def run_genome_pnf_scalper(genome, config, data, pip, spread_pips,
                           max_bars=5000, max_hold=50, r_gamma=0.01, start_offset=0):
    """P&F scalper genome evaluation (7 inputs, 2 outputs, short-only)."""
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    result = evaluate_pnf_scalper_jit(
        data["pnf_close"], data["pnf_actual_price"], data["_pnf_mc"],
        data["_pnf_col_len"],
        data["_h1_support"], data["_h1_resistance"],
        data["_h1_atr"], data["_median_h1_atr"],
        data["_m1_cc1"], data["_m5_cc1"], data["_h1_cc1"],
        data["_pnf_to_h1"], data["_pnf_to_m1"], data["_pnf_to_m5"],
        pip, spread_pips, max_bars, max_hold, r_gamma,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight,
        output_indices, start_offset,
    )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars,
     avg_pnl_mae, n_long, n_short) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
        "avg_pnl_mae": round(float(avg_pnl_mae), 3),
        "n_long": int(n_long), "n_short": int(n_short),
    }


def precompute_pnf_scalper_indicators(data, spread_pips):
    """Precompute P&F scalper indicators (7-input). No look-ahead."""
    pip = data["pip"]
    min_swing = 3.0 * spread_pips * pip

    # Momentum consistency (reuse existing helper)
    data["_pnf_mc"] = _precompute_pnf_mc(data["pnf_close"], 5)

    # H1 S/R (causal — computed on H1 data only, no future info)
    sup, res = _precompute_h1_sr_levels(
        data["ind_h1_high"], data["ind_h1_low"], data["ind_h1_close"], min_swing)
    data["_h1_support"] = sup
    data["_h1_resistance"] = res

    # H1 ATR — compute from H1 high/low/close if not already present
    h1_high = data["ind_h1_high"]
    h1_low = data["ind_h1_low"]
    h1_close = data["ind_h1_close"]
    n_h1 = data["n_h1"]
    h1_atr = np.zeros(n_h1, dtype=np.float64)
    period = 14
    # True Range
    for i in range(n_h1):
        if i == 0:
            h1_atr[i] = h1_high[i] - h1_low[i]
        else:
            tr1 = h1_high[i] - h1_low[i]
            tr2 = abs(h1_high[i] - h1_close[i - 1])
            tr3 = abs(h1_low[i] - h1_close[i - 1])
            tr = max(tr1, max(tr2, tr3))
            if i < period:
                h1_atr[i] = h1_atr[i - 1] + (tr - h1_atr[i - 1]) / (i + 1)
            else:
                h1_atr[i] = h1_atr[i - 1] + (tr - h1_atr[i - 1]) / period
    data["_h1_atr"] = h1_atr
    valid_atr = h1_atr[h1_atr > 0]
    data["_median_h1_atr"] = float(np.nanmedian(valid_atr)) if len(valid_atr) > 0 else 1.0

    # P&F column length — boxes in current column up to this point
    pnf_column_id = data["pnf_column_id"]
    n_pnf = data["n_pnf"]
    pnf_col_len = np.zeros(n_pnf, dtype=np.float64)
    if n_pnf > 0:
        count = 1
        pnf_col_len[0] = 1.0
        for i in range(1, n_pnf):
            if pnf_column_id[i] == pnf_column_id[i - 1]:
                count += 1
            else:
                count = 1
            pnf_col_len[i] = float(count)
    data["_pnf_col_len"] = pnf_col_len

    # Close-to-close changes for M1 bars
    # If tick_tf was "1min", m5_close (which is really tick close) IS M1 close
    m1_close = data["m5_close"]  # tick-level close (M1 when tick_tf="1min")
    n_m1 = len(m1_close)
    m1_cc1 = np.zeros(n_m1, dtype=np.float64)
    for i in range(1, n_m1):
        m1_cc1[i] = m1_close[i] - m1_close[i - 1]
    data["_m1_cc1"] = m1_cc1

    # Close-to-close changes for M5 bars
    m5_close = data["ind_m5_close"]
    n_m5 = len(m5_close)
    m5_cc1 = np.zeros(n_m5, dtype=np.float64)
    for i in range(1, n_m5):
        m5_cc1[i] = m5_close[i] - m5_close[i - 1]
    data["_m5_cc1"] = m5_cc1

    # Close-to-close changes for H1 bars
    h1_cc1 = np.zeros(n_h1, dtype=np.float64)
    for i in range(1, n_h1):
        h1_cc1[i] = h1_close[i] - h1_close[i - 1]
    data["_h1_cc1"] = h1_cc1

    # Map P&F boxes → H1 bars causally via source timestamps
    pnf_source = data.get("pnf_source_idx", None)
    if pnf_source is not None:
        max_s5 = max(float(pnf_source[-1]), 1.0) if n_pnf > 0 else 1.0
        pnf_to_h1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            frac = pnf_source[i] / max_s5
            h1_idx = int(frac * (n_h1 - 1))
            pnf_to_h1[i] = max(0, h1_idx - 1)
        data["_pnf_to_h1"] = pnf_to_h1
    else:
        pnf_to_h1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            pnf_to_h1[i] = max(0, int(i / max(n_pnf, 1) * (n_h1 - 1)) - 1)
        data["_pnf_to_h1"] = pnf_to_h1

    # Map P&F boxes → M1 bars causally
    n_m1_bars = len(data["m5_close"])  # tick-level = M1 when tick_tf="1min"
    if pnf_source is not None:
        max_s5 = max(float(pnf_source[-1]), 1.0) if n_pnf > 0 else 1.0
        pnf_to_m1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            frac = pnf_source[i] / max_s5
            m1_idx = int(frac * (n_m1_bars - 1))
            pnf_to_m1[i] = max(0, m1_idx - 1)
        data["_pnf_to_m1"] = pnf_to_m1
    else:
        pnf_to_m1 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            pnf_to_m1[i] = max(0, int(i / max(n_pnf, 1) * (n_m1_bars - 1)) - 1)
        data["_pnf_to_m1"] = pnf_to_m1

    # Map P&F boxes → M5 bars causally
    n_m5_bars = data["n_ind_m5"]
    if pnf_source is not None:
        max_s5 = max(float(pnf_source[-1]), 1.0) if n_pnf > 0 else 1.0
        pnf_to_m5 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            frac = pnf_source[i] / max_s5
            m5_idx = int(frac * (n_m5_bars - 1))
            pnf_to_m5[i] = max(0, m5_idx - 1)
        data["_pnf_to_m5"] = pnf_to_m5
    else:
        pnf_to_m5 = np.zeros(n_pnf, dtype=np.int64)
        for i in range(n_pnf):
            pnf_to_m5[i] = max(0, int(i / max(n_pnf, 1) * (n_m5_bars - 1)) - 1)
        data["_pnf_to_m5"] = pnf_to_m5

    # Ensure actual prices exist
    if "pnf_actual_price" not in data:
        data["pnf_actual_price"] = data["pnf_close"].copy()

    return data


# ═══════════════════════════════════════════════════════════════════════════
# HEIKEN ASHI TIME-BASED EVALUATOR
# Operates on M5 or H1 bars (not P&F). Inputs: ha_dir (+1/-1), UPnL.
# 2-output mode: ENTER vs CLOSE (long-only or short-only)
# 3-output mode: BUY vs SELL vs CLOSE (both directions)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_ha_dir(o, h, l, c, n):
    """Compute Heiken Ashi direction array: +1 (bullish) or -1 (bearish)."""
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n, dtype=np.float64)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_dir = np.empty(n, dtype=np.float64)
    for i in range(n):
        ha_dir[i] = 1.0 if ha_c[i] >= ha_o[i] else -1.0
    return ha_dir


@njit(cache=True)
def evaluate_ha_2out_jit(
    ha_dir, mid_close,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
    is_long,
):
    """HA time-based cadence, 2 outputs: ENTER vs CLOSE (long-only or short-only).
    Inputs: [ha_dir, UPnL_tanh].
    """
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 5, 5)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0
    mae = 0.0
    direction = 1 if is_long else -1

    for i in range(start_bar, end_bar):
        # Compute UPnL
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        # Network inputs
        values[0] = ha_dir[i]
        values[1] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_enter = values[output_indices[0]]
        out_close = values[output_indices[1]]

        # Update MFE/MAE
        if position != 0:
            pnl_now = (mid_close[i] - entry_price) / pip * position - spread_pips
            if pnl_now > mfe:
                mfe = pnl_now
            if -pnl_now > mae:
                mae = -pnl_now

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_mfes[n_trades] = mfe
                trade_maes[n_trades] = mae
                trade_bars[n_trades] = i - entry_bar
                trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                n_trades += 1
            position = 0
            mfe = 0.0
            mae = 0.0

        # Decision
        if position == 0:
            if out_enter > out_close:
                position = direction
                entry_price = mid_close[i]
                entry_bar = i
        else:
            if out_close > out_enter:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = i - entry_bar
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    n_trades += 1
                position = 0
                mfe = 0.0
                mae = 0.0

    # Close any open position at end
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_mfes[n_trades] = mfe
            trade_maes[n_trades] = mae
            trade_bars[n_trades] = (end_bar - 1) - entry_bar
            trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pnls = trade_pnls[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades):
        total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades):
        var += (pnls[j] - mean_pnl) ** 2
    std = (var / n_trades) ** 0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades ** 0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0:
            wins += 1
    win_rate = 100.0 * wins / n_trades
    avg_mfe = avg_mae = avg_cap = avg_bars_val = 0.0
    for j in range(n_trades):
        avg_mfe += trade_mfes[j]
        avg_mae += trade_maes[j]
        avg_cap += trade_captures[j]
        avg_bars_val += trade_bars[j]
    avg_mfe /= n_trades
    avg_mae /= n_trades
    avg_cap /= n_trades
    avg_bars_val /= n_trades
    cum = peak_eq = max_dd = 0.0
    for j in range(n_trades):
        cum += pnls[j]
        if cum > peak_eq:
            peak_eq = cum
        dd = peak_eq - cum
        if dd > max_dd:
            max_dd = dd

    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars_val)


@njit(cache=True)
def evaluate_ha_3out_jit(
    ha_dir, mid_close,
    pip, spread_pips, max_bars, max_hold, r_gamma,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """HA time-based cadence, 3 outputs: BUY vs SELL vs CLOSE.
    Inputs: [ha_dir, UPnL_tanh].
    When flat: highest of BUY/SELL wins (must also beat CLOSE).
    When in position: CLOSE > max(BUY,SELL) -> close. Direction flip allowed.
    """
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 5, 5)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_mfes = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    trade_bars = np.zeros(max_trades)
    trade_captures = np.zeros(max_trades)
    n_trades = 0

    position = 0  # -1, 0, +1
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0
    mae = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        values[0] = ha_dir[i]
        values[1] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_close = values[output_indices[2]]

        # Update MFE/MAE
        if position != 0:
            pnl_now = (mid_close[i] - entry_price) / pip * position - spread_pips
            if pnl_now > mfe:
                mfe = pnl_now
            if -pnl_now > mae:
                mae = -pnl_now

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_mfes[n_trades] = mfe
                trade_maes[n_trades] = mae
                trade_bars[n_trades] = i - entry_bar
                trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                n_trades += 1
            position = 0
            mfe = 0.0
            mae = 0.0

        # Decision
        if position == 0:
            if out_buy > out_close and out_buy > out_sell:
                position = 1
                entry_price = mid_close[i]
                entry_bar = i
            elif out_sell > out_close and out_sell > out_buy:
                position = -1
                entry_price = mid_close[i]
                entry_bar = i
        else:
            if out_close > out_buy and out_close > out_sell:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = i - entry_bar
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    n_trades += 1
                position = 0
                mfe = 0.0
                mae = 0.0
            elif position == 1 and out_sell > out_buy and out_sell > out_close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = i - entry_bar
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    n_trades += 1
                position = -1
                entry_price = mid_close[i]
                entry_bar = i
                mfe = 0.0
                mae = 0.0
            elif position == -1 and out_buy > out_sell and out_buy > out_close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_mfes[n_trades] = mfe
                    trade_maes[n_trades] = mae
                    trade_bars[n_trades] = i - entry_bar
                    trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
                    n_trades += 1
                position = 1
                entry_price = mid_close[i]
                entry_bar = i
                mfe = 0.0
                mae = 0.0

    # Close any open position at end
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_mfes[n_trades] = mfe
            trade_maes[n_trades] = mae
            trade_bars[n_trades] = (end_bar - 1) - entry_bar
            trade_captures[n_trades] = pnl / (mfe + 0.1) if mfe > 0.1 else 0.0
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pnls = trade_pnls[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades):
        total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades):
        var += (pnls[j] - mean_pnl) ** 2
    std = (var / n_trades) ** 0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades ** 0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0:
            wins += 1
    win_rate = 100.0 * wins / n_trades
    avg_mfe = avg_mae = avg_cap = avg_bars_val = 0.0
    for j in range(n_trades):
        avg_mfe += trade_mfes[j]
        avg_mae += trade_maes[j]
        avg_cap += trade_captures[j]
        avg_bars_val += trade_bars[j]
    avg_mfe /= n_trades
    avg_mae /= n_trades
    avg_cap /= n_trades
    avg_bars_val /= n_trades
    cum = peak_eq = max_dd = 0.0
    for j in range(n_trades):
        cum += pnls[j]
        if cum > peak_eq:
            peak_eq = cum
        dd = peak_eq - cum
        if dd > max_dd:
            max_dd = dd

    return (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
            avg_mfe, avg_mae, avg_cap, max_dd, avg_bars_val)


def run_genome_ha(genome, config, ha_dir, mid_close, pip, spread_pips,
                  max_bars=50000, max_hold=200, r_gamma=0.01, start_offset=0,
                  is_long=True, n_outputs=2):
    """Run HA genome evaluation. Supports 2-output and 3-output modes."""
    (n_inputs, n_out, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight,
     output_indices) = extract_network(genome, config)

    if n_outputs == 2:
        result = evaluate_ha_2out_jit(
            ha_dir, mid_close, pip, spread_pips, max_bars, max_hold, r_gamma,
            n_inputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight,
            output_indices, start_offset, is_long,
        )
    else:
        result = evaluate_ha_3out_jit(
            ha_dir, mid_close, pip, spread_pips, max_bars, max_hold, r_gamma,
            n_inputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight,
            output_indices, start_offset,
        )

    (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
     avg_mfe, avg_mae, avg_cap, max_dd, avg_bars) = result

    return {
        "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
        "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
        "avg_mae": round(float(avg_mae), 1), "avg_capture": round(float(avg_cap), 3),
        "max_dd": round(float(max_dd), 1), "avg_bars": round(float(avg_bars), 1),
    }
