#!/usr/bin/env python3
"""
Multi-pair NEAT strategy container — runs all 12 pairs for one account.

Replaces 12 separate systemd services per account with 1 container.
Subscribes to P&F box events from the curator, runs NEAT network forward
pass per pair, executes trades via OANDA.

Configuration via environment variables:
  OANDA_ACCOUNT: OANDA account ID
  STRATEGY_NAME: e.g., "label_long", "neat2_short", "strength_long"
  NEAT_GENOME: Genome pkl filename (in /data/models/)
  NEAT_INPUT_MODE: "pnf", "mtf_mc", or "strength"
  NEAT_BOX_SIZE: P&F box size in pips (5 or 15)
  NEAT_REVERSAL: P&F reversal count (2 or 3)
  NEAT_DIRECTION: "long" or "short"
  NEAT_EMERGENCY_SL: Emergency SL in pips
  NEAT_EMERGENCY_TP: Emergency TP in pips
  ER_GATE_THRESHOLD: If >0, block entries when er_norm > threshold (e.g. 0.35 = ranging only)
"""

import os
import sys
import time
import signal
import pickle
import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from lib.pair_config import PAIRS, ALL_PAIR_NAMES, get_pair
from lib.broker_adapter import BrokerAdapter, OrderResult
from lib.oanda_adapter import OANDAAdapter
from lib.zmq_protocol import (
    Subscriber, MARKET_PUB, ALLOCATION_PUB,
    MSG_PNF_BOX, MSG_INDICATOR, MSG_KALMAN, MSG_CANDLE, MSG_RISK_OVERLAY,
    make_topic,
)
from lib.db import get_trades_db, init_trades_schema, TradeDBSender
from lib.sizing import StrategyAllocator, is_weekend_entry_blocked, is_weekend_close_time
from lib.notify import notify_open, notify_close, _send as notify_send

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(os.environ.get("STRATEGY_NAME", "neat"))

# MFE trailing threshold
MFE_TRAIL_STEP_PIPS = 5  # Trail SL/TP every N pips of MFE improvement
SL_TP_UPDATE_MAX_RETRIES = 3
EQUITY_REFRESH_INTERVAL = 300  # seconds
RECONCILE_INTERVAL = 60  # Check broker for SL/TP fills every 60s

# S5 fast exit — software-side position protection (5-second cadence)
# Per-pair soft stop: max(15, round(avg_mae[pair] * 3.0)) from OOS backtest
PAIR_SOFT_STOP_PIPS: dict = {
    'CHF_JPY': 27, 'EUR_JPY': 22, 'NZD_JPY': 20, 'AUD_JPY': 18,
    'CAD_JPY': 17, 'GBP_JPY': 16, 'USD_JPY': 15,
    'GBP_USD': 17, 'EUR_USD': 15, 'AUD_USD': 15, 'NZD_USD': 15, 'EUR_GBP': 15,
}
DEFAULT_SOFT_STOP_PIPS = 15

# Per-pair broker trailing stop config (from OOS MAE analysis, +10% margin)
# Trail distance = 1.5 × avg_MAE × 1.1, Min target = 2 × trail distance × 1.1
PAIR_TRAIL_CONFIG: dict = {
    'EUR_JPY': {'trail': 14, 'target': 29},
    'USD_JPY': {'trail': 11, 'target': 21},
    'GBP_JPY': {'trail': 13, 'target': 25},
    'AUD_JPY': {'trail':  8, 'target': 17},
    'CAD_JPY': {'trail': 11, 'target': 22},
    'CHF_JPY': {'trail': 13, 'target': 27},
    'NZD_JPY': {'trail':  9, 'target': 19},
    'EUR_USD': {'trail':  8, 'target': 15},
    'GBP_USD': {'trail':  9, 'target': 19},
    'AUD_USD': {'trail':  6, 'target': 11},
    'NZD_USD': {'trail':  7, 'target': 13},
    'EUR_GBP': {'trail':  5, 'target': 10},
}
DEFAULT_TRAIL_DISTANCE = 8
DEFAULT_TRAIL_TARGET = 15

# Legacy S5 trail (kept for backward compat, overridden by PAIR_TRAIL_CONFIG)
S5_TRAIL_ACTIVATE_PIPS = 15
S5_TRAIL_DISTANCE_PIPS = 8


# ─── NEAT Network ──────────────────────────────────────────────────────────

def _gauss_activation(x):
    import math
    return math.exp(-x * x)

def _sin_activation(x):
    import math
    return math.sin(x)

def _cos_activation(x):
    import math
    return math.cos(x)

def _tanh_activation(x):
    import math
    return math.tanh(x)

def _sech_activation(x):
    import math
    return 1.0 / math.cosh(max(min(x, 50), -50))

def _dog_activation(x):
    import math
    return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)

def _gabor_activation(x):
    import math
    return math.exp(-2*x*x) * math.cos(2*math.pi*x)

def _sinc_activation(x):
    import math
    return math.sin(math.pi*x)/(math.pi*x) if abs(x) > 1e-7 else 1.0

def _morlet_activation(x):
    import math
    return math.sin(x) * math.exp(-x*x/2)

# Aliases without underscore prefix — required for pickle compatibility with
# genomes trained in scripts that define these as module-level (no underscore).
gauss_activation = _gauss_activation
sin_activation   = _sin_activation
cos_activation   = _cos_activation
tanh_activation  = _tanh_activation
sech_activation  = _sech_activation
dog_activation   = _dog_activation
gabor_activation = _gabor_activation
sinc_activation  = _sinc_activation
morlet_activation = _morlet_activation


def load_genome(genome_path: str, config_path: Optional[str] = None):
    """Load NEAT genome and create feed-forward network."""
    import neat

    with open(genome_path, "rb") as f:
        saved = pickle.load(f)

    if "config" in saved and saved["config"] is not None:
        config = saved["config"]
    else:
        if config_path is None:
            # Auto-detect 3 vs 4 input config based on genome connections
            models_dir = os.environ.get("NEAT_MODELS_DIR",
                        os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            if not os.path.isdir(models_dir):
                models_dir = "/data/models"
            has_4_inputs = any(k[0] == -4 for k in saved["genome"].connections)
            if has_4_inputs:
                config_path = os.path.join(models_dir, "neat_config_4in.ini")
            else:
                config_path = os.path.join(models_dir, "neat_config.ini")
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path,
        )

    # Register all wavelet activations
    for name, fn in [('gauss', _gauss_activation), ('sin', _sin_activation),
                     ('cos', _cos_activation), ('tanh', _tanh_activation),
                     ('sech', _sech_activation), ('dog', _dog_activation),
                     ('gabor', _gabor_activation), ('sinc', _sinc_activation),
                     ('morlet', _morlet_activation)]:
        try:
            config.genome_config.add_activation(name, fn)
        except Exception:
            pass

    net = neat.nn.FeedForwardNetwork.create(saved["genome"], config)
    gen = saved.get("generation", "?")
    n_nodes = len(saved["genome"].nodes)
    n_conn = sum(1 for c in saved["genome"].connections.values() if c.enabled)
    logger.info(f"Loaded genome: {os.path.basename(genome_path)} "
                f"(gen {gen}, {n_nodes} nodes, {n_conn} connections)")
    return net


# ─── Per-Pair State ────────────────────────────────────────────────────────

@dataclass
class PairState:
    """Trading state for one pair within this container."""
    pair: str
    open_trade_id: Optional[str] = None
    open_direction: int = 0         # +1 long, -1 short, 0 flat
    open_entry_price: float = 0.0
    open_units: int = 0
    open_time: Optional[str] = None
    running_mfe: float = 0.0
    running_mae: float = 0.0
    last_exit_box_id: Optional[int] = None  # Same-box re-entry block
    last_broker_sl: Optional[float] = None
    broker_trailing_active: bool = False  # True once broker-side trailing stop is live
    completed_trades: list = field(default_factory=list)

    # Shadow trade for perf gate recovery
    shadow_trade: Optional[dict] = None

    # Latest indicator values (from curator)
    last_mc: float = 0.0
    last_h1_sr: float = 0.0
    last_mtf_mc_d: float = 0.0
    last_mtf_mc_dd: float = 0.0
    last_bid: float = 0.0
    last_ask: float = 0.0
    last_mid: float = 0.0

    # ASI-MC state (computed locally from S5 candles)
    asi_mc_d: float = 0.0
    asi_mc_dd: float = 0.0
    er_norm: float = 0.0  # V3: Kaufman ER(60 M5 bars) arctan-normalized
    # V7 indicators
    v7_bb_width: float = 0.0
    v7_stoch_d: float = 0.0
    v7_macd_hist: float = 0.0
    v7_range_pos: float = 0.0
    v7_aroon_osc: float = 0.0
    # Feature set indicators (S1, sets A-F)
    v7_sb_a: float = 0.0
    v7_tec_5: float = 0.0
    v7_h1_slope: float = 0.0
    v7_gap_norm: float = 0.0
    v7_hl_price: float = 0.0
    s5_buffer_o: list = field(default_factory=list)
    s5_buffer_h: list = field(default_factory=list)
    s5_buffer_l: list = field(default_factory=list)
    s5_buffer_c: list = field(default_factory=list)
    s5_tick_count: int = 0  # Throttle: run network every 12 ticks (= M1 cadence)

    # Range bar gate: only trigger network when price moves N pips from anchor
    range_bar_anchor: float = 0.0  # Price at last decision point

    def get_regime_rvol(self, window: int = 60) -> Optional[float]:
        """Compute annualized realized volatility from S5 close buffer.
        Returns None if insufficient data."""
        closes = self.s5_buffer_c
        if len(closes) < window + 1:
            return None
        import numpy as np
        arr = np.array(closes[-window - 1:], dtype=np.float64)
        returns = np.diff(np.log(arr))
        # Annualize from S5 bars: 12 per minute * 60 * 24 * 252
        return float(np.std(returns) * np.sqrt(12 * 60 * 24 * 252))


# ─── Multi-Pair NEAT Runner ───────────────────────────────────────────────

class NeatMultiPairRunner:
    """Runs NEAT genome across all 12 pairs for one account."""

    def __init__(self):
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Config from environment (fail fast on missing)
        for var in ("OANDA_ACCOUNT", "STRATEGY_NAME", "NEAT_GENOME"):
            if var not in os.environ:
                raise ValueError(f"{var} env var required")
        self.account_id = os.environ["OANDA_ACCOUNT"]
        self.strategy_name = os.environ["STRATEGY_NAME"]
        self.input_mode = os.environ.get("NEAT_INPUT_MODE", "pnf")
        self.box_size = int(os.environ.get("NEAT_BOX_SIZE", "5"))
        self.reversal = int(os.environ.get("NEAT_REVERSAL", "3"))
        self.direction = 1 if os.environ.get("NEAT_DIRECTION", "long") == "long" else -1
        self.emergency_sl = int(os.environ.get("NEAT_EMERGENCY_SL", "50"))
        self.emergency_tp = int(os.environ.get("NEAT_EMERGENCY_TP", "32"))
        self.max_hold_hours = float(os.environ.get("NEAT_MAX_HOLD_HOURS", "0"))  # 0 = disabled
        self.allowed_pairs = os.environ.get("NEAT_PAIRS", "").split(",") if os.environ.get("NEAT_PAIRS") else []
        self.range_bar_pips = float(os.environ.get("NEAT_RANGE_BAR_PIPS", "0"))  # 0 = disabled, 10 = 10-pip gate
        self.er_gate = float(os.environ.get("ER_GATE_THRESHOLD", "0"))  # 0 = disabled; e.g. 0.35 blocks entries when er_norm > 0.35 (trending)

        self.pnf_config = f"{self.box_size}pip_rev{self.reversal}"
        self.consensus_confidence = float(os.environ.get("NEAT_CONSENSUS_CONFIDENCE", "0.3"))

        # ── Consensus mode: multiple input modes with separate genome maps ──
        # Format: "mode1:pair1=g1.pkl,pair2=g2.pkl|mode2:pair1=g3.pkl,..."
        consensus_str = os.environ.get("NEAT_CONSENSUS", "")
        self.consensus_voices = []
        if consensus_str:
            for voice_str in consensus_str.split("|"):
                voice_str = voice_str.strip()
                if ":" not in voice_str:
                    continue
                voice_mode, voice_map_str = voice_str.split(":", 1)
                voice_genomes = {}
                for entry in voice_map_str.split(","):
                    entry = entry.strip()
                    if ":" not in entry:
                        continue
                    p, gn = entry.split(":", 1)
                    genome_path = os.path.join("/data/models", gn.strip())
                    voice_genomes[p.strip()] = load_genome(genome_path)
                    logger.info(f"  Consensus voice {voice_mode}: {p.strip()} → {gn.strip()}")
                self.consensus_voices.append({
                    "mode": voice_mode.strip(),
                    "genomes": voice_genomes,
                })
            if self.consensus_voices:
                logger.info(f"Consensus mode: {len(self.consensus_voices)} voices, "
                            f"confidence gate={self.consensus_confidence}")

        # Load genome(s)
        # Per-pair mode: NEAT_GENOME_MAP="EUR_JPY:eur_jpy.pkl,USD_JPY:usd_jpy.pkl,..."
        # Multi-genome: NEAT_GENOME="a.pkl,b.pkl" (comma-separated, all pairs)
        # Single genome: NEAT_GENOME="model.pkl"
        self.pair_genome_map: Dict[str, tuple] = {}  # pair -> net tuple
        genome_map_str = os.environ.get("NEAT_GENOME_MAP", "")
        if genome_map_str:
            for entry in genome_map_str.split(","):
                entry = entry.strip()
                if ":" not in entry:
                    continue
                pair, gn = entry.split(":", 1)
                pair = pair.strip()
                gn = gn.strip()
                genome_path = os.path.join("/data/models", gn)
                net = load_genome(genome_path)
                self.pair_genome_map[pair] = net
                logger.info(f"  Per-pair genome: {pair} → {gn}")
            logger.info(f"Per-pair genome mode: {len(self.pair_genome_map)} pair-specific genomes")

        genome_names = os.environ["NEAT_GENOME"].split(",")
        self.genomes = []
        for gn in genome_names:
            gn = gn.strip()
            genome_path = os.path.join("/data/models", gn)
            net = load_genome(genome_path)
            self.genomes.append({"name": gn, "net": net})
        self.net = self.genomes[0]["net"]  # Primary/fallback genome

        # Broker
        self.broker = OANDAAdapter(account_id=self.account_id)

        # Sizing allocator
        try:
            self.allocator = StrategyAllocator(self.strategy_name)
            logger.info(f"Sizing allocator: '{self.strategy_name}' config loaded")
        except KeyError:
            try:
                self.allocator = StrategyAllocator("neat_pf")
                logger.info(f"Sizing allocator: '{self.strategy_name}' not found, using 'neat_pf' fallback")
            except KeyError:
                self.allocator = None
                logger.warning("No sizing allocator — using 1 unit")

        # Per-pair state — multi-genome: keyed by (genome_idx, pair)
        self.multi_genome = len(self.genomes) > 1
        active_pairs = [p.strip() for p in self.allowed_pairs] if self.allowed_pairs else ALL_PAIR_NAMES
        self.active_pair_set = set(active_pairs)
        self.pairs: Dict[str, PairState] = {
            pair: PairState(pair=pair) for pair in active_pairs
        }
        # For multi-genome mode: separate position state per genome per pair
        if self.multi_genome:
            self.genome_pairs: Dict[int, Dict[str, PairState]] = {}
            for gi, g in enumerate(self.genomes):
                self.genome_pairs[gi] = {
                    pair: PairState(pair=pair) for pair in active_pairs
                }
            logger.info(f"Multi-genome mode: {len(self.genomes)} genomes loaded")

        # Trade records sent via ZMQ to portfolio_mgr (single DB writer)
        self.trade_db = TradeDBSender()

        self._last_equity_refresh = 0.0
        self._last_reconcile = 0.0
        self._risk_overlay = {}  # Latest risk overlay from portfolio_mgr

        logger.info(f"NEAT Multi-Pair Runner: {self.strategy_name}")
        logger.info(f"  Account: {self.account_id}")
        logger.info(f"  Config: {self.pnf_config}, direction={'long' if self.direction > 0 else 'short'}")
        logger.info(f"  Input mode: {self.input_mode}")
        logger.info(f"  Emergency SL/TP: +/-{self.emergency_sl}/{self.emergency_tp} pips")
        logger.info(f"  Max hold: {self.max_hold_hours}h" if self.max_hold_hours > 0 else "  Max hold: disabled")
        logger.info(f"  Pairs: {list(self.active_pair_set)}" if self.allowed_pairs else "  Pairs: all 12")

    def _get_net_for_pair(self, pair: str):
        """Return the correct genome network for this pair.
        Per-pair map takes priority, then falls back to primary genome."""
        if self.pair_genome_map and pair in self.pair_genome_map:
            return self.pair_genome_map[pair]
        return self.net

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum}, shutting down...")
        self._shutdown = True

    def _compute_inputs(self, pair: str, state: PairState, mode_override: str = None) -> list:
        """Compute NEAT network inputs based on input mode."""
        upnl = 0.0
        if state.open_trade_id and state.last_mid > 0:
            pip = get_pair(pair).pip
            price_diff = (state.last_mid - state.open_entry_price) * state.open_direction
            upnl = math.tanh(price_diff / (20.0 * pip))

        mode = mode_override or self.input_mode

        if mode == "pnf":
            return [state.last_mc, state.last_h1_sr, upnl]
        elif mode == "mtf_mc":
            return [state.last_mtf_mc_d, state.last_mtf_mc_dd, upnl]
        elif mode == "strength":
            return [0.0, 0.0, state.last_mtf_mc_d, state.last_mtf_mc_dd, upnl]
        elif mode == "asi_mc":
            return [state.asi_mc_d, state.asi_mc_dd, upnl]
        elif mode in ("asi_mc_v3", "asi_mc_v3_h1"):
            return [state.asi_mc_d, state.asi_mc_dd, state.er_norm, upnl]
        elif mode == "asi_mc_v7_h1":
            return [state.v7_bb_width * 20.0,
                    state.v7_stoch_d,
                    max(-1.0, min(1.0, state.v7_macd_hist / 2.0)),
                    state.v7_range_pos,
                    state.v7_aroon_osc,
                    state.asi_mc_d,
                    upnl]
        elif mode == "s1_h1":
            return [state.asi_mc_d, state.asi_mc_dd, state.er_norm, state.v7_sb_a, upnl]
        # Feature set modes (setA-setF): read from V7 + unified fields
        elif mode == "setA_h1":
            return [state.v7_tec_5, state.v7_bb_width * 20.0, state.v7_h1_slope, upnl]
        elif mode == "setB_h1":
            return [state.v7_tec_5, state.v7_stoch_d, state.v7_range_pos, upnl]
        elif mode == "setC_h1":
            return [state.v7_tec_5, state.v7_bb_width * 20.0,
                    max(-1.0, min(1.0, state.v7_gap_norm / 3.0)), upnl]
        elif mode == "setD_h1":
            return [state.v7_tec_5, state.v7_bb_width * 20.0, state.v7_h1_slope, state.v7_stoch_d, upnl]
        elif mode == "setE_h1":
            return [state.v7_tec_5, state.v7_bb_width * 20.0, state.v7_range_pos, state.v7_stoch_d, upnl]
        elif mode == "setF_h1":
            return [state.v7_tec_5, state.v7_h1_slope, state.v7_hl_price, state.v7_stoch_d, upnl]
        return [0.0, 0.0, upnl]

    def _on_pnf_box(self, pair: str, boxes: list):
        """Handle new P&F boxes — run NEAT network and decide action."""
        if self.multi_genome:
            # Multi-genome: each genome processes boxes independently
            for gi, g in enumerate(self.genomes):
                state = self.genome_pairs[gi][pair]
                # Copy indicator values from shared state
                shared = self.pairs[pair]
                state.last_mc = shared.last_mc
                state.last_h1_sr = shared.last_h1_sr
                state.last_mtf_mc_d = shared.last_mtf_mc_d
                state.last_mtf_mc_dd = shared.last_mtf_mc_dd
                state.last_bid = shared.last_bid
                state.last_ask = shared.last_ask
                state.last_mid = shared.last_mid
                self._process_boxes_for_genome(pair, state, boxes, g["net"], g["name"])
        else:
            state = self.pairs[pair]
            net = self._get_net_for_pair(pair)
            self._process_boxes_for_genome(pair, state, boxes, net, self.genomes[0]["name"])

    def _process_boxes_for_genome(self, pair: str, state: PairState,
                                   boxes: list, net, genome_name: str):
        """Run NEAT forward pass for one genome on P&F boxes."""
        for box in boxes:
            inputs = self._compute_inputs(pair, state)
            if any(math.isnan(v) or math.isinf(v) for v in inputs):
                logger.warning(f"NaN/inf in inputs for {pair}, skipping: {inputs}")
                continue
            output = net.activate(inputs)

            # 2-output mode: output[0]=ENTER, output[1]=CLOSE
            if output[0] > output[1]:
                action = "ENTER"
            else:
                action = "CLOSE"

            has_position = state.open_trade_id is not None

            # Weekend guard
            if action == "ENTER" and is_weekend_entry_blocked():
                continue

            # Same-box re-entry block
            if action == "ENTER" and not has_position:
                box_id = box.get("column_id")
                if state.last_exit_box_id is not None and box_id == state.last_exit_box_id:
                    logger.debug(f"SAME-BOX BLOCK {pair}: box {box_id}")
                    continue
                self._open_trade(pair, state, box)
            elif action == "CLOSE" and has_position:
                state.last_exit_box_id = box.get("column_id")
                self._close_trade(pair, state, "NETWORK_CLOSE")

    def _on_asi_mc_tick(self, pair: str, state: PairState):
        """Handle ASI-MC tick — 3-output: BUY/SELL/FLATTEN.
        Supports consensus mode (multiple voices) or single genome."""

        if self.consensus_voices:
            # Consensus mode: sum outputs from all voices
            sum_buy, sum_sell, sum_flat = 0.0, 0.0, 0.0
            n_voices = 0
            for voice in self.consensus_voices:
                if pair not in voice["genomes"]:
                    continue
                voice_inputs = self._compute_inputs(pair, state, mode_override=voice["mode"])
                if any(math.isnan(v) or math.isinf(v) for v in voice_inputs):
                    continue
                voice_net = voice["genomes"][pair]
                voice_out = voice_net.activate(voice_inputs)
                sum_buy += voice_out[0]
                sum_sell += voice_out[1]
                sum_flat += voice_out[2]
                n_voices += 1
            if n_voices == 0:
                return
            out_buy, out_sell, out_flat = sum_buy, sum_sell, sum_flat
            # Confidence gate: require margin between top 2
            vals = sorted([out_buy, out_sell, out_flat], reverse=True)
            if vals[0] - vals[1] < self.consensus_confidence * n_voices:
                out_flat = max(out_buy, out_sell) + 0.01  # force FLATTEN on low confidence
        else:
            # Single genome mode
            inputs = self._compute_inputs(pair, state)
            if any(math.isnan(v) or math.isinf(v) for v in inputs):
                return
            net = self._get_net_for_pair(pair)
            output = net.activate(inputs)
            out_buy, out_sell, out_flat = output[0], output[1], output[2]
        has_position = state.open_trade_id is not None

        if is_weekend_entry_blocked():
            if has_position and is_weekend_close_time():
                self._close_trade(pair, state, "WEEKEND_CLOSE")
            return

        er_ok = self._er_gate_allows_entry(pair, state)

        if not has_position:
            # Flat → enter if BUY or SELL dominates
            if er_ok and out_buy > out_sell and out_buy > out_flat:
                self._open_trade_direction(pair, state, 1)
            elif er_ok and out_sell > out_buy and out_sell > out_flat:
                self._open_trade_direction(pair, state, -1)
        else:
            # In position — match training: CLOSE or FLIP
            if out_flat > out_buy and out_flat > out_sell:
                self._close_trade(pair, state, "NETWORK_CLOSE")
            elif state.open_direction == 1 and out_sell > out_buy and out_sell > out_flat:
                self._close_trade(pair, state, "NETWORK_FLIP")
                if er_ok:
                    self._open_trade_direction(pair, state, -1)
            elif state.open_direction == -1 and out_buy > out_sell and out_buy > out_flat:
                self._close_trade(pair, state, "NETWORK_FLIP")
                if er_ok:
                    self._open_trade_direction(pair, state, 1)

        # Update MFE/MAE + trail
        self._update_mfe_mae(pair, state)

    def _er_gate_allows_entry(self, pair: str, state: PairState) -> bool:
        """Returns False when ER_GATE_THRESHOLD is set and er_norm exceeds it (trending market)."""
        if self.er_gate <= 0.0:
            return True
        if state.er_norm > self.er_gate:
            logger.debug(f"ER_GATE {pair}: er_norm={state.er_norm:.3f} > {self.er_gate:.2f} — entry blocked")
            return False
        return True

    def _check_risk_overlay(self, pair: str, direction: int) -> Optional[str]:
        """Check global risk overlay. Returns block reason or None if clear."""
        ro = self._risk_overlay
        if not ro:
            return None  # No overlay received yet — allow trading

        # Total position cap
        if ro.get("total_positions", 0) >= ro.get("max_positions", 60):
            return f"GLOBAL_POS_CAP_{ro['total_positions']}/{ro['max_positions']}"

        # Global DD throttle — if scale is 0, block entirely
        if ro.get("global_dd_scale", 1.0) <= 0.0:
            return f"GLOBAL_DD_{ro.get('daily_dd_pct', 0):.1f}%"

        # Pair+direction concentration block
        dir_str = "long" if direction > 0 else "short"
        key = f"{pair}_{dir_str}"
        blocked_pairs = ro.get("blocked_pairs", {})
        if key in blocked_pairs:
            return f"RISK_{blocked_pairs[key]}"

        return None

    def _get_risk_dd_scale(self) -> float:
        """Get global DD scale multiplier from risk overlay (1.0 = normal, 0.0 = blocked)."""
        return self._risk_overlay.get("global_dd_scale", 1.0) if self._risk_overlay else 1.0

    def _open_trade_direction(self, pair: str, state: PairState, direction: int):
        """Open a trade in given direction (+1 long, -1 short). Used by ASI-MC mode."""
        # Global risk overlay check
        block_reason = self._check_risk_overlay(pair, direction)
        if block_reason:
            logger.info(f"RISK_BLOCK {pair} {'LONG' if direction > 0 else 'SHORT'}: {block_reason}")
            return

        pip = get_pair(pair).pip
        entry_est = state.last_mid
        if entry_est <= 0:
            return

        # Sizing — FX_VALIDATION_UNITS overrides allocator
        validation_units = os.environ.get("FX_VALIDATION_UNITS")
        if validation_units:
            units = int(validation_units)
        elif self.allocator:
            try:
                pair_trades = [{"pnl_pips": t.get("pnl_pips", 0)}
                               for t in state.completed_trades]
                units, debug = self.allocator.compute_units(
                    pair=pair, entry_price=entry_est,
                    risk_pips=self.emergency_sl,
                    direction=direction,
                    completed_trades=pair_trades,
                    open_positions={},
                    regime_rvol=state.get_regime_rvol(),
                )
                if units <= 0:
                    return
            except Exception:
                units = 1
        else:
            units = 1

        signed_units = units * direction
        sl_pips = self.emergency_sl * pip * (-direction)
        tp_pips = self.emergency_tp * pip * direction
        sl_price = entry_est + sl_pips
        tp_price = entry_est + tp_pips

        result = self.broker.place_market_order(pair, signed_units,
                                                 sl_price=sl_price, tp_price=tp_price)

        if result.success and result.trade_id:
            state.open_trade_id = result.trade_id
            state.open_direction = direction
            state.open_entry_price = result.fill_price if result.fill_price and result.fill_price > 0 else entry_est
            state.open_units = abs(signed_units)
            state.open_time = str(datetime.now(timezone.utc))
            state.running_mfe = 0.0
            state.running_mae = 0.0
            state.last_broker_sl = sl_price

            d = "LONG" if direction > 0 else "SHORT"
            logger.info(f"OPEN {pair} {d} {abs(signed_units)}u @ {state.open_entry_price:.5f}")
            notify_open(self.strategy_name, pair, d, abs(signed_units), state.open_entry_price, account=self.account_id)

    def _update_mfe_mae(self, pair: str, state: PairState):
        """Update MFE/MAE and trail broker SL/TP when MFE improves."""
        if not state.open_trade_id or state.last_mid <= 0:
            return

        pip = get_pair(pair).pip
        spread = (state.last_ask - state.last_bid) / pip if state.last_bid > 0 else 0
        pnl_pips = (state.last_mid - state.open_entry_price) * state.open_direction / pip - spread

        prev_mfe = state.running_mfe
        if pnl_pips > state.running_mfe:
            state.running_mfe = pnl_pips
        if -pnl_pips > state.running_mae:
            state.running_mae = -pnl_pips

        # Per-pair broker trailing stop: once MFE hits min target, set broker trailing stop
        # and let it ride. One API call — broker owns it from there.
        cfg = PAIR_TRAIL_CONFIG.get(pair, {'trail': DEFAULT_TRAIL_DISTANCE, 'target': DEFAULT_TRAIL_TARGET})
        if (state.running_mfe >= cfg['target']
                and not state.broker_trailing_active):
            success = self._activate_broker_trailing_stop(
                state.open_trade_id, pair, cfg['trail'])
            if success:
                state.broker_trailing_active = True
                logger.info(f"BROKER_TRAIL {pair}: trail={cfg['trail']}p activated at MFE={state.running_mfe:.1f}p "
                            f"(target={cfg['target']}p) — broker owns it")

        # Also update shadow trade MFE/MAE
        if state.shadow_trade is not None:
            st = state.shadow_trade
            s_pnl = (state.last_mid - st["entry_price"]) * st["direction"] / pip - spread
            if s_pnl > st.get("running_mfe", 0.0):
                st["running_mfe"] = s_pnl
            if -s_pnl > st.get("running_mae", 0.0):
                st["running_mae"] = -s_pnl

    def _s5_fast_exit_check(self, pair: str, state: PairState, bid: float, ask: float):
        """Fast exit check on S5 candle — software-side stop + trailing TP.
        Runs every 5 seconds, much faster than M5 network evaluation."""
        if not state.open_trade_id or bid <= 0 or ask <= 0:
            return

        pip = get_pair(pair).pip
        mid = (bid + ask) / 2
        spread_pips = (ask - bid) / pip
        pnl_pips = (mid - state.open_entry_price) * state.open_direction / pip - spread_pips

        # Update MFE/MAE on S5 tick
        if pnl_pips > state.running_mfe:
            state.running_mfe = pnl_pips
        if -pnl_pips > state.running_mae:
            state.running_mae = -pnl_pips

        # Soft stop — per-pair threshold, close before broker SL fires (less slippage)
        soft_stop = PAIR_SOFT_STOP_PIPS.get(pair, DEFAULT_SOFT_STOP_PIPS)
        if -pnl_pips >= soft_stop:
            logger.info(f"S5_SOFT_STOP {pair}: pnl={pnl_pips:+.1f}p >= -{soft_stop}p")
            self._close_trade(pair, state, "S5_SOFT_STOP")
            return

        # Trailing TP handled by broker-side trailing stop (set in _update_mfe_mae)
        # No software trailing needed — broker owns it after activation

    def _update_sl_tp_verified(self, trade_id: str, pair: str,
                                sl_price: float, tp_price: float) -> bool:
        """Update broker SL/TP with 3x retry and read-back verification."""
        for attempt in range(SL_TP_UPDATE_MAX_RETRIES):
            try:
                self.broker.modify_trade_sl_tp(trade_id,
                                                sl_price=sl_price, tp_price=tp_price)
                # Read-back verification
                trade_details = self.broker.get_trade_details(trade_id)
                if trade_details:
                    actual_sl = getattr(trade_details, 'sl_price', None)
                    if actual_sl is not None and abs(actual_sl - sl_price) > 1e-5:
                        logger.warning(f"SL mismatch {pair}: sent {sl_price:.5f}, "
                                      f"got {actual_sl:.5f} (attempt {attempt+1})")
                        if attempt < SL_TP_UPDATE_MAX_RETRIES - 1:
                            time.sleep(0.5 * (2 ** attempt))
                            continue
                        # Persistent failure — alert via Telegram
                        try:
                            notify_send(f"⚠️ SL/TP VERIFY FAILED {pair} trade {trade_id}")
                        except Exception:
                            pass
                        return False
                return True
            except Exception as e:
                logger.warning(f"SL/TP update error {pair} (attempt {attempt+1}): {e}")
                if attempt < SL_TP_UPDATE_MAX_RETRIES - 1:
                    time.sleep(0.5 * (2 ** attempt))
        return False

    def _activate_broker_trailing_stop(self, trade_id: str, pair: str,
                                        distance_pips: float) -> bool:
        """Switch from static SL to broker-side trailing stop.

        Called once when MFE first crosses S5_TRAIL_ACTIVATE_PIPS.  After this
        the broker trails the SL server-side — no further modify calls needed.
        Uses 3× retry + read-back to confirm the broker registered it.
        """
        for attempt in range(SL_TP_UPDATE_MAX_RETRIES):
            try:
                success = self.broker.modify_trade_trailing_stop(
                    trade_id, pair, distance_pips)
                if not success:
                    raise RuntimeError("modify returned False")

                # Read-back: confirm trailingStopLoss is registered
                trade_details = self.broker.get_trade_details(trade_id)
                if trade_details:
                    actual_dist = getattr(trade_details, 'trailing_stop_distance', None)
                    pip = get_pair(pair).pip
                    expected = distance_pips * pip
                    if actual_dist is not None and abs(actual_dist - expected) > pip * 0.5:
                        logger.warning(
                            f"TRAIL_STOP mismatch {pair}: sent {expected:.5f}, "
                            f"got {actual_dist:.5f} (attempt {attempt+1})")
                        if attempt < SL_TP_UPDATE_MAX_RETRIES - 1:
                            time.sleep(0.5 * (2 ** attempt))
                            continue
                        try:
                            notify_send(
                                f"⚠️ BROKER TRAIL VERIFY FAILED {pair} trade {trade_id}")
                        except Exception:
                            pass
                        return False
                return True
            except Exception as e:
                logger.warning(
                    f"Broker trailing stop error {pair} (attempt {attempt+1}): {e}")
                if attempt < SL_TP_UPDATE_MAX_RETRIES - 1:
                    time.sleep(0.5 * (2 ** attempt))
        return False

    def _open_trade(self, pair: str, state: PairState, box: dict):
        """Open a new trade for this pair."""
        # Global risk overlay check
        block_reason = self._check_risk_overlay(pair, self.direction)
        if block_reason:
            logger.info(f"RISK_BLOCK {pair} {'LONG' if self.direction > 0 else 'SHORT'}: {block_reason}")
            return

        pip = get_pair(pair).pip
        mid = state.last_mid if state.last_mid > 0 else box.get("mid_price", 0)
        if mid <= 0:
            return

        # Spread check
        spread_config = get_pair(pair).max_entry_spread
        if state.last_bid > 0 and state.last_ask > 0:
            spread = (state.last_ask - state.last_bid) / pip
            if spread > spread_config:
                logger.info(f"SPREAD BLOCK {pair}: {spread:.1f}p > {spread_config}p")
                return

        entry_est = state.last_ask if self.direction > 0 else state.last_bid
        if entry_est <= 0:
            entry_est = mid

        # Sizing — FX_VALIDATION_UNITS overrides allocator for parity testing
        validation_units = os.environ.get("FX_VALIDATION_UNITS")
        units = int(validation_units) if validation_units else 1
        if not validation_units and self.allocator:
            try:
                pair_trades = [{"pnl_pips": t.get("pnl_pips", 0)}
                               for t in state.completed_trades]
                units, debug = self.allocator.compute_units(
                    pair=pair, entry_price=entry_est,
                    risk_pips=self.box_size * 10,
                    direction=self.direction,
                    completed_trades=pair_trades,
                    open_positions={},
                    regime_rvol=state.get_regime_rvol(),
                )
                if units <= 0:
                    # Perf gate blocked — track as shadow trade
                    state.shadow_trade = {
                        "entry_price": entry_est,
                        "direction": self.direction,
                        "entry_time": str(datetime.now(timezone.utc)),
                        "running_mfe": 0.0, "running_mae": 0.0,
                    }
                    logger.info(f"SHADOW {pair}: perf gate blocked, tracking shadow trade")
                    return
                logger.info(f"SIZING {pair}: units={units} "
                           f"equity=${debug.get('equity', 0):.2f}")
            except Exception as e:
                logger.warning(f"Sizing error {pair}: {e}, using 1 unit")
                units = 1

        signed_units = units * self.direction

        # Emergency SL/TP
        sl_pips = self.emergency_sl * pip * (-self.direction)
        tp_pips = self.emergency_tp * pip * self.direction
        sl_price = entry_est + sl_pips
        tp_price = entry_est + tp_pips

        result = self.broker.place_market_order(
            pair, signed_units,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        if result.success and result.trade_id:
            state.open_trade_id = result.trade_id
            state.open_direction = self.direction
            state.open_entry_price = result.fill_price if result.fill_price and result.fill_price > 0 else entry_est
            state.open_units = abs(signed_units)
            state.open_time = str(datetime.now(timezone.utc))
            state.running_mfe = 0.0
            state.running_mae = 0.0
            state.last_broker_sl = sl_price
            logger.info(f"OPENED {pair} {'LONG' if self.direction > 0 else 'SHORT'} "
                       f"{abs(signed_units)}u @ {state.open_entry_price} id={result.trade_id}")
            notify_open(self.strategy_name, pair,
                        "LONG" if self.direction > 0 else "SHORT",
                        abs(signed_units), state.open_entry_price,
                        tp=tp_price, sl=sl_price,
                        extra=f"id={result.trade_id}", account=self.account_id)

    def _close_trade(self, pair: str, state: PairState, reason: str):
        """Close the open trade for this pair."""
        if not state.open_trade_id:
            return

        result = self.broker.close_trade(state.open_trade_id)
        pip = get_pair(pair).pip
        exit_price = result.fill_price if result.fill_price and result.fill_price > 0 else state.last_mid
        if exit_price <= 0:
            logger.error(f"BAD EXIT PRICE {pair}: fill={result.fill_price}, last_mid={state.last_mid}, using entry")
            exit_price = state.open_entry_price  # Fallback: record as 0 P/L rather than garbage
        pnl_pips = (exit_price - state.open_entry_price) * state.open_direction / pip

        logger.info(f"CLOSED {pair} {reason}: {pnl_pips:+.1f}p "
                   f"(MFE={state.running_mfe:.1f}p, MAE={state.running_mae:.1f}p)")
        notify_close(self.strategy_name, pair,
                     "LONG" if state.open_direction > 0 else "SHORT",
                     state.open_units, state.open_entry_price, exit_price,
                     pnl_pips, reason=reason,
                     extra=f"MFE={state.running_mfe:.1f}p | MAE={state.running_mae:.1f}p", account=self.account_id)

        # Record trade
        capture_ratio = pnl_pips / state.running_mfe if state.running_mfe > 0 else 0
        trade_record = {
            "pair": pair, "direction": state.open_direction,
            "entry_price": state.open_entry_price, "exit_price": exit_price,
            "pnl_pips": round(pnl_pips, 1), "exit_reason": reason,
            "mfe_pips": round(state.running_mfe, 1),
            "mae_pips": round(state.running_mae, 1),
            "capture_ratio": round(capture_ratio, 2),
        }
        state.completed_trades.append(trade_record)

        # Send trade to DB writer via ZMQ (portfolio_mgr is sole DuckDB writer)
        try:
            hours_held = (
                round((datetime.now(timezone.utc) - datetime.fromisoformat(
                    state.open_time.replace("Z", "+00:00"))).total_seconds() / 3600, 2)
                if state.open_time else 0
            )
            self.trade_db.send_trade(
                strategy=self.strategy_name, pair=pair, account_id=self.account_id,
                trade_id=state.open_trade_id, direction=state.open_direction,
                entry_price=state.open_entry_price, exit_price=exit_price,
                entry_time=state.open_time, exit_time=str(datetime.now(timezone.utc)),
                pnl_pips=pnl_pips, exit_reason=reason, hours_held=hours_held,
                units=state.open_units, mfe_pips=state.running_mfe,
                mae_pips=state.running_mae, capture_ratio=capture_ratio,
            )
        except Exception as e:
            logger.error(f"Failed to send trade to DB writer: {e}", exc_info=True)

        # Reset state
        state.open_trade_id = None
        state.open_direction = 0
        state.open_entry_price = 0.0
        state.open_units = 0
        state.last_broker_sl = None
        state.broker_trailing_active = False

    def _close_shadow(self, pair: str, state: PairState, reason: str):
        """Close shadow trade and record for perf gate recovery."""
        st = state.shadow_trade
        if st is None:
            return
        pip = get_pair(pair).pip
        exit_price = state.last_mid
        pnl_pips = (exit_price - st["entry_price"]) * st["direction"] / pip

        state.completed_trades.append({
            "pair": pair, "direction": st["direction"],
            "entry_price": st["entry_price"], "exit_price": exit_price,
            "pnl_pips": round(pnl_pips, 1), "exit_reason": f"SHADOW_{reason}",
            "mfe_pips": round(st.get("running_mfe", 0), 1),
            "mae_pips": round(st.get("running_mae", 0), 1),
        })
        state.shadow_trade = None
        logger.info(f"SHADOW CLOSED {pair} {reason}: {pnl_pips:+.1f}p")

    def run(self):
        """Main event loop — subscribe to curator events and dispatch."""
        logger.info("Starting NEAT multi-pair event loop...")

        # Subscribe to relevant P&F boxes + indicators
        topics = []
        for pair in self.active_pair_set:
            topics.append(f"pnf_box.{pair}.{self.pnf_config}")
            topics.append(f"indicator.{pair}")
            topics.append(f"candle.{pair}.S5")  # S5 fast exit for open positions
            if self.input_mode == "asi_mc_v3_h1":
                topics.append(f"indicator.{pair}.H1")
        if self.input_mode == "strength":
            topics.append("kalman.")

        market_sub = Subscriber(MARKET_PUB, topics=topics)
        risk_sub = Subscriber(ALLOCATION_PUB, topics=[make_topic(MSG_RISK_OVERLAY)])
        logger.info(f"Subscribed to {len(topics)} market topics + risk overlay")

        # Adopt existing broker positions on startup
        self._reconcile_positions()

        # Touch alive file for Docker healthcheck
        self._touch_alive()

        poll_count = 0
        _last_weekend_check = 0
        while not self._shutdown:
            try:
                # Weekend guard: check every 30s regardless of message flow
                _now = time.time()
                if _now - _last_weekend_check >= 30:
                    _last_weekend_check = _now
                    if is_weekend_close_time():
                        self._close_all_weekend()

                # Drain risk overlay updates (non-blocking)
                while True:
                    risk_msg = risk_sub.receive(timeout_ms=0)
                    if risk_msg is None:
                        break
                    _, payload = risk_msg
                    if payload.get("type") == MSG_RISK_OVERLAY:
                        self._risk_overlay = payload

                result = market_sub.receive(timeout_ms=1000)
                if result is None:
                    # Periodic tasks on idle
                    self._periodic_tasks()
                    continue

                topic, msg = result
                msg_type = msg.get("type")
                pair = msg.get("pair")

                if msg_type == "pnf_box" and pair in self.pairs:
                    config = msg.get("config")
                    if config == self.pnf_config:
                        self._on_pnf_box(pair, msg.get("boxes", []))

                elif msg_type == "indicator" and pair in self.pairs:
                    granularity = msg.get("granularity", "")
                    state = self.pairs[pair]

                    if granularity == "H1" and self.input_mode in ("asi_mc_v3_h1", "asi_mc_v7_h1", "s1_h1"):
                        # H1 indicator message — update state + run NEAT network
                        state.asi_mc_d = msg.get("asi_mc_d", 0.0)
                        state.asi_mc_dd = msg.get("asi_mc_dd", 0.0)
                        state.er_norm = msg.get("er_norm", 0.0)
                        # V7 + feature set fields
                        state.v7_bb_width = msg.get("bb_width", 0.0)
                        state.v7_stoch_d = msg.get("stoch_d", 0.0)
                        state.v7_macd_hist = msg.get("macd_hist", 0.0)
                        state.v7_range_pos = msg.get("range_pos_30", 0.0)
                        state.v7_aroon_osc = msg.get("aroon_osc", 0.0)
                        state.v7_sb_a = msg.get("sb_a", 0.0)
                        state.v7_tec_5 = msg.get("tec_5", 0.0)
                        state.v7_h1_slope = msg.get("h1_slope", 0.0)
                        state.v7_gap_norm = msg.get("gap_norm", 0.0)
                        state.v7_hl_price = msg.get("hl_price", 0.0)
                        bid = msg.get("bid", state.last_bid)
                        ask = msg.get("ask", state.last_ask)
                        if bid > 0 and ask > 0:
                            state.last_mid = (bid + ask) / 2
                            state.last_bid = bid
                            state.last_ask = ask

                        self._update_mfe_mae(pair, state)

                        if (self.max_hold_hours > 0 and state.open_trade_id
                                and state.open_time):
                            try:
                                opened = datetime.fromisoformat(
                                    state.open_time.replace("Z", "+00:00"))
                                held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                                if held >= self.max_hold_hours:
                                    logger.warning(f"MAX HOLD {pair}: {held:.1f}h")
                                    self._close_trade(pair, state, "MAX_HOLD")
                            except Exception:
                                pass

                        # Run NEAT on H1 cadence
                        if state.asi_mc_d != 0.0 or state.asi_mc_dd != 0.0:
                            self._on_asi_mc_tick(pair, state)

                    elif granularity == "" and self.input_mode in ("asi_mc", "asi_mc_v3"):
                        # S5 indicator — ASI-MC mode: update state + run NEAT
                        state.asi_mc_d = msg.get("asi_mc_d", 0.0)
                        state.asi_mc_dd = msg.get("asi_mc_dd", 0.0)
                        state.er_norm = msg.get("er_norm", 0.0)
                        bid = msg.get("bid", state.last_bid)
                        ask = msg.get("ask", state.last_ask)
                        if bid > 0 and ask > 0:
                            state.last_mid = (bid + ask) / 2
                            state.last_bid = bid
                            state.last_ask = ask

                        self._update_mfe_mae(pair, state)

                        if (self.max_hold_hours > 0 and state.open_trade_id
                                and state.open_time):
                            try:
                                opened = datetime.fromisoformat(
                                    state.open_time.replace("Z", "+00:00"))
                                held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                                if held >= self.max_hold_hours:
                                    logger.warning(f"MAX HOLD {pair}: {held:.1f}h")
                                    self._close_trade(pair, state, "MAX_HOLD")
                            except Exception:
                                pass

                        # Run network — with optional range bar gate
                        if state.asi_mc_d != 0.0 or state.asi_mc_dd != 0.0:
                            if self.range_bar_pips > 0 and state.last_mid > 0:
                                pip = get_pair(pair).pip
                                if state.range_bar_anchor == 0:
                                    state.range_bar_anchor = state.last_mid
                                move_pips = abs(state.last_mid - state.range_bar_anchor) / pip
                                if move_pips >= self.range_bar_pips:
                                    state.range_bar_anchor = state.last_mid
                                    self._on_asi_mc_tick(pair, state)
                            else:
                                self._on_asi_mc_tick(pair, state)

                    elif granularity == "" and self.input_mode in ("asi_mc_v3_h1", "asi_mc_v7_h1", "s1_h1"):
                        # S5 indicator in H1 mode — update bid/ask + MFE/MAE only (no NEAT eval)
                        bid = msg.get("bid", state.last_bid)
                        ask = msg.get("ask", state.last_ask)
                        if bid > 0 and ask > 0:
                            state.last_mid = (bid + ask) / 2
                            state.last_bid = bid
                            state.last_ask = ask
                        self._update_mfe_mae(pair, state)

                        if (self.max_hold_hours > 0 and state.open_trade_id
                                and state.open_time):
                            try:
                                opened = datetime.fromisoformat(
                                    state.open_time.replace("Z", "+00:00"))
                                held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                                if held >= self.max_hold_hours:
                                    logger.warning(f"MAX HOLD {pair}: {held:.1f}h")
                                    self._close_trade(pair, state, "MAX_HOLD")
                            except Exception:
                                pass

                    else:
                        # Default: update shared state for pnf/mtf_mc/strength modes
                        state.last_mc = msg.get(f"mc_{self.pnf_config}", 0.0)
                        state.last_h1_sr = msg.get("h1_support", 0.0)
                        state.last_mtf_mc_d = msg.get("mtf_mc_d", 0.0)
                        state.last_mtf_mc_dd = msg.get("mtf_mc_dd", 0.0)
                        state.last_bid = msg.get("bid", state.last_bid)
                        state.last_ask = msg.get("ask", state.last_ask)
                        if state.last_bid > 0 and state.last_ask > 0:
                            state.last_mid = (state.last_bid + state.last_ask) / 2

                        self._update_mfe_mae(pair, state)

                        if (self.max_hold_hours > 0 and state.open_trade_id
                                and state.open_time):
                            try:
                                from datetime import datetime, timezone
                                opened = datetime.fromisoformat(
                                    state.open_time.replace("Z", "+00:00"))
                                held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                                if held >= self.max_hold_hours:
                                    logger.warning(f"MAX HOLD {pair}: {held:.1f}h >= {self.max_hold_hours}h")
                                    self._close_trade(pair, state, "MAX_HOLD")
                            except Exception:
                                pass

                elif msg_type == "kalman":
                    pass  # TODO: Update strength inputs from Kalman

                elif msg_type == "candle" and pair in self.pairs:
                    # S5 fast exit — check open positions every 5 seconds
                    state = self.pairs[pair]
                    if state.open_trade_id:
                        bid = msg.get("bid_c", 0)
                        ask = msg.get("ask_c", 0)
                        if bid > 0 and ask > 0:
                            self._s5_fast_exit_check(pair, state, bid, ask)

                poll_count += 1
                if poll_count % 1000 == 0:
                    n_open = sum(1 for s in self.pairs.values() if s.open_trade_id)
                    n_trades = sum(len(s.completed_trades) for s in self.pairs.values())
                    logger.info(f"Heartbeat: {poll_count} events, {n_open} open, {n_trades} trades")
                    self._touch_alive()

            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)
                time.sleep(1)

        # Weekend guard: close all on shutdown if market closing
        if is_weekend_close_time():
            self._close_all_weekend()

        market_sub.close()
        risk_sub.close()
        logger.info("NEAT multi-pair runner shutdown complete.")

    def _touch_alive(self):
        """Touch /tmp/alive for Docker healthcheck."""
        try:
            with open("/tmp/alive", "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

    def _periodic_tasks(self):
        """Run on idle — equity refresh, weekend guard, broker reconciliation."""
        now = time.time()

        # Equity refresh for allocator
        if self.allocator and now - self._last_equity_refresh >= EQUITY_REFRESH_INTERVAL:
            try:
                self.allocator.refresh_equity(self.broker)
            except Exception as e:
                logger.debug(f"Equity refresh failed: {e}")
            self._last_equity_refresh = now

        # Detect broker-side closes (SL/TP fills) every RECONCILE_INTERVAL
        if now - self._last_reconcile >= RECONCILE_INTERVAL:
            self._detect_broker_closes()
            self._last_reconcile = now

        # Weekend guard: close all positions
        if is_weekend_close_time():
            self._close_all_weekend()

    def _close_all_weekend(self):
        """Close all open positions for weekend guard."""
        for pair, state in self.pairs.items():
            if state.open_trade_id:
                self._close_trade(pair, state, "WEEKEND_CLOSE")
            if state.shadow_trade:
                self._close_shadow(pair, state, "WEEKEND_CLOSE")

    def _reconcile_positions(self):
        """Check broker for existing positions and adopt them."""
        trades = self.broker.get_open_trades()
        adopted = 0
        for t in trades:
            pair = t.instrument
            if pair in self.pairs:
                state = self.pairs[pair]
                state.open_trade_id = t.trade_id
                state.open_direction = 1 if t.units > 0 else -1
                state.open_entry_price = t.entry_price
                state.open_units = abs(t.units)
                state.open_time = t.open_time
                adopted += 1
                logger.info(f"Adopted {pair} {'LONG' if t.units > 0 else 'SHORT'} "
                           f"{abs(t.units)}u @ {t.entry_price} id={t.trade_id}")
        logger.info(f"Reconciliation: {adopted} positions adopted from broker")

    def _detect_broker_closes(self):
        """Detect trades closed by broker (SL/TP fills) that we still think are open."""
        # Get currently open trades from broker
        try:
            broker_trades = self.broker.get_open_trades()
        except Exception:
            return  # Skip on error, retry next interval

        broker_trade_ids = {str(t.trade_id) for t in broker_trades}

        for pair, state in self.pairs.items():
            if not state.open_trade_id:
                continue
            if str(state.open_trade_id) in broker_trade_ids:
                continue

            # Trade we think is open is NOT on the broker — broker closed it (SL/TP)
            pip = get_pair(pair).pip
            # Try to get actual close price from broker trade history
            exit_price = 0.0
            exit_reason = "BROKER_CLOSED"
            try:
                trade_details = self.broker.get_trade_details(state.open_trade_id)
                if trade_details:
                    close_price = getattr(trade_details, 'close_price', None)
                    realized_pl = getattr(trade_details, 'realizedPL', None)
                    if close_price and float(close_price) > 0:
                        exit_price = float(close_price)
                    # Detect if it was SL or TP
                    if realized_pl is not None and float(realized_pl) < 0:
                        exit_reason = "BROKER_SL"
                    elif realized_pl is not None and float(realized_pl) > 0:
                        exit_reason = "BROKER_TP"
            except Exception:
                pass

            if exit_price <= 0:
                # Fallback: use last known mid price (best market estimate)
                # NEVER use SL price as fallback — causes phantom losses when
                # trades are closed externally (e.g. flatten-all command)
                exit_price = state.last_mid if state.last_mid > 0 else state.open_entry_price

            pnl_pips = (exit_price - state.open_entry_price) * state.open_direction / pip

            logger.info(f"CLOSED {pair} {exit_reason}: {pnl_pips:+.1f}p "
                       f"(MFE={state.running_mfe:.1f}p, MAE={state.running_mae:.1f}p)")
            notify_close(self.strategy_name, pair,
                        "LONG" if state.open_direction > 0 else "SHORT",
                        state.open_units, state.open_entry_price, exit_price,
                        pnl_pips, reason=exit_reason,
                        extra=f"MFE={state.running_mfe:.1f}p | MAE={state.running_mae:.1f}p",
                        account=self.account_id)

            # Record trade
            capture_ratio = pnl_pips / state.running_mfe if state.running_mfe > 0 else 0
            trade_record = {
                "pair": pair, "direction": state.open_direction,
                "entry_price": state.open_entry_price, "exit_price": exit_price,
                "pnl_pips": round(pnl_pips, 1), "exit_reason": exit_reason,
                "mfe_pips": round(state.running_mfe, 1),
                "mae_pips": round(state.running_mae, 1),
                "capture_ratio": round(capture_ratio, 2),
            }
            state.completed_trades.append(trade_record)

            try:
                hours_held = (
                    round((datetime.now(timezone.utc) - datetime.fromisoformat(
                        state.open_time.replace("Z", "+00:00"))).total_seconds() / 3600, 2)
                    if state.open_time else 0
                )
                self.trade_db.send_trade(
                    strategy=self.strategy_name, pair=pair, account_id=self.account_id,
                    trade_id=state.open_trade_id, direction=state.open_direction,
                    entry_price=state.open_entry_price, exit_price=exit_price,
                    entry_time=state.open_time, exit_time=str(datetime.now(timezone.utc)),
                    pnl_pips=pnl_pips, exit_reason=exit_reason, hours_held=hours_held,
                    units=state.open_units, mfe_pips=state.running_mfe,
                    mae_pips=state.running_mae, capture_ratio=capture_ratio,
                )
            except Exception as e:
                logger.error(f"Failed to send broker-close trade to DB: {e}")

            # Reset state
            state.open_trade_id = None
            state.open_direction = 0
            state.open_entry_price = 0.0
            state.open_units = 0
            state.last_broker_sl = None
            state.broker_trailing_active = False


def main():
    runner = NeatMultiPairRunner()
    runner.run()


if __name__ == "__main__":
    main()
