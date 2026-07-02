#!/usr/bin/env python3
"""
FX-Core Data Curator — single source of truth for all market data.

Responsibilities:
  1. Maintain 1 OANDA price stream for all 12 pairs
  2. Poll candles at scheduled intervals (S5/M1/M5/H1/D1)
  3. Compute all indicators incrementally (P&F, MC, zigzag, ATR, MTF-MC, Kalman, Asian)
  4. Write to DuckDB (fx.duckdb)
  5. Publish events via ZMQ for strategy containers
  6. Respond to health checks for Docker

This replaces 99 independent OANDA connections with 1 stream + scheduled polls.
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime, timezone

import v20

from lib.pair_config import PAIRS, ALL_PAIR_NAMES, PNF_CONFIGS, get_pair
from lib.indicators import PnFBuilder, ZigzagSR, ATR, MTFMC, AsianRange, KalmanStrength, ASIMC, SwingStructure
from lib.zmq_protocol import (
    Publisher, HealthResponder,
    MARKET_PUB, HEALTH_REP,
    make_topic, MSG_CANDLE, MSG_PNF_BOX, MSG_INDICATOR, MSG_KALMAN, MSG_HEARTBEAT,
)
from lib.db import get_fx_db, init_fx_schema
from lib.oanda_adapter import OANDAAdapter

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [curator] %(message)s",
)
logger = logging.getLogger("curator")


class DataCurator:
    """Central data curator — streams, computes, publishes, persists."""

    def __init__(self):
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # OANDA client (for candle polling + stream)
        self.oanda = OANDAAdapter()

        # DuckDB — brief connections only (open/write/close) to avoid
        # holding the single-writer lock across Docker PID namespaces.
        db = get_fx_db(read_only=False)
        init_fx_schema(db)
        db.close()
        logger.info("DuckDB schema initialized")

        # ZMQ publisher + health responder
        self.publisher = Publisher(MARKET_PUB)
        self.health = HealthResponder(HEALTH_REP)
        logger.info(f"ZMQ publisher bound: {MARKET_PUB}")

        # ─── Initialize indicators for all pairs ───────────────────────

        # P&F builders: {(pair, config_name): PnFBuilder}
        self.pnf_builders = {}
        for pair_name, pair_cfg in PAIRS.items():
            for pnf_cfg in PNF_CONFIGS:
                key = (pair_name, pnf_cfg["name"])
                self.pnf_builders[key] = PnFBuilder(
                    box_size_pips=pnf_cfg["box_size_pips"],
                    reversal=pnf_cfg["reversal"],
                    pip=pair_cfg.pip,
                )

        # H1 zigzag S/R: {pair: ZigzagSR}
        self.zigzag = {}
        for pair_name, pair_cfg in PAIRS.items():
            min_swing = 3.0 * pair_cfg.median_spread_pips * pair_cfg.pip
            self.zigzag[pair_name] = ZigzagSR(min_swing=min_swing)

        # ATR(14): {(pair, timeframe): ATR}
        self.atr = {}
        for pair_name in ALL_PAIR_NAMES:
            self.atr[(pair_name, "M5")] = ATR(14)
            self.atr[(pair_name, "H1")] = ATR(14)

        # MTF-MC: {pair: MTFMC}
        self.mtf_mc = {}
        for pair_name, pair_cfg in PAIRS.items():
            self.mtf_mc[pair_name] = MTFMC(pip=pair_cfg.pip)

        # ASI-MC: {pair: ASIMC} — ASI → MC(D)/MC(dD) on M5 OHLC
        self.asi_mc = {pair: ASIMC() for pair in ALL_PAIR_NAMES}
        # Swing structure: {pair: SwingStructure} — TopsBots SB-A/SB-P/ERP/HHHL on M5 OHLC
        self.swing = {pair: SwingStructure() for pair in ALL_PAIR_NAMES}
        # S5→M5 resample accumulators: {pair: {o, h, l, c, count}}
        self._m5_accum = {pair: {"o": 0, "h": -1e10, "l": 1e10, "c": 0, "count": 0}
                          for pair in ALL_PAIR_NAMES}

        # H1 indicator tracking: last published H1 bar timestamp per pair
        self._last_h1_indicator_ts = {pair: "" for pair in ALL_PAIR_NAMES}

        # V7 indicator buffers: rolling M5 close/high/low for computing
        # bb_width, stoch_d, macd_hist, range_pos_30, aroon_osc at H1 bar close
        self._v7_m5_close = {pair: [] for pair in ALL_PAIR_NAMES}
        self._v7_m5_high = {pair: [] for pair in ALL_PAIR_NAMES}
        self._v7_m5_low = {pair: [] for pair in ALL_PAIR_NAMES}

        # Asian range: {pair: AsianRange}
        self.asian = {pair: AsianRange() for pair in ALL_PAIR_NAMES}

        # Kalman strength (shared across all pairs)
        kalman_pairs = [
            (p, PAIRS[p].base, PAIRS[p].quote) for p in ALL_PAIR_NAMES
            if PAIRS[p].base in ("EUR", "USD", "GBP", "JPY", "AUD", "NZD")
            and PAIRS[p].quote in ("EUR", "USD", "GBP", "JPY", "AUD", "NZD")
        ]
        kalman_currencies = list(set(
            c for p in kalman_pairs for c in (p[1], p[2])
        ))
        self.kalman_h1 = KalmanStrength(kalman_currencies, kalman_pairs)
        self.kalman_d1 = KalmanStrength(kalman_currencies, kalman_pairs)

        # Timing
        self._last_s5_poll = 0
        self._last_h1_poll = 0
        self._last_d1_poll = 0
        self._last_health_check = 0
        self._last_heartbeat = 0

        logger.info(f"Curator initialized: {len(ALL_PAIR_NAMES)} pairs, "
                    f"{len(self.pnf_builders)} P&F charts, "
                    f"{len(self.zigzag)} zigzag, "
                    f"{len(self.atr)} ATR, "
                    f"{len(self.mtf_mc)} MTF-MC")

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum}, shutting down...")
        self._shutdown = True

    # ─── Warmup ────────────────────────────────────────────────────────

    def warmup(self):
        """Fetch historical data and prime all indicators."""
        logger.info("=" * 60)
        logger.info("WARMUP: Priming indicators from OANDA history...")
        logger.info("=" * 60)

        # H1 bars for zigzag + ATR + Kalman + Asian range
        logger.info("Fetching H1 history (200 bars per pair)...")
        h1_closes = {}
        for pair in ALL_PAIR_NAMES:
            candles = self.oanda.get_candles(pair, count=200, granularity="H1")
            for c in candles:
                bar = {"open": c["open"], "high": c["high"],
                       "low": c["low"], "close": c["close"]}
                self.zigzag[pair].update_from_h1_bar(bar)
                self.atr[(pair, "H1")].update(bar)

                # Asian range
                ts = c["timestamp"]
                hour = int(str(ts)[11:13]) if isinstance(ts, str) else ts.hour
                self.asian[pair].update_from_h1(bar, hour)

            if candles:
                h1_closes[pair] = candles[-1]["close"]
            logger.info(f"  {pair}: {len(candles)} H1 bars, "
                       f"S={self.zigzag[pair].state.support:.3f}, "
                       f"R={self.zigzag[pair].state.resistance:.3f}")

        # Kalman warmup from H1
        if h1_closes:
            self.kalman_h1.update(h1_closes)
            self.kalman_h1.warmup_done = True
            logger.info(f"  Kalman H1 warmed up: {self.kalman_h1.get_ranks()}")

        # M1 bars for P&F priming (1000 per pair)
        logger.info("Fetching M1 history (1000 bars per pair) for P&F...")
        for pair in ALL_PAIR_NAMES:
            candles = self.oanda.get_candles(pair, count=1000, granularity="M1")
            for c in candles:
                mid = c["close"]
                ts = str(c["timestamp"])
                for pnf_cfg in PNF_CONFIGS:
                    key = (pair, pnf_cfg["name"])
                    self.pnf_builders[key].process_price(mid, ts)
            logger.info(f"  {pair}: {len(candles)} M1 bars → "
                       f"{sum(len(self.pnf_builders[(pair, cfg['name'])].state.box_history) for cfg in PNF_CONFIGS)} total P&F boxes")

        # M5 bars for ASI-MC priming
        logger.info("Fetching M5 history (1000 bars per pair) for ASI-MC + SwingStructure...")
        for pair in ALL_PAIR_NAMES:
            candles = self.oanda.get_candles(pair, count=1000, granularity="M5")
            for c in candles:
                self.asi_mc[pair].append_m5(c["open"], c["high"], c["low"], c["close"])
                self.swing[pair].append_m5(c["open"], c["high"], c["low"], c["close"])
                self._v7_m5_close[pair].append(c["close"])
                self._v7_m5_high[pair].append(c["high"])
                self._v7_m5_low[pair].append(c["low"])
            if candles:
                mc_d, mc_dd = self.asi_mc[pair].compute()
                sw = self.swing[pair].compute()
                logger.info(f"  {pair}: {len(candles)} M5 bars, ASI-MC [{mc_d:.3f}, {mc_dd:.3f}] "
                            f"SB-A={sw['sb_a']:+.1f} ERP-P={sw['erp_p']:.2f} "
                            f"HH={sw['hh_price']:.0f}/HL={sw['hl_price']:.0f}")

        # S5 bars for MTF-MC buffer priming
        logger.info("Fetching S5 history for MTF-MC buffer...")
        for pair in ALL_PAIR_NAMES:
            pip = PAIRS[pair].pip
            candles = self.oanda.get_candles(pair, count=1000, granularity="S5")
            for c in candles:
                self.mtf_mc[pair].append_s5(c["close"])

                # Also accumulate into H1 for zigzag (from S5)
                self.zigzag[pair].accumulate_s5(c)

                # P&F from S5 too (fills gap between M1 end and now)
                for pnf_cfg in PNF_CONFIGS:
                    key = (pair, pnf_cfg["name"])
                    self.pnf_builders[key].process_price(c["close"], str(c["timestamp"]))

            logger.info(f"  {pair}: {len(candles)} S5 bars, MTF-MC buffer={len(self.mtf_mc[pair].s5_buffer)}")

        logger.info("=" * 60)
        logger.info("WARMUP COMPLETE")
        logger.info("=" * 60)

    # ─── Live Polling ──────────────────────────────────────────────────

    def _poll_s5(self):
        """Poll S5 candles for all pairs, update P&F + MTF-MC, publish events."""
        db_rows = []  # Batch candle rows for a single brief DB write

        for pair in ALL_PAIR_NAMES:
            candles = self.oanda.get_candles(pair, count=5, granularity="S5")
            if not candles:
                continue

            pip = PAIRS[pair].pip
            for c in candles:
                mid = c["close"]
                ts_str = str(c["timestamp"])

                # Update MTF-MC buffer
                self.mtf_mc[pair].append_s5(mid)

                # Accumulate S5 → M5 for ASI-MC
                acc = self._m5_accum[pair]
                acc["count"] += 1
                if acc["count"] == 1:
                    acc["o"] = c["open"]
                    acc["h"] = c["high"]
                    acc["l"] = c["low"]
                else:
                    if c["high"] > acc["h"]:
                        acc["h"] = c["high"]
                    if c["low"] < acc["l"]:
                        acc["l"] = c["low"]
                acc["c"] = c["close"]
                if acc["count"] >= 12:  # 12 S5 = 1 M5
                    self.asi_mc[pair].append_m5(acc["o"], acc["h"], acc["l"], acc["c"])
                    self.swing[pair].append_m5(acc["o"], acc["h"], acc["l"], acc["c"])
                    # V7 buffers
                    self._v7_m5_close[pair].append(acc["c"])
                    self._v7_m5_high[pair].append(acc["h"])
                    self._v7_m5_low[pair].append(acc["l"])
                    if len(self._v7_m5_close[pair]) > 200:
                        self._v7_m5_close[pair] = self._v7_m5_close[pair][-200:]
                        self._v7_m5_high[pair] = self._v7_m5_high[pair][-200:]
                        self._v7_m5_low[pair] = self._v7_m5_low[pair][-200:]

                    # Compute ASI-MC now (after append) and embed in MSG_CANDLE so
                    # strategies consuming M5 candles get same-bar mc_d without waiting
                    # for the MSG_INDICATOR that is published later in this poll cycle.
                    _m5_asi_d, _m5_asi_dd = self.asi_mc[pair].compute()

                    # Publish M5 candle event (needed by Impulse strategy)
                    m5_topic = make_topic(MSG_CANDLE, pair, "M5")
                    self.publisher.publish(m5_topic, {
                        "type": MSG_CANDLE,
                        "pair": pair,
                        "granularity": "M5",
                        "ts": ts_str,
                        "o": acc["o"], "h": acc["h"], "l": acc["l"], "c": acc["c"],
                        "bid_c": c.get("bid_c", mid), "ask_c": c.get("ask_c", mid),
                        "volume": 0,
                        "asi_mc_d":  _m5_asi_d,   # same-bar mc_d for SBA IronNet
                        "asi_mc_dd": _m5_asi_dd,
                    })

                    acc["o"] = 0; acc["h"] = -1e10; acc["l"] = 1e10; acc["c"] = 0; acc["count"] = 0

                # Update P&F builders and collect new boxes
                for pnf_cfg in PNF_CONFIGS:
                    key = (pair, pnf_cfg["name"])
                    new_boxes = self.pnf_builders[key].process_price(mid, ts_str)

                    if new_boxes:
                        # Publish P&F box event
                        box_dicts = [{
                            "box_id": b.column_id * 1000 + len(self.pnf_builders[key].state.box_history),
                            "column_id": b.column_id,
                            "direction": b.direction,
                            "level": b.level,
                            "mid_price": b.mid_price,
                            "ts": b.timestamp,
                        } for b in new_boxes]

                        topic = make_topic(MSG_PNF_BOX, pair, config=pnf_cfg["name"])
                        self.publisher.publish(topic, {
                            "type": MSG_PNF_BOX,
                            "pair": pair,
                            "config": pnf_cfg["name"],
                            "boxes": box_dicts,
                        })

                # Accumulate into H1 zigzag
                self.zigzag[pair].accumulate_s5(c)

                # Publish candle event
                topic = make_topic(MSG_CANDLE, pair, "S5")
                self.publisher.publish(topic, {
                    "type": MSG_CANDLE,
                    "pair": pair,
                    "granularity": "S5",
                    "ts": ts_str,
                    "o": c["open"], "h": c["high"], "l": c["low"], "c": c["close"],
                    "bid_c": c.get("bid_c", mid), "ask_c": c.get("ask_c", mid),
                    "volume": c.get("volume", 0),
                })

                # Collect row for batch DB write
                db_rows.append([pair, "S5", ts_str,
                               c["open"], c["high"], c["low"], c["close"],
                               c.get("bid_c", mid), c.get("ask_c", mid), c.get("volume", 0)])

            # Publish indicator snapshot after processing all candles for this pair
            mc_d, mc_dd = self.mtf_mc[pair].compute()
            zz = self.zigzag[pair].state
            atr_h1 = self.atr.get((pair, "H1"), ATR()).value
            asian = self.asian[pair]

            # Latest bid/ask from most recent S5 candle
            last_bid = candles[-1].get("bid_c", mid) if candles else mid
            last_ask = candles[-1].get("ask_c", mid) if candles else mid

            indicator_msg = {
                "type": MSG_INDICATOR,
                "pair": pair,
                "ts": str(datetime.now(timezone.utc)),
                "bid": last_bid,
                "ask": last_ask,
                "h1_support": zz.support,
                "h1_resistance": zz.resistance,
                "h1_zz_dir": zz.zz_direction,
                "atr14_h1": atr_h1,
                "mtf_mc_d": mc_d,
                "mtf_mc_dd": mc_dd,
                "asian_high": asian.high,
                "asian_low": asian.low,
                "asian_mid": asian.mid,
            }

            # ASI-MC
            asi_d, asi_dd = self.asi_mc[pair].compute()
            indicator_msg["asi_mc_d"] = asi_d
            indicator_msg["asi_mc_dd"] = asi_dd

            # Swing structure (TopsBots SB-A/SB-P + ERP + HHHL)
            sw = self.swing[pair].compute()
            indicator_msg["sb_a"]      = sw["sb_a"]
            indicator_msg["erp_a"]     = sw["erp_a"]
            indicator_msg["hh_asi"]    = sw["hh_asi"]
            indicator_msg["hl_asi"]    = sw["hl_asi"]
            indicator_msg["erp_p"]     = sw["erp_p"]
            indicator_msg["hh_price"]  = sw["hh_price"]
            indicator_msg["hl_price"]  = sw["hl_price"]
            indicator_msg["d_erp_p"]   = sw["d_erp_p"]   # 1h velocity of price range pos
            indicator_msg["d_erp_a"]   = sw["d_erp_a"]   # 1h velocity of ASI range pos

            # ASI-MC V3: ER_norm — Kaufman ER(60 M5 bars), arctan-normalized
            _m5c = self.asi_mc[pair].m5_c
            if len(_m5c) >= 61:
                import numpy as _np
                _c = _np.array(_m5c[-61:], dtype=_np.float64)
                _net = abs(float(_c[-1]) - float(_c[0]))
                _path = float(_np.sum(_np.abs(_np.diff(_c))))
                _er = _net / _path if _path > 0 else 0.0
                indicator_msg["er_norm"] = float(_np.arctan(_er / 0.3) / (_np.pi / 2))
            else:
                indicator_msg["er_norm"] = 0.0

            # Add MC per P&F config
            for pnf_cfg in PNF_CONFIGS:
                key = (pair, pnf_cfg["name"])
                mc_val = self.pnf_builders[key].compute_mc()
                indicator_msg[f"mc_{pnf_cfg['name']}"] = mc_val

            topic = make_topic(MSG_INDICATOR, pair)
            self.publisher.publish(topic, indicator_msg)

        # Batch write all candles to DuckDB (brief connection)
        if db_rows:
            try:
                db = get_fx_db(read_only=False)
                db.begin()
                for row in db_rows:
                    db.execute(
                        "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        row)
                db.commit()
                db.close()
            except Exception as e:
                try:
                    db.rollback()
                    db.close()
                except Exception:
                    pass
                if "Constraint" not in type(e).__name__:
                    logger.warning(f"DB batch write error: {e}")

    def _poll_h1(self):
        """Poll H1 candles for zigzag + ATR + Kalman update."""
        h1_closes = {}
        for pair in ALL_PAIR_NAMES:
            candles = self.oanda.get_candles(pair, count=2, granularity="H1")
            if not candles:
                continue

            for c in candles:
                bar = {"open": c["open"], "high": c["high"],
                       "low": c["low"], "close": c["close"]}
                self.zigzag[pair].update_from_h1_bar(bar)
                self.atr[(pair, "H1")].update(bar)

                ts = c["timestamp"]
                hour = int(str(ts)[11:13]) if isinstance(ts, str) else ts.hour
                self.asian[pair].update_from_h1(bar, hour)

            # Publish latest H1 candle event (needed by Impulse/Fade strategies)
            last_h1 = candles[-1]
            h1_topic = make_topic(MSG_CANDLE, pair, "H1")
            self.publisher.publish(h1_topic, {
                "type": MSG_CANDLE,
                "pair": pair,
                "granularity": "H1",
                "ts": str(last_h1["timestamp"]),
                "o": last_h1["open"], "h": last_h1["high"],
                "l": last_h1["low"], "c": last_h1["close"],
                "bid_c": last_h1.get("bid_c", last_h1["close"]),
                "ask_c": last_h1.get("ask_c", last_h1["close"]),
                "volume": last_h1.get("volume", 0),
            })

            h1_closes[pair] = candles[-1]["close"]

            # Publish H1 indicator message (M5-computed values at H1 cadence)
            # Matches training data: M5 indicators resampled to H1 via .last()
            last_ts = str(candles[-1]["timestamp"])
            if last_ts != self._last_h1_indicator_ts.get(pair, ""):
                self._last_h1_indicator_ts[pair] = last_ts
                asi_d, asi_dd = self.asi_mc[pair].compute()
                _m5c = self.asi_mc[pair].m5_c
                if len(_m5c) >= 61:
                    import numpy as _np
                    _c = _np.array(_m5c[-61:], dtype=_np.float64)
                    _net = abs(float(_c[-1]) - float(_c[0]))
                    _path = float(_np.sum(_np.abs(_np.diff(_c))))
                    _er = _net / _path if _path > 0 else 0.0
                    er_norm = float(_np.arctan(_er / 0.3) / (_np.pi / 2))
                else:
                    er_norm = 0.0

                # V7 indicators from M5 buffers (bb_width, stoch_d, macd_hist, range_pos_30, aroon_osc)
                v7 = {}
                m5c = self._v7_m5_close[pair]
                m5h = self._v7_m5_high[pair]
                m5l = self._v7_m5_low[pair]
                if len(m5c) >= 30:
                    import numpy as _np
                    _c = _np.array(m5c[-30:], dtype=_np.float64)
                    _h = _np.array(m5h[-30:], dtype=_np.float64)
                    _l = _np.array(m5l[-30:], dtype=_np.float64)
                    # BB width
                    _sma = float(_np.mean(_c[-20:]))
                    _std = float(_np.std(_c[-20:]))
                    v7["bb_width"] = (_sma + 2*_std - (_sma - 2*_std)) / _sma if _sma > 0 else 0.0
                    # Stochastic %D (14,3)
                    _lo14 = float(_np.min(_l[-14:]))
                    _hi14 = float(_np.max(_h[-14:]))
                    _rng14 = _hi14 - _lo14
                    _k = (float(_c[-1]) - _lo14) / _rng14 if _rng14 > 0 else 0.5
                    _k2 = (float(_c[-2]) - float(_np.min(_l[-15:-1]))) / max(float(_np.max(_h[-15:-1])) - float(_np.min(_l[-15:-1])), 1e-10)
                    _k3 = (float(_c[-3]) - float(_np.min(_l[-16:-2]))) / max(float(_np.max(_h[-16:-2])) - float(_np.min(_l[-16:-2])), 1e-10)
                    v7["stoch_d"] = (_k + _k2 + _k3) / 3.0
                    # MACD histogram / ATR
                    _c26 = _np.array(m5c[-30:], dtype=_np.float64)
                    _ema12 = float(_np.mean(_c26[-12:]))  # simplified
                    _ema26 = float(_np.mean(_c26[-26:]))
                    _macd = _ema12 - _ema26
                    _tr = _np.maximum(_h[-14:] - _l[-14:], _np.abs(_h[-14:] - _np.roll(_c[-14:], 1)))
                    _atr = float(_np.mean(_tr[1:]))
                    v7["macd_hist"] = _macd / _atr if _atr > 0 else 0.0
                    # Range position (30-bar)
                    _hh30 = float(_np.max(_h))
                    _ll30 = float(_np.min(_l))
                    _rng30 = _hh30 - _ll30
                    v7["range_pos_30"] = (float(_c[-1]) - _ll30) / _rng30 if _rng30 > 0 else 0.5
                    # Aroon oscillator (25-bar)
                    if len(m5h) >= 26:
                        _h25 = _np.array(m5h[-26:], dtype=_np.float64)
                        _l25 = _np.array(m5l[-26:], dtype=_np.float64)
                        _hh_idx = int(_np.argmax(_h25[1:]) + 1)
                        _ll_idx = int(_np.argmin(_l25[1:]) + 1)
                        v7["aroon_osc"] = (_hh_idx - _ll_idx) / 25.0
                    else:
                        v7["aroon_osc"] = 0.0

                    # TEC_5: signed Kaufman ER (5-bar)
                    if len(m5c) >= 6:
                        _c6 = _np.array(m5c[-6:], dtype=_np.float64)
                        _net_tec = float(_c6[-1] - _c6[0])  # signed
                        _path_tec = float(_np.sum(_np.abs(_np.diff(_c6))))
                        _raw_tec = _net_tec / _path_tec if _path_tec > 0 else 0.0
                        v7["tec_5"] = float(_np.arctan(_raw_tec / 0.3) / (_np.pi / 2))
                    else:
                        v7["tec_5"] = 0.0

                    # H1_slope: linreg slope on last 3 H1 closes (36 M5 bars apart)
                    if len(m5c) >= 37:
                        _vals = _np.array([m5c[-36], m5c[-24], m5c[-12]], dtype=_np.float64)
                        _slope = (_vals[2] - _vals[0]) / 2.0  # simple linreg slope
                        _rng_sl = float(_vals.max() - _vals.min())
                        _norm = _slope / _rng_sl if _rng_sl > 0 else 0.0
                        v7["h1_slope"] = float(_np.arctan(_norm * 3.0) / (_np.pi / 2))
                    else:
                        v7["h1_slope"] = 0.0

                    # gap_norm: gap between last 2 M5 bars / prev range
                    if len(m5c) >= 3 and len(m5h) >= 3:
                        _prev_rng = m5h[-2] - m5l[-2]
                        v7["gap_norm"] = (m5c[-1] - m5c[-2]) / _prev_rng if _prev_rng > 0 else 0.0
                    else:
                        v7["gap_norm"] = 0.0

                    # Swing structure features (sb_a, hl_price)
                    sw = self.swing[pair].compute()
                    v7["sb_a"] = float(sw.get("sb_a", 0.0))
                    v7["hl_price"] = float(sw.get("hl_price", 0.0))

                h1_ind_msg = {
                    "type": MSG_INDICATOR,
                    "pair": pair,
                    "granularity": "H1",
                    "ts": str(datetime.now(timezone.utc)),
                    "h1_bar_ts": last_ts,
                    "bid": last_h1.get("bid_c", last_h1["close"]),
                    "ask": last_h1.get("ask_c", last_h1["close"]),
                    "asi_mc_d": asi_d,
                    "asi_mc_dd": asi_dd,
                    "er_norm": er_norm,
                }
                h1_ind_msg.update(v7)
                h1_ind_topic = make_topic(MSG_INDICATOR, pair, "H1")
                self.publisher.publish(h1_ind_topic, h1_ind_msg)
                logger.info(f"H1 indicator {pair}: mc_d={asi_d:.3f} er={er_norm:.3f} bb_w={v7.get('bb_width',0):.4f} stoch={v7.get('stoch_d',0):.3f}")

        # Update Kalman H1
        if h1_closes:
            strengths = self.kalman_h1.update(h1_closes)
            if strengths is not None:
                ranks = self.kalman_h1.get_ranks()
                topic = make_topic(MSG_KALMAN, granularity="H1")
                self.publisher.publish(topic, {
                    "type": MSG_KALMAN,
                    "granularity": "H1",
                    "ts": str(datetime.now(timezone.utc)),
                    "strengths": strengths,
                    "ranks": ranks,
                })
                logger.info(f"Kalman H1 updated: ranks={ranks}")

    # ─── Main Loop ─────────────────────────────────────────────────────

    # ─── Streaming ──────────────────────────────────────────────────

    def _start_price_stream(self):
        """Start OANDA price stream in a background thread."""
        self._stream_thread = threading.Thread(target=self._price_stream_loop, daemon=True)
        self._stream_active = False
        self._stream_thread.start()

    def _price_stream_loop(self):
        """Background thread: receive streaming prices from OANDA."""
        instruments = ",".join(ALL_PAIR_NAMES)
        while not self._shutdown:
            try:
                logger.info(f"Starting price stream for {len(ALL_PAIR_NAMES)} pairs...")
                stream_ctx = self.oanda.create_stream_context()
                response = stream_ctx.pricing.stream(
                    self.oanda.account_id,
                    instruments=instruments,
                    snapshot=False,
                )
                self._stream_active = True
                tick_count = 0
                for msg_type, msg in response.parts():
                    if self._shutdown:
                        break
                    if msg_type == "pricing.ClientPrice":
                        self._on_stream_tick(msg)
                        tick_count += 1
                    # pricing.PricingHeartbeat = keep-alive (ignore)
                # Stream ended (market closed or disconnect)
                logger.info(f"Price stream ended after {tick_count} ticks, reconnecting in 30s...")
                self._stream_active = False
                time.sleep(30)
            except Exception as e:
                logger.warning(f"Price stream error: {e}, reconnecting in 30s...")
                self._stream_active = False
                time.sleep(30)
        self._stream_active = False

    def _on_stream_tick(self, price):
        """Process a streaming tick — update indicators and publish."""
        try:
            pair = price.instrument
            if pair not in PAIRS:
                return

            # Extract bid/ask/mid
            bids = price.bids
            asks = price.asks
            if not bids or not asks:
                return
            bid = float(bids[0].price)
            ask = float(asks[0].price)
            mid = (bid + ask) / 2
            ts_str = str(price.time)[:23]

            pip = PAIRS[pair].pip

            # Publish tick as S5 candle (strategies use for fast exit)
            # M5 bars, indicators, P&F, DuckDB handled by _poll_s5 (always running)
            self.publisher.publish(make_topic(MSG_CANDLE, pair, "S5"), {
                "type": MSG_CANDLE, "pair": pair, "granularity": "S5",
                "ts": ts_str,
                "o": mid, "h": mid, "l": mid, "c": mid,
                "bid_c": bid, "ask_c": ask, "volume": 0,
            })

        except Exception as e:
            logger.debug(f"Stream tick error {pair}: {e}")

    def run(self):
        """Main entry point."""
        logger.info("FX-Core Data Curator starting...")
        self.warmup()

        # Start streaming thread
        self._start_price_stream()
        logger.info("Price stream started in background thread")

        logger.info("Entering main loop...")
        poll_count = 0
        last_successful_poll = time.time()
        STALL_TIMEOUT = 180  # exit if no poll completes in 3 min → Docker restarts

        while not self._shutdown:
            now = time.time()

            # Stall watchdog: if poll loop has been stuck for > 3 min, exit cleanly
            if now - last_successful_poll > STALL_TIMEOUT:
                logger.error(f"Poll loop stalled for >{STALL_TIMEOUT}s — exiting for Docker restart")
                break

            # S5 poll ALWAYS runs — handles M5 bars, indicators, DuckDB writes
            # Stream adds tick-level S5 candles for fast exit (complementary)
            if now - self._last_s5_poll >= 5:
                self._poll_s5()
                self._last_s5_poll = now
                poll_count += 1
                last_successful_poll = time.time()

            # H1 poll every 5 minutes (check for new H1 bar)
            if now - self._last_h1_poll >= 300:
                self._poll_h1()
                self._last_h1_poll = now

            # Health check responder (non-blocking)
            if now - self._last_health_check >= 1:
                self.health.check_and_respond(timeout_ms=10)
                self._last_health_check = now

            # Heartbeat (every 30s)
            if now - self._last_heartbeat >= 30:
                stream_status = "STREAMING" if self._stream_active else "POLLING"
                self.publisher.publish("heartbeat", {
                    "type": MSG_HEARTBEAT,
                    "ts": str(datetime.now(timezone.utc)),
                    "poll_count": poll_count,
                    "pairs": len(ALL_PAIR_NAMES),
                    "pnf_charts": len(self.pnf_builders),
                    "warmup_complete": True,
                    "stream_active": self._stream_active,
                })
                self._last_heartbeat = now

                if poll_count % 12 == 0:
                    logger.info(f"Heartbeat: {stream_status}, poll #{poll_count}, "
                               f"uptime {(now - self._start_time)/3600:.1f}h")

            time.sleep(1)

        logger.info("Curator shutting down...")
        self.publisher.close()
        self.health.close()
        logger.info("Curator shutdown complete.")

    @property
    def _start_time(self):
        if not hasattr(self, '__start_time'):
            self.__start_time = time.time()
        return self.__start_time


def main():
    curator = DataCurator()
    curator.run()


if __name__ == "__main__":
    main()
