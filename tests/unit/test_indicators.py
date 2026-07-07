"""Tests for indicator engine — must produce identical output to scalper's live traders."""

import pytest
import math
from lib.indicators import PnFBuilder, PnFBox, ZigzagSR, ATR, MTFMC, AsianRange


class TestPnFBuilder:
    """P&F box building must match scalper/unified/live_neat_pf_unified.py exactly."""

    def _make_builder(self, box_pips=5, reversal=3, pip=0.01):
        return PnFBuilder(box_size_pips=box_pips, reversal=reversal, pip=pip)

    def test_first_price_creates_one_box(self):
        b = self._make_builder()
        boxes = b.process_price(184.321, "2026-01-01T00:00:00")
        assert len(boxes) == 1
        assert boxes[0].direction == 1
        assert boxes[0].column_id == 0

    def test_snap_to_box(self):
        b = self._make_builder(box_pips=5, pip=0.01)
        # box_size = 0.05
        assert b._snap_to_box(184.321) == pytest.approx(184.30, abs=0.001)
        assert b._snap_to_box(184.05) == pytest.approx(184.05, abs=0.001)
        assert b._snap_to_box(184.049) == pytest.approx(184.00, abs=0.001)

    def test_continuation_up(self):
        b = self._make_builder(box_pips=10, pip=0.01)
        # box_size = 0.10. Initialize snaps 184.00 to 184.00 (level)
        b.process_price(184.00, "t0")
        # From 184.00, go to 184.20 = delta = 0.20 / 0.10 = 2 boxes
        # But first box already at 184.00, so delta = (184.20 - 184.00) / 0.10 = 2
        # Actually int((184.20 - 184.00) / 0.10) = int(2.0) = 2, but the state
        # level is 184.00, so we get boxes at 184.10 and 184.20 = 2 boxes
        # Wait: 184.20 - 184.00 = 0.20, / 0.10 = 2.0, int(2.0) = 2 ✓
        # Hmm, let me use a cleaner number
        boxes = b.process_price(184.25, "t1")  # +0.25 / 0.10 = 2.5 → int(2) = 2 boxes
        assert len(boxes) == 2
        assert all(box.direction == 1 for box in boxes)

    def test_continuation_down(self):
        b = self._make_builder(box_pips=10, pip=0.01)
        b.process_price(184.00, "t0")
        # Need to go down — force initial direction
        b.state.direction = -1
        boxes = b.process_price(183.70, "t1")  # -0.30 = 3 boxes
        assert len(boxes) == 3
        assert all(box.direction == -1 for box in boxes)

    def test_reversal(self):
        b = self._make_builder(box_pips=10, reversal=3, pip=0.01)
        b.process_price(184.00, "t0")  # Initialize
        b.process_price(184.30, "t1")  # 3 boxes up
        boxes = b.process_price(183.70, "t2")  # -6 boxes = reversal (>= 3)
        assert len(boxes) > 0
        assert boxes[-1].direction == -1
        assert b.state.column_id == 1  # New column

    def test_no_movement_no_boxes(self):
        b = self._make_builder(box_pips=10, pip=0.01)
        b.process_price(184.00, "t0")
        boxes = b.process_price(184.05, "t1")  # < 1 box
        assert len(boxes) == 0

    def test_mc_computation(self):
        b = self._make_builder(box_pips=5, pip=0.01)
        # Build 10 ascending boxes
        for i in range(10):
            b.process_price(184.00 + i * 0.05, f"t{i}")
        mc = b.compute_mc(n_lags=5)
        assert mc == pytest.approx(1.0)  # All up = MC=1.0

    def test_mc_returns_zero_insufficient_data(self):
        b = self._make_builder()
        b.process_price(184.00, "t0")
        assert b.compute_mc() == 0.0

    def test_box_history_trimmed(self):
        b = self._make_builder(box_pips=1, pip=0.01)
        # Create many boxes
        for i in range(300):
            b.process_price(100.00 + i * 0.01, f"t{i}")
        assert len(b.state.box_history) <= PnFBuilder.MAX_BOX_HISTORY

    def test_config_name(self):
        b = self._make_builder(box_pips=5, reversal=3)
        assert b.config_name == "5pip_rev3"
        b2 = self._make_builder(box_pips=15, reversal=2)
        assert b2.config_name == "15pip_rev2"


class TestZigzagSR:
    """H1 zigzag S/R must match live_neat_pf_unified.py _update_zigzag() exactly."""

    def test_first_bar_sets_sr(self):
        zz = ZigzagSR(min_swing=0.051)
        zz.update_from_h1_bar({"open": 184.0, "high": 184.5, "low": 183.5, "close": 184.3})
        assert zz.state.support == 183.5
        assert zz.state.resistance == 184.5

    def test_swing_detection(self):
        zz = ZigzagSR(min_swing=0.10)
        # Bar 1: establish range
        zz.update_from_h1_bar({"open": 100, "high": 100.2, "low": 99.8, "close": 100.1})
        # Bar 2: strong move up
        zz.update_from_h1_bar({"open": 100.1, "high": 100.5, "low": 100.0, "close": 100.4})
        # Bar 3: strong reversal down — should trigger zigzag
        zz.update_from_h1_bar({"open": 100.4, "high": 100.4, "low": 100.1, "close": 100.2})
        assert zz.state.zz_direction != 0

    def test_s5_accumulation_to_h1(self):
        """S5 bars should accumulate into H1 and trigger zigzag on hour boundary."""
        from datetime import datetime
        zz = ZigzagSR(min_swing=0.05)

        # 720 S5 bars = 1 hour (but we just need 2 hours worth)
        # Simulate hour 10
        for i in range(10):
            bar = {"timestamp": datetime(2026, 3, 25, 10, 0, i * 5),
                   "open": 184.0, "high": 184.0 + i * 0.01,
                   "low": 184.0 - 0.01, "close": 184.0 + i * 0.01}
            result = zz.accumulate_s5(bar)
            assert result is None  # Same hour, no completed bar

        # Cross into hour 11
        bar = {"timestamp": datetime(2026, 3, 25, 11, 0, 0),
               "open": 184.1, "high": 184.2, "low": 184.0, "close": 184.15}
        result = zz.accumulate_s5(bar)
        assert result is not None  # Completed H1 bar returned


class TestATR:
    """ATR(14) Wilder smoothing."""

    def test_atr_warmup(self):
        atr = ATR(period=14)
        for i in range(14):
            atr.update({"high": 100 + i, "low": 99 + i, "close": 99.5 + i})
        assert atr.value > 0

    def test_atr_first_bar(self):
        atr = ATR(period=14)
        val = atr.update({"high": 101, "low": 99, "close": 100})
        assert atr._count == 1
        assert atr.prev_close == 100

    def test_atr_wilder_smoothing(self):
        """After warmup, ATR uses Wilder smoothing: (prev × 13 + tr) / 14"""
        atr = ATR(period=3)
        atr.update({"high": 102, "low": 98, "close": 100})   # tr=4
        atr.update({"high": 103, "low": 99, "close": 101})   # tr=4
        atr.update({"high": 104, "low": 100, "close": 102})  # tr=4, ATR=4.0
        assert atr.value == pytest.approx(4.0, abs=0.1)
        # Next bar with different range
        atr.update({"high": 103, "low": 101, "close": 102})  # tr=2
        # Wilder: (4.0 × 2 + 2) / 3 = 3.33
        assert atr.value == pytest.approx(3.33, abs=0.1)


class TestMTFMC:
    """Multi-timeframe momentum consistency."""

    def test_returns_zeros_with_insufficient_data(self):
        mc = MTFMC(pip=0.01)
        for i in range(50):
            mc.append_s5(184.0 + i * 0.001)
        d, dd = mc.compute()
        assert d == 0.0
        assert dd == 0.0

    def test_returns_values_with_enough_data(self):
        mc = MTFMC(pip=0.01)
        # 200 trending S5 bars
        for i in range(200):
            mc.append_s5(184.0 + i * 0.01)
        d, dd = mc.compute()
        assert d != 0.0  # Should have some momentum signal

    def test_buffer_trimmed(self):
        mc = MTFMC(pip=0.01)
        for i in range(6000):
            mc.append_s5(184.0)
        assert len(mc.s5_buffer) <= MTFMC.MAX_BUFFER


class TestAsianRange:
    """Asian session range from H1 bars."""

    def test_range_computed_at_london_open(self):
        ar = AsianRange()
        # Asian hours (0-6)
        for h in range(7):
            ar.update_from_h1(
                {"open": 100, "high": 100 + h, "low": 99 - h, "close": 100},
                bar_hour=h
            )
        # London open (hour 7) — should finalize
        ar.update_from_h1({"open": 100, "high": 101, "low": 99, "close": 100}, bar_hour=7)
        assert ar.high == 106  # max(100+0, ..., 100+6)
        assert ar.low == 93    # min(99-0, ..., 99-6)
        assert ar.mid == pytest.approx((106 + 93) / 2)

    def test_no_range_before_london(self):
        ar = AsianRange()
        ar.update_from_h1({"open": 100, "high": 101, "low": 99, "close": 100}, bar_hour=3)
        assert ar.high == 0.0  # Not finalized yet
