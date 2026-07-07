"""Tests for ZMQ message protocol — topic construction and serialization."""

import pytest
from lib.zmq_protocol import make_topic


class TestTopicConstruction:
    """Topic strings must be deterministic for subscriber filtering."""

    def test_candle_topic(self):
        assert make_topic("candle", "EUR_JPY", "S5") == "candle.EUR_JPY.S5"

    def test_pnf_box_topic(self):
        assert make_topic("pnf_box", "EUR_JPY", config="5pip_rev3") == "pnf_box.EUR_JPY.5pip_rev3"

    def test_kalman_topic(self):
        assert make_topic("kalman", granularity="H1") == "kalman.H1"

    def test_allocation_topic(self):
        assert make_topic("allocation") == "allocation"

    def test_indicator_topic(self):
        assert make_topic("indicator", "GBP_USD") == "indicator.GBP_USD"

    def test_heartbeat_topic(self):
        assert make_topic("heartbeat") == "heartbeat"
