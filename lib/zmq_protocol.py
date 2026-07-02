"""
ZMQ message protocol — defines all message types exchanged between containers.

All messages are msgpack-encoded dicts with a mandatory 'type' field.
Published as multipart ZMQ frames: [topic_bytes, msgpack_bytes].

Socket topology:
  Curator   --PUB--> ipc:///tmp/zmq/market.ipc      (candles, indicators, P&F boxes, Kalman)
  Portfolio --PUB--> ipc:///tmp/zmq/allocation.ipc   (allocation weights, margin budgets)
  Strategies --PUB--> ipc:///tmp/zmq/trades.ipc      (trade open/close events)
  Curator   --REP--> ipc:///tmp/zmq/health.ipc       (ping/pong for Docker healthcheck)
"""

import msgpack
import zmq
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any


# ─── Socket Paths ──────────────────────────────────────────────────────────

MARKET_PUB = "ipc:///tmp/zmq/market.ipc"
ALLOCATION_PUB = "ipc:///tmp/zmq/allocation.ipc"
TRADES_PUB = "ipc:///tmp/zmq/trades.ipc"
TRADES_DB_PULL = "ipc:///tmp/zmq/trades_db.ipc"   # portfolio_mgr PULL (single DB writer)
HEALTH_REP = "ipc:///tmp/zmq/health.ipc"

# ZMQ high-water mark (max queued messages before dropping)
HWM = 10000


# ─── Message Types ─────────────────────────────────────────────────────────

# Curator publishes
MSG_CANDLE = "candle"
MSG_PNF_BOX = "pnf_box"
MSG_INDICATOR = "indicator"
MSG_KALMAN = "kalman"
MSG_HEARTBEAT = "heartbeat"

# Portfolio manager publishes
MSG_ALLOCATION = "allocation"
MSG_ACCOUNT = "account"
MSG_RISK_OVERLAY = "risk_overlay"
MSG_PORTFOLIO_STATE = "portfolio_state"

# Strategies publish
MSG_TRADE_EVENT = "trade_event"


# ─── Topic Construction ───────────────────────────────────────────────────

def make_topic(msg_type: str, pair: Optional[str] = None,
               granularity: Optional[str] = None,
               config: Optional[str] = None) -> str:
    """Build a ZMQ topic string for filtering.

    Examples:
        make_topic("candle", "EUR_JPY", "S5")  -> "candle.EUR_JPY.S5"
        make_topic("pnf_box", "EUR_JPY", config="5pip_rev3") -> "pnf_box.EUR_JPY.5pip_rev3"
        make_topic("kalman", granularity="H1") -> "kalman.H1"
        make_topic("allocation") -> "allocation"
    """
    parts = [msg_type]
    if pair:
        parts.append(pair)
    if granularity:
        parts.append(granularity)
    if config:
        parts.append(config)
    return ".".join(parts)


# ─── Publisher / Subscriber Helpers ────────────────────────────────────────

class Publisher:
    """ZMQ PUB socket wrapper with msgpack serialization."""

    def __init__(self, endpoint: str, context: Optional[zmq.Context] = None):
        self.ctx = context or zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, HWM)
        # Ensure IPC directory exists; clean up stale socket
        if endpoint.startswith("ipc://"):
            import os
            ipc_path = endpoint.replace("ipc://", "")
            os.makedirs(os.path.dirname(ipc_path), exist_ok=True)
            if os.path.exists(ipc_path):
                os.unlink(ipc_path)
        self.socket.bind(endpoint)
        self._lock = threading.Lock()   # ZMQ sockets are NOT thread-safe; serialize concurrent publishes

    def publish(self, topic: str, message: dict):
        """Publish a message with topic prefix for subscriber filtering.

        Thread-safe: the curator publishes from both its stream thread and its poll loop on the same
        socket. Without this lock, concurrent send_multipart calls interleave frames and produce
        garbled multi-topic messages that subscribers can't unpack (ExtraData)."""
        payload = msgpack.packb(message, use_bin_type=True)
        with self._lock:
            self.socket.send_multipart([topic.encode(), payload])

    def close(self):
        self.socket.close()


class Subscriber:
    """ZMQ SUB socket wrapper with msgpack deserialization."""

    def __init__(self, endpoint: str, topics: Optional[List[str]] = None,
                 context: Optional[zmq.Context] = None):
        self.ctx = context or zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, HWM)
        self.socket.connect(endpoint)

        # Subscribe to specified topics, or all if none given
        if topics:
            for t in topics:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, t)
        else:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def receive(self, timeout_ms: int = -1) -> Optional[tuple]:
        """Receive a message. Returns (topic_str, message_dict) or None on timeout.

        Args:
            timeout_ms: -1 for blocking, 0 for non-blocking, >0 for timeout in ms.
        """
        if timeout_ms >= 0:
            if not self.socket.poll(timeout_ms):
                return None

        frames = self.socket.recv_multipart()
        if len(frames) < 2:
            return None
        topic = frames[0].decode()
        message = msgpack.unpackb(frames[1], raw=False)
        return topic, message

    def close(self):
        self.socket.close()


class HealthResponder:
    """REP socket for Docker healthcheck (ping/pong)."""

    def __init__(self, endpoint: str = HEALTH_REP,
                 context: Optional[zmq.Context] = None):
        self.ctx = context or zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.REP)
        if endpoint.startswith("ipc://"):
            import os
            ipc_path = endpoint.replace("ipc://", "")
            os.makedirs(os.path.dirname(ipc_path), exist_ok=True)
            if os.path.exists(ipc_path):
                os.unlink(ipc_path)
        self.socket.bind(endpoint)

    def check_and_respond(self, timeout_ms: int = 100) -> bool:
        """Non-blocking: check for ping, respond with pong. Returns True if handled."""
        if self.socket.poll(timeout_ms):
            msg = self.socket.recv()
            self.socket.send(b"pong")
            return True
        return False

    def close(self):
        self.socket.close()
