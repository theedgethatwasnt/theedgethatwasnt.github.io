"""Incremental (O(1) per bar) TopsBots swing detector.

Implements the same 3-stage algorithm as topsbots_swings() in lib/swing_indicators.py
but processes one bar at a time with O(1) update cost. Output is IDENTICAL to the
batch algorithm given the same chronological bar sequence.

WHY THIS EXISTS
---------------
The batch TopsBots algorithm is history-path-dependent: the confirmed swing set at bar N
depends on the entire history from bar 0. This creates a training/live gap if training
uses full-history batch while live uses a sliding window of recent bars (the IronNet V1
bug — only 47% SBA agreement at 500-bar buffer).

The fix is to run the SAME algorithm in both training and live:
  - Training: walk bars left-to-right through IncrementalTopsBots, record features at
    each step. Produces exactly the features the live system would have seen.
  - Live: continue from the serialized state (no history replay needed).

THE 1-BAR LOOKAHEAD
--------------------
Stage 1 of TopsBots requires: h[i] > h[i-1] AND h[i] > h[i+1].
Bar N-1 can only be confirmed as a Stage 1 swing when bar N closes, giving its h[i+1].
This means the swing state at bar N reflects swings confirmed through bar N-1.

This is strictly MORE CAUSAL than the batch algorithm (which appears to use lookahead
in training), but it's the CORRECT behavior for live trading. Using IncrementalTopsBots
in training ensures the learned representation matches what the live system observes.

ALGORITHM DETAILS
-----------------
Stage 1: local extreme at bar N-1, confirmed when bar N arrives.
  HIGH: h[N-1] > h[N-2] AND h[N-1] > h[N]
  LOW:  l[N-1] < l[N-2] AND l[N-1] < l[N]

Stage 2: alternation — consecutive same-type Stage 1 candidates are compressed to
  the best (max for H, min for L). A pending candidate is finalized when an
  opposite-type candidate arrives.

Stage 3: exceeding-extremes gate.
  H accepted if: lh is None OR val > lh OR glh (gate opened by previous L)
  L accepted if: ll is None OR val < ll OR glh2 (gate opened by previous H)
  Accepting H: sets glh=False, glh2=True, updates act_h
  Accepting L: sets glh2=False, glh=True, updates act_l

STATE ENCODING (5-state SBA)
-----------------------------
  +2: mid > act_h (breakout above HSP)
  +1: act_l <= mid <= act_h, last confirmed swing was LSP (ascending context)
   0: no confirmed swings yet
  -1: act_l <= mid <= act_h, last confirmed swing was HSP (descending context)
  -2: mid < act_l (breakout below LSP)

Normalized SBA = state * 0.5 → {-1, -0.5, 0, +0.5, +1}

SERIALIZATION
-------------
to_dict() / from_dict(d) round-trip the complete state to a JSON-compatible dict.
No bar history is stored — the compact state (14 scalar fields) is all that's needed
to resume O(1) updates without replaying any bars.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple


class IncrementalTopsBots:
    """O(1)-per-bar TopsBots swing detector with serializable state.

    Usage::

        tb = IncrementalTopsBots()
        for h, l, mid in bars:
            state, erp, act_h, act_l = tb.update(h, l, mid)

        # Serialize after training to seed the live system:
        state_dict = tb.to_dict()

        # Restore in live service (no history replay needed):
        tb = IncrementalTopsBots.from_dict(state_dict)
        state, erp, act_h, act_l = tb.update(next_h, next_l, next_mid)

    Parameters for update():
        h   : high of the arriving bar
        l   : low  of the arriving bar
        mid : midpoint / close, used only for state encoding (not swing detection)

    Returns:
        state : int {-2, -1, 0, +1, +2}
        erp   : float, (mid - act_l) / (act_h - act_l), unclamped, or NaN
        act_h : float or None — last confirmed HSP level
        act_l : float or None — last confirmed LSP level
    """

    __slots__ = (
        '_h1', '_l1', '_h2', '_l2',
        '_pend_type', '_pend_val', '_pend_idx',
        '_lh', '_ll', '_glh', '_glh2',
        '_act_h', '_act_l', '_cur_last', '_n_bars',
    )

    def __init__(self) -> None:
        # ── 1-bar lookahead buffer ────────────────────────────────────────────
        # _h1/_l1 = bar N-1 (pending Stage 1 check)
        # _h2/_l2 = bar N-2 (left-neighbour for Stage 1)
        self._h1: Optional[float] = None
        self._l1: Optional[float] = None
        self._h2: Optional[float] = None
        self._l2: Optional[float] = None

        # ── Stage 2: pending candidate ────────────────────────────────────────
        # At most one Stage 2 candidate is live at any time.
        # Finalized when an opposite-type Stage 1 extreme arrives.
        self._pend_type: Optional[str] = None   # 'H' | 'L' | None
        self._pend_val:  Optional[float] = None
        self._pend_idx:  Optional[int]   = None

        # ── Stage 3: exceeding-extremes gate ─────────────────────────────────
        self._lh:   Optional[float] = None   # last confirmed HSP value
        self._ll:   Optional[float] = None   # last confirmed LSP value
        self._glh:  bool = False             # gate: True → next H acceptance unlocked (set by L)
        self._glh2: bool = False             # gate: True → next L acceptance unlocked (set by H)

        # ── Active S/R levels (last Stage 3 output) ───────────────────────────
        self._act_h:    Optional[float] = None
        self._act_l:    Optional[float] = None
        self._cur_last: int = 0   # +1 = last was LSP (ascending), -1 = last was HSP

        self._n_bars: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self, h: float, l: float, mid: float
    ) -> Tuple[int, float, Optional[float], Optional[float]]:
        """Process one incoming bar. Returns (state, erp, act_h, act_l)."""
        # Check bar N-1 as a Stage 1 candidate (now that bar N has arrived)
        if self._h1 is not None and self._h2 is not None:
            if self._h1 > self._h2 and self._h1 > h:
                self._s1_candidate(self._n_bars - 1, 'H', self._h1)
            if self._l1 < self._l2 and self._l1 < l:
                self._s1_candidate(self._n_bars - 1, 'L', self._l1)

        # Advance lookahead buffer
        self._h2, self._l2 = self._h1, self._l1
        self._h1, self._l1 = h, l
        self._n_bars += 1

        return (*self._encode(mid), self._act_h, self._act_l)

    def finalize(self) -> None:
        """Flush any pending Stage 2 candidate through Stage 3.

        Call after the last bar in a training corpus to ensure the final
        pending swing is accepted (if it passes Stage 3). This matches the
        batch behaviour where topsbots_swings() processes every raw candidate.

        In live (infinite stream) there is no final bar, so this is a no-op.
        """
        self._flush_pending()

    @property
    def act_h(self) -> Optional[float]:
        """Most recent confirmed HSP (active resistance level)."""
        return self._act_h

    @property
    def act_l(self) -> Optional[float]:
        """Most recent confirmed LSP (active support level)."""
        return self._act_l

    # ── Internals ─────────────────────────────────────────────────────────────

    def _s1_candidate(self, idx: int, typ: str, val: float) -> None:
        """Route a Stage 1 extreme through Stage 2 → Stage 3 logic."""
        if self._pend_type is None:
            self._pend_type, self._pend_val, self._pend_idx = typ, val, idx
        elif self._pend_type == typ:
            # Same-type run: Stage 2 compression — keep the more extreme value
            if (typ == 'H' and val > self._pend_val) or \
               (typ == 'L' and val < self._pend_val):
                self._pend_val, self._pend_idx = val, idx
        else:
            # New type: finalize current pending, start fresh
            self._flush_pending()
            self._pend_type, self._pend_val, self._pend_idx = typ, val, idx

    def _flush_pending(self) -> None:
        """Pass the pending Stage 2 candidate through Stage 3 gate."""
        if self._pend_type is None:
            return
        typ, val = self._pend_type, self._pend_val
        if typ == 'H':
            if self._lh is None or val > self._lh or self._glh:
                self._lh   = val
                self._act_h = val
                self._cur_last = -1
                self._glh  = False
                self._glh2 = True
        else:  # 'L'
            if self._ll is None or val < self._ll or self._glh2:
                self._ll   = val
                self._act_l = val
                self._cur_last = 1
                self._glh2 = False
                self._glh  = True
        self._pend_type = self._pend_val = self._pend_idx = None

    def _encode(self, mid: float) -> Tuple[int, float]:
        """Map mid vs active levels to (state, erp). Matches compute_swing_features()."""
        ah, al = self._act_h, self._act_l
        if ah is not None and al is not None:
            rng = ah - al
            if mid > ah:
                return 2, (mid - al) / rng if rng > 0 else 1.0
            if mid < al:
                return -2, (mid - al) / rng if rng > 0 else 0.0
            return self._cur_last, (mid - al) / rng if rng > 0 else 0.5
        if ah is not None:
            return (-1 if mid <= ah else 2), math.nan
        if al is not None:
            return (1 if mid >= al else -2), math.nan
        return 0, math.nan

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize complete state to a JSON-compatible dict (14 scalar fields).

        This compact state is sufficient to resume O(1) updates on the next bar
        without replaying any history. Suitable for storage in DuckDB or JSON files.

        Round-trip guarantee: IncrementalTopsBots.from_dict(tb.to_dict()) produces
        an instance that generates byte-identical output on subsequent bars.
        """
        return {
            'h1': self._h1, 'l1': self._l1,
            'h2': self._h2, 'l2': self._l2,
            'pend_type': self._pend_type,
            'pend_val':  self._pend_val,
            'pend_idx':  self._pend_idx,
            'lh': self._lh, 'll': self._ll,
            'glh': self._glh, 'glh2': self._glh2,
            'act_h': self._act_h, 'act_l': self._act_l,
            'cur_last': self._cur_last,
            'n_bars': self._n_bars,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'IncrementalTopsBots':
        """Restore from a dict produced by to_dict(). No bar replay required."""
        obj = cls.__new__(cls)
        obj._h1 = d['h1'];         obj._l1 = d['l1']
        obj._h2 = d['h2'];         obj._l2 = d['l2']
        obj._pend_type = d['pend_type']
        obj._pend_val  = d['pend_val']
        obj._pend_idx  = d['pend_idx']
        obj._lh   = d['lh'];       obj._ll   = d['ll']
        obj._glh  = d['glh'];      obj._glh2 = d['glh2']
        obj._act_h = d['act_h'];   obj._act_l = d['act_l']
        obj._cur_last = d['cur_last']
        obj._n_bars   = d['n_bars']
        return obj

    def __repr__(self) -> str:
        return (
            f"IncrementalTopsBots(n={self._n_bars}, "
            f"act_h={self._act_h}, act_l={self._act_l}, "
            f"pend={self._pend_type}/{self._pend_val})"
        )
