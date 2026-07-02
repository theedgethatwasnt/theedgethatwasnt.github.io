"""
ontick_engine.py -- deterministic incremental rolling-window engine (tick-time).

CLEAN ROOM. Does NOT import or reference the bb_fade family.

A RollingWindow maintains, over the trailing `window_seconds` of TICK-TIME
(driven by tick.time only -- never wall-clock), per tick (t_ms, mid):

  - rolling mean and std via running sum + sumsq (add new, subtract evicted).
        std = sqrt(max(0, sumsq/n - mean**2))                  O(1) per tick
  - rolling high / low via monotonic deques (sliding-window max/min)  O(1) amortized
  - open  = mid of the OLDEST in-window tick (front of the raw deque)
  - close = latest mid (most recent tick)
  - bands = mean +/- K*std

The window holds every tick whose timestamp is within (t_ms - window_seconds*1000, t_ms].
Eviction is by tick-time of the newest tick, so warmup/replay is fully deterministic
and independent of arrival wall-clock.

The engine is K-agnostic for the raw state; bands at a given K are a pure function
of (mean, std), exposed via band_at(K) so one engine can serve multiple K in a sweep
deterministically (state is identical regardless of K).
"""
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class WindowState:
    n: int
    mean: float
    std: float
    high: float
    low: float
    open: float
    close: float
    t_ms: int

    def band(self, K: float):
        """Upper/lower band at multiplier K. Pure function of mean,std."""
        return self.mean + K * self.std, self.mean - K * self.std


class RollingWindow:
    """Incremental rolling window over trailing `window_seconds` of tick-time."""

    def __init__(self, window_seconds: float):
        self.window_ms = int(round(window_seconds * 1000))
        # raw tick storage (ordered oldest -> newest): (t_ms, mid)
        self._ticks = deque()
        self._sum = 0.0
        self._sumsq = 0.0
        # monotonic deques store indices into a running counter so we can
        # evict from the front by matching the evicted tick's seq.
        # We store (seq, value); decreasing deque for max, increasing for min.
        self._maxdq = deque()  # (seq, mid) decreasing mid
        self._mindq = deque()  # (seq, mid) increasing mid
        self._seq = 0          # monotonically increasing id per tick added
        # parallel seq stored with each raw tick for eviction matching
        self._seqs = deque()

    def update(self, t_ms: int, mid: float) -> WindowState:
        seq = self._seq
        self._seq += 1

        # --- add new tick ---
        self._ticks.append((t_ms, mid))
        self._seqs.append(seq)
        self._sum += mid
        self._sumsq += mid * mid

        # monotonic max deque (strictly decreasing values from front)
        while self._maxdq and self._maxdq[-1][1] <= mid:
            self._maxdq.pop()
        self._maxdq.append((seq, mid))
        # monotonic min deque (strictly increasing values from front)
        while self._mindq and self._mindq[-1][1] >= mid:
            self._mindq.pop()
        self._mindq.append((seq, mid))

        # --- evict ticks older than window (by tick-time of newest tick) ---
        cutoff = t_ms - self.window_ms
        while self._ticks and self._ticks[0][0] <= cutoff:
            old_t, old_mid = self._ticks.popleft()
            old_seq = self._seqs.popleft()
            self._sum -= old_mid
            self._sumsq -= old_mid * old_mid
            # pop from monotonic deques if their front is this evicted seq
            if self._maxdq and self._maxdq[0][0] == old_seq:
                self._maxdq.popleft()
            if self._mindq and self._mindq[0][0] == old_seq:
                self._mindq.popleft()

        n = len(self._ticks)
        if n == 0:
            # should not happen (we just added one) but guard
            return WindowState(0, mid, 0.0, mid, mid, mid, mid, t_ms)

        mean = self._sum / n
        var = self._sumsq / n - mean * mean
        std = math.sqrt(var) if var > 0.0 else 0.0
        high = self._maxdq[0][1]
        low = self._mindq[0][1]
        open_ = self._ticks[0][1]
        close = self._ticks[-1][1]
        return WindowState(n, mean, std, high, low, open_, close, t_ms)
