"""
Zone Recovery Engine — Python port of Roni_cBot_Prod.cs
Faithful translation + ATR-calibrated extension.

Split-account design: long legs tracked in account_long, short legs in account_short.
This mirrors the two-OANDA-account approach required to work around OANDA no-hedging.

Core mechanics (from cBot):
  - Entry: random direction, base_unit volume
  - Zone: entry ± half_zone (locked at cycle start)
  - Targets: zone boundary ± target_beyond
  - PingPong: when price crosses zone boundary, add leg sized to break-even at target
  - Exit: price hits target → close all; or max_legs reached

P&L note: all computations are in NORMALIZED PIPS where 1 pip = 1 base_unit.
E.g., a 2-base_unit leg earning 5 pips contributes 10 normalized pip-units.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


PIP = 0.0001            # EUR_USD pip size (price change per pip)
TICK = 0.00001          # EUR_USD tick size


@dataclass
class Leg:
    bar_idx: int
    price: float
    direction: int      # +1=long, -1=short
    volume: float       # in base units (1.0 = 1 base_unit)


@dataclass
class CycleResult:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    legs: List[dict]
    net_pnl_pips: float        # after spread
    gross_pnl_pips: float      # before spread
    max_drawdown_pips: float   # worst unrealized normalized pips
    duration_bars: int
    exit_reason: str           # "target", "max_legs", "eod", "init_target"
    long_legs: List[dict] = field(default_factory=list)
    short_legs: List[dict] = field(default_factory=list)


def get_pl_pips_at_target(legs: List[Leg], target_price: float, pip_size: float = PIP) -> float:
    """Aggregate P&L in normalized pips for all legs if closed at target_price.

    Normalized pip = pips × volume_in_base_units.
    """
    total = 0.0
    for leg in legs:
        if leg.direction == 1:
            pips2target = (target_price - leg.price) / pip_size
        else:
            pips2target = (leg.price - target_price) / pip_size
        total += pips2target * leg.volume   # volume already in base units
    return total


def get_volume_to_break_even(
    current_pl_pips: float,
    pips_to_target: float,
    profit_factor: float = 1.19,
    min_volume: float = 1.0,
) -> float:
    """Compute base_units needed on next leg to break even (or better) at target.

    Math:
      new_leg_gain = V × pips_to_target
      current_pl_pips + V × pips_to_target = 0  (break-even condition)
      V = -current_pl_pips / pips_to_target × profit_factor
    """
    if current_pl_pips >= 0.0:
        return 0.0
    if pips_to_target <= 0:
        return min_volume
    raw = -current_pl_pips / pips_to_target * profit_factor
    return max(math.ceil(raw), min_volume)


class ZoneRecoveryEngine:
    """Zone recovery strategy engine.

    Two modes:
      mode='classic'  — fixed pips (faithful to Roni_cBot params)
      mode='atr'      — ATR-calibrated zone/target with entry filters
    """

    def __init__(
        self,
        mode: str = "classic",
        # Classic mode (in pips)
        half_zone_pips: float = 10.25,
        target_beyond_pips: float = 6.0,
        init_target_pips: float = 2.7,
        # ATR mode multipliers
        atr_zone_mult: float = 0.5,       # zone half-width = mult × ATR_short
        atr_target_mult: float = 1.5,     # escape target = mult × ATR_long
        ez_ratio_min: float = 6.0,
        ez_ratio_max: float = 15.0,
        spread_min_mult: float = 3.0,     # zone must be > N × spread
        # Common
        base_unit: float = 1.0,           # all volumes relative to this unit
        max_legs: int = 10,
        max_volume_per_leg: float = 32.0, # hard cap: prevents geometric explosion
        sizing_mode: str = "dynamic",     # "dynamic", "convex", "linear"
        convex_exponent: float = 1.5,
        profit_factor: float = 1.19,
        # Costs (pips per round trip per base_unit)
        spread_pips: float = 1.4,
        pip_size: float = PIP,
    ):
        self.mode = mode
        self.half_zone_pips = half_zone_pips
        self.target_beyond_pips = target_beyond_pips
        self.init_target_pips = init_target_pips
        self.atr_zone_mult = atr_zone_mult
        self.atr_target_mult = atr_target_mult
        self.ez_ratio_min = ez_ratio_min
        self.ez_ratio_max = ez_ratio_max
        self.spread_min_mult = spread_min_mult
        self.base_unit = base_unit
        self.max_legs = max_legs
        self.max_volume_per_leg = max_volume_per_leg
        self.sizing_mode = sizing_mode
        self.convex_exponent = convex_exponent
        self.profit_factor = profit_factor
        self.spread_pips = spread_pips
        self.pip_size = pip_size

    def _compute_zone_params(
        self, entry_price: float, atr_short: float = 0.0, atr_long: float = 0.0
    ) -> Optional[tuple]:
        """Return (half_zone_price, target_beyond_price) or None if filters fail."""
        if self.mode == "classic":
            return (
                self.half_zone_pips * self.pip_size,
                self.target_beyond_pips * self.pip_size,
            )

        if atr_short <= 0 or atr_long <= 0:
            return None

        half_zone = self.atr_zone_mult * atr_short
        target_beyond = self.atr_target_mult * atr_long
        zone_width = 2 * half_zone
        ez_ratio = target_beyond / zone_width

        if not (self.ez_ratio_min <= ez_ratio <= self.ez_ratio_max):
            return None

        spread_price = self.spread_pips * self.pip_size
        if zone_width < self.spread_min_mult * spread_price:
            return None

        return half_zone, target_beyond

    def _next_leg_volume(
        self, leg_number: int, all_legs: List[Leg], target_price: float, pips_to_target: float
    ) -> float:
        """Compute volume (in base_units) for the next leg."""
        if self.sizing_mode == "dynamic":
            current_pl = get_pl_pips_at_target(all_legs, target_price, self.pip_size)
            vol = get_volume_to_break_even(current_pl, pips_to_target, self.profit_factor)
            vol = max(vol, 1.0)
        elif self.sizing_mode == "convex":
            vol = max(1.0, leg_number ** self.convex_exponent)
        else:  # linear
            vol = float(leg_number)
        # Cap to prevent geometric explosion on long ping-pong sequences
        return min(vol, self.max_volume_per_leg)

    def simulate_on_ohlc(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        atr_short: np.ndarray = None,
        atr_long: np.ndarray = None,
        rng: np.random.RandomState = None,
        entry_mask: np.ndarray = None,
        direction_signal: np.ndarray = None,
    ) -> List[CycleResult]:
        """Simulate zone recovery on OHLC bars.

        Intra-bar crossing detection using H/L:
          - bullish bar (close >= open): H visited before L
          - bearish bar (close < open):  L visited before H

        entry_mask: boolean array — only start new cycles on True bars (session filter).
        direction_signal: int8 array of +1/-1 — first-leg direction per bar (None = random).

        Returns list of CycleResult, one per completed cycle.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        n = len(close_arr)
        results = []

        if atr_short is None:
            atr_short = np.ones(n)
        if atr_long is None:
            atr_long = np.ones(n)

        pip = self.pip_size
        spread = self.spread_pips

        i = 0
        while i < n:
            if np.isnan(atr_short[i]) or np.isnan(atr_long[i]):
                i += 1
                continue

            if entry_mask is not None and not entry_mask[i]:
                i += 1
                continue

            entry_price = close_arr[i]
            params = self._compute_zone_params(entry_price, atr_short[i], atr_long[i])
            if params is None:
                i += 1
                continue

            half_zone, target_beyond = params
            hz_pips = half_zone / pip
            tb_pips = target_beyond / pip

            # Locked zone geometry
            lower_zone = entry_price - half_zone
            upper_zone = entry_price + half_zone
            upper_target = upper_zone + target_beyond
            lower_target = lower_zone - target_beyond

            if direction_signal is not None:
                direction = int(direction_signal[i])
            else:
                direction = rng.choice([-1, 1])

            # First leg: 1 base_unit
            first_leg = Leg(bar_idx=i, price=entry_price, direction=direction, volume=1.0)
            all_legs: List[Leg] = [first_leg]
            long_legs: List[Leg] = [first_leg] if direction == 1 else []
            short_legs: List[Leg] = [first_leg] if direction == -1 else []

            entry_bar = i
            worst_dd_pips = 0.0
            exit_reason = "eod"
            exit_bar = i
            exit_price = entry_price

            i += 1
            closed = False
            last_zone_crossed = None  # (bar_idx, "upper"|"lower") — prevent double entries

            while i < n and not closed:
                bar_open = open_arr[i]
                bar_high = high_arr[i]
                bar_low = low_arr[i]
                bar_close = close_arr[i]
                bullish = bar_close >= bar_open

                # Track basket P&L at bar close for drawdown
                basket_pips = get_pl_pips_at_target(all_legs, bar_close, pip)
                if basket_pips < worst_dd_pips:
                    worst_dd_pips = basket_pips

                # Initial target: first-leg-only exit (clean profit)
                if len(all_legs) == 1:
                    single_pips = direction * (bar_close - all_legs[0].price) / pip
                    if single_pips >= self.init_target_pips:
                        exit_reason = "init_target"
                        exit_price = bar_close
                        exit_bar = i
                        closed = True
                        break

                # Determine intra-bar visit order: (extreme_price, is_high)
                if bullish:
                    bar_sequence = [(bar_high, True), (bar_low, False)]
                else:
                    bar_sequence = [(bar_low, False), (bar_high, True)]

                for bar_extreme, is_high in bar_sequence:
                    if closed:
                        break

                    # Check target hits
                    if is_high and bar_high >= upper_target:
                        exit_reason = "target"
                        exit_price = upper_target
                        exit_bar = i
                        closed = True
                        break
                    if not is_high and bar_low <= lower_target:
                        exit_reason = "target"
                        exit_price = lower_target
                        exit_bar = i
                        closed = True
                        break

                    # Upper zone crossing → add Buy leg (on long account)
                    if is_high and bar_high >= upper_zone:
                        key = (i, "upper")
                        if key != last_zone_crossed:
                            last_zone_crossed = key
                            if len(all_legs) >= self.max_legs:
                                exit_reason = "max_legs"
                                exit_price = bar_close
                                exit_bar = i
                                closed = True
                                break
                            vol = self._next_leg_volume(
                                len(long_legs) + 1, all_legs, upper_target, tb_pips
                            )
                            new_leg = Leg(bar_idx=i, price=upper_zone, direction=1, volume=vol)
                            all_legs.append(new_leg)
                            long_legs.append(new_leg)

                    # Lower zone crossing → add Sell leg (on short account)
                    if not is_high and bar_low <= lower_zone:
                        key = (i, "lower")
                        if key != last_zone_crossed:
                            last_zone_crossed = key
                            if len(all_legs) >= self.max_legs:
                                exit_reason = "max_legs"
                                exit_price = bar_close
                                exit_bar = i
                                closed = True
                                break
                            vol = self._next_leg_volume(
                                len(short_legs) + 1, all_legs, lower_target, tb_pips
                            )
                            new_leg = Leg(bar_idx=i, price=lower_zone, direction=-1, volume=vol)
                            all_legs.append(new_leg)
                            short_legs.append(new_leg)

                if not closed:
                    i += 1

            # ── Compute cycle results ──────────────────────────────────────
            if not all_legs:
                continue

            ep = exit_price
            gross_pips = get_pl_pips_at_target(all_legs, ep, pip)

            # Spread cost: each leg pays spread on open and close
            total_volume = sum(l.volume for l in all_legs)
            spread_cost_pips = total_volume * spread  # one way (open cost); close is at exit

            net_pips = gross_pips - spread_cost_pips

            results.append(CycleResult(
                entry_bar=entry_bar,
                exit_bar=exit_bar,
                entry_price=entry_price,
                exit_price=ep,
                legs=[{
                    "bar": l.bar_idx, "price": l.price,
                    "dir": l.direction, "vol": l.volume,
                } for l in all_legs],
                net_pnl_pips=net_pips,
                gross_pnl_pips=gross_pips,
                max_drawdown_pips=worst_dd_pips,
                duration_bars=exit_bar - entry_bar,
                exit_reason=exit_reason,
                long_legs=[{"bar": l.bar_idx, "price": l.price, "vol": l.volume}
                           for l in long_legs],
                short_legs=[{"bar": l.bar_idx, "price": l.price, "vol": l.volume}
                            for l in short_legs],
            ))

            if not closed:
                break

        return results
