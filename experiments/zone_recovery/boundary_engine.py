"""
Boundary-entry zone recovery engine — break-even sizing (Roni cBot logic).

Geometry:
  LONG entry:  entry_price = upper_zone; lower_zone = entry - zone_width
               upper_target = entry + target_beyond; lower_target = lower_zone - target_beyond
  SHORT entry: entry_price = lower_zone; upper_zone = entry + zone_width
               lower_target = entry - target_beyond; upper_target = upper_zone + target_beyond

Sizing (replaces convex):
  Upper crossing → add LONG only if net P&L at upper_target < 0.
  Lower crossing → add SHORT only if net P&L at lower_target < 0.
  Volume = ceil(-net_pips / target_pips * profit_factor), min 1.
  If already profitable at that target → skip (no rebalancing needed).
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from engine import Leg, CycleResult, get_pl_pips_at_target, PIP


def _net_at_target(legs: List[Leg], target: float, pip: float, spread: float) -> float:
    gross = get_pl_pips_at_target(legs, target, pip)
    cost  = sum(l.volume for l in legs) * spread
    return gross - cost


def _breakeven_vol(legs: List[Leg], target: float, pip: float,
                   spread: float, target_pips: float, profit_factor: float) -> float:
    net = _net_at_target(legs, target, pip, spread)
    if net >= 0:
        return 0.0
    return max(1.0, math.ceil(-net / target_pips * profit_factor))


class BoundaryZoneEngine:
    """Zone recovery with boundary-entry geometry and break-even sizing."""

    def __init__(
        self,
        zone_width_pips: float = 35.0,
        target_beyond_pips: float = 60.0,
        max_legs: int = 10,
        profit_factor: float = 1.19,
        spread_pips: float = 1.4,
        pip_size: float = PIP,
        # legacy params accepted but ignored (for sweep compatibility)
        convex_exponent: float = 1.5,
        max_volume_per_leg: float = 32.0,
    ):
        self.zone_width_pips   = zone_width_pips
        self.target_beyond_pips = target_beyond_pips
        self.max_legs          = max_legs
        self.profit_factor     = profit_factor
        self.spread_pips       = spread_pips
        self.pip_size          = pip_size

    def simulate_on_ohlc(
        self,
        open_arr, high_arr, low_arr, close_arr,
        atr_short=None, atr_long=None,
        rng=None,
        entry_mask=None,
        direction_signal=None,
    ) -> List[CycleResult]:

        if rng is None:
            rng = np.random.RandomState(42)

        n       = len(close_arr)
        pip     = self.pip_size
        zone_w  = self.zone_width_pips * pip
        tgt_b   = self.target_beyond_pips * pip
        tgt_p   = self.target_beyond_pips   # pips (scalar)
        spread  = self.spread_pips
        pf      = self.profit_factor
        results = []

        i = 0
        while i < n:
            if entry_mask is not None and not entry_mask[i]:
                i += 1
                continue

            entry_price = close_arr[i]
            direction   = int(direction_signal[i]) if direction_signal is not None \
                          else rng.choice([-1, 1])

            if direction == 1:
                upper_zone   = entry_price
                lower_zone   = entry_price - zone_w
                upper_target = entry_price + tgt_b
                lower_target = lower_zone  - tgt_b
            else:
                lower_zone   = entry_price
                upper_zone   = entry_price + zone_w
                lower_target = entry_price - tgt_b
                upper_target = upper_zone  + tgt_b

            first_leg = Leg(bar_idx=i, price=entry_price, direction=direction, volume=1.0)
            all_legs: List[Leg] = [first_leg]
            long_legs  = [first_leg] if direction == 1 else []
            short_legs = [first_leg] if direction == -1 else []

            entry_bar = i
            worst_dd  = 0.0
            exit_reason = "eod"
            exit_bar    = i
            exit_price  = entry_price
            last_zone_crossed = last_zone_bar = None

            i += 1
            closed = False

            while i < n and not closed:
                bar_high  = high_arr[i]
                bar_low   = low_arr[i]
                bar_close = close_arr[i]
                bullish   = bar_close >= open_arr[i]

                basket = get_pl_pips_at_target(all_legs, bar_close, pip)
                if basket < worst_dd:
                    worst_dd = basket

                seq = [(bar_high, True), (bar_low, False)] if bullish \
                      else [(bar_low, False), (bar_high, True)]

                for extreme, is_high in seq:
                    if closed:
                        break

                    # Target exits
                    if is_high and bar_high >= upper_target:
                        exit_reason, exit_price, exit_bar = "target", upper_target, i
                        closed = True; break
                    if not is_high and bar_low <= lower_target:
                        exit_reason, exit_price, exit_bar = "target", lower_target, i
                        closed = True; break

                    # Upper boundary → add LONG only if adverse at upper target
                    if is_high and bar_high >= upper_zone:
                        if not (last_zone_crossed == "upper" and last_zone_bar == i):
                            last_zone_crossed, last_zone_bar = "upper", i
                            vol = _breakeven_vol(all_legs, upper_target, pip, spread, tgt_p, pf)
                            if vol > 0:
                                if len(all_legs) >= self.max_legs:
                                    exit_reason, exit_price, exit_bar = "max_legs", bar_close, i
                                    closed = True; break
                                leg = Leg(i, upper_zone, 1, vol)
                                all_legs.append(leg); long_legs.append(leg)

                    # Lower boundary → add SHORT only if adverse at lower target
                    if not is_high and bar_low <= lower_zone:
                        if not (last_zone_crossed == "lower" and last_zone_bar == i):
                            last_zone_crossed, last_zone_bar = "lower", i
                            vol = _breakeven_vol(all_legs, lower_target, pip, spread, tgt_p, pf)
                            if vol > 0:
                                if len(all_legs) >= self.max_legs:
                                    exit_reason, exit_price, exit_bar = "max_legs", bar_close, i
                                    closed = True; break
                                leg = Leg(i, lower_zone, -1, vol)
                                all_legs.append(leg); short_legs.append(leg)

                if not closed:
                    i += 1

            if not all_legs:
                continue

            gross = get_pl_pips_at_target(all_legs, exit_price, pip)
            spread_cost = sum(l.volume for l in all_legs) * spread
            net = gross - spread_cost

            results.append(CycleResult(
                entry_bar=entry_bar, exit_bar=exit_bar,
                entry_price=entry_price, exit_price=exit_price,
                legs=[{"bar": l.bar_idx, "price": l.price,
                       "dir": l.direction, "vol": l.volume} for l in all_legs],
                net_pnl_pips=net, gross_pnl_pips=gross,
                max_drawdown_pips=worst_dd,
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
