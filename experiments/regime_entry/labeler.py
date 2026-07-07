"""Triple-barrier labeler on mid-price S5 path (master doc §11, §24.1).

Barriers in absolute pips from entry_price. SL checked before TP within each
S5 bar (conservative, PREREGISTRATION.md). Timeout exits at the last close of
the horizon slice. Gross mid-price P&L; costs are a separate layer (§24.1).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def label_trade(highs, lows, closes, start, end, entry_price, direction,
                tp_pips, sl_pips, pip):
    if end <= start:
        return -9, np.nan, 0, np.nan, np.nan
    if direction == 1:
        tp = entry_price + tp_pips * pip
        sl = entry_price - sl_pips * pip
    else:
        tp = entry_price - tp_pips * pip
        sl = entry_price + sl_pips * pip
    mfe = 0.0
    mae = 0.0
    for i in range(start, end):
        fav = (highs[i] - entry_price) / pip if direction == 1 else (entry_price - lows[i]) / pip
        adv = (entry_price - lows[i]) / pip if direction == 1 else (highs[i] - entry_price) / pip
        if fav > mfe: mfe = fav
        if adv > mae: mae = adv
        if direction == 1:
            if lows[i] <= sl:
                return -1, -sl_pips, i - start + 1, mfe, mae
            if highs[i] >= tp:
                return 1, tp_pips, i - start + 1, mfe, mae
        else:
            if highs[i] >= sl:
                return -1, -sl_pips, i - start + 1, mfe, mae
            if lows[i] <= tp:
                return 1, tp_pips, i - start + 1, mfe, mae
    exit_pips = (closes[end - 1] - entry_price) / pip * direction
    return 0, exit_pips, end - start, mfe, mae
