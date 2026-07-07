import numpy as np
from labeler import label_trade

PIP = 1e-4

def _mk(n, base=1.1000):
    h = np.full(n, base); l = np.full(n, base); c = np.full(n, base)
    return h, l, c

def test_tp_hit_long():
    h, l, c = _mk(10)
    h[3] = 1.1000 + 4 * PIP                       # TP=3.2 touched at bar 3
    lab, ex, held, mfe, mae = label_trade(h, l, c, 0, 10, 1.1000, 1, 3.2, 6.4, PIP)
    assert lab == 1 and abs(ex - 3.2) < 1e-9 and held == 4

def test_sl_hit_short():
    h, l, c = _mk(10)
    h[2] = 1.1000 + 7 * PIP                       # short SL=6.4 above entry
    lab, ex, *_ = label_trade(h, l, c, 0, 10, 1.1000, -1, 3.2, 6.4, PIP)
    assert lab == -1 and abs(ex + 6.4) < 1e-9

def test_same_bar_ambiguity_sl_first():
    h, l, c = _mk(10)
    h[1] = 1.1000 + 4 * PIP                       # both barriers inside bar 1
    l[1] = 1.1000 - 7 * PIP
    lab, ex, *_ = label_trade(h, l, c, 0, 10, 1.1000, 1, 3.2, 6.4, PIP)
    assert lab == -1                              # conservative: SL first

def test_timeout_exits_at_last_close():
    h, l, c = _mk(10)
    c[9] = 1.1000 + 1 * PIP
    lab, ex, held, *_ = label_trade(h, l, c, 0, 10, 1.1000, 1, 3.2, 6.4, PIP)
    assert lab == 0 and abs(ex - 1.0) < 1e-9 and held == 10

def test_empty_slice_returns_sentinel():
    h, l, c = _mk(5)
    lab, *_ = label_trade(h, l, c, 3, 3, 1.1000, 1, 3.2, 6.4, PIP)
    assert lab == -9

def test_random_walk_recovers_667_baseline():
    """Gambler's ruin: P(TP first)=SL/(TP+SL)=2/3 at TP:SL=1:2, zero cost."""
    rng = np.random.default_rng(42)
    wins = 0; dec = 0
    for _ in range(4000):
        steps = rng.normal(0, 0.8 * PIP, 3000)    # fine steps, barriers resolve
        c = 1.1000 + np.cumsum(steps)
        h = c; l = c
        lab, *_ = label_trade(h, l, c, 0, 3000, 1.1000, 1, 3.2, 6.4, PIP)
        if lab == 1: wins += 1; dec += 1
        elif lab == -1: dec += 1
    wr = wins / dec
    assert abs(wr - 2/3) < 0.03, wr
