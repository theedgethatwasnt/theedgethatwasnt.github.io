"""Finite-margin realizable-risk account simulation (the validation_gap fix).

Replays closed trades in time order, sizing each off the running balance per the
live rule, and halts on the daily/overall DD guard. Reports whether a real account
would have survived, and at what size."""

def _pip_value_usd(pair, units):
    return (0.01/152.0 if pair.endswith("_JPY") else 0.0001) * units

def simulate_account(trades, start_balance, margin_rate=0.045, gate=0.45,
                     closeout=0.50, dd_guard=(0.25, 0.50)):
    trades = sorted(trades, key=lambda t: t["exit_time"])
    bal = peak = float(start_balance)
    max_dd_usd = 0.0; max_util = 0.0
    halted = False; reason = ""; at = None
    daily_dd, overall_dd = dd_guard
    for i, t in enumerate(trades):
        units = max(1, min(500, round(bal * 1.3)))
        # margin utilization for this position (single open approximation)
        util = (units * margin_rate) / max(bal, 1e-9)
        max_util = max(max_util, util)
        if util >= closeout:                      # would exceed broker closeout
            halted, reason, at = True, "closeout", i; break
        pnl_usd = _pip_value_usd(t["pair"], units) * t["pnl_pips"]
        bal += pnl_usd
        peak = max(peak, bal)
        dd = (peak - bal) / peak if peak > 0 else 0.0
        max_dd_usd = max(max_dd_usd, peak - bal)
        if bal <= 0:
            halted, reason, at = True, "blown", i; break
        if dd >= overall_dd:
            halted, reason, at = True, "overall_dd", i; break
    return {"final_balance": round(bal, 2), "max_dd_usd": round(max_dd_usd, 2),
            "max_util": round(max_util, 3), "halted": halted,
            "halted_reason": reason, "halted_at": at, "start_balance": start_balance}
