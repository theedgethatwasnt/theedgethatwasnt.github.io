"""
bb_sizing.py — position sizing for the 4-TF BB re-entry book, per-account (one TF each, acct 001-004).
Risk-parity: size by target $/pip (so JPY pairs scale correctly), NAV-proportional via a DD-target
(worst historical per-TF drawdown should equal DD_FRAC of NAV), 1-unit floor (OANDA min), MAX_UNITS
cap, then the live margin gate (lib/sizing.py, 45% util) refines under concurrency.
Answers: at $5 NAV, how many units? And does it scale to any account size?
"""
import numpy as np

WORST_DD_PIPS={"5min":3574,"15min":1703,"1h":2264,"4h":1483}   # per-TF max historical DD (5.3y)
MAX_UNITS=100000

def pip_value_per_unit(pair, price):
    """USD value of a 1-pip move for 1 unit."""
    if pair.endswith("JPY"): return 0.01/price            # XXX/JPY: pip=0.01, /quote-rate
    if pair=="EUR_GBP":      return 0.0001*1.27            # quote GBP -> *GBPUSD
    return 0.0001                                          # USD-quoted majors

def size(nav, tf, pair, price, dd_frac=0.20, leverage=30):
    """units for one position. dd_frac = fraction of NAV the worst historical DD may consume."""
    dollar_per_pip = dd_frac*nav / WORST_DD_PIPS[tf]       # risk-parity $/pip from DD budget
    units = dollar_per_pip / pip_value_per_unit(pair, price)
    return int(max(1, min(MAX_UNITS, round(units))))

PRICES={"EUR_USD":1.08,"USD_JPY":150.0,"GBP_JPY":190.0}
def main():
    print("BB book sizing — units per position (risk-parity, DD-target). OANDA floor = 1 unit.\n")
    for dd in (0.20,0.10):
        print(f"=== DD tolerance {int(dd*100)}% of NAV (worst historical per-TF DD) ===")
        print(f"  {'NAV':>6} | {'TF':>5} | {'EUR_USD':>8} {'USD_JPY':>8} {'GBP_JPY':>8}  units/position")
        for nav in (5,20,50,100,500):
            for tf in ("5min","15min","1h","4h"):
                eu=size(nav,tf,"EUR_USD",PRICES["EUR_USD"],dd)
                uj=size(nav,tf,"USD_JPY",PRICES["USD_JPY"],dd)
                gj=size(nav,tf,"GBP_JPY",PRICES["GBP_JPY"],dd)
                print(f"  ${nav:>4} | {tf:>5} | {eu:>8} {uj:>8} {gj:>8}")
            print()
    # the $5 question, explicit
    print("ANSWER @ $5 NAV (20% DD), EUR_USD:")
    for tf in ("5min","15min","1h","4h"):
        u=size(5,tf,"EUR_USD",1.08,0.20)
        dpp=u*pip_value_per_unit("EUR_USD",1.08)
        print(f"  {tf:>5}: {u} units  (= ${dpp:.5f}/pip; at ~{34.6/4:.1f} pips/day this acct ~ ${dpp*34.6/4:.4f}/day)")

if __name__=="__main__": main()
