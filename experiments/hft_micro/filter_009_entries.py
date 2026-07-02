"""
009 (post-shock retrace) diagnostic — would an efficiency / spread_rel / ATR
regime filter at ENTRY have skipped the recent dead timeouts while keeping the
TP winners?

Context: (a) showed the TP captures stopped because favorable-excursion (MFE)
amplitude collapsed in wk24 (p90 MFE 17.4→9.8, 0% reached the 20-pip TP). This
asks whether a *pre-entry, causal* regime feature separates the dead timeouts
(MFE<10, drift to scratch) from the productive trades (MFE>=15 / TP).

Features at entry (from the last COMPLETED M5 bar before entry_time, OANDA fetch):
  - atr14_pips : ATR(14) on M5 in pips  (volatility — the (a) suspect)
  - eff        : directional efficiency |net|/path over last 12 M5 bars
  - spread_rel : M5 spread / rolling-mean spread(48)

HONEST CAVEAT: this is an IN-SAMPLE / hindsight split on 247 live trades. A
filter that helps here is a HYPOTHESIS to validate OOS (or forward on paper),
not a deployable result.
"""
import json, os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv; load_dotenv(".env")
except Exception:
    pass
import v20

PAIRS = ["USD_JPY", "GBP_JPY", "AUD_JPY", "EUR_JPY"]
TRADES = json.load(open("/tmp/trades_009.json"))
KEY = os.environ["OANDA_API_KEY"]
ACCT = os.environ.get("OANDA_ACCOUNT_ID_009") or os.environ.get("OANDA_ACCOUNT_ID_001", "")


def pip_of(p):
    return 0.01 if "JPY" in p else 0.0001


def fetch_m5(pair):
    """Fetch M5 mid OHLC + bid/ask close over the trade span via from/to chunks."""
    ctx = v20.Context("api-fxtrade.oanda.com", 443, token=KEY)
    out = []
    start = datetime(2026, 5, 24, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) - timedelta(minutes=10)   # never request future toTime
    a = start
    while a < end:
        b = min(a + timedelta(days=5), end)
        resp = ctx.instrument.candles(
            pair, granularity="M5", price="MBA",
            fromTime=a.isoformat().replace("+00:00", "Z"),
            toTime=b.isoformat().replace("+00:00", "Z"))
        cs = resp.body.get("candles", []) if hasattr(resp.body, "get") else resp.body["candles"]
        for c in cs:
            if not c.complete:
                continue
            out.append((str(c.time), float(c.mid.o), float(c.mid.h), float(c.mid.l),
                        float(c.mid.c), float(c.bid.c), float(c.ask.c)))
        a = b
    df = pd.DataFrame(out, columns=["t", "o", "h", "l", "c", "bid", "ask"]).drop_duplicates("t")
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df.sort_values("t").reset_index(drop=True)


def features(df, pair):
    pip = pip_of(pair)
    c = df["c"].values; h = df["h"].values; l = df["l"].values
    pc = np.roll(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]); tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14).mean().values / pip
    dmid = np.diff(c, prepend=c[0]); abscum = np.cumsum(np.abs(dmid))
    N = 12
    net = np.full(len(c), np.nan); path = np.full(len(c), np.nan)
    net[N:] = np.abs(c[N:] - c[:-N]); path[N:] = abscum[N:] - abscum[:-N]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(path > 0, net / path, np.nan)
    sp = (df["ask"].values - df["bid"].values) / pip
    sp_avg = pd.Series(sp).rolling(48, min_periods=10).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        sp_rel = np.where(sp_avg > 0, sp / sp_avg, np.nan)
    df["atr"] = atr; df["eff"] = eff; df["sp_rel"] = sp_rel
    return df


def main():
    feat = {}
    for p in PAIRS:
        if not any(t[2] == p for t in TRADES):
            continue
        print(f"fetching M5 {p} …")
        feat[p] = features(fetch_m5(p), p)
        print(f"  {p}: {len(feat[p])} bars {feat[p]['t'].min()} → {feat[p]['t'].max()}")

    rows = []
    for et, xt, pair, d, pnl, mfe, mae, reason in TRADES:
        df = feat.get(pair)
        if df is None:
            continue
        entry = pd.to_datetime(et, utc=True)
        prior = df[df["t"] < entry]
        if len(prior) < 50:
            continue
        b = prior.iloc[-1]
        rows.append(dict(pair=pair, entry=entry, pnl=pnl, mfe=mfe, mae=mae, reason=reason,
                         atr=b["atr"], eff=b["eff"], sp_rel=b["sp_rel"]))
    R = pd.DataFrame(rows).dropna(subset=["atr", "eff", "sp_rel"])
    R["productive"] = R["mfe"] >= 15        # reached most of the 20p TP
    R["dead"] = (R["reason"] == "timeout") & (R["mfe"] < 10)
    print(f"\nmatched {len(R)}/{len(TRADES)} trades to entry-regime features")
    print(f"feature coverage to {max(f['t'].max() for f in feat.values())}\n")

    # per-week entry-regime (ties directly to finding (a): wk24 = low-vol regime?)
    R["wk"] = R["entry"].dt.strftime("%Y-%W")
    print("=== entry-regime by week: n, entryATR, eff, sp_rel, TP%, productive% ===")
    for wk, g in R.groupby("wk"):
        print(f"  {wk}: n={len(g):>3} atr={g['atr'].mean():.2f} eff={g['eff'].mean():.2f} "
              f"sp_rel={g['sp_rel'].mean():.2f} "
              f"tp%={100*(g['reason']=='tp').mean():.0f} prod%={100*g['productive'].mean():.0f}")
    print()

    print("=== entry-regime by outcome (mean) ===")
    print(R.groupby(R["productive"].map({True: "productive(MFE>=15)", False: "other"}))
          [["atr", "eff", "sp_rel", "pnl", "mfe"]].mean().round(2))
    print()
    print("dead timeouts (MFE<10):", int(R["dead"].sum()), "/", len(R),
          "| their mean atr=%.2f eff=%.2f sp_rel=%.2f"
          % (R.loc[R["dead"], "atr"].mean(), R.loc[R["dead"], "eff"].mean(),
             R.loc[R["dead"], "sp_rel"].mean()))

    # filter sweep — keep trades passing a threshold; report net pips + TP retention
    print("\n=== filter sweep (in-sample/hindsight) — net pips & productive-retention ===")
    base_net = R["pnl"].sum(); base_prod = int(R["productive"].sum())
    print(f"  NO FILTER: n={len(R)} net={base_net:+.1f}p productive={base_prod} "
          f"deadkept={int(R['dead'].sum())}")
    for feat_name, op, thrs in [("atr", ">=", [4, 6, 8, 10, 12]),
                                ("eff", ">=", [0.2, 0.3, 0.4, 0.5]),
                                ("eff", "<=", [0.3, 0.4, 0.5]),
                                ("sp_rel", "<=", [0.9, 1.0, 1.1, 1.3])]:
        for thr in thrs:
            keep = R[feat_name] >= thr if op == ">=" else R[feat_name] <= thr
            sub = R[keep]
            if len(sub) < 20:
                continue
            print(f"  {feat_name}{op}{thr}: n={len(sub):>3} "
                  f"net={sub['pnl'].sum():+8.1f}p  net/trade={sub['pnl'].mean():+5.2f}  "
                  f"prod={int(sub['productive'].sum()):>2}/{base_prod}  "
                  f"deadkept={int(sub['dead'].sum()):>3}/{int(R['dead'].sum())}")


if __name__ == "__main__":
    main()
