"""IS gate synthesis (plan Task 7 Step 1) — confirmatory cell across pairs + arm contrasts.

For each pair: against/with/coin arms on the confirmatory geometry (t32, h2),
hiER bucket vs all-ER, gross and net-ECN-1x expectancy, decided WR, and the
Amendment-2 operative contrasts (against − coin). Point estimates only; CIs
are an OOS-unseal deliverable. IS rows only — OOS stays sealed.
"""
import sys
import pandas as pd
from analysis import analyze, TP_LEVELS

PAIRS = ["EUR_USD", "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP",
         "EUR_JPY", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY"]


def cell(tbl, bucket):
    row = tbl[tbl["bucket"] == bucket]
    if len(row) == 0:
        return None
    return row.iloc[0]


def main(pairs):
    recs = []
    for pair in pairs:
        res = analyze(pair)  # IS only
        for arm in ("against", "with", "coin"):
            tbl = res[(arm, "t32", "h2")]
            hi = cell(tbl, "hiER")
            if hi is None:
                continue
            recs.append({
                "pair": pair, "arm": arm,
                "n": int(hi["n"]), "n_dec": int(hi["n_decided"]),
                "wr": hi["wr"], "timeout": hi["timeout_share"],
                "gross": hi["gross_mean"],
                "net_ecn": hi["net_ecn_1x"], "net_ecn_15": hi["net_ecn_1.5x"],
                "net_oanda": hi["net_oanda_1x"],
                "fifo_frac": res.get("fifo_realized_frac"),
            })
        print(f"{pair} done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv("synth_is_confirmatory.csv", index=False)

    piv = df.pivot_table(index="pair", columns="arm",
                         values=["wr", "gross", "net_ecn"], aggfunc="first")
    piv[("wr", "against_minus_coin")] = piv[("wr", "against")] - piv[("wr", "coin")]
    piv[("gross", "against_minus_coin")] = piv[("gross", "against")] - piv[("gross", "coin")]
    piv[("net_ecn", "against_minus_coin")] = piv[("net_ecn", "against")] - piv[("net_ecn", "coin")]

    print("\n=== Confirmatory cell (t32/h2, hiER) — IS only ===")
    print(df.round(4).to_string(index=False))
    print("\n=== against − coin contrasts ===")
    print(piv.round(4).to_string())
    n_pos_gross = (piv[("gross", "against_minus_coin")] > 0).sum()
    n_pos_net = (piv[("net_ecn", "against")] > 0).sum()
    print(f"\npairs with against>coin gross delta: {n_pos_gross}/{len(piv)}")
    print(f"pairs with against net-ECN>0:        {n_pos_net}/{len(piv)}")


if __name__ == "__main__":
    main(sys.argv[1:] or PAIRS)
