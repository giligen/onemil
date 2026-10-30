"""Cell 1,722 (EXPLORATORY, post hoc): execution levers on the live HOD paper book; appends to RESULT_1721.md.

(a) outcome by entry-slip bucket, (b) exit slip by exit_reason, (c) EOD exits at the bell, (d) entry-quote recording check.
Reads fills_1721.csv and a trades.db copy (--db).
"""
import argparse, os, sqlite3
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def log(m):
    """Flushed progress."""
    print(m, flush=True)


def tbl(rows, hdr):
    """Markdown table."""
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def bucket_rows(d, label):
    """One row per entry-slip bucket for frame d."""
    b = pd.cut(d.entry_slip_bps, [-1e9, 5, 10, 15, 1e9], labels=["<=5", "5-10", "10-15", ">15"])
    out = []
    for k, g in d.groupby(b, observed=False):
        out.append([label, k, len(g), f"{g.gross_R.mean():+.3f}", f"{g.R.mean():+.3f}", f"{(g.pnl > 0).mean():.0%}", f"{(g.exit_kind == 'target').mean():.0%}"])
    return out


def main():
    """Compute (a)-(d) and append to RESULT_1721.md."""
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); a = ap.parse_args()
    c = pd.read_csv(os.path.join(HERE, "fills_1721.csv")); v = c[c.valid_r]
    odd = v.trade_date.str[-2:].astype(int) % 2 == 1
    L = ["", "## Cell 1,722 — execution levers (exploratory, post hoc)", "",
         "Not pre-registered; descriptive, no pass bar. Buckets use entry slip = fill - pattern level in bps (n=%d R-valid)." % len(v), "",
         "### (a) outcome by entry-slip bucket"]
    rows = bucket_rows(v, "pooled") + bucket_rows(v[odd], "A") + bucket_rows(v[~odd], "B")
    L.append(tbl(rows, ["half", "slip bps", "n", "gross@levels R", "net R", "win %", "target-hit %"]))
    L.append("\nslip quantiles p10/p50/p90: " + "/".join(f"{x:.1f}" for x in v.entry_slip_bps.quantile([.1, .5, .9])))
    L += ["", "### (b) exit slip by exit_reason (exit slip = intended level - actual exit; positive = cost; eod has no intended level)"]
    c["exit_slip_bps"] = (c.exit_level - c.exit_price) / c.exit_level * 1e4
    rows = []
    for k, g in c.groupby("exit_reason"):
        gv = g[g.valid_r]
        rows.append([k, len(g), f"{g.exit_slip_bps.mean():+.1f}", f"{gv.exit_cost_R.mean():+.3f}", g.exit_pricing_method.value_counts(dropna=False).to_dict()])
    L.append(tbl(rows, ["exit_reason", "n", "mean slip bps", "mean slip R", "exit_pricing_method"]))
    rows = []
    for (k, m), g in c[c.exit_kind != "eod"].groupby(["exit_kind", "exit_pricing_method"], dropna=False):
        gv = g[g.valid_r]
        rows.append([k, m, len(g), f"{g.exit_slip_bps.mean():+.1f}", f"{gv.exit_cost_R.mean():+.3f}", f"{gv.exit_cost_R.sum():+.1f}"])
    L += ["", "By kind x method (sum R = contribution to the 0.037 R/trade total):", tbl(rows, ["kind", "method", "n", "mean bps", "mean R", "sum R"])]
    e = c[c.exit_kind == "eod"]; ev = e[e.valid_r]
    held = (pd.to_datetime(e.exited_at, utc=True, format="ISO8601") - pd.to_datetime(e.filled_at, utc=True, format="ISO8601")).dt.total_seconds() / 60
    L += ["", "### (c) EOD exits (n=%d)" % len(e),
          f"R at exit p10/p50/p90: {ev.R.quantile(.1):+.2f}/{ev.R.quantile(.5):+.2f}/{ev.R.quantile(.9):+.2f}; above +1 R: {(ev.R > 1).sum()} of {len(ev)} "
          f"(mean R of those {ev[ev.R > 1].R.mean():+.2f}, ${ev[ev.R > 1].pnl.sum():+.0f}); mean minutes held {held.mean():.0f} (median {held.median():.0f}, n {held.notna().sum()})."]
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    q = pd.read_sql("select account, count(*) n, count(entry_quote_bid) bid, count(entry_quote_ask) ask, count(entry_quote_spread) spr, "
                    "count(entry_fill_quote_bid) fbid, count(entry_fill_quote_ask) fask, count(exit_quote_bid) xbid from trades where strategy='hod_break' group by account", con)
    q2 = pd.read_sql("select count(*) n, count(entry_quote_bid) bid, count(entry_quote_ask) ask from trades where strategy='orb_breakout' or strategy like 'orb%'", con)
    L += ["", "### (d) entry quote recording (hod_break rows, all dates)", tbl(q.values.tolist(), list(q.columns)),
          "ORB rows for comparison (n / non-null bid / ask): " + " / ".join(str(x) for x in q2.iloc[0].tolist())]
    open(os.path.join(HERE, "RESULT_1721.md"), "a").write("\n".join(L) + "\n")
    log("\n".join(L))


main()
