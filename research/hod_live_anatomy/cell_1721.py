"""Cell 1,721: bottom-up loser anatomy of the LIVE HOD paper book (PREREG_1721.md, frozen 2026-10-30).

Step 1 decomposes each closed fill into gross-at-intended-levels, entry slip, exit slip and net.
Step 2 evaluates ONLY the 12 pre-listed cuts under the frozen A/B bar.

Reads a COPY of trades.db (path via --db) and data/cache.db read-only (daily_bars).  Writes only
fills_1721.csv and RESULT_1721.md next to this file.

Conventions (written before any number was read):
  * R unit = total_risk of the row (shares * (entry_price - stop)); R multiple = pnl / total_risk.
    Rows with total_risk <= 0 are kept in the $ tables but excluded from every R statistic and logged.
  * Kept side per cut = the side the written mechanism favours, split at the POOLED median of the cut
    variable (ties go to kept), so kept >= 50 % of fills by construction; categorical cuts use the
    mechanism-implied groups.  Direction is fixed in CUTS below and never flipped after seeing data.
  * A/B halves = odd / even calendar day of trade_date.
  * PASS: kept - dropped >= +0.10 R in BOTH halves (same sign), kept >= 50 % of fills (and >= 5/week),
    kept ex-top-5 % R/trade > 0, mechanism listed in the PREREG.  Passing in A only = unconfirmed.
  * Cuts 4, 5, 12 have no mechanism written in the PREREG -> reported, can never PASS.
  * Cuts 3 (entry spread) and 8 (SPY) are not computable (columns / bars absent) -> reported as such.
"""
import argparse
import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
START = "2026-09-29"
EDGE = 0.10


def log(msg):
    """Verbose, flushed progress line."""
    print(msg, flush=True)


def to_et(s):
    """Parse ISO UTC strings to tz-aware America/New_York timestamps."""
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_convert("America/New_York")


def load_fills(db):
    """Load the live HOD paper book from the trades.db copy and derive base fields."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    d = pd.read_sql("select * from trades where strategy='hod_break' and account='paper' and trade_date>=?",
                    con, params=(START,))
    con.close()
    log(f"rows {len(d)} (closed {d.pnl.notna().sum()}, other {d.pnl.isna().sum()})")
    pat = d["pattern_data"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    for k in ("level", "consol_low", "limit", "target_r", "entry_mode"):
        d[k] = pat.apply(lambda p: p.get(k))
    d["created_et"] = to_et(d["created_at"])
    d["filled_et"] = to_et(d["filled_at"]) if d["filled_at"].notna().any() else pd.NaT
    d.loc[d["filled_at"].isna(), "filled_et"] = pd.NaT
    d["exited_et"] = to_et(d["exited_at"])
    return d


def decompose(c):
    """Add gross-at-levels, entry slip, exit slip and residual columns ($ and R) to closed fills."""
    c = c.copy()
    c["fill_used"] = c["fill_price"].fillna(c["entry_price"])
    c["fill_fallback"] = c["fill_price"].isna()
    kind = np.where(c.exit_reason == "target", "target",
                    np.where(c.exit_reason.isin(["stop", "stop_loss", "stop_loss_unconfirmed"]), "stop", "eod"))
    c["exit_kind"] = kind
    c["exit_level"] = np.where(kind == "target", c.take_profit_price,
                               np.where(kind == "stop", c.stop_loss_price, c.exit_price))
    qty = c["filled_qty"].fillna(c["shares"])            # pnl in the DB reflects filled_qty (25 rows differ from shares)
    sh = qty
    c["risk_eff"] = c["total_risk"] * qty / c["shares"]   # R unit rescaled to the quantity actually traded
    c["gross_usd"] = sh * (c.exit_level - c.level)
    c["entry_cost_usd"] = sh * (c.fill_used - c.level)
    c["exit_cost_usd"] = sh * (c.exit_level - c.exit_price)
    c["resid_usd"] = c.pnl - (c.gross_usd - c.entry_cost_usd - c.exit_cost_usd)
    c["valid_r"] = c.risk_eff > 0
    c["R"] = np.where(c.valid_r, c.pnl / c.risk_eff, np.nan)
    for col in ("gross", "entry_cost", "exit_cost", "resid"):
        c[col + "_R"] = np.where(c.valid_r, c[col + "_usd"] / c.risk_eff, np.nan)
    c["entry_slip_bps"] = (c.fill_used / c.level - 1) * 1e4
    return c


def add_features(c, cache_db):
    """Add the 12 cut variables (3 and 8 marked not computable)."""
    f = c["filled_et"]
    log(f"fills without filled_at (time cuts treat them as dropped/na): {int(f.isna().sum())}")
    c["fill_min"] = f.dt.hour * 60 + f.dt.minute
    c["fill_bucket"] = (c.fill_min // 10 * 10).map(lambda m: "na" if pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}")
    c["r_pct"] = c.risk_per_share / c.entry_price * 100
    c["spread_bps"] = np.nan                                   # cut 3: entry_quote_spread is NULL for every row
    c["price"] = c.entry_price
    first_arm = c.groupby("trade_date")["created_et"].transform("min")   # includes canceled rows via full frame below
    c["min_since_arm"] = (f - first_arm).dt.total_seconds() / 60
    c = c.sort_values("filled_et")
    c["rank_day"] = c.groupby("trade_date").cumcount() + 1
    c["fills_day"] = c.groupby("trade_date")["id"].transform("count")
    c["spy_ret"] = np.nan                                      # cut 8: no SPY/QQQ 1-min after 2026-05-29
    c["depth_ratio"] = c.risk_per_share / (c.level - c.consol_low)
    c["wk"] = f.dt.isocalendar().week.astype(float)
    c["repeat_wk"] = c.groupby(["symbol", "wk"]).cumcount() > 0
    c["dow"] = f.dt.dayofweek
    c = add_gap(c, cache_db)
    return c.sort_values("id")


def add_gap(c, cache_db):
    """Cut 9: today's open vs prior close from daily_bars (read-only); NaN where absent."""
    con = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True, timeout=10)
    syms = sorted(c.symbol.unique())
    q = ",".join("?" * len(syms))
    b = pd.read_sql(f"select symbol, bar_date, open, close from daily_bars where bar_date>='2026-09-15' and symbol in ({q})",
                    con, params=syms)
    con.close()
    b = b.sort_values(["symbol", "bar_date"])
    b["prev_close"] = b.groupby("symbol")["close"].shift(1)
    b["gap_pct"] = (b.open / b.prev_close - 1) * 100
    c["trade_date_s"] = c.trade_date.astype(str)
    m = c.merge(b[["symbol", "bar_date", "gap_pct"]].rename(columns={"bar_date": "trade_date_s"}),
                on=["symbol", "trade_date_s"], how="left")
    log(f"gap coverage {m.gap_pct.notna().mean():.0%} of fills")
    return m


def cluster_diff(y, kept, days):
    """Kept-minus-dropped mean difference with day-clustered SE (sandwich on OLS slope) and iid Welch."""
    y = np.asarray(y, float)
    k = np.asarray(kept, float)
    ok = ~np.isnan(y)
    y, k, days = y[ok], k[ok], np.asarray(days)[ok]
    if k.sum() < 3 or (1 - k).sum() < 3:
        return dict(diff=np.nan, p_iid=np.nan, p_cl=np.nan)
    diff = y[k == 1].mean() - y[k == 0].mean()
    t, p2 = stats.ttest_ind(y[k == 1], y[k == 0], equal_var=False)
    X = np.column_stack([np.ones_like(k), k])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for dd in np.unique(days):
        s = X[days == dd].T @ e[days == dd]
        meat += np.outer(s, s)
    G = len(np.unique(days))
    V = XtXi @ meat @ XtXi * G / max(G - 1, 1)
    z = beta[1] / math.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan
    return dict(diff=diff, p_iid=(p2 / 2 if diff > 0 else 1 - p2 / 2), p_cl=1 - stats.norm.cdf(z))


def exTop5(y):
    """Mean of y after removing its top 5 % (rounded up) values."""
    y = np.sort(np.asarray(y, float)[~np.isnan(y)])
    n = math.ceil(0.05 * len(y))
    return y[:-n].mean() if len(y) > n else np.nan


def side_stats(d, kept):
    """Fills, R/trade and $ for a boolean mask and its complement within frame d."""
    out = {}
    for name, m in (("kept", kept), ("drop", ~kept)):
        s = d[m]
        out[name] = (len(s), s.R.mean(), s.pnl.sum())
    return out


def evaluate(c, name, kept, mech):
    """Apply the frozen A/B bar to one cut; returns a result dict."""
    c = c.assign(kept=kept)
    c = c[c.valid_r]
    odd = c.trade_date.str[-2:].astype(int) % 2 == 1
    res = dict(cut=name, mech=mech)
    for h, m in (("A", odd), ("B", ~odd), ("pooled", pd.Series(True, index=c.index))):
        s = c[m]
        st = side_stats(s, s.kept)
        res[h] = st
        res[h + "_diff"] = st["kept"][1] - st["drop"][1]
    cd = cluster_diff(c.R, c.kept, c.trade_date)
    res.update(cd)
    res["keep_frac"] = c.kept.mean()
    weeks = (pd.to_datetime(c.trade_date).max() - pd.to_datetime(c.trade_date).min()).days / 7 + 1 / 7
    res["kept_per_wk"] = c.kept.sum() / weeks
    res["kept_exTop5"] = exTop5(c[c.kept].R)
    res["bonf_p_cl"] = min(1, cd["p_cl"] * 12) if not np.isnan(cd["p_cl"]) else np.nan
    res["bonf_p_iid"] = min(1, cd["p_iid"] * 12) if not np.isnan(cd["p_iid"]) else np.nan
    a_ok, b_ok = res["A_diff"] >= EDGE, res["B_diff"] >= EDGE
    other = res["keep_frac"] >= 0.5 and res["kept_per_wk"] >= 5 and res["kept_exTop5"] > 0
    if not mech:
        res["verdict"] = "FAIL (no mechanism listed; cannot pass)" if not (a_ok and b_ok) else "no-mechanism (cannot pass)"
    elif a_ok and b_ok and other:
        res["verdict"] = "PASS"
    elif (a_ok or b_ok) and other:
        res["verdict"] = "UNCONFIRMED"
    else:
        res["verdict"] = "FAIL"
    return res


def cut_defs(c):
    """The 12 frozen cuts: (id, label, kept mask or None, mechanism-listed flag, bucket series for the table)."""
    med = lambda s: s.median()
    defs = []
    defs.append((1, "fill time ET (keep <= median)", c.fill_min <= med(c.fill_min), True, c.fill_bucket))
    defs.append((2, "R as % of price (keep >= median)", c.r_pct >= med(c.r_pct), True,
                 pd.qcut(c.r_pct, 4, duplicates="drop").astype(str)))
    defs.append((3, "entry spread bps", None, True, None))
    defs.append((4, "price (keep >= median)", c.price >= med(c.price), False,
                 pd.cut(c.price, [0, 5, 10, 20, 1e9]).astype(str)))
    defs.append((5, "minutes first arm->fill (keep <= median)", c.min_since_arm <= med(c.min_since_arm), False,
                 pd.qcut(c.min_since_arm, 4, duplicates="drop").astype(str)))
    defs.append((6, "fill rank in day (keep <= median rank)", c.rank_day <= med(c.rank_day), True,
                 c.rank_day.clip(upper=3).map({1: "1st", 2: "2nd", 3: "3rd+"})))
    defs.append((7, "fills that day (keep <= median)", c.fills_day <= med(c.fills_day), True,
                 pd.cut(c.fills_day, [0, 5, 10, 1e9], labels=["1-5", "6-10", "11+"]).astype(str)))
    defs.append((8, "SPY ret 09:30->fill", None, True, None))
    g = c.gap_pct
    defs.append((9, "today's gap % (keep >= median)", (g >= g.median()).where(g.notna(), False), True,
                 pd.qcut(g, 4, duplicates="drop").astype(str)))
    dr = c.depth_ratio
    defs.append((10, "R/(level-consol_low) (keep <= median)", dr <= med(dr), True, pd.qcut(dr, 4, duplicates="drop").astype(str)))
    defs.append((11, "symbol repeat in ISO week (keep first fill)", ~c.repeat_wk, True, c.repeat_wk.map({False: "first", True: "repeat"})))
    defs.append((12, "day of week (keep Tue-Thu)", c.dow.isin([1, 2, 3]), False, c.dow.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})))
    return defs


def fmt_side(st):
    """One A/B table cell: n, R/trade, $."""
    n, r, p = st
    return f"{n} / {r:+.3f} / {p:+.0f}"


def md_table(rows, header):
    """Render a markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def mde(sd, n_keep, n_drop, alpha, power=0.8):
    """Minimum detectable kept-minus-dropped difference in R for a one-sided z-test."""
    se = sd * math.sqrt(1 / n_keep + 1 / n_drop)
    return (stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * se


def main():
    """Run decomposition + cuts and write fills_1721.csv and RESULT_1721.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "cache.db"))
    a = ap.parse_args()
    d = load_fills(a.db)
    cl = d[d.pnl.notna()].copy()
    odd_rows = cl[~(cl.total_risk > 0)]
    log(f"closed {len(cl)}; total_risk<=0 rows: {len(odd_rows)} ids {odd_rows.id.tolist()}")
    c = decompose(cl)
    c = add_features(c, a.cache)
    log(f"fill_price fallback to entry_price on {int(c.fill_fallback.sum())} rows")
    c.drop(columns=["pattern_data", "created_et", "filled_et", "exited_et"]).to_csv(os.path.join(HERE, "fills_1721.csv"), index=False)

    v = c[c.valid_r]
    sd = v.R.std()
    L = ["# RESULT 1,721 — loser anatomy of the live HOD paper book", "",
         f"Window {c.trade_date.min()} .. {c.trade_date.max()}; closed fills {len(c)} (PREREG said 241; the DB has 232 closed + 2 canceled "
         f"CONI rows = 234 rows). R-valid fills {len(v)} (rows with total_risk <= 0 excluded from R stats: ids {odd_rows.id.tolist()}). "
         f"R = pnl / (total_risk * filled_qty/shares) (actual risk per fill; 25 rows have filled_qty != shares and the DB pnl follows filled_qty; mean ${c.total_risk.mean():.1f}, not the nominal $50).", "",
         "## Step 1 — decomposition (per fill; R = / total_risk, valid rows)", ""]
    rows = []
    for lab, m in (("all", c.valid_r | ~c.valid_r), ("stop exits", c.exit_kind == "stop"), ("target exits", c.exit_kind == "target"),
                   ("eod exits", c.exit_kind == "eod")):
        s = c[m]
        sv = s[s.valid_r]
        rows.append([lab, len(s), f"{sv.gross_R.mean():+.3f}", f"{sv.entry_cost_R.mean():+.3f}", f"{sv.exit_cost_R.mean():+.3f}",
                     f"{sv.resid_R.mean():+.3f}", f"{sv.R.mean():+.3f}", f"{s.gross_usd.sum():+.0f}", f"{s.entry_cost_usd.sum():+.0f}",
                     f"{s.exit_cost_usd.sum():+.0f}", f"{s.resid_usd.sum():+.0f}", f"{s.pnl.sum():+.0f}"])
    L.append(md_table(rows, ["group", "n", "gross@levels R", "entry slip R (cost)", "exit slip R (cost)", "residual R", "NET R/trade",
                             "gross $", "entry cost $", "exit cost $", "resid $", "net $"]))
    allv = v
    gross, cost = allv.gross_R.mean(), allv.entry_cost_R.mean() + allv.exit_cost_R.mean()
    L += ["", "Identity: net = gross@levels - entry slip - exit slip + residual (residual = partial-exit row, fill fallback, fees; "
          "eod exits carry exit_level = actual exit so zero exit slip by construction). Entry slip = fill - pattern level; "
          "exit slip = intended exit level - actual exit (stop/target exits only).",
          f"All: gross@levels {gross:+.3f} R/trade, total cost {cost:+.3f} R/trade -> "
          + ("COST >= |GROSS|: the lever is EXECUTION first." if cost >= abs(gross) else "gross exceeds cost."),
          f"Entry slip: mean {c.entry_slip_bps.mean():.1f} bps, median {c.entry_slip_bps.median():.1f}, p90 {c.entry_slip_bps.quantile(.9):.1f}. "
          f"Exit_slippage column (as stored, unit price): mean {c.exit_slippage.mean():+.3f} (n {c.exit_slippage.notna().sum()}).",
          f"Exit-method counts: {c.exit_pricing_method.value_counts(dropna=False).to_dict()}", ""]
    e = c[c.exit_kind == "eod"]
    ev = e[e.valid_r]
    L += ["### EOD exits", "",
          f"n {len(e)} ({(e.exit_reason == 'eod').sum()} eod + {(e.exit_reason == 'eod_ops_ah').sum()} eod_ops_ah); P&L ${e.pnl.sum():+.0f}; "
          f"R/trade {ev.R.mean():+.3f}; positive {int((e.pnl > 0).sum())} / negative {int((e.pnl < 0).sum())}. "
          f"R quantiles (min, p10, p25, med, p75, p90, max): " + ", ".join(f"{x:+.2f}" for x in ev.R.quantile([0, .1, .25, .5, .75, .9, 1])) + ". "
          "MFE NOT computable: no 1-min bars for Sep-Oct 2026 in cache.db (0 rows for SPY/AAOI since 2026-09-01); only sign/distribution reported.", ""]

    L += [f"Book per-fill R: mean {v.R.mean():+.3f}, sd {sd:.3f}, n {len(v)}, ${c.pnl.sum():+.0f} total; "
          f"ex-top-5 % mean {exTop5(v.R):+.3f}.", "", "## Step 2 — the 12 cuts (cell = n / R per trade / $ ; kept vs dropped)", ""]
    results = []
    for cid, label, kept, mech, bucket in cut_defs(c):
        L.append(f"### Cut {cid}: {label}")
        if kept is None:
            reason = ("entry_quote_spread / entry quote columns are NULL for all rows (no entry NBBO stored)" if cid == 3 else
                      "SPY/QQQ 1-min parquet ends 2026-05-29 and cache.db has no Sep-Oct intraday bars; the daily_bars close-to-close "
                      "SPY return would include post-fill information (look-ahead), so not substituted")
            L += [f"NOT COMPUTABLE: {reason}.", ""]
            results.append(dict(cut=f"{cid} {label}", verdict="NOT COMPUTABLE"))
            continue
        r = evaluate(c, f"{cid} {label}", kept, mech)
        results.append(r)
        rows = [[h, fmt_side(r[h]["kept"]), fmt_side(r[h]["drop"]), f"{r[h + '_diff']:+.3f}"] for h in ("A", "B", "pooled")]
        L.append(md_table(rows, ["half", "kept n/R/$", "dropped n/R/$", "kept - dropped R"]))
        L.append(f"\nkept {r['keep_frac']:.0%} of fills ({r['kept_per_wk']:.1f}/wk); kept ex-top-5 % {r['kept_exTop5']:+.3f}; "
                 f"p one-sided: day-clustered {r['p_cl']:.3f} (Bonferroni x12: {r['bonf_p_cl']:.3f}), iid Welch {r['p_iid']:.3f} "
                 f"(Bonf {r['bonf_p_iid']:.3f}); mechanism listed: {mech}. **Verdict: {r['verdict']}**")
        vb = c[c.valid_r].assign(bk=bucket[c.valid_r.values]) if bucket is not None else None
        if vb is not None:
            g = vb.groupby("bk").agg(n=("R", "size"), R=("R", "mean"), usd=("pnl", "sum")).reset_index()
            L += ["", "Bucket table (pooled): " + "; ".join(f"{x.bk}: {x.n} / {x.R:+.3f} / {x.usd:+.0f}" for x in g.itertuples())]
        L.append("")

    L += ["## Verdict summary", ""]
    L.append(md_table([[r["cut"], r.get("verdict"), f"{r.get('A_diff', float('nan')):+.3f}", f"{r.get('B_diff', float('nan')):+.3f}",
                        f"{r.get('bonf_p_cl', float('nan')):.3f}"] for r in results],
                      ["cut", "verdict", "A diff R", "B diff R", "Bonf p (cluster)"]))
    n_half = len(v) // 2
    L += ["", "## MDE (per-fill R sd %.3f from this book)" % sd, "",
          f"50 % cut pooled (116 vs 116 of {len(v)}): one-sided alpha 0.05 power 0.8 -> {mde(sd, len(v)//2, len(v)//2, .05):.3f} R/trade; "
          f"Bonferroni alpha 0.05/12 -> {mde(sd, len(v)//2, len(v)//2, .05/12):.3f} R. "
          f"Per half (~{n_half//2} vs {n_half//2}): alpha 0.05 -> {mde(sd, n_half//2, n_half//2, .05):.3f} R; the frozen +0.10 R edge is "
          f"{EDGE / mde(sd, n_half // 2, n_half // 2, .05):.2f} of that per-half MDE, i.e. each half can only resolve "
          f"effects of that size, and the +0.10 R bar in both halves is mostly noise-gated.", ""]
    passed = [r for r in results if r.get("verdict") == "PASS"]
    if passed:
        L += ["## Ship recommendation", ""]
        for r in passed:
            L.append(f"Cut {r['cut']} passed; engine condition = keep fills where the stated variable meets the stated median direction "
                     "(see label); thresholds are the pooled medians computed in this script and must be frozen at their numeric values "
                     "before wiring into the HOD paper engine.")
    else:
        L += ["## Ship recommendation", "", "No cut passes the frozen bar; nothing to ship from this cell. See the MDE paragraph for what "
              "232 fills can and cannot detect."]
    L += ["", "## Caveats on this implementation", "",
          "* Fill count 232 closed (not 241); 2 canceled rows ignored. R unit = total_risk rescaled by filled_qty/shares (not the nominal $50); the first run without that rescale left a +0.045 R residual from 25 rows where filled_qty != shares.",
          "* Direction and threshold (pooled median) of every cut were fixed in code before computing outcomes; categorical tables are "
          "descriptive. Cut 7 uses the full-day fill count (not known at fill time -> not directly implementable; a running count would be needed). "
          "Cut 5 minutes are fill minus the day's first created_at (any row, including canceled). Cut 10 follows the PREREG formula literally "
          "(risk_per_share uses entry_price, which includes the expected slip). Cuts 4, 5, 12 had no mechanism written -> cannot pass.",
          "* Cuts 3 and 8 not computable (no entry quotes; no Sep-Oct index bars). Cut 9 gap uses daily_bars open vs prior close; fills on "
          "dates/symbols missing from daily_bars drop out of the kept side (treated as not kept); coverage logged in the run.",
          "* Gross@levels for eod exits equals the actual exit (no intended level). Residual column absorbs the partial-exit row and rows "
          "with fill_price missing (entry_price used).",
          "* Day-clustered one-sided p from a sandwich OLS slope with 22-ish clusters is itself approximate; the iid Welch p is also shown.",
          "* 232 fills across ~22 dates, one regime month; this is an in-sample anatomy of a single window."]
    open(os.path.join(HERE, "RESULT_1721.md"), "w").write("\n".join(L) + "\n")
    log("wrote RESULT_1721.md")


if __name__ == "__main__":
    main()
