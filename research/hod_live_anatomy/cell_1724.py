"""Cell 1,724: time-aware exits on the live HOD paper book (PREREG_1724.md incl. Amendment 1, frozen 2026-10-30).

Replays the live exit rule on Alpaca 1-min SIP bars (free data API, no orders), checks the replay against
the realized ledger, then scores six frozen exit variants paired against the live replay:
  V1 time stop 14:00 | V2 take-the-profit 14:00 | V3 protect (BE) 14:00 | V4 1.5R target
  V5 decaying target (2.0R until 12:00 -> 0.5R at 15:55) | V6 (2.0R until 11:00 -> 0.25R at 15:55)

Inputs : fills_1721.csv (231 R-valid fills).   Bars: cache/bars_1724/{date}_{symbol}.parquet (atomic writes).
Outputs: RESULT_1724.md, variants_1724.csv next to this file.   Writes nothing else.  No orders, no DB writes.

Conventions (see RESULT caveats): bar minute m = minutes since 00:00 ET of the bar START; only bars STRICTLY AFTER the
fill minute are walked; within a bar the stop is checked before the target; stop fills at min(open, stop) - 12 bps;
target fills at the target; EOD exit = close of the 15:55 bar (last close <= 15:55 if that bar is absent);
R = (exit - fill_used) / risk_per_share, risk_per_share = risk_eff / filled_qty (cell 1,721 definition).
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache", "bars_1724")
SLIP = 12e-4            # measured 12 bps exit slip (PREREG)
M_1400, M_1555, M_1600 = 840, 955, 960
ET = "America/New_York"


def log(msg):
    """Verbose, flushed progress line."""
    print(msg, flush=True)


# ---------------------------------------------------------------- data
def load_fills():
    """Load the R-valid fills and derive entry minute, fill_used, per-share risk."""
    c = pd.read_csv(os.path.join(HERE, "fills_1721.csv"))
    c = c[c.valid_r].copy()
    c["fill_used"] = c.fill_price.fillna(c.entry_price)
    c["qty"] = c.filled_qty.fillna(c.shares)
    c["rps"] = c.risk_eff / c.qty
    t = pd.to_datetime(c.filled_at.fillna(c.created_at), utc=True, format="ISO8601").dt.tz_convert(ET)
    c["entry_min"] = t.dt.hour * 60 + t.dt.minute
    c["entry_proxy"] = c.filled_at.isna()
    log(f"fills {len(c)}; entry time proxied by created_at for {int(c.entry_proxy.sum())}")
    return c.reset_index(drop=True)


def fetch_one(client, sym, day, start_min):
    """Fetch 1-min SIP bars for (sym, day) from the fill minute to 16:00 ET."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    d = pd.Timestamp(day, tz=ET)
    s = d + pd.Timedelta(minutes=int(start_min))
    e = d + pd.Timedelta(minutes=M_1600)
    req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=s.tz_convert("UTC"),
                           end=e.tz_convert("UTC"), feed=DataFeed.SIP, limit=10000)
    bars = client.get_stock_bars(req).data.get(sym, [])
    rows = []
    for b in bars:
        t = pd.Timestamp(b.timestamp).tz_convert(ET)
        rows.append((t.hour * 60 + t.minute, float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
    return pd.DataFrame(rows, columns=["m", "o", "h", "l", "c", "v"])


def ensure_bars(c):
    """Fetch every missing (date, symbol) once (earliest fill minute among duplicates); resumable, atomic."""
    os.makedirs(CACHE, exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from alpaca.data.historical import StockHistoricalDataClient
    key, sec = os.environ.get("ALPACA_HOD_API_KEY"), os.environ.get("ALPACA_HOD_API_SECRET")
    if not key or not sec:
        raise SystemExit("ERROR: ALPACA_HOD_API_KEY/SECRET missing - cannot fetch bars")
    client = StockHistoricalDataClient(key, sec)
    g = c.groupby(["trade_date", "symbol"]).entry_min.min().reset_index()
    todo = [(r.trade_date, r.symbol, r.entry_min) for r in g.itertuples()
            if not os.path.exists(os.path.join(CACHE, f"{r.trade_date}_{r.symbol}.parquet"))]
    log(f"symbol-days {len(g)}, to fetch {len(todo)}")
    for i, (day, sym, sm) in enumerate(todo, 1):
        for attempt in range(3):
            try:
                df = fetch_one(client, sym, day, sm)
                break
            except Exception as ex:                                   # retry, then LOST (logged ERROR)
                log(f"WARNING fetch {day} {sym} attempt {attempt + 1}: {ex}")
                time.sleep(1.5)
        else:
            log(f"ERROR LOST {day} {sym}: 3 failed attempts")
            continue
        p = os.path.join(CACHE, f"{day}_{sym}.parquet")
        df.to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        if i % 20 == 0 or i == len(todo):
            log(f"  fetched {i}/{len(todo)} ({day} {sym}: {len(df)} bars)")
        time.sleep(0.2)


def read_bars(day, sym):
    """Cached bars for a symbol-day as a DataFrame sorted by minute (empty if absent)."""
    p = os.path.join(CACHE, f"{day}_{sym}.parquet")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["m", "o", "h", "l", "c", "v"])
    return pd.read_parquet(p).sort_values("m").reset_index(drop=True)


# ---------------------------------------------------------------- replay
def walk(b, m_from, m_to, stop, tgt_fn):
    """Walk bars with m_from <= m < m_to (m <= 955). Stop checked first (gap-through at the open), then target.
    Returns (kind, price, minute) or None if the position survives the window."""
    w = b[(b.m >= m_from) & (b.m < m_to) & (b.m <= M_1555)]
    for r in w.itertuples():
        if r.l <= stop:
            return "stop", min(r.o, stop) * (1 - SLIP), r.m
        t = tgt_fn(r.m)
        if t is not None and r.h >= t:
            return "target", t, r.m
    return None


def eod_price(b, fallback):
    """Close of the 15:55 bar, else last close at or before 15:55, else `fallback`; flag whether degraded."""
    w = b[b.m <= M_1555]
    if w.empty:
        return fallback, True
    last = w.iloc[-1]
    return float(last.c), bool(last.m != M_1555)


def decay_target(entry, rps, m1, floor):
    """Target price function: 2.0R until minute m1, then linear to `floor` R at 15:55 (never below floor)."""
    def f(m):
        r = 2.0 if m < m1 else max(floor, 2.0 - (2.0 - floor) * (m - m1) / (M_1555 - m1))
        return entry + r * rps
    return f


def replay_fill(r, b):
    """Return dict of (kind, price) for the live rule and every variant for one fill."""
    fu, rps, m0 = r.fill_used, r.rps, int(r.entry_min)
    stop0, tp = r.stop_loss_price, r.take_profit_price
    live_t = lambda m: tp
    eodp, degraded = eod_price(b, fu)

    def finish(res):
        """Exit (kind, price) from a walk result, falling through to the EOD flat."""
        return (res[0], res[1]) if res else ("eod", eodp)

    out = {"eod_degraded": degraded}
    out["live"] = finish(walk(b, m0 + 1, M_1600, stop0, live_t))
    # --- 14:00 family: positions still open at the 14:00 decision (replayed live rule, entered before 14:00)
    pre = walk(b, m0 + 1, M_1400, stop0, live_t) if m0 < M_1400 else "late"
    open_1400 = pre is None
    out["open_1400"] = open_1400
    pre_b = b[b.m < M_1400]
    cl_1359 = float(pre_b.iloc[-1].c) if len(pre_b) else fu
    open_r = (cl_1359 - fu) / rps
    out["open_r_1400"] = open_r if open_1400 else np.nan
    post = b[(b.m >= M_1400) & (b.m <= M_1555)]
    p14 = float(post.iloc[0].o) * (1 - SLIP) if len(post) else None
    out["v1"] = ("time", p14) if (open_1400 and p14 is not None) else out["live"]
    out["v2"] = out["v1"] if (open_1400 and p14 is not None and open_r >= 0) else out["live"]
    if open_1400 and open_r >= 0.5:
        out["v3"] = finish(walk(b, M_1400, M_1600, fu, live_t))        # stop moves to break-even at the fill
    else:
        out["v3"] = out["live"]
    # --- target-schedule family
    out["v4"] = finish(walk(b, m0 + 1, M_1600, stop0, lambda m, e=r.entry_price: e + 1.5 * rps))
    out["v5"] = finish(walk(b, m0 + 1, M_1600, stop0, decay_target(r.entry_price, rps, 720, 0.5)))
    out["v6"] = finish(walk(b, m0 + 1, M_1600, stop0, decay_target(r.entry_price, rps, 660, 0.25)))
    return out


def build_table(c):
    """Replay every fill; return per-fill table with R columns and coverage."""
    rows, lost = [], 0
    for r in c.itertuples():
        b = read_bars(r.trade_date, r.symbol)
        exp = max(0, M_1555 - int(r.entry_min))                          # minutes m0+1 .. 955
        have = int(((b.m > r.entry_min) & (b.m <= M_1555)).sum())
        row = dict(id=r.id, trade_date=r.trade_date, symbol=r.symbol, realized_R=r.R, ledger_kind=r.exit_kind,
                   entry_min=r.entry_min, entry_proxy=r.entry_proxy, expected_min=exp, bars_present=have,
                   coverage=have / exp if exp else np.nan)
        if b.empty:
            lost += 1
            rows.append(row)
            continue
        o = replay_fill(r, b)
        row.update(open_1400=o["open_1400"], open_r_1400=o["open_r_1400"], eod_degraded=o["eod_degraded"],
                   live_kind=o["live"][0])
        for k in ("live", "v1", "v2", "v3", "v4", "v5", "v6"):
            row[f"{k}_R"] = (o[k][1] - r.fill_used) / r.rps
        rows.append(row)
    t = pd.DataFrame(rows)
    log(f"replayed {len(t)} fills; no-bars (LOST) {lost}")
    return t


# ---------------------------------------------------------------- stats
def ex_top5(y):
    """Mean after removing the top 5 % (rounded up) values."""
    y = np.sort(np.asarray(y, float))
    n = math.ceil(0.05 * len(y))
    return y[:-n].mean() if len(y) > n else np.nan


def tstat(y):
    """One-sample t of the mean against 0 (each fill is one trade)."""
    y = np.asarray(y, float)
    return y.mean() / (y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 2 and y.std() > 0 else np.nan


def score(t, k, half):
    """Paired score of variant k vs live replay on rows of t (half in {None,'A','B'})."""
    d = t if half is None else t[t.half == half]
    dl = (d[f"{k}_R"] - d["live_R"]).values
    cap = np.minimum(d[f"{k}_R"], 2.0).values - np.minimum(d["live_R"], 2.0).values
    se = dl.std(ddof=1) / math.sqrt(len(dl)) if len(dl) > 2 else np.nan
    return dict(n=len(dl), touched=int((np.abs(dl) > 1e-9).sum()), mean=dl.mean(), t=tstat(dl), extop5=ex_top5(dl),
                capped=cap.mean(), se=se)


def main():
    """Fetch, replay, gate, score and write RESULT_1724.md / variants_1724.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    c = load_fills()
    if not a.no_fetch:
        ensure_bars(c)
    t = build_table(c)
    t = t[t.live_R.notna()].copy()
    dates = sorted(t.trade_date.unique())
    t["half"] = t.trade_date.map({d: ("A" if i % 2 == 0 else "B") for i, d in enumerate(dates)})
    cov_all = t.bars_present.sum() / t.expected_min.sum()
    t["miss"] = 1 - t.coverage
    win, los = t[t.realized_R > 0], t[t.realized_R <= 0]
    gap_pp = (win.miss.mean() - los.miss.mean()) * 100
    low = t.groupby(["trade_date", "symbol"]).coverage.mean()
    n_low = int((low < 0.9).sum())
    gate_ok = cov_all >= 0.80 and abs(gap_pp) <= 5
    log(f"coverage {cov_all:.1%}; symbol-days <90% {n_low}/{len(low)}; miss gap W-L {gap_pp:+.2f}pp; gate {'OK' if gate_ok else 'VOID'}")
    t["reason_match"] = t.live_kind == t.ledger_kind
    t["r_close"] = (t.live_R - t.realized_R).abs() <= 0.15
    agree_reason, agree_r = t.reason_match.mean(), t.r_close.mean()
    log(f"replay agreement: reason {agree_reason:.1%}, within 0.15R {agree_r:.1%}")
    xt = pd.crosstab(t.ledger_kind, t.live_kind)
    t.to_csv(os.path.join(HERE, "variants_1724.csv"), index=False)
    trusted = min(agree_reason, agree_r) >= 0.8
    L = ["# RESULT cell 1,724 - time-aware exits on the live HOD paper book", "",
         f"Script `cell_1724.py`; PREREG_1724.md incl. Amendment 1. n = {len(t)} fills, {len(dates)} trade dates, "
         "Alpaca 1-min SIP bars (fill minute -> 16:00 ET). Mean R realized: "
         f"{t.realized_R.mean():+.3f}; live replay: {t.live_R.mean():+.3f}.", "",
         "## Coverage", "",
         f"* symbol-days {len(low)}; with < 90 % of expected minutes: {n_low}",
         f"* minutes present / expected (fill+1 .. 15:55): {cov_all:.1%}",
         f"* missing-minute share, winners (realized R>0, n {len(win)}) {win.miss.mean():.1%} vs losers (n {len(los)}) "
         f"{los.miss.mean():.1%}; gap {gap_pp:+.2f} pp",
         f"* gate (cov >= 80 % and |gap| <= 5 pp): **{'MET' if gate_ok else 'VOID'}**", ""]
    if not gate_ok:
        L.append("**All variants VOID per the PREREG coverage gate.**\n")
    L += ["## Live-rule replay vs ledger", "",
          f"* exit-reason match: {agree_reason:.1%}; |replay R - realized R| <= 0.15: {agree_r:.1%} "
          f"(trusted bar 80 %: {'PASS' if trusted else 'FAIL - PREREG says the cell stops here; variants below are for the record only'})",
          "* crosstab ledger (rows) x replay (cols):", "", "```", xt.to_string(), "```", ""]
    L += ["## Variants (paired Δ = variant R - live-replay R, per fill)", "",
          "| variant | n | touched | Δ mean | t | Δ ex-top5% | Δ capped(+2R) | A | B | 95% CI | MDE(2se) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"v1": "V1 time stop 14:00", "v2": "V2 take-profit 14:00", "v3": "V3 protect BE 14:00", "v4": "V4 1.5R target",
             "v5": "V5 decay 12:00->0.5R", "v6": "V6 decay 11:00->0.25R"}
    verdicts = {}
    for k, nm in names.items():
        p, sa, sb = score(t, k, None), score(t, k, "A"), score(t, k, "B")
        ok = bool(gate_ok and trusted and sa["mean"] >= 0.10 and sb["mean"] >= 0.10 and p["t"] >= 2 and p["extop5"] > 0)
        verdicts[k] = (ok, p["extop5"])
        L.append(f"| {nm} | {p['n']} | {p['touched']} | {p['mean']:+.3f} | {p['t']:.2f} | {p['extop5']:+.3f} | {p['capped']:+.3f} | "
                 f"{sa['mean']:+.3f} (n {sa['n']}) | {sb['mean']:+.3f} (n {sb['n']}) | "
                 f"[{p['mean'] - 1.96 * p['se']:+.3f}, {p['mean'] + 1.96 * p['se']:+.3f}] | {2 * p['se']:.3f} | "
                 f"{'**PASS**' if ok else 'FAIL' if gate_ok else 'VOID'} |")
    n14 = int(t.open_1400.sum())
    L += ["", f"Positions open at 14:00 in the replay (entered before 14:00, not yet exited): {n14} of {len(t)}; "
          f"of those with open P&L >= 0 R: {int((t.open_r_1400 >= 0).sum())}, >= +0.5 R: {int((t.open_r_1400 >= 0.5).sum())}.",
          "Pass bar: Δ >= +0.10 R in BOTH halves, pooled t >= 2, Δ ex-top-5 % > 0, coverage gate met. "
          "'touched' counts fills whose variant R differs from the live replay (nonzero Δ).", ""]
    winners = [k for k, (ok, _) in verdicts.items() if ok]
    if winners:
        best = max(winners, key=lambda k: verdicts[k][1])
        L += [f"**Passing: {', '.join(winners)}; ship {best} (higher ex-top-5 % Δ).**", "",
              "ENGINE CONDITION: (to be written by the coordinator from the variant definition in the PREREG).", ""]
    else:
        L += ["**No variant passes.** Δ with CI and the MDE at this n are above; no parameter sweeps.", ""]
    L += ["## Caveats on this implementation", "",
          "* Minute bars cannot order events inside a bar: only bars STRICTLY AFTER the fill minute are walked (same-minute stops/targets are missed); within a bar the stop is checked before the target (conservative).",
          "* Stop fill = min(bar open, stop) x (1 - 12 bps) (gap-through at the open is an addition to the frozen 'stop - 12 bps'); target fills at the target price even when the bar gaps above it.",
          "* EOD exit = close of the 15:55 bar with no extra cost; the ledger's real EOD fills (incl. 4 `eod_ops_ah` after-hours exits) carry their own slippage, so replay-vs-ledger R differs there.",
          "* Missing minute bars mean no SIP trade that minute (thin names), not necessarily missing data; a stop/target touch cannot be seen in a minute with no trade, and the 14:00 open-P&L uses the last close before 14:00.",
          "* 14:00 decision = last closed bar is the 13:59 bar; exit at the open of the first bar at/after 14:00, minus 12 bps. 'Open at 14:00' is taken from the replayed live rule, not the ledger.",
          "* V4/V5/V6 target level = entry_price (the intended level, same basis as the live target) + R x risk_per_share, so V5/V6 equal the live target while R_t = 2.0; the PREREG text says 'fill + target x R' - using the actual fill would add entry-slip noise to Δ. Decaying target uses the target at the bar's START minute (conservative, causal).",
          "* R = (exit - fill_used)/risk_per_share, risk_per_share = risk_eff/filled_qty; fill_used = actual fill, else entry_price (3 rows). Entry minute for rows with no filled_at = created_at minute.",
          "* One same-name-same-day duplicate shares one bar file fetched from the earliest fill minute.",
          "* Paired t is iid over fills; fills on the same day are correlated, so t is optimistic. 6 variants scored, no multiplicity correction.",
          "* No independent rebuild was run inside this cell's budget; replay agreement with the ledger is the only external check."]
    open(os.path.join(HERE, "RESULT_1724.md"), "w").write("\n".join(L) + "\n")
    log("wrote RESULT_1724.md and variants_1724.csv")


if __name__ == "__main__":
    main()
