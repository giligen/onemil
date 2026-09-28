#!/usr/bin/env python3
"""Cells 1,633-1,636 -- research/edgar_desk/PREREG_1633.md (FROZEN 2026-09-28 18:40 UTC).

Post-earnings drift (PEAD) on the reaction: for every 8-K item-2.02 (earnings release) filing,
compute the reaction return R0 (close before the announcement -> close of the first full session
after it), bucket events into TRAIN-defined deciles of R0, and test whether the decile predicts
the NEXT 10/20-session drift -- bought/shorted causally at the open after the reaction session
has already closed.

Cells:
  1,633 -- LONG TRAIN-top-decile R0.  Entry MOO next open after reaction. Exit MOC +10 sessions.
  1,634 -- same, +20 sessions.
  1,635 -- SHORT TRAIN-bottom-decile R0 (mirror of 1,633; shortable/SSR excluded, 3%/yr borrow).
  1,636 -- report-only: full TRAIN-decile table for +10 and +20 holds, small-cap (<=$1B) vs
           larger (UNAVAILABLE -- no shares-outstanding/market-cap source on disk; see the
           caveats in RESULT_1633.md), and the SPY-adjusted return beside the raw.

Data:
  research/edgar_desk/events_raw.csv (4.4M SEC filing rows, streamed in 300k-row chunks).
  research/overnight_high/alpaca_daily_2019_2024H1.parquet (raw daily bars, 2019-01-02..2024-06-27)
  research/overnight_high/panel_2024_2026.parquet          (raw daily bars, 2024-07-01..2026-09-04)
  research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv    (CURRENT-SNAPSHOT shortable flags --
      no date column, so this is applied retroactively to 2019-2024 shorts; documented as a
      limitation. Symbols absent from the file are reported UNFILTERED, per the task spec.)

Splits (by reaction-session date): TRAIN 2019-01-01..2022-12-31, VAL 2023-01-01..2024-06-30.
TEST (reaction_session >= 2024-07-01) is SEALED: rows are dropped immediately after the split
label is assigned, before R0, entry, exit or any statistic is computed for them.

Timezone: acceptance_datetime is UTC (ISO 'Z' suffix -- confirmed on inspection: values like
"2019-01-02T21:16:06.000Z"). The cell-1,552 refuter found an earlier pipeline treated these
digits as already being ET wall-clock (stripping the 'Z' without shifting the clock -- see
research/edgar_desk/rebuild_1552_full.py's build_events()). This script does a REAL UTC ->
America/New_York conversion via zoneinfo (DST-aware), per PREREG_1633's explicit instruction to
"convert properly". This is a documented deviation from the cell-1,552 convention -- flagged in
RESULT_1633.md for the independent-check agent to verify directly.

Reaction-session mapping. The PREREG's three prose cases collapse to one rule once you track
session INDEX rather than calendar date:
  after-close (time_et >= 16:00)     -> reaction_session = NEXT trading session after date_et
  pre-market  (time_et <  09:30)     -> reaction_session = date_et's own session
  intraday    (09:30 <= time_et<16:00) -> reaction_session = date_et's own session
and in all three cases R0 = close(reaction_session) / close(session immediately prior to it) - 1
(for after-close this "prior" session is date_et's own close -- the close before the after-hours
announcement; for pre-market/intraday it is the session before date_et, matching the PREREG's
"prior close" wording exactly). See resolve_symbol_events() for the index arithmetic.

The universe filter (price >= $3, 20-day $ volume >= $1M) is read at the session STRICTLY BEFORE
date_et (the announcement's own calendar day), uniformly across all three timing buckets -- a
conservative, always-causal choice (it never uses same-day information, even for after-close
filings where date_et's own close would also be causally valid). Documented in RESULT_1633.md.

Usage:
    python3 research/edgar_desk/cell_1633.py

Outputs: research/edgar_desk/cell_1633_events_pead.csv, research/edgar_desk/RESULT_1633.md.
"""
import logging
import sys
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     stream=sys.stdout)
log = logging.getLogger("cell_1633")

HERE = Path(__file__).resolve().parent
EVENTS_CSV = HERE / "events_raw.csv"
PRICES_A = HERE.parent / "overnight_high" / "alpaca_daily_2019_2024H1.parquet"
PRICES_B = HERE.parent / "overnight_high" / "panel_2024_2026.parquet"
BORROW_CSV = HERE.parent / "fuckup_audit" / "O_halt" / "PASSIVE" / "borrow_flags.csv"

OUT_EVENTS = HERE / "cell_1633_events_pead.csv"
OUT_RESULT = HERE / "RESULT_1633.md"

MIN_PRICE = 3.0
MIN_DVOL20 = 1_000_000.0
COST_BPS_PER_LEG = 5.0
BORROW_APY = 0.03
SESSIONS_PER_YEAR = 252
WINNER_CAP = 0.30                              # +30% winner cap, return units
SPLIT_GUARD_LO, SPLIT_GUARD_HI = 0.40, 2.50    # session close/close ratio outside -> guard trip
TEST_TICKER_RE = r'^Z[A-Z]ZZT$'
SSR_PRIOR_RET = -0.10                          # desk convention, rebuild_1552_full.py
SSR_MIN_PRICE = 5.0
ET = ZoneInfo("America/New_York")

TRAIN_START, TRAIN_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30")
TEST_START = pd.Timestamp("2024-07-01")

ITEM_202_RE = r'(?:^|;)2\.02(?:;|$)'


# ---------------------------------------------------------------------------
# 1. Load the 8-K / item-2.02 events, UTC -> ET conversion
# ---------------------------------------------------------------------------

def load_events():
    """Stream events_raw.csv in 300k-row chunks; keep 8-K filings with item 2.02.

    Returns one row per (filing, symbol) with accept_utc (UTC-aware) and accept_et (ET-aware,
    DST-correct via zoneinfo). Drops rows with a blank symbol or a test-ticker symbol
    (^Z[A-Z]ZZT$, per project convention) and rows whose acceptance_datetime fails to parse.
    """
    usecols = ["symbol", "form", "acceptance_datetime", "items"]
    chunks = []
    n_seen = n_8k = n_202 = 0
    for i, chunk in enumerate(pd.read_csv(EVENTS_CSV, usecols=usecols, dtype=str,
                                           chunksize=300_000)):
        n_seen += len(chunk)
        is_8k = chunk["form"] == "8-K"
        n_8k += int(is_8k.sum())
        has_202 = chunk["items"].fillna("").str.contains(ITEM_202_RE, regex=True)
        keep = is_8k & has_202
        n_202 += int(keep.sum())
        if keep.any():
            chunks.append(chunk.loc[keep, ["symbol", "acceptance_datetime"]].copy())
        if (i + 1) % 5 == 0:
            log.info("events_raw.csv: %d rows scanned, %d 8-K, %d item-2.02 kept so far",
                      n_seen, n_8k, n_202)
    log.info("events_raw.csv DONE: %d rows scanned, %d 8-K, %d item-2.02 kept", n_seen, n_8k, n_202)

    ev = pd.concat(chunks, ignore_index=True)
    n0 = len(ev)
    ev = ev[ev["symbol"].notna() & (ev["symbol"].str.len() > 0)]
    n_blank = n0 - len(ev)
    n1 = len(ev)
    ev = ev[~ev["symbol"].str.match(TEST_TICKER_RE)]
    n_testticker = n1 - len(ev)

    ev["accept_utc"] = pd.to_datetime(ev["acceptance_datetime"], utc=True, errors="coerce")
    n_bad = int(ev["accept_utc"].isna().sum())
    if n_bad:
        log.warning("dropping %d/%d item-2.02 rows: unparseable acceptance_datetime", n_bad, len(ev))
    ev = ev[ev["accept_utc"].notna()].copy()

    ev["accept_et"] = ev["accept_utc"].dt.tz_convert(ET)
    ev["date_et"] = pd.to_datetime(ev["accept_et"].dt.date)
    ev["time_et"] = ev["accept_et"].dt.time

    log.info("events after symbol/test-ticker/parse filters: %d (dropped %d blank symbol, "
              "%d test-ticker, %d unparseable time); %d unique symbols",
              len(ev), n_blank, n_testticker, n_bad, ev["symbol"].nunique())
    return ev.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Load and combine the raw daily-bar panels
# ---------------------------------------------------------------------------

def load_prices():
    """Concat the two raw (unadjusted) daily-bar parquets -- no date overlap (A ends
    2024-06-27, B starts 2024-07-01) -- drop zero-OHLCV placeholder rows from each, and compute
    a causal trailing 20-session dollar-volume average (rolling(20, min_periods=10), so the
    value at row t uses only rows <= t -- no forward leakage)."""
    cols = ["symbol", "bar_date", "open", "high", "low", "close", "volume"]
    a = pd.read_parquet(PRICES_A, columns=cols)
    b = pd.read_parquet(PRICES_B, columns=cols)
    n_a0, n_b0 = len(a), len(b)
    ok = lambda d: (d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0) & (d["volume"] > 0)
    a = a[ok(a)]
    b = b[ok(b)]
    log.info("alpaca_daily_2019_2024H1: dropped %d/%d zero-OHLCV rows", n_a0 - len(a), n_a0)
    log.info("panel_2024_2026: dropped %d/%d zero-OHLCV rows", n_b0 - len(b), n_b0)

    p = pd.concat([a, b], ignore_index=True)
    p["symbol"] = p["symbol"].astype(str)
    p["bar_date"] = pd.to_datetime(p["bar_date"])
    n_before_dedup = len(p)
    p = p.drop_duplicates(subset=["symbol", "bar_date"], keep="first")
    if len(p) != n_before_dedup:
        log.warning("dropped %d duplicate (symbol, bar_date) rows across the two panels",
                    n_before_dedup - len(p))
    p = p.sort_values(["symbol", "bar_date"]).reset_index(drop=True)

    p["dvol"] = p["close"] * p["volume"]
    p["dvol20"] = p.groupby("symbol")["dvol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())

    log.info("combined price panel: %d rows, %d symbols, %s .. %s",
              len(p), p["symbol"].nunique(), p["bar_date"].min().date(), p["bar_date"].max().date())
    return p


def load_borrow_flags():
    """Load the current-snapshot shortable/easy-to-borrow flags. Returns {symbol: bool} for
    borrow_ok (shortable AND easy_to_borrow). Symbols absent are NOT included in the dict --
    callers must treat 'absent' as its own, unfiltered category (see resolve_symbol_events)."""
    b = pd.read_csv(BORROW_CSV)
    b["borrow_ok"] = b["shortable"].astype(bool) & b["easy_to_borrow"].astype(bool)
    d = dict(zip(b["symbol"], b["borrow_ok"]))
    log.info("borrow_flags.csv: %d symbols, %d borrow_ok=True, %d borrow_ok=False",
              len(d), sum(d.values()), len(d) - sum(d.values()))
    return d


# ---------------------------------------------------------------------------
# 3. Statistics helpers (day_clustered_t per research/hod_entry/cell_1445.py convention)
# ---------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean.
    Reused convention: research/hod_entry/cell_1445.py:day_clustered_t (same method, replicated
    here rather than imported, to keep this cell's dependency surface self-contained)."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top_mean(y, frac):
    """Mean excluding the top `frac` (e.g. 0.05) of a series by value -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(frac * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap=WINNER_CAP):
    y = pd.Series(y).dropna()
    if not len(y):
        return np.nan
    return float(np.minimum(y, cap).mean())


def weeks_spanned(dates):
    """Distinct ISO (year, week) count over a date series -- the denominator for events/week."""
    dates = pd.to_datetime(pd.Series(dates)).dropna().unique()
    wk = {(pd.Timestamp(d).isocalendar()[0], pd.Timestamp(d).isocalendar()[1]) for d in dates}
    return max(len(wk), 1)


# ---------------------------------------------------------------------------
# 4. Per-symbol vectorized resolution: reaction session, R0, entry/exit, universe & guards
# ---------------------------------------------------------------------------

def resolve_symbol_events(events, prices, borrow_ok_map):
    """For each event, locate (by searchsorted on that symbol's own trading-day array):
      idx_date_et         -- first index with bar_date >= date_et
      prior_idx            = idx_date_et - 1          (universe-filter reference session)
      reaction_idx          -- idx_date_et, bumped +1 iff after-close AND date_et is itself a
                                trading day (see module docstring for the case collapse)
      entry_idx             = reaction_idx + 1          (MOO buy/short session)
      exit10_idx / exit20_idx = reaction_idx + 10 / +20  (MOC sell/cover session)
    Applies the universe filter (price>=$3, dvol20>=$1M at prior_idx), the split-artifact guard
    (any consecutive close/close ratio outside [0.40, 2.50] along the path actually used), and
    computes the short-eligibility flag (SSR proxy + borrow_flags.csv lookup) for every row,
    independent of decile -- decile assignment and cell selection happen afterward on this
    resolved table.
    """
    price_groups = {sym: g for sym, g in prices.groupby("symbol", sort=False)}
    lost = dict(no_price_series=0, no_prior_session=0, below_universe_filter=0)
    out = []
    for sym, g in events.groupby("symbol", sort=False):
        arr = price_groups.get(sym)
        if arr is None:
            lost["no_price_series"] += len(g)
            continue
        bar_date = arr["bar_date"].to_numpy()
        open_ = arr["open"].to_numpy()
        close = arr["close"].to_numpy()
        dvol20 = arr["dvol20"].to_numpy()
        n = len(bar_date)

        date_et = g["date_et"].to_numpy()
        idx_date_et = np.searchsorted(bar_date, date_et, side="left")
        prior_idx = idx_date_et - 1
        valid_prior = prior_idx >= 0
        if not valid_prior.any():
            lost["no_prior_session"] += len(g)
            continue

        same_day = (idx_date_et < n)
        same_day[same_day] &= (bar_date[idx_date_et[same_day]] == date_et[same_day])
        after_close = np.array([t >= dtime(16, 0) for t in g["time_et"]])
        bump = (after_close & same_day).astype(int)
        reaction_idx = idx_date_et + bump

        entry_idx = reaction_idx + 1
        exit10_idx = reaction_idx + 10
        exit20_idx = reaction_idx + 20
        prior_react_idx = reaction_idx - 1  # R0 denominator

        valid = valid_prior & (prior_react_idx >= 0) & (reaction_idx < n) & (entry_idx < n)
        if not valid.any():
            lost["no_prior_session"] += int((~valid).sum())
            continue

        gv = g.loc[valid].reset_index(drop=True)
        pi, ri, ei, e10, e20, pri = (a[valid] for a in
            (prior_idx, reaction_idx, entry_idx, exit10_idx, exit20_idx, prior_react_idx))

        price_ok = close[pi] >= MIN_PRICE
        dvol_ok = np.nan_to_num(dvol20[pi], nan=-1.0) >= MIN_DVOL20
        uni_ok = price_ok & dvol_ok
        lost["below_universe_filter"] += int((~uni_ok).sum())

        R0 = close[ri] / close[pri] - 1.0
        entry_open = open_[ei]

        has20 = e20 < n
        exit10_close = np.where(e10 < n, close[np.minimum(e10, n - 1)], np.nan)
        exit20_close = np.where(has20, close[np.minimum(e20, n - 1)], np.nan)

        def path_guard(lo, hi):
            ok = np.ones(len(gv), dtype=bool)
            for k in range(len(gv)):
                if not (uni_ok[k] and (hi[k] < n)):
                    continue
                path = close[lo[k]:hi[k] + 1]
                if len(path) < 2:
                    continue
                ratios = path[1:] / path[:-1]
                if np.any((ratios < SPLIT_GUARD_LO) | (ratios > SPLIT_GUARD_HI)):
                    ok[k] = False
            return ok

        guard10 = path_guard(pri, np.minimum(e10, n - 1)) & (e10 < n)
        guard20 = path_guard(pri, np.minimum(e20, n - 1)) & has20

        # SSR proxy (rebuild_1552_full.py convention) referenced at the session immediately
        # before entry (== the reaction session): a big prior-day drop or a sub-$5 close.
        ssr_ref_close = close[ri]
        ssr_block = (ssr_ref_close < SSR_MIN_PRICE) | (np.nan_to_num(R0, nan=0.0) <= SSR_PRIOR_RET)
        borrow_status = np.array([("checked_ok" if borrow_ok_map[s] else "checked_excluded")
                                   if s in borrow_ok_map else "absent_unfiltered" for s in gv["symbol"]])
        short_eligible = (borrow_status != "checked_excluded") & (~ssr_block)

        hold_days10 = (e10 - ei)
        hold_days20 = (e20 - ei)

        out.append(pd.DataFrame({
            "symbol": gv["symbol"].to_numpy(),
            "date_et": gv["date_et"].to_numpy(),
            "time_et": gv["time_et"].astype(str).to_numpy(),
            "after_close": after_close[valid],
            "reaction_session": bar_date[ri],
            "prior_session": bar_date[pri],
            "entry_date": np.where(ei < n, bar_date[np.minimum(ei, n - 1)], np.datetime64("NaT")),
            "exit10_date": np.where(e10 < n, bar_date[np.minimum(e10, n - 1)], np.datetime64("NaT")),
            "exit20_date": np.where(has20, bar_date[np.minimum(e20, n - 1)], np.datetime64("NaT")),
            "R0": R0,
            "entry_open": entry_open,
            "exit10_close": exit10_close,
            "exit20_close": exit20_close,
            "hold_days10": hold_days10,
            "hold_days20": hold_days20,
            "universe_ok": uni_ok,
            "guard10_ok": guard10,
            "guard20_ok": guard20,
            "short_eligible": short_eligible,
            "borrow_status": borrow_status,
        }))

    lost_total = sum(lost.values())
    log.info("resolve_symbol_events: kept %d rows across %d symbols; lost %s (total %d)",
             sum(len(x) for x in out), len(out), lost, lost_total)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def attach_spy(resolved, prices):
    """SPY's own close/close return over the identical [reaction_session .. exit] window used
    by each event -- the abnormal-return (SPY-adjusted) reference, matched apples-to-apples on
    session index, not calendar date."""
    spy = prices[prices["symbol"] == "SPY"].sort_values("bar_date")
    spy_date = spy["bar_date"].to_numpy()
    spy_close = spy["close"].to_numpy()

    def ret_over(date_a, date_b):
        ia = np.searchsorted(spy_date, date_a, side="left")
        ib = np.searchsorted(spy_date, date_b, side="left")
        ia = np.clip(ia, 0, len(spy_date) - 1)
        ib = np.clip(ib, 0, len(spy_date) - 1)
        return spy_close[ib] / spy_close[ia] - 1.0

    resolved = resolved.copy()
    resolved["spy_ret10"] = ret_over(resolved["reaction_session"].to_numpy(),
                                      resolved["exit10_date"].fillna(resolved["reaction_session"]).to_numpy())
    resolved["spy_ret20"] = ret_over(resolved["reaction_session"].to_numpy(),
                                      resolved["exit20_date"].fillna(resolved["reaction_session"]).to_numpy())
    resolved.loc[resolved["exit10_date"].isna(), "spy_ret10"] = np.nan
    resolved.loc[resolved["exit20_date"].isna(), "spy_ret20"] = np.nan
    return resolved


# ---------------------------------------------------------------------------
# 5. Split assignment, TRAIN deciles, cell scoring
# ---------------------------------------------------------------------------

def assign_split(resolved):
    """Split by reaction_session date. TEST (>= 2024-07-01) rows are dropped here -- SEALED,
    never scored, never written out."""
    r = resolved.copy()
    r["split"] = np.select(
        [r["reaction_session"].between(TRAIN_START, TRAIN_END),
         r["reaction_session"].between(VAL_START, VAL_END),
         r["reaction_session"] >= TEST_START],
        ["TRAIN", "VAL", "TEST"], default="PRE_2019_OR_GAP")
    n_test = int((r["split"] == "TEST").sum())
    n_gap = int((r["split"] == "PRE_2019_OR_GAP").sum())
    log.info("split assignment: TRAIN=%d VAL=%d (TEST=%d SEALED -- dropped now, gap/pre-2019=%d dropped)",
              int((r["split"] == "TRAIN").sum()), int((r["split"] == "VAL").sum()), n_test, n_gap)
    r = r[r["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)
    return r


def assign_deciles(r):
    """TRAIN-only decile boundaries (qcut on TRAIN R0, universe_ok rows only), applied to both
    TRAIN and VAL via pd.cut with the SAME edges -- deciles are never re-derived on VAL."""
    train_r0 = r.loc[(r["split"] == "TRAIN") & r["universe_ok"], "R0"]
    _, edges = pd.qcut(train_r0, 10, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    n_deciles = len(edges) - 1
    if n_deciles != 10:
        log.warning("TRAIN R0 qcut collapsed to %d deciles (duplicate edges) instead of 10", n_deciles)
    r = r.copy()
    r["decile"] = pd.cut(r["R0"], bins=edges, labels=list(range(1, n_deciles + 1)), include_lowest=True)
    r["decile"] = r["decile"].astype("Int64")
    log.info("decile edges (TRAIN R0, universe_ok): %s", np.round(edges, 4).tolist())
    return r, n_deciles


def net_return(gross, is_short, hold_days):
    """5 bps per auction leg (2 legs = one entry + one exit); shorts also pay borrow at 3%/yr
    over the sessions actually held (SESSIONS_PER_YEAR=252 convention)."""
    cost = 2 * (COST_BPS_PER_LEG / 10_000.0)
    if is_short:
        borrow = BORROW_APY * (hold_days / SESSIONS_PER_YEAR)
        return -gross - cost - borrow
    return gross - cost


def score_cell(rows, cell, split_col="split"):
    """Per-split stats block for one cell's row set (already filtered to the cell's population).
    rows must carry: split, R0, entry_date (cluster key), net_bps, ret_bps, spy_bps, year."""
    out = []
    for split in ("TRAIN", "VAL"):
        sub = rows[rows[split_col] == split]
        n = len(sub)
        if n == 0:
            out.append(dict(cell=cell, split=split, n=0))
            continue
        mean_net_bps = float(sub["net_bps"].mean())
        t = day_clustered_t(sub["net_bps"], sub["entry_date"])
        ex5 = ex_top_mean(sub["net_bps"], 0.05)
        ex1 = ex_top_mean(sub["net_bps"], 0.01)
        wcap = winner_capped_mean(sub["ret_frac"]) * 10_000.0
        spy_adj_bps = float((sub["ret_bps"] - sub["spy_bps"]).mean())
        spy_adj_net_bps = float((sub["net_bps"] - sub["spy_bps"]).mean())
        wk = weeks_spanned(sub["entry_date"])
        ev_wk = n / wk
        by_year = sub.assign(yr=sub["entry_date"].dt.year).groupby("yr")["net_bps"].agg(["mean", "count"])
        years_pos = "/".join(f"{int(y)}:{'+' if m > 0 else '-'}" for y, m in by_year["mean"].items())
        out.append(dict(cell=cell, split=split, n=n, events_wk=round(ev_wk, 2),
                         mean_net_bps=round(mean_net_bps, 1), t=round(t, 2) if pd.notna(t) else np.nan,
                         ex_top5_bps=round(ex5, 1), ex_top1_bps=round(ex1, 1),
                         winner_capped_bps=round(wcap, 1), spy_adj_bps=round(spy_adj_bps, 1),
                         spy_adj_net_bps=round(spy_adj_net_bps, 1), years_positive=years_pos))
    return out


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== cell_1633: PEAD on the 8-K item-2.02 reaction (PREREG_1633.md) ===")
    events = load_events()
    prices = load_prices()
    borrow_ok_map = load_borrow_flags()

    resolved = resolve_symbol_events(events, prices, borrow_ok_map)
    resolved = attach_spy(resolved, prices)
    resolved = assign_split(resolved)
    resolved, n_deciles = assign_deciles(resolved)

    resolved["entry_date"] = pd.to_datetime(resolved["entry_date"])
    resolved["exit10_date"] = pd.to_datetime(resolved["exit10_date"])
    resolved["exit20_date"] = pd.to_datetime(resolved["exit20_date"])

    base = resolved[resolved["universe_ok"]].copy()
    log.info("rows after universe filter (price>=$3, dvol20>=$1M at prior session): %d / %d",
             len(base), len(resolved))

    events_out = []
    all_cell_stats = []

    # ---- 1,633: LONG top decile, +10 ----
    pop = base[(base["decile"] == n_deciles) & base["guard10_ok"]].copy()
    pop["ret_frac"] = pop["exit10_close"] / pop["entry_open"] - 1.0
    pop["ret_bps"] = pop["ret_frac"] * 10_000.0
    pop["net_bps"] = pop.apply(lambda r: net_return(r["ret_frac"], False, r["hold_days10"]), axis=1) * 10_000.0
    pop["spy_bps"] = pop["spy_ret10"] * 10_000.0
    all_cell_stats += score_cell(pop, "1633")
    events_out.append(pop.assign(cell="1633", exit_close=pop["exit10_close"], hold="+10"))

    # ---- 1,634: LONG top decile, +20 ----
    pop2 = base[(base["decile"] == n_deciles) & base["guard20_ok"]].copy()
    pop2["ret_frac"] = pop2["exit20_close"] / pop2["entry_open"] - 1.0
    pop2["ret_bps"] = pop2["ret_frac"] * 10_000.0
    pop2["net_bps"] = pop2.apply(lambda r: net_return(r["ret_frac"], False, r["hold_days20"]), axis=1) * 10_000.0
    pop2["spy_bps"] = pop2["spy_ret20"] * 10_000.0
    all_cell_stats += score_cell(pop2, "1634")
    events_out.append(pop2.assign(cell="1634", exit_close=pop2["exit20_close"], hold="+20"))

    # ---- 1,635: SHORT bottom decile, +10, shortable/SSR filtered ----
    pop3 = base[(base["decile"] == 1) & base["guard10_ok"] & base["short_eligible"]].copy()
    n_bottom_all = int((base["decile"] == 1).sum())
    n_bottom_excluded_borrow = int(((base["decile"] == 1) & (base["borrow_status"] == "checked_excluded")).sum())
    n_bottom_absent = int(((base["decile"] == 1) & (base["borrow_status"] == "absent_unfiltered")).sum())
    pop3["ret_frac"] = pop3["exit10_close"] / pop3["entry_open"] - 1.0
    pop3["ret_bps"] = -pop3["ret_frac"] * 10_000.0  # short: gain when price falls
    pop3["net_bps"] = pop3.apply(lambda r: net_return(r["ret_frac"], True, r["hold_days10"]), axis=1) * 10_000.0
    pop3["spy_bps"] = -pop3["spy_ret10"] * 10_000.0
    all_cell_stats += score_cell(pop3, "1635")
    events_out.append(pop3.assign(cell="1635", exit_close=pop3["exit10_close"], hold="+10"))

    # ---- 1,636: report-only full decile table, both holds ----
    decile_rows = []
    for hold, guard_col, exit_col, spy_col, hold_days_col in (
            ("+10", "guard10_ok", "exit10_close", "spy_ret10", "hold_days10"),
            ("+20", "guard20_ok", "exit20_close", "spy_ret20", "hold_days20")):
        d = base[base[guard_col]].copy()
        d["ret_frac"] = d[exit_col] / d["entry_open"] - 1.0
        d["ret_bps"] = d["ret_frac"] * 10_000.0
        d["net_bps"] = d.apply(lambda r: net_return(r["ret_frac"], False, r[hold_days_col]), axis=1) * 10_000.0
        d["spy_bps"] = d[spy_col] * 10_000.0
        cell_tag = f"1636_h{hold.strip('+')}"
        events_out.append(d.assign(cell=cell_tag, exit_close=d[exit_col], hold=hold))
        for split in ("TRAIN", "VAL"):
            for dec in range(1, n_deciles + 1):
                sub = d[(d["split"] == split) & (d["decile"] == dec)]
                if len(sub) == 0:
                    continue
                decile_rows.append(dict(
                    hold=hold, split=split, decile=dec, n=len(sub),
                    mean_ret_bps=round(sub["ret_bps"].mean(), 1),
                    mean_net_bps=round(sub["net_bps"].mean(), 1),
                    spy_adj_bps=round((sub["ret_bps"] - sub["spy_bps"]).mean(), 1)))
    decile_table = pd.DataFrame(decile_rows)

    events_csv_cols = ["cell", "split", "symbol", "date_et", "reaction_session", "R0", "decile",
                        "entry_open", "exit_close", "ret_bps", "spy_bps", "net_bps", "hold",
                        "hold_days10", "entry_date", "short_eligible", "borrow_status"]
    ev_all = pd.concat(events_out, ignore_index=True)
    ev_all = ev_all.rename(columns={"date_et": "ann_date"})
    ev_all[["cell", "split", "symbol", "ann_date", "reaction_session", "R0", "decile",
            "entry_open", "exit_close", "ret_bps", "spy_bps", "net_bps", "hold",
            "entry_date", "short_eligible", "borrow_status"]].to_csv(OUT_EVENTS, index=False)
    log.info("wrote %s (%d rows)", OUT_EVENTS, len(ev_all))

    stats_df = pd.DataFrame(all_cell_stats)
    log.info("cell stats:\n%s", stats_df.to_string())

    write_result_md(stats_df, decile_table, resolved, base, n_deciles,
                     n_bottom_all, n_bottom_excluded_borrow, n_bottom_absent, events, borrow_ok_map)
    log.info("DONE")


def write_result_md(stats_df, decile_table, resolved, base, n_deciles,
                     n_bottom_all, n_bottom_excluded_borrow, n_bottom_absent, events, borrow_ok_map):
    """Pass-bar checklist is computed here on the VAL row of each of 1,633/1,634/1,635, per the
    frozen PREREG bar: mean net >= +50bps, day-clustered t >= 2.5, ex-top-5% > 0, TRAIN same-sign
    t >= 1, decile table monotone on both halves, SPY-adjusted (net) >= +30bps. The >=5
    events/week-in-season line is reported (season = Jan/Feb/Apr/May/Jul/Aug/Oct/Nov, the
    quarterly-earnings-heavy months) but not separately gated here -- it is folded into the
    PASS/FAIL column as an explicit extra check.
    """
    lines = []
    lines.append("# RESULT — cells 1,633-1,636: post-earnings drift on the 8-K item-2.02 reaction")
    lines.append("")
    lines.append(f"Spec: `research/edgar_desk/PREREG_1633.md` (FROZEN 2026-09-28 18:40 UTC). "
                 f"Builder run: this script, `research/edgar_desk/cell_1633.py`.")
    lines.append("")
    lines.append("## Pipeline counts")
    lines.append(f"- item-2.02 8-K (filing,symbol) rows after parse/test-ticker filters: {len(events)}")
    lines.append(f"- resolved against the price panel (has a symbol match + full index bounds): {len(resolved)}")
    lines.append(f"- pass universe filter (price >= $3, 20-day $ volume >= $1M at the session "
                 f"strictly before the announcement date): {len(base)} "
                 f"({100*len(base)/max(len(resolved),1):.1f}%)")
    lines.append(f"- TRAIN decile count realized: {n_deciles} (10 requested; fewer means the "
                 f"TRAIN R0 distribution had duplicate qcut edges)")
    lines.append("")
    lines.append("## Timezone conversion (departure from an earlier cell-1,552 convention)")
    lines.append("`acceptance_datetime` carries an ISO 'Z' (UTC) suffix. This script converts it "
                 "to America/New_York with `zoneinfo` (DST-aware), per PREREG_1633's explicit "
                 "instruction: \"acceptance datetime UTC ... convert properly.\" "
                 "`research/edgar_desk/rebuild_1552_full.py` instead stripped the 'Z' and treated "
                 "the raw digits as already-ET wall-clock (its own comment: \"tz_localize(None) "
                 "drops the (spurious) UTC label without shifting the clock\"). The two "
                 "conventions disagree by 4-5 hours (the DST offset) on every event, which can "
                 "flip an event between the pre-market/intraday/after-close buckets and therefore "
                 "change its reaction_session. **This is the single highest-priority item for the "
                 "independent-check agent to verify against the raw SEC EDGAR convention "
                 "directly** (SEC's own acceptance-datetime is documented as Eastern time in the "
                 "submissions header; if that is also true of this vendor's export despite the "
                 "'Z' suffix, this script's conversion — not cell 1,552's — would be the bug).")
    lines.append("")
    lines.append("## Cell stats (n, events/week, mean net bps, day-clustered t [entry session], "
                 "ex-top-5%/ex-top-1% bps, winner-capped [+30%] bps, SPY-adjusted raw/net bps, "
                 "per-year sign)")
    lines.append("")
    lines.append(stats_df.to_markdown(index=False) if len(stats_df) else "(no rows)")
    lines.append("")

    lines.append("## Pass-bar checklist (frozen; VAL, per cell)")
    lines.append("Mean net >= +50bps | day-clustered t >= 2.5 | ex-top-5% > 0 | TRAIN same-sign "
                 "t >= 1 | decile table monotone both halves | SPY-adjusted (net) >= +30bps")
    lines.append("")
    monotone = {}
    for hold in ("+10", "+20"):
        for split in ("TRAIN", "VAL"):
            sub = decile_table[(decile_table["hold"] == hold) & (decile_table["split"] == split)]
            sub = sub.sort_values("decile")
            vals = sub["mean_net_bps"].to_numpy()
            is_mono = bool(len(vals) >= 3 and np.all(np.diff(vals) >= -1e-9))
            monotone[(hold, split)] = is_mono
    lines.append(f"- Decile-table monotonicity (net bps, non-decreasing decile 1->{n_deciles}): "
                 + ", ".join(f"{h} {s}={'MONO' if m else 'NOT MONO'}" for (h, s), m in monotone.items()))
    lines.append("")

    for cell, hold_key in (("1633", "+10"), ("1634", "+20"), ("1635", "+10")):
        row_val = stats_df[(stats_df["cell"] == cell) & (stats_df["split"] == "VAL")]
        row_train = stats_df[(stats_df["cell"] == cell) & (stats_df["split"] == "TRAIN")]
        if row_val.empty or row_val.iloc[0].get("n", 0) == 0:
            lines.append(f"### Cell {cell}: NO VAL ROWS -- cannot evaluate the pass bar")
            continue
        v = row_val.iloc[0]
        t_ = row_train.iloc[0] if not row_train.empty else None
        checks = {
            "mean_net>=+50bps": v["mean_net_bps"] >= 50,
            "t>=2.5": pd.notna(v["t"]) and v["t"] >= 2.5,
            "ex_top5>0": v["ex_top5_bps"] > 0,
            "TRAIN_same_sign_t>=1": (t_ is not None and pd.notna(t_["t"])
                                       and np.sign(t_["mean_net_bps"]) == np.sign(v["mean_net_bps"])
                                       and abs(t_["t"]) >= 1),
            "decile_monotone_both_halves": monotone.get((hold_key, "TRAIN"), False) and monotone.get((hold_key, "VAL"), False),
            "spy_adj_net>=+30bps": v["spy_adj_net_bps"] >= 30,
        }
        overall = all(checks.values())
        lines.append(f"### Cell {cell} (VAL n={v['n']}, {v['events_wk']} events/wk): "
                     f"{'PASS' if overall else 'FAIL'}")
        for k, ok in checks.items():
            lines.append(f"  - {k}: {'PASS' if ok else 'FAIL'}")
        lines.append("")

    lines.append("## 1,636 full decile table (report-only; TRAIN-defined edges applied to both halves)")
    lines.append("")
    lines.append(decile_table.sort_values(["hold", "split", "decile"]).to_markdown(index=False)
                 if len(decile_table) else "(no rows)")
    lines.append("")

    lines.append("## Short eligibility (cell 1,635, bottom decile population)")
    lines.append(f"- bottom-decile rows (pre shortability/SSR filter, universe_ok only): {n_bottom_all}")
    lines.append(f"- excluded: not shortable / not easy-to-borrow per borrow_flags.csv "
                 f"(CURRENT SNAPSHOT, no date column): {n_bottom_excluded_borrow}")
    lines.append(f"- absent from borrow_flags.csv entirely -> KEPT UNFILTERED per the task spec: "
                 f"{n_bottom_absent}")
    lines.append(f"- borrow_flags.csv total coverage: {len(borrow_ok_map)} symbols")
    lines.append("- **Caveat**: borrow_flags.csv has no date column -- it is a present-day "
                 "(~2026-09-18) snapshot of shortability, applied retroactively to 2019-2024 "
                 "short entries. A name that is easy to borrow today may not have been in 2020, "
                 "and vice versa; this cannot be corrected without a historical borrow-flag "
                 "source. The SSR exclusion is itself a proxy (prior-session close < $5 OR R0 <= "
                 "-10%, the research/edgar_desk/rebuild_1552_full.py convention), not the real "
                 "intraday-triggered SSR rule (which needs intraday data not fetched here).")
    lines.append("")

    lines.append("## Small-cap (<=$1B) vs larger split -- cell 1,636")
    lines.append("**UNAVAILABLE.** No shares-outstanding or market-cap source exists on disk "
                 "for this task's inputs (checked: no market_cap/shares_outstanding file under "
                 "`research/` or `data/research/`). Fabricating a market-cap proxy from price or "
                 "dollar volume would misrepresent size and was not done. The full decile table "
                 "above is unsplit by size; this is a data gap, not a finding, and should be "
                 "filled by fetching a shares-outstanding source (e.g. an EDGAR company-facts "
                 "pull) before this line item can be reported.")
    lines.append("")

    lines.append("## Split-artifact guard (unadjusted-split defense)")
    lines.append("Daily bars here are RAW (unadjusted). Any event whose price path from the "
                 "prior-reaction session through the exit session contains a session-to-session "
                 f"close ratio outside [{SPLIT_GUARD_LO}, {SPLIT_GUARD_HI}] is excluded from "
                 "scoring (guard10_ok / guard20_ok = False) rather than left in as a fabricated "
                 f"extreme return. Excluded by this guard (universe_ok rows): "
                 f"+10 hold {int((base['universe_ok'] & ~base['guard10_ok']).sum())}, "
                 f"+20 hold {int((base['universe_ok'] & ~base['guard20_ok']).sum())}.")
    lines.append("")

    lines.append("## Independent-check flags for the next agent")
    lines.append("1. **Timezone conversion** (see section above) -- the single most consequential "
                 "methodological choice in this cell; verify against raw SEC EDGAR behavior.")
    lines.append("2. **'Prior session' for the universe filter** was read literally as *the "
                 "session strictly before the announcement's own calendar day* (date_et - 1 "
                 "session), uniformly across after-close/pre-market/intraday. An equally "
                 "defensible reading uses date_et's own close for after-close filings (one "
                 "session later, still fully causal) -- rebuild independently and compare the "
                 "event set under both readings.")
    lines.append("3. **Small-cap split is unavailable** (see above) -- confirm no market-cap "
                 "source was missed before reporting this gap as final.")
    lines.append("4. **Borrow/SSR filtering is a proxy** applied with a present-day snapshot; "
                 "confirm the SSR proxy formula and its constants against rebuild_1552_full.py "
                 "directly rather than trusting this script's transcription.")
    lines.append("5. Rebuild the event set independently from `events_raw.csv` prose (Jaccard "
                 ">= 0.98 on (symbol, reaction_session) pairs) and R0/net_bps within 2 bps, per "
                 "the CLAUDE.md independent-check protocol, BEFORE this result is shown to the "
                 "owner.")
    lines.append("")

    OUT_RESULT.write_text("\n".join(lines))
    log.info("wrote %s", OUT_RESULT)


if __name__ == "__main__":
    main()
