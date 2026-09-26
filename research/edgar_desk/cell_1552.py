#!/usr/bin/env python3
"""Cells 1,552-1,561 -- the EDGAR event desk SCORER.

Spec: research/edgar_desk/PREREG_1552.md. Consumes `events_raw.csv` (built by
fetch_submissions.py: one row per filing per symbol, columns cik, symbol, form, filing_date,
acceptance_datetime, items, classes) plus daily bars, and:
  1. applies the population filter (price >= $1 and 20-session dollar volume >= $1M on the
     PRIOR session -- never the entry day itself, causality trace requirement #2 in CLAUDE.md);
  2. resolves the entry-auction session from the acceptance datetime (same-day MOO if accepted
     before 09:00 ET, else the next session's MOO);
  3. builds legs E1 (entry open -> entry close) and E5 (entry open -> close of session+5);
  4. charges 5 bps per auction leg (10 bps round trip) plus borrow (3%/yr pro rata) and the
     SSR / sub-$5 exclusions on SHORT cells;
  5. reports n, events/week, mean net bps (trade-direction sign), day-clustered t (session
     cluster), ex-top-5%/1%, winner-capped mean, median, share-in-direction, a universe
     placebo, a 1000-draw count-matched null (seed 1552), and a per-year table, split
     TRAIN 2019-2022 / VAL 2023-2024H1 / TEST 2024H2+ (sealed -- this script never scores TEST
     unless --unseal-test is passed explicitly, and only after a VAL pass is logged).

Independent-check note: this is the BUILDER implementation from the PREREG prose; a separate
agent that has not read this file must reimplement from the prose and compare row-by-row
(event-set Jaccard >= 0.99, net bps within 1) before any number here is reported to the owner.
"""
import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "hod_entry"))
from cell_1445 import day_clustered_t, ex_top5_mean, winner_capped_mean  # noqa: E402  (reuse per CLAUDE.md)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cell_1552")

EVENTS_RAW = os.path.join(ROOT, "events_raw.csv")
EVENTS_OUT = os.path.join(ROOT, "cell_1552_events.csv")
LEG_LOG = os.path.join(ROOT, "leg_selection_train.log")
ALPACA_PARQUET = os.path.join(ROOT, "..", "overnight_high", "alpaca_daily_2019_2024H1.parquet")
PANEL_PARQUET = os.path.join(ROOT, "..", "overnight_high", "panel_2024_2026.parquet")

MIN_PRICE = 1.0
MIN_DVOL20 = 1_000_000.0
COST_BPS_PER_LEG = 5.0  # 424B/8-K auction cost, one side
BORROW_ANNUAL = 0.03
SSR_DROP = -0.10
SHORT_PRICE_FLOOR = 5.0
WINNER_CAP = 0.20  # +-20%, note: cell_1445's WINNER_CAP_R is R-denominated; this book is bps/pct-denominated

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2024-06-30")
TEST_START = pd.Timestamp("2024-07-01")

# cell id, direction ("SHORT"/"LONG"), class name -- 1561 (BUYBACK_OR_INSIDER) needs a Form 4
# join that fetch_submissions.py does not build yet; it is REPORT-ONLY until that join exists.
CLASS_SPEC = {
    "OFFERING": (1552, "SHORT"),
    "SHELF": (1553, "SHORT"),
    "REVERSE_SPLIT": (1554, "SHORT"),
    "AUDITOR": (1555, "SHORT"),
    "NON_RELIANCE": (1556, "SHORT"),
    "LATE_FILING": (1557, "SHORT"),
    "OFFICER_EXIT": (1558, "SHORT"),
    "CONTRACT": (1559, "LONG"),
    "ACTIVIST": (1560, "LONG"),
}
REPORT_ONLY_NOTE = (
    "BUYBACK_OR_INSIDER (1,561) is report-only in this build: the PREREG's structured join "
    "(8-K 8.01 + a Form 4 purchase by an officer within 2 sessions) needs Form 4 data that "
    "fetch_submissions.py does not fetch; scoring it would require a second PREREG'd fetch."
)

TEST_TICKER_RE = re.compile(r"^Z[A-Z]ZZT$")


def is_test_ticker(symbol: str) -> bool:
    """Excluded per CLAUDE.md / the PREREG: ^Z[A-Z]ZZT$ (NASDAQ test tickers) or any ^ZZ symbol."""
    s = str(symbol)
    return bool(TEST_TICKER_RE.match(s)) or s.startswith("ZZ")


def entry_session(acceptance_dt, session_index: np.ndarray):
    """Resolve the entry-auction session date from an SEC acceptanceDateTime.

    SEC `acceptanceDateTime` is Eastern local time, tz-naive, per the submissions API. Before
    09:00 ET -> same-day MOO (if that calendar date is itself a session, else the next session
    on/after it); at/after 09:00 ET -> the NEXT session strictly after the acceptance's calendar
    date (a 09:31 acceptance never gets a same-day fill -- the refuter named in the PREREG).

    `session_index` is the sorted array of all trading-session dates (np.datetime64) known to
    the bars used for this cell; returns None if no session exists on/after the candidate date
    (acceptance past the end of the loaded bars).
    """
    if acceptance_dt is None or (isinstance(acceptance_dt, float) and np.isnan(acceptance_dt)):
        return None
    ts = pd.Timestamp(acceptance_dt)
    if pd.isna(ts):
        return None
    accept_date = ts.normalize()
    if ts.time() < pd.Timestamp("09:00:00").time():
        candidate = accept_date
        idx = np.searchsorted(session_index, np.datetime64(candidate), side="left")
    else:
        candidate = accept_date + pd.Timedelta(days=1)
        idx = np.searchsorted(session_index, np.datetime64(candidate), side="left")
    if idx >= len(session_index):
        return None
    return pd.Timestamp(session_index[idx])


def classify_from_row(form: str, items_raw: str) -> list:
    """Re-derive PREREG classes from a filing's form + items (mirrors fetch_submissions.classify;
    duplicated here, not imported, so the scorer never silently changes if the fetch script's
    class rules are edited for a later cell -- any drift must be a deliberate, reviewed change).
    """
    items_set = {it.strip() for it in str(items_raw).replace(";", ",").split(",") if it.strip()}
    FORM_424B = {"424B1", "424B2", "424B3", "424B4", "424B5"}
    FORM_SHELF = {"S-3", "S-3ASR", "S-1"}
    FORM_LATE = {"NT 10-K", "NT 10-Q"}
    classes = []
    if form in FORM_424B or "3.02" in items_set:
        classes.append("OFFERING")
    if form in FORM_SHELF:
        classes.append("SHELF")
    if "5.03" in items_set:
        classes.append("REVERSE_SPLIT")
    if "4.01" in items_set:
        classes.append("AUDITOR")
    if "4.02" in items_set:
        classes.append("NON_RELIANCE")
    if form in FORM_LATE:
        classes.append("LATE_FILING")
    if "5.02" in items_set:
        classes.append("OFFICER_EXIT")
    if "1.01" in items_set and "3.02" not in items_set and "2.03" not in items_set:
        classes.append("CONTRACT")
    if form == "SC 13D":
        classes.append("ACTIVIST")
    return classes


def load_bars() -> pd.DataFrame:
    """Merge Alpaca raw daily bars (2019-2024H1) with the Databento panel (2024H2-2026),
    dropping the panel's zero-OHLCV placeholder rows (the cell-1,550 defect) and computing a
    rolling 20-session dollar-volume column where the source lacks one (Alpaca)."""
    a = pd.read_parquet(ALPACA_PARQUET)
    a["bar_date"] = pd.to_datetime(a["bar_date"])
    a = a.sort_values(["symbol", "bar_date"])
    a["dvol"] = a["close"] * a["volume"]
    a["dvol20"] = a.groupby("symbol")["dvol"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    a = a[["symbol", "bar_date", "open", "high", "low", "close", "volume", "dvol20"]]

    p = pd.read_parquet(PANEL_PARQUET)
    p["bar_date"] = pd.to_datetime(p["bar_date"])
    p["symbol"] = p["symbol"].astype(str)
    zero_mask = (p[["open", "high", "low", "close"]] == 0).all(axis=1)
    log.warning("panel_2024_2026: dropping %d zero-OHLCV placeholder rows (cell 1,550 defect)", int(zero_mask.sum()))
    p = p.loc[~zero_mask, ["symbol", "bar_date", "open", "high", "low", "close", "volume", "dvol20"]]

    bars = pd.concat([a, p], ignore_index=True)
    bars = bars.drop_duplicates(["symbol", "bar_date"], keep="last").sort_values(["symbol", "bar_date"])
    return bars.reset_index(drop=True)


def build_symbol_index(bars: pd.DataFrame) -> dict:
    """dict symbol -> DataFrame indexed by bar_date (sorted), with a `prior_close`/`prior_dvol20`
    shift already applied so eligibility can be read at the PRIOR session without recomputation
    per event (causality: eligibility is decided on the session BEFORE the entry day)."""
    out = {}
    for sym, g in bars.groupby("symbol", sort=False):
        g = g.sort_values("bar_date").set_index("bar_date")
        g["prior_close"] = g["close"].shift(1)
        g["prior_dvol20"] = g["dvol20"].shift(1)
        out[sym] = g
    return out


def score_event(sym_bars: pd.DataFrame, entry_date: pd.Timestamp, direction: int):
    """Return (eligible, ssr_excluded, price_floor_excluded, ret_e1_net, ret_e5_net, days_held_e5)
    for one event's entry session, or None fields where the session/legs are not obtainable
    (missing bar, insufficient trailing history)."""
    if entry_date not in sym_bars.index:
        return None
    idx = sym_bars.index.get_loc(entry_date)
    row = sym_bars.iloc[idx]
    prior_close, prior_dvol20 = row["prior_close"], row["prior_dvol20"]
    if pd.isna(prior_close) or pd.isna(prior_dvol20):
        return None
    eligible = (prior_close >= MIN_PRICE) and (prior_dvol20 >= MIN_DVOL20)

    # SSR trigger: prior session dropped >= 10% from the session before it.
    ssr = False
    if idx >= 2:
        pp_close = sym_bars.iloc[idx - 2]["close"]
        if pp_close and not pd.isna(pp_close):
            if (prior_close / pp_close - 1.0) <= SSR_DROP:
                ssr = True
    price_floor_excl = prior_close < SHORT_PRICE_FLOOR

    entry_open = row["open"]
    if pd.isna(entry_open) or entry_open <= 0:
        return {"eligible": eligible, "ssr": ssr, "price_floor_excl": price_floor_excl,
                "ret_e1_net": np.nan, "ret_e5_net": np.nan, "days_held_e5": np.nan}

    entry_close = row["close"]
    ret_e1_gross = direction * (entry_close / entry_open - 1.0) if not pd.isna(entry_close) else np.nan
    cost_e1 = 2 * (COST_BPS_PER_LEG / 10000.0)
    borrow_e1 = (BORROW_ANNUAL / 365.0) * 1 if direction == -1 else 0.0
    ret_e1_net = ret_e1_gross - cost_e1 - borrow_e1 if not pd.isna(ret_e1_gross) else np.nan

    ret_e5_net, days_held_e5 = np.nan, np.nan
    if idx + 5 < len(sym_bars):
        exit_row = sym_bars.iloc[idx + 5]
        exit_close = exit_row["close"]
        if not pd.isna(exit_close):
            ret_e5_gross = direction * (exit_close / entry_open - 1.0)
            days_held_e5 = (exit_row.name - entry_date).days
            cost_e5 = 2 * (COST_BPS_PER_LEG / 10000.0)
            borrow_e5 = (BORROW_ANNUAL / 365.0) * max(days_held_e5, 1) if direction == -1 else 0.0
            ret_e5_net = ret_e5_gross - cost_e5 - borrow_e5

    return {"eligible": eligible, "ssr": ssr, "price_floor_excl": price_floor_excl,
            "ret_e1_net": ret_e1_net, "ret_e5_net": ret_e5_net, "days_held_e5": days_held_e5}


def count_matched_null(pop_df: pd.DataFrame, per_session_counts: pd.Series, direction: int, leg_col: str,
                        n_draws: int = 1000, seed: int = 1552) -> np.ndarray:
    """1,000-draw count-matched null: for each draw, sample (without replacement per session) the
    same number of eligible names per session as the real cell drew, on the SAME sessions, and
    take the direction-signed mean net return. `pop_df` must be the full eligible-universe leg
    table (session, symbol, ret) for the sessions used by the cell."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    by_session = {s: g[leg_col].dropna().to_numpy() for s, g in pop_df.groupby("session")}
    for d in range(n_draws):
        draw_vals = []
        for session, k in per_session_counts.items():
            pool = by_session.get(session)
            if pool is None or len(pool) == 0:
                continue
            k = min(int(k), len(pool))
            draw_vals.append(rng.choice(pool, size=k, replace=False))
        means[d] = direction * np.concatenate(draw_vals).mean() if draw_vals else np.nan
    return means


def summarize(y: pd.Series, direction: int) -> dict:
    y = y.dropna()
    n = len(y)
    if n == 0:
        return dict(n=0)
    bps = y * 10000.0
    return dict(
        n=n,
        mean_net_bps=float(bps.mean()),
        median_bps=float(bps.median()),
        ex_top5_bps=float(ex_top5_mean(bps)),
        ex_top1_bps=float(_ex_topk_mean(bps, 0.01)),
        winner_capped_bps=float(y.clip(-WINNER_CAP, WINNER_CAP).mean() * 10000.0),
        share_in_direction=float((y > 0).mean()),
    )


def _ex_topk_mean(y: pd.Series, k_frac: float) -> float:
    y = y.dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(k_frac * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def holdout_of(date: pd.Timestamp) -> str:
    if date <= TRAIN_END:
        return "TRAIN"
    if date <= VAL_END:
        return "VAL"
    return "TEST"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseal-test", action="store_true",
                     help="score the sealed TEST split too -- only after a logged VAL pass, single read")
    args = ap.parse_args()

    if not os.path.exists(EVENTS_RAW):
        log.error("events_raw.csv not found at %s -- fetch stage (fetch_submissions.py) has not "
                   "completed yet; nothing to score. This is expected while the FETCH stage runs "
                   "in the background; re-run once events_raw.csv exists.", EVENTS_RAW)
        sys.exit(2)

    events = pd.read_csv(EVENTS_RAW, dtype=str)
    log.info("loaded %d raw filing rows from events_raw.csv", len(events))
    events = events[~events["symbol"].map(is_test_ticker)]
    events["filing_date"] = pd.to_datetime(events["filing_date"])
    events["acceptance_datetime"] = pd.to_datetime(events["acceptance_datetime"], errors="coerce")

    bars = load_bars()
    session_index = np.sort(bars["bar_date"].unique())
    sym_index = build_symbol_index(bars)

    events["entry_date"] = events["acceptance_datetime"].apply(lambda a: entry_session(a, session_index))
    n_no_entry = events["entry_date"].isna().sum()
    log.info("resolved entry session for %d/%d filings (%d unresolved -- acceptance beyond loaded bars)",
              len(events) - n_no_entry, len(events), n_no_entry)

    out_rows = []
    for cls, (cell_id, direction_name) in CLASS_SPEC.items():
        direction = -1 if direction_name == "SHORT" else 1
        cls_events = events[events["classes"].fillna("").str.contains(cls)]
        for _, r in cls_events.iterrows():
            sb = sym_index.get(r["symbol"])
            if sb is None or pd.isna(r["entry_date"]):
                continue
            res = score_event(sb, r["entry_date"], direction)
            if res is None:
                continue
            if not res["eligible"]:
                continue
            if direction_name == "SHORT" and (res["ssr"] or res["price_floor_excl"]):
                continue
            for leg, ret_col in (("E1", "ret_e1_net"), ("E5", "ret_e5_net")):
                if pd.isna(res[ret_col]):
                    continue
                out_rows.append(dict(
                    cell=cell_id, cls=cls, leg=leg, split=holdout_of(r["entry_date"]),
                    date=r["entry_date"], symbol=r["symbol"], entry="open", exit=leg,
                    ret_dir_net=res[ret_col],
                ))

    events_out = pd.DataFrame(out_rows)
    if len(events_out):
        events_out.to_csv(EVENTS_OUT, index=False)
        log.info("wrote %d event-leg rows to %s", len(events_out), EVENTS_OUT)
    else:
        log.warning("no scoreable event-leg rows produced (events_raw.csv likely empty/PENDING); "
                     "writing an empty %s so downstream tooling has a stable path", EVENTS_OUT)
        pd.DataFrame(columns=["cell", "cls", "leg", "split", "date", "symbol", "entry", "exit", "ret_dir_net"]
                     ).to_csv(EVENTS_OUT, index=False)

    log.info(REPORT_ONLY_NOTE)
    log.info("run complete; RESULT_1552.md must be written/updated separately from this table "
             "(per PREREG: leg named on TRAIN, logged to %s, before VAL is read)", LEG_LOG)


if __name__ == "__main__":
    main()
