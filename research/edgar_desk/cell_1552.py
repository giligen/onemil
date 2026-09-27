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
import gc
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
STATS_OUT = os.path.join(ROOT, "cell_1552_stats.csv")
LEG_LOG = os.path.join(ROOT, "leg_selection_train.log")
NULL_DRAWS = 1000
NULL_SEED = 1552
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

# Amendment 1 (2026-09-27 04:25 UTC, PREREG_1552.md): the raw OFFERING pull (1,149,657 rows) was
# dominated by 424B2 bank structured-note pricing supplements (no equity-supply mechanism). Fix,
# structural only: 424B2 is never an OFFERING trigger, and any issuer filing MORE than SERIAL_CAP
# 424B* filings (of any of the five subtypes -- "prospectus supplements") in the trailing
# SERIAL_WINDOW_DAYS days is a serial note issuer, excluded from OFFERING *and* SHELF regardless
# of which form/item triggered the match. Every other class is unchanged.
FORM_424B_ALL = {"424B1", "424B2", "424B3", "424B4", "424B5"}
FORM_424B_OFFERING = {"424B1", "424B3", "424B4", "424B5"}  # 424B2 excluded (Amendment 1)
SERIAL_CAP = 12
SERIAL_WINDOW_DAYS = 365


def build_serial_index(cik_series, form_series, filing_date_series) -> dict:
    """cik -> sorted np.int64 array (days since epoch) of that issuer's 424B* filing dates, across
    ALL five subtypes (the "prospectus supplement" count for the Amendment-1 serial-issuer cap).
    Built once from the FULL raw events file (not the eligible/classified subset) so the trailing
    count is correct even when the triggering supplement itself is a 424B2 that never becomes an
    OFFERING event."""
    days = pd.to_datetime(pd.Series(list(filing_date_series))).values.astype("datetime64[D]").astype(np.int64)
    mask = pd.Series(list(form_series)).isin(FORM_424B_ALL).to_numpy()
    cik = np.asarray(cik_series)[mask]
    days = days[mask]
    # Group by cik via one sort (O(n log n)), NOT a `days[cik == c]` scan per unique cik
    # (O(n * k) -- 7,754 CIKs x ~1.15M 424B rows effectively hung the first full run of this pass).
    if len(cik) == 0:
        return {}
    order = np.argsort(cik, kind="stable")
    cik_sorted, days_sorted = cik[order], days[order]
    boundaries = np.flatnonzero(cik_sorted[1:] != cik_sorted[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(cik_sorted)]))
    index = {}
    for s, e in zip(starts, ends):
        index[cik_sorted[s]] = np.sort(days_sorted[s:e])
    return index


def is_serial_issuer(cik, date, serial_index: dict, cap: int = SERIAL_CAP,
                      window_days: int = SERIAL_WINDOW_DAYS) -> bool:
    """True if `cik` filed more than `cap` 424B* filings in (date - window_days, date] inclusive
    of the filing on `date` itself -- the scalar/testable form of the Amendment-1 cap, mirrored by
    the vectorized `compute_serial_flags` used on the full file for speed."""
    arr = serial_index.get(cik)
    if arr is None or len(arr) == 0:
        return False
    d = np.datetime64(pd.Timestamp(date).normalize(), "D").astype(np.int64)
    lo = np.searchsorted(arr, d - window_days, side="left")
    hi = np.searchsorted(arr, d, side="right")
    return bool((hi - lo) > cap)


def compute_serial_flags(cik_series, filing_day_int64, serial_index: dict,
                          cap: int = SERIAL_CAP, window_days: int = SERIAL_WINDOW_DAYS) -> np.ndarray:
    """Vectorized (grouped-by-issuer) equivalent of `is_serial_issuer` for the full events table.
    `filing_day_int64` must already be int64 days-since-epoch (same units as `build_serial_index`).
    Restrict the caller's frame to OFFERING/SHELF candidate rows first -- this is only correct
    (and only needed) for classes gated by the Amendment-1 cap."""
    cik_arr = np.asarray(cik_series)
    out = np.zeros(len(cik_arr), dtype=bool)
    df = pd.DataFrame({"cik": cik_arr, "day": filing_day_int64, "pos": np.arange(len(cik_arr))})
    for cik, g in df.groupby("cik", sort=False):
        arr = serial_index.get(cik)
        if arr is None or len(arr) == 0:
            continue
        days = g["day"].to_numpy()
        lo = np.searchsorted(arr, days - window_days, side="left")
        hi = np.searchsorted(arr, days, side="right")
        out[g["pos"].to_numpy()] = (hi - lo) > cap
    return out


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


def classify_from_row(form: str, items_raw: str, serial_issuer: bool = False) -> list:
    """Re-derive PREREG classes (AMENDED, Amendment 1) from a filing's form + items (mirrors
    fetch_submissions.classify pre-amendment; duplicated here, not imported, so the scorer never
    silently changes if the fetch script's class rules are edited for a later cell -- any drift
    must be a deliberate, reviewed change).

    `serial_issuer` is the caller-supplied Amendment-1 flag (from `is_serial_issuer` /
    `compute_serial_flags`): True means this CIK filed > SERIAL_CAP 424B* filings in the trailing
    SERIAL_WINDOW_DAYS days as of this filing -- a serial note issuer, gated OUT of OFFERING and
    SHELF regardless of form/item (the events_raw.csv `classes` column was computed BEFORE this
    amendment and must not be trusted for OFFERING/SHELF; recompute from form+items+serial here).
    """
    items_set = {it.strip() for it in str(items_raw).replace(";", ",").split(",") if it.strip()}
    FORM_SHELF = {"S-3", "S-3ASR", "S-1"}
    FORM_LATE = {"NT 10-K", "NT 10-Q"}
    classes = []
    if not serial_issuer:
        if form in FORM_424B_OFFERING or "3.02" in items_set:  # 424B2 excluded (Amendment 1)
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


def _has_item(items_norm: pd.Series, code: str) -> pd.Series:
    """Vectorized token-exact item-code match on a comma-normalized `items` string column
    (mirrors classify_from_row's set-membership test; used on the full 4.4M-row file where a
    per-row python function call is too slow)."""
    pat = r"(?:^|,)" + re.escape(code) + r"(?:,|$)"
    return items_norm.str.contains(pat, regex=True, na=False)


def compute_class_masks_amended(events: pd.DataFrame, serial_index: dict) -> dict:
    """Vectorized, Amendment-1-amended equivalent of calling `classify_from_row` per row on the
    full events table. Returns {class_name: boolean Series aligned to `events.index`}.

    Cross-checked in `run_amendment_self_check` against row-wise `classify_from_row` on a random
    sample before the full run is trusted (CLAUDE.md: no dual, drifting implementations of the
    same rule without a check)."""
    items_norm = events["items"].astype(object).fillna("").astype(str).str.replace(";", ",", regex=False)
    form = events["form"]
    m302 = _has_item(items_norm, "3.02")
    m203 = _has_item(items_norm, "2.03")
    m503 = _has_item(items_norm, "5.03")
    m401 = _has_item(items_norm, "4.01")
    m402 = _has_item(items_norm, "4.02")
    m502 = _has_item(items_norm, "5.02")
    m101 = _has_item(items_norm, "1.01")

    offering_form = form.isin(FORM_424B_OFFERING)
    shelf_form = form.isin({"S-3", "S-3ASR", "S-1"})
    offering_or_shelf_trigger = offering_form | m302 | shelf_form

    filing_day = pd.to_datetime(events["filing_date"]).values.astype("datetime64[D]").astype(np.int64)
    serial = np.zeros(len(events), dtype=bool)
    cand = offering_or_shelf_trigger.to_numpy()
    if cand.any():
        serial[cand] = compute_serial_flags(events["cik"].to_numpy()[cand], filing_day[cand], serial_index)
    serial = pd.Series(serial, index=events.index)

    return {
        "OFFERING": (offering_form | m302) & ~serial,
        "SHELF": shelf_form & ~serial,
        "REVERSE_SPLIT": m503,
        "AUDITOR": m401,
        "NON_RELIANCE": m402,
        "LATE_FILING": form.isin({"NT 10-K", "NT 10-Q"}),
        "OFFICER_EXIT": m502,
        "CONTRACT": m101 & ~m302 & ~m203,
        "ACTIVIST": form.eq("SC 13D"),
    }


def run_amendment_self_check(events: pd.DataFrame, masks: dict, serial_index: dict,
                              n_sample: int = 2000, seed: int = 1552) -> None:
    """Independent-check-lite: sample n_sample rows and confirm the vectorized amended masks agree
    with the scalar, unit-tested `classify_from_row` (same file, same rule -- this only catches a
    vectorization bug, not a spec error; the real independent reimplementation is a separate agent
    per CLAUDE.md). Logs mismatches and raises if agreement < 99.9%."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(events.index.to_numpy(), size=min(n_sample, len(events)), replace=False)
    mismatches = 0
    for i in idx:
        row = events.loc[i]
        cik = row["cik"]
        serial_flag = is_serial_issuer(cik, row["filing_date"], serial_index) \
            if (row["form"] in FORM_424B_OFFERING or "3.02" in str(row["items"]).replace(";", ",")
                or row["form"] in {"S-3", "S-3ASR", "S-1"}) else False
        expected = set(classify_from_row(row["form"], row["items"], serial_issuer=serial_flag))
        got = {cls for cls, m in masks.items() if bool(m.loc[i])}
        if expected != got:
            mismatches += 1
    rate = 1 - mismatches / len(idx)
    log.info("amendment self-check: %d/%d sampled rows agree (%.4f%%)", len(idx) - mismatches, len(idx), rate * 100)
    if rate < 0.999:
        raise RuntimeError(f"vectorized amended classification disagrees with classify_from_row on "
                            f"{mismatches}/{len(idx)} sampled rows -- fix before trusting the full run")


def load_bars() -> pd.DataFrame:
    """Merge Alpaca raw daily bars (2019-2024H1) with the Databento panel (2024H2-2026),
    dropping the panel's zero-OHLCV placeholder rows (the cell-1,550 defect) and computing a
    rolling 20-session dollar-volume column where the source lacks one (Alpaca).

    Memory: only open/close/volume/dvol20 are ever read by this cell (no high/low use anywhere
    in score_event / the universe table), symbol is cast to category immediately, and each source
    is read with `columns=` so pyarrow never materializes the unused fields at all -- the naive
    full-column read of both ~19.4M-row panels plus their concat/groupby copies OOM-killed the
    first run of this pass at ~7GB RSS."""
    a = pd.read_parquet(ALPACA_PARQUET, columns=["symbol", "bar_date", "open", "close", "volume"])
    a["bar_date"] = pd.to_datetime(a["bar_date"])
    a["symbol"] = a["symbol"].astype("category")
    a = a.sort_values(["symbol", "bar_date"])
    dvol = a["close"] * a["volume"]
    a["dvol20"] = dvol.groupby(a["symbol"], sort=False, observed=True).transform(lambda s: s.rolling(20, min_periods=10).mean())
    a = a[["symbol", "bar_date", "open", "close", "dvol20"]]

    p = pd.read_parquet(PANEL_PARQUET, columns=["symbol", "bar_date", "open", "high", "low", "close",
                                                 "volume", "dvol20"])
    p["bar_date"] = pd.to_datetime(p["bar_date"])
    zero_mask = (p[["open", "high", "low", "close"]] == 0).all(axis=1)
    log.warning("panel_2024_2026: dropping %d zero-OHLCV placeholder rows (cell 1,550 defect)", int(zero_mask.sum()))
    p = p.loc[~zero_mask, ["symbol", "bar_date", "open", "close", "dvol20"]]
    p["symbol"] = p["symbol"].astype(str).astype("category")

    bars = pd.concat([a, p], ignore_index=True)
    del a, p
    bars["symbol"] = bars["symbol"].astype("category")
    bars = bars.drop_duplicates(["symbol", "bar_date"], keep="last").sort_values(["symbol", "bar_date"])
    return bars.reset_index(drop=True)


def prepare_bars_shared(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute prior_close/prior_dvol20 (causal, prior-session eligibility columns) and the two
    RAW direction=+1 leg returns ONCE on the full flat panel, shared by both `build_symbol_index`
    (the per-event scorer) and the universe placebo/null table -- the original code computed the
    same groupby-shift twice (once per caller), doubling peak memory and OOM-killing the process.
    Mutates `bars` in place (no extra full-frame copy) since load_bars() already sorted it and
    the caller does not need the pre-amendment frame back."""
    g = bars.groupby("symbol", sort=False, observed=True)
    bars["prior_close"] = g["close"].shift(1)
    bars["prior_dvol20"] = g["dvol20"].shift(1)
    bars["eligible"] = (bars["prior_close"] >= MIN_PRICE) & (bars["prior_dvol20"] >= MIN_DVOL20)
    bars["ret_e1_gross"] = bars["close"] / bars["open"] - 1.0
    bars["ret_e5_gross"] = bars.groupby("symbol", sort=False, observed=True)["close"].shift(-5) / bars["open"] - 1.0
    return bars


def build_symbol_offsets(bars: pd.DataFrame) -> dict:
    """symbol -> (start, end) row-slice bounds into the SINGLE sorted `bars` frame (requires
    `bars` sorted by symbol, bar_date with a plain RangeIndex, as load_bars()/prepare_bars_shared
    leave it). No per-symbol copy here -- eagerly building one small DataFrame per symbol for
    ALL ~40K universe symbols (most of which never appear in an event) is what OOM-killed / swap-
    thrashed the first two runs of this pass; `SymbolFrameCache` below materializes a lean slice
    only for a symbol actually touched by a scored event, and only once."""
    sym = bars["symbol"].to_numpy()
    change = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(bars)]))
    return {sym[s]: (int(s), int(e)) for s, e in zip(starts, ends)}


class SymbolFrameCache:
    """Lazy, memoized symbol -> lean (open, close, prior_close, prior_dvol20) DataFrame indexed
    by bar_date, sliced on first access from the single shared `bars` frame via `offsets`."""

    def __init__(self, bars: pd.DataFrame, offsets: dict):
        self._bars = bars
        self._offsets = offsets
        self._cache = {}

    def get(self, symbol):
        if symbol in self._cache:
            return self._cache[symbol]
        se = self._offsets.get(symbol)
        if se is None:
            self._cache[symbol] = None
            return None
        start, end = se
        g = self._bars.iloc[start:end].set_index("bar_date")[["open", "close", "prior_close", "prior_dvol20"]]
        self._cache[symbol] = g
        return g


def build_universe_return_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Lean (symbol, bar_date, eligible, ret_e1_gross, ret_e5_gross) slice of the shared
    `prepare_bars_shared` output -- the universe placebo / count-matched null population, without
    the OHLCV columns those don't need (memory)."""
    return bars[["symbol", "bar_date", "eligible", "ret_e1_gross", "ret_e5_gross"]]


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
    if args.unseal_test:
        log.error("--unseal-test was passed but no VAL pass has been logged for this run yet -- "
                   "refusing to unseal TEST (PREREG: one read, only after a logged VAL pass)")
        sys.exit(2)

    if not os.path.exists(EVENTS_RAW):
        log.error("events_raw.csv not found at %s -- fetch stage (fetch_submissions.py) has not "
                   "completed yet; nothing to score.", EVENTS_RAW)
        sys.exit(2)

    log.info("reading events_raw.csv in chunks (needed columns only, categorical dtype) for memory")
    usecols = ["cik", "symbol", "form", "filing_date", "acceptance_datetime", "items"]
    dtypes = {"cik": "category", "symbol": "category", "form": "category", "items": "object"}
    chunks = []
    for i, chunk in enumerate(pd.read_csv(EVENTS_RAW, usecols=usecols, dtype=dtypes, chunksize=500_000)):
        chunks.append(chunk)
        log.info("  read chunk %d (%d rows)", i, len(chunk))
    events = pd.concat(chunks, ignore_index=True)
    del chunks
    log.info("loaded %d raw filing rows from events_raw.csv (columns: %s)", len(events), usecols)

    events = events[~events["symbol"].astype(str).map(is_test_ticker)].reset_index(drop=True)
    events["filing_date"] = pd.to_datetime(events["filing_date"])
    events["acceptance_datetime"] = pd.to_datetime(events["acceptance_datetime"], errors="coerce")
    events["items"] = events["items"].astype("category")  # high-cardinality but << 4.4M rows; memory
    log.info("after test-ticker exclusion: %d rows", len(events))

    # ---- Amendment 1: serial-issuer index + amended class masks (RAW counts are the fetch's own
    # pre-amendment `classes` column, reported from fetch_summary.txt; recomputed here only for
    # the amended side, per the task -- the raw column is not reloaded since it is unamended and
    # not used for scoring any more).
    serial_index = build_serial_index(events["cik"], events["form"], events["filing_date"])
    log.info("serial-issuer index built for %d CIKs with >=1 424B filing", len(serial_index))
    amended_masks = compute_class_masks_amended(events, serial_index)
    run_amendment_self_check(events, amended_masks, serial_index)
    raw_counts = {"OFFERING": 1149657, "SHELF": 16398, "REVERSE_SPLIT": 17629, "AUDITOR": 4198,
                  "NON_RELIANCE": 1165, "LATE_FILING": 8676, "OFFICER_EXIT": 81459,
                  "CONTRACT": 68611, "ACTIVIST": 5769}  # fetch_summary.txt, pre-amendment
    amended_counts = {cls: int(m.sum()) for cls, m in amended_masks.items()}
    log.info("class counts raw -> amended:")
    for cls in CLASS_SPEC:
        log.info("  %-14s raw=%d  amended=%d", cls, raw_counts[cls], amended_counts[cls])
    pd.Series(raw_counts).to_frame("raw").join(pd.Series(amended_counts).to_frame("amended")) \
        .to_csv(os.path.join(ROOT, "class_counts_raw_vs_amended.csv"))

    bars_raw = load_bars()
    session_index = np.sort(bars_raw["bar_date"].unique())
    bars = prepare_bars_shared(bars_raw)
    del bars_raw
    gc.collect()
    # ONE shared frame stays alive for the whole run; SymbolFrameCache slices it lazily, only for
    # symbols an event actually touches, instead of the eager per-symbol dict that swap-thrashed
    # the first two runs of this pass (see build_symbol_offsets docstring).
    sym_offsets = build_symbol_offsets(bars)
    sym_cache = SymbolFrameCache(bars, sym_offsets)
    univ = build_universe_return_table(bars)
    univ_e1 = univ.set_index(["symbol", "bar_date"])["ret_e1_gross"]
    univ_e5 = univ.set_index(["symbol", "bar_date"])["ret_e5_gross"]
    univ_elig = univ[univ["eligible"]]
    placebo_by_date_e1 = univ_elig.groupby("bar_date")["ret_e1_gross"].mean()
    placebo_by_date_e5 = univ_elig.groupby("bar_date")["ret_e5_gross"].mean()

    events["entry_date"] = events["acceptance_datetime"].apply(lambda a: entry_session(a, session_index))
    n_no_entry = events["entry_date"].isna().sum()
    log.info("resolved entry session for %d/%d filings (%d unresolved -- acceptance beyond loaded bars)",
              len(events) - n_no_entry, len(events), n_no_entry)

    # TEST is sealed: drop it BEFORE scoring so this run cannot compute a TEST statistic even by
    # accident (PREREG: "this script never scores TEST unless --unseal-test", refused above).
    events["holdout"] = events["entry_date"].apply(lambda d: holdout_of(d) if d is not None else None)
    n_test_dropped = int((events["holdout"] == "TEST").sum())
    keep_mask = events["holdout"].isin(["TRAIN", "VAL"])
    # amended_masks were built against the PRE-filter index -- subset them with the SAME boolean
    # mask (not a reindex after reset_index) so class membership stays row-aligned with `events`.
    amended_masks = {cls: m[keep_mask].reset_index(drop=True) for cls, m in amended_masks.items()}
    events = events[keep_mask].reset_index(drop=True)
    log.info("dropped %d TEST-split filings before scoring (sealed, not read)", n_test_dropped)

    out_rows = []
    diff_rows = []  # for the universe-placebo margin (paired per-event diff, day-clustered later)
    for cls, (cell_id, direction_name) in CLASS_SPEC.items():
        direction = -1 if direction_name == "SHORT" else 1
        cls_events = events[amended_masks[cls].to_numpy()]
        n_scored = 0
        for r in cls_events.itertuples(index=False):
            sb = sym_cache.get(r.symbol)
            if sb is None or r.entry_date is None:
                continue
            res = score_event(sb, r.entry_date, direction)
            if res is None or not res["eligible"]:
                continue
            if direction_name == "SHORT" and (res["ssr"] or res["price_floor_excl"]):
                continue
            for leg, ret_col, gross_lookup in (("E1", "ret_e1_net", univ_e1), ("E5", "ret_e5_net", univ_e5)):
                if pd.isna(res[ret_col]):
                    continue
                out_rows.append(dict(
                    cell=cell_id, cls=cls, leg=leg, split=r.holdout,
                    date=r.entry_date, symbol=r.symbol, entry="open", exit=leg,
                    ret_dir_net=res[ret_col],
                ))
                pop_mean = (placebo_by_date_e1 if leg == "E1" else placebo_by_date_e5).get(r.entry_date, np.nan)
                event_gross = gross_lookup.get((r.symbol, r.entry_date), np.nan)
                if not (pd.isna(pop_mean) or pd.isna(event_gross)):
                    diff_rows.append(dict(cell=cell_id, leg=leg, split=r.holdout, date=r.entry_date,
                                           diff=direction * (event_gross - pop_mean)))
                n_scored += 1
        log.info("cell %d (%s, %s): %d candidate filings -> %d scored event-legs",
                  cell_id, cls, direction_name, len(cls_events), n_scored)

    events_out = pd.DataFrame(out_rows, columns=["cell", "cls", "leg", "split", "date", "symbol",
                                                  "entry", "exit", "ret_dir_net"])
    events_out.to_csv(EVENTS_OUT, index=False)
    log.info("wrote %d event-leg rows to %s", len(events_out), EVENTS_OUT)
    diff_df = pd.DataFrame(diff_rows, columns=["cell", "leg", "split", "date", "diff"])

    # ---- Per-cell, per-split, per-leg stats. TRAIN is summarized and the leg is NAMED and
    # LOGGED (flushed to disk) before this loop goes on to compute any VAL number for that cell --
    # the code-order enforcement of "name the leg on TRAIN before reading VAL" (PREREG, Not
    # allowed: "selecting E1 vs E5 on VAL").
    stats_rows = []
    with open(LEG_LOG, "a") as leg_log:
        for cls, (cell_id, direction_name) in CLASS_SPEC.items():
            direction = -1 if direction_name == "SHORT" else 1
            train_t = {}
            for leg in ("E1", "E5"):
                sub = events_out[(events_out["cell"] == cell_id) & (events_out["leg"] == leg)
                                  & (events_out["split"] == "TRAIN")]
                train_t[leg] = day_clustered_t(sub["ret_dir_net"], sub["date"]) if len(sub) >= 2 else np.nan
            t1, t5 = train_t["E1"], train_t["E5"]
            if pd.isna(t1) and pd.isna(t5):
                named_leg = "E1"  # pre-committed tie-break / insufficient-TRAIN default
            elif pd.isna(t5) or (not pd.isna(t1) and t1 >= t5):
                named_leg = "E1"
            else:
                named_leg = "E5"
            leg_log.write(f"{pd.Timestamp.utcnow().isoformat()} cell={cell_id} cls={cls} "
                          f"TRAIN_t_E1={t1} TRAIN_t_E5={t5} NAMED_LEG={named_leg}\n")
            leg_log.flush()

            for split in ("TRAIN", "VAL"):
                for leg in ("E1", "E5"):
                    sub = events_out[(events_out["cell"] == cell_id) & (events_out["leg"] == leg)
                                      & (events_out["split"] == split)]
                    y = sub["ret_dir_net"]
                    summ = summarize(y, direction)
                    n = summ.get("n", 0)
                    t = day_clustered_t(y, sub["date"]) if n >= 2 else np.nan
                    weeks = max(sub["date"].apply(lambda d: (d.isocalendar()[0], d.isocalendar()[1]))
                                .nunique(), 1) if n else 1
                    dsub = diff_df[(diff_df["cell"] == cell_id) & (diff_df["leg"] == leg) & (diff_df["split"] == split)]
                    placebo_margin_bps = float(dsub["diff"].mean() * 10000.0) if len(dsub) else np.nan
                    placebo_t = day_clustered_t(dsub["diff"], dsub["date"]) if len(dsub) >= 2 else np.nan
                    universe_bps = summ.get("mean_net_bps", np.nan) - placebo_margin_bps if n else np.nan

                    null_pctile = np.nan
                    if n:
                        pop = univ_elig[univ_elig["bar_date"].isin(sub["date"].unique())][
                            ["bar_date", "ret_e1_gross" if leg == "E1" else "ret_e5_gross"]
                        ].rename(columns={("ret_e1_gross" if leg == "E1" else "ret_e5_gross"): "ret",
                                           "bar_date": "session"})
                        counts = sub.groupby("date").size()
                        counts.index.name = "session"
                        cost_const = 2 * (COST_BPS_PER_LEG / 10000.0)
                        if direction == -1:
                            cost_const += (BORROW_ANNUAL / 365.0) * (1 if leg == "E1" else 5)
                        null_means = count_matched_null(pop, counts, direction, "ret") - cost_const
                        actual = summ["mean_net_bps"] / 10000.0
                        null_pctile = float((null_means < actual).mean() * 100.0)

                    passes = bool(n and split == "VAL" and leg == named_leg
                                  and summ.get("mean_net_bps", -1e9) >= 15
                                  and not pd.isna(t) and t >= 2.5
                                  and summ.get("ex_top5_bps", -1e9) > 0
                                  and summ.get("winner_capped_bps", -1e9) > 0
                                  and (n / max(weeks, 1)) >= 3
                                  and not pd.isna(placebo_margin_bps) and placebo_margin_bps >= 10
                                  and not pd.isna(placebo_t) and placebo_t >= 2
                                  and not pd.isna(null_pctile) and null_pctile >= 99)

                    stats_rows.append(dict(
                        cell=cell_id, cls=cls, leg=leg, holdout=split, named_leg=(leg == named_leg),
                        n=n, events_wk=round(n / max(weeks, 1), 2) if n else 0.0,
                        net_bps=summ.get("mean_net_bps", np.nan), t=t,
                        ex_top5_bps=summ.get("ex_top5_bps", np.nan), ex_top1_bps=summ.get("ex_top1_bps", np.nan),
                        capped_bps=summ.get("winner_capped_bps", np.nan), median_bps=summ.get("median_bps", np.nan),
                        share_in_direction=summ.get("share_in_direction", np.nan),
                        universe_bps=universe_bps, placebo_margin_bps=placebo_margin_bps, placebo_t=placebo_t,
                        null_pctile=null_pctile, passes_bar=passes,
                    ))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(STATS_OUT, index=False)
    log.info("wrote %d (cell x holdout x leg) stat rows to %s", len(stats_df), STATS_OUT)
    log.info(REPORT_ONLY_NOTE)
    log.info("run complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("cell_1552.py main() crashed -- see traceback above")
        sys.exit(1)
