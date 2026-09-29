#!/usr/bin/env python3
"""Cells 1,646-1,648 -- research/edgar_desk/PREREG_1646.md (FROZEN 2026-09-29 06:15 UTC).

Pre-announcement run-up (the earnings announcement premium): buy ahead of a firm's OWN
predictable earnings date, sell before or through the event. The expected date E for a
firm-quarter is estimated PURELY from that firm's own item-2.02 8-K history -- the same-quarter
release ~1 year earlier (R1) plus the year-over-year drift observed between R1 and the release
~1 year before that (R2) -- never the actual date of the release being predicted. There is no
fiscal-period field on disk (events_raw.csv has no reportDate/period column), so "same fiscal
quarter" is operationalised as "nearest release in this firm's own history to (target - 365
days), within +/- 45 days" -- a quarter is ~91 days wide, so 45 days cannot cross into an
adjacent quarter's anchor while comfortably tolerating leap years and a few days of calendar
drift.

Cells:
  1,646 -- PRE-WINDOW. Buy MOO session E-5, sell MOC session E-1 -- unless the REAL 8-K's own
           reaction session arrives on or before E-1, in which case the position is closed at
           that (earlier) session's close instead (early-arrival rule -- a real-time risk exit
           triggered by news that has already landed, NOT a forecast input: it never changes
           E, the entry, or which firm-quarters are traded).
  1,647 -- THROUGH-EVENT. Same entry, sell MOC session E+1 unconditionally (classic premium,
           event risk included by construction).
  1,648 -- report-only: 1,647 split by TRAIN-cut terciles of R1's OWN prior-year announcement-
           window volume ratio (the mechanism -- premium concentrated where volume jumps).

Data:
  research/edgar_desk/events_raw.csv (373 MB; streamed in 300k-row chunks, item-2.02 8-Ks only).
  research/overnight_high/alpaca_daily_2019_2024H1.parquet (2019-01-02..2024-06-27) +
  research/overnight_high/panel_2024_2026.parquet          (2024-07-01..2026-09-04)
  (raw daily bars; zero-OHLCV placeholder rows dropped; SPY present in both panels -- no Alpaca
  fetch fallback needed, confirmed on inspection before writing this script).

Splits (by ENTRY session, i.e. session E-5 -- the session PREREG names for day-clustered t):
  TRAIN 2019-01-01..2022-12-31, VAL 2023-01-01..2024-06-30, TEST >= 2024-07-01 SEALED (dropped
  immediately after the split label is assigned; no statistic is ever computed on it here).

Timezone: acceptance_datetime is UTC ('...Z' suffix) -- load_events() below reproduces cell_1633's
verbatim UTC -> America/New_York conversion via zoneinfo (DST-aware). The cell-1,552 refuter found
an earlier pipeline treated these digits as already being ET wall-clock; this cell does a REAL
conversion, per PREREG_1633/1646's explicit instruction.

Firm-quarter formation (form_expected_dates): for symbol S's own sorted, deduplicated release
dates d[0..m-1], and for each i:
  R1 = nearest_anchor(d, i)          -- nearest d[j], j<i, to d[i]-365d, within +/-45d; else SKIP.
  R2 = nearest_anchor(d, R1)         -- nearest d[k], k<R1, to d[R1]-365d, within +/-45d.
  if R2 missing:  E = d[R1] + 365d                              (no drift term available)
  else:           drift = (d[R1]-d[R2]) - 365d
                  if |drift| > 7d: SKIP (last two years' dates disagree too much to trust)
                  else:            E = d[R1] + (d[R1]-d[R2])     (= d[R1] + 365d + drift)
R1 and R2 are always ~1 and ~2 years in i's past -- causality (filings used to form E accepted
long before session E-6) holds by construction, not by a runtime filter; a diagnostic assertion
below still checks it on every formed row and would ERROR-log any violation (none expected).

Duplicate/amended 8-Ks: item-2.02 filings for the same symbol within 14 calendar days of a
previously kept release are collapsed into that release (a true next quarter is ~91 days out);
the EARLIEST acceptance in the group is kept as the disclosure moment (build_release_calendar).

Universe: price >= $3 and 20-day dollar volume >= $5M at session E-6, on THIS symbol's own daily
bar array (same per-symbol-array convention as cell_1633's resolve_symbol_events). Size split
(<= $1B vs larger) is UNAVAILABLE -- no shares-outstanding/market-cap source on disk (same
documented gap as cell_1633/RESULT_1633.md); reported as such, not silently omitted.

Volume-ratio mechanism split (1,648): announcement-window volume ratio = mean dollar volume over
the 3 sessions centered on R1 (R1-1..R1+1) divided by the 20-day dollar-volume average 6 sessions
before R1 (mirrors the E-6 universe reference, applied to R1 instead of E) -- entirely a function
of history before the current firm-quarter's own E-6, so it is causal for THIS event by the same
argument as R1/R2 themselves. TRAIN-only tercile cutpoints, applied to VAL via the same edges
(pd.qcut on TRAIN, then pd.cut everywhere -- never re-derived on VAL, per cell_1633 convention).

Usage:
    python3 research/edgar_desk/cell_1646.py

Outputs: research/edgar_desk/events_1646.csv, research/edgar_desk/RESULT_1646_build.md.
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
log = logging.getLogger("cell_1646")

HERE = Path(__file__).resolve().parent
EVENTS_CSV = HERE / "events_raw.csv"
PRICES_A = HERE.parent / "overnight_high" / "alpaca_daily_2019_2024H1.parquet"
PRICES_B = HERE.parent / "overnight_high" / "panel_2024_2026.parquet"

OUT_EVENTS = HERE / "events_1646.csv"
OUT_RESULT = HERE / "RESULT_1646_build.md"

MIN_PRICE = 3.0
MIN_DVOL20 = 5_000_000.0
COST_BPS_PER_LEG = 5.0
SESSIONS_PER_YEAR = 252
WINNER_CAP = 0.30
SPLIT_GUARD_LO, SPLIT_GUARD_HI = 0.40, 2.50    # cell_1633 convention: any consecutive close/close
                                                # ratio outside this band along the path used is a
                                                # split/reverse-split artifact, not a real return
TEST_TICKER_RE = r'^Z[A-Z]ZZT$'
ET = ZoneInfo("America/New_York")

TRAIN_START, TRAIN_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30")
TEST_START = pd.Timestamp("2024-07-01")

YEAR_DAYS = 365
MATCH_TOL_DAYS = 45      # +/- half a quarter around the "1 year earlier" anchor search
DRIFT_GATE_DAYS = 7      # skip firm-quarter if the last two years' dates disagree by more
DUP_MERGE_DAYS = 14      # collapse same-quarter duplicate/amended 8-Ks filed within this window
MAX_WINDOW_CALENDAR_DAYS = 20   # corpse gate: E-6..E+1 is nominally 7 sessions (~9-12 calendar
                                # days with holidays); a wider span means this symbol's own bar
                                # array jumped a real trading gap (halt, ticker reuse/collision,
                                # zero-volume days silently dropped by load_prices) -- discovered
                                # from FLG/HAPN/VSXY producing 150x-300x "returns" in the first
                                # run (idx_E+1 landed 4+ months after idx_E-6); same family as the
                                # ORB corpse-gate defect (stale bar > 4 days), applied here to the
                                # whole entry/exit window rather than a single bar
ITEM_202_RE = r'(?:^|;)2\.02(?:;|$)'


# ---------------------------------------------------------------------------
# 1. Load the 8-K / item-2.02 events, UTC -> ET conversion (reused verbatim from cell_1633.py's
#    load_events(), parsing only -- see PREREG_1646.md / task instructions).
# ---------------------------------------------------------------------------

def load_events():
    """Stream events_raw.csv in 300k-row chunks; keep 8-K filings with item 2.02.

    Returns one row per (filing, symbol) with accept_utc (UTC-aware) and accept_et (ET-aware,
    DST-correct via zoneinfo). Drops rows with a blank symbol or a test-ticker symbol
    (^Z[A-Z]ZZT$) and rows whose acceptance_datetime fails to parse.
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


def load_prices():
    """Concat the two raw (unadjusted) daily-bar parquets -- no date overlap (A ends 2024-06-27,
    B starts 2024-07-01) -- drop zero-OHLCV placeholder rows from each, and compute a causal
    trailing 20-session dollar-volume average (rolling(20, min_periods=10): value at row t uses
    only rows <= t, no forward leakage). Reused verbatim from cell_1633.py, parsing only."""
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
    if (p["symbol"] == "SPY").sum() == 0:
        log.error("SPY absent from the combined price panel -- SPY-adjustment cannot be computed "
                  "and no Alpaca fetch fallback is implemented in this build; downstream code will "
                  "report spy_adj columns as NaN / UNAVAILABLE, not silently zero")
    return p


# ---------------------------------------------------------------------------
# 2. Statistics helpers (day_clustered_t per research/hod_entry/cell_1445.py convention, reused
#    by re-implementation -- cell_1633.py's own stated convention -- to keep this cell
#    self-contained).
# ---------------------------------------------------------------------------

def day_clustered_t(y, day):
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
    dates = pd.to_datetime(pd.Series(dates)).dropna().unique()
    wk = {(pd.Timestamp(d).isocalendar()[0], pd.Timestamp(d).isocalendar()[1]) for d in dates}
    return max(len(wk), 1)


def mde_line(net_bps):
    """MDE = SD(per-event net bps) / sqrt(n) * 2.5 -- the PREREG's frozen MDE formula, printed
    beside every verdict rather than only once at freeze time (recomputed on the realised n/SD)."""
    y = pd.Series(net_bps).dropna()
    n = len(y)
    if n < 2:
        return np.nan, n
    return float(y.std(ddof=1) / np.sqrt(n) * 2.5), n


# ---------------------------------------------------------------------------
# 3. Release calendar: one row per distinct quarterly release per symbol
# ---------------------------------------------------------------------------

def build_release_calendar(events, merge_gap_days=DUP_MERGE_DAYS):
    """Collapse item-2.02 filings within `merge_gap_days` of a previously kept release (by
    consecutive date_et gap, per symbol) into ONE release -- duplicate/amended 8-Ks refiled a few
    days later, not a new quarter (~91 days out). Keeps the row with the EARLIEST accept_et in
    each group as the disclosure moment."""
    ev = events.sort_values(["symbol", "date_et", "accept_et"]).reset_index(drop=True)
    gap_days = ev.groupby("symbol")["date_et"].diff().dt.days
    new_release = gap_days.isna() | (gap_days > merge_gap_days)
    ev["_grp"] = new_release.cumsum()
    idx_min = ev.groupby(["symbol", "_grp"], sort=False)["accept_et"].idxmin()
    reps = (ev.loc[idx_min, ["symbol", "date_et", "accept_et", "time_et"]]
              .sort_values(["symbol", "date_et"]).reset_index(drop=True))
    log.info("release calendar: %d item-2.02 rows -> %d distinct releases (%d collapsed as "
             "duplicate/amended within %dd); %d symbols",
             len(events), len(reps), len(events) - len(reps), merge_gap_days, reps["symbol"].nunique())
    return reps


# ---------------------------------------------------------------------------
# 4. Expected-date formation E = R1 + (R1 - R2), causal by construction (R1/R2 always in i's past)
# ---------------------------------------------------------------------------

def nearest_anchor(dates, i, target_days=YEAR_DAYS, tol=MATCH_TOL_DAYS):
    """Index j < i in sorted `dates` nearest to dates[i] - target_days, within +/- tol days, or
    None. Only ever searches dates[:i] -- never the row being predicted itself or anything later."""
    if i == 0:
        return None
    target = dates[i] - np.timedelta64(target_days, "D")
    sub = dates[:i]
    pos = int(np.searchsorted(sub, target))
    best_j, best_diff = None, None
    for c in (pos - 1, pos):
        if 0 <= c < len(sub):
            diff = abs((sub[c] - target) / np.timedelta64(1, "D"))
            if diff <= tol and (best_diff is None or diff < best_diff):
                best_j, best_diff = c, diff
    return best_j


def form_expected_dates(reps):
    """Per symbol, walk the sorted release array and try to form E for every release using ONLY
    earlier releases of the SAME firm (see module docstring for the R1/R2/drift-gate rule)."""
    out = []
    n_no_r1 = n_drift_gate = n_r1_only = n_with_drift = n_causality_violation = 0
    n_total = len(reps)
    for sym, g in reps.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        d = g["date_et"].to_numpy()
        accept = g["accept_et"].to_numpy()
        tET = g["time_et"].to_numpy()
        for i in range(len(d)):
            r1 = nearest_anchor(d, i)
            if r1 is None:
                n_no_r1 += 1
                continue
            r2 = nearest_anchor(d, r1)
            if r2 is None:
                E = d[r1] + np.timedelta64(YEAR_DAYS, "D")
                drift = np.nan
                n_r1_only += 1
            else:
                drift = float((d[r1] - d[r2]) / np.timedelta64(1, "D") - YEAR_DAYS)
                if abs(drift) > DRIFT_GATE_DAYS:
                    n_drift_gate += 1
                    continue
                E = d[r1] + (d[r1] - d[r2])
                n_with_drift += 1
            if (pd.Timestamp(E) - pd.Timestamp(d[r1])).days < 300:
                n_causality_violation += 1
                log.error("causality check failed for %s row %d: E %.10s too close to its own "
                          "R1 %.10s -- dropping this firm-quarter", sym, i, str(E), str(d[r1]))
                continue
            out.append(dict(symbol=sym, actual_date=pd.Timestamp(d[i]),
                             actual_accept_et=accept[i], actual_time_et=tET[i],
                             E=pd.Timestamp(E), r1_date=pd.Timestamp(d[r1]),
                             r2_date=(pd.Timestamp(d[r2]) if r2 is not None else pd.NaT),
                             drift_days=drift))
    log.info("expected-date formation: %d releases -> no-R1(skip)=%d, drift-gate-skip(>%dd)=%d, "
             "formed R1-only(no drift term)=%d, formed with drift=%d, causality_violation=%d",
             n_total, n_no_r1, DRIFT_GATE_DAYS, n_drift_gate, n_r1_only, n_with_drift,
             n_causality_violation)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 5. Resolve sessions/prices per symbol, universe filter, early-arrival rule, R1 volume ratio
# ---------------------------------------------------------------------------

def resolve_trades(expected, prices):
    """Map each (symbol, E) firm-quarter instance onto THIS symbol's own trading-day array:
    idx_E (first session >= E), E-6/E-5/E-1/E+1 as offsets from it, the universe filter at E-6,
    the actual release's own reaction session (after-close bump, cell_1633 convention) for the
    early-arrival exit rule and the expected-date hit rate, and R1's announcement-window volume
    ratio for cell 1,648."""
    price_groups = {sym: g for sym, g in prices.groupby("symbol", sort=False)}
    lost = dict(no_price_series=0, e_window_out_of_range=0, session_gap_too_wide=0,
                split_guard=0, below_universe=0)
    n_early_arrival_clip = 0
    out = []
    for sym, g in expected.groupby("symbol", sort=False):
        arr = price_groups.get(sym)
        if arr is None:
            lost["no_price_series"] += len(g)
            continue
        bar_date = arr["bar_date"].to_numpy()
        open_ = arr["open"].to_numpy()
        close = arr["close"].to_numpy()
        dvol20 = arr["dvol20"].to_numpy()
        dvol = arr["close"].to_numpy() * arr["volume"].to_numpy()
        n = len(bar_date)
        if n == 0:
            lost["no_price_series"] += len(g)
            continue

        g = g.reset_index(drop=True)
        E = g["E"].to_numpy()
        idx_E = np.searchsorted(bar_date, E, side="left")
        idx_E6, idx_E5, idx_E1m, idx_E1p = idx_E - 6, idx_E - 5, idx_E - 1, idx_E + 1

        actual_date = g["actual_date"].to_numpy()
        idx_actual = np.clip(np.searchsorted(bar_date, actual_date, side="left"), 0, n - 1)
        same_day = bar_date[idx_actual] == actual_date
        after_close = np.array([t >= dtime(16, 0) for t in g["actual_time_et"]])
        idx_reaction = idx_actual + (after_close & same_day).astype(int)
        idx_reaction = np.clip(idx_reaction, 0, n - 1)

        r1_date = g["r1_date"].to_numpy()
        idx_r1 = np.searchsorted(bar_date, r1_date, side="left")

        for k in range(len(g)):
            if not (0 <= idx_E6[k] and idx_E5[k] >= 0 and idx_E1p[k] < n and idx_E5[k] <= idx_E1m[k]):
                lost["e_window_out_of_range"] += 1
                continue

            window_days = (bar_date[idx_E1p[k]] - bar_date[idx_E6[k]]) / np.timedelta64(1, "D")
            if window_days > MAX_WINDOW_CALENDAR_DAYS:
                lost["session_gap_too_wide"] += 1        # corpse gate -- see MAX_WINDOW_CALENDAR_DAYS
                continue

            path = close[idx_E6[k]:idx_E1p[k] + 1]
            path_ratios = path[1:] / path[:-1]
            if np.any((path_ratios < SPLIT_GUARD_LO) | (path_ratios > SPLIT_GUARD_HI)):
                lost["split_guard"] += 1     # unadjusted split/reverse-split inside E-6..E+1
                continue

            price_ok = close[idx_E6[k]] >= MIN_PRICE
            dvol_ok = np.nan_to_num(dvol20[idx_E6[k]], nan=-1.0) >= MIN_DVOL20
            if not (price_ok and dvol_ok):
                lost["below_universe"] += 1
                continue

            entry_idx = int(idx_E5[k])
            exit1646_default_idx = int(idx_E1m[k])
            exit1647_idx = int(idx_E1p[k])
            react_idx = int(idx_reaction[k])

            early_arrival = react_idx <= exit1646_default_idx
            if early_arrival:
                exit1646_idx = react_idx
                if exit1646_idx < entry_idx:
                    n_early_arrival_clip += 1
                    exit1646_idx = entry_idx      # same-session round trip; never exit before entry
            else:
                exit1646_idx = exit1646_default_idx

            hit_window = exit1646_default_idx <= react_idx <= exit1647_idx

            r1i = int(idx_r1[k])
            vol_ratio = np.nan
            if 6 <= r1i and r1i + 1 < n:
                base_dvol = dvol20[r1i - 6]
                if base_dvol and not np.isnan(base_dvol) and base_dvol > 0:
                    vol_ratio = float(dvol[r1i - 1:r1i + 2].mean() / base_dvol)

            out.append(dict(
                symbol=sym, actual_date=pd.Timestamp(actual_date[k]), E=pd.Timestamp(E[k]),
                r1_date=pd.Timestamp(r1_date[k]), r2_date=g["r2_date"].to_numpy()[k],
                drift_days=g["drift_days"].to_numpy()[k],
                entry_date=pd.Timestamp(bar_date[entry_idx]), entry_open=float(open_[entry_idx]),
                exit1646_date=pd.Timestamp(bar_date[exit1646_idx]),
                exit1646_close=float(close[exit1646_idx]),
                exit1647_date=pd.Timestamp(bar_date[exit1647_idx]),
                exit1647_close=float(close[exit1647_idx]),
                early_arrival=bool(early_arrival), hit_window=bool(hit_window),
                vol_ratio=vol_ratio))

    lost_total = sum(lost.values())
    log.info("resolve_trades: kept %d across %d symbols; lost %s (total %d); early-arrival exit "
             "clipped to the entry session (news before entry) %d times",
             len(out), expected["symbol"].nunique(), lost, lost_total, n_early_arrival_clip)
    return pd.DataFrame(out)


def spy_series(prices):
    spy = prices[prices["symbol"] == "SPY"].sort_values("bar_date")
    if spy.empty:
        return None, None
    return spy["bar_date"].to_numpy(), spy["close"].to_numpy()


def spy_ret(spy_date, spy_close, date_a, date_b):
    if spy_date is None:
        return np.full(len(date_a), np.nan)
    ia = np.clip(np.searchsorted(spy_date, date_a, side="left"), 0, len(spy_date) - 1)
    ib = np.clip(np.searchsorted(spy_date, date_b, side="left"), 0, len(spy_date) - 1)
    return spy_close[ib] / spy_close[ia] - 1.0


# ---------------------------------------------------------------------------
# 6. Split assignment, TRAIN-only tercile edges, cell scoring
# ---------------------------------------------------------------------------

def assign_split(r):
    """Split by ENTRY session (E-5) -- the session PREREG names for day-clustered t. TEST
    (entry_date >= 2024-07-01) is SEALED: dropped here, never scored, never written out."""
    r = r.copy()
    r["split"] = np.select(
        [r["entry_date"].between(TRAIN_START, TRAIN_END),
         r["entry_date"].between(VAL_START, VAL_END),
         r["entry_date"] >= TEST_START],
        ["TRAIN", "VAL", "TEST"], default="PRE_2019_OR_GAP")
    n_test = int((r["split"] == "TEST").sum())
    n_gap = int((r["split"] == "PRE_2019_OR_GAP").sum())
    log.info("split assignment (by entry session): TRAIN=%d VAL=%d (TEST=%d SEALED -- dropped "
             "now, gap/pre-2019=%d dropped)", int((r["split"] == "TRAIN").sum()),
             int((r["split"] == "VAL").sum()), n_test, n_gap)
    return r[r["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)


def assign_tercile(r):
    """TRAIN-only tercile edges on vol_ratio (non-NaN rows), applied to both TRAIN and VAL via
    the SAME edges (pd.qcut on TRAIN, pd.cut everywhere) -- never re-derived on VAL."""
    train_vr = r.loc[(r["split"] == "TRAIN") & r["vol_ratio"].notna(), "vol_ratio"]
    if len(train_vr) < 30:
        log.warning("only %d TRAIN rows with a valid vol_ratio -- tercile edges may be unstable",
                    len(train_vr))
    _, edges = pd.qcut(train_vr, 3, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    n_terciles = len(edges) - 1
    r = r.copy()
    r["tercile"] = pd.cut(r["vol_ratio"], bins=edges, labels=list(range(1, n_terciles + 1)),
                           include_lowest=True)
    log.info("vol_ratio tercile edges (TRAIN): %s -> %d terciles", np.round(edges, 3).tolist(),
             n_terciles)
    return r, n_terciles


def net_return_bps(ret_frac):
    """5 bps per auction leg, 2 legs (MOO entry + MOC exit), long-only -- no borrow term."""
    return (ret_frac - 2 * (COST_BPS_PER_LEG / 10_000.0)) * 10_000.0


def score_cell(rows, cell, ret_col, spy_col):
    """Per-split stats block. rows must carry: split, entry_date (cluster key), net_bps (=ret_col
    net of cost, already in bps), spy_col (SPY return over the identical window, in bps)."""
    out = []
    for split in ("TRAIN", "VAL"):
        sub = rows[rows["split"] == split]
        n = len(sub)
        if n == 0:
            out.append(dict(cell=cell, split=split, n=0))
            continue
        net_bps = sub[ret_col]
        mean_net_bps = float(net_bps.mean())
        t = day_clustered_t(net_bps, sub["entry_date"])
        ex5 = ex_top_mean(net_bps, 0.05)
        ex1 = ex_top_mean(net_bps, 0.01)
        wcap = winner_capped_mean(net_bps / 10_000.0) * 10_000.0
        spy_adj_net_bps = float((net_bps - sub[spy_col]).mean())
        wk = weeks_spanned(sub["entry_date"])
        ev_wk = n / wk
        hit_rate = float(sub["hit_window"].mean())
        early_arrival_share = float(sub["early_arrival"].mean())
        mde, _ = mde_line(net_bps)
        by_year = sub.assign(yr=sub["entry_date"].dt.year).groupby("yr")[ret_col].agg(["mean", "count"])
        years_pos = "/".join(f"{int(y)}:{'+' if m > 0 else '-'}" for y, m in by_year["mean"].items())
        out.append(dict(cell=cell, split=split, n=n, events_wk=round(ev_wk, 2),
                         mean_net_bps=round(mean_net_bps, 1), t=round(t, 2) if pd.notna(t) else np.nan,
                         ex_top5_bps=round(ex5, 1), ex_top1_bps=round(ex1, 1),
                         winner_capped_bps=round(wcap, 1), spy_adj_net_bps=round(spy_adj_net_bps, 1),
                         hit_rate=round(hit_rate, 3), early_arrival_share=round(early_arrival_share, 3),
                         mde_bps=round(mde, 1) if pd.notna(mde) else np.nan, years_positive=years_pos))
    return out


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== cell_1646: pre-announcement run-up (PREREG_1646.md) ===")
    events = load_events()
    prices = load_prices()
    spy_date, spy_close = spy_series(prices)

    reps = build_release_calendar(events)
    expected = form_expected_dates(reps)
    resolved = resolve_trades(expected, prices)
    resolved = assign_split(resolved)

    resolved["ret1646_frac"] = resolved["exit1646_close"] / resolved["entry_open"] - 1.0
    resolved["ret1647_frac"] = resolved["exit1647_close"] / resolved["entry_open"] - 1.0
    resolved["net1646_bps"] = net_return_bps(resolved["ret1646_frac"])
    resolved["net1647_bps"] = net_return_bps(resolved["ret1647_frac"])
    resolved["spy1646_bps"] = spy_ret(spy_date, spy_close, resolved["entry_date"].to_numpy(),
                                       resolved["exit1646_date"].to_numpy()) * 10_000.0
    resolved["spy1647_bps"] = spy_ret(spy_date, spy_close, resolved["entry_date"].to_numpy(),
                                       resolved["exit1647_date"].to_numpy()) * 10_000.0

    resolved, n_terciles = assign_tercile(resolved)

    stats = []
    stats += score_cell(resolved, "1646", "net1646_bps", "spy1646_bps")
    stats += score_cell(resolved, "1647", "net1647_bps", "spy1647_bps")

    tercile_rows = []
    for split in ("TRAIN", "VAL"):
        for tc in range(1, n_terciles + 1):
            sub = resolved[(resolved["split"] == split) & (resolved["tercile"] == tc)]
            if len(sub) == 0:
                continue
            tercile_rows.append(dict(split=split, tercile=tc, n=len(sub),
                                      mean_net1647_bps=round(sub["net1647_bps"].mean(), 1),
                                      t=round(day_clustered_t(sub["net1647_bps"], sub["entry_date"]), 2)))
    tercile_table = pd.DataFrame(tercile_rows)
    log.info("1,648 volume-tercile table:\n%s", tercile_table.to_string())

    per_quarter_rows = []
    for split in ("TRAIN", "VAL"):
        sub = resolved[resolved["split"] == split].assign(
            yq=lambda d: d["entry_date"].dt.year.astype(str) + "Q" + d["entry_date"].dt.quarter.astype(str))
        for yq, g in sub.groupby("yq"):
            per_quarter_rows.append(dict(split=split, year_quarter=yq, n=len(g),
                                          mean_net1646_bps=round(g["net1646_bps"].mean(), 1),
                                          mean_net1647_bps=round(g["net1647_bps"].mean(), 1)))
    per_quarter_table = pd.DataFrame(per_quarter_rows).sort_values(["split", "year_quarter"])

    events_cols = ["symbol", "split", "actual_date", "E", "r1_date", "r2_date", "drift_days",
                   "entry_date", "entry_open", "exit1646_date", "exit1646_close",
                   "exit1647_date", "exit1647_close", "early_arrival", "hit_window",
                   "vol_ratio", "tercile", "ret1646_frac", "ret1647_frac", "net1646_bps",
                   "net1647_bps", "spy1646_bps", "spy1647_bps"]
    resolved[events_cols].to_csv(OUT_EVENTS, index=False)
    log.info("wrote %s (%d rows)", OUT_EVENTS, len(resolved))

    stats_df = pd.DataFrame(stats)
    log.info("cell stats:\n%s", stats_df.to_string())

    write_result_md(stats_df, tercile_table, per_quarter_table, resolved, n_terciles)
    log.info("DONE")


def write_result_md(stats_df, tercile_table, per_quarter_table, resolved, n_terciles):
    """Pass bar (frozen, VAL, per cell): mean net >= +40bps, day-clustered t >= 2.5, ex-top-5% > 0,
    >= 20 events/week in season, TRAIN same-sign t >= 1, SPY-adjusted (net) >= +25bps, volume-
    tercile table monotone on both halves, hit rate >= 70%."""
    def row(cell, split):
        r = stats_df[(stats_df["cell"] == cell) & (stats_df["split"] == split)]
        return r.iloc[0] if len(r) else None

    val1646, train1646 = row("1646", "VAL"), row("1646", "TRAIN")
    val1647, train1647 = row("1647", "VAL"), row("1647", "TRAIN")

    def bar_line(cell, val, train):
        if val is None or val.get("n", 0) == 0:
            return f"| {cell} | NO VAL EVENTS | -- | -- | -- | -- | -- | -- | -- | FAIL (no data) |"
        mean_ok = val["mean_net_bps"] >= 40
        t_ok = pd.notna(val["t"]) and val["t"] >= 2.5
        ex5_ok = pd.notna(val["ex_top5_bps"]) and val["ex_top5_bps"] > 0
        freq_ok = val["events_wk"] >= 20
        train_ok = (train is not None and train.get("n", 0) > 0 and pd.notna(train["t"])
                    and np.sign(train["t"]) == np.sign(val["t"] if pd.notna(val["t"]) else 0)
                    and train["t"] >= 1)
        spy_ok = pd.notna(val["spy_adj_net_bps"]) and val["spy_adj_net_bps"] >= 25
        hit_ok = val["hit_rate"] >= 0.70
        passed = mean_ok and t_ok and ex5_ok and freq_ok and train_ok and spy_ok and hit_ok
        verdict = "PASS" if passed else "FAIL"
        return (f"| {cell} | n={val['n']} | {val['mean_net_bps']:.1f} ({'OK' if mean_ok else 'no'}) "
                f"| t={val['t']} ({'OK' if t_ok else 'no'}) | ex5={val['ex_top5_bps']:.1f} "
                f"({'OK' if ex5_ok else 'no'}) | {val['events_wk']:.2f}/wk ({'OK' if freq_ok else 'no'}) "
                f"| TRAIN t={train['t'] if train is not None else 'NA'} ({'OK' if train_ok else 'no'}) "
                f"| spy_adj={val['spy_adj_net_bps']:.1f} ({'OK' if spy_ok else 'no'}) "
                f"| hit={val['hit_rate']:.2f} ({'OK' if hit_ok else 'no'}) | **{verdict}** |")

    tc_val = tercile_table[tercile_table["split"] == "VAL"].sort_values("tercile")
    tc_train = tercile_table[tercile_table["split"] == "TRAIN"].sort_values("tercile")
    mono_val = bool(tc_val["mean_net1647_bps"].is_monotonic_increasing) if len(tc_val) == n_terciles else False
    mono_train = bool(tc_train["mean_net1647_bps"].is_monotonic_increasing) if len(tc_train) == n_terciles else False

    n_causal_flag = int((resolved["E"] - resolved["r1_date"]).dt.days.lt(300).sum())

    lines = []
    lines.append("# RESULT 1,646-1,648 -- pre-announcement run-up (BUILD, independent check pending)")
    lines.append("")
    lines.append(f"Total resolved/traded firm-quarters (TRAIN+VAL): {len(resolved)}. "
                 f"Causality-check violations (E within 300d of its own R1): {n_causal_flag} "
                 "(expected 0 -- E is always R1 + ~365d by construction).")
    lines.append("")
    lines.append("## Pass bar (frozen; VAL, per cell) -- mean net >=40bps, t>=2.5, ex-top-5%>0, "
                 ">=20 ev/wk, TRAIN same-sign t>=1, SPY-adj net >=25bps, hit rate >=70%")
    lines.append("")
    lines.append("| cell | n | mean net bps | t | ex-top-5% | events/wk | TRAIN t | SPY-adj net | "
                 "hit rate | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(bar_line("1646", val1646, train1646))
    lines.append(bar_line("1647", val1647, train1647))
    lines.append("")
    lines.append(f"1,648 (report-only, mechanism split of 1,647): volume-tercile table monotone "
                 f"on VAL = {mono_val}, on TRAIN = {mono_train}.")
    lines.append("")
    lines.append("## Full stats (TRAIN + VAL, both cells)")
    lines.append("")
    lines.append(stats_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 1,648 -- volume-ratio tercile table (mean net bps, 1,647 exit rule)")
    lines.append("")
    lines.append(tercile_table.to_markdown(index=False) if len(tercile_table) else "(empty)")
    lines.append("")
    lines.append("## Per-quarter table (net bps, both cells)")
    lines.append("")
    lines.append(per_quarter_table.to_markdown(index=False) if len(per_quarter_table) else "(empty)")
    lines.append("")
    lines.append("## Size split (<=$1B vs larger)")
    lines.append("")
    lines.append("UNAVAILABLE -- no shares-outstanding/market-cap source on disk (same documented "
                 "gap as cell_1633/RESULT_1633.md). Not computed; not silently defaulted.")
    lines.append("")
    lines.append("## Caveats for the independent-check agent (read as an adversary)")
    lines.append("")
    lines.append("1. **\"Same fiscal quarter\" is inferred, not observed.** events_raw.csv has no "
                 "reportDate/period-of-report column, so R1/R2 are found by nearest-date matching "
                 "(target +/-45d) rather than an explicit fiscal-quarter key. An item-2.02 8-K that "
                 "is NOT a regular quarterly release (e.g. a preliminary/ad hoc results filing) can "
                 "pollute a firm's release calendar and get selected as an anchor; not filtered here.")
    lines.append("2. **Duplicate/amendment merge is a 14-day heuristic** (consecutive-gap cumsum, "
                 "not a true greedy-interval merge from the last KEPT release) -- see "
                 "build_release_calendar docstring for the exact edge case this can miss.")
    lines.append("3. **Early-arrival clip**: when the real 8-K's reaction session lands ON OR BEFORE "
                 "the planned entry session (E-5), cell 1,646 closes on the entry session itself "
                 "(same-day round trip) rather than dropping the event -- see resolve_trades' "
                 "n_early_arrival_clip count in the run log. This keeps n intact but can print a "
                 "near-zero-duration trade for a badly-estimated E; hit_window/hit rate still "
                 "penalizes these appropriately.")
    lines.append("4. **Volume-ratio window (1,648) is a documented choice, not in the PREREG's "
                 "numeric detail**: mean dollar volume over R1-1..R1+1 divided by dvol20 at R1-6. "
                 "A different window could change the tercile table materially -- rebuild agent "
                 "should treat this as a named parameter to vary, not assume it is the only choice.")
    lines.append("5. **SPY-adjustment**: SPY IS present in both parquet panels (confirmed before "
                 "writing this script) -- no Alpaca fetch fallback was implemented or needed.")
    lines.append("6. **Corpse gate (session_gap_too_wide) added after the first run found 150x-300x "
                 "\"returns\"** on FLG/HAPN/VSXY: their own per-symbol bar array has a multi-month "
                 "gap (real halt, or a recycled ticker used by a different company later -- both "
                 "seen on inspection) spanning the same calendar window that load_prices()'s "
                 "zero-OHLCV drop silently removes non-trading days, so `idx_E+1` landed on a bar "
                 "many months after `idx_E-6`. Fixed by requiring bar_date[idx_E+1] - "
                 f"bar_date[idx_E-6] <= {MAX_WINDOW_CALENDAR_DAYS} calendar days (a nominal 7-session "
                 "window is ~9-12 days with holidays); see resolve_trades' session_gap_too_wide lost "
                 "count. A cell_1633-convention SPLIT_GUARD (any consecutive close/close ratio "
                 "outside [0.40, 2.50] along the E-6..E+1 path) is also applied (split_guard lost "
                 "count) -- it caught a real WMT 3-for-1 split (2024-02-26) that had produced a "
                 "-6,510bps 'trade' in the first guarded run. Not screened beyond this: intra-day "
                 "halts/circuit breakers and genuine >50% single-name crashes (biotech trial "
                 "failures, meme-stock reversals) are real returns, kept as-is, and still drive the "
                 "tail-dependence gap between mean and ex-top-5% below -- read that gap as real "
                 "single-name risk, not further data corruption, unless the independent check finds "
                 "otherwise.")
    lines.append("7. **TEST is sealed**: 2024-07-01 onward was dropped immediately after split "
                 "assignment (assign_split), before any statistic in this file. Nothing above used it.")
    OUT_RESULT.write_text("\n".join(lines) + "\n")
    log.info("wrote %s (%d lines)", OUT_RESULT, len(lines))


if __name__ == "__main__":
    main()
