#!/usr/bin/env python3
"""Cell 1,701 step 1 -- earnings-session calendar + candidate universe for pool E1.

Builds the point-in-time candidate list for PREREG_1701.md (frozen 2026-10-02, implemented
here exactly as written -- no filter added after seeing numbers): every 8-K / 8-K-A filing
with Item 2.02 (earnings release), mapped to its event session (the first regular NYSE
session whose 09:30 ET open is >= 15 minutes after SEC acceptance), screened to a
point-in-time common-stock, liquid universe. Writes 1701_candidates.csv and
1701_completeness.md. Does NOT open cache.db or bars_sip.db -- bar backfill is a separate,
not-yet-launched step (1701_backfill_queue.sh).

Trading-day calendar: no pandas_market_calendars / exchange_calendars installed on this node
(checked 2026-10-02). Trading days 2024-07-01..2026-09-30 are the union of bar_date values
actually present in the Databento EQUS.SUMMARY daily panel -- a real trading session is
exactly a day that panel has bars for (the same self-contained-calendar trick as
research/edgar_desk/rebuild_1633.py's SPY proxy, generalised to "any bar exists"). The read
filter pads back to filing_date >= 2024-06-01 solely to resolve the late-June-2024 boundary
rollover into the 07-01 window open; for that short pad period (the panel has no bars yet) we
use NYSE business days minus the one NYSE holiday that falls in that window (Juneteenth,
2024-06-19) -- the only non-weekend closure, so no calendar package is needed for 18 days.
Early closes are not separately flagged in this step (the session rule depends only on the
09:30 ET open, which an early close does not move); later steps that walk intraday exits must
still consult a half-day list.

Memory: events_raw.csv is 4.4M rows. Read via pd.read_csv(chunksize=...), filtered to
(8-K or 8-K/A) AND Item 2.02 AND filing_date >= 2024-06-01 inside each chunk before
concatenation, per the node's 2-CPU / ~1.8 GB free RAM budget.
"""
from __future__ import annotations

import bisect
import logging
import re
import sys
import time as _time
from datetime import date, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from research.scripts.pit_listings import PitListings
except ImportError:
    sys.path.insert(0, str(ROOT / "research" / "scripts"))
    from pit_listings import PitListings  # type: ignore

EVENTS_CSV = ROOT / "research/edgar_desk/events_raw.csv"
PANEL_PATHS = (
    ROOT / "data/research/databento/equs_daily_2024H2.parquet",
    ROOT / "data/research/databento/equs_daily_2025_2026.parquet",
)
OUT_DIR = ROOT / "research/orb_earn"
CANDIDATES_CSV = OUT_DIR / "1701_candidates.csv"
COMPLETENESS_MD = OUT_DIR / "1701_completeness.md"
LOG_PATH = OUT_DIR / "1701_calendar.log"

EVENTS_COLS = ["cik", "symbol", "form", "filing_date", "acceptance_datetime", "items"]
READ_FLOOR = "2024-06-01"          # filing_date read-time floor (pads for boundary rollover)
WINDOW_LO = pd.Timestamp("2024-07-01")
WINDOW_HI = pd.Timestamp("2026-09-30")
HALF_SPLIT = pd.Timestamp("2025-06-30")   # half A ends here, half B starts 2025-07-01
MIN_PRIOR_CLOSE = 5.0
MIN_ADV20_DOLLAR = 5_000_000.0
TEST_TICKER_RE = re.compile(r"^Z[A-Z]ZZT$")
CHUNKSIZE = 500_000

log = logging.getLogger("1701_calendar")


def setup_logging() -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def read_8k_202_events() -> tuple[pd.DataFrame, int]:
    """Chunked scan of events_raw.csv -> rows that are 8-K/8-K-A, item 2.02, filing_date floor."""
    log.info("reading %s in chunks of %d ...", EVENTS_CSV, CHUNKSIZE)
    chunks, total_rows, kept_rows = [], 0, 0
    t0 = _time.time()
    reader = pd.read_csv(EVENTS_CSV, usecols=EVENTS_COLS, dtype=str, chunksize=CHUNKSIZE,
                          keep_default_na=False)
    for i, chunk in enumerate(reader, 1):
        total_rows += len(chunk)
        f = chunk[chunk["form"].isin(["8-K", "8-K/A"])
                  & chunk["items"].str.contains("2.02", regex=False)
                  & (chunk["filing_date"] >= READ_FLOOR)]
        kept_rows += len(f)
        chunks.append(f)
        log.info("chunk %d: +%d rows read (cum %d), +%d kept (cum %d) elapsed %.1fs",
                  i, len(chunk), total_rows, len(f), kept_rows, _time.time() - t0)
    events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=EVENTS_COLS)
    log.info("events scan done: %d rows read, %d matched 8-K(/A)+item2.02+filing_date>=%s in %.1fs",
              total_rows, len(events), READ_FLOOR, _time.time() - t0)
    return events, total_rows


def load_panel() -> pd.DataFrame:
    """Databento daily panel (bar_date,symbol,close,volume) with prior_close/adv20 per symbol."""
    log.info("loading Databento daily panel(s) ...")
    frames = []
    for p in PANEL_PATHS:
        d = pd.read_parquet(p, columns=["bar_date", "symbol", "close", "volume"])
        log.info("  %s: %d rows", p.name, len(d))
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel["bar_date"] = pd.to_datetime(panel["bar_date"])
    panel["symbol"] = panel["symbol"].astype("category")
    panel = panel.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    panel["prior_close"] = panel.groupby("symbol")["close"].shift(1)
    panel["adv20_dollar"] = (panel.groupby("symbol")["dollar_vol"]
                              .transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean()))
    log.info("panel built: %d rows, %d distinct symbols, %d distinct sessions (%s..%s)",
              len(panel), panel["symbol"].nunique(), panel["bar_date"].nunique(),
              panel["bar_date"].min().date(), panel["bar_date"].max().date())
    return panel


def trading_day_calendar(panel: pd.DataFrame) -> list[date]:
    """Sorted real trading days: panel bar_dates (2024-07-01+) plus a June-2024 pad."""
    panel_days = sorted(panel["bar_date"].dt.date.unique().tolist())
    june_pad = [d.date() for d in pd.bdate_range("2024-06-01", "2024-06-30")
                if d.date() != date(2024, 6, 19)]  # Juneteenth, the one NYSE holiday in the pad
    all_days = sorted(set(panel_days) | set(june_pad))
    log.info("trading-day calendar: %d sessions (%d June-2024 pad days ahead of the panel start %s)",
              len(all_days), len(set(june_pad) - set(panel_days)), panel_days[0])
    return all_days


def event_session(threshold_et: pd.Timestamp, trading_days: list) -> date | None:
    """First trading day whose 09:30 ET open is >= `threshold_et` (= acceptance + 15 min)."""
    d = threshold_et.date()
    floor_date = d if threshold_et.time() <= dtime(9, 30) else d + timedelta(days=1)
    i = bisect.bisect_left(trading_days, floor_date)
    if i >= len(trading_days):
        return None
    return trading_days[i]


def release_slot(t_et: pd.Timestamp) -> str:
    tt = t_et.time()
    if tt < dtime(9, 30):
        return "premarket"
    if tt < dtime(16, 0):
        return "intraday_prior"
    return "after_close"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()
    t0 = _time.time()
    log.info("=== 1701_calendar starting ===")

    events, events_read = read_8k_202_events()
    reasons: dict[str, int] = {}

    events["symbol"] = events["symbol"].str.strip()
    no_symbol_mask = events["symbol"].eq("")
    reasons["no_symbol"] = int(no_symbol_mask.sum())
    events = events[~no_symbol_mask].copy()

    events["acceptance_utc"] = pd.to_datetime(events["acceptance_datetime"], utc=True, errors="coerce")
    bad_ts = events["acceptance_utc"].isna()
    if bad_ts.any():
        log.warning("%d rows have unparseable acceptance_datetime -- dropped", int(bad_ts.sum()))
        events = events[~bad_ts].copy()
    events["acceptance_et"] = events["acceptance_utc"].dt.tz_convert("America/New_York")
    events["release_slot"] = events["acceptance_et"].apply(release_slot)

    panel = load_panel()
    trading_days = trading_day_calendar(panel)

    threshold = events["acceptance_et"] + pd.Timedelta(minutes=15)
    log.info("computing event session for %d rows (bisect over %d trading days) ...",
              len(events), len(trading_days))
    events["session"] = threshold.apply(lambda t: event_session(t, trading_days))
    events["session_ts"] = pd.to_datetime(events["session"])

    oow_mask = events["session_ts"].isna() | (events["session_ts"] < WINDOW_LO) | (events["session_ts"] > WINDOW_HI)
    reasons["out_of_window"] = int(oow_mask.sum())
    events = events[~oow_mask].copy()
    log.info("session computed + windowed: %d candidates remain", len(events))

    events = events.sort_values(["symbol", "session_ts", "acceptance_utc"])
    dup_mask = events.duplicated(subset=["symbol", "session_ts"], keep="first")
    reasons["dup"] = int(dup_mask.sum())
    events = events[~dup_mask].copy()

    log.info("PIT common-stock screen via pit_listings (coverage %s) ...", PitListings().coverage)
    pit = PitListings()
    test_ticker_mask = events["symbol"].str.match(TEST_TICKER_RE)
    not_common_mask = pd.Series(False, index=events.index)
    events["session_month"] = events["session_ts"].dt.strftime("%Y%m")
    for month, idx in events.groupby("session_month").groups.items():
        sample_date = events.loc[idx, "session_ts"].iloc[0]
        common = pit.common_stock_symbols(sample_date)
        not_common_mask.loc[idx] = ~events.loc[idx, "symbol"].isin(common)
    not_common_mask = not_common_mask | test_ticker_mask
    reasons["not_common"] = int(not_common_mask.sum())
    events = events[~not_common_mask].copy()
    events["pit_common"] = True

    events = events.merge(
        panel[["symbol", "bar_date", "prior_close", "adv20_dollar"]],
        left_on=["symbol", "session_ts"], right_on=["symbol", "bar_date"], how="left")
    not_listed_mask = events["bar_date"].isna()
    reasons["not_listed"] = int(not_listed_mask.sum())
    events = events[~not_listed_mask].copy()

    price_mask = events["prior_close"].isna() | (events["prior_close"] < MIN_PRIOR_CLOSE)
    reasons["price"] = int(price_mask.sum())
    events = events[~price_mask].copy()

    adv_mask = events["adv20_dollar"].isna() | (events["adv20_dollar"] < MIN_ADV20_DOLLAR)
    reasons["adv"] = int(adv_mask.sum())
    events = events[~adv_mask].copy()

    events["half"] = np.where(events["session_ts"] <= HALF_SPLIT, "A", "B")

    out = pd.DataFrame({
        "symbol": events["symbol"],
        "cik": events["cik"],
        "session": events["session_ts"].dt.strftime("%Y-%m-%d"),
        "acceptance_et": events["acceptance_et"].apply(lambda t: t.isoformat()),
        "release_slot": events["release_slot"],
        "half": events["half"],
        "prior_close": events["prior_close"].round(4),
        "adv20_dollar": events["adv20_dollar"].round(2),
        "pit_common": events["pit_common"],
        "form": events["form"],
    }).sort_values(["session", "symbol"]).reset_index(drop=True)
    out.to_csv(CANDIDATES_CSV, index=False)
    log.info("wrote %s (%d candidates)", CANDIDATES_CSV, len(out))

    write_completeness(events_read, reasons, out)
    log.info("=== 1701_calendar DONE in %.1fs ===", _time.time() - t0)
    return 0


def write_completeness(events_read: int, reasons: dict, out: pd.DataFrame) -> None:
    half_counts = out["half"].value_counts().reindex(["A", "B"], fill_value=0)
    sessions = pd.to_datetime(out["session"])
    full_weeks = pd.period_range(start=WINDOW_LO, end=WINDOW_HI, freq="W-SUN")
    wk_counts = sessions.dt.to_period("W-SUN").value_counts().reindex(full_weeks, fill_value=0)
    sample = out.sample(min(5, len(out)), random_state=1).sort_values("session") if len(out) else out
    sample_lines = []
    for _, r in sample.iterrows():
        sample_lines.append(f"  {r['symbol']} session={r['session']} acceptance_et={r['acceptance_et']}")

    lines = [
        "# 1701_calendar completeness (cell 1,701 step 1)",
        "",
        f"events_raw.csv rows read: {events_read}",
        f"drop reasons (sequential, each on the rows remaining after prior drops):",
        f"  no_symbol: {reasons.get('no_symbol', 0)}",
        f"  out_of_window (session outside 2024-07-01..2026-09-30): {reasons.get('out_of_window', 0)}",
        f"  dup (same symbol+session, later acceptance): {reasons.get('dup', 0)}",
        f"  not_common (incl. test-ticker ^Z[A-Z]ZZT$ and non-'C' security_type): {reasons.get('not_common', 0)}",
        f"  not_listed (no Databento bar on the event session day): {reasons.get('not_listed', 0)}",
        f"  price (prior_close < $5 or unknown): {reasons.get('price', 0)}",
        f"  adv (adv20_dollar < $5M or unknown): {reasons.get('adv', 0)}",
        f"final candidates: {len(out)}",
        "",
        f"candidates per half: A (2024-07..2025-06)={int(half_counts['A'])}  B (2025-07..2026-09)={int(half_counts['B'])}",
        "",
        "candidates per week (full calendar incl. zero weeks):",
        f"  mean={wk_counts.mean():.2f}  median={wk_counts.median():.1f}  p90={wk_counts.quantile(0.9):.1f}  "
        f"max={int(wk_counts.max())}  share_weeks>=10={ (wk_counts >= 10).mean():.2%}",
        "",
        "UTC->ET conversion check (5 sample candidate rows):",
        *sample_lines,
        "",
        f"symbol-session count step 2 must backfill (one event session of minute bars each): {len(out)}",
        "",
        "Caveats: early closes not separately flagged (session rule uses the 09:30 open only, "
        "unaffected by an early close); EDGAR ticker spelling matched as-is against Databento "
        "raw_symbol with no class-share normalisation -- any mismatch drops under not_common/not_listed.",
    ]
    COMPLETENESS_MD.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", COMPLETENESS_MD)


if __name__ == "__main__":
    sys.exit(main())
