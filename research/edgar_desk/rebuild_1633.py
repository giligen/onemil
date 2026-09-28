"""Independent rebuild of PREREG_1633.md (cells 1,633-1,635): post-earnings drift on the
reaction (8-K item 2.02).

Written from the PREREG prose only -- the original cell_1633.py, cell_1633_events_pead.csv,
RESULT_1633.md and the v1 desk cell_1552.py were never opened while writing this file. This
is the "independent reimplementation" step required by CLAUDE.md before any research number
ships to the owner.

Mechanism (Bernard & Thomas 1989; Brandt/Kishore/Santa-Clara/Venkatachalam 2008): after an
earnings announcement price keeps drifting in the direction of the initial reaction. R0 (the
reaction) is the announcement-window return; PEAD says a buy-the-top-decile / short-the-
bottom-decile strategy on R0 should show positive forward drift over the following weeks.

Data:
  - research/edgar_desk/events_raw.csv        4.4M SEC filings (cik,symbol,form,filing_date,
    acceptance_datetime[UTC],items,primary_document,classes) -- streamed in chunks.
  - research/overnight_high/alpaca_daily_2019_2024H1.parquet  raw daily bars 2019-01-02 ..
    2024-06-27, 36,288 symbols (symbol,bar_date,open,high,low,close,volume).
  - research/overnight_high/panel_2024_2026.parquet           daily bars 2024-07-01 ..
    2026-09-04 (same OHLCV + precomputed extras we do not use); zero-OHLCV rows dropped.
  - research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv     a CURRENT-DAY snapshot of
    (symbol,tradable,shortable,easy_to_borrow,exchange) -- no date column, no SSR column.

Reaction-session mapping (own implementation, the exact refuter the 1,552 desk flagged: UTC
read as ET): acceptance_datetime -> America/New_York. Trading calendar = SPY's own bar_date
series (SPY trades every real session, so it is a clean, self-contained NYSE calendar with no
external dependency). For an acceptance at ET date D, ET time T:
  - let D_eff = first calendar day >= D  (rolls weekend/holiday filings forward to the next
    real session -- unstated in the prose for non-trading-day filings, but the only sensible
    generalisation of "the first full session after the announcement")
  - if D_eff == D (D is itself a trading day) and T >= 16:00 ET  -> reaction_session is the
    NEXT calendar day after D_eff   (after-close filing)
  - otherwise                                                    -> reaction_session = D_eff
    (this single rule covers both the pre-market-before-09:30 case and the intraday case,
    which the PREREG defines identically: "that session's close vs the prior close")
  R0 = close(reaction_session) / close(prior trading day) - 1.

Universe: close >= $3 and trailing-20-session average daily dollar volume >= $1M, both
measured at the prior session (the R0 baseline day -- known before the reaction, no lookahead).

Execution: MOO entry at the open of the session AFTER the reaction session; MOC exit at the
close of session reaction+10 (cells 1,633/1,635) or reaction+20 (cell 1,634). Both legs are
real auction order types (MOO/MOC) at prices the market actually prints -- not a level touch,
so this satisfies the CLAUDE.md obtainability check by construction. Cost: 5 bps/leg (10 bps
round trip). Shorts (1,635) additionally pay 3%/yr borrow over the actual hold in calendar
days, and are reported BOTH filtered to borrow_flags.csv "shortable"==True (partial coverage,
NOT point-in-time, no SSR field at all -> SSR is simply not applied, stated here explicitly
per CLAUDE.md's "read the report's own red flags") AND fully unfiltered, per the PREREG's own
"where absent report the short cell as unfiltered and say so".

Deciles: the 9 interior cutoffs (10th..90th percentile of R0) are fit on TRAIN-eligible events
ONLY and then applied mechanically to VAL -- never refit on VAL ("not allowed: choosing the
decile or the hold on VAL"). TEST (reaction_session >= 2024-07-01) is SEALED: rows in that
window are dropped immediately after split assignment and never appear in any later
computation, table, or file written by this script.
"""
import sys
import time as _time
import logging
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     stream=sys.stdout)
log = logging.getLogger("rebuild_1633")

# ---------------------------------------------------------------------------------------
# Config (frozen from PREREG_1633.md -- do not tune on VAL)
# ---------------------------------------------------------------------------------------
ROOT = Path("/home/ec2-user/onemil")
EVENTS_CSV = ROOT / "research/edgar_desk/events_raw.csv"
PRICES_A = ROOT / "research/overnight_high/alpaca_daily_2019_2024H1.parquet"
PRICES_B = ROOT / "research/overnight_high/panel_2024_2026.parquet"
BORROW_FLAGS = ROOT / "research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv"
OUT_DIR = ROOT / "research/edgar_desk"

PRICE_MIN = 3.0
DVOL20_MIN = 1_000_000.0
DVOL_WINDOW = 20
COST_BPS_PER_LEG = 5.0
ROUNDTRIP_COST_FRAC = 2 * COST_BPS_PER_LEG / 10_000.0
BORROW_ANNUAL = 0.03
HOLD_A = 10   # cells 1,633 / 1,635
HOLD_B = 20   # cell 1,634
WINNER_CAP = 0.30
AFTERCLOSE_MINUTES = 16 * 60  # 16:00 ET

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2024-06-30")
TEST_START = pd.Timestamp("2024-07-01")   # SEALED -- everything >= this is dropped, never scored

CSV_CHUNK = 300_000


# ---------------------------------------------------------------------------------------
# Stats helpers -- same method as research/hod_entry/cell_1445.py (day_clustered_t is used
# verbatim from that module's source per the task's pointer; reproduced here rather than
# imported to avoid pulling in that script's unrelated module-level state).
# ---------------------------------------------------------------------------------------
def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean.

    Identical implementation to research/hod_entry/cell_1445.py::day_clustered_t.
    `day` must be a pd.Series sharing y's original (pre-dropna) index.
    """
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap_frac):
    """Mean of fractional returns after capping each at +cap_frac (tail-robustness check)."""
    y = pd.Series(y).dropna()
    if not len(y):
        return np.nan
    return float(y.clip(upper=cap_frac).mean())


# ---------------------------------------------------------------------------------------
# Step 1: stream events_raw.csv, keep form == '8-K' with item 2.02 in the ';'-delimited list
# ---------------------------------------------------------------------------------------
def load_events():
    log.info("Streaming %s in %d-row chunks ...", EVENTS_CSV, CSV_CHUNK)
    usecols = ["cik", "symbol", "form", "filing_date", "acceptance_datetime", "items"]
    dtype = {"cik": str, "symbol": str, "form": str, "filing_date": str,
              "acceptance_datetime": str, "items": str}
    item_re = r"(?:^|;)2\.02(?:;|$)"
    keep = []
    n_scanned = 0
    n_8k_202 = 0
    n_8ka_202 = 0  # 8-K/A amendments with item 2.02 -- excluded by the exact form=='8-K' match
    forms_seen = {}
    t0 = _time.time()
    for i, chunk in enumerate(pd.read_csv(EVENTS_CSV, usecols=usecols, dtype=dtype,
                                           chunksize=CSV_CHUNK)):
        n_scanned += len(chunk)
        has_202 = chunk["items"].fillna("").str.contains(item_re, regex=True)
        is_8k = chunk["form"] == "8-K"
        keep.append(chunk.loc[is_8k & has_202])
        n_8k_202 += int((is_8k & has_202).sum())
        n_8ka_202 += int((~is_8k & has_202 & chunk["form"].fillna("").str.startswith("8-K")).sum())
        for f, c in chunk.loc[has_202, "form"].value_counts().items():
            forms_seen[f] = forms_seen.get(f, 0) + int(c)
        if (i + 1) % 5 == 0:
            log.info("  ... %d rows scanned, %d form=8-K item-2.02 matches so far (%.1fs)",
                      n_scanned, n_8k_202, _time.time() - t0)
    events = pd.concat(keep, ignore_index=True)
    log.info("Scanned %d rows total. form=='8-K' & item 2.02: %d rows.", n_scanned, n_8k_202)
    log.info("8-K/A (or other 8-K-prefixed) amendments with item 2.02 excluded by the exact "
              "form=='8-K' match: %d (this is the duplicate-8-K/amendment refuter from the "
              "PREREG's own independent-check list).", n_8ka_202)
    log.info("All form values seen among item-2.02 filings: %s", forms_seen)

    # Parse acceptance_datetime (UTC) -> ET. This is the exact refuter the 1,552 desk found:
    # the raw string was read as if it were already ET.
    acc_utc = pd.to_datetime(events["acceptance_datetime"], utc=True, errors="coerce")
    n_bad_ts = int(acc_utc.isna().sum())
    if n_bad_ts:
        log.warning("DROPPING %d events with unparseable acceptance_datetime", n_bad_ts)
    events = events.loc[acc_utc.notna()].copy()
    acc_utc = acc_utc.loc[acc_utc.notna()]
    acc_et = acc_utc.dt.tz_convert("America/New_York")
    events["acceptance_et"] = acc_et.dt.tz_localize(None)
    events["et_date"] = events["acceptance_et"].dt.normalize()
    events["et_minutes"] = events["acceptance_et"].dt.hour * 60 + events["acceptance_et"].dt.minute

    # Dedupe: same (symbol, cik) filing multiple 8-K/2.02s close together (amendments,
    # multi-item duplicates within a quarter) should count once. Group by (symbol, et_date)
    # is too coarse across genuinely different quarters, so first collapse EXACT duplicate
    # rows, then collapse same-symbol filings landing on the SAME et_date, keeping the
    # earliest acceptance (the first, price-moving release).
    n_before = len(events)
    events = events.drop_duplicates(subset=["cik", "acceptance_datetime", "items"])
    events = events.sort_values("acceptance_et").drop_duplicates(subset=["symbol", "et_date"],
                                                                    keep="first")
    log.info("Deduped %d -> %d events (exact dupes + same-symbol/same-ET-date collapsed to the "
              "earliest acceptance).", n_before, len(events))
    return events.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Step 2: prices -- combine both parquets, build the SPY trading calendar
# ---------------------------------------------------------------------------------------
def load_prices(event_symbols):
    log.info("Loading price parquets ...")
    cols = ["symbol", "bar_date", "open", "close", "volume"]
    a = pd.read_parquet(PRICES_A, columns=cols)
    b = pd.read_parquet(PRICES_B, columns=cols)
    n_b_raw = len(b)
    zero_mask = (b[["open", "close", "volume"]] == 0).any(axis=1)
    b = b.loc[~zero_mask].copy()
    log.info("panel_2024_2026: dropped %d/%d zero-OHLCV rows as instructed.",
              int(zero_mask.sum()), n_b_raw)

    a["bar_date"] = pd.to_datetime(a["bar_date"]).dt.normalize()
    b["bar_date"] = pd.to_datetime(b["bar_date"]).dt.normalize()

    spy_cal = pd.concat([a.loc[a.symbol == "SPY", "bar_date"],
                          b.loc[b.symbol == "SPY", "bar_date"]]).drop_duplicates().sort_values()
    calendar = spy_cal.values.astype("datetime64[D]")
    log.info("SPY trading calendar: %d sessions, %s .. %s", len(calendar),
              calendar.min(), calendar.max())
    gaps = np.diff(calendar).astype("timedelta64[D]").astype(int)
    big_gaps = np.where(gaps > 4)[0]
    if len(big_gaps):
        log.warning("%d calendar gaps > 4 days (holidays or data holes) -- largest %d days "
                     "around %s. Known: the alpaca_daily file ends 2024-06-27 and "
                     "panel_2024_2026 starts 2024-07-01, so 2024-06-28 (a real Friday trading "
                     "day) is missing from BOTH sources; any event needing that exact date "
                     "will simply skip it via the calendar.", len(big_gaps), int(gaps.max()),
                     calendar[big_gaps[np.argmax(gaps[big_gaps])]])

    keep_syms = set(event_symbols) | {"SPY"}
    prices = pd.concat([a.loc[a.symbol.isin(keep_syms)], b.loc[b.symbol.isin(keep_syms)]],
                        ignore_index=True)
    prices = prices.drop_duplicates(subset=["symbol", "bar_date"], keep="first")
    prices = prices.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    log.info("Price panel restricted to %d event symbols + SPY: %d rows.",
              len(keep_syms) - 1, len(prices))

    prices["dollar_vol"] = prices["close"] * prices["volume"]
    prices["dvol20"] = (prices.groupby("symbol", sort=False)["dollar_vol"]
                         .rolling(DVOL_WINDOW, min_periods=DVOL_WINDOW).mean()
                         .reset_index(level=0, drop=True))
    return prices, calendar


# ---------------------------------------------------------------------------------------
# Step 3: reaction-session mapping + session offsets (fully vectorized on the SPY calendar)
# ---------------------------------------------------------------------------------------
def map_sessions(events, calendar):
    et_date = events["et_date"].values.astype("datetime64[D]")
    idx0 = np.searchsorted(calendar, et_date, side="left")
    idx0 = np.clip(idx0, 0, len(calendar) - 1)
    d_eff = calendar[idx0]
    is_trading_day = d_eff == et_date
    after_close = is_trading_day & (events["et_minutes"].values >= AFTERCLOSE_MINUTES)
    reaction_idx = np.where(after_close, idx0 + 1, idx0)

    out_of_range = reaction_idx >= len(calendar)
    if out_of_range.any():
        log.warning("%d events map past the end of the calendar (dropping).", out_of_range.sum())
    reaction_idx = np.clip(reaction_idx, 0, len(calendar) - 1)

    events = events.copy()
    events["reaction_idx"] = reaction_idx
    events["reaction_session"] = calendar[reaction_idx]
    events["prior_idx"] = reaction_idx - 1
    events["entry_idx"] = reaction_idx + 1
    events["exit10_idx"] = reaction_idx + HOLD_A
    events["exit20_idx"] = reaction_idx + HOLD_B
    events["bucket"] = np.where(after_close, "afterclose",
                                 np.where(events["et_minutes"].values < 570, "premarket",
                                          "intraday"))

    valid = (events["prior_idx"] >= 0) & (events["exit20_idx"] < len(calendar)) & (~out_of_range)
    n_bad = int((~valid).sum())
    if n_bad:
        log.warning("Dropping %d events with session offsets outside the calendar range.", n_bad)
    events = events.loc[valid].copy()
    for col in ["prior_idx", "entry_idx", "exit10_idx", "exit20_idx"]:
        events[col.replace("_idx", "_session")] = calendar[events[col].values]
    return events


# ---------------------------------------------------------------------------------------
# Step 4: attach prices, universe filter, split assignment (TEST dropped immediately)
# ---------------------------------------------------------------------------------------
def attach_prices_and_filter(events, prices):
    px = prices.set_index(["symbol", "bar_date"])
    close_s = px["close"]
    open_s = px["open"]
    dvol_s = px["dvol20"]

    def lookup(series, symbols, dates, label):
        key = list(zip(symbols, dates))
        vals = series.reindex(key)
        return vals.to_numpy()

    sym = events["symbol"].to_numpy()
    events["prior_close"] = lookup(close_s, sym, events["prior_session"], "prior_close")
    events["dvol20"] = lookup(dvol_s, sym, events["prior_session"], "dvol20")
    events["reaction_close"] = lookup(close_s, sym, events["reaction_session"], "reaction_close")
    events["entry_open"] = lookup(open_s, sym, events["entry_session"], "entry_open")
    events["exit10_close"] = lookup(close_s, sym, events["exit10_session"], "exit10_close")
    events["exit20_close"] = lookup(close_s, sym, events["exit20_session"], "exit20_close")

    n0 = len(events)
    have_core = events[["prior_close", "reaction_close", "entry_open"]].notna().all(axis=1)
    log.info("Price coverage: %d/%d events have prior/reaction/entry prices (%.1f%%).",
              int(have_core.sum()), n0, 100 * have_core.mean())
    events["delisted_10"] = have_core & events["exit10_close"].isna()
    events["delisted_20"] = have_core & events["exit20_close"].isna()
    if have_core.sum():
        log.info("  of those, missing exit10 price (delisted/halted, kept at -100%% for the "
                  "long legs): %d ; missing exit20: %d",
                  int(events.loc[have_core, "delisted_10"].sum()),
                  int(events.loc[have_core, "delisted_20"].sum()))
    events = events.loc[have_core].copy()

    events["R0"] = events["reaction_close"] / events["prior_close"] - 1.0

    price_ok = events["prior_close"] >= PRICE_MIN
    dvol_ok = events["dvol20"] >= DVOL20_MIN
    log.info("Universe filter: price>=$3 keeps %d/%d; dvol20>=$1M keeps %d/%d (dvol20 NaN "
              "for %d events with <20 prior sessions of history).",
              int(price_ok.sum()), len(events), int(dvol_ok.fillna(False).sum()), len(events),
              int(events["dvol20"].isna().sum()))
    events = events.loc[price_ok & dvol_ok].copy()
    log.info("After universe filter: %d events.", len(events))

    events["split"] = np.select(
        [events["reaction_session"].astype("datetime64[ns]").between(TRAIN_START, TRAIN_END),
         events["reaction_session"].astype("datetime64[ns]").between(VAL_START, VAL_END)],
        ["TRAIN", "VAL"], default="TEST")
    n_test = int((events["split"] == "TEST").sum())
    log.info("Split counts before sealing TEST: %s", events["split"].value_counts().to_dict())
    events = events.loc[events["split"] != "TEST"].copy()
    assert not (events["split"] == "TEST").any()
    log.info("SEALED %d TEST-window events dropped (reaction_session >= %s) -- never scored.",
              n_test, TEST_START.date())
    return events.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Step 5: TRAIN-only decile cutoffs, applied mechanically to VAL
# ---------------------------------------------------------------------------------------
def assign_deciles(events):
    train_r0 = events.loc[events["split"] == "TRAIN", "R0"]
    qs = [i / 10 for i in range(1, 10)]
    cuts = train_r0.quantile(qs).to_numpy()
    log.info("TRAIN R0 decile cutoffs (10..90pct): %s", np.round(cuts, 4))
    events = events.copy()
    events["decile"] = np.searchsorted(cuts, events["R0"].to_numpy(), side="right") + 1
    return events, cuts


# ---------------------------------------------------------------------------------------
# Step 6: returns, costs, SPY-adjustment
# ---------------------------------------------------------------------------------------
def compute_returns(events, prices):
    spy = prices.loc[prices.symbol == "SPY", ["bar_date", "open", "close"]].set_index("bar_date")
    spy_open = spy["open"]
    spy_close = spy["close"]

    events = events.copy()
    events["gross_long_10"] = events["exit10_close"] / events["entry_open"] - 1.0
    events["gross_long_20"] = events["exit20_close"] / events["entry_open"] - 1.0
    events.loc[events["delisted_10"], "gross_long_10"] = -1.0
    events.loc[events["delisted_20"], "gross_long_20"] = -1.0
    events["net_long_10_bps"] = (events["gross_long_10"] - ROUNDTRIP_COST_FRAC) * 10_000
    events["net_long_20_bps"] = (events["gross_long_20"] - ROUNDTRIP_COST_FRAC) * 10_000

    hold_days = (pd.to_datetime(events["exit10_session"]) -
                 pd.to_datetime(events["entry_session"])).dt.days
    borrow_frac = BORROW_ANNUAL * (hold_days / 365.0)
    events["gross_short_10"] = 1.0 - events["exit10_close"] / events["entry_open"]
    events.loc[events["delisted_10"], "gross_short_10"] = np.nan  # excluded, see docstring
    events["net_short_10_bps"] = (events["gross_short_10"] - ROUNDTRIP_COST_FRAC -
                                   borrow_frac) * 10_000

    spy_entry = lambda s: spy_open.reindex(s).to_numpy()
    spy_exit10 = lambda s: spy_close.reindex(s).to_numpy()
    spy_exit20 = spy_exit10
    events["spy_gross_10"] = spy_exit10(events["exit10_session"]) / spy_entry(events["entry_session"]) - 1.0
    events["spy_gross_20"] = spy_exit20(events["exit20_session"]) / spy_entry(events["entry_session"]) - 1.0
    events["abn_net_long_10_bps"] = events["net_long_10_bps"] - events["spy_gross_10"] * 10_000
    events["abn_net_long_20_bps"] = events["net_long_20_bps"] - events["spy_gross_20"] * 10_000
    return events


def attach_borrow(events):
    bf = pd.read_csv(BORROW_FLAGS)
    bf = bf.drop_duplicates(subset="symbol", keep="first").set_index("symbol")
    log.info("borrow_flags.csv: %d symbols, %d shortable==True, %d shortable==False. NOTE: "
              "no date column (current-day snapshot, not point-in-time) and NO SSR column at "
              "all, so SSR exclusion is NOT applied anywhere in this rebuild.",
              len(bf), int((bf["shortable"] == True).sum()), int((bf["shortable"] == False).sum()))
    events = events.copy()
    matched = events["symbol"].isin(bf.index)
    events["shortable_known"] = matched
    events["shortable_flag"] = events["symbol"].map(bf["shortable"])
    log.info("Bottom-decile short population: %d/%d symbols found in borrow_flags.csv; of "
              "those %d are shortable==True.", int(matched.sum()), len(events),
              int((events["shortable_flag"] == True).sum()))
    return events


# ---------------------------------------------------------------------------------------
# Step 7: reporting
# ---------------------------------------------------------------------------------------
def decile_table(events, split):
    sub = events.loc[events["split"] == split]
    rows = []
    for d in range(1, 11):
        g = sub.loc[sub["decile"] == d]
        rows.append({
            "decile": d, "n": len(g), "mean_R0_bps": g["R0"].mean() * 10_000,
            "mean_gross10_bps": g["gross_long_10"].mean() * 10_000,
            "mean_gross20_bps": g["gross_long_20"].mean() * 10_000,
        })
    return pd.DataFrame(rows)


def cell_stats(y_bps_frac_series, day_series, split_label, cell_label, hold_days_span):
    """y_bps_frac_series: net bps already computed. Returns a dict of report fields."""
    y = pd.Series(y_bps_frac_series)
    n = int(y.notna().sum())
    if n == 0:
        return {"split": split_label, "cell": cell_label, "n": 0}
    t = day_clustered_t(y, day_series)
    weeks = hold_days_span / 7.0
    return {
        "split": split_label, "cell": cell_label, "n": n,
        "events_per_week": n / weeks if weeks > 0 else np.nan,
        "mean_net_bps": float(y.mean()),
        "day_clustered_t": t,
        "ex_top5pct_bps": ex_top5_mean(y),
        "winner_capped_30pct_bps": winner_capped_mean(y / 10_000.0, WINNER_CAP) * 10_000,
        "median_bps": float(y.median()),
        "pct_positive": float((y > 0).mean()),
    }


def main():
    t_start = _time.time()
    events_raw = load_events()
    prices, calendar = load_prices(events_raw["symbol"].unique())
    events = map_sessions(events_raw, calendar)
    events = attach_prices_and_filter(events, prices)
    events, train_cuts = assign_deciles(events)
    events = compute_returns(events, prices)
    events = attach_borrow(events)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["cik", "symbol", "form", "filing_date", "acceptance_et", "bucket",
                "reaction_session", "prior_session", "entry_session", "exit10_session",
                "exit20_session", "R0", "split", "decile", "prior_close", "dvol20",
                "entry_open", "exit10_close", "exit20_close", "delisted_10", "delisted_20",
                "gross_long_10", "gross_long_20", "net_long_10_bps", "net_long_20_bps",
                "gross_short_10", "net_short_10_bps", "shortable_known", "shortable_flag",
                "spy_gross_10", "spy_gross_20", "abn_net_long_10_bps", "abn_net_long_20_bps"]
    events_out_path = OUT_DIR / "rebuild_1633_events_pead.csv"
    events[out_cols].to_csv(events_out_path, index=False)
    log.info("Wrote %s (%d rows).", events_out_path, len(events))

    train_dt = decile_table(events, "TRAIN")
    val_dt = decile_table(events, "VAL")
    log.info("TRAIN decile table:\n%s", train_dt.to_string(index=False))
    log.info("VAL decile table:\n%s", val_dt.to_string(index=False))

    top_cut = train_cuts[-1]   # 90th pct
    bot_cut = train_cuts[0]    # 10th pct

    results = []
    for split, s_start, s_end in [("TRAIN", TRAIN_START, TRAIN_END), ("VAL", VAL_START, VAL_END)]:
        span_days = (s_end - s_start).days
        sub = events.loc[events["split"] == split]
        top = sub.loc[sub["R0"] >= top_cut]
        bot_all = sub.loc[sub["R0"] <= bot_cut]
        bot_filt = bot_all.loc[bot_all["shortable_flag"] == True]

        r = cell_stats(top["net_long_10_bps"], top["entry_session"], split, "1633_long_top_h10",
                        span_days)
        results.append(r)
        r = cell_stats(top["net_long_20_bps"], top["entry_session"], split, "1634_long_top_h20",
                        span_days)
        results.append(r)
        r = cell_stats(bot_filt["net_short_10_bps"], bot_filt["entry_session"], split,
                        "1635_short_bot_h10_FILTERED_shortable", span_days)
        results.append(r)
        r = cell_stats(bot_all["net_short_10_bps"], bot_all["entry_session"], split,
                        "1635_short_bot_h10_UNFILTERED", span_days)
        results.append(r)

    results_df = pd.DataFrame(results)
    log.info("Cell results:\n%s", results_df.to_string(index=False))

    # Per-year breakdown for cell 1633 (both splits), a PREREG-requested diagnostic.
    events["year"] = pd.to_datetime(events["reaction_session"]).dt.year
    peryear = (events.loc[events["R0"] >= top_cut]
               .groupby(["split", "year"])["net_long_10_bps"]
               .agg(["count", "mean"]).reset_index())
    log.info("Cell 1633 per-year:\n%s", peryear.to_string(index=False))

    write_report(events, train_dt, val_dt, results_df, train_cuts, peryear, t_start)
    results_df.to_csv(OUT_DIR / "rebuild_1633_cell_results.csv", index=False)
    train_dt.to_csv(OUT_DIR / "rebuild_1633_decile_train.csv", index=False)
    val_dt.to_csv(OUT_DIR / "rebuild_1633_decile_val.csv", index=False)
    log.info("DONE in %.1fs", _time.time() - t_start)


def write_report(events, train_dt, val_dt, results_df, train_cuts, peryear, t_start):
    n_train = int((events["split"] == "TRAIN").sum())
    n_val = int((events["split"] == "VAL").sum())
    val_1633 = results_df.loc[(results_df.split == "VAL") &
                               (results_df.cell == "1633_long_top_h10")].iloc[0]
    train_1633 = results_df.loc[(results_df.split == "TRAIN") &
                                 (results_df.cell == "1633_long_top_h10")].iloc[0]

    lines = []
    lines.append("# REBUILD_1633.md -- independent rebuild of PREREG_1633.md (cells 1,633-1,635)")
    lines.append("")
    lines.append("Built from the PREREG prose only. cell_1633.py, cell_1633_events_pead.csv, "
                  "RESULT_1633.md and cell_1552.py were never opened while writing "
                  "rebuild_1633.py -- this is the independent-reimplementation leg of the "
                  "CLAUDE.md pre-ship check, not a comparison against the original (a "
                  "trade-by-trade Jaccard/bps diff against cell_1633_events_pead.csv must be "
                  "done separately by whoever can see both files).")
    lines.append("")
    lines.append("## Event population")
    lines.append(f"- Eligible (universe-filtered) events, TRAIN+VAL: {len(events)} "
                  f"(TRAIN {n_train}, VAL {n_val}). TEST (reaction_session >= "
                  f"{TEST_START.date()}) was dropped immediately after split assignment and "
                  "never scored, per SEALED.")
    lines.append(f"- TRAIN R0 decile cutoffs (10th..90th pct): "
                  f"{[round(float(c), 4) for c in train_cuts]}")
    lines.append("")
    lines.append("## TRAIN decile table (R0 and forward gross return by TRAIN-fit decile)")
    lines.append(train_dt.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")
    lines.append("## VAL decile table (same TRAIN cutoffs applied out-of-sample)")
    lines.append(val_dt.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")
    lines.append("## Cells 1,633 / 1,634 / 1,635 -- per split")
    lines.append(results_df.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Cell 1,633 per-year (top decile, 10-session hold, net bps)")
    lines.append(peryear.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")
    lines.append("## Pass-bar check (frozen, VAL, cell 1,633)")
    lines.append(f"- Mean net >= +50 bps/event: VAL = {val_1633.mean_net_bps:.1f} bps -> "
                  f"{'PASS' if val_1633.mean_net_bps >= 50 else 'FAIL'}")
    lines.append(f"- day-clustered t >= 2.5: VAL t = {val_1633.day_clustered_t:.2f} -> "
                  f"{'PASS' if (val_1633.day_clustered_t == val_1633.day_clustered_t and val_1633.day_clustered_t >= 2.5) else 'FAIL'}")
    lines.append(f"- ex-top-5% > 0: VAL = {val_1633.ex_top5pct_bps:.1f} bps -> "
                  f"{'PASS' if val_1633.ex_top5pct_bps > 0 else 'FAIL'}")
    lines.append(f"- >= 5 events/week in season: VAL = {val_1633.events_per_week:.2f}/wk -> "
                  f"{'PASS' if val_1633.events_per_week >= 5 else 'FAIL'}")
    lines.append(f"- TRAIN same sign, t >= 1: TRAIN mean = {train_1633.mean_net_bps:.1f} bps, "
                  f"t = {train_1633.day_clustered_t:.2f} -> "
                  f"{'PASS' if (train_1633.mean_net_bps > 0 and train_1633.day_clustered_t >= 1) else 'FAIL'}")
    mono = val_dt["mean_gross10_bps"].is_monotonic_increasing
    lines.append(f"- decile table monotone on VAL (10-session gross): {mono}")
    lines.append("")
    lines.append("## Caveats (read as an adversary, per CLAUDE.md)")
    lines.append("- **borrow_flags.csv is not point-in-time.** It is a single current-day "
                  "snapshot (no date column) joined onto TRAIN/VAL events by symbol only; a "
                  "name's shortability in 2019-2023 may differ from today's snapshot. Cell "
                  "1,635 is reported both FILTERED (shortable==True in the snapshot) and "
                  "UNFILTERED (all bottom-decile events), per the task's own instruction for "
                  "when the borrow data is imperfect/absent.")
    lines.append("- **SSR is not represented at all** in borrow_flags.csv (columns: symbol, "
                  "tradable, shortable, easy_to_borrow, exchange). No SSR exclusion is applied "
                  "anywhere -- this is a real gap against the PREREG's 'shortable/SSR excluded', "
                  "not a filter that was silently skipped.")
    lines.append("- **Delistings inside the hold are kept at -100%** for the long legs "
                  "(1,633/1,634) when a symbol has core prices (prior/reaction/entry) but no "
                  "exit price; the short leg (1,635) EXCLUDES those events instead of assuming "
                  "a clean +100% cover, since a delisting's actual short P&L depends on the "
                  "bankruptcy/wind-down process and is not reliably inferable from daily bars.")
    lines.append("- **Known 1-day data hole: 2024-06-28.** alpaca_daily_2019_2024H1.parquet "
                  "ends 2024-06-27; panel_2024_2026.parquet starts 2024-07-01. 2024-06-28 was a "
                  "real Friday trading session missing from both files, so it is simply absent "
                  "from the SPY calendar built here; any event landing exactly on it is "
                  "invisible to this rebuild (logged, not silently large -- see the run log).")
    lines.append("- **No market-cap split.** Cell 1,636's small-cap (<=$1B) vs larger cut needs "
                  "shares outstanding, which is not in any of the datasets handed to this task; "
                  "out of scope here and not attempted (no market-cap number is reported).")
    lines.append("- **Obtainability:** both legs are MOO/MOC auction orders at prices the "
                  "market actually printed, not a level touch, so this satisfies the CLAUDE.md "
                  "obtainability check by construction.")
    lines.append("- **Reaction-session mapping for non-trading-day (weekend/holiday) filings** "
                  "is not stated in the PREREG prose; this rebuild rolls them forward to the "
                  "next real session under the same rule as a pre-market filing (see the "
                  "module docstring in rebuild_1633.py). This is a genuine judgment call an "
                  "independent implementation could reasonably make differently.")
    lines.append(f"- Dedup collapsed same-symbol/same-ET-date 8-K/2.02 filings to the earliest "
                  "acceptance (handles amendments/multi-part same-day filings); exact form=='8-K' "
                  "matching (vs '8-K/A') independently excludes amendment forms.")
    lines.append("")
    lines.append(f"Runtime: {_time.time() - t_start:.1f}s.")
    (OUT_DIR / "REBUILD_1633.md").write_text("\n".join(lines))
    log.info("Wrote %s", OUT_DIR / "REBUILD_1633.md")


if __name__ == "__main__":
    main()
