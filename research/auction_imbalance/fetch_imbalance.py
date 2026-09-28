#!/usr/bin/env python3
"""FETCH stage -- research/auction_imbalance/PREREG_1630.md (FROZEN 2026-09-28 18:35 UTC),
Amendment 1 (2026-09-28, main session): sample window narrowed to the price panel's own coverage,
spend cap raised to $90, and the reference-quote pull extended into a through-touch check.
Cells 1,630-1,632 (closing-auction imbalance). Idea 2 of research/IDEAS_20260928.md.

Builds the three yearly point-in-time liquid lists (300 names each, by panel dvol20 as of the
first session of 2024 / 2025 / 2026), determines each name's imbalance-publishing venue
(XNAS.ITCH vs XNYS.PILLAR) by a bookend probe, then pulls for every session 2024-07-01..2026-09-04
(Amendment 1: the price panel's own coverage -- close/next_open below are NOT re-fetched, so a
session outside the panel's range has no exit price anyway; sessions before 2024-07-01 pulled
under the original spec stay cached on disk UNUSED -- excluded from both the `run` day loop and
`consolidate`, never re-spent, never re-pulled):
  * schema `imbalance` on the assigned venue, 15:45:00-16:00:00 ET (the closing-auction imbalance
    publication window -- 15:50 NYSE / 15:55 Nasdaq -- and its updates through the close).
  * schema `bbo-1m` on the assigned venue, ~15:48:50-15:58:10 ET (Amendment 1: widened from the
    original two 20s reference-mid windows to cover every ET minute mark 15:49..15:58 in one pull
    -- 15:50/15:55 are the pre-publication reference mids the PREREG needs, 15:51-15:52/15:56-15:57
    are the +1min/+2min through-touch checks Amendment 1 adds, the rest is buffer; bbo-1m's one
    snapshot/symbol/minute lands exactly on the minute boundary -- verified live 9/28 -- so a plain
    continuous minute range replaces the old two narrow windows with no precision loss and
    negligible extra cost, ~$0.00003/symbol-window).
This is the FETCH stage only: no signal, no decile, no trade simulation. The official close and
next-open prices are already in the panel (columns `close`, `next_open`) and are NOT re-fetched.

Cost discipline: metadata.get_cost is called before EVERY timeseries.get_range; every such call's
projected cost is logged and added to a running total persisted in dbn_cache/spend.json; the
script hard-stops (exit 2) the instant that running total would exceed SPEND_CAP_USD, before
issuing the get_range. Every pull is cached immediately as its own shard parquet file under
dbn_cache/shards/{imbalance,quotes}/{day}_{venue}.parquet (skipped on re-run if present) --
resumable by construction; `--mode run` can be killed and re-launched at no data loss and no
re-spend on completed shards. `--mode consolidate` (also run automatically at the end of `run`)
concatenates the shards dated within [SAMPLE_START, SAMPLE_END] into the two final files:
dbn_cache/imbalance.parquet, dbn_cache/quotes.parquet (one file, both the reference mids and the
touch checks, distinguished by the `minute` ET-clock column).

Modes (run in order the first time; each is idempotent / cheap to re-run):
    plan          pure-python: build the 3 liquid lists from the panel, no API calls.
    probe-schema  tiny real pull (1 session, 5 symbols) on both schemas; prints/logs the actual
                  columns Databento returns, so the canonical-field mapping below is verified
                  before any money is spent on the full run.
    probe-venue   bookend (first + last session) batched probe of the `imbalance` schema for the
                  WHOLE deduped symbol universe on both venues in one call per (day, venue); a
                  symbol's venue = whichever side returned more rows; neither -> dropped (WARNING).
    estimate      samples 5 representative sessions, projects the total cost for the full run.
    run           the full resumable fetch. --start/--end override the session range (smoke-test
                  a few days before committing to the full sample).
    consolidate   glob the shards into the two final parquet files and print counts.

Usage:
    python3 research/auction_imbalance/fetch_imbalance.py plan
    python3 research/auction_imbalance/fetch_imbalance.py probe-schema
    python3 research/auction_imbalance/fetch_imbalance.py probe-venue
    python3 research/auction_imbalance/fetch_imbalance.py estimate
    python3 research/auction_imbalance/fetch_imbalance.py run --start 2024-07-01 --end 2024-07-03
    python3 research/auction_imbalance/fetch_imbalance.py run
    python3 research/auction_imbalance/fetch_imbalance.py consolidate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

REPO = '/home/ec2-user/onemil'
sys.path.insert(0, REPO)
os.chdir(REPO)
load_dotenv(os.path.join(REPO, '.env'))

import databento as db  # noqa: E402

ET = ZoneInfo('America/New_York')
HERE = os.path.join(REPO, 'research/auction_imbalance')
CACHE = os.path.join(HERE, 'dbn_cache')
SHARDS_IMB = os.path.join(CACHE, 'shards', 'imbalance')
SHARDS_QT = os.path.join(CACHE, 'shards', 'quotes')
PANEL = os.path.join(REPO, 'research/overnight_high/panel_2024_2026.parquet')
SPEND_JSON = os.path.join(CACHE, 'spend.json')
LOG_FILE = os.path.join(CACHE, 'fetch_log.txt')
IMBALANCE_PARQUET = os.path.join(CACHE, 'imbalance.parquet')
QUOTES_PARQUET = os.path.join(CACHE, 'quotes.parquet')  # Amendment 1: renamed from ref_quotes.parquet
                                                          # -- one file, both ref mids and touch checks
LISTS_JSON = os.path.join(CACHE, 'liquid_lists.json')
VENUE_CSV = os.path.join(CACHE, 'venue_map.csv')
ESTIMATE_JSON = os.path.join(CACHE, 'cost_estimate.json')
SENTINEL_DONE = os.path.join(CACHE, 'FETCH_DONE_1630')

SAMPLE_START = '2024-07-01'  # Amendment 1 (2026-09-28): the price panel's own coverage start --
                              # sessions before this stay cached on disk from the original spec,
                              # UNUSED (excluded from both the `run` day loop and `consolidate`)
SAMPLE_END = '2026-09-04'    # Amendment 1: the price panel's own coverage end (close/next_open
                              # for the exit are sourced from the panel, never re-fetched)
SPEND_CAP_USD = 100.0        # Amendment 2: raised from $90 (main session, 2026-09-28 19:50 UTC; the $90 run halts ~6 weeks short)
N_LIQUID = 300
VENUES = ('XNAS.ITCH', 'XNYS.PILLAR')
# bbo-1m (one top-of-book snapshot/symbol/minute), not mbp-1 (every book update): verified live
# 9/28 on 5 mega-cap names, same 5m20s window -- mbp-1 $0.0417 / 128,684 rows vs bbo-1m $0.00003 /
# 30 rows, and bbo-1m's snapshot lands exactly on the minute mark (19:50:00.000 UTC = 15:50:00 ET),
# square in the middle of both 20s reference windows. The task lists bbo-1m as the first choice
# and mbp-1 only "for the two 10-second reference windows" -- at >1000x the cost for no precision
# gain here (a minute-boundary snapshot IS the pre-publication quote), bbo-1m is the one used.
QUOTES_SCHEMA = 'bbo-1m'
# Amendment 1 (2026-09-28): one continuous ET minute range 15:49..15:58 replaces the original two
# narrow 20s windows -- 15:50/15:55 remain the pre-publication reference mids (NYSE/Nasdaq), and
# 15:51-15:52/15:56-15:57 are the new +1min/+2min through-touch checks (fill assumed only if the
# mid is touched within 60s per PREREG_1630.md); the rest (15:49, 15:53, 15:54, 15:58) is buffer.
# +-10s safety margin at each end of the pulled range, same convention as the original windows, in
# case get_range's start/end are exclusive on either edge -- a bbo-1m snapshot lands exactly on the
# minute boundary so the margin costs nothing extra (no additional minute marks fall inside it).
QUOTE_WINDOW_ET = (15, 48, 50, 15, 58, 10)

for d in (CACHE, SHARDS_IMB, SHARDS_QT):
    os.makedirs(d, exist_ok=True)


def log(msg: str) -> None:
    line = f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------------------
# spend ledger -- every get_cost call is logged; only get_range spends real $.
# ---------------------------------------------------------------------------

def load_spend() -> dict:
    if os.path.exists(SPEND_JSON):
        with open(SPEND_JSON) as f:
            spend = json.load(f)
        if spend.get('cap_usd') != SPEND_CAP_USD:
            log(f'NOTE spend ledger cap_usd was ${spend.get("cap_usd")}, syncing to the current '
                f'SPEND_CAP_USD ${SPEND_CAP_USD:.2f} (Amendment 1) -- the enforced cap is always '
                f'the SPEND_CAP_USD constant, this field is informational only')
            spend['cap_usd'] = SPEND_CAP_USD
        return spend
    return {'total_usd': 0.0, 'cap_usd': SPEND_CAP_USD, 'purchases': []}


def save_spend(spend: dict) -> None:
    with open(SPEND_JSON, 'w') as f:
        json.dump(spend, f, indent=2)


def record_purchase(spend: dict, desc: str, cost: float) -> float:
    spend['total_usd'] = round(spend['total_usd'] + cost, 6)
    spend['purchases'].append({
        'ts': datetime.now(timezone.utc).isoformat(), 'desc': desc,
        'cost_usd': round(cost, 6), 'running_total_usd': spend['total_usd'],
    })
    save_spend(spend)
    return spend['total_usd']


def get_cost_safe(client, retries=6, **kw) -> float:
    for attempt in range(retries):
        try:
            return float(client.metadata.get_cost(**kw))
        except Exception as exc:
            log(f'WARNING get_cost attempt {attempt + 1}/{retries} failed '
                f'({kw.get("dataset")}/{kw.get("schema")} {kw.get("start")}..{kw.get("end")}): '
                f'{str(exc)[:150]}')
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'get_cost failed permanently for {kw.get("dataset")}/{kw.get("schema")} '
                        f'{kw.get("start")}')


def get_range_safe(client, retries=5, **kw) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            data = client.timeseries.get_range(**kw)
            df = data.to_df()
            # databento's to_df() carries the event time as the DatetimeIndex (named ts_event),
            # not a column -- reset once here so every caller can treat 'ts_event' uniformly as
            # a column (verified live: ohlcv-1d/imbalance/mbp-1 all index this way).
            if df.index.name is not None:
                df = df.reset_index()
            return df
        except Exception as exc:
            log(f'WARNING get_range attempt {attempt + 1}/{retries} failed '
                f'({kw.get("dataset")}/{kw.get("schema")} {kw.get("start")}..{kw.get("end")}): '
                f'{str(exc)[:150]}')
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'get_range failed permanently for {kw.get("dataset")}/{kw.get("schema")} '
                        f'{kw.get("start")}')


def priced_pull(client, spend: dict, desc: str, **kw):
    """get_cost -> cap check -> get_range -> log, in that order, EVERY time. Returns None (and
    leaves the caller's loop to stop) if pulling would breach SPEND_CAP_USD -- the pull is never
    issued in that case, so the ledger's total_usd always reflects money actually put at risk."""
    cost = get_cost_safe(client, **kw)
    projected = spend['total_usd'] + cost
    if projected > SPEND_CAP_USD:
        log(f'ERROR spend cap ${SPEND_CAP_USD:.2f} would be exceeded by "{desc}" '
            f'(est ${cost:.4f}, running total ${spend["total_usd"]:.4f} -> ${projected:.4f}) -- STOP, no pull issued')
        return None
    df = get_range_safe(client, **kw)
    total = record_purchase(spend, desc, cost)
    log(f'  {desc}: est ${cost:.4f}, {len(df):,} rows, running total ${total:.4f}')
    return df


# ---------------------------------------------------------------------------
# time / session helpers
# ---------------------------------------------------------------------------

def et_window_utc(day: str, h0, m0, s0, h1, m1, s1):
    d = datetime.strptime(day, '%Y-%m-%d')
    a = d.replace(hour=h0, minute=m0, second=s0, tzinfo=ET).astimezone(timezone.utc)
    b = d.replace(hour=h1, minute=m1, second=s1, tzinfo=ET).astimezone(timezone.utc)
    return a.strftime('%Y-%m-%dT%H:%M:%S'), b.strftime('%Y-%m-%dT%H:%M:%S')


CALENDAR_CSV = os.path.join(CACHE, 'trading_calendar.csv')


def ensure_trading_calendar(client, spend) -> list:
    """The authoritative NYSE/Nasdaq session list for the fetch window, sourced from Databento
    itself (AAPL ohlcv-1d on XNAS.ITCH -- Nasdaq and NYSE share one US equity holiday calendar,
    so any liquid name's daily-bar dates are the full trading calendar) rather than the panel
    (which starts 2024-07-01, see build_liquid_lists) or a hand-maintained holiday table (a
    silent transcription error there would corrupt every session downstream). Cached once."""
    import re as _re

    if os.path.exists(CALENDAR_CSV):
        return sorted(pd.read_csv(CALENDAR_CSV, dtype=str).bar_date.unique())

    def plus1(d):
        return datetime.fromordinal(datetime.strptime(d, '%Y-%m-%d').date().toordinal() + 1).strftime('%Y-%m-%d')

    # ohlcv-1d is cheap (one row/session) but its own availability can lag "today" by a session or
    # two (finalization delay) even when the intraday schemas we actually need are already caught
    # up -- observed live 2026-09-28: ohlcv-1d capped at 2026-09-26T00:00Z (i.e. through the
    # 2026-09-25 session) while `imbalance`/`mbp-1` were already available through 2026-09-27. Ask
    # once for the FULL requested range; if Databento 422s with the dataset's real upper bound,
    # parse it out of the message and retry clamped to that -- auto-adapts on any future re-run
    # rather than hardcoding today's specific cutoff.
    end_excl = plus1(SAMPLE_END)
    cost = None
    try:
        cost = client.metadata.get_cost(dataset='XNAS.ITCH', schema='ohlcv-1d', symbols=['AAPL'],
                                         stype_in='raw_symbol', start=SAMPLE_START, end=end_excl)
    except Exception as exc:
        m = _re.search(r'available between [^ ]+ and (\d{4}-\d{2}-\d{2})', str(exc))
        if not m:
            raise
        clamped_end = m.group(1)
        log(f'NOTE ohlcv-1d only finalized through {clamped_end} as of now (today={datetime.now(timezone.utc).date()}); '
            f'clamping the calendar pull there and back-filling {clamped_end}..{SAMPLE_END} from the '
            f'`imbalance` schema (already available later) below')
        end_excl = clamped_end
    df = priced_pull(client, spend, f'trading-calendar AAPL ohlcv-1d {SAMPLE_START}..{end_excl}',
                      dataset='XNAS.ITCH', schema='ohlcv-1d', symbols=['AAPL'],
                      stype_in='raw_symbol', start=SAMPLE_START, end=end_excl)
    if df is None or not len(df):
        raise RuntimeError('trading-calendar pull failed or returned 0 rows -- cannot derive sessions')
    ts_col = 'ts_event' if 'ts_event' in df.columns else 'ts_recv'
    # daily-schema gotcha (documented, see the team's databento-pit-universe reference): ts_event
    # for ohlcv-1d is 00:00 UTC of the TRADE date itself -- converting to ET shifts every date back
    # one day (19:00-20:00 ET the prior evening). Take the UTC date as-is; do NOT tz_convert(ET).
    days = sorted(pd.to_datetime(df[ts_col], utc=True).dt.strftime('%Y-%m-%d').unique())

    # tail gap (if any) between the ohlcv-1d cutoff and SAMPLE_END: check each weekday directly
    # against the `imbalance` schema (AAPL, XNAS.ITCH, the actual 15:45-16:00 window) -- >=1 row
    # means a real session (an auction imbalance was published), 0 rows means holiday/no session.
    gap_start = days[-1] if days else SAMPLE_START
    candidate = plus1(gap_start)
    while candidate <= SAMPLE_END:
        dow = datetime.strptime(candidate, '%Y-%m-%d').weekday()
        if dow < 5:  # Mon-Fri only; weekends never trade
            a, b = et_window_utc(candidate, 15, 45, 0, 16, 0, 0)
            chk = priced_pull(client, spend, f'trading-calendar tail-check AAPL imbalance {candidate}',
                               dataset='XNAS.ITCH', schema='imbalance', symbols=['AAPL'],
                               stype_in='raw_symbol', start=a, end=b)
            if chk is None:
                break  # spend cap -- stop extending, keep what we have
            if len(chk):
                days.append(candidate)
                log(f'NOTE tail-check: {candidate} confirmed a session (imbalance published)')
            else:
                log(f'NOTE tail-check: {candidate} has no AAPL imbalance record -- treated as a non-session (holiday)')
        candidate = plus1(candidate)

    days = sorted(set(days))
    pd.DataFrame({'bar_date': days}).to_csv(CALENDAR_CSV, index=False)
    log(f'trading calendar: {len(days)} sessions {days[0]}..{days[-1]} -> {CALENDAR_CSV}')
    return days


def sessions(lo: str, hi: str) -> list:
    if not os.path.exists(CALENDAR_CSV):
        raise RuntimeError(f'{CALENDAR_CSV} missing -- run `calendar` mode first (needs a client, '
                            f'spends a few cents) before plan/probe-venue/estimate/run can list sessions')
    s = sorted(pd.read_csv(CALENDAR_CSV, dtype=str).bar_date.unique())
    return [x for x in s if lo <= x <= hi]


# ---------------------------------------------------------------------------
# mode: plan -- the 3 point-in-time liquid lists, no API calls
# ---------------------------------------------------------------------------

def build_liquid_lists() -> dict:
    df = pd.read_parquet(PANEL, columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close',
                                          'volume', 'dvol20'])
    df['symbol'] = df['symbol'].astype(str)
    df['bar_date'] = df['bar_date'].astype(str)
    lo, hi = df.bar_date.min(), df.bar_date.max()
    log(f'panel covers {lo}..{hi}, {len(df):,} rows, {df.symbol.nunique():,} unique symbols')
    out = {}
    for year in (2024, 2025, 2026):
        yr_days = sorted(df.loc[df.bar_date.str.startswith(str(year)), 'bar_date'].unique())
        if not yr_days:
            log(f'WARNING no sessions for {year} in the panel -- year skipped')
            continue
        # "first session of the year" normally means yr_days[0], but dvol20 is a 20-session
        # TRAILING average: at the panel's own first day (2024-07-01 -- the panel starts there,
        # not 2024-01-01, see the calendar/panel-coverage caveat in FETCH_1630.md) there is no
        # trailing history yet and dvol20 is NaN for every row. Root cause, not a special case:
        # walk forward to the first day THIS year with real dvol20 coverage on a majority of rows.
        first_day = None
        for cand in yr_days:
            cov = df.loc[df.bar_date == cand, 'dvol20'].notna().mean()
            if cov >= 0.5:
                first_day = cand
                break
        if first_day is None:
            log(f'WARNING {year}: no session has >=50% dvol20 coverage (checked {len(yr_days)} days) -- year skipped')
            continue
        if first_day != yr_days[0]:
            log(f'NOTE {year}: calendar-year first session is {yr_days[0]} but dvol20 has no trailing '
                f'history that early (panel starts {df.bar_date.min()}); using {first_day} (first '
                f'session with >=50% dvol20 coverage) as the point-in-time "as of" date instead')
        day_df = df[df.bar_date == first_day].copy()
        before = len(day_df)
        day_df = day_df[(day_df.open > 0) & (day_df.high > 0) & (day_df.low > 0)
                         & (day_df.close > 0) & (day_df.volume > 0)]
        dropped_zero = before - len(day_df)
        day_df = day_df.dropna(subset=['dvol20'])
        day_df = day_df.sort_values('dvol20', ascending=False)
        top = day_df.head(N_LIQUID)[['symbol', 'dvol20']].reset_index(drop=True)
        if len(top) < N_LIQUID:
            log(f'WARNING {year}: only {len(top)} names available with valid dvol20 on {first_day} '
                f'(< {N_LIQUID} requested)')
        log(f'{year}: first session {first_day}, {before} rows -> {dropped_zero} zero-OHLCV dropped '
            f'-> {len(top)} liquid names kept (dvol20 range ${top.dvol20.min():,.0f}..'
            f'${top.dvol20.max():,.0f})')
        out[str(year)] = {
            'as_of': first_day,
            'n': len(top),
            'symbols': top['symbol'].tolist(),
            'dvol20': {r.symbol: float(r.dvol20) for r in top.itertuples()},
        }
    union = sorted(set().union(*[set(v['symbols']) for v in out.values()]))
    log(f'union across the 3 yearly lists: {len(union)} unique symbols')
    out['_union'] = union
    out['_panel_range'] = {'min': lo, 'max': hi}
    with open(LISTS_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {LISTS_JSON}')
    return out


def load_lists() -> dict:
    if not os.path.exists(LISTS_JSON):
        return build_liquid_lists()
    with open(LISTS_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# mode: probe-schema -- verify actual Databento column names before spending
# ---------------------------------------------------------------------------

def cmd_probe_schema(client, spend):
    probe_syms = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA']
    day = sessions(SAMPLE_START, SAMPLE_END)[-1]  # last REAL session, not the raw (maybe weekend) bound
    log(f'=== probe-schema: {probe_syms} on {day} ===')
    for schema, window in (
        ('imbalance', (15, 45, 0, 16, 0, 0)),
        (QUOTES_SCHEMA, QUOTE_WINDOW_ET),
    ):
        start, end = et_window_utc(day, *window)
        for venue in VENUES:
            desc = f'probe-schema {schema} {venue} {day}'
            df = priced_pull(client, spend, desc, dataset=venue, schema=schema, symbols=probe_syms,
                              stype_in='raw_symbol', start=start, end=end)
            if df is None:
                log('ERROR spend cap hit during probe-schema -- aborting')
                return 2
            if len(df):
                log(f'{venue}/{schema} columns: {list(df.columns)}')
                log(f'{venue}/{schema} sample row: {df.iloc[0].to_dict()}')
            else:
                log(f'{venue}/{schema}: 0 rows for this probe (symbols may list on the other venue)')
    log('probe-schema done -- inspect dbn_cache/fetch_log.txt for column names, then adjust '
        'CANDIDATE_COLS below if needed before `run`.')
    return 0


# ---------------------------------------------------------------------------
# mode: probe-venue -- bookend batched probe -> venue_map.csv
# ---------------------------------------------------------------------------

def cmd_probe_venue(client, spend):
    """Venue = primary listing exchange, read from the already-purchased Databento EQUS.SUMMARY
    definition feed (research/scripts/pit_listings.py, 2024-07..2026-09, MIC per symbol per
    month) -- NOT inferred by comparing imbalance-schema row counts across venues. That row-count
    heuristic was tried first and is WRONG: XNAS.ITCH returns a uniform ~330 imbalance rows for
    EVERY requested symbol regardless of primary listing (e.g. JNJ/WMT/PG/BAC/BRK.B, all really
    XNYS, each showed exactly 330 rows on XNAS.ITCH on 2026-09-25 -- a placeholder/cross-listed
    artifact of that feed, not a real Nasdaq closing-cross imbalance), so "more rows wins" mis-
    routed several liquid NYSE names to the wrong venue. pit_listings costs nothing further (spend
    already sunk 9/18) and is authoritative; this costs zero additional Databento $." No `spend`
    call is made here."""
    lists = load_lists()
    universe = lists['_union']
    log(f'=== venue (pit_listings, MIC->venue): {len(universe)} symbols ===')
    sys.path.insert(0, REPO)
    from research.scripts.pit_listings import PitListings  # noqa: E402
    pit = PitListings()
    lo, hi = pit.coverage
    log(f'pit_listings coverage: {lo}..{hi}')
    mic_to_venue = {'XNAS': 'XNAS.ITCH', 'XNYS': 'XNYS.PILLAR'}
    # most-recent-year-first: a listing venue can change, so prefer the freshest as-of date
    # available for a symbol that also falls inside pit_listings' bought coverage window.
    as_of_by_year = {y: lists[y]['as_of'] for y in ('2026', '2025', '2024') if y in lists}

    def lookup(sym):
        for y, as_of in as_of_by_year.items():
            if sym not in lists[y]['symbols']:
                continue
            try:
                mic = pit.listing_exchange(sym, as_of)
            except KeyError:
                continue  # as_of date outside the bought window (e.g. a pre-2024-07 as_of)
            if mic:
                return mic, as_of
        return None, None

    venue_map, mic_other, missing = {}, {}, []
    for s in universe:
        mic, as_of = lookup(s)
        if mic is None:
            missing.append(s)
        elif mic in mic_to_venue:
            venue_map[s] = mic_to_venue[mic]
        else:
            mic_other[s] = mic
    n_nasdaq = sum(1 for v in venue_map.values() if v == 'XNAS.ITCH')
    n_nyse = len(venue_map) - n_nasdaq
    log(f'venue map: {len(venue_map)} resolved ({n_nasdaq} XNAS.ITCH / {n_nyse} XNYS.PILLAR); '
        f'{len(mic_other)} on a MIC this study does not fetch (dropped): {mic_other}; '
        f'{len(missing)} with no pit_listings record on their as-of date (dropped): {missing}')
    pd.DataFrame({'symbol': list(venue_map.keys()), 'venue': list(venue_map.values())}) \
        .to_csv(VENUE_CSV, index=False)
    pd.DataFrame({'symbol': list(missing) + list(mic_other.keys()),
                  'reason': ['no_pit_record'] * len(missing) + [f'mic_{m}' for m in mic_other.values()]}) \
        .to_csv(os.path.join(CACHE, 'venue_unresolved.csv'), index=False)
    log(f'wrote {VENUE_CSV} and dbn_cache/venue_unresolved.csv')
    return 0


def load_venue_map() -> dict:
    if not os.path.exists(VENUE_CSV):
        raise RuntimeError('venue_map.csv missing -- run `probe-venue` first')
    df = pd.read_csv(VENUE_CSV, dtype=str)
    return dict(zip(df.symbol, df.venue))


# ---------------------------------------------------------------------------
# mode: estimate -- sample days, project the full-sample cost
# ---------------------------------------------------------------------------

def cmd_estimate(client, spend):
    venue_map = load_venue_map()
    by_venue = {v: [] for v in VENUES}
    for s, v in venue_map.items():
        by_venue[v].append(s)
    all_days = sessions(SAMPLE_START, SAMPLE_END)
    n_days = len(all_days)
    sample_idx = sorted(set([0, n_days // 4, n_days // 2, (3 * n_days) // 4, n_days - 1]))
    sample_days = [all_days[i] for i in sample_idx]
    log(f'=== estimate: {n_days} sessions total {SAMPLE_START}..{SAMPLE_END}, '
        f'sampling {len(sample_days)}: {sample_days} ===')
    per_day_costs = []
    for day in sample_days:
        day_total = 0.0
        imb_start, imb_end = et_window_utc(day, 15, 45, 0, 16, 0, 0)
        qt_start, qt_end = et_window_utc(day, *QUOTE_WINDOW_ET)
        for venue, syms in by_venue.items():
            if not syms:
                continue
            c1 = get_cost_safe(client, dataset=venue, schema='imbalance', symbols=syms,
                                stype_in='raw_symbol', start=imb_start, end=imb_end)
            c2 = get_cost_safe(client, dataset=venue, schema=QUOTES_SCHEMA, symbols=syms,
                                stype_in='raw_symbol', start=qt_start, end=qt_end)
            log(f'  {day} {venue}: imbalance ${c1:.4f} + {QUOTES_SCHEMA} ${c2:.4f} ({len(syms)} syms)')
            day_total += c1 + c2
        per_day_costs.append(day_total)
    avg_day = sum(per_day_costs) / len(per_day_costs)
    projected_total = avg_day * n_days
    log(f'sampled per-day cost: {[round(c, 4) for c in per_day_costs]}, avg ${avg_day:.4f}/day')
    log(f'PROJECTED full-sample cost ({n_days} sessions): ${projected_total:.2f} '
        f'(cap ${SPEND_CAP_USD:.2f})')
    result = {'n_days': n_days, 'sample_days': sample_days, 'per_day_costs': per_day_costs,
              'avg_day_usd': avg_day, 'projected_total_usd': projected_total, 'cap_usd': SPEND_CAP_USD,
              'safe_to_run': projected_total < SPEND_CAP_USD * 0.9}
    with open(ESTIMATE_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    log(f'wrote {ESTIMATE_JSON} -- safe_to_run={result["safe_to_run"]}')
    if not result['safe_to_run']:
        log(f'ERROR projected cost ${projected_total:.2f} is not safely under the ${SPEND_CAP_USD:.2f} '
            f'cap (>90% margin) -- do NOT run the full fetch without narrowing scope')
        return 2
    return 0


# ---------------------------------------------------------------------------
# mode: run -- the full resumable fetch
# ---------------------------------------------------------------------------

# candidate raw-field names Databento may use; first present wins. Verified against a real probe
# in probe-schema (see dbn_cache/fetch_log.txt); kept as a list of candidates so a name mismatch
# degrades to "canonical column all-NaN + WARNING" rather than a crash.
IMB_CANDIDATES = {
    'ref_px': ['ref_price', 'cont_book_clr_price', 'auct_interest_clr_price'],
    'paired_qty': ['paired_qty'],
    'imbalance_qty': ['total_imbalance_qty', 'imbalance_qty', 'market_imbalance_qty'],
    'side': ['side'],
}
QT_CANDIDATES = {
    'bid_px': ['bid_px_00', 'bid_px'],
    'ask_px': ['ask_px_00', 'ask_px'],
}


def first_present(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def shard_path(kind: str, day: str, venue: str) -> str:
    d = SHARDS_IMB if kind == 'imbalance' else SHARDS_QT
    return os.path.join(d, f'{day}_{venue.replace(".", "-")}.parquet')


def et_date_of(ts_utc_series: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_utc_series, utc=True).dt.tz_convert(ET).dt.strftime('%Y-%m-%d')


def tag_quote_window(ts_utc_series: pd.Series) -> pd.Series:
    """Amendment 1: tag each bbo-1m snapshot with its ET clock minute 'HH:MM' rather than the old
    fixed {1550, 1555} vocabulary -- the pull now spans a continuous range (QUOTE_WINDOW_ET) so any
    minute in it (reference mid, through-touch check, or buffer) gets a real label instead of being
    dropped. A snapshot lands exactly on the minute boundary (verified live 9/28), so a plain
    strftime is exact -- no window-matching arithmetic needed."""
    ts = pd.to_datetime(ts_utc_series, utc=True).dt.tz_convert(ET)
    return ts.dt.strftime('%H:%M')


def process_imbalance_df(df: pd.DataFrame, day: str, venue: str) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    out = df.copy()
    out['venue'] = venue
    out['date'] = day
    if 'symbol' not in out.columns:
        log(f'WARNING {venue} {day} imbalance: no "symbol" column in the response '
            f'(columns={list(df.columns)}) -- shard written with raw fields only, unusable for '
            f'per-symbol joins until resolved')
    ts_col = 'ts_event' if 'ts_event' in out.columns else ('ts_recv' if 'ts_recv' in out.columns else None)
    out['ts'] = out[ts_col] if ts_col else pd.NaT
    for canon, cands in IMB_CANDIDATES.items():
        src = first_present(out, cands)
        out[canon] = out[src] if src else pd.NA
        if src is None:
            log(f'WARNING {venue} {day} imbalance: none of {cands} present for canonical "{canon}"')
    keep_front = ['symbol', 'date', 'venue', 'ts', 'ref_px', 'paired_qty', 'imbalance_qty', 'side']
    keep_front = [c for c in keep_front if c in out.columns]
    rest = [c for c in out.columns if c not in keep_front]
    return out[keep_front + rest]


def process_quotes_df(df: pd.DataFrame, day: str, venue: str) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    out = df.copy()
    out['venue'] = venue
    out['date'] = day
    ts_col = 'ts_event' if 'ts_event' in out.columns else ('ts_recv' if 'ts_recv' in out.columns else None)
    out['ts'] = out[ts_col] if ts_col else pd.NaT
    out['minute'] = tag_quote_window(out['ts']) if ts_col else pd.NA
    # safety net only -- QUOTE_WINDOW_ET already bounds the pull to ~15:49-15:58 ET, so this should
    # normally drop nothing; guards against a stray row if the API's edge behaviour ever changes.
    out = out[(out['minute'] >= '15:49') & (out['minute'] <= '15:58')].copy()
    if not len(out):
        return out
    bid_c = first_present(out, QT_CANDIDATES['bid_px'])
    ask_c = first_present(out, QT_CANDIDATES['ask_px'])
    out['bid_px'] = out[bid_c] if bid_c else pd.NA
    out['ask_px'] = out[ask_c] if ask_c else pd.NA
    if bid_c is None or ask_c is None:
        log(f'WARNING {venue} {day} {QUOTES_SCHEMA}: bid/ask columns not found (had {list(df.columns)}) -- '
            f'mid will be NaN')
    try:
        out['mid'] = (pd.to_numeric(out['bid_px'], errors='coerce')
                       + pd.to_numeric(out['ask_px'], errors='coerce')) / 2.0
    except Exception:
        out['mid'] = pd.NA
    if 'symbol' not in out.columns:
        log(f'WARNING {venue} {day} {QUOTES_SCHEMA}: no "symbol" column in the response')
    keep_front = ['symbol', 'date', 'venue', 'minute', 'ts', 'bid_px', 'ask_px', 'mid']
    keep_front = [c for c in keep_front if c in out.columns]
    rest = [c for c in out.columns if c not in keep_front]
    return out[keep_front + rest]


def cmd_run(client, spend, start, end):
    venue_map = load_venue_map()
    by_venue = {v: [] for v in VENUES}
    for s, v in venue_map.items():
        by_venue[v].append(s)
    days = sessions(start, end)
    log(f'=== run: {len(days)} sessions {start}..{end}, venues '
        f'{[(v, len(by_venue[v])) for v in VENUES]} ===')
    n_skip_imb = n_pull_imb = n_skip_qt = n_pull_qt = 0
    for i, day in enumerate(days):
        for venue, syms in by_venue.items():
            if not syms:
                continue
            sp = shard_path('imbalance', day, venue)
            if os.path.exists(sp):
                n_skip_imb += 1
            else:
                imb_start, imb_end = et_window_utc(day, 15, 45, 0, 16, 0, 0)
                df = priced_pull(client, spend, f'imbalance {venue} {day} ({len(syms)} syms)',
                                  dataset=venue, schema='imbalance', symbols=syms,
                                  stype_in='raw_symbol', start=imb_start, end=imb_end)
                if df is None:
                    log(f'STOP at day {i + 1}/{len(days)} ({day}) -- spend cap reached during imbalance pull')
                    cmd_consolidate()
                    return 2
                process_imbalance_df(df, day, venue).to_parquet(sp, index=False)
                n_pull_imb += 1
            sq = shard_path('quotes', day, venue)
            if os.path.exists(sq):
                n_skip_qt += 1
            else:
                qt_start, qt_end = et_window_utc(day, *QUOTE_WINDOW_ET)
                df = priced_pull(client, spend, f'{QUOTES_SCHEMA} {venue} {day} ({len(syms)} syms)',
                                  dataset=venue, schema=QUOTES_SCHEMA, symbols=syms,
                                  stype_in='raw_symbol', start=qt_start, end=qt_end)
                if df is None:
                    log(f'STOP at day {i + 1}/{len(days)} ({day}) -- spend cap reached during {QUOTES_SCHEMA} pull')
                    cmd_consolidate()
                    return 2
                process_quotes_df(df, day, venue).to_parquet(sq, index=False)
                n_pull_qt += 1
        if (i + 1) % 20 == 0 or i == len(days) - 1:
            log(f'progress {i + 1}/{len(days)} sessions ({day}); imbalance pulled={n_pull_imb} '
                f'skipped={n_skip_imb}; {QUOTES_SCHEMA} pulled={n_pull_qt} skipped={n_skip_qt}; '
                f'spend ${spend["total_usd"]:.4f}')
    log(f'run complete: {len(days)} sessions. imbalance pulled={n_pull_imb} skipped(cached)={n_skip_imb}; '
        f'{QUOTES_SCHEMA} pulled={n_pull_qt} skipped(cached)={n_skip_qt}; total spend ${spend["total_usd"]:.4f}')
    cmd_consolidate()
    if start == SAMPLE_START and end == SAMPLE_END:
        with open(SENTINEL_DONE, 'w') as f:
            f.write(f'{datetime.now(timezone.utc).isoformat()} FETCH_DONE_1630: {len(days)} sessions '
                    f'{start}..{end}, total spend ${spend["total_usd"]:.4f}\n')
        log(f'wrote completion sentinel {SENTINEL_DONE}')
    else:
        log(f'run range {start}..{end} is not the full configured sample ({SAMPLE_START}..{SAMPLE_END}) '
            f'-- no completion sentinel written')
    return 0


def cmd_consolidate():
    """Amendment 1: shards are filtered to the CURRENT [SAMPLE_START, SAMPLE_END] before being
    concatenated -- the original spec pulled from 2024-01-02, so shards dated before the amended
    2024-07-01 start are real cached data that must stay on disk UNUSED, never entering the
    consolidated file, per Amendment 1."""
    for kind, shard_dir, out_path in (
        ('imbalance', SHARDS_IMB, IMBALANCE_PARQUET),
        ('quotes', SHARDS_QT, QUOTES_PARQUET),
    ):
        all_files = sorted(glob.glob(os.path.join(shard_dir, '*.parquet')))
        files = [fp for fp in all_files
                 if SAMPLE_START <= os.path.basename(fp)[:10] <= SAMPLE_END]
        n_out_of_range = len(all_files) - len(files)
        if n_out_of_range:
            log(f'consolidate {kind}: {n_out_of_range} shard(s) outside {SAMPLE_START}..{SAMPLE_END} '
                f'left unused on disk (Amendment 1 -- not re-spent, not consolidated)')
        if not files:
            log(f'consolidate {kind}: no in-range shards yet')
            continue
        parts = []
        for fp in files:
            try:
                p = pd.read_parquet(fp)
                if len(p):
                    parts.append(p)
            except Exception as exc:
                log(f'WARNING could not read shard {fp}: {str(exc)[:150]}')
        if not parts:
            log(f'consolidate {kind}: {len(files)} shard files, all empty')
            continue
        full = pd.concat(parts, ignore_index=True, sort=False)
        full.to_parquet(out_path, index=False)
        n_sym = full['symbol'].nunique() if 'symbol' in full.columns else float('nan')
        n_day = full['date'].nunique() if 'date' in full.columns else float('nan')
        log(f'consolidate {kind}: {len(files)} shards -> {out_path} ({len(full):,} rows, '
            f'{n_sym} symbols, {n_day} sessions)')
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['plan', 'calendar', 'probe-schema', 'probe-venue', 'estimate',
                                      'run', 'consolidate'])
    ap.add_argument('--start', default=SAMPLE_START)
    ap.add_argument('--end', default=SAMPLE_END)
    args = ap.parse_args()

    if args.mode == 'plan':
        build_liquid_lists()
        return 0
    if args.mode == 'consolidate':
        return cmd_consolidate()

    if 'DATABENTO_API_KEY' not in os.environ or not os.environ['DATABENTO_API_KEY']:
        log('ERROR DATABENTO_API_KEY missing from environment/.env -- refusing to run '
            '(production code never falls back to mock data)')
        return 1
    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    spend = load_spend()
    log(f'--- mode={args.mode} start (spend so far ${spend["total_usd"]:.4f} / cap ${SPEND_CAP_USD:.2f}) ---')
    if spend['total_usd'] > SPEND_CAP_USD:
        log(f'ERROR spend ledger already ${spend["total_usd"]:.4f} > cap ${SPEND_CAP_USD:.2f} -- refusing to spend more')
        return 2

    ensure_trading_calendar(client, spend)  # every remaining mode uses sessions() directly or indirectly
    if args.mode == 'calendar':
        rc = 0
    elif args.mode == 'probe-schema':
        rc = cmd_probe_schema(client, spend)
    elif args.mode == 'probe-venue':
        rc = cmd_probe_venue(client, spend)
    elif args.mode == 'estimate':
        rc = cmd_estimate(client, spend)
    elif args.mode == 'run':
        rc = cmd_run(client, spend, args.start, args.end)
    else:
        rc = 1
    log(f'--- mode={args.mode} exit {rc} (spend now ${spend["total_usd"]:.4f}) ---')
    return rc


if __name__ == '__main__':
    sys.exit(main())
