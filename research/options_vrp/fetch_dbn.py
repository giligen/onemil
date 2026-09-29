#!/usr/bin/env python3
"""PREREG_1567 v3 (Amendment 3) FETCH stage — Databento OPRA.PILLAR cbbo-1m.

Executes ONLY the fetch: Monday option ladders (strike selection quotes), the definition
schema (strike/expiry/right per instrument), the full-life cbbo-1m of every put contract in
the superset around the 0.20- and 0.30-delta strikes (plus its $10-below partner), and the SPY
underlying prices needed for strike selection and daily/exit marks, for both the PANEL
(2024-02-05..2026-08-17 entries) and the EXTENSION (2013-04-08..2023-12-25 entries).

Cost discipline (owner rule, hard cap $150 total for this stage):
  * every purchase is preceded by metadata.get_cost and logged to opt_cache/dbn/spend.json
    (running total); a purchase that would push the total over the cap is SKIPPED and logged
    as an ERROR, never silently dropped.
  * every result is cached as parquet under opt_cache/dbn/ and is resumable: a rerun skips any
    (Monday, leg) already on disk.

Known, disclosed data gap (reported here, not hidden): Databento's own equities datasets
(ARCX.PILLAR, XNAS.ITCH, DBEQ.BASIC, EQUS.*) all start 2018-05-01 or later, and Alpaca's stock
history starts 2016-01-04 (verified empirically 2026-09-28: daily bars for SPY are empty before
that date on this account). Neither source can price the SPY underlying for 2013-04-08 through
2015-12-31. Those entry Mondays are still walked for the OPTION side (OPRA covers 2013-04-01
onward) but SPY spot is NaN and flagged `spot_source=GAP`; strike selection for those weeks
cannot be done from a minute spot price in this stage (see FETCH_DBN.md) and is left to the BUILD
stage to resolve (e.g. put-call parity from the same cbbo-1m chain) or to exclude.

Usage:
  python3 research/options_vrp/fetch_dbn.py --cost-only     # estimate total cost, no purchase
  python3 research/options_vrp/fetch_dbn.py                 # full resumable run
  python3 research/options_vrp/fetch_dbn.py --stage spy      # SPY underlying only
  python3 research/options_vrp/fetch_dbn.py --stage mondays  # Monday ladders only
  python3 research/options_vrp/fetch_dbn.py --stage legs     # per-leg full-life only

Legs-stage backlog resume (PULL_REPLAN_20260929 #5): the 'legs'/'all' stages rebuild leg_plan
from mondays.parquet's own cached rows for every entry Monday in [--start, --end] (default the
2016-2023 extension backlog), REGARDLESS of whether this run's 'mondays' stage touched that
Monday -- this is what makes a `--stage legs` rerun drain already-selected-but-never-pulled legs
instead of silently seeing an empty leg_plan. Before any per-leg cost check or purchase, a DRY
PLAN line is logged (Mondays in range, legs needed/cached/to-pull, projected $ at the ledger's own
mean $/leg); `--cost-only` stops right after that line (zero network calls -- it does not even
call metadata.get_cost).
  python3 research/options_vrp/fetch_dbn.py --stage legs --batched --start 2016-01-04 --end 2023-12-25 --cost-only
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_CACHE = os.path.join(HERE, 'opt_cache')
CACHE_DIR = os.path.join(HERE, 'opt_cache', 'dbn')
LEGS_DIR = os.path.join(CACHE_DIR, 'legs')
SPEND_PATH = os.path.join(CACHE_DIR, 'spend.json')
MONDAYS_PATH = os.path.join(CACHE_DIR, 'mondays.parquet')
SPY_PATH = os.path.join(CACHE_DIR, 'spy_prices.parquet')
DEFS_PATH = os.path.join(CACHE_DIR, 'definitions.parquet')
os.makedirs(LEGS_DIR, exist_ok=True)

SPEND_CAP_USD = 150.0
ET = ZoneInfo('America/New_York')
UTC = dt.timezone.utc

PANEL_START, PANEL_END = dt.date(2024, 2, 5), dt.date(2026, 8, 17)
EXT_START, EXT_END = dt.date(2013, 4, 8), dt.date(2023, 12, 25)
SPY_EQUITY_GAP_START, SPY_EQUITY_GAP_END = dt.date(2013, 4, 8), dt.date(2015, 12, 31)  # neither source covers this
ALPACA_EQUITY_START = dt.date(2016, 1, 4)  # verified empirically 2026-09-28
DBN_EQUITY_START = dt.date(2018, 5, 1)     # ARCX.PILLAR / XNAS.ITCH verified 2026-09-28; SPY listed ARCX -> use ARCX

DELTAS = [0.20, 0.30]
WIDTH = 10.0
DTE_LO, DTE_HI, DTE_TARGET = 38, 52, 45
STRIKE_STEP = 1.0
SUPERSET_BAND = 3  # strikes either side of each delta target

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(os.path.join(HERE, 'fetch_dbn.log')), logging.StreamHandler()],
)
log = logging.getLogger('fetch_dbn')

load_dotenv(os.path.join(ROOT, '.env'))
try:
    import databento as db
except ImportError:
    log.error("databento package not installed -- cannot run the option fetch")
    raise

_API_KEY = os.environ.get('DATABENTO_API_KEY')
if not _API_KEY:
    log.error("DATABENTO_API_KEY missing from .env -- cannot run the option fetch")
    raise SystemExit(1)
CLIENT = db.Historical(_API_KEY)


# --------------------------------------------------------------------------------------
# Spend guard -- every Databento purchase goes through this. Never call timeseries.get_range
# directly elsewhere in this file.
# --------------------------------------------------------------------------------------
def load_spend():
    if os.path.exists(SPEND_PATH):
        with open(SPEND_PATH) as f:
            return json.load(f)
    return {'total_usd': 0.0, 'purchases': [], 'skipped': []}


def save_spend(state):
    tmp = SPEND_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, SPEND_PATH)


SPEND = load_spend()


def guarded_get_range(cost_only, **kwargs):
    """Cost-checked, spend-capped, logged Databento purchase.

    Returns a DBNStore on success, None if skipped (cap hit) or on error (logged).
    """
    label = f"{kwargs.get('dataset')}/{kwargs.get('schema')} " \
            f"{kwargs.get('symbols')} {kwargs.get('start')}..{kwargs.get('end')}"
    try:
        cost = CLIENT.metadata.get_cost(**kwargs)
    except Exception as e:
        log.error(f"get_cost FAILED for {label}: {e}")
        return None
    projected = SPEND['total_usd'] + cost
    if projected > SPEND_CAP_USD:
        log.error(
            f"SPEND CAP HIT: current ${SPEND['total_usd']:.4f} + ${cost:.4f} = "
            f"${projected:.4f} > ${SPEND_CAP_USD} -- SKIPPING {label}"
        )
        SPEND['skipped'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label})
        save_spend(SPEND)
        return None
    if cost_only:
        log.info(f"[cost-only] would pull ${cost:.5f} (running total would be ${projected:.4f}) {label}")
        SPEND['total_usd'] = projected  # cost-only still tallies the PROJECTED total, purchases list marked dry
        SPEND['purchases'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label, 'dry_run': True})
        save_spend(SPEND)
        return None
    try:
        data = CLIENT.timeseries.get_range(**kwargs)
    except Exception as e:
        log.error(f"get_range FAILED for {label} (cost was ${cost:.4f}, NOT charged to running total): {e}")
        return None
    SPEND['total_usd'] = projected
    SPEND['purchases'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label})
    save_spend(SPEND)
    log.info(f"PULLED ${cost:.5f} (total ${projected:.4f} / ${SPEND_CAP_USD}) {label}")
    return data


# --------------------------------------------------------------------------------------
# NYSE trading-day calendar (no external calendar package on this node -- verified 2026-09-28).
# Fixed + floating holidays, nearest-weekday observed rule for the fixed ones. Good Friday via
# the Gauss/Meeus computus for Easter Sunday.
# --------------------------------------------------------------------------------------
def _easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        d = dt.date(year, 12, 31)
    else:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - dt.timedelta(days=offset)


def _observed(d):
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(year):
    hols = {
        _observed(dt.date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),          # MLK day, 3rd Monday Jan
        _nth_weekday(year, 2, 0, 3),          # Washington's birthday, 3rd Monday Feb
        _easter_sunday(year) - dt.timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),            # Memorial Day, last Monday May
        _nth_weekday(year, 9, 0, 1),          # Labor Day, 1st Monday Sept
        _nth_weekday(year, 11, 3, 4),         # Thanksgiving, 4th Thursday Nov
        _observed(dt.date(year, 7, 4)),
        _observed(dt.date(year, 12, 25)),
    }
    if year >= 2022:
        hols.add(_observed(dt.date(year, 6, 19)))  # Juneteenth, NYSE since 2022
    return hols


_HOL_CACHE = {}


def is_trading_day(d):
    if d.weekday() >= 5:
        return False
    if d.year not in _HOL_CACHE:
        _HOL_CACHE[d.year] = nyse_holidays(d.year)
    return d not in _HOL_CACHE[d.year]


def entry_session(monday):
    """First trading session of the ISO week containing `monday` (a date on a Monday)."""
    d = monday
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def entry_mondays(start, end):
    out = []
    d = start - dt.timedelta(days=start.weekday())  # snap to that week's Monday
    while d <= end:
        sess = entry_session(d)
        if start <= sess <= end + dt.timedelta(days=7):
            out.append(sess)
        d += dt.timedelta(days=7)
    return sorted(set(out))


# --------------------------------------------------------------------------------------
# Black-Scholes put price / delta / implied vol (bisection -- no scipy dependency needed here,
# though scipy is present on this node; bisection is simpler to audit for an independent rebuild).
# --------------------------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_put_delta(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol_put(price, S, K, T, r, q, lo=1e-4, hi=5.0, iters=60):
    if price <= 0 or T <= 0:
        return None
    flo = bs_put_price(S, K, T, r, q, lo) - price
    fhi = bs_put_price(S, K, T, r, q, hi) - price
    if flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = bs_put_price(S, K, T, r, q, mid) - price
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Stage 1: SPY underlying. Alpaca for 2016-01-04.. (already cached 2024-01-02..2026-09-25 by
# fetch_options.py); the 2013-04-08..2015-12-31 gap is unfillable by any source checked here
# (Databento equities datasets start 2018-05-01; Alpaca starts 2016-01-04) and is written to
# spy_prices.parquet with spot=NaN, source='GAP'.
# --------------------------------------------------------------------------------------
def fetch_spy_prices(cost_only):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    rows = []
    # Reuse the existing Alpaca cache (2024-01-02..2026-09-25) built by fetch_options.py -- no re-pull.
    old_daily = os.path.join(OLD_CACHE, 'spy_daily.parquet')
    old_minute = os.path.join(OLD_CACHE, 'spy_minute.parquet')
    daily_cached = pd.read_parquet(old_daily) if os.path.exists(old_daily) else pd.DataFrame()
    minute_cached = pd.read_parquet(old_minute) if os.path.exists(old_minute) else pd.DataFrame()
    log.info(f"reused existing Alpaca SPY cache: {len(daily_cached)} daily rows, {len(minute_cached)} minute rows")

    sessions_needed = [d for d in _all_sessions(EXT_START, PANEL_END) if d < dt.date(2024, 1, 2)]
    gap_days = [d for d in sessions_needed if d < ALPACA_EQUITY_START]
    fetch_days = [d for d in sessions_needed if d >= ALPACA_EQUITY_START]
    log.info(f"SPY underlying: {len(gap_days)} sessions in the unfillable gap "
             f"({SPY_EQUITY_GAP_START}..{ALPACA_EQUITY_START - dt.timedelta(days=1)}), "
             f"{len(fetch_days)} sessions to fetch from Alpaca ({ALPACA_EQUITY_START}..2024-01-01)")

    for d in gap_days:
        rows.append({'day': d.isoformat(), 'spot_10': np.nan, 'spot_16': np.nan, 'close': np.nan, 'source': 'GAP'})

    if fetch_days and not cost_only:
        client = StockHistoricalDataClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'])
        start_dt = dt.datetime.combine(fetch_days[0], dt.time(0, 0))
        end_dt = dt.datetime.combine(fetch_days[-1] + dt.timedelta(days=1), dt.time(0, 0))
        req = StockBarsRequest(symbol_or_symbols='SPY', timeframe=TimeFrame.Minute, start=start_dt, end=end_dt)
        try:
            mdf = client.get_stock_bars(req).df.reset_index()
        except Exception as e:
            log.error(f"Alpaca minute pull failed for {fetch_days[0]}..{fetch_days[-1]}: {e}")
            mdf = pd.DataFrame()
        if len(mdf):
            mdf['timestamp'] = pd.to_datetime(mdf['timestamp'], utc=True).dt.tz_convert(ET)
            mdf['day'] = mdf['timestamp'].dt.date
            mdf['hm'] = mdf['timestamp'].dt.strftime('%H:%M')
            by_day = mdf.groupby('day')
            for d in fetch_days:
                if d not in by_day.groups:
                    rows.append({'day': d.isoformat(), 'spot_10': np.nan, 'spot_16': np.nan, 'close': np.nan, 'source': 'ALPACA_MISSING'})
                    continue
                day_df = by_day.get_group(d)
                r10 = day_df[day_df['hm'] == '10:00']
                r16 = day_df[day_df['hm'] == '15:59']  # last RTH minute bar timestamped 15:59
                rows.append({
                    'day': d.isoformat(),
                    'spot_10': float(r10['close'].iloc[0]) if len(r10) else np.nan,
                    'spot_16': float(r16['close'].iloc[0]) if len(r16) else np.nan,
                    'close': np.nan,
                    'source': 'ALPACA',
                })
        else:
            for d in fetch_days:
                rows.append({'day': d.isoformat(), 'spot_10': np.nan, 'spot_16': np.nan, 'close': np.nan, 'source': 'ALPACA_MISSING'})

    new_df = pd.DataFrame(rows)
    # Fold in the reused 2024-01-02..2026-09-25 cache (spot_10 from minute, close from daily).
    reuse_rows = []
    if len(daily_cached):
        m10 = pd.DataFrame()
        if len(minute_cached):
            minute_cached['t'] = pd.to_datetime(minute_cached.get('t', minute_cached.get('timestamp')), utc=True).dt.tz_convert(ET)
            minute_cached['day'] = minute_cached['t'].dt.date.astype(str)
            minute_cached['hm'] = minute_cached['t'].dt.strftime('%H:%M')
            m10 = minute_cached[minute_cached['hm'] == '10:00'][['day', 'c']].rename(columns={'c': 'spot_10'})
            m16 = minute_cached[minute_cached['hm'] == '15:59'][['day', 'c']].rename(columns={'c': 'spot_16'})
        for _, r in daily_cached.iterrows():
            day = str(r['day'])
            s10 = m10[m10['day'] == day]['spot_10'].iloc[0] if len(m10) and (m10['day'] == day).any() else np.nan
            s16 = m16[m16['day'] == day]['spot_16'].iloc[0] if len(minute_cached) and (m16['day'] == day).any() else np.nan
            reuse_rows.append({'day': day, 'spot_10': s10, 'spot_16': s16, 'close': r['c'], 'source': 'ALPACA_REUSED'})
    reuse_df = pd.DataFrame(reuse_rows)

    out = pd.concat([new_df, reuse_df], ignore_index=True).drop_duplicates(subset='day').sort_values('day')
    if not cost_only:
        out.to_parquet(SPY_PATH, index=False)
        log.info(f"wrote {SPY_PATH}: {len(out)} rows "
                 f"({(out['source']=='GAP').sum()} GAP, {(out['spot_10'].isna() & (out['source']!='GAP')).sum()} missing-but-should-have-data)")
    return out


def _all_sessions(start, end):
    d = start
    out = []
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------------------
# Stage 2: Monday ladders. For each entry Monday: definition schema (strike/expiry/right) +
# cbbo-1m 10:00-10:02 ET for SPY.OPT (parent) puts of the chosen 45-DTE expiry.
# --------------------------------------------------------------------------------------
def _et_window_utc(day, hh, mm, span_min=2):
    start_et = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=ET)
    end_et = start_et + dt.timedelta(minutes=span_min)
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def _pick_expiry(expiries, entry_date):
    cands = [(e, (e - entry_date).days) for e in expiries if DTE_LO <= (e - entry_date).days <= DTE_HI]
    if not cands:
        return None
    return min(cands, key=lambda x: abs(x[1] - DTE_TARGET))[0]


def fetch_mondays(cost_only, spy_df):
    if os.path.exists(MONDAYS_PATH):
        done = pd.read_parquet(MONDAYS_PATH)
        done_days = set(done['entry_date'].unique())
    else:
        done = pd.DataFrame()
        done_days = set()

    all_mondays = entry_mondays(EXT_START, EXT_END) + entry_mondays(PANEL_START, PANEL_END)
    todo = [d for d in all_mondays if d.isoformat() not in done_days]
    log.info(f"Monday ladders: {len(all_mondays)} entry sessions total, {len(todo)} not yet cached")

    spy_by_day = {r['day']: r for _, r in spy_df.iterrows()} if spy_df is not None else {}
    new_rows = []
    leg_plan = {}  # osi -> {'strike':.., 'expiry':.., 'right':'P', 'first_seen': date}

    for entry_date in todo:
        day_str = entry_date.isoformat()
        spot = spy_by_day.get(day_str, {}).get('spot_10', np.nan)
        start_utc, end_utc = _et_window_utc(entry_date, 10, 0, 2)

        # Definitions require a window starting at UTC midnight or the daily snapshot can be
        # missing (verified empirically 2026-09-28, BentoWarning); pull the whole day.
        day_start_utc = dt.datetime.combine(entry_date, dt.time(0, 0), tzinfo=UTC)
        day_end_utc = dt.datetime.combine(entry_date + dt.timedelta(days=1), dt.time(0, 0), tzinfo=UTC)
        defs = guarded_get_range(
            cost_only, dataset='OPRA.PILLAR', schema='definition', stype_in='parent',
            symbols=['SPY.OPT'], start=day_start_utc, end=day_end_utc,
        )
        if defs is None:
            continue
        ddf = defs.to_df()
        puts = ddf[ddf['instrument_class'].astype(str).str.upper().str.startswith('P')] if 'instrument_class' in ddf else ddf
        if 'raw_symbol' not in puts.columns or not len(puts):
            log.warning(f"{day_str}: no put definitions returned, skipping ladder")
            continue
        puts = puts.copy()
        puts['expiration'] = pd.to_datetime(puts['expiration']).dt.date
        puts['strike'] = puts['strike_price'].astype(float) / 1e9 if puts['strike_price'].max() > 1e6 else puts['strike_price'].astype(float)

        expiry = _pick_expiry(sorted(puts['expiration'].unique()), entry_date)
        if expiry is None:
            log.warning(f"{day_str}: no expiry in [{DTE_LO},{DTE_HI}] DTE, skipping")
            continue
        chain = puts[puts['expiration'] == expiry]

        quotes = guarded_get_range(
            cost_only, dataset='OPRA.PILLAR', schema='cbbo-1m', stype_in='parent',
            symbols=['SPY.OPT'], start=start_utc, end=end_utc,
        )
        if quotes is None:
            continue
        qdf = quotes.to_df()
        if not len(qdf):
            log.warning(f"{day_str}: empty cbbo-1m bar, VOID")
            continue
        qdf = qdf.merge(chain[['raw_symbol', 'strike', 'expiration']], left_on='symbol', right_on='raw_symbol', how='inner')
        first_bar = qdf.sort_values('ts_event').groupby('symbol').first().reset_index()

        if np.isfinite(spot):
            band_mask = (first_bar['strike'] >= 0.80 * spot) & (first_bar['strike'] <= spot)
            strikes_by_leg = first_bar[band_mask]
            if not len(strikes_by_leg):
                strikes_by_leg = first_bar
        else:
            strikes_by_leg = first_bar

        for _, r in strikes_by_leg.iterrows():
            new_rows.append({
                'entry_date': day_str, 'symbol': r['symbol'], 'strike': r['strike'], 'expiry': str(expiry),
                'bid': r.get('bid_px_00', np.nan), 'ask': r.get('ask_px_00', np.nan),
                'bid_sz': r.get('bid_sz_00', np.nan), 'ask_sz': r.get('ask_sz_00', np.nan),
                'spot_10': spot,
            })
            leg_plan.setdefault(r['symbol'], {'strike': float(r['strike']), 'expiry': str(expiry), 'first_seen': day_str})

        # Superset selection needs delta/IV; only computable where spot and DTE are finite.
        if np.isfinite(spot):
            T = (expiry - entry_date).days / 365.0
            deltas = []
            for _, r in strikes_by_leg.iterrows():
                bid, ask = r.get('bid_px_00', np.nan), r.get('ask_px_00', np.nan)
                if not (np.isfinite(bid) and np.isfinite(ask)) or bid <= 0:
                    continue
                mid = 0.5 * (bid + ask)
                iv = implied_vol_put(mid, spot, float(r['strike']), T, 0.045, 0.013)
                if iv is None:
                    continue
                delta = bs_put_delta(spot, float(r['strike']), T, 0.045, 0.013, iv)
                deltas.append((float(r['strike']), abs(delta)))
            for target in DELTAS:
                if not deltas:
                    break
                strike0 = min(deltas, key=lambda x: abs(x[1] - target))[0]
                for k in range(-SUPERSET_BAND, SUPERSET_BAND + 1):
                    kk = strike0 + k * STRIKE_STEP
                    for partner in (kk, kk - WIDTH):
                        row = chain[np.isclose(chain['strike'], partner)]
                        if len(row):
                            osi = row['raw_symbol'].iloc[0]
                            leg_plan.setdefault(osi, {'strike': float(partner), 'expiry': str(expiry), 'first_seen': day_str})

    if new_rows:
        add = pd.DataFrame(new_rows)
        out = pd.concat([done, add], ignore_index=True) if len(done) else add
        if not cost_only:
            out.to_parquet(MONDAYS_PATH, index=False)
            log.info(f"wrote {MONDAYS_PATH}: {len(out)} rows total ({len(add)} new)")
    else:
        out = done
    return out, leg_plan


# --------------------------------------------------------------------------------------
# Stage 3: per-leg full-life cbbo-1m for the superset (both PANEL and EXTENSION selections
# feed the same leg_plan; already-cached OSIs are skipped).
# --------------------------------------------------------------------------------------
def _leg_window(info):
    """(life_start, life_end) in UTC for one leg_plan entry -- identical formula used by both
    the per-leg and batched fetch paths, so the two are byte-for-byte comparable in cost."""
    expiry = dt.date.fromisoformat(info['expiry'])
    first_seen = dt.date.fromisoformat(info['first_seen'])
    life_start = dt.datetime.combine(first_seen, dt.time(9, 30), tzinfo=ET).astimezone(UTC)
    life_end = dt.datetime.combine(min(expiry, dt.date.today()) + dt.timedelta(days=1), dt.time(0, 0), tzinfo=ET).astimezone(UTC)
    return life_start, life_end


LEG_LABEL_RE = re.compile(r"^OPRA\.PILLAR/cbbo-1m \[('[^']*')\]")


def ledger_mean_cost_per_leg():
    """Mean $/leg from the spend ledger's own single-leg cbbo-1m purchases (same method as
    PULL_REPLAN_20260929 #1): a label matches only when its symbol list is exactly one raw_symbol,
    which excludes both the 'SPY.OPT' parent-symbol Monday-ladder overhead calls (definition +
    the 10:00 cbbo-1m snapshot) and multi-leg batched purchases (not attributable to one leg).
    Used only to PROJECT cost for legs not yet pulled -- never charged, never gates the spend cap.
    """
    costs = []
    for p in SPEND.get('purchases', []):
        mo = LEG_LABEL_RE.match(p.get('label', ''))
        if mo and mo.group(1) != "'SPY.OPT'":
            costs.append(p['cost_usd'])
    if not costs:
        log.warning("ledger_mean_cost_per_leg: no single-leg cbbo-1m purchase in spend.json yet -- "
                    "projected $ will be $0.00, NOT a real cost estimate")
        return 0.0
    return sum(costs) / len(costs)


def rebuild_leg_plan_from_mondays(mondays_df, start, end):
    """Rebuild leg_plan directly from mondays.parquet's own cached rows for every entry Monday in
    [start, end], regardless of whether THIS run's fetch_mondays() call touched that Monday.

    Fixes the blocker in PULL_REPLAN_20260929 #5: fetch_mondays() only ever returns leg_plan
    entries for Mondays it freshly processes ('todo'); once a Monday's rows are already on disk,
    its legs were silently dropped from every later run's leg_plan, so the 16,102-leg 2016-2023
    backlog (already selected, never pulled) was invisible to `--stage legs`. This restores
    exactly the band-mask legs that are literally rows in mondays.parquet -- it does NOT recompute
    the delta-targeted superset partners (that needs a fresh per-Monday option-definition pull,
    out of scope here; see PULL_REPLAN #3).

    Returns (leg_plan, n_mondays) where n_mondays is the count of distinct entry Mondays in range
    that are present in mondays_df (whether or not every one contributes a still-uncached leg).
    """
    if mondays_df is None or not len(mondays_df):
        log.warning(f"rebuild_leg_plan_from_mondays: mondays_df empty/missing -- 0 legs rebuilt "
                    f"for [{start}..{end}]")
        return {}, 0
    df = mondays_df.copy()
    df['_entry_date'] = pd.to_datetime(df['entry_date']).dt.date
    in_range = df[(df['_entry_date'] >= start) & (df['_entry_date'] <= end)].sort_values('_entry_date')
    leg_plan = {}
    for _, r in in_range.iterrows():
        leg_plan.setdefault(r['symbol'], {
            'strike': float(r['strike']), 'expiry': str(r['expiry']), 'first_seen': str(r['_entry_date']),
        })
    n_mondays = int(in_range['_entry_date'].nunique())
    log.info(f"rebuild_leg_plan_from_mondays: {n_mondays} Mondays in [{start}..{end}], "
             f"{len(leg_plan)} unique OSIs (band-mask legs) rebuilt from {MONDAYS_PATH}")
    return leg_plan, n_mondays


def print_dry_plan(leg_plan, start, end, n_mondays):
    """Log the dry-plan line required before ANY per-leg cost check or purchase. Purely local
    (mondays.parquet rows already in leg_plan + on-disk opt_cache/dbn/legs/ + the ledger's own
    history) -- zero network calls, so this is safe to log even when the caller is about to stop
    for --cost-only.
    """
    legs_needed = len(leg_plan)
    legs_cached = sum(1 for osi in leg_plan if os.path.exists(os.path.join(LEGS_DIR, f"{osi.strip()}.parquet")))
    legs_to_pull = legs_needed - legs_cached
    mean_cost = ledger_mean_cost_per_leg()
    projected = legs_to_pull * mean_cost
    log.info(
        f"DRY PLAN [{start}..{end}]: {n_mondays} Mondays in range, {legs_needed} legs needed, "
        f"{legs_cached} legs cached, {legs_to_pull} legs to pull, projected ${projected:.2f} "
        f"at ${mean_cost:.5f}/leg (ledger mean)"
    )
    return {'n_mondays': n_mondays, 'legs_needed': legs_needed, 'legs_cached': legs_cached,
            'legs_to_pull': legs_to_pull, 'mean_cost_per_leg': mean_cost, 'projected_usd': projected}


def fetch_legs(cost_only, leg_plan):
    todo = {osi: info for osi, info in leg_plan.items() if not os.path.exists(os.path.join(LEGS_DIR, f"{osi.strip()}.parquet"))}
    log.info(f"per-leg full-life: {len(leg_plan)} legs in plan, {len(todo)} not yet cached")
    for osi, info in sorted(todo.items()):
        life_start, life_end = _leg_window(info)
        data = guarded_get_range(
            cost_only, dataset='OPRA.PILLAR', schema='cbbo-1m', stype_in='raw_symbol',
            symbols=[osi.strip()], start=life_start, end=life_end,
        )
        if data is None:
            continue
        df = data.to_df()
        if not cost_only:
            df.to_parquet(os.path.join(LEGS_DIR, f"{osi.strip()}.parquet"), index=False)
            log.info(f"leg {osi.strip()}: {len(df)} bars -> {LEGS_DIR}/{osi.strip()}.parquet")


def split_batch_frame(df, todo_osis):
    """Split one batched cbbo-1m DataFrame (multiple raw_symbol legs) into {osi: sub_df}.

    Databento returns the requested raw_symbol back in the 'symbol' column when stype_in is
    'raw_symbol' and more than one symbol was requested. Legs present in `todo_osis` but absent
    from the returned frame are reported (data gap / no quotes), never silently dropped.
    """
    by_osi = {}
    if 'symbol' not in df.columns:
        # A single-leg batch (group of 1) can come back without a symbol column on some client
        # versions; there is only one possible owner for every row in that case.
        if len(todo_osis) == 1:
            return {todo_osis[0]: df}
        log.error(f"batched frame has no 'symbol' column and {len(todo_osis)} legs were requested -- cannot split")
        return by_osi
    stripped = {osi.strip(): osi for osi in todo_osis}
    for sym, sub in df.groupby('symbol'):
        osi = stripped.get(str(sym).strip())
        if osi is None:
            log.warning(f"batched frame returned unrequested symbol {sym!r}, ignoring")
            continue
        by_osi[osi] = sub
    missing = [osi for osi in todo_osis if osi not in by_osi]
    for osi in missing:
        log.warning(f"batched frame: no rows returned for leg {osi.strip()} (data gap, not cached)")
    return by_osi


def fetch_legs_batched(cost_only, leg_plan):
    """Same result as fetch_legs (one parquet per OSI under LEGS_DIR, same spend ledger via
    guarded_get_range), but ONE timeseries.get_range per entry Monday (grouped by the leg's
    (first_seen, expiry) -- constant per Monday by construction, so every leg in a group shares
    the exact same life window) instead of one call per leg. Resumable: a leg already cached is
    dropped from its group before the call; a group with nothing left to fetch is skipped with no
    get_cost call at all (unlike fetch_legs, which still iterates leg-by-leg, this never even
    prices an already-cached leg).
    """
    todo = {osi: info for osi, info in leg_plan.items() if not os.path.exists(os.path.join(LEGS_DIR, f"{osi.strip()}.parquet"))}
    log.info(f"batched per-leg full-life: {len(leg_plan)} legs in plan, {len(todo)} not yet cached")

    groups = {}
    for osi, info in todo.items():
        key = (info['first_seen'], info['expiry'])
        groups.setdefault(key, []).append(osi)

    log.info(f"batched: {len(todo)} legs remaining grouped into {len(groups)} entry-Monday batches")
    for (first_seen, expiry), osis in sorted(groups.items()):
        life_start, life_end = _leg_window({'first_seen': first_seen, 'expiry': expiry})
        symbols = [osi.strip() for osi in sorted(osis)]
        data = guarded_get_range(
            cost_only, dataset='OPRA.PILLAR', schema='cbbo-1m', stype_in='raw_symbol',
            symbols=symbols, start=life_start, end=life_end,
        )
        if data is None:
            continue
        df = data.to_df()
        if cost_only:
            continue
        by_osi = split_batch_frame(df, osis)
        for osi, sub in by_osi.items():
            sub.to_parquet(os.path.join(LEGS_DIR, f"{osi.strip()}.parquet"), index=False)
        log.info(f"batch {first_seen} (expiry {expiry}): {len(by_osi)}/{len(osis)} legs written -> {LEGS_DIR}/")


def write_report(spy_df, mondays_df, leg_plan):
    n_gap = int((spy_df['source'] == 'GAP').sum()) if spy_df is not None and len(spy_df) else 0
    n_legs_cached = len([f for f in os.listdir(LEGS_DIR) if f.endswith('.parquet')])
    lines = [
        "# FETCH_DBN — PREREG_1567 v3 (Amendment 3) fetch stage report",
        "",
        f"Generated: {dt.datetime.now(UTC).isoformat()}",
        "",
        "## Spend",
        f"Total charged: ${SPEND['total_usd']:.4f} of ${SPEND_CAP_USD} cap "
        f"({len(SPEND['purchases'])} purchases logged, {len(SPEND['skipped'])} skipped at the cap).",
        "",
        "## SPY underlying",
        f"{len(spy_df) if spy_df is not None else 0} sessions written to opt_cache/dbn/spy_prices.parquet. "
        f"{n_gap} sessions ({SPY_EQUITY_GAP_START}..{ALPACA_EQUITY_START - dt.timedelta(days=1)}) are an UNFILLABLE "
        "GAP: Databento's equities datasets (ARCX.PILLAR, XNAS.ITCH, DBEQ.BASIC, EQUS.*) all start "
        "2018-05-01 or later and Alpaca's stock history starts 2016-01-04 (verified empirically, "
        "empty response before that date). 2016-01-04..2023-12-31 sourced from Alpaca minute bars "
        "(10:00 / 15:59 ET); 2024-01-02..2026-09-25 reused from the existing Alpaca cache built by "
        "fetch_options.py (no re-pull, no re-charge).",
        "",
        "## Monday ladders",
        f"{len(mondays_df) if mondays_df is not None else 0} (entry_date, symbol) rows written to "
        "opt_cache/dbn/mondays.parquet.",
        "",
        "## Per-leg full life",
        f"{len(leg_plan)} unique OSIs in the superset plan (0.20-/0.30-delta strikes ± "
        f"{SUPERSET_BAND} strikes, and each one's $10-below partner); {n_legs_cached} cached under "
        "opt_cache/dbn/legs/.",
        "",
        "## Gaps / caveats",
        "* Entry Mondays inside the SPY-price gap (2013-04-08..2015-12-31) have spot_10=NaN; the "
        "  Monday-ladder fetch for those weeks pulls the OPTION side only (definition + cbbo-1m are "
        "  available from 2013-04-01) but cannot select the delta-targeted superset without a spot "
        "  price -- resolving that (e.g. put-call parity from the same chain) is BUILD-stage work, "
        "  not fetched here.",
        "* A cycle/Monday is only in mondays.parquet if both its definition and cbbo-1m pulls "
        "  succeeded and returned a non-empty 10:00 bar; anything else is logged as WARNING/ERROR "
        "  in fetch_dbn.log, never silently dropped.",
        "* This run may have stopped partway through the plan if the $150 cap or the step budget "
        "  was reached first; rerun the same command to resume (already-cached Mondays/legs are "
        "  skipped).",
    ]
    with open(os.path.join(HERE, 'FETCH_DBN.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.info("wrote FETCH_DBN.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-only', action='store_true', help='estimate cost only, no purchases')
    ap.add_argument('--stage', choices=['spy', 'mondays', 'legs', 'all'], default='all')
    ap.add_argument('--batched', action='store_true',
                     help='per-leg full-life stage: one get_range call per entry Monday (all its '
                          'still-uncached legs as the symbols list) instead of one call per leg')
    ap.add_argument('--start', default='2016-01-04',
                     help='entry-Monday range start (ISO date), inclusive -- bounds which Mondays '
                          "mondays.parquet's rows are rebuilt into leg_plan for the legs/all "
                          'stages. Default = the 2016-2023 extension backlog (excludes the '
                          '2013-2015 SPY-price-gap Mondays).')
    ap.add_argument('--end', default='2023-12-25',
                     help='entry-Monday range end (ISO date), inclusive.')
    args = ap.parse_args()
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    log.info(f"=== fetch_dbn.py start (stage={args.stage}, cost_only={args.cost_only}, "
             f"batched={args.batched}, range=[{start_date}..{end_date}], "
             f"running spend ${SPEND['total_usd']:.4f}) ===")

    spy_df = None
    if args.stage in ('spy', 'all'):
        spy_df = fetch_spy_prices(args.cost_only)
    elif os.path.exists(SPY_PATH):
        spy_df = pd.read_parquet(SPY_PATH)

    mondays_df, leg_plan = None, {}
    if args.stage in ('mondays', 'all'):
        mondays_df, leg_plan = fetch_mondays(args.cost_only, spy_df)
    elif os.path.exists(MONDAYS_PATH):
        mondays_df = pd.read_parquet(MONDAYS_PATH)

    if args.stage in ('legs', 'all'):
        rebuilt, n_mondays = rebuild_leg_plan_from_mondays(mondays_df, start_date, end_date)
        for osi, info in rebuilt.items():
            leg_plan.setdefault(osi, info)
        print_dry_plan(leg_plan, start_date, end_date, n_mondays)
        if args.cost_only:
            log.info("cost-only: stopping before any per-leg cost check or purchase "
                      "(see DRY PLAN line above) -- no metadata.get_cost call made")
        elif leg_plan:
            if args.batched:
                fetch_legs_batched(args.cost_only, leg_plan)
            else:
                fetch_legs(args.cost_only, leg_plan)

    write_report(spy_df, mondays_df, leg_plan)
    log.info(f"=== fetch_dbn.py done (total spend ${SPEND['total_usd']:.4f}) ===")


if __name__ == '__main__':
    main()
