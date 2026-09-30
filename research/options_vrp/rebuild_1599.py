#!/usr/bin/env python3
"""Independent rebuild of PREREG_1567 v3 (Amendment 3), cells 1,599-1,606.

Built from research/options_vrp/PREREG_1567.md prose ONLY (the base spec + Amendment 2,
2a, 3 text). Does not read cell_1599.py, its outputs, or the v1/v2 cell scripts.

Ladder: weekly SPY bull put credit spread, sold Monday 10:00 ET, expiry nearest 45 DTE
(38-52 window). Grid: delta in {0.20, 0.30}, width $10, management in {A, B}, gate in
{none, IV>=15%} = 8 cells. Sizing: contracts = floor((B/6) / ((W-credit)*100)), B=$6,500,
equity fixed $65,000.

Data: Databento OPRA.PILLAR cbbo-1m (consolidated NBBO, 1-min bars), parent symbol
SPY.OPT. Entry fill = sold leg BID, bought leg ASK of the 10:00 bar (Amendment 3). Cycle
VOID only if the entry 10:00 bar lacks a two-sided quote for either leg. SPY spot: cached
Alpaca parquet (2024-01+) for the PANEL; for the EXTENSION (2013-2023), Alpaca equities
back to 2016-01-04, and put-call parity from the same options chain for 2013-04-08 to
2015-12-31 (neither Alpaca nor Databento equities cover that gap -- disclosed).

Usage:
  python3 rebuild_1599.py --stage smoke               # one Monday, sanity check
  python3 rebuild_1599.py --stage panel                # TRAIN+VAL fetch+build, both deltas
  python3 rebuild_1599.py --stage select                # TRAIN selection + VAL report
  python3 rebuild_1599.py --stage extension --delta 0.X # EXTENSION fetch+build, one delta
  python3 rebuild_1599.py --stage report                # final REBUILD_1599.md + csvs
  python3 rebuild_1599.py --stage all                    # chain everything, resumable
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, 'opt_cache', 'dbn')
LEGS_DIR = os.path.join(CACHE, 'legs')
SNAP_DIR = os.path.join(CACHE, 'snapshots')
os.makedirs(LEGS_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
SPEND_PATH = os.path.join(CACHE, 'spend.json')

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(os.path.join(HERE, 'rebuild_1599.log')), logging.StreamHandler()])
log = logging.getLogger('rebuild_1599')

load_dotenv(os.path.join(ROOT, '.env'))
import databento as db
_KEY = os.environ.get('DATABENTO_API_KEY')
if not _KEY:
    log.error("DATABENTO_API_KEY missing -- cannot fetch")
    raise SystemExit(1)
CLIENT = db.Historical(_KEY)

ET = ZoneInfo('America/New_York')
UTC = dt.timezone.utc

SPEND_CAP = 150.0
_spend_lock = threading.Lock()


def _load_spend():
    if os.path.exists(SPEND_PATH):
        with open(SPEND_PATH) as f:
            return json.load(f)
    return {'total_usd': 0.0, 'purchases': [], 'skipped': []}


def _save_spend(s):
    tmp = SPEND_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=2, default=str)
    os.replace(tmp, SPEND_PATH)


import re
_AVAIL_END_RE = re.compile(r"available up to '([\d\- :+]+)'")


def guarded_range(cost_only=False, **kw):
    """metadata.get_cost then timeseries.get_range, spend-capped and logged. Thread-safe.
    On a 'data_end_after_available_end' error, clips `end` to the reported available end and
    retries once (handles the live data-lag boundary automatically instead of guessing it)."""
    label = f"{kw.get('dataset')}/{kw.get('schema')} {kw.get('symbols')} {kw.get('start')}..{kw.get('end')}"
    with _spend_lock:
        state = _load_spend()
        try:
            cost = CLIENT.metadata.get_cost(**kw)
        except Exception as e:
            m = _AVAIL_END_RE.search(str(e))
            if m and 'end' in kw and pd.Timestamp(kw['start']) < pd.Timestamp(m.group(1).strip()):
                new_end = pd.Timestamp(m.group(1).strip()).isoformat()
                kw = dict(kw, end=new_end)
                label = f"{kw.get('dataset')}/{kw.get('schema')} {kw.get('symbols')} {kw.get('start')}..{kw.get('end')} (clipped)"
                try:
                    cost = CLIENT.metadata.get_cost(**kw)
                except Exception as e2:
                    log.error(f"get_cost FAILED after clip {label}: {e2}")
                    return None
            else:
                log.error(f"get_cost FAILED {label}: {e}")
                return None
        projected = state['total_usd'] + cost
        if projected > SPEND_CAP:
            log.error(f"SPEND CAP: ${state['total_usd']:.4f}+${cost:.4f}=${projected:.4f} > ${SPEND_CAP} SKIP {label}")
            state['skipped'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label})
            _save_spend(state)
            return None
        if cost_only:
            state['total_usd'] = projected
            state['purchases'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label, 'dry_run': True})
            _save_spend(state)
            return None
    try:
        data = CLIENT.timeseries.get_range(**kw)
    except Exception as e:
        log.error(f"get_range FAILED {label} (cost ${cost:.4f} not charged): {e}")
        return None
    with _spend_lock:
        state = _load_spend()
        state['total_usd'] = state['total_usd'] + cost
        state['purchases'].append({'ts': dt.datetime.now(UTC).isoformat(), 'cost_usd': cost, 'label': label})
        _save_spend(state)
    return data


# --------------------------------------------------------------------------------------
# OSI parsing / BS math
# --------------------------------------------------------------------------------------
def parse_osi(raw_symbol):
    s = raw_symbol
    root = s[0:6].strip()
    yy, mm, dd = int(s[6:8]), int(s[8:10]), int(s[10:12])
    right = s[12]
    strike = int(s[13:21]) / 1000.0
    expiry = dt.date(2000 + yy, mm, dd)
    return root, expiry, right, strike


R_RATE = 0.045
Q_DIV = 0.013


def bs_put_price(S, K, T, sigma, r=R_RATE, q=Q_DIV):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_put_delta(S, K, T, sigma, r=R_RATE, q=Q_DIV):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * norm.cdf(-d1)


def implied_vol_put(mid, S, K, T):
    if mid <= 0 or T <= 0:
        return None
    intrinsic = max(K - S, 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    try:
        return brentq(lambda sig: bs_put_price(S, K, T, sig) - mid, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Calendars / dates
# --------------------------------------------------------------------------------------
def et_window_utc(day, hh, mm, span_min=2):
    """[start,end) UTC for hh:mm ET on `day` (handles DST via zoneinfo)."""
    start_et = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)
    end_et = start_et + dt.timedelta(minutes=span_min)
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def mondays_between(start, end):
    d = start
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    out = []
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=7)
    return out


def nearest_friday_expiry(monday, dte_lo=38, dte_hi=52, dte_target=45):
    """SPY weeklies expire Fridays (some Wed); we search candidate Fridays in [lo,hi] DTE."""
    best = None
    for delta in range(dte_lo, dte_hi + 1):
        cand = monday + dt.timedelta(days=delta)
        if cand.weekday() == 4:  # Friday
            if best is None or abs(delta - dte_target) < abs(best[1] - dte_target):
                best = (cand, delta)
    return best  # (expiry_date, dte) or None


# --------------------------------------------------------------------------------------
# SPY spot
# --------------------------------------------------------------------------------------
_spy_daily = None
_spy_minute = None


def load_spy_panel():
    global _spy_daily, _spy_minute
    _spy_daily = pd.read_parquet(os.path.join(HERE, 'opt_cache', 'spy_daily.parquet'))
    _spy_daily['day'] = pd.to_datetime(_spy_daily['day']).dt.date
    _spy_minute = pd.read_parquet(os.path.join(HERE, 'opt_cache', 'spy_minute.parquet'))
    _spy_minute['t'] = pd.to_datetime(_spy_minute['t'], utc=True)


def spy_spot_at(day, hh, mm):
    """SPY price at hh:mm ET on `day` from cached Alpaca minute bars (panel period)."""
    if _spy_minute is None:
        load_spy_panel()
    start_u, end_u = et_window_utc(day, hh, mm, span_min=1)
    row = _spy_minute[(_spy_minute['t'] >= start_u) & (_spy_minute['t'] < end_u)]
    if len(row):
        return float(row.iloc[0]['c'])
    return None


def spy_close(day):
    if _spy_daily is None:
        load_spy_panel()
    row = _spy_daily[_spy_daily['day'] == day]
    if len(row):
        return float(row.iloc[0]['c'])
    return None


# --------------------------------------------------------------------------------------
# Fetch: Monday chain snapshot (definitions + cbbo-1m, 10:00-10:02 ET)
# --------------------------------------------------------------------------------------
def fetch_monday_snapshot(monday, cost_only=False):
    """Return DataFrame [raw_symbol, expiry, right, strike, bid, ask] for all SPY puts
    quoted in the 10:00-10:02 ET window on `monday`. Cached as parquet, resumable."""
    path = os.path.join(SNAP_DIR, f"{monday.isoformat()}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    start_u, end_u = et_window_utc(monday, 10, 0, span_min=2)
    kw = dict(dataset='OPRA.PILLAR', schema='cbbo-1m', symbols=['SPY.OPT'], stype_in='parent',
              start=start_u.isoformat(), end=end_u.isoformat())
    store = guarded_range(cost_only=cost_only, **kw)
    if store is None:
        return None
    df = store.to_df()
    if df.empty:
        df.to_parquet(path)
        return df
    if 'symbol' not in df.columns:
        # fall back to a definition pull for the same window to map instrument_id->raw_symbol
        dkw = dict(dataset='OPRA.PILLAR', schema='definition', symbols=['SPY.OPT'], stype_in='parent',
                   start=start_u.isoformat(), end=end_u.isoformat())
        dstore = guarded_range(cost_only=cost_only, **dkw)
        if dstore is None:
            return None
        ddf = dstore.to_df()
        symmap = ddf.drop_duplicates('instrument_id').set_index('instrument_id')['raw_symbol']
        df['symbol'] = df['instrument_id'].map(symmap)
    df = df.dropna(subset=['symbol'])
    bidcol = 'bid_px_00' if 'bid_px_00' in df.columns else 'bid_px'
    askcol = 'ask_px_00' if 'ask_px_00' in df.columns else 'ask_px'
    parsed = df['symbol'].apply(parse_osi)
    df['root'] = parsed.apply(lambda t: t[0])
    df['expiry'] = parsed.apply(lambda t: t[1])
    df['right'] = parsed.apply(lambda t: t[2])
    df['strike'] = parsed.apply(lambda t: t[3])
    df = df[df['root'] == 'SPY']
    g = df.groupby('symbol').last().reset_index()
    out = g[['symbol', 'expiry', 'strike', 'right']].copy()
    out['bid'] = g[bidcol].astype(float)
    out['ask'] = g[askcol].astype(float)
    # Databento prices may be fixed-point 1e-9; detect by magnitude
    if out['ask'].median() > 1000:
        out['bid'] /= 1e9
        out['ask'] /= 1e9
    out = out[(out['bid'] > 0) & (out['ask'] > 0) & (out['ask'] >= out['bid'])]
    out.to_parquet(path)
    return out


def fetch_leg_life(raw_symbol, entry_day, life_end_day, cost_only=False):
    """Full-life cbbo-1m for one contract, cached parquet. Returns df[t,bid,ask] (ET-local ts).

    PATCH (2026-09-30, agent, cell 1,599 verdict task): fetch_dbn.py (the builder) caches the
    identical full-life leg pull under a DIFFERENT key -- bare OSI, no entry-day suffix, e.g.
    'opt_cache/dbn/legs/SPY   260925P00756000.parquet' vs this function's own
    'SPY___260925P00756000_2026-08-10.parquet'. Before this patch, every leg not already fetched
    by THIS script's own prior partial run (1,427 of 20,872) would fall through to a live
    guarded_range() call and re-purchase data already paid for by fetch_dbn.py ($53.66 sunk) --
    forbidden by this task's no-new-spend instruction. Read-only fallback added; no fetch
    behaviour changed for genuinely uncached legs. Caveat (disclosed, not fixed -- would need a
    re-fetch to fix): fetch_dbn.py's leg parquet is saved with to_parquet(index=False), which
    drops the DatetimeIndex (ts_recv) that databento's to_df() sets; only the coarser, sometimes
    stale/NaT 'ts_event' column survives (measured: 7-95% non-null across a sample of 8 cached
    legs). Any leg served from this fallback inherits that same staleness for bar_at() minute
    lookups, so builder/rebuild agreement on management-A exit timing is NOT fully independent
    for those legs -- entry/strike-selection (rebuild's own 10:00 snapshot fetch, independently
    sourced) and management-B settlement (SPY close, independently sourced) are unaffected.
    """
    safe = raw_symbol.replace(' ', '_')
    path = os.path.join(LEGS_DIR, f"{safe}_{entry_day.isoformat()}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    shared_path = os.path.join(LEGS_DIR, f"{raw_symbol.strip()}.parquet")
    if os.path.exists(shared_path):
        raw = pd.read_parquet(shared_path)
        if raw.empty:
            out = pd.DataFrame(columns=['t', 'bid', 'ask'])
        else:
            bidcol = 'bid_px_00' if 'bid_px_00' in raw.columns else 'bid_px'
            askcol = 'ask_px_00' if 'ask_px_00' in raw.columns else 'ask_px'
            ts = pd.to_datetime(raw['ts_event'], utc=True)
            out = pd.DataFrame({'t': ts, 'bid': raw[bidcol].astype(float).values,
                                 'ask': raw[askcol].astype(float).values})
            if out['ask'].median() > 1000:
                out['bid'] /= 1e9
                out['ask'] /= 1e9
            out = out.dropna(subset=['t'])
            out = out[(out['bid'] > 0) & (out['ask'] > 0) & (out['ask'] >= out['bid'])]
        out.to_parquet(path)
        return out
    start_u = dt.datetime(entry_day.year, entry_day.month, entry_day.day, tzinfo=ET).astimezone(UTC)
    end_u = (dt.datetime(life_end_day.year, life_end_day.month, life_end_day.day, tzinfo=ET)
             + dt.timedelta(days=1)).astimezone(UTC)
    avail_end = dt.datetime.now(UTC) - dt.timedelta(days=2)
    if end_u > avail_end:
        end_u = avail_end
    if start_u >= end_u:
        log.warning(f"{raw_symbol}: entry {entry_day} not yet inside the available data window -- skipping (PENDING)")
        pd.DataFrame(columns=['t', 'bid', 'ask']).to_parquet(path)
        return pd.read_parquet(path)
    kw = dict(dataset='OPRA.PILLAR', schema='cbbo-1m', symbols=[raw_symbol], stype_in='raw_symbol',
              start=start_u.isoformat(), end=end_u.isoformat())
    store = guarded_range(cost_only=cost_only, **kw)
    if store is None:
        return None
    df = store.to_df()
    if df.empty:
        pd.DataFrame(columns=['t', 'bid', 'ask']).to_parquet(path)
        return pd.read_parquet(path)
    bidcol = 'bid_px_00' if 'bid_px_00' in df.columns else 'bid_px'
    askcol = 'ask_px_00' if 'ask_px_00' in df.columns else 'ask_px'
    ts = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['ts_event'], utc=True)
    out = pd.DataFrame({'t': pd.to_datetime(ts, utc=True), 'bid': df[bidcol].astype(float).values,
                         'ask': df[askcol].astype(float).values})
    if out['ask'].median() > 1000:
        out['bid'] /= 1e9
        out['ask'] /= 1e9
    out = out[(out['bid'] > 0) & (out['ask'] > 0) & (out['ask'] >= out['bid'])]
    out.to_parquet(path)
    return out


def parity_spot(snap, expiry, dte, r=R_RATE, q=Q_DIV):
    """Put-call parity spot estimate for entry Mondays where no equity source exists
    (2013-04-08..2015-12-31, disclosed data gap): S = (C-P)*e^(qT) + K*e^((q-r)T).
    Uses the strike whose call/put mids are closest (nearest ATM). Returns None if no
    common strike has both a call and a put quote."""
    T = dte / 365.0
    ch = snap[snap['expiry'] == expiry].copy()
    calls = ch[ch['right'] == 'C'].set_index('strike')
    puts = ch[ch['right'] == 'P'].set_index('strike')
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return None
    cmid = (calls.loc[common, 'bid'] + calls.loc[common, 'ask']) / 2.0
    pmid = (puts.loc[common, 'bid'] + puts.loc[common, 'ask']) / 2.0
    k = (cmid - pmid).abs().idxmin()
    S = (cmid[k] - pmid[k]) * math.exp(q * T) + k * math.exp((q - r) * T)
    return float(S)


def bar_at(df, day, hh, mm):
    if df is None or df.empty:
        return None
    start_u, end_u = et_window_utc(day, hh, mm, span_min=1)
    row = df[(df['t'] >= start_u) & (df['t'] < end_u)]
    if len(row):
        r = row.iloc[0]
        return float(r['bid']), float(r['ask'])
    return None


# --------------------------------------------------------------------------------------
# Cycle construction (one Monday, one delta target) -> row dict or None(skip)/VOID row
# --------------------------------------------------------------------------------------
FEE_PER_CONTRACT = 0.03  # regulatory fee, both legs, both open+close
EQUITY = 65000.0
B = 0.10 * EQUITY
WIDTH = 10.0


def build_cycle(monday, delta_target, snap, spot10, trading_days):
    exp_info = nearest_friday_expiry(monday)
    if exp_info is None:
        return {'status': 'NO_EXPIRY', 'entry_day': monday, 'delta_target': delta_target}
    expiry, dte = exp_info
    T = dte / 365.0
    chain = snap[(snap['expiry'] == expiry) & (snap['right'] == 'P')].copy()
    if chain.empty or spot10 is None:
        return {'status': 'VOID', 'entry_day': monday, 'delta_target': delta_target, 'reason': 'no_chain_or_spot'}
    chain['mid'] = (chain['bid'] + chain['ask']) / 2.0
    ivs, deltas = [], []
    for _, row in chain.iterrows():
        iv = implied_vol_put(row['mid'], spot10, row['strike'], T)
        if iv is None:
            ivs.append(np.nan); deltas.append(np.nan); continue
        ivs.append(iv)
        deltas.append(abs(bs_put_delta(spot10, row['strike'], T, iv)))
    chain['iv'] = ivs
    chain['abs_delta'] = deltas
    valid = chain.dropna(subset=['abs_delta'])
    if valid.empty:
        return {'status': 'VOID', 'entry_day': monday, 'delta_target': delta_target, 'reason': 'no_iv_solve'}
    atm_row = valid.iloc[(valid['strike'] - spot10).abs().argsort().iloc[0]]
    atm_iv = float(atm_row['iv'])
    short_row = valid.iloc[(valid['abs_delta'] - delta_target).abs().argsort().iloc[0]]
    short_strike = float(short_row['strike'])
    target_long_strike = short_strike - WIDTH
    long_candidates = chain.iloc[(chain['strike'] - target_long_strike).abs().argsort()]
    long_row = long_candidates.iloc[0]
    long_strike = float(long_row['strike'])
    if short_row['bid'] <= 0 or short_row['ask'] <= 0 or long_row['bid'] <= 0 or long_row['ask'] <= 0:
        return {'status': 'VOID', 'entry_day': monday, 'delta_target': delta_target, 'reason': 'no_two_sided_quote'}
    credit = short_row['bid'] - long_row['ask']  # sell at bid, buy at ask
    if credit <= 0:
        return {'status': 'VOID', 'entry_day': monday, 'delta_target': delta_target, 'reason': 'negative_credit'}
    risk_per_contract = (WIDTH - credit) * 100.0
    contracts = math.floor((B / 6.0) / risk_per_contract) if risk_per_contract > 0 else 0
    if contracts <= 0:
        return {'status': 'ZERO_SIZE', 'entry_day': monday, 'delta_target': delta_target,
                'expiry': expiry, 'short_strike': short_strike, 'long_strike': long_strike, 'credit': credit,
                'atm_iv': atm_iv}
    entry_fee = 2 * FEE_PER_CONTRACT * contracts
    return {
        'status': 'OPEN', 'entry_day': monday, 'delta_target': delta_target, 'expiry': expiry, 'dte': dte,
        'short_symbol': short_row['symbol'], 'long_symbol': long_row['symbol'],
        'short_strike': short_strike, 'long_strike': long_strike, 'credit': credit,
        'contracts': contracts, 'risk_per_contract': risk_per_contract, 'entry_fee': entry_fee,
        'atm_iv': atm_iv, 'spot10': spot10,
    }


def observed_days(short_life, long_life, lo, hi):
    """Calendar days actually observed in BOTH legs' quote data (has a 15:59 bar), between
    (lo, hi] -- avoids depending on an external market-holiday calendar."""
    def days_of(df):
        if df is None or df.empty:
            return set()
        local = df['t'].dt.tz_convert(ET)
        return set(local.dt.date[(local.dt.hour == 15) & (local.dt.minute == 59)])
    both = days_of(short_life) & days_of(long_life)
    return sorted(d for d in both if lo < d <= hi)


def next_observed_day(short_life, long_life, after_day, hi):
    """Next day after `after_day` (<= hi) where BOTH legs have a 10:00 bar."""
    def days_of(df):
        if df is None or df.empty:
            return set()
        local = df['t'].dt.tz_convert(ET)
        return set(local.dt.date[(local.dt.hour == 10) & (local.dt.minute == 0)])
    both = sorted(d for d in (days_of(short_life) & days_of(long_life)) if after_day < d <= hi)
    return both[0] if both else None


def run_management(cyc, mgmt, spy_close_fn, short_life=None, long_life=None):
    """Given an OPEN cycle dict, simulate management A or B. `short_life`/`long_life` may be
    pre-fetched (shared across the two management variants of the same cycle) or left None
    to fetch here. Adds (on a COPY): exit_day, exit_reason, exit_credit_cost, pnl, hold_days."""
    cyc = dict(cyc)
    entry_day, expiry = cyc['entry_day'], cyc['expiry']
    life_end = min(expiry, entry_day + dt.timedelta(days=60))
    if short_life is None:
        short_life = fetch_leg_life(cyc['short_symbol'], entry_day, life_end)
    if long_life is None:
        long_life = fetch_leg_life(cyc['long_symbol'], entry_day, life_end)
    cyc['short_life_rows'] = 0 if short_life is None else len(short_life)
    cyc['long_life_rows'] = 0 if long_life is None else len(long_life)
    days = observed_days(short_life, long_life, entry_day, expiry)
    contracts, credit0 = cyc['contracts'], cyc['credit']
    if mgmt == 'B':
        s_close = spy_close_fn(expiry)
        if s_close is None:
            cyc.update(status='VOID', reason='no_spot_at_expiry')
            return cyc
        short_intrinsic = max(cyc['short_strike'] - s_close, 0.0)
        long_intrinsic = max(cyc['long_strike'] - s_close, 0.0)
        exit_cost = short_intrinsic - long_intrinsic
        exit_fee = 0.0  # settlement, no commission modeled
        pnl_per_contract = (credit0 - exit_cost) * 100.0 - cyc['entry_fee'] / contracts * 0
        cyc.update(status='CLOSED', exit_day=expiry, exit_reason='EXPIRY_SETTLE',
                    exit_credit_cost=exit_cost, exit_fee=exit_fee)
    else:  # management A
        exit_day, exit_reason, exit_cost = None, None, None
        for d in days:
            if d == expiry:
                s_close = spy_close_fn(expiry)
                if s_close is not None:
                    exit_cost = max(cyc['short_strike'] - s_close, 0.0) - max(cyc['long_strike'] - s_close, 0.0)
                    exit_day, exit_reason = expiry, 'EXPIRY_SETTLE'
                break
            sb = bar_at(short_life, d, 15, 59)
            lb = bar_at(long_life, d, 15, 59)
            if sb is None or lb is None:
                continue
            mark_cost = ((sb[0] + sb[1]) / 2.0) - ((lb[0] + lb[1]) / 2.0)  # mid-mid mark
            dte_left = (expiry - d).days
            if mark_cost <= 0.5 * credit0:
                nd = next_observed_day(short_life, long_life, d, expiry)
                if nd is None:
                    continue
                sn = bar_at(short_life, nd, 10, 0)
                ln = bar_at(long_life, nd, 10, 0)
                if sn and ln:
                    exit_cost = sn[1] - ln[0]  # buy back short at ASK, sell long at BID
                    exit_day, exit_reason = nd, 'PROFIT_50PCT'
                    break
            if mark_cost >= 2.0 * credit0:
                nd = next_observed_day(short_life, long_life, d, expiry)
                if nd is None:
                    continue
                sn = bar_at(short_life, nd, 10, 0)
                ln = bar_at(long_life, nd, 10, 0)
                if sn and ln:
                    exit_cost = sn[1] - ln[0]
                    exit_day, exit_reason = nd, 'STOP_2X'
                    break
            if dte_left <= 21:
                nd = next_observed_day(short_life, long_life, d, expiry)
                if nd is None:
                    continue
                sn = bar_at(short_life, nd, 10, 0)
                ln = bar_at(long_life, nd, 10, 0)
                if sn and ln:
                    exit_cost = sn[1] - ln[0]
                    exit_day, exit_reason = nd, 'DTE21'
                    break
        if exit_cost is None:
            s_close = spy_close_fn(expiry)
            if s_close is None:
                cyc.update(status='VOID', reason='no_exit_found')
                return cyc
            exit_cost = max(cyc['short_strike'] - s_close, 0.0) - max(cyc['long_strike'] - s_close, 0.0)
            exit_day, exit_reason = expiry, 'EXPIRY_SETTLE_FALLBACK'
        cyc.update(status='CLOSED', exit_day=exit_day, exit_reason=exit_reason, exit_credit_cost=exit_cost)
    exit_fee = 2 * FEE_PER_CONTRACT * contracts if cyc['exit_reason'] != 'EXPIRY_SETTLE' else 0.0
    pnl = ((cyc['credit'] - cyc['exit_credit_cost']) * 100.0 * contracts) - cyc['entry_fee'] - exit_fee
    cyc['pnl'] = pnl
    cyc['hold_days'] = (cyc['exit_day'] - cyc['entry_day']).days
    return cyc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                     choices=['smoke', 'panel', 'select', 'extension', 'report', 'all'])
    ap.add_argument('--cost-only', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--delta', type=float, default=None)
    ap.add_argument('--max-mondays', type=int, default=None)
    args = ap.parse_args()
    from rebuild_1599_run import dispatch
    dispatch(args)
