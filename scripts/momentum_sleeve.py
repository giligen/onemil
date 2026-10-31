#!/usr/bin/env python3
"""Momentum sleeve runner -- weekly risk-adjusted 12-1 momentum, top 20, equal weight, PAPER only.

Every first trading day of the week the sleeve resets ALL 20 names to 1/20 of its own equity
(sleeve cash + sum qty * last price). Selection lives in trading/momentum_sleeve.py (pure, parity-tested
against research/momentum_weekly/recon/A_holdings.csv). This file does the I/O:

  fetch adjusted daily bars (Alpaca, adjustment=ALL) -> parquet cache -> select -> delta orders ->
  sells (fractional qty, market DAY) polled to fill -> buys (notional, market DAY) -> state/ledger/weekly
  CSVs -> ONE [MOM] Telegram summary.

Default is DRY-RUN (nothing submitted). ``--submit`` refuses unless the account is PAPER and the session is
a regular trading day within 09:31-15:30 ET. Keys: ALPACA_MOM_API_KEY / ALPACA_MOM_API_SECRET (dedicated paper account).
State: data/momentum_sleeve/state.json. Ledgers: logs/momentum_sleeve_ledger.csv, logs/momentum_sleeve_weekly.csv.

Usage:
  python3 scripts/momentum_sleeve.py [--asof YYYY-MM-DD] [--n 20] [--equity-start 20000]
                                     [--skip-fetch] [--force] [--submit]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import logging
import os
import re
import sys
import time
import urllib.request
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import Config                                           # noqa: E402
from monitoring.cron_guard import attach_telegram_error_handler, run_guarded  # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError   # noqa: E402
from notifications.telegram_notifier import TelegramNotifier         # noqa: E402
from tom_sleeve import assert_paper_account, _to_date                # noqa: E402  (shared paper guard)
from trading import momentum_sleeve as ms                            # noqa: E402
from trading import momentum_live as ml                              # noqa: E402

logger = logging.getLogger('momentum_sleeve')

ET = ZoneInfo('US/Eastern')
DATA_DIR = os.path.join(ROOT, 'data', 'momentum_sleeve')
STATE_PATH = os.path.join(DATA_DIR, 'state.json')
LEDGER_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_ledger.csv')
WEEKLY_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_weekly.csv')
LEDGER_FIELDS = ['date', 'symbol', 'side', 'qty', 'avg_price', 'notional', 'official_open',
                 'slip_bps_vs_open', 'client_order_id', 'size_pct']
WEEKLY_FIELDS = ['date', 'equity', 'cash', 'n_names', 'turnover_usd', 'names_in', 'names_out', 'spy_close']
TG_PREFIX = '[MOM]'
SHADOW_GATE_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_shadow_gate.csv')
SHADOW_GATE_FIELDS = ['run_date', 'asof', 'vix', 'vix3m', 'ratio', 'percentile', 'gate_on', 'equity', 'scale']
CBOE_URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv'
GUARD_TOP = 40                # the guard line reports removed names that would have ranked in the top 40
LOOKBACK_DAYS = 420
BATCH_SIZE = 200
MIN_FREE_BYTES = 1 * 1024 ** 3
KEEP_CACHES = 2
SUBMIT_START, SUBMIT_END = dtime(9, 31), dtime(15, 30)
POLL_SECONDS, POLL_TIMEOUT = 2.0, 180.0
COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
INVALID_SYM_RE = re.compile(r'invalid symbol:\s*([^"\s]+)')
DD_KILL = 0.40
COMPLETENESS_MIN = 0.98      # --submit REFUSES when under this share of LIQUID names (ADV20 >= LIQUID_ADV) has an asof bar
LOST_MAX = 0.05              # ... or when more than this share of the requested symbols was LOST in the fetch
LIQUID_ADV = 200_000_000.0   # liquid = ADV20 dollar volume over the 20 sessions before asof


# --------------------------------------------------------------------------- calendar helpers

def trading_sessions(client: AlpacaClient, start: date, end: date) -> List[date]:
    """Sorted NYSE sessions in [start, end]. Raises AlpacaAPIError / RuntimeError on failure or empty."""
    cal = client.get_market_calendar(start, end)
    sessions = sorted(_to_date(d['date']) for d in cal)
    if not sessions:
        raise RuntimeError(f"calendar returned zero sessions for {start}..{end}")
    return sessions


def last_completed_session(sessions: List[date], now_et: datetime) -> date:
    """Last session whose close has passed: today only once it is after 16:30 ET, else the prior one."""
    today = now_et.date()
    done = [s for s in sessions if s < today or (s == today and now_et.time() >= dtime(16, 30))]
    if not done:
        raise RuntimeError("no completed session found in the calendar window")
    return done[-1]


def is_rebalance_session(today: date, week_sessions: List[date]) -> bool:
    """True iff ``today`` is the first trading session of its ISO week (Monday, or the next session)."""
    return bool(week_sessions) and today == min(week_sessions)


CATCHUP_LAST_WEEKDAY = 2      # Mon=0 .. Wed=2: a missed rotation is caught up Mon-Wed; Thu/Fri wait for Monday


def rotation_due(today: date, week_sessions: List[date], last_rebalance: Optional[str]) -> Tuple[bool, str]:
    """(due, reason) -- the weekly rotation rule with catch-up (Fable 2026-10-31, after the missed 10/26 rotation).

    DUE on a session iff the state's last completed rotation (``last_rebalance``, 'YYYY-MM-DD' or None) is not
    in today's ISO week AND today is Mon-Wed. On the first session of the week with last week's rotation done
    this is exactly the old first-session rule; a Tuesday/Wednesday after a crashed or refused Monday is a
    catch-up; Thursday/Friday never trade (the next Monday is the rotation)."""
    if today not in week_sessions:
        return False, "not a trading session"
    last = None
    if last_rebalance:
        try:
            last = date.fromisoformat(str(last_rebalance)[:10])
        except ValueError:
            logger.warning("momentum_sleeve: state last_rebalance %r unreadable -- treated as no rotation yet",
                           last_rebalance)
    week = today.isocalendar()[1]
    if last is not None and last.isocalendar()[:2] == today.isocalendar()[:2]:
        return False, f"rotation already done this week ({last}, week {week})"
    if today.weekday() > CATCHUP_LAST_WEEKDAY:
        return False, f"{today:%A}: a missed rotation (last {last}) waits for Monday"
    if is_rebalance_session(today, week_sessions):
        return True, f"first session of week {week}"
    return True, f"catch-up (last {last}, week {week})"


def check_submit_window(now_et: datetime, is_session: bool) -> None:
    """--submit guard: regular session day and 09:31-15:30 ET, else RuntimeError."""
    if not is_session:
        raise RuntimeError("--submit refused: today is not a regular trading session")
    if not (SUBMIT_START <= now_et.time() <= SUBMIT_END):
        raise RuntimeError(f"--submit refused: {now_et:%H:%M} ET is outside 09:31-15:30 ET")


# --------------------------------------------------------------------------- assets / bars

def fetch_assets(client: AlpacaClient) -> pd.DataFrame:
    """Active, tradable us_equity assets (symbol, name). Raises if the call fails (no silent empty)."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = client.get_assets_tolerant(req, timeout=90, timeout_retries=2, rate_limit_retries=5)
    df = pd.DataFrame([dict(symbol=a.symbol, name=a.name or '', tradable=bool(a.tradable)) for a in assets])
    if df.empty:
        raise RuntimeError("get_all_assets returned no active us_equity assets")
    return df[df['tradable']][['symbol', 'name']].reset_index(drop=True)


def bars_to_df(barset) -> pd.DataFrame:
    """BarSet -> flat DataFrame (COLS), float32 prices; empty frame if no data."""
    raw = getattr(barset, 'df', None)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLS)
    df = raw.reset_index().rename(columns={'timestamp': 'bar_date'})
    df['bar_date'] = pd.to_datetime(df['bar_date']).dt.tz_localize(None).dt.normalize()
    df = df[COLS].copy()
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = df[c].astype('float32')
    return df


def fetch_batch_resilient(client: AlpacaClient, batch: List[str], start: pd.Timestamp, end: pd.Timestamp,
                          max_strip: int = 250) -> Tuple[Optional[object], List[str], List[str]]:
    """Fetch one batch with adjustment=ALL. A symbol is dropped ONLY when the API names it invalid
    (one at a time, never the whole batch -- see feedback_fetch_completeness_gate). Returns
    (barset_or_None, remaining, removed)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    remaining, removed = list(batch), []
    for _ in range(max_strip):
        if not remaining:
            return None, remaining, removed
        req = StockBarsRequest(symbol_or_symbols=remaining, timeframe=TimeFrame.Day, start=start, end=end,
                               adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        try:
            barset = client._call_with_timeout(lambda r=req: client.data_client.get_stock_bars(r),
                                               'momentum_fetch_batch', timeout=300, timeout_retries=2,
                                               rate_limit_retries=6)
            return barset, remaining, removed
        except Exception as e:
            m = INVALID_SYM_RE.search(str(e))
            bad = m.group(1) if m else None
            if bad and bad in remaining:
                remaining.remove(bad)
                removed.append(bad)
                logger.warning("momentum fetch: API rejected %s as invalid -- stripped, retrying the other %d",
                               bad, len(remaining))
                continue
            raise
    logger.error("momentum fetch: exceeded %d invalid-symbol strips, giving up on %d symbols",
                 max_strip, len(remaining))
    return None, remaining, removed


def fetch_panel(client: AlpacaClient, symbols: List[str], asof: date) -> Tuple[pd.DataFrame, List[str]]:
    """Adjusted daily bars for ``symbols`` + SPY over the last LOOKBACK_DAYS up to ``asof``.

    Returns (panel, lost_symbols). CUSIP-shaped (digit-leading) symbols are pre-filtered and counted LOST.
    """
    syms = sorted(set(symbols) | {'SPY'})
    cusip = [s for s in syms if re.match(r'^[0-9]', s)]
    if cusip:
        logger.warning("momentum fetch: %d CUSIP-shaped symbols pre-filtered (LOST reason cusip_like), e.g. %s",
                       len(cusip), cusip[:3])
    syms = [s for s in syms if s not in set(cusip)]
    start = pd.Timestamp(asof) - pd.Timedelta(days=LOOKBACK_DAYS)
    end = pd.Timestamp(asof) + pd.Timedelta(days=1)
    frames, lost = [], list(cusip)
    batches = [syms[i:i + BATCH_SIZE] for i in range(0, len(syms), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"disk free {free / 1e9:.2f} GB below the 1 GB floor -- aborting fetch")
        try:
            barset, remaining, removed = fetch_batch_resilient(client, batch, start, end)
        except Exception as e:
            logger.error("momentum fetch: batch %d/%d failed (%s) -- %d symbols LOST", i + 1, len(batches), e,
                         len(batch))
            lost += batch
            continue
        lost += removed
        if barset is None:
            lost += remaining
            continue
        df = bars_to_df(barset)
        got = set(df['symbol'])
        lost += [s for s in remaining if s not in got]   # requested, valid, but no bars returned
        frames.append(df)
        if (i + 1) % 10 == 0 or i + 1 == len(batches):
            logger.info("momentum fetch: %d/%d batches, %d LOST so far", i + 1, len(batches), len(lost))
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    return panel, sorted(set(lost))


def liquid_symbols(panel: pd.DataFrame, asof: date) -> set:
    """Symbols with ADV20 >= LIQUID_ADV over exactly the 20 sessions BEFORE ``asof`` (close * volume mean).
    These always print a bar, so a missing asof bar on one of them means stale data."""
    if not len(panel):
        return set()
    sub = panel[pd.to_datetime(panel['bar_date']) < pd.Timestamp(asof)][['symbol', 'bar_date', 'close', 'volume']]
    sub = sub.sort_values(['symbol', 'bar_date'], kind='stable').drop_duplicates(['symbol', 'bar_date'], keep='last')
    sub = sub[sub.groupby('symbol', sort=False, observed=True).cumcount(ascending=False) < ms.ADV_WINDOW]
    agg = (sub['close'].astype('float64') * sub['volume'].astype('float64')).groupby(
        sub['symbol'], observed=True).agg(['mean', 'count'])
    return set(agg.index[(agg['count'] == ms.ADV_WINDOW) & (agg['mean'] >= LIQUID_ADV)])


def completeness_info(n_requested: int, panel: pd.DataFrame, lost: List[str], asof: date) -> Dict:
    """Completeness result the --submit gate reads: {asof, requested, with_asof_bar, ratio, lost, lost_share,
    n_liquid, liquid_with_asof, liquid_ratio}. ``ratio`` (all requested names) is informational only: ~10 % of
    the listings simply do not trade on a given day. The gate tests lost_share and liquid_ratio."""
    has_asof = (set(panel.loc[pd.to_datetime(panel['bar_date']) == pd.Timestamp(asof), 'symbol'])
                if len(panel) else set())
    liquid = liquid_symbols(panel, asof)
    lost = sorted(set(lost))
    return {'asof': str(asof), 'requested': int(n_requested), 'with_asof_bar': len(has_asof),
            'ratio': len(has_asof) / max(n_requested, 1), 'lost': lost,
            'lost_share': len(lost) / max(n_requested, 1), 'n_liquid': len(liquid),
            'liquid_with_asof': len(liquid & has_asof),
            'liquid_ratio': len(liquid & has_asof) / len(liquid) if liquid else 0.0}


def completeness_path(asof: date) -> str:
    """Completeness JSON path, beside the panel cache (``cache_file(asof)`` + '.completeness.json')."""
    return cache_file(asof) + '.completeness.json'


def write_completeness(asof: date, n_requested: int, panel: pd.DataFrame, lost: List[str]) -> Dict:
    """Persist the completeness result atomically beside the panel cache; returns it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    info = completeness_info(n_requested, panel, lost, asof)
    path = completeness_path(asof)
    with open(path + '.tmp', 'w') as f:
        json.dump(info, f)
    os.replace(path + '.tmp', path)
    return info


def read_completeness(asof: date) -> Optional[Dict]:
    """The persisted completeness result for ``asof``; None (WARNING) when missing or unreadable."""
    path = completeness_path(asof)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("momentum_sleeve: completeness file %s unreadable (%s)", path, e)
        return None


def completeness_refusal(info: Optional[Dict], asof: date, held, selected, missing_bar=()) -> Optional[str]:
    """Reason to REFUSE the rebalance, or None. Refuse when the result is missing / for another asof, or (a) the
    LOST share exceeds LOST_MAX, or (b) the asof-bar coverage of LIQUID names is under COMPLETENESS_MIN (stale data
    would enter the ranking), or (c) any HELD or newly selected symbol was LOST or has no asof bar."""
    if info is None:
        return f"completeness file missing for {asof}"
    if info.get('asof') != str(asof):
        return f"completeness file is for another asof ({info.get('asof')} != {asof})"
    if info['lost_share'] > LOST_MAX:
        return f"LOST share {info['lost_share']:.3f} > {LOST_MAX:.2f} ({len(info['lost'])}/{info['requested']})"
    if info['liquid_ratio'] < COMPLETENESS_MIN:
        return (f"liquid asof coverage {info['liquid_ratio']:.3f} < {COMPLETENESS_MIN:.2f} "
                f"({info['liquid_with_asof']}/{info['n_liquid']})")
    hit = sorted((set(held) | set(selected)) & (set(info.get('lost', [])) | set(missing_bar)))
    if hit:
        return f"held/selected symbols LOST or without an asof bar: {hit}"
    return None


def completeness_line(n_requested: int, panel: pd.DataFrame, lost: List[str], asof: date) -> str:
    """One-line completeness gate; ERROR when the LOST share or the liquid-name asof coverage fails the gate."""
    info = completeness_info(n_requested, panel, lost, asof)
    line = (f"COMPLETENESS: requested {n_requested} symbols, {panel['symbol'].nunique() if len(panel) else 0} "
            f"with bars, {info['with_asof_bar']} with an {asof} bar ({info['ratio']:.0%}), LOST {len(lost)} "
            f"({info['lost_share']:.1%}, max {LOST_MAX:.0%}), liquid {info['liquid_with_asof']}/{info['n_liquid']} "
            f"with an asof bar ({info['liquid_ratio']:.1%}, min {COMPLETENESS_MIN:.0%})")
    ok = info['lost_share'] <= LOST_MAX and info['liquid_ratio'] >= COMPLETENESS_MIN
    (logger.info if ok else logger.error)(line)
    return line


def cache_file(asof: date) -> str:
    """Parquet cache path for a signal date."""
    return os.path.join(DATA_DIR, f"daily_{asof:%Y%m%d}.parquet")


def prune_caches(keep: int = KEEP_CACHES) -> None:
    """Keep only the newest ``keep`` daily parquet files."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'daily_*.parquet')))
    for f in files[:-keep]:
        os.remove(f)
        if os.path.exists(f + '.completeness.json'):
            os.remove(f + '.completeness.json')
        logger.info("momentum_sleeve: pruned old cache %s", os.path.basename(f))


def write_cache_atomic(panel: pd.DataFrame, asof: date) -> str:
    """Atomic parquet write (tmp + rename) then prune."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = cache_file(asof)
    tmp = path + '.tmp'
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    prune_caches()
    return path


# --------------------------------------------------------------------------- state / ledgers

def load_state(path: str, equity_start: float) -> Dict:
    """Sleeve state {cash, positions{sym: qty}, peak_equity, last_rebalance}; fresh at ``equity_start``."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'cash': float(equity_start), 'positions': {}, 'peak_equity': float(equity_start),
            'last_rebalance': None}


def save_state(state: Dict, path: str) -> None:
    """Atomic JSON write."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_csv(path: str, fields: List[str], row: Dict) -> None:
    """Append one row, header first if the file is new. An existing file whose header lacks some of ``fields``
    (a column added later) is rewritten first with the new header, old rows keeping blanks (WARNING logged)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    if not new:
        with open(path, newline='') as f:
            old = list(csv.DictReader(f))
            f.seek(0)
            header = next(csv.reader(f), [])
        if header != fields:
            logger.warning("momentum_sleeve: migrating %s header %s -> %s", path, header, fields)
            tmp = path + '.tmp'
            with open(tmp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                w.writeheader()
                w.writerows(old)
            os.replace(tmp, path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def mode_paths(live: bool) -> Dict[str, str]:
    """State / ledger / weekly / gate file paths: the PAPER module globals, or the separate LIVE files.

    Read at call time so tests (and the paper default, byte-identical to before) see the patched globals."""
    if live:
        return {'state': ml.LIVE_STATE_PATH, 'ledger': ml.LIVE_LEDGER_PATH, 'weekly': ml.LIVE_WEEKLY_PATH,
                'gate': ml.LIVE_SHADOW_GATE_PATH}
    return {'state': STATE_PATH, 'ledger': LEDGER_PATH, 'weekly': WEEKLY_PATH, 'gate': SHADOW_GATE_PATH}


def assert_live_account(client: AlpacaClient) -> None:
    """LIVE guard (mirror of ``assert_paper_account``): refuse a paper client or a paper ('PA...') account.

    Logs at most 4 characters of the account number."""
    if getattr(client, 'is_paper', True):
        raise RuntimeError("--live requires AlpacaClient.is_paper False -- refusing to run")
    acct = client.trading_client.get_account()
    number = str(getattr(acct, 'account_number', '') or '')
    if not number or number.startswith('PA'):
        raise RuntimeError(f"--live account number {number[:4]!r}... looks like a paper account -- refusing to run")
    logger.info("momentum_sleeve: LIVE account %s*** verified (not PA)", number[:4])


def sleeve_equity(state: Dict, prices: Dict[str, float]) -> float:
    """cash + sum qty * price; a held symbol without a price is valued at 0 with a WARNING."""
    eq = float(state['cash'])
    for sym, qty in state['positions'].items():
        px = prices.get(sym)
        if px is None or px != px:
            logger.warning("momentum_sleeve: no price for held %s -- valued at 0 in sleeve equity", sym)
            continue
        eq += qty * px
    return eq


QTY_DP = 9            # Alpaca fractional precision; the state must never round a quantity UP past the broker's
DUST_QTY = 1e-6       # a broker remainder below this is dust, not a position


def floor_qty(q: float) -> float:
    """Truncate (never round up) to the broker's 9-decimal precision: a sell rounded up by 1e-7 is rejected
    with 40310000 'insufficient qty available' (2026-10-02 forced run: 2.365105 requested, 2.365104893 held)."""
    return math.floor(float(q) * 10 ** QTY_DP) / 10 ** QTY_DP


def broker_sleeve_cash(client: AlpacaClient, equity_start: float) -> float:
    """Sleeve cash derived from the broker's own record: equity_start - sum(filled 'mom-' buys) + sum(filled
    'mom-' sells). Survives a crashed run, a forced re-run and a lost state file."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.common.enums import Sort
    cash, after, seen = float(equity_start), None, set()
    while True:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, direction=Sort.ASC, after=after)
        page = list(client.trading_client.get_orders(req))
        new = [o for o in page if str(o.id) not in seen]
        for o in new:
            seen.add(str(o.id))
            if not str(getattr(o, 'client_order_id', '') or '').startswith('mom-'):
                continue
            qty, avg = float(o.filled_qty or 0), float(o.filled_avg_price or 0)
            side = str(getattr(o.side, 'value', o.side))
            cash += qty * avg if side == 'sell' else -qty * avg
        if len(page) < 500 or not new:
            return cash
        after = max(o.submitted_at for o in page)


def sync_state_from_broker(client: AlpacaClient, state: Dict, equity_start: float) -> None:
    """The dedicated paper account is the source of truth: positions = the broker's positions (dust dropped),
    cash = equity_start +/- every filled 'mom-' order. Differences from the stored state are logged WARNING
    (they mean a crashed or forced run); a broker failure is a WARNING and the stored state is kept."""
    try:
        pos = {sym: q for sym, q in broker_qty(client).items() if q > DUST_QTY}
        cash = broker_sleeve_cash(client, equity_start)
    except Exception as e:
        logger.warning("momentum_sleeve: broker sync unavailable (%s) -- using the stored state as is", e)
        return
    old_pos, old_cash = state.get('positions', {}), float(state.get('cash', equity_start))
    diff = sorted(k for k in set(pos) | set(old_pos) if abs(pos.get(k, 0.0) - old_pos.get(k, 0.0)) > 1e-4)
    if diff or abs(cash - old_cash) > 0.5:
        logger.warning("momentum_sleeve: state resynced from the broker -- positions changed for %s; cash "
                       "%.2f -> %.2f", diff or 'none', old_cash, cash)
    state['positions'], state['cash'] = pos, cash


def broker_marks(client: AlpacaClient, only=None) -> Dict[str, float]:
    """Current price per symbol from the broker's open positions (the mark for the post-trade equity).
    ``only`` (LIVE): the sleeve's own symbols -- every other position is dropped unseen.
    A failure is a WARNING and returns {} -- the caller then falls back to the signal-date closes."""
    try:
        return {p.symbol: float(p.current_price) for p in client.trading_client.get_all_positions()
                if getattr(p, 'current_price', None) is not None and (only is None or p.symbol in only)}
    except Exception as e:
        logger.warning("momentum_sleeve: broker marks unavailable (%s) -- sleeve equity marked at the "
                       "signal-date closes instead of current prices", e)
        return {}


def client_order_id(day: date, symbol: str, side: str, tag: str = '', prefix: str = ml.PAPER_COID_PREFIX) -> str:
    """'mom-<YYYYMMDD>-<SYM>-<s|b>[-<tag>]' ('moml-' prefix on the LIVE path). ``tag`` is empty for the scheduled weekly run (one id per symbol,
    side and day = idempotent re-run after a crash) and a run stamp under --force, so a deliberate second run on
    the same day gets fresh ids instead of re-booking the first run's fills."""
    if side not in ('sell', 'buy'):
        raise ValueError(f"bad side {side!r}")
    base = f"{prefix}{day:%Y%m%d}-{symbol}-{'s' if side == 'sell' else 'b'}"
    return f"{base}-{tag}" if tag else base


# --------------------------------------------------------------------------- broker I/O

def get_existing_order(client: AlpacaClient, coid: str):
    """The broker order with this client_order_id, or None if it does not exist (404)."""
    try:
        return client.trading_client.get_order_by_client_id(coid)
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            return None
        raise


def _order_exists(client: AlpacaClient, coid: str) -> bool:
    """True when the broker already has an order with this client_order_id; a lookup failure is a WARNING and
    counts as 'not there' (the conservative side: the buy-power check then includes the order)."""
    try:
        return get_existing_order(client, coid) is not None
    except Exception as e:
        logger.warning("momentum_sleeve: order lookup %s failed (%s) -- counted as a new order", coid, e)
        return False


def submit_market(client: AlpacaClient, symbol: str, side: str, coid: str,
                  qty: Optional[float] = None, notional: Optional[float] = None):
    """Market DAY order. Sells use fractional ``qty``; buys use ``notional`` (or whole ``qty`` if non-fractionable). Idempotent on ``coid``:
    an existing order is returned unchanged (logged) instead of re-submitting."""
    existing = get_existing_order(client, coid)
    if existing is not None:
        logger.warning("momentum_sleeve: order %s already exists at the broker -- skipping submit", coid)
        return existing
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    kw = dict(symbol=symbol, side=OrderSide.SELL if side == 'sell' else OrderSide.BUY,
              time_in_force=TimeInForce.DAY, client_order_id=coid)
    if side == 'sell':
        kw['qty'] = floor_qty(qty)
    elif qty is not None:
        kw['qty'] = int(qty)              # non-fractionable buy: whole shares
    else:
        kw['notional'] = round(float(notional), 2)
    return client.trading_client.submit_order(MarketOrderRequest(**kw))


def poll_fills(client: AlpacaClient, coids: List[str], timeout: float = POLL_TIMEOUT,
               interval: float = POLL_SECONDS) -> Dict[str, Dict]:
    """Poll each client order until terminal or timeout. Returns {coid: {status, qty, avg}}.
    Non-filled terminal states and timeouts are logged ERROR."""
    out, deadline = {}, time.time() + timeout
    pending = list(coids)
    while pending:
        for coid in list(pending):
            o = get_existing_order(client, coid)
            status = str(getattr(getattr(o, 'status', ''), 'value', getattr(o, 'status', ''))) if o else 'missing'
            if status in ('filled', 'canceled', 'expired', 'rejected', 'missing'):
                qty = float(getattr(o, 'filled_qty', 0) or 0) if o else 0.0
                avg = float(getattr(o, 'filled_avg_price', 0) or 0) if o else 0.0
                out[coid] = {'status': status, 'qty': qty, 'avg': avg}
                if status != 'filled':
                    logger.error("momentum_sleeve: order %s ended %s (filled %.6f)", coid, status, qty)
                pending.remove(coid)
        if pending:
            if time.time() >= deadline:
                for coid in pending:
                    logger.warning("momentum_sleeve: order %s still open after %.0fs -- state NOT updated for it yet",
                                   coid, timeout)
                    out[coid] = {'status': 'timeout', 'qty': 0.0, 'avg': 0.0}
                break
            time.sleep(interval)
    return out


PENDING_MAX_DAYS = 10                                 # a still-open pending order older than this is dropped (WARNING)
TERMINAL_STATUSES = ('filled', 'canceled', 'expired', 'rejected')


def book_fills(state: Dict, fills: List[Dict], live: bool, final: bool = True) -> None:
    """The ONE place a fill is booked into the sleeve state (poll loop and late-fill reconcile share it).

    LIVE: cash and positions are state-tracked, so the fill is applied (``ml.apply_fills``). PAPER: the dedicated
    account is the source of truth and ``sync_state_from_broker`` rebuilds cash/positions at every run, so only the
    ``booked_coids`` marker is written. Either way the ids are appended to ``state['booked_coids']`` -- unless ``final`` is False (a
    partial fill of an order still working: more may come, the reconcile books the increment later)."""
    if live:
        ml.apply_fills(state, fills)
    if final:
        state['booked_coids'] = sorted(set(state.get('booked_coids', [])) | {f['client_order_id'] for f in fills})


def write_ledger_rows(client: AlpacaClient, fills: List[Dict], ledger_path: str, day: date, size_pct) -> float:
    """Append one ledger row per fill (slippage vs the official open of ``day``); returns the fills' turnover."""
    opens = official_opens(client, [f['symbol'] for f in fills], day)
    turnover = 0.0
    for f in fills:
        op = opens.get(f['symbol'])
        slip = ''
        if op:
            raw = (f['avg_price'] / op - 1.0) * 1e4
            slip = round(raw if f['side'] == 'buy' else -raw, 1)
        append_csv(ledger_path, LEDGER_FIELDS, {
            'date': str(day), 'symbol': f['symbol'], 'side': f['side'], 'qty': f['qty'],
            'avg_price': f['avg_price'], 'notional': round(f['notional'], 2),
            'official_open': op if op else '', 'slip_bps_vs_open': slip,
            'client_order_id': f['client_order_id'], 'size_pct': size_pct})
        turnover += f['notional']
    return turnover


def record_pending(state: Dict, coids: Dict[str, str], res: Dict[str, Dict], phase: str, today: date) -> List[str]:
    """Remember every order the poll gave up on (status 'timeout') in ``state['pending_coids']`` so that the next
    reconcile can book its late fill. Returns those client order ids."""
    open_ids = [c for c in coids if res.get(c, {}).get('status') == 'timeout']
    for c in open_ids:
        state.setdefault('pending_coids', {})[c] = {'symbol': coids[c], 'side': phase, 'date': str(today)}
    return open_ids


def reconcile_late_fills(client: AlpacaClient, state: Dict, today: date, live: bool) -> int:
    """Book the fills of orders that were still open when an earlier poll timed out (docs/mom_sleeve_late_fill_
    booking_spec_20261031.md). Candidates are ``state['pending_coids']`` ({coid: symbol, side, date}, written by
    ``record_pending``); each is looked up at the broker. A filled / partially filled order with filled_qty > 0 is
    booked exactly like a poll fill (``book_fills`` + a ledger row), once: a partial fill books its filled quantity
    and later calls book only the increment. A terminal order leaves the pending list; a lookup failure keeps it.
    State is saved right after booking. Returns the number of fills booked."""
    pending = state.get('pending_coids') or {}
    if not pending:
        return 0
    paths = mode_paths(live)
    booked_n, changed = 0, False
    for coid, info in sorted(pending.items()):
        if coid in state.get('booked_coids', []):
            pending.pop(coid)
            changed = True
            continue
        try:
            o = get_existing_order(client, coid)
        except Exception as e:
            logger.warning("momentum_sleeve: late-fill lookup %s failed (%s) -- kept pending", coid, e)
            continue
        age = (today - date.fromisoformat(info['date'])).days
        if o is None:
            if age > PENDING_MAX_DAYS:
                logger.warning("momentum_sleeve: pending order %s unknown at the broker after %d days -- dropped",
                               coid, age)
                pending.pop(coid)
                changed = True
            continue
        status = str(getattr(getattr(o, 'status', ''), 'value', getattr(o, 'status', '')))
        fq, avg = float(getattr(o, 'filled_qty', 0) or 0), float(getattr(o, 'filled_avg_price', 0) or 0)
        dq = fq - float(info.get('booked_qty', 0.0))
        if dq > 1e-9 and avg > 0:
            dn = fq * avg - float(info.get('booked_notional', 0.0))
            px = dn / dq
            fill = {'symbol': info['symbol'], 'side': info['side'], 'qty': dq, 'avg_price': px, 'notional': dn,
                    'client_order_id': coid}
            book_fills(state, [fill], live, final=status in TERMINAL_STATUSES)
            info['booked_qty'], info['booked_notional'] = fq, fq * avg
            write_ledger_rows(client, [fill], paths['ledger'], date.fromisoformat(info['date']), '')
            logger.info("momentum_sleeve: late fill booked %s qty %s @ %s", coid, dq, px)
            booked_n += 1
            changed = True
        if status in TERMINAL_STATUSES:
            if status != 'filled' and dq <= 1e-9 and fq <= 0:
                logger.warning("momentum_sleeve: pending order %s ended %s unfilled -- nothing to book", coid, status)
            pending.pop(coid)
            changed = True
        elif age > PENDING_MAX_DAYS:
            logger.warning("momentum_sleeve: pending order %s still %s after %d days -- dropped", coid, status, age)
            pending.pop(coid)
            changed = True
    if not pending:
        state.pop('pending_coids', None)
    if changed:
        save_state(state, paths['state'])
    return booked_n


def settle_open_orders(client: AlpacaClient, state: Dict, coids: Dict[str, str], res: Dict[str, Dict],
                       phase: str, today: date, live: bool) -> None:
    """After a poll: remember the orders it gave up on, re-check them once right away (a fill that landed in the
    last seconds is booked now) and WARN about those that stay open (booked at the next run's reconcile)."""
    open_ids = record_pending(state, coids, res, phase, today)
    if not open_ids:
        return
    reconcile_late_fills(client, state, today, live)
    still = [c for c in open_ids if c in (state.get('pending_coids') or {})]
    if still:
        logger.warning("momentum_sleeve: %d order(s) still open after %.0fs -- will be booked at the next run: %s",
                       len(still), POLL_TIMEOUT, ' '.join(still))


def official_opens(client: AlpacaClient, symbols: List[str], day: date) -> Dict[str, float]:
    """Today's daily-bar open per symbol (raw) when available; {} + WARNING on failure."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                               start=pd.Timestamp(day), end=pd.Timestamp(day) + pd.Timedelta(days=1),
                               feed=DataFeed.SIP)
        df = bars_to_df(client._call_with_timeout(lambda: client.data_client.get_stock_bars(req),
                                                  'official_opens', timeout=60))
        return {r.symbol: float(r.open) for r in df.itertuples()}
    except Exception as e:
        logger.warning("momentum_sleeve: official open lookup failed (%s) -- slip_bps_vs_open left blank", e)
        return {}


def broker_qty(client: AlpacaClient) -> Dict[str, float]:
    """{symbol: qty} of every broker position (fractional-safe; AlpacaClient.get_open_positions int-casts)."""
    return {p.symbol: float(p.qty) for p in client.trading_client.get_all_positions()}


# --------------------------------------------------------------------------- the run

def http_get_text(url: str, timeout: float = 60) -> str:
    """GET ``url`` and return the body as text (the CBOE history CSVs); raises on any failure."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research'})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def fetch_cboe_close(name: str) -> pd.Series:
    """CBOE daily history of index ``name`` ('VIX' / 'VIX3M') as a close series indexed by date (1700q parse)."""
    d = pd.read_csv(io.StringIO(http_get_text(CBOE_URL.format(name=name))))
    d['DATE'] = pd.to_datetime(d['DATE'], format='%m/%d/%Y')
    return d.set_index('DATE')['CLOSE'].astype(float).rename(name)


def shadow_gate_info(asof) -> Optional[Dict]:
    """Term-structure gate at ``asof``; None (WARNING) on any fetch / parse / history failure.

    It never blocks the rebalance: a missing gate trades full size (``ms.gate_scale``). A gate computed on a close
    older than ``asof`` (the CBOE file not yet updated) is still used, with a WARNING naming the date it read."""
    try:
        info = ms.term_structure_gate(fetch_cboe_close('VIX'), fetch_cboe_close('VIX3M'), asof)
    except Exception as e:                      # network, schema, empty file: the gate is reported n/a
        logger.warning("momentum_sleeve: term-structure gate unavailable (%s) -- reported n/a", e)
        return None
    if info and info['date'] < pd.Timestamp(asof).date():
        logger.warning("momentum_sleeve: term-structure gate is STALE -- last CBOE close %s, signal date %s",
                       info['date'], pd.Timestamp(asof).date())
    return info


def gate_label(info: Optional[Dict], asof=None) -> str:
    """'gate ON (p21)' / 'gate OFF (p55)' / 'gate n/a (full size)' / 'gate STALE ON (p21)' (CBOE close older than
    ``asof``) for the log and the [MOM] Telegram line."""
    if not info:
        return 'gate n/a (full size)'
    stale = asof is not None and info['date'] < pd.Timestamp(asof).date()
    return f"gate {'STALE ' if stale else ''}{'ON' if info['gate_on'] else 'OFF'} (p{info['percentile'] * 100:.0f})"


def size_label(scale: float) -> str:
    """'size 100%' / 'size 50%' for the plan line, the ledger and the [MOM] Telegram line."""
    return f"size {scale * 100:.0f}%"


def append_shadow_gate(run_day: date, info: Optional[Dict], equity: float, scale: float = 1.0,
                       path: Optional[str] = None) -> None:
    """One row per run to the gate CSV incl. the applied size ``scale`` (skipped, WARNING, when the gate is n/a)."""
    if not info:
        logger.warning("momentum_sleeve: no shadow-gate row written for %s (gate n/a)", run_day)
        return
    append_csv(path or SHADOW_GATE_PATH, SHADOW_GATE_FIELDS, {
        'run_date': str(run_day), 'asof': str(info['date']), 'vix': round(info['vix'], 4),
        'vix3m': round(info['vix3m'], 4), 'ratio': round(info['ratio'], 6),
        'percentile': round(info['percentile'], 4), 'gate_on': info['gate_on'], 'equity': round(equity, 2),
        'scale': scale})


def guard_line(events: Dict[str, tuple], top_unguarded: List[str]) -> str:
    """The per-run guard INFO line: guard-removed names among the unguarded top 40, with reason and event date."""
    hit = [f"{s} {events[s][0]} {events[s][1]}" for s in top_unguarded[:GUARD_TOP] if s in events]
    return "guard: " + ("; ".join(hit) if hit else "none in the top 40")


def build_plan(panel: pd.DataFrame, assets: pd.DataFrame, asof: date, n: int, state: Dict,
               marks: Optional[Dict[str, float]] = None, scale: float = 1.0):
    """(selected, feat, equity, targets_usd, orders, prices) for this run -- pure given the inputs.

    The SIGNAL uses the closes of ``asof``. The SIZING (sleeve equity, each held name's value, trim/exit
    quantities) uses ``marks`` -- the broker's current prices -- where given, so the weekly reset lands every
    name at 1/N at today's prices, as the research build does at Monday's open. Without marks (dry run before
    the open, or a broker failure) the signal-date closes are used (2026-10-02 forced run: sizing on the prior
    close left names between $873 and $1,062 instead of equal)."""
    feat = ms.risk_adjusted_momentum(panel, asof)
    elig = [s for s in ms.eligible_universe(panel, asof, assets) if s != 'SPY']
    raw_elig = [s for s in ms.eligible_universe(panel, asof, assets, guard=False) if s != 'SPY']
    top_raw = ms.select_top(feat.loc[raw_elig, 'signal'], GUARD_TOP)
    logger.info("momentum_sleeve: %s", guard_line(ms.hygiene_events(panel, asof), top_raw))
    selected = ms.select_top(feat.loc[elig, 'signal'], n)
    prices = {s: float(p) for s, p in feat['close'].items()}
    prices.update({s: float(p) for s, p in (marks or {}).items() if p and p == p})
    equity = sleeve_equity(state, prices)
    targets = ms.target_dollars(selected, equity, n, scale)
    current = {s: q * prices[s] for s, q in state['positions'].items() if s in prices and q > 0}
    held_without_price = [s for s, q in state['positions'].items() if q > 0 and s not in prices]
    if held_without_price:
        logger.error("momentum_sleeve: held %s have no bar on %s -- cannot size their exit; handle manually",
                     held_without_price, asof)
    orders = ms.rebalance_orders(current, targets)
    return selected, feat, equity, targets, orders, prices


def apply_fractionable(client: AlpacaClient, order: Dict, prices: Dict[str, float]) -> None:
    """Buy order on a non-fractionable asset -> ``whole_qty`` = floor(notional / price) and ``residual`` USD left
    uninvested (selection is unchanged, the BT has no fractionable filter). Idempotent. A failed asset lookup is a
    WARNING and the order stays notional-based; a 0-share result is kept (0) and skipped by ``execute``."""
    if order['side'] != 'buy' or 'whole_qty' in order or order.get('fractionable_checked'):
        return
    order['fractionable_checked'] = True
    sym = order['symbol']
    try:
        fractionable = bool(client.get_asset_tolerant(sym).fractionable)
    except Exception as e:
        logger.warning("momentum_sleeve: %s asset lookup failed (%s) -- assuming fractionable, notional order", sym, e)
        return
    if fractionable:
        return
    qty = int(math.floor(order['notional'] / prices[sym]))
    order['whole_qty'], order['residual'] = qty, order['notional'] - qty * prices[sym]
    logger.warning("momentum_sleeve: %s is non-fractionable -- whole shares %d (residual $%.2f of $%.2f)",
                   sym, qty, order['residual'], order['notional'])


def execute(client: AlpacaClient, orders: List[Dict], prices: Dict[str, float], state: Dict,
            today: date, tag: str = '') -> List[Dict]:
    """Submit sells, poll to fill, then buys, poll. Returns the fills; the caller resyncs ``state`` from the
    broker afterwards. Sell quantities come from the BROKER's position (a full exit sells exactly what is there,
    a trim is floored to 9 decimals). One rejected order is logged ERROR and does not abort the others."""
    fills: List[Dict] = []
    bqty = broker_qty(client)
    for phase in ('sell', 'buy'):
        batch = [o for o in orders if o['side'] == phase]
        coids = {}
        for o in batch:
            sym = o['symbol']
            coid = client_order_id(today, sym, phase, tag)
            try:
                if phase == 'sell':
                    held = bqty.get(sym, 0.0)
                    if held <= DUST_QTY:
                        logger.warning("momentum_sleeve: %s sell skipped -- the broker holds %.9f", sym, held)
                        continue
                    qty = held if o['full_exit'] else min(held, floor_qty(o['notional'] / prices[sym]))
                    submit_market(client, sym, 'sell', coid, qty=qty)
                else:
                    apply_fractionable(client, o, prices)
                    if o.get('whole_qty') == 0:
                        logger.warning("momentum_sleeve: %s non-fractionable and $%.2f buys 0 shares -- skipped",
                                       sym, o['notional'])
                        continue
                    submit_market(client, sym, 'buy', coid, qty=o.get('whole_qty'), notional=o['notional'])
            except Exception as e:
                logger.error("momentum_sleeve: %s %s order %s REJECTED (%s) -- continuing with the rest",
                             phase, sym, coid, e)
                continue
            coids[coid] = sym
        res = poll_fills(client, list(coids)) if coids else {}
        settle_open_orders(client, state, coids, res, phase, today, live=False)
        for coid, sym in coids.items():
            r = res[coid]
            if r['qty'] <= 0:
                continue
            fills.append({'symbol': sym, 'side': phase, 'qty': r['qty'], 'avg_price': r['avg'],
                          'notional': r['qty'] * r['avg'], 'client_order_id': coid})
    return fills


def execute_live(client: AlpacaClient, orders: List[Dict], prices: Dict[str, float], state: Dict,
                 today: date, tag: str = '', notes: Optional[List[str]] = None, on_phase=None) -> List[Dict]:
    """LIVE (shared account) twin of ``execute``: the sleeve STATE decides what is sold and what cash exists.

    * sells never exceed the STATE qty (broker more -> state qty only + WARN; broker less -> what is there + WARN
      and the state qty is set to the broker's); a symbol not in the state is never sold;
    * after the sells the account's buying power must cover the whole buy side, else ERROR + a note and NO buys;
    * orders carry the ``moml-`` client_order_id prefix;
    * fills are booked into ``state`` (cash and positions) after each phase, then ``on_phase(state)`` persists it.
    One rejected order is logged ERROR and does not abort the others. ``notes`` collects the Telegram lines."""
    notes = notes if notes is not None else []
    fills: List[Dict] = []
    bqty = broker_qty(client)
    day_key = f"{today:%Y%m%d}"
    booked = set(c for c in state.get('booked_coids', []) if day_key in c)   # today's ids only (pruned daily)
    for phase in ('sell', 'buy'):
        batch = [o for o in orders if o['side'] == phase]
        if phase == 'buy' and batch:
            for b in batch:
                apply_fractionable(client, b, prices)
            # A buy whose id is already at the broker (a crashed earlier run today) consumed its buying power
            # already: counting it again would refuse every buy and leave those fills unbooked in the state.
            fresh = [b for b in batch if not _order_exists(client, client_order_id(
                today, b['symbol'], 'buy', tag, prefix=ml.LIVE_COID_PREFIX))]
            need = sum((o['whole_qty'] * prices[o['symbol']]) if o.get('whole_qty') is not None else o['notional']
                       for o in fresh)
            bp = float(client.trading_client.get_account().buying_power)
            if not ml.buying_power_ok(bp, need):
                msg = f"buying power ${bp:,.0f} < buy notional ${need:,.0f} -- NO buys this rotation (sells stand)"
                logger.error("momentum_sleeve: %s", msg)
                notes.append(msg)
                break
        coids = {}
        for o in batch:
            sym = o['symbol']
            coid = client_order_id(today, sym, phase, tag, prefix=ml.LIVE_COID_PREFIX)
            try:
                if phase == 'sell':
                    st_qty = float(state['positions'].get(sym, 0.0))
                    if st_qty <= DUST_QTY:
                        logger.warning("momentum_sleeve: %s sell skipped -- not in the sleeve state", sym)
                        continue
                    want = None if o['full_exit'] else floor_qty(o['notional'] / prices[sym])
                    qty, warn = ml.sell_quantity(st_qty, bqty.get(sym, 0.0), want)
                    if warn:
                        logger.warning("momentum_sleeve: %s %s", sym, warn)
                        notes.append(f"{sym}: {warn}")
                    if bqty.get(sym, 0.0) < st_qty - 1e-6:
                        state['positions'][sym] = bqty.get(sym, 0.0)
                    if qty <= DUST_QTY:
                        continue
                    submit_market(client, sym, 'sell', coid, qty=qty if not o['full_exit'] else floor_qty(qty))
                else:
                    if o.get('whole_qty') == 0:
                        logger.warning("momentum_sleeve: %s non-fractionable and $%.2f buys 0 shares -- skipped",
                                       sym, o['notional'])
                        continue
                    submit_market(client, sym, 'buy', coid, qty=o.get('whole_qty'), notional=o['notional'])
            except Exception as e:
                logger.error("momentum_sleeve: %s %s order %s REJECTED (%s) -- continuing with the rest",
                             phase, sym, coid, e)
                continue
            coids[coid] = sym
        res = poll_fills(client, list(coids)) if coids else {}
        settle_open_orders(client, state, coids, res, phase, today, live=True)
        booked |= {c for c in state.get('booked_coids', []) if day_key in c}   # fills booked by the re-check
        phase_fills = []
        for coid, sym in coids.items():
            r = res[coid]
            if r['qty'] <= 0:
                continue
            if coid in booked:
                # Same-day re-run after a crash: submit_market returned the existing order -- its fill is
                # already in the state; booking it again would double the position and the cash move.
                logger.warning("momentum_sleeve: %s fill already booked in the sleeve state -- not booked twice",
                               coid)
                continue
            phase_fills.append({'symbol': sym, 'side': phase, 'qty': r['qty'], 'avg_price': r['avg'],
                                'notional': r['qty'] * r['avg'], 'client_order_id': coid})
        book_fills(state, phase_fills, live=True)
        booked |= {f['client_order_id'] for f in phase_fills}
        state['booked_coids'] = sorted(booked)
        fills.extend(phase_fills)
        if on_phase is not None:
            on_phase(state)
    return fills


def format_orders(selected: List[str], feat: pd.DataFrame, equity: float, orders: List[Dict],
                  scale: float = 1.0) -> str:
    """Human-readable target list + orders for the dry run (target per name = scale * equity / N)."""
    lines = [f"sleeve equity ${equity:,.2f}; {size_label(scale)}; target per name "
             f"${scale * equity / max(len(selected), 1):,.2f}", "TOP:"]
    for i, s in enumerate(selected, 1):
        lines.append(f"  {i:2d} {s:<6} signal {feat.loc[s, 'signal']:.3f}  close {feat.loc[s, 'close']:.2f}")
    lines.append(f"ORDERS ({len(orders)}), sells first:")
    for o in orders:
        whole = (f" (whole {o['whole_qty']} sh, residual ${o['residual']:,.2f})" if 'whole_qty' in o else '')
        lines.append(f"  {o['side'].upper():<4} {o['symbol']:<6} ${o['notional']:,.2f}"
                     f"{' (full exit)' if o['full_exit'] else ''}{whole}")
    return "\n".join(lines)


def notify(notifier: Optional[TelegramNotifier], msg: str, prefix: str = TG_PREFIX) -> None:
    """ONE [MOM] (LIVE: [MOM LIVE]) Telegram message; a failure is a WARNING, never fatal."""
    if notifier is None:
        return
    try:
        notifier.send_message_sync(f"{prefix} {msg}")
    except Exception as e:
        logger.warning("momentum_sleeve: Telegram send failed (non-fatal): %s", e)


def live_guards(client: AlpacaClient, args, state: Dict, selected: List[str], orders: List[Dict],
                prices: Dict[str, float], equity: float) -> Tuple[List[Dict], bool, List[str]]:
    """LIVE-only plan guards on the shared account. Returns (orders, rail_hit, lines for the log/Telegram).

    * the drawdown rail (sleeve equity <= 0.75 x start) replaces the plan by full exits of the STATE names;
    * buy collisions (a selected name the broker holds outside the state) lose their buy -- cash stays cash;
    * the count of non-sleeve positions is reported, never their symbols."""
    lines: List[str] = []
    all_qty = broker_qty(client)
    _, ignored = ml.sleeve_view(all_qty, state['positions'])
    lines.append(f"{ignored} non-sleeve positions ignored")
    rail = ml.drawdown_breached(equity, state['equity_start'])
    if rail:
        cur = {s: q * prices[s] for s, q in state['positions'].items() if s in prices and q > 0}
        orders = ms.rebalance_orders(cur, {})
        lines.append(f"RAIL: sleeve equity ${equity:,.0f} <= {ml.DD_PAUSE_FRAC:.0%} of ${state['equity_start']:,.0f} -> "
                     f"SELL EVERYTHING the state holds and PAUSE")
        logger.error("momentum_sleeve: drawdown rail hit -- liquidating the sleeve")
        return orders, True, lines
    collisions = ml.find_collisions(selected, all_qty, state['positions'])
    for sym in collisions:
        lines.append(f"COLLISION {sym}: held manually — skipped")
        logger.warning("momentum_sleeve: COLLISION %s: held manually -- skipped", sym)
    return ml.drop_collision_buys(orders, collisions), False, lines


def live_dry_lines(client: AlpacaClient, state: Dict, orders: List[Dict], prices: Dict[str, float], args) -> str:
    """Dry-run lines for the LIVE plan: notional, sleeve cash, buying-power check, cadence (dry run only)."""
    buy = sum(o['notional'] for o in orders if o['side'] == 'buy')
    sell = sum(o['notional'] for o in orders if o['side'] == 'sell')
    try:
        bp = float(client.trading_client.get_account().buying_power)
        bp_txt = (f"buying power ${bp:,.0f} vs buy notional ${buy:,.0f}: "
                  f"{'OK' if ml.buying_power_ok(bp + sell, buy) else 'INSUFFICIENT (buys would be refused)'}")
    except Exception as e:
        logger.error("momentum_sleeve: buying power unavailable (%s)", e)
        bp_txt = "buying power unavailable"
    eq = sleeve_equity(state, prices)
    _, weeks = ml.cadence(ml.read_weekly(mode_paths(True)['weekly']), state['equity_start'], datetime.now(ET).date())
    warn = f" WARNING >= {args.cadence_warn_weeks}" if weeks >= args.cadence_warn_weeks else ""
    return (f"LIVE PLAN: sells ${sell:,.0f} buys ${buy:,.0f} | sleeve cash (state) ${state['cash']:,.2f} | "
            f"{bp_txt} | sleeve equity ${eq:,.0f} (start ${state['equity_start']:,.0f}, rail at "
            f"${ml.DD_PAUSE_FRAC * state['equity_start']:,.0f}) | weeks since a strong week {weeks}{warn}\n"
            f"NOTHING WRITTEN (dry run)")


def submit_live(args, client, notifier, state, orders, prices, feat, panel, asof, today, now_utc, scale, gate,
                gate_txt, live_notes, paths) -> int:
    """LIVE submit: verify the account is NOT paper, execute from the state, book fills, rails, files, Telegram."""
    assert_live_account(client)
    names_before = set(state['positions'])
    rail = any(l.startswith('RAIL:') for l in live_notes)
    run_tag = f"f{now_utc:%H%M%S}" if args.force else ''
    notes: List[str] = [l for l in live_notes if not l.endswith('non-sleeve positions ignored')]
    save = lambda st: save_state(st, paths['state'])   # noqa: E731
    fills = execute_live(client, orders, prices, state, today, run_tag, notes, on_phase=save)
    state['last_rebalance'], state['last_rebalance_run'] = str(today), datetime.now(timezone.utc).isoformat()
    # Persist the rotation marker NOW, as the paper path does: a crash in the ledger / marks code below must not
    # let the 14:46 retry re-plan this rotation.
    save_state(state, paths['state'])
    turnover = write_ledger_rows(client, fills, paths['ledger'], today, round(scale * 100))
    eq_after = sleeve_equity(state, {**prices, **broker_marks(client, only=set(state['positions']) | names_before)})
    state['peak_equity'] = max(float(state.get('peak_equity', eq_after)), eq_after)
    dd = eq_after / state['peak_equity'] - 1.0
    if rail:
        state['paused'] = True
        state['paused_reason'] = (f"drawdown rail {today}: sleeve equity ${eq_after:,.0f} <= "
                                  f"{ml.DD_PAUSE_FRAC:.0%} of ${state['equity_start']:,.0f}")
        notes.append(f"PAUSED -- {state['paused_reason']}; clear with: {ml.PAUSE_CLEAR_CMD}")
    save_state(state, paths['state'])
    rows = ml.read_weekly(paths['weekly'])
    week_r, weeks = ml.cadence(rows, state['equity_start'], today, eq_after)
    spy = feat['close'].get('SPY') if 'SPY' in feat.index else float(
        panel.loc[(panel.symbol == 'SPY') & (panel.bar_date == pd.Timestamp(asof)), 'close'].iloc[0])
    names_in, names_out = sorted(set(state['positions']) - names_before), sorted(names_before - set(state['positions']))
    append_csv(paths['weekly'], ml.LIVE_WEEKLY_FIELDS, {
        'date': str(today), 'equity': round(eq_after, 2), 'cash': round(state['cash'], 2),
        'n_names': len(state['positions']), 'turnover_usd': round(turnover, 2),
        'names_in': ' '.join(names_in), 'names_out': ' '.join(names_out), 'spy_close': spy,
        'week_r': round(week_r, 2), 'weeks_since_strong': weeks})
    append_shadow_gate(today, gate, eq_after, scale, path=paths['gate'])
    if weeks >= args.cadence_warn_weeks:
        msg = f"CADENCE: {weeks} weeks since a strong week (>= +{ml.STRONG_WEEK_R:.0f}R) -- warn at {args.cadence_warn_weeks}"
        logger.warning("momentum_sleeve: %s", msg)
        notes.append(msg)
    kill = f" KILL RULE: drawdown {dd:.1%} > 40% -> HALVE the sleeve" if dd < -DD_KILL else ""
    extra = (" | " + " | ".join(notes)) if notes else ""
    notify(notifier, f"{today} equity ${eq_after:,.0f} (dd {dd:.1%} from peak) names {len(state['positions'])} "
                     f"in {len(names_in)} out {len(names_out)} turnover ${turnover:,.0f} fills {len(fills)}/{len(orders)} "
                     f"week {week_r:+.1f}R {gate_txt}{kill}{extra}", ml.LIVE_TG_PREFIX)
    return 0


def run(args, client: AlpacaClient, notifier: Optional[TelegramNotifier], now_utc: datetime) -> int:
    """Whole run; returns the process exit code."""
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    week_start = today - timedelta(days=today.weekday())
    week_sessions = trading_sessions(client, week_start, today)
    is_session = today in week_sessions
    live = getattr(args, 'live', False) is True   # strict: a MagicMock args object is the paper path
    paths = mode_paths(live)
    tg = ml.LIVE_TG_PREFIX if live else TG_PREFIX
    if not args.force:
        due, reason = rotation_due(today, week_sessions, (ml.read_json(paths['state']) or {}).get('last_rebalance'))
        if not due:
            print(f"not a rebalance session — no-op ({reason})")
            return 0
        logger.info("momentum_sleeve: rotation due: %s", reason)
    if args.submit and is_session and now_et.time() < SUBMIT_START:
        # The EDT/EST twin cron lines: in EST the 13:45 UTC line lands at 08:45 ET. Not an error -- the 14:45 line
        # submits (an exception here reached Telegram as a 'crashed' ERROR every EST rotation day).
        msg = f"submit window opens {SUBMIT_START:%H:%M} ET (now {now_et:%H:%M} ET) — no-op; the later cron line submits"
        logger.info("momentum_sleeve: %s", msg)
        print(msg)
        return 0
    if args.submit:
        check_submit_window(now_et, is_session)
    sessions = trading_sessions(client, today - timedelta(days=14), today)
    asof = pd.Timestamp(args.asof).date() if args.asof else last_completed_session(sessions, now_et)
    state = load_state(paths['state'], args.equity_start)
    if not live:
        sync_state_from_broker(client, state, args.equity_start)
    reconcile_late_fills(client, state, today, live)   # book fills of orders left open by an earlier run's poll
    if live:
        state.setdefault('equity_start', float(args.equity_start))
        if state['equity_start'] != float(args.equity_start):
            logger.warning("momentum_sleeve: --equity-start %.0f differs from the state's %.0f -- the STATE value rules",
                           args.equity_start, state['equity_start'])
        if state.get('paused'):
            msg = (f"PAUSED -- {state.get('paused_reason', 'no reason recorded')}; refusing to trade. "
                   f"Clear with: {ml.PAUSE_CLEAR_CMD}")
            logger.error("momentum_sleeve: %s", msg)
            print(msg)
            if args.submit:
                notify(notifier, f"{today} {msg}", tg)
            return 3
    if args.submit and state.get('last_rebalance') == str(today) and not args.force:
        print(f"already rebalanced on {today} — no-op")
        return 0

    assets = fetch_assets(client)
    cpath = cache_file(asof)
    if args.skip_fetch and os.path.exists(cpath):
        panel = pd.read_parquet(cpath)
        logger.info("momentum_sleeve: reusing cache %s (%d rows)", cpath, len(panel))
        info = read_completeness(asof)
    else:
        if args.skip_fetch:
            logger.warning("momentum_sleeve: --skip-fetch but %s missing -- fetching", cpath)
        panel, lost = fetch_panel(client, list(assets['symbol']), asof)
        info = None
        if len(panel):
            write_cache_atomic(panel, asof)
            panel['bar_date'] = pd.to_datetime(panel['bar_date'])
            info = write_completeness(asof, len(assets) + 1, panel, lost)
    if args.skip_fetch and info is not None:
        print(f"COMPLETENESS (from the prefetch): requested {info['requested']}, {info['with_asof_bar']} with an "
              f"{asof} bar ({info['ratio']:.0%}), LOST {len(info['lost'])} ({info['lost_share']:.1%}, max "
              f"{LOST_MAX:.0%}), liquid {info['liquid_with_asof']}/{info['n_liquid']} ({info['liquid_ratio']:.1%}, "
              f"min {COMPLETENESS_MIN:.0%})")
    else:
        print(completeness_line(len(assets) + 1, panel, info['lost'] if info else [], asof))
    if panel.empty:
        logger.error("momentum_sleeve: empty panel -- aborting")
        return 1
    panel['bar_date'] = pd.to_datetime(panel['bar_date'])

    gate = shadow_gate_info(pd.Timestamp(asof))
    scale = ms.gate_scale(gate, args.gate)
    gate_txt = f"{gate_label(gate, asof)} {size_label(scale)}"
    marks = broker_marks(client, only=set(state['positions'])) if live else broker_marks(client)
    selected, feat, equity, targets, orders, prices = build_plan(panel, assets, asof, args.n, state,
                                                                 marks=marks, scale=scale)
    logger.info("momentum_sleeve: %s  ratio %s", gate_txt,
                f"{gate['ratio']:.4f} (VIX {gate['vix']:.2f} / VIX3M {gate['vix3m']:.2f}, {gate['date']})" if gate
                else "unavailable")
    print(f"asof {asof}  today {today}  mode {'LIVE ' if live else ''}{'SUBMIT' if args.submit else 'DRY-RUN'}  {gate_txt}")
    live_notes: List[str] = []
    if live:
        orders, rail_line, live_notes = live_guards(client, args, state, selected, orders, prices, equity)
    for o in orders:
        apply_fractionable(client, o, prices)
    print(format_orders(selected, feat, equity, orders, scale))
    if live:
        for line in live_notes:
            print(line)
        if not args.submit:
            print(live_dry_lines(client, state, orders, prices, args))
    refusal = completeness_refusal(info, asof, set(state['positions']), selected,
                                   missing_bar=set(state['positions']) - set(feat.index))
    if refusal:
        label = 'MOM LIVE' if live else 'MOM'
        logger.error("%s REFUSED: completeness %s -- positions left as they are (no sells, no buys)", label, refusal)
        print(f"{label} REFUSED: completeness {refusal}")
        if args.submit:
            notify(notifier, f"{today} REFUSED: completeness {refusal} -- no orders placed, book unchanged", tg)
        return 2
    if not args.submit:
        return 0

    if live:
        return submit_live(args, client, notifier, state, orders, prices, feat, panel, asof, today, now_utc,
                           scale, gate, gate_txt, live_notes, paths)
    assert_paper_account(client)
    names_before = set(state['positions'])
    run_tag = f"f{now_utc:%H%M%S}" if args.force else ''
    fills = execute(client, orders, prices, state, today, run_tag)
    # Persist the rebalance marker NOW (the 14:45 cron is a retry): a crash in the sync / ledger / marking code
    # below must not let the retry re-plan against a pre-trade state.
    state['last_rebalance'], state['last_rebalance_run'] = str(today), datetime.now(timezone.utc).isoformat()
    save_state(state, STATE_PATH)
    sync_state_from_broker(client, state, args.equity_start)
    turnover = write_ledger_rows(client, fills, LEDGER_PATH, today, round(scale * 100))
    state['last_rebalance'] = str(today)
    # Mark at CURRENT broker prices: the signal-date closes are a session stale (2026-10-02 forced run marked
    # $20,134 on a $19,989 book and would have set a false equity peak for the drawdown rule).
    eq_after = sleeve_equity(state, {**prices, **broker_marks(client)})
    state['peak_equity'] = max(float(state.get('peak_equity', eq_after)), eq_after)
    dd = eq_after / state['peak_equity'] - 1.0
    save_state(state, STATE_PATH)
    spy = feat['close'].get('SPY') if 'SPY' in feat.index else float(
        panel.loc[(panel.symbol == 'SPY') & (panel.bar_date == pd.Timestamp(asof)), 'close'].iloc[0])
    names_in, names_out = sorted(set(state['positions']) - names_before), sorted(names_before - set(state['positions']))
    append_csv(WEEKLY_PATH, WEEKLY_FIELDS, {
        'date': str(today), 'equity': round(eq_after, 2), 'cash': round(state['cash'], 2),
        'n_names': len(state['positions']), 'turnover_usd': round(turnover, 2),
        'names_in': ' '.join(names_in), 'names_out': ' '.join(names_out), 'spy_close': spy})
    append_shadow_gate(today, gate, eq_after, scale)
    kill = f" KILL RULE: drawdown {dd:.1%} > 40% -> HALVE the sleeve" if dd < -DD_KILL else ""
    notify(notifier, f"{today} equity ${eq_after:,.0f} (dd {dd:.1%} from peak) names {len(state['positions'])} "
                     f"in {len(names_in)} out {len(names_out)} turnover ${turnover:,.0f} fills {len(fills)}/{len(orders)} {gate_txt}{kill}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI parser (``--gate`` defaults to 'half': PAPER account only, the account guard refuses a live key)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asof', help='signal date YYYY-MM-DD (default: last completed trading day)')
    ap.add_argument('--n', type=int, default=ms.DEFAULT_N)
    ap.add_argument('--equity-start', type=float, default=None,
                    help='sleeve starting equity (paper default 20000; REQUIRED with --live: the slice, e.g. 10000)')
    ap.add_argument('--live', action='store_true',
                    help='run on the LIVE shared account (ALPACA_API_KEY) with the state-isolated sleeve; '
                         'separate state/ledger/weekly/gate files, [MOM LIVE] Telegram, moml- order ids')
    ap.add_argument('--cadence-warn-weeks', type=int, default=ml.DEFAULT_CADENCE_WARN_WEEKS,
                    help='LIVE: WARN when this many weeks passed since a strong (>= +5R) week')
    ap.add_argument('--skip-fetch', action='store_true', help='reuse the cached parquet for --asof')
    ap.add_argument('--force', action='store_true', help='run on a non-rebalance day')
    ap.add_argument('--submit', action='store_true', help='place orders (PAPER, or LIVE with --live; default: dry-run)')
    ap.add_argument('--gate', choices=[ms.GATE_MODE_OFF, ms.GATE_MODE_SHADOW, ms.GATE_MODE_HALF],
                    default=ms.GATE_MODE_HALF,
                    help="term-structure gate: off, shadow (log only) or half (half size when VIX/VIX3M pct < 20%%)")
    ap.add_argument('--verbose', action='store_true')
    return ap


def main_live(args, cfg) -> int:
    """LIVE entry: the owner's live key (ALPACA_API_KEY), paper=False, non-PA account assertion."""
    key, secret = os.environ.get('ALPACA_API_KEY', ''), os.environ.get('ALPACA_API_SECRET', '')
    if not key or not secret:
        logger.error("momentum_sleeve: ALPACA_API_KEY/SECRET not set -- cannot run --live")
        return 1
    client = AlpacaClient(key, secret, paper=False)
    try:
        assert_live_account(client)
    except RuntimeError as e:
        logger.error("momentum_sleeve: refusing to run -- %s", e)
        return 1
    notifier = None
    if args.submit:
        if cfg.telegram_bot_token and cfg.telegram_chat_id:
            notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
        else:
            logger.warning("momentum_sleeve: Telegram not configured -- proceeding without notifications")
    return run(args, client, notifier, datetime.now(timezone.utc))


def main() -> int:
    """CLI entry."""
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    attach_telegram_error_handler()   # non-crash ERRORs reach Telegram too
    cfg = Config()   # loads .env
    if args.live and args.equity_start is None:
        logger.error("momentum_sleeve: --live requires an explicit --equity-start (the slice, e.g. 10000)")
        return 1
    if args.equity_start is None:
        args.equity_start = 20000.0
    if args.live:
        return main_live(args, cfg)
    # Dedicated sleeve paper account (owner 2026-10-02): its positions never mix with the ORB/HOD engines'
    # broker lines. No fallback to another account — a missing key is an error, not a reason to trade elsewhere.
    mom_key, mom_secret = os.environ.get('ALPACA_MOM_API_KEY', ''), os.environ.get('ALPACA_MOM_API_SECRET', '')
    if not mom_key or not mom_secret:
        logger.error("momentum_sleeve: ALPACA_MOM_API_KEY/SECRET not set — cannot run")
        return 1
    client = AlpacaClient(mom_key, mom_secret, paper=True)   # paper ONLY; going live is a deliberate code change
    try:
        assert_paper_account(client)
    except RuntimeError as e:
        logger.error("momentum_sleeve: refusing to run — %s", e)
        return 1
    notifier = None
    if args.submit:
        if cfg.telegram_bot_token and cfg.telegram_chat_id:
            notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
        else:
            logger.warning("momentum_sleeve: Telegram not configured — proceeding without notifications")
    return run(args, client, notifier, datetime.now(timezone.utc))


if __name__ == '__main__':
    sys.exit(run_guarded(main, 'momentum_sleeve'))
