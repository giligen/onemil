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
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError   # noqa: E402
from notifications.telegram_notifier import TelegramNotifier         # noqa: E402
from tom_sleeve import assert_paper_account, _to_date                # noqa: E402  (shared paper guard)
from trading import momentum_sleeve as ms                            # noqa: E402

logger = logging.getLogger('momentum_sleeve')

ET = ZoneInfo('US/Eastern')
DATA_DIR = os.path.join(ROOT, 'data', 'momentum_sleeve')
STATE_PATH = os.path.join(DATA_DIR, 'state.json')
LEDGER_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_ledger.csv')
WEEKLY_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_weekly.csv')
LEDGER_FIELDS = ['date', 'symbol', 'side', 'qty', 'avg_price', 'notional', 'official_open',
                 'slip_bps_vs_open', 'client_order_id']
WEEKLY_FIELDS = ['date', 'equity', 'cash', 'n_names', 'turnover_usd', 'names_in', 'names_out', 'spy_close']
TG_PREFIX = '[MOM]'
SHADOW_GATE_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_shadow_gate.csv')
SHADOW_GATE_FIELDS = ['run_date', 'asof', 'vix', 'vix3m', 'ratio', 'percentile', 'gate_on', 'equity']
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
    assets = client._call_with_timeout(lambda: client.trading_client.get_all_assets(req), 'get_all_assets',
                                       timeout=90, timeout_retries=2, rate_limit_retries=5)
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


def completeness_line(n_requested: int, panel: pd.DataFrame, lost: List[str], asof: date) -> str:
    """One-line completeness gate. Logs ERROR when under 90 % of requested symbols have an asof bar."""
    on_asof = int((panel['bar_date'] == pd.Timestamp(asof)).sum()) if len(panel) else 0
    ratio = on_asof / max(n_requested, 1)
    line = (f"COMPLETENESS: requested {n_requested} symbols, {panel['symbol'].nunique() if len(panel) else 0} "
            f"with bars, {on_asof} with an {asof} bar ({ratio:.0%}), LOST {len(lost)}")
    (logger.error if ratio < 0.5 else logger.info)(line)
    return line


def cache_file(asof: date) -> str:
    """Parquet cache path for a signal date."""
    return os.path.join(DATA_DIR, f"daily_{asof:%Y%m%d}.parquet")


def prune_caches(keep: int = KEEP_CACHES) -> None:
    """Keep only the newest ``keep`` daily parquet files."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'daily_*.parquet')))
    for f in files[:-keep]:
        os.remove(f)
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
    """Append one row, header first if the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


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


def broker_marks(client: AlpacaClient) -> Dict[str, float]:
    """Current price per symbol from the broker's open positions (the mark for the post-trade equity).
    A failure is a WARNING and returns {} -- the caller then falls back to the signal-date closes."""
    try:
        return {p.symbol: float(p.current_price) for p in client.trading_client.get_all_positions()
                if getattr(p, 'current_price', None) is not None}
    except Exception as e:
        logger.warning("momentum_sleeve: broker marks unavailable (%s) -- sleeve equity marked at the "
                       "signal-date closes instead of current prices", e)
        return {}


def client_order_id(day: date, symbol: str, side: str, tag: str = '') -> str:
    """'mom-<YYYYMMDD>-<SYM>-<s|b>[-<tag>]'. ``tag`` is empty for the scheduled weekly run (one id per symbol,
    side and day = idempotent re-run after a crash) and a run stamp under --force, so a deliberate second run on
    the same day gets fresh ids instead of re-booking the first run's fills."""
    if side not in ('sell', 'buy'):
        raise ValueError(f"bad side {side!r}")
    base = f"mom-{day:%Y%m%d}-{symbol}-{'s' if side == 'sell' else 'b'}"
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


def submit_market(client: AlpacaClient, symbol: str, side: str, coid: str,
                  qty: Optional[float] = None, notional: Optional[float] = None):
    """Market DAY order. Sells use fractional ``qty``; buys use ``notional``. Idempotent on ``coid``:
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
                    logger.error("momentum_sleeve: order %s still open after %.0fs -- state NOT updated for it",
                                 coid, timeout)
                    out[coid] = {'status': 'timeout', 'qty': 0.0, 'avg': 0.0}
                break
            time.sleep(interval)
    return out


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
    """Shadow term-structure gate at ``asof``; None (WARNING) on any fetch / parse / history failure.

    It never blocks or changes the rebalance: nothing in order building reads this."""
    try:
        return ms.term_structure_gate(fetch_cboe_close('VIX'), fetch_cboe_close('VIX3M'), asof)
    except Exception as e:                      # network, schema, empty file: the gate is informational only
        logger.warning("momentum_sleeve: shadow term-structure gate unavailable (%s) -- reported n/a", e)
        return None


def gate_label(info: Optional[Dict]) -> str:
    """'gate ON (p21)' / 'gate OFF (p55)' / 'gate n/a' for the log and the [MOM] Telegram line."""
    if not info:
        return 'gate n/a'
    return f"gate {'ON' if info['gate_on'] else 'OFF'} (p{info['percentile'] * 100:.0f})"


def append_shadow_gate(run_day: date, info: Optional[Dict], equity: float) -> None:
    """One row per run to the shadow-gate CSV (skipped, WARNING, when the gate is n/a)."""
    if not info:
        logger.warning("momentum_sleeve: no shadow-gate row written for %s (gate n/a)", run_day)
        return
    append_csv(SHADOW_GATE_PATH, SHADOW_GATE_FIELDS, {
        'run_date': str(run_day), 'asof': str(info['date']), 'vix': round(info['vix'], 4),
        'vix3m': round(info['vix3m'], 4), 'ratio': round(info['ratio'], 6),
        'percentile': round(info['percentile'], 4), 'gate_on': info['gate_on'], 'equity': round(equity, 2)})


def guard_line(events: Dict[str, tuple], top_unguarded: List[str]) -> str:
    """The per-run guard INFO line: guard-removed names among the unguarded top 40, with reason and event date."""
    hit = [f"{s} {events[s][0]} {events[s][1]}" for s in top_unguarded[:GUARD_TOP] if s in events]
    return "guard: " + ("; ".join(hit) if hit else "none in the top 40")


def build_plan(panel: pd.DataFrame, assets: pd.DataFrame, asof: date, n: int, state: Dict,
               marks: Optional[Dict[str, float]] = None):
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
    targets = ms.target_dollars(selected, equity, n)
    current = {s: q * prices[s] for s, q in state['positions'].items() if s in prices and q > 0}
    held_without_price = [s for s, q in state['positions'].items() if q > 0 and s not in prices]
    if held_without_price:
        logger.error("momentum_sleeve: held %s have no bar on %s -- cannot size their exit; handle manually",
                     held_without_price, asof)
    orders = ms.rebalance_orders(current, targets)
    return selected, feat, equity, targets, orders, prices


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
                    submit_market(client, sym, 'buy', coid, notional=o['notional'])
            except Exception as e:
                logger.error("momentum_sleeve: %s %s order %s REJECTED (%s) -- continuing with the rest",
                             phase, sym, coid, e)
                continue
            coids[coid] = sym
        res = poll_fills(client, list(coids)) if coids else {}
        for coid, sym in coids.items():
            r = res[coid]
            if r['qty'] <= 0:
                continue
            fills.append({'symbol': sym, 'side': phase, 'qty': r['qty'], 'avg_price': r['avg'],
                          'notional': r['qty'] * r['avg'], 'client_order_id': coid})
    return fills


def format_orders(selected: List[str], feat: pd.DataFrame, equity: float, orders: List[Dict]) -> str:
    """Human-readable target list + orders for the dry run."""
    lines = [f"sleeve equity ${equity:,.2f}; target per name ${equity / max(len(selected), 1):,.2f}", "TOP:"]
    for i, s in enumerate(selected, 1):
        lines.append(f"  {i:2d} {s:<6} signal {feat.loc[s, 'signal']:.3f}  close {feat.loc[s, 'close']:.2f}")
    lines.append(f"ORDERS ({len(orders)}), sells first:")
    for o in orders:
        lines.append(f"  {o['side'].upper():<4} {o['symbol']:<6} ${o['notional']:,.2f}"
                     f"{' (full exit)' if o['full_exit'] else ''}")
    return "\n".join(lines)


def notify(notifier: Optional[TelegramNotifier], msg: str) -> None:
    """ONE [MOM] Telegram message; a failure is a WARNING, never fatal."""
    if notifier is None:
        return
    try:
        notifier.send_message_sync(f"{TG_PREFIX} {msg}")
    except Exception as e:
        logger.warning("momentum_sleeve: Telegram send failed (non-fatal): %s", e)


def run(args, client: AlpacaClient, notifier: Optional[TelegramNotifier], now_utc: datetime) -> int:
    """Whole run; returns the process exit code."""
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    week_start = today - timedelta(days=today.weekday())
    week_sessions = trading_sessions(client, week_start, today)
    is_session = today in week_sessions
    if not args.force and not (is_session and is_rebalance_session(today, week_sessions)):
        print("not a rebalance session — no-op")
        return 0
    if args.submit:
        check_submit_window(now_et, is_session)
    sessions = trading_sessions(client, today - timedelta(days=14), today)
    asof = pd.Timestamp(args.asof).date() if args.asof else last_completed_session(sessions, now_et)
    state = load_state(STATE_PATH, args.equity_start)
    sync_state_from_broker(client, state, args.equity_start)
    if args.submit and state.get('last_rebalance') == str(today) and not args.force:
        print(f"already rebalanced on {today} — no-op")
        return 0

    assets = fetch_assets(client)
    cpath = cache_file(asof)
    if args.skip_fetch and os.path.exists(cpath):
        panel = pd.read_parquet(cpath)
        logger.info("momentum_sleeve: reusing cache %s (%d rows)", cpath, len(panel))
        lost: List[str] = []
    else:
        if args.skip_fetch:
            logger.warning("momentum_sleeve: --skip-fetch but %s missing -- fetching", cpath)
        panel, lost = fetch_panel(client, list(assets['symbol']), asof)
        if len(panel):
            write_cache_atomic(panel, asof)
    print(completeness_line(len(assets) + 1, panel, lost, asof))
    if panel.empty:
        logger.error("momentum_sleeve: empty panel -- aborting")
        return 1
    panel['bar_date'] = pd.to_datetime(panel['bar_date'])

    selected, feat, equity, targets, orders, prices = build_plan(panel, assets, asof, args.n, state,
                                                                 marks=broker_marks(client))
    gate = shadow_gate_info(pd.Timestamp(asof))
    gate_txt = gate_label(gate)
    logger.info("momentum_sleeve: %s  ratio %s", gate_txt,
                f"{gate['ratio']:.4f} (VIX {gate['vix']:.2f} / VIX3M {gate['vix3m']:.2f}, {gate['date']})" if gate
                else "unavailable")
    print(f"asof {asof}  today {today}  mode {'SUBMIT' if args.submit else 'DRY-RUN'}  {gate_txt}")
    print(format_orders(selected, feat, equity, orders))
    if not args.submit:
        return 0

    assert_paper_account(client)
    names_before = set(state['positions'])
    run_tag = f"f{now_utc:%H%M%S}" if args.force else ''
    fills = execute(client, orders, prices, state, today, run_tag)
    sync_state_from_broker(client, state, args.equity_start)
    opens = official_opens(client, [f['symbol'] for f in fills], today)
    turnover = 0.0
    for f in fills:
        op = opens.get(f['symbol'])
        slip = ''
        if op:
            raw = (f['avg_price'] / op - 1.0) * 1e4
            slip = round(raw if f['side'] == 'buy' else -raw, 1)
        append_csv(LEDGER_PATH, LEDGER_FIELDS, {
            'date': str(today), 'symbol': f['symbol'], 'side': f['side'], 'qty': f['qty'],
            'avg_price': f['avg_price'], 'notional': round(f['notional'], 2),
            'official_open': op if op else '', 'slip_bps_vs_open': slip,
            'client_order_id': f['client_order_id']})
        turnover += f['notional']
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
    append_shadow_gate(today, gate, eq_after)
    kill = f" KILL RULE: drawdown {dd:.1%} > 40% -> HALVE the sleeve" if dd < -DD_KILL else ""
    notify(notifier, f"{today} equity ${eq_after:,.0f} (dd {dd:.1%} from peak) names {len(state['positions'])} "
                     f"in {len(names_in)} out {len(names_out)} turnover ${turnover:,.0f} fills {len(fills)}/{len(orders)} {gate_txt}{kill}")
    return 0


def main() -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asof', help='signal date YYYY-MM-DD (default: last completed trading day)')
    ap.add_argument('--n', type=int, default=ms.DEFAULT_N)
    ap.add_argument('--equity-start', type=float, default=20000.0)
    ap.add_argument('--skip-fetch', action='store_true', help='reuse the cached parquet for --asof')
    ap.add_argument('--force', action='store_true', help='run on a non-rebalance day')
    ap.add_argument('--submit', action='store_true', help='place PAPER orders (default: dry-run)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = Config()   # loads .env
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
    sys.exit(main())
