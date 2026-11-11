#!/usr/bin/env python3
"""Hourly Telegram summary per strategy book (spec: docs/hourly_summary_spec_20261009.md).

ONE message per run with today's state of every book: ORB, TOM (QQQ, only when active), HOD, MOM and a single
LIVE line (owner's manual account, report-only: day P&L + equity, never positions).
The ORB paper account is SHARED with the TOM sleeve and with third-party (`micha-*`) orders: they are attributed by
ORDER ORIGIN (see `classify_order`), not by symbol, and third-party activity gets its own report-only EXT line.

READ-ONLY. This script never submits, cancels or modifies an order and never writes the DB. It reads, per account,
the account object (equity / last_equity / account number), today's CLOSED orders, the open positions and the 1W
daily portfolio history.

Numbers
  * Day P&L: HOD / MOM / LIVE = equity - last_equity of the account. ORB, TOM and EXT share one account, so they
    are split by ORDER ORIGIN: `tom-*` client ids = TOM, `orb-*` ids or an order id in the trades.db ORB ledger
    (strategy='orb', this week) = ORB, everything else = EXT (third party, report-only). Positions: ORB iff an open
    ORB ledger row or an ORB fill today, TOM iff a `tom-` fill this week on the symbol, else EXT. Realized from
    today's fills (buy VWAP vs sell VWAP x min(bought, sold) qty, `filled_qty` / `filled_avg_price` only, never
    `qty`) + `unrealized_intraday_pl` of the open positions. The account total is the cross-check; a WARNING is
    logged when |ORB + TOM + EXT - account| > $5 (typically a position opened on a previous day and closed today:
    its sell has no matching buy in today's orders).
  * Week to date: sum of the account's daily `profit_loss` from `get_portfolio_history(period='1W', timeframe='1D')`
    (HOD / MOM / LIVE-style single-book accounts). ORB and TOM share one account, so their week is computed from
    the orders API since Monday 00:00 ET (realized per symbol as above, split by order origin).
    Alpaca stamps each daily bar with the NEXT UTC date (verified 2026-10-09: the bar stamped Fri 10/9 holds the
    Thursday 10/8 session and its `equity` equals today's `last_equity`), so a bar stamped D belongs to the
    previous weekday; sessions Monday..yesterday come from the history, TODAY comes from equity - last_equity.
  * Day boundary = 00:00 ET of the (rehearsal) date, converted to UTC (spec said 00:00 UTC; that would also
    include yesterday's 20:00-24:00 ET after-hours orders).

Usage
    python3 scripts/hourly_summary.py                 # send ONE Telegram message (stdout if Telegram unset)
    python3 scripts/hourly_summary.py --dry-run       # print, do not send
    python3 scripts/hourly_summary.py --now-et "2026-10-09 12:00"   # header / day boundary for tests

Cron (installed by the owner, NOT by this script; 10:05-16:05 ET on EDT, shifts to 09:05-15:05 ET on EST):
    5 14-20 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/hourly_summary.py >> logs/hourly_summary.log 2>&1
"""
import argparse
import html
import logging
import os
import re
import sqlite3
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from trading.ramp_stage import ORB_DAY_ONE, ORB_DAY_ONE_SUFFIX  # noqa: E402  (ONE constant for ORB day one)

from config import Config                                          # noqa: E402
from data_sources.alpaca_client import AlpacaClient                # noqa: E402
from notifications.telegram_notifier import TelegramNotifier       # noqa: E402
from trading import momentum_live as mom_live                      # noqa: E402

logger = logging.getLogger("hourly_summary")

ET = ZoneInfo("America/New_York")
MAX_LINES = 25
ORB_STAGE_USD = 10_000.0          # ORB % is quoted on the $10K stage
CROSS_CHECK_TOLERANCE = 5.0       # |ORB + TOM + EXT - account day P&L| above this -> WARNING
MAX_OPEN_NAMES = 6                # open-position names listed per line
TOM_SYMBOL = "QQQ"                # legacy single-symbol TOM split; attribution is now by order origin
TOM_PREFIX = "tom-"               # TOM sleeve client_order_id prefix
ORB_PREFIX = "orb-"               # ORB add-on / unwind client_order_id prefix (entries carry plain-UUID ids)
TRADES_DB_PATH = os.path.join(ROOT, 'data', 'trades.db')
ORB_LEG_COLUMNS = ('sl_leg_id', 'tp_leg_id', 'exit_order_id')   # ledger columns added to the ORB order-id set if present
MINUS = "−"
ORDERS_LIMIT = 500
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# book -> (env key var, env secret var); the book order is the message order
BOOK_ENV = {
    "ORB": ("ALPACA_ORB_API_KEY", "ALPACA_ORB_API_SECRET"),
    "HOD": ("ALPACA_HOD_API_KEY", "ALPACA_HOD_API_SECRET"),
    "MOM": ("ALPACA_MOM_API_KEY", "ALPACA_MOM_API_SECRET"),
}
LIVE_ENV = ("ALPACA_API_KEY", "ALPACA_API_SECRET")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_money(x: float, plus: bool = True) -> str:
    """Whole-dollar money with a unicode minus: +$241 / −$79 / $0 (sign dropped when it rounds to zero)."""
    r = int(round(abs(x)))
    if r == 0:
        return "$0"
    sign = MINUS if x < 0 else ("+" if plus else "")
    return f"{sign}${r:,}"


def fmt_pct(pnl: float, base: float) -> str:
    """P&L as a percent of `base` with one decimal; '0.0%' when the base is not positive."""
    if base is None or base <= 0:
        return "0.0%"
    v = 100.0 * pnl / base
    if round(abs(v), 1) == 0:
        return "0.0%"
    return f"{MINUS if v < 0 else '+'}{abs(v):.1f}%"


def money_pct(x: float, base: Optional[float]) -> str:
    """'+$53 (+2.3%)': money plus its percent of `base`; the bare money when the base is missing or not positive
    (a percent with no base is a false label -- never '(0.0%)')."""
    if base is None or base <= 0:
        return fmt_money(x)
    return f"{fmt_money(x)} ({fmt_pct(x, base)})"


def fmt_k(x: float) -> str:
    """Compact market value: $19.5K / $850."""
    return f"${x / 1000:.1f}K" if abs(x) >= 1000 else f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Paper guard
# ---------------------------------------------------------------------------

class PaperGuardError(RuntimeError):
    """A strategy account failed the paper check (kept distinct from API faults, which are n/a + WARNING)."""


def assert_paper_client(client: AlpacaClient, account_number: str) -> None:
    """Refuse a strategy account that is not paper (defense in depth, copied from tom_sleeve.assert_paper_account).

    Checks the wrapper flag, the SDK base URL AND that the account number starts with "PA" (Alpaca paper
    accounts); a live key pasted into a strategy slot trips the third check. Raises PaperGuardError.
    """
    if not getattr(client, 'is_paper', False):
        raise PaperGuardError("AlpacaClient.is_paper is False - refusing to read a strategy book from a live account")
    base_url = getattr(getattr(client, 'trading_client', None), '_base_url', None)
    base_url_str = str(getattr(base_url, 'value', base_url) or '')
    if 'paper' not in base_url_str.lower():
        raise PaperGuardError(f"broker base URL does not look like paper ({base_url_str!r})")
    if not str(account_number or '').upper().startswith("PA"):
        raise PaperGuardError("account number does not start with 'PA' - not a paper account")


# ---------------------------------------------------------------------------
# Data access (read-only)
# ---------------------------------------------------------------------------

def _f(v, default: float = 0.0) -> float:
    """float() that maps None / '' to `default`."""
    if v is None or v == "":
        return default
    return float(v)


def _side(order) -> str:
    """'buy' / 'sell' from an order whose side may be an enum or a string."""
    raw = getattr(order, 'side', '')
    s = str(getattr(raw, 'value', raw)).lower()
    return 'buy' if 'buy' in s else ('sell' if 'sell' in s else s)


def fetch_account(client: AlpacaClient) -> Dict:
    """Account snapshot: equity, last_equity, account_number (read-only)."""
    a = client.trading_client.get_account()
    return {'equity': _f(a.equity), 'last_equity': _f(a.last_equity),
            'account_number': str(getattr(a, 'account_number', '') or '')}


def fetch_filled_orders(client: AlpacaClient, after_utc: datetime) -> List[Dict]:
    """Today's CLOSED orders with filled_qty > 0 as plain dicts (symbol, side, qty, price).

    Uses `filled_qty` / `filled_avg_price` (a partially filled, then cancelled order is CLOSED with its filled
    part); unfilled orders are dropped. A full page (500) logs a WARNING because older fills may be missing.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    orders = client.trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after_utc, limit=ORDERS_LIMIT))
    if len(orders) >= ORDERS_LIMIT:
        logger.warning("hourly_summary: orders page full (%d) - today's fills may be truncated", len(orders))
    out = []
    for o in orders:
        qty = _f(getattr(o, 'filled_qty', None))
        if qty <= 0:
            continue
        out.append({'symbol': str(o.symbol), 'side': _side(o), 'qty': qty,
                    'price': _f(getattr(o, 'filled_avg_price', None)),
                    'order_id': str(getattr(o, 'id', '') or ''),
                    'client_order_id': str(getattr(o, 'client_order_id', '') or ''),
                    'filled_date': _filled_date_et(getattr(o, 'filled_at', None))})
    return out


def _filled_date_et(filled_at) -> str:
    """ISO ET date of an order's fill time ('' when the broker object carries none) - the ORB day-one filter key."""
    if filled_at is None:
        return ''
    try:
        return filled_at.astimezone(ET).date().isoformat()
    except (AttributeError, ValueError, OSError):
        logger.warning("hourly_summary: unreadable filled_at %r - fill date unknown", filled_at)
        return ''


def since_orb_day_one(fills: List[Dict]) -> List[Dict]:
    """ORB fills dated before ORB_DAY_ONE are pre-parity (A2) and dropped; an unknown date is kept (and was warned about)."""
    day_one = ORB_DAY_ONE.isoformat()
    return [f for f in fills if not f.get('filled_date') or f['filled_date'] >= day_one]


def fetch_positions(client: AlpacaClient) -> List[Dict]:
    """Open positions as plain dicts: symbol, qty, market_value, cost_basis, unrealized_pl, intraday_pl."""
    out = []
    for p in client.trading_client.get_all_positions():
        out.append({'symbol': str(p.symbol), 'qty': _f(p.qty), 'market_value': _f(p.market_value),
                    'cost_basis': _f(getattr(p, 'cost_basis', None)),
                    'unrealized_pl': _f(getattr(p, 'unrealized_pl', None)),
                    'intraday_pl': _f(getattr(p, 'unrealized_intraday_pl', None))})
    return out


def fetch_history(client: AlpacaClient, period: str = '1W') -> List[Dict]:
    """`period` / 1D portfolio history as [{'stamp': UTC date, 'profit_loss': float, 'equity': float}] (read-only)."""
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    h = client.trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe='1D'))
    out = []
    for ts, pl, eq in zip(h.timestamp or [], h.profit_loss or [], h.equity or []):
        out.append({'stamp': datetime.fromtimestamp(ts, timezone.utc).date(),
                    'profit_loss': _f(pl), 'equity': _f(eq)})
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def load_orb_ledger(today: date, db_path: Optional[str] = None) -> (set, set):
    """(order ids of ORB rows since this week's Monday, symbols of open ORB rows) from trades.db, opened READ-ONLY.

    Bracket-leg id columns (ORB_LEG_COLUMNS) are included when the table has them. A missing / unreadable ledger
    logs an ERROR and returns empty sets: every fill then reports as EXT instead of silently as ORB.
    """
    path = db_path or TRADES_DB_PATH
    monday = (today - timedelta(days=today.weekday())).isoformat()
    ids: set = set()
    open_syms: set = set()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
            legs = [c for c in ORB_LEG_COLUMNS if c in cols]
            sel = ", ".join(["order_id"] + legs)
            for row in conn.execute(f"SELECT {sel} FROM trades WHERE strategy='orb' AND trade_date >= ?", (monday,)):
                ids.update(str(v) for v in row if v)
            for (sym,) in conn.execute("SELECT symbol FROM trades WHERE strategy='orb' AND order_status='filled' "
                                       "AND exit_price IS NULL"):
                open_syms.add(str(sym))
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error("hourly_summary: ORB ledger %s unreadable (%s) - ALL fills will be attributed EXT", path, e)
    return ids, open_syms


def classify_order(client_order_id: str, order_id: str, orb_order_ids) -> str:
    """Origin of one order: 'TOM' (`tom-` client id), 'ORB' (`orb-` client id or order id in the ORB ledger), else 'EXT'."""
    cid = client_order_id or ''
    if cid.startswith(TOM_PREFIX):
        return 'TOM'
    if cid.startswith(ORB_PREFIX) or (order_id and order_id in orb_order_ids):
        return 'ORB'
    return 'EXT'


def classify_position(symbol: str, fills_by_book: Dict[str, set], open_orb_symbols) -> str:
    """Owner of an open position: ORB iff an open ORB ledger row or an ORB fill today, TOM iff a `tom-` fill (today
    or this week) on the symbol, else EXT. `fills_by_book` = {'ORB': symbols filled today, 'TOM': symbols filled this week}."""
    if symbol in open_orb_symbols or symbol in fills_by_book.get('ORB', ()):
        return 'ORB'
    if symbol in fills_by_book.get('TOM', ()):
        return 'TOM'
    return 'EXT'


def ext_prefixes(fills: List[Dict]) -> List[str]:
    """Distinct client-id prefixes ('micha-hf') of the EXT fills, for the INFO log; '(none)' for an empty id."""
    out = set()
    for f in fills:
        m = re.match(r'[A-Za-z_]+(?:-[A-Za-z_]{2,})*', f.get('client_order_id') or '')
        out.add(m.group(0) if m else '(none)')
    return sorted(out)


def split_books(orb: Dict) -> (Dict[str, Dict], Dict[str, Dict]):
    """Split the shared ORB account into ORB / TOM / EXT by order origin.

    Returns (day_stats, week_stats), each {'ORB'|'TOM'|'EXT': book_stats}. Fills are classified per order; the
    positions once (week fills decide TOM ownership, today's fills + the ledger decide ORB ownership).
    """
    ids, open_syms = orb.get('orb_order_ids', set()), orb.get('open_orb_symbols', set())

    def tag(fills: List[Dict]) -> Dict[str, List[Dict]]:
        by = {'ORB': [], 'TOM': [], 'EXT': []}
        for f in fills:
            by[classify_order(f.get('client_order_id', ''), f.get('order_id', ''), ids)].append(f)
        return by

    day, week = tag(orb['fills']), tag(orb['week_fills'])
    day['ORB'], week['ORB'] = since_orb_day_one(day['ORB']), since_orb_day_one(week['ORB'])
    fills_by_book = {'ORB': {f['symbol'] for f in day['ORB']},
                     'TOM': {f['symbol'] for f in week['TOM']} | {f['symbol'] for f in day['TOM']}}
    pos_by = {'ORB': [], 'TOM': [], 'EXT': []}
    for p in orb['positions']:
        pos_by[classify_position(p['symbol'], fills_by_book, open_syms)].append(p)
    if day['EXT']:
        logger.info("hourly_summary: EXT prefixes: %s", ", ".join(ext_prefixes(day['EXT'])))
    every = lambda s: True
    return ({b: book_stats(b, day[b], pos_by[b], every) for b in day},
            {b: book_stats(b, week[b], pos_by[b], every) for b in week})


def per_symbol_fills(fills: List[Dict]) -> Dict[str, Dict]:
    """Aggregate fills per symbol: bought / sold qty and notional, fills, realized $ on min(bought, sold) qty."""
    agg: Dict[str, Dict] = {}
    for f in fills:
        s = agg.setdefault(f['symbol'], {'buy_qty': 0.0, 'buy_val': 0.0, 'sell_qty': 0.0, 'sell_val': 0.0,
                                         'fills': 0, 'realized': 0.0, 'closed_qty': 0.0})
        s['fills'] += 1
        key = 'buy' if f['side'] == 'buy' else 'sell'
        s[f'{key}_qty'] += f['qty']
        s[f'{key}_val'] += f['qty'] * f['price']
    for s in agg.values():
        closed = min(s['buy_qty'], s['sell_qty'])
        s['closed_qty'] = closed
        if closed > 0:
            s['realized'] = (s['sell_val'] / s['sell_qty'] - s['buy_val'] / s['buy_qty']) * closed
    return agg


def session_date(stamp: date) -> date:
    """The trading session a daily history bar holds: Alpaca stamps the NEXT UTC date, so stamp minus one weekday."""
    sess = stamp - timedelta(days=1)
    while sess.weekday() >= 5:
        sess -= timedelta(days=1)
    return sess


def week_to_date(history: List[Dict], today: date, day_pnl_today: float) -> float:
    """Sessions Monday..yesterday from the history + today's account day P&L.

    A bar stamped D holds the previous weekday's session (Alpaca stamps the NEXT UTC date), so the session date
    is D minus one weekday; only sessions >= this week's Monday and < today count.
    """
    monday = today - timedelta(days=today.weekday())
    total = 0.0
    for bar in history:
        sess = session_date(bar['stamp'])
        if monday <= sess < today:
            total += bar['profit_loss']
    return total + day_pnl_today


def check_history_alignment(name: str, history: List[Dict], last_equity: float) -> bool:
    """The last history bar must close at the account's last_equity (it holds YESTERDAY's session).

    Returns False and logs a WARNING when it does not (Alpaca changed its stamping, or the history is stale), in
    which case the week-to-date number cannot be trusted.
    """
    if not history:
        logger.warning("hourly_summary: %s portfolio history is empty - week to date unavailable", name)
        return False
    if abs(history[-1]['equity'] - last_equity) > 1.0:
        logger.warning("hourly_summary: %s last history bar equity %.2f != account last_equity %.2f - "
                       "week-to-date date stamping may be off", name, history[-1]['equity'], last_equity)
        return False
    return True


def book_stats(name: str, fills: List[Dict], positions: List[Dict], include) -> Dict:
    """Per-book numbers for the symbols where include(symbol) is True: fills, realized, open, winners/losers."""
    sel_fills = [f for f in fills if include(f['symbol'])]
    sel_pos = [p for p in positions if include(p['symbol'])]
    agg = per_symbol_fills(sel_fills)
    realized = sum(s['realized'] for s in agg.values())
    closed = {sym: s['realized'] for sym, s in agg.items() if s['closed_qty'] > 0}
    closed_notional = {sym: s['buy_val'] for sym, s in agg.items() if s['closed_qty'] > 0}
    return {
        'name': name,
        'n_fills': len(sel_fills),
        'bought_notional': sum(s['buy_val'] for s in agg.values()),
        'realized': realized,
        'closed': closed,
        'closed_notional': closed_notional,
        'positions': sel_pos,
        'intraday_open': sum(p['intraday_pl'] for p in sel_pos),
        'open_pl': sum(p['unrealized_pl'] for p in sel_pos),
        'open_mv': sum(abs(p['market_value']) for p in sel_pos),
        'active': bool(sel_fills or sel_pos),
    }


def collect_book(name: str, client: AlpacaClient, after_utc: datetime, today: date,
                 db_path: Optional[str] = None) -> Dict:
    """Read one strategy account and return its raw data (account, fills, positions, history).

    The ORB account is shared (TOM + third-party orders): its trades.db ledger (order ids of this week's ORB rows,
    symbols of open ORB rows) is attached as `orb_order_ids` / `open_orb_symbols` for the order-origin split.
    """
    acct = fetch_account(client)
    assert_paper_client(client, acct['account_number'])
    history = fetch_history(client)
    check_history_alignment(name, history, acct['last_equity'])
    fills = fetch_filled_orders(client, after_utc)
    try:
        week_fills = fetch_filled_orders(client, week_start_utc(today))
    except Exception as e:
        if name == 'ORB':    # the ORB / TOM week split cannot exist without the week's fills: the book goes n/a
            raise
        logger.warning("hourly_summary: %s week fills fetch failed (%s) - its wk figure stays bare $", name, e)
        week_fills = None
    data = {'account': acct, 'fills': fills, 'week_fills': week_fills,
            'positions': fetch_positions(client), 'history': history}
    if name == 'ORB':
        data['orb_order_ids'], data['open_orb_symbols'] = load_orb_ledger(today, db_path)
    return data


def collect_mom_live(client: AlpacaClient, today: date, state_path: Optional[str] = None,
                     weekly_path: Optional[str] = None) -> Optional[Dict]:
    """MOM-LIVE inputs: the sleeve STATE plus marks of the state's own names ONLY (the broker positions are
    filtered inside ``sleeve_marks`` -- no other position is kept, counted or logged). None when no state file."""
    state = mom_live.read_json(state_path or mom_live.LIVE_STATE_PATH)
    if state is None:
        return None
    marks, prev = mom_live.sleeve_marks(client.trading_client.get_all_positions(), state)
    return {'state': state, 'marks': marks, 'prev': prev, 'day': today,
            'weekly': mom_live.read_weekly(weekly_path or mom_live.LIVE_WEEKLY_PATH)}


def collect_live(client: AlpacaClient) -> Dict:
    """LIVE (owner's manual account): account object only - no orders, no positions are ever read."""
    return {'account': fetch_account(client)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _open_names(positions: List[Dict], key: str = 'intraday_pl') -> str:
    """'open N: SYM +x SYM −y ...' (largest |P&L| first, MAX_OPEN_NAMES then '+k more'); 'flat' when empty."""
    if not positions:
        return "flat"
    ranked = sorted(positions, key=lambda p: -abs(p[key]))
    names = " ".join(f"{html.escape(p['symbol'])} {money_pct(p[key], abs(p.get('cost_basis') or 0.0))}"
                     for p in ranked[:MAX_OPEN_NAMES])
    extra = len(ranked) - MAX_OPEN_NAMES
    return f"open {len(positions)}: {names}" + (f" +{extra} more" if extra > 0 else "")


def _best_worst(closed: Dict[str, float], notional: Optional[Dict[str, float]] = None) -> str:
    """'best X +n (+p%)' only when a closed winner exists, 'worst Y −n (−p%)' only when a closed loser exists;
    the percent is on that symbol's bought notional (`notional`), omitted when it is unknown."""
    notional = notional or {}
    parts = []
    if closed:
        best = max(closed, key=closed.get)
        worst = min(closed, key=closed.get)
        if closed[best] > 0:
            parts.append(f"best {html.escape(best)} {money_pct(closed[best], notional.get(best))}")
        if closed[worst] < 0:
            parts.append(f"worst {html.escape(worst)} {money_pct(closed[worst], notional.get(worst))}")
    return " | ".join(parts)


def format_book_line(label: str, day_pnl: float, base: Optional[float], stats: Dict, wk: Optional[float],
                     wk_suffix: str = "", wk_base: Optional[float] = None) -> str:
    """One ORB / HOD style line: day P&L (% of `base`), fills, open positions, best/worst closed, week to date
    (% of `wk_base`, bare $ when None)."""
    parts = [f"<b>{label}</b> {money_pct(day_pnl, base)}", f"{stats['n_fills']} fills", _open_names(stats['positions'])]
    bw = _best_worst(stats['closed'], stats.get('closed_notional'))
    if bw:
        parts.append(bw)
    line = " | ".join(parts)
    if wk is not None:
        line += f"   wk {money_pct(wk, wk_base)}{wk_suffix}"
    return line


def format_mom_line(day_pnl: float, equity: float, stats: Dict, wk: Optional[float]) -> str:
    """MOM (weekly book) line: day P&L on equity, open P&L on its cost basis (market value - open P&L), number of
    names, week to date on the start-of-week equity (equity - wk)."""
    open_cost = stats['open_mv'] - stats['open_pl']
    line = (f"<b>MOM</b> {money_pct(day_pnl, equity)} | open {money_pct(stats['open_pl'], open_cost)}"
            f" on {fmt_k(stats['open_mv'])}, {len(stats['positions'])} names")
    if wk is not None:
        line += f"   wk {money_pct(wk, equity - wk)}"
    return line


def format_tom_line(stats: Dict, wk: Optional[float] = None, wk_base: Optional[float] = None) -> str:
    """TOM (QQQ) line: day P&L on the QQQ cost basis, fills, QQQ position, week to date."""
    pnl = stats['realized'] + stats['intraday_open']
    cost = sum(abs(p['cost_basis']) for p in stats['positions']) or stats['bought_notional']
    pos = "flat"
    if stats['positions']:
        pos = "hold " + " ".join(f"{html.escape(p['symbol'])} {p['qty']:g}" for p in stats['positions'])
    line = f"<b>TOM</b> {money_pct(pnl, cost)} | {stats['n_fills']} fills | {pos}"
    return line + (f"   wk {money_pct(wk, wk_base or cost)}" if wk is not None else "")


def format_ext_line(stats: Dict) -> str:
    """EXT (third-party `micha-*` orders on the ORB account) line: report-only, day P&L, fills, open symbols."""
    pnl = stats['realized'] + stats['intraday_open']
    names = " ".join(html.escape(p['symbol']) for p in stats['positions'])
    pos = f"open {len(stats['positions'])}: {names}" if stats['positions'] else "flat"
    return f"<b>EXT (micha)</b> {money_pct(pnl, stats['bought_notional'])} | {stats['n_fills']} fills | {pos}"


def na_line(label: str, why: str) -> str:
    """Line for a book whose account could not be read; no exception text goes into the message."""
    return f"<b>{label}</b> n/a ({why})"


def enforce_line_limit(lines: List[str], limit: int = MAX_LINES) -> List[str]:
    """Hard cap on message length: keep the header and the first limit-2 lines, then a truncation note."""
    if len(lines) <= limit:
        return lines
    return lines[:limit - 1] + [f"... {len(lines) - limit + 1} lines cut"]


def build_message(now_et: datetime, data: Dict[str, Optional[Dict]], errors: Dict[str, str]) -> str:
    """Compose the single HTML message from per-book raw data (None = book failed, reason in `errors`).

    data keys: 'ORB' (also feeds TOM and EXT), 'HOD', 'MOM', 'LIVE'. Logs a WARNING when ORB + TOM + EXT disagrees with the
    shared account's day P&L by more than CROSS_CHECK_TOLERANCE.
    """
    today = now_et.date()
    header = f"\U0001F4CA <b>HOURLY {now_et:%H:%M} ET ({WEEKDAY_NAMES[today.weekday()]} {today.month}/{today.day})</b>"
    lines: List[str] = []

    orb = data.get('ORB')
    if orb is None:
        lines.append(na_line("ORB", errors.get('ORB', 'api error')))
    else:
        acct = orb['account']
        acct_day = acct['equity'] - acct['last_equity']
        day_s, week_s = split_books(orb)
        orb_s, tom_s, ext_s = day_s['ORB'], day_s['TOM'], day_s['EXT']
        orb_pnl = orb_s['realized'] + orb_s['intraday_open']
        tom_pnl = tom_s['realized'] + tom_s['intraday_open']
        ext_pnl = ext_s['realized'] + ext_s['intraday_open']
        if abs(orb_pnl + tom_pnl + ext_pnl - acct_day) > CROSS_CHECK_TOLERANCE:
            logger.warning("hourly_summary: ORB+TOM+EXT split %.2f differs from account day P&L %.2f by more than $%.0f "
                           "(a position opened on a previous day and closed today has no buy in today's orders)",
                           orb_pnl + tom_pnl + ext_pnl, acct_day, CROSS_CHECK_TOLERANCE)
        # ORB, TOM and EXT share one account, so the account's portfolio history mixes them: week to date for each
        # is the realized P&L of the week's fills of its own orders + the open positions' intraday P&L.
        orb_wk_s, tom_wk_s = week_s['ORB'], week_s['TOM']
        lines.append(format_book_line(f"ORB {ORB_DAY_ONE_SUFFIX}", orb_pnl, ORB_STAGE_USD, orb_s,
                                      orb_wk_s['realized'] + orb_wk_s['intraday_open'], wk_base=ORB_STAGE_USD))
        if tom_s['active']:   # hidden unless a TOM position or a tom- fill today (week-only fills have no cost basis in the window)
            lines.append(format_tom_line(tom_s, tom_wk_s['realized'] + tom_wk_s['intraday_open']))
        if ext_s['active'] and (ext_pnl or ext_s['n_fills'] or ext_s['positions']):   # report-only, never a book
            lines.append(format_ext_line(ext_s))

    hod = data.get('HOD')
    if hod is None:
        lines.append(na_line("HOD", errors.get('HOD', 'api error')))
    else:
        acct = hod['account']
        day = acct['equity'] - acct['last_equity']
        st = book_stats('HOD', hod['fills'], hod['positions'], lambda s: True)
        wk_base = None
        if hod.get('week_fills') is None:
            logger.warning("hourly_summary: HOD week fills unavailable - wk figure stays bare $ (no notional base)")
        else:
            wk_base = book_stats('HOD', hod['week_fills'], hod['positions'], lambda s: True)['bought_notional']
        lines.append(format_book_line("HOD", day, st['bought_notional'], st,
                                      week_to_date(hod['history'], today, day), wk_base=wk_base))

    mom = data.get('MOM')
    if mom is None:
        lines.append(na_line("MOM", errors.get('MOM', 'api error')))
    else:
        acct = mom['account']
        day = acct['equity'] - acct['last_equity']
        st = book_stats('MOM', mom['fills'], mom['positions'], lambda s: True)
        lines.append(format_mom_line(day, acct['equity'], st, week_to_date(mom['history'], today, day)))

    live = data.get('LIVE')
    if live is None:
        lines.append(na_line("LIVE", errors.get('LIVE', 'api error')))
    else:
        a = live['account']
        lines.append(f"<b>LIVE</b> {money_pct(a['equity'] - a['last_equity'], a['last_equity'])} | equity ${a['equity']:,.0f}")

    if 'MOM-LIVE' in data:   # absent state file = the sleeve was never live = line omitted (not NO-DATA)
        ml_data = data['MOM-LIVE']
        if ml_data is None:
            lines.append(na_line("MOM-LIVE", errors.get('MOM-LIVE', 'api error')))
        else:
            lines.append("<b>" + html.escape(mom_live.mom_live_line(
                ml_data['state'], ml_data['marks'], ml_data['prev'], ml_data['weekly'], ml_data['day'])) + "</b>")

    return header + "\n" + "\n".join(enforce_line_limit(lines, MAX_LINES - 1))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def gather(cfg: Config, after_utc: datetime, today: date) -> (Dict, Dict):
    """Read every book; a failing book is logged at WARNING and reported as n/a, the others still go out."""
    data: Dict[str, Optional[Dict]] = {}
    errors: Dict[str, str] = {}
    for name, (kvar, svar) in BOOK_ENV.items():
        key, secret = os.environ.get(kvar, ''), os.environ.get(svar, '')
        if not key or not secret:
            logger.warning("hourly_summary: %s/%s not set - %s line n/a", kvar, svar, name)
            data[name], errors[name] = None, "no keys"
            continue
        try:
            data[name] = collect_book(name, AlpacaClient(key, secret, paper=True), after_utc, today)
        except PaperGuardError as e:
            logger.error("hourly_summary: %s refused by the paper guard: %s", name, e)
            data[name], errors[name] = None, "not paper"
        except Exception as e:
            logger.warning("hourly_summary: %s read failed: %s", name, e)
            data[name], errors[name] = None, "api error"
    key, secret = os.environ.get(LIVE_ENV[0], ''), os.environ.get(LIVE_ENV[1], '')
    if not key or not secret:
        logger.warning("hourly_summary: %s/%s not set - LIVE line n/a", *LIVE_ENV)
        data['LIVE'], errors['LIVE'] = None, "no keys"
    else:
        try:
            live_client = AlpacaClient(key, secret, paper=False)
            data['LIVE'] = collect_live(live_client)
            if os.path.exists(mom_live.LIVE_STATE_PATH):
                try:
                    data['MOM-LIVE'] = collect_mom_live(live_client, today)
                except Exception as e:
                    logger.warning("hourly_summary: MOM-LIVE read failed: %s", e)
                    data['MOM-LIVE'], errors['MOM-LIVE'] = None, "api error"
        except Exception as e:
            logger.warning("hourly_summary: LIVE read failed: %s", e)
            data['LIVE'], errors['LIVE'] = None, "api error"
    return data, errors


def parse_now_et(arg: Optional[str]) -> datetime:
    """Current ET time, or the `--now-et 'YYYY-MM-DD HH:MM'` rehearsal time."""
    if arg:
        return datetime.strptime(arg, "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return datetime.now(ET)


def week_start_utc(today: date) -> datetime:
    """Monday 00:00 ET of today's week, in UTC (lower bound of the week-to-date orders query)."""
    monday = today - timedelta(days=today.weekday())
    return datetime.combine(monday, dtime(0, 0), tzinfo=ET).astimezone(timezone.utc)


def day_start_utc(now_et: datetime) -> datetime:
    """00:00 ET of now_et's date, in UTC (lower bound of 'today's orders')."""
    return datetime.combine(now_et.date(), dtime(0, 0), tzinfo=ET).astimezone(timezone.utc)


def deliver(message: str, notifier: Optional[TelegramNotifier], dry_run: bool) -> Optional[bool]:
    """Send the message (True/False = send_message_sync result) or print it (dry run / Telegram unset -> None)."""
    if dry_run:
        print(message)
        return None
    if notifier is None:
        logger.warning("hourly_summary: Telegram not configured (TELEGRAM_BOT_TOKEN/CHAT_ID empty) - printing")
        print(message)
        return None
    ok = notifier.send_message_sync(message)
    if ok:
        logger.info("hourly_summary: Telegram send_message_sync returned True")
    else:
        logger.error("hourly_summary: Telegram send_message_sync returned %s - message NOT delivered", ok)
    return bool(ok)


def main() -> int:
    """CLI entry. Exit 0 after any send attempt; non-zero only on a programming error."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help="print the message, do not send")
    ap.add_argument('--once', action='store_true', help="single run (the default; accepted for cron clarity)")
    ap.add_argument('--now-et', help="header / day boundary override: 'YYYY-MM-DD HH:MM' ET")
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = Config()   # loads .env
    now_et = parse_now_et(args.now_et)
    data, errors = gather(cfg, day_start_utc(now_et), now_et.date())
    message = build_message(now_et, data, errors)
    notifier = None
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
    deliver(message, notifier, args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
