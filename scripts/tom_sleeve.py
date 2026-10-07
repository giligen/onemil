#!/usr/bin/env python3
"""Turn-of-month index sleeve — PAPER execution rehearsal only.

Cell 1,651 (`research/index_overnight/PREREG_1649.md`, judged in `RESULT_1649.md`): buy MOC on the
last NYSE trading session of the calendar month, sell MOC on the third trading session of the next
month, SPY/QQQ/IWM, $20,000 notional each. The cell PASSED the frozen letter of the PREREG (2016-23
pooled +32.1 bps/event, 6/8 years and 3/3 ETFs positive) but 83% of the mean is carried by the top 5%
of events and the ex-top-5% return is BELOW four ordinary nights (RESULT_1649.md, "Adequacy and the
tail"). Per the PREREG's frozen consequence: PAPER MOC/MOO legs as an execution rehearsal from the
next event; capital only on the owner's word with the tail line in front of him. This script is that
rehearsal — it must never place a real order.

Runs ONLY on the PAPER account keyed by ALPACA_ORB_API_KEY / ALPACA_ORB_API_SECRET
(ALPACA_ORB_PAPER=true in .env) and refuses to run against anything else (`assert_paper_account`).
Idempotent by client_order_id ('tom-<YYYYMM>-<SYM>-in' / 'tom-<YYYYMM>-<SYM>-out', <YYYYMM> = the
month the action itself falls in) and by the CSV ledger's notion of an open sleeve position — safe to
invoke repeatedly, from cron, in the same window, or on both daily UTC cron ticks across DST.

2026-09-30 entry incident (first-ever entry) and the fix: the 19:45 UTC (15:45 ET) tick submitted
three MOC BUYs (TimeInForce.CLS) at 19:45:04-05 UTC. All three were ACCEPTED at submission (no
reject — 5 min inside Alpaca's documented "CLS orders submitted after 3:50pm ET are rejected" rule,
https://docs.alpaca.markets/us/docs/orders-at-alpaca, and inside every cited exchange cutoff: NYSE
15:50 ET, Nasdaq 15:55 ET, NYSE Arca 15:59 ET per alpaca.markets/learn/13-order-types-you-should-
know-about and the NYSE closing-auction fact sheet — submission timing is therefore RULED OUT as the
cause). QQQ (Nasdaq-listed) got a closing-cross print and FILLED @ 739.55 at 19:59:59.73 UTC. SPY and
IWM (NYSE Arca-listed) got no print, sat as open orders, and were marked EXPIRED ~30s AFTER the close
(20:00:29.96 / 20:00:37.27 UTC respectively) — confirmed by reading the raw order objects (status,
submitted_at, expired_at) via the read-only Alpaca API. Alpaca's own docs only say "any unfilled
order after the close will be cancelled" with no venue detail, so the exact internal reason the Arca
leg's auction print never arrived (paper-simulation gap vs a real Arca-side quirk) is NOT confirmed
with Alpaca support — this is N=1. Consequence, since the failure is AT the close, not submission:
MOC/CLS stays for QQQ (empirically reliable); SPY/IWM switch to a marketable DAY limit (NOT CLS, so
it does not depend on the closing auction at all) submitted in a narrow LATE_WINDOW just before the
close so the fill still tracks the closing price closely. See MOC_RELIABLE_SYMBOLS /
FALLBACK_LIMIT_SYMBOLS below.

2026-10-06 exit incident (docs/tom_sleeve_partial_fill_fix_20261007.md): the catch-up MOC SELL
`tom-202610-QQQ-out-r1` (26 QQQ) filled 25 of 26 @ 759.76 and Alpaca closed it `expired` with
filled_qty=25; the fill check read only `status`, booked "unfilled", and the 19:58 UTC tick never ran the
cancel+market fallback because 10/6 is not a turn-of-month session. Fixes: the fill check reads
filled_qty (ledger status `partial_fill`, qty = the filled part, remainder stays owed); the owed quantity
is the BROKER's position (capped by the ledger's open qty); any live tom exit order gets the late
fallback on any session (catch-up days included); `--reconcile` backfills a missing partial_fill row
from the orders API (ledger write only, never an order).

Usage:
    python3 scripts/tom_sleeve.py              # act only if now is in the MOC window (15:40-15:55 ET,
                                                 # QQQ) or the late fallback window (15:57-15:59 ET,
                                                 # SPY/IWM) on an entry/exit day; otherwise checks
                                                 # pending fills and exits 0
    python3 scripts/tom_sleeve.py --dry-run     # print intended actions, submit nothing, no Telegram
    python3 scripts/tom_sleeve.py --status      # print the ledger + current open sleeve, no orders
    python3 scripts/tom_sleeve.py --reconcile   # append any missing partial_fill ledger row (no orders)

Cron (added by the main session, NOT by this script): run FOUR times daily on weekdays so BOTH
15:45 ET (MOC window) and 15:58 ET (late fallback window) each fall inside one of the ticks in both
DST regimes (EDT = UTC-4, EST = UTC-5) — the script itself decides whether "now" is actually in a
window, so a miss on any tick is a harmless no-op; the post-close tick (the OTHER hour's :45) is also
when check_pending_fills reports final status/reason for the day's entries:
    45,58 19,20 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/tom_sleeve.py >> logs/tom_sleeve_cron.log 2>&1
"""
import argparse
import calendar as calendar_mod
import csv
import logging
import os
import re
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config                                          # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError  # noqa: E402
from notifications.telegram_notifier import TelegramNotifier        # noqa: E402

logger = logging.getLogger('tom_sleeve')

SYMBOLS = ('SPY', 'QQQ', 'IWM')
NOTIONAL_PER_SYMBOL = 20_000.0
MIN_BUYING_POWER = 70_000.0
ET = ZoneInfo('US/Eastern')
WINDOW_START = dtime(15, 40)
WINDOW_END = dtime(15, 55)
# 2026-09-30 evidence (module docstring): QQQ's MOC/CLS order got a Nasdaq closing-cross print and
# filled; SPY's and IWM's did not and expired ~30s after the close. MOC/CLS stays ONLY for the symbol
# it is proven to work for; the other two use the fallback marketable-limit leg in LATE_WINDOW.
MOC_RELIABLE_SYMBOLS = ('QQQ',)
FALLBACK_LIMIT_SYMBOLS = ('SPY', 'IWM')
LATE_WINDOW_START = dtime(15, 57)
LATE_WINDOW_END = dtime(15, 59)
FALLBACK_LIMIT_BUFFER = 0.003  # marketable cushion (30 bps) over the latest trade so the DAY limit crosses the spread
LEDGER_PATH = os.path.join(ROOT, 'logs', 'tom_sleeve_ledger.csv')
LEDGER_FIELDS = ['date', 'action', 'symbol', 'qty', 'ref_price', 'order_id', 'status',
                  'client_order_id', 'timestamp_utc', 'partial_entry', 'deviation']
TG_PREFIX = '[TOM]'
_sleep = time.sleep  # indirection so tests can skip the cancel-confirm poll
LIVE_STATUSES = ('new', 'accepted', 'pending_new', 'partially_filled', 'pending_cancel', 'pending_replace',
                 'accepted_for_bidding', 'held')  # order still working at the broker: blocks a resubmit
DEAD_STATUSES = ('expired', 'canceled', 'rejected')  # terminal WITHOUT a fill: the leg is still owed
PARTIAL_STATUS = 'partial_fill'  # ledger fill-check status: terminal order, 0 < filled_qty < qty; remainder owed
CANCEL_POLL_SECONDS = 10
_ORDER_ACTIONS = ('entry', 'exit', 'late_fallback_replace')
ACTIVE_STATUSES = ('submitted', 'filled')  # ledger statuses that count as "this leg is in effect"


# ---------------------------------------------------------------------------
# Calendar: turn-of-month entry/exit day detection
# ---------------------------------------------------------------------------

def _to_date(v) -> date:
    """Normalize an Alpaca calendar 'date' field (date or datetime) to a plain date."""
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    return v.date() if isinstance(v, datetime) else v


def _month_bounds(year: int, month: int) -> (date, date):
    """First and last calendar-day of a given (year, month)."""
    first = date(year, month, 1)
    last_day = calendar_mod.monthrange(year, month)[1]
    return first, date(year, month, last_day)


def month_trading_sessions(alpaca_client: AlpacaClient, year: int, month: int) -> List[date]:
    """All NYSE trading sessions inside one calendar month, ascending.

    Returns [] and logs ERROR on an API failure or an empty month — a real calendar month always
    has multiple NYSE sessions, so an empty result means the calendar call itself is broken, not
    that there is nothing to trade.
    """
    first, last = _month_bounds(year, month)
    try:
        cal = alpaca_client.get_market_calendar(first, last)
    except AlpacaAPIError as e:
        logger.error(f"tom_sleeve: get_market_calendar({year}-{month:02d}) failed: {e}")
        return []
    sessions = sorted(_to_date(d['date']) for d in cal)
    if not sessions:
        logger.error(f"tom_sleeve: {year}-{month:02d} has zero trading sessions per Alpaca — "
                      f"treating as a calendar API problem, not a real empty month")
    return sessions


def classify_day(today: date, alpaca_client: AlpacaClient) -> str:
    """'entry' if `today` is the LAST trading session of its month, 'exit' if it is the THIRD
    trading session of its month (which closes the sleeve opened on the prior month's entry),
    else 'none'. One calendar call per invocation.
    """
    sessions = month_trading_sessions(alpaca_client, today.year, today.month)
    if not sessions:
        return 'none'
    if today == sessions[-1]:
        return 'entry'
    if len(sessions) >= 3 and today == sessions[2]:
        return 'exit'
    return 'none'


def in_action_window(now_et: datetime) -> bool:
    """True iff the ET wall-clock time falls in the 15:40-15:55 MOC submission window
    (MOC_RELIABLE_SYMBOLS only — see module docstring, 2026-09-30 SPY/IWM incident)."""
    return WINDOW_START <= now_et.time() <= WINDOW_END


def in_late_window(now_et: datetime) -> bool:
    """True iff the ET wall-clock time falls in the 15:57-15:59 late fallback window
    (FALLBACK_LIMIT_SYMBOLS only): a marketable DAY limit submitted just before the close so the
    fill still tracks the closing price without depending on the closing auction."""
    return LATE_WINDOW_START <= now_et.time() <= LATE_WINDOW_END


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def assert_paper_account(alpaca_client: AlpacaClient) -> None:
    """Refuse to proceed unless this is unambiguously the paper account.

    Defense in depth: checks both the wrapper's own `is_paper` flag (config-driven — could be
    wrong if ALPACA_ORB_PAPER is misset) and the broker SDK's actual base URL (so a bad flag alone
    can't route this script at live money). Raises RuntimeError; callers MUST NOT catch this and
    continue — it is the one guard nothing else backstops.
    """
    if not getattr(alpaca_client, 'is_paper', False):
        raise RuntimeError("AlpacaClient.is_paper is False — refusing to run against a live account")
    base_url = getattr(getattr(alpaca_client, 'trading_client', None), '_base_url', None)
    base_url_str = str(getattr(base_url, 'value', base_url) or '')
    if 'paper' not in base_url_str.lower():
        raise RuntimeError(f"broker base URL does not look like paper ({base_url_str!r}) — refusing to run")


def assert_buying_power(alpaca_client: AlpacaClient, minimum: float = MIN_BUYING_POWER) -> float:
    """Refuse to enter new positions if the paper account's buying power is below `minimum`."""
    bp = alpaca_client.get_buying_power()
    if bp < minimum:
        raise RuntimeError(f"buying power ${bp:,.0f} is below the ${minimum:,.0f} floor — refusing to trade")
    return bp


def _client_order_id(today: date, symbol: str, tag: str) -> str:
    """'tom-<YYYYMM>-<SYM>-<in|out>', tagged with the month the action itself falls in."""
    if tag not in ('in', 'out'):
        raise ValueError(f"bad client_order_id tag: {tag!r}")
    if symbol not in SYMBOLS:
        raise ValueError(f"tom_sleeve only trades {SYMBOLS}, not {symbol!r}")
    return f"tom-{today:%Y%m}-{symbol}-{tag}"


def shares_for_notional(price: float, notional: float) -> int:
    """Whole shares buyable at `price` for at most `notional` dollars (floored, never over target)."""
    if price is None or price <= 0:
        return 0
    return int(notional // price)


# ---------------------------------------------------------------------------
# Ledger (logs/tom_sleeve_ledger.csv) — the local record of every action taken
# ---------------------------------------------------------------------------

def read_ledger(path: str = LEDGER_PATH) -> List[Dict]:
    """Return every row ever appended, oldest first, or [] if the ledger does not exist yet."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def append_ledger_row(row: Dict, path: str = LEDGER_PATH) -> None:
    """Append one row, writing the header first if the file is new. Append-only: a resolved fill
    check adds a NEW row rather than rewriting history, so the CSV is always a full audit trail.
    """
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _leg(client_order_id: str) -> str:
    """'in' or 'out' for 'tom-YYYYMM-SYM-in|out[-rN|-mkt]'; '' if the id is not a tom id."""
    parts = (client_order_id or '').split('-')
    return parts[3] if len(parts) >= 4 and parts[0] == 'tom' and parts[3] in ('in', 'out') else ''


def open_tom_symbols(ledger_rows: List[Dict]) -> Set[str]:
    """Symbols the ledger believes the sleeve currently holds. Rows are read in append (chronological)
    order: an active entry-leg row adds the symbol, an active exit-leg row removes it; a *_fillcheck
    row that resolved WITHOUT a fill undoes the tentative row before it -- an unfilled entry (the
    2026-09-30 SPY/IWM case) was never entered, an unfilled EXIT (the 2026-10-05 QQQ MOC) means the
    position is still at the broker and still owed. A `partial_fill` row keeps the symbol held either
    way: a partial ENTRY holds the filled part, a partial EXIT leaves the remainder owed.
    """
    held: Set[str] = set()
    for row in ledger_rows:
        action = row.get('action') or ''
        status = row.get('status')
        leg = _leg(row.get('client_order_id')) or ('in' if action.startswith('entry') else 'out')
        if action in _ORDER_ACTIONS and status in ACTIVE_STATUSES:
            (held.add if leg == 'in' else held.discard)(row['symbol'])
        elif action.endswith('_fillcheck') and status != 'filled':
            (held.discard if leg == 'in' and status != PARTIAL_STATUS else held.add)(row['symbol'])
    return held


def _row_qty(row: Dict) -> Optional[int]:
    """The row's qty as int, or None (with a WARNING) when it is missing / not a number."""
    try:
        return int(float(row['qty']))
    except (KeyError, ValueError, TypeError):
        logger.warning(f"tom_sleeve: ledger row without a usable qty ({row.get('action')} {row.get('symbol')} "
                        f"{row.get('client_order_id')}) — ignored for the open-qty count")
        return None


def ledger_open_qty(ledger_rows: List[Dict], symbol: str) -> Optional[int]:
    """Shares of `symbol` the ledger believes the sleeve still holds: the latest entry's qty (the filled
    part for a partial entry, 0 for an entry that never filled) minus every exit `partial_fill` row
    (counted once per client_order_id). None if there is no entry row (nothing to cross-check).
    """
    open_qty: Optional[int] = None
    counted: Set[str] = set()
    for row in ledger_rows:
        if row.get('symbol') != symbol:
            continue
        action, status, coid = row.get('action') or '', row.get('status'), row.get('client_order_id') or ''
        leg = _leg(coid) or ('in' if action.startswith('entry') else 'out')
        qty = _row_qty(row)
        if qty is None:
            continue
        if action in _ORDER_ACTIONS and leg == 'in' and status in ACTIVE_STATUSES:
            open_qty = qty
        elif action.endswith('_fillcheck') and leg == 'in' and status != 'filled':
            open_qty = qty if status == PARTIAL_STATUS else 0
        elif action.endswith('_fillcheck') and leg == 'out' and status == PARTIAL_STATUS and coid not in counted:
            counted.add(coid)
            open_qty = None if open_qty is None else max(open_qty - qty, 0)
    return open_qty


# ---------------------------------------------------------------------------
# Telegram (best-effort; never fatal)
# ---------------------------------------------------------------------------

def notify(notifier: Optional[TelegramNotifier], msg: str) -> None:
    """Send one [TOM]-prefixed line. A send failure is logged WARNING (not ERROR — an unreachable
    Telegram API is an expected, recoverable condition, never a reason to fail the run) and swallowed.
    """
    if notifier is None:
        return
    try:
        notifier.send_message_sync(f"{TG_PREFIX} {msg}")
    except Exception as e:
        logger.warning(f"tom_sleeve: Telegram send failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------------

def submit_moc_buy_order(alpaca_client: AlpacaClient, symbol: str, qty: int, client_order_id: str) -> Dict:
    """MOC BUY via the same TradingClient AlpacaClient already constructed (paper/live routing and
    credentials stay single-sourced through the wrapper). AlpacaClient.submit_moc_sell_order only
    covers the SELL side (its existing callers are all exits); TOM's entry leg is a BUY, so this
    mirrors that method's request shape for the other side rather than widening the shared wrapper
    class for one sleeve's entry leg.
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"tom_sleeve only trades {SYMBOLS}, not {symbol!r}")
    if not client_order_id.startswith('tom-'):
        raise ValueError(f"refusing to submit a non-tom client_order_id: {client_order_id!r}")
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    try:
        request = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.CLS, order_class=OrderClass.SIMPLE,
            client_order_id=client_order_id,
        )
        order = alpaca_client.trading_client.submit_order(request)
        result = {
            'id': str(getattr(order, 'id', '') or ''),
            'status': str(getattr(getattr(order, 'status', ''), 'value', getattr(order, 'status', '')) or 'unknown'),
            'symbol': symbol,
            'qty': qty,
        }
        logger.info(f"tom_sleeve: MOC buy submitted: {symbol} BUY {qty} — id={result['id']} status={result['status']}")
        return result
    except AlpacaAPIError:
        raise
    except Exception as e:
        logger.error(f"tom_sleeve: MOC buy submit failed for {symbol}: {e}")
        raise AlpacaAPIError(f"Failed to submit MOC buy for {symbol}: {e}")


def submit_fallback_limit_buy_order(alpaca_client: AlpacaClient, symbol: str, qty: int, price: float,
                                     client_order_id: str) -> Dict:
    """FALLBACK_LIMIT_SYMBOLS entry leg: a DAY limit order priced FALLBACK_LIMIT_BUFFER above the
    latest trade so it is marketable (crosses the spread) in continuous trading. Deliberately NOT
    TimeInForce.CLS — the 2026-09-30 incident (module docstring) showed the closing auction itself
    never returned a fill for SPY/IWM, so any CLS-based order (MOC or LOC) risks the same silent
    expiry; a plain marketable DAY order submitted in LATE_WINDOW sidesteps the auction entirely
    while still filling within ~1-2 minutes of the close.
    """
    if symbol not in FALLBACK_LIMIT_SYMBOLS:
        raise ValueError(f"fallback limit entry is only for {FALLBACK_LIMIT_SYMBOLS}, not {symbol!r}")
    if not client_order_id.startswith('tom-'):
        raise ValueError(f"refusing to submit a non-tom client_order_id: {client_order_id!r}")
    if price is None or price <= 0:
        raise ValueError(f"refusing a fallback limit order for {symbol} with a non-positive price: {price!r}")
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    limit_price = round(price * (1 + FALLBACK_LIMIT_BUFFER), 2)
    try:
        request = LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, limit_price=limit_price,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.SIMPLE,
            client_order_id=client_order_id,
        )
        order = alpaca_client.trading_client.submit_order(request)
        result = {
            'id': str(getattr(order, 'id', '') or ''),
            'status': str(getattr(getattr(order, 'status', ''), 'value', getattr(order, 'status', '')) or 'unknown'),
            'symbol': symbol,
            'qty': qty,
        }
        logger.info(f"tom_sleeve: fallback limit buy submitted: {symbol} BUY {qty} <= ${limit_price:.2f} "
                     f"(CLS auction bypassed — id={result['id']} status={result['status']})")
        return result
    except AlpacaAPIError:
        raise
    except Exception as e:
        logger.error(f"tom_sleeve: fallback limit buy submit failed for {symbol}: {e}")
        raise AlpacaAPIError(f"Failed to submit fallback limit buy for {symbol}: {e}")


# ---------------------------------------------------------------------------
# Fill-aware idempotency, late fallback and next-session catch-up (2026-10-05 QQQ MOC incident)
# ---------------------------------------------------------------------------

def _norm_status(raw) -> str:
    """'OrderStatus.NEW' / enum / 'new' -> 'new'."""
    return str(getattr(raw, 'value', raw) or 'unknown').split('.')[-1].lower()


def resolve_fill(order: Dict, planned_qty) -> tuple:
    """(status, filled_qty, avg_price) of a broker order dict. A terminal order that did not end `filled`
    but has 0 < filled_qty < planned qty resolves to PARTIAL_STATUS (the 2026-10-06 QQQ MOC: expired,
    filled 25 of 26); filled_qty >= planned resolves to `filled`; no filled_qty keeps the raw status."""
    status = _norm_status(order.get('status'))
    filled = float(order.get('filled_qty') or 0)
    avg = order.get('filled_avg_price')
    if status in DEAD_STATUSES and filled > 0:
        try:
            planned = float(planned_qty or order.get('qty') or 0)
        except (TypeError, ValueError):
            planned = 0.0
        if planned <= 0:
            logger.warning(f"tom_sleeve: {status} order has filled_qty={filled:g} but no planned qty to compare "
                            f"— cannot classify, keeping {status!r}")
        elif filled >= planned:
            status = 'filled'
        else:
            status = PARTIAL_STATUS
    return status, int(filled), avg


def _base_coid(coid: str) -> str:
    """Strip a '-rN' / '-mkt' retry suffix: the leg's original client_order_id."""
    return re.sub(r'-(r\d+|mkt)$', '', coid)


def chain_state(alpaca_client: AlpacaClient, base: str, ledger_rows: List[Dict],
                open_orders: Dict[str, str]) -> Dict:
    """State of one leg across its whole id chain (base, base-r1.., base-mkt). `open_orders` maps
    broker-open client_order_id -> order id. Returns {'state': none|live|filled|dead, 'coid', 'order_id',
    'retries'}: filled > live > dead > none. A terminal ledger *_fillcheck row is trusted, otherwise the
    broker is asked (get_order); a failed lookup is ERROR-logged and treated as LIVE so we never
    double-submit on an unknown."""
    chain, planned = {}, {}
    for r in ledger_rows:
        c = r.get('client_order_id') or ''
        if r.get('action') in _ORDER_ACTIONS and (c == base or _base_coid(c) == base):
            chain[c] = r.get('order_id')
            planned[c] = r.get('qty')
    for c, oid in open_orders.items():
        if c == base or _base_coid(c) == base:
            chain.setdefault(c, oid)
    retries = sum(1 for c in chain if re.search(r'-r\d+$', c))
    if not chain:
        return {'state': 'none', 'coid': None, 'order_id': None, 'retries': 0}
    dead = None
    for c, oid in chain.items():
        fc = [r for r in ledger_rows if r.get('client_order_id') == c and (r.get('action') or '').endswith('_fillcheck')]
        from_ledger, filled_qty, avg_price = False, 0, None
        if c in open_orders:
            status = 'new'
        elif fc:
            status, from_ledger = _norm_status(fc[-1].get('status')), True
            filled_qty, avg_price = _row_qty(fc[-1]) or 0, fc[-1].get('ref_price')
        else:
            try:
                status, filled_qty, avg_price = resolve_fill(alpaca_client.get_order(oid), planned.get(c))
            except Exception as e:
                logger.error(f"tom_sleeve: get_order({oid}) for {c} failed ({e}) — treating as LIVE, no resubmit")
                status = 'new'
        if status == 'filled':
            return {'state': 'filled', 'coid': c, 'order_id': oid, 'retries': retries}
        if status in DEAD_STATUSES or status == PARTIAL_STATUS:
            dead = {'state': 'dead', 'coid': c, 'order_id': oid, 'retries': retries, 'status': status,
                    'from_ledger': from_ledger, 'filled_qty': filled_qty, 'avg_price': avg_price}
        else:
            if status not in LIVE_STATUSES:
                logger.warning(f"tom_sleeve: {c} has unrecognised status {status!r} — treating as live")
            return {'state': 'live', 'coid': c, 'order_id': oid, 'retries': retries, 'status': status}
    return dead


def _ledger_row(today, action, symbol, qty, ref, order_id, status, coid, now_utc, deviation='',
                partial_entry='') -> Dict:
    """One ledger row dict."""
    return {'date': str(today), 'action': action, 'symbol': symbol, 'qty': qty, 'ref_price': ref,
            'order_id': order_id, 'status': status, 'client_order_id': coid,
            'timestamp_utc': now_utc.isoformat(timespec='seconds'), 'partial_entry': partial_entry,
            'deviation': deviation}


def close_out_dead(state: Dict, symbol: str, qty, today, now_utc: datetime, ledger_path: str,
                   status: Optional[str] = None) -> None:
    """Write the terminal *_fillcheck row for a leg id we are about to replace, BEFORE the replacement
    row, so open_tom_symbols() ends in the right state and check_pending_fills() never re-resolves it."""
    leg = _leg(state['coid'])
    action = 'entry_fillcheck' if leg == 'in' else 'exit_fillcheck'
    status = status or state.get('status', 'expired')
    if status == PARTIAL_STATUS:
        if state.get('from_ledger'):
            return  # the partial_fill row is already in the ledger: a second one would double-count the fill
        avg = state.get('avg_price')
        append_ledger_row(_ledger_row(today, action, symbol, state.get('filled_qty'),
                                      f"{float(avg):.4f}" if avg else '', state['order_id'], PARTIAL_STATUS,
                                      state['coid'], now_utc, partial_entry='true' if leg == 'in' else 'false'),
                          ledger_path)
        return
    append_ledger_row(_ledger_row(today, action, symbol, qty, '', state['order_id'], status, state['coid'],
                                  now_utc), ledger_path)


def cancel_and_confirm(alpaca_client: AlpacaClient, order_id: str) -> str:
    """Cancel then poll get_order up to CANCEL_POLL_SECONDS. Returns the terminal status
    ('canceled'/'expired'/'rejected'/'filled') or 'unconfirmed' (caller must NOT submit a replacement)."""
    try:
        alpaca_client.cancel_order(order_id)
    except Exception as e:
        logger.error(f"tom_sleeve: cancel_order({order_id}) raised: {e}")
    for _ in range(CANCEL_POLL_SECONDS):
        try:
            st = _norm_status(alpaca_client.get_order(order_id).get('status'))
        except Exception as e:
            logger.warning(f"tom_sleeve: cancel poll get_order({order_id}) failed: {e}")
            st = 'unknown'
        if st in DEAD_STATUSES or st == 'filled':
            return st
        _sleep(1)
    return 'unconfirmed'


def submit_market_buy_order(alpaca_client: AlpacaClient, symbol: str, qty: int, client_order_id: str) -> Dict:
    """Plain DAY market BUY (late-fallback entry replacement)."""
    if symbol not in SYMBOLS or not client_order_id.startswith('tom-'):
        raise ValueError(f"refusing market buy {symbol!r} {client_order_id!r}")
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    try:
        order = alpaca_client.trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE, client_order_id=client_order_id))
    except AlpacaAPIError:
        raise
    except Exception as e:
        logger.error(f"tom_sleeve: market buy submit failed for {symbol}: {e}")
        raise AlpacaAPIError(f"Failed to submit market buy for {symbol}: {e}")
    return {'id': str(getattr(order, 'id', '') or ''), 'status': _norm_status(getattr(order, 'status', '')),
            'symbol': symbol, 'qty': qty}


def owed_exits(ledger_rows: List[Dict], today: date, alpaca_client: AlpacaClient,
               open_orders: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
    """Exit legs owed from an EARLIER session: {symbol: {'base': original exit coid, 'sessions': trading
    sessions since the chain's FIRST exit}}. A symbol is owed when the ledger still holds it (the exit never
    filled, or only partly) OR when an exit order of its chain is still LIVE at the broker -- the latter is
    the catch-up-day late fallback (2026-10-06: r1 was live at 15:58 ET on a non-TOM day)."""
    open_orders = open_orders or {}
    held = open_tom_symbols(ledger_rows)
    first_exit: Dict[str, Dict] = {}
    for r in ledger_rows:
        if r.get('action') == 'entry':
            first_exit.pop(r['symbol'], None)
        elif r.get('action') == 'exit':
            cur = first_exit.get(r['symbol'])
            if cur is None or _base_coid(cur['client_order_id']) != _base_coid(r['client_order_id']):
                first_exit[r['symbol']] = r
    out = {}
    for sym, r in first_exit.items():
        base = _base_coid(r['client_order_id'])
        live = any(_base_coid(c) == base for c in open_orders)
        if (sym not in held and not live) or _to_date(r['date']) >= today:
            continue
        d0 = _to_date(r['date'])
        try:
            n = len([c for c in alpaca_client.get_market_calendar(d0, today) if _to_date(c['date']) > d0])
        except Exception as e:
            logger.error(f"tom_sleeve: calendar lookup for catch-up failed ({e}) — counting weekdays")
            n = sum(1 for k in range(1, (today - d0).days + 1) if (d0 + timedelta(k)).weekday() < 5)
        out[sym] = {'base': base, 'sessions': max(n, 1)}
    return out


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run(alpaca_client: AlpacaClient, notifier: Optional[TelegramNotifier], now_utc: datetime,
        dry_run: bool = False, ledger_path: str = LEDGER_PATH) -> str:
    """One pass: act on an entry/exit day inside the window, else no-op. Returns a one-line summary
    (also what --dry-run prints). Safe to call repeatedly — every side effect is idempotency-checked.
    """
    assert_paper_account(alpaca_client)

    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    if not (in_action_window(now_et) or in_late_window(now_et)):
        msg = (f"{now_et.isoformat(timespec='minutes')} outside both the 15:40-15:55 ET MOC window "
               f"and the 15:57-15:59 ET late fallback window — no-op")
        logger.info(f"tom_sleeve: {msg}")
        return msg

    ledger_rows = read_ledger(ledger_path)
    action = classify_day(today, alpaca_client)
    try:
        open_orders = {o['client_order_id']: o.get('id') for o in alpaca_client.get_open_orders()
                       if o.get('client_order_id')}
    except AlpacaAPIError as e:
        logger.error(f"tom_sleeve: get_open_orders failed, proceeding on ledger state alone: {e}")
        open_orders = {}
    owed = owed_exits(ledger_rows, today, alpaca_client, open_orders) if action != 'exit' else {}
    if action == 'none' and not owed:
        stray = sorted(c for c in open_orders if c.startswith('tom-'))
        if stray:
            logger.warning(f"tom_sleeve: live tom order(s) {stray} at the broker on a non-TOM day with no "
                            f"tracked owed exit in the ledger — NOT handled, check the ledger")
        msg = f"{today} is not a turn-of-month entry or exit session — no-op"
        logger.info(f"tom_sleeve: {msg}")
        return msg

    if alpaca_client.is_short_trading_day(today):
        msg = (f"{today} is a short trading day — the 15:40-15:55 ET window assumes a 16:00 close "
               f"and the CLS cutoff has likely passed — skipped, no orders sent")
        logger.warning(f"tom_sleeve: {msg}")
        return msg

    held = open_tom_symbols(ledger_rows)

    moc_window = in_action_window(now_et)
    late_window = in_late_window(now_et)
    lines: List[str] = []
    if action == 'entry':
        assert_buying_power(alpaca_client)
        try:
            prices = alpaca_client.get_latest_trades(list(SYMBOLS))
        except AlpacaAPIError as e:
            logger.error(f"tom_sleeve: get_latest_trades failed, cannot size entries: {e}")
            return f"{today} entry day — price fetch failed, no orders sent ({e})"

        for symbol in SYMBOLS:
            is_moc = symbol in MOC_RELIABLE_SYMBOLS
            coid = _client_order_id(today, symbol, 'in')
            st = chain_state(alpaca_client, coid, ledger_rows, open_orders)
            if st['state'] == 'filled':
                logger.info(f"tom_sleeve: {st['coid']} filled — skip (idempotent)")
                continue
            if st['state'] == 'live':
                if late_window and not st['coid'].endswith('-mkt'):
                    line = _late_replace(alpaca_client, notifier, st, symbol, 'in', None, today, now_utc,
                                         ledger_path, dry_run)
                    if line:
                        lines.append(line)
                else:
                    logger.info(f"tom_sleeve: {st['coid']} live ({st['status']}) — skip (idempotent)")
                continue
            if is_moc and not moc_window and not (late_window and st['state'] == 'dead'):
                logger.info(f"tom_sleeve: {symbol} MOC window not active this tick — skip")
                continue
            if not is_moc and not late_window:
                logger.info(f"tom_sleeve: {symbol} late fallback window not active this tick — skip")
                continue
            if st['state'] == 'none' and symbol in held:
                logger.warning(f"tom_sleeve: {symbol} tom-position already held per ledger — "
                                f"skipping re-entry (was a prior exit missed?)")
                continue
            price = (prices.get(symbol) or {}).get('price', 0.0)
            if not price or price <= 0:
                logger.error(f"tom_sleeve: no usable latest price for {symbol} — skipping entry")
                continue
            qty = shares_for_notional(price, NOTIONAL_PER_SYMBOL)
            if qty <= 0:
                logger.error(f"tom_sleeve: computed qty<=0 for {symbol} at ${price:.2f} — skipping")
                continue
            use_moc = is_moc and moc_window
            order_kind = 'MOC' if use_moc else 'LIMIT'
            send_coid = coid
            if st['state'] == 'dead':
                send_coid = f"{coid}-r{st['retries'] + 1}"
                logger.warning(f"tom_sleeve: {st['coid']} {st['status']} unfilled — resubmitting as {send_coid}")
            if dry_run:
                lines.append(f"DRY-RUN would BUY {order_kind} {qty} {symbol} (~${price:.2f} -> "
                              f"~${qty * price:,.0f}) id={send_coid}")
                continue
            if st['state'] == 'dead':
                close_out_dead(st, symbol, qty, today, now_utc, ledger_path)
            if use_moc:
                result = submit_moc_buy_order(alpaca_client, symbol, qty, send_coid)
            else:
                result = submit_fallback_limit_buy_order(alpaca_client, symbol, qty, price, send_coid)
            append_ledger_row(_ledger_row(today, 'entry', symbol, qty, f"{price:.4f}", result['id'],
                                           'submitted', send_coid, now_utc), ledger_path)
            msg = (f"BUY {order_kind} {qty} {symbol} submitted (~${qty * price:,.0f}) "
                   f"id={result['id']} status={result['status']}")
            notify(notifier, msg)
            lines.append(msg)
        for sym, info in _unreentered_entries(ledger_rows, today).items():
            logger.warning(f"tom_sleeve: {sym} entry owed since {info} was NOT re-entered (entry window gone)")
    # Exit leg: today's exit session targets (all held) plus any exit owed from an earlier session.
    targets: Dict[str, Dict] = {}
    if action == 'exit':
        for symbol in SYMBOLS:
            base = _client_order_id(today, symbol, 'out')
            tracked = any(r.get('client_order_id') and _base_coid(r['client_order_id']) == base for r in ledger_rows)
            if symbol in held or tracked or base in open_orders:
                targets[symbol] = {'base': base, 'sessions': 0}
            else:
                logger.warning(f"tom_sleeve: no tracked tom position for {symbol} — nothing to exit, skip")
    targets.update(owed)
    if targets:
        try:
            positions = {p['symbol']: p['qty'] for p in alpaca_client.get_open_positions()}
        except AlpacaAPIError as e:
            logger.error(f"tom_sleeve: get_open_positions failed, cannot verify exit quantities: {e}")
            return f"{today} exit day — position fetch failed, no orders sent ({e})"
        for symbol, info in targets.items():
            line = _process_exit(alpaca_client, notifier, symbol, info, positions, ledger_rows, open_orders,
                                 today, now_utc, moc_window, late_window, ledger_path, dry_run)
            if line:
                lines.append(line)

    if not lines:
        return f"{today} {action} day — nothing to do (all idempotent / skipped, see log)"
    return "; ".join(lines)


def _unreentered_entries(ledger_rows: List[Dict], today: date) -> Dict[str, str]:
    """Earlier-session entries that resolved unfilled and were never replaced (symbol -> date)."""
    held = open_tom_symbols(ledger_rows)
    out = {}
    for r in ledger_rows:
        if (r.get('action') == 'entry_fillcheck' and r.get('status') != 'filled'
                and r['symbol'] not in held and _to_date(r['date']) < today):
            out[r['symbol']] = r['date']
    return out


def _late_replace(alpaca_client, notifier, st, symbol, leg, qty, today, now_utc, ledger_path, dry_run) -> Optional[str]:
    """Late window, live-but-unfilled order: cancel, confirm, then MARKET (`<coid>-mkt`).
    Cancel not confirmed -> ERROR + Telegram, NO second order."""
    mkt = f"{_base_coid(st['coid'])}-mkt"
    if qty is None:  # entry: reuse the ledger qty of the original order
        qty = next((int(float(r['qty'])) for r in read_ledger(ledger_path) if r.get('client_order_id') == st['coid']), 0)
    verb = 'BUY' if leg == 'in' else 'SELL'
    if dry_run:
        return f"DRY-RUN would CANCEL {st['coid']} ({st['status']}) then {verb} MARKET {qty} {symbol} id={mkt}"
    status = cancel_and_confirm(alpaca_client, st['order_id'])
    if status == 'filled':
        logger.info(f"tom_sleeve: {st['coid']} filled while cancelling — nothing to replace")
        return None
    if status == 'unconfirmed':
        msg = f"{symbol} {leg} order {st['coid']} UNFILLED, cancel not confirmed in {CANCEL_POLL_SECONDS}s — NOT replacing"
        logger.error(f"tom_sleeve: {msg}")
        notify(notifier, msg)
        return msg
    close_out_dead(st, symbol, qty, today, now_utc, ledger_path, status='canceled')
    if leg == 'in':
        result = submit_market_buy_order(alpaca_client, symbol, qty, mkt)
    else:
        result = alpaca_client.submit_market_sell_order(symbol, qty, client_order_id=mkt)
    append_ledger_row(_ledger_row(today, 'late_fallback_replace', symbol, qty, '', result['id'], 'submitted',
                                   mkt, now_utc), ledger_path)
    msg = (f"{symbol} {leg} order {st['coid']} unfilled at the close — cancelled, {verb} MARKET {qty} "
           f"submitted as {mkt} id={result['id']}")
    logger.warning(f"tom_sleeve: {msg}")
    notify(notifier, msg)
    return msg


def _process_exit(alpaca_client, notifier, symbol, info, positions, ledger_rows, open_orders, today, now_utc,
                  moc_window, late_window, ledger_path, dry_run) -> Optional[str]:
    """One symbol's exit leg: fill-aware idempotency, late market fallback, catch-up tagging."""
    base = info['base']
    deviation = f"late_exit_{info['sessions']}_sessions" if info['sessions'] else ''
    broker_qty = positions.get(symbol, 0)
    ledger_qty = ledger_open_qty(ledger_rows, symbol)
    # Owed = what the broker actually holds, never more than the ledger says is ours (never the original qty).
    qty = broker_qty if ledger_qty is None else min(broker_qty, ledger_qty)
    if ledger_qty is not None and broker_qty != ledger_qty:
        logger.warning(f"tom_sleeve: {symbol} broker qty {broker_qty} != ledger open qty {ledger_qty} "
                        f"— owed qty = min = {qty} (the broker position is the source of truth)")
    st = chain_state(alpaca_client, base, ledger_rows, open_orders)
    if st['state'] == 'filled':
        logger.info(f"tom_sleeve: {st['coid']} filled — skip (idempotent)")
        return None
    if qty <= 0 and st['state'] != 'live':
        logger.error(f"tom_sleeve: ledger shows an open {symbol} tom position (ledger qty {ledger_qty}) but "
                      f"the broker reports {broker_qty} shares — nothing to sell")
        return None
    if st['state'] == 'live':
        if late_window and not st['coid'].endswith('-mkt'):
            return _late_replace(alpaca_client, notifier, st, symbol, 'out', qty, today, now_utc, ledger_path, dry_run)
        logger.info(f"tom_sleeve: {st['coid']} live ({st['status']}) — skip (idempotent)")
        return None
    if not (moc_window or late_window):
        return None
    send_coid = base
    if st['state'] == 'dead':
        send_coid = f"{base}-r{st['retries'] + 1}"
        logger.warning(f"tom_sleeve: {st['coid']} {st['status']} "
                        f"({'filled ' + str(st['filled_qty']) if st['status'] == PARTIAL_STATUS else 'unfilled'}) "
                        f"— resubmitting the owed {qty} as {send_coid}")
    if deviation:
        logger.warning(f"tom_sleeve: {symbol} exit owed since an earlier session — {deviation}")
    use_moc = moc_window
    kind = 'MOC' if use_moc else 'MARKET'
    if dry_run:
        return f"DRY-RUN would SELL {kind} {qty} {symbol} id={send_coid}" + (f" deviation={deviation}" if deviation else '')
    if st['state'] == 'dead':
        close_out_dead(st, symbol, qty, today, now_utc, ledger_path)
    if use_moc:
        result = alpaca_client.submit_moc_sell_order(symbol, qty, client_order_id=send_coid)
    else:
        result = alpaca_client.submit_market_sell_order(symbol, qty, client_order_id=send_coid)
    append_ledger_row(_ledger_row(today, 'exit', symbol, qty, '', result['id'], 'submitted', send_coid, now_utc,
                                   deviation), ledger_path)
    msg = f"SELL {kind} {qty} {symbol} submitted id={result['id']} status={result['status']}" + \
          (f" [{deviation}]" if deviation else '')
    notify(notifier, msg)
    return msg


# Human-readable reason per terminal status, logged and Telegram'd at every fill check so a partial
# entry is never silent — see module docstring (2026-09-30 SPY/IWM incident) for the evidence behind
# the 'expired' text specifically.
_TERMINAL_STATUS_REASON = {
    'filled': 'filled',
    PARTIAL_STATUS: 'partially filled',
    'expired': ('unfilled — order was ACCEPTED at submission (no reject) but returned no fill by '
                'end of day; Alpaca marks unmatched orders EXPIRED after the close with no reason '
                'field on the order object (docs.alpaca.markets/us/docs/orders-at-alpaca only says '
                '"any unfilled order after the close will be cancelled")'),
    'canceled': 'canceled before it could fill (broker- or user-initiated)',
    'rejected': 'rejected by the broker or exchange at or after submission',
}


def _fillcheck_row(row: Dict, status: str, qty, ref_price: str, partial_entry: str) -> Dict:
    """The `<action>_fillcheck` ledger row that resolves order-row `row` (date and coid of the order)."""
    return {'date': row['date'], 'action': f"{row['action']}_fillcheck", 'symbol': row['symbol'], 'qty': qty,
            'ref_price': ref_price, 'order_id': row['order_id'], 'status': status,
            'client_order_id': row['client_order_id'],
            'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'partial_entry': partial_entry}


def _partial_message(row: Dict, filled_qty: int, fill_price) -> str:
    """Telegram/log line for a partially filled order: filled/total, price, what stays owed."""
    total = _row_qty(row) or 0
    price = f" @ ${float(fill_price):.2f}" if fill_price else ""
    if _leg(row.get('client_order_id')) == 'in':
        return (f"{row['symbol']} {row['action']} order PARTIAL FILL {filled_qty}/{total}{price} — "
                f"[PARTIAL ENTRY: the sleeve holds the filled {filled_qty} only]")
    return (f"{row['symbol']} {row['action']} order PARTIAL FILL {filled_qty}/{total}{price} — "
            f"remaining {max(total - filled_qty, 0)} stays owed (sold next session from the broker position)")


def check_pending_fills(alpaca_client: AlpacaClient, notifier: Optional[TelegramNotifier],
                         ledger_path: str = LEDGER_PATH) -> List[str]:
    """Resolve ledger rows still marked 'submitted' against the broker's current order status and
    notify + log a follow-up row (status AND a human-readable reason, never just the bare status —
    see _TERMINAL_STATUS_REASON) for anything that has since become terminal (filled/canceled/
    rejected/expired). An entry that resolves to anything other than 'filled' is flagged
    partial_entry='true' on that row — the ledger's honest record of what this month's sleeve
    position actually is, since open_tom_symbols() only treats a 'filled'/'submitted' entry as held.
    Cheap and read-mostly, so it runs on every non-dry-run invocation, in or out of either action
    window: MOC orders are still pending at 15:55 ET when submitted, so it is the LATER daily cron
    tick (after the 16:00 ET close, e.g. the 20:45 UTC tick in EDT) that actually reports fills — by
    design, since the four daily cron ticks (19:45, 19:58, 20:45, 20:58 UTC) straddle both windows
    and the close.
    """
    rows = read_ledger(ledger_path)
    # A coid is "still pending" if its most recent row says 'submitted' and no *_fillcheck row exists yet.
    fillcheck_done = {r['client_order_id'] for r in rows if r.get('action', '').endswith('_fillcheck')}
    pending = [r for r in rows if r.get('status') == 'submitted' and r['client_order_id'] not in fillcheck_done]

    messages = []
    for row in pending:
        order_id = row.get('order_id')
        if not order_id:
            continue
        try:
            order = alpaca_client.get_order(order_id)
        except AlpacaAPIError as e:
            logger.warning(f"tom_sleeve: fill check for {order_id} ({row['symbol']}) failed: {e} — will retry next run")
            continue
        status, filled_qty, fill_price = resolve_fill(order, row.get('qty'))
        if status in ('filled', 'canceled', 'rejected', 'expired', PARTIAL_STATUS):
            is_entry = _leg(row.get('client_order_id')) == 'in'
            partial = is_entry and status != 'filled'
            reason = _TERMINAL_STATUS_REASON.get(status)
            if reason is None:
                reason = f"unrecognized terminal status {status!r} — treating as a reason gap, not a fill"
                logger.warning(f"tom_sleeve: fill check — {row['symbol']} {row['action']} resolved to "
                                f"{status!r}, which has no documented reason text (see _TERMINAL_STATUS_REASON)")
            append_ledger_row(_fillcheck_row(
                row, status, filled_qty if status == PARTIAL_STATUS else row['qty'],
                f"{fill_price:.4f}" if fill_price else row.get('ref_price', ''), 'true' if partial else 'false'),
                ledger_path)
            if status == PARTIAL_STATUS:
                msg = _partial_message(row, filled_qty, fill_price)
            else:
                msg = (f"{row['symbol']} {row['action']} order {status}"
                       + (f" @ ${fill_price:.2f}" if fill_price else "") + f" — {reason}")
            if status == PARTIAL_STATUS:
                logger.warning(f"tom_sleeve: fill check — {msg}")
            elif partial:
                msg += " [PARTIAL ENTRY: this symbol is NOT in the sleeve this month]"
                logger.warning(f"tom_sleeve: fill check — {msg}")
            else:
                logger.info(f"tom_sleeve: fill check — {msg}")
            notify(notifier, msg)
            messages.append(msg)
        else:
            logger.info(f"tom_sleeve: fill check — {row['symbol']} {row['action']} order still {status}")
    return messages


def reconcile_ledger(alpaca_client: AlpacaClient, notifier: Optional[TelegramNotifier],
                     ledger_path: str = LEDGER_PATH, dry_run: bool = False) -> List[str]:
    """Backfill missing `partial_fill` rows. For every ledger order row whose fill check resolved WITHOUT a
    fill (expired/canceled), ask the orders API; if the order in fact filled 0 < filled_qty < qty and no
    partial_fill row exists for it, append one (filled qty, avg price). Idempotent. Writes the ledger only --
    NEVER places or cancels an order. A full fill mislabelled unfilled is WARNING-logged, not rewritten. A
    partial row is refused (ERROR) when a LATER order row exists for the symbol: appended last it would
    re-add the symbol to the held set. Returns one message per row appended (or, in dry_run, that would be)."""
    rows = read_ledger(ledger_path)
    msgs: List[str] = []
    for idx, row in enumerate(rows):
        coid = row.get('client_order_id') or ''
        if row.get('action') not in _ORDER_ACTIONS or not row.get('order_id'):
            continue
        fcs = [r for r in rows if r.get('client_order_id') == coid and (r.get('action') or '').endswith('_fillcheck')]
        if not fcs or any(r.get('status') in ('filled', PARTIAL_STATUS) for r in fcs):
            continue
        try:
            status, filled_qty, avg = resolve_fill(alpaca_client.get_order(row['order_id']), row.get('qty'))
        except AlpacaAPIError as e:
            logger.warning(f"tom_sleeve: reconcile get_order({row['order_id']}) for {coid} failed: {e}")
            continue
        if status == 'filled':
            logger.warning(f"tom_sleeve: reconcile — {coid} is FILLED at the broker but the ledger says "
                            f"{fcs[-1].get('status')}; not auto-corrected, review the ledger")
            continue
        if status != PARTIAL_STATUS:
            continue
        if any(r.get('symbol') == row['symbol'] and r.get('action') in _ORDER_ACTIONS
               and r.get('client_order_id') != coid for r in rows[idx + 1:]):
            logger.error(f"tom_sleeve: reconcile — {coid} filled {filled_qty} but a LATER order row exists for "
                          f"{row['symbol']}; NOT appending (it would corrupt the held state), fix by hand")
            continue
        is_entry = _leg(coid) == 'in'
        msg = _partial_message(row, filled_qty, avg)
        if dry_run:
            msgs.append(f"DRY-RUN would append partial_fill: {msg}")
            continue
        append_ledger_row(_fillcheck_row(row, PARTIAL_STATUS, filled_qty, f"{float(avg):.4f}" if avg else '',
                                         'true' if is_entry else 'false'), ledger_path)
        logger.warning(f"tom_sleeve: reconcile — appended partial_fill row: {msg}")
        notify(notifier, msg)
        msgs.append(msg)
    return msgs


def print_status(alpaca_client: Optional[AlpacaClient], ledger_path: str = LEDGER_PATH) -> None:
    """Print the ledger tail and the sleeve's believed-open positions; cross-check against the
    broker's actual SPY/QQQ/IWM holdings when a client is available. Read-only.
    """
    rows = read_ledger(ledger_path)
    print(f"tom_sleeve ledger: {len(rows)} rows ({ledger_path})")
    for r in rows[-20:]:
        print(f"  {r['date']} {r['action']:16s} {r['symbol']:4s} qty={r['qty']:>6} "
              f"ref={r.get('ref_price', ''):>10} {r['status']:10s} {r['order_id']}")
    held = open_tom_symbols(rows)
    print(f"open tom sleeve (per ledger): {sorted(held) if held else '(flat)'}")
    if alpaca_client is not None:
        try:
            positions = {p['symbol']: p['qty'] for p in alpaca_client.get_open_positions() if p['symbol'] in SYMBOLS}
            print(f"broker SPY/QQQ/IWM positions: {positions if positions else '(flat)'}")
        except AlpacaAPIError as e:
            logger.warning(f"tom_sleeve: --status could not fetch broker positions: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help="print intended actions, submit nothing")
    ap.add_argument('--status', action='store_true', help="print the ledger and open sleeve, take no action")
    ap.add_argument('--reconcile', action='store_true',
                    help="append any missing partial_fill ledger row from the orders API (ledger write only, "
                         "NO orders); with --dry-run just print")
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--now-et', help="rehearsal only (needs --dry-run): pretend now is 'YYYY-MM-DD HH:MM' ET")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    cfg = Config()
    if not cfg.alpaca_orb_api_key or not cfg.alpaca_orb_api_secret:
        logger.error("tom_sleeve: ALPACA_ORB_API_KEY/SECRET not set — cannot run")
        return 1
    alpaca_client = AlpacaClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret, paper=cfg.alpaca_orb_paper)

    notifier = None
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
    else:
        logger.warning("tom_sleeve: Telegram not configured (TELEGRAM_BOT_TOKEN/CHAT_ID empty) — "
                        "proceeding without notifications")

    try:
        assert_paper_account(alpaca_client)
    except RuntimeError as e:
        logger.error(f"tom_sleeve: refusing to run — {e}")
        return 1

    if args.status:
        print_status(alpaca_client)
        return 0

    if args.reconcile:
        msgs = reconcile_ledger(alpaca_client, None, dry_run=args.dry_run)  # no Telegram: a ledger correction
        print("\n".join(msgs) if msgs else "reconcile: ledger already matches the orders API — nothing to append")
        return 0

    now_utc = datetime.now(timezone.utc)
    if args.now_et:
        if not args.dry_run:
            logger.error("tom_sleeve: --now-et is only allowed with --dry-run")
            return 1
        now_utc = datetime.strptime(args.now_et, '%Y-%m-%d %H:%M').replace(tzinfo=ET).astimezone(timezone.utc)
    summary = run(alpaca_client, notifier, now_utc, dry_run=args.dry_run)
    print(summary)
    if not args.dry_run:
        check_pending_fills(alpaca_client, notifier)
    return 0


if __name__ == '__main__':
    sys.exit(main())
