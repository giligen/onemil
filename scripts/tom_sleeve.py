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

Usage:
    python3 scripts/tom_sleeve.py              # act only if now is in the MOC window (15:40-15:55 ET,
                                                 # QQQ) or the late fallback window (15:57-15:59 ET,
                                                 # SPY/IWM) on an entry/exit day; otherwise checks
                                                 # pending fills and exits 0
    python3 scripts/tom_sleeve.py --dry-run     # print intended actions, submit nothing, no Telegram
    python3 scripts/tom_sleeve.py --status      # print the ledger + current open sleeve, no orders

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
import sys
from datetime import date, datetime, time as dtime, timezone
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
                  'client_order_id', 'timestamp_utc', 'partial_entry']
TG_PREFIX = '[TOM]'
ACTIVE_STATUSES = ('submitted', 'filled')  # ledger statuses that count as "this leg is in effect"


# ---------------------------------------------------------------------------
# Calendar: turn-of-month entry/exit day detection
# ---------------------------------------------------------------------------

def _to_date(v) -> date:
    """Normalize an Alpaca calendar 'date' field (date or datetime) to a plain date."""
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


def open_tom_symbols(ledger_rows: List[Dict]) -> Set[str]:
    """Symbols the ledger believes the sleeve currently holds: an 'entry' row with no later 'exit'
    row for that symbol. Rows are read in file (append) order, which is chronological.
    """
    held: Set[str] = set()
    for row in ledger_rows:
        action = row.get('action')
        status = row.get('status')
        if action == 'entry' and status in ACTIVE_STATUSES:
            held.add(row['symbol'])
        elif action == 'exit' and status in ACTIVE_STATUSES:
            held.discard(row['symbol'])
        elif action == 'entry_fillcheck' and status != 'filled':
            # The entry order resolved WITHOUT filling (canceled/rejected/expired — the 2026-09-30
            # SPY/IWM case). It was tentatively "held" by the 'submitted' row above; honesty means
            # this symbol was never actually entered this month, so clear it (else it is falsely
            # locked out of every future month's entry — open_tom_symbols would never see a matching
            # 'exit' row to clear a position that was never really opened).
            held.discard(row['symbol'])
    return held


def _last_entry_qty(ledger_rows: List[Dict], symbol: str) -> Optional[int]:
    """Most recently logged entry quantity for `symbol`, for cross-checking against the broker's
    actual held quantity before an exit. None if no entry row is found (nothing to cross-check).
    """
    for row in reversed(ledger_rows):
        if row.get('symbol') == symbol and row.get('action') == 'entry' and row.get('status') in ACTIVE_STATUSES:
            try:
                return int(float(row['qty']))
            except (KeyError, ValueError, TypeError):
                return None
    return None


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

    action = classify_day(today, alpaca_client)
    if action == 'none':
        msg = f"{today} is not a turn-of-month entry or exit session — no-op"
        logger.info(f"tom_sleeve: {msg}")
        return msg

    if alpaca_client.is_short_trading_day(today):
        msg = (f"{today} is a short trading day — the 15:40-15:55 ET window assumes a 16:00 close "
               f"and the CLS cutoff has likely passed — skipped, no orders sent")
        logger.warning(f"tom_sleeve: {msg}")
        return msg

    ledger_rows = read_ledger(ledger_path)
    held = open_tom_symbols(ledger_rows)
    existing_coids = {r['client_order_id'] for r in ledger_rows}
    try:
        open_broker_coids = {o['client_order_id'] for o in alpaca_client.get_open_orders() if o.get('client_order_id')}
    except AlpacaAPIError as e:
        logger.error(f"tom_sleeve: get_open_orders failed, proceeding on ledger state alone: {e}")
        open_broker_coids = set()

    lines: List[str] = []
    if action == 'entry':
        assert_buying_power(alpaca_client)
        try:
            prices = alpaca_client.get_latest_trades(list(SYMBOLS))
        except AlpacaAPIError as e:
            logger.error(f"tom_sleeve: get_latest_trades failed, cannot size entries: {e}")
            return f"{today} entry day — price fetch failed, no orders sent ({e})"

        moc_window = in_action_window(now_et)
        late_window = in_late_window(now_et)
        for symbol in SYMBOLS:
            is_moc = symbol in MOC_RELIABLE_SYMBOLS
            if is_moc and not moc_window:
                logger.info(f"tom_sleeve: {symbol} MOC window not active this tick — skip")
                continue
            if not is_moc and not late_window:
                logger.info(f"tom_sleeve: {symbol} late fallback window not active this tick — skip")
                continue
            coid = _client_order_id(today, symbol, 'in')
            if coid in existing_coids or coid in open_broker_coids:
                logger.info(f"tom_sleeve: {coid} already exists — skip (idempotent)")
                continue
            if symbol in held:
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
            order_kind = 'MOC' if is_moc else 'LIMIT'
            if dry_run:
                lines.append(f"DRY-RUN would BUY {order_kind} {qty} {symbol} (~${price:.2f} -> "
                              f"~${qty * price:,.0f}) id={coid}")
                continue
            if is_moc:
                result = submit_moc_buy_order(alpaca_client, symbol, qty, coid)
            else:
                result = submit_fallback_limit_buy_order(alpaca_client, symbol, qty, price, coid)
            append_ledger_row({
                'date': str(today), 'action': 'entry', 'symbol': symbol, 'qty': qty,
                'ref_price': f"{price:.4f}", 'order_id': result['id'], 'status': 'submitted',
                'client_order_id': coid, 'timestamp_utc': now_utc.isoformat(timespec='seconds'),
                'partial_entry': '',
            }, ledger_path)
            msg = (f"BUY {order_kind} {qty} {symbol} submitted (~${qty * price:,.0f}) "
                   f"id={result['id']} status={result['status']}")
            notify(notifier, msg)
            lines.append(msg)
    else:  # exit
        try:
            positions = {p['symbol']: p['qty'] for p in alpaca_client.get_open_positions()}
        except AlpacaAPIError as e:
            logger.error(f"tom_sleeve: get_open_positions failed, cannot verify exit quantities: {e}")
            return f"{today} exit day — position fetch failed, no orders sent ({e})"

        for symbol in SYMBOLS:
            coid = _client_order_id(today, symbol, 'out')
            if coid in existing_coids or coid in open_broker_coids:
                logger.info(f"tom_sleeve: {coid} already exists — skip (idempotent)")
                continue
            if symbol not in held:
                logger.warning(f"tom_sleeve: no tracked tom position for {symbol} — nothing to exit, skip")
                continue
            qty = positions.get(symbol, 0)
            ledger_qty = _last_entry_qty(ledger_rows, symbol)
            if qty <= 0:
                logger.error(f"tom_sleeve: ledger shows an open {symbol} tom position (entry qty "
                              f"{ledger_qty}) but the broker reports {qty} shares — nothing to sell")
                continue
            if ledger_qty is not None and qty != ledger_qty:
                logger.warning(f"tom_sleeve: {symbol} broker qty {qty} != logged entry qty {ledger_qty} "
                                f"— selling the broker qty (source of truth for what is actually held)")
            if dry_run:
                lines.append(f"DRY-RUN would SELL MOC {qty} {symbol} id={coid}")
                continue
            result = alpaca_client.submit_moc_sell_order(symbol, qty, client_order_id=coid)
            append_ledger_row({
                'date': str(today), 'action': 'exit', 'symbol': symbol, 'qty': qty,
                'ref_price': '', 'order_id': result['id'], 'status': 'submitted',
                'client_order_id': coid, 'timestamp_utc': now_utc.isoformat(timespec='seconds'),
                'partial_entry': '',
            }, ledger_path)
            msg = f"SELL MOC {qty} {symbol} submitted id={result['id']} status={result['status']}"
            notify(notifier, msg)
            lines.append(msg)

    if not lines:
        return f"{today} {action} day — nothing to do (all idempotent / skipped, see log)"
    return "; ".join(lines)


# Human-readable reason per terminal status, logged and Telegram'd at every fill check so a partial
# entry is never silent — see module docstring (2026-09-30 SPY/IWM incident) for the evidence behind
# the 'expired' text specifically.
_TERMINAL_STATUS_REASON = {
    'filled': 'filled',
    'expired': ('unfilled — order was ACCEPTED at submission (no reject) but returned no fill by '
                'end of day; Alpaca marks unmatched orders EXPIRED after the close with no reason '
                'field on the order object (docs.alpaca.markets/us/docs/orders-at-alpaca only says '
                '"any unfilled order after the close will be cancelled")'),
    'canceled': 'canceled before it could fill (broker- or user-initiated)',
    'rejected': 'rejected by the broker or exchange at or after submission',
}


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
        status = order.get('status', 'unknown')
        if status in ('filled', 'canceled', 'rejected', 'expired'):
            fill_price = order.get('filled_avg_price')
            is_entry = row.get('action') == 'entry'
            partial = is_entry and status != 'filled'
            reason = _TERMINAL_STATUS_REASON.get(status)
            if reason is None:
                reason = f"unrecognized terminal status {status!r} — treating as a reason gap, not a fill"
                logger.warning(f"tom_sleeve: fill check — {row['symbol']} {row['action']} resolved to "
                                f"{status!r}, which has no documented reason text (see _TERMINAL_STATUS_REASON)")
            append_ledger_row({
                'date': row['date'], 'action': f"{row['action']}_fillcheck", 'symbol': row['symbol'],
                'qty': row['qty'], 'ref_price': f"{fill_price:.4f}" if fill_price else row.get('ref_price', ''),
                'order_id': order_id, 'status': status, 'client_order_id': row['client_order_id'],
                'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'partial_entry': 'true' if partial else 'false',
            }, ledger_path)
            msg = (f"{row['symbol']} {row['action']} order {status}"
                   + (f" @ ${fill_price:.2f}" if fill_price else "") + f" — {reason}")
            if partial:
                msg += " [PARTIAL ENTRY: this symbol is NOT in the sleeve this month]"
                logger.warning(f"tom_sleeve: fill check — {msg}")
            else:
                logger.info(f"tom_sleeve: fill check — {msg}")
            notify(notifier, msg)
            messages.append(msg)
        else:
            logger.info(f"tom_sleeve: fill check — {row['symbol']} {row['action']} order still {status}")
    return messages


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
    ap.add_argument('--verbose', action='store_true')
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

    now_utc = datetime.now(timezone.utc)
    summary = run(alpaca_client, notifier, now_utc, dry_run=args.dry_run)
    print(summary)
    if not args.dry_run:
        check_pending_fills(alpaca_client, notifier)
    return 0


if __name__ == '__main__':
    sys.exit(main())
