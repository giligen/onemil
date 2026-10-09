#!/usr/bin/env python3
"""Hourly Telegram summary per strategy book (spec: docs/hourly_summary_spec_20261009.md).

ONE message per run with today's state of every book: ORB, TOM (QQQ, only when active), HOD, MOM and a single
LIVE line (owner's manual account, report-only: day P&L + equity, never positions).

READ-ONLY. This script never submits, cancels or modifies an order and never writes the DB. It reads, per account,
the account object (equity / last_equity / account number), today's CLOSED orders, the open positions and the 1W
daily portfolio history.

Numbers
  * Day P&L: HOD / MOM / LIVE = equity - last_equity of the account. ORB and TOM share one account, so they are
    split by symbol (TOM = QQQ only): realized from today's fills (buy VWAP vs sell VWAP x min(bought, sold) qty,
    `filled_qty` / `filled_avg_price` only, never `qty`) + `unrealized_intraday_pl` of the open positions. The
    account total is the cross-check; a WARNING is logged when |ORB + TOM - account| > $5 (typically a position
    opened on a previous day and closed today: its sell has no matching buy in today's orders).
  * Week to date: sum of the account's daily `profit_loss` from `get_portfolio_history(period='1W', timeframe='1D')`
    (HOD / MOM / LIVE-style single-book accounts). ORB and TOM share one account, so their week is computed from
    the orders API since Monday 00:00 ET (realized per symbol as above, QQQ -> TOM, the rest -> ORB).
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
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config                                          # noqa: E402
from data_sources.alpaca_client import AlpacaClient                # noqa: E402
from notifications.telegram_notifier import TelegramNotifier       # noqa: E402

logger = logging.getLogger("hourly_summary")

ET = ZoneInfo("America/New_York")
MAX_LINES = 25
ORB_STAGE_USD = 10_000.0          # ORB % is quoted on the $10K stage
CROSS_CHECK_TOLERANCE = 5.0       # |ORB + TOM - account day P&L| above this -> WARNING
MAX_OPEN_NAMES = 6                # open-position names listed per line
TOM_SYMBOL = "QQQ"
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
                    'price': _f(getattr(o, 'filled_avg_price', None))})
    return out


def fetch_positions(client: AlpacaClient) -> List[Dict]:
    """Open positions as plain dicts: symbol, qty, market_value, cost_basis, unrealized_pl, intraday_pl."""
    out = []
    for p in client.trading_client.get_all_positions():
        out.append({'symbol': str(p.symbol), 'qty': _f(p.qty), 'market_value': _f(p.market_value),
                    'cost_basis': _f(getattr(p, 'cost_basis', None)),
                    'unrealized_pl': _f(getattr(p, 'unrealized_pl', None)),
                    'intraday_pl': _f(getattr(p, 'unrealized_intraday_pl', None))})
    return out


def fetch_history(client: AlpacaClient) -> List[Dict]:
    """1W / 1D portfolio history as [{'stamp': UTC date, 'profit_loss': float, 'equity': float}] (read-only)."""
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    h = client.trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period='1W', timeframe='1D'))
    out = []
    for ts, pl, eq in zip(h.timestamp or [], h.profit_loss or [], h.equity or []):
        out.append({'stamp': datetime.fromtimestamp(ts, timezone.utc).date(),
                    'profit_loss': _f(pl), 'equity': _f(eq)})
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

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


def week_to_date(history: List[Dict], today: date, day_pnl_today: float) -> float:
    """Sessions Monday..yesterday from the history + today's account day P&L.

    A bar stamped D holds the previous weekday's session (Alpaca stamps the NEXT UTC date), so the session date
    is D minus one weekday; only sessions >= this week's Monday and < today count.
    """
    monday = today - timedelta(days=today.weekday())
    total = 0.0
    for bar in history:
        sess = bar['stamp'] - timedelta(days=1)
        while sess.weekday() >= 5:
            sess -= timedelta(days=1)
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
    return {
        'name': name,
        'n_fills': len(sel_fills),
        'bought_notional': sum(s['buy_val'] for s in agg.values()),
        'realized': realized,
        'closed': closed,
        'positions': sel_pos,
        'intraday_open': sum(p['intraday_pl'] for p in sel_pos),
        'open_pl': sum(p['unrealized_pl'] for p in sel_pos),
        'open_mv': sum(abs(p['market_value']) for p in sel_pos),
        'active': bool(sel_fills or sel_pos),
    }


def collect_book(name: str, client: AlpacaClient, after_utc: datetime, today: date) -> Dict:
    """Read one strategy account and return its raw data (account, fills, positions, history)."""
    acct = fetch_account(client)
    assert_paper_client(client, acct['account_number'])
    history = fetch_history(client)
    check_history_alignment(name, history, acct['last_equity'])
    return {'account': acct, 'fills': fetch_filled_orders(client, after_utc),
            'week_fills': fetch_filled_orders(client, week_start_utc(today)),
            'positions': fetch_positions(client), 'history': history}


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
    names = " ".join(f"{html.escape(p['symbol'])} {fmt_money(p[key])}" for p in ranked[:MAX_OPEN_NAMES])
    extra = len(ranked) - MAX_OPEN_NAMES
    return f"open {len(positions)}: {names}" + (f" +{extra} more" if extra > 0 else "")


def _best_worst(closed: Dict[str, float]) -> str:
    """'best X +n' only when a closed winner exists, 'worst Y −n' only when a closed loser exists."""
    parts = []
    if closed:
        best = max(closed, key=closed.get)
        worst = min(closed, key=closed.get)
        if closed[best] > 0:
            parts.append(f"best {html.escape(best)} {fmt_money(closed[best])}")
        if closed[worst] < 0:
            parts.append(f"worst {html.escape(worst)} {fmt_money(closed[worst])}")
    return " | ".join(parts)


def format_book_line(label: str, day_pnl: float, pct: str, stats: Dict, wk: Optional[float],
                     wk_suffix: str = "") -> str:
    """One ORB / HOD style line: day P&L, fills, open positions, best/worst closed, week to date."""
    parts = [f"<b>{label}</b> {fmt_money(day_pnl)} ({pct})", f"{stats['n_fills']} fills", _open_names(stats['positions'])]
    bw = _best_worst(stats['closed'])
    if bw:
        parts.append(bw)
    line = " | ".join(parts)
    if wk is not None:
        line += f"   wk {fmt_money(wk)}{wk_suffix}"
    return line


def format_mom_line(day_pnl: float, equity: float, stats: Dict, wk: Optional[float]) -> str:
    """MOM (weekly book) line: day P&L, open P&L on market value, number of names, week to date."""
    line = (f"<b>MOM</b> {fmt_money(day_pnl)} ({fmt_pct(day_pnl, equity)}) | open {fmt_money(stats['open_pl'])}"
            f" on {fmt_k(stats['open_mv'])}, {len(stats['positions'])} names")
    if wk is not None:
        line += f"   wk {fmt_money(wk)}"
    return line


def format_tom_line(stats: Dict, wk: Optional[float] = None) -> str:
    """TOM (QQQ) line: day P&L on the QQQ cost basis, fills, QQQ position, week to date."""
    pnl = stats['realized'] + stats['intraday_open']
    cost = sum(abs(p['cost_basis']) for p in stats['positions']) or stats['bought_notional']
    pos = "flat"
    if stats['positions']:
        pos = "hold " + " ".join(f"{html.escape(p['symbol'])} {p['qty']:g}" for p in stats['positions'])
    line = f"<b>TOM</b> {fmt_money(pnl)} ({fmt_pct(pnl, cost)}) | {stats['n_fills']} fills | {pos}"
    return line + (f"   wk {fmt_money(wk)}" if wk is not None else "")


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

    data keys: 'ORB' (also feeds TOM), 'HOD', 'MOM', 'LIVE'. Logs a WARNING when ORB + TOM disagrees with the
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
        orb_s = book_stats('ORB', orb['fills'], orb['positions'], lambda s: s != TOM_SYMBOL)
        tom_s = book_stats('TOM', orb['fills'], orb['positions'], lambda s: s == TOM_SYMBOL)
        orb_pnl = orb_s['realized'] + orb_s['intraday_open']
        tom_pnl = tom_s['realized'] + tom_s['intraday_open']
        if abs(orb_pnl + tom_pnl - acct_day) > CROSS_CHECK_TOLERANCE:
            logger.warning("hourly_summary: ORB+TOM split %.2f differs from account day P&L %.2f by more than $%.0f "
                           "(a position opened on a previous day and closed today has no buy in today's orders)",
                           orb_pnl + tom_pnl, acct_day, CROSS_CHECK_TOLERANCE)
        # ORB and TOM share one account, so the account's portfolio history mixes them: week to date for each is
        # the realized P&L of the week's fills for its symbols + the open positions' intraday P&L.
        orb_wk_s = book_stats('ORB', orb['week_fills'], orb['positions'], lambda s: s != TOM_SYMBOL)
        tom_wk_s = book_stats('TOM', orb['week_fills'], orb['positions'], lambda s: s == TOM_SYMBOL)
        lines.append(format_book_line("ORB", orb_pnl, fmt_pct(orb_pnl, ORB_STAGE_USD), orb_s,
                                      orb_wk_s['realized'] + orb_wk_s['intraday_open']))
        if tom_s['active']:   # hidden unless a QQQ position or a QQQ fill today (week-only fills have no cost basis in the window)
            lines.append(format_tom_line(tom_s, tom_wk_s['realized'] + tom_wk_s['intraday_open']))

    hod = data.get('HOD')
    if hod is None:
        lines.append(na_line("HOD", errors.get('HOD', 'api error')))
    else:
        acct = hod['account']
        day = acct['equity'] - acct['last_equity']
        st = book_stats('HOD', hod['fills'], hod['positions'], lambda s: True)
        lines.append(format_book_line("HOD", day, fmt_pct(day, st['bought_notional']), st,
                                      week_to_date(hod['history'], today, day)))

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
        lines.append(f"<b>LIVE</b> {fmt_money(a['equity'] - a['last_equity'])} | equity ${a['equity']:,.0f}")

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
            data['LIVE'] = collect_live(AlpacaClient(key, secret, paper=False))
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
