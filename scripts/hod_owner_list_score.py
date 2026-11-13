#!/usr/bin/env python3
"""Scores the HOD owner-in-the-loop list (spec docs/hod_owner_list_spec_20261112.md section 2; Friday 17:00 ET cron).

The owner's picks = filled BUY orders on the LIVE account whose client_order_id carries none of our prefixes. Each pick
is IN-LIST (the symbol was on a list that day with list time <= the order time) or OFF-LIST. Realised P&L per
(day, symbol) is FIFO over the owner's own fills (marked at the day's close from daily_bars when a position is still
open); the machine side is the HOD paper book's R on the same (date, symbol). READ-ONLY toward the broker (orders and
FILL activities); writes only reports/hod_owner_list_score_<date>.md (not in --dry-run) and sends one Telegram message.

Usage: python3 scripts/hod_owner_list_score.py --since YYYY-MM-DD [--dry-run]
"""
import argparse
import logging
import os
import sqlite3
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); os.chdir(ROOT)

import hod_owner_list as hl   # noqa: E402

logger = logging.getLogger('hod_owner_list_score')

ET = hl.ET
ORDERS_URL = 'https://api.alpaca.markets/v2/orders'
FILLS_URL = 'https://api.alpaca.markets/v2/account/activities/FILL'
# Every client_order_id prefix our own code submits (grep of trading/, 2026-11-12: hod-rest/hod..., orb-/orb-tp-/orb-addbe-,
# moml-/mom-, tom-, ign-stage-, r2g-, ops-flatten-, bf-, onemil-probe-; micha-* are third-party orders on the shared ORB
# account). A prefix missing here would count one of OUR orders as an owner pick: add it when a new book ships.
OUR_PREFIXES = ('hod-', 'hod', 'orb-', 'bf-', 'bull', 'moml-', 'mom-', 'tom-', 'micha-', 'micha-hf-', 'micha-vid-',
                'r2g', 'ign-', 'ign', 'macd', 'ops-', 'onemil-probe-', 'nightly')


# ---------------------------------------------------------------- orders and picks
def _filled(o: dict) -> bool:
    """True when the order has filled quantity."""
    try:
        return float(o.get('filled_qty') or 0) > 0
    except (TypeError, ValueError):
        return False


def is_ours(order: dict) -> bool:
    """True when the client_order_id carries one of OUR_PREFIXES."""
    return str(order.get('client_order_id') or '').startswith(OUR_PREFIXES)


def is_owner_pick(order: dict) -> bool:
    """A pick = filled BUY whose client_order_id carries none of our prefixes."""
    return order.get('side') == 'buy' and _filled(order) and not is_ours(order)


def fetch_orders(since: str, key: str, secret: str, session=None, limit: int = 500) -> list:
    """All closed orders (nested legs included) created after `since`, ascending, paginated by `after`=last created_at; de-duplicated by id."""
    seen, out, after = set(), [], since
    while True:
        page = hl.alpaca_get(ORDERS_URL, key, secret, {'status': 'closed', 'after': after, 'limit': limit, 'direction': 'asc', 'nested': 'true'}, session)
        new = [o for o in page if o['id'] not in seen]
        for o in new:
            seen.add(o['id']); out.append(o)
        if len(page) < limit or not new:
            return out
        after = page[-1]['created_at']


def fetch_fills(since: str, key: str, secret: str, session=None, page_size: int = 100) -> list:
    """All FILL activities after `since`, ascending, paginated by page_token (the last activity id)."""
    out, token = [], None
    while True:
        params = {'after': since, 'direction': 'asc', 'page_size': page_size}
        if token:
            params['page_token'] = token
        page = hl.alpaca_get(FILLS_URL, key, secret, params, session)
        out.extend(page)
        if len(page) < page_size:
            return out
        token = page[-1]['id']


def _et_date(ts: str) -> str:
    """ET calendar date of an ISO timestamp."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(ET).strftime('%Y-%m-%d')


def owner_order_ids(orders: list) -> set:
    """Ids of every filled non-prefixed order AND its nested legs (a bracket's legs inherit the parent's owner)."""
    ids = set()
    for o in orders:
        if is_ours(o):
            continue
        if _filled(o):
            ids.add(o['id'])
        for leg in o.get('legs') or []:
            if _filled(leg):
                ids.add(leg['id'])
    return ids


def classify_pick(pick: dict, ledger: list) -> str:
    """IN-LIST when (ET date of pick['ts'], symbol) is in the ledger with list time (HHMM ET) <= the order time, else OFF-LIST."""
    ts = datetime.fromisoformat(pick['ts'].replace('Z', '+00:00')).astimezone(ET)
    day = ts.strftime('%Y-%m-%d')
    for r in ledger:
        if r['date'] == day and r['symbol'] == pick['symbol']:
            listed = datetime.strptime(f"{day} {r['at']}", '%Y-%m-%d %H%M').replace(tzinfo=ET)
            if listed <= ts:
                return 'IN-LIST'
    return 'OFF-LIST'


# ---------------------------------------------------------------- P&L
def fifo_pnl(fills: list, mark=None) -> dict:
    """FIFO P&L over fills [{side, qty, price}] in time order; an open remainder is marked at `mark`.

    Returns realized, unrealized, pnl, open_qty (signed), bought_notional. Shorts work symmetrically (sell first).
    """
    lots = deque()                      # [signed_qty, price]
    realized, bought = 0.0, 0.0
    for f in fills:
        q = float(f['qty']) * (1 if f['side'] == 'buy' else -1)
        px = float(f['price'])
        if q > 0:
            bought += q * px
        while q and lots and (lots[0][0] > 0) != (q > 0):
            lot = lots[0]
            m = min(abs(lot[0]), abs(q))
            realized += m * (px - lot[1]) * (1 if lot[0] > 0 else -1)
            lot[0] -= m * (1 if lot[0] > 0 else -1)
            q -= m * (1 if q > 0 else -1)
            if abs(lot[0]) < 1e-9:
                lots.popleft()
        if abs(q) > 1e-9:
            lots.append([q, px])
    open_qty = sum(l[0] for l in lots)
    unreal = 0.0
    if lots:
        if mark is None:
            logger.warning('open position of %.0f shares and no mark price - unrealized counted as 0', open_qty)
        else:
            unreal = sum(l[0] * (mark - l[1]) for l in lots)
    return dict(realized=realized, unrealized=unreal, pnl=realized + unreal, open_qty=open_qty, bought_notional=bought)


def day_close(symbol: str, date: str, db_path: str = hl.CACHE_DB):
    """The day's close from daily_bars (None if absent)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        row = con.execute('SELECT close FROM daily_bars WHERE symbol=? AND bar_date=?', (symbol, date)).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def load_machine_r(since: str, db_path: str = hl.TRADES_DB) -> dict:
    """(date, symbol) -> R (pnl / total_risk) of the CLOSED paper hod_break rows since `since` (latest row per name)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = con.execute("SELECT trade_date, symbol, pnl, total_risk FROM trades WHERE strategy='hod_break' AND account='paper' "
                           "AND trade_date>=? AND order_status='closed' AND pnl IS NOT NULL AND total_risk>0 ORDER BY id", (since,)).fetchall()
    finally:
        con.close()
    return {(d, s): p / r for d, s, p, r in rows}


# ---------------------------------------------------------------- report
def _mean(xs: list):
    """Mean or None."""
    return sum(xs) / len(xs) if xs else None


def build_report(picks: list, ledger: list, machine: dict, since: str) -> dict:
    """Summary numbers + the Telegram/markdown text. picks: [{date, symbol, kind, pnl, bought_notional}]."""
    n_in = sum(1 for p in picks if p['kind'] == 'IN-LIST')
    n_off = len(picks) - n_in
    pnl = sum(p['pnl'] for p in picks)
    notional = sum(p['bought_notional'] for p in picks)
    hit = lambda ps: (sum(1 for p in ps if p['pnl'] > 0) / len(ps)) if ps else None          # noqa: E731
    picked = {(p['date'], p['symbol']) for p in picks}
    listed = {(r['date'], r['symbol']) for r in ledger if r['date'] >= since}
    m_picked = [machine[k] for k in picked if k in machine]
    m_other = [machine[k] for k in listed - picked if k in machine]
    m_all = [machine[k] for k in listed if k in machine]
    rep = dict(n_picks=len(picks), n_in=n_in, n_off=n_off, owner_pnl=pnl, owner_ret_pct=(pnl / notional * 100) if notional else None,
               owner_hit_rate=hit(picks), hit_in=hit([p for p in picks if p['kind'] == 'IN-LIST']),
               hit_off=hit([p for p in picks if p['kind'] == 'OFF-LIST']),
               machine_r_picked=_mean(m_picked), machine_r_not_picked=_mean(m_other),
               machine_hit_rate=(sum(1 for r in m_all if r > 0) / len(m_all)) if m_all else None, n_list_names=len(listed))
    pct = lambda v: 'n/a' if v is None else f'{v * 100:.0f}%'                                  # noqa: E731
    num = lambda v, f: 'n/a' if v is None else format(v, f)                                    # noqa: E731
    lines = [f'[HOD-LIST SCORE since {since}] owner picks {len(picks)} (in-list {n_in} / off-list {n_off})',
             f"owner P&L {'-' if pnl < 0 else '+'}${abs(pnl):,.0f} = {num(rep['owner_ret_pct'], '+.2f')}% on ${notional:,.0f} bought; hit rate {pct(rep['owner_hit_rate'])} "
             f"(in-list {pct(rep['hit_in'])}, off-list {pct(rep['hit_off'])})",
             f"machine R on the picked names {num(rep['machine_r_picked'], '+.2f')} (n {len(m_picked)}); "
             f"on the list names NOT picked {num(rep['machine_r_not_picked'], '+.2f')} (n {len(m_other)}, the list's base rate); "
             f"machine hit rate {pct(rep['machine_hit_rate'])} over {len(m_all)} traded list names of {len(listed)}"]
    if len(picks) < 5:
        lines.append('n < 5, no verdict')
    lines.append('')
    lines.append('date       symbol  kind      pnl$     notional$')
    for p in sorted(picks, key=lambda p: (p['date'], p['symbol'])):
        lines.append(f"{p['date']} {p['symbol']:<7} {p['kind']:<9} {p['pnl']:+9.0f} {p['bought_notional']:10.0f}")
    rep['text'] = '\n'.join(lines)
    return rep


def collect_picks(since: str, key: str, secret: str, ledger: list, session=None) -> list:
    """Live-account orders and fills -> classified picks with realised P&L per (day, symbol)."""
    orders = fetch_orders(since, key, secret, session)
    ids = owner_order_ids(orders)
    first_ts = {}
    for o in orders:
        if is_owner_pick(o):
            ts = o.get('filled_at') or o.get('submitted_at') or o['created_at']
            k = (_et_date(ts), o['symbol'])
            first_ts[k] = min(first_ts.get(k, ts), ts)
    by_day = defaultdict(list)
    for f in fetch_fills(since, key, secret, session):
        if f.get('order_id') in ids:
            by_day[(_et_date(f['transaction_time']), f['symbol'])].append(f)
    picks = []
    for (day, sym), ts in sorted(first_ts.items()):
        fills = sorted(by_day.get((day, sym), []), key=lambda f: f['transaction_time'])
        mark = day_close(sym, day) or (float(fills[-1]['price']) if fills else None)
        r = fifo_pnl(fills, mark)
        picks.append(dict(date=day, symbol=sym, ts=ts, pnl=r['pnl'], bought_notional=r['bought_notional']))
        picks[-1]['kind'] = classify_pick(picks[-1], ledger)
    return picks


def main(argv: list = None) -> int:
    """CLI entry: score, print (dry) or write the report and send one Telegram message."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--since', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s')
    from dotenv import load_dotenv
    load_dotenv('/home/ec2-user/onemil/.env')
    key, secret = os.getenv('ALPACA_API_KEY', ''), os.getenv('ALPACA_API_SECRET', '')
    if not key or not secret:
        logger.error('ALPACA_API_KEY/SECRET missing - the owner picks cannot be read')
        return 1
    ledger = [r for r in hl.read_ledger() if r['date'] >= args.since]
    rep = build_report(collect_picks(args.since, key, secret, ledger), ledger, load_machine_r(args.since), args.since)
    if args.dry_run:
        print(rep['text'])
        return 0
    today = datetime.now(ET).strftime('%Y-%m-%d')
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    with open(os.path.join(ROOT, 'reports', f'hod_owner_list_score_{today}.md'), 'w') as fh:
        fh.write(f'# HOD owner list score since {args.since}\n\n```\n{rep["text"]}\n```\n')
    from config import Config
    cfg = Config()
    ok = hl.TelegramNotifier(bot_token=cfg.telegram_bot_token, chat_id=cfg.telegram_chat_id).send_message_sync(rep['text'], parse_mode=None)
    if not ok:
        logger.error('HOD-LIST SCORE: Telegram send returned False')
    else:
        # 2026-11-13: proof line — the first Friday run logged nothing on success, so the log could not show the
        # report went out (same gap as hod_owner_list.py, fixed the same morning).
        logger.info('HOD-LIST SCORE: sent since %s (%d chars, report reports/hod_owner_list_score_%s.md)',
                    args.since, len(rep['text']), today)
    return 0 if ok else 1


if __name__ == '__main__':
    from monitoring.cron_guard import run_guarded
    sys.exit(run_guarded(main, 'hod_owner_list_score'))
