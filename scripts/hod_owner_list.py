#!/usr/bin/env python3
"""HOD owner-in-the-loop list (spec docs/hod_owner_list_spec_20261112.md, idea #25 of research/hod_entry/IDEAS_HOD_20261112.md).

Three times a session (09:40, 10:30, 14:00 ET) the owner receives the machine's armed HOD-break symbols with context,
picks 1-2 by eye on his live account, and `hod_owner_list_score.py` measures his hit rate against the machine's on the
same names. READ-ONLY toward the broker and the databases: Alpaca data snapshot with the HOD paper key, the trader
journal, `data/trades.db`, `data/cache.db`, the parity ledger. The only files it writes are `logs/hod_owner_list.csv`
(the ledger the scorer reads) and, via the cron redirect, `logs/hod_owner_list.log`. `--dry-run` writes/sends nothing.

Usage: python3 scripts/hod_owner_list.py --at {0940,1030,1400} [--dry-run] [--date YYYY-MM-DD]
"""
import argparse
import csv
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); os.chdir(ROOT)

from notifications.telegram_notifier import TelegramNotifier   # noqa: E402

logger = logging.getLogger('hod_owner_list')

ET = ZoneInfo('America/New_York')
LEDGER = os.path.join(ROOT, 'logs', 'hod_owner_list.csv')
TRADES_DB = os.path.join(ROOT, 'data', 'trades.db')
CACHE_DB = os.path.join(ROOT, 'data', 'cache.db')
PARITY_LEDGER = os.path.join(ROOT, 'logs', 'hod_live_parity_ledger.csv')
SNAPSHOT_URL = 'https://data.alpaca.markets/v2/stocks/snapshots'
MAX_ROWS = 14
MAX_SPREAD_BPS = 100   # wider than 1 % is not a pick the owner can make at the touch
SESSION_MINUTES = 390
OPEN_MINUTE = 570                     # 09:30 ET
LEDGER_FIELDS = ['date', 'at', 'symbol', 'level', 'last', 'dist_pct', 'trigger', 'stop', 'r_pct', 'day_pct', 'rvol',
                 'spread_bps', 'rearms', 'qualified', 'status', 'machine_r', 'pnl']
FOOTER = 'pick by symbol on the live account; the ledger scores you Friday'

_DRY_RX = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \|.*?\[HOD(?: DRY)?\] ARMED (\S+) level ([\d.]+) trigger ([\d.]+) '
                     r'limit ([\d.]+) stop ([\d.]+)')
_LIVE_RX = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \|.*?\[HOD\] (\S+): LIVE ARMED stop ([\d.]+) limit ([\d.]+)')


# ---------------------------------------------------------------- parsing and arithmetic
def parse_arms(lines: list, cutoff_utc: str = None) -> dict:
    """symbol -> latest arm {level, trigger, limit, stop, rearms, ts} from journal lines.

    `[HOD DRY] ARMED SYM level L trigger T limit X stop S` is one arm (hod_break_engine.py:1022); the live order's
    `[HOD] SYM: LIVE ARMED stop T limit X` (:1871) carries no level/stop and is used only for a symbol that has no
    dry-tagged arm (level/stop None). `rearms` = arms of that symbol minus one. `cutoff_utc` ('YYYY-MM-DD HH:MM:SS')
    drops arms logged after the list time. Timestamps are the app's own (UTC) body timestamps.
    """
    dry, live = {}, {}
    for ln in lines:
        m = _DRY_RX.search(ln)
        if m:
            dry.setdefault(m.group(2), []).append((m.group(1), m.groups()[2:]))
            continue
        m = _LIVE_RX.search(ln)
        if m:
            live.setdefault(m.group(2), []).append((m.group(1), m.groups()[2:]))
    arms = {}
    for sym, evs in dry.items():
        evs = sorted(e for e in evs if cutoff_utc is None or e[0] <= cutoff_utc)
        if evs:
            ts, (lv, tr, lim, st) = evs[-1]
            arms[sym] = dict(level=float(lv), trigger=float(tr), limit=float(lim), stop=float(st), rearms=len(evs) - 1, ts=ts)
    for sym, evs in live.items():
        evs = sorted(e for e in evs if cutoff_utc is None or e[0] <= cutoff_utc)
        if evs and sym not in arms:
            ts, (tr, lim) = evs[-1]
            arms[sym] = dict(level=None, trigger=float(tr), limit=float(lim), stop=None, rearms=len(evs) - 1, ts=ts)
    return arms


def rvol(day_volume, adv20, minutes_elapsed) -> float:
    """Day volume / (adv20 x fraction of the 390-minute session elapsed); fraction floored at 1 minute, capped at 1. None if unknown."""
    if not day_volume or not adv20:
        return None
    frac = min(1.0, max(1.0, float(minutes_elapsed)) / SESSION_MINUTES)
    return float(day_volume) / (float(adv20) * frac)


def spread_bps(bid, ask) -> float:
    """(ask - bid) / mid in basis points; None for a missing or crossed/zero quote."""
    if not bid or not ask or ask < bid:
        return None
    return (ask - bid) / ((ask + bid) / 2.0) * 1e4


# ---------------------------------------------------------------- read-only data sources
def _status_of(row: dict) -> str:
    """Human status of one hod_break trades row: ARMED / FILLED @px / STOPPED / TARGET / EOD / CLOSED:<reason>."""
    st = (row.get('order_status') or '').lower()
    if st in ('filled', 'partially_filled'):
        return f"FILLED @{float(row.get('fill_price') or 0):.2f}"
    if st == 'closed':
        why = (row.get('exit_reason') or '').lower()
        if why.startswith('stop') or why == 'squeeze':
            return 'STOPPED'
        if why.startswith('target'):
            return 'TARGET'
        if why.startswith('eod'):
            return 'EOD'
        return f'CLOSED:{why or "?"}'
    return 'ARMED'


def load_trade_status(date: str, db_path: str = TRADES_DB) -> dict:
    """symbol -> {status, r, pnl} from the PAPER hod_break rows of `date` (latest row per symbol); R = pnl / total_risk."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM trades WHERE strategy='hod_break' AND trade_date=? AND (account='paper' OR account IS NULL) "
                           "ORDER BY id", (date,)).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        r = dict(r)
        status = _status_of(r)
        pnl = r.get('pnl')
        risk = r.get('total_risk')
        closed = status.startswith(('STOPPED', 'TARGET', 'EOD', 'CLOSED'))
        out[r['symbol']] = dict(status=status, pnl=pnl, r=(pnl / risk) if (closed and pnl is not None and risk) else None)
    return out


def load_adv20(symbols: list, date: str, db_path: str = CACHE_DB) -> dict:
    """symbol -> mean volume of the last 20 daily bars strictly before `date`; symbols with no bars are absent."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    out = {}
    try:
        for s in symbols:
            vols = [v for (v,) in con.execute('SELECT volume FROM daily_bars WHERE symbol=? AND bar_date<? ORDER BY bar_date DESC LIMIT 20', (s, date))]
            if vols:
                out[s] = sum(vols) / len(vols)
    finally:
        con.close()
    return out


def load_parity(date: str, path: str = PARITY_LEDGER) -> dict:
    """symbol -> {qualified, cf_floor_stop_px} from the live parity ledger rows of `date` (latest row wins); {} if absent."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get('date') == date:
                out[r['symbol']] = dict(qualified=r.get('scanner_qualified_at_arm', ''), cf_floor_stop_px=r.get('cf_floor_stop_px', ''))
    return out


def alpaca_get(url: str, key: str, secret: str, params: dict, session=None):
    """GET with Alpaca key headers; returns parsed JSON, raises RuntimeError on a non-200 (never swallowed here)."""
    session = session or requests.Session()
    resp = session.get(url, headers={'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}, params=params, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f'GET {url} -> HTTP {resp.status_code}: {str(resp.text)[:200]}')
    return resp.json()


def fetch_snapshots(symbols: list, key: str, secret: str, session=None) -> dict:
    """symbol -> {last, bid, ask, day_open, prev_close, day_volume} in ONE snapshot call (SIP feed); {} + WARNING on failure."""
    if not symbols:
        return {}
    try:
        data = alpaca_get(SNAPSHOT_URL, key, secret, {'symbols': ','.join(symbols), 'feed': 'sip'}, session)
    except Exception as e:  # noqa: BLE001 - rows still print, snapshot fields blank
        logger.warning('snapshot fetch failed (%s) - rows carry no quote/volume fields', e)
        return {}
    out = {}
    for sym, s in (data.get('snapshots', data) if isinstance(data, dict) else {}).items():
        if not isinstance(s, dict):
            continue
        q, t, d, p = s.get('latestQuote') or {}, s.get('latestTrade') or {}, s.get('dailyBar') or {}, s.get('prevDailyBar') or {}
        out[sym] = dict(last=t.get('p'), bid=q.get('bp'), ask=q.get('ap'), day_open=d.get('o'), prev_close=p.get('c'), day_volume=d.get('v'))
    return out


# ---------------------------------------------------------------- rows, message, ledger
def build_rows(date: str, at: str, arms: dict, trades: dict, parity: dict, snaps: dict, adv20: dict, minutes_elapsed: float) -> list:
    """One dict per armed symbol with every field of the spec (blank/None when its source is missing)."""
    rows = []
    for sym, a in arms.items():
        s, t, p = snaps.get(sym, {}), trades.get(sym, {}), parity.get(sym, {})
        last, level = s.get('last'), a['level']
        prev, dopen = s.get('prev_close'), s.get('day_open')
        rows.append(dict(
            date=date, at=at, symbol=sym, level=level, last=last, trigger=a['trigger'], stop=a['stop'], rearms=a['rearms'],
            dist_pct=((last - level) / level * 100) if (last and level) else None,
            r_pct=((a['trigger'] - a['stop']) / a['trigger'] * 100) if (a['stop'] and a['trigger']) else None,
            day_pct=((last - prev) / prev * 100) if (last and prev) else None,
            rvol=rvol(s.get('day_volume'), adv20.get(sym), minutes_elapsed),
            spread_bps=spread_bps(s.get('bid'), s.get('ask')),
            qualified=p.get('qualified', ''), status=t.get('status', 'ARMED'), machine_r=t.get('r'), pnl=t.get('pnl')))
    return rows


def _f(v, fmt: str, dash: str = '-') -> str:
    """Format a possibly-missing number."""
    return dash if v is None else format(v, fmt)


def format_message(rows: list, at: str) -> str:
    """The Telegram text (plain, no HTML): header counts over ALL rows, then the <= 14 rows closest to their level, then the footer.

    Header: `[HOD-LIST HH:MM ET] <armed today> armed · <filled (incl. closed)> filled · <closed> closed <sum $>`.
    """
    if not rows:
        return None
    filled = [r for r in rows if r['status'] != 'ARMED']
    closed = [r for r in rows if r['status'].startswith(('STOPPED', 'TARGET', 'EOD', 'CLOSED'))]
    total = sum(r.get('pnl') or 0.0 for r in closed)
    head = f"[HOD-LIST {at[:2]}:{at[2:]} ET] {len(rows)} armed · {len(filled)} filled · {len(closed)} closed {'-' if total < 0 else '+'}${abs(total):,.0f}"
    # Only LIVE candidates take a slot: still armed or filled-and-open, spread tradeable. Closed rows and wide
    # spreads are counted in the header but never shown — the owner picks what can still be bought.
    live = [r for r in rows if r['status'] in ('ARMED', 'FILLED') or r['status'].startswith('FILLED')]
    live = [r for r in live if r.get('spread_bps') is None or r['spread_bps'] <= MAX_SPREAD_BPS]
    ordered = sorted(live, key=lambda r: abs(r['dist_pct']) if r.get('dist_pct') is not None else 1e9)[:MAX_ROWS]
    lines = [head]
    for r in ordered:
        lines.append(f"{r['symbol']} lvl {_f(r['level'], '.2f')} last {_f(r['last'], '.2f')} dist {_f(r['dist_pct'], '+.1f')}% "
                     f"r {_f(r['r_pct'], '.1f')}% day {_f(r['day_pct'], '+.0f')}% rvol {_f(r['rvol'], '.1f')}x "
                     f"sp {_f(r['spread_bps'], '.0f')}bp re{r['rearms']} {'Q' if str(r['qualified']) in ('1', 'True') else '-'} {r['status']}"
                     + (f" R {r['machine_r']:+.1f}" if r.get('machine_r') is not None else ''))
    if len(live) > len(ordered):
        lines.append(f'(+{len(live) - len(ordered)} further live symbols not shown)')
    lines.append(FOOTER)
    return '\n'.join(lines)


def publish(rows: list, at: str, notifier=None) -> bool:
    """Send ONE plain-text Telegram message. Zero rows -> one INFO line, nothing sent. Returns True iff a message went out."""
    if not rows:
        logger.info('HOD-LIST %s: no armed symbols today - nothing sent', at)
        return False
    if notifier is None:
        from config import Config
        cfg = Config()
        notifier = TelegramNotifier(bot_token=cfg.telegram_bot_token, chat_id=cfg.telegram_chat_id)
    ok = notifier.send_message_sync(format_message(rows, at), parse_mode=None)
    if not ok:
        logger.error('HOD-LIST %s: Telegram send returned False', at)
    return bool(ok)


def append_ledger(rows: list, date: str, at: str, path: str = None) -> None:
    """Append every row to the CSV ledger (header once). The scorer reads this file."""
    path = path or LEDGER
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r[k]) for k in LEDGER_FIELDS} | {'date': date, 'at': at})


def read_ledger(path: str = None) -> list:
    """All ledger rows as dicts of strings ([] when the file does not exist)."""
    path = path or LEDGER
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- orchestration
def minutes_elapsed_at(date: str, at: str, now: datetime = None) -> float:
    """Minutes of the session elapsed at the list time (or now, when `date` is today and the list time is still ahead)."""
    m = int(at[:2]) * 60 + int(at[2:]) - OPEN_MINUTE
    now = now or datetime.now(ET)
    if now.strftime('%Y-%m-%d') == date:
        m = min(m, now.hour * 60 + now.minute - OPEN_MINUTE)
    return float(m)


def gather_rows(date: str, at: str, now: datetime = None) -> list:
    """Read the journal, trades.db, parity ledger, snapshot and adv20 and return the rows for `date` at list time `at`."""
    from hod_break_eod_check import journal            # unions journald with logs/session_archive
    cutoff = datetime.strptime(f'{date} {at}', '%Y-%m-%d %H%M').replace(tzinfo=ET).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    arms = parse_arms(journal(date), cutoff_utc=cutoff)
    if not arms:
        return []
    syms = sorted(arms)
    key, secret = os.getenv('ALPACA_HOD_API_KEY', ''), os.getenv('ALPACA_HOD_API_SECRET', '')
    if not key or not secret:
        logger.error('ALPACA_HOD_API_KEY/SECRET missing - rows carry no quote/volume fields')
    snaps = fetch_snapshots(syms, key, secret) if key and secret else {}
    return build_rows(date, at, arms, load_trade_status(date), load_parity(date), snaps, load_adv20(syms, date),
                      minutes_elapsed_at(date, at, now))


def main(argv: list = None) -> int:
    """CLI entry: build the list, print (dry) or append the ledger + send (real)."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--at', required=True, choices=['0940', '1030', '1400'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--date', default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s')
    from dotenv import load_dotenv
    load_dotenv('/home/ec2-user/onemil/.env')
    date = args.date or datetime.now(ET).strftime('%Y-%m-%d')
    rows = gather_rows(date, args.at)
    if args.dry_run:
        print(format_message(rows, args.at) or f'HOD-LIST {args.at}: no armed symbols on {date} - nothing would be sent')
        return 0
    if not rows:
        logger.info('HOD-LIST %s: no armed symbols on %s - nothing sent', args.at, date)
        return 0
    append_ledger(rows, date, args.at)
    return 0 if publish(rows, args.at) else 1


if __name__ == '__main__':
    from monitoring.cron_guard import run_guarded
    sys.exit(run_guarded(main, 'hod_owner_list'))
