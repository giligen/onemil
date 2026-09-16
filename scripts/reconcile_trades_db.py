#!/usr/bin/env python3
"""Reconcile trades.db against the broker's own fill history (2026-09-16).

READ THIS BEFORE RUNNING IT AGAIN. The first run of this script was WRONG and was rolled back. 95 of the 99 rows it
targeted ALREADY carried a correct P&L and were merely left in `filled` status because the engine never advanced the
status field. Overwriting them with broker-derived exits moved lifetime P&L by -16,721. The defect was a STATUS bug, not
a missing-P&L bug. The safe procedure, now implemented as --status-only (the default):
  * a row that already has a pnl  -> advance order_status to 'closed'. NEVER touch its numbers.
  * a row with no pnl at all      -> try the broker; if nothing matches, mark 'closed_unverified'.
Two further traps found the hard way:
  * `sell_short` fills are the OWNER'S manual shorts on this shared account. Matching one to our long fabricates an exit.
    Only side == 'sell' may close one of our longs.
  * trades.db is in WAL mode. `cp trades.db backup` does NOT snapshot it, and restoring the main file lets the -wal
    replay the writes you were trying to undo. Use `VACUUM INTO` for a backup, or revert row by row through SQLite.

Found by the owner brief: 99 rows from 2026-03 onward are still marked `filled`/`pending_new` with no exit written, while
the broker is FLAT. Lifetime P&L could therefore not be stated from the DB. This walks every such row, finds the matching
SELL fill(s) at the broker for that symbol on or after the trade date, and writes exit_price / exited_at / pnl /
order_status='closed'. Rows with no broker sell are marked 'closed_unverified' with a note, never silently left open.

Read-only against the broker. Writes ONLY to the trades table. --dry lists what it would do and changes nothing.
Usage: python3 scripts/reconcile_trades_db.py [--dry] [--since 2026-01-01]
"""
import json, os, sqlite3, sys, urllib.parse, urllib.request
from collections import defaultdict
from datetime import datetime, timezone
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
DRY = '--dry' in sys.argv
STATUS_ONLY = '--reconcile-pnl' not in sys.argv     # default: fix the status field only, never rewrite an existing P&L
SINCE = sys.argv[sys.argv.index('--since') + 1] if '--since' in sys.argv else '2026-01-01'
OPEN_STATES = ('filled', 'partially_filled', 'pending_new', 'exit_pending_verification')
cfg = Config()
BASE = 'https://paper-api.alpaca.markets' if cfg.alpaca_paper else 'https://api.alpaca.markets'
HDR = {'APCA-API-KEY-ID': cfg.alpaca_api_key, 'APCA-API-SECRET-KEY': cfg.alpaca_api_secret}


def api(path):
    req = urllib.request.Request(BASE + path, headers=HDR)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def all_fills(after):
    """Every FILL/PARTIAL_FILL activity at or after `after`, paged."""
    out = []; page_token = None
    while True:
        q = {'activity_types': 'FILL', 'page_size': '100', 'after': after}
        if page_token: q['page_token'] = page_token
        rows = api('/v2/account/activities?' + urllib.parse.urlencode(q))
        if not rows: break
        out.extend(rows)
        if len(rows) < 100: break
        page_token = rows[-1].get('id')
        if page_token is None: break
    return out


def main():
    fills = all_fills(SINCE + 'T00:00:00Z')
    sells = defaultdict(list)
    for f in fills:
        # ONLY plain 'sell' closes one of our longs. 'sell_short' is a SHORT ENTRY — on this shared account those are the
        # owner's manual trades (feedback_owner_manual_trades_untouchable) and matching one to our long fabricates an exit.
        if f.get('side') == 'sell':
            sells[f.get('symbol')].append(f)
    for v in sells.values():
        v.sort(key=lambda x: x.get('transaction_time', ''))
    print(f'broker fills since {SINCE}: {len(fills)} | symbols with sells: {len(sells)}', flush=True)
    con = sqlite3.connect(f'{ROOT}/data/trades.db', timeout=60)
    rows = con.execute(f"select id, strategy, symbol, trade_date, shares, fill_price, entry_price, order_status from trades "
                       f"where order_status in ({','.join('?' * len(OPEN_STATES))}) and trade_date >= ? order by trade_date", (*OPEN_STATES, SINCE)).fetchall()
    print(f'DB rows to reconcile: {len(rows)}', flush=True)
    used = defaultdict(set); consumed = {}; n_fixed = n_unver = 0
    for tid, strat, sym, day, sh, fp, ep, st in rows:
        existing = con.execute("select pnl from trades where id=?", (tid,)).fetchone()[0]
        if STATUS_ONLY and existing is not None:
            n_fixed += 1
            print(f'  {strat:10s} {sym:6s} {day} already has pnl {existing:+.2f} — advancing status to closed, numbers untouched')
            if not DRY: con.execute("update trades set order_status='closed' where id=?", (tid,))
            continue
        entry = float(fp or ep or 0); qty = int(sh or 0)
        # same-day sells first (these books are intraday), then later days; never reuse a fill already consumed
        pool = [f for f in sells.get(sym, []) if (f.get('transaction_time') or '')[:10] >= day and id(f) not in used[sym]]
        pool.sort(key=lambda f: ((f.get('transaction_time') or '')[:10] != day, f.get('transaction_time') or ''))
        got = 0; notional = 0.0; when = None
        for f in pool:
            q = int(float(f.get('qty') or 0)) - consumed.get(id(f), 0); px = float(f.get('price') or 0)
            if q <= 0 or px <= 0: continue
            take = min(q, qty - got); got += take; notional += take * px; when = f.get('transaction_time')
            consumed[id(f)] = consumed.get(id(f), 0) + take
            if consumed[id(f)] >= int(float(f.get('qty') or 0)): used[sym].add(id(f))
            if got >= qty: break
        if got > 0 and entry > 0:
            px = notional / got; pnl = (px - entry) * got
            upd = dict(order_status='closed' if got >= qty else 'closed_partial', exit_price=round(px, 4),
                       exited_at=when, pnl=round(pnl, 2), pnl_pct=round((px / entry - 1) * 100, 3), exit_reason='broker_reconciled')
            n_fixed += 1
            print(f'  {strat:10s} {sym:6s} {day} x{qty} entry {entry:.4f} -> exit {px:.4f} pnl {pnl:+.2f} ({got}/{qty})')
            if not DRY:
                con.execute("update trades set order_status=?, exit_price=?, exited_at=?, pnl=?, pnl_pct=?, exit_reason=? where id=?",
                            (upd['order_status'], upd['exit_price'], upd['exited_at'], upd['pnl'], upd['pnl_pct'], upd['exit_reason'], tid))
        else:
            n_unver += 1
            print(f'  {strat:10s} {sym:6s} {day} x{qty} NO BROKER SELL FOUND — marking closed_unverified')
            if not DRY:
                con.execute("update trades set order_status='closed_unverified', exit_reason='no_broker_sell' where id=?", (tid,))
    if not DRY: con.commit()
    tot = con.execute("select round(coalesce(sum(pnl),0),2) from trades").fetchone()[0]
    per = con.execute("select strategy, count(*), round(coalesce(sum(pnl),0),2) from trades where pnl is not null group by strategy").fetchall()
    print(f"\n{'DRY RUN — nothing written' if DRY else 'WRITTEN'} | reconciled {n_fixed}, unverified {n_unver}")
    print('DB lifetime P&L now:', tot, '| per book:', per)
    con.close()


if __name__ == '__main__':
    main()
