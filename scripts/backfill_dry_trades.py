#!/usr/bin/env python3
"""Backfill the `dry_trades` table from pre-existing HOD-break dry-run sources.

Defect (owner 9/28, "hod dry-run is not in the DB???"): before this script + the engine wiring in
trading/hod_break_engine.py::_record_dry_fill/_close_dry_trade_db, every HOD-break dry-run fill/exit
lived ONLY in CSV ledgers and journald/logs/session_archive, rebuilt on every read by
scripts/hod_dry_ledger.py (slow, and journal rotation already silently changed a past day's book once —
project_dry_ledger_archive_sep2026). This script inserts the HISTORY (everything before this fix shipped)
into `dry_trades` so scripts/hod_dry_ledger.py can read it directly.

Two independent sources, each tagged with its own `source` so they are never silently mixed:

  * source='backfill_ledger' — logs/hod_dry_entry_ledger.csv, the entry ledger `_append_dry_ledger`
    already writes on every dry fill. Gives entry_ts/entry_px/stop_px/target_px exactly; joined against
    logs/hod_dry_counterfactuals.csv (`_write_cf_row`'s exit ledger) by (symbol, date, fill_ts) for
    exit_px/exit_reason/exit_ts/r_multiple when available. Tolerant to the file's mixed row shape: the
    header is the OLD 13-column shape (no `live`), but every row since 2026-09-26 has a 14th value
    (`live`) appended without the header being rewritten (docs note in the module CLAUDE.md history) —
    parsed positionally, not via csv.DictReader.

  * source='backfill_journal' — scripts/hod_dry_ledger.py's own parse_dry_run_line/analyze_day (REUSED,
    not duplicated), which run scripts/hod_break_eod_check.py per day and read its "DRY-RUN EXECUTABLE
    book" summary line. Coarser: only (symbol, trade_date, r_multiple) per trade, no entry/exit
    price/timestamp (that script reports a day-level $ total, not a per-trade split) — entry_px, exit_px,
    entry_ts, exit_ts, pnl_usd are left NULL for these rows on purpose.

Idempotent: before inserting, checks existing dry_trades rows for the same (strategy, trade_date,
symbol, source) — and additionally entry_ts for backfill_ledger rows, since a symbol can fire more than
once a day. Already-present rows are skipped and counted, never re-inserted or duplicated.

Usage:
    python3 scripts/backfill_dry_trades.py --dry-run                      # print counts only, no DB writes
    python3 scripts/backfill_dry_trades.py --dry-run --ledger-only        # skip the slow journal reconstruction
    python3 scripts/backfill_dry_trades.py --dry-run --start 2026-09-25 --end 2026-09-28

The real (non-dry-run) backfill is run by the owner/main session, never by this agent (token discipline).
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from persistence.database import Database  # noqa: E402
from trading.hod_break_engine import STRATEGY_NAME  # noqa: E402 - 'hod_break'

DEFAULT_ENTRY_LEDGER = 'logs/hod_dry_entry_ledger.csv'
DEFAULT_CF_LEDGER = 'logs/hod_dry_counterfactuals.csv'
DEFAULT_JOURNAL_START = '2026-09-14'  # scripts/hod_dry_ledger.py's own default

# The entry ledger's base row shape (`_append_dry_ledger` in trading/hod_break_engine.py), positional —
# the file's header may be missing the trailing `live` column (see module docstring).
_ENTRY_COLS = ['date', 'symbol', 'arm_ts', 'cross_ts', 'level', 'trigger', 'limit', 'ask',
               'filled', 'fill_px', 'stop', 'target', 'tape_accurate', 'live']


def _f(v):
    """float() that treats '' / None as None instead of raising."""
    if v is None or v == '':
        return None
    return float(v)


def parse_entry_ledger(path: str = DEFAULT_ENTRY_LEDGER) -> list:
    """Tolerant positional parse of the entry ledger: returns one dict per FILLED row (filled=='1'),
    keyed by _ENTRY_COLS. Rows with 13 or 14 values are both accepted (see module docstring); anything
    else is skipped with a printed warning (never raises)."""
    rows = []
    if not os.path.exists(path):
        print(f"WARNING: entry ledger not found: {path}")
        return rows
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r, None)
        for i, raw in enumerate(r, start=2):
            if len(raw) not in (13, 14):
                print(f"WARNING: {path}:{i}: unexpected column count {len(raw)} — skipped")
                continue
            d = dict(zip(_ENTRY_COLS, raw))
            if d.get('filled') != '1':
                continue
            rows.append(d)
    return rows


def parse_cf_ledger(path: str = DEFAULT_CF_LEDGER) -> dict:
    """logs/hod_dry_counterfactuals.csv keyed by (symbol, date, fill_ts) -> exit dict. Empty dict (with a
    WARNING) if the file is absent — exits simply stay unknown for every backfill_ledger row."""
    out = {}
    if not os.path.exists(path):
        print(f"WARNING: counterfactual ledger not found: {path} — backfilled entries will have NULL exits")
        return out
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            key = (row['symbol'], row['date'], row['fill_ts'])
            out[key] = row
    return out


def build_ledger_rows(entry_path: str = DEFAULT_ENTRY_LEDGER, cf_path: str = DEFAULT_CF_LEDGER) -> list:
    """One dry_trades-shaped dict per FILLED entry-ledger row, exit fields joined from the cf ledger."""
    cf = parse_cf_ledger(cf_path)
    out = []
    for d in parse_entry_ledger(entry_path):
        entry_px = _f(d['fill_px'])
        stop_px = _f(d['stop'])
        exit_row = cf.get((d['symbol'], d['date'], d['cross_ts']))
        exit_px = exit_reason = exit_ts = r_multiple = None
        if exit_row:
            exit_px = _f(exit_row.get('exit_px'))
            exit_reason = exit_row.get('exit_reason') or None
            exit_ts = exit_row.get('exit_ts') or None
            if exit_px is not None and entry_px is not None and stop_px is not None and entry_px != stop_px:
                r_multiple = (exit_px - entry_px) / (entry_px - stop_px)
        out.append({
            'strategy': STRATEGY_NAME, 'trade_date': d['date'], 'symbol': d['symbol'],
            'entry_ts': d['cross_ts'], 'entry_px': entry_px, 'shares': None,
            'stop_px': stop_px, 'target_px': _f(d['target']),
            'exit_ts': exit_ts, 'exit_px': exit_px, 'exit_reason': exit_reason,
            'r_multiple': r_multiple, 'pnl_usd': None, 'risk_usd': None,
            'source': 'backfill_ledger',
        })
    return out


def build_journal_rows(start: str, end: str) -> list:
    """REUSES scripts/hod_dry_ledger.py's parse_dry_run_line/analyze_day (never duplicated) to walk
    each weekday from `start` to `end` and reconstruct (symbol, trade_date, r_multiple) rows from the
    engine's own logged "DRY-RUN EXECUTABLE book" summary line (journalctl + session_archive). Per-trade
    entry/exit price and timestamp are not available from this source — left NULL. Skips (with a printed
    warning) any day analyze_day cannot parse, e.g. a day with no signals or a subprocess failure."""
    from scripts.hod_dry_ledger import analyze_day  # local import: heavy (subprocess + Alpaca) module

    out = []
    d0 = datetime.strptime(start, '%Y-%m-%d').date()
    d1 = datetime.strptime(end, '%Y-%m-%d').date()
    day = d0
    while day <= d1:
        if day.weekday() < 5:
            day_str = day.strftime('%Y-%m-%d')
            try:
                result = analyze_day(day_str)
            except Exception as e:
                print(f"WARNING: analyze_day({day_str}) raised: {e} — skipped")
                result = None
            if result:
                _trades, _r, _usd, symbols, _is_green = result
                for sym, r in symbols:
                    out.append({
                        'strategy': STRATEGY_NAME, 'trade_date': day_str, 'symbol': sym,
                        'entry_ts': None, 'entry_px': None, 'shares': None,
                        'stop_px': None, 'target_px': None,
                        'exit_ts': None, 'exit_px': None, 'exit_reason': None,
                        'r_multiple': r, 'pnl_usd': None, 'risk_usd': None,
                        'source': 'backfill_journal',
                    })
        day += timedelta(days=1)
    return out


def _norm_ts(v):
    """Normalize a timestamp for equality comparison: sqlite3's PARSE_DECLTYPES hands a TIMESTAMP column
    back as a datetime object, while a freshly-parsed CSV row still holds the raw ISO string — both must
    compare equal for the same instant. None passes through unchanged."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return v
    return v


def _already_present(db: Database, row: dict) -> bool:
    """Idempotency check: same (strategy, trade_date, symbol, source), and for backfill_ledger rows also
    the same entry_ts (a symbol can fire more than once a day)."""
    existing = db.get_dry_trades(row['strategy'], start=row['trade_date'], end=row['trade_date'])
    for e in existing:
        if e['symbol'] != row['symbol'] or e['source'] != row['source']:
            continue
        if row['source'] == 'backfill_ledger' and _norm_ts(e.get('entry_ts')) != _norm_ts(row.get('entry_ts')):
            continue
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='print what would be inserted; no DB writes')
    ap.add_argument('--ledger-only', action='store_true', help='skip the journal reconstruction (slow: one subprocess per day)')
    ap.add_argument('--start', default=DEFAULT_JOURNAL_START)
    ap.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
    ap.add_argument('--entry-ledger', default=DEFAULT_ENTRY_LEDGER)
    ap.add_argument('--cf-ledger', default=DEFAULT_CF_LEDGER)
    args = ap.parse_args()

    rows = build_ledger_rows(args.entry_ledger, args.cf_ledger)
    print(f"entry ledger: {len(rows)} filled rows parsed from {args.entry_ledger}")
    if not args.ledger_only:
        jrows = build_journal_rows(args.start, args.end)
        print(f"journal reconstruction: {len(jrows)} trade rows parsed for {args.start}..{args.end}")
        rows += jrows
    else:
        print("journal reconstruction skipped (--ledger-only)")

    db = Database()  # opens data/trades.db; migration 16 creates dry_trades if missing (empty table, idempotent)
    inserted = skipped = failed = 0
    for row in rows:
        if _already_present(db, row):
            skipped += 1
            continue
        if args.dry_run:
            inserted += 1
            continue
        rid = db.insert_dry_entry(row)
        if rid is None:
            failed += 1
            continue
        inserted += 1
        if row['exit_px'] is not None or row['r_multiple'] is not None:
            db.close_dry_trade(rid, exit_ts=row['exit_ts'], exit_px=row['exit_px'],
                                exit_reason=row['exit_reason'], r_multiple=row['r_multiple'],
                                pnl_usd=row['pnl_usd'])

    verb = 'would insert' if args.dry_run else 'inserted'
    print(f"{verb}: {inserted}  already present (skipped): {skipped}  failed: {failed}  total parsed: {len(rows)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
