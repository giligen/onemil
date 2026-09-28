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
    entry_ts, exit_ts are left NULL for these rows on purpose. pnl_usd and risk_usd ARE derived: since
    R = pnl_usd / risk_usd by definition and the day's aggregate $ and signed R total are both known (the
    engine sizes every trade in a day off the same configured risk_usd), a day's implied risk_usd =
    day_usd / day_r_total, and each trade's pnl_usd = r_multiple * that risk_usd. When a day's signed R
    total is exactly 0 (can't invert), falls back to DEFAULT_RISK_USD (a WARNING is logged — this is an
    approximation, not the true per-trade risk).

Cross-source dedup (defect 9/28: a symbol/day with BOTH a ledger row and a journal row was inserted
TWICE, because the old idempotency key — strategy+trade_date+symbol+entry_ts — never matched across
sources: the ledger has a real entry_ts, the journal has none. `upsert_rows`/`_same_trade` now match
across sources on strategy+symbol+trade_date with entry_ts within 120s (or a wildcard match when either
side lacks one, which is always true for a journal row) — see their docstrings for the "journal wins"
and re-run-is-idempotent rules.

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

# Per-trade risk used ONLY to derive a backfill_journal row's pnl_usd/risk_usd when a day's signed R
# total is exactly 0 and the day's aggregate $ can't be inverted (see build_journal_rows). Matches
# trading/hod_break_engine.py's HodBreakEngine cfg default (cfg.get('risk_usd', 100.0)) is NOT used here
# on purpose — the LIVE config through this backfill window (config.yaml) was risk_usd: 50.
DEFAULT_RISK_USD = 50.0


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
    entry/exit price and timestamp are not available from this source — left NULL. pnl_usd/risk_usd ARE
    derived per trade from the day's aggregate $ and signed R total (see module docstring); skips (with a
    printed warning) any day analyze_day cannot parse, e.g. a day with no signals or a subprocess failure."""
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
                _trades, day_r, day_usd, symbols, _is_green = result
                if day_r:
                    day_risk_usd = day_usd / day_r  # R = pnl_usd / risk_usd, inverted at the day level
                else:
                    day_risk_usd = DEFAULT_RISK_USD
                    if symbols:
                        print(f"WARNING: {day_str}: day R total is 0 — can't derive risk_usd from $, "
                              f"falling back to the configured default ${DEFAULT_RISK_USD:.0f}")
                for sym, r in symbols:
                    out.append({
                        'strategy': STRATEGY_NAME, 'trade_date': day_str, 'symbol': sym,
                        'entry_ts': None, 'entry_px': None, 'shares': None,
                        'stop_px': None, 'target_px': None,
                        'exit_ts': None, 'exit_px': None, 'exit_reason': None,
                        'r_multiple': r, 'pnl_usd': round(r * day_risk_usd, 2),
                        'risk_usd': round(day_risk_usd, 2),
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


def _same_trade(a: dict, b: dict) -> bool:
    """True when two dry_trades-shaped dicts (an existing DB row and a freshly-parsed source row, in
    either order) are the SAME real trade: same strategy, symbol, trade_date, and entry timestamps within
    120s of each other. The journal reconstruction has no per-trade entry_ts (day-level only — see
    build_journal_rows); when either side lacks one, strategy+symbol+trade_date alone decides (a
    same-symbol re-entry on one day was not observed in the backfill window)."""
    if (a['strategy'] != b['strategy'] or a['symbol'] != b['symbol']
            or str(a['trade_date']) != str(b['trade_date'])):
        return False
    ta, tb = _norm_ts(a.get('entry_ts')), _norm_ts(b.get('entry_ts'))
    if ta is None or tb is None:
        return True
    return abs((ta - tb).total_seconds()) <= 120


def _find_match(existing: list, row: dict):
    """First row in `existing` (already-in-DB dry_trades rows for this strategy+trade_date) that is the
    SAME trade as `row` per _same_trade, regardless of source. None if no match."""
    for e in existing:
        if _same_trade(e, row):
            return e
    return None


def _merge_fields(row: dict) -> dict:
    """Non-None values from `row`'s dry_trades columns — the fields worth writing onto a matched existing
    row. Never overwrites a real stored value with a NULL."""
    cols = ('entry_ts', 'entry_px', 'shares', 'stop_px', 'target_px', 'exit_ts', 'exit_px',
            'exit_reason', 'r_multiple', 'pnl_usd', 'risk_usd')
    return {c: row[c] for c in cols if row.get(c) is not None}


def _field_changed(existing_val, new_val, col: str) -> bool:
    """Equality check for one dry_trades column, normalizing entry_ts/exit_ts (see _norm_ts)."""
    if col in ('entry_ts', 'exit_ts'):
        return _norm_ts(existing_val) != _norm_ts(new_val)
    return existing_val != new_val


def upsert_rows(db: Database, rows: list, dry_run: bool) -> tuple:
    """Insert/update `rows` (ledger + journal, any order) into dry_trades, deduplicating ACROSS sources
    by _same_trade so re-running — or a ledger/journal pair describing one real trade — never produces
    two rows for it. Journal candidates are applied first: a matching ledger candidate found afterwards
    is dropped (skipped), never inserted as a second row and never allowed to downgrade an already-won
    journal row — "journal wins". A journal candidate MAY update an existing row of any source (a
    same-source re-run healing a previously-NULL field such as pnl_usd, or a ledger-only OPEN row now
    resolved by the journal), promoting its `source` too. A row with no match at all is inserted as-is (a
    ledger-only row this way stays OPEN unless its own cf_ledger join already resolved an exit).

    Returns (inserted, updated, skipped, failed) counts."""
    inserted = updated = skipped = failed = 0
    ordered = sorted(rows, key=lambda r: r['source'] != 'backfill_journal')  # journal candidates first
    for row in ordered:
        existing = db.get_dry_trades(row['strategy'], start=row['trade_date'], end=row['trade_date'])
        match = _find_match(existing, row)

        if match is None:
            if dry_run:
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
            continue

        # A ledger candidate never overrides an already-won journal row.
        if row['source'] != 'backfill_journal' and match['source'] != row['source']:
            skipped += 1
            continue

        changed = {c: v for c, v in _merge_fields(row).items() if _field_changed(match.get(c), v, c)}
        if match['source'] != row['source']:
            changed['source'] = row['source']
        if not changed:
            skipped += 1
            continue
        if dry_run:
            updated += 1
            continue
        db.update_dry_entry(match['id'], **changed)
        updated += 1

    return inserted, updated, skipped, failed


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
    inserted, updated, skipped, failed = upsert_rows(db, rows, args.dry_run)

    verb = 'would insert' if args.dry_run else 'inserted'
    verb2 = 'would update' if args.dry_run else 'updated'
    print(f"{verb}: {inserted}  {verb2}: {updated}  already correct (skipped): {skipped}  "
          f"failed: {failed}  total parsed: {len(rows)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
