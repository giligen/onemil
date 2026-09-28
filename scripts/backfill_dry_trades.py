#!/usr/bin/env python3
"""Backfill the `dry_trades` table from pre-existing HOD-break dry-run sources.

Defect (owner 9/28, "hod dry-run is not in the DB???"): before this script + the engine wiring in
trading/hod_break_engine.py::_record_dry_fill/_close_dry_trade_db, every HOD-break dry-run fill/exit
lived ONLY in CSV ledgers and journald/logs/session_archive, rebuilt on every read by
scripts/hod_dry_ledger.py (slow, and journal rotation already silently changed a past day's book once —
project_dry_ledger_archive_sep2026). This script inserts the HISTORY (everything before this fix shipped)
into `dry_trades` so scripts/hod_dry_ledger.py can read it directly.

Three sources, each tagged with its own `source` so they are never silently mixed:

  * source='counterfactual_all' — logs/hod_dry_entry_ledger.csv, the entry ledger `_append_dry_ledger`
    already writes on every dry fill. Gives entry_ts/entry_px/stop_px/target_px exactly; joined against
    logs/hod_dry_counterfactuals.csv (`_write_cf_row`'s exit ledger) by (symbol, date, fill_ts) for
    exit_px/exit_reason/exit_ts/r_multiple when available. Tolerant to the file's mixed row shape: the
    header is the OLD 13-column shape (no `live`), but every row since 2026-09-26 has a 14th value
    (`live`) appended without the header being rewritten (docs note in the module CLAUDE.md history) —
    parsed positionally, not via csv.DictReader.

    NOT the capped dry book (renamed from 'backfill_ledger' 9/28 — diagnosis of "9/28 shows no capped
    dry entries"): the resting-fill simulator that writes this ledger (trading/hod_break_engine.py
    ~890-935) has NO max_per_day/max_concurrent check at all — those caps only gate the LIVE order path
    (`_arm_live_order`, lines ~1267-1270/1556-1563, `if not self.dry_run`). With log_counterfactuals on
    (shipped 9/26; logs/hod_dry_counterfactuals.csv has no rows before 9/28), EVERY dry fill opens a
    counterfactual watch — so every row here is the UNCAPPED all-armed population, never the ~12/day
    4-concurrent book. See build_capped_replay_rows for the actual capped book. A 13-column row (2026-09-25
    and earlier) predates log_counterfactuals entirely — no cf-ledger row can ever resolve it, so it is
    EXCLUDED (not stored with a fabricated r_multiple=0.0 outcome).

  * source='backfill_journal' — scripts/hod_dry_ledger.py's own parse_dry_run_line/analyze_day (REUSED,
    not duplicated), which run scripts/hod_break_eod_check.py per day and read its "DRY-RUN EXECUTABLE
    book" summary line. This line already applies trading.hod_break.run_book's cap (first max_per_day,
    max_concurrent) — it IS a capped book, for 2026-09-14..09-25 (before log_counterfactuals existed).
    Coarser: only (symbol, trade_date, r_multiple) per trade, no entry/exit price/timestamp (that script
    reports a day-level $ total, not a per-trade split) — entry_px, exit_px, entry_ts, exit_ts are left
    NULL for these rows on purpose. pnl_usd and risk_usd ARE derived: since R = pnl_usd / risk_usd by
    definition and the day's aggregate $ and signed R total are both known (the engine sizes every trade
    in a day off the same configured risk_usd), a day's implied risk_usd = day_usd / day_r_total, and
    each trade's pnl_usd = r_multiple * that risk_usd. When a day's signed R total is exactly 0 (can't
    invert), falls back to DEFAULT_RISK_USD (a WARNING is logged — this is an approximation, not the
    true per-trade risk).

  * source='replay_capped' — build_capped_replay_rows: the CAPPED book for 2026-09-26 onward (once
    log_counterfactuals made the ledger uncapped, see above). Replays the RESOLVED rows of the
    'counterfactual_all' population (a fill needs a matching cf-ledger exit to be replayable — an
    open/unresolved arm has no exit_m to free its slot, and is excluded, counted) through
    trading.hod_break.run_book — the SAME cap-simulation helper the backtest and hod_break_eod_check.py
    use (CLAUDE.md: ONE spec for backtest and live) — first-come by ARM time (not fill time), symbol
    tiebreak, max_per_day/max_concurrent from config.yaml's hod_break settings (12/4 as of 2026-09-28).
    Uses the ACTUAL exit leg (cf ledger's own exit_px/exit_reason/exit_ts, resolved against
    actual_stop/actual_target) — never cf_floor_stop_px/cf_stoplimit_*, which is a different, unrelated
    what-if (a floor-stop counterfactual). pnl_usd = r_multiple * DEFAULT_RISK_USD ("the day's risk" —
    config.yaml risk_usd was a flat $50 through this whole window).

'counterfactual_all' and 'replay_capped' are fully re-derived from the CSV ledgers on every run, so
main() makes them idempotent by DELETE-then-INSERT (Database.delete_dry_trades), scoped to their own
source tag and the date range just parsed — never touching 'backfill_journal' rows.

Cross-source dedup for 'counterfactual_all' + 'backfill_journal' ONLY (defect 9/28: a symbol/day with
BOTH a ledger row and a journal row was inserted TWICE, because the old idempotency key —
strategy+trade_date+symbol+entry_ts — never matched across sources: the ledger has a real entry_ts, the
journal has none). `upsert_rows`/`_same_trade` match across sources on strategy+symbol+trade_date with
entry_ts within 120s (or a wildcard match when either side lacks one, which is always true for a journal
row) — see their docstrings for the "journal wins" and re-run-is-idempotent rules. 'replay_capped' rows
are NEVER passed through upsert_rows: they are an intentionally separate, overlapping view of the same
underlying fills (a capped subset) and must not be merged against 'counterfactual_all' rows describing
the identical fill.

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
from trading.hod_break import run_book  # noqa: E402 - ONE cap-simulation helper (CLAUDE.md: ONE spec)
from trading.hod_break_engine import STRATEGY_NAME  # noqa: E402 - 'hod_break'

DEFAULT_ENTRY_LEDGER = 'logs/hod_dry_entry_ledger.csv'
DEFAULT_CF_LEDGER = 'logs/hod_dry_counterfactuals.csv'
DEFAULT_JOURNAL_START = '2026-09-14'  # scripts/hod_dry_ledger.py's own default

# config.yaml hod_break settings as of 2026-09-28 (owner-set live caps) — passed to trading.hod_break.
# run_book by build_capped_replay_rows. Not read from config.yaml directly: this script must not import
# the live trading config (token-discipline/blast-radius boundary for an agent that must never touch
# config.yaml); pass explicit --max-per-day/--max-concurrent if the live config changes.
DEFAULT_MAX_PER_DAY = 12
DEFAULT_MAX_CONCURRENT = 4

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
    keyed by _ENTRY_COLS. Only 14-value rows are accepted. A 13-value row (2026-09-25 and earlier)
    predates the `live` column AND log_counterfactuals (logs/hod_dry_counterfactuals.csv carries no row
    before 2026-09-28) — it has no possible resolved outcome, so it is EXCLUDED here rather than stored
    with a fabricated r_multiple=0.0 (module docstring). Anything else (not 13 or 14) is skipped with a
    printed warning. Never raises."""
    rows = []
    if not os.path.exists(path):
        print(f"WARNING: entry ledger not found: {path}")
        return rows
    n_excluded_13 = 0
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r, None)
        for i, raw in enumerate(r, start=2):
            if len(raw) == 13:
                n_excluded_13 += 1
                continue
            if len(raw) != 14:
                print(f"WARNING: {path}:{i}: unexpected column count {len(raw)} — skipped")
                continue
            d = dict(zip(_ENTRY_COLS, raw))
            if d.get('filled') != '1':
                continue
            rows.append(d)
    if n_excluded_13:
        print(f"NOTE: {path}: excluded {n_excluded_13} 13-column row(s) (pre-log_counterfactuals, "
              f"outcome-less — see module docstring)")
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
    """One dry_trades-shaped dict per FILLED entry-ledger row, exit fields joined from the cf ledger.
    source='counterfactual_all' — the UNCAPPED all-armed population, NOT the capped dry book (module
    docstring). For the capped book see build_capped_replay_rows."""
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
            'source': 'counterfactual_all',
        })
    return out


def build_capped_replay_rows(entry_path: str = DEFAULT_ENTRY_LEDGER, cf_path: str = DEFAULT_CF_LEDGER,
                              max_per_day: int = DEFAULT_MAX_PER_DAY,
                              max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> list:
    """The CAPPED dry book, replayed from the UNCAPPED 'counterfactual_all' population (see module
    docstring: the live resting-fill simulator has no max_per_day/max_concurrent check at all — those
    caps only gate the separate LIVE-order path). Uses trading.hod_break.run_book — the SAME
    cap-simulation helper the backtest and scripts/hod_break_eod_check.py's "DRY-RUN EXECUTABLE book"
    line use (CLAUDE.md: ONE spec for backtest and live, parity by construction) — never a bespoke cap
    loop.

    Ordering is BY ARM TIME (`arm_ts`), not fill time — first-come per the owner's spec — matching
    run_book's own tie-break (entry minute, then symbol alphabetically). Only rows with a RESOLVED
    counterfactual exit are eligible: an open/never-resolved arm has no exit_m to free its concurrency
    slot and is excluded (counted, printed).

    Exit uses the ACTUAL leg — the cf ledger's own exit_px/exit_reason/exit_ts columns, which
    trading/hod_break_engine.py's _update_cf_watch resolves against actual_stop/actual_target — NEVER
    cf_floor_stop_px/cf_stoplimit_*, which is a different, unrelated floor-stop counterfactual.

    pnl_usd = r_multiple * DEFAULT_RISK_USD ("the day's risk" — config.yaml risk_usd was a flat $50
    through this whole window, same convention build_journal_rows uses). Returns dry_trades-shaped
    dicts, source='replay_capped'."""
    from scripts.hod_break_eod_check import _et_minute  # local: avoid this script's lighter paths pulling it in

    cf = parse_cf_ledger(cf_path)
    candidates = []
    n_unresolved = 0
    for d in parse_entry_ledger(entry_path):
        exit_row = cf.get((d['symbol'], d['date'], d['cross_ts']))
        exit_px = _f(exit_row.get('exit_px')) if exit_row else None
        entry_px, stop_px = _f(d['fill_px']), _f(d['stop'])
        entry_m = _et_minute(d['arm_ts'])
        exit_m = _et_minute(exit_row.get('exit_ts')) if exit_row else None
        if exit_row is None or exit_px is None or entry_px is None or stop_px is None \
                or entry_px == stop_px or entry_m is None or exit_m is None:
            n_unresolved += 1
            continue
        r_multiple = (exit_px - entry_px) / (entry_px - stop_px)
        candidates.append({
            'trade_date': d['date'], 'symbol': d['symbol'], 'entry_m': entry_m, 'exit_m': exit_m,
            'entry_ts': d['arm_ts'], 'exit_ts': exit_row.get('exit_ts'),
            'entry_px': entry_px, 'stop_px': stop_px, 'target_px': _f(d['target']),
            'exit_px': exit_px, 'exit_reason': exit_row.get('exit_reason') or None,
            'r_multiple': r_multiple,
        })
    if n_unresolved:
        print(f"NOTE: {entry_path}: {n_unresolved} filled row(s) excluded from the capped replay "
              f"(no resolved counterfactual exit — can't causally free a concurrency slot)")

    book_rows = [(c['trade_date'], c['entry_m'], c['exit_m'], c['symbol'], c) for c in candidates]
    taken = run_book(book_rows, max_per_day, max_concurrent)

    out = []
    for _day, _entry_m, _exit_m, _sym, c in taken:
        out.append({
            'strategy': STRATEGY_NAME, 'trade_date': c['trade_date'], 'symbol': c['symbol'],
            'entry_ts': c['entry_ts'], 'entry_px': c['entry_px'], 'shares': None,
            'stop_px': c['stop_px'], 'target_px': c['target_px'],
            'exit_ts': c['exit_ts'], 'exit_px': c['exit_px'], 'exit_reason': c['exit_reason'],
            'r_multiple': c['r_multiple'], 'pnl_usd': round(c['r_multiple'] * DEFAULT_RISK_USD, 2),
            'risk_usd': DEFAULT_RISK_USD, 'source': 'replay_capped',
        })
    return out


def _date_bounds(rows: list) -> tuple:
    """(min, max) trade_date among `rows`, or (None, None) if empty — the [start, end] window a
    delete-and-rebuild of this batch's own rows must be scoped to."""
    dates = [r['trade_date'] for r in rows if r.get('trade_date')]
    return (min(dates), max(dates)) if dates else (None, None)


def insert_fresh_rows(db: Database, rows: list, dry_run: bool) -> tuple:
    """Plain insert, no merge/match — paired with Database.delete_dry_trades for delete-and-rebuild
    idempotency on a source that is fully re-derived from CSV ledgers every run ('counterfactual_all',
    'replay_capped'): the caller deletes any existing rows of this source+date-range first, so every
    row here is guaranteed new; matching against other sources (as upsert_rows does for
    'backfill_journal') would wrongly collapse an intentionally-separate population into one row.
    Returns (inserted, failed)."""
    inserted = failed = 0
    for row in rows:
        if dry_run:
            inserted += 1
            continue
        rid = db.insert_dry_entry(row)
        if rid is None:
            failed += 1
            continue
        inserted += 1
        if row.get('exit_px') is not None or row.get('r_multiple') is not None:
            db.close_dry_trade(rid, exit_ts=row.get('exit_ts'), exit_px=row.get('exit_px'),
                                exit_reason=row.get('exit_reason'), r_multiple=row.get('r_multiple'),
                                pnl_usd=row.get('pnl_usd'))
    return inserted, failed


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
    ap.add_argument('--max-per-day', type=int, default=DEFAULT_MAX_PER_DAY)
    ap.add_argument('--max-concurrent', type=int, default=DEFAULT_MAX_CONCURRENT)
    args = ap.parse_args()

    cf_all_rows = build_ledger_rows(args.entry_ledger, args.cf_ledger)
    print(f"entry ledger: {len(cf_all_rows)} filled rows parsed from {args.entry_ledger} "
          f"(source=counterfactual_all — the uncapped all-armed population)")

    replay_rows = build_capped_replay_rows(args.entry_ledger, args.cf_ledger,
                                            args.max_per_day, args.max_concurrent)
    print(f"capped replay: {len(replay_rows)} trade(s) survive the {args.max_concurrent}-concurrent/"
          f"{args.max_per_day}-per-day cap (source=replay_capped)")

    db = Database()  # opens data/trades.db; migration 16 creates dry_trades if missing (empty table, idempotent)

    # 'counterfactual_all' and 'replay_capped' are fully re-derived from the CSVs every run — idempotent
    # by delete-then-insert, scoped to their own source tag and this run's own date range only.
    cf_lo, cf_hi = _date_bounds(cf_all_rows)
    if cf_lo:
        if not args.dry_run:
            deleted = db.delete_dry_trades(STRATEGY_NAME, 'counterfactual_all', start=cf_lo, end=cf_hi)
            print(f"counterfactual_all: deleted {deleted} existing row(s) in [{cf_lo}, {cf_hi}] before rebuild")
        else:
            print(f"counterfactual_all: would delete existing rows in [{cf_lo}, {cf_hi}] before rebuild")
    cf_inserted, cf_failed = insert_fresh_rows(db, cf_all_rows, args.dry_run)

    rp_lo, rp_hi = _date_bounds(replay_rows)
    if rp_lo:
        if not args.dry_run:
            deleted = db.delete_dry_trades(STRATEGY_NAME, 'replay_capped', start=rp_lo, end=rp_hi)
            print(f"replay_capped: deleted {deleted} existing row(s) in [{rp_lo}, {rp_hi}] before rebuild")
        else:
            print(f"replay_capped: would delete existing rows in [{rp_lo}, {rp_hi}] before rebuild")
    rp_inserted, rp_failed = insert_fresh_rows(db, replay_rows, args.dry_run)

    # 'backfill_journal' keeps its existing merge-across-sources upsert (journal wins over
    # counterfactual_all for the SAME real trade, 2026-09-14..09-25 legacy window) — unchanged.
    if not args.ledger_only:
        jrows = build_journal_rows(args.start, args.end)
        print(f"journal reconstruction: {len(jrows)} trade rows parsed for {args.start}..{args.end}")
    else:
        jrows = []
        print("journal reconstruction skipped (--ledger-only)")
    j_inserted, j_updated, j_skipped, j_failed = upsert_rows(db, jrows, args.dry_run)

    verb = 'would insert' if args.dry_run else 'inserted'
    print(f"{verb}: counterfactual_all={cf_inserted} (failed {cf_failed})  "
          f"replay_capped={rp_inserted} (failed {rp_failed})  "
          f"backfill_journal={j_inserted}/updated={j_updated}/skipped={j_skipped} (failed {j_failed})")

    if args.dry_run:
        return 0

    # By-source counts + the most recent day's capped replay detail (owner-facing summary).
    all_rows = db.get_dry_trades(STRATEGY_NAME)
    by_source: dict = {}
    for r in all_rows:
        by_source.setdefault(r['source'], []).append(r)
    print("\nBy-source row counts after this run:")
    for src in sorted(by_source):
        rs = by_source[src]
        # 'backfill_journal' rows have no per-trade exit_ts (day-level source, build_journal_rows'
        # docstring) — exit_ts-or-r_multiple is "resolved", matching hod_dry_ledger.py's predicate.
        closed = [x for x in rs if x.get('exit_ts') is not None or x.get('r_multiple') is not None]
        print(f"  {src}: {len(rs)} row(s), {len(closed)} closed")

    if replay_rows:
        last_day = max(r['trade_date'] for r in replay_rows)
        # trade_date comes back from sqlite as a datetime.date (detect_types=PARSE_DECLTYPES,
        # persistence/database.py) — str() it before comparing to last_day (a plain CSV string).
        day_rows = [r for r in by_source.get('replay_capped', []) if str(r['trade_date']) == last_day]
        day_r = sum(r['r_multiple'] for r in day_rows if r.get('r_multiple') is not None)
        syms = sorted(r['symbol'] for r in day_rows)
        print(f"\nCapped replay for {last_day}: n={len(day_rows)}  R={day_r:+.2f}  symbols={syms}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
