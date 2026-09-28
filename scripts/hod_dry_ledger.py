#!/usr/bin/env python3
"""HOD-break DRY-RUN ledger: print day-by-day and cumulative book since a start date.

Prints: date, trades, R, $, green/red, per-symbol list, then totals with worst/best day, mean R per trade,
and the count of still-OPEN rows (excluded from trades/R/$ — see ledger_from_db).

Usage: python3 scripts/hod_dry_ledger.py [--start YYYY-MM-DD] [--json]
Default start: 2026-09-14. Each weekday from start to today shows one line with its DRY-RUN EXECUTABLE book.
"""
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ET = ZoneInfo('US/Eastern')

DRY_RUN_RE = re.compile(
    r'DRY-RUN EXECUTABLE book.*?:\s*(\d+)\s+trades?,\s+([-+]?[\d.]+)R,\s+\$([-+]?[\d,]+)\s*\|\s*(\[.*\])'
)

# dry_trades.source values that make up the CAPPED dry book (scripts/backfill_dry_trades.py module
# docstring): 'backfill_journal' already applies trading.hod_break.run_book's cap when it's built
# (2026-09-14..09-25 legacy window); 'replay_capped' replays the same cap for 2026-09-26 onward. Every
# OTHER source ('counterfactual_all', and the engine's own real-time 'live_dry' insert — both written by
# the resting-fill simulator, which has NO max_per_day/max_concurrent check — see that script's
# diagnosis note) is the UNCAPPED all-armed population: reported on its own line, NEVER summed into the
# dry book (9/28 defect: the two were conflated, showing 33 all-armed counterfactual fills as if they
# were the ~12/day capped book).
DRY_BOOK_SOURCES = {'backfill_journal', 'replay_capped'}


def parse_dry_run_line(output: str):
    """Parse the '  DRY-RUN EXECUTABLE book' line; return (trades, R, USD, symbols_with_r, is_green) or None.

    Line format: "  DRY-RUN EXECUTABLE book (first 12/day, 4 concurrent, logged sizes): 7 trades, +9.6R,
    $+922 | [('CRWD', 1.1), ('FBL', 2.26), ...]"
    """
    for line in output.split('\n'):
        if 'DRY-RUN EXECUTABLE book' in line:
            m = DRY_RUN_RE.search(line)
            if m:
                trades = int(m.group(1))
                r = float(m.group(2))
                usd = float(m.group(3).replace(',', ''))
                symbols_str = m.group(4)
                try:
                    symbols = ast.literal_eval(symbols_str)  # [('FBL', -1.08), ...]
                except (ValueError, SyntaxError):
                    symbols = []
                is_green = r > 0
                return (trades, r, usd, symbols, is_green)
    if 'no signals today' in output:
        # Early-return path in hod_break_eod_check.py: zero signals, no DRY-RUN line at all.
        return (0, 0.0, 0.0, [], False)
    return None


def analyze_day(day_str: str):
    """Call hod_break_eod_check.py for a day and extract DRY-RUN EXECUTABLE metrics."""
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/hod_break_eod_check.py', day_str],
            capture_output=True, text=True, timeout=150, cwd=ROOT
        )
        if result.returncode != 0:
            print(f"  [{day_str}] hod_break_eod_check.py exited {result.returncode}: "
                  f"{result.stderr.strip()[-300:]}", file=sys.stderr)
            return None
        parsed = parse_dry_run_line(result.stdout)
        if parsed is None:
            print(f"  [{day_str}] WARNING: no DRY-RUN EXECUTABLE line found in output "
                  f"(no trading data for this day?)", file=sys.stderr)
        return parsed
    except subprocess.TimeoutExpired:
        print(f"  [{day_str}] ERROR: hod_break_eod_check.py timed out after 150s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [{day_str}] ERROR: {e}", file=sys.stderr)
        return None


def _read_dry_trades(start_date_str: str, end_date: 'datetime.date', db=None):
    """Shared DB read for ledger_from_db/armed_population_from_db. Returns the raw `dry_trades` rows for
    the window, or None on a read failure (never on merely-empty — callers distinguish). `db` is
    injectable for tests; production always uses the real Database() (opens data/trades.db)."""
    from persistence.database import Database
    try:
        db = db or Database()
        return db.get_dry_trades('hod_break', start=start_date_str, end=end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"[db] could not read dry_trades ({e}) — use --rebuild", file=sys.stderr)
        return None


def _ledger_rows_to_table(rows: list) -> list:
    """Group `dry_trades` rows (already filtered to one population — the capped dry book or the
    all-armed population, see DRY_BOOK_SOURCES) into the per-day
    (date_str, trades, R, USD, symbols_with_r, status, open_n) table both ledger_from_db and
    armed_population_from_db print.

    Only CLOSED rows count toward trades/R/$/symbols — an open dry-run position has no realized R or
    pnl_usd yet, and the defect this fixes (9/28: a day showing 130 trades / -5.2R / $0 instead of 67 /
    +6.2R / +$891) was exactly stale/duplicate rows being counted as if resolved. "Closed" is exit_ts
    not null OR r_multiple not null, not exit_ts alone (found rebuilding the 9/28 fix: build_journal_rows
    rows have NO per-trade exit_ts by construction — day-level source, see that function's docstring —
    so an exit_ts-only test silently showed every 'backfill_journal' trade as open/uncounted forever).
    Open rows are still tallied, via `open_n`, so they are visible rather than silently dropped."""
    by_date: dict = {}
    for r in rows:
        d = str(r['trade_date'])
        by_date.setdefault(d, []).append(r)
    table = []
    for d in sorted(by_date):
        day_rows = by_date[d]
        closed = [r for r in day_rows if r.get('exit_ts') is not None or r.get('r_multiple') is not None]
        open_n = len(day_rows) - len(closed)
        trades = len(closed)
        dr = sum(float(r['r_multiple']) for r in closed if r.get('r_multiple') is not None)
        dusd = sum(float(r['pnl_usd']) for r in closed if r.get('pnl_usd') is not None)
        symbols = [(r['symbol'], round(float(r['r_multiple']), 2)) for r in closed if r.get('r_multiple') is not None]
        is_green = dr > 0
        status = 'FLAT' if trades == 0 or dr == 0 else ('GREEN' if is_green else 'RED')
        table.append((d, trades, dr, dusd, symbols, status, open_n))
    return table


def ledger_from_db(start_date_str: str, end_date: 'datetime.date', db=None):
    """Fast path (9/28 "hod dry-run is not in the DB???"): read the day-by-day CAPPED dry book straight
    from `dry_trades` instead of re-parsing journalctl/session_archive on every call. Returns a list of
    (date_str, trades, R, USD, symbols_with_r, status, open_n) tuples, or None if unreachable/empty so
    callers can fall back to --rebuild.

    Only rows whose source is in DRY_BOOK_SOURCES count as the dry book (9/28 defect: the resting-fill
    simulator's uncapped all-armed population — source 'counterfactual_all'/'live_dry' — was being
    reported as if it were the ~12/day 4-concurrent book; see armed_population_from_db for that
    population, reported separately, never summed here)."""
    rows = _read_dry_trades(start_date_str, end_date, db)
    if not rows:
        return None
    dry_rows = [r for r in rows if r.get('source') in DRY_BOOK_SOURCES]
    if not dry_rows:
        return None
    return _ledger_rows_to_table(dry_rows)


def armed_population_from_db(start_date_str: str, end_date: 'datetime.date', db=None):
    """The UNCAPPED all-armed population — every `dry_trades` row whose source is NOT in
    DRY_BOOK_SOURCES (see that constant's docstring) — same per-day table shape as ledger_from_db, over
    the same window. None if unreachable/empty. Reported on its own line, NEVER summed with the capped
    dry book (ledger_from_db)."""
    rows = _read_dry_trades(start_date_str, end_date, db)
    if not rows:
        return None
    armed_rows = [r for r in rows if r.get('source') not in DRY_BOOK_SOURCES]
    if not armed_rows:
        return None
    return _ledger_rows_to_table(armed_rows)


def _print_report(ledger: list, skipped: list, json_output: bool) -> None:
    """All stdout for one run: per-day lines (or JSON), then totals including the open-row count. Takes
    the fully-built `ledger` (list of (date, trades, R, USD, symbols, status, open_n) tuples, from either
    ledger_from_db or the journal-reconstruction fallback in main()) so it is testable without a DB or
    subprocess."""
    if json_output:
        print(json.dumps([{
            'date': d, 'trades': t, 'R': r, 'USD': u, 'status': st, 'open': o,
            'symbols': [{'symbol': s, 'R': sr} for s, sr in syms]
        } for d, t, r, u, syms, st, o in ledger], indent=2))
        return

    print(f"{'Date':<12} {'Trades':>6} {'R':>7} {'$':>10} {'Status':<5} Symbols (R)")
    print('-' * 100)

    total_trades = 0; total_r = 0.0; total_usd = 0.0; total_open = 0; green_days = 0; all_daily_rs = []
    worst_day = (None, float('inf')); best_day = (None, float('-inf'))

    for day_str, trades, dr, dusd, symbols, status, open_n in ledger:
        sym_str = ', '.join(f"{s} {sr:+.2f}" for s, sr in symbols) if symbols else '(none)'
        print(f"{day_str}  {trades:6d} {dr:+7.1f} {dusd:+10,.0f}  {status:<5}  {sym_str}")
        total_trades += trades
        total_r += dr
        total_usd += dusd
        total_open += open_n
        all_daily_rs.append(dr)
        if status == 'GREEN':
            green_days += 1
        if dr < worst_day[1]:
            worst_day = (day_str, dr)
        if dr > best_day[1]:
            best_day = (day_str, dr)

    print('-' * 100)
    total_days = len(ledger)
    if skipped:
        print(f"(skipped {len(skipped)} day(s) with no parseable DRY-RUN line: {', '.join(skipped)})")
    if total_days == 0:
        print("No data.")
        return
    green_pct = f"{green_days}/{total_days}"
    mean_r_per_trade = total_r / total_trades if total_trades else 0.0
    se_r = (np.std(all_daily_rs, ddof=1) / np.sqrt(len(all_daily_rs))) if len(all_daily_rs) > 1 else 0.0

    print(f"TOTAL:       {total_trades:6d} {total_r:+7.1f} {total_usd:+10,.0f}  {green_pct:<5}")
    print(f"open: {total_open}")
    print(f"Worst day: {worst_day[0]} ({worst_day[1]:+.1f}R)")
    print(f"Best day:  {best_day[0]} ({best_day[1]:+.1f}R)")
    print(f"Mean R/trade: {mean_r_per_trade:+.2f} ± {se_r:.2f}")


def _print_armed_population_line(armed: list) -> None:
    """One summary line for the UNCAPPED all-armed population (armed_population_from_db) — deliberately
    NOT a per-day table like _print_report's, so it can never be visually mistaken for, or summed with,
    the capped dry book."""
    if not armed:
        return
    total_trades = sum(t for _, t, _, _, _, _, _ in armed)
    total_r = sum(r for _, _, r, _, _, _, _ in armed)
    total_usd = sum(u for _, _, _, u, _, _, _ in armed)
    total_open = sum(o for _, _, _, _, _, _, o in armed)
    print(f"\nALL-ARMED population (uncapped — source counterfactual_all/live_dry, NOT the dry book), "
          f"{armed[0][0]}..{armed[-1][0]}: {total_trades} closed  {total_r:+.1f}R  ${total_usd:+,.0f}  open: {total_open}")


def main():
    argv = list(sys.argv[1:])
    start_date_str = '2026-09-14'
    json_output = False
    rebuild = False

    i = 0
    while i < len(argv):
        if argv[i] == '--start' and i + 1 < len(argv):
            start_date_str = argv[i + 1]
            i += 2
        elif argv[i] == '--json':
            json_output = True
            i += 1
        elif argv[i] == '--rebuild':
            rebuild = True
            i += 1
        else:
            i += 1

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    today = datetime.now(timezone.utc).astimezone(ET).date()

    ledger = []  # (date_str, trades, R, USD, symbols_with_r, status, open_n)
    skipped = []

    armed = None
    if not rebuild:
        ledger = ledger_from_db(start_date_str, today)
        armed = armed_population_from_db(start_date_str, today)
    if ledger is None or rebuild:
        if not rebuild:
            print("[db] dry_trades empty for this range — falling back to journal reconstruction "
                  "(pass --rebuild to force this path)", file=sys.stderr)
        ledger = []
        current_date = start_date
        while current_date <= today:
            if current_date.weekday() < 5:  # weekday only
                day_str = current_date.strftime('%Y-%m-%d')
                result = analyze_day(day_str)
                if result:
                    trades, dr, dusd, symbols, is_green = result
                    status = 'FLAT' if trades == 0 or dr == 0 else ('GREEN' if is_green else 'RED')
                    ledger.append((day_str, trades, dr, dusd, symbols, status, 0))  # no open-row concept here
                else:
                    skipped.append(day_str)
            current_date += timedelta(days=1)

    _print_report(ledger, skipped, json_output)
    if not rebuild and not json_output:
        _print_armed_population_line(armed)


if __name__ == '__main__':
    main()
