"""Ramp STAGE START — the one documented source of "when did this stage begin".

A ramp checker measures stage P&L, stage sessions and stage parity defects over
a window that begins at the stage start. Both `scripts/bf_ramp_check.py` and
`scripts/orb_ramp_check.py` used to hardcode that date as their book's LAUNCH,
so a config change that is a new stage by the scaling plan's own definition
(docs/scaling_plan_2026.md) silently kept counting the old stage's trades,
sessions and parity defects.

A new stage starts when the CONFIG THE BOOK TRADES changes materially — new
size, new slot count, new gate set. History is never rewritten: the old stage
row stays in this table (and old reds stay in logs/green_streak.json) — they
simply fall outside the current stage's window.

    from trading import ramp_stage
    start, reason = ramp_stage.current('orb')

`--stage-start YYYY-MM-DD` on either checker overrides the table (reason
"CLI override"). To open a NEW stage, append a row here and to the book's
playbook table (docs/bf_p1_ramp.md / docs/orb_p1_style_ramp_proposal.md) in
the same commit — the reason string is what the checker prints.

Why `trading/` and not `scripts/`: same argument as trading/ramp_freeze.py —
ONE spec, consumed by entry points in more than one import root.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

BOOKS = ('bf', 'orb')

# KEEP IN SYNC with docs/bf_p1_ramp.md and docs/orb_p1_style_ramp_proposal.md.
# Chronological; the LAST row of a book is its current stage. Never delete a
# row — an old date is history, and `--stage-start <old date>` must keep working.
STAGE_STARTS: Dict[str, List[Dict[str, str]]] = {
    'bf': [
        {'start': '2026-09-07',
         'reason': 'P1 live launch — L0 $150 (docs/bf_p1_ramp.md)'},
        {'start': '2026-09-21',
         'reason': 'ADV-gate-off stage (min_daily_volume 0) — ZERO live '
                   'trades under the config that boots Monday '
                   '(docs/scaling_plan_2026.md)'},
    ],
    'orb': [
        {'start': '2026-08-17',
         'reason': 'B+ live — 3 slots, $10,000 budget'},
        {'start': '2026-09-21',
         'reason': '8-slot + catalyst-off + latency-fix + 50bps stage'},
    ],
}


def _normalize_book(book: str) -> str:
    b = (book or '').strip().lower()
    if b not in BOOKS:
        raise ValueError(f"unknown book {book!r} — expected one of {BOOKS}")
    return b


def history(book: str) -> List[Dict[str, str]]:
    """Every stage start recorded for `book`, oldest first."""
    return list(STAGE_STARTS[_normalize_book(book)])


def current(book: str) -> Tuple[str, str]:
    """(start ISO date, reason) for this book's CURRENT stage."""
    row = history(book)[-1]
    return row['start'], row['reason']


def previous(book: str) -> Tuple[str, str]:
    """(start, reason) of the stage before the current one, for reporting."""
    rows = history(book)
    if len(rows) < 2:
        return rows[0]['start'], rows[0]['reason']
    return rows[-2]['start'], rows[-2]['reason']


def resolve(book: str, override: str = None) -> Tuple[str, str]:
    """Stage start to measure from: the CLI override if given, else the table.

    A malformed override is a hard error — a silently-wrong window would
    silently mis-measure every gate.
    """
    if override:
        date.fromisoformat(override)      # raises on a malformed date
        return override, 'CLI override (--stage-start)'
    return current(book)


def add_stage_start_arg(parser, book: str) -> None:
    """Attach `--stage-start` to a ramp checker, documenting the default."""
    start, reason = current(book)
    parser.add_argument(
        '--stage-start', default=None, metavar='YYYY-MM-DD',
        help=f'measure the stage from this date '
             f'(default {start} — {reason}; table: trading/ramp_stage.py)')


def line(book: str, start: str, reason: str) -> str:
    """The one line a checker prints under its stage header."""
    prev_start, _ = previous(book)
    tail = (f'; the previous stage started {prev_start} and its trades, '
            f'sessions and parity defects are OUT of this window'
            if prev_start != start else '')
    return f"  stage start {start} — {reason}{tail}"
