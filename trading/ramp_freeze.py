"""Ramp FREEZE state — the Gate-1 parity breach latch (docs/scaling_plan_2026.md).

A FREEZE means "we do not know what we are running": size stays exactly where
it is, entries continue, and the stage clock STOPS until the breach is
explained. It is never a demotion.

Why this module lives in `trading/` and not `scripts/`: it is ONE spec shared
by four entry points that live in two different import roots — the parity
producers (`scripts/daily_green_check.py`, `scripts/bf_decision_parity.py`)
SET it and the ramp checkers (`scripts/{bf,orb}_ramp_check.py`) READ it. Every
one of those already puts the repo ROOT on sys.path (they import `config`,
`trading.*`, `data_sources.*`), so `trading/` is the one place all four can
import from, and it is this repo's established home for a single shared spec
(trading/two_tier_filter.py, trading/bf_trail.py, trading/orb_pm_mult.py).
`scripts/report_common.py` is a REPORTING helper imported only by report
scripts; freeze state is trading governance, not reporting.

State file: logs/ramp_freeze.json

    {"bf":  {"frozen": true, "since": "2026-09-18",
             "reason": "...", "by": "ec2-user",
             "frozen_dates": ["2026-09-18"], "history": [...]},
     "orb": {...}}

Clearing a freeze is MANUAL and logged (who/why) — never automatic.
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = ROOT / 'logs' / 'ramp_freeze.json'
TELEGRAM_SCRIPT = ROOT / 'scripts' / 'send_telegram_alert.py'
TELEGRAM_PREFIX = '[RAMP FREEZE]'
BOOKS = ('bf', 'orb')


@dataclass
class FreezeState:
    """One book's freeze record."""
    book: str
    frozen: bool = False
    since: Optional[str] = None
    reason: Optional[str] = None
    by: Optional[str] = None
    frozen_dates: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {'frozen': self.frozen, 'since': self.since,
                'reason': self.reason, 'by': self.by,
                'frozen_dates': sorted(set(self.frozen_dates)),
                'history': self.history}

    def line(self) -> str:
        """The one line a ramp checker prints when this book is frozen."""
        return (f"FROZEN since {self.since}: {self.reason} "
                f"(set by {self.by}; clear with "
                f"--clear-freeze {self.book} \"<reason>\")")


def _normalize_book(book: str) -> str:
    b = (book or '').strip().lower()
    if b not in BOOKS:
        raise ValueError(f"unknown book {book!r} — expected one of {BOOKS}")
    return b


def current_user() -> str:
    """Best-effort identity for the audit trail (never raises)."""
    for env in ('SUDO_USER', 'USER', 'LOGNAME'):
        v = os.environ.get(env)
        if v:
            return v
    try:
        return getpass.getuser()
    except Exception as e:  # noqa: BLE001 - identity is best-effort only
        logger.warning(f"could not determine current user ({e}) — 'unknown'")
        return 'unknown'


def _resolve(path: Optional[Path]) -> Path:
    """Late-bound state path: module-level FREEZE_PATH unless overridden.

    Resolved at CALL time (not as a default argument) so tests and tools can
    redirect the state file by setting ramp_freeze.FREEZE_PATH.
    """
    return Path(path) if path is not None else FREEZE_PATH


def load_state(path: Optional[Path] = None) -> Dict[str, FreezeState]:
    """Read the whole freeze file. Missing/corrupt -> all books unfrozen.

    A corrupt file is reported loudly (it is state we rely on) but must not
    crash a report tool.
    """
    path = _resolve(path)
    raw: Dict = {}
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.error(f"ramp freeze state unreadable at {path} ({e}) — "
                     f"treating every book as UNFROZEN; fix the file")
    out: Dict[str, FreezeState] = {}
    for book in BOOKS:
        d = raw.get(book) or {}
        out[book] = FreezeState(
            book=book,
            frozen=bool(d.get('frozen')),
            since=d.get('since'),
            reason=d.get('reason'),
            by=d.get('by'),
            frozen_dates=list(d.get('frozen_dates') or []),
            history=list(d.get('history') or []),
        )
    return out


def save_state(state: Dict[str, FreezeState],
               path: Optional[Path] = None) -> None:
    """Persist the whole freeze file (atomic-ish: write temp, replace)."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps({b: s.to_dict() for b, s in state.items()},
                              indent=1, sort_keys=True))
    tmp.replace(p)


def get(book: str, path: Optional[Path] = None) -> FreezeState:
    """This book's freeze record (never None)."""
    return load_state(path)[_normalize_book(book)]


def is_frozen(book: str, path: Optional[Path] = None) -> bool:
    return get(book, path).frozen


def send_freeze_telegram(book: str, reason: str, since: str,
                         script: Path = TELEGRAM_SCRIPT) -> bool:
    """Fire-and-forget '[RAMP FREEZE]' Telegram. Never raises."""
    msg = (f"{TELEGRAM_PREFIX} {book.upper()} FROZEN since {since}: {reason}\n"
           f"Size stays put, entries continue, the stage clock STOPS. "
           f"Clear manually once explained.")
    try:
        res = subprocess.run([sys.executable, str(script), msg],
                             capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            logger.warning(f"freeze Telegram not sent (rc={res.returncode}): "
                           f"{(res.stderr or res.stdout or '').strip()[:200]}")
            return False
        return True
    except Exception as e:  # noqa: BLE001 - alerting must never break a check
        logger.warning(f"freeze Telegram failed ({e}) — freeze state is "
                       f"persisted regardless")
        return False


def set_freeze(book: str, reason: str, day: Optional[str] = None,
               by: Optional[str] = None, path: Optional[Path] = None,
               notify: bool = True) -> FreezeState:
    """Freeze `book` for a parity breach on `day` (default: today).

    Idempotent on `since`: re-freezing an already-frozen book keeps the
    original freeze date (the stage clock stopped THEN) and records the new
    reason in history. `day` is added to frozen_dates so the session counter
    can exclude it permanently, even after the freeze is cleared.
    """
    book = _normalize_book(book)
    day = day or date.today().isoformat()
    by = by or current_user()
    state = load_state(path)
    s = state[book]
    was_frozen = s.frozen
    if not was_frozen:
        s.since = day
        s.frozen = True
    s.reason = reason
    s.by = by
    if day not in s.frozen_dates:
        s.frozen_dates.append(day)
    s.history.append({'action': 'freeze', 'day': day, 'reason': reason,
                      'by': by, 'at_utc': datetime.now(timezone.utc).isoformat()})
    save_state(state, path)
    logger.error(f"RAMP FREEZE set on {book.upper()} ({day}): {reason} "
                 f"[by {by}] — stage clock stopped, size unchanged")
    if notify and not was_frozen:
        send_freeze_telegram(book, reason, s.since or day)
    return s


def clear_freeze(book: str, reason: str, by: Optional[str] = None,
                 path: Optional[Path] = None,
                 day: Optional[str] = None) -> FreezeState:
    """MANUAL freeze clear — logs who and why. Never called automatically.

    frozen_dates are KEPT: those sessions did not count toward the stage
    minimum and must not retroactively start counting.
    """
    book = _normalize_book(book)
    if not (reason or '').strip():
        raise ValueError("clearing a freeze requires a reason")
    by = by or current_user()
    day = day or date.today().isoformat()
    state = load_state(path)
    s = state[book]
    if not s.frozen:
        logger.warning(f"{book.upper()} was not frozen — clear is a no-op "
                       f"(reason: {reason})")
    s.history.append({'action': 'clear', 'day': day, 'reason': reason,
                      'by': by, 'was_frozen_since': s.since,
                      'at_utc': datetime.now(timezone.utc).isoformat()})
    s.frozen = False
    s.since = None
    s.reason = None
    s.by = by
    save_state(state, path)
    logger.warning(f"RAMP FREEZE CLEARED on {book.upper()} by {by}: {reason}")
    return s


def _weekdays(start: str, end: str) -> List[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    for i in range((d1 - d0).days + 1):
        d = date.fromordinal(d0.toordinal() + i)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out


def frozen_sessions(book: str, start: str, end: str,
                    path: Optional[Path] = None) -> List[str]:
    """Sessions in [start, end] that must NOT count toward the stage minimum.

    Two sources, unioned:
      * recorded frozen_dates (the days a breach was detected), and
      * every weekday from `since` to `end` while the book is STILL frozen —
        a freeze that is never cleared freezes the clock, not just its first
        day.
    """
    book = _normalize_book(book)
    s = get(book, path)
    days = {d for d in s.frozen_dates if start <= d <= end}
    if s.frozen and s.since:
        lo = max(s.since, start)
        if lo <= end:
            days |= set(_weekdays(lo, end))
    return sorted(days)


def unfrozen_sessions(book: str, sessions: Sequence[str],
                      path: Optional[Path] = None) -> List[str]:
    """`sessions` minus the frozen ones (the stage clock actually accrued)."""
    if not sessions:
        return []
    frozen = set(frozen_sessions(book, min(sessions), max(sessions), path))
    return [d for d in sessions if d not in frozen]


def add_clear_freeze_arg(parser) -> None:
    """Attach the shared `--clear-freeze BOOK "reason"` flag to a CLI."""
    parser.add_argument(
        '--clear-freeze', nargs=2, metavar=('BOOK', 'REASON'),
        help='MANUALLY clear a parity freeze (bf|orb) — logs who and why. '
             'Frozen sessions stay excluded from the stage clock.')


def handle_clear_freeze(argv_pair, path: Optional[Path] = None) -> str:
    """Execute `--clear-freeze BOOK REASON`; returns the line to print."""
    book, reason = argv_pair
    s = clear_freeze(book, reason, path=path)
    return (f"FREEZE CLEARED on {book.upper()} by {s.by}: {reason} "
            f"(frozen sessions stay excluded from the stage clock)")
