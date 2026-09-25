"""Live guardrail — cumulative-ledger auto-pause (docs/live_guardrails_spec_20260925.md).

Why: ORB went live 2026-05-19 and lost -$5,281 on 123 fills through 9/23 while
the backtest for the same months was positive. The `LATENCY TRIPWIRE` warning
fired for weeks and only wrote a log line, and every config change (9/17
resume, 9/21 catalyst-off) restarted the ramp's stage clock, so no rule ever
saw the CUMULATIVE live loss. The owner's words: "a bug lost money for five
months before it was flagged."

This module is the ONE spec shared by the daily check (`scripts/guardrail.py`)
and the live engines (their boot-time pre-open check), so a pause decided by
one is seen by the other — parity by construction, per CLAUDE.md's "ONE spec
for backtest and live" rule applied here to "ONE spec for the daily check and
the engine".

Ledger: `live_record()` reads every closed fill for a book from `data/trades.db`
since that book's FIRST LIVE FILL EVER (`first_fill_date`) — never a stage
start, never a config-change date, so a config change cannot reset what the
guardrail has seen (the exact defect this spec fixes).

Pause rule (frozen, spec G1): PAUSE iff, of the three, ANY fires:
  1. trailing-40-fill mean R <= the BT band's p5 for that book, with >= 20
     fills feeding that mean (`trading/ramp_bt_band.py` supplies the band).
  2. trailing-20-session $ <= -3 x stage_risk_usd x SESSION_MULT[book]
     (8 for orb, 4 for bull_flag).
  3. any single session <= -6 x stage_risk_usd.

State file: data/guardrail_state.json (override with env ONEMIL_GUARDRAIL_STATE,
    e.g. tests — see tests/conftest.py's autouse fixture)
    {"orb": {"paused_by_guardrail": true, "rule": "...", "reason": "...",
             "at_utc": "...", "numbers": {...}, "history": [...]},
     "bull_flag": {...}}

Clearing a pause is MANUAL and logged: `scripts/guardrail.py --clear BOOK "reason"`.
`hod_break` is PAUSABLE (docs/hod_live_resting_orders_spec_20260925.md, 2026-09-25): its
resting-order live path can place real orders, so the same G1 rule now applies to it,
scaled to ITS OWN risk (scripts/guardrail.py stage_risk_usd reads hod_break.risk_usd, not
bull_flag's trading.risk_per_trade). With zero live fills `LedgerStats.trailing_40_mean_r`
is None and both $ thresholds are 0 vs 0 — the rule cannot fire until hod_break's first
live fill, matching the spec's "pause-capable from its first fill" without a separate gate.
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TRADES_DB = ROOT / 'data' / 'trades.db'
STATE_PATH = ROOT / 'data' / 'guardrail_state.json'
TELEGRAM_SCRIPT = ROOT / 'scripts' / 'send_telegram_alert.py'
TELEGRAM_PREFIX = '[GUARDRAIL]'

#: Books with a live (or dry-live) order path. `hod_break` is ledgered but
#: never auto-paused (see module docstring).
BOOKS = ('orb', 'bull_flag', 'hod_break')
#: Books `scripts/guardrail.py --check` is allowed to pause.
PAUSABLE_BOOKS = ('orb', 'bull_flag', 'hod_break')

#: spec: "x8 (ORB) / x4 (BF)" in the trailing-20-session $ rule. hod_break added 2026-09-25
#: (docs/hod_live_resting_orders_spec_20260925.md): no session-count precedent of its own yet,
#: so it uses bull_flag's x4 (its max_per_day=8/max_concurrent=4 caps are closer to bull_flag's
#: cadence than ORB's 8-slot book) — a judgment call, not a backtested number; revisit once
#: hod_break has its own live session history.
SESSION_MULT = {'orb': 8, 'bull_flag': 4, 'hod_break': 4}
TRAILING_R_WINDOW = 40
TRAILING_R_MIN_FILLS = 20     # rule 1 only fires with >= this many R values in the window
TRAILING_SESSION_WINDOW = 20
SESSION_DOLLAR_MULT = 3       # rule 2: -3x stage risk x SESSION_MULT[book]
SINGLE_SESSION_MULT = 6       # rule 3: -6x stage risk

RULE_BAND_P5 = 'trailing_40_band_p5'
RULE_TRAILING_20_SESSION = 'trailing_20_session_usd'
RULE_SINGLE_SESSION = 'single_session_usd'


def _normalize_book(book: str) -> str:
    b = (book or '').strip().lower()
    if b not in BOOKS:
        raise ValueError(f"unknown book {b!r} — expected one of {BOOKS}")
    return b


@dataclass
class LedgerStats:
    """One book's cumulative live ledger since its first-ever fill."""
    book: str
    n_fills: int
    total_usd: float
    mean_r: Optional[float]
    trailing_40_mean_r: Optional[float]
    trailing_40_n: int
    trailing_20_session_usd: float
    worst_month: Optional[str]
    worst_month_usd: Optional[float]
    worst_session_date: Optional[str]
    worst_session_usd: Optional[float]
    first_fill_date: Optional[str]
    sessions: List[str] = field(default_factory=list)

    def line(self) -> str:
        """The one line the EOD report and guardrail.py print per book."""
        if self.n_fills == 0:
            return f"{self.book}: no live fills yet"
        r_txt = f"{self.trailing_40_mean_r:+.3f}" if self.trailing_40_mean_r is not None else "n/a"
        return (f"{self.book}: n={self.n_fills} since {self.first_fill_date} | "
                f"total ${self.total_usd:+,.0f} | trailing-{self.trailing_40_n}-fill mean R {r_txt} | "
                f"trailing-20-session ${self.trailing_20_session_usd:+,.0f} | "
                f"worst month {self.worst_month or 'n/a'} ${self.worst_month_usd or 0:+,.0f} | "
                f"worst session {self.worst_session_date or 'n/a'} ${self.worst_session_usd or 0:+,.0f}")


@dataclass
class PauseCheck:
    """The pause verdict for one book, carrying the numbers that triggered it."""
    book: str
    should_pause: bool
    rule: Optional[str]
    detail: str
    stats: LedgerStats


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def first_fill_date(book: str, db_path: Path = TRADES_DB) -> Optional[str]:
    """This book's first-ever closed live fill date.

    The ledger start a stage/config change must NEVER move (spec: "never
    reset by stage or config") — callers always derive `since` from this,
    not from a ramp-stage or config-change timestamp.
    """
    book = _normalize_book(book)
    if not Path(db_path).exists():
        logger.warning(f"guardrail: trades db missing at {db_path} — {book} "
                       f"treated as having no live fills yet")
        return None
    conn = _connect_ro(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(trade_date) AS d FROM trades "
            "WHERE strategy=? AND pnl IS NOT NULL", (book,)).fetchone()
    finally:
        conn.close()
    return row['d'] if row and row['d'] else None


def load_fills(book: str, db_path: Path = TRADES_DB,
               since: Optional[str] = None) -> List[Dict]:
    """Every closed fill for `book`, since its first-ever fill (or `since`). READ-ONLY."""
    book = _normalize_book(book)
    since = since if since is not None else first_fill_date(book, db_path)
    if since is None:
        return []
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(
            "SELECT trade_date, symbol, pnl, entry_price, stop_loss_price, "
            "shares, total_risk, exited_at "
            "FROM trades WHERE strategy=? AND trade_date>=? AND pnl IS NOT NULL "
            "ORDER BY trade_date, exited_at", (book, since)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def trade_risk_usd(row: Dict, stage_risk_usd: float) -> float:
    """That fill's own $ risk: `total_risk` when stored (it IS |entry-stop| x
    shares, computed at order time), else reconstructed as
    |entry_price - stop_loss_price| x shares, else the book's current stage
    risk — logged, since this is the fallback the spec calls out by name."""
    total_risk = row.get('total_risk')
    if total_risk not in (None, 0, 0.0):
        return abs(float(total_risk))
    entry, stop, shares = row.get('entry_price'), row.get('stop_loss_price'), row.get('shares')
    if entry is not None and stop is not None and shares:
        risk = abs(float(entry) - float(stop)) * abs(float(shares))
        if risk > 0:
            return risk
    logger.warning(
        f"guardrail: fill {row.get('trade_date')} {row.get('symbol')} has no "
        f"usable total_risk or entry/stop/shares — using stage risk "
        f"${stage_risk_usd:.0f} as its R denominator")
    return float(stage_risk_usd)


def trade_r(row: Dict, stage_risk_usd: float) -> Optional[float]:
    """pnl / that fill's own risk; None if pnl is missing or risk resolves to 0."""
    pnl = row.get('pnl')
    if pnl is None:
        return None
    risk = trade_risk_usd(row, stage_risk_usd)
    if not risk:
        logger.warning("guardrail: trade risk resolved to 0 — R undefined for this fill")
        return None
    return float(pnl) / risk


def live_record(book: str, stage_risk_usd: float = 0.0, db_path: Path = TRADES_DB,
                fills: Optional[List[Dict]] = None) -> LedgerStats:
    """The book's cumulative live ledger since its first-ever fill.

    `fills` lets a caller (tests, or a prefetched list) skip the DB read —
    `live_record` itself never assumes production `data/trades.db`.
    """
    book = _normalize_book(book)
    rows = fills if fills is not None else load_fills(book, db_path)
    if not rows:
        return LedgerStats(book=book, n_fills=0, total_usd=0.0, mean_r=None,
                           trailing_40_mean_r=None, trailing_40_n=0,
                           trailing_20_session_usd=0.0, worst_month=None,
                           worst_month_usd=None, worst_session_date=None,
                           worst_session_usd=None, first_fill_date=None, sessions=[])
    r_values: List[float] = []
    monthly: Dict[str, float] = {}
    session_pnl: Dict[str, float] = {}
    total = 0.0
    for row in rows:
        pnl = float(row.get('pnl') or 0.0)
        total += pnl
        td = row.get('trade_date') or ''
        session_pnl[td] = session_pnl.get(td, 0.0) + pnl
        month = td[:7]
        if month:
            monthly[month] = monthly.get(month, 0.0) + pnl
        r = trade_r(row, stage_risk_usd)
        if r is not None:
            r_values.append(r)
    sessions = sorted(session_pnl)
    trailing_sessions = sessions[-TRAILING_SESSION_WINDOW:]
    trailing_20_session_usd = sum(session_pnl[d] for d in trailing_sessions)
    trailing_40 = r_values[-TRAILING_R_WINDOW:]
    trailing_40_mean_r = (sum(trailing_40) / len(trailing_40)) if trailing_40 else None
    worst_month, worst_month_usd = (min(monthly.items(), key=lambda kv: kv[1])
                                    if monthly else (None, None))
    worst_session_date, worst_session_usd = min(session_pnl.items(), key=lambda kv: kv[1])
    return LedgerStats(
        book=book, n_fills=len(rows), total_usd=total,
        mean_r=(sum(r_values) / len(r_values)) if r_values else None,
        trailing_40_mean_r=trailing_40_mean_r, trailing_40_n=len(trailing_40),
        trailing_20_session_usd=trailing_20_session_usd,
        worst_month=worst_month, worst_month_usd=worst_month_usd,
        worst_session_date=worst_session_date, worst_session_usd=worst_session_usd,
        first_fill_date=rows[0].get('trade_date'), sessions=sessions,
    )


def band_p5_for(book: str, n: int, **band_kwargs) -> Optional[float]:
    """This book's BT band p5 for a trailing sample of `n` fills.

    Thin wrapper over `trading/ramp_bt_band.py` (the same reference each ramp
    checker already uses) so `evaluate_pause` never has to know how the band
    is built — only its p5. Returns None (NO-DATA) rather than raising, since
    a missing/unreadable reference must never crash the guardrail check.
    """
    from trading import ramp_bt_band as band_mod
    if n <= 0:
        return None
    try:
        if book == 'orb':
            ref = band_mod.orb_reference(band_kwargs.get('orb_catalyst_veto', False))
            r_values = band_mod.load_reference_r(ref, 'orb')
        elif book == 'bull_flag':
            ref = band_mod.bf_reference(band_kwargs.get('bf_min_daily_volume', 200_000))
            r_values = band_mod.load_reference_r(ref, 'bf')
        else:
            logger.warning(f"guardrail: no BT band reference for book {book!r}")
            return None
    except Exception as e:  # noqa: BLE001 - a decision aid must not crash
        logger.error(f"guardrail: BT band reference failed for {book} ({e}) — rule 1 NO-DATA")
        return None
    band = band_mod.bootstrap_band(r_values, n) if r_values else None
    return band.p5 if band else None


def evaluate_pause(stats: LedgerStats, stage_risk_usd: float,
                   band_p5: Optional[float]) -> PauseCheck:
    """The frozen pause rule (spec G1): OR of three thresholds, checked in order.

    A book outside PAUSABLE_BOOKS is answered honestly (never pauses) rather
    than raising, so a reporting loop over all of BOOKS stays simple.
    """
    book = stats.book
    if book not in PAUSABLE_BOOKS:
        return PauseCheck(book, False, None,
                          f"{book} is reported only — never auto-paused", stats)

    mult = SESSION_MULT[book]
    trailing_20_threshold = -SESSION_DOLLAR_MULT * stage_risk_usd * mult
    single_threshold = -SINGLE_SESSION_MULT * stage_risk_usd

    if (band_p5 is not None and stats.trailing_40_mean_r is not None
            and stats.trailing_40_n >= TRAILING_R_MIN_FILLS
            and stats.trailing_40_mean_r <= band_p5):
        return PauseCheck(
            book, True, RULE_BAND_P5,
            f"trailing-{stats.trailing_40_n}-fill mean R {stats.trailing_40_mean_r:+.3f} "
            f"<= BT band p5 {band_p5:+.3f}", stats)

    if stats.trailing_20_session_usd <= trailing_20_threshold:
        return PauseCheck(
            book, True, RULE_TRAILING_20_SESSION,
            f"trailing-20-session ${stats.trailing_20_session_usd:+,.0f} <= "
            f"${trailing_20_threshold:+,.0f} (-{SESSION_DOLLAR_MULT}x stage risk "
            f"${stage_risk_usd:,.0f} x{mult})", stats)

    if stats.worst_session_usd is not None and stats.worst_session_usd <= single_threshold:
        return PauseCheck(
            book, True, RULE_SINGLE_SESSION,
            f"session {stats.worst_session_date} ${stats.worst_session_usd:+,.0f} <= "
            f"${single_threshold:+,.0f} (-{SINGLE_SESSION_MULT}x stage risk "
            f"${stage_risk_usd:,.0f})", stats)

    return PauseCheck(book, False, None, "within all three thresholds", stats)


# ---------------------------------------------------------------- state file

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


def resolve_state_path(path: Optional[Path]) -> Path:
    """Resolve the state-file path for a call.

    Explicit `path` wins. Else `ONEMIL_GUARDRAIL_STATE` (set by tests via
    `tests/conftest.py` so no test ever reads/writes production state). Else
    the current `STATE_PATH` module attribute (read at CALL time, not bound
    as a def-time default, so `monkeypatch.setattr(gr, 'STATE_PATH', ...)`
    still works)."""
    if path is not None:
        return Path(path)
    env = os.environ.get('ONEMIL_GUARDRAIL_STATE')
    return Path(env) if env else Path(STATE_PATH)


def load_state(path: Optional[Path] = None) -> Dict[str, Dict]:
    """Read the whole guardrail state file. Missing/corrupt -> no books paused."""
    path = resolve_state_path(path)
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        logger.error(f"guardrail state unreadable at {path} ({e}) — treating "
                     f"every book as UNPAUSED; fix the file")
        return {}


def save_state(state: Dict[str, Dict], path: Optional[Path] = None) -> None:
    """Persist the whole state file (write temp, replace — atomic-ish)."""
    p = resolve_state_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(p)


def is_paused(book: str, path: Optional[Path] = None) -> bool:
    """True if `book`'s pre-open check must refuse real orders."""
    book = _normalize_book(book)
    return bool(load_state(path).get(book, {}).get('paused_by_guardrail', False))


def send_guardrail_telegram(text: str, script: Path = TELEGRAM_SCRIPT) -> bool:
    """Fire-and-forget guardrail Telegram. Never raises."""
    try:
        res = subprocess.run([sys.executable, str(script), text],
                             capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            logger.warning(f"guardrail Telegram not sent (rc={res.returncode}): "
                           f"{(res.stderr or res.stdout or '').strip()[:200]}")
            return False
        return True
    except Exception as e:  # noqa: BLE001 - alerting must never break a check
        logger.warning(f"guardrail Telegram failed ({e}) — state is persisted regardless")
        return False


def pause_book(check: PauseCheck, stage_risk_usd: float, path: Optional[Path] = None,
               notify: bool = True) -> Dict:
    """Latch a PAUSE into the state file, log ERROR, send the Telegram.

    Idempotent: re-pausing an already-paused book keeps the original `at_utc`
    and appends the new numbers to history, so the boot-time check (which
    only reads `paused_by_guardrail`) sees no flapping.
    """
    if not check.should_pause:
        raise ValueError(f"pause_book called on a non-pausing check for {check.book}")
    state = load_state(path)
    now = datetime.now(timezone.utc).isoformat()
    entry = state.get(check.book, {})
    already = bool(entry.get('paused_by_guardrail'))
    entry.update({
        'paused_by_guardrail': True,
        'rule': check.rule,
        'reason': check.detail,
        'at_utc': entry.get('at_utc') if already else now,
        'numbers': {
            'n_fills': check.stats.n_fills,
            'total_usd': check.stats.total_usd,
            'trailing_40_mean_r': check.stats.trailing_40_mean_r,
            'trailing_40_n': check.stats.trailing_40_n,
            'trailing_20_session_usd': check.stats.trailing_20_session_usd,
            'worst_session_date': check.stats.worst_session_date,
            'worst_session_usd': check.stats.worst_session_usd,
            'stage_risk_usd': stage_risk_usd,
        },
    })
    history = entry.setdefault('history', [])
    history.append({'action': 'pause', 'at_utc': now, 'rule': check.rule, 'reason': check.detail})
    state[check.book] = entry
    save_state(state, path)
    logger.error(f"[GUARDRAIL] {check.book} PAUSED: {check.detail}")
    if notify and not already:
        send_guardrail_telegram(f"{TELEGRAM_PREFIX} {check.book} PAUSED: {check.detail}")
    return entry


def clear_pause(book: str, reason: str, by: Optional[str] = None,
                path: Optional[Path] = None) -> Dict:
    """MANUAL pause clear — logs who and why. Never called automatically."""
    book = _normalize_book(book)
    if not (reason or '').strip():
        raise ValueError("clearing a guardrail pause requires a reason")
    by = by or current_user()
    now = datetime.now(timezone.utc).isoformat()
    state = load_state(path)
    entry = state.get(book, {})
    if not entry.get('paused_by_guardrail'):
        logger.warning(f"{book} was not paused — clear is a no-op (reason: {reason})")
    history = entry.setdefault('history', [])
    history.append({'action': 'clear', 'reason': reason, 'by': by, 'at_utc': now})
    entry['paused_by_guardrail'] = False
    entry['cleared_reason'] = reason
    entry['cleared_by'] = by
    entry['cleared_at_utc'] = now
    state[book] = entry
    save_state(state, path)
    logger.warning(f"[GUARDRAIL] {book} CLEARED by {by}: {reason}")
    return entry
