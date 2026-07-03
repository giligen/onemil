"""Shared helpers for the daily green-check + weekly report Telegrams.

The daily check computes the day's OPERATIONAL-GREEN verdict — the unit the
2026-07-06 ramp policy advances on — and persists a streak file that
`orb_ramp_check.py` and the weekly report read. The weekly report is the
owner's decision document (P&L, gate progress, edge capture, flags).

All loaders are top-level functions so tests can monkeypatch them.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trading.exit_reasons import is_known, needs_reconcile  # noqa: E402

STREAK_PATH = ROOT / 'logs' / 'green_streak.json'
SELECTION_AUDIT = ROOT / 'logs' / 'orb_selection_audit.jsonl'
STAGE_RISK_USD = 1000.0      # Stage 0 — keep in sync with orb.yaml sizing
MODEL_RISK_USD = 3000.0      # BT $100K-model risk (study convention)
STAGE_SCALE = STAGE_RISK_USD / MODEL_RISK_USD
RAMP_START = '2026-05-18'
ADVANCE_LOSS_FLOOR = -7500.0   # Stage 0: -1 x (daily limit 1500 x 5)
GREEN_SESSIONS_NEEDED = 10

# Live-side reject reasons that legitimately explain a BT pick not trading
# (live-only gates BT cannot model). pdr_veto is NOT here: BT applies the
# same veto, so a live pdr_veto on a BT-selected pick would be a parity bug.
EXPLAINED_LIVE_REJECTS = {'spread_gate', 'insufficient_bp',
                          'fcfs_other_strategy', 'halted'}


# ---------------------------------------------------------------------------
# Loaders (monkeypatch points)
# ---------------------------------------------------------------------------

def latest_bt_trades_csv() -> Optional[str]:
    paths = sorted(glob.glob(str(ROOT / 'analysis_results' / 'orb_static_lock_trades.csv')))
    return paths[-1] if paths else None


def load_bt_selected(day: str) -> List[Dict]:
    """BT-defended selection rows (post Q1-filter, post PDR-veto) for a date."""
    import pandas as pd
    path = latest_bt_trades_csv()
    if not path:
        return []
    df = pd.read_csv(path)
    df = df[df['date'].astype(str).str.startswith(day)]
    return df.to_dict('records')


def bt_data_max_date() -> Optional[str]:
    """Latest date present in the BT trades CSV — staleness detector.
    The nightly orb-backtest regenerates it; if its max date < the day
    under check, BT-parity is UNKNOWN (skipped), not clean."""
    import pandas as pd
    path = latest_bt_trades_csv()
    if not path:
        return None
    try:
        return str(pd.read_csv(path, usecols=['date'])['date'].max())[:10]
    except Exception:
        return None


def load_live_rows(day: str, strategy: Optional[str] = None) -> List[Dict]:
    """All trade rows (any status — placements included) for a date."""
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM trades WHERE trade_date = ?"
    args = [day]
    if strategy:
        q += " AND strategy = ?"
        args.append(strategy)
    rows = [dict(r) for r in conn.execute(q, args)]
    conn.close()
    return rows


def journal_grep(pattern: str, day: str) -> List[str]:
    """Journald lines for onemil-trader on a UTC date (best effort —
    retention is ~2 days; empty result is not proof of absence)."""
    try:
        out = subprocess.run(
            ['journalctl', '-u', 'onemil-trader', '--since', f'{day} 00:00',
             '--until', f'{day} 23:59', '--no-pager'],
            capture_output=True, text=True, timeout=30).stdout
        return [ln for ln in out.splitlines() if pattern in ln]
    except Exception:
        return []


def read_selection_audit(day: str) -> List[Dict]:
    recs = []
    if SELECTION_AUDIT.exists():
        for line in SELECTION_AUDIT.read_text().splitlines():
            try:
                r = json.loads(line)
                if str(r.get('ts_utc', '')).startswith(day):
                    recs.append(r)
            except Exception:
                continue
    return recs


# ---------------------------------------------------------------------------
# The daily verdict
# ---------------------------------------------------------------------------

def green_verdict(day: str) -> Dict:
    """Compute the operational-green verdict for one trading day.

    Hard gates (any failure = RED day for the ramp streak):
      A. every closed trade's exit_reason is known + attributed
      B. no rows stuck in exit_pending_verification
      C. every BT-selected pick was at least ORDERED live, or its absence
         is explained by a live-only gate (spread, buying power, conflict)
    Informational (reported, not gating — journald retention makes absence
    unprovable): touchgo REKEY / negative-age tripwires.
    """
    reasons: List[str] = []
    checks: Dict[str, str] = {}

    rows = load_live_rows(day)
    closed = [r for r in rows if r.get('exit_price') is not None]
    bad_exit = [r['symbol'] for r in closed
                if not is_known(r.get('exit_reason'))
                or needs_reconcile(r.get('exit_reason'))]
    checks['exits'] = 'OK' if not bad_exit else f"unattributed: {bad_exit}"
    if bad_exit:
        reasons.append(f"unattributed exits: {bad_exit}")

    stuck = [r['symbol'] for r in rows
             if r.get('order_status') == 'exit_pending_verification']
    checks['pending_verification'] = 'OK' if not stuck else f"stuck: {stuck}"
    if stuck:
        reasons.append(f"exit_pending_verification: {stuck}")

    bt_max = bt_data_max_date()
    bt_stale = bt_max is None or bt_max < day
    bt_syms = set() if bt_stale else {r['symbol'] for r in load_bt_selected(day)}
    live_orb_syms = {r['symbol'] for r in rows if r.get('strategy') == 'orb'}
    missing = bt_syms - live_orb_syms
    unexplained = []
    if missing:
        audit = read_selection_audit(day)
        audited = set()
        for rec in audit:
            audited |= {e.get('sym') for e in rec.get('ranked', [])}
        explained_lines = journal_grep('skipped —', day) + \
            journal_grep('spread', day)
        for sym in sorted(missing):
            if any(sym in ln and any(k in ln for k in
                   ('spread', 'buying power', 'other strategy', 'halted'))
                   for ln in explained_lines):
                continue
            unexplained.append(sym)
    if bt_stale:
        checks['bt_parity'] = (f"SKIPPED — BT data stale (max={bt_max}); "
                               f"nightly orb-backtest hasn't covered {day} yet")
    elif unexplained:
        checks['bt_parity'] = f"BT picks never ordered: {unexplained}"
        reasons.append(f"BT picks never ordered live: {unexplained}")
    else:
        checks['bt_parity'] = 'OK'

    rekey = journal_grep('REKEY', day) + journal_grep('negative-age', day)
    checks['touchgo_tripwires'] = 'OK' if not rekey else f"{len(rekey)} tripwire line(s)"

    return {
        'day': day,
        'green': not reasons,
        'reasons': reasons,
        'checks': checks,
        'n_live_rows': len(rows),
        'n_bt_selected': len(bt_syms),
        'bt_stale': bt_stale,
    }


# ---------------------------------------------------------------------------
# Streak persistence — the number the ramp gate reads
# ---------------------------------------------------------------------------

def streak_update(day: str, green: bool, reasons: List[str],
                  path: Path = STREAK_PATH) -> int:
    """Record the day's verdict; return the current consecutive-green count.

    Streak counts consecutive green TRADING days (weekend/holiday gaps
    don't break it — only a recorded red does). Re-running the same day
    overwrites that day's record (idempotent).
    """
    records: List[Dict] = []
    if path.exists():
        try:
            records = json.loads(path.read_text()).get('days', [])
        except Exception:
            records = []
    records = [r for r in records if r.get('day') != day]
    records.append({'day': day, 'green': bool(green), 'reasons': reasons})
    records.sort(key=lambda r: r['day'])
    records = records[-60:]
    streak = 0
    for r in reversed(records):
        if r['green']:
            streak += 1
        else:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {'streak': streak, 'needed': GREEN_SESSIONS_NEEDED, 'days': records},
        indent=1))
    return streak


def read_streak(path: Path = STREAK_PATH) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# P&L helpers
# ---------------------------------------------------------------------------

def realized_pnl(day: str) -> Dict[str, float]:
    """Realized P&L per strategy for a date (closed rows only)."""
    out: Dict[str, float] = {}
    for r in load_live_rows(day):
        if r.get('exit_price') is None or r.get('entry_price') is None:
            continue
        pnl = (r['exit_price'] - r['entry_price']) * (r.get('shares') or 0)
        out[r.get('strategy') or '?'] = out.get(r.get('strategy') or '?', 0.0) + pnl
    return out


def cumulative_orb_since(start: str) -> float:
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    row = conn.execute(
        """SELECT COALESCE(SUM((exit_price-entry_price)*shares), 0)
           FROM trades WHERE strategy='orb' AND trade_date >= ?
             AND exit_price IS NOT NULL""", (start,)).fetchone()
    conn.close()
    return float(row[0])


def send_telegram(msg: str) -> bool:
    """Send unless running under pytest. Returns True on success."""
    if os.environ.get('PYTEST_CURRENT_TEST'):
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / '.env'))
        from notifications.telegram_notifier import TelegramNotifier
        n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'),
                             os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
        n.send_message_sync(msg, parse_mode='HTML')
        return True
    except Exception:
        return False


def prev_trading_day_utc(today: Optional[date] = None) -> str:
    """Most recent weekday (approximation; holiday-running is harmless —
    a no-data day yields an empty-but-green verdict that's overwritten)."""
    d = today or datetime.utcnow().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()
