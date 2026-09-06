#!/usr/bin/env python3
"""BF P1 live ramp — eligibility checker (decision aid; changes no config).

Gates are in units of base risk (u = trading.risk_per_trade) so they hold at
every stage. Playbook: docs/bf_p1_ramp.md. Reads config.yaml (current stage
= risk_per_trade) and trades.db (strategy='bull_flag', closed trades since
the stage start), prints stage P&L, trades, sessions, streak, rail hits and
the ADVANCE / HOLD / DEMOTE / PAUSE verdict.

Usage:
  python scripts/bf_ramp_check.py                       # stage started at LAUNCH (2026-09-07)
  python scripts/bf_ramp_check.py --stage-start 2026-10-01
  python scripts/bf_ramp_check.py --verbose             # per-trade table
"""
import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'config.yaml'
TRADES_DB = ROOT / 'data' / 'trades.db'
LAUNCH = '2026-09-07'

# KEEP IN SYNC with docs/bf_p1_ramp.md
STAGES = [
    {'name': 'L0', 'risk': 150},
    {'name': 'L1', 'risk': 400},
    {'name': 'L2', 'risk': 1000},
    {'name': 'L3', 'risk': 2000},
]
RAILS_U = {'daily': -5.0, 'weekly': -7.0, 'month_pause': -8.0}
ADVANCE = {'min_trades': 8, 'early_trades': 6, 'early_pnl_u': 4.0, 'min_sessions': 15, 'rail_free_sessions': 10}
DEMOTE = {'pnl_u': -6.0, 'streak': 5}
PAUSE = {'pnl_u': -8.0}


@dataclass
class StageStats:
    base: float
    pnl: float
    trades: int
    sessions: int
    losing_streak: int
    worst_day: float
    worst_week: float
    daily_rail_hits: int
    weekly_rail_hit: bool
    parity_flags: int

    @property
    def pnl_u(self) -> float:
        return self.pnl / self.base if self.base else 0.0


def stage_for_risk(risk: float) -> Dict:
    """Current stage = the ladder row whose base risk matches risk_per_trade."""
    for s in STAGES:
        if abs(s['risk'] - risk) < 1e-6:
            return s
    return {'name': f'custom(${risk:.0f})', 'risk': risk}


def next_stage(cur: Dict) -> Optional[Dict]:
    names = [s['name'] for s in STAGES]
    if cur['name'] in names and names.index(cur['name']) + 1 < len(STAGES):
        return STAGES[names.index(cur['name']) + 1]
    return None


def compute_stats(trades: List[Dict], base: float, sessions: int) -> StageStats:
    """Pure: closed BF trades (dicts with trade_date, pnl, exit_pending_verification)."""
    pnl = sum(float(t.get('pnl') or 0) for t in trades)
    by_day: Dict[str, float] = {}
    by_week: Dict[str, float] = {}
    streak = best = 0
    for t in sorted(trades, key=lambda x: (x['trade_date'], x.get('exited_at') or '')):
        p = float(t.get('pnl') or 0)
        d = str(t['trade_date'])[:10]
        by_day[d] = by_day.get(d, 0.0) + p
        wk = date.fromisoformat(d).isocalendar()[:2]
        by_week[wk] = by_week.get(wk, 0.0) + p
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
    worst_day = min(by_day.values()) if by_day else 0.0
    worst_week = min(by_week.values()) if by_week else 0.0
    daily_rail = RAILS_U['daily'] * base
    weekly_rail = RAILS_U['weekly'] * base
    return StageStats(
        base=base, pnl=pnl, trades=len(trades), sessions=sessions, losing_streak=best,
        worst_day=worst_day, worst_week=worst_week,
        daily_rail_hits=sum(1 for v in by_day.values() if v <= daily_rail),
        weekly_rail_hit=any(v <= weekly_rail for v in by_week.values()),
        parity_flags=sum(1 for t in trades if t.get('exit_pending_verification')),
    )


def verdict(s: StageStats) -> str:
    """ADVANCE / HOLD / DEMOTE / PAUSE per docs/bf_p1_ramp.md."""
    if s.pnl_u <= PAUSE['pnl_u']:
        return 'PAUSE'
    if s.pnl_u <= DEMOTE['pnl_u'] or s.losing_streak >= DEMOTE['streak'] or s.weekly_rail_hit:
        return 'DEMOTE'
    enough = (s.trades >= ADVANCE['min_trades']
              or (s.trades >= ADVANCE['early_trades'] and s.pnl_u >= ADVANCE['early_pnl_u']))
    if (s.pnl > 0 and enough and s.sessions >= ADVANCE['min_sessions']
            and s.parity_flags == 0 and s.daily_rail_hits == 0):
        return 'ADVANCE'
    return 'HOLD'


def load_trades(since: str) -> List[Dict]:
    conn = sqlite3.connect(str(TRADES_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT trade_date, symbol, pnl, exit_reason, exited_at, "
        "CASE WHEN order_status='exit_pending_verification' THEN 1 ELSE 0 END "
        "AS exit_pending_verification "
        "FROM trades WHERE strategy='bull_flag' AND trade_date>=? AND pnl IS NOT NULL "
        "ORDER BY trade_date, exited_at", (since,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def session_count(since: str) -> int:
    """Weekday sessions from `since` through today (holidays counted — conservative)."""
    d0, d1 = date.fromisoformat(since), date.today()
    return sum(1 for i in range((d1 - d0).days + 1)
               if (d0.fromordinal(d0.toordinal() + i)).weekday() < 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage-start', default=LAUNCH)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    base = float(cfg['trading']['risk_per_trade'])
    cur = stage_for_risk(base)
    trades = load_trades(a.stage_start)
    s = compute_stats(trades, base, session_count(a.stage_start))
    v = verdict(s)
    nxt = next_stage(cur)
    print(f"BF P1 ramp — stage {cur['name']} (base ${base:.0f}) since {a.stage_start}")
    print(f"  stage P&L ${s.pnl:,.0f} = {s.pnl_u:+.2f}u | trades {s.trades} | sessions {s.sessions} | "
          f"losing streak {s.losing_streak} | worst day ${s.worst_day:,.0f} | worst week ${s.worst_week:,.0f}")
    print(f"  rails: daily hits {s.daily_rail_hits} | weekly hit {s.weekly_rail_hit} | parity flags {s.parity_flags}")
    print(f"  VERDICT: {v}" + (f" → {nxt['name']} (${nxt['risk']}) — set risk_per_trade + rails + daily_loss_limit together" if v == 'ADVANCE' and nxt else ''))
    if v == 'HOLD':
        need = []
        if s.pnl <= 0: need.append('stage P&L > 0')
        if s.trades < ADVANCE['min_trades']: need.append(f"trades {s.trades}/{ADVANCE['min_trades']} (or 6 with ≥ +4u)")
        if s.sessions < ADVANCE['min_sessions']: need.append(f"sessions {s.sessions}/{ADVANCE['min_sessions']}")
        if s.parity_flags: need.append('parity flags = 0')
        if s.daily_rail_hits: need.append('no daily rail hit')
        print('  holding on: ' + '; '.join(need))
    if a.verbose:
        for t in trades:
            print(f"  {t['trade_date']} {t['symbol']:6s} {float(t['pnl'] or 0):8.0f} {t.get('exit_reason') or ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
