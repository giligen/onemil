#!/usr/bin/env python3
"""BF P1 live ramp — eligibility checker (decision aid; changes no config).

Gates are in units of base risk (u = trading.risk_per_trade) so they hold at
every stage. Playbook: docs/bf_p1_ramp.md. Reads config.yaml (current stage
= risk_per_trade) and trades.db (strategy='bull_flag', closed trades since
the stage start), prints stage P&L, trades, sessions, streak, rail hits and
the ADVANCE / HOLD / DEMOTE / PAUSE verdict.

Gate-2 of docs/scaling_plan_2026.md adds two columns on top of the P&L rules:
  * above-water EX-MONSTER — stage P&L with the single best trade removed
    ("green weeks over monsters": one +8R trade is not a stage), and
  * BT band — realized stage R/trade inside the backtest's bootstrap
    [p5, p10, p90] band for this n (trading/ramp_bt_band.py).
A parity FREEZE (trading/ramp_freeze.py) stops the stage clock and blocks
ADVANCE regardless of P&L.

Usage:
  python scripts/bf_ramp_check.py                       # stage start from trading/ramp_stage.py
  python scripts/bf_ramp_check.py --stage-start 2026-10-01   # override the table
  python scripts/bf_ramp_check.py --verbose             # per-trade table
  python scripts/bf_ramp_check.py --clear-freeze bf "cache rebuilt, parity re-verified"
"""
import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'config.yaml'
TRADES_DB = ROOT / 'data' / 'trades.db'

from trading import ramp_bt_band as band_mod  # noqa: E402  (needs ROOT on sys.path)
from trading import ramp_pool  # noqa: E402  (the pooled ADVISORY line)
from trading import ramp_freeze  # noqa: E402
from trading import ramp_stage  # noqa: E402  (the stage-start table)

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
    # Gate-2 additions (docs/scaling_plan_2026.md items 3 and 4)
    pnl_ex_monster: float = 0.0
    best_trade_pnl: float = 0.0
    live_mean_r: Optional[float] = None
    band: Optional[band_mod.Band] = None
    band_status: str = band_mod.NO_DATA
    frozen: bool = False
    frozen_sessions: int = 0

    @property
    def pnl_u(self) -> float:
        return self.pnl / self.base if self.base else 0.0

    @property
    def pnl_ex_monster_u(self) -> float:
        return self.pnl_ex_monster / self.base if self.base else 0.0

    @property
    def above_water_ex_monster(self) -> bool:
        return self.pnl_ex_monster > 0


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


def compute_stats(trades: List[Dict], base: float, sessions: int,
                  bt_r: Optional[Sequence[float]] = None,
                  frozen: bool = False, frozen_sessions: int = 0) -> StageStats:
    """Pure: closed BF trades (dicts with trade_date, pnl, exit_pending_verification).

    bt_r: the backtest's per-trade R distribution (trading/ramp_bt_band.py).
          None/empty -> the BT-band gate reads NO-DATA, which blocks ADVANCE
          (an unscored gate is never a passed gate).
    frozen / frozen_sessions: Gate-1 freeze state; `sessions` must already
          EXCLUDE the frozen ones (the stage clock stops while frozen).
    """
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
    pnls = [float(t.get('pnl') or 0) for t in trades]
    best_trade = max(pnls) if pnls else 0.0
    # Live R per trade = pnl / the stage base risk — the same normalization
    # the BT reference uses (pnl / $2,000), so the two are comparable.
    live_r = [p / base for p in pnls] if base else []
    mean_r = sum(live_r) / len(live_r) if live_r else None
    bnd = band_mod.bootstrap_band(bt_r or [], len(live_r)) if live_r else None
    return StageStats(
        base=base, pnl=pnl, trades=len(trades), sessions=sessions, losing_streak=best,
        worst_day=worst_day, worst_week=worst_week,
        daily_rail_hits=sum(1 for v in by_day.values() if v <= daily_rail),
        weekly_rail_hit=any(v <= weekly_rail for v in by_week.values()),
        parity_flags=sum(1 for t in trades if t.get('exit_pending_verification')),
        pnl_ex_monster=pnl - best_trade,
        best_trade_pnl=best_trade,
        live_mean_r=mean_r,
        band=bnd,
        band_status=band_mod.classify(mean_r, bnd),
        frozen=frozen,
        frozen_sessions=frozen_sessions,
    )


def verdict(s: StageStats) -> str:
    """ADVANCE / HOLD / DEMOTE / PAUSE per docs/bf_p1_ramp.md + scaling_plan_2026 Gate 2."""
    if s.pnl_u <= PAUSE['pnl_u']:
        return 'PAUSE'
    below_p5_after_8 = (s.band_status == band_mod.BELOW_P5
                        and s.trades >= ADVANCE['min_trades'])
    if (s.pnl_u <= DEMOTE['pnl_u'] or s.losing_streak >= DEMOTE['streak']
            or s.weekly_rail_hit or below_p5_after_8):
        return 'DEMOTE'
    enough = (s.trades >= ADVANCE['min_trades']
              or (s.trades >= ADVANCE['early_trades'] and s.pnl_u >= ADVANCE['early_pnl_u']))
    if (s.pnl > 0 and enough and s.sessions >= ADVANCE['min_sessions']
            and s.parity_flags == 0 and s.daily_rail_hits == 0
            and s.above_water_ex_monster and s.band_status == band_mod.IN_BAND
            and not s.frozen):
        return 'ADVANCE'
    return 'HOLD'


def load_trades(since: str) -> List[Dict]:
    """Closed BF trades since `since` — READ-ONLY (a checker never writes)."""
    conn = sqlite3.connect(f"file:{TRADES_DB}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT trade_date, symbol, pnl, exit_reason, exited_at, "
        "CASE WHEN order_status='exit_pending_verification' THEN 1 ELSE 0 END "
        "AS exit_pending_verification "
        "FROM trades WHERE strategy='bull_flag' AND trade_date>=? AND pnl IS NOT NULL "
        "ORDER BY trade_date, exited_at", (since,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def session_dates(since: str, today: Optional[date] = None) -> List[str]:
    """Weekday sessions from `since` through today (holidays counted — conservative)."""
    d0, d1 = date.fromisoformat(since), today or date.today()
    return [date.fromordinal(d0.toordinal() + i).isoformat()
            for i in range((d1 - d0).days + 1)
            if date.fromordinal(d0.toordinal() + i).weekday() < 5]


def session_count(since: str) -> int:
    """Kept for callers that only want the raw weekday count."""
    return len(session_dates(since))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ramp_stage.add_stage_start_arg(ap, 'bf')
    ap.add_argument('--verbose', action='store_true')
    ramp_freeze.add_clear_freeze_arg(ap)
    a = ap.parse_args()
    if a.clear_freeze:
        print(ramp_freeze.handle_clear_freeze(a.clear_freeze))
        return 0
    stage_start, stage_reason = ramp_stage.resolve('bf', a.stage_start)
    cfg = yaml.safe_load(open(CONFIG))
    base = float(cfg['trading']['risk_per_trade'])
    cur = stage_for_risk(base)
    trades = load_trades(stage_start)
    all_sessions = session_dates(stage_start)
    fz = ramp_freeze.get('bf')
    live_sessions = ramp_freeze.unfrozen_sessions('bf', all_sessions)
    adv_gate = int(cfg.get('scanner', {}).get('min_daily_volume', 200_000))
    ref = band_mod.bf_reference(adv_gate)
    bt_r = band_mod.load_reference_r(ref, 'bf')
    s = compute_stats(trades, base, len(live_sessions), bt_r=bt_r,
                      frozen=fz.frozen,
                      frozen_sessions=len(all_sessions) - len(live_sessions))
    v = verdict(s)
    nxt = next_stage(cur)
    print(f"BF P1 ramp — stage {cur['name']} (base ${base:.0f}) since {stage_start}")
    print(ramp_stage.line('bf', stage_start, stage_reason))
    print(f"  stage P&L ${s.pnl:,.0f} = {s.pnl_u:+.2f}u | trades {s.trades} | sessions {s.sessions} | "
          f"losing streak {s.losing_streak} | worst day ${s.worst_day:,.0f} | worst week ${s.worst_week:,.0f}")
    print(f"  rails: daily hits {s.daily_rail_hits} | weekly hit {s.weekly_rail_hit} | parity flags {s.parity_flags}")
    print(f"  ex-monster: ${s.pnl_ex_monster:,.0f} = {s.pnl_ex_monster_u:+.2f}u "
          f"(best trade ${s.best_trade_pnl:,.0f} removed) → "
          f"{'ABOVE WATER' if s.above_water_ex_monster else 'NOT above water'} "
          f"(advance requires > 0)")
    print(band_mod.band_line(s.band_status, s.live_mean_r, s.band, ref))
    print(band_mod.bf_basis_comparison_line(ref, s.band.n if s.band else s.trades,
                                            s.live_mean_r))
    print(ramp_pool.advisory_line())
    if s.frozen:
        print(f"  {fz.line()}")
    if s.frozen_sessions:
        print(f"  stage clock: {s.frozen_sessions} frozen session(s) excluded "
              f"({len(all_sessions)} weekdays → {s.sessions} counted)")
    print(f"  VERDICT: {v}" + (f" → {nxt['name']} (${nxt['risk']}) — set risk_per_trade + rails + daily_loss_limit together" if v == 'ADVANCE' and nxt else ''))
    if v == 'HOLD':
        need = []
        if s.frozen: need.append('PARITY FREEZE cleared (manual)')
        if s.pnl <= 0: need.append('stage P&L > 0')
        if not s.above_water_ex_monster: need.append('stage P&L ex-monster > 0')
        if s.band_status != band_mod.IN_BAND:
            need.append(f"BT band IN-BAND (now {s.band_status})")
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
