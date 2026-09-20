#!/usr/bin/env python3
"""ORB budget ramp — eligibility checker on POSITIVE realized P&L (owner
adopted 2026-09-07; docs/orb_p1_style_ramp_proposal.md replaces the
cushion ladder of docs/orb_rollout_plan.md). Decision aid — changes nothing.

Stage = orb.yaml sizing.account_budget_usd. Reads trades.db (strategy='orb',
closed fills since the stage start), logs/green_streak.json (fill-parity
reasons per session) and the fills' own entry slippage vs the 30 bps model.

Gate-2 of docs/scaling_plan_2026.md adds two columns on top of the P&L rules:
  * above-water EX-MONSTER — stage P&L with the single best fill removed, and
  * BT band — realized stage R/trade inside the backtest's bootstrap
    [p5, p10, p90] band for this n (trading/ramp_bt_band.py; the reference
    book follows orb.yaml's catalyst-veto state and is named in the output).
A parity FREEZE (trading/ramp_freeze.py, set by daily_green_check.py) stops
the stage clock and blocks ADVANCE regardless of P&L.

Usage:
  python scripts/orb_ramp_check.py                        # stage start from trading/ramp_stage.py
  python scripts/orb_ramp_check.py --stage-start 2026-10-01 [--verbose]   # override
  python scripts/orb_ramp_check.py --clear-freeze orb "mult drift explained + fixed"
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ORB_YAML = ROOT / 'orb.yaml'
TRADES_DB = ROOT / 'data' / 'trades.db'
GREEN = ROOT / 'logs' / 'green_streak.json'

from trading import ramp_bt_band as band_mod  # noqa: E402  (needs ROOT on sys.path)
from trading import ramp_pool  # noqa: E402  (the pooled ADVISORY line)
from trading import ramp_freeze  # noqa: E402
from trading import ramp_stage  # noqa: E402  (the stage-start table)

# KEEP IN SYNC with docs/orb_p1_style_ramp_proposal.md (budget, slots=3)
STAGES = [
    {'name': 'S0', 'budget': 10000, 'daily_limit': -750, 'month_pause': None},
    {'name': 'S1', 'budget': 30000, 'daily_limit': -1500, 'month_pause': -3000},
    {'name': 'S2', 'budget': 60000, 'daily_limit': -3000, 'month_pause': -6000},
    {'name': 'S3', 'budget': 100000, 'daily_limit': -5000, 'month_pause': -10000},
]
ADVANCE = {'min_fills': 8, 'min_sessions': 15, 'limit_free_sessions': 10,
           'entry_slip_model_bps': 30.0, 'slip_tolerance_bps': 10.0}
DEMOTE = {'pnl_pct_of_budget': -6.0, 'streak': 5, 'limit_hits': 2, 'slip_2x_fills': 3}
PAUSE = {'pnl_pct_of_budget': -8.0}
PARITY_MARKERS = ('BT picks never ordered live', 'fill-parity')


@dataclass
class StageStats:
    budget: float
    pnl: float
    fills: int
    sessions: int
    losing_streak: int
    limit_hits: int
    limit_hit_last10: bool
    parity_defects: int
    mean_entry_slip_bps: Optional[float]
    slip_2x_fills: int
    # Gate-2 additions (docs/scaling_plan_2026.md items 3 and 4)
    pnl_ex_monster: float = 0.0
    best_fill_pnl: float = 0.0
    live_mean_r: Optional[float] = None
    band: Optional[band_mod.Band] = None
    band_status: str = band_mod.NO_DATA
    frozen: bool = False
    frozen_sessions: int = 0

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.budget * 100 if self.budget else 0.0

    @property
    def pnl_ex_monster_pct(self) -> float:
        return self.pnl_ex_monster / self.budget * 100 if self.budget else 0.0

    @property
    def above_water_ex_monster(self) -> bool:
        return self.pnl_ex_monster > 0


def stage_for_budget(budget: float) -> Dict:
    for s in STAGES:
        if abs(s['budget'] - budget) < 1e-6:
            return s
    return {'name': f'custom(${budget:,.0f})', 'budget': budget, 'daily_limit': None, 'month_pause': None}


def next_stage(cur: Dict) -> Optional[Dict]:
    names = [s['name'] for s in STAGES]
    if cur['name'] in names and names.index(cur['name']) + 1 < len(names):
        return STAGES[names.index(cur['name']) + 1]
    return None


def compute_stats(fills: List[Dict], budget: float, daily_limit: Optional[float],
                  sessions: int, session_dates: List[str], parity_reasons: Dict[str, List[str]],
                  bt_r: Optional[Sequence[float]] = None,
                  frozen: bool = False, frozen_sessions: int = 0) -> StageStats:
    """Pure. fills: dicts with trade_date, pnl, entry_price (trigger), fill_price,
    total_risk (shares x risk_per_share — the live 1R).
    parity_reasons: {day: [reason strings]} from green_streak.json.

    bt_r: the backtest's per-trade R distribution (trading/ramp_bt_band.py).
          None/empty -> the BT-band gate reads NO-DATA, which blocks ADVANCE.
    frozen / frozen_sessions: Gate-1 freeze state; `sessions` must already
          EXCLUDE the frozen ones (the stage clock stops while frozen)."""
    pnl = sum(float(f.get('pnl') or 0) for f in fills)
    by_day: Dict[str, float] = {}
    streak = best = 0
    slips = []
    pnls: List[float] = []
    live_r: List[float] = []
    for f in sorted(fills, key=lambda x: (x['trade_date'], x.get('exited_at') or '')):
        p = float(f.get('pnl') or 0)
        pnls.append(p)
        # Live 1R = the planned dollar risk of the fill (entry - stop) x shares,
        # the same normalization as the BT's pnl_pct / range_size_pct.
        try:
            risk = float(f.get('total_risk') or 0)
            if risk > 0:
                live_r.append(p / risk)
        except (TypeError, ValueError):
            pass
        by_day[f['trade_date']] = by_day.get(f['trade_date'], 0.0) + p
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
        try:
            trig, fp = float(f['entry_price']), float(f['fill_price'])
            if trig > 0 and fp > 0:
                slips.append((fp - trig) / trig * 1e4)
        except (TypeError, ValueError, KeyError):
            pass
    hits = [d for d, v in by_day.items() if daily_limit is not None and v <= daily_limit]
    last10 = set(session_dates[-ADVANCE['limit_free_sessions']:])
    parity = sum(1 for d, rs in parity_reasons.items()
                 if d in set(session_dates) and any(m in r for r in rs for m in PARITY_MARKERS))
    mean_slip = sum(slips) / len(slips) if slips else None
    best_fill = max(pnls) if pnls else 0.0
    mean_r = sum(live_r) / len(live_r) if live_r else None
    bnd = band_mod.bootstrap_band(bt_r or [], len(live_r)) if live_r else None
    return StageStats(
        budget=budget, pnl=pnl, fills=len(fills), sessions=sessions, losing_streak=best,
        limit_hits=len(hits), limit_hit_last10=any(d in last10 for d in hits),
        parity_defects=parity, mean_entry_slip_bps=mean_slip,
        slip_2x_fills=sum(1 for s in slips if s > 2 * ADVANCE['entry_slip_model_bps']),
        pnl_ex_monster=pnl - best_fill, best_fill_pnl=best_fill,
        live_mean_r=mean_r, band=bnd,
        band_status=band_mod.classify(mean_r, bnd),
        frozen=frozen, frozen_sessions=frozen_sessions,
    )


def verdict(s: StageStats) -> str:
    """HOLD / ADVANCE / DEMOTE / PAUSE per the ramp + scaling_plan_2026 Gate 2."""
    if s.pnl_pct <= PAUSE['pnl_pct_of_budget']:
        return 'PAUSE'
    below_p5_after_8 = (s.band_status == band_mod.BELOW_P5
                        and s.fills >= ADVANCE['min_fills'])
    if (s.pnl_pct <= DEMOTE['pnl_pct_of_budget'] or s.losing_streak >= DEMOTE['streak']
            or s.limit_hits >= DEMOTE['limit_hits'] or s.slip_2x_fills >= DEMOTE['slip_2x_fills']
            or below_p5_after_8):
        return 'DEMOTE'
    slip_ok = (s.mean_entry_slip_bps is not None
               and s.mean_entry_slip_bps <= ADVANCE['entry_slip_model_bps'] + ADVANCE['slip_tolerance_bps'])
    if (s.pnl > 0 and s.fills >= ADVANCE['min_fills'] and s.sessions >= ADVANCE['min_sessions']
            and s.parity_defects == 0 and not s.limit_hit_last10 and slip_ok
            and s.above_water_ex_monster and s.band_status == band_mod.IN_BAND
            and not s.frozen):
        return 'ADVANCE'
    return 'HOLD'


def load_fills(since: str) -> List[Dict]:
    """Closed ORB fills since `since` — READ-ONLY (a checker never writes)."""
    conn = sqlite3.connect(f"file:{TRADES_DB}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT trade_date, symbol, pnl, entry_price, fill_price, total_risk, "
        "exited_at, exit_reason "
        "FROM trades WHERE strategy='orb' AND trade_date>=? AND pnl IS NOT NULL "
        "ORDER BY trade_date, exited_at", (since,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_parity(since: str) -> Dict[str, List[str]]:
    try:
        days = json.load(open(GREEN)).get('days', [])
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: {GREEN} unreadable ({e}) — parity treated as UNKNOWN (blocks ADVANCE)")
        return {'__unknown__': ['fill-parity: unknown']}
    return {d['day']: list(d.get('reasons') or []) for d in days if d.get('day', '') >= since}


def sessions_since(since: str) -> List[str]:
    d0, d1 = date.fromisoformat(since), date.today()
    out = []
    for i in range((d1 - d0).days + 1):
        d = date.fromordinal(d0.toordinal() + i)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ramp_stage.add_stage_start_arg(ap, 'orb')
    ap.add_argument('--verbose', action='store_true')
    ramp_freeze.add_clear_freeze_arg(ap)
    a = ap.parse_args()
    if a.clear_freeze:
        print(ramp_freeze.handle_clear_freeze(a.clear_freeze))
        return 0
    stage_start, stage_reason = ramp_stage.resolve('orb', a.stage_start)
    cfg = yaml.safe_load(open(ORB_YAML))
    budget = float(cfg['sizing']['account_budget_usd'])
    cur = stage_for_budget(budget)
    daily_limit = float(cfg.get('risk', {}).get('daily_loss_limit_usd', cur['daily_limit'] or 0) or 0) or None
    fills = load_fills(stage_start)
    sess = sessions_since(stage_start)
    parity = load_parity(stage_start)
    if '__unknown__' in parity:
        parity = {d: parity['__unknown__'] for d in sess}
    fz = ramp_freeze.get('orb')
    live_sess = ramp_freeze.unfrozen_sessions('orb', sess)
    veto_on = bool(cfg.get('filter', {}).get('catalyst_veto', {}).get('enabled'))
    ref = band_mod.orb_reference(veto_on)
    bt_r = band_mod.load_reference_r(ref, 'orb')
    s = compute_stats(fills, budget, daily_limit, len(live_sess), sess, parity,
                      bt_r=bt_r, frozen=fz.frozen,
                      frozen_sessions=len(sess) - len(live_sess))
    v = verdict(s); nxt = next_stage(cur)
    slip = f"{s.mean_entry_slip_bps:.0f} bps" if s.mean_entry_slip_bps is not None else 'n/a (no fills)'
    print(f"ORB ramp — stage {cur['name']} (budget ${budget:,.0f}) since {stage_start}")
    print(ramp_stage.line('orb', stage_start, stage_reason))
    print(f"  stage P&L ${s.pnl:,.0f} = {s.pnl_pct:+.2f}% of budget | fills {s.fills} | sessions {s.sessions} | "
          f"losing streak {s.losing_streak} | entry slip mean {slip} (model {ADVANCE['entry_slip_model_bps']:.0f}) | "
          f">2x-model fills {s.slip_2x_fills}")
    print(f"  daily-limit hits {s.limit_hits} (last 10 sessions: {s.limit_hit_last10}) | parity defects {s.parity_defects}")
    print(f"  ex-monster: ${s.pnl_ex_monster:,.0f} = {s.pnl_ex_monster_pct:+.2f}% of budget "
          f"(best fill ${s.best_fill_pnl:,.0f} removed) → "
          f"{'ABOVE WATER' if s.above_water_ex_monster else 'NOT above water'} "
          f"(advance requires > 0)")
    print(band_mod.band_line(s.band_status, s.live_mean_r, s.band, ref))
    print(ramp_pool.advisory_line())
    if s.frozen:
        print(f"  {fz.line()}")
    if s.frozen_sessions:
        print(f"  stage clock: {s.frozen_sessions} frozen session(s) excluded "
              f"({len(sess)} weekdays → {s.sessions} counted)")
    print(f"  VERDICT: {v}" + (f" → {nxt['name']} (${nxt['budget']:,.0f}, daily limit {nxt['daily_limit']}) — "
                              f"set sizing.account_budget_usd + risk.daily_loss_limit_usd together" if v == 'ADVANCE' and nxt else ''))
    if v == 'HOLD':
        need = []
        if s.frozen: need.append('PARITY FREEZE cleared (manual)')
        if s.pnl <= 0: need.append('stage P&L > 0')
        if not s.above_water_ex_monster: need.append('stage P&L ex-monster > 0')
        if s.band_status != band_mod.IN_BAND:
            need.append(f"BT band IN-BAND (now {s.band_status})")
        if s.fills < ADVANCE['min_fills']: need.append(f"fills {s.fills}/{ADVANCE['min_fills']}")
        if s.sessions < ADVANCE['min_sessions']: need.append(f"sessions {s.sessions}/{ADVANCE['min_sessions']}")
        if s.parity_defects: need.append(f"parity defects {s.parity_defects} → 0")
        if s.limit_hit_last10: need.append('no daily-limit hit in last 10 sessions')
        if s.mean_entry_slip_bps is None or s.mean_entry_slip_bps > ADVANCE['entry_slip_model_bps'] + ADVANCE['slip_tolerance_bps']:
            need.append(f"entry slip ≤ {ADVANCE['entry_slip_model_bps'] + ADVANCE['slip_tolerance_bps']:.0f} bps")
        print('  holding on: ' + '; '.join(need))
    if a.verbose:
        for f in fills:
            print(f"  {f['trade_date']} {f['symbol']:6s} {float(f['pnl'] or 0):8.0f} {f.get('exit_reason') or ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
