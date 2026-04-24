#!/usr/bin/env python3
"""ORB live ramp eligibility checker.

Reads the current orb.yaml config + trades DB + git log, then prints:
  - Current stage (inferred from account_budget_usd)
  - Cushion: cumulative realized P&L since live-ramp start
  - Days in current stage (from last 'ORB ramp:' commit on orb.yaml)
  - Whether you're eligible to ADVANCE to the next stage
  - Whether any DEMOTION trigger is active

This tool is a DECISION AID — it does not change any config itself.
See `docs/orb_rollout_plan.md` for the full playbook.

Usage:
    python3 scripts/orb_ramp_check.py           # summary
    python3 scripts/orb_ramp_check.py --verbose # show per-trade history
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
ORB_YAML = ROOT / 'orb.yaml'
TRADES_DB = ROOT / 'data' / 'trades.db'

# The ramp stages — KEEP IN SYNC with docs/orb_rollout_plan.md
STAGES = [
    # (stage_idx, account_budget_usd, risk_per_trade_usd, daily_loss_limit_usd,
    #  cushion_to_advance, min_days_in_stage)
    (0,  30000, 1000, -1500,   5000, 10),
    (1,  50000, 1500, -2500,  10000, 10),
    (2,  80000, 2400, -4000,  18000, 15),
    (3, 120000, 3600, -6000,  30000, 20),
    (4, 174000, 5200, -8800,   None, None),  # terminal
]

HARD_STOP_PCT_OF_CASH = -0.15     # halt if realized ≤ -15% of starting cash
DEMOTE_DD_PCT_OF_PEAK = -0.20     # demote if realized drops 20% from peak
DEMOTE_CONSEC_RED_DAYS = 3        # demote on 3 consecutive red days


@dataclass
class Stage:
    idx: int
    account_budget_usd: int
    risk_per_trade_usd: int
    daily_loss_limit_usd: int
    cushion_to_advance: Optional[int]
    min_days_in_stage: Optional[int]


def _stage_by_idx(i: int) -> Stage:
    return Stage(*STAGES[i])


def _current_stage_from_yaml() -> Optional[Stage]:
    """Infer the active stage by EXACT match on account_budget_usd in orb.yaml.
    Returns None if the config doesn't match any ramp stage — e.g. the user
    is still on the $100K BT-validated baseline and hasn't started the ramp."""
    with open(ORB_YAML) as f:
        cfg = yaml.safe_load(f)
    budget = int(cfg['sizing']['account_budget_usd'])
    for row in STAGES:
        if row[1] == budget:
            return Stage(*row)
    return None


def _ramp_start_date() -> Optional[date]:
    """Best-effort: first date with an ORB trade in the DB.

    Fallback: sentinel file `data/orb_ramp_start.txt` if you want to
    override (useful when switching from paper to live, to reset the
    cushion math)."""
    override = ROOT / 'data' / 'orb_ramp_start.txt'
    if override.exists():
        try:
            return date.fromisoformat(override.read_text().strip())
        except Exception:
            pass
    if not TRADES_DB.exists():
        return None
    conn = sqlite3.connect(TRADES_DB)
    try:
        row = conn.execute(
            "SELECT MIN(trade_date) FROM trades WHERE strategy='orb'"
        ).fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None
    finally:
        conn.close()


def _last_ramp_commit_date() -> Optional[date]:
    """Date of the most recent commit touching orb.yaml with 'ORB ramp:' prefix.
    Used to compute days-in-current-stage."""
    try:
        out = subprocess.check_output(
            ['git', 'log', '--format=%ci', '--grep=ORB ramp:', '-n', '1',
             '--', str(ORB_YAML)],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            return None
        # '%ci' = "2026-04-24 15:30:00 +0000" — take the date portion
        return date.fromisoformat(out.split()[0])
    except Exception:
        return None


def _realized_pnl_series(since: date):
    """Return list of (trade_date, pnl) for closed ORB trades since `since`."""
    if not TRADES_DB.exists():
        return []
    conn = sqlite3.connect(TRADES_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT trade_date, pnl FROM trades "
            "WHERE strategy='orb' AND trade_date >= ? "
            "AND exit_price IS NOT NULL AND pnl IS NOT NULL "
            "ORDER BY trade_date, id",
            (since.isoformat(),),
        )
        return [(date.fromisoformat(r['trade_date']), float(r['pnl']))
                for r in cur.fetchall()]
    finally:
        conn.close()


def _trading_days_between(a: date, b: date) -> int:
    """Rough count of weekdays (Mon-Fri) between a and b inclusive. Not
    holiday-aware — close enough for the min-days-in-stage gate."""
    if a > b:
        return 0
    n = 0
    d = a
    while d <= b:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def _consecutive_red_days(by_day_pnl: list) -> int:
    """Longest trailing run of red (pnl < 0) days from the end of the list.
    by_day_pnl: list of (date, pnl_sum_for_day) sorted chronologically."""
    n = 0
    for _, pnl in reversed(by_day_pnl):
        if pnl < 0:
            n += 1
        else:
            break
    return n


def main():
    p = argparse.ArgumentParser(description="ORB live ramp eligibility checker")
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print per-trade and per-day P&L detail')
    p.add_argument('--starting-cash', type=float, default=80000.0,
                   help='Your real account starting cash (default: $80K)')
    args = p.parse_args()

    # --- Current state ---
    stage = _current_stage_from_yaml()
    ramp_start = _ramp_start_date()
    stage_change_date = _last_ramp_commit_date() or ramp_start
    today = date.today()

    # Not on a ramp stage — user is still on the BT-validated baseline
    # (or a custom budget). Print what to do and bail.
    if stage is None:
        with open(ORB_YAML) as f:
            cfg = yaml.safe_load(f)
        budget = int(cfg['sizing']['account_budget_usd'])
        print("=" * 70)
        print(f"ORB live ramp check — {today.isoformat()}")
        print("=" * 70)
        print(f"NOT ON RAMP — orb.yaml account_budget_usd=${budget:,} doesn't "
              f"match any\n  stage in docs/orb_rollout_plan.md.")
        print()
        print("If you are ready to go LIVE, edit orb.yaml to Stage 0 values:")
        s0 = _stage_by_idx(0)
        print(f"  sizing.account_budget_usd:   {s0.account_budget_usd}")
        print(f"  sizing.risk_per_trade_usd:   {s0.risk_per_trade_usd}")
        print(f"  risk.daily_loss_limit_usd:   {s0.daily_loss_limit_usd}")
        print(f"Commit with message prefix 'ORB ramp: Stage 0 (live launch)' so")
        print(f"the check script can track days-in-stage from here on.")
        print()
        print("If you're deliberately on the $100K BT-validated config for")
        print("research, that's fine — this script only applies once you've")
        print("entered the ramp.")
        return 0

    if ramp_start is None:
        print("No ORB trades in DB yet — nothing to check. Ship stage 0 + run.")
        return 0

    trades = _realized_pnl_series(ramp_start)
    cushion = sum(pnl for _, pnl in trades)

    # Aggregate to per-day for consecutive-red and peak tracking
    by_day: dict = {}
    for d, pnl in trades:
        by_day[d] = by_day.get(d, 0.0) + pnl
    day_series = sorted(by_day.items())

    # Peak realized + current drawdown
    running = 0.0
    peak = 0.0
    for d, pnl in day_series:
        running += pnl
        peak = max(peak, running)
    drawdown_from_peak = cushion - peak  # ≤ 0

    consec_red = _consecutive_red_days(day_series)
    days_in_stage = _trading_days_between(stage_change_date or ramp_start, today)
    total_trading_days = _trading_days_between(ramp_start, today)

    # --- Advancement ---
    next_stage = (_stage_by_idx(stage.idx + 1)
                  if stage.idx + 1 < len(STAGES) else None)
    advance_blockers = []
    if next_stage is not None and stage.cushion_to_advance is not None:
        if cushion < stage.cushion_to_advance:
            advance_blockers.append(
                f"cushion ${cushion:,.0f} < required ${stage.cushion_to_advance:,}"
            )
        if stage.min_days_in_stage and days_in_stage < stage.min_days_in_stage:
            advance_blockers.append(
                f"{days_in_stage} trading day(s) in stage < required "
                f"{stage.min_days_in_stage}"
            )
        if peak > 0 and drawdown_from_peak / peak <= -0.08:
            advance_blockers.append(
                f"current DD {drawdown_from_peak / peak * 100:.1f}% "
                f"from peak > 8% health-check ceiling"
            )

    # --- Demotion triggers ---
    demote_triggers = []
    if peak > 0 and (drawdown_from_peak / peak) <= DEMOTE_DD_PCT_OF_PEAK:
        demote_triggers.append(
            f"realized P&L down {drawdown_from_peak / peak * 100:.1f}% "
            f"from peak ${peak:,.0f} (threshold: {DEMOTE_DD_PCT_OF_PEAK*100:.0f}%)"
        )
    if consec_red >= DEMOTE_CONSEC_RED_DAYS:
        demote_triggers.append(
            f"{consec_red} consecutive red days "
            f"(threshold: {DEMOTE_CONSEC_RED_DAYS})"
        )

    # --- Hard stop ---
    hard_stop = cushion <= args.starting_cash * HARD_STOP_PCT_OF_CASH
    hard_stop_threshold = args.starting_cash * HARD_STOP_PCT_OF_CASH

    # --- Output ---
    print("=" * 70)
    print(f"ORB live ramp check — {today.isoformat()}")
    print("=" * 70)
    print(f"Ramp started:           {ramp_start} ({total_trading_days} trading days)")
    print(f"Current stage:          {stage.idx} "
          f"(budget=${stage.account_budget_usd:,}, risk=${stage.risk_per_trade_usd:,}/trade)")
    print(f"Stage entered:          {stage_change_date} "
          f"({days_in_stage} trading days ago)")
    print(f"Realized P&L (cushion): ${cushion:+,.2f}")
    print(f"Peak realized:          ${peak:+,.2f}")
    print(f"Current DD from peak:   ${drawdown_from_peak:+,.2f}  "
          f"({drawdown_from_peak / peak * 100 if peak else 0:+.1f}%)")
    print(f"Consecutive red days:   {consec_red}")
    print()

    # Hard stop
    if hard_stop:
        print("🛑 HARD STOP TRIGGERED")
        print(f"   Realized ${cushion:+,.2f} ≤ ${hard_stop_threshold:+,.2f} "
              f"({HARD_STOP_PCT_OF_CASH*100:.0f}% of ${args.starting_cash:,.0f} cash)")
        print(f"   STOP TRADING. See docs/orb_rollout_plan.md § Hard stop.")
        print()

    # Demotion
    if demote_triggers:
        print(f"⚠️  DEMOTION TRIGGERED — step down to stage {max(0, stage.idx - 1)}:")
        for t in demote_triggers:
            print(f"   - {t}")
        print("   Apply via docs/orb_rollout_plan.md § Demotion procedure.")
        print()

    # Advancement
    if next_stage is None:
        print(f"✓ At terminal stage {stage.idx} — no further advancement.")
    elif not advance_blockers:
        print(f"✅ ELIGIBLE TO ADVANCE → stage {next_stage.idx}")
        print(f"   budget ${stage.account_budget_usd:,} → ${next_stage.account_budget_usd:,}")
        print(f"   risk   ${stage.risk_per_trade_usd:,}/trade → ${next_stage.risk_per_trade_usd:,}/trade")
        print(f"   Apply via docs/orb_rollout_plan.md § How to apply a stage change")
    else:
        print(f"⏳ Not yet eligible for stage {next_stage.idx}. Blockers:")
        for b in advance_blockers:
            print(f"   - {b}")

    print()
    if args.verbose and trades:
        print("Per-day P&L:")
        print(f"  {'date':12s}  {'day P&L':>12s}  {'cumulative':>12s}")
        cum = 0.0
        for d, pnl in day_series:
            cum += pnl
            print(f"  {d.isoformat()}  {pnl:>+12.2f}  {cum:>+12.2f}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
