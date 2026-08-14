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
# Stage -1 ("Pre-0") is the half-size live data-collection phase that
# precedes the formal ramp. It uses scripts/orb_pre0_daily.py for its
# own monitoring; the standard ramp script recognizes it but applies
# different gates.
# 2026-07-06 policy revision (owner-approved; see
# docs/ramp_policy_proposal_jul2026.md + docs/orb_rollout_plan.md):
# cushion_to_advance is RETIRED. Advancement = operational-green sessions
# + LOSS FLOOR (stage P&L >= -1 x weekly loss budget) + min days +
# slippage parity. Demotion = operational failure OR stage P&L <
# -2 x weekly loss budget. BT-consistent drawdown is NOT a demotion
# trigger (codifies the two June-2026 overrides).
STAGES = [
    # (stage_idx, account_budget_usd, risk_per_trade_usd, daily_loss_limit_usd,
    #  min_days_in_stage)
    (-1, 15000,  500, -750,  10),  # Pre-0 — see orb_pre0_daily.py for richer gating
    (0,  30000, 1000, -1500, 10),
    (1,  50000, 1500, -2500, 10),
    (2,  80000, 2400, -4000, 15),
    (3, 120000, 3600, -6000, 20),
    (4, 174000, 5200, -8800, None),  # terminal
]

HARD_STOP_PCT_OF_CASH = -0.15      # halt if realized ≤ -15% of starting cash
LOSS_FLOOR_WEEKS = 1.0             # advance-blocker floor: -1 x (daily limit x 5)
DEMOTE_LOSS_WEEKS = 2.0            # demotion floor: -2 x (daily limit x 5)
OPERATIONAL_GREEN_SESSIONS = 10    # consecutive clean sessions to advance


@dataclass
class Stage:
    idx: int
    account_budget_usd: int
    risk_per_trade_usd: int
    daily_loss_limit_usd: int
    min_days_in_stage: Optional[int]

    @property
    def weekly_loss_budget(self) -> float:
        """One full losing week at stage size (daily limit x 5, negative)."""
        return self.daily_loss_limit_usd * 5

    @property
    def advance_loss_floor(self) -> float:
        return LOSS_FLOOR_WEEKS * self.weekly_loss_budget

    @property
    def demote_loss_floor(self) -> float:
        return DEMOTE_LOSS_WEEKS * self.weekly_loss_budget


PRE_STAGE_IDX = -1  # Pre-Stage-0 LIVE half-size data-collection phase
TERMINAL_STAGE_IDX = 4


def _stage_by_idx(i: int) -> Optional[Stage]:
    """Look up Stage by its semantic idx (-1 for Pre-0, 0-4 for the ramp).

    Note: list position != stage idx since Pre-0 is idx=-1.
    """
    for row in STAGES:
        if row[0] == i:
            return Stage(*row)
    return None


def is_pre_stage(stage: Stage) -> bool:
    """True if `stage` is the Pre-Stage-0 LIVE phase."""
    return stage.idx == PRE_STAGE_IDX


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
    """Date of the most recent 'ORB ramp:' commit — the stage-entry marker.
    Used to compute days-in-current-stage AND the stage P&L window.

    8/14 audit fix: the old version filtered by path `-- orb.yaml`, but
    orb.yaml is GITIGNORED — no commit ever touches it, so this always
    returned None and the 'stage' silently became the whole ramp. That
    blended Pre-0 profit into Stage-0 P&L (+$2,057 flattering a true
    −$6,973 to −$4,917) — pro-advancement bias on the above-water rule."""
    try:
        out = subprocess.check_output(
            ['git', 'log', '--format=%ci|%s', '--grep=ORB ramp:', '-n', '20'],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            return None
        for line in out.splitlines():
            ci, _, subject = line.partition('|')
            # Only STAGE-TRANSITION commits mark stage entry ('→ Stage N'
            # / 'Stage N (live launch)') — 'ORB ramp:' policy commits
            # (e.g. 9669c02 operational-gate adoption) do not.
            if '→ Stage' in subject or 'live launch' in subject:
                # '%ci' = "2026-06-04 06:33:05 +0000" — date portion
                return date.fromisoformat(ci.split()[0])
        return None
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
    # STAGE-scoped P&L for the advancement/demotion gates (8/14 audit
    # fix: gating on the whole-ramp cushion blended Pre-0 profit into
    # "stage P&L" — pro-advancement bias on the above-water rule).
    # Whole-ramp cushion is still what the hard stop keys on.
    stage_pnl = sum(pnl for d, pnl in trades if d >= stage_change_date)

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
    # Stage idx semantics: -1 (Pre-0) → 0 (Stage 0) → 1 → 2 → 3 → 4 (terminal).
    next_stage = _stage_by_idx(stage.idx + 1) if stage.idx < TERMINAL_STAGE_IDX else None
    advance_blockers = []
    if next_stage is not None and stage.min_days_in_stage is not None:
        # Loss floor (replaces cushion): block only if performing WORSE
        # than one full losing week at stage size.
        if stage_pnl < stage.advance_loss_floor:
            advance_blockers.append(
                f"stage P&L ${stage_pnl:,.0f} below loss floor "
                f"${stage.advance_loss_floor:,.0f} (-1x weekly loss budget)"
            )
        # Above-water rule (owner decision 2026-07-23): never advance a
        # stage while realized stage P&L is negative — advance on proof,
        # and proof includes being ahead. Stricter than the loss floor;
        # both checked so the blocker messages stay specific.
        if stage_pnl < 0:
            advance_blockers.append(
                f"stage P&L ${stage_pnl:,.0f} under water — hold until "
                f"realized stage P&L > 0 (owner rule 2026-07-23)"
            )
        if stage.min_days_in_stage and days_in_stage < stage.min_days_in_stage:
            advance_blockers.append(
                f"{days_in_stage} trading day(s) in stage < required "
                f"{stage.min_days_in_stage}"
            )
        # Green streak — computed by scripts/daily_green_check.py (21:30 UTC
        # weekdays) into logs/green_streak.json. Falls back to a manual-check
        # reminder if the file doesn't exist yet.
        try:
            import json as _json
            _st = _json.loads(
                (Path(__file__).resolve().parent.parent / 'logs' /
                 'green_streak.json').read_text())
            if _st.get('streak', 0) < OPERATIONAL_GREEN_SESSIONS:
                advance_blockers.append(
                    f"operational-green streak {_st.get('streak', 0)}"
                    f"/{OPERATIONAL_GREEN_SESSIONS} (daily_green_check)")
        except Exception:
            advance_blockers.append(
                f"MANUAL CHECK: {OPERATIONAL_GREEN_SESSIONS} consecutive "
                f"operational-green sessions (no green_streak.json yet — "
                f"daily_green_check.py cron populates it)"
            )
        advance_blockers.append(
            "MANUAL CHECK: median entry slippage <= BT+10bps "
            "(analyze_orb_slippage.py)"
        )

    # --- Demotion triggers (2026-07-06: operational failures OR deep
    # loss floor ONLY — BT-consistent drawdown is NOT a trigger) ---
    demote_triggers = []
    if stage_pnl < stage.demote_loss_floor:
        demote_triggers.append(
            f"stage P&L ${stage_pnl:,.0f} below demotion floor "
            f"${stage.demote_loss_floor:,.0f} (-2x weekly loss budget)"
        )
    if consec_red >= 5:
        demote_triggers.append(
            f"{consec_red} consecutive red days — not an auto-demote under "
            f"the 2026-07-06 policy, but check the BT percentile bands "
            f"before dismissing"
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
    print(f"Realized P&L (ramp):    ${cushion:+,.2f}")
    print(f"Realized P&L (STAGE):   ${stage_pnl:+,.2f}  <- gates key on this")
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
        if is_pre_stage(stage):
            # Pre-0 demotion = revert to paper / research config, not "Stage -2"
            print(f"⚠️  DEMOTION TRIGGERED — revert Pre-Stage-0 → paper:")
            print(f"   Restore orb.yaml to research config ($100K budget) and")
            print(f"   re-investigate before retrying live.")
        else:
            print(f"⚠️  DEMOTION TRIGGERED — step down to stage {max(0, stage.idx - 1)}:")
        for t in demote_triggers:
            print(f"   - {t}")
        print("   Apply via docs/orb_rollout_plan.md § Demotion procedure.")
        print()

    # Pre-Stage-0 has richer gating (slippage + drift) — defer
    if is_pre_stage(stage):
        print(f"ℹ️  Pre-Stage-0 has richer promotion gating (slippage, drift).")
        print(f"   Run: python3 scripts/orb_pre0_daily.py")
        print(f"   for the full eligibility check.")
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
