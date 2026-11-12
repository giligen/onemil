"""Daily operational-green check → one-line Telegram + the ONE Amendment-2 streak file.

ONE counter (2026-11-12, docs/green_check_one_counter_spec_20261112.md). A session is GREEN iff (a) exits are
attributed and nothing is stuck, (b) the pick-parity verdict for the day (scripts/orb_pick_parity.run_parity) has zero
`unexplained` difference and no live fill outside the final set, and (c) the nightly BT for the day exists. The BT is
WAITED for (``--wait-bt-min``, poll 60 s); still absent = RED with the reason, never "skipped". A vendor no-range day
(every pick `range_missing`) is NOT COUNTED: recorded, streak unchanged. The streak counts only days >= ORB_DAY_ONE.

Runs AFTER the BT build and the pick-parity run (crontab 23:20 UTC). The streak is the advancement gate of the ramp
policy (orb_ramp_check.py reads it).

Usage: python3 scripts/daily_green_check.py [--date|--day YYYY-MM-DD] [--no-telegram] [--dry-run] [--wait-bt-min 45]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orb_pick_parity as opp
import report_common as rc
from trading import ramp_freeze

logger = logging.getLogger('daily_green_check')

BT_POLL_SECONDS = 60

# Gate-1 HARD parity items (docs/scaling_plan_2026.md): a reason line that
# starts with one of these means "we do not know what we are running" and
# FREEZES the ORB ramp (size unchanged, entries continue, stage clock stops).
# Everything else that reds a day (e.g. service-uptime CONTEXT) is a P&L /
# operations event, not a parity breach.
HARD_PARITY_MARKERS = (
    'pm_mult drift vs recompute',      # recorded vs recomputed sizing mult
    'pick parity',                     # selection / composite parity (named reasons, scripts/orb_pick_parity.py)
    'fill-parity',                     # BT filled, live never filled
    'unattributed exits',              # exit parity
    'exit_pending_verification',       # unverified exit rows
    'floored-stop drift',              # live monitored an unvalidated stop
)


def parity_breaches(reasons) -> list:
    """The subset of red-day reasons that are Gate-1 HARD parity breaches."""
    return [r for r in reasons
            if any(m in r for m in HARD_PARITY_MARKERS)]


def freeze_owned_by_day(state, day: str) -> bool:
    """True when the ORB freeze's latest `freeze` event was raised by the check of `day` itself (a re-run of that
    day may re-judge it); a freeze raised by another day's check is never touched by this day's verdict."""
    events = [h for h in state.history if h.get('action') == 'freeze']
    return bool(events) and events[-1].get('day') == day


def reconcile_freeze(day: str, breaches: list, dry_run: bool = False, notify: bool = True) -> str:
    """Re-judge the ORB ramp freeze against the day's CURRENT hard breaches (2026-10-06).

    * breaches -> freeze (or re-freeze): the stored reason becomes the surviving breaches only, so a reason that
      listed retired attributions (HOD rows) now says what is really left (e.g. a true 'BT picks never ordered').
    * no breaches and the freeze was raised by THIS day's check -> cleared by code with the reason recorded; a
      freeze raised by another day, or a manual one, stays (`--clear-freeze` is the only path for those).
    Returns one `RAMP FREEZE orb: before -> after` line (state before / after) for the log."""
    before = ramp_freeze.get('orb')
    b_txt = f"frozen since {before.since}: {before.reason}" if before.frozen else "not frozen"
    if breaches:
        if dry_run:
            return f"RAMP FREEZE orb DRY RUN: {b_txt} -> would FREEZE: {breaches}"
        after = ramp_freeze.set_freeze('orb', '; '.join(breaches), day=day, notify=notify)
        return f"RAMP FREEZE orb: {b_txt} -> frozen since {after.since}: {after.reason}"
    if before.frozen and freeze_owned_by_day(before, day):
        why = (f"{day} re-run: no hard parity breach left (ORB-only exit attribution, no true "
               f"'BT pick never ordered'); was: {before.reason}")
        if dry_run:
            return f"RAMP FREEZE orb DRY RUN: {b_txt} -> would CLEAR ({why})"
        ramp_freeze.clear_freeze('orb', why, by='daily_green_check', day=day)
        return f"RAMP FREEZE orb: {b_txt} -> cleared by code ({why})"
    return f"RAMP FREEZE orb: {b_txt} -> unchanged (no breach on {day}; freeze not raised by {day}'s check)"


def wait_for_bt(day: str, wait_min: float, sleep=None, bt_max_fn=None) -> bool:
    """Poll `bt_data_max_date() >= day` every BT_POLL_SECONDS for up to `wait_min` minutes; True when the BT is there."""
    sleep = sleep or time.sleep
    bt_max_fn = bt_max_fn or rc.bt_data_max_date
    deadline_polls = int(wait_min * 60 // BT_POLL_SECONDS)
    for i in range(deadline_polls + 1):
        m = bt_max_fn()
        if m is not None and m >= day:
            return True
        if i < deadline_polls:
            print(f"{day}: nightly BT not there yet (max={m}) — waiting {BT_POLL_SECONDS}s "
                  f"({i + 1}/{deadline_polls})", flush=True)
            sleep(BT_POLL_SECONDS)
    return False


def is_no_range_day(res: dict) -> bool:
    """True for a vendor no-range day: the live side ranked NOTHING (empty final set), BT has picks, none of the
    differences is unexplained, and the pick parity names every BT pick `range_missing` (no range = no ranking)."""
    missing = {d['symbol'] for d in res.get('differences', []) if d['reason'] == 'range_missing'}
    picks = set(res.get('bt_top8') or [])
    return bool(picks) and not res.get('live_final') and not res.get('n_unexplained') and picks <= missing


def pick_parity_verdict(day: str) -> dict:
    """The pick-parity result of `day` (never emits its own ERROR: this check owns the one alarm)."""
    return opp.run_parity(day, emit=False)


def apply_pick_parity(v: dict, res: dict) -> bool:
    """Fold the pick-parity result into verdict `v` (RED on an unexplained difference / fill outside the final set).
    Returns True when the day is NOT COUNTED (a no-range vendor day). A BT side the pick-parity run could not read
    AFTER the nightly BT was waited for is OUR defect: RED with the reason and one ERROR, never 'not counted'."""
    if not res.get('bt_available'):
        why = res.get('bt_why', 'unknown')
        v['reasons'].append(f"pick parity BT side unavailable for {v.get('day')}: {why} — the day is RED")
        v['checks']['pick_parity'] = f"BT side unavailable ({why})"
        v['green'] = False
        logger.error("ORB green check %s: pick-parity BT side unavailable (%s) although the nightly BT exists — "
                     "bt_ranked / book path defect; the day is RED", v.get('day'), why)
        return False
    if is_no_range_day(res):
        return True
    bad = sorted({d['symbol'] for d in res['differences'] if d['reason'] == 'unexplained'})
    leaks = list(res.get('fills_outside_final', []))
    if res['n_unexplained'] or leaks:
        v['reasons'].append(f"pick parity: {res['n_unexplained']} unexplained {bad}, live fills outside the final set "
                            f"{leaks} — {res.get('summary') or opp.summary_line(res)}")
        v['green'] = False
    v['checks']['pick_parity'] = res.get('summary') or opp.summary_line(res)
    return False


def build_message(v: dict, streak: int, pnl: dict,
                  sizing_txt: str = '') -> str:
    day = v['day']
    pnl_txt = '  '.join(f"{k} ${x:+,.0f}" for k, x in sorted(pnl.items())) or 'no closed trades'
    if v.get('not_counted'):
        return (f"⚪ <b>[NOT COUNTED] {day}</b> — vendor no-range day, streak unchanged at {streak}. {pnl_txt}"
                + (f"\n{sizing_txt}" if sizing_txt else ''))
    if v['green']:
        msg = (f"✅ <b>{rc.green_label(streak)} {day}</b> — "
               f"exits attributed, pick parity clean ({v['n_bt_selected']} BT picks). {pnl_txt}")
        return msg + (f"\n{sizing_txt}" if sizing_txt else '')
    lines = [f"🔴 <b>[RED DAY] {day} — streak reset</b>"]
    for r in v['reasons']:
        lines.append(f"• {r}")
    lines.append(f"checks: {v['checks']}")
    lines.append(f"P&L: {pnl_txt}")
    if sizing_txt:
        lines.append(sizing_txt)
    lines.append("Ramp streak reset to 0 — investigate before next session.")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', '--day', dest='date', default=None, help='YYYY-MM-DD (default: last weekday)')
    ap.add_argument('--no-telegram', action='store_true')
    ap.add_argument('--wait-bt-min', type=float, default=45.0,
                    help='minutes to wait (poll 60 s) for the nightly BT of the day before judging it RED')
    ap.add_argument('--dry-run', action='store_true',
                    help='compute + print only: no streak write, no telegram '
                         '(use for smoke tests / re-inspection of past days)')
    ap.add_argument('--force-downgrade', action='store_true',
                    help='allow a green→red re-adjudication of an already-'
                         'recorded day (normally blocked: journal evidence '
                         'decays and false-reds the streak)')
    args = ap.parse_args(argv)

    day = args.date or rc.prev_trading_day_utc()
    bt_ok = wait_for_bt(day, args.wait_bt_min)
    if not bt_ok:
        logger.error("ORB green check %s: nightly BT missing — look at onemil-orb-backtest.timer (21:30 UTC) / "
                     "nightly_bt_update.sh (22:30 UTC); the day is RED", day)
    v = rc.green_verdict(day)
    if v['n_live_rows'] == 0 and v['n_bt_selected'] == 0 and bt_ok:
        # Zero fills + zero BT picks is NOT the same as a holiday
        # (2026-07-30 finding: 7/23, 7/24 and 7/28 were live trading
        # days with scored candidates and vetoes but zero fills — the
        # old bail skipped ALL operational checks incl. decision parity
        # on exactly those days and left holes in the streak record).
        # If the engine scored anything that day, it traded — proceed.
        scored = len(rc.journal_grep('ORB SCORED', day)) \
            + len(rc.journal_grep('below filter threshold', day))
        if scored == 0:
            print(f"{day}: no live rows, no BT picks, no scored "
                  f"candidates — genuine non-trading day, not recorded",
                  flush=True)
            return 0
        print(f"{day}: zero-fill trading day ({scored} scored lines) — "
              f"running operational checks anyway", flush=True)
    # Sizing attribution (2026-07-13 mult ships): a recorded pm_mult that
    # doesn't recompute from its recorded inputs is a code bug — HARD gate.
    # News drift (live fetch gap) is soft: fail-open is correct behavior.
    attr = rc.sizing_attribution(day)
    if attr['mult_mismatches']:
        v['reasons'].append(
            f"pm_mult drift vs recompute: {attr['mult_mismatches']}")
        v['green'] = False
    # Pick parity (the ONE selection / composite parity definition, named reasons).
    not_counted = False
    if bt_ok:
        res = pick_parity_verdict(day)
        print(res.get('summary') or opp.summary_line(res), flush=True)
        not_counted = apply_pick_parity(v, res)
        if not_counted:
            logger.warning("ORB green check %s NOT COUNTED (streak unchanged): %s", day,
                           'vendor no-range day' if res.get('bt_available') else
                           f"BT side unavailable ({res.get('bt_why', 'unknown')})")
    v['not_counted'] = not_counted
    if args.dry_run:
        existing = rc.read_streak()
        streak = existing.get('streak', 0) if existing else 0
        print(f"DRY RUN — streak file untouched (currently {streak})",
              flush=True)
    else:
        streak = rc.streak_update(day, v['green'] and not not_counted, [] if not_counted else v['reasons'],
                                  allow_downgrade=args.force_downgrade, counted=not not_counted)
    pnl = rc.realized_pnl(day)
    # Pool-level Benzinga indexing-lag audit (soft — vendor latency is not
    # an operational bug, but persistent lag erodes the news-gate edge and
    # must be visible the same evening).
    lag_txt = rc.news_lag_line(rc.news_lag_audit(day))
    # BF kill-rail state (Discipline Program Phase 1) — same DB sums the
    # live rails read + the month-pause flag file. Always shown (EoD-dive
    # visibility); a breach reads loudly but is not a RED — the rails
    # firing is the system working, not an operational bug.
    bf_txt = rc.bf_rails_line(rc.bf_rails_status(day))
    sizing_txt = '\n'.join(
        x for x in (rc.sizing_block(attr), lag_txt, bf_txt) if x)
    # Gate-1: a HARD parity breach FREEZES the ORB ramp; a re-run of the same day that finds no breach left
    # clears the freeze it raised itself (reconcile_freeze); every other clear is
    # `orb_ramp_check.py --clear-freeze orb "<reason>"`.
    breaches = [] if not_counted else parity_breaches(v['reasons'])
    print(reconcile_freeze(day, breaches, dry_run=args.dry_run,
                           notify=not args.no_telegram), flush=True)
    msg = build_message(v, streak, pnl, sizing_txt=sizing_txt)
    print(msg, flush=True)
    if not args.no_telegram and not args.dry_run:
        rc.send_telegram(msg)
    return 0 if (v['green'] or not_counted) else 1


if __name__ == '__main__':
    sys.exit(main())
