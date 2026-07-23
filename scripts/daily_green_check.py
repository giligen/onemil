"""Daily operational-green check → one-line Telegram + streak file.

Runs weekdays 21:30 UTC (after the 20:30 nightly orb-backtest regenerates
the day's BT ground truth). Exception-shaped: a single compact line when
green, a loud block when red. The streak it persists IS the advancement
gate of the 2026-07-06 ramp policy (orb_ramp_check.py reads it).

Consolidates (after their 7/10 validation window) the separate observer +
touchgo-debug daily messages.

Usage: python3 scripts/daily_green_check.py [--date YYYY-MM-DD] [--no-telegram]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report_common as rc


def build_message(v: dict, streak: int, pnl: dict,
                  sizing_txt: str = '') -> str:
    day = v['day']
    pnl_txt = '  '.join(f"{k} ${x:+,.0f}" for k, x in sorted(pnl.items())) or 'no closed trades'
    if v['green']:
        parity = (f"BT parity clean ({v['n_bt_selected']} BT picks)"
                  if not v.get('bt_stale')
                  else "⚠ BT parity SKIPPED (nightly BT data stale)")
        msg = (f"✅ <b>[GREEN {streak}/{rc.GREEN_SESSIONS_NEEDED}] {day}</b> — "
               f"exits attributed, {parity}. {pnl_txt}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='YYYY-MM-DD (default: last weekday)')
    ap.add_argument('--no-telegram', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='compute + print only: no streak write, no telegram '
                         '(use for smoke tests / re-inspection of past days)')
    ap.add_argument('--force-downgrade', action='store_true',
                    help='allow a green→red re-adjudication of an already-'
                         'recorded day (normally blocked: journal evidence '
                         'decays and false-reds the streak)')
    args = ap.parse_args()

    day = args.date or rc.prev_trading_day_utc()
    v = rc.green_verdict(day)
    if v['n_live_rows'] == 0 and v['n_bt_selected'] == 0:
        print(f"{day}: no live rows and no BT picks — non-trading day, "
              f"not recorded", flush=True)
        return 0
    # Sizing attribution (2026-07-13 mult ships): a recorded pm_mult that
    # doesn't recompute from its recorded inputs is a code bug — HARD gate.
    # News drift (live fetch gap) is soft: fail-open is correct behavior.
    attr = rc.sizing_attribution(day)
    if attr['mult_mismatches']:
        v['reasons'].append(
            f"pm_mult drift vs recompute: {attr['mult_mismatches']}")
        v['green'] = False
    # Field-level decision parity (2026-07-17, z-param desync class):
    # live composites must NUMERICALLY match the BT code path — a
    # decision-RELEVANT divergence is a code/param bug and HARD-fails
    # the day. Decision-irrelevant drift on deep rejects (vendor bar
    # revisions, 7/21 VIVK class) warns without resetting the streak.
    dp = rc.decision_parity(day)
    if dp['mismatches']:
        v['reasons'].append(
            f"composite drift BT vs live ({len(dp['mismatches'])} of "
            f"{dp['n_compared']}): {dp['mismatches'][:4]}")
        v['green'] = False
    if dp.get('warnings'):
        print(f"soft composite drift (deep rejects, no decision impact): "
              f"{dp['warnings'][:4]}", flush=True)
    if args.dry_run:
        existing = rc.read_streak()
        streak = existing.get('streak', 0) if existing else 0
        print(f"DRY RUN — streak file untouched (currently {streak})",
              flush=True)
    else:
        streak = rc.streak_update(day, v['green'], v['reasons'],
                                  allow_downgrade=args.force_downgrade)
    pnl = rc.realized_pnl(day)
    # Pool-level Benzinga indexing-lag audit (soft — vendor latency is not
    # an operational bug, but persistent lag erodes the news-gate edge and
    # must be visible the same evening).
    lag_txt = rc.news_lag_line(rc.news_lag_audit(day))
    sizing_txt = '\n'.join(x for x in (rc.sizing_block(attr), lag_txt) if x)
    msg = build_message(v, streak, pnl, sizing_txt=sizing_txt)
    print(msg, flush=True)
    if not args.no_telegram and not args.dry_run:
        rc.send_telegram(msg)
    return 0 if v['green'] else 1


if __name__ == '__main__':
    sys.exit(main())
