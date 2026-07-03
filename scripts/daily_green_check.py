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


def build_message(v: dict, streak: int, pnl: dict) -> str:
    day = v['day']
    pnl_txt = '  '.join(f"{k} ${x:+,.0f}" for k, x in sorted(pnl.items())) or 'no closed trades'
    if v['green']:
        parity = (f"BT parity clean ({v['n_bt_selected']} BT picks)"
                  if not v.get('bt_stale')
                  else "⚠ BT parity SKIPPED (nightly BT data stale)")
        return (f"✅ <b>[GREEN {streak}/{rc.GREEN_SESSIONS_NEEDED}] {day}</b> — "
                f"exits attributed, {parity}. {pnl_txt}")
    lines = [f"🔴 <b>[RED DAY] {day} — streak reset</b>"]
    for r in v['reasons']:
        lines.append(f"• {r}")
    lines.append(f"checks: {v['checks']}")
    lines.append(f"P&L: {pnl_txt}")
    lines.append("Ramp streak reset to 0 — investigate before next session.")
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='YYYY-MM-DD (default: last weekday)')
    ap.add_argument('--no-telegram', action='store_true')
    args = ap.parse_args()

    day = args.date or rc.prev_trading_day_utc()
    v = rc.green_verdict(day)
    if v['n_live_rows'] == 0 and v['n_bt_selected'] == 0:
        print(f"{day}: no live rows and no BT picks — non-trading day, "
              f"not recorded", flush=True)
        return 0
    streak = rc.streak_update(day, v['green'], v['reasons'])
    pnl = rc.realized_pnl(day)
    msg = build_message(v, streak, pnl)
    print(msg, flush=True)
    if not args.no_telegram:
        rc.send_telegram(msg)
    return 0 if v['green'] else 1


if __name__ == '__main__':
    sys.exit(main())
