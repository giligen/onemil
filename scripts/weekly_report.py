"""Weekly owner report → Telegram. Fridays 21:45 UTC.

The decision document: P&L per day per strategy, ramp-gate progress
(green streak + loss-floor headroom), edge capture (BT-at-stage-scale vs
live, missed BT picks, runner-capture), monster watch, flags.

Usage: python3 scripts/weekly_report.py [--end YYYY-MM-DD] [--no-telegram]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report_common as rc

MONSTER_LIVE_USD = 3000.0   # ~3R at Stage-0 $1K risk


def week_days(end: date):
    monday = end - timedelta(days=end.weekday())
    return [(monday + timedelta(days=i)).isoformat()
            for i in range(5) if monday + timedelta(days=i) <= end]


def build_report(end: date) -> str:
    days = week_days(end)
    lines = [f"<b>[WEEKLY] {days[0]} → {days[-1]}</b>", ""]

    # --- P&L per day per strategy ---
    tot: dict = {}
    lines.append("<b>P&L (realized)</b>")
    for d in days:
        pnl = rc.realized_pnl(d)
        for k, v in pnl.items():
            tot[k] = tot.get(k, 0.0) + v
        day_total = sum(pnl.values())
        detail = '  '.join(f"{k} {v:+,.0f}" for k, v in sorted(pnl.items()))
        lines.append(f"  {d[5:]}: ${day_total:+,.0f}  ({detail or 'flat'})")
    week_total = sum(tot.values())
    lines.append(f"  <b>week: ${week_total:+,.0f}</b>  "
                 + '  '.join(f"{k} {v:+,.0f}" for k, v in sorted(tot.items())))

    # --- Ramp gate progress ---
    lines.append("")
    lines.append("<b>Ramp (Stage 0 → 1)</b>")
    st = rc.read_streak()
    if st:
        lines.append(f"  green sessions: <b>{st['streak']}/{st['needed']}</b>")
        reds = [r for r in st.get('days', [])[-10:] if not r['green']]
        if reds:
            lines.append(f"  recent reds: " + '; '.join(
                f"{r['day']}: {', '.join(r['reasons'])[:80]}" for r in reds))
    else:
        lines.append("  green streak: no data yet (daily check starts Monday)")
    cum = rc.cumulative_orb_since(rc.RAMP_START)
    headroom = cum - rc.ADVANCE_LOSS_FLOOR
    lines.append(f"  ORB since ramp-start: ${cum:+,.0f}  "
                 f"(loss-floor headroom ${headroom:+,.0f} above {rc.ADVANCE_LOSS_FLOOR:+,.0f})")

    # --- Edge capture: BT at stage scale vs live ---
    lines.append("")
    lines.append("<b>Edge capture (BT stage-scaled vs live ORB)</b>")
    bt_week = []
    for d in days:
        bt_week.extend(rc.load_bt_selected(d))
    bt_stage = sum(r.get('_sized_pnl', 0.0) for r in bt_week) * rc.STAGE_SCALE
    live_orb = tot.get('orb', 0.0)
    lines.append(f"  BT (~stage-scaled): ${bt_stage:+,.0f} on {len(bt_week)} picks | "
                 f"live: ${live_orb:+,.0f} | gap ${live_orb - bt_stage:+,.0f}")
    # missed BT picks
    missed = []
    for d in days:
        bt_syms = {r['symbol'] for r in rc.load_bt_selected(d)}
        live_syms = {r['symbol'] for r in rc.load_live_rows(d, strategy='orb')}
        missed += [f"{s}({d[5:]})" for s in sorted(bt_syms - live_syms)]
    lines.append(f"  BT picks never ordered live: {', '.join(missed) if missed else 'none'}")
    # runner capture: best BT pick of the week vs what live banked on it
    if bt_week:
        best = max(bt_week, key=lambda r: r.get('_sized_pnl', 0))
        b_day = str(best['date'])[:10]
        live_rows = [r for r in rc.load_live_rows(b_day, strategy='orb')
                     if r['symbol'] == best['symbol'] and r.get('exit_price')]
        banked = sum((r['exit_price'] - r['entry_price']) * (r.get('shares') or 0)
                     for r in live_rows)
        lines.append(f"  runner-capture: best BT pick {best['symbol']} {b_day[5:]} "
                     f"${best.get('_sized_pnl', 0) * rc.STAGE_SCALE:+,.0f} stage-scaled → "
                     f"live banked ${banked:+,.0f}"
                     + ("" if live_rows else " (NOT TRADED)"))

    # --- Monster watch ---
    lines.append("")
    import sqlite3
    conn = sqlite3.connect(rc.ROOT / 'data' / 'trades.db', timeout=15)
    row = conn.execute(
        """SELECT MAX(trade_date) FROM trades WHERE strategy='orb'
             AND exit_price IS NOT NULL
             AND (exit_price-entry_price)*shares >= ?""",
        (MONSTER_LIVE_USD,)).fetchone()
    conn.close()
    if row and row[0]:
        lines.append(f"<b>Monster watch</b>: last ≥${MONSTER_LIVE_USD:,.0f} ORB win {row[0]} "
                     f"(base rate: clean monsters ~3-4/yr — patience is the position)")
    else:
        lines.append(f"<b>Monster watch</b>: none ≥${MONSTER_LIVE_USD:,.0f} yet since ramp "
                     f"(expected wait at base rates: months, not weeks)")

    # --- Flags ---
    lines.append("")
    flags = []
    if cum < rc.ADVANCE_LOSS_FLOOR:
        flags.append(f"stage P&L below advance loss floor")
    if cum < 2 * rc.ADVANCE_LOSS_FLOOR:
        flags.append(f"DEMOTION floor breached")
    if st and st['streak'] == 0 and st.get('days'):
        flags.append("streak at 0 — red day this week")
    lines.append("<b>Flags</b>: " + ('; '.join(flags) if flags else 'none'))
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (default: today)')
    ap.add_argument('--no-telegram', action='store_true')
    args = ap.parse_args()
    end = date.fromisoformat(args.end) if args.end else date.today()
    msg = build_report(end)
    print(msg, flush=True)
    if not args.no_telegram:
        rc.send_telegram(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
