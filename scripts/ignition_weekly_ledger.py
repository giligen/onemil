#!/usr/bin/env python3
"""Ignition weekly Σrr ledger — THE committed aggregation.

8/30 audit: the headline weekly table (research/ignition_week_aug17_
review.md) was assembled ad hoc in-session — no committed code grouped
by week, summed rr, or produced the flat-$ column, and six scripts could
each print "the ignition numbers" with three different maths. This
script is the single reproducible producer, built on the SAME per-trade
math as research/scripts/ignition_bt_replay.py (imported, not copied).

Era discipline: journals before 2026-08-14 predate the trigger-bar
re-key and carry era-lossy catalyst fields (news-resolution rate swung
1.5%-100% across the 7/24 gate-order refactor) — they are EXCLUDED by
default and cannot be compared against post-8/14 weeks. --include-legacy
prints them in a separate, clearly-labeled section.

Usage:
    python3 scripts/ignition_weekly_ledger.py [--start YYYY-MM-DD]
        [--risk-usd 50] [--include-legacy]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ERA_FLOOR = date(2026, 8, 14)   # trigger-bar re-key (70109bc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None)
    ap.add_argument('--risk-usd', type=float, default=50.0)
    ap.add_argument('--include-legacy', action='store_true')
    args = ap.parse_args()

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'ignition_bt_replay',
        os.path.join(os.path.dirname(__file__), '..', 'research',
                     'scripts', 'ignition_bt_replay.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    replay_day = _mod.replay_day

    days = sorted(
        p.split('ignition_shadow_')[1].split('.jsonl')[0]
        for p in glob.glob('logs/ignition_shadow_*.jsonl'))
    if args.start:
        days = [d for d in days if d >= args.start]

    weeks: dict = defaultdict(lambda: {'rr': 0.0, 'n': 0, 'days': 0,
                                       'legacy': False})
    for d in days:
        dd = date.fromisoformat(d)
        legacy = dd < ERA_FLOOR
        if legacy and not args.include_legacy:
            continue
        try:
            out = replay_day(d, verbose=False)
        except Exception as e:
            print(f"{d}: replay failed ({e}) — day EXCLUDED (a gap is a "
                  f"finding, not a zero)")
            continue
        if out is None:
            continue
        iso = dd.isocalendar()
        key = f"{iso[0]}-wk{iso[1]:02d}" + (' [LEGACY-ERA]' if legacy else '')
        wk = weeks[key]
        wk['days'] += 1
        for t in out['kept']:
            wk['rr'] += float(t['rr'])
            wk['n'] += 1
        wk['legacy'] = legacy

    print(f"{'week':<22} {'days':>4} {'trades':>6} {'Σrr':>8} "
          f"{'flat-$'+format(args.risk_usd, '.0f'):>10}")
    for key in sorted(weeks):
        wk = weeks[key]
        print(f"{key:<22} {wk['days']:>4} {wk['n']:>6} {wk['rr']:>+8.2f} "
              f"{wk['rr'] * args.risk_usd:>+10.2f}")
    if not args.include_legacy:
        n_legacy = sum(1 for d in days if date.fromisoformat(d) < ERA_FLOOR)
        if n_legacy:
            print(f"\n({n_legacy} pre-{ERA_FLOOR} journal days excluded — "
                  f"era-lossy catalyst fields; --include-legacy to show "
                  f"separately, NEVER comparable)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
