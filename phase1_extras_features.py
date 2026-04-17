#!/usr/bin/env python3
"""Phase 1: feature-lift on the extras subset (max_intraday_change_pre_entry 10-19%).

Uses the cache CSV augmented by phase0_intraday_change.py. Restricts analysis
to a_eligible==0 AND max_intraday_change_pre_entry >= 10, which is the 165-trade
subset where O takes trades A doesn't.

For each candidate feature, compute:
  - TRAIN median-split: above-median WR vs below-median WR
  - VALIDATE median-split: same
  - Sign agreement on WR_lift AND pnl_lift (only keep sign-consistent features)

TRAIN = 2025 Jan-Jul, VALIDATE = 2025 Aug-Oct, HOLDOUT = 2025 Nov-Dec + Q1 2026.

Features tried:
  - All the Stage-A features from feature_hunt.py
  - max_intraday_change_pre_entry itself (is within-extras change predictive?)
  - entry_minute (time of day)
"""
from __future__ import annotations
import csv
from collections import defaultdict
from statistics import median
from typing import Callable


IN_2025 = '/tmp/variant_runner/bt_O_f6_2025_with_intraday.csv'
IN_Q1   = '/tmp/variant_runner/bt_O_f6_q1_with_intraday.csv'

TRAIN_MONTHS    = {f"2025-{m:02d}" for m in range(1, 8)}
VALIDATE_MONTHS = {f"2025-{m:02d}" for m in range(8, 11)}
HOLDOUT_2025    = {"2025-11", "2025-12"}


def f(v):
    try:
        return float(v) if v not in (None, '', 'None') else None
    except (ValueError, TypeError):
        return None


def parse_entry_minute(r):
    t = r.get('entry_time_et') or ''
    if not t:
        return None
    try:
        hh, mm, _ = t.split(':')
        return int(hh) * 60 + int(mm)
    except ValueError:
        return None


FEATURES: list[tuple[str, Callable]] = [
    ("qf_pole_gain_pct",           lambda r: f(r.get('qf_pole_gain_pct'))),
    ("qf_pole_bars",               lambda r: f(r.get('qf_pole_bars'))),
    ("qf_vwap_dist_pct",           lambda r: f(r.get('qf_vwap_dist_pct'))),
    ("qf_fill_vwap_dist_pct",      lambda r: f(r.get('qf_fill_vwap_dist_pct'))),
    ("qf_gap_pct",                 lambda r: f(r.get('qf_gap_pct'))),
    ("qf_spy_return_pct",          lambda r: f(r.get('qf_spy_return_pct'))),
    ("daily_range_pct",            lambda r: f(r.get('daily_range_pct'))),
    ("avg_volume_20d",             lambda r: f(r.get('avg_volume_20d'))),
    ("conviction_mult",            lambda r: f(r.get('conviction_mult'))),
    ("macd_zone_mult",             lambda r: f(r.get('macd_zone_mult'))),
    ("conv_pole_gain",             lambda r: f(r.get('conv_pole_gain'))),
    ("conv_flag_tightness",        lambda r: f(r.get('conv_flag_tightness'))),
    ("conv_vol_ratio",             lambda r: f(r.get('conv_vol_ratio'))),
    ("conv_spy_regime",            lambda r: f(r.get('conv_spy_regime'))),
    ("conv_retracement",           lambda r: f(r.get('conv_retracement'))),
    ("conv_raw_score",             lambda r: f(r.get('conv_raw_score'))),
    ("spy_3d_range",               lambda r: f(r.get('spy_3d_range'))),
    ("entry_price",                lambda r: f(r.get('entry_price'))),
    ("pole_gain_over_bars",        lambda r: (
        (f(r.get('qf_pole_gain_pct')) or 0) / max(f(r.get('qf_pole_bars')) or 1, 1)
    )),
    ("max_intraday_change_pre",    lambda r: f(r.get('max_intraday_change_pre_entry'))),
    ("entry_minute",               parse_entry_minute),
]


def split_by_month(rows):
    tr, va, ho25 = [], [], []
    for r in rows:
        m = r['date'][:7]
        if m in TRAIN_MONTHS:    tr.append(r)
        elif m in VALIDATE_MONTHS: va.append(r)
        elif m in HOLDOUT_2025:  ho25.append(r)
    return tr, va, ho25


def median_split_stats(rows, fn):
    vals = []
    for r in rows:
        v = fn(r)
        if v is None:
            continue
        try:
            pnl = float(r['pnl'])
        except ValueError:
            continue
        vals.append((v, pnl))
    if len(vals) < 20:
        return None
    vals.sort(key=lambda x: x[0])
    med = median(v for v, _ in vals)
    above = [p for v, p in vals if v > med]
    below = [p for v, p in vals if v <= med]
    if not above or not below:
        return None
    a_wr = sum(1 for p in above if p > 0) / len(above) * 100
    b_wr = sum(1 for p in below if p > 0) / len(below) * 100
    return {
        'n': len(vals), 'median': med,
        'above_n': len(above), 'below_n': len(below),
        'above_wr': a_wr, 'below_wr': b_wr,
        'wr_lift': a_wr - b_wr,
        'above_avg': sum(above) / len(above),
        'below_avg': sum(below) / len(below),
        'pnl_lift': sum(above)/len(above) - sum(below)/len(below),
    }


def main() -> int:
    rows_2025 = list(csv.DictReader(open(IN_2025)))
    rows_q1 = list(csv.DictReader(open(IN_Q1)))

    def is_extra(r):
        e = r.get('a_eligible')
        ic = f(r.get('max_intraday_change_pre_entry'))
        return e == '0' and ic is not None and 10.0 <= ic < 20.0

    extras_25 = [r for r in rows_2025 if is_extra(r)]
    extras_q1 = [r for r in rows_q1 if is_extra(r)]
    tr, va, ho25 = split_by_month(extras_25)
    print(f"Extras — TRAIN={len(tr)}  VAL={len(va)}  HOLDOUT 2025 Nov-Dec={len(ho25)}  Q1 2026={len(extras_q1)}")

    def metrics(rs):
        if not rs: return (0, 0, 0)
        pnl = sum(float(r['pnl']) for r in rs)
        wins = sum(1 for r in rs if float(r['pnl']) > 0)
        return len(rs), pnl, wins / len(rs) * 100
    for name, rs in [('TRAIN', tr), ('VAL', va), ('HOLD25', ho25), ('HOLD Q1', extras_q1)]:
        n, pnl, wr = metrics(rs)
        print(f"  {name:<8} n={n:>3}  pnl=${pnl:>+9,.0f}  WR {wr:>5.1f}%")

    print(f"\n{'='*112}")
    print(f"  Extras feature lift — TRAIN vs VALIDATE median split")
    print(f"{'='*112}")
    print(f"  {'feature':<24} {'TRAIN wr_lift  pnl_lift (n a/b)':<36} "
          f"{'VAL wr_lift  pnl_lift (n a/b)':<36} sign")
    print("-" * 112)
    results = []
    for name, fn in FEATURES:
        t = median_split_stats(tr, fn)
        v = median_split_stats(va, fn)
        if t is None or v is None:
            continue
        sign_wr = (t['wr_lift'] > 0) == (v['wr_lift'] > 0)
        sign_pnl = (t['pnl_lift'] > 0) == (v['pnl_lift'] > 0)
        ok = '✓' if sign_wr and sign_pnl else ('≈' if sign_wr or sign_pnl else '✗')
        print(f"  {name:<24} {t['wr_lift']:>+6.1f}pt ${t['pnl_lift']:>+6.0f} ({t['above_n']}/{t['below_n']})  "
              f"{v['wr_lift']:>+6.1f}pt ${v['pnl_lift']:>+6.0f} ({v['above_n']}/{v['below_n']})  {ok}")
        results.append((name, fn, t, v, sign_wr and sign_pnl))

    print(f"\n=== Top sign-consistent features (|TRAIN wr_lift| desc) ===")
    good = [r for r in results if r[4]]
    good.sort(key=lambda r: abs(r[2]['wr_lift']), reverse=True)
    for name, fn, t, v, _ in good[:8]:
        direction = 'HIGH→WIN' if t['wr_lift'] > 0 else 'LOW→WIN'
        print(f"  {name:<24} {direction}  TRAIN {t['wr_lift']:+.1f}pt ${t['pnl_lift']:+.0f}  "
              f"VAL {v['wr_lift']:+.1f}pt ${v['pnl_lift']:+.0f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
