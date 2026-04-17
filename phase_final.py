#!/usr/bin/env python3
"""Unified two-tier filter with extras multiplier cap.

Applies:
  1. T = -0.50 filter on extras (sign-consistent composite: low conviction,
     low vwap dist, low fill vwap dist, early entry_minute)
  2. Extras multiplier cap: clamp effective mult to 1.5x for extras-tier
     trades (conviction/MACD mults overleverage when applied to weaker 10-19%
     setups).

Reports final metrics vs A_f6 on 2025 + Q1 2026. Post-hoc, no cache rebuild.
"""
from __future__ import annotations
import csv
from collections import defaultdict
import statistics as stats


IN_2025 = '/tmp/variant_runner/bt_O_f6_2025_with_intraday.csv'
IN_Q1   = '/tmp/variant_runner/bt_O_f6_q1_with_intraday.csv'

TRAIN_MONTHS    = {f"2025-{m:02d}" for m in range(1, 8)}
VALIDATE_MONTHS = {f"2025-{m:02d}" for m in range(8, 11)}
HOLDOUT_2025    = {"2025-11", "2025-12"}

FILTER_T = -0.50   # accepted from Phase 2 sweep
EXTRAS_MULT_CAP = 1.5  # clamp effective sizing multiplier for extras
BASE_RISK = 200.0  # dollars per trade at 1x


def f(v):
    try:
        return float(v) if v not in (None, '', 'None') else None
    except (ValueError, TypeError):
        return None


def entry_minute(r):
    t = r.get('entry_time_et') or ''
    if not t:
        return None
    try:
        hh, mm, _ = t.split(':')
        return int(hh) * 60 + int(mm)
    except ValueError:
        return None


EXTRAS_FEATURES = [
    ('conviction_mult',        -1, lambda r: f(r.get('conviction_mult'))),
    ('qf_vwap_dist_pct',       -1, lambda r: f(r.get('qf_vwap_dist_pct'))),
    ('qf_fill_vwap_dist_pct',  -1, lambda r: f(r.get('qf_fill_vwap_dist_pct'))),
    ('entry_minute',           -1, entry_minute),
]


def fit_params(rows):
    params = {}
    for name, sign, fn in EXTRAS_FEATURES:
        vals = [fn(r) for r in rows if fn(r) is not None]
        if len(vals) < 10:
            params[name] = {'mean': 0, 'std': 1, 'sign': sign}
            continue
        m = stats.mean(vals)
        s = stats.stdev(vals) if len(vals) > 1 else 1.0
        params[name] = {'mean': m, 'std': s or 1.0, 'sign': sign}
    return params


def score(row, params):
    total = 0.0
    count = 0
    for name, sign, fn in EXTRAS_FEATURES:
        v = fn(row)
        if v is None:
            continue
        p = params[name]
        z = (v - p['mean']) / p['std']
        total += sign * z
        count += 1
    return total / max(count, 1)


def tier(r):
    if r.get('a_eligible') == '1':
        return 'A'
    ic = f(r.get('max_intraday_change_pre_entry'))
    if ic is None:
        return '?'
    if 10.0 <= ic < 20.0:
        return 'E'
    if ic < 10.0:
        return 'edge'
    return '?'


def effective_mult(r):
    """Infer sizing multiplier from shares/base_shares at $200 risk."""
    try:
        risk_ps = float(r['entry_price']) - float(r['stop_loss'])
        if risk_ps <= 0: return 1.0
        base = BASE_RISK / risk_ps
        if base <= 0: return 1.0
        return max(0.01, float(r['shares']) / base)
    except (ValueError, ZeroDivisionError, KeyError):
        return 1.0


def apply_mult_cap(r, cap):
    """Return adjusted pnl if effective mult > cap, else original.

    unit_pnl = pnl / eff_mult
    new_pnl  = unit_pnl * min(eff_mult, cap)
    """
    try:
        pnl = float(r['pnl'])
    except (ValueError, KeyError):
        return 0.0
    eff = effective_mult(r)
    if eff <= cap:
        return pnl
    return pnl * cap / eff


def metrics(trades):
    """trades: list of (date, entry_time_et, pnl_adjusted) tuples."""
    if not trades:
        return {'n': 0, 'pnl': 0, 'wr': 0, 'pf': 0, 'max_dd': 0,
                'strip10': 0, 'days': 0, 'pos_days': 0}
    wins = sum(1 for _, _, p in trades if p > 0)
    pnl = sum(p for _, _, p in trades)
    gw = sum(p for _, _, p in trades if p > 0)
    gl = abs(sum(p for _, _, p in trades if p <= 0))
    pf = gw / gl if gl > 0 else 999
    eq = peak = dd = 0
    for _, _, p in sorted(trades):
        eq += p
        if eq > peak: peak = eq
        if peak - eq > dd: dd = peak - eq
    by_day = defaultdict(float)
    for d, _, p in trades:
        by_day[d] += p
    days_sorted = sorted(by_day.values(), reverse=True)
    strip10 = sum(days_sorted[10:]) if len(days_sorted) > 10 else sum(days_sorted)
    pos_days = sum(1 for v in by_day.values() if v > 0)
    return {
        'n': len(trades), 'pnl': pnl, 'wr': wins/len(trades)*100, 'pf': pf,
        'max_dd': dd, 'strip10': strip10, 'days': len(by_day), 'pos_days': pos_days,
    }


def apply_two_tier(rows, params, T, extras_cap):
    """Apply filter + extras multiplier cap. Returns list of (date, entry_time, pnl)."""
    trades = []
    for r in rows:
        t = tier(r)
        if t == 'A' or t == 'edge':
            try:
                trades.append((r['date'], r['entry_time_et'], float(r['pnl'])))
            except (ValueError, KeyError):
                pass
        elif t == 'E':
            if score(r, params) >= T:
                trades.append((r['date'], r['entry_time_et'], apply_mult_cap(r, extras_cap)))
    return trades


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--in-2025', type=str, default=IN_2025)
    p.add_argument('--in-q1',   type=str, default=IN_Q1)
    args = p.parse_args()
    rows_2025 = list(csv.DictReader(open(args.in_2025)))
    rows_q1 = list(csv.DictReader(open(args.in_q1)))

    train_extras = [r for r in rows_2025 if r['date'][:7] in TRAIN_MONTHS and tier(r) == 'E']
    params = fit_params(train_extras)
    print(f"Extras z-score params (fit on TRAIN={len(train_extras)} extras):")
    for name, p in params.items():
        print(f"  {name:<24} mean={p['mean']:+.3f}  std={p['std']:.3f}")
    print(f"\nFilter T = {FILTER_T}  (extras with composite score >= T are kept)")
    print(f"Extras multiplier cap = {EXTRAS_MULT_CAP}x  (above-cap extras get pnl re-scaled)\n")

    # Show contribution of each change
    print("=" * 110)
    print(f"  Ablation — building up from RAW to final configuration")
    print(f"{'='*110}")
    print(f"  {'config':<50} {'2025':<28}  {'Q1 2026':<22}")
    print(f"  {'':<50} {'n   WR   PF   P&L     DD':<28}  {'n   WR   PF   P&L':<22}")
    print("-" * 110)

    configs = [
        ('O_f6 RAW (no filter, no cap)',               -1e9, 1e9),
        ('O_f6 + extras-mult cap 1.5x only',           -1e9, EXTRAS_MULT_CAP),
        (f'O_f6 + filter T={FILTER_T}, kept mults',        FILTER_T, 1e9),
        (f'O_f6 + filter T={FILTER_T} + extras cap 1.5x', FILTER_T, EXTRAS_MULT_CAP),
    ]
    for label, T, cap in configs:
        t25 = apply_two_tier(rows_2025, params, T, cap)
        tq1 = apply_two_tier(rows_q1, params, T, cap)
        m25 = metrics(t25)
        mq1 = metrics(tq1)
        print(f"  {label:<50} {m25['n']:>3} {m25['wr']:>4.1f}% {m25['pf']:>4.2f} "
              f"${m25['pnl']:>+7,.0f} ${m25['max_dd']:>5,.0f}  "
              f"{mq1['n']:>3} {mq1['wr']:>4.1f}% {mq1['pf']:>4.2f} ${mq1['pnl']:>+6,.0f}")

    # Final result
    final_25 = apply_two_tier(rows_2025, params, FILTER_T, EXTRAS_MULT_CAP)
    final_q1 = apply_two_tier(rows_q1, params, FILTER_T, EXTRAS_MULT_CAP)
    m25 = metrics(final_25)
    mq1 = metrics(final_q1)

    print(f"\n{'='*80}")
    print(f"  Final — O_f6 + two-tier filter (T={FILTER_T}) + extras cap ({EXTRAS_MULT_CAP}x)")
    print(f"{'='*80}")
    print(f"  {'set':<22} {'n':>4} {'WR':>7} {'PF':>5} {'P&L':>10} {'DD':>7} {'strip10':>8}  {'days':>4} {'pos_days':>5}")
    print(f"  A_f6 2025 (reference)     83   60.2%  3.50   +54,572   2,502    +11,221   —    —")
    print(f"  O_f6 2025 FINAL          {m25['n']:>4} {m25['wr']:>6.1f}% {m25['pf']:>5.2f} "
          f"{m25['pnl']:>+10,.0f} {m25['max_dd']:>7,.0f} {m25['strip10']:>+8,.0f}  "
          f"{m25['days']:>4} {m25['pos_days']:>5}")
    print(f"  A_f6 Q1 2026 (reference)   —     —     —     +4,495      —       —    —    —")
    print(f"  O_f6 Q1 2026 FINAL       {mq1['n']:>4} {mq1['wr']:>6.1f}% {mq1['pf']:>5.2f} "
          f"{mq1['pnl']:>+10,.0f} {mq1['max_dd']:>7,.0f} {mq1['strip10']:>+8,.0f}  "
          f"{mq1['days']:>4} {mq1['pos_days']:>5}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
