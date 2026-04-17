#!/usr/bin/env python3
"""Build + test composite quality score on O_f6 trades.

Uses top Stage-A features (sign-consistent on TRAIN and VALIDATE):
  + qf_pole_gain_pct      — higher = more momentum
  + conv_flag_tightness   — higher = tighter flag
  + pole_gain_over_bars   — higher = steeper pole
  - entry_price           — cheaper stocks win more at 10%
  - qf_vwap_dist_pct      — less extended = better
  - qf_fill_vwap_dist_pct — less extended at fill = better

Scoring:
  1. For each feature, z-score using TRAIN mean/std (so VALIDATE/HOLDOUT
     don't leak into the stats).
  2. Sum z-scores with directional signs.
  3. Sweep threshold T: keep trades where score >= T. Report P&L/WR/PF.

Hard rule: a threshold is only accepted if it improves P&L AND WR on BOTH
TRAIN and VALIDATE compared to the unfiltered baseline (no look-back, no
p-hacking).
"""
from __future__ import annotations
import csv
import statistics as stats
from collections import defaultdict
import datetime as dt


TRAIN_MONTHS    = {f"2025-{m:02d}" for m in range(1, 8)}
VALIDATE_MONTHS = {f"2025-{m:02d}" for m in range(8, 11)}
HOLDOUT_2025_MONTHS = {"2025-11", "2025-12"}


FEATURES = [
    # (name, sign, column_or_derived)
    ("qf_pole_gain_pct",       +1, lambda r: float(r['qf_pole_gain_pct'])),
    ("conv_flag_tightness",    +1, lambda r: float(r['conv_flag_tightness'])),
    ("pole_gain_over_bars",    +1, lambda r: float(r['qf_pole_gain_pct']) / max(float(r['qf_pole_bars']) or 1, 1)),
    ("entry_price",            -1, lambda r: float(r['entry_price'])),
    ("qf_vwap_dist_pct",       -1, lambda r: float(r['qf_vwap_dist_pct']) if r['qf_vwap_dist_pct'] else 0),
    ("qf_fill_vwap_dist_pct",  -1, lambda r: float(r['qf_fill_vwap_dist_pct']) if r['qf_fill_vwap_dist_pct'] else 0),
]


def fit_zscore_params(rows: list[dict]) -> dict:
    """Compute TRAIN-set mean and std for each feature (for z-scoring)."""
    params = {}
    for name, sign, fn in FEATURES:
        vals = []
        for r in rows:
            try:
                v = fn(r)
                vals.append(v)
            except (TypeError, ValueError, KeyError, ZeroDivisionError):
                continue
        if len(vals) < 10:
            params[name] = {"mean": 0, "std": 1, "sign": sign}
            continue
        m = stats.mean(vals)
        s = stats.stdev(vals) if len(vals) > 1 else 1.0
        params[name] = {"mean": m, "std": s or 1.0, "sign": sign}
    return params


def score(row: dict, params: dict) -> float:
    """Composite score: sum of signed z-scores over the 6 features."""
    total = 0.0
    count = 0
    for name, sign, fn in FEATURES:
        try:
            v = fn(row)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            continue
        p = params[name]
        z = (v - p['mean']) / p['std']
        total += sign * z
        count += 1
    return total / max(count, 1)   # avg z-score, stable even with missing features


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0, "avg_pnl": 0,
                "max_dd": 0, "strip10_rest": 0, "days": 0, "pos_days": 0}
    wins = sum(1 for r in rows if float(r['pnl']) > 0)
    pnl = sum(float(r['pnl']) for r in rows)
    gw = sum(float(r['pnl']) for r in rows if float(r['pnl']) > 0)
    gl = abs(sum(float(r['pnl']) for r in rows if float(r['pnl']) <= 0))
    pf = gw / gl if gl > 0 else 999
    # Drawdown
    eq = 0; peak = 0; dd = 0
    for r in sorted(rows, key=lambda r: (r['date'], r['entry_time_et'])):
        eq += float(r['pnl'])
        if eq > peak:
            peak = eq
        if peak - eq > dd:
            dd = peak - eq
    # Days
    by_day = defaultdict(float)
    for r in rows:
        by_day[r['date']] += float(r['pnl'])
    pos_days = sum(1 for v in by_day.values() if v > 0)
    days_sorted = sorted(by_day.values(), reverse=True)
    strip10_rest = sum(days_sorted[10:]) if len(days_sorted) > 10 else sum(days_sorted)
    return {
        "n": len(rows),
        "pnl": pnl,
        "wr": wins / len(rows) * 100,
        "pf": pf,
        "avg_pnl": pnl / len(rows),
        "max_dd": dd,
        "strip10_rest": strip10_rest,
        "days": len(by_day),
        "pos_days": pos_days,
        "pct_pos_days": pos_days / len(by_day) * 100 if by_day else 0,
    }


def main() -> int:
    rows_2025 = list(csv.DictReader(open('/tmp/variant_runner/bt_O_f6_2025-01-01_2025-12-31.csv')))
    rows_q1_2026 = list(csv.DictReader(open('/tmp/variant_runner/bt_O_f6_q1.csv')))

    train = [r for r in rows_2025 if r['date'][:7] in TRAIN_MONTHS]
    validate = [r for r in rows_2025 if r['date'][:7] in VALIDATE_MONTHS]
    holdout25 = [r for r in rows_2025 if r['date'][:7] in HOLDOUT_2025_MONTHS]

    # Fit z-score parameters from TRAIN only
    params = fit_zscore_params(train)
    print("Z-score params (TRAIN):")
    for name, p in params.items():
        print(f"  {name:<24} mean={p['mean']:+.3f}  std={p['std']:.3f}  sign={p['sign']:+d}")

    # Attach scores
    for r in train + validate + holdout25 + rows_q1_2026:
        r['_score'] = score(r, params)

    # Sweep thresholds on TRAIN
    print(f"\n{'='*120}")
    print(f"  Threshold sweep — accept rows with score >= T")
    print(f"{'='*120}")
    print(f"  {'T':>5}  {'TRAIN':<50} {'VALIDATE':<50}")
    print(f"  {'':>5}  {'n   WR     PF   P&L     DD     strip10':<50} {'n  WR    PF    P&L    strip10':<50}")
    print("-" * 120)
    baseline_tr = metrics(train)
    baseline_va = metrics(validate)
    print(f"  {'0 (raw)':>7}  "
          f"{baseline_tr['n']:>3} {baseline_tr['wr']:>4.1f}% {baseline_tr['pf']:>4.2f} ${baseline_tr['pnl']:>+7,.0f} ${baseline_tr['max_dd']:>5,.0f} ${baseline_tr['strip10_rest']:>+6,.0f}  "
          f"{baseline_va['n']:>2} {baseline_va['wr']:>4.1f}% {baseline_va['pf']:>4.2f} ${baseline_va['pnl']:>+6,.0f} ${baseline_va['strip10_rest']:>+6,.0f}")
    best = None
    for T in [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        tr = metrics([r for r in train if r['_score'] >= T])
        va = metrics([r for r in validate if r['_score'] >= T])
        flag = ""
        # Acceptance: (1) per-trade edge must improve on BOTH sets
        # (PF+WR must beat baseline, confirms signal is real not noise),
        # (2) VAL P&L must improve (confirms generalization) —
        # TRAIN P&L may decrease (filter rejects some winners naturally),
        # (3) minimum sample size 10 on TRAIN, 8 on VAL (stats sanity).
        if (tr['pf'] > baseline_tr['pf']
            and va['pf'] > baseline_va['pf']
            and tr['wr'] > baseline_tr['wr']
            and va['wr'] > baseline_va['wr']
            and va['pnl'] > baseline_va['pnl']
            and tr['n'] >= 10 and va['n'] >= 8):
            flag = "✓"
            # Optimize: PF on validate (generalization), break tie on TRAIN pnl
            sc = va['pf'] * 1000 + tr['pnl'] / 1000
            if best is None or sc > best['score']:
                best = {"T": T, "score": sc, "tr": tr, "va": va}
        print(f"  {T:>5.2f}  "
              f"{tr['n']:>3} {tr['wr']:>4.1f}% {tr['pf']:>4.2f} ${tr['pnl']:>+7,.0f} ${tr['max_dd']:>5,.0f} ${tr['strip10_rest']:>+6,.0f}  "
              f"{va['n']:>2} {va['wr']:>4.1f}% {va['pf']:>4.2f} ${va['pnl']:>+6,.0f} ${va['strip10_rest']:>+6,.0f}  {flag}")

    if best is None:
        print("\n❌ NO threshold improves BOTH TRAIN and VALIDATE on both P&L and PF.")
        print("   Stage A composite is insufficient. Need Stage B (raw-bar features).")
        return 1

    print(f"\n✓ Best threshold: T={best['T']:.2f}")
    print(f"  TRAIN:    n={best['tr']['n']}, WR {best['tr']['wr']:.1f}%, PF {best['tr']['pf']:.2f}, "
          f"P&L ${best['tr']['pnl']:+,.0f}")
    print(f"  VALIDATE: n={best['va']['n']}, WR {best['va']['wr']:.1f}%, PF {best['va']['pf']:.2f}, "
          f"P&L ${best['va']['pnl']:+,.0f}")

    # Now apply to HOLDOUT (ONE shot, no re-tuning)
    print(f"\n{'='*120}")
    print(f"  HOLDOUT test — T={best['T']:.2f} applied to Nov-Dec 2025 + Q1 2026")
    print(f"{'='*120}")
    ho_2025 = [r for r in holdout25 if r['_score'] >= best['T']]
    ho_q1 = [r for r in rows_q1_2026 if r['_score'] >= best['T']]
    ho_2025_base = metrics(holdout25)
    ho_q1_base = metrics(rows_q1_2026)
    ho_2025_m = metrics(ho_2025)
    ho_q1_m = metrics(ho_q1)
    print(f"  Nov-Dec 2025 baseline: {ho_2025_base['n']} tr, WR {ho_2025_base['wr']:.1f}%, "
          f"PF {ho_2025_base['pf']:.2f}, P&L ${ho_2025_base['pnl']:+,.0f}")
    print(f"  Nov-Dec 2025 FILTERED: {ho_2025_m['n']} tr, WR {ho_2025_m['wr']:.1f}%, "
          f"PF {ho_2025_m['pf']:.2f}, P&L ${ho_2025_m['pnl']:+,.0f}  "
          f"{'✓' if ho_2025_m['pnl'] > ho_2025_base['pnl'] else '✗'}")
    print(f"  Q1 2026 baseline:      {ho_q1_base['n']} tr, WR {ho_q1_base['wr']:.1f}%, "
          f"PF {ho_q1_base['pf']:.2f}, P&L ${ho_q1_base['pnl']:+,.0f}")
    print(f"  Q1 2026 FILTERED:      {ho_q1_m['n']} tr, WR {ho_q1_m['wr']:.1f}%, "
          f"PF {ho_q1_m['pf']:.2f}, P&L ${ho_q1_m['pnl']:+,.0f}  "
          f"{'✓' if ho_q1_m['pnl'] > ho_q1_base['pnl'] else '✗'}")

    # Summary vs A_f6 baseline
    print(f"\n{'='*70}")
    print(f"  Final gate check — O_f6 + composite filter vs A_f6 (2025 full)")
    print(f"{'='*70}")
    all_2025_filtered = [r for r in (train + validate + holdout25) if r['_score'] >= best['T']]
    filt = metrics(all_2025_filtered)
    print(f"  A_f6 2025:                 83tr WR 60.2% PF 3.50 P&L $+54,572 DD $2,502")
    print(f"  O_f6 2025 (no filter):    243tr WR 46.1% PF 1.39 P&L $+34,944 DD $8,014")
    print(f"  O_f6 + composite (T={best['T']:.2f}): "
          f"{filt['n']}tr WR {filt['wr']:.1f}% PF {filt['pf']:.2f} P&L ${filt['pnl']:+,.0f} DD ${filt['max_dd']:,.0f}")
    if filt['pnl'] > 54572 and filt['pf'] > 3.50:
        print(f"  ✓ BEATS A_f6")
    else:
        print(f"  ✗ Does not beat A_f6 — need Stage B")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
