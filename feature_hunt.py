#!/usr/bin/env python3
"""Feature-lift analysis for 10%-threshold bull flag signal quality.

Split:
  TRAIN    = 2025 Jan–Jul (for feature hunt + composite-score tuning)
  VALIDATE = 2025 Aug–Oct  (direction-of-lift must match TRAIN; else scrap)
  HOLDOUT  = 2025 Nov–Dec + Q1 2026 (touched once at end)

Stage A: analyze features already in the cache CSV columns.
Stage B (next script): compute additional features from raw 1-min bars.

Output: train-set feature-lift ranking. For each feature:
  - above_median WR vs below_median WR (lift in pts)
  - above_median avg_pnl vs below_median avg_pnl
  - signs on TRAIN and VALIDATE must agree for feature to proceed
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from statistics import median
from typing import Callable, Iterable


TRAIN_MONTHS    = {f"2025-{m:02d}" for m in range(1, 8)}
VALIDATE_MONTHS = {f"2025-{m:02d}" for m in range(8, 11)}


def load(path: str) -> list[dict]:
    return list(csv.DictReader(open(path)))


def split_2025(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train = [r for r in rows if r['date'][:7] in TRAIN_MONTHS]
    validate = [r for r in rows if r['date'][:7] in VALIDATE_MONTHS]
    holdout = [r for r in rows if r['date'][:7] in {'2025-11', '2025-12'}]
    return train, validate, holdout


def median_split_stats(rows: list[dict], feature: Callable[[dict], float]) -> dict:
    """Split rows into above/below median of feature, return WR and avg_pnl."""
    vals = []
    for r in rows:
        try:
            v = feature(r)
            if v is None:
                continue
            vals.append((v, float(r['pnl'])))
        except (TypeError, ValueError, KeyError):
            continue
    if len(vals) < 20:
        return {"n": len(vals), "error": "too few samples"}
    vals.sort(key=lambda x: x[0])
    med = median(v for v, _ in vals)
    above = [p for v, p in vals if v > med]
    below = [p for v, p in vals if v <= med]
    if not above or not below:
        return {"n": len(vals), "error": "no split"}
    above_wr = sum(1 for p in above if p > 0) / len(above) * 100
    below_wr = sum(1 for p in below if p > 0) / len(below) * 100
    above_avg = sum(above) / len(above)
    below_avg = sum(below) / len(below)
    return {
        "n": len(vals),
        "median": med,
        "above_n": len(above),
        "below_n": len(below),
        "above_wr": above_wr,
        "below_wr": below_wr,
        "wr_lift": above_wr - below_wr,     # + means higher feature → higher WR
        "above_avg_pnl": above_avg,
        "below_avg_pnl": below_avg,
        "pnl_lift": above_avg - below_avg,
    }


def float_or_none(v):
    try:
        return float(v) if v != '' else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Feature definitions (existing cache columns — Stage A)
# ---------------------------------------------------------------------------

FEATURES: list[tuple[str, Callable[[dict], float]]] = [
    ("qf_pole_bars",            lambda r: float_or_none(r.get('qf_pole_bars'))),
    ("qf_pole_gain_pct",        lambda r: float_or_none(r.get('qf_pole_gain_pct'))),
    ("qf_vwap_dist_pct",        lambda r: float_or_none(r.get('qf_vwap_dist_pct'))),
    ("qf_fill_vwap_dist_pct",   lambda r: float_or_none(r.get('qf_fill_vwap_dist_pct'))),
    ("qf_gap_pct",              lambda r: float_or_none(r.get('qf_gap_pct'))),
    ("qf_spy_return_pct",       lambda r: float_or_none(r.get('qf_spy_return_pct'))),
    ("daily_range_pct",         lambda r: float_or_none(r.get('daily_range_pct'))),
    ("avg_volume_20d",          lambda r: float_or_none(r.get('avg_volume_20d'))),
    ("conviction_mult",         lambda r: float_or_none(r.get('conviction_mult'))),
    ("macd_zone_mult",          lambda r: float_or_none(r.get('macd_zone_mult'))),
    ("conv_pole_gain",          lambda r: float_or_none(r.get('conv_pole_gain'))),
    ("conv_flag_tightness",     lambda r: float_or_none(r.get('conv_flag_tightness'))),
    ("conv_vol_ratio",          lambda r: float_or_none(r.get('conv_vol_ratio'))),
    ("conv_spy_regime",         lambda r: float_or_none(r.get('conv_spy_regime'))),
    ("conv_retracement",        lambda r: float_or_none(r.get('conv_retracement'))),
    ("conv_raw_score",          lambda r: float_or_none(r.get('conv_raw_score'))),
    ("conv_vwap_dist",          lambda r: float_or_none(r.get('conv_vwap_dist'))),
    ("conv_gap_fading",         lambda r: float_or_none(r.get('conv_gap_fading'))),
    ("spy_3d_range",            lambda r: float_or_none(r.get('spy_3d_range'))),
    # Derived (still from cache columns, no bar loading needed)
    ("entry_price",             lambda r: float_or_none(r.get('entry_price'))),
    ("pole_gain_over_bars",     lambda r: (
        float_or_none(r.get('qf_pole_gain_pct')) or 0
    ) / max(float_or_none(r.get('qf_pole_bars')) or 1, 1)),
    ("entry_minute_et",         lambda r: (
        int(r['entry_time_et'][:2]) * 60 + int(r['entry_time_et'][3:5])
        if r.get('entry_time_et') else None
    )),
]


def report(tr_rows, va_rows, name):
    print(f"\n{'='*100}")
    print(f"  Feature lift — {name}")
    print(f"{'='*100}")
    print(f"  {'feature':<26} {'TRAIN':<32} {'VALIDATE':<32} {'sign-ok':<8}")
    print(f"  {'':<26} {'WR_lift  pnl_lift  (n=T/B)':<32} {'WR_lift  pnl_lift  (n=T/B)':<32}")
    print("-" * 100)
    results = []
    for fname, fn in FEATURES:
        tr = median_split_stats(tr_rows, fn)
        va = median_split_stats(va_rows, fn)
        if "error" in tr or "error" in va:
            continue
        sign_ok = ((tr['wr_lift'] > 0) == (va['wr_lift'] > 0))
        # We want same sign AND same direction on pnl_lift
        sign_ok_pnl = ((tr['pnl_lift'] > 0) == (va['pnl_lift'] > 0))
        ok = "✓" if sign_ok and sign_ok_pnl else ("≈" if sign_ok or sign_ok_pnl else "✗")
        print(f"  {fname:<26} "
              f"{tr['wr_lift']:>+6.1f}pt ${tr['pnl_lift']:>+7.0f} ({tr['above_n']}/{tr['below_n']})  "
              f"{va['wr_lift']:>+6.1f}pt ${va['pnl_lift']:>+7.0f} ({va['above_n']}/{va['below_n']})  "
              f"{ok}")
        results.append({
            "feature": fname, "tr": tr, "va": va,
            "sign_ok": sign_ok, "sign_ok_pnl": sign_ok_pnl,
        })
    # Rank by abs(TRAIN wr_lift) but only among sign-ok features
    print("\n=== Top features by |TRAIN wr_lift|, sign-consistent on VALIDATE ===")
    consistent = [r for r in results if r['sign_ok'] and r['sign_ok_pnl']]
    consistent.sort(key=lambda r: abs(r['tr']['wr_lift']), reverse=True)
    for r in consistent[:10]:
        print(f"  {r['feature']:<26} TRAIN lift {r['tr']['wr_lift']:>+5.1f}pt / "
              f"${r['tr']['pnl_lift']:>+6.0f}   "
              f"VAL lift {r['va']['wr_lift']:>+5.1f}pt / ${r['va']['pnl_lift']:>+6.0f}")


def main() -> int:
    rows = load('/tmp/variant_runner/bt_O_f6_2025-01-01_2025-12-31.csv')
    train, validate, holdout = split_2025(rows)
    print(f"TRAIN {len(train)} trades / VALIDATE {len(validate)} / HOLDOUT 2025 {len(holdout)}")
    report(train, validate, "Stage A (existing cache columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
