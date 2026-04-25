"""Stage 2.2 — Combined filter test: loose detector + Q4/Q5 + $5-$20 price.

Stage 2.1 stratification on TRAIN+VAL (n=72 fires, loose detector) found:
  - $10-$20 price bucket: 48% WR, rm3 = -$76 (tail-robust)
  - Q4 quintile: 46% WR, rm3 = -$705
  - <$5 stocks: 13% WR (avoid)
  - Q3/Q2: <20% WR (avoid)

This script tests the combined filter:
  loose detector  +  quintile in {Q4, Q5}  +  entry_price in [5, 20]

OOS validation: HOQ1+ slice was held back from stratification — this is the
blind test. If HOQ1+ shows positive lift with positive top-3 removal, we
have a hypothesis-stage candidate (still small N, would need paper validation).

Reads the saved CSV from study_orb_add_bullflag_segmented.py rather than
re-running the BT (cheaper).
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


TRAIN_END = '2025-06-30'
VAL_END = '2025-12-31'


def _calmar(pnls):
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v); mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


def evaluate_filter(rdf: pd.DataFrame, filter_label: str, mask):
    """Apply mask to fires, report walk-forward + tail-dep."""
    rdf = rdf.copy()
    rdf['date'] = pd.to_datetime(rdf['date'])
    fires_all = rdf[rdf['flag_detected']].copy()
    fires = fires_all[mask(fires_all)].copy() if len(fires_all) else fires_all
    print(f"\n{'='*100}")
    print(f"  FILTER: {filter_label}")
    print(f"{'='*100}")
    if len(fires) == 0:
        print("  (no fires after filter)")
        return
    print(f"  Total fires after filter: {len(fires)} of {len(fires_all)}  "
          f"({len(fires)/len(fires_all)*100:.1f}% pass-through)")
    overall_wr = fires['add_was_winner'].mean() * 100
    print(f"  Overall WR: {overall_wr:.1f}%")

    slices = [
        ('TRAIN',  '2025-01-01', TRAIN_END),
        ('VAL',    '2025-07-01', VAL_END),
        ('HOQ1+',  '2026-01-01', '2030-12-31'),
        ('FULL',   '2025-01-01', '2030-12-31'),
    ]
    print(f"\n  {'Slice':<7} {'Fires':>6} {'WR':>6} {'Sum':>10} {'rm3':>10} "
          f"{'rm5':>10} {'avg/fire':>10} {'Combined Δ vs V0':>18}")
    print('  ' + '-' * 90)
    for slice_name, lo, hi in slices:
        sub = fires[(fires['date'] >= lo) & (fires['date'] <= hi)]
        sub_all = rdf[(rdf['date'] >= lo) & (rdf['date'] <= hi)]
        if len(sub) == 0:
            print(f"  {slice_name:<7} {0:>6}   --     --        --        --        --")
            continue
        n = len(sub)
        wr = sub['add_was_winner'].mean() * 100
        s = sub['add_sized_pnl'].sum()
        sorted_fires = sub.sort_values('add_sized_pnl', ascending=False)
        rm3 = sorted_fires.iloc[3:]['add_sized_pnl'].sum() if n >= 3 else 0
        rm5 = sorted_fires.iloc[5:]['add_sized_pnl'].sum() if n >= 5 else 0
        avg = sub['add_sized_pnl'].mean()
        # Combined effect with V0: include ALL trades' V0 + only filtered fires' add
        v0_total = sub_all['v0_sized_pnl'].sum()
        combined_total = v0_total + s
        delta = combined_total - v0_total  # = s, but kept for clarity
        print(f"  {slice_name:<7} {n:>6} {wr:>5.1f}% ${s:>+8,.0f} ${rm3:>+8,.0f} "
              f"${rm5:>+8,.0f} ${avg:>+8,.0f} ${delta:>+15,.0f}")


def main():
    # Load latest segmented CSV (loose detector)
    candidates = sorted(glob.glob('analysis_results/orb_add_bullflag_seg_loose_*.csv'))
    if not candidates:
        print("ERROR: no loose-detector CSV found. Run study_orb_add_bullflag_segmented.py first.")
        return
    csv = candidates[-1]
    print(f"Loading {csv}")
    rdf = pd.read_csv(csv)
    rdf['date'] = pd.to_datetime(rdf['date'])
    print(f"Total trades: {len(rdf)}")
    print(f"Total fires (any quintile, any price): {int(rdf['flag_detected'].sum())}")

    # Filter 1: Q4/Q5 only
    evaluate_filter(rdf, "Q4 or Q5",
                     lambda f: f['quintile'].isin(['Q4', 'Q5']))

    # Filter 2: $5-$20 price only
    evaluate_filter(rdf, "$5 <= price < $20",
                     lambda f: (f['entry_price'] >= 5) & (f['entry_price'] < 20))

    # Filter 3: $10-$20 only (best stratum)
    evaluate_filter(rdf, "$10 <= price < $20",
                     lambda f: (f['entry_price'] >= 10) & (f['entry_price'] < 20))

    # Filter 4: combined Q4/Q5 + $5-$20
    evaluate_filter(rdf, "Q4/Q5 AND $5 <= price < $20",
                     lambda f: f['quintile'].isin(['Q4', 'Q5']) &
                                (f['entry_price'] >= 5) & (f['entry_price'] < 20))

    # Filter 5: tightest — Q4/Q5 + $10-$20
    evaluate_filter(rdf, "Q4/Q5 AND $10 <= price < $20",
                     lambda f: f['quintile'].isin(['Q4', 'Q5']) &
                                (f['entry_price'] >= 10) & (f['entry_price'] < 20))

    # Filter 6: Q4 only + $5-$20
    evaluate_filter(rdf, "Q4 only AND $5 <= price < $20",
                     lambda f: (f['quintile'] == 'Q4') &
                                (f['entry_price'] >= 5) & (f['entry_price'] < 20))

    # Filter 7: BASELINE — all fires for comparison
    evaluate_filter(rdf, "BASELINE — all loose fires (no segment filter)",
                     lambda f: pd.Series([True] * len(f), index=f.index))


if __name__ == '__main__':
    main()
