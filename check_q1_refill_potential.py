"""Check whether the Q1 filter leaves alpha on the table by failing to refill slots
with unused Q2/Q3/Q4/Q5 candidates blocked by dedup or threshold.

For every selection day in the V0 BT, build the FULL pool of candidates that
passed the composite threshold, simulate the dedup+top-4 selection both with
and without Q1, and count:
  - Days where Q1 filter caused a slot to drop (4→3 etc)
  - On those days: how many Q2-Q5 candidates passed threshold but weren't picked,
    and WHY (slot full vs dedup-blocked)
  - Whether any of those Q2-Q5 unused-candidates would have refilled the slot
    if Q1 was removed (i.e., did dedup change at all)
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
    ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


N = 4
ACCOUNT = 100_000.0
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
EXIT_SLIP_BPS = 10.0
TRAIN_END = '2025-06-30'
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def select_top4(day_df, exclude_q1: bool):
    """Mirror study_orb_pipeline_static_lock.py selection. Returns list of dicts:
    [{'symbol', 'quintile', 'composite', 'family', 'sgroup', 'picked', 'reject_reason'}].
    """
    d = day_df.copy()
    if exclude_q1:
        d = d[d['_quintile'] != 'Q1']
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    seen_fam = set(); seen_sup = set()
    out = []
    picked_count = 0
    for _, r in d.iterrows():
        fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
        rec = {'symbol': r['symbol'], 'quintile': r['_quintile'],
               'composite': r['_composite'], 'family': fam, 'sgroup': sup,
               'picked': False, 'reject_reason': None}
        if picked_count >= N:
            rec['reject_reason'] = 'slot_full'
        elif fam and fam in seen_fam:
            rec['reject_reason'] = f'dedup_family:{fam}'
        elif sup and sup in seen_sup:
            rec['reject_reason'] = f'dedup_supergroup:{sup}'
        else:
            rec['picked'] = True
            picked_count += 1
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
        out.append(rec)
    return out


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct',
                                     'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    # Standard pipeline scaffolding to compute composite + quintile
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train_k = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END) &
                  (df['_composite'] >= FILTER_THRESHOLD)]
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)

    # Per-day analysis
    days_data = []
    for day, dg in kept.groupby('date'):
        v0 = select_top4(dg, exclude_q1=False)
        v0noq1 = select_top4(dg, exclude_q1=True)
        v0_picked = sum(1 for r in v0 if r['picked'])
        v0noq1_picked = sum(1 for r in v0noq1 if r['picked'])
        # Which Q1 trades did V0 pick?
        v0_q1_picks = [r for r in v0 if r['picked'] and r['quintile'] == 'Q1']
        # In V0, count rejections of non-Q1 candidates
        v0_non_q1_rejected = [r for r in v0
                               if not r['picked'] and r['quintile'] != 'Q1']
        # In V0_noQ1, count rejections of non-Q1 candidates
        v0noq1_non_q1_rejected = [r for r in v0noq1
                                   if not r['picked'] and r['quintile'] != 'Q1']
        # Did Q1 removal cause a slot to drop?
        slot_dropped = v0_picked > v0noq1_picked
        days_data.append({
            'date': day,
            'kept_count': len(dg),
            'q1_count_in_pool': int((dg['_quintile'] == 'Q1').sum()),
            'q2plus_count_in_pool': int((dg['_quintile'] != 'Q1').sum()),
            'v0_picked': v0_picked,
            'v0noq1_picked': v0noq1_picked,
            'slot_dropped': slot_dropped,
            'q1_picks_in_v0': len(v0_q1_picks),
            'q2plus_unused_in_v0': len(v0_non_q1_rejected),
            'q2plus_unused_in_v0noq1': len(v0noq1_non_q1_rejected),
            # Of non-Q1 candidates rejected in V0_noQ1, why?
            'q2plus_dedup_blocked': sum(1 for r in v0noq1_non_q1_rejected
                                          if r['reject_reason'] and 'dedup' in r['reject_reason']),
            'q2plus_slot_full': sum(1 for r in v0noq1_non_q1_rejected
                                      if r['reject_reason'] == 'slot_full'),
        })

    dd = pd.DataFrame(days_data)
    print(f"\nTotal trading days analyzed: {len(dd)}")
    print(f"Days where V0 picked at least 1 Q1 trade: "
          f"{(dd['q1_picks_in_v0'] > 0).sum()}")
    print(f"Days where Q1 filter dropped a slot: {dd['slot_dropped'].sum()}")
    print(f"Days where Q1 filter did NOT drop a slot: {(~dd['slot_dropped']).sum()}"
          f" (Q1 wasn't picked in V0, OR refill happened)")

    # The critical check: on slot-dropped days, were there MORE non-Q1 candidates
    # in the pool that didn't get picked?
    drop_days = dd[dd['slot_dropped']].copy()
    print(f"\n{'='*80}")
    print(f"  SLOT-DROPPED DAYS — could Q2+ candidates have refilled?")
    print(f"{'='*80}")
    print(f"  Total slot-dropped days: {len(drop_days)}")
    print(f"  Avg Q2-Q5 unused in V0_noQ1 on these days: "
          f"{drop_days['q2plus_unused_in_v0noq1'].mean():.2f}")
    print(f"  Days with >=1 Q2-Q5 candidate unused after V0_noQ1: "
          f"{(drop_days['q2plus_unused_in_v0noq1'] >= 1).sum()}")
    print(f"  Of those unused: dedup-blocked vs slot-full breakdown")
    print(f"    Dedup-blocked Q2-Q5: total {drop_days['q2plus_dedup_blocked'].sum()}")
    print(f"    Slot-full Q2-Q5:     total {drop_days['q2plus_slot_full'].sum()}")

    # Show sample slot-dropped days with their unused non-Q1 candidates
    print(f"\n  Sample 10 slot-dropped days with Q2+ candidates left over:")
    sample = drop_days[drop_days['q2plus_unused_in_v0noq1'] >= 1].head(10)
    if len(sample) > 0:
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', '{:,.0f}'.format)
        print(sample[['date', 'kept_count', 'q1_picks_in_v0', 'v0_picked',
                       'v0noq1_picked', 'q2plus_unused_in_v0noq1',
                       'q2plus_dedup_blocked', 'q2plus_slot_full']].to_string(index=False))

    # And — are there any days where V0_noQ1 has MORE picks than V0?
    # (i.e., refill DID happen via dedup change)
    refill_days = dd[dd['v0noq1_picked'] > dd['v0_picked']]
    print(f"\n{'='*80}")
    print(f"  Days where V0_noQ1 picked MORE than V0 (refill happened): "
          f"{len(refill_days)}")
    print(f"{'='*80}")
    if len(refill_days) > 0:
        print(refill_days[['date', 'v0_picked', 'v0noq1_picked',
                            'q1_picks_in_v0', 'q2plus_unused_in_v0',
                            'q2plus_unused_in_v0noq1']].to_string(index=False))

    # Drill into a slot-dropped day to understand what's actually happening
    print(f"\n{'='*80}")
    print(f"  DRILL-DOWN — first slot-dropped day with Q2+ candidates unused")
    print(f"{'='*80}")
    if len(sample) > 0:
        first_day = sample.iloc[0]['date']
        day_pool = kept[kept['date'] == first_day]
        v0_sel = select_top4(day_pool, exclude_q1=False)
        v0nq_sel = select_top4(day_pool, exclude_q1=True)
        print(f"Date: {first_day}")
        print(f"\nV0 selection (with Q1):")
        for r in v0_sel:
            mark = '✓ PICKED' if r['picked'] else f'  {r["reject_reason"]}'
            print(f"  {r['symbol']:<8} {r['quintile']:<3} comp={r['composite']:.3f} "
                  f"fam={r['family']!r:<20} sg={r['sgroup']!r:<15} {mark}")
        print(f"\nV0_noQ1 selection (without Q1):")
        for r in v0nq_sel:
            mark = '✓ PICKED' if r['picked'] else f'  {r["reject_reason"]}'
            print(f"  {r['symbol']:<8} {r['quintile']:<3} comp={r['composite']:.3f} "
                  f"fam={r['family']!r:<20} sg={r['sgroup']!r:<15} {mark}")


if __name__ == '__main__':
    main()
