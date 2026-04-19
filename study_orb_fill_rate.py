"""Measure ACTUAL fill rate on top-4 ORB picks — the BT's blind spot.

The validated defended BT (`study_orb_100k_defended.py`) operates on a features
CSV that contains ONLY trades where entry triggered. This implicitly assumes
100% fill rate — every top-4 pick has a P&L because every pick is
pre-filtered to the entered subset.

In PROD, we pick top-4 at 9:35 BEFORE knowing which will trigger. Some never
fire (price doesn't cross range_high in 60 min) → 0 P&L, time-stop cancel at
10:35 ET.

This script:
  1. Extract features at 9:35 for the FULL universe (entered AND non-entered).
  2. Simulate entry per pair (entered flag).
  3. Apply the defended pipeline (composite filter → Q4-preferred rank → dedup → cap 4).
  4. Per-day: count n_placed (top-4) and n_filled (entered=True subset of top-4).
  5. Report fill-rate distribution + P&L degradation vs BT's 100%-fill assumption.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df, simulate_orb_trade, CACHE_DB
from study_orb_broad import load_broad_universe
from study_orb_features import (
    extract_features, load_daily_bars_frame, load_spy_intraday,
)
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


MAX_KEEP = 4                  # top-K per day
FILTER_THRESHOLD_VAL = 0.0    # composite z cutoff
ACCOUNT = 100_000
N_MAX = 4
RISK = 3000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

TRAIN_START, TRAIN_END = '2025-01-01', '2025-06-30'  # H1 2025 z-fit window


# -----------------------------------------------------------------------------
# Build full-universe feature DataFrame (entered AND non-entered)
# -----------------------------------------------------------------------------

def build_full_feature_df() -> pd.DataFrame:
    """Compute features at 9:35 for EVERY (sym, date) in broad universe.
    Returns DataFrame with `entered` column (True/False)."""
    print("Loading broad universe...")
    universe = load_broad_universe()
    n_pairs = sum(len(v) for v in universe.values())
    print(f"  {n_pairs:,} (symbol, date) pairs across {len(universe)} days")

    print("Loading daily bars + SPY context...")
    daily = load_daily_bars_frame()
    daily_by_sym: Dict[str, pd.DataFrame] = {
        s: g.reset_index(drop=True) for s, g in daily.groupby('symbol')
    }
    spy_daily = daily.loc[daily['symbol'] == 'SPY'].reset_index(drop=True)
    spy_intraday = load_spy_intraday()

    print(f"Bulk-loading 1-min bars for {n_pairs:,} pairs...")
    t0 = datetime.now()
    db = Database(db_path=CACHE_DB)
    pair_list: List[Tuple[str, str]] = [
        (s, d) for d, syms in universe.items() for s in syms
    ]
    raw = db.get_intraday_bars_bulk(pair_list)
    db.close()
    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {
        k: _bars_to_df(v) for k, v in raw.items()
    }
    print(f"  Loaded {len(bars_cache):,} in {(datetime.now()-t0).total_seconds():.0f}s")

    print("Computing features + simulating entry per pair...")
    rows = []
    n_total = 0
    n_feats = 0
    n_entered = 0
    for date_str, syms in sorted(universe.items()):
        for sym in syms:
            n_total += 1
            bars = bars_cache.get((sym, date_str))
            if bars is None or bars.empty:
                continue
            feats = extract_features(bars, sym, date_str, daily_by_sym, spy_intraday, spy_daily)
            if feats is None:
                continue
            n_feats += 1
            trade = simulate_orb_trade(
                bars, sym, date_str, 'orb_fill_rate',
                range_minutes=5, entry_mode='touch', stop_mode='range_low',
                target_mult=2.0, time_stop_minutes=60,
            )
            if trade.entered:
                n_entered += 1
            rows.append({
                'symbol': sym, 'date': date_str,
                'entered': bool(trade.entered),
                'range_size_pct_of_entry': (
                    (trade.range_high - trade.range_low) / trade.range_high * 100.0
                    if trade.entered and trade.range_high > 0 else feats.get('range_size_pct', 0)
                ),
                'entry_price': trade.entry_price if trade.entered else 0.0,
                'pnl': trade.pnl if trade.entered else 0.0,
                'pnl_pct': trade.pnl_pct if trade.entered else 0.0,
                **feats,
            })
    print(f"  Processed: {n_total:,}  Features OK: {n_feats:,}  Entered: {n_entered:,} "
          f"({n_entered/max(n_feats,1)*100:.1f}%)")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Apply defended pipeline and measure fill rate
# -----------------------------------------------------------------------------

def apply_pipeline_and_measure(full_df: pd.DataFrame) -> Dict:
    """Run composite filter → Q4-pref rank → dedup → top-K. Measure fill rate."""
    full_df = full_df.copy()
    full_df['date'] = pd.to_datetime(full_df['date'])

    # Fit z-score params on H1 2025 training slice
    train = full_df[(full_df['date'] >= TRAIN_START) & (full_df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    full_df['composite'] = composite_score(full_df, params)

    # Keep candidates passing composite threshold
    kept = full_df[full_df['composite'] >= FILTER_THRESHOLD_VAL].copy()

    # Fit quintile cutoffs on train_kept (same as BT pipeline)
    train_kept = kept[(kept['date'] >= TRAIN_START) & (kept['date'] <= TRAIN_END)].copy()
    cutoffs = fit_quintile_cutoffs(train_kept['composite'])
    kept['quintile'] = assign_quintile(kept['composite'], cutoffs)
    kept['_q_rank'] = kept['quintile'].map(Q_ORDER)

    # Per-day rank + dedup + top-K
    day_stats = []
    for date, grp in kept.groupby('date'):
        grp = grp.sort_values(['_q_rank', 'composite'], ascending=[True, False])
        seen_fam = set()
        seen_sup = set()
        picked = []
        for _, r in grp.iterrows():
            fam = symbol_family(r['symbol'])
            sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam:
                continue
            if sup and sup in seen_sup:
                continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            picked.append(r)
            if len(picked) >= MAX_KEEP:
                break
        if not picked:
            continue
        picked_df = pd.DataFrame(picked)
        n_placed = len(picked_df)
        n_filled = int(picked_df['entered'].sum())
        fill_rate = n_filled / n_placed if n_placed else 0.0
        day_stats.append({
            'date': date,
            'n_placed': n_placed,
            'n_filled': n_filled,
            'fill_rate': fill_rate,
            'total_pnl_filled_only': float(picked_df.loc[picked_df['entered'], 'pnl'].sum()),
            'quintile_mix': picked_df['quintile'].value_counts().to_dict(),
        })
    return {
        'day_stats': pd.DataFrame(day_stats),
        'full_kept': kept,
        'cutoffs': cutoffs,
        'z_params': params,
    }


def main():
    t0 = datetime.now()
    print(f"[{t0.isoformat(timespec='seconds')}] ORB fill-rate backtest")
    print("=" * 78)

    full = build_full_feature_df()
    print(f"\nFull feature DataFrame: {len(full):,} rows "
          f"({full['entered'].sum():,} entered / {len(full)-full['entered'].sum():,} not-entered)")

    result = apply_pipeline_and_measure(full)
    day_stats = result['day_stats']

    print(f"\n{'='*78}")
    print("FILL RATE DISTRIBUTION (top-{MAX_KEEP} picks per day)".replace('{MAX_KEEP}', str(MAX_KEEP)))
    print(f"{'='*78}")
    if len(day_stats) == 0:
        print("No days with picks — check filter threshold or data.")
        return

    print(f"Total trading days with ≥1 pick: {len(day_stats)}")
    print(f"Total orders placed (top-K picks): {day_stats['n_placed'].sum():,}")
    print(f"Total orders filled:               {day_stats['n_filled'].sum():,}")
    overall_fill = day_stats['n_filled'].sum() / day_stats['n_placed'].sum()
    print(f"Overall fill rate:                 {overall_fill*100:.1f}%")
    print()
    print(f"  Daily fill-rate distribution:")
    print(f"    min:     {day_stats['fill_rate'].min()*100:5.1f}%")
    print(f"    p10:     {day_stats['fill_rate'].quantile(0.10)*100:5.1f}%")
    print(f"    p25:     {day_stats['fill_rate'].quantile(0.25)*100:5.1f}%")
    print(f"    median:  {day_stats['fill_rate'].median()*100:5.1f}%")
    print(f"    p75:     {day_stats['fill_rate'].quantile(0.75)*100:5.1f}%")
    print(f"    p90:     {day_stats['fill_rate'].quantile(0.90)*100:5.1f}%")
    print(f"    max:     {day_stats['fill_rate'].max()*100:5.1f}%")
    print()
    zero_fill_days = (day_stats['n_filled'] == 0).sum()
    all_fill_days = (day_stats['n_filled'] == day_stats['n_placed']).sum()
    print(f"  Days with 0% fill rate: {zero_fill_days} ({zero_fill_days/len(day_stats)*100:.1f}%)")
    print(f"  Days with 100% fill rate: {all_fill_days} ({all_fill_days/len(day_stats)*100:.1f}%)")
    print()
    # Distribution of fill counts given n_placed
    print(f"  Fills per 'full-day' (n_placed = {MAX_KEEP}):")
    full_days = day_stats[day_stats['n_placed'] == MAX_KEEP]
    if len(full_days) > 0:
        for n_f in sorted(full_days['n_filled'].unique()):
            cnt = (full_days['n_filled'] == n_f).sum()
            print(f"    {n_f}/{MAX_KEEP} fills: {cnt} days ({cnt/len(full_days)*100:.0f}%)")

    # Detailed worst days (by fill rate)
    print(f"\n{'='*78}")
    print("10 WORST FILL-RATE DAYS (not counting 0-placement days)")
    print(f"{'='*78}")
    worst = day_stats.nsmallest(10, 'fill_rate')[['date', 'n_placed', 'n_filled', 'fill_rate']]
    for _, r in worst.iterrows():
        print(f"  {r['date'].date()}  placed={int(r['n_placed'])}  "
              f"filled={int(r['n_filled'])}  "
              f"rate={r['fill_rate']*100:5.1f}%")

    # P&L degradation vs BT
    print(f"\n{'='*78}")
    print("P&L IMPLICATIONS")
    print(f"{'='*78}")
    bt_pnl_if_all_fill = float(result['full_kept'].loc[result['full_kept']['entered'], 'pnl'].sum())
    # Approximate BT P&L by summing pnl of the top-K picks (if filled)
    # We already computed this per day in day_stats['total_pnl_filled_only']
    prod_pnl = day_stats['total_pnl_filled_only'].sum()
    # Approximate "BT-assumed" P&L = sum of pnl for top-K picks REGARDLESS of entry
    # (treating non-entered picks as if they'd been filled at range_high, 0 P&L since exit=stop immediately)
    # Actually since they didn't enter at all, their true P&L = 0. So prod_pnl IS the correct number.
    # Still — the BT pipeline implicitly picks from ENTERED-ONLY subset, so its top-K is DIFFERENT.
    # Compute BT-style P&L: same pipeline on entered-only subset.
    entered_df = result['full_kept'][result['full_kept']['entered']].copy()
    entered_df = entered_df.sort_values(['date', '_q_rank', 'composite'], ascending=[True, True, False])
    bt_total = 0.0
    bt_n = 0
    for date, grp in entered_df.groupby('date'):
        seen_fam = set()
        seen_sup = set()
        picked = []
        for _, r in grp.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            picked.append(r)
            if len(picked) >= MAX_KEEP:
                break
        bt_total += sum(r['pnl'] for r in picked)
        bt_n += len(picked)
    print(f"  BT-style P&L (picks from entered-only; implicit 100% fill): ${bt_total:+,.0f}  ({bt_n} trades)")
    print(f"  PROD-style P&L (picks from full; actual fills only):        ${prod_pnl:+,.0f}  "
          f"({int(day_stats['n_filled'].sum())} trades)")
    if bt_total != 0:
        print(f"  Ratio: {prod_pnl/bt_total*100:.1f}% of BT projection")

    # Save per-day stats
    os.makedirs('analysis_results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    day_stats.to_csv(f'analysis_results/orb_fill_rate_daily_{ts}.csv', index=False)
    print(f"\nDaily stats saved: analysis_results/orb_fill_rate_daily_{ts}.csv")
    print(f"Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == '__main__':
    main()
