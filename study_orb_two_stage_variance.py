"""Variance/stability analysis: PROD vs two-stage variants.

Beyond total P&L, compare:
- Weekly P&L distribution (mean, std, % weeks positive)
- Monthly P&L distribution (% months positive)
- Largest single-day loss
- Max consecutive losing days
- Sharpe ratio (annualized weekly)
- Sortino (downside-only deviation)
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import glob
from datetime import timedelta
from persistence.database import Database
from study_orb import _bars_to_df

# Reuse the pipeline
from study_orb_two_stage import run_pipeline, split_pnls

# Top-4 candidates to compare for stability (PROD + 3 strongest contenders).
# Picked from prior R-grid + two-stage results — HOLDOUT-aware here, but
# stability metrics are independent of P&L magnitude so we focus on
# distribution shape rather than absolute return.
VARIANTS = [
    ('PROD (1.5R / 1.0R)',           0.00, 1.50, 1.00),
    ('R-grid winner (1.75/0.5)',     0.00, 1.75, 0.50),
    ('2stg best-HOLDOUT (0.75/1.5/1.0)', 0.75, 1.50, 1.00),
    ('2stg balanced (0.75/1.5/0.5)', 0.75, 1.50, 0.50),
]


def _stability(sel: pd.DataFrame, period: str = 'W'):
    """Compute weekly or monthly stability metrics."""
    sel = sel.copy()
    sel['date'] = pd.to_datetime(sel['date'])
    daily = sel.groupby(sel['date'].dt.date)['_sized_pnl'].sum()
    daily.index = pd.to_datetime(daily.index)
    if period == 'W':
        agg = daily.resample('W').sum()
    elif period == 'M':
        agg = daily.resample('ME').sum()
    else:
        agg = daily
    if len(agg) < 2:
        return {}
    mean = agg.mean(); std = agg.std()
    sharpe = mean / std * (52 ** 0.5) if std > 0 else 0  # annualized weekly
    pos = (agg > 0).sum() / len(agg) * 100
    neg = (agg < 0).sum() / len(agg) * 100
    flat = (agg == 0).sum() / len(agg) * 100
    # largest single-period loss
    worst = agg.min()
    best = agg.max()
    # max consecutive negatives
    max_consec_neg = 0; cur = 0
    for v in agg:
        if v < 0:
            cur += 1
            max_consec_neg = max(max_consec_neg, cur)
        else:
            cur = 0
    # downside-only std (Sortino)
    downside = agg[agg < 0]
    downside_std = downside.std() if len(downside) > 1 else (downside.iloc[0] if len(downside) else 0)
    sortino = mean / abs(downside_std) * (52 ** 0.5) if downside_std and abs(downside_std) > 0 else 0
    return {
        'n_periods': len(agg),
        'mean': mean,
        'std': std,
        'sharpe': sharpe,
        'sortino': sortino,
        'pct_positive': pos,
        'pct_negative': neg,
        'pct_flat': flat,
        'worst': worst,
        'best': best,
        'max_consec_neg': max_consec_neg,
    }


def main():
    csv_path = sorted(
        p for p in glob.glob('analysis_results/orb_features_*.csv')
        if 'corrmatrix' not in p
    )[-1]
    print(f'Reading: {csv_path}')
    from study_orb_filter import FILTER_FEATURES
    df = pd.read_csv(csv_path)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # For each variant, compute HOLDOUT-period stability metrics
    print('\nHOLDOUT (Jan-Apr 2026) stability comparison:')
    print('=' * 130)
    print(f'{"Variant":<28} | {"P&L":>10} | {"WeekMean":>9} {"WeekStd":>9} {"Sharpe":>8} {"Sortino":>8} | '
          f'{"%Pos":>6} {"%Neg":>6} | {"Worst":>10} {"Best":>10} {"MaxNeg":>7}')
    print('-' * 130)

    rows = []
    for label, be, arm, lock in VARIANTS:
        sel = run_pipeline(df, bars_cache, be_arm_r=be, arm_r=arm, lock_r=lock)
        if len(sel) == 0:
            continue
        # HOLDOUT only
        sel = sel.copy()
        sel['date'] = pd.to_datetime(sel['date'])
        sel = sel[(sel['date'] >= '2026-01-01') & (sel['date'] <= '2026-04-30')]
        if len(sel) == 0:
            continue
        total_pnl = sel['_sized_pnl'].sum()
        w = _stability(sel, 'W')
        m = _stability(sel, 'M')
        if not w:
            continue
        rows.append((label, total_pnl, w, m))
        print(f'{label:<28} | ${total_pnl:>+9,.0f} | '
              f'${w["mean"]:>+8,.0f} ${w["std"]:>8,.0f} {w["sharpe"]:>8.2f} {w["sortino"]:>8.2f} | '
              f'{w["pct_positive"]:>5.1f}% {w["pct_negative"]:>5.1f}% | '
              f'${w["worst"]:>+9,.0f} ${w["best"]:>+9,.0f} {w["max_consec_neg"]:>7}')

    print('\n\nMonthly distribution (4-month HOLDOUT):')
    print('=' * 100)
    print(f'{"Variant":<28} | {"%MoPos":>7} {"WorstMo":>10} {"BestMo":>10} {"MaxConsecNegMo":>15}')
    print('-' * 100)
    for label, total, w, m in rows:
        if not m:
            continue
        print(f'{label:<28} | {m["pct_positive"]:>6.1f}% '
              f'${m["worst"]:>+9,.0f} ${m["best"]:>+9,.0f}  {m["max_consec_neg"]:>13}')


if __name__ == '__main__':
    main()
