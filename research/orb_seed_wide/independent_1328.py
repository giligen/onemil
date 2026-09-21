#!/usr/bin/env python3
"""Independent rebuild of cell 1328: union of pools."""
import pandas as pd
import numpy as np
from datetime import datetime

# Read CSVs
base_path = "/home/ec2-user/onemil/research/orb_seed_wide/out/"
wide_csv = pd.read_csv(base_path + "orb_features_20260920_2142.csv", keep_default_na=False, na_values=[""])
comb_features = pd.read_csv(base_path + "runCOMB_features.csv", keep_default_na=False, na_values=[""])
prod_book = pd.read_csv(base_path + "runB_true.csv", keep_default_na=False, na_values=[""])
comb_book = pd.read_csv(base_path + "runCOMB_true.csv", keep_default_na=False, na_values=[""])

# Task A: Verify combined seed definition
def meets_comb_criteria(row):
    ep = row['entry_price']
    gp = row['gap_pct']
    if ep < 3 or ep > 50:
        return False
    if ep <= 30 and gp >= 4:
        return True
    if 30 < ep <= 50 and 3 <= gp < 5:
        return True
    return False

wide_seed = wide_csv[wide_csv.apply(meets_comb_criteria, axis=1)].copy()
wide_seed_set = set(zip(wide_seed['symbol'], wide_seed['date']))
comb_features_set = set(zip(comb_features['symbol'], comb_features['date']))

only_wide = wide_seed_set - comb_features_set
only_comb = comb_features_set - wide_seed_set
both = wide_seed_set & comb_features_set

print("Task A: Seed Verification")
print(f"  Wide CSV seed rows: {len(wide_seed_set)}")
print(f"  COMB features rows: {len(comb_features_set)}")
print(f"  Only in wide: {len(only_wide)}")
print(f"  Only in COMB: {len(only_comb)}")
print(f"  In both: {len(both)}")

# Task B: Build UNION book and score
prod_picks = prod_book[prod_book['entered'] == 1].copy()
prod_set = set(zip(prod_picks['date'], prod_picks['symbol']))

# Add-on picks
comb_picks = comb_book[comb_book['entered'] == 1].copy()
addon_picks = comb_picks[~comb_picks.apply(lambda r: (r['date'], r['symbol']) in prod_set, axis=1)].copy()

# Sort by date and _composite descending
addon_picks = addon_picks.sort_values(['date', '_composite'], ascending=[True, False])

# Admit at most 8 - prod_picks_per_date per date
admitted = []
prod_per_date = prod_picks.groupby('date').size()
for date, group in addon_picks.groupby('date'):
    prod_count = prod_per_date.get(date, 0)
    admit_count = min(len(group), 8 - prod_count)
    if admit_count > 0:
        admitted.append(group.head(admit_count))

admitted_picks = pd.concat(admitted, ignore_index=True) if admitted else pd.DataFrame()

union_book = pd.concat([prod_picks, admitted_picks], ignore_index=True).sort_values(['date', '_composite'], ascending=[True, False])

# Assert no date >= 2026-06-01
cutoff = pd.Timestamp('2026-06-01')
assert (pd.to_datetime(union_book['date']) < cutoff).all(), "Found rows >= 2026-06-01"

# Score: separate TRAIN (2025) and VAL (2026-01-01..2026-05-31)
union_book['date_ts'] = pd.to_datetime(union_book['date'])
prod_picks['date_ts'] = pd.to_datetime(prod_picks['date'])
if len(admitted_picks) > 0:
    admitted_picks['date_ts'] = pd.to_datetime(admitted_picks['date'])
else:
    admitted_picks['date_ts'] = pd.NaT

def compute_stats(book, label):
    if len(book) == 0:
        return {
            'n': 0, 'mean_r': 0, 'total_pnl': 0, 'fills_per_week': 0,
            'weekly_max_dd_r': 0
        }

    n = len(book)
    mean_r = book['_sized_pnl'].mean() / 375
    total_pnl = book['_sized_pnl'].sum()

    # Fills per week
    book_tmp = book.copy()
    book_tmp['week'] = book_tmp['date_ts'].dt.isocalendar().week
    book_tmp['year'] = book_tmp['date_ts'].dt.isocalendar().year
    weeks = book_tmp.drop_duplicates(['year', 'week']).shape[0]
    fills_per_week = n / weeks if weeks > 0 else 0

    # Weekly max drawdown
    weekly_pnl = book_tmp.groupby(['year', 'week'])['_sized_pnl'].sum() / 375
    cumsum = weekly_pnl.cumsum()
    runmax = cumsum.expanding().max()
    dd = cumsum - runmax
    weekly_max_dd_r = dd.min()

    return {
        'n': n, 'mean_r': mean_r, 'total_pnl': total_pnl, 'fills_per_week': fills_per_week,
        'weekly_max_dd_r': weekly_max_dd_r
    }

results = {}
for split in ['TRAIN', 'VAL']:
    if split == 'TRAIN':
        prod_split = prod_picks[prod_picks['date_ts'].dt.year == 2025].copy()
        union_split = union_book[union_book['date_ts'].dt.year == 2025].copy()
    else:  # VAL
        prod_split = prod_picks[(prod_picks['date_ts'] >= '2026-01-01') & (prod_picks['date_ts'] <= '2026-05-31')].copy()
        union_split = union_book[(union_book['date_ts'] >= '2026-01-01') & (union_book['date_ts'] <= '2026-05-31')].copy()

    prod_stats = compute_stats(prod_split, f'{split} PROD')
    union_stats = compute_stats(union_split, f'{split} UNION')

    # Admitted rows in this split
    admitted_split = admitted_picks[admitted_picks['date_ts'].dt.year == 2025] if split == 'TRAIN' else admitted_picks[(admitted_picks['date_ts'] >= '2026-01-01') & (admitted_picks['date_ts'] <= '2026-05-31')]
    n_added = len(admitted_split)
    mean_r_added = admitted_split['_sized_pnl'].mean() / 375 if n_added > 0 else 0
    total_pnl_added = admitted_split['_sized_pnl'].sum() if n_added > 0 else 0

    # Mean R after removing top 5%
    if n_added > 0:
        pct_5 = max(1, int(np.ceil(n_added * 0.05)))
        top_5pct_threshold = admitted_split['_sized_pnl'].nlargest(pct_5).min()
        mean_r_no_top5 = admitted_split[admitted_split['_sized_pnl'] < top_5pct_threshold]['_sized_pnl'].mean() / 375
    else:
        mean_r_no_top5 = 0

    results[split] = {
        'prod': prod_stats,
        'union': union_stats,
        'added_n': n_added,
        'added_mean_r': mean_r_added,
        'added_total_pnl': total_pnl_added,
        'added_mean_r_no_top5': mean_r_no_top5
    }

# Union uplift
train_prod_total = results['TRAIN']['prod']['total_pnl']
train_union_total = results['TRAIN']['union']['total_pnl']
val_prod_total = results['VAL']['prod']['total_pnl']
val_union_total = results['VAL']['union']['total_pnl']

train_uplift = train_union_total - train_prod_total
val_uplift = val_union_total - val_prod_total
both_raised = train_uplift > 0 and val_uplift > 0

# Write markdown
md = f"""# Independent 1328: Union of Pools

## Task A — Seed Definition Verification
| Metric | Count |
|--------|-------|
| Wide CSV seed (criteria met) | {len(wide_seed_set)} |
| COMB features (input) | {len(comb_features_set)} |
| Only in wide | {len(only_wide)} |
| Only in COMB | {len(only_comb)} |
| In both | {len(both)} |

## Task B — Union Book Results

### Production vs Union by Split

| Split | Book | n | Mean R | Total $ | Fills/Week | Weekly Max DD (R) |
|-------|------|---|--------|---------|------------|------------------|
| TRAIN | Production | {results['TRAIN']['prod']['n']} | {results['TRAIN']['prod']['mean_r']:.3f} | ${results['TRAIN']['prod']['total_pnl']:.0f} | {results['TRAIN']['prod']['fills_per_week']:.2f} | {results['TRAIN']['prod']['weekly_max_dd_r']:.3f} |
| TRAIN | Union | {results['TRAIN']['union']['n']} | {results['TRAIN']['union']['mean_r']:.3f} | ${results['TRAIN']['union']['total_pnl']:.0f} | {results['TRAIN']['union']['fills_per_week']:.2f} | {results['TRAIN']['union']['weekly_max_dd_r']:.3f} |
| VAL | Production | {results['VAL']['prod']['n']} | {results['VAL']['prod']['mean_r']:.3f} | ${results['VAL']['prod']['total_pnl']:.0f} | {results['VAL']['prod']['fills_per_week']:.2f} | {results['VAL']['prod']['weekly_max_dd_r']:.3f} |
| VAL | Union | {results['VAL']['union']['n']} | {results['VAL']['union']['mean_r']:.3f} | ${results['VAL']['union']['total_pnl']:.0f} | {results['VAL']['union']['fills_per_week']:.2f} | {results['VAL']['union']['weekly_max_dd_r']:.3f} |

### Added Rows (Addon Picks Not in Production)

| Split | n | Mean R | Total $ | Mean R (ex-top 5%) |
|-------|---|--------|---------|-------------------|
| TRAIN | {results['TRAIN']['added_n']} | {results['TRAIN']['added_mean_r']:.3f} | ${results['TRAIN']['added_total_pnl']:.0f} | {results['TRAIN']['added_mean_r_no_top5']:.3f} |
| VAL | {results['VAL']['added_n']} | {results['VAL']['added_mean_r']:.3f} | ${results['VAL']['added_total_pnl']:.0f} | {results['VAL']['added_mean_r_no_top5']:.3f} |

## Summary
Union book raised total $ on {'both' if both_raised else 'one or neither'} splits. TRAIN: ${train_uplift:+.0f}, VAL: ${val_uplift:+.0f}.
"""

with open('/home/ec2-user/onemil/research/orb_seed_wide/INDEPENDENT_1328.md', 'w') as f:
    f.write(md)

print("\nTask B: Union Book Scoring")
print(f"TRAIN - Prod: n={results['TRAIN']['prod']['n']}, Union: n={results['TRAIN']['union']['n']}, Added: {results['TRAIN']['added_n']}")
print(f"VAL   - Prod: n={results['VAL']['prod']['n']}, Union: n={results['VAL']['union']['n']}, Added: {results['VAL']['added_n']}")
print(f"Union total $ TRAIN: ${train_union_total:.0f} vs prod ${train_prod_total:.0f} (uplift ${train_uplift:+.0f})")
print(f"Union total $ VAL:   ${val_union_total:.0f} vs prod ${val_prod_total:.0f} (uplift ${val_uplift:+.0f})")
print(f"Both splits raised: {both_raised}")
print("\nResult written to INDEPENDENT_1328.md")
