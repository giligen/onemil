"""Ignition capcheck — annotate a capsim trades CSV like analyze.py does
(era, monsters, anchor, same-day anchor cohort, complex_conf) so resting_sim
and the live comparison can consume any PART (2026-09-05, roll-forward).

Usage: python3 annotate_trades.py IN_TRADES.csv OUT_ANNOTATED.csv
"""
import csv
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_asset_class import load_class_map, underlying_anchor  # noqa: E402
from trading.orb_csv import read_orb_csv  # noqa: E402


def annotate(t: pd.DataFrame) -> pd.DataFrame:
    """Add era / monster flags / anchor / cohort / complex_conf columns."""
    for c in t.columns:
        if c not in ('day', 'symbol', 'reason'):
            t[c] = pd.to_numeric(t[c], errors='coerce')
    t = t[t['symbol'].astype(str) != ''].reset_index(drop=True)
    t['ym'] = t['day'].str[:7]
    t['era'] = pd.cut(pd.to_datetime(t['day']).astype('int64'),
                      bins=[0, pd.Timestamp('2025-07-01').value, pd.Timestamp('2026-01-01').value,
                            pd.Timestamp('2100-01-01').value], labels=['25H1', '25H2', '2026'])
    t['monster2'] = t['rr'] >= 2.0
    t['monster3'] = t['rr'] >= 3.0
    names = {r['symbol']: r['name'] for r in csv.DictReader(
        open(f'{ROOT}/data/research/orb_asset_class_map_20260711.csv', newline=''))}
    cmap = load_class_map()
    t['anchor'] = t['symbol'].map({s: underlying_anchor(s, names.get(s), cmap) for s in t['symbol'].unique()})
    sz = t[t['anchor'].notna()].groupby(['day', 'anchor']).size().rename('coh')
    t = t.merge(sz, left_on=['day', 'anchor'], right_index=True, how='left')
    t['complex_conf'] = t['coh'].fillna(0) >= 2
    return t


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    t = annotate(read_orb_csv(src, dtype={'symbol': str}))
    t.to_csv(dst, index=False)
    print(f"{src}: {len(t):,} trades | baseline ${t['pnl'].sum():+,.0f} | "
          f"complex-confirmed {int(t['complex_conf'].sum())} -> {dst}")


if __name__ == '__main__':
    main()
