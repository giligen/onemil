"""Cell 1,684 finalize: load every pool/production book, score, union, cadence-report, apply the
PREREG pass bar, write 1684_pool_books.csv and print the full material for RESULT_1684.md.
(Loads 1684_score.py by path with importlib since a module name can't start with a digit.)
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'

spec = importlib.util.spec_from_file_location('score1684', OUT / '1684_score.py')
score1684 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score1684)
load_book, stats, fmt = score1684.load_book, score1684.stats, score1684.fmt
union_book, cadence_report = score1684.union_book, score1684.cadence_report
from datetime import date  # noqa: E402

IN_LO, IN_HI = date(2025, 1, 1), date(2026, 9, 18)
OUT_LO, OUT_HI = date(2024, 7, 1), date(2024, 12, 31)

BOOKS = {
    ('in_regime', 'prod'): OUT / 'fastpath/prod_true.csv',
    ('in_regime', 'idea1'): OUT / 'fastpath/idea1_true.csv',
    ('in_regime', 'idea2'): OUT / 'fastpath/idea2_true.csv',
    ('in_regime', 'idea10'): OUT / 'fresh_in_regime/idea10_true.csv',
    ('in_regime', 'idea11'): OUT / 'fresh_in_regime/idea11_true.csv',
    ('out_regime', 'prod'): ROOT / 'research/orb_2024/book_1415_liveexit.csv',
    ('out_regime', 'idea1'): OUT / 'fresh_out_regime/idea1_true.csv',
    ('out_regime', 'idea2'): OUT / 'fresh_out_regime/idea2_true.csv',
    ('out_regime', 'idea10'): OUT / 'fresh_out_regime/idea10_true.csv',
    ('out_regime', 'idea11'): OUT / 'fresh_out_regime/idea11_true.csv',
    ('out_regime_2023', 'prod'): ROOT / 'research/orb_2023/book_1418_liveexit.csv',  # context only
}

all_rows = []
loaded = {}
for (win, pool), path in BOOKS.items():
    df = load_book(path)
    loaded[(win, pool)] = df
    if df is not None and len(df):
        tag = df.copy()
        tag['window'] = win
        tag['pool'] = pool
        all_rows.append(tag[['window', 'pool', 'date', 'symbol', 'entry_price', '_sized_pnl', 'R', '_composite']])
    print(f"{win:16s} {pool:8s} n={0 if df is None else len(df):5d}  {path}")

if all_rows:
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / '1684_pool_books.csv', index=False)
    print(f"\nwrote {OUT / '1684_pool_books.csv'} rows={sum(len(r) for r in all_rows)}")

print("\n=== per-pool per-window stats ===")
for win in ('in_regime', 'out_regime', 'out_regime_2023'):
    for pool in ('prod', 'idea1', 'idea2', 'idea10', 'idea11'):
        df = loaded.get((win, pool))
        if df is None:
            continue
        if win == 'in_regime' and len(df):
            for half, sel in (('2025', df.date.dt.year == 2025), ('2026', df.date.dt.year == 2026)):
                print(fmt(stats(df[sel], f'{win}/{pool}/{half}')))
        print(fmt(stats(df, f'{win}/{pool}/FULL')))

print("\n=== unions (pool + production, independent_1328 method) ===")
for win, lo, hi in (('in_regime', IN_LO, IN_HI), ('out_regime', OUT_LO, OUT_HI)):
    prod = loaded.get((win, 'prod'))
    print(f"\n--- {win} production alone ---")
    print(fmt(stats(prod, f'{win}/prod')))
    print(cadence_report(prod, f'{win}/prod', lo, hi))
    for pool in ('idea1', 'idea2', 'idea10', 'idea11'):
        pooldf = loaded.get((win, pool))
        if pooldf is None or len(pooldf) == 0:
            print(f"\n{win}/{pool}: no pool rows, skip union")
            continue
        union, addon, raw_overlap = union_book(prod, pooldf)
        print(f"\n{win}/{pool}: raw_overlap_with_prod={raw_overlap:.1%} added_after_excl={len(addon)}")
        print(fmt(stats(addon, f'{win}/{pool}/ADDED-TO-UNION')))
        print(fmt(stats(union, f'{win}/{pool}/UNION')))
        print(cadence_report(union, f'{win}/{pool}/UNION', lo, hi))
        daily = union.groupby(union['date'].dt.date)['R'].sum()
        worst_day = daily.idxmin()
        print(f"  shared worst day: {worst_day} R={daily.loc[worst_day]:+.2f} "
              f"(prod-alone worst day same date: "
              f"{prod[prod.date.dt.date == worst_day]['R'].sum() if prod is not None else 0:+.2f})")
