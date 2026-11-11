"""Window table: PIT (seeded) book vs the published current-params book. Realized $ = _sized_pnl on entered fills."""
import pandas as pd
D = 'research/book_review_20261111/'
pit = pd.read_csv(D + 'orb_bplus_book_pit_full.csv', keep_default_na=False, na_values=[''])
pub = pd.read_csv('analysis_results/orb_bplus_book.csv', keep_default_na=False, na_values=[''])
WIN = [('4wk', '2026-10-14'), ('6wk', '2026-09-30'), ('6mo', '2026-05-11')]


def stats(d):
    """fills, $, ex-top-1 fill $, ex-top-5% fills $ (top ceil(5%) fills removed)."""
    f = d[d['entered'] == 1].sort_values('_sized_pnl', ascending=False)
    n = len(f); k5 = max(1, round(0.05 * n)) if n else 0
    return n, f['_sized_pnl'].sum(), f['_sized_pnl'].iloc[1:].sum(), f['_sized_pnl'].iloc[k5:].sum(), k5


for name, s in WIN:
    a = stats(pit[pit['date'] >= s]); b = stats(pub[pub['date'] >= s])
    mix = pit[pit['date'] >= s].drop_duplicates('date')['params_source'].value_counts().to_dict()
    print(f"{name} since {s}: PIT fills {a[0]} ${a[1]:+,.0f} ex1 ${a[2]:+,.0f} ex5% ${a[3]:+,.0f} (k={a[4]}) | "
          f"PUB fills {b[0]} ${b[1]:+,.0f} ex1 ${b[2]:+,.0f} ex5% ${b[3]:+,.0f} | days by params_source {mix}")
print('full-book days by source:', pit.drop_duplicates('date')['params_source'].value_counts().to_dict(),
      'rows', len(pit), pit['date'].min(), pit['date'].max())
