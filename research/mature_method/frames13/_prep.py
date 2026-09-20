"""F40 — pull the two small lookups OUT of parquet.

`pd.read_parquet` reserves ~1.3 GB of VIRTUAL address space for its arena the first time it is
called, which alone is half this node's `ulimit -v 3000000`.  The scorer therefore reads CSVs.
"""
import pandas as pd

D13 = '/home/ec2-user/onemil/research/mature_method/frames13'
PX = '/home/ec2-user/onemil/research/multiday/data/prices_by_year'

pd.read_parquet(f'{D13}/symbols.parquet').to_csv(f'{D13}/symbols.csv', index=False)
fr = []
for y in range(2016, 2027):
    s = pd.read_parquet(f'{PX}/all/year={y}.parquet', columns=['symbol', 'date', 'close'])
    s = s[s.symbol.astype(str) == 'SPY'][['date', 'close']].copy()
    s['date'] = s.date.astype(str)
    fr.append(s)
pd.concat(fr, ignore_index=True).drop_duplicates('date').sort_values('date') \
  .to_csv(f'{D13}/spy_daily.csv', index=False)
print('symbols.csv + spy_daily.csv written')
