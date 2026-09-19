import pandas as pd
pd.set_option('display.width', 240)
D = 'research/mature_method/hod_preopen_regime'
c = pd.read_csv(f'{D}/cells.csv'); c['cell'] = c.cell.str.strip()
c = c.drop_duplicates(['cell', 'split'])
p = c.pivot(index='cell', columns='split')
sel = ['B0', 'B2'] + [x for x in p.index if x.startswith('A') and '[B2]' in x] \
      + [x for x in p.index if x.startswith('F5M') and '[B2]' in x] \
      + [x for x in p.index if x.startswith('M') and '[B2]' in x]
z = pd.concat([p['per_wk'], p['green'], p['total'], p['gross'], p['net'], p['mdd'], p['worst']],
              axis=1, keys=['wk', 'grn', 'tot', 'gross', 'net', 'mdd', 'worst']).loc[sel]
print(z.round(3).to_string())
n = pd.read_csv(f'{D}/nulls.csv')
print('\noutside-band summary:\n', n.outside.value_counts().to_string())
