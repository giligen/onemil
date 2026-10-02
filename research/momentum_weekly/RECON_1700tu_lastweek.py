"""Last-week check + cost-gap split by cause (reads dump outputs + 3 panel days)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
H = '/home/ec2-user/onemil/research/momentum_weekly/'
P = lambda *a: print(*a, flush=True)
t = pq.read_table(H + 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'open', 'close'], filters=[('bar_date', '>=', pd.Timestamp('2026-09-21').date())]).to_pandas()
t['bar_date'] = pd.to_datetime(t.bar_date); P(t.groupby('bar_date').size().to_string())
hb = pd.read_csv(H + 'RECON_1700tu_A_guarded_hold.csv', parse_dates=['date']); h = hb[(hb.date == '2026-09-28') & (hb.kept >= 0)]
d28 = set(t[t.bar_date == '2026-09-28'].symbol); P('A holdings at 2026-09-28:', len(h), 'with a 9/28 bar:', int(h.sym.isin(d28).sum()))
# cost split on common name-weeks (cost as fraction of that book's pre-trade equity)
wa = pd.read_csv(H + 'RECON_1700tu_A_guarded_weeks.csv', parse_dates=['date']).set_index('date'); wb = pd.read_csv(H + 'RECON_1700tu_B_guarded_weeks.csv', parse_dates=['date']).set_index('date')
ha = pd.read_csv(H + 'RECON_1700tu_A_guarded_hold.csv', parse_dates=['date']); hb = pd.read_csv(H + 'RECON_1700tu_B_guarded_hold.csv', parse_dates=['date'])
ha['cf'] = ha.cost / ha.date.map(wa.eq_pre); hb['cf'] = hb.cost / hb.date.map(wb.eq_pre)
m = ha.merge(hb, on=['date', 'sym'], suffixes=('_A', '_B')); m = m[m.date >= '2017-02-06']; yrs = 9.64
m['hi15'] = m.rate_B > 0.0015
for nm, g in (('rate_B > 15bp (A clipped)', m[m.hi15]), ('rate_B <= 15bp (proxy day differs only)', m[~m.hi15])):
    P(f'{nm}: rows {len(g)}  cost-gap (B-A) {((g.cf_B - g.cf_A).sum()) / yrs * 100:.3f} %/yr  mean rate A {g.rate_A.mean() * 1e4:.1f}bp B {g.rate_B.mean() * 1e4:.1f}bp')
P('sold-only (exits) rows: A cost', ha[(ha.kept == -1) & (ha.date >= '2017-02-06')].cf.sum() / yrs * 100, 'B', hb[(hb.kept == -1) & (hb.date >= '2017-02-06')].cf.sum() / yrs * 100)
P('total cost drag %/yr A', ha[ha.date >= '2017-02-06'].cf.sum() / yrs * 100, 'B', hb[hb.date >= '2017-02-06'].cf.sum() / yrs * 100)
P('share of common rows where rate_B>rate_A', float((m.rate_B > m.rate_A).mean()), ' rate_B>15bp share', float(m.hi15.mean()))
print(wa.tail(3).to_string(), flush=True); print(wb.tail(3).to_string(), flush=True)
