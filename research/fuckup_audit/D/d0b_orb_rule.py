#!/usr/bin/env python3
"""D0b — the ORB two-leg rule on the broad F6/F8 population (PRE-REGISTERED before running, 2026-09-16 22:20 UTC).

Why: D0 found news ALONE inert. The validated ORB separator is `has_news AND pm_dollar_vol > $5,816,688`
(research/orb_machine_rules.md); premarket dollars were not in candidates3 and are now in D/pm_bars.db (100% coverage).
Population: Stage A's population (acore.load), families F6 {} / F8 N=15 / F8 N=30, entries >= 10:00 (entry_m >= 600),
cost contract (c), exit = hold (D0's pre-registered exit) and 2r reported too.
Buckets: combo = news & pm_hi; pm_only; news_only; neither. news = n_prev15_to_0930 > 0 (causal, before 09:30);
pm_hi = pm_dollar_vol > 5816688 (src none -> 0, a real zero).
Decision rule: the combo bucket is adopted as a Stage-D1 filter only if (combo mean - rest mean) >= +0.05R with a
Welch t >= 2 on TRAIN and the sign agrees on VAL; TEST is read once, after, and reported whatever it says.
Cells: 3 families x 4 buckets x 2 exits per-trade = 24, plus the booked combo-vs-all rows (3 families x 2 exits) = 6.
"""
import os, sys, sqlite3, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A')
import acore
D = 'research/fuckup_audit/D'; PM_CUT = 5816688.0
c = acore.load()
c = c[c.fam.isin(['F6', 'F8']) & (c.entry_m >= 600)].reset_index(drop=True)
print('rows', len(c), 'keys', sorted(c.key.unique()), flush=True)
news = pd.read_csv(f'{D}/news_presence.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
news['has_news'] = (news.n_prev15_to_0930.fillna(0) > 0).astype(int)
pm = pd.read_sql('select symbol, day, pm_dollar_vol, src from pm', sqlite3.connect(f'file:{D}/pm_bars.db?mode=ro', uri=True))
pm['pm_hi'] = (pm.pm_dollar_vol.fillna(0) > PM_CUT).astype(int)
c = c.merge(news[['day', 'symbol', 'has_news']], on=['day', 'symbol'], how='left').merge(pm[['day', 'symbol', 'pm_hi']], on=['day', 'symbol'], how='left')
print('coverage news', c.has_news.notna().mean().round(4), 'pm', c.pm_hi.notna().mean().round(4), flush=True)
c['has_news'] = c.has_news.fillna(0).astype(int); c['pm_hi'] = c.pm_hi.fillna(0).astype(int)
c['bucket'] = np.select([(c.has_news == 1) & (c.pm_hi == 1), c.pm_hi == 1, c.has_news == 1], ['combo', 'pm_only', 'news_only'], 'neither')


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb)


L = ['# D0b — ORB two-leg rule (news AND premarket $ > $5.82M) on the broad F6/F8 population, entries >= 10:00', '',
     f'rows {len(c):,} | bucket shares {c.bucket.value_counts(normalize=True).round(3).to_dict()}', '']
rows = []
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        col = f'c_{tag}'
        for sp in ('TRAIN', 'VAL', 'TEST'):
            d = dk[dk.split == sp]
            for b in ('combo', 'pm_only', 'news_only', 'neither'):
                v = d[d.bucket == b][col]; rest = d[d.bucket != b][col]
                if len(v) < 20: continue
                rows.append(dict(key=key, exit=tag, split=sp, bucket=b, n=len(v), meanR=round(v.mean(), 3),
                                 se=round(v.std(ddof=1) / np.sqrt(len(v)), 3), rest_mean=round(rest.mean(), 3),
                                 diff=round(v.mean() - rest.mean(), 3), t_diff=round(welch(v, rest), 2), WR=round((v > 0).mean() * 100, 1)))
T = pd.DataFrame(rows)
pd.set_option('display.width', 220)
L += ['## per-trade, by bucket (diff = bucket minus the rest; t = Welch)', '', T.to_string(index=False), '']
# decision rule on the combo bucket
L += ['## pre-registered decision: combo adopted iff TRAIN diff >= +0.05 with t >= 2 AND VAL diff > 0', '']
for (key, tag), g in T[T.bucket == 'combo'].groupby(['key', 'exit']):
    tr = g[g.split == 'TRAIN']; va = g[g.split == 'VAL']; te = g[g.split == 'TEST']
    ok = len(tr) and len(va) and tr.iloc[0]['diff'] >= 0.05 and tr.iloc[0].t_diff >= 2 and va.iloc[0]['diff'] > 0
    L.append(f"- {key} {tag}: TRAIN diff {tr.iloc[0]['diff'] if len(tr) else 'n/a'} (t {tr.iloc[0].t_diff if len(tr) else 'n/a'}), "
             f"VAL diff {va.iloc[0]['diff'] if len(va) else 'n/a'}, TEST diff {te.iloc[0]['diff'] if len(te) else 'n/a'} -> {'ADOPT' if ok else 'no'}")
# booked: combo-only book vs the unselected book
L += ['', '## booked (run_book 12/4): combo-only vs all candidates, contract (c)', '']
brows = []
weeks = acore.week_index(c)
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        for sub, x in (('all', dk), ('combo', dk[dk.bucket == 'combo']), ('pm_hi', dk[dk.pm_hi == 1])):
            for sp in ('TRAIN', 'VAL', 'TEST'):
                t = acore.book_rows(x[x.split == sp], tag)
                if t is None: continue
                brows.append(dict(key=key, exit=tag, subset=sub, split=sp, **acore.stats(t, 'c', weeks[sp])))
B = pd.DataFrame(brows)
L += [B.to_string(index=False), '', 'cells: 24 per-trade + 18 booked (all/combo/pm_hi x 3 fam x 2 exits); TEST read once after the rule above.']
open(f'{D}/d0b_orb_rule.md', 'w').write('\n'.join(L)); T.to_csv(f'{D}/d0b_cells.csv', index=False); B.to_csv(f'{D}/d0b_booked.csv', index=False)
print('\n'.join(L), flush=True)
