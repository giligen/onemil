#!/usr/bin/env python3
"""D1 step 7 — the univariate two-leg ORB rule on candidates4 (PREREG §0.7; D0b re-run, ALL DAY and by time band).

Rule: `has_news AND pm_dollar_vol > $5,816,688` (research/orb_machine_rules.md). D0b measured it on candidates3 with
entries >= 10:00 and found the combo bucket WORSE everywhere. Here: candidates4, ALL DAY, per family incl. the
declared extra F8 N=5, hold exit (target p), contract (c), per split and per time band.

Buckets: combo / pm_only / news_only / neither, on rows where BOTH legs are known; rows where either leg is
uncovered form their own `missing` bucket and are excluded from the combo-vs-rest comparison (coverage reported).
Decision rule (pre-registered, not an adoption by default): carry the rule forward only if TRAIN diff >= +0.05R with
Welch t >= 2 AND the VAL sign agrees.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D1')
import d1_core as K

D1 = 'research/fuckup_audit/D1'
BANDS = [(569, 600, '09:30-10:00'), (600, 720, '10:00-12:00'), (720, 842, '12:00+')]

c = K.load()
c = c[c.net_p.notna()].reset_index(drop=True)
weeks = K.week_index(c)
known = (c.f_news_missing == 0) & (c.f_pm_missing == 0)
c['bucket'] = np.where(~known, 'missing',
                       np.select([(c.f_news_pre > 0) & (c.f_pm_hi == 1), c.f_pm_hi == 1, c.f_news_pre > 0],
                                 ['combo', 'pm_only', 'news_only'], 'neither'))
print('coverage (both legs known):', round(float(known.mean()), 4), flush=True)
print(c.bucket.value_counts(normalize=True).round(4).to_string(), flush=True)


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if (va + vb) > 0 else np.nan


rows = []
for key, dk in c.groupby('key'):
    for band_lo, band_hi, band in [(569, 842, 'all-day')] + BANDS:
        db = dk[(dk.sig_m > band_lo) & (dk.sig_m <= band_hi)]
        for sp in ('TRAIN', 'VAL', 'TEST'):
            d = db[db.split == sp]
            kn = d[d.bucket != 'missing']
            for b in ('combo', 'pm_only', 'news_only', 'neither', 'missing'):
                v = d[d.bucket == b].net_p
                rest = (kn[kn.bucket != b].net_p if b != 'missing' else kn.net_p)
                if len(v) < 20 or len(rest) < 20:
                    continue
                rows.append(dict(key=key, band=band, split=sp, bucket=b, n=len(v), meanR=round(v.mean(), 3),
                                 se=round(v.std(ddof=1) / np.sqrt(len(v)), 3), rest_mean=round(rest.mean(), 3),
                                 diff=round(v.mean() - rest.mean(), 3), t_diff=round(welch(v, rest), 2),
                                 WR=round((v > 0).mean() * 100, 1)))
T = pd.DataFrame(rows)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 900)

L = ['# D1 step 7 — the two-leg ORB rule (news AND premarket $ > $5.82M) on candidates4, ALL DAY and by time band',
     '', f'rows {len(c):,} (primary target population) | coverage both legs {float(known.mean()):.4f}',
     '', 'bucket shares: ' + str(c.bucket.value_counts(normalize=True).round(4).to_dict()), '',
     '## per-trade net R by bucket (diff = bucket minus the covered rest; t = Welch)', '',
     T.to_string(index=False), '',
     '## pre-registered decision: carry the combo rule forward iff TRAIN diff >= +0.05 with t >= 2 AND VAL diff > 0',
     '']
for (key, band), g in T[T.bucket == 'combo'].groupby(['key', 'band']):
    tr = g[g.split == 'TRAIN']; va = g[g.split == 'VAL']; te = g[g.split == 'TEST']
    ok = len(tr) and len(va) and tr.iloc[0]['diff'] >= 0.05 and tr.iloc[0].t_diff >= 2 and va.iloc[0]['diff'] > 0
    L.append(f"- {key} [{band}]: TRAIN diff {tr.iloc[0]['diff'] if len(tr) else 'n/a'} "
             f"(t {tr.iloc[0].t_diff if len(tr) else 'n/a'}, n {int(tr.iloc[0].n) if len(tr) else 0}), "
             f"VAL diff {va.iloc[0]['diff'] if len(va) else 'n/a'}, "
             f"TEST diff {te.iloc[0]['diff'] if len(te) else 'n/a'} -> {'CARRY' if ok else 'no'}")

# booked: combo-only book vs the unselected book (all-day)
brows = []
for key, dk in c.groupby('key'):
    for sub, x in (('all', dk), ('combo', dk[dk.bucket == 'combo']), ('pm_hi', dk[dk.f_pm_hi == 1]),
                   ('news', dk[dk.f_news_pre > 0])):
        for sp in ('TRAIN', 'VAL', 'TEST'):
            t = K.book(x[x.split == sp], 'p')
            if t is None or len(t) < 20:
                continue
            brows.append(dict(key=key, subset=sub, split=sp, **K.stats(t, weeks[sp])))
B = pd.DataFrame(brows)
L += ['', '## booked (run_book 12/4), all-day, target p: combo-only / pm_hi / news vs all candidates', '',
      B.to_string(index=False), '',
      f'cells: 5 families x 4 buckets x 4 bands x 3 splits reported ({len(T)} rows with n>=20); '
      f'the 5 DECLARED cells are the all-day combo-vs-rest test per family.']
open(f'{D1}/d1_orb.md', 'w').write('\n'.join(L))
T.to_csv(f'{D1}/d1_orb_cells.csv', index=False); B.to_csv(f'{D1}/d1_orb_booked.csv', index=False)
print('\n'.join(L[:10]), flush=True)
print('wrote d1_orb.md', flush=True)
