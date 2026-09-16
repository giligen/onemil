#!/usr/bin/env python3
"""A3 — intraday market state at the ENTRY minute (H3 + H5). Entries >= 10:00 only, contract (c).

Features, all causal at the entry minute:
  iwm_ret / spy_ret : (close of the LAST CLOSED 1-min bar before the entry minute / the 09:30 bar's open - 1) * 100,
                      from research/lit_review_2026/etf_1min.db (read-only).
  breadth_so_far    : APPROXIMATION — share of THAT day's scoring-population candidates (deduped on
                      (day, symbol, entry_m), all 26 family-configs) that signalled at an EARLIER minute and had
                      dist_open_pct > 0. It is breadth inside the study's own candidate population, not the market's,
                      and each candidate's dist_open_pct is measured at ITS OWN entry minute, not at the split minute.
                      Requires >= 5 earlier candidates that day, else NaN.
Terciles are cut on TRAIN and the SAME edges applied to VAL. Sign splits are >= 0 / < 0.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A')
import acore
A = 'research/fuckup_audit/A'
KEYS = ['F8 {"N": 5}', 'F8 {"N": 30}', 'F6 {}', 'F1 {"P": 0.12}']
MIN_ENTRY = 600

# ---------- index state ----------
con = sqlite3.connect(f'file:{ROOT}/research/lit_review_2026/etf_1min.db?mode=ro', uri=True)
b = pd.read_sql("select symbol, t, o, c from bars where symbol in ('IWM','SPY') and t >= '2024-12-15'", con)
con.close()
ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
b['day'] = ts.dt.strftime('%Y-%m-%d'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
op = b[b.m == 570].set_index(['symbol', 'day']).o
b = b[(b.m >= 570) & (b.m <= 900)].copy()
b['o570'] = list(op.reindex(list(zip(b.symbol, b.day))))
b['ret'] = (b.c / b.o570 - 1) * 100
idx = {s: b[b.symbol == s].set_index(['day', 'm']).ret for s in ('IWM', 'SPY')}
print('etf rows', len(b), 'days', b.day.nunique(), flush=True)

# ---------- breadth ----------
p = pd.read_csv(f'{A}/pop_a.csv', usecols=['day', 'symbol', 'entry_m', 'dist_open_pct'],
                dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
p = p.drop_duplicates(['day', 'symbol', 'entry_m']).sort_values(['day', 'entry_m'])
p['pos'] = (p.dist_open_pct > 0).astype(int)
g = p.groupby('day')
p['cum_n'] = g.cumcount()
p['cum_pos'] = g.pos.cumsum() - p.pos
# collapse to (day, entry_m) -> counts of everything signalled strictly EARLIER
ends = p.groupby(['day', 'entry_m']).agg(n_before=('cum_n', 'min'), pos_before=('cum_pos', 'min'))
print('breadth index rows', len(ends), flush=True)


def breadth(day, m):
    try:
        r = ends.loc[(day, m)]
    except KeyError:
        return np.nan
    return r.pos_before / r.n_before if r.n_before >= 5 else np.nan


c = acore.load(keys=KEYS)
c = c[c.entry_m >= MIN_ENTRY]
WK = acore.week_index(c)
rows = []
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        bk = {}
        for sp in ('TRAIN', 'VAL'):
            t = acore.book_rows(dk[dk.split == sp], tag)
            if t is None: continue
            t['iwm'] = [idx['IWM'].get((d, m - 1), np.nan) for d, m in zip(t.day, t.em)]
            t['spy'] = [idx['SPY'].get((d, m - 1), np.nan) for d, m in zip(t.day, t.em)]
            t['brd'] = [breadth(d, m) for d, m in zip(t.day, t.em)]
            bk[sp] = t
        if 'TRAIN' not in bk: continue
        for feat in ('iwm', 'spy', 'brd'):
            q = bk['TRAIN'][feat].quantile([1/3, 2/3]).values
            for sp, t in bk.items():
                v = t[feat]
                buckets = [('T1 low', v <= q[0]), ('T2 mid', (v > q[0]) & (v <= q[1])), ('T3 high', v > q[1])]
                if feat != 'brd':
                    buckets += [('sign -', v < 0), ('sign +', v >= 0)]
                for lab, msk in buckets:
                    x = t[msk & v.notna()]
                    if not len(x): continue
                    se = x.c.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                    rows.append(dict(key=key, exit=tag, feat=feat, bucket=lab, split=sp, n=len(x),
                                     cut_lo=round(float(q[0]), 3), cut_hi=round(float(q[1]), 3),
                                     meanR=round(float(x.c.mean()), 3), se=round(float(se), 4),
                                     t=round(float(x.c.mean() / se), 2) if se == se and se else np.nan,
                                     WR=round(float((x.c > 0).mean() * 100), 1), nan_share=round(float(v.isna().mean()), 3)))
R = pd.DataFrame(rows)
R.to_csv(f'{A}/a3_cells.csv', index=False)
piv = R.pivot_table(index=['key', 'exit', 'feat', 'bucket'], columns='split', values=['n', 'meanR', 't'])
piv.columns = [f'{a}_{b}' for a, b in piv.columns]
piv = piv.reset_index()
piv['sign_agree'] = np.sign(piv.meanR_TRAIN) == np.sign(piv.meanR_VAL)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
agree = piv.sign_agree.mean()
L = ['# A3 — intraday market state at the entry minute, entries >= 10:00, contract (c)', '',
     f'cells: {len(R)} bucket x split rows | {piv.shape[0]} buckets | TRAIN->VAL sign agreement {agree:.2f}', '',
     piv[['key', 'exit', 'feat', 'bucket', 'n_TRAIN', 'meanR_TRAIN', 't_TRAIN', 'n_VAL', 'meanR_VAL', 't_VAL',
          'sign_agree']].to_string(index=False), '',
     'by feature, share of buckets whose TRAIN and VAL means share a sign:', '',
     piv.groupby('feat').sign_agree.agg(['mean', 'size']).round(2).to_string(), '']
open(f'{A}/a3_tables.md', 'w').write('\n'.join(L))
print('\n'.join(L)); print('DONE', flush=True)
