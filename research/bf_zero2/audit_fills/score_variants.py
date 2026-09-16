#!/usr/bin/env python3
"""Score every fill correction: apply it to the WHOLE F6 candidate pool, re-run the book rule
(run_book, 4 concurrent / 4 per day, first-come, causal freeing) and report mean net R per split.

Every correction is scored on the same pool, so the deltas are attributable.
Cost model: half a spread, in R units, charged on every exit that is not the resting target. The
spread is either the study's flat 40 bps or a per-price-band table measured from the REAL SIP NBBO in
the book's own fill minute (audit_fills/book_spreads.csv).  → audit_fills/variants.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_fills'
SPLITS = ('TRAIN', 'VAL', 'TEST')
BANDS = [5, 10, 20, 50, 1e9]

R = pd.read_csv(f'{A}/pool_rewalk.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
have2 = os.path.exists(f'{A}/pool_rewalk2.csv')
if have2:
    R2 = pd.read_csv(f'{A}/pool_rewalk2.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    R = R.merge(R2.drop(columns=['entry_m']), on=['day', 'symbol'], how='left')
print(f'rewalk rows {len(R):,} | rewalk2 merged: {have2}', flush=True)

# ---------- spread tables ----------
tab = {}
s = pd.read_csv(f'{A}/book_spreads.csv')
s = s[s.n_quotes > 0].copy()
s['sp_pct'] = s.spread_med / s.price * 100
s['mid_dev'] = (s.mid_med / s.price - 1) * 100
clean = s[(s.n_quotes >= 50) & (s.mid_dev.abs() <= 2)]          # actively quoted, quotes consistent with the tape
for name, d in (('measured_all', s), ('measured_clean', clean)):
    tab[name] = d.groupby(pd.cut(d.price, BANDS), observed=True).sp_pct.agg(['size', 'median', 'mean'])
print('\nmeasured full spread (% of price) in the book\'s own fill minute:')
for k, v in tab.items(): print(k, '\n', v.round(3).to_string())


def band_spread(price, which, stat):
    t = tab[which]
    m = {str(i): float(t.loc[i, stat]) for i in t.index}
    return pd.cut(price, BANDS).astype(str).map(m).astype(float)


def net(df, tag, spread_pct=0.40, per_band=None):
    rp = df[f'rpct_{tag}'] if f'rpct_{tag}' in df.columns else df.r_pct
    rp = pd.to_numeric(rp, errors='coerce').fillna(df.r_pct)
    sp = spread_pct if per_band is None else band_spread(df.entry, *per_band)
    half = 0.5 * sp / rp.clip(lower=0.05)
    return df[f'rr_{tag}'] - np.where(df[f'why_{tag}'] == 'target', 0.0, half)


def book(df, tag, netcol='netv', xmcol=None, emcol='entry_m'):
    xm = xmcol or f'xm_{tag}'
    d = df[df[netcol].notna() & df[xm].notna()]
    rows = [(r.day, int(getattr(r, emcol)), int(getattr(r, xm)), r.symbol, getattr(r, netcol), r.wk) for r in d.itertuples()]
    bk = pd.DataFrame(run_book(rows, 4, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    bk['split'] = np.where(bk.day < '2026-01-01', 'TRAIN', np.where(bk.day < '2026-06-01', 'VAL', 'TEST'))
    out = {}
    for sp in SPLITS:
        x = bk[bk.split == sp]; w = x.groupby('wk').net.sum()
        out[sp] = dict(n=len(x), meanR=round(x.net.mean(), 3), t=round(x.net.mean() / (x.net.std() / np.sqrt(len(x))), 2),
                       WR=round((x.net > 0).mean() * 100, 1), sumR=round(x.net.sum(), 1), green=f'{int((w > 0).sum())}/{len(w)}')
    return out, bk


CASES = [
    ('BASE - the study as published', R, 'base', dict()),
    ('C1 entry = max(level x 1.003, entry-bar OPEN)', R, 'g', dict()),
    ('C2 stop charged inside the entry bar', R, 'eb', dict()),
    ('C3a stop fill 25 bps through', R, 's25', dict()),
    ('C3b stop fill 50 bps through', R, 's50', dict()),
    ('C3c stop fill 100 bps through', R, 's100', dict()),
    ('C5 15:55 exit 10 bps below the open', R, 'eod10', dict()),
]
if have2:
    CASES += [('C8e entry slip 50 bps instead of 30', R, 'e50', dict()),
              ('C8e entry slip 75 bps (half the measured median spread)', R, 'e75', dict()),
              ('C8e entry slip 150 bps (half the measured mean spread)', R, 'e150', dict())]
CASES += [
    ('C6 drop positions > 1% of the 5-min $ volume', R[R.notional_frac_dv5 <= 0.01], 'base', dict()),
    ('C6b drop positions > 5% of the 5-min $ volume', R[R.notional_frac_dv5 <= 0.05], 'base', dict()),
]
halt = R.why_base == 'eod_notape'
Rh = R.copy(); Rh.loc[halt, 'rr_base'] = -1.0; Rh.loc[halt, 'why_base'] = 'halt'
CASES += [
    ('C7 tape-ends-early names booked at -1R', Rh, 'base', dict()),
    ('C7b tape-ends-early names excluded', R[~halt], 'base', dict()),
    ('C8 exit cost = MEASURED median spread (actively quoted)', R, 'base', dict(per_band=('measured_clean', 'median'))),
    ('C8b exit cost = MEASURED mean spread (actively quoted)', R, 'base', dict(per_band=('measured_clean', 'mean'))),
    ('C8c exit cost = MEASURED median spread (all sampled)', R, 'base', dict(per_band=('measured_all', 'median'))),
]

rows = []
for label, df, tag, kw in CASES:
    d = df.copy(); d['netv'] = net(d, tag, **kw)
    res, _ = book(d, tag)
    rows.append(dict(correction=label, **{f'{s}_{k}': res[s][k] for s in SPLITS for k in ('n', 'meanR', 't', 'WR', 'sumR', 'green')}))
    print(f'{label:58s} ' + ' | '.join(f"{s} {res[s]['meanR']:+.3f} (n {res[s]['n']})" for s in SPLITS), flush=True)


def stacked(label, tag, per_band, liq_thr=0.01, halt_penalty=True):
    d = R[R.notional_frac_dv5 <= liq_thr].copy()
    if halt_penalty:
        hm = d[f'why_{tag}'] == 'eod_notape'
        d.loc[hm, f'rr_{tag}'] = -1.0
    d['netv'] = net(d, tag, per_band=per_band)
    res, bk = book(d, tag)
    rows.append(dict(correction=label, **{f'{s}_{k}': res[s][k] for s in SPLITS for k in ('n', 'meanR', 't', 'WR', 'sumR', 'green')}))
    print(f'{label:58s} ' + ' | '.join(f"{s} {res[s]['meanR']:+.3f} (n {res[s]['n']})" for s in SPLITS), flush=True)
    return bk


bk_all = stacked('ALL corrections, STUDY cost (40 bps)', 'all', None)
bk_allc = stacked('ALL corrections + MEASURED median spread', 'all', ('measured_clean', 'median'))
bk_allc.to_csv(f'{A}/book_all_corrections.csv', index=False)
if have2:
    stacked('ALL + 75 bps entry slip + MEASURED median spread', 'allx', ('measured_clean', 'median'))
    stacked('ALL + 75 bps entry slip + MEASURED mean spread', 'allx', ('measured_clean', 'mean'))
# a middle book: only the two corrections that are beyond argument (gap-through entry, real median spread)
d = R.copy(); d['netv'] = net(d, 'g', per_band=('measured_clean', 'median'))
res, _ = book(d, 'g')
rows.append(dict(correction='MID: C1 + measured median spread only', **{f'{s}_{k}': res[s][k] for s in SPLITS for k in ('n', 'meanR', 't', 'WR', 'sumR', 'green')}))
print('MID: ' + ' | '.join(f"{s} {res[s]['meanR']:+.3f}" for s in SPLITS), flush=True)

T = pd.DataFrame(rows)
T.to_csv(f'{A}/variants.csv', index=False)
pd.set_option('display.width', 250)
print('\n' + T[['correction'] + [f'{s}_meanR' for s in SPLITS] + [f'{s}_n' for s in SPLITS] + [f'{s}_t' for s in SPLITS]].to_string(index=False))
print('DONE', flush=True)
