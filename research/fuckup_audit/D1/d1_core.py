#!/usr/bin/env python3
"""D1 shared core — D0's d0_core.py, extended to two stop/exit targets.

The book is the live spec's: `trading.hod_break.run_book(rows, 12, 4)` — first come by entry minute, alphabetical
tie-break, 12 fills a day, 4 concurrent, causal slot freeing. A SELECTION rule is a filter applied to the day's
candidates BEFORE run_book; the book itself never changes. Weeks per split are counted on the whole candidate table
(a week the cell misses scores 0).

Targets (PREREG §0.2): 'p' = hold-to-close with the touch stop (primary); 's' = the 2R stop-1% variant (secondary,
its own R). Each has its own population mask: net_<t> is NaN where that variant's r_pct < 1.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.hod_break import run_book                                    # noqa: E402

D1 = f'{ROOT}/research/fuckup_audit/D1'
FEAT = f'{D1}/feat.csv'
KEYS = ['F6 {}', 'F8 {"N": 30}', 'F14 {"N": 15}', 'F11 {"base": "F6"}']
EXTRA_KEY = 'F8 {"N": 5}'                        # declared extra: the two-leg univariate cell only
TARGETS = ('p', 's')
TGT_LABEL = {'p': 'hold / touch stop', 's': '2R / stop-1%'}
SPLITS = ('TRAIN', 'VAL', 'TEST')
DT = {'symbol': str, 'day': str, 'key': str, 'fam': str, 'cfg': str, 'split': str, 'month': str, 'wk': str,
      'why_p': str, 'why_s': str}


def load(keys=None):
    c = pd.read_csv(FEAT, dtype=DT, keep_default_na=False, na_values=[''])
    if keys is not None:
        c = c[c.key.isin(keys)].reset_index(drop=True)
    return c


def feat_cols(c, drop=()):
    return [k for k in c.columns if k.startswith('f_') and k not in drop]


def week_index(c):
    return {s: sorted(c[c.split == s].wk.unique()) for s in SPLITS}


def book(x, tgt='p', netcol=None):
    """run_book(12, 4) over a candidate subset for one target. Returns the taken rows."""
    nc = netcol or f'net_{tgt}'
    y = x[x[nc].notna() & x[f'xm_{tgt}'].notna()]
    if not len(y):
        return None
    rows = [(r.day, int(r.next_entry_m), int(getattr(r, f'xm_{tgt}')), r.symbol, float(getattr(r, nc)),
             getattr(r, f'why_{tgt}'), r.wk, r.month, r.split) for r in y.itertuples()]
    t = run_book(rows, 12, 4)
    if not t:
        return None
    return pd.DataFrame(t, columns=['day', 'em', 'xm', 'symbol', 'net', 'why', 'wk', 'month', 'split'])


def stats(t, weeks, col='net'):
    v = t[col].values
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk')[col].sum().reindex(weeks).fillna(0)
    mix = t.why.value_counts(normalize=True)
    return dict(n=n, tpw=round(n / len(weeks), 1), meanR=round(float(v.mean()), 4),
                se=round(float(se), 4), t=round(float(v.mean() / se), 2) if se == se and se else np.nan,
                mde=round(float(2.8 * se), 4) if se == se else np.nan,
                WR=round(float((v > 0).mean() * 100), 1), wkR=round(float(w.mean()), 2),
                green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1),
                mix_stop=round(float(mix.get('stop', 0)), 3), mix_eod=round(float(mix.get('eod', 0)), 3),
                mix_target=round(float(mix.get('target', 0)), 3))


def tail_stats(t, weeks, col='net'):
    """Tail dependence: top 1% / top 5% of booked trades removed, winners capped at +3R."""
    v = t[col].values
    out = {}
    for q, lab in ((0.99, 'cut1'), (0.95, 'cut5')):
        thr = np.quantile(v, q)
        x = t[t[col] < thr]
        out[f'{lab}_meanR'] = round(float(x[col].mean()), 4) if len(x) else np.nan
        out[f'{lab}_wkR'] = round(float(x.groupby('wk')[col].sum().reindex(weeks).fillna(0).mean()), 2) if len(x) else np.nan
    cp = t.copy(); cp[col] = cp[col].clip(upper=3.0)
    out['cap3_meanR'] = round(float(cp[col].mean()), 4)
    out['cap3_wkR'] = round(float(cp.groupby('wk')[col].sum().reindex(weeks).fillna(0).mean()), 2)
    return out
