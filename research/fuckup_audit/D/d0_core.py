#!/usr/bin/env python3
"""D0 shared core — the feature table, the book, the per-cell statistics.

The book is Stage A's / the live spec's: `trading.hod_break.run_book(rows, 12, 4)` — first come by
entry minute, alphabetical tie-break, 12 fills a day, 4 concurrent, causal slot freeing.
A SELECTION rule is a filter applied to the day's candidates BEFORE run_book; the book itself never
changes. Weeks per split are counted on the whole candidate table (a week the cell misses scores 0).
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.hod_break import run_book                                    # noqa: E402

D = f'{ROOT}/research/fuckup_audit/D'
FEAT = f'{D}/feat.csv'
KEYS = ['F6 {}', 'F8 {"N": 30}', 'F8 {"N": 15}']
SPLITS = ('TRAIN', 'VAL', 'TEST')


def load():
    c = pd.read_csv(FEAT, dtype={'symbol': str, 'day': str, 'key': str, 'fam': str, 'cfg': str,
                                 'split': str, 'month': str, 'wk': str, 'why_hold': str},
                    keep_default_na=False, na_values=[''])
    return c


def feat_cols(c, drop=()):
    return [k for k in c.columns if k.startswith('f_') and k not in drop]


def week_index(c):
    """Week labels per split, counted on the WHOLE table (score4/Stage A convention)."""
    return {s: sorted(c[c.split == s].wk.unique()) for s in SPLITS}


def book(x):
    """run_book(12, 4) over a candidate subset. Returns the taken rows as a DataFrame."""
    if not len(x):
        return None
    rows = [(r.day, int(r.entry_m), int(r.exit_m_hold), r.symbol, float(r.net), int(r.win),
             r.why_hold, r.wk, r.month, r.split) for r in x.itertuples()]
    t = run_book(rows, 12, 4)
    if not t:
        return None
    return pd.DataFrame(t, columns=['day', 'em', 'xm', 'symbol', 'net', 'win', 'why', 'wk', 'month', 'split'])


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
