#!/usr/bin/env python3
"""Stage I — the shared contract: cost (c), dedupe, run_book(12,4), split stats.

Identical arithmetic to H/F6/f6_pdr_book.py; re-stated here so the stacked book is scored by ONE code path.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
if os.getcwd() != ROOT:
    os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.hod_break import run_book  # noqa: E402

D = 'research/fuckup_audit/I'
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}
ENTRY = 0.25
PRIORITY = ['F6', 'F14', 'F11', 'F8N30']
STACKS = {'S0': ['F6'], 'S1': ['F6', 'F14'], 'S2': ['F6', 'F14', 'F11'],
          'S3': ['F6', 'F14', 'F11', 'F8N30']}


def load(path=f'{D}/pop_i.csv'):
    c = pd.read_csv(path, dtype={'symbol': str, 'day': str, 'tag': str},
                    keep_default_na=False, na_values=[''])
    c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
    c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
    c['mo'] = c.day.str[:7]
    half = 0.5 * (c.spread_cc_bps / 1e4 * 100) / c.next_r_pct.clip(lower=0.05)
    c['half'] = half
    for tag in ('hold', '2r'):
        c[f'net_{tag}'] = c[f'next_rr_{tag}'] - ENTRY * half - half * c[f'next_why_{tag}'].map(RATIO).fillna(0.875)
    c['prio'] = c.tag.map({t: i for i, t in enumerate(PRIORITY)})
    return c


def weeks_of(c):
    return {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}


def dedupe(c, fams):
    """One row per (symbol, day): the highest-priority family, then the earliest entry minute."""
    x = c[c.tag.isin(fams)]
    x = x.sort_values(['day', 'symbol', 'prio', 'next_entry_m'])
    return x.drop_duplicates(['day', 'symbol'], keep='first')


def book(x, tag):
    rows = [(r.day, int(r.next_entry_m), int(getattr(r, f'next_exit_m_{tag}')), r.symbol,
             getattr(r, f'net_{tag}'), getattr(r, f'next_rr_{tag}'), getattr(r, f'next_why_{tag}'),
             r.wk, r.mo, r.tag, r.Index) for r in x.itertuples()]
    t = run_book(rows, 12, 4)
    return pd.DataFrame(t, columns=['day', 'em', 'xm', 'symbol', 'net', 'gross', 'why', 'wk', 'mo', 'tag', 'idx'])


def stats(t, sp, weeks, extra_mix=True):
    v = t.net.values
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk').net.sum().reindex(weeks[sp]).fillna(0)
    m = t.groupby('mo').net.sum()
    s = np.sort(v)
    out = dict(split=sp, n=n, tpw=round(n / len(weeks[sp]), 1), meanR=round(v.mean(), 3),
               gross=round(t.gross.mean(), 3), t=round(v.mean() / se, 2), WR=round((v > 0).mean() * 100, 1),
               wkR=round(float(w.mean()), 2), green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1),
               mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 1),
               moG=f'{int((m > 0).sum())}/{len(m)}',
               ex5=round(s[:int(n * 0.95)].mean(), 3) if n > 20 else np.nan,
               cap3=round(np.minimum(v, 3).mean(), 3))
    if extra_mix:
        mix = t.tag.value_counts()
        out['mix'] = ' '.join(f'{k}:{int(vv)}' for k, vv in mix.items())
    return out
