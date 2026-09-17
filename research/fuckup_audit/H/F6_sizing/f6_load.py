#!/usr/bin/env python3
"""Stage H/F6_sizing step 1 — extract the F6 population from C/pop_c.csv and reproduce the C book numbers.

Population (PLAN §1 / B/score5.py): fam F6, next_entry present and >= 5, next_entry_m <= 841,
next_r_pct >= 1.0, range_so_far_pct >= 5.  Cost contract (c): half = 0.5*(spread_cc_bps/100)/max(r_pct,0.05);
net = rr - 0.25*half - half*{stop .875, lock .875, eod .412, target .875, none .875}[why].
Book = trading.hod_break.run_book(rows, 12, 4).
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book

D = 'research/fuckup_audit/H/F6_sizing'
SRC = 'research/fuckup_audit/C/pop_c.csv'
OUT = f'{D}/pop_f6.csv'
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}
ENTRY_MULT_NEXT = 0.25

KEEP = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'level', 'stop', 'price', 'range_so_far_pct', 'rv_adv',
        'gap_pct', 'adv20', 'spread_pct', 'spread_cc_bps', 'prev_day_range_pct', 'asset_class',
        'next_entry', 'next_entry_m', 'next_r_pct',
        'next_rr_2r', 'next_why_2r', 'next_exit_m_2r',
        'next_rr_hold', 'next_why_hold', 'next_exit_m_hold']


def load():
    if os.path.exists(OUT):
        return pd.read_csv(OUT, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    parts, n_in, t0 = [], 0, time.time()
    for ch in pd.read_csv(SRC, usecols=KEEP, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                          keep_default_na=False, na_values=[''], chunksize=200_000, low_memory=True):
        n_in += len(ch)
        ch = ch[ch.fam == 'F6']
        if not len(ch):
            continue
        ok = (ch.range_so_far_pct >= 5) & ch.next_entry.notna() & (ch.next_entry >= 5) \
             & (ch.next_entry_m <= 841) & (ch.next_r_pct >= 1.0)
        ch = ch[ok]
        if len(ch):
            parts.append(ch)
        print(f'{n_in:,} read | kept {sum(len(p) for p in parts):,} | {(time.time()-t0)/60:.1f}m', flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT, index=False)
    return d


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


def net_r(d, rr, why, spread_mult=1.0):
    half = 0.5 * (d.spread_cc_bps * spread_mult / 100.0) / d.next_r_pct.clip(lower=0.05)
    return d[rr] - ENTRY_MULT_NEXT * half - half * d[why].map(EXIT_RATIO).fillna(0.875)


def book(d, rr, why, xm, spread_mult=1.0):
    """Returns the booked trades (run_book 12/4) with net R attached."""
    x = d.copy()
    x['net'] = net_r(x, rr, why, spread_mult)
    x['xm'] = x[xm]
    x = x[x.net.notna() & x.xm.notna()]
    rows = [(r.day, int(r.next_entry_m), int(r.xm), r.symbol, float(r.net), r.Index) for r in x.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'idx'])
    return t


if __name__ == '__main__':
    d = load()
    d['split'] = split_of(d.day.values)
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
    print(f'F6 population rows: {len(d):,}   ' + str(d.split.value_counts().to_dict()), flush=True)
    for name, (rr, why, xm) in {'hold-to-close': ('next_rr_hold', 'next_why_hold', 'next_exit_m_hold'),
                                '2R close-fill': ('next_rr_2r', 'next_why_2r', 'next_exit_m_2r')}.items():
        for sp in ('TRAIN', 'VAL', 'TEST'):
            t = book(d[d.split == sp], rr, why, xm)
            sd = t.net.std(ddof=1)
            print(f'{name:14s} {sp:5s} n={len(t):5d} meanR={t.net.mean():+.4f} '
                  f't={t.net.mean()/(sd/np.sqrt(len(t))):+.2f}', flush=True)
