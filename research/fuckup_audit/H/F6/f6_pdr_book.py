#!/usr/bin/env python3
"""F6-PDR — the owner's book, declared 2026-09-17 15:10 UTC BEFORE this run, TEST read ONCE here.

Definition (all causal at the signal bar): F6 red-to-green (opens below prev close; signal = first 1-min bar whose high
reaches prev close x 1.003); prev_day_range_pct >= 8 (ORB's shipped day-2-continuation veto, NOT derived from these
losers — H/F6/REPORT.md found it the only rule that replicated on VAL); range_so_far_pct >= 5 on bars before the signal
(baked into pop_c's F6 rows); price >= 5; R >= 1% of price; entry <= 14:01; fill = next bar's open <= level x 1.006;
stop = lowest low before entry (touch); exits: HOLD to 15:55 (primary) and 2R-on-close (secondary); contract (c) as
B/score5.py; run_book(12, 4). Cells read here: 2 exits x {PDR>=8, all} = 4, TEST once for both PDR cells.
Outputs: H/F6/f6_pdr_book.md, f6_pdr_trades_<exit>.csv."""
import os, sys, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
D = 'research/fuckup_audit'
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}; ENTRY = 0.25
USE = ['day', 'symbol', 'fam', 'price', 'prev_day_range_pct', 'range_so_far_pct', 'spread_cc_bps', 'next_entry', 'next_entry_m',
       'next_r_pct', 'next_rr_hold', 'next_why_hold', 'next_exit_m_hold', 'next_rr_2r', 'next_why_2r', 'next_exit_m_2r']
parts = []
for ch in pd.read_csv(f'{D}/C/pop_c.csv', usecols=USE, chunksize=400_000, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}):
    parts.append(ch[(ch.fam == 'F6') & ch.next_entry.notna()])
c = pd.concat(parts, ignore_index=True)
c = c[(c.price >= 5) & (c.next_r_pct >= 1) & (c.next_entry_m <= 841) & (c.range_so_far_pct >= 5)].reset_index(drop=True)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str); c['mo'] = c.day.str[:7]
half = 0.5 * (c.spread_cc_bps / 1e4 * 100) / c.next_r_pct.clip(lower=0.05)
for tag in ('hold', '2r'):
    c[f'net_{tag}'] = c[f'next_rr_{tag}'] - ENTRY * half - half * c[f'next_why_{tag}'].map(RATIO).fillna(0.875)
WEEKS = {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
print('rows', len(c), 'PDR>=8 share', round((c.prev_day_range_pct >= 8).mean(), 3), flush=True)


def book(x, tag):
    rows = [(r.day, int(r.next_entry_m), int(getattr(r, f'next_exit_m_{tag}')), r.symbol, getattr(r, f'net_{tag}'),
             getattr(r, f'next_rr_{tag}'), getattr(r, f'next_why_{tag}'), r.wk, r.mo) for r in x.itertuples()]
    t = run_book(rows, 12, 4)
    return pd.DataFrame(t, columns=['day', 'em', 'xm', 'symbol', 'net', 'gross', 'why', 'wk', 'mo'])


def stats(t, sp):
    v = t.net.values; n = len(v); se = v.std(ddof=1) / np.sqrt(n)
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0)
    s = np.sort(v)
    return dict(split=sp, n=n, tpw=round(n / len(WEEKS[sp]), 1), meanR=round(v.mean(), 3), gross=round(t.gross.mean(), 3), t=round(v.mean() / se, 2),
                WR=round((v > 0).mean() * 100, 1), stopP=round((t.why == 'stop').mean() * 100, 1), wkR=round(w.mean(), 2), green=round((w > 0).mean(), 2),
                worst=round(w.min(), 1), ex5=round(s[:int(n * 0.95)].mean(), 3), cap3=round(np.minimum(v, 3).mean(), 3),
                mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 1))


L = ['# F6-PDR — declared book, TEST read once (2026-09-17)', '', open(__file__).read().split('"""')[1], '']
for tag in ('hold', '2r'):
    for lab, x in (('PDR>=8', c[c.prev_day_range_pct >= 8]), ('all', c)):
        rows = []
        for sp in ('TRAIN', 'VAL', 'TEST'):
            t = book(x[x.split == sp], tag); rows.append(stats(t, sp))
            if lab == 'PDR>=8': t.to_csv(f'{D}/H/F6/f6_pdr_trades_{tag}_{sp}.csv', index=False)
        L += [f'## exit {tag} | {lab}', pd.DataFrame(rows).to_string(index=False), '']
        if lab == 'PDR>=8':
            allt = pd.concat([book(x[x.split == sp], tag) for sp in ('TRAIN', 'VAL', 'TEST')])
            m = allt.groupby('mo').net.agg(['sum', 'count']); m['green'] = m['sum'] > 0
            L += ['monthly R: ' + ' | '.join(f"{i} {r['sum']:+.1f} ({int(r['count'])})" for i, r in m.iterrows()), f"months green {int(m.green.sum())}/{len(m)}", '']
open(f'{D}/H/F6/f6_pdr_book.md', 'w').write('\n'.join(L)); print('\n'.join(L[3:]), flush=True)
