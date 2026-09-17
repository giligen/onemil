#!/usr/bin/env python3
"""R9 - implementation E: sensitivities. (1) NASDAQ test tickers (ZVZZT/ZJZZT/ZXZZT) removed; (2) fills whose
bar is not the next CLOCK minute (the live order lives ~20s and is cancelled at the next bar close) removed;
(3) tail tests. Re-books from e_cands.csv, so no tape is re-read."""
import os, sys, re
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book

OUT = 'research/fuckup_audit/H/F6_reconcile'
CC_CSV = 'research/lit_review_2026/cost_curve.csv'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())
TESTTICK = re.compile(r'^Z[A-Z]ZZT$')

d = pd.read_csv(CC_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
d = d[(d.n_q > 0) & d.spread.notna() & (d.price > 0)].copy(); d['bps'] = d.spread / d.price * 1e4
CC = {k: float(v) for k, v in d.groupby(['pb', 'hb']).bps.median().items()}; del d


def cc_bps(p, m):
    pb = '$5-10' if p <= 10 else '$10-20' if p <= 20 else '$20-50' if p <= 50 else '$50-200' if p <= 200 else '$200+'
    hb = ('09:30-09:35' if m <= 575 else '09:35-10:00' if m <= 600 else '10:00-11:00' if m <= 660
          else '11:00-13:00' if m <= 780 else '13:00+')
    return CC.get((pb, hb), np.nan)


K = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}
c = RD(f'{OUT}/e_cands.csv')
c['cc_bps'] = [cc_bps(p, m) for p, m in zip(c.entry, c.entry_m)]
c['half'] = 0.5 * (c.cc_bps / 100.0) / c.r_pct.clip(lower=0.05)
for mode in ('hold', 'r2', 'partial'):
    lc = [sum(float(w) * h * K[t] for w, t in (x.split(':') for x in str(legs).split(';')))
          for legs, h in zip(c[f'{mode}_legs'], c.half)]
    c[f'net_{mode}'] = c[f'{mode}_grossR'] - 0.25 * c.half - np.array(lc)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c['mo'] = c.day.str[:7]
c['is_test_tick'] = c.symbol.str.match(TESTTICK)
c['next_clock'] = c.entry_m == c.sig_m + 1
WEEKS = {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}


def book(x, mode):
    rr = [(r.day, int(r.entry_m), int(getattr(r, f'{mode}_exit_m')), r.symbol, getattr(r, f'net_{mode}'),
           getattr(r, f'{mode}_grossR'), getattr(r, f'{mode}_exit_type'), r.wk, r.mo) for r in x.itertuples()]
    return pd.DataFrame(run_book(rr, 12, 4),
                        columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'gross', 'why', 'wk', 'mo'])


def stats(t, sp, lab, mode):
    v = t.net.values; n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0)
    s = np.sort(v)
    return dict(variant=lab, exit=mode, split=sp, n=n, tpw=round(n / len(WEEKS[sp]), 1),
                meanR=round(v.mean(), 4), t=round(v.mean() / se, 2), WR=round((v > 0).mean() * 100, 1),
                wkR=round(w.mean(), 2), green=round((w > 0).mean(), 2), worst=round(w.min(), 1),
                ex1=round(s[:max(int(round(n * 0.99)), 1)].mean(), 4),
                ex5=round(s[:max(int(round(n * 0.95)), 1)].mean(), 4),
                cap3=round(np.minimum(v, 3).mean(), 4), totR=round(v.sum(), 1),
                mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 1))


VAR = {'E (all)': c,
       'E ex test-tickers': c[~c.is_test_tick],
       'E ex test-tickers, fill=next clock minute': c[(~c.is_test_tick) & c.next_clock]}
rows = []
mon = {}
for lab, x in VAR.items():
    for mode in ('hold', 'r2', 'partial'):
        bs = []
        for sp in ('TRAIN', 'VAL', 'TEST'):
            t = book(x[x.split == sp], mode); bs.append(t)
            rows.append(stats(t, sp, lab, mode))
        at = pd.concat(bs)
        mon[(lab, mode)] = at.groupby('mo').net.agg(['sum', 'count'])
        if lab == 'E ex test-tickers':
            at.to_csv(f'{OUT}/e_trades_extick_{mode}.csv', index=False)
st = pd.DataFrame(rows)
st.to_csv(f'{OUT}/e_stats.csv', index=False)

L = ['# R9 - implementation E sensitivities', '',
     f'E candidates: {len(c)}  | test-ticker rows: {int(c.is_test_tick.sum())} '
     f'| fill NOT the next clock minute: {int((~c.next_clock).sum())} ({(~c.next_clock).mean()*100:.1f}%)', '']
for mode in ('hold', 'r2', 'partial'):
    L += [f'## exit {mode}', st[st.exit == mode].drop(columns=['exit']).to_string(index=False), '']
for lab in VAR:
    for mode in ('hold', 'r2', 'partial'):
        m = mon[(lab, mode)]
        L += [f'monthly [{lab} | {mode}]: ' + ' | '.join(f"{i} {r['sum']:+.1f} ({int(r['count'])})" for i, r in m.iterrows())
              + f"   green {int((m['sum'] > 0).sum())}/{len(m)}"]
    L.append('')
open(f'{OUT}/r9_e_sensitivity.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
