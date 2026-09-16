#!/usr/bin/env python3
"""score3 — the corrected day-trading scorer. Three bugs in score2 are fixed here; each is stated so the numbers can be
compared with the earlier (void) runs.

BUG 1 — the causal filter was three times too strict. The study universe is "day range >= 5%", which is known only at the
close, so F5-F10 signals need a causal guarantee of membership. score2 used `dist_open_pct >= 5` (entry 5% above the open).
The correct and much weaker guarantee is `range_so_far_pct >= 5`: if the high-low range UP TO THE SIGNAL BAR is already 5%,
the full day's range is at least 5% whatever happens next. Both are causal; the second keeps 734K rows instead of 229K.

BUG 2 — the entry cost was counted twice. Pass 1 already fills at `level x 1.003` (30 bps through the level, i.e. an ask-ish
fill). score2 then charged another half-spread (20 bps) on entry. Here the entry slip in the price IS the entry cost, and
only the exit pays a half-spread, and only when the exit is not the resting target.

BUG 3 — only the +2R close-fill exit was scored. The hold-to-close exit with a -1R stop (rr_e4, the spec in the day-trading
literature) was never run through the book. Both are scored here.

Also: the book is run at 4 slots (the live constraint) AND 20 slots, because at 4 names the standard error per trade is
larger than the effect sizes the literature reports. Splits fixed, TEST read once. → score3_tables.md"""
import os, sys, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
D = 'research/bf_zero2'
MIN_TPW, MIN_N, MAX_SPLITS, LAST_ENTRY, PRICE_FLOOR = 5.0, 300, 3, 840, 5.0
NUM = ['entry_m', 'exit_m_e1c', 'price', 'r_pct', 'dist_open_pct', 'range_so_far_pct', 'rv_profile', 'rv_clock', 'gap_pct', 'adv20', 'bar_vol_x',
       'above_vwap', 'prev_range_pct', 'pm_vol', 'pm_high_pct', 'bars_per_min', 'spy_5m_ret', 'is_wrapper', 'coh_by_t', 'rr_e1c', 'rr_e4', 'dist_20d_high_pct']
c = pd.read_csv(f'{D}/candidates_full.csv', usecols=lambda k: k in set(['day', 'symbol', 'fam', 'cfg', 'why_e1c', 'why_e4'] + NUM),
                dtype={'symbol': 'category', 'day': 'category', 'fam': 'category', 'cfg': 'category', 'why_e1c': 'category', 'why_e4': 'category'},
                keep_default_na=False, na_values=[''], low_memory=True)
c['day'] = c.day.astype(str); c['symbol'] = c.symbol.astype(str)
c = c[(c.price >= PRICE_FLOOR) & (c.entry_m <= LAST_ENTRY + 1) & (c.r_pct >= 1.0)]
NEED = ~c.fam.isin(['F1', 'F2', 'F3', 'F4'])
before = int(NEED.sum())
c = c[~(NEED & ~(c.range_so_far_pct >= 5))].reset_index(drop=True)
print(f'BUG 1 fix: causal condition range_so_far >= 5% keeps {int((~c.fam.isin(["F1","F2","F3","F4"])).sum()):,} of {before:,} F5-F10 rows '
      f'(score2 kept 229,265 with dist_open >= 5%)', flush=True)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c['key'] = c.fam.astype(str) + ' ' + c.cfg.astype(str)
half = 0.5 * 0.40 / c.r_pct.clip(lower=0.05)                     # BUG 2 fix: 20 bps of half-spread, EXIT only, in R units
c['net_e1c'] = c.rr_e1c - np.where(c.why_e1c == 'target', 0.0, half)
c['net_e4'] = c.rr_e4 - half                                     # hold-to-close always exits at the market
c['exit_m_e4'] = 955
WEEKS = {s: c[c.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}
print(f'rows {len(c):,} | keys {c.key.nunique()} | weeks {WEEKS}', flush=True)


def book(d, exit_tag, split, N):
    col, xm = f'net_{exit_tag}', f'exit_m_{exit_tag}'
    x = d[(d.split == split) & d[col].notna()]
    if len(x) < 50: return None
    rows = [(r.day, int(r.entry_m), int(getattr(r, xm)), r.symbol, getattr(r, col), r.wk) for r in x.itertuples()]
    t = pd.DataFrame(run_book(rows, N, N), columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'wk'])
    if not len(t): return None
    nw = WEEKS[split]; w = t.groupby('wk').net.sum().reindex(sorted(c[c.split == split].wk.unique())).fillna(0)
    return dict(n=len(t), tpw=round(len(t) / nw, 1), meanR=round(t.net.mean(), 3), t=round(t.net.mean() / (t.net.std() / np.sqrt(len(t))), 2),
                WR=round((t.net > 0).mean() * 100, 1), wkR=round(float(w.mean()), 1), green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1))


L = ['# score3 — corrected day-trading scorer (causal condition fixed, entry cost no longer double-counted, hold-to-close scored)', '']
rows = []
for key, dk in c.groupby('key', observed=True):
    for tag in ('e4', 'e1c'):
        for N in (4, 20):
            for sp in ('TRAIN', 'VAL', 'TEST'):
                r = book(dk, tag, sp, N)
                if r: rows.append(dict(key=key, exit={'e4': 'hold-to-close -1R', 'e1c': '+2R close-fill'}[tag], slots=N, split=sp, **r))
T1 = pd.DataFrame(rows); T1.to_csv(f'{D}/score3_step1.csv', index=False)
pd.set_option('display.width', 250)
tr = T1[(T1.split == 'TRAIN')].sort_values('wkR', ascending=False)
L += ['## STEP 1 — every family-config x exit x book size, TRAIN, net of the EXIT half-spread only', '', tr.head(30).to_string(index=False), '']
print(tr.head(25).to_string(index=False), flush=True)
keep = tr[(tr.meanR > 0) & (tr.tpw >= MIN_TPW)]
L.append(f'TRAIN positive with >= {MIN_TPW} trades/week: {len(keep)} of {len(tr)}')
# carry every TRAIN-positive config to VAL and TEST without any further tuning
res = []
for _, k in keep.iterrows():
    tag = 'e4' if k.exit.startswith('hold') else 'e1c'; dk = c[c.key == k.key]
    va = book(dk, tag, 'VAL', int(k.slots)); te = book(dk, tag, 'TEST', int(k.slots))
    res.append(dict(key=k.key, exit=k.exit, slots=k.slots, TRAIN_meanR=k.meanR, TRAIN_wkR=k.wkR, TRAIN_t=k.t,
                    VAL_meanR=va['meanR'] if va else None, VAL_wkR=va['wkR'] if va else None, VAL_t=va['t'] if va else None, VAL_green=va['green'] if va else None,
                    TEST_meanR=te['meanR'] if te else None, TEST_wkR=te['wkR'] if te else None, TEST_t=te['t'] if te else None, TEST_green=te['green'] if te else None))
R = pd.DataFrame(res).sort_values('TRAIN_wkR', ascending=False)
L += ['', '## STEPS 2-3 — the same configs on VAL and TEST, no re-tuning', '', R.to_string(index=False)]
print('\n' + R.head(25).to_string(index=False), flush=True)
surv = R[(R.VAL_meanR > 0) & (R.TEST_meanR > 0)]
L += ['', f'## positive on all three splits: {len(surv)}', surv.to_string(index=False) if len(surv) else '(none)']
print(f'\npositive on all three splits: {len(surv)}', flush=True)
if len(surv): print(surv.to_string(index=False), flush=True)
open(f'{D}/score3_tables.md', 'w').write('\n'.join(L)); print('DONE', flush=True)
