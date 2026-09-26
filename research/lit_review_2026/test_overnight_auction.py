#!/usr/bin/env python3
"""Re-test of the overnight family with an AUCTION-appropriate cost. A close→open trade executes MOC then MOO: it pays the
auction clearing price, not the quoted spread, so charging 12–40 bps round trip (the earlier runs) is wrong by construction.
Here the round trip is charged at 2, 5 and 10 bps and the book is run at 4, 10, 25 and 50 names so the slot limit is visible.
Rules (all long-only, all inputs known at the 15:59 close): V = top-decile volume/ADV20; H = close at a new 252-day high on
>= 1.5x volume; B = both. Universe: close >= $5, 20-day dollar volume >= $10M (auction depth). Splits fixed. → overnight_auction.md"""
import os, re, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet',
                    columns=['symbol', 'bar_date', 'close', 'high', 'dvol20', 'adv20', 'ret_on_next', 'ret_id_next', 'vol_ratio', 'high52'])
bad = [c for c in d.symbol.cat.categories if re.match(r'^Z[VWX]ZZ|^ZZ', str(c))] if hasattr(d.symbol, 'cat') else []
if bad: d = d[~d.symbol.isin(bad)]
d = d[(d.close >= 5) & (d.dvol20 >= 1e7) & d.ret_on_next.notna() & (d.ret_on_next.abs() <= 0.5) & d.vol_ratio.notna()]
d['split'] = np.where(d.bar_date < '2026-01-01', 'TRAIN', np.where(d.bar_date < '2026-06-01', 'VAL', 'TEST'))
d['wk'] = pd.to_datetime(d.bar_date).dt.to_period('W-FRI').astype(str)
thr = d[d.split == 'TRAIN'].vol_ratio.quantile(0.9)
sets = {'V volume shock': d[d.vol_ratio >= thr].assign(score=lambda x: x.vol_ratio),
        'H new 252d high': d[d.high52.notna() & (d.close >= d.high52) & (d.vol_ratio >= 1.5)].assign(score=lambda x: x.vol_ratio),
        'B both': d[(d.vol_ratio >= thr) & d.high52.notna() & (d.close >= d.high52)].assign(score=lambda x: x.vol_ratio)}
L = ['# Overnight family with auction costs (close→open), by book size and cost', '',
     f'universe: close >= $5, 20-day dollar volume >= $10M; volume-shock decile cut on TRAIN at {thr:.2f}x ADV20', '']
for name, s in sets.items():
    L.append(f'## {name} — n available per day: ' + ', '.join(f'{sp} {len(s[s.split==sp])/s[s.split==sp].bar_date.nunique():.0f}' for sp in ('TRAIN','VAL','TEST')))
    for N in (4, 10, 25, 50):
        row = [f'  top-{N:<3d}']
        for sp in ('TRAIN', 'VAL', 'TEST'):
            x = s[s.split == sp].sort_values(['bar_date', 'score'], ascending=[True, False]).groupby('bar_date').head(N)
            if len(x) < 50: row.append(f'{sp} n/a'); continue
            gross = x.ret_on_next.mean(); t = gross / (x.ret_on_next.std() / np.sqrt(len(x)))
            nw = x.wk.nunique(); per_wk = x.groupby('wk').ret_on_next.mean().mean() * 5
            row.append(f'{sp} gross {gross*1e4:+5.1f} (t {t:+4.1f}) net@2bp {(gross-0.0002)*1e4:+5.1f} @5bp {(gross-0.0005)*1e4:+5.1f} @10bp {(gross-0.0010)*1e4:+5.1f}')
        L.append(' | '.join(row))
    L.append('')
# weekly R at $100 risk for the surviving configuration: N names x $ per name, overnight move sd
best = sets['V volume shock']
L += ['## what it is worth, honestly', '']
for N in (25, 50):
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = best[best.split == sp].sort_values(['bar_date', 'score'], ascending=[True, False]).groupby('bar_date').head(N)
        if len(x) < 50: continue
        w = x.groupby('wk').ret_on_next.mean() - 0.0005
        cap = 60000; per_name = cap / N
        L.append(f'N={N} {sp}: mean weekly return on the book {w.mean()*100*5:+.2f}% (5 sessions) → ${w.mean()*5*cap:+,.0f}/week on $60K at 1x, weekly sd ${w.std()*np.sqrt(5)*cap:,.0f}, weeks green {int((w>0).sum())}/{len(w)}')
open('research/lit_review_2026/overnight_auction.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
