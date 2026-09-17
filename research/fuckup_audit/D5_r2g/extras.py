"""D5 stage 3 — the two questions the main tables raised.

(i) Is the sequence effect (1st entry of the day) anything other than time of
    day?  seq x hour cross-tab, and "first two entries" restricted to 09:xx.
(ii) Where do the worse-than-1R losses come from?  The stop-fill model books
    min(stop, bar open) x 0.999, so a tiny R on a gapping thin tape becomes a
    multi-R loss.  Stop distance is known AT THE FILL, so a floor is a
    live-computable rule.  Declared cells: stop_dist_pct >= 2 / 3 / 4.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []
CELLS = 0


def w(s=''):
    OUT.append(s)


def main():
    global CELLS
    d = pd.read_csv(os.path.join(HERE, 'trades_classed.csv'), keep_default_na=False)
    d['hour'] = d.entry_min // 60
    d['seq_b'] = np.where(d.seq >= 4, '4+', d.seq.astype(str))
    d['loser'] = d.net_R < 0
    MONTHS = {'TRAIN': 12.0, 'VAL': 5.0, 'TEST': 3.13}

    w('---')
    w()
    w('## 6. Two follow-ups the tables forced')
    w()
    w('### 6a. Is "first entry of the day" anything but the 09:3x bar?')
    w()
    w('The book is FIRST-COME (12/day, 4 concurrent), so `seq` is just entry '
      'order in time. Cross-tab of net R per trade, seq ordinal x entry hour, '
      'ALL splits pooled and then per split for the 09:xx column only.')
    w()
    w('| ordinal | 09:xx n | 09:xx R/tr | 10:xx n | 10:xx R/tr | 11:xx+ n | 11:xx+ R/tr |')
    w('|---|---|---|---|---|---|---|')
    for v in ['1', '2', '3', '4+']:
        row = [v]
        for lo, hi in ((9, 9), (10, 10), (11, 23)):
            g = d[(d.seq_b == v) & (d.hour >= lo) & (d.hour <= hi)]
            row += [str(len(g)), f'{g.net_R.mean():+.3f}' if len(g) else '-']
            CELLS += 1
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('Within the 09:xx hour only, per split:')
    w()
    w('| ordinal | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |')
    w('|---|---|---|---|---|---|---|')
    for v in ['1', '2', '3', '4+']:
        row = [v]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & (d.seq_b == v) & (d.hour == 9)]
            row += [str(len(g)), f'{g.net_R.mean():+.3f}' if len(g) else '-']
            CELLS += 1
        w('| ' + ' | '.join(row) + ' |')
    w()

    w('### 6b. The worse-than-1R losses')
    w()
    w('A stop exit books `min(stop, bar open) x 0.999`. When the next bar opens '
      'BELOW the stop, the loss is bigger than the 1R the stop nominally risked. '
      'The rule floor is `R >= 1% of entry`, so a 1.1%-wide stop on a thin tape '
      'turns a 4% gap-down bar into a -3.6R print.')
    w()
    bad = d[d.net_R <= -1.5]
    w(f'- trades with net R <= -1.5: **{len(bad)}** of {len(d)} ({len(bad)/len(d)*100:.1f}%), '
      f'**{bad.net_R.sum():+.1f} R** — {bad.net_R.sum()/d[d.loser].net_R.sum()*100:.1f}% '
      f'of ALL loser R, against a book total of {d.net_R.sum():+.1f} R.')
    w(f'- their median stop distance **{bad.stop_dist_pct.median():.2f}%** of entry '
      f'vs **{d.stop_dist_pct.median():.2f}%** for the book; median 5-min $ volume '
      f'**${bad.dollar_vol_5m.median():,.0f}** vs **${d.dollar_vol_5m.median():,.0f}**.')
    w(f'- {int((bad.stop_dist_pct < 3).sum())} of {len(bad)} had a stop closer than 3% '
      f'of the entry price; {int((bad.dollar_vol_5m < 200000).sum())} were thin.')
    w()
    w('| stop distance band | n | R/trade | total R | mean net R of its losers |')
    w('|---|---|---|---|---|')
    for lo, hi, lab in ((0, 2, '< 2%'), (2, 3, '2-3%'), (3, 4, '3-4%'),
                        (4, 6, '4-6%'), (6, 9, '6-9%'), (9, 1e9, '>= 9%')):
        g = d[(d.stop_dist_pct >= lo) & (d.stop_dist_pct < hi)]
        gl = g[g.loser]
        w(f'| {lab} | {len(g)} | {g.net_R.mean():+.3f} | {g.net_R.sum():+.1f} | '
          f'{gl.net_R.mean():+.3f} |')
        CELLS += 1
    w()
    w('**Declared cell: a minimum stop distance** (known at the fill — the stop is '
      'the running low, the entry is the fill). Subset filter, not a re-book.')
    w()
    w('| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | '
      'TEST n | TEST R/tr | TEST $/mo |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for thr in (0.0, 2.0, 3.0, 4.0):
        lab = 'as booked' if thr == 0 else f'stop distance >= {thr:.0f}% of entry'
        row = [lab]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & (d.stop_dist_pct >= thr)]
            row += [str(len(g)), f'{g.net_R.mean():+.4f}',
                    f'${g.net_R.sum()*300/MONTHS[s]:,.0f}']
            CELLS += 1
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('**Declared cell: stop floor combined with the sequence rule.**')
    w()
    w('| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | '
      'TEST n | TEST R/tr | TEST $/mo |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    combos = [('seq <= 2', d.seq <= 2),
              ('stop >= 3%', d.stop_dist_pct >= 3.0),
              ('seq <= 2 AND stop >= 3%', (d.seq <= 2) & (d.stop_dist_pct >= 3.0)),
              ('seq <= 2 AND stop >= 3% AND 09:xx',
               (d.seq <= 2) & (d.stop_dist_pct >= 3.0) & (d.hour == 9))]
    for lab, m in combos:
        row = [lab]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & m]
            row += [str(len(g)), f'{g.net_R.mean():+.4f}',
                    f'${g.net_R.sum()*300/MONTHS[s]:,.0f}']
            CELLS += 1
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('Tail check on the surviving rule (`seq <= 2 AND stop >= 3%`): mean net R '
      'with the top 1% and top 5% of trades removed, and with winners capped at +3R.')
    w()
    w('| split | n | R/tr | ex-top-1% | ex-top-5% | winners capped +3R |')
    w('|---|---|---|---|---|---|')
    m = (d.seq <= 2) & (d.stop_dist_pct >= 3.0)
    for s in ('TRAIN', 'VAL', 'TEST'):
        g = d[(d.split == s) & m].sort_values('net_R')
        n = len(g)
        k1, k5 = max(1, int(round(n * .01))), max(1, int(round(n * .05)))
        w(f'| {s} | {n} | {g.net_R.mean():+.4f} | {g.net_R.iloc[:n-k1].mean():+.4f} | '
          f'{g.net_R.iloc[:n-k5].mean():+.4f} | '
          f'{np.minimum(g.net_R.values, 3.0).mean():+.4f} |')
        CELLS += 3
    w()
    w('Monthly net R of `seq <= 2 AND stop >= 3%`:')
    w()
    g = d[m].copy()
    g['mon'] = g.day.str[:7]
    mm = g.groupby('mon').agg(n=('net_R', 'size'), R=('net_R', 'sum'))
    w('| ' + ' | '.join(mm.index) + ' |')
    w('|' + '---|' * len(mm))
    w('| ' + ' | '.join(f'{r.R:+.1f}' for r in mm.itertuples()) + ' |')
    w()
    w(f'Months green **{int((mm.R>0).sum())}/{len(mm)}**; worst month '
      f'**{mm.R.min():+.1f} R** (= ${mm.R.min()*300:,.0f} at $300 risk); '
      f'mean month **{mm.R.mean():+.2f} R** (${mm.R.mean()*300:,.0f}).')
    w()
    print('EXTRA CELLS:', CELLS)
    open(os.path.join(HERE, 'extras.md'), 'w').write('\n'.join(OUT))
    open(os.path.join(HERE, 'cells.txt'), 'a').write(f'extras (6a/6b): {CELLS}\n')


if __name__ == '__main__':
    main()
