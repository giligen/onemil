#!/usr/bin/env python3
"""Stage H / F6 — METHOD steps 2 and 3: the candidate filters and the stack, TRAIN only.

Every candidate is a BUCKET VETO on a feature that is known at the CLOSE of the signal bar, i.e.
before the engine's next-open order exists.  Filters are applied BEFORE `run_book(12,4)`: this book
is first-come (not a ranked 09:35 batch like ORB), so a vetoed signal simply is not ordered and the
slot stays open for the next signal that arrives — that IS the live semantics, and it is why there
is no "no-refill" question here.

Cuts are coarse (round numbers or quartile boundaries rounded), never a fine sweep, and EVERY
(feature x cut) cell is written to `h2_cells.csv` so the multiplicity denominator is honest.

VAL and TEST are NOT read by this script.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h2_filters.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C

H = C.H
V = 'hold'
NET = f'net_{V}'
pd.set_option('display.width', 250)

# --- the candidate grid.  name -> (prose, lambda on the frame)
CUTS = []


def add(feat, name, fn, prose):
    CUTS.append(dict(feat=feat, name=name, fn=fn, prose=prose))


add('vwap_dist_pct', 'vwap>0', lambda d: d.vwap_dist_pct > 0,
    'the reclaim level is ABOVE the running VWAP of the day so far')
add('vwap_dist_pct', 'vwap>0.5', lambda d: d.vwap_dist_pct > 0.5, 'level > 0.5% above the day VWAP')
add('vwap_dist_pct', 'vwap>1', lambda d: d.vwap_dist_pct > 1.0, 'level > 1% above the day VWAP')
add('prev_day_range_pct', 'pdr>=5', lambda d: d.prev_day_range_pct >= 5, 'prior day range >= 5%')
add('prev_day_range_pct', 'pdr>=8', lambda d: d.prev_day_range_pct >= 8, 'prior day range >= 8% (the shipped ORB PDR veto)')
add('prev_day_range_pct', 'pdr>=15', lambda d: d.prev_day_range_pct >= 15, 'prior day range >= 15%')
add('close_over_level_pct', 'col<=0.25', lambda d: d.close_over_level_pct <= 0.25, 'signal bar closed <= 0.25% above the level')
add('close_over_level_pct', 'col<=0.5', lambda d: d.close_over_level_pct <= 0.5, 'signal bar closed <= 0.5% above the level')
add('close_over_level_pct', 'col<=1', lambda d: d.close_over_level_pct <= 1.0, 'signal bar closed <= 1% above the level')
add('close_over_level_pct', 'col<=2', lambda d: d.close_over_level_pct <= 2.0, 'signal bar closed <= 2% above the level')
add('sig_body_pct', 'body<0.5', lambda d: d.sig_body_pct < 0.5, 'signal bar body < +0.5%')
add('sig_body_pct', 'body<1', lambda d: d.sig_body_pct < 1.0, 'signal bar body < +1%')
add('sig_body_pct', 'body<2', lambda d: d.sig_body_pct < 2.0, 'signal bar body < +2%')
add('sig_close_pos', 'clpos<0.5', lambda d: d.sig_close_pos.fillna(0.5) < 0.5, 'signal bar closed in the lower half of its own range')
add('sig_close_pos', 'clpos<0.75', lambda d: d.sig_close_pos.fillna(0.5) < 0.75, 'signal bar closed below the top quarter of its range')
add('close_confirm', 'noconfirm', lambda d: d.close_confirm == 0, 'the signal bar did NOT close above the level (wick reclaim)')
add('spy_ret', 'spy>=-0.3', lambda d: d.spy_ret.fillna(0) >= -0.3, 'SPY not more than 0.3% below its own 09:30 open at the signal minute')
add('spy_ret', 'spy>=0', lambda d: d.spy_ret.fillna(0) >= 0, 'SPY at or above its 09:30 open at the signal minute')
add('consol_bars', 'cb<41', lambda d: d.consol_bars < 41, 'fewer than 41 bars since the day low')
add('consol_bars', 'cb<92', lambda d: d.consol_bars < 92, 'fewer than 92 bars since the day low')
add('rv_adv', 'rv<0.215', lambda d: d.rv_adv.fillna(0) < 0.215, 'cumulative volume below 21.5% of ADV20')
add('price', 'px<=50', lambda d: d.price <= 50, 'level at or below $50')
add('range_so_far_pct', 'rsf>=8', lambda d: d.range_so_far_pct >= 8, 'pre-signal day range >= 8% (tighter than the book floor)')
add('n_touches', 'touch>=1', lambda d: d.n_touches >= 1, 'the level was touched at least once before the signal bar')
add('next_entry_m', 'after0935', lambda d: d.next_entry_m >= 575, 'no entries in the first five minutes')
add('dist_open_pct', 'dopen>=2.2', lambda d: d.dist_open_pct >= 2.2, 'level at least 2.2% above the 09:30 open')

STACK = ['pdr>=8', 'vwap>0', 'col<=1']     # frozen in REPORT.md step 4 before VAL is read


def era(day):
    return np.where(day < '2025-07-01', 'H1', 'H2')


def prep(pop):
    d = C.load(pop)
    d['close_over_level_pct'] = (d.sig_c / d.level - 1.0) * 100.0
    return d, C.weeks_of(pop)


def row(tag, d, wk, split='TRAIN'):
    st, t = C.book_stats(d, split, wk, V)
    if st is None:
        return None
    t = t.copy()
    t['era'] = era(t.day.values)
    h1, h2 = t[t.era == 'H1'], t[t.era == 'H2']
    st = dict(cell=tag, **st)
    st['H1'] = round(float(h1[NET].mean()), 4) if len(h1) > 15 else np.nan
    st['H2'] = round(float(h2[NET].mean()), 4) if len(h2) > 15 else np.nan
    st['bothpos'] = bool(st['H1'] > 0 and st['H2'] > 0)
    return st, t


def fmt(r, keys):
    def g(k):
        v = r.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ''
        return f'{v:+.4f}' if k in ('meanR', 'grossR', 'H1', 'H2', 'ex1', 'ex5', 'cap3') else str(v)
    return '| ' + ' | '.join(g(k) for k in keys) + ' |'


KEYS = ['cell', 'n', 'tpw', 'meanR', 't', 'WR', 'stopP', 'wkR', 'green', 'worstWk', 'mdd',
        'H1', 'H2', 'ex5', 'cap3']
HDR = ['cell', 'n', 'tr/wk', 'net R', 't', 'WR%', 'stop%', 'wk R', 'green', 'worst wk', 'MDD',
       'H1', 'H2', 'ex-top5%', 'cap +3R']


def main():
    L = ['# Stage H / F6 — steps 2 and 3: candidate filters and the stack (TRAIN only)', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Book: F6 red-to-green, next-open fill, hold to 15:55 with the touch stop, contract (c), '
         'run_book(12,4), `range_so_far_pct >= 5`, 09:30-14:01, price >= $5, R >= 1% of price. '
         'Filters are applied BEFORE the book (first-come semantics: a vetoed signal is never ordered '
         'and the slot stays open for the next arrival). VAL and TEST are not read here.', '']
    allcells = []
    for pop in ('Q', 'P'):
        d, wk = prep(pop)
        x = C.scoreable(d, V, floor=True)
        base, bt = row('BASELINE (no filter)', x, wk)
        L += [f'## universe {pop}', '', '### step 2 — every candidate cut, one at a time', '',
              '| ' + ' | '.join(HDR) + ' |', '|' + '|'.join(['---'] * len(HDR)) + '|', fmt(base, KEYS)]
        allcells.append(dict(pop=pop, kind='baseline', **base))
        for c in CUTS:
            m = c['fn'](x)
            r = row(f'{c["name"]}  ({m.mean():.0%} kept)', x[m], wk)
            if r is None:
                L.append(f'| {c["name"]} | (no book) | | | | | | | | | | | | | |')
                continue
            st, _ = r
            L.append(fmt(st, KEYS))
            allcells.append(dict(pop=pop, kind='single', feat=c['feat'], cut=c['name'],
                                 kept=round(float(m.mean()), 3), **st))
        L.append('')

        # ---------------- step 3 — the stack, cumulative
        L += ['### step 3 — the stack, cumulative (the order is the frozen order)', '',
              '| ' + ' | '.join(HDR) + ' |', '|' + '|'.join(['---'] * len(HDR)) + '|', fmt(base, KEYS)]
        cur = x
        mask = pd.Series(True, index=x.index)
        by = {c['name']: c for c in CUTS}
        for i, nm in enumerate(STACK, 1):
            mask &= by[nm]['fn'](x)
            r = row(f'+{nm}  ({mask.mean():.0%} kept)', x[mask], wk)
            if r is None:
                L.append(f'| +{nm} | (no book) |')
                continue
            st, tt = r
            L.append(fmt(st, KEYS))
            allcells.append(dict(pop=pop, kind=f'stack{i}', cut=nm, kept=round(float(mask.mean()), 3), **st))
            cur = x[mask]
        # leave-one-out of the final stack
        L += ['', '### step 3b — leave-one-out of the frozen stack', '',
              '| ' + ' | '.join(HDR) + ' |', '|' + '|'.join(['---'] * len(HDR)) + '|']
        for drop in STACK:
            m2 = pd.Series(True, index=x.index)
            for nm in STACK:
                if nm != drop:
                    m2 &= by[nm]['fn'](x)
            r = row(f'stack minus {drop}', x[m2], wk)
            if r:
                L.append(fmt(r[0], KEYS))
                allcells.append(dict(pop=pop, kind='loo', cut=drop, **r[0]))
        # the vetoed cohort itself, booked on its own
        L += ['', '### step 3c — the vetoed cohort, booked on its own (what the stack removes)', '',
              '| ' + ' | '.join(HDR) + ' |', '|' + '|'.join(['---'] * len(HDR)) + '|']
        keep = pd.Series(True, index=x.index)
        for nm in STACK:
            keep &= by[nm]['fn'](x)
        r = row('REMOVED cohort', x[~keep], wk)
        if r:
            L.append(fmt(r[0], KEYS))
            allcells.append(dict(pop=pop, kind='removed', **r[0]))
        L += ['', '### step 3d — per month, the stacked book vs the baseline', '',
              '| month | base n | base R | stacked n | stacked R |', '|---|---:|---:|---:|---:|']
        sb = C.book(x[keep], 'TRAIN', wk, V)
        a = bt.assign(mon=bt.day.str[:7]).groupby('mon')[NET].agg(['size', 'sum'])
        b = sb.assign(mon=sb.day.str[:7]).groupby('mon')[NET].agg(['size', 'sum'])
        for mo in sorted(set(a.index) | set(b.index)):
            L.append(f'| {mo} | {int(a["size"].get(mo, 0))} | {a["sum"].get(mo, 0):+.1f} | '
                     f'{int(b["size"].get(mo, 0))} | {b["sum"].get(mo, 0):+.1f} |')
        L.append('')
        sb.to_csv(f'{H}/h2_stack_train_{pop}.csv', index=False)

    pd.DataFrame(allcells).to_csv(f'{H}/h2_cells.csv', index=False)
    L += ['## cells looked at in step 2/3', '',
          f'{len(CUTS)} single cuts x 2 universes = {len(CUTS)*2}, plus {len(STACK)*2} cumulative stack '
          f'steps, {len(STACK)*2} leave-one-out, 2 removed-cohort and 2 baseline books = '
          f'**{len(CUTS)*2 + len(STACK)*4 + 4} cells in this step**. Step 1 looked at 561 bucket cells.', '']
    open(f'{H}/h2_filters.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
