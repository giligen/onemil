#!/usr/bin/env python3
"""Stage H / F6 — METHOD steps 4 and 5: VAL once, then TEST once, on the FROZEN stack only.

The stack is frozen in writing in `H/F6/REPORT.md` §4 BEFORE this script is run.  It is read from
FROZEN below, which must match the report verbatim.

  step 4 (VAL)   the frozen stack applied to VAL, per-filter contribution, the vetoed cohort on VAL,
                 and the pass rule: VAL mean net R improves vs the unfiltered VAL book AND the vetoed
                 cohort is negative on VAL AND VAL mean net R > 0 with >= 55% of weeks green.
  step 5 (TEST)  only with H_READ_TEST=1 and only after step 4 is written down: week-by-week,
                 per-month, tail tests, the search-adjusted permutation p over every BOOK cell this
                 stage looked at, and the money line at $100 and $400 risk.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h3_val_test.py
       H_READ_TEST=1 ... (only after the freeze AND step 4 are in REPORT.md)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C
import h2_filters as F
import h2b_stacks as S

H = C.H
V, NET = 'hold', 'net_hold'
READ_TEST = os.environ.get('H_READ_TEST') == '1'
NPERM = int(os.environ.get('H_PERM', '500'))

# ----------------------------------------------------------------- THE FROZEN STACK (REPORT.md §4)
FROZEN = ['pdr>=8', 'vwap>0', 'col<=0.25']
PRIMARY = 'Q'


def perm_pvalue(cells, nperm, seed=11):
    """Search-adjusted permutation p, exactly Stage C/E's: flip the sign of every trade of a random
    half of the DAYS (day blocks keep the within-day correlation), recompute each cell's TRAIN t,
    take the MAX over all cells; p = share of draws whose max exceeds the observed max."""
    rng = np.random.default_rng(seed)
    obs = max(c['t'] for c in cells.values())
    mx = []
    for _ in range(nperm):
        best = -np.inf
        for c in cells.values():
            flip = rng.integers(0, 2, len(c['bydayN'])) * 2 - 1
            v = np.repeat(flip, c['bydayN']) * c['trades']
            sd = v.std(ddof=1)
            if sd > 0:
                best = max(best, v.mean() / (sd / np.sqrt(len(v))))
        mx.append(best)
    mx = np.array(mx)
    return float((mx >= obs).mean()), obs, float(np.percentile(mx, 95))


def main():
    by = {c['name']: c for c in F.CUTS}
    L = ['# Stage H / F6 — steps 4 and 5: VAL, then TEST, on the frozen stack', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         '**The frozen stack** (REPORT.md §4, written before this ran): `' + '` + `'.join(FROZEN) + '`.',
         f'Primary universe **{PRIMARY}**; the other is reported alongside as the twin.', '']
    cells, out = {}, []
    for pop in ('Q', 'P'):
        d, wk = F.prep(pop)
        x = C.scoreable(d, V, floor=True)
        m = pd.Series(True, index=x.index)
        masks = {}
        for k in FROZEN:
            m &= by[k]['fn'](x)
            masks[k] = m.copy()
        splits = ['TRAIN', 'VAL'] + (['TEST'] if READ_TEST else [])
        L += [f'## universe {pop}', '', '| split | book | ' + ' | '.join(F.HDR[1:]) + ' |',
              '|---|---|' + '|'.join(['---'] * (len(F.HDR) - 1)) + '|']
        for sp in splits:
            for tag, xx in (('baseline', x), ('FROZEN stack', x[m]), ('vetoed cohort', x[~m])):
                r = F.row(tag, xx, wk, sp)
                if r is None:
                    L.append(f'| {sp} | {tag} | (no book) |')
                    continue
                st, tt = r
                L.append(f'| {sp} | {tag} | ' + F.fmt(st, F.KEYS)[2:])
                out.append(dict(pop=pop, split=sp, book=tag, **st))
                if sp == 'TEST' and tag == 'FROZEN stack' and pop == PRIMARY:
                    tt.to_csv(f'{H}/h3_book_TEST_{pop}.csv', index=False)
                if sp == 'VAL' and tag == 'FROZEN stack':
                    tt.to_csv(f'{H}/h3_book_VAL_{pop}.csv', index=False)
                if sp == 'TRAIN' and tag == 'FROZEN stack':
                    tt.to_csv(f'{H}/h3_book_TRAIN_{pop}.csv', index=False)
            # per-filter contribution, cumulative
            for k in FROZEN:
                r = F.row(f'cum +{k}', x[masks[k]], wk, sp)
                if r:
                    L.append(f'| {sp} | cum +{k} | ' + F.fmt(r[0], F.KEYS)[2:])
                    out.append(dict(pop=pop, split=sp, book=f'cum +{k}', **r[0]))
            # each filter's own vetoed bucket, on this split
            for k in FROZEN:
                r = F.row(f'bucket vetoed by {k}', x[~by[k]['fn'](x)], wk, sp)
                if r:
                    L.append(f'| {sp} | vetoed by {k} | ' + F.fmt(r[0], F.KEYS)[2:])
                    out.append(dict(pop=pop, split=sp, book=f'vetoed by {k}', **r[0]))
        L.append('')
        # week and month tables on each split for the frozen stack
        for sp in splits:
            t = C.book(x[m], sp, wk, V)
            if t is None:
                continue
            w = t.groupby('wk')[NET].sum().reindex(wk[sp]).fillna(0.0)
            mo = t.assign(mon=t.day.str[:7]).groupby('mon')[NET].agg(['size', 'sum', 'mean']).round(3)
            L += [f'### {pop} / {sp} — the frozen book, per month', '',
                  '| month | n | R | mean |', '|---|---:|---:|---:|']
            for k, r in mo.iterrows():
                L.append(f'| {k} | {int(r["size"])} | {r["sum"]:+.1f} | {r["mean"]:+.3f} |')
            L += ['', f'weeks {len(w)}, green {float((w > 0).mean()):.2f}, mean {w.mean():+.2f}R, '
                      f'min {w.min():+.1f}R, max {w.max():+.1f}R, '
                      f'worst 5 weeks {w.nsmallest(5).sum():+.1f}R', '',
                  '| week | R |', '|---|---:|']
            for k, v in w.items():
                L.append(f'| {k} | {v:+.2f} |')
            L.append('')

    # --------------------------------------------------- the permutation null over this stage's books
    d, wk = F.prep(PRIMARY)
    x = C.scoreable(d, V, floor=True)

    def reg(tag, xx):
        r = F.row(tag, xx, wk, 'TRAIN')
        if r is None:
            return
        st, tt = r
        g = tt.groupby('day')[NET]
        cells[tag] = dict(t=st['t'], trades=tt[NET].values, bydayN=g.size().values)

    reg('BASELINE', x)
    for c in F.CUTS:
        reg(c['name'], x[c['fn'](x)])
    for nm, names in S.STACKS.items():
        mm = pd.Series(True, index=x.index)
        for k in names:
            mm &= by[k]['fn'](x)
        reg(nm, x[mm])
    mm = pd.Series(True, index=x.index)
    for k in FROZEN:
        mm &= by[k]['fn'](x)
    reg('FROZEN', x[mm])
    p, obs, q95 = perm_pvalue(cells, NPERM)
    L += ['## permutation, search-adjusted over every BOOK cell of this stage', '',
          f'{len(cells)} book cells on universe {PRIMARY}, {NPERM} day-label sign-flip draws: '
          f'observed max TRAIN t **{obs:.2f}**, null 95th pct **{q95:.2f}**, **p = {p:.3f}**.', '']

    pd.DataFrame(out).to_csv(f'{H}/h3_results{"_TEST" if READ_TEST else ""}.csv', index=False)
    open(f'{H}/h3_val{"_test" if READ_TEST else ""}.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
