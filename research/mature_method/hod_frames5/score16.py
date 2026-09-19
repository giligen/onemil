#!/usr/bin/env python3
"""hod_frames5 / F16 — THE PORTFOLIO: concurrency correlation and anchor de-duplication.

Cells exactly as declared in PREREG.md §3 (committed 7144d3c, before any cell was scored).
The ADMISSION is untouched in every cell — same signals, same prices, same stops, same exits.
Only the slot allocator changes.  TEST sealed.  Read-only.  One process.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import (ROOT, D5, D4, S, S2, SPLITS, RISK, load_breaks4, sigset5, admit,   # noqa: E402
                     book_ranked, book_portfolio, attach_instrument, clustered_t, halves,
                     Sheet)

ADV_EDGES = [0, 10e6, 50e6, 200e6, 1e9, 1e18]
ADV_LAB = ['<$10M', '$10-50M', '$50-200M', '$200M-1B', '>=$1B']


def build():
    br = load_breaks4()
    S.build_impute(S2.load_pop())
    s = sigset5(admit(br, pd.Series(True, index=br.index)))
    s = attach_instrument(s)
    # venue: Databento point-in-time listing exchange, joined from the pass-4 artifact (100 % cov)
    v = pd.read_csv(f'{D4}/sig4_inst.csv', usecols=['day', 'symbol', 'entry_m', 'venue'],
                    dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    s = s.merge(v.drop_duplicates(['day', 'symbol', 'entry_m']), on=['day', 'symbol', 'entry_m'],
                how='left')
    s['advb'] = pd.cut(s.adv_dollar, ADV_EDGES, labels=ADV_LAB).astype(object)
    # the CAUSAL cohort (hod_frames4/supp15 definition): how many same-anchor candidates of the day
    # have broken AT OR BEFORE this signal's entry minute, this one included.
    s = s.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)
    s['cohort_causal'] = np.nan
    ok = s.anchor.notna()
    ranks = []
    for _, d in s[ok].groupby(['day', 'anchor'], sort=False):
        e = d.entry_m.values
        ranks.append(pd.Series([(e <= x).sum() for x in e], index=d.index))
    if ranks:
        s.loc[ok, 'cohort_causal'] = pd.concat(ranks).reindex(s.index[ok]).values
    return s


def parity(s):
    """PREREG §1: book_portfolio(key=None) must reproduce book_ranked row-for-row."""
    a = book_ranked(s, 12, 4); b = book_portfolio(s, 12, 4, key=None)
    assert len(a) == len(b) and set(a.index) == set(b.index), \
        f'PORTFOLIO SLOT-MACHINE PARITY FAILED: {len(a)} vs {len(b)}'
    print(f'   slot-machine parity: book_portfolio(key=None) == book_ranked  '
          f'({len(a)} == {len(b)} rows, identical set)  ASSERTED', flush=True)
    return a


def concentration(b, s):
    """F16-diag — the concentration diagnostic, printed BEFORE any cell."""
    print('\n== F16-diag  THE CONCENTRATION DIAGNOSTIC (no bar) ==', flush=True)
    print('| split | occupied slot-minutes | >=2 share an anchor | >=2 a venue | >=2 an ADV$ bucket |'
          ' max same-anchor conc | booked rows with a same-anchor sibling open |')
    print('|' + '|'.join(['-' * 6] * 7) + '|')
    for sp in SPLITS:
        d = b[b.split == sp]
        occ = dup_a = dup_v = dup_b = 0; mx = 0; rows_dup = 0
        for day, g in d.groupby('day'):
            iv = list(zip(g.entry_m.astype(int), g.exit_m.astype(int),
                          g.anchor.astype(object), g.venue.astype(object), g.advb.astype(object)))
            lo = min(x[0] for x in iv); hi = max(x[1] for x in iv)
            for m in range(lo, hi + 1):
                op = [x for x in iv if x[0] <= m <= x[1]]
                if not op:
                    continue
                occ += 1
                for j, col in ((2, 'a'), (3, 'v'), (4, 'b')):
                    vals = [x[j] for x in op if isinstance(x[j], str)]
                    c = pd.Series(vals).value_counts() if vals else pd.Series(dtype=int)
                    if len(c) and c.iloc[0] >= 2:
                        if col == 'a':
                            dup_a += 1; mx = max(mx, int(c.iloc[0]))
                        elif col == 'v':
                            dup_v += 1
                        else:
                            dup_b += 1
            for x in iv:
                if isinstance(x[2], str) and sum(
                        1 for y in iv if y is not x and isinstance(y[2], str) and y[2] == x[2]
                        and y[0] <= x[1] and x[0] <= y[1]) >= 1:
                    rows_dup += 1
        print(f'| {sp} | {occ} | {dup_a / max(occ,1):.1%} | {dup_v / max(occ,1):.1%} | '
              f'{dup_b / max(occ,1):.1%} | {mx} | {rows_dup}/{len(d)} ({rows_dup/max(len(d),1):.1%}) |',
              flush=True)
    print('\n   top anchors in the BOOKED set:', flush=True)
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'     {sp}: {dict(d.anchor.value_counts().head(8))}', flush=True)
    print(f'\n   pre-book composition: wrapper {100*(s.asset_class=="wrapper").mean():.1f}% | '
          f'stock {100*(s.asset_class=="stock").mean():.1f}% | '
          f'unknown {100*(~s.asset_class.isin(["wrapper","stock"])).mean():.1f}%', flush=True)
    print(f'   booked   composition: wrapper {100*(b.asset_class=="wrapper").mean():.1f}% | '
          f'stock {100*(b.asset_class=="stock").mean():.1f}%', flush=True)


def main():
    s = build()
    print(f'\npre-book signals {len(s)} | anchors {s.anchor.nunique()} | '
          f'venues {s.venue.nunique()} | ADV$ buckets {s.advb.nunique()}', flush=True)
    print('\n== 0. reproduction + parity ==', flush=True)
    b2 = parity(s)
    for sp in SPLITS:
        w = S.week_stats(b2, sp)
        print(f'   B2 {sp:5s} n {w["n"]:5d} {w["per_wk"]:5.1f}/wk gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} green {w["green"]:.1f}% total ${w["total"]:+,.0f}', flush=True)

    concentration(b2, s)

    sh = Sheet()
    print('\n== F16 CELLS (admission untouched; only the slot allocator changes) ==', flush=True)
    sh.show('F16-base  B2 (reference)', b2)
    sh.show('F16-a1  1/anchor, 4 slots', book_portfolio(s, 12, 4, 'anchor', 1))
    sh.show('F16-a2  2/anchor, 4 slots', book_portfolio(s, 12, 4, 'anchor', 2))
    sh.show('F16-a1n 1/anchor, 4 slots, NO REFILL',
            book_portfolio(s, 12, 4, 'anchor', 1, consume_on_reject=True))
    sh.show('F16-a6  1/anchor, 6 slots', book_portfolio(s, 12, 6, 'anchor', 1))
    sh.show('F16-a8  1/anchor, 8 slots', book_portfolio(s, 12, 8, 'anchor', 1))
    sh.show('F16-v2  2/venue, 4 slots', book_portfolio(s, 12, 4, 'venue', 2))
    sh.show('F16-b2  2/ADV$ bucket, 4 slots', book_portfolio(s, 12, 4, 'advb', 2))
    m1 = s[s.cohort_causal >= 2]
    sh.show('F16-m1  CONCENTRATED (causal sibling)', book_ranked(m1, 12, 4))
    sh.show('F16-m2  CONCENTRATED x 1/anchor', book_portfolio(m1, 12, 4, 'anchor', 1))

    # the pre-committed reproduction expectation for the mirror
    print('\n   [F16-m1 pre-committed expectation, hod_frames4 §3.3 causal row] '
          'gross -0.013 / +0.154, $-2,284 / $+2,099, H1 -0.120, 6.4 / 9.7 tr/wk', flush=True)

    # week shape, the thing the frame claims to move
    print('\n== F16 week shape (the frame\'s own claim: shape, not gross) ==', flush=True)
    print('| cell | split | green % | red streak | worst wk $ | best wk $ | MDD $ | '
          'best wk share of total |')
    print('|' + '|'.join(['-' * 6] * 8) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            w = S.week_stats(b, sp)
            share = (w['best'] / w['total'] * 100) if w['total'] > 0 else np.nan
            print(f'| {name} | {sp} | {w["green"]:.1f} | {w["redstreak"]} | {w["worst"]:.0f} | '
                  f'{w["best"]:.0f} | {w["mdd"]:.0f} | {share:.0f}% |', flush=True)

    sh.nulls(f'{D5}/nulls16.csv')
    sh.dump(f'{D5}/cells16.csv')

    # MDE + ex-top-5 % on every cell (rails 11 and the tail test)
    print('\n== MDE (80 % power, net R/trade) and the tail ==', flush=True)
    print('| cell | split | n | net | MDE | ex-top5% net | ex-top1% net |')
    print('|' + '|'.join(['-' * 6] * 7) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            d = b[b.split == sp]
            if len(d) < 5:
                continue
            mde = 2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d)))
            e5 = float(d.net[d.net <= d.net.quantile(0.95)].mean())
            e1 = float(d.net[d.net <= d.net.quantile(0.99)].mean())
            print(f'| {name} | {sp} | {len(d)} | {d.net.mean():+.3f} | {mde:.3f} | {e5:+.3f} | '
                  f'{e1:+.3f} |', flush=True)
    print('\nDONE F16', flush=True)


if __name__ == '__main__':
    main()
