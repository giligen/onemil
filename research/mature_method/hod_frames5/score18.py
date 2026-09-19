#!/usr/bin/env python3
"""hod_frames5 / F18 — THE POPULATION: is `B2` the book we should be asking about?

Cells exactly as declared in PREREG.md §4 (committed 7144d3c, before any cell was scored).
Three sub-questions: (1) wrappers vs common vs mixed, (2) the price floor + the live-parity check,
(3) the two spread gates with the cost MEASURED.  TEST sealed.  Read-only.  One process.

`--measured` re-runs the whole sheet with `nbbo5.csv` (the dedicated fetch) merged.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import (ROOT, D5, S, S2, SPLITS, RISK, load_breaks5, sigset5, admit,   # noqa: E402
                     book_ranked, attach_instrument, clustered_t, halves, Sheet)

MEASURED = '--measured' in sys.argv
TAG = 'measured' if MEASURED else 'asis'


def build():
    extra = (f'{D5}/nbbo5.csv',) if MEASURED else ()
    br = load_breaks5(extra_nbbo=extra)
    S.build_impute(S2.load_pop())
    FIRST = admit(br, pd.Series(True, index=br.index))
    return FIRST


def pop(FIRST, floor=20.0, fr=0.15, bps=100.0, cls=None):
    s = sigset5(FIRST, min_price=floor, max_frac_r=fr, max_bps=bps)
    s = attach_instrument(s)
    if cls is not None:
        s = s[s.asset_class == cls]
    return s


def signal_level(s):
    """Gross R at the SIGNAL level — before the slot rule (F18-R1: 'is B2 the pattern?')."""
    out = {}
    for sp in SPLITS:
        d = s[s.split == sp]
        out[sp] = (len(d), float(d.rr.mean()) if len(d) else np.nan,
                   float(d.net.mean()) if len(d) else np.nan)
    tr = s[s.split == 'TRAIN']
    h1 = tr[tr.day < '2025-07-01'].rr.mean() if len(tr) else np.nan
    h2 = tr[tr.day >= '2025-07-01'].rr.mean() if len(tr) else np.nan
    return out, h1, h2


def weekly_table(b, name):
    print(f'\n   weekly dollar path — {name} (live sizing, $100 risk)', flush=True)
    for sp in SPLITS:
        d = b[b.split == sp]
        w = d.groupby('wk').pnl.sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
        q = w.tail(26)
        print(f'     {sp} last 26 weeks: ' + ' '.join(f'{v:+.0f}' for v in q.values), flush=True)
        print(f'     {sp} green {int((w>0).sum())}/{len(w)}  mean {w.mean():+.0f}  '
              f'worst {w.min():+.0f}  best {w.max():+.0f}  total {w.sum():+.0f}', flush=True)


def main():
    FIRST = build()
    print(f'\n=== F18 ({TAG.upper()} cost arm) ===', flush=True)

    # ---------------------------------------------------------------- availability audit (R2)
    print('\n== F18-R2 availability audit: `asset_class` on the enlarged populations ==', flush=True)
    print('| population | rows | identified | miss on winners | miss on losers | gap | verdict |')
    print('|' + '|'.join(['-' * 6] * 7) + '|')
    for fl in (5.0, 10.0, 20.0, 30.0, 50.0):
        s = pop(FIRST, floor=fl)
        idn = s.asset_class.isin(['stock', 'wrapper'])
        win = s.rr > 0
        mw = float((~idn)[win].mean() * 100); ml = float((~idn)[~win].mean() * 100)
        print(f'| floor ${fl:.0f} | {len(s)} | {idn.mean():.1%} | {mw:.1f}% | {ml:.1f}% | '
              f'{abs(mw-ml):.1f} pp | {"ok" if idn.mean() >= 0.5 and abs(mw-ml) <= 5 else "VOID"} |',
              flush=True)

    # ---------------------------------------------------------------- F18-R1 structural report
    print('\n== F18-R1 the BASE GROSS of every population choice, at the SIGNAL level, before any '
          'admission rule ==', flush=True)
    print('| population | TRAIN n | TRAIN gross | TRAIN net | VAL n | VAL gross | VAL net | '
          'H1 gross | H2 gross | era-consistent |')
    print('|' + '|'.join(['-' * 6] * 10) + '|')
    r1 = []
    for lab, kw in [('mixed  $20 (B2 population)', dict()),
                    ('mixed  $5', dict(floor=5.0)), ('mixed  $10', dict(floor=10.0)),
                    ('mixed  $30', dict(floor=30.0)), ('mixed  $50', dict(floor=50.0)),
                    ('WRAP   $20', dict(cls='wrapper')), ('COMMON $20', dict(cls='stock')),
                    ('WRAP   $5', dict(floor=5.0, cls='wrapper')),
                    ('COMMON $5', dict(floor=5.0, cls='stock')),
                    ('WRAP   $10', dict(floor=10.0, cls='wrapper')),
                    ('COMMON $10', dict(floor=10.0, cls='stock')),
                    ('WRAP   $30', dict(floor=30.0, cls='wrapper')),
                    ('WRAP   $50', dict(floor=50.0, cls='wrapper')),
                    ('mixed  $20 gates OFF', dict(fr=None, bps=None)),
                    ('mixed  $20 frac_r 0.08', dict(fr=0.08)),
                    ('mixed  $20 frac_r 0.25', dict(fr=0.25)),
                    ('mixed  $20 frac_r 0.40', dict(fr=0.40))]:
        s = pop(FIRST, **kw)
        o, h1, h2 = signal_level(s)
        ok = all(v == v and v > 0 for v in (h1, h2, o['VAL'][1]))
        print(f'| {lab} | {o["TRAIN"][0]} | {o["TRAIN"][1]:+.3f} | {o["TRAIN"][2]:+.3f} | '
              f'{o["VAL"][0]} | {o["VAL"][1]:+.3f} | {o["VAL"][2]:+.3f} | {h1:+.3f} | {h2:+.3f} | '
              f'{ok} |', flush=True)
        r1.append(dict(population=lab, train_n=o['TRAIN'][0], train_gross=o['TRAIN'][1],
                       train_net=o['TRAIN'][2], val_n=o['VAL'][0], val_gross=o['VAL'][1],
                       val_net=o['VAL'][2], h1=h1, h2=h2, era_ok=ok))
    pd.DataFrame(r1).to_csv(f'{D5}/f18_r1_{TAG}.csv', index=False)

    # ---------------------------------------------------------------- the cells
    sh = Sheet()
    print('\n== F18 CELLS (booked 12/day x 4 concurrent, the shipped slot rule) ==', flush=True)
    sh.show('F18-base  mixed $20 [=B2]', book_ranked(pop(FIRST), 12, 4))
    sh.show('F18-p1  WRAPPERS ONLY $20', book_ranked(pop(FIRST, cls='wrapper'), 12, 4))
    sh.show('F18-p2  COMMON ONLY $20', book_ranked(pop(FIRST, cls='stock'), 12, 4))
    for fl in (5.0, 10.0, 30.0, 50.0):
        sh.show(f'F18-f{fl:.0f}  mixed floor ${fl:.0f}', book_ranked(pop(FIRST, floor=fl), 12, 4))
    for fl in (5.0, 10.0, 30.0, 50.0):
        sh.show(f'F18-w{fl:.0f}  WRAPPERS floor ${fl:.0f}',
                book_ranked(pop(FIRST, floor=fl, cls='wrapper'), 12, 4))
    for fr in (0.08, 0.25, 0.40):
        sh.show(f'F18-s{int(fr*100):02d}  spread <= {fr:.0%} of R',
                book_ranked(pop(FIRST, fr=fr), 12, 4))
    sh.show('F18-soff  both spread gates OFF', book_ranked(pop(FIRST, fr=None, bps=None), 12, 4))

    sh.nulls(f'{D5}/nulls18_{TAG}.csv')
    sh.dump(f'{D5}/cells18_{TAG}.csv')

    print('\n== MDE, the tail, and the cost actually paid ==', flush=True)
    print('| cell | split | n | /wk | net | MDE | ex-top5% | ex-top1% | cost R | imputed % |')
    print('|' + '|'.join(['-' * 6] * 10) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            d = b[b.split == sp]
            if len(d) < 5:
                continue
            mde = 2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d)))
            print(f'| {name} | {sp} | {len(d)} | {len(d)/S.NW[sp]:.1f} | {d.net.mean():+.3f} | '
                  f'{mde:.3f} | {d.net[d.net <= d.net.quantile(0.95)].mean():+.3f} | '
                  f'{d.net[d.net <= d.net.quantile(0.99)].mean():+.3f} | '
                  f'{(d.rr - d.net).mean():+.3f} | {d.imputed.mean()*100:.0f}% |', flush=True)

    # the already-there / arrived-after diagnostic on the candidate populations
    print('\n== the already-there / arrived-after diagnostic (reported, never an exclusion) ==',
          flush=True)
    for name in ('F18-base  mixed $20 [=B2]', 'F18-p1  WRAPPERS ONLY $20', 'F18-p2  COMMON ONLY $20'):
        b = sh.books[name]
        for sp in SPLITS:
            d = b[b.split == sp]
            aw = d[d.rng_sig >= 10]; aa = d[d.rng_sig < 10]
            print(f'   {name} {sp}: already-wide n {len(aw)} gross {aw.rr.mean():+.3f} | '
                  f'arrived-after n {len(aa)} gross {aa.rr.mean():+.3f}', flush=True)

    # the ship-bar treatment for the declared candidate
    for name in ('F18-p1  WRAPPERS ONLY $20',):
        weekly_table(sh.books[name], name)

    print('\nDONE F18 ' + TAG, flush=True)


if __name__ == '__main__':
    main()
