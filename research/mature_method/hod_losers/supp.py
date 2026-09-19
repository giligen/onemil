#!/usr/bin/env python3
"""hod_losers — supplementary: WHY the nominated rules fail, and the MDE of this pass.
Diagnostics only; no cell is scored here."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
import cells as C            # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUT = open(f'{D}/supp.txt', 'w')
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    ex = pd.read_csv(f'{D}/exits.csv', **RD).drop_duplicates(['day', 'symbol', 'entry_m'])
    pa = pd.read_csv(f'{D}/path.csv', **RD).drop_duplicates(['day', 'symbol', 'entry_m'])
    SG = {}
    for nm in ('B0', 'B2'):
        sg = S2.sig_set(pop, **S2.BASES[nm])
        sg = sg[sg.split.isin(S.SPLITS)].merge(ex, on=['day', 'symbol', 'entry_m'], how='left')
        sg = sg.merge(pa[['day', 'symbol', 'entry_m', 'mfe10', 'mfe5', 'touch_n', 'consol_bars']],
                      on=['day', 'symbol', 'entry_m'], how='left')
        SG[nm] = sg[sg.rr_base.notna()].reset_index(drop=True)

    p('=' * 110)
    p('S1. WHY THE TIME STOP FAILS — the 0.77 R separation is real and un-monetisable')
    p('=' * 110)
    for nm in ('B0',):
        sg = SG[nm]
        for sp in S.SPLITS:
            d = sg[sg.split == sp]
            t = d[d.mfe10 < 0.25]
            p(f'  [{nm} {sp}] pre-book signals {len(d)}; the time stop fires on {len(t)} '
              f'({len(t)/len(d)*100:.0f}%)')
            p(f'     on the FIRED subset: gross under the shipped exit {t.rr_base.mean():+.3f}, '
              f'gross at the fill+10 close {t.rr_t10.mean():+.3f}  -> the rule SAVES '
              f'{t.rr_t10.mean() - t.rr_base.mean():+.3f} R a fired trade')
            p(f'     the bleed is ALREADY BOOKED: median R at the fill+10 close '
              f'{t.rr_t10.median():+.3f}; the stop is at -1.0 R, so the cut happens '
              f'{(t.rr_t10.mean() + 1.0):.2f} R into a 1 R loss')
            p(f'     freed slots: mean hold shrinks '
              f'{(d.exit_m_base - d.entry_m).mean():.0f} -> {(d.exit_m_t10 - d.entry_m).mean():.0f} min, '
              f'which is why the book grows 31.8 -> 34.5 trades a week')
    p('\n  the counterfactual that separates the two effects (pre-book, no slots): '
      'mean gross under each exit over the WHOLE signal set')
    for nm in ('B0', 'B2'):
        sg = SG[nm]
        for sp in S.SPLITS:
            d = sg[sg.split == sp]
            row = '  '.join(f'{v} {d[f"rr_{v}"].mean():+.3f}'
                            for v in ('base', 't10', 't5', 'ruleD', 'shape', 'be', 't10be'))
            p(f'    {nm} {sp:5s} n {len(d):5d}  {row}')

    p('\n' + '=' * 110)
    p('S2. THE ADDED TRADES — what the freed slots buy')
    p('=' * 110)
    for nm in ('B0',):
        base = C.book(C.recost(SG[nm].assign(rr=SG[nm].rr_base, why=SG[nm].why_base,
                                             exit_m=SG[nm].exit_m_base)))
        t10 = C.book(C.recost(SG[nm].assign(rr=SG[nm].rr_t10, why=SG[nm].why_t10,
                                            exit_m=SG[nm].exit_m_t10)))
        kb = set(zip(base.day, base.symbol, base.entry_m))
        kt = set(zip(t10.day, t10.symbol, t10.entry_m))
        add = t10[[k not in kb for k in zip(t10.day, t10.symbol, t10.entry_m)]]
        drop = base[[k not in kt for k in zip(base.day, base.symbol, base.entry_m)]]
        p(f'  [{nm}] the T10 book adds {len(add)} trades the shipped book never took and drops '
          f'{len(drop)}')
        for sp in S.SPLITS:
            a = add[add.split == sp]; dr = drop[drop.split == sp]
            p(f'    {sp:5s} ADDED n {len(a):4d} gross {a.rr.mean():+.3f} net {a.net.mean():+.3f} '
              f'$ {a.pnl.sum():+,.0f}   |   DROPPED n {len(dr):4d} $ {dr.pnl.sum():+,.0f}')

    p('\n' + '=' * 110)
    p('S3. MDE — the smallest per-trade effect this pass could have resolved (80 % power, two-sided 5 %)')
    p('=' * 110)
    for nm in ('B0', 'B2'):
        sg = SG[nm]
        for sp in S.SPLITS:
            d = sg[sg.split == sp]
            sd = d.rr_base.std(ddof=1)
            p(f'  {nm} {sp:5s} pre-book n {len(d):5d} sd(rr) {sd:.3f} -> MDE '
              f'{2.8 * sd / np.sqrt(len(d)):.3f} R')
    for nm, sub, lbl in (('B0', lambda s: s.dist_20d_high_pct >= 0, 'P14 selected subset'),
                         ('B0', lambda s: s.touch_n < 5, 'P11 selected subset'),
                         ('B0', lambda s: s.consol_bars < 8, 'P13 selected subset')):
        sg = SG[nm]; sg = sg[sub(sg)]
        for sp in S.SPLITS:
            d = sg[sg.split == sp]
            if len(d) < 30:
                continue
            p(f'  {nm} {sp:5s} {lbl}: n {len(d):5d} -> MDE '
              f'{2.8 * d.rr_base.std(ddof=1) / np.sqrt(len(d)):.3f} R')
    p('\n  green-week MDE: +-19.0 pp over 53 TRAIN weeks, +-28.9 pp over 23 VAL weeks '
      '(unchanged from hod_break §10 / hod_filter_stack §10 -- same weeks, same book).')

    p('\n' + '=' * 110)
    p('S4. P14 — the one cell that improves week shape, against its own count-matched null')
    p('=' * 110)
    p('  TRAIN observed 45.3 % vs null mean 46.7 % [39.6, 54.7]  -> BELOW the null MEAN')
    p('  VAL   observed 47.8 % vs null mean 44.6 % [34.8, 56.5]  -> inside')
    p('  i.e. the whole green-week gain of the 20-day-high gate is what cutting the pick count from')
    p('  31.8 to 10.5 trades a week does to a book of this mean and variance. It is arithmetic.')
    OUT.close()


if __name__ == '__main__':
    main()
