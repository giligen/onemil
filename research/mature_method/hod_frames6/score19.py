#!/usr/bin/env python3
"""hod_frames6 / F19 — ERA COMPARABILITY.  Cells exactly as declared in PREREG.md.

Rebuilds the three eras as POINT-IN-TIME populations (Databento EQUS.SUMMARY definition feed) and
re-reads the base book and the four H2/VAL-positive, H1-negative objects on the instrument set
present in ALL THREE eras.  Then the frame's own question: is "H1-negative" the EDGE failing or the
UNIVERSE differing?
"""
import gc, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import (D6, ROOT, S, S2, SPLITS, sigset5, book_ranked, admit, clustered_t,  # noqa: E402
                     load_breaks4, attach_instrument, mde)

ERAS = {'H1-2025': ('2025-01', '2025-06'), 'H2-2025': ('2025-07', '2025-12'),
        'VAL':     ('2026-01', '2026-05')}


def era_of(day):
    return 'H1-2025' if day < '2025-07-01' else ('H2-2025' if day < '2026-01-01' else 'VAL')


def months(a, b):
    return [str(p) for p in pd.period_range(a, b, freq='M')]


def era_stats(s, tag, sheet):
    """Three-era gross/net/dollars/frequency for a SIGNAL set, booked with the shipped slot rule."""
    b = book_ranked(s, 12, 4)
    b = b.assign(era=[era_of(d) for d in b.day])
    out = []
    for e in ERAS:
        d = b[b.era == e]
        nw = len(pd.period_range(f'{ERAS[e][0]}-01', f'{ERAS[e][1]}-28', freq='W-FRI'))
        if not len(d):
            out.append(dict(cell=tag, era=e, n=0)); continue
        w = d.groupby('wk').pnl.sum()
        row = dict(cell=tag, era=e, n=len(d), per_wk=len(d) / nw, gross=float(d.rr.mean()),
                   net=float(d.net.mean()), total=float(d.pnl.sum()),
                   green=float((w > 0).mean() * 100), tc=clustered_t(d), mde=mde(d, 'rr'))
        out.append(row)
    sheet.extend(out)
    g = {r['era']: r for r in out}
    print(f'| {tag:<34s} |' + ''.join(
        f" {g[e].get('n', 0):5d} | {g[e].get('per_wk', float('nan')):5.1f} | "
        f"{g[e].get('gross', float('nan')):+.3f} | {g[e].get('total', float('nan')):+8.0f} |"
        for e in ERAS), flush=True)
    return out


def main():
    print('== F19 · point-in-time listing sets (stage A artifact `pitsets6.csv`) ==', flush=True)
    ps = pd.read_csv(f'{D6}/pitsets6.csv', dtype={'symbol': str})
    L = {e: frozenset(ps.symbol[ps[e]]) for e in ERAS}
    INTER = frozenset(ps.symbol[ps.INTERSECT])
    EVER = frozenset(ps.symbol)
    print('  PIT definition coverage 202407..202609 — the whole TRAIN+VAL window is INSIDE it; '
          'the XNAS.ITCH 2018-2024 fallback is not needed and is not used.', flush=True)
    for e in ERAS:
        print(f'  {e}: listed throughout = {len(L[e])} symbols', flush=True)
    print(f'  INTERSECT (listed throughout all three eras) = {len(INTER)} symbols\n', flush=True)

    br = load_breaks4(verbose=False)
    S.build_impute(S2.load_pop())
    s = sigset5(admit(br, pd.Series(True, index=br.index)))
    del br; gc.collect()
    s = attach_instrument(s)
    s['era'] = [era_of(d) for d in s.day]
    s['listed_any'] = s.symbol.isin(EVER)
    s['in_inter'] = s.symbol.isin(INTER)

    # ------------------------------------------------------------------ F19-0 structural report
    print('== F19-0  structural report (per era, the candidate stream) ==', flush=True)
    print('| era | signals | distinct symbols | wrappers | commons | UNCHECKABLE (no PIT record) | '
          'median price | median ADV$ | listed-in-all-3 share of signals |', flush=True)
    print('|---|---|---|---|---|---|---|---|---|', flush=True)
    for e in ERAS:
        d = s[s.era == e]
        u = d.drop_duplicates('symbol')
        unk = int((~u.listed_any).sum())
        print(f'| {e} | {len(d)} | {len(u)} | {int((u.asset_class == "wrapper").sum())} | '
              f'{int((u.asset_class == "stock").sum())} | {unk} ({unk / len(u) * 100:.1f} %) | '
              f'${d.price.median():.2f} | ${d.adv_dollar.median() / 1e6:.0f}M | '
              f'{d.in_inter.mean() * 100:.1f} % |', flush=True)
    print('\n  wrappers listed throughout all three eras vs first seen later:', flush=True)
    for cls in ('wrapper', 'stock'):
        u = s[s.asset_class == cls].drop_duplicates('symbol')
        print(f'    {cls}: {len(u)} distinct; in INTERSECT {int(u.in_inter.sum())} '
              f'({u.in_inter.mean() * 100:.1f} %); in H1 set {int(u.symbol.isin(L["H1-2025"]).sum())} '
              f'({u.symbol.isin(L["H1-2025"]).mean() * 100:.1f} %)', flush=True)
    print('', flush=True)

    # ------------------------------------------------------------------ the cells
    hdr = '| cell                               |' + ''.join(
        f' {e:>5s} n | /wk   | gross  | $        |' for e in ERAS)
    sheet = []
    print('== F19 · the cells (three-era table; the object is the UNIVERSE, not a new rule) ==',
          flush=True)
    print(hdr, flush=True)
    print('|' + '|'.join(['-' * 6] * 13) + '|', flush=True)

    era_stats(s, 'F19-ref  base B2 (as-is)', sheet)
    era_stats(s[s.in_inter], 'F19-b1   base, INTERSECT', sheet)
    era_stats(s[~s.in_inter], 'F19-b2   base, NOT in INTERSECT', sheet)
    era_stats(s[s.in_inter & (s.asset_class == 'wrapper')], 'F19-o1   wrappers, INTERSECT', sheet)
    era_stats(s[s.in_inter & (s.spy_r5_pct > 0)], 'F19-o2   SPY 09:35 gate, INTERSECT', sheet)
    p80 = float(s[s.split == 'TRAIN'].dollar_frac.quantile(0.80))
    era_stats(s[s.in_inter & (s.dollar_frac >= p80)],
              f'F19-o3   dollar_frac>=p80({p80:.1f}), INT', sheet)
    era_stats(s[s.in_inter & (s.price >= 30.0)], 'F19-o4   price>=$30, INTERSECT', sheet)
    # the same four objects WITHOUT the intersection, for the like-for-like read
    era_stats(s[s.asset_class == 'wrapper'], 'F19-o1r  wrappers, as-is', sheet)
    era_stats(s[s.spy_r5_pct > 0], 'F19-o2r  SPY 09:35 gate, as-is', sheet)
    era_stats(s[s.dollar_frac >= p80], 'F19-o3r  dollar_frac>=p80, as-is', sheet)
    era_stats(s[s.price >= 30.0], 'F19-o4r  price>=$30, as-is', sheet)

    # ------------------------------------------------------------------ F19-x1
    print('\n== F19-x1  the H1 book on the H2-listed universe vs the H1-only names ==', flush=True)
    h1 = s[s.era == 'H1-2025']
    for lab, sub in (('H1, all names (reference)', h1),
                     ('H1, names listed throughout H2-2025', h1[h1.symbol.isin(L['H2-2025'])]),
                     ('H1, names listed in all three eras', h1[h1.in_inter]),
                     ('H1, names NOT listed throughout H2', h1[~h1.symbol.isin(L['H2-2025'])])):
        b = book_ranked(sub, 12, 4)
        nw = len(pd.period_range('2025-01-01', '2025-06-30', freq='W-FRI'))
        w = b.groupby('wk').pnl.sum() if len(b) else pd.Series(dtype=float)
        print(f'  {lab:<38s} n {len(b):4d} ({len(b) / nw:4.1f}/wk)  gross {b.rr.mean():+.4f}  '
              f'net {b.net.mean():+.4f}  $ {b.pnl.sum():+8.0f}  green {(w > 0).mean() * 100:5.1f} % '
              f' clust t {clustered_t(b):+.2f}  MDE {mde(b, "rr"):.3f}', flush=True)
        sheet.append(dict(cell='F19-x1', era=lab, n=len(b), gross=float(b.rr.mean()),
                          net=float(b.net.mean()), total=float(b.pnl.sum()),
                          green=float((w > 0).mean() * 100), tc=clustered_t(b), mde=mde(b, 'rr')))

    # ------------------------------------------------------------------ the pre-committed rule
    df = pd.DataFrame(sheet)
    df.to_csv(f'{D6}/cells19.csv', index=False)
    print('\n== F19 · THE PRE-COMMITTED RE-OPEN RULE ==', flush=True)
    print('  RE-OPEN iff on INTERSECT: (a) H1 gross >= 0, (b) H1/H2/VAL same-signed positive, '
          '(c) >= 10 tr/wk on BOTH splits.', flush=True)
    print('| object | H1 gross | H2 gross | VAL gross | min /wk | (a) | (b) | (c) | VERDICT |',
          flush=True)
    print('|---|---|---|---|---|---|---|---|---|', flush=True)
    for tag in [t for t in df.cell.unique() if t.startswith('F19-') and 'x1' not in t]:
        d = df[df.cell == tag].set_index('era')
        try:
            g = [float(d.loc[e, 'gross']) for e in ERAS]
            fw = min(float(d.loc[e, 'per_wk']) for e in ERAS)
        except (KeyError, ValueError):
            continue
        a = g[0] >= 0; bb = all(x > 0 for x in g); cc = fw >= 10.0
        v = 'RE-OPENED (listing caveat)' if (a and bb and cc) else 'stays dead'
        print(f'| {tag} | {g[0]:+.3f} | {g[1]:+.3f} | {g[2]:+.3f} | {fw:.1f} | {a} | {bb} | {cc} | '
              f'**{v}** |', flush=True)
    print('\ncells19.csv written', flush=True)


if __name__ == '__main__':
    main()
