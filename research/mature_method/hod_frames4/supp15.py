#!/usr/bin/env python3
"""hod_frames4 / F15 supplementary — the CAUSALITY TRACE on F15-c3 (`anchor_cohort >= 2`).

`anchor_cohort` as scored counts every same-anchor signal of the WHOLE session, including siblings
that break AFTER ours.  That is not knowable at the decision bar.  This file traces the field, builds
the CAUSAL version (siblings that have ALREADY broken at or before our entry minute), and reports the
decomposition, the concentration and the tail test.  No cell here is promoted; the causal cell is the
honest re-read of a declared cell whose field failed its own causality trace.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import D4, S, SPLITS, clustered_t, halves, book_ranked   # noqa: E402


def row(name, b):
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        print(f'| {name:<40s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
              f'{w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | {w["green"]:5.1f} | '
              f'{w["total"]:+8.0f} | {w["mdd"]:+8.0f} | {w["ex5"]:+.3f} |', flush=True)
    g, n, ok = halves(b)
    print(f'     halves H1 {g[0]:+.3f} | H2 {g[1]:+.3f} | VAL {g[2]:+.3f} -> same-signed +: {ok}',
          flush=True)
    return b


def main():
    s = pd.read_csv(f'{D4}/sig4_inst.csv', dtype={'symbol': str, 'day': str, 'wk': str,
                                                  'split': str, 'why': str, 'anchor': str,
                                                  'asset_class': str, 'venue': str},
                    keep_default_na=False, na_values=[''])
    s = s.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)

    # ---- the CAUSAL cohort ------------------------------------------------------------------
    s['cohort_causal'] = np.nan
    ok = s.anchor.notna()
    grp = s[ok].groupby(['day', 'anchor'], sort=False)
    ranks = []
    for _, d in grp:
        e = d.entry_m.values
        ranks.append(pd.Series([(e <= x).sum() for x in e], index=d.index))
    if ranks:
        s.loc[ok, 'cohort_causal'] = pd.concat(ranks).reindex(s.index[ok]).values

    print('== C1 — does the declared field use the future? ==')
    c3 = s[s.anchor_cohort >= 2]
    print(f'   F15-c3 signals {len(c3)}; of these the sibling had ALREADY broken at our entry '
          f'minute in {int((c3.cohort_causal >= 2).sum())} ({(c3.cohort_causal>=2).mean():.1%}) and '
          f'broke LATER in {int((c3.cohort_causal < 2).sum())} ({(c3.cohort_causal<2).mean():.1%}).')
    print('   -> the declared field is NOT computable at the decision bar for the second group. '
          'The causal cell below is the honest version.\n')

    print('| cell                                     | split | n     | /wk   | grossR | netR   |'
          '  t    | tc    | green | total $  | MDD $    | ex5    |')
    print('|' + '|'.join(['-' * 6] * 12) + '|')
    row('F15-c3 [declared] anchor_cohort>=2', book_ranked(s[s.anchor_cohort >= 2], 12, 4))
    row('F15-c3c [CAUSAL] cohort_causal>=2', book_ranked(s[s.cohort_causal >= 2], 12, 4))
    row('  ... sibling broke LATER only', book_ranked(
        s[(s.anchor_cohort >= 2) & (s.cohort_causal < 2)], 12, 4))
    row('  ... cohort==1 (alone all session)', book_ranked(s[s.anchor_cohort == 1], 12, 4))
    row('F15-c3c x stock rows only', book_ranked(
        s[(s.cohort_causal >= 2) & (s.asset_class == 'stock')], 12, 4))
    row('F15-c3c x wrapper rows only', book_ranked(
        s[(s.cohort_causal >= 2) & (s.asset_class == 'wrapper')], 12, 4))
    row('F15-c3c x spy_r5>0', book_ranked(s[(s.cohort_causal >= 2) & (s.spy_r5_pct > 0)], 12, 4))

    print('\n== C2 — composition of the declared cell ==')
    for sp in SPLITS:
        d = c3[c3.split == sp]
        print(f'   {sp}: rows {len(d)} | stock {int((d.asset_class=="stock").sum())} | wrapper '
              f'{int((d.asset_class=="wrapper").sum())} | distinct anchors {d.anchor.nunique()} | '
              f'top anchor {d.anchor.value_counts().head(3).to_dict()}')

    print('\n== C3 — concentration and the tail test (the causal cell) ==')
    b = book_ranked(s[s.cohort_causal >= 2], 12, 4)
    for sp in SPLITS:
        d = b[b.split == sp]
        wk = d.groupby('wk').pnl.sum().sort_values(ascending=False)
        tot = d.pnl.sum()
        n99 = d.net.quantile(0.99); n95 = d.net.quantile(0.95)
        print(f'   {sp}: total ${tot:+.0f} | best week ${wk.iloc[0]:+.0f} '
              f'({wk.iloc[0]/tot*100 if tot else np.nan:.0f} % of it) | top-3 weeks '
              f'${wk.head(3).sum():+.0f} | net {d.net.mean():+.3f} -> ex-top-1 % '
              f'{d.net[d.net<=n99].mean():+.3f} -> ex-top-5 % {d.net[d.net<=n95].mean():+.3f}')
        mo = d.groupby(d.day.str[:7]).pnl.sum()
        print(f'        months green {int((mo>0).sum())}/{len(mo)} | worst month ${mo.min():+.0f}')

    print('\n== C4 — count-matched null on the causal cell ==')
    for sp in SPLITS:
        obs, mu_, p5, p95 = S.null_band(b, sp)
        o = 'ABOVE' if obs > p95 else ('below' if obs < p5 else 'inside')
        print(f'   {sp}: green {obs:.1f} % vs null {mu_:.1f} [{p5:.1f}, {p95:.1f}] -> {o}')

    print('\n== C5 — MDE of the causal cell ==')
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'   {sp}: n {len(d)}, net {d.net.mean():+.3f}, 80 %-power MDE '
              f'{2.80*d.net.std(ddof=1)/np.sqrt(len(d)):.3f} R/trade')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
