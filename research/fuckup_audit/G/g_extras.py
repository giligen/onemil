#!/usr/bin/env python3
"""Stage G extras — the descriptive counts the REPORT needs, plus the two audits the run exposed:

 1. how many signals the SSR exclusion removes, and how many the no-chase cap refuses to fill;
 2. the cost decomposition (gross minus net, in R) per family;
 3. the ETF leak: `trading/orb_asset_class` excludes LEVERAGED wrappers, not plain funds, so a
    borrowable-looking "stock" can be an index ETF. The share is measured against the Alpaca asset
    name dump and the 24 primary cells are re-scored with those names removed (a POST-HOC block of
    24 cells, counted as such);
 4. S5 attention minus control, in R, on the same book.

Writes ONLY research/fuckup_audit/G/extras.md.
Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/G/g_extras.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/G')
import score_short as S                                    # the scorer's contract, reused verbatim

G = f'{ROOT}/research/fuckup_audit/G'
ASSETS = f'{ROOT}/data/research/databento/alpaca_assets_all_20260905.csv'
FUND_RE = re.compile(
    r'\bETF\b|\bETN\b|\bFund\b|\bIndex\b|\bSPDR\b|iShares|Vanguard|Invesco|WisdomTree|ProShares|'
    r'Direxion|Global X|First Trust|VanEck|Amplify|Roundhill|YieldMax|Defiance|Simplify|Xtrackers|'
    r'GraniteShares|Innovator|Pacer|ALPS |Franklin .*ETF|Schwab .*ETF|\bUnit Trust\b|'
    r'\bClosed[- ]End\b|\bPortfolio\b', re.I)
L = []


def log(m):
    print(m, flush=True)
    L.append(m)


def main():
    d = S.load()
    weeks = {s: sorted(d[d.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}

    # ---- 1. what the population filters remove
    raw = pd.read_csv(f'{G}/candidates_short.csv',
                      usecols=['fam', 'cfg', 'split', 'ssr', 'entry', 'in_u12', 'attn_grp'],
                      dtype={'fam': str, 'cfg': str, 'split': str, 'attn_grp': str},
                      keep_default_na=False, na_values=[''])
    raw['key'] = raw.fam + ' ' + raw.cfg
    log('## 1. Signals, fills and the SSR exclusion')
    log('')
    rows = []
    for k in S.FAM_KEYS:
        x = raw[raw.key == k]
        x = x[(x.attn_grp == 'attention') if k == S.S5 else (x.in_u12 == 1)]
        sc = d[d.key == k]
        sc = sc[(sc.attn_grp == 'attention') if k == S.S5 else (sc.in_u12 == 1)]
        rows.append(dict(key=k, signals=len(x),
                         filled=int(x.entry.notna().sum()),
                         fill_rate=f'{x.entry.notna().mean()*100:.1f}%',
                         scoreable=len(sc),
                         ssr_signals=int(x.ssr.sum()),
                         ssr_pct=f'{x.ssr.mean()*100:.1f}%',
                         scoreable_after_ssr=int((sc.ssr == 0).sum())))
    log(pd.DataFrame(rows).to_string(index=False))
    log('')
    log('`filled` = the next-open (S5: the 09:35 open) fill passed the no-chase cap and left a stop '
        'above the entry; `scoreable` adds entry >= $10, entry_m <= 14:01 and r_pct >= 1.0.')

    # ---- 2. cost decomposition, per family, on the booked primary cells
    log('')
    log('## 2. What the cost contract costs, in R (booked TRAIN trades, UA, hold / S5 1030)')
    log('')
    rows = []
    for k in S.FAM_KEYS:
        lab, rr, why, xm = (S.EXITS_S5 if k == S.S5 else S.EXITS_PATTERN)[0]
        x = d[(d.key == k) & S.universe_mask(d, k, 'UA') & (d.ssr == 0)].copy()
        if not len(x):
            continue
        x['net'] = S.net_r(x, rr, why, False)
        x['gross'] = x[rr]
        x['why'] = x[why]
        x['xm'] = x[xm]
        st, tr = S.book_stats(x, 'TRAIN', weeks)
        if st is None:
            continue
        rows.append(dict(key=k, exit=lab, n=st['n'], gross=st['grossR'], net=st['meanR'],
                         cost=round(st['grossR'] - st['meanR'], 4),
                         med_spread_bps=round(float(x.spread_cc_bps.median()), 1),
                         med_r_pct=round(float(x.r_pct.median()), 2),
                         stopP=st['stopP']))
    log(pd.DataFrame(rows).to_string(index=False))

    # ---- 3. the ETF leak
    log('')
    log('## 3. The ETF leak in `asset_class == stock`')
    log('')
    a = pd.read_csv(ASSETS, dtype=str, keep_default_na=False, na_values=[''])
    nm = dict(zip(a.symbol, a.name.fillna('')))
    syms = sorted(d.symbol.unique())
    fund = {s for s in syms if FUND_RE.search(nm.get(s, '') or '')}
    unknown = [s for s in syms if s not in nm]
    log(f'- {len(fund):,} of {len(syms):,} distinct G symbols carry a fund/ETF token in the Alpaca '
        f'asset name ({len(fund)/len(syms)*100:.1f}%); {len(unknown):,} symbols have no name row '
        f'(they cannot be audited and are LEFT IN).')
    d['is_fund'] = d.symbol.isin(fund)
    log(f'- they are **{d.is_fund.mean()*100:.1f}% of the scoreable rows** '
        f'({int(d.is_fund.sum()):,} of {len(d):,}); by family: ' +
        ', '.join(f'{k} {d[d.key == k].is_fund.mean()*100:.1f}%' for k in S.FAM_KEYS))
    log(f'- regex used: `{FUND_RE.pattern}`')
    log('')
    log('### The 24 primary cells re-scored with those names removed (POST-HOC, +24 cells)')
    log('')
    res = S.grid(d[~d.is_fund], weeks, 'no_funds')
    T = pd.DataFrame(res).sort_values('TRAIN_meanR', ascending=False)
    log(T[['key', 'uni', 'exit', 'TRAIN_n', 'TRAIN_tpw', 'TRAIN_meanR', 'TRAIN_grossR', 'TRAIN_t',
           'VAL_n', 'VAL_meanR', 'VAL_t']].to_string(index=False))
    g1 = T[(T.TRAIN_meanR > 0) & (T.TRAIN_t >= S.G1_T) & (T.TRAIN_tpw >= S.MIN_TPW)]
    log('')
    log(f'**G1 without the fund names: {len(g1)} of {len(T)}**')

    # ---- 4. S5 attention minus control, in R
    log('')
    log('## 4. S5 attention minus control, same machinery, in R')
    log('')
    rows = []
    for uni in ('UA', 'UB'):
        for lab, rr, why, xm in S.EXITS_S5:
            r = dict(uni=uni, exit=lab)
            for grp in ('attention', 'control'):
                x = d[(d.key == S.S5) & S.universe_mask(d, S.S5, uni, grp) & (d.ssr == 0)].copy()
                if not len(x):
                    continue
                x['net'] = S.net_r(x, rr, why, False)
                x['gross'] = x[rr]
                x['why'] = x[why]
                x['xm'] = x[xm]
                for sp in ('TRAIN', 'VAL'):
                    st, _ = S.book_stats(x, sp, weeks)
                    r[f'{grp[:4]}_{sp}'] = st['meanR'] if st else None
            for sp in ('TRAIN', 'VAL'):
                a1, c1 = r.get(f'atte_{sp}'), r.get(f'cont_{sp}')
                r[f'diff_{sp}'] = round(a1 - c1, 4) if (a1 is not None and c1 is not None) else None
            rows.append(r)
    log(pd.DataFrame(rows).to_string(index=False))
    open(f'{G}/extras.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
