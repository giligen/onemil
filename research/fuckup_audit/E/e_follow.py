#!/usr/bin/env python3
"""Stage E follow-ups, all declared in the task brief before they were run:

  A. the H6 question directly — the SAME causal candidate table split by whether the row would have
     passed the old `range_so_far_pct >= 5` floor, and by time band (early window vs later), booked.
  B. the `news_only` bucket BOOKED (PLAN §1 pre-registration: hold exit, per family) — 10 cells.
  C. for every family whose `news_only` per-trade effect passes the sign-agreement rule
     (TRAIN diff >= +0.05 R, t >= 2, VAL sign agrees): PM$ decile monotonicity inside the news
     bucket, and the Simpson's checks within price band and within time band.
  D. the Stage-C comparison: the same family x fill x exit cells on the >=5%-range universe
     (C/score5_results.csv) next to Stage E's.

Writes E/score_e_follow.md and E/score_e_newsonly.csv.  Read-only elsewhere.
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/E')
from trading.hod_break import run_book
import e_score as S                      # the scorer's contract, imported not copied

E = f'{ROOT}/research/fuckup_audit/E'
RD = dict(keep_default_na=False, na_values=[''])
pd.set_option('display.width', 250)


def log(s):
    print(f'{time.strftime("%H:%M:%S")} {s}', flush=True)


def book_row(x, split, weeks, ex, label):
    st, tr = S.book_stats(x, split, weeks, ex)
    if st is None:
        return None
    return dict(cell=label, split=split, **st)


def main():
    n_in, data = S.load()
    allr = pd.concat([data['next'], data['rest']], ignore_index=True)
    weeks = {s: sorted(allr[allr.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    L = ['# Stage E — follow-ups (H6 early-window question, news_only booked, monotonicity, Stage-C diff)', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")} | contract (c) | run_book(12,4)', '']

    d = data['next']

    # ---------------------------------------------------------------- A. the H6 question
    L += ['## A. Would the old `range_so_far_pct >= 5` floor have kept the better rows?', '',
          'Same table, same tape, same cost contract; `passed_floor` = the row the >=5%-range design could '
          'have expressed, `below_floor` = the row it discarded by construction. Booked separately '
          '(each is its own 12/4 book), `all` population, fill next, exit hold.', '',
          '| family | split | subset | n | tr/wk | mean net R | gross R | t | WR | stop% |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in S.FAM_KEYS:
        dk = d[d.key == key]
        for sp in ('TRAIN', 'VAL'):
            for nm, m in (('passed_floor', dk.range_so_far_pct >= 5), ('below_floor', dk.range_so_far_pct < 5)):
                r = book_row(dk[m], sp, weeks, 'hold', nm)
                if r:
                    L.append(f"| {key} | {sp} | {nm} | {r['n']} | {r['tpw']} | {r['meanR']} | {r['grossR']} | "
                             f"{r['t']} | {r['WR']} | {r['stopP']} |")
    L.append('')
    L += ['### share of the causal population below the old floor', '',
          pd.crosstab(d.key, d.range_so_far_pct >= 5, normalize='index').mul(100).round(1).to_string(), '']

    L += ['## A2. Booked by time band — the early window the causal universe exists to test', '',
          '`all` population, fill next, exit hold, each band booked on its own (so trades/week are '
          'band trades/week, and the bands do NOT sum to the all-day book).', '',
          '| family | split | band | n | tr/wk | mean net R | gross R | t | WR | stop% |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in S.FAM_KEYS:
        dk = d[d.key == key]
        for sp in ('TRAIN', 'VAL'):
            for _, _, nm in S.BANDS:
                r = book_row(dk[dk.band == nm], sp, weeks, 'hold', nm)
                if r:
                    L.append(f"| {key} | {sp} | {nm} | {r['n']} | {r['tpw']} | {r['meanR']} | {r['grossR']} | "
                             f"{r['t']} | {r['WR']} | {r['stopP']} |")
    L.append('')

    # ---------------------------------------------------------------- B. news_only booked
    rows = []
    for fill in S.FILLS:
        df = data[fill]
        for key in S.FAM_KEYS:
            if fill == 'rest' and key.split(' ')[0] in S.CLOSE_TRIGGERED:
                continue
            x = df[(df.key == key) & (df.bucket == 'news_only')]
            st, tr = S.book_stats(x, 'TRAIN', weeks, 'hold')
            if st is None:
                continue
            sv, _ = S.book_stats(x, 'VAL', weeks, 'hold')
            rows.append(dict(key=key, fill=fill, **{f'TR_{k}': v for k, v in st.items()},
                             **{f'VA_{k}': (sv or {}).get(k) for k in
                                ('n', 'tpw', 'meanR', 'grossR', 't', 'wkR', 'wkSE', 'green')}))
    NO = pd.DataFrame(rows).sort_values('TR_meanR', ascending=False)
    NO.to_csv(f'{E}/score_e_newsonly.csv', index=False)
    g1 = NO[(NO.TR_meanR > 0) & (NO.TR_t >= S.G1_T) & (NO.TR_tpw >= S.MIN_TPW)]
    L += ['## B. The `news_only` bucket AS A BOOK (PLAN §1 pre-registration; hold exit) — 10 cells', '',
          NO.to_string(index=False), '',
          f'G1 passes: **{len(g1)} of {len(NO)}**', '']

    # ---------------------------------------------------------------- C. sign-agreement survivors
    B = pd.read_csv(f'{E}/score_e_buckets.csv', **RD)
    nb = B[(B.bucket == 'news_only') & (B.scope == 'all-day')]
    piv = nb.pivot_table(index='key', columns='split', values=['diff', 't'])
    surv = []
    for key in S.FAM_KEYS:
        try:
            dt, tt, dv = piv.loc[key, ('diff', 'TRAIN')], piv.loc[key, ('t', 'TRAIN')], piv.loc[key, ('diff', 'VAL')]
        except KeyError:
            continue
        if dt >= 0.05 and tt >= 2.0 and np.sign(dv) == np.sign(dt):
            surv.append(key)
    L += ['## C. The pre-registered sign-agreement rule on `news_only` (TRAIN diff >= +0.05 R, t >= 2, VAL sign agrees)',
          '', nb[['key', 'split', 'n', 'meanR', 'rest_meanR', 'diff', 't']].to_string(index=False), '',
          f'**survivors: {surv if surv else "(none)"}**', '']

    for key in surv:
        dk = d[(d.key == key)]
        nk = dk[dk.b_news == 1]
        L += [f'### C1. `{key}` — is the effect MONOTONE in the PM$ level inside the news bucket?', '',
              'deciles of `pm_dollar_vol` among news-carrying rows (NaN premarket is its own row); '
              'the ORB gate says HIGH PM$ should be the good bucket.', '']
        for sp in ('TRAIN', 'VAL'):
            s = nk[nk.split == sp].copy()
            if len(s) < 100:
                continue
            s['dec'] = pd.qcut(s.pm_dollar_vol.rank(method='first'), 10, labels=False, duplicates='drop')
            s.loc[s.pm_dollar_vol.isna(), 'dec'] = -1
            tb = s.groupby('dec').net_hold.agg(['size', 'mean']).round(4)
            tb['pm_med'] = s.groupby('dec').pm_dollar_vol.median().round(0)
            L += [f'**{sp}**', '', tb.to_string(), '']
            good = tb.drop(index=-1, errors='ignore')
            if len(good) > 2:
                rho = np.corrcoef(good.index.values.astype(float), good['mean'].values)[0, 1]
                L += [f'rank correlation decile vs mean net R: **{rho:+.2f}** '
                      f'(the ORB gate predicts strongly positive)', '']
        L += [f'### C2. `{key}` — Simpson\'s check: the news_only-minus-rest difference WITHIN price band '
              f'and WITHIN time band', '',
              '| split | slice | n(news_only) | news_only R | rest R | diff | t |', '|---|---|---:|---:|---:|---:|---:|']
        dk = dk.copy()
        dk['pb'] = pd.cut(dk.price, [0, 10, 20, 50, 100, 1e9], labels=['5-10', '10-20', '20-50', '50-100', '100+'])
        for sp in ('TRAIN', 'VAL', 'TEST'):
            s = dk[dk.split == sp]
            for slicer, vals in (('price', s.pb), ('band', s.band)):
                for v in sorted(set(vals.dropna().astype(str))):
                    ss = s[vals.astype(str) == v]
                    a = ss[ss.bucket == 'news_only'].net_hold.dropna().values
                    b = ss[ss.bucket != 'news_only'].net_hold.dropna().values
                    if len(a) < 30 or len(b) < 30:
                        continue
                    dm, tt = S.welch(a, b)
                    L.append(f'| {sp} | {slicer} {v} | {len(a)} | {a.mean():.4f} | {b.mean():.4f} | '
                             f'{dm:+.4f} | {tt:.2f} |')
        L.append('')
        # sensitivity: drop rows whose premarket value is unknown (the 11% availability residual)
        k2 = dk[dk.pm_dollar_vol.notna()]
        L += ['**sensitivity — the same all-day difference with the 11% pm-unknown rows dropped**', '',
              '| split | n(news_only) | news_only R | rest R | diff | t |', '|---|---:|---:|---:|---:|---:|']
        for sp in ('TRAIN', 'VAL', 'TEST'):
            s = k2[k2.split == sp]
            a = s[s.bucket == 'news_only'].net_hold.dropna().values
            b = s[s.bucket != 'news_only'].net_hold.dropna().values
            if len(a) < 30:
                continue
            dm, tt = S.welch(a, b)
            L.append(f'| {sp} | {len(a)} | {a.mean():.4f} | {b.mean():.4f} | {dm:+.4f} | {tt:.2f} |')
        L.append('')
        # the trade list the headline sample will be drawn from
        samp = dk[(dk.split == 'TRAIN') & (dk.bucket == 'news_only')][['day', 'symbol', 'entry_m', 'net_hold']]
        samp.drop_duplicates(['day', 'symbol']).sample(min(30, len(samp)), random_state=7).to_csv(
            f'{E}/score_e_headline_sample.csv', index=False)

    # ---------------------------------------------------------------- D. Stage-C comparison
    try:
        C = pd.read_csv(f'{ROOT}/research/fuckup_audit/C/score5_results.csv', **RD)
        C = C[C.key.isin(S.FAM_KEYS) & C.outcome.isin(['hold-to-close', '2R close-fill'])]
        C['exit'] = C.outcome.map({'hold-to-close': 'hold', '2R close-fill': '2r'})
        Ecsv = pd.read_csv(f'{E}/score_e_results.csv', **RD)
        Ea = Ecsv[Ecsv['pop'] == 'all']
        M = Ea.merge(C[['key', 'fill', 'exit', 'TRAIN_meanR', 'TRAIN_t', 'TRAIN_tpw', 'VAL_meanR', 'VAL_t']],
                     on=['key', 'fill', 'exit'], how='left', suffixes=('_E', '_C'))
        M['d_TRAIN'] = (M.TR_meanR - M.TRAIN_meanR).round(4)
        L += ['## D. Stage E (causal universe, all-day) vs Stage C (>=5%-range universe, all-day) — '
              'the same family x fill x exit cells, `all` population', '',
              M[['key', 'fill', 'exit', 'TR_meanR', 'TRAIN_meanR', 'd_TRAIN', 'TR_t', 'TRAIN_t',
                 'TR_tpw', 'TRAIN_tpw', 'VA_meanR', 'VAL_meanR']].to_string(index=False), '',
              f'mean TRAIN difference (E minus C) over {M.d_TRAIN.notna().sum()} matched cells: '
              f'**{M.d_TRAIN.mean():+.4f} R**', '']
    except Exception as e:
        L += [f'## D. Stage-C comparison unavailable: {e}', '']

    open(f'{E}/score_e_follow.md', 'w').write('\n'.join(L))
    log(f'wrote {E}/score_e_follow.md')


if __name__ == '__main__':
    main()
