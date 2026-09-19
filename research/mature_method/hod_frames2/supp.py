#!/usr/bin/env python3
"""hod_frames2 — supplementary diagnostics (S1..S6).  No decision cell here; every number below is
descriptive and is labelled as such in REPORT.md.  Read-only.  One process."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import importlib.util                                      # noqa: E402
import score as S            # noqa: E402  (hod_break/score.py — the cost model + book + week stats)
import score2 as S2          # noqa: E402
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

_sp = importlib.util.spec_from_file_location(
    'hf2_score', f'{ROOT}/research/mature_method/hod_frames2/score.py')
_hf2 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_hf2)
sigset, daily_ctx, clustered_t = _hf2.sigset, _hf2.daily_ctx, _hf2.clustered_t

D = f'{ROOT}/research/mature_method/hod_frames2'
RD = dict(dtype={'symbol': str, 'day': str, 'why_n': str}, keep_default_na=False, na_values=[''])
SPLITS = ('TRAIN', 'VAL')


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    br = pd.read_csv(f'{D}/breaks2.csv', **RD)
    br = br[~br.day.isin(S.EARLY_CLOSE)]
    br = br[~br.symbol.map(lambda s: is_test_ticker(str(s)))]
    br['split'] = S.split_of(br.day.values)
    br['wk'] = pd.to_datetime(br.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    br = br.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']],
                  on=['day', 'symbol', 'entry_m'], how='left')
    br = br.merge(daily_ctx(), on=['day', 'symbol'], how='left')
    br = br.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)
    br['rng_after'] = br.rng_day - br.rng_sig
    br['dollar_frac'] = br.cum_dollar / br.adv_dollar.replace(0, np.nan) * 100
    br['add30_ratio'] = np.where(br.break_m >= 630,
                                 (br.rng_sig - br.rng_30) / br.rng_first30.replace(0, np.nan), np.nan)
    br['rng_own'] = br.rng_sig / br.med_rng.replace(0, np.nan)
    PRE = sigset(br); PRE0 = PRE[PRE.n_prior == 0]

    print('\n== S1 — F5 at the SIGNAL level (pre-book gross; the 12/4 slot rule is NOT applied) ==')
    print('| admission | split | signals | /wk | gross R | +/- | WR | median entry minute | imp % |')
    F5 = {'first break': PRE.n_prior == 0,
          '2nd, prev STOPPED': (PRE.n_prior == 1) & (PRE.prev_stopped == 1),
          '2nd, prev back<5': (PRE.n_prior == 1) & (PRE.prev_back5 == 1),
          '2nd, prev back<15': (PRE.n_prior == 1) & (PRE.prev_back15 == 1),
          '2nd, stopped OR back15': (PRE.n_prior == 1) & ((PRE.prev_stopped == 1) | (PRE.prev_back15 == 1)),
          'ANY re-break': PRE.n_prior >= 1, '3rd+': PRE.n_prior >= 2}
    for nm, mk in F5.items():
        x = PRE[mk]
        for sp in SPLITS:
            d = x[x.split == sp]
            if len(d) < 3:
                continue
            se = d.rr.std(ddof=1) / np.sqrt(len(d))
            print(f'| {nm:<24s} | {sp} | {len(d):5d} | {len(d)/S.NW[sp]:5.1f} | {d.rr.mean():+.3f} | '
                  f'{se:.3f} | {(d.rr>0).mean():.1%} | {int(d.entry_m.median())} | '
                  f'{d.imputed.mean()*100:.0f} |')

    print('\n== S2 — gate separation (kept minus rejected) at the declared threshold, on the B2 '
          'pre-book set (RUNBOOK step 5) ==')
    print('| field | thr | split | n kept | n rej | gross kept | gross rej | d gross | iid t |')
    FIELDS = [('shelf_share', 2), ('shelf_share', 5), ('shelf_share', 10), ('shelf_share', 20),
              ('shelf_bars', 5), ('shelf_bars', 10), ('shelf_bars', 20), ('hod_age_bars', 20),
              ('dollar_frac', 10), ('dollar_frac', 25), ('dollar_frac', 50),
              ('exp5_n', 3), ('exp5_n', 6), ('add30_ratio', 1.0), ('add30_ratio', 2.0),
              ('rng_own', 0.5), ('rng_own', 1.0), ('rng_own', 1.5)]
    for f, thr in FIELDS:
        for sp in SPLITS:
            d = PRE0[(PRE0.split == sp) & PRE0[f].notna()]
            k, r = d[d[f] >= thr], d[d[f] < thr]
            if len(k) < 5 or len(r) < 5:
                continue
            dm = k.rr.mean() - r.rr.mean()
            se = np.sqrt(k.rr.var(ddof=1) / len(k) + r.rr.var(ddof=1) / len(r))
            print(f'| {f} | {thr} | {sp} | {len(k)} | {len(r)} | {k.rr.mean():+.3f} | '
                  f'{r.rr.mean():+.3f} | {dm:+.3f} | {dm/se if se else np.nan:+.2f} |')

    print('\n== S3 — the two-cohort enrichment of every declared admission (share of trades on '
          '>=10 %-range days; the hod_fresh diagnostic) ==')
    print('| admission | TRAIN >=10 % | VAL >=10 % | TRAIN n | VAL n |')
    ADM = {'B2 first break': PRE0.index}
    for f, thr in FIELDS:
        ADM[f'{f}>={thr}'] = PRE0[PRE0[f].notna() & (PRE0[f] >= thr)].index
    for nm, idx in ADM.items():
        d = PRE0.loc[idx]
        t = d[d.split == 'TRAIN']; v = d[d.split == 'VAL']
        if len(t) < 5 or len(v) < 5:
            continue
        print(f'| {nm} | {(t.rng_day>=10).mean():.0%} | {(v.rng_day>=10).mean():.0%} | {len(t)} | {len(v)} |')

    print('\n== S4 — weekly dollar path, VAL, of the three best-dollar cells per frame '
          '(cells.csv order) ==')
    cf = pd.read_csv(f'{D}/cells.csv')
    for sp in SPLITS:
        best = cf[(cf.split == sp) & (cf.per_wk >= 10)].sort_values('total', ascending=False).head(4)
        print(f'  {sp} best-dollar cells at >=10/wk: ' +
              ' | '.join(f'{r.cell} ${r.total:,.0f} ({r.green:.0f}% green, tc {r.tc:+.2f})'
                         for r in best.itertuples()))
    print('\n== S5 — MDE on green weeks (the count-matched null width) is in nulls.csv ==')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
