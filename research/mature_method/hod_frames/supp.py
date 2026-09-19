#!/usr/bin/env python3
"""hod_frames — supplementary diagnostics (no new decision cell; each is a described number)."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames'
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
S.SPLITS = S2.SPLITS = SPLITS = ('TRAIN', 'VAL')


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    sig = S2.sig_set(pop, **S2.BASES['B2'])
    b2 = S.apply_book(sig, 12, 4)
    rng = pd.read_csv(f'{D}/range.csv', **RD).set_index(['day', 'symbol'])
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])[['day', 'spy_r5_pct']]
    b = b2.join(rng[['rng_720', 'rng_day']], on=['day', 'symbol']).join(df.set_index('day'), on='day')

    print('== S1. THE TWO-COHORT FINDING, TESTED DIRECTLY ON THIS BOOK ==')
    print('bf_zero §6b: EOD-range>=10% days +0.43R, others -0.55R, every split. Here, on the B2 book,')
    print('the SAME oracle split (EOD RTH range, UNOBSERVABLE at the signal -- this is the ceiling of')
    print('anything Frame 2 could reach):')
    print('| split | cohort | n | grossR | netR | green % | total $ |')
    print('|---|---|---|---|---|---|---|')
    for sp in SPLITS:
        for lab, m in (('EOD range >= 10%', b.rng_day >= 10), ('EOD range < 10%', b.rng_day < 10)):
            d = b[(b.split == sp) & m.fillna(False)]
            w = S.week_stats(d, sp)
            print(f'| {sp} | {lab} | {len(d)} | {d.rr.mean():+.3f} | {d.net.mean():+.3f} | '
                  f'{w["green"]:.1f} | {w["total"]:.0f} |')
    print('\n   (pre-book signal set, same split:)')
    s = sig.join(rng[['rng_day']], on=['day', 'symbol'])
    for sp in SPLITS:
        d = s[s.split == sp]
        print(f'   {sp}: >=10% n {int((d.rng_day>=10).sum())} grossR '
              f'{d[d.rng_day>=10].rr.mean():+.4f} | <10% n {int((d.rng_day<10).sum())} grossR '
              f'{d[d.rng_day<10].rr.mean():+.4f}')

    print('\n== S2. FRAME 2 — where the frequency goes ==')
    for T in (660, 720, 780):
        d = sig[sig.entry_m > T]
        print(f'  B2 book signals with entry_m > {T}: TRAIN {len(d[d.split=="TRAIN"])} '
              f'({len(d[d.split=="TRAIN"])/S.NW["TRAIN"]:.1f}/wk) VAL {len(d[d.split=="VAL"])} '
              f'({len(d[d.split=="VAL"])/S.NW["VAL"]:.1f}/wk)  [the time restriction alone, '
              f'before any range filter]')

    print('\n== S3. FRAME 3 — S4 null re-specified (the declared day-level shuffle is wrong for a '
          'TRADE-level key) ==')
    rgen = np.random.default_rng(23)
    z = b.copy(); k = (z.rng_720 >= 7.0).fillna(False)
    z['pnl'] = z.net * S.RISK * np.where(k, 2.0, 0.5)
    for sp in SPLITS:
        d = z[z.split == sp]; kk = k[z.split == sp].values
        obs = S.week_stats(d, sp)
        tot, grn = [], []
        for _ in range(2000):
            p = d.net.values * S.RISK * np.where(rgen.permutation(kk), 2.0, 0.5)
            w = pd.Series(p).groupby(d.wk.values).sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
            tot.append(w.sum()); grn.append((w > 0).mean() * 100)
        print(f'  S4 {sp:5s} TRADE-level shuffle: total$ obs {obs["total"]:+8.0f} null '
              f'{np.mean(tot):+8.0f} [{np.percentile(tot,5):+.0f}, {np.percentile(tot,95):+.0f}]  '
              f'green% obs {obs["green"]:.1f} null {np.mean(grn):.1f} '
              f'[{np.percentile(grn,5):.1f}, {np.percentile(grn,95):.1f}]')

    print('\n== S4. FRAME 1 — the short leg\'s cost and borrow, and what the long book pays ==')
    sh = pd.read_csv(f'{D}/short.csv', **RD)
    sh = sh[~sh.day.isin(S.EARLY_CLOSE)]
    sh = sh[~sh.symbol.map(is_test_ticker)]
    sh['split'] = S.split_of(sh.day.values)
    fb = sh[(sh.trig == 'fb') & (sh.stop_var == 'mfe') & sh.rr_tls.notna() & (sh.r_pct_s >= 1.0) &
            (sh.short_px >= 20.0)]
    for sp in SPLITS:
        d = fb[fb.split == sp]
        print(f'  {sp}: short R as % of price median {d.r_pct_s.median():.2f} (long book '
              f'{sig[sig.split==sp].r_pct.median():.2f}); short R/long R median '
              f'{(d.r_short/d.r_long).median():.2f}')
    bfl = pd.read_csv(f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv',
                      dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    B = set(bfl[(bfl.shortable.astype(str) == 'True') &
                (bfl.easy_to_borrow.astype(str) == 'True')].symbol)
    known = set(bfl.symbol)
    u = sh.drop_duplicates(['day', 'symbol', 'entry_m', 'trig'])
    print(f'  borrow: of {u.symbol.nunique()} distinct short symbols, '
          f'{len([x for x in u.symbol.unique() if x in known])} are in today\'s Alpaca asset list '
          f'and {len([x for x in u.symbol.unique() if x in B])} are shortable AND easy_to_borrow; '
          f'signal-weighted tradeable share {u.symbol.isin(B).mean():.1%} '
          f'(absent-from-list share {1-u.symbol.isin(known).mean():.1%} counted NOT borrowable)')

    print('\n== S5. MDE per frame (80% power, per trade, on net R) ==')
    for lab, d in (('B2 pre-book (the long population)', sig),
                   ('Frame 1 fb/mfe pre-book short population', fb)):
        for sp in SPLITS:
            x = d[d.split == sp]
            col = x.net if 'net' in x else None
            v = x.net if col is not None else x.rr_tls
            print(f'  {lab:42s} {sp:5s} n {len(x):5d}  MDE {2.80*v.std(ddof=1)/np.sqrt(len(x)):.3f} R')


if __name__ == '__main__':
    main()
