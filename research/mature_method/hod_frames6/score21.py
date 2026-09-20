#!/usr/bin/env python3
"""hod_frames6 / F21 — SURVIVORSHIP.  Cells exactly as declared in PREREG.md.

Every one of the 989 prior cells intersected the break stream with TODAY's `daily_bars`.  This
quantifies that intersection on the point-in-time HOD universe (Databento EQUS.SUMMARY daily panel),
measures the DIRECTION of the bias for a long-only intraday book that is flat by 15:55, and applies
the pre-committed materiality rule.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import D6, ROOT, attach_instrument                                # noqa: E402
from research.scripts.pit_listings import is_test_ticker                       # noqa: E402

DB = f'{ROOT}/data/research/databento'
MIN_PREV_CLOSE, MIN_ADV20 = 17.0, 100_000
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'          # TRAIN+VAL only — TEST sealed


def main():
    print('== F21 · building the point-in-time HOD universe ==', flush=True)
    parts = []
    for f in ('equs_daily_2024H2.parquet', 'equs_daily_2025_2026.parquet'):
        d = pd.read_parquet(f'{DB}/{f}', columns=['bar_date', 'symbol', 'open', 'high', 'low',
                                                  'close', 'volume'])
        for c in ('open', 'high', 'low', 'close', 'volume'):
            d[c] = d[c].astype('float32')
        d['symbol'] = d.symbol.astype('category')
        parts.append(d)
    u = pd.concat(parts, ignore_index=True)
    del parts, d
    u['symbol'] = u.symbol.astype(str)
    u = u[(u.symbol != 'None') & (u.symbol != 'nan') & (u.bar_date <= DAY_HI)]
    u = u[~u.symbol.map(is_test_ticker)]
    u = u.sort_values(['symbol', 'bar_date'], kind='mergesort').reset_index(drop=True)
    u['symbol'] = u.symbol.astype('category')
    pos = u.groupby('symbol', sort=False, observed=True).cumcount().values
    u['prev_close'] = u.close.shift(1).astype('float32')
    # 20-session mean volume, strictly prior.  The panel starts 2024-07 and the window 2025-01, so
    # every name listed before the window has >= 120 prior sessions; requiring a FULL 20 rather than
    # the engine's min_periods=15 can only differ for a name in its first sessions, and that
    # difference is stated rather than hidden.
    u['adv20'] = u.volume.rolling(20).mean().shift(1).astype('float32')
    bad = pos < 21
    u.loc[bad, ['prev_close', 'adv20']] = np.nan
    pit = u[(u.bar_date >= DAY_LO) & (u.prev_close >= MIN_PREV_CLOSE) &
            (u.adv20 >= MIN_ADV20)].copy()
    del u
    pit['symbol'] = pit.symbol.astype(str)
    pit['year'] = pit.bar_date.str[:4]
    print(f'  PIT HOD-universe symbol-days {len(pit):,} '
          f'({pit.symbol.nunique():,} distinct symbols, {pit.bar_date.nunique()} sessions)',
          flush=True)

    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    pit['present'] = pit.symbol.isin(dbs)
    print(f'  today\'s `daily_bars` carries {len(dbs):,} distinct symbols\n', flush=True)

    # ------------------------------------------------------------------ F21-1
    print('== F21-1  share of PIT HOD-universe symbol-days ABSENT from today\'s daily_bars ==',
          flush=True)
    print('| period | symbol-days | absent | share | distinct symbols | absent symbols |', flush=True)
    print('|---|---|---|---|---|---|', flush=True)
    rows = []
    for lab, d in list(pit.groupby('year')) + [('ALL', pit)]:
        ab = int((~d.present).sum())
        na = d[~d.present].symbol.nunique()
        print(f'| {lab} | {len(d):,} | {ab:,} | **{ab / len(d) * 100:.2f} %** | '
              f'{d.symbol.nunique():,} | {na:,} |', flush=True)
        rows.append(dict(cell='F21-1', period=lab, n=len(d), absent=ab, share=ab / len(d) * 100,
                         n_sym=d.symbol.nunique(), n_absent_sym=na))
    mo = pit.groupby(pit.bar_date.str[:7]).present.apply(lambda x: (~x).mean() * 100)
    print(f'\n  monthly absent share: min {mo.min():.2f} % ({mo.idxmin()}), '
          f'max {mo.max():.2f} % ({mo.idxmax()}), median {mo.median():.2f} %', flush=True)

    # ------------------------------------------------------------------ F21-2
    print('\n== F21-2  wrapper vs common, and by price / ADV$ band ==', flush=True)
    syms = pd.DataFrame({'symbol': sorted(pit.symbol.unique()), 'day': DAY_LO})
    syms = attach_instrument(syms)
    cls = dict(zip(syms.symbol, syms.asset_class))
    pit['asset_class'] = pit.symbol.map(cls)
    print('| cohort | symbol-days | absent | share |', flush=True)
    print('|---|---|---|---|', flush=True)
    for c, d in pit.groupby('asset_class'):
        ab = int((~d.present).sum())
        print(f'| {c} | {len(d):,} | {ab:,} | {ab / len(d) * 100:.2f} % |', flush=True)
        rows.append(dict(cell='F21-2', period=c, n=len(d), absent=ab, share=ab / len(d) * 100))
    pit['pb'] = pd.cut(pit.prev_close, [0, 20, 30, 50, 100, 1e9],
                       labels=['<$20', '$20-30', '$30-50', '$50-100', '$100+'])
    pit['ab'] = pd.cut(pit.adv20 * pit.prev_close, [0, 10e6, 50e6, 200e6, 1e18],
                       labels=['<$10M', '$10-50M', '$50-200M', '>=$200M'])
    for col in ('pb', 'ab'):
        for c, d in pit.groupby(col, observed=True):
            ab = int((~d.present).sum())
            print(f'| {col}={c} | {len(d):,} | {ab:,} | {ab / len(d) * 100:.2f} % |', flush=True)

    # ------------------------------------------------------------------ F21-3 direction
    print('\n== F21-3  DIRECTION — what the absent names\' own sessions look like ==', flush=True)
    pit['oc'] = (pit.close / pit.open - 1.0) * 100
    pit['rng'] = (pit.high - pit.low) / pit.open * 100
    print('| cohort | n | mean open->close % | median | mean range % | share of days +5 % | '
          'share of days -5 % |', flush=True)
    print('|---|---|---|---|---|---|---|', flush=True)
    for lab, d in (('PRESENT in daily_bars', pit[pit.present]),
                   ('ABSENT from daily_bars', pit[~pit.present])):
        print(f'| {lab} | {len(d):,} | {d.oc.mean():+.3f} | {d.oc.median():+.3f} | '
              f'{d.rng.mean():.2f} | {(d.oc >= 5).mean() * 100:.2f} % | '
              f'{(d.oc <= -5).mean() * 100:.2f} % |', flush=True)
        rows.append(dict(cell='F21-3', period=lab, n=len(d), oc_mean=float(d.oc.mean()),
                         rng_mean=float(d.rng.mean())))
    a, b = pit[pit.present].oc, pit[~pit.present].oc
    dm = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    print(f'\n  absent minus present, mean open->close: {dm:+.3f} pp (SE {se:.3f}, t {dm / se:+.2f})',
          flush=True)
    print('  Interpretation rail: a long-only book flat by 15:55 earns the INTRADAY path, not the '
          'overnight gap; a positive difference means the survivor filter dropped days that would '
          'have HELPED, a negative one means it dropped days that would have HURT.', flush=True)

    # -------------------------------------------------- F21-3b  the CANDIDATE-generating subset
    print('\n== F21-3b  the absent share among CANDIDATE-generating days (range >= 5 %) ==',
          flush=True)
    print('  Breaks come from movers, not from the average symbol-day, so the share that matters is', flush=True)
    print('  the one on days wide enough to produce a >= 5 %-above-open high.', flush=True)
    print('| cohort | symbol-days | absent | share | mean open->close % of the absent |', flush=True)
    print('|---|---|---|---|---|', flush=True)
    for lab, thr in (('all days', 0.0), ('range >= 5 %', 5.0), ('range >= 10 %', 10.0)):
        d = pit[pit.rng >= thr]
        ab = d[~d.present]
        print(f'| {lab} | {len(d):,} | {len(ab):,} | {len(ab) / len(d) * 100:.2f} % | '
              f'{ab.oc.mean():+.3f} (present {d[d.present].oc.mean():+.3f}) |', flush=True)
        rows.append(dict(cell='F21-3b', period=lab, n=len(d), absent=len(ab),
                         share=len(ab) / len(d) * 100, oc_mean=float(ab.oc.mean())))

    # ------------------------------------------------------------------ F21-4 materiality
    print('\n== F21-4  materiality — do the absent names have 1-minute bars at all? ==', flush=True)
    absent_syms = sorted(pit[~pit.present].symbol.unique())
    print(f'  {len(absent_syms)} distinct absent symbols; probing the study\'s own bar stores',
          flush=True)
    found_c = found_s = 0
    if absent_syms:
        con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
        chunk = 500
        for i in range(0, len(absent_syms), chunk):
            cc = absent_syms[i:i + chunk]
            q = ('select count(distinct symbol) n from intraday_bars_1min where symbol in ('
                 + ','.join('?' * len(cc)) + ') and bar_date between ? and ?')
            found_c += int(pd.read_sql(q, con, params=cc + [DAY_LO, DAY_HI]).n.iloc[0])
        con.close()
        sp = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True,
                             timeout=120)
        for i in range(0, len(absent_syms), chunk):
            cc = absent_syms[i:i + chunk]
            q = ('select count(distinct symbol) n from bars where symbol in ('
                 + ','.join('?' * len(cc)) + ') and day between ? and ?')
            found_s += int(pd.read_sql(q, sp, params=cc + [DAY_LO, DAY_HI]).n.iloc[0])
        sp.close()
    print(f'  1-min bars present for {found_c} of them in cache.db, {found_s} in bars_sip.db',
          flush=True)
    rows.append(dict(cell='F21-4', period='absent_with_1min_bars', n=len(absent_syms),
                     absent=found_c + found_s))

    # ------------------------------------------------------------------ the pre-committed rule
    sh = {r['period']: r['share'] for r in rows if r['cell'] == 'F21-1'}
    small = all(sh[y] < 5.0 for y in ('2025', '2026'))
    print('\n== F21 · THE PRE-COMMITTED MATERIALITY RULE ==', flush=True)
    print(f'  absent share 2025 {sh["2025"]:.2f} % · 2026 {sh["2026"]:.2f} % — '
          f'both < 5 %: {small}', flush=True)
    print(f'  direction: absent minus present open->close {dm:+.3f} pp (t {dm / se:+.2f})',
          flush=True)
    verdict = ('NOT load-bearing' if small else 'LOAD-BEARING — every era-consistency verdict in '
                                                'eleven passes is computed on a survivor set')
    print(f'\n  >>> SURVIVORSHIP: **{verdict}** <<<\n', flush=True)
    pd.DataFrame(rows).to_csv(f'{D6}/cells21.csv', index=False)
    print('cells21.csv written', flush=True)


if __name__ == '__main__':
    main()
