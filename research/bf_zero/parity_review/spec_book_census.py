#!/usr/bin/env python3
"""Census of the spec's executable book (research/bf_zero/spec_book.csv): which bar source served each traded
symbol-day under B.load_bars' precedence (cache.db first, then topup.db / pit_bars_1min.db / bars.db — the three
side DBs are all Databento EQUS.MINI ohlcv-1m), whether the day's RTH bars start at 09:30, rv/HOD stats by source,
and Databento-only symbol names. Side DBs are queried per key (PK lookups); a key absent from every side DB was
served by cache.db (Alpaca SIP) — verified on the TEST split with direct cache.db lookups. Read-only."""
import os, sys, sqlite3, time
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
OUT = f'{ROOT}/research/bf_zero/parity_review'
t0 = time.time()
bk = pd.read_csv(f'{ROOT}/research/bf_zero/spec_book.csv', dtype={'symbol': str})
sides = [(os.path.basename(p), sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=10)) for p in
         (f'{ROOT}/research/ignition_capcheck/topup.db', f'{ROOT}/data/research/databento/pit_bars_1min.db', f'{ROOT}/research/bf_zero/bars.db') if os.path.exists(p)]
rows = []
for r in bk.itertuples():
    src = 'cache.db(Alpaca SIP)'; first = None; n = 0
    for name, con in sides:
        q = con.execute("select min(t), count(*) from bars where symbol=? and day=?", (r.symbol, r.day)).fetchone()
        if q and q[1]: src = name + '(Databento EQUS.MINI)'; n = q[1]; first = q[0]; break
    fm = None
    if first:
        t = pd.Timestamp(first); t = t.tz_localize('UTC') if t.tzinfo is None else t
        et = t.tz_convert('America/New_York'); fm = et.hour * 60 + et.minute
    rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, source=src, n_bars=n, first_minute=fm, entry_m=r.entry_m, rr=r.rr, rv=r.rv_profile, adv20=r.adv20, level=r.level))
S = pd.DataFrame(rows)
print(f'side-db pass {time.time() - t0:.0f}s', flush=True)
# verify the cache attribution + 09:30 presence on the TEST split (direct cache.db PK/index lookups)
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=5)
test_cache = S[(S.split == 'TEST') & S.source.str.startswith('cache')]
miss = 0; no930 = 0; checked = 0
for r in test_cache.itertuples():
    try:
        q = cache.execute("select min(timestamp), count(*) from intraday_bars_1min where symbol=? and bar_date=?", (r.symbol, r.day)).fetchone()
    except Exception as e:
        print('cache lookup failed', e); break
    checked += 1
    if not q or not q[1]: miss += 1; continue
    S.loc[r.Index, 'n_bars'] = q[1]
    ts = pd.read_sql("select timestamp from intraday_bars_1min where symbol=? and bar_date=?", cache, params=(r.symbol, r.day)).timestamp
    m = pd.to_datetime(ts, utc=True).dt.tz_convert('America/New_York').pipe(lambda t: t.dt.hour * 60 + t.dt.minute)
    fm = int(m[m >= 570].min()) if (m >= 570).any() else None
    S.loc[r.Index, 'first_minute'] = fm
    if fm != 570: no930 += 1
print(f'cache verification (TEST split, {checked} cache-attributed trades): not found in cache.db {miss}, first RTH bar != 09:30: {no930} | {time.time() - t0:.0f}s', flush=True)
S.to_csv(f'{OUT}/spec_book_sources.csv', index=False)
lines = [f'=== spec book census: {len(S)} trades']
lines.append('source x split:\n' + pd.crosstab(S.source, S.split).to_string())
lines.append('by source: n, meanR, rv median, first_minute!=09:30 count:\n' + S.groupby('source').agg(n=('rr', 'size'), meanR=('rr', 'mean'), rv_med=('rv', 'median'), rv_p90=('rv', lambda x: x.quantile(.9)),
                                                                                  no930=('first_minute', lambda x: int(((x.notna()) & (x != 570)).sum()))).round(3).to_string())
hy = S[S.symbol.str.contains('-', regex=False) | S.symbol.str.contains(' ', regex=False)]
lines.append(f'Databento-only symbol names (hyphen/space) in the book: {len(hy)} trades: {hy.symbol.unique()[:12].tolist()}')
lines.append(f'weekly R share from Databento-sourced trades: {S[S.source.str.contains("Databento")].rr.sum():+.1f} of {S.rr.sum():+.1f} total; TEST: {S[(S.split == "TEST") & S.source.str.contains("Databento")].rr.sum():+.1f} of {S[S.split == "TEST"].rr.sum():+.1f}')
print('\n'.join(lines), flush=True)
open(f'{OUT}/spec_book_census.txt', 'w').write('\n'.join(lines) + '\n')
