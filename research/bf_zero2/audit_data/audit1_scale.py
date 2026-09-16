#!/usr/bin/env python3
"""ATTACK 1+2: price-scale mismatch between the Databento daily file (prev_close source) and the
Alpaca-SIP minute bars, and corporate-action days. Read-only. Writes audit_data/*.csv."""
import os, sqlite3, sys, json
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
A = 'research/bf_zero2/audit_data'
os.makedirs(A, exist_ok=True)

print('load daily parquet', flush=True)
daily = pd.read_parquet('data/research/databento/equs_daily_2025_2026.parquet',
                        columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna() & (daily.symbol.astype(str).str.strip() != '')]
daily = daily.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = daily.groupby('symbol')
daily['prev_close'] = g.close.shift(1)
daily['prev_date'] = g.bar_date.shift(1)
print('daily rows', len(daily), flush=True)

book = pd.read_csv(f'research/bf_zero2/f6_2r_book.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False)
print('book trades', len(book), flush=True)

uni = pd.read_csv('research/bf_zero/universe.csv', dtype={'symbol': str, 'bar_date': str}, keep_default_na=False)
rng = np.random.default_rng(7)
samp = uni.iloc[rng.choice(len(uni), size=25000, replace=False)][['symbol', 'bar_date']].copy()
samp.columns = ['symbol', 'day']
samp['src'] = 'sample'
bk = book[['symbol', 'day']].copy(); bk['src'] = 'book'
keys = pd.concat([bk, samp], ignore_index=True).drop_duplicates(['symbol', 'day'])
print('keys to check', len(keys), flush=True)

cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=180)
sip = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=180)


def minute_stats(day, syms):
    """{symbol: (first_rth_open, rth_high, rth_low, last_rth_close, nbars, store)} replicating load_bars precedence."""
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    a = pd.read_sql(q, cache, params=[day] + list(syms))
    a['store'] = 'cache'
    left = [s for s in syms if s not in set(a.symbol)]
    if left:
        b = pd.read_sql("select symbol, t, o, h, l, c from bars where day=?", sip, params=[day])
        b = b[b.symbol.isin(left)]
        b['store'] = 'sip'
        a = pd.concat([a, b], ignore_index=True) if len(a) else b
    if not len(a):
        return out
    ts = pd.to_datetime(a.t, utc=True).dt.tz_convert('America/New_York')
    a['m'] = (ts.dt.hour * 60 + ts.dt.minute).values
    a = a[(a.m >= 570) & (a.m < 960)].sort_values(['symbol', 'm']).drop_duplicates(['symbol', 'm'])
    for s, gg in a.groupby('symbol'):
        out[s] = (float(gg.o.iloc[0]), float(gg.h.max()), float(gg.l.min()), float(gg.c.iloc[-1]),
                  len(gg), gg.store.iloc[0], int(gg.m.iloc[0]))
    return out


rows = []
days = sorted(keys.day.unique())
for n, day in enumerate(days):
    syms = keys[keys.day == day].symbol.tolist()
    try:
        st = minute_stats(day, syms)
    except Exception as e:
        print('ERR', day, e, flush=True); continue
    for s in syms:
        if s in st:
            o1, hh, ll, cc, nb, store, m0 = st[s]
            rows.append(dict(day=day, symbol=s, min_open=o1, min_high=hh, min_low=ll, min_close=cc,
                             nbars=nb, store=store, first_m=m0))
    if n % 25 == 0:
        print(f'{n}/{len(days)} {day} rows {len(rows)}', flush=True)

M = pd.DataFrame(rows)
print('matched minute rows', len(M), flush=True)
D = daily[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'prev_close', 'prev_date']].rename(
    columns={'bar_date': 'day', 'open': 'd_open', 'high': 'd_high', 'low': 'd_low', 'close': 'd_close', 'volume': 'd_vol'})
M = M.merge(D, on=['symbol', 'day'], how='left')
M = M.merge(keys, on=['symbol', 'day'], how='left')
M['ratio_open'] = M.d_open / M.min_open
M['ratio_close'] = M.d_close / M.min_close
M.to_csv(f'{A}/scale_check.csv', index=False)


def rep(tag, x):
    r = x.ratio_open.dropna()
    rc = x.ratio_close.dropna()
    print(f'\n== {tag}  n={len(x)} with daily={len(r)}')
    print('  open ratio  pct |r-1|>0.1%%: %.2f%%  >0.5%%: %.2f%%  >1%%: %.2f%%  >5%%: %.2f%%  >20%%: %.2f%%' % tuple(
        100 * (np.abs(r - 1) > t).mean() for t in (0.001, 0.005, 0.01, 0.05, 0.20)))
    print('  close ratio pct |r-1|>0.1%%: %.2f%%  >0.5%%: %.2f%%  >1%%: %.2f%%  >5%%: %.2f%%  >20%%: %.2f%%' % tuple(
        100 * (np.abs(rc - 1) > t).mean() for t in (0.001, 0.005, 0.01, 0.05, 0.20)))
    print('  open ratio quantiles', r.quantile([.001, .01, .25, .5, .75, .99, .999]).round(5).to_dict())


rep('BOOK trades', M[M.src == 'book'])
rep('RANDOM universe sample', M[M.src == 'sample'])
print('\nfirst RTH minute distribution (book):', M[M.src == 'book'].first_m.value_counts().head(5).to_dict(), flush=True)
print('store mix book:', M[M.src == 'book'].store.value_counts().to_dict(), flush=True)
print('store mix sample:', M[M.src == 'sample'].store.value_counts().to_dict(), flush=True)
print('DONE', flush=True)
