"""Which store served each causal-superset symbol-day (day high >= open*1.05, all prices) under the loader order of
build_candidates.load_bars: cache.db -> topup.db -> pit_bars_1min.db -> bars.db. Writes superset_provenance.csv."""
import os, sqlite3
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
u = pd.read_csv('research/bf_zero/universe.csv', usecols=['symbol', 'bar_date', 'open', 'high'], dtype={'symbol': str}, keep_default_na=False)
for k in ('open', 'high'): u[k] = pd.to_numeric(u[k], errors='coerce')
sup = u[u.high >= u.open * 1.05][['symbol', 'bar_date', 'open']].reset_index(drop=True)
print('causal superset', len(sup), flush=True)
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
side = [('topup.db', f'{ROOT}/research/ignition_capcheck/topup.db'), ('pit_bars_1min.db', f'{ROOT}/data/research/databento/pit_bars_1min.db'), ('bars.db', f'{ROOT}/research/bf_zero/bars.db')]
cons = [(n, sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=120)) for n, p in side]
src = []
by_day = {d: set(g.symbol) for d, g in sup.groupby('bar_date')}
served = {}
for i, (d, syms) in enumerate(sorted(by_day.items())):
    have = {s for s in syms if cache.execute("select 1 from intraday_bars_1min where symbol=? and bar_date=? limit 1", (s, d)).fetchone()}
    for s in have: served[(s, d)] = 'cache.db'
    left = syms - have
    for n, con in cons:
        if not left: break
        got = {r[0] for r in con.execute("select distinct symbol from bars where day=?", (d,)).fetchall()} & left
        for s in got: served[(s, d)] = n
        left -= got
    for s in left: served[(s, d)] = 'none'
    if i % 50 == 0: print(f'{i + 1}/{len(by_day)} {d}', flush=True)
sup['src'] = [served[(s, d)] for s, d in zip(sup.symbol, sup.bar_date)]
sup.to_csv('research/bf_zero/parity_review/superset_provenance.csv', index=False)
print(sup.src.value_counts().to_dict()); print('open>=5:', sup[sup.open >= 5].src.value_counts().to_dict()); print('open>=19:', sup[sup.open >= 19].src.value_counts().to_dict())
