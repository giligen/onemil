"""Which store served each causal-superset symbol-day (day high >= open*1.05, all prices) under the loader order of
build_candidates.load_bars: cache.db -> topup.db -> pit_bars_1min.db -> bars.db. Resumable (per-day part file).
Writes superset_provenance.csv (symbol, bar_date, open, src)."""
import os, sqlite3
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); PR = 'research/bf_zero/parity_review'; PART = f'{PR}/superset_provenance.part.csv'
u = pd.read_csv('research/bf_zero/universe.csv', usecols=['symbol', 'bar_date', 'open', 'high'], dtype={'symbol': str}, keep_default_na=False)
for k in ('open', 'high'): u[k] = pd.to_numeric(u[k], errors='coerce')
sup = u[u.high >= u.open * 1.05][['symbol', 'bar_date', 'open']].reset_index(drop=True); del u
print('causal superset', len(sup), flush=True)
done_days = set(pd.read_csv(PART, usecols=['bar_date'], dtype=str).bar_date.unique()) if os.path.exists(PART) else set()
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
side = [('topup.db', f'{ROOT}/research/ignition_capcheck/topup.db'), ('pit_bars_1min.db', f'{ROOT}/data/research/databento/pit_bars_1min.db'), ('bars.db', f'{ROOT}/research/bf_zero/bars.db')]
cons = [(n, sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=120)) for n, p in side]
days = sorted(sup.bar_date.unique()); todo = [d for d in days if d not in done_days]
print(f'days {len(days)} done {len(done_days)} todo {len(todo)}', flush=True)
for i, d in enumerate(todo):
    g = sup[sup.bar_date == d]; syms = set(g.symbol); served = {}
    have = {s for s in syms if cache.execute("select 1 from intraday_bars_1min where symbol=? and bar_date=? limit 1", (s, d)).fetchone()}
    for s in have: served[s] = 'cache.db'
    left = syms - have
    for n, con in cons:
        if not left: break
        got = {r[0] for r in con.execute("select distinct symbol from bars where day=?", (d,)).fetchall()} & left
        for s in got: served[s] = n
        left -= got
    for s in left: served[s] = 'none'
    g.assign(src=[served[s] for s in g.symbol]).to_csv(PART, mode='a', header=not os.path.exists(PART), index=False)
    if i % 25 == 0: print(f'{i + 1}/{len(todo)} {d}', flush=True)
out = pd.read_csv(PART, dtype={'symbol': str}, keep_default_na=False).drop_duplicates(['symbol', 'bar_date'])
out.to_csv(f'{PR}/superset_provenance.csv', index=False)
print(out.src.value_counts().to_dict()); print('open>=5:', out[out.open >= 5].src.value_counts().to_dict()); print('open>=19:', out[out.open >= 19].src.value_counts().to_dict()); print('DONE', flush=True)
