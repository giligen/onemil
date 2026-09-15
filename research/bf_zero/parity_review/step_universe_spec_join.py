import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import pyarrow as pa
pa.set_cpu_count(1); pa.set_io_thread_count(1)
import pandas as pd, numpy as np, pyarrow.parquet as pq
os.chdir('/home/ec2-user/onemil')
S = '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad'
PQ = 'data/research/databento/equs_daily_2025_2026.parquet'


def load(syms, vol_pos=False):
    t = pq.read_table(PQ, columns=['symbol', 'bar_date', 'open', 'close', 'volume'], filters=[('symbol', 'in', list(syms))], use_threads=False)
    d = t.to_pandas(); d['bar_date'] = d.bar_date.astype(str).str[:10]
    if vol_pos: d = d[d.volume > 0]
    d = d.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = d.groupby('symbol', sort=False)
    d['prev_close'] = g.close.shift(1).values; d['nprev'] = g.cumcount().values
    sv = g.volume.shift(1)
    for mp in (20, 5, 1):
        d[f'adv20_{mp}'] = sv.groupby(d.symbol, sort=False).rolling(20, min_periods=mp).mean().reset_index(level=0, drop=True).values
    return d


def lookup(d, keys, cols):
    idx = {k: i for i, k in enumerate(zip(d.symbol.values, d.bar_date.values))}
    out = {c: np.full(len(keys), np.nan) for c in cols}
    vals = {c: d[c].values for c in cols}
    for j, k in enumerate(keys):
        i = idx.get(k)
        if i is not None:
            for c in cols: out[c][j] = vals[c][i]
    return out


mode = sys.argv[1]
if mode == 'A':
    T = pd.read_csv('research/bf_zero/spec_trades.csv', dtype={'symbol': str}, keep_default_na=False)
    for k in ('entry', 'level', 'price', 'adv20', 'rr', 'entry_m'): T[k] = pd.to_numeric(T[k], errors='coerce')
    T = T[T.level >= 20].reset_index(drop=True); syms = sorted(T.symbol.unique()); print('spec level>=20 signals', len(T), 'symbols', len(syms), flush=True)
    d = load(syms)
    L = lookup(d, list(zip(T.symbol.values, T.day.values)), ['prev_close', 'nprev', 'adv20_20', 'adv20_5', 'open'])
    for c in L: T[c] = L[c]
    print('unmatched in parquet', int(np.isnan(T.prev_close).sum()))
    for th in (17, 10, 5):
        k = T.prev_close < th
        print(f'prev_close<{th}: {int(k.sum())} ({k.mean()*100:.2f}%) meanR {T.rr[k].mean():+.3f} | rest n {int((~k).sum())} meanR {T.rr[~k].mean():+.3f}')
    for nm, lo in (('2026', '2026-01-01'), ('TEST', '2026-06-01')):
        x = T[T.day >= lo]; print(f'{nm}: n={len(x)} prev<17 {int((x.prev_close<17).sum())} ({(x.prev_close<17).mean()*100:.2f}%) prev<10 {int((x.prev_close<10).sum())} nprev<10 {int((x.nprev<10).sum())}')
    print('nprev<10:', int((T.nprev < 10).sum()), ' nprev<20:', int((T.nprev < 20).sum()), ' nprev<5:', int((T.nprev < 5).sum()))
    for c_ in ('adv20_20', 'adv20_5'):
        print(c_, 'matches spec adv20 (rtol 1e-6):', int(np.isclose(T.adv20, T[c_], rtol=1e-6).sum()), '/', len(T))
    T[['day', 'symbol', 'entry_m', 'level', 'entry', 'prev_close', 'open', 'adv20', 'adv20_20', 'adv20_5', 'nprev', 'rr']].to_csv(f'{S}/spec20_join.csv', index=False)
    B = pd.read_csv('research/bf_zero/spec_book.csv', dtype={'symbol': str}, keep_default_na=False); B['level'] = pd.to_numeric(B.level); B['rr'] = pd.to_numeric(B.rr)
    B = B[B.level >= 20].reset_index(drop=True)
    L = lookup(d, list(zip(B.symbol.values, B.day.values)), ['prev_close', 'nprev'])
    for c in L: B[c] = L[c]
    k = B.prev_close < 17
    print('spec_book (8/4) level>=20 n', len(B), 'prev<17', int(k.sum()), f'({k.mean()*100:.2f}%) meanR {B.rr[k].mean():+.3f}', 'prev<10', int((B.prev_close < 10).sum()), 'nprev<10', int((B.nprev < 10).sum()))
    # also the 12/day book variant is not stored; population numbers above are the superset
elif mode == 'B':
    u = pd.read_csv('research/bf_zero/universe.csv', dtype={'symbol': str}, keep_default_na=False)
    for c_ in ('open', 'adv20', 'prev_vol'): u[c_] = pd.to_numeric(u[c_], errors='coerce')
    rs = np.random.RandomState(0); syms = sorted(rs.choice(u.symbol.unique(), 600, replace=False))
    um = u[u.symbol.isin(syms)].reset_index(drop=True)
    for vp in (False, True):
        d = load(syms, vol_pos=vp)
        L = lookup(d, list(zip(um.symbol.values, um.bar_date.values)), ['prev_close', 'nprev', 'adv20_20', 'adv20_5', 'adv20_1'])
        print(f'--- parquet volume>0 filter={vp}: sample rows', len(um), 'unmatched', int(np.isnan(L['prev_close']).sum()))
        for c_ in ('adv20_20', 'adv20_5', 'adv20_1'):
            print(' ', c_, 'matches universe adv20:', int(np.isclose(um.adv20.values, L[c_], rtol=1e-6).sum()), '/', len(um))
        n = L['nprev']; print('  nprev: <5', int((n < 5).sum()), '<10', int((n < 10).sum()), '<20', int((n < 20).sum()), 'min', np.nanmin(n))
        bad = ~np.isclose(um.adv20.values, L['adv20_5'], rtol=1e-6) & ~np.isnan(L['adv20_5'])
        if bad.sum():
            i = np.flatnonzero(bad)[:3]; print('  mismatch examples', [(um.symbol[j], um.bar_date[j], um.adv20[j], L['adv20_5'][j], L['nprev'][j]) for j in i])
