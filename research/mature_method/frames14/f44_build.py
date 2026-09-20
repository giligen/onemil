#!/usr/bin/env python3
"""F44 stage 1 — the eight 15:55 conditioners, attached to frames13's A13 overnight panel.

Every field is computable from the PIT daily panel strictly at or before the 16:00 cross of day t
(PREREG §3.2).  Built per year into `feat_<year>.parquet`, restricted to the A13 keys so the node
never holds the 17.4M-row panel and a feature frame at the same time.

  python3 f44_build.py <year>
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames13')

D13 = f'{ROOT}/research/mature_method/frames13'
D14 = f'{ROOT}/research/mature_method/frames14'
PX = f'{ROOT}/research/multiday/data/prices_by_year'
UNI = f'{ROOT}/research/multiday/data/universe.parquet'
Y0, Y1 = 2016, 2026

_SYMS = None


def syms():
    global _SYMS
    if _SYMS is None:
        u = pd.read_parquet(UNI, columns=['symbol'])
        _SYMS = pd.Index(sorted(set(u.symbol.astype(str))))
    return _SYMS


def _panel(y, tail=None, head=None):
    """(sid, d, o, h, l, c, v) for year y from the ADJUSTED panel + the RAW volume, sorted."""
    a = pd.read_parquet(f'{PX}/all/year={y}.parquet',
                        columns=['symbol', 'date', 'open', 'high', 'low', 'close'])
    r = pd.read_parquet(f'{PX}/raw/year={y}.parquet', columns=['symbol', 'date', 'volume'])
    assert len(a) == len(r), f'{y}: adjusted and raw panels differ in length'
    sid = syms().get_indexer(pd.Index(a.symbol.astype(str))).astype(np.int32)
    day = pd.to_datetime(a.date).values.astype('datetime64[D]').astype(np.int32)
    keep = sid >= 0
    if tail or head:
        u = np.unique(day)
        keep = keep & ((day >= u[-tail]) if tail else (day <= u[head - 1]))
    out = dict(sid=sid[keep], d=day[keep])
    for k in ('open', 'high', 'low', 'close'):
        out[k[0]] = a[k].values[keep].astype(np.float64)
    out['v'] = r.volume.values[keep].astype(np.float64)
    del a, r
    order = np.lexsort((out['d'], out['sid']))
    return {k: v[order] for k, v in out.items()}


def rolling_prev_mean(x, sid, n=20):
    """Mean of x over the n rows STRICTLY BEFORE each row, within each symbol; NaN before that."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    pos = np.arange(len(x))
    start = np.concatenate([[0], np.flatnonzero(sid[1:] != sid[:-1]) + 1])
    first = np.repeat(start, np.diff(np.concatenate([start, [len(sid)]])))
    k = pos - first
    ok = k >= n
    out = np.full(len(x), np.nan)
    out[ok] = (cs[pos[ok]] - cs[pos[ok] - n]) / float(n)
    return out


def rolling_max_incl(x, sid, n=20):
    """Max of x over the n rows ENDING at each row (inclusive), within each symbol."""
    out = np.full(len(x), np.nan)
    start = np.concatenate([[0], np.flatnonzero(sid[1:] != sid[:-1]) + 1])
    ends = np.concatenate([start[1:], [len(sid)]])
    for s, e in zip(start, ends):
        seg = x[s:e]
        if len(seg) < n:
            continue
        w = np.lib.stride_tricks.sliding_window_view(seg, n).max(axis=1)
        out[s + n - 1:e] = w
    return out


def spy_day():
    """SPY's own close-to-close return by date, from the adjusted panel."""
    fr = []
    for y in range(Y0, Y1 + 1):
        d = pd.read_parquet(f'{PX}/all/year={y}.parquet', columns=['symbol', 'date', 'close'])
        fr.append(d[d.symbol.astype(str) == 'SPY'])
    s = pd.concat(fr).sort_values('date')
    s['d'] = pd.to_datetime(s.date).values.astype('datetime64[D]').astype(np.int32)
    s['spy_ret'] = s.close.astype(float).pct_change() * 100
    return s.set_index('d').spy_ret


def a13_keys():
    """The A13 key set (common, close >= $20, ADV$ >= $10M, CA rail, TEST cut) from frames13."""
    import f40
    X = f40.load()
    kinds = pd.Index(['common', 'fund', 'wrapper'])
    m = ((X['kind_i'] == int(kinds.get_loc('common'))) & (X['close'] >= 20.0) &
         (X['adv20'] >= 1e7))
    k = (X['sid'][m].astype(np.int64) * 100000 + X['d'][m].astype(np.int64))
    print(f'  A13 keys {len(k):,} (frames13 reports 3,310,743)', flush=True)
    assert len(k) == 3310743, f'A13 does not reproduce ({len(k):,})'
    return np.sort(k)


def build(y, keys, spy):
    spec = [(z, dict(tail=25) if z == y - 1 else (dict(head=1) if z == y + 1 else {}))
            for z in (y - 1, y, y + 1) if Y0 <= z <= Y1]
    P = [_panel(z, **kw) for z, kw in spec]
    X = {k: np.concatenate([p[k] for p in P]) for k in P[0]}
    del P
    order = np.lexsort((X['d'], X['sid']))
    X = {k: v[order] for k, v in X.items()}
    sid, d, o, h, l, c, v = (X[k] for k in ('sid', 'd', 'o', 'h', 'l', 'c', 'v'))
    same = np.empty(len(sid), bool); same[0] = False; same[1:] = sid[1:] == sid[:-1]
    prevc = np.where(same, np.roll(c, 1), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        day_ret = (c / prevc - 1.0) * 100
        day_range = (h - l) / prevc * 100
        close_pos = (c - l) / (h - l)
        gap = (o / prevc - 1.0) * 100
        v20 = rolling_prev_mean(v, sid, 20)
        rv = v / v20
        adv20 = rolling_prev_mean(c * v, sid, 20)
        dollar_frac = (c * v) / adv20
        hi20 = rolling_max_incl(h, sid, 20)
        hi20_prox = c / hi20
    key = sid.astype(np.int64) * 100000 + d.astype(np.int64)
    pos = np.searchsorted(keys, key)
    hit = (pos < len(keys)) & (keys[np.clip(pos, 0, len(keys) - 1)] == key)
    f = pd.DataFrame(dict(
        key=key[hit], d=d[hit],
        V1_day_ret=day_ret[hit], V2_day_range=day_range[hit], V3_close_pos=close_pos[hit],
        V4_rv=rv[hit], V5_dollar_frac=dollar_frac[hit], V7_gap=gap[hit],
        V8_hi20_prox=hi20_prox[hit])).astype({'key': np.int64, 'd': np.int32})
    f['V6_spy_day'] = spy.reindex(f.d.values).values
    f.to_parquet(f'{D14}/feat_{y}.parquet', index=False)
    print(f'  {y}: {len(f):,} A13 rows with features '
          f'(availability: ' + ', '.join(
              f'{k.split("_")[0]} {f[k].notna().mean():.0%}' for k in f.columns
              if k.startswith('V')) + ')', flush=True)


if __name__ == '__main__':
    y = int(sys.argv[1])
    build(y, a13_keys(), spy_day())
