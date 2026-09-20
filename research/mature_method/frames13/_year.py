"""F40 — the per-year overnight builder, numpy-only in the hot path.

Split out of `f40.py` so one year runs in one short-lived process under `ulimit -v 3000000`:
`python3 _year.py <year>` writes `on_<year>.parquet` and exits.  The two price panels are
row-identical by construction (`research/multiday/DATA.md` §2) and that is ASSERTED here, so no
merge is needed — the expensive operation on this node.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from research.scripts.pit_listings import is_test_ticker      # noqa: E402

D13 = f'{ROOT}/research/mature_method/frames13'
PX = f'{ROOT}/research/multiday/data/prices_by_year'
UNI = f'{ROOT}/research/multiday/data/universe.parquet'
PB_EDGES = [0, 5, 20, 50, 200, np.inf]
PB_LAB = ['<5', '5-20', '20-50', '50-200', '>=200']
AB_EDGES = [0, 1e6, 1e7, 1e8, np.inf]
AB_LAB = ['<1M', '1-10M', '10-100M', '>=100M']
Y0, Y1 = 2016, 2026

_SYMS = None


def syms():
    global _SYMS
    if _SYMS is None:
        u = pd.read_parquet(UNI, columns=['symbol'])
        _SYMS = pd.Index(sorted(set(u.symbol.astype(str))))
    return _SYMS


def _panel(y, adj, tail=None, head=None):
    """(sid, d, open, close, volume) for year `y` from one adjustment panel, sorted by (sid, d).

    `tail` keeps only the last N sessions of the year, `head` only the first N — the neighbouring
    years are needed ONLY for adv20's 20-session lookback and for the last night's next open, and
    carrying them whole is what blows the node's 3 GB virtual cap.
    """
    cols = ['symbol', 'date', 'open', 'close'] + ([] if adj else ['volume'])
    d = pd.read_parquet(f'{PX}/{"all" if adj else "raw"}/year={y}.parquet', columns=cols)
    sid = syms().get_indexer(pd.Index(d.symbol.astype(str))).astype(np.int32)
    day = pd.to_datetime(d.date).values.astype('datetime64[D]').astype(np.int32)
    keep = sid >= 0
    if tail or head:
        u = np.unique(day)
        cut = u[-tail] if tail else None
        top = u[head - 1] if head else None
        keep = keep & ((day >= cut) if tail else (day <= top))
    out = dict(sid=sid[keep], d=day[keep],
               o=d.open.values[keep].astype(np.float64),
               c=d.close.values[keep].astype(np.float64))
    if not adj:
        out['v'] = d.volume.values[keep].astype(np.float64)
    del d
    order = np.lexsort((out['d'], out['sid']))
    return {k: v[order] for k, v in out.items()}


def build(y):
    spec = [(z, dict(tail=25) if z == y - 1 else (dict(head=1) if z == y + 1 else {}))
            for z in (y - 1, y, y + 1) if Y0 <= z <= Y1]
    R = [_panel(z, False, **kw) for z, kw in spec]
    A = [_panel(z, True, **kw) for z, kw in spec]
    sid = np.concatenate([r['sid'] for r in R])
    d = np.concatenate([r['d'] for r in R])
    c = np.concatenate([r['c'] for r in R])
    o = np.concatenate([r['o'] for r in R])
    v = np.concatenate([r['v'] for r in R])
    sid_a = np.concatenate([a['sid'] for a in A])
    d_a = np.concatenate([a['d'] for a in A])
    ca_ = np.concatenate([a['c'] for a in A])
    oa = np.concatenate([a['o'] for a in A])
    del R, A
    assert len(sid) == len(sid_a) and np.array_equal(sid, sid_a) and np.array_equal(d, d_a), \
        'the raw and adjusted panels are NOT row-identical — DATA.md §2 violated'
    del sid_a, d_a
    order = np.lexsort((d, sid))
    sid, d, c, o, v, ca_, oa = (a[order] for a in (sid, d, c, o, v, ca_, oa))

    same = np.empty(len(sid), bool)
    same[:-1] = sid[:-1] == sid[1:]
    same[-1] = False
    nxt_o = np.where(same, np.roll(o, -1), np.nan)
    nxt_oa = np.where(same, np.roll(oa, -1), np.nan)
    nxt_d = np.where(same, np.roll(d, -1), -1)

    # adv20 = mean raw dollar volume over the 20 sessions STRICTLY BEFORE d
    dv = c * v
    cs = np.concatenate([[0.0], np.cumsum(dv)])
    pos = np.arange(len(sid))
    start = np.concatenate([[0], np.flatnonzero(sid[1:] != sid[:-1]) + 1])
    first = np.repeat(start, np.diff(np.concatenate([start, [len(sid)]])))
    k = pos - first                                   # index within the symbol
    ok = k >= 20
    adv = np.full(len(sid), np.nan)
    adv[ok] = (cs[pos[ok]] - cs[pos[ok] - 20]) / 20.0

    lo = np.datetime64(f'{y}-01-01').astype('datetime64[D]').astype(np.int32)
    hi = np.datetime64(f'{y + 1}-01-01').astype('datetime64[D]').astype(np.int32)
    m = ((d >= lo) & (d < hi) & np.isfinite(nxt_o) & np.isfinite(adv) &
         (c >= 1.0) & (ca_ > 0) & (nxt_oa > 0))
    on = ((nxt_oa[m] / ca_[m] - 1.0) * 100.0).astype(np.float32)
    onr = ((nxt_o[m] / c[m] - 1.0) * 100.0).astype(np.float32)
    dd = d[m]
    sid_m = sid[m].astype(np.int32)
    cl = c[m].astype(np.float32)
    ad = adv[m].astype(np.float64)
    ni = np.maximum(nxt_d[m] - dd, 1).astype(np.int16)
    del sid, d, c, o, v, ca_, oa, nxt_o, nxt_oa, nxt_d, adv, dv, cs, pos, start, first, k, m, same
    # `sid` (not the string) is written: a 2.3M-row object column of python strings is what
    # exceeds the node's virtual cap.  `symbols.parquet` is the map, written once.
    x = pd.DataFrame(dict(sid=sid_m, d=dd.astype(np.int32), close=cl, adv20=ad,
                          on_pct=on, ca=np.abs(on - onr) > 1.0, nights=ni))
    x.to_parquet(f'{D13}/on_{y}.parquet', index=False)
    print(f'  {y}: {len(x):,} name-nights  mean on_pct {x.on_pct.mean():+.4f} % '
          f'| corporate-action rows {int(x.ca.sum()):,}', flush=True)


if __name__ == '__main__':
    if not os.path.exists(f'{D13}/symbols.parquet'):
        u = pd.read_parquet(UNI, columns=['symbol', 'kind'])
        u['symbol'] = u.symbol.astype(str)
        u = u[~u.symbol.map(is_test_ticker)]
        sm = pd.DataFrame(dict(sid=np.arange(len(syms()), dtype=np.int32), symbol=syms()))
        sm = sm.merge(u, on='symbol', how='left')
        sm.to_parquet(f'{D13}/symbols.parquet', index=False)
    build(int(sys.argv[1]))
