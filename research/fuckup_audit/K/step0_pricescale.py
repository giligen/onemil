#!/usr/bin/env python3
"""Stage K step 0 — the PLAN.md §1 price-scale check, plus the asset-class map coverage audit.

WHY (CLAUDE.md check 3): every K family compares a Databento daily-panel price with another
Databento daily-panel price, but the universe gates (dvol20, adv20) and the capacity numbers are
quoted against the prices the live account would see (Alpaca, RAW).  If the Databento daily file is
split/dividend ADJUSTED and the Alpaca daily bars are RAW, a gap/return computed on the panel is a
different number from the one the engine would have seen, and adjusted prices silently fabricate
setups (an 8% "gap" that is a 2-for-1 split).  So: 200 random (symbol, day) keys present in BOTH
`data/research/databento/equs_daily_2025_2026.parquet` and `data/cache.db::daily_bars` are compared
on OPEN, HIGH, LOW, CLOSE and VOLUME, and the disagreement distribution is reported.

Also audits the asset-class map (`data/research/orb_asset_class_map_20260711.csv`, a 2026-07-11
dump) against the panel's symbols: the PREREG's "common stock only" rule is only usable if the map
covers the universe, and a 2026-07 dump cannot name a symbol that delisted in 2025 -> requiring map
membership is itself a SURVIVORSHIP filter.  Coverage is reported per split before the rule is used.

READ-ONLY everywhere except research/fuckup_audit/K/.
"""
import os
import random
import re
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
K = f'{ROOT}/research/fuckup_audit/K'
PANEL_RAW = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
PANEL = f'{ROOT}/research/lit_review_2026/daily_panel.parquet'
CACHE = f'{ROOT}/data/cache.db'
CLASSMAP = f'{ROOT}/data/research/orb_asset_class_map_20260711.csv'
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
SPLITS = [('TRAIN', '2025-01-02', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-09-11')]
N_KEYS = 200
SEED = 17


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def split_of(d):
    for n, a, b in SPLITS:
        if a <= d <= b:
            return n
    return ''


def trailing_median(vals, starts, ends, win, minp):
    """Causal trailing median: out[i] uses vals[i-win .. i-1] inside its own symbol block.

    Same convention as research/fuckup_audit/E/universes.py::_trailing(how='median') and therefore
    the same dvol20 definition every earlier stage used, only shifted by one row (no same-day data).
    """
    out = np.full(len(vals), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        v = vals[a:b].astype('float64')
        n = len(v)
        if n < minp + 1:
            continue
        for i in range(minp, min(win, n)):
            out[a + i] = np.median(v[max(0, i - win):i])
        if n > win:
            w = sw(v, win)[:n - win]
            out[a + win:a + n] = np.median(w, axis=1)
    return out


def main():
    L = ['# Stage K step 0 — price scale + asset-class map coverage', '',
         f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by '
         '`research/fuckup_audit/K/step0_pricescale.py`.', '']

    # ------------------------------------------------------------------ panel
    log('reading daily panel (symbol, bar_date, ohlcv)')
    d = pd.read_parquet(PANEL, columns=['symbol', 'bar_date', 'open', 'high', 'low',
                                        'close', 'volume'])
    # symbol stays CATEGORICAL: 5M python strings do not fit in `ulimit -v 1500000`
    cats = np.asarray(d.symbol.cat.categories, dtype=object)
    scode = d.symbol.cat.codes.to_numpy()
    d = d.drop(columns=['symbol'])
    bdate = d.bar_date.to_numpy()
    log(f'panel {len(d):,} rows, {len(cats):,} symbols, {len(np.unique(bdate))} days')

    # ------------------------------------------------------------------ price scale
    log(f'price-scale: sampling {N_KEYS} keys present in both stores')
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=180)
    random.seed(SEED)
    idx = list(range(len(d)))
    random.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= N_KEYS:
            break
        s = cats[scode[i]]
        b = bdate[i]
        if TEST_TICKER.match(s):
            continue
        r = con.execute('select open, high, low, close, volume from daily_bars '
                        'where symbol=? and bar_date=?', (s, b)).fetchone()
        if not r:
            continue
        rows.append(dict(symbol=s, bar_date=b,
                         p_open=float(d.open.iat[i]), p_high=float(d.high.iat[i]),
                         p_low=float(d.low.iat[i]), p_close=float(d.close.iat[i]),
                         p_vol=float(d.volume.iat[i]),
                         a_open=float(r[0]), a_high=float(r[1]), a_low=float(r[2]),
                         a_close=float(r[3]), a_vol=float(r[4])))
    con.close()
    ps = pd.DataFrame(rows)
    for k in ('open', 'high', 'low', 'close'):
        ps[f'd_{k}'] = (ps[f'p_{k}'] / ps[f'a_{k}'] - 1.0) * 100.0
    ps['d_vol'] = (ps.p_vol / ps.a_vol - 1.0) * 100.0
    ps['split'] = ps.bar_date.map(split_of)
    ps.to_csv(f'{K}/pricescale.csv', index=False)
    log(f'price-scale rows {len(ps)}')

    L.append('## Price-scale check (PLAN.md §1 / CLAUDE.md check 3)')
    L.append('')
    L.append(f'{len(ps)} random (symbol, day) keys present in BOTH the Databento daily panel and '
             '`data/cache.db::daily_bars` (Alpaca, the prices the live account sees). '
             '`diff = panel / alpaca - 1`, in percent.')
    L.append('')
    L.append('| field | within 0.01% | within 0.1% | within 1% | median abs | p95 abs | max abs |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for k in ('open', 'high', 'low', 'close', 'vol'):
        g = ps[f'd_{k}'].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(g):
            continue
        L.append(f'| {k} | {(g.abs() <= 0.01).mean() * 100:.1f}% | {(g.abs() <= 0.1).mean() * 100:.1f}% '
                 f'| {(g.abs() <= 1.0).mean() * 100:.1f}% | {g.abs().median():.4f}% | '
                 f'{g.abs().quantile(0.95):.4f}% | {g.abs().max():.3f}% |')
    L.append('')
    bad = ps[ps.d_close.abs() > 1.0]
    L.append(f'Keys disagreeing by more than 1% on the CLOSE: {len(bad)} of {len(ps)} '
             f'({len(bad) / max(len(ps), 1) * 100:.1f}%).')
    if len(bad):
        L.append('')
        L.append('| symbol | day | panel close | alpaca close | diff % |')
        L.append('|---|---|---:|---:|---:|')
        for _, r in bad.head(20).iterrows():
            L.append(f'| {r.symbol} | {r.bar_date} | {r.p_close:.4f} | {r.a_close:.4f} | '
                     f'{r.d_close:+.2f}% |')
    L.append('')
    L.append('Rows: `K/pricescale.csv`.')
    L.append('')

    # ------------------------------------------------------------------ class map
    log('asset-class map coverage')
    cm = pd.read_csv(CLASSMAP, dtype=str, keep_default_na=False, na_values=[''])
    cls = dict(zip(cm.symbol, cm.asset_class))
    L.append('## Asset-class map coverage (PREREG "common stock only")')
    L.append('')
    L.append(f'`{os.path.relpath(CLASSMAP, ROOT)}`: {len(cm):,} symbols '
             f'({(cm.asset_class == "stock").sum():,} stock, '
             f'{(cm.asset_class == "wrapper").sum():,} wrapper, '
             f'{(~cm.asset_class.isin(["stock", "wrapper"])).sum():,} other).')
    L.append('')
    L.append('The map is a **2026-07-11 dump of live Alpaca assets**. A symbol that delisted in '
             '2025 cannot be in it, so *requiring* map membership is itself a survivorship filter. '
             'Coverage is measured on the liquid slice this stage trades (20-day median dollar '
             'volume >= $10M, close >= $5).')
    L.append('')
    dvol = d.close.to_numpy('float64') * d.volume.to_numpy('float64')
    closes = d.close.to_numpy('float64')
    del d
    # the panel is already sorted by (symbol, bar_date) -- build_daily_panel.py:14
    edges = np.flatnonzero(np.diff(scode)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(scode)]))
    med = trailing_median(dvol, starts, ends, 20, 10)
    keep = (med >= 1e7) & (closes >= 5.0)
    liq = pd.DataFrame({'symbol': pd.Categorical.from_codes(scode[keep], cats),
                        'bar_date': bdate[keep]})
    del dvol, closes, med, keep
    liq['split'] = pd.Series(liq.bar_date).map(split_of)
    cls_of_cat = np.array([cls.get(str(c), 'not_in_map') for c in cats], dtype=object)
    liq['cls'] = cls_of_cat[liq.symbol.cat.codes.to_numpy()]
    L.append('| split | symbol-days | distinct symbols | stock | wrapper | other-in-map | not in map |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for nm, _a, _b in SPLITS + [('ALL', '', '')]:
        x = liq if nm == 'ALL' else liq[liq.split == nm]
        n = len(x)
        if not n:
            continue
        cnt = x.cls.value_counts()
        other = n - int(cnt.get('stock', 0)) - int(cnt.get('wrapper', 0)) - int(cnt.get('not_in_map', 0))
        L.append(f'| {nm} | {n:,} | {x.symbol.nunique():,} | '
                 f'{cnt.get("stock", 0) / n * 100:.1f}% | {cnt.get("wrapper", 0) / n * 100:.1f}% | '
                 f'{other / n * 100:.1f}% | {cnt.get("not_in_map", 0) / n * 100:.1f}% |')
    L.append('')
    nim = liq[liq.cls == 'not_in_map']
    last_seen = {str(cats[scode[e - 1]]): bdate[e - 1] for e in ends}
    nim_syms = sorted(str(s) for s in nim.symbol.unique())
    deli = sum(1 for s in nim_syms if last_seen.get(s, '') < '2026-07-11')
    L.append(f'Symbols in the liquid slice that are NOT in the map: {len(nim_syms):,} '
             f'({len(nim):,} symbol-days). Of those, {deli:,} have no panel bar after 2026-07-11 '
             '(i.e. they are gone by the dump date) — this is the survivorship channel the rule '
             'opens. `K/classmap_missing.csv` lists them.')
    ndays = nim.symbol.astype(str).value_counts()
    pd.DataFrame(dict(symbol=nim_syms,
                      last_panel_day=[last_seen.get(s, '') for s in nim_syms],
                      n_liquid_days=[int(ndays.get(s, 0)) for s in nim_syms])
                 ).to_csv(f'{K}/classmap_missing.csv', index=False)
    L.append('')
    L.append('**Decision taken here and carried into the scoring (pre-registered before any P&L '
             'was computed):** the PREREG rule (`asset_class == stock`) is the PRIMARY universe; '
             'a secondary universe `stock or not-in-map` (drop only POSITIVELY identified '
             'wrappers) is carried alongside every cell so the survivorship cost of the primary '
             'rule is visible rather than assumed away. Neither is tuned on a result.')
    L.append('')

    with open(f'{K}/step0.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    log(f'wrote {K}/step0.md')


if __name__ == '__main__':
    main()
