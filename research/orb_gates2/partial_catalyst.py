#!/usr/bin/env python3
"""G6 / G8 — the catalyst PARTIAL (`min_cohort: 2 -> 1`), derived row-exactly.

The catalyst veto is applied POST-selection with NO refill (a vetoed pick's slot
stays empty), so the catalyst-ON book is EXACTLY the catalyst-OFF book minus the
vetoed rows, with identical sizing on every surviving row.  This module rebuilds
the veto decision with the SHIPPED shared helpers
(`trading.orb_catalyst_veto`, `trading.orb_asset_class.underlying_anchor`, the
shipped class map, the same tri-state raw-news source) and applies it to a
catalyst-OFF book at an arbitrary `min_cohort`.

PREREG §2a validation gate (must pass before G6/G8 are quoted):
    derive(G3, min_cohort=2)  ==  G0      row-for-row and to the cent
    derive(G5, min_cohort=2)  ==  G4      row-for-row and to the cent
"""
from __future__ import annotations

import csv as _csv
import glob as _glob
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv                      # noqa: E402
from trading.orb_asset_class import (DEFAULT_CLASS_MAP,        # noqa: E402
                                     load_class_map, underlying_anchor)
from trading.orb_catalyst_veto import (anchor_cohort_counts,   # noqa: E402
                                       catalyst_veto_applies)

D = f'{ROOT}/research/orb_gates2'
FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'
NEEDED = ['range_high', 'range_low', 'entry_price', 'pnl_pct', 'range_size_pct']

_CACHE: dict = {}


def _universe() -> pd.DataFrame:
    """The candidate frame the pipeline computes cohorts over (its `df`)."""
    if 'df' not in _CACHE:
        df = read_orb_csv(FEATURES)
        df = df.dropna(subset=['pnl', 'date', 'pnl_pct', 'range_size_pct',
                               'entry_price'])
        df['date'] = pd.to_datetime(df['date'])
        _CACHE['df'] = df
    return _CACHE['df']


def _anchors() -> dict:
    if 'anchors' not in _CACHE:
        names = {}
        with open(DEFAULT_CLASS_MAP, newline='') as fh:
            for row in _csv.DictReader(fh):
                names[row['symbol']] = row.get('name', '')
        cmap = load_class_map()
        df = _universe()
        _CACHE['anchors'] = {s: underlying_anchor(s, names.get(s), cmap)
                             for s in set(df['symbol'])}
    return _CACHE['anchors']


def _cohorts() -> dict:
    if 'cohorts' not in _CACHE:
        df = _universe()
        a = _anchors()
        day = df.assign(_a=df['symbol'].map(a))
        _CACHE['cohorts'] = {d: anchor_cohort_counts(g['_a'])
                             for d, g in day.groupby(
                                 day['date'].dt.strftime('%Y-%m-%d'))}
    return _CACHE['cohorts']


def _raw_news() -> dict:
    """Tri-state own-ticker premarket news; absent pair -> None -> fail-open."""
    if 'news' not in _CACHE:
        n = {}
        for p in sorted(_glob.glob(f'{ROOT}/data/research/orb_news_catalyst_*.csv')):
            for _, r in read_orb_csv(p).iterrows():
                n[(r['symbol'], r['day'])] = (r['n_articles'] or 0) > 0
        _CACHE['news'] = n
    return _CACHE['news']


def derive(book_off: str, min_cohort: int, out: str | None = None) -> pd.DataFrame:
    """Apply the catalyst veto at `min_cohort` to a catalyst-OFF book."""
    b = read_orb_csv(book_off)
    dt = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    anch = b['symbol'].map(_anchors())
    news = _raw_news()
    coh = _cohorts()
    veto = [catalyst_veto_applies(news.get((s, d)), a, coh.get(d, {}), min_cohort)
            for s, d, a in zip(b['symbol'], dt, anch)]
    kept = b[~pd.Series(veto, index=b.index)].copy()
    if out:
        kept.to_csv(out, index=False)
    return kept


def _cmp(derived: pd.DataFrame, ref_path: str, label: str) -> bool:
    ref = read_orb_csv(ref_path)
    d = derived.reset_index(drop=True)
    same = len(d) == len(ref) and \
        set(zip(d['symbol'], d['date'].astype(str))) == \
        set(zip(ref['symbol'], ref['date'].astype(str))) and \
        abs(d['_sized_pnl'].sum() - ref['_sized_pnl'].sum()) < 1e-6
    print(f'  {label}: derived {len(d)} / ref {len(ref)} picks, '
          f'${d["_sized_pnl"].sum():,.6f} vs ${ref["_sized_pnl"].sum():,.6f} '
          f'-> {"PASS" if same else "FAIL"}')
    return same


if __name__ == '__main__':
    ok = True
    for dump in ('meas', 'asis'):
        print(f'[{dump}] PREREG §2a validation gate (min_cohort=2 must reproduce '
              f'the shipped rule):')
        ok &= _cmp(derive(f'{D}/book_G3_{dump}.csv', 2), f'{D}/book_G0_{dump}.csv',
                   'derive(G3,mc=2) == G0')
        ok &= _cmp(derive(f'{D}/book_G5_{dump}.csv', 2), f'{D}/book_G4_{dump}.csv',
                   'derive(G5,mc=2) == G4')
    if not ok:
        raise SystemExit('VALIDATION GATE FAILED — G6/G8 are NOT MEASURED (PREREG §2a)')
    for dump in ('meas', 'asis'):
        g6 = derive(f'{D}/book_G3_{dump}.csv', 1, f'{D}/book_G6_{dump}.csv')
        g8 = derive(f'{D}/book_G5_{dump}.csv', 1, f'{D}/book_G8_{dump}.csv')
        print(f'[{dump}] G6 {len(g6)} picks  ${g6["_sized_pnl"].sum():,.0f}   '
              f'G8 {len(g8)} picks  ${g8["_sized_pnl"].sum():,.0f}')
    print('PARTIAL CATALYST DONE')
