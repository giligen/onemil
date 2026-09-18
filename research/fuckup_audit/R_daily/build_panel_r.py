#!/usr/bin/env python3
"""Stage R_daily step 2 — ONE continuous daily panel 2018-05 → 2026-09 in the `K/build_k.py` layout.

Sources (none modified):
    XNAS.ITCH  2018-05-01..2023-12-29   N_databento/N3/xnas_daily.parquet
    XNAS.ITCH  2024-01-02..2024-06-28   R_daily/xnas_daily_2024H1.parquet     (bought today)
    EQUS.SUM.  2024-07-01..2024-12-31   data/research/databento/equs_daily_2024H2.parquet
    EQUS.SUM.  2025-01-02..2026-09-04   data/research/databento/equs_daily_2025_2026.parquet

ITCH volume is the Nasdaq venue subset of consolidated volume, so it is multiplied by
1/VENUE_SHARE (seam.md §2, calibrated on the one overlapping month) — one $10M/day rule then means
the same thing on both sides of the seam.

Symbols kept: the point-in-time Nasdaq-listed common-stock union (pit_xnas_common.csv) PLUS any
symbol that ever reaches $10M of scaled daily dollar volume (the survivorship control needs the
delisted liquid names).  A symbol that never reaches $10M on a single day can never have a 20-day
MEDIAN of $10M, so this pre-filter cannot remove a universe member — it is a superset.

Derived fields use `research/lit_review_2026/build_daily_panel.py` definitions, verbatim:
    prev_close = close.shift(1) · adv20 = volume.shift(1).rolling(20, min_periods=10).mean()
    ret_on = open/prev_close - 1 · vol_ratio = volume/adv20 · ret5 = close/close.shift(5) - 1
    high52 = high.shift(1).rolling(250, min_periods=60).max()

Output: R_daily/daily_panel_2018_2026.parquet
"""
from __future__ import annotations

import gc
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

R = 'research/fuckup_audit/R_daily'
VENUE_SHARE = 0.3694                     # seam.md §2 — ITCH volume / consolidated volume, median
SRC = [('research/fuckup_audit/N_databento/N3/xnas_daily.parquet', 'itch'),
       (f'{R}/xnas_daily_2024H1.parquet', 'itch'),
       ('data/research/databento/equs_daily_2024H2.parquet', 'equs'),
       ('data/research/databento/equs_daily_2025_2026.parquet', 'equs')]
COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
OUT = f'{R}/daily_panel_2018_2026.parquet'
MIN_EVER_DVOL = 1e7


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def block_edges(codes):
    e = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate(([0], e)), np.concatenate((e, [len(codes)]))


def shift_k(v, starts, ends, k):
    out = np.full(len(v), np.nan, dtype='float64')
    for a, b in zip(starts, ends):
        if b - a > k:
            out[a + k:b] = v[a:b - k]
    return out


def trailing_mean(v, starts, ends, win, minp):
    out = np.full(len(v), np.nan, dtype='float64')
    for a, b in zip(starts, ends):
        x = v[a:b].astype('float64')
        n = len(x)
        if n <= minp:
            continue
        cs = np.concatenate(([0.0], np.nancumsum(x)))
        cnt = np.concatenate(([0], np.cumsum(np.isfinite(x))))
        i = np.arange(minp, n)
        lo = np.maximum(i - win, 0)
        s = cs[i] - cs[lo]
        k = cnt[i] - cnt[lo]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[a + minp:b] = np.where(k >= minp, s / np.where(k > 0, k, 1), np.nan)
    return out


def trailing_max(v, starts, ends, win, minp):
    out = np.full(len(v), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        x = v[a:b].astype('float64')
        n = len(x)
        if n <= minp:
            continue
        for i in range(minp, min(win, n)):
            seg = x[max(0, i - win):i]
            out[a + i] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        if n > win:
            out[a + win:a + n] = np.nanmax(sw(x, win)[:n - win], axis=1)
    return out


def keep_symbols():
    """Every symbol with >= 10 sessions of $10M+ scaled dollar volume — a strict SUPERSET of the
    universe: a 20-day MEDIAN of $10M needs at least 10 days of $10M+ inside that window, so a
    symbol with fewer than 10 such days in its whole life can never be a universe member.  The PIT
    Nasdaq-listed common-stock test is applied later, at universe time, on this set."""
    cnt = {}
    for path, kind in SRC:
        f = pq.ParquetFile(path)
        scale = 1.0 / VENUE_SHARE if kind == 'itch' else 1.0
        for g in range(f.metadata.num_row_groups):
            d = f.read_row_group(g, columns=['symbol', 'close', 'volume']).to_pandas()
            dv = (d.close.to_numpy('float64') * d.volume.to_numpy('float64') * scale)
            ok = np.isfinite(dv) & (dv >= MIN_EVER_DVOL)
            s = pd.Series(ok.astype('int32')).groupby(d.symbol.to_numpy()).sum()
            for sym, v in s.items():
                if v:
                    cnt[sym] = cnt.get(sym, 0) + int(v)
            del d, dv, ok, s
        log(f'  scanned {path} ({f.metadata.num_row_groups} row groups)')
        del f
        gc.collect()
    liq = {s for s, v in cnt.items() if v >= 10}
    ps = pd.read_csv(f'{R}/pit_xnas_common.csv', keep_default_na=False, na_values=[''])
    pit = set(ps.symbol)
    out = liq & pit
    # The panel carries ONLY point-in-time Nasdaq-listed common stocks.  The non-PIT liquid names
    # (5,793 of them) are overwhelmingly NYSE/ARCA listings, whose ITCH open and close are
    # off-primary prints — including them would not be a survivorship control, it would be a
    # different and broken universe.  The survivorship channel is quantified in REPORT.md instead.
    log(f'liquid symbols (>=10 sessions >= $10M): {len(liq):,}; PIT Nasdaq commons among them '
        f'{len(out):,}; PIT commons never liquid enough {len(pit - liq):,}; liquid non-PIT (not '
        f'carried) {len(liq - pit):,}')
    return out


def main() -> int:
    keep = keep_symbols()
    sym_map, day_map = {}, {}
    scode_p, dcode_p = [], []
    arr_p = {k: [] for k in ('open', 'high', 'low', 'close', 'volume')}
    for path, kind in SRC:
        f = pq.ParquetFile(path)
        scale = 1.0 / VENUE_SHARE if kind == 'itch' else 1.0
        n = 0
        for g in range(f.metadata.num_row_groups):
            d = f.read_row_group(g, columns=COLS).to_pandas()
            m = d.symbol.isin(keep).to_numpy()
            d = d[m]
            if not len(d):
                continue
            scode_p.append(np.fromiter((sym_map.setdefault(s, len(sym_map)) for s in d.symbol),
                                       dtype='int32', count=len(d)))
            dcode_p.append(np.fromiter((day_map.setdefault(str(x)[:10], len(day_map))
                                        for x in d.bar_date), dtype='int32', count=len(d)))
            for c in ('open', 'high', 'low', 'close'):
                arr_p[c].append(d[c].to_numpy('float64').astype('float32'))
            arr_p['volume'].append((d.volume.to_numpy('float64') * scale).astype('float32'))
            n += len(d)
            del d
        log(f'{path}: kept {n:,} rows')
        del f
        gc.collect()

    scode = np.concatenate(scode_p)
    dcode = np.concatenate(dcode_p)
    del scode_p, dcode_p
    arr = {}
    for c in list(arr_p):
        arr[c] = np.concatenate(arr_p[c])
        del arr_p[c]
        gc.collect()
    del arr_p
    syms = np.array(list(sym_map), dtype=object)
    days_raw = np.array(list(day_map), dtype=object)
    uniq_days = np.array(sorted(day_map), dtype=object)
    remap = np.argsort(np.argsort(days_raw)).astype('int32')
    dcode = remap[dcode]
    gc.collect()

    order = np.lexsort((dcode, scode))
    scode, dcode = scode[order], dcode[order]
    for c in arr:
        arr[c] = arr[c][order]
    del order
    gc.collect()
    dup = np.flatnonzero((np.diff(scode) == 0) & (np.diff(dcode) == 0))
    log(f'duplicate (symbol, bar_date) rows dropped: {len(dup):,}')
    if len(dup):
        m = np.ones(len(scode), dtype=bool)
        m[dup] = False
        scode, dcode = scode[m], dcode[m]
        for c in arr:
            arr[c] = arr[c][m]
        del m
    del dup
    gc.collect()
    log(f'panel {len(scode):,} rows  {len(syms):,} symbols  {len(uniq_days)} sessions '
        f'{uniq_days[0]}..{uniq_days[-1]}')

    starts, ends = block_edges(scode)
    log(f'{len(starts):,} symbol blocks — deriving fields')
    prev_close = shift_k(arr['close'], starts, ends, 1).astype('float32')
    adv20 = trailing_mean(arr['volume'], starts, ends, 20, 10).astype('float32')
    log('adv20 done')
    c5 = shift_k(arr['close'], starts, ends, 5).astype('float32')
    with np.errstate(invalid='ignore', divide='ignore'):
        ret5 = (arr['close'] / c5 - 1.0).astype('float32')
        ret_on = (arr['open'] / prev_close - 1.0).astype('float32')
        vol_ratio = (arr['volume'] / adv20).astype('float32')
    del c5
    gc.collect()
    high52 = trailing_max(arr['high'], starts, ends, 250, 60).astype('float32')
    log('high52 done')
    gc.collect()

    cols = {'symbol': pa.DictionaryArray.from_arrays(pa.array(scode, type=pa.int32()),
                                                     pa.array([str(x) for x in syms])),
            'bar_date': pa.DictionaryArray.from_arrays(pa.array(dcode, type=pa.int32()),
                                                       pa.array([str(x) for x in uniq_days]))}
    for k in ('open', 'high', 'low', 'close', 'volume'):
        cols[k] = pa.array(arr[k])
    for k, v in (('prev_close', prev_close), ('adv20', adv20), ('ret5', ret5),
                 ('high52', high52), ('ret_on', ret_on), ('vol_ratio', vol_ratio)):
        cols[k] = pa.array(v)
    hi_finite = np.isfinite(high52)
    del arr, prev_close, adv20, ret5, ret_on, vol_ratio, high52
    gc.collect()
    tbl = pa.table(cols)
    del cols
    gc.collect()
    pq.write_table(tbl, OUT)
    del tbl
    gc.collect()
    log(f'wrote {OUT}: {len(scode):,} rows ({os.path.getsize(OUT) / 1e6:.1f} MB)')
    dstr = np.asarray(uniq_days)[dcode]
    for cut in ('2019-01-02', '2020-01-02', '2024-01-02', '2024-07-01', '2025-07-01'):
        sel = dstr[dstr >= cut]
        if not len(sel):
            continue
        first = sel.min()
        m = dstr == first
        log(f'  {first}: {int(m.sum()):,} rows, high52 finite {hi_finite[m].mean() * 100:.1f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
