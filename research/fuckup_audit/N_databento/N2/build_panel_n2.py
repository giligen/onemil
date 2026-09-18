#!/usr/bin/env python3
"""Stage N2 step 2 — the daily panel for K2 over 2024-07 .. 2026-09.

Concatenates the NEW `data/research/databento/equs_daily_2024H2.parquet` with the existing
`data/research/databento/equs_daily_2025_2026.parquet` (neither is modified) and derives the
columns `K/build_k.py::load_panel` reads, with the SAME definitions as
`research/lit_review_2026/build_daily_panel.py` (the panel Stage K ran on):

    prev_close = close.shift(1)
    adv20      = volume.shift(1).rolling(20, min_periods=10).mean()
    ret_on     = open / prev_close - 1
    vol_ratio  = volume / adv20
    ret5       = close / close.shift(5) - 1
    high52     = high.shift(1).rolling(250, min_periods=60).max()

Nothing else is changed: with 128 extra sessions in front of 2025-01-02, `high52` is a FULL
250-session lookback from 2025-07 onward — the reason for the pull.

Everything is held as dictionary codes + float32 arrays: 6.5M python strings do not fit in
`ulimit -v 3000000`.

Output: research/fuckup_audit/N_databento/N2/daily_panel_2024H2_2026.parquet
"""
from __future__ import annotations

import gc
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N2 = 'research/fuckup_audit/N_databento/N2'
SRC = ['data/research/databento/equs_daily_2024H2.parquet',
       'data/research/databento/equs_daily_2025_2026.parquet']
OUT = f'{N2}/daily_panel_2024H2_2026.parquet'
COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def block_edges(codes):
    e = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate(([0], e)), np.concatenate((e, [len(codes)]))


def shift_k(v, starts, ends, k):
    """v[i-k] inside the symbol block, NaN before."""
    out = np.full(len(v), np.nan, dtype='float64')
    for a, b in zip(starts, ends):
        if b - a > k:
            out[a + k:b] = v[a:b - k]
    return out


def trailing_mean(v, starts, ends, win, minp):
    """mean of v[i-win .. i-1] inside the block, NaN until `minp` prior bars exist."""
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
            m = np.where(k >= minp, s / np.where(k > 0, k, 1), np.nan)
        out[a + minp:b] = m
    return out


def trailing_max(v, starts, ends, win, minp):
    """max of v[i-win .. i-1] inside the block, NaN until `minp` prior bars exist."""
    out = np.full(len(v), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        x = v[a:b].astype('float64')
        n = len(x)
        if n <= minp:
            continue
        hi = min(win, n)
        for i in range(minp, hi):
            seg = x[max(0, i - win):i]
            out[a + i] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        if n > win:
            w = sw(x, win)[:n - win]
            out[a + win:a + n] = np.nanmax(w, axis=1)
    return out


def main() -> int:
    tabs = []
    for p in SRC:
        t = pq.read_table(p, columns=COLS)
        log(f'{p}: {t.num_rows:,} rows '
            f'{pc.min(t.column("bar_date")).as_py()}..{pc.max(t.column("bar_date")).as_py()}')
        tabs.append(t)
    t = pa.concat_tables(tabs)
    del tabs
    gc.collect()

    sym = t.column('symbol').cast(pa.string()).dictionary_encode().combine_chunks()
    scode = sym.indices.to_numpy(zero_copy_only=False).astype('int32')
    snull = sym.indices.is_null().to_numpy(zero_copy_only=False)
    scode = np.where(snull, -1, scode).astype('int32')
    syms = np.array(sym.dictionary.to_pylist(), dtype=object)
    del sym
    gc.collect()
    day = t.column('bar_date').cast(pa.string()).dictionary_encode().combine_chunks()
    dcode = day.indices.to_numpy(zero_copy_only=False).astype('int32')
    days_raw = np.array([str(x)[:10] for x in day.dictionary.to_pylist()], dtype=object)
    del day
    gc.collect()
    arr = {k: t.column(k).to_numpy(zero_copy_only=False).astype('float32')
           for k in ('open', 'high', 'low', 'close', 'volume')}
    del t
    gc.collect()

    # chronological day codes
    uniq_days = np.array(sorted(set(days_raw)), dtype=object)
    remap = {d: i for i, d in enumerate(uniq_days)}
    dcode = np.array([remap[d] for d in days_raw], dtype='int32')[dcode]
    blank = np.array([str(s).strip() == '' for s in syms])
    keep = (scode >= 0) & ~blank[np.maximum(scode, 0)]
    log(f'rows with a usable symbol: {int(keep.sum()):,} of {len(scode):,}')
    scode, dcode = scode[keep], dcode[keep]
    for k in arr:
        arr[k] = arr[k][keep]
    del keep
    gc.collect()

    order = np.lexsort((dcode, scode))
    scode, dcode = scode[order], dcode[order]
    for k in arr:
        arr[k] = arr[k][order]
    del order
    gc.collect()
    dup = np.flatnonzero((np.diff(scode) == 0) & (np.diff(dcode) == 0))
    log(f'duplicate (symbol, bar_date) rows: {len(dup):,}')
    if len(dup):
        m = np.ones(len(scode), dtype=bool)
        m[dup] = False
        scode, dcode = scode[m], dcode[m]
        for k in arr:
            arr[k] = arr[k][m]
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

    # written column by column: consolidating 11 float columns into one pandas block needs an
    # extra 271 MB and this node is capped at 3 GB of address space.
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
    for cut in ('2025-01-02', '2025-07-01', '2026-01-02', '2026-06-01'):
        first = dstr[dstr >= cut].min()
        m = dstr == first
        log(f'  high52 finite on {first}: {hi_finite[m].mean() * 100:.1f}% '
            f'of {int(m.sum()):,} rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
