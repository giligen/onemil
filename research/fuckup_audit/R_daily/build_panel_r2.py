#!/usr/bin/env python3
"""R_daily step 2b — the SPLIT-ADJUSTED panel (REPORT_v2).

`build_panel_r.py` wrote the raw vendor prices.  Both Databento files are unadjusted, so every
forward/reverse split is a fabricated overnight return: in a 1-10 day long-only book a 2:1 forward
split reads as a -50% loss and a 1:20 reverse split as a +1,900% gain, and — worse — a split inside
a FEATURE window (K4's 20-day mean overnight, K2's 250-day high, K5's 50-day high / 20-day SMA,
K3's 5-day return) fabricates the SELECTION itself.

This rebuilds the same panel with three data-integrity corrections, each reported and none of them
a change to the pre-registered trading rule:

  1. SPLIT ADJUSTMENT.  `corporate_actions.csv` (Alpaca corporate-actions endpoint, 2018-2026, the
     source named in the task's preference order (b) — the Databento `pit_definition` files carry
     instrument definitions only and no corporate-actions schema is entitled).  For a row dated d,
     factor(sym, d) = product of `ratio` over every event with ex_date > d; prices are divided by it
     and volume multiplied, so returns across the ex-date become real returns.  Applied identically
     on BOTH sides of the ITCH/EQUS seam.
  2. BAD-PRINT OPEN.  After adjustment, an `open` more than 50% away from the previous close whose
     own day's CLOSE is within 20% of that previous close is not an auction print — it is an
     extended-session outlier (the ITCH daily bar aggregates 04:00-20:00).  MPWR 2022-10-14 is the
     worked case: open $4.26 against a $334.95 prior close and a $310.05 close.  A market-on-open
     order could never have obtained it (CLAUDE.md fill-realism rule 1b), so the `open` of such a row
     is set to NaN: no entry can be simulated there, and `ret_on`/`gap` for that row are missing
     rather than wrong.  The rest of the row (high/low/close/volume) is kept, so no rolling window
     loses a real bar.  The rule is symmetric in direction and is applied to every row of the panel.
  3. A `close_raw` column is carried.  Back-adjustment is a LOOK-AHEAD for a PRICE gate: a
     reverse split multiplies every earlier price up, so a $0.80 penny stock that later did a 1:20
     reverse split would show as $16 in 2020 and sail through PREREG's `close >= $5`.  The gate must
     be applied to the price that actually traded, so the panel keeps the unadjusted close beside
     the adjusted one and `run_r2.py` gates on `close_raw`.  (The $10M dollar-volume gate needs no
     such care: close x volume is invariant under the adjustment.)  A `bad_open` column is carried
     too, so the audit can count the blanked opens.

Output: R_daily/daily_panel_2018_2026_adj.parquet (same layout as the raw panel).
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
VENUE_SHARE = 0.3694
SRC = [('research/fuckup_audit/N_databento/N3/xnas_daily.parquet', 'itch'),
       (f'{R}/xnas_daily_2024H1.parquet', 'itch'),
       ('data/research/databento/equs_daily_2024H2.parquet', 'equs'),
       ('data/research/databento/equs_daily_2025_2026.parquet', 'equs')]
COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
RAW_PANEL = f'{R}/daily_panel_2018_2026.parquet'
OUT = f'{R}/daily_panel_2018_2026_adj.parquet'
CA = f'{R}/corporate_actions.csv'
BADPRINT_GAP = 0.50          # |open/prev_close - 1| above this ...
BADPRINT_CLOSE = 0.20        # ... while the day's close came back inside this of prev_close

sys.path.insert(0, f'{R}')
from build_panel_r import block_edges, shift_k, trailing_mean, trailing_max  # noqa: E402


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def load_events():
    """{symbol: (ex_dates ascending, suffix product of ratios from that index on)}."""
    ca = pd.read_csv(CA, keep_default_na=False, na_values=[''])
    ca = ca[ca.ratio.notna() & (ca.ratio > 0) & (ca.ratio != 1.0)]
    ca = ca.sort_values(['symbol', 'ex_date'])
    out = {}
    for sym, g in ca.groupby('symbol'):
        d = g.ex_date.astype(str).to_numpy()
        r = g.ratio.to_numpy('float64')
        suf = np.concatenate((np.cumprod(r[::-1])[::-1], [1.0]))   # suf[i] = prod(r[i:])
        out[str(sym)] = (d, suf)
    log(f'corporate actions: {len(ca):,} split-like events over {len(out):,} symbols')
    return out


def factors(syms, dates, ev):
    """Back-adjustment factor per row: prices / f, volume * f."""
    f = np.ones(len(syms), dtype='float64')
    uniq, inv = np.unique(syms, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    inv_s = inv[order]
    edge = np.flatnonzero(np.diff(inv_s)) + 1
    starts = np.concatenate(([0], edge))
    ends = np.concatenate((edge, [len(inv_s)]))
    n_touched = 0
    for a, b in zip(starts, ends):
        e = ev.get(str(uniq[inv_s[a]]))
        if e is None:
            continue
        d, suf = e
        rr = order[a:b]
        v = suf[np.searchsorted(d, dates[rr], side='right')]
        f[rr] = v
        n_touched += int((v != 1.0).sum())
    return f, n_touched


def keep_symbols():
    """The raw panel's own symbol set — dollar volume (close x volume) is invariant under a split
    adjustment, so the `>= 10 sessions of $10M+` pre-filter and the PIT intersection select exactly
    the same symbols as `build_panel_r.keep_symbols`."""
    col = pq.read_table(RAW_PANEL, columns=['symbol']).column('symbol')
    s = set()
    for i in range(col.num_chunks):
        ch = col.chunk(i)
        s.update(ch.dictionary.to_pylist() if pa.types.is_dictionary(ch.type)
                 else ch.to_pylist())
    del col, ch
    gc.collect()
    log(f'symbol set from the raw panel: {len(s):,}')
    return s


def main() -> int:
    ev = load_events()
    keep = keep_symbols()
    sym_map, day_map = {}, {}
    scode_p, dcode_p = [], []
    arr_p = {k: [] for k in ('open', 'high', 'low', 'close', 'volume', 'close_raw')}
    n_adj_rows = 0
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
            sy = d.symbol.astype(str).to_numpy()
            dt = np.array([str(x)[:10] for x in d.bar_date], dtype=object)
            fac, nt = factors(sy, dt, ev)
            n_adj_rows += nt
            scode_p.append(np.fromiter((sym_map.setdefault(s, len(sym_map)) for s in sy),
                                       dtype='int32', count=len(d)))
            dcode_p.append(np.fromiter((day_map.setdefault(x, len(day_map)) for x in dt),
                                       dtype='int32', count=len(d)))
            arr_p['close_raw'].append(d['close'].to_numpy('float64').astype('float32'))
            for c in ('open', 'high', 'low', 'close'):
                arr_p[c].append((d[c].to_numpy('float64') / fac).astype('float32'))
            arr_p['volume'].append((d.volume.to_numpy('float64') * scale * fac).astype('float32'))
            n += len(d)
            del d, sy, dt, fac
        log(f'{path}: kept {n:,} rows')
        del f
        gc.collect()
    log(f'rows whose price scale was adjusted: {n_adj_rows:,}')
    del ev, keep
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
    prev_close = shift_k(arr['close'], starts, ends, 1).astype('float32')

    # ---- bad-print opens (see the module docstring) --------------------------------------------
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.abs(arr['open'] / prev_close - 1.0)
        bad_open = np.isfinite(g) & (g > BADPRINT_GAP)
        del g
        gc.collect()
        g = np.abs(arr['close'] / prev_close - 1.0)
        bad_open &= np.isfinite(g) & (g < BADPRINT_CLOSE)
        del g
        gc.collect()
    log(f'bad-print opens blanked: {int(bad_open.sum()):,} rows '
        f'({bad_open.mean() * 100:.4f}% of the panel)')
    bp = pd.DataFrame(dict(symbol=[str(syms[s]) for s in scode[bad_open]],
                           bar_date=[str(uniq_days[d]) for d in dcode[bad_open]],
                           open=arr['open'][bad_open], prev_close=prev_close[bad_open],
                           close=arr['close'][bad_open]))
    bp.sort_values('bar_date').to_csv(f'{R}/bad_print_opens.csv', index=False, float_format='%.6g')
    arr['open'] = np.where(bad_open, np.nan, arr['open']).astype('float32')
    gc.collect()

    adv20 = trailing_mean(arr['volume'], starts, ends, 20, 10).astype('float32')
    c5 = shift_k(arr['close'], starts, ends, 5).astype('float32')
    with np.errstate(invalid='ignore', divide='ignore'):
        ret5 = (arr['close'] / c5 - 1.0).astype('float32')
        ret_on = (arr['open'] / prev_close - 1.0).astype('float32')
        vol_ratio = (arr['volume'] / adv20).astype('float32')
    del c5
    gc.collect()
    high52 = trailing_max(arr['high'], starts, ends, 250, 60).astype('float32')
    gc.collect()

    cols = {'symbol': pa.DictionaryArray.from_arrays(pa.array(scode, type=pa.int32()),
                                                     pa.array([str(x) for x in syms])),
            'bar_date': pa.DictionaryArray.from_arrays(pa.array(dcode, type=pa.int32()),
                                                       pa.array([str(x) for x in uniq_days]))}
    for k in ('open', 'high', 'low', 'close', 'volume', 'close_raw'):
        cols[k] = pa.array(arr[k])
    for k, v in (('prev_close', prev_close), ('adv20', adv20), ('ret5', ret5),
                 ('high52', high52), ('ret_on', ret_on), ('vol_ratio', vol_ratio)):
        cols[k] = pa.array(v)
    cols['bad_open'] = pa.array(bad_open)
    del arr, prev_close, adv20, ret5, ret_on, vol_ratio, high52
    gc.collect()
    tbl = pa.table(cols)
    del cols
    gc.collect()
    pq.write_table(tbl, OUT)
    log(f'wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
