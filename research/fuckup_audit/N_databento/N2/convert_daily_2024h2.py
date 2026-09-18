#!/usr/bin/env python3
"""Stage N2 step 1b — DBN -> parquet for the 2024H2 EQUS.SUMMARY daily pull.

The pull was made with `symbols='ALL_SYMBOLS'`, whose DBN carries no symbol mappings, so the
instrument_id -> raw_symbol map is resolved separately (`symbology_2024h2.json`, free) and joined
per date interval.  `bar_date` is the UTC date of `ts_event` — the convention of the existing
`data/research/databento/equs_daily_2025_2026.parquet` (its first bar_date is 2025-01-02, the first
session of 2025; converting to ET would shift every bar a day earlier).

Output: data/research/databento/equs_daily_2024H2.parquet, SAME column layout as the 2025-2026 file.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
import databento as db  # noqa: E402

N2 = 'research/fuckup_audit/N_databento/N2'
DBN = 'data/research/databento/equs_summary_ohlcv1d_ALL_20240701_20241231.dbn.zst'
SYMJSON = f'{N2}/symbology_2024h2.json'
OUT = 'data/research/databento/equs_daily_2024H2.parquet'
REF = 'data/research/databento/equs_daily_2025_2026.parquet'


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    store = db.DBNStore.from_file(DBN)
    df = store.to_df(price_type='float').reset_index()
    log(f'{len(df):,} records')
    ts = pd.to_datetime(df['ts_event'], utc=True)
    bar_date = ts.dt.strftime('%Y-%m-%d')

    # instrument_id -> symbol, per [d0, d1) interval
    j = json.load(open(SYMJSON))['result']
    rows = []
    for sym, ivs in j.items():
        for iv in ivs:
            rows.append((int(iv['s']), iv['d0'], iv['d1'], sym))
    m = pd.DataFrame(rows, columns=['instrument_id', 'd0', 'd1', 'symbol'])
    log(f'symbology: {len(m):,} intervals, {m.symbol.nunique():,} symbols, '
        f'{m.instrument_id.nunique():,} ids')
    m.to_csv(f'{N2}/equs_instrument_symbol_map_2024h2.csv', index=False)

    d = pd.DataFrame({'bar_date': bar_date.to_numpy(),
                      'instrument_id': df['instrument_id'].to_numpy('uint32'),
                      'open': df['open'].to_numpy('float64'),
                      'high': df['high'].to_numpy('float64'),
                      'low': df['low'].to_numpy('float64'),
                      'close': df['close'].to_numpy('float64'),
                      'volume': df['volume'].to_numpy('uint64')})
    del df
    d = d.merge(m[['instrument_id', 'd0', 'd1', 'symbol']], on='instrument_id', how='left')
    ok = (d.symbol.notna()) & (d.bar_date >= d.d0) & (d.bar_date < d.d1)
    log(f'interval-valid rows {int(ok.sum()):,} of {len(d):,} '
        f'({int((~ok).sum()):,} dropped/unmapped)')
    d.loc[~ok, 'symbol'] = np.nan
    d = d.drop(columns=['d0', 'd1']).drop_duplicates(subset=['bar_date', 'instrument_id'],
                                                     keep='first')
    out = d[['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low', 'close', 'volume']]
    out = out.sort_values(['bar_date', 'symbol'], kind='mergesort').reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    log(f'wrote {OUT}: {len(out):,} rows  {out.bar_date.min()}..{out.bar_date.max()}  '
        f'{out.symbol.nunique():,} symbols  {out.bar_date.nunique()} sessions  '
        f'({os.path.getsize(OUT) / 1e6:.1f} MB)')
    ref = pd.read_parquet(REF, filters=[('bar_date', '==', '2025-01-02')])
    log(f'layout: cols equal={list(ref.columns) == list(out.columns)}  '
        f'dtypes equal={[str(x) for x in ref.dtypes] == [str(x) for x in out.dtypes]}  '
        f'ref rows on 2025-01-02 {len(ref):,} vs new rows on {out.bar_date.max()} '
        f'{int((out.bar_date == out.bar_date.max()).sum()):,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
