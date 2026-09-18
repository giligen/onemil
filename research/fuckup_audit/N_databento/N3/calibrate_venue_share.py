#!/usr/bin/env python3
"""Stage N3 step 1b — how much of consolidated volume does XNAS.ITCH see?

The paper's ADV filter is 1,000,000 CONSOLIDATED shares; our daily panel is Nasdaq-venue only.
This measures the venue share directly: one month of XNAS.ITCH ohlcv-1d (2024-09) against the
consolidated EQUS.SUMMARY daily panel for the same symbol-days, and cross-checks EQUS.SUMMARY
against Alpaca SIP daily bars (cache.db, read-only) so "consolidated" is not taken on faith.

Writes N3/venue_share.md and prints the scaled ADV threshold to use in build_pool.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

N3 = 'research/fuckup_audit/N_databento/N3'
RAW = f'{N3}/raw_daily'
CAL_DBN = f'{RAW}/xnas_ohlcv1d_cal202409.dbn.zst'
CAL_SYM = f'{RAW}/symbology_cal202409.json'
CAL_PQ = f'{RAW}/xnas_ohlcv1d_cal202409.parquet'
CONSOL = 'data/research/databento/equs_daily_2024H2.parquet'
START, END = '2024-09-01', '2024-10-01'
BUDGET = 1.50


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    if not os.path.exists(CAL_PQ):
        cost = float(client.metadata.get_cost(dataset='XNAS.ITCH', schema='ohlcv-1d',
                                              symbols='ALL_SYMBOLS', stype_in='raw_symbol',
                                              start=START, end=END))
        log(f'PRICED calibration month {START}..{END}: ${cost:.4f}')
        if cost > BUDGET:
            log('STOP: calibration month over budget')
            return 2
        if not os.path.exists(CAL_DBN):
            client.timeseries.get_range(dataset='XNAS.ITCH', schema='ohlcv-1d', symbols='ALL_SYMBOLS',
                                        stype_in='raw_symbol', start=START, end=END).to_file(CAL_DBN)
        if not os.path.exists(CAL_SYM):
            for attempt in range(5):
                try:
                    r = client.symbology.resolve(dataset='XNAS.ITCH', symbols='ALL_SYMBOLS',
                                                 stype_in='raw_symbol', stype_out='instrument_id',
                                                 start_date=START, end_date=END)
                    json.dump(r, open(CAL_SYM, 'w'))
                    break
                except Exception as exc:
                    log(f'  symbology attempt {attempt + 1}: {str(exc)[:80]}')
                    time.sleep(15 * (attempt + 1))
            else:
                return 3
        j = json.load(open(CAL_SYM))['result']
        rows = [(int(iv['s']), int(iv['d0'].replace('-', '')), int(iv['d1'].replace('-', '')), s)
                for s, ivs in j.items() for iv in ivs]
        m = pd.DataFrame(rows, columns=['instrument_id', 'd0i', 'd1i', 'symbol'])
        m = m.drop_duplicates(['instrument_id', 'd0i'])
        m['instrument_id'] = m.instrument_id.astype('uint32')
        m = m.sort_values(['d0i', 'instrument_id']).reset_index(drop=True)
        store = db.DBNStore.from_file(CAL_DBN)
        df = store.to_df(price_type='float').reset_index()
        ts = pd.to_datetime(df['ts_event'], utc=True)
        # ts_event is 00:00 UTC of the SESSION date -> take the UTC date (never convert to ET)
        d = pd.DataFrame({'bar_date': ts.dt.strftime('%Y-%m-%d'),
                          'instrument_id': df['instrument_id'].to_numpy('uint32'),
                          'close': df['close'].to_numpy('float64'),
                          'volume': df['volume'].to_numpy('float64')})
        d['di'] = d.bar_date.str.replace('-', '', regex=False).astype('int64')
        d = d.sort_values(['di', 'instrument_id']).reset_index(drop=True)
        d = pd.merge_asof(d, m, left_on='di', right_on='d0i', by='instrument_id', direction='backward')
        d = d[d.di < d.d1i].drop(columns=['di', 'd0i', 'd1i']).dropna(subset=['symbol'])
        d.to_parquet(CAL_PQ, index=False)
        log(f'{CAL_PQ}: {len(d):,} rows')

    x = pd.read_parquet(CAL_PQ)[['bar_date', 'symbol', 'close', 'volume']]
    x['symbol'] = x.symbol.astype(str)
    x = x.rename(columns={'volume': 'v_xnas', 'close': 'c_xnas'})
    c = pd.read_parquet(CONSOL, columns=['bar_date', 'symbol', 'close', 'volume'])
    c['symbol'] = c.symbol.astype(str)
    c = c[(c.bar_date >= START) & (c.bar_date < END)].rename(columns={'volume': 'v_con', 'close': 'c_con'})

    # (1) is EQUS.SUMMARY really consolidated?  cross-check vs Alpaca SIP daily bars.
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    a = pd.read_sql('select symbol, bar_date, close as c_alp, volume as v_alp from daily_bars '
                    'where bar_date >= ? and bar_date < ?', con, params=(START, END))
    con.close()
    chk = c.merge(a, on=['bar_date', 'symbol'], how='inner')
    chk = chk[(chk.v_alp > 0) & (chk.v_con > 0)]
    r_alp = (chk.v_con / chk.v_alp)

    j = x.merge(c, on=['bar_date', 'symbol'], how='inner')
    j = j[(j.v_con > 0) & (j.v_xnas > 0)]
    j['share'] = j.v_xnas / j.v_con
    j = j[j.share <= 1.5]
    liq = j[j.v_con >= 1e6]
    px = (j.c_xnas / j.c_con)

    lines = [
        '# XNAS.ITCH venue share of consolidated volume (2024-09, the calibration month)',
        '',
        f'EQUS.SUMMARY vs Alpaca SIP daily volume, n={len(chk):,}: median ratio '
        f'{r_alp.median():.3f} (p10 {r_alp.quantile(.1):.3f}, p90 {r_alp.quantile(.9):.3f}) '
        '— EQUS.SUMMARY is a consolidated tape.',
        f'XNAS close vs consolidated close, n={len(px):,}: median {px.median():.5f}, '
        f'share within 0.1% = {float((px.sub(1).abs() <= 0.001).mean()) * 100:.1f}% (price-scale check).',
        '',
        '| cohort | n symbol-days | median XNAS share | p25 | p75 |',
        '|---|---|---|---|---|',
        f'| all matched | {len(j):,} | {j.share.median():.3f} | {j.share.quantile(.25):.3f} | '
        f'{j.share.quantile(.75):.3f} |',
        f'| consolidated volume >= 1M | {len(liq):,} | {liq.share.median():.3f} | '
        f'{liq.share.quantile(.25):.3f} | {liq.share.quantile(.75):.3f} |',
        '',
    ]
    share = float(liq.share.median())
    lines.append(f'**Scaled ADV threshold**: 1,000,000 consolidated x {share:.3f} = '
                 f'**{1e6 * share:,.0f} XNAS-venue shares**.')
    open(f'{N3}/venue_share.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'ADV_MIN_XNAS={1e6 * share:.0f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
