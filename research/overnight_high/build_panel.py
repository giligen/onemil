#!/usr/bin/env python3
"""PREREG_1550 step 2 — ONE panel schema for both samples.

EXTENSION panel <- research/overnight_high/alpaca_daily_2019_2024H1.parquet (Alpaca daily bars,
2019-01-02..2024-06-28, survivorship-corrected universe per fetch_alpaca_daily.py).
PANEL <- data/research/databento/equs_daily_2024H2.parquet + equs_daily_2025_2026.parquet
(EQUS.SUMMARY daily bars, delisted names included by construction).

Schema (both outputs, identical columns): symbol, bar_date, open, high, low, close, volume,
adv20, dvol20, vol_ratio, high252, next_open, ret_on_next.

Convention chosen (PREREG_1550 explicit requirement to state it):
  * adv20  = mean(volume) over the PRIOR 20 sessions (shift(1).rolling(20, min_periods=10)) --
    "ADV20 through t-1", matching the existing `research/lit_review_2026/build_daily_panel.py`
    convention (min_periods=10 tolerates the first-3-weeks warm-up rather than voiding it).
  * dvol20 = mean(close*volume) over the same PRIOR 20 sessions, same min_periods.
  * vol_ratio = volume_t / adv20_t (today's volume against yesterday-and-before ADV -- today's
    volume can never leak into its own denominator).
  * high252 = MAX of the prior 252 sessions' CLOSE (shift(1).rolling(252, min_periods=252).max())
    -- the PREREG rule text says "the highest close of the prior 252 sessions" (close, not high;
    252, not 250; strictly prior, not including t) and the refuter explicitly re-checks this field
    for look-ahead, so min_periods is set to the FULL window (252) rather than the looser 60/10
    used for adv20/dvol20: a 252-day-high flag computed from a 40-day warm-up window is not a
    252-day high. This differs from the legacy `high52` field in build_daily_panel.py (used HIGH,
    250 sessions, min_periods=60) -- that field is not reused here.
  * next_open = open_{t+1} (shift(-1)); ret_on_next = next_open / close_t - 1 (raw close, raw
    next open, no adjustment -- the price-scale refuter checks this on the actual trades).

Test tickers and the $5 / $10M universe gate are NOT applied here (this is the raw derived-field
panel); cell_1550.py applies the rule's universe and exclusions per the PREREG.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger('build_panel')

ADV_MIN_PERIODS = 10
HIGH252_WINDOW = 252
HIGH252_MIN_PERIODS = 252


def derive_fields(d: pd.DataFrame) -> pd.DataFrame:
    """Add adv20/dvol20/vol_ratio/high252/next_open/ret_on_next to a symbol,bar_date,OHLCV frame."""
    d = d[d.symbol.notna() & (d.symbol.astype(str).str.strip() != '')].copy()
    d['bar_date'] = d.bar_date.astype(str).str[:10]
    for k in ('open', 'high', 'low', 'close', 'volume'):
        d[k] = pd.to_numeric(d[k], errors='coerce').astype('float64')
    d = d.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    d = d.drop_duplicates(subset=['symbol', 'bar_date']).sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    d['symbol'] = d['symbol'].astype('category')
    d['_dollar_vol'] = d['close'] * d['volume']
    g = d.groupby('symbol', observed=True)
    d['adv20'] = g['volume'].transform(lambda s: s.shift(1).rolling(20, min_periods=ADV_MIN_PERIODS).mean())
    d['dvol20'] = g['_dollar_vol'].transform(lambda s: s.shift(1).rolling(20, min_periods=ADV_MIN_PERIODS).mean())
    d['vol_ratio'] = d['volume'] / d['adv20']
    d = d.drop(columns=['_dollar_vol'])
    d['high252'] = g['close'].transform(lambda s: s.shift(1).rolling(HIGH252_WINDOW, min_periods=HIGH252_MIN_PERIODS).max())
    d['next_open'] = g['open'].shift(-1)
    d['ret_on_next'] = d['next_open'] / d['close'] - 1
    for k in ('adv20', 'dvol20', 'vol_ratio', 'high252', 'next_open', 'ret_on_next'):
        d[k] = d[k].astype('float64')
    keep = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume',
            'adv20', 'dvol20', 'vol_ratio', 'high252', 'next_open', 'ret_on_next']
    return d[keep]


def build_extension():
    src = os.path.join(HERE, 'alpaca_daily_2019_2024H1.parquet')
    d = pd.read_parquet(src)
    out = derive_fields(d)
    out = out[(out.bar_date >= '2019-01-02') & (out.bar_date <= '2024-06-28')]
    dest = os.path.join(HERE, 'panel_extension.parquet')
    out.to_parquet(dest, index=False)
    log.info('EXTENSION panel: %d rows, %d symbols, %s..%s -> %s',
              len(out), out.symbol.nunique(), out.bar_date.min(), out.bar_date.max(), dest)
    return out


def build_panel_2024_2026():
    cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
    p1 = pd.read_parquet(os.path.join(ROOT, 'data/research/databento/equs_daily_2024H2.parquet'), columns=cols)
    p2 = pd.read_parquet(os.path.join(ROOT, 'data/research/databento/equs_daily_2025_2026.parquet'), columns=cols)
    d = pd.concat([p1, p2], ignore_index=True)
    dupe_dates = d.duplicated(subset=['symbol', 'bar_date']).sum()
    if dupe_dates:
        log.warning('%d duplicate (symbol,bar_date) rows across the two Databento parquets -- keeping last', dupe_dates)
    out = derive_fields(d)
    dest = os.path.join(HERE, 'panel_2024_2026.parquet')
    out.to_parquet(dest, index=False)
    log.info('PANEL (2024-07..2026-09): %d rows, %d symbols, %s..%s -> %s',
              len(out), out.symbol.nunique(), out.bar_date.min(), out.bar_date.max(), dest)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    build_extension()
    build_panel_2024_2026()


if __name__ == '__main__':
    main()
