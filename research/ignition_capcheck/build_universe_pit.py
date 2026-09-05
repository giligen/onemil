"""Ignition capcheck — POINT-IN-TIME universe (2026-09-05).

Same prefilter as build_universe.py, applied to the Databento EQUS.SUMMARY
daily bars (every US equity that existed on each date, delisted names
included) instead of cache.db daily_bars (seeded Dec-24 from TODAY's
listings = survivorship). Writes universe_raw_pit.csv and prints the
diff vs universe_raw.csv: candidate symbol-days the cache-based universe
never saw, by month, split into "symbol entirely absent from daily_bars"
(dead/delisted) vs "symbol present but that day missing".

Prefilter (superset of the book universe by construction):
  open >= 1.95, gap < 5.5% (or no prev), high >= open*1.09, volume*high >= $2M
"""
import os
import sqlite3
import sys

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
PARQUET = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
OUT = f'{ROOT}/research/ignition_capcheck/universe_raw_pit.csv'
OLD = f'{ROOT}/research/ignition_capcheck/universe_raw.csv'
START, END = '2025-01-01', '2026-09-04'


def build_pit_universe(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply the capcheck prefilter to a point-in-time daily frame."""
    d = daily.sort_values(['symbol', 'bar_date']).copy()
    d['prev_close'] = d.groupby('symbol')['close'].shift(1)
    d = d[(d['bar_date'] >= START) & (d['bar_date'] <= END)]
    gap = (d['open'] - d['prev_close']) / d['prev_close'] * 100.0
    keep = d[(d['open'] >= 1.95)
             & (d['prev_close'].isna() | (gap < 5.5))
             & (d['high'] >= d['open'] * 1.09)
             & (d['volume'] * d['high'] >= 2_000_000)]
    return keep[['symbol', 'bar_date', 'open', 'high', 'close', 'volume', 'prev_close']]


def main() -> None:
    daily = pd.read_parquet(PARQUET)
    # Databento raw symbols: drop test/odd instruments; keep plain tickers
    # comparable to Alpaca's (share classes use '.', e.g. BRK.B — kept).
    daily = daily[daily['symbol'].str.match(r'^[A-Z]{1,5}(\.[A-Z])?$', na=False)]
    daily = daily[daily['volume'] > 0]
    print(f"databento daily: {len(daily):,} rows, {daily['symbol'].nunique():,} symbols")
    pit = build_pit_universe(daily)
    pit.to_csv(OUT, index=False)
    print(f"PIT UNIVERSE: {len(pit):,} symbol-days, {pit['symbol'].nunique():,} symbols "
          f"{pit['bar_date'].min()}..{pit['bar_date'].max()} -> {OUT}")

    if not os.path.exists(OLD):
        print(f"(no {OLD} to diff against)")
        return
    old = pd.read_csv(OLD, usecols=['symbol', 'bar_date'])
    old_keys = set(zip(old['symbol'], old['bar_date']))
    pit['in_old'] = [(s, d) in old_keys for s, d in zip(pit['symbol'], pit['bar_date'])]
    conn = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=60)
    cache_syms = set(pd.read_sql("select distinct symbol from daily_bars", conn)['symbol'])
    conn.close()
    pit['sym_in_cache'] = pit['symbol'].isin(cache_syms)
    pit['month'] = pit['bar_date'].str[:7]
    miss = pit[~pit['in_old']]
    old_end = old['bar_date'].max()
    print(f"\nold universe ends {old_end}; comparing months <= {old_end[:7]}")
    cmp = pit[pit['bar_date'] <= old_end]
    by = cmp.groupby('month').agg(
        pit=('symbol', 'size'),
        in_old=('in_old', 'sum'),
        missing=('in_old', lambda s: int((~s).sum())),
        missing_dead_symbol=('sym_in_cache', lambda s: int((~s[~cmp.loc[s.index, 'in_old']]).sum())),
    )
    by['missing_pct'] = (by['missing'] / by['pit'] * 100).round(1)
    print(by.to_string())
    tot = len(cmp); tot_miss = int((~cmp['in_old']).sum())
    dead = int((~cmp.loc[~cmp['in_old'], 'sym_in_cache']).sum())
    print(f"\nTOTAL through {old_end}: PIT {tot:,} symbol-days; cache-based universe "
          f"missed {tot_miss:,} ({tot_miss / max(tot, 1) * 100:.1f}%), of which "
          f"{dead:,} are on symbols absent from daily_bars entirely (delisted/renamed).")
    miss[['symbol', 'bar_date', 'open', 'high', 'close', 'volume', 'prev_close',
          'sym_in_cache']].to_csv(f'{ROOT}/research/ignition_capcheck/universe_pit_missing.csv',
                                  index=False)
    print(f"missing symbol-days -> research/ignition_capcheck/universe_pit_missing.csv")


if __name__ == '__main__':
    main()
