"""ORB — point-in-time universe survivorship diff (2026-09-05).

Applies study_orb_broad.load_broad_universe's gate (gap >= 5% open vs prev
close, prev-day volume >= 500K, open in [$3, $30]) to the Databento
EQUS.SUMMARY daily bars (every US equity that existed on each date) and
to cache.db daily_bars, and reports per month how many candidate
symbol-days the cache-based BT universe could never see — split into
"symbol absent from daily_bars entirely" (delisted/renamed/listed after
the universe seed) vs "symbol present, day missing".

Writes research/orb_entered_inclusive/orb_universe_pit_missing.csv.
"""
import sqlite3
import sys

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
PARQUET = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
OUT = f'{ROOT}/research/orb_entered_inclusive/orb_universe_pit_missing.csv'
START, END = '2025-01-01', '2026-09-04'
MIN_GAP_PCT, MIN_PREV_DAY_VOL, MIN_OPEN, MAX_OPEN = 5.0, 500_000, 3.0, 30.0
TEST_TICKERS = {'ZVZZT'}


def orb_candidates(daily: pd.DataFrame) -> pd.DataFrame:
    """The load_broad_universe gate on a (symbol, bar_date, open, close, volume) frame."""
    d = daily.sort_values(['symbol', 'bar_date']).copy()
    g = d.groupby('symbol')
    d['prev_close'] = g['close'].shift(1)
    d['prev_vol'] = g['volume'].shift(1)
    d = d[(d['bar_date'] >= START) & (d['bar_date'] <= END)]
    gap = (d['open'] - d['prev_close']) / d['prev_close'] * 100.0
    keep = d[d['prev_close'].notna() & (d['prev_close'] > 0) & (gap >= MIN_GAP_PCT)
             & (d['prev_vol'] >= MIN_PREV_DAY_VOL)
             & (d['open'] >= MIN_OPEN) & (d['open'] <= MAX_OPEN)]
    return keep[['symbol', 'bar_date', 'open', 'high', 'close', 'volume', 'prev_close', 'prev_vol']]


def main() -> None:
    daily = pd.read_parquet(PARQUET)
    daily = daily[daily['symbol'].str.match(r'^[A-Z]{1,5}(\.[A-Z])?$', na=False)
                  & ~daily['symbol'].isin(TEST_TICKERS) & (daily['volume'] > 0)]
    pit = orb_candidates(daily)
    del daily
    print(f"PIT ORB candidates: {len(pit):,} symbol-days, {pit['symbol'].nunique():,} symbols")

    conn = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    # Same gate in SQL (load_broad_universe's query minus the intraday-exists
    # clause) — pulling the whole table into pandas OOMs under the 3.5GB cap.
    cb = pd.read_sql("""
        WITH r AS (
            SELECT symbol, bar_date, open,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_close,
                   LAG(volume) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_vol
            FROM daily_bars WHERE bar_date >= '2024-12-15')
        SELECT symbol, bar_date FROM r
        WHERE bar_date BETWEEN ? AND ? AND prev_close IS NOT NULL AND prev_close > 0
          AND (open - prev_close) / prev_close * 100 >= ? AND prev_vol >= ?
          AND open BETWEEN ? AND ?""", conn,
        params=[START, END, MIN_GAP_PCT, MIN_PREV_DAY_VOL, MIN_OPEN, MAX_OPEN])
    cache_syms = set(pd.read_sql("select distinct symbol from daily_bars", conn)['symbol'])
    print(f"cache-based ORB candidates (no intraday-exists clause): {len(cb):,}")
    cb_keys = set(zip(cb['symbol'], cb['bar_date']))
    conn.close()

    pit['in_cache'] = [(s, d) in cb_keys for s, d in zip(pit['symbol'], pit['bar_date'])]
    pit['sym_in_cache'] = pit['symbol'].isin(cache_syms)
    pit['month'] = pit['bar_date'].str[:7]
    miss = pit[~pit['in_cache']]
    tab = pit.groupby('month').agg(pit=('symbol', 'size'), in_cache=('in_cache', 'sum'))
    tab['missing'] = tab['pit'] - tab['in_cache']
    tab['missing_dead_symbol'] = miss.groupby('month')['sym_in_cache'].apply(lambda s: int((~s).sum()))
    tab['missing_dead_symbol'] = tab['missing_dead_symbol'].fillna(0).astype(int)
    tab['missing_pct'] = (tab['missing'] / tab['pit'] * 100).round(1)
    print(tab.to_string())
    n_dead = int((~miss['sym_in_cache']).sum())
    print(f"\nTOTAL: PIT {len(pit):,}; cache-based universe missed {len(miss):,} "
          f"({len(miss) / len(pit) * 100:.1f}%), {n_dead:,} on symbols absent from daily_bars.")
    print("top missing symbols:", miss['symbol'].value_counts().head(15).to_dict())
    miss.drop(columns=['in_cache']).to_csv(OUT, index=False)
    print(f"missing symbol-days -> {OUT}")


if __name__ == '__main__':
    main()
