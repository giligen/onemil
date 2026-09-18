#!/usr/bin/env python3
"""Multi-day DATA stage, step 3 — quantify the survivorship hole in the Alpaca universe.

The panel's universe is TODAY's Alpaca asset list, so every name that delisted
between 2016 and now is missing. This script measures how big that hole is on the
two point-in-time overlaps we own, and writes the union of the delisted names so
the family stage can do the PIT re-run:

  (a) Databento `pit_definition` (EQUS.SUMMARY definitions, ALL exchanges),
      monthly 2024-07 → 2026-09 — `research/scripts/pit_listings.PitListings`.
      For each month: how many names listed THEN are absent from the Alpaca
      universe TODAY.
  (b) XNAS.ITCH `ohlcv-1d` on disk, 2018-05 → 2024-06
      (`research/fuckup_audit/N_databento/N3/xnas_daily.parquet` +
       `research/fuckup_audit/R_daily/xnas_daily_2024H1.parquet`).
      Symbols are resolved through Databento's daily symbology map, which was
      repaired once (N3 `fix_symbol_map.py`: ITCH re-assigns instrument_ids every
      day, a stale map silently renames symbols) — the parquets read here are the
      repaired ones, and a residual mis-map shows up as a "delisted" name that
      never existed, so the tables below are an UPPER bound on attrition.
      This feed is the Nasdaq VENUE tape (every name that traded on Nasdaq, not
      only Nasdaq-listed), which is exactly the panel the PIT re-run needs.

Outputs
  survivorship_pit.csv      per-month table for (a)
  survivorship_xnas.csv     per-year table for (b)
  delisted_names.parquet    union: symbol, last_seen, source, exchange
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(REPO))

from research.scripts.pit_listings import PitListings, is_test_ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
UNIVERSE = HERE / 'universe.parquet'
XNAS_FILES = [
    REPO / 'research/fuckup_audit/N_databento/N3/xnas_daily.parquet',
    REPO / 'research/fuckup_audit/R_daily/xnas_daily_2024H1.parquet',
]
# Alpaca exchange label -> Databento listing MIC, for the like-for-like overlap.
MIC_OF = {'NASDAQ': 'XNAS', 'NYSE': 'XNYS', 'ARCA': 'ARCX', 'BATS': 'BATS', 'AMEX': 'XASE'}


import re  # noqa: E402

# Nasdaq ITCH marks non-common classes with a suffix character ('+' warrant,
# '=' unit, '^' preferred, '~' when-issued, '#'/'$'/'*' test & special). Those
# names are NOT in our research universe by construction, so counting them as
# "delisted" would inflate the attrition rate. This keeps roots and class shares.
PLAIN_SYMBOL_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')


def norm(sym: str) -> str:
    """Databento/EDGAR class-share separators (' ', '-') -> Alpaca's '.'."""
    return (sym or '').replace(' ', '.').replace('-', '.').upper()


def is_plain_equity_symbol(sym: str) -> bool:
    """True for a plain common-stock ticker (no ITCH warrant/unit/preferred suffix)."""
    return bool(PLAIN_SYMBOL_RE.match(sym))


def main() -> None:
    uni = pd.read_parquet(UNIVERSE)
    today = {norm(s) for s in uni['symbol']}
    # Like-for-like denominator: EVERY active Alpaca asset, unfiltered. A name
    # absent from `alive` is gone from the broker; a name absent from `today` but
    # present in `alive` was merely filtered out by our class rules.
    alive_path = UNIVERSE.with_name('assets_all_active.parquet')
    alive = {norm(s) for s in pd.read_parquet(alive_path)['symbol']} if alive_path.exists() \
        else set(today)
    print(f'Alpaca research universe today: {len(today)} symbols; '
          f'all active assets: {len(alive)}', flush=True)

    # ---------------- (a) Databento point-in-time definitions ----------------
    pit = PitListings()
    first, last = pit.coverage
    months = pd.period_range(first, last, freq='M')
    rows = []
    for per in months:
        date = per.to_timestamp().strftime('%Y-%m-%d')
        df = pit._month(per.strftime('%Y%m'))
        df = df[~df['raw_symbol'].map(is_test_ticker)]
        common = df[df['security_type'] == 'C']
        for label, sub in (('all_types', df), ('common_stock', common)):
            syms = {norm(s) for s in sub['raw_symbol']}
            missing = syms - today
            gone = syms - alive
            rows.append({'month': per.strftime('%Y-%m'), 'scope': label,
                         'listed': len(syms),
                         'absent_universe': len(missing),
                         'absent_broker': len(gone),
                         'pct_absent_universe': round(100 * len(missing) / max(len(syms), 1), 2),
                         'pct_absent_broker': round(100 * len(gone) / max(len(syms), 1), 2)})
    pit_tbl = pd.DataFrame(rows)
    pit_tbl.to_csv(HERE / 'survivorship_pit.csv', index=False)
    print(pit_tbl[pit_tbl.scope == 'common_stock'].to_string(index=False), flush=True)

    # last seen month of every PIT name that is gone today
    seen: dict[str, tuple[str, str]] = {}
    for per in months:
        df = pit._month(per.strftime('%Y%m'))
        df = df[(df['security_type'] == 'C') & (~df['raw_symbol'].map(is_test_ticker))]
        for sym, exch in zip(df['raw_symbol'], df['exchange']):
            n = norm(sym)
            if n not in alive:
                seen[n] = (per.strftime('%Y-%m-%d'), str(exch))
    pit_gone = pd.DataFrame(
        [{'symbol': s, 'last_seen': d, 'exchange': e, 'source': 'databento_pit'}
         for s, (d, e) in seen.items()])
    print(f'(a) distinct common-stock names listed 2024-07..2026-09 and absent today: '
          f'{len(pit_gone)}', flush=True)

    # ---------------- (b) XNAS.ITCH daily tape on disk ----------------
    # Streamed in record batches: 13.4M rows never enter one frame (node rule).
    import pyarrow.dataset as pads

    stats: dict[tuple[int, str], list[float]] = {}   # (year, symbol) -> [sessions, $vol]
    last_seen_map: dict[str, str] = {}
    for f in XNAS_FILES:
        if not f.exists():
            print(f'WARNING: {f} missing — XNAS overlap incomplete', flush=True)
            continue
        scanner = pads.dataset(f, format='parquet').scanner(
            columns=['bar_date', 'symbol', 'close', 'volume'], batch_size=200_000)
        for batch in scanner.to_batches():
            d = batch.to_pandas()
            d = d[(d['volume'] > 0) & (d['close'] > 0)]
            if d.empty:
                continue
            d['bar_date'] = pd.to_datetime(d['bar_date'])
            uniq = {s: norm(s) for s in d['symbol'].unique()
                    if not is_test_ticker(s) and is_plain_equity_symbol(norm(s))}
            d = d[d['symbol'].isin(uniq)]
            d['n'] = d['symbol'].map(uniq)
            d['year'] = d['bar_date'].dt.year
            d['dollars'] = d['close'] * d['volume']
            agg = d.groupby(['year', 'n']).agg(sessions=('close', 'size'),
                                               dollars=('dollars', 'sum'))
            for (year, n), row in agg.iterrows():
                cur = stats.setdefault((year, n), [0.0, 0.0])
                cur[0] += row['sessions']
                cur[1] += row['dollars']
            gone_rows = d[~d['n'].isin(alive)]
            for n, dt in gone_rows.groupby('n')['bar_date'].max().items():
                iso = dt.strftime('%Y-%m-%d')
                if last_seen_map.get(n, '') < iso:
                    last_seen_map[n] = iso
            del d, agg

    rows = []
    years = sorted({y for y, _ in stats})
    for y in years:
        names = {n: v for (yy, n), v in stats.items() if yy == y}
        liquid = {n for n, v in names.items() if v[0] >= 20 and v[1] / v[0] >= 100_000}
        rows.append({
            'year': y,
            'symbols': len(names),
            'absent_broker': len(set(names) - alive),
            'absent_universe': len(set(names) - today),
            'liquid_symbols': len(liquid),
            'liquid_absent_broker': len(liquid - alive),
        })
    per_year = pd.DataFrame(rows)
    per_year['pct_absent_broker'] = (100 * per_year['absent_broker'] /
                                     per_year['symbols']).round(2)
    per_year['pct_liquid_absent'] = (100 * per_year['liquid_absent_broker'] /
                                     per_year['liquid_symbols']).round(2)
    per_year.to_csv(HERE / 'survivorship_xnas.csv', index=False)
    print(per_year.to_string(index=False), flush=True)

    last_seen = pd.DataFrame(sorted(last_seen_map.items()),
                             columns=['symbol', 'last_seen'])
    last_seen['exchange'] = 'XNAS_tape'
    last_seen['source'] = 'xnas_itch_daily'

    union = pd.concat([pit_gone, last_seen], ignore_index=True)
    union = (union.sort_values('last_seen')
                  .drop_duplicates('symbol', keep='last')
                  .sort_values('symbol').reset_index(drop=True))
    union.to_parquet(HERE / 'delisted_names.parquet', index=False)
    print(f'union of delisted/absent names: {len(union)} -> delisted_names.parquet', flush=True)


if __name__ == '__main__':
    main()
