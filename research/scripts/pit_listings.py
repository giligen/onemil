#!/usr/bin/env python3
"""Point-in-time listing facts, from the Databento EQUS.SUMMARY `definition` feed.

Bought 2026-09-18 ($35.27, 2024-07 → 2026-09, 27 monthly parquets under
`data/research/databento/pit_definition/`). It exists to retire ONE class of
false positive that has already cost this program a TEST result:

  * test tickers  — `ZVZZT` and its five siblings are NASDAQ test symbols that
    trade synthetic prints. One of them was +46.8R on a single day and WAS the
    entire TEST "profit" of the F6 red-to-green book (H/F6_reconcile/REPORT.md).
  * survivorship  — a universe built from TODAY's symbol list silently drops
    every name that later delisted. This file lists what was listed THEN.
  * venue         — `listing_exchange` says which venue is the primary listing,
    the fact that caps any primary-venue feed (the LULD `status` study, O_halt).

Deliberately NOT a price source and NOT a tradability rule: it answers "did this
symbol exist on this date, what kind of instrument was it, and where was it
listed". Membership rules stay in each study's own pre-registration.

Usage
-----
    from research.scripts.pit_listings import PitListings
    pit = PitListings()
    pit.is_test_ticker('ZVZZT')            -> True
    pit.was_listed('SIEB', '2025-03-14')   -> True/False
    pit.listing_exchange('AAPL', '2025-03-14') -> 'XNAS'
    pit.listed_symbols('2025-03-14')       -> frozenset of raw symbols
    pit.common_stock_symbols('2025-03-14') -> frozenset (security_type 'C')

Every accessor is month-cached; a date outside the bought window raises rather
than guessing, because a silent empty universe is exactly the failure this file
was bought to prevent.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# NASDAQ test symbols: ZAZZT ZBZZT ZCZZT ZJZZT ZVZZT ZWZZT and any future sibling.
TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT')

DEFAULT_DIR = Path(__file__).resolve().parents[2] / 'data' / 'research' / 'databento' / 'pit_definition'


def is_test_ticker(symbol: str) -> bool:
    """True for a NASDAQ test symbol, which must never enter a research universe."""
    return bool(TEST_TICKER_RE.match(symbol or ''))


class PitListings:
    """Read-only accessor over the monthly point-in-time definition parquets."""

    def __init__(self, directory: Path | str = DEFAULT_DIR):
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f'point-in-time listings not found at {self.dir} — '
                'run the EQUS.SUMMARY definition pull before using this helper')
        self._months = sorted(p.stem.split('_')[1] for p in self.dir.glob('def_*.parquet'))
        if not self._months:
            raise FileNotFoundError(f'no def_YYYYMM.parquet under {self.dir}')

    @property
    def coverage(self) -> tuple[str, str]:
        """(first, last) month present, as 'YYYYMM' — the honest bounds of any claim."""
        return self._months[0], self._months[-1]

    def _month_key(self, date) -> str:
        key = pd.Timestamp(date).strftime('%Y%m')
        if key not in self._months:
            raise KeyError(
                f'{key} is outside the bought point-in-time window '
                f'{self._months[0]}..{self._months[-1]} — do not fall back to '
                'the present-day symbol list, that IS the survivorship bug')
        return key

    @lru_cache(maxsize=48)
    def _month(self, key: str) -> pd.DataFrame:
        """One month of definition records, deduped to the last record per symbol."""
        df = pd.read_parquet(self.dir / f'def_{key}.parquet')
        return df.drop_duplicates(subset='raw_symbol', keep='last')

    def listed_symbols(self, date) -> frozenset[str]:
        """Every symbol with a definition record in `date`'s month, test tickers removed."""
        df = self._month(self._month_key(date))
        return frozenset(s for s in df['raw_symbol'] if not is_test_ticker(s))

    def common_stock_symbols(self, date) -> frozenset[str]:
        """Listed symbols whose security_type is common stock ('C'), test tickers removed."""
        df = self._month(self._month_key(date))
        common = df[df['security_type'] == 'C']['raw_symbol']
        return frozenset(s for s in common if not is_test_ticker(s))

    def was_listed(self, symbol: str, date) -> bool:
        """True if `symbol` carried a definition record in `date`'s month."""
        return symbol in self.listed_symbols(date)

    def listing_exchange(self, symbol: str, date) -> str | None:
        """Primary listing venue MIC ('XNAS', 'XNYS', 'ARCX', 'BATS', ...) or None."""
        df = self._month(self._month_key(date))
        hit = df.loc[df['raw_symbol'] == symbol, 'exchange']
        return None if hit.empty else str(hit.iloc[0])

    def security_type(self, symbol: str, date) -> str | None:
        """Databento security_type ('C' common, 'Q'/'O'/'P' others) or None if unlisted."""
        df = self._month(self._month_key(date))
        hit = df.loc[df['raw_symbol'] == symbol, 'security_type']
        return None if hit.empty else str(hit.iloc[0])

    def screen(self, symbols, date) -> tuple[frozenset[str], dict[str, str]]:
        """Split `symbols` into (kept, {dropped: reason}) — the one call a study needs.

        Drops test tickers and names with no definition record on that date. Every
        drop carries its reason, because a silent universe shrink is the failure
        mode this helper exists to prevent (CLAUDE.md fallback-logging rule).
        """
        listed = self.listed_symbols(date)
        kept, dropped = set(), {}
        for s in symbols:
            if is_test_ticker(s):
                dropped[s] = 'test_ticker'
            elif s not in listed:
                dropped[s] = 'not_listed_on_date'
            else:
                kept.add(s)
        return frozenset(kept), dropped
