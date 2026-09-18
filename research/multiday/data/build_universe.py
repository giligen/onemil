#!/usr/bin/env python3
"""Multi-day DATA stage, step 1 — the tradeable universe.

Pulls every ACTIVE, tradable US equity asset from Alpaca (the same REST path as
`data_sources/alpaca_client.get_all_tradeable_assets`, read-only) and records the
fields the multi-day program needs:

  symbol, name, exchange, asset_class, marginable, shortable, easy_to_borrow,
  fractionable, status, kind (common | wrapper | etf_or_fund | excluded)

Two deliberate deviations from the live universe rule:
  * leveraged/inverse wrappers are KEPT (`exclude_leveraged=False`) — the ORB
    universe rule (2026-09-05 owner GO); they are flagged, so a family can drop
    them without a re-pull.
  * warrants / units / preferred / rights are dropped by the shared
    `AlpacaClient._is_common_stock` name rule, so the panel is common stock +
    funds only.

Test tickers (`ZVZZT` and siblings) are dropped via
`research.scripts.pit_listings.is_test_ticker` — standing rule (PLAN §1).

Output: research/multiday/data/universe.parquet  (SIC is added later by
`edgar_earnings.py`; this file is the seed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(REPO))
load_dotenv(REPO / '.env')

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetAssetsRequest  # noqa: E402
from alpaca.trading.enums import AssetClass, AssetStatus  # noqa: E402

from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from research.scripts.pit_listings import is_test_ticker  # noqa: E402

OUT = Path(__file__).resolve().parent / 'universe.parquet'

# Name keywords that mark a pooled vehicle rather than an operating company.
FUND_KEYWORDS = (
    'ETF', 'FUND', 'TRUST', 'INDEX', 'SHARES', 'PORTFOLIO', 'ETN',
    'SPDR', 'ISHARES', 'INVESCO', 'VANGUARD', 'PROSHARES', 'DIREXION',
    'GRANITESHARES', 'DEFIANCE', 'ROUNDHILL', 'YIELDMAX', 'TIDAL',
)


def classify(symbol: str, name: str) -> str:
    """Bucket an asset: 'excluded' (warrant/unit/pref/right), 'wrapper', 'fund', 'common'."""
    if not AlpacaClient._is_common_stock(symbol, name, exclude_leveraged=False):
        return 'excluded'
    if not AlpacaClient._is_common_stock(symbol, name, exclude_leveraged=True):
        return 'wrapper'
    upper = (name or '').upper()
    if any(k in upper for k in FUND_KEYWORDS):
        return 'fund'
    return 'common'


def main() -> None:
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_API_SECRET')
    if not key or not secret:
        raise SystemExit('ALPACA_API_KEY/SECRET missing from .env — cannot build the universe')

    client = TradingClient(key, secret, paper=False)
    assets = client.get_all_assets(GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
    print(f'active US equity assets returned: {len(assets)}', flush=True)

    # Every active asset, unfiltered — the like-for-like denominator for the
    # survivorship overlaps (a name absent HERE is truly gone from Alpaca; a name
    # absent from universe.parquet may merely be a warrant we filtered out).
    pd.DataFrame([{'symbol': a.symbol,
                   'name': a.name or '',
                   'exchange': a.exchange.value if a.exchange else '',
                   'tradable': bool(a.tradable)} for a in assets]).to_parquet(
        OUT.with_name('assets_all_active.parquet'), index=False)

    rows, n_untradable, n_test, n_excluded = [], 0, 0, 0
    for a in assets:
        if not a.tradable:
            n_untradable += 1
            continue
        sym = a.symbol
        if is_test_ticker(sym):
            n_test += 1
            continue
        kind = classify(sym, a.name or '')
        if kind == 'excluded':
            n_excluded += 1
            continue
        rows.append({
            'symbol': sym,
            'name': a.name or '',
            'exchange': a.exchange.value if a.exchange else '',
            'kind': kind,
            'marginable': bool(getattr(a, 'marginable', False)),
            'shortable': bool(getattr(a, 'shortable', False)),
            'easy_to_borrow': bool(getattr(a, 'easy_to_borrow', False)),
            'fractionable': bool(getattr(a, 'fractionable', False)),
        })

    df = pd.DataFrame(rows).sort_values('symbol').reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    print(f'dropped: untradable={n_untradable} test_ticker={n_test} '
          f'warrant/unit/pref/right={n_excluded}', flush=True)
    print(f'kept {len(df)} symbols -> {OUT}', flush=True)
    print('\nby exchange:\n', df['exchange'].value_counts().to_string(), flush=True)
    print('\nby kind:\n', df['kind'].value_counts().to_string(), flush=True)
    print('\nflags:\n', df[['marginable', 'shortable', 'easy_to_borrow']].sum().to_string(),
          flush=True)


if __name__ == '__main__':
    main()
