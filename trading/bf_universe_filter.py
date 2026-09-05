"""Bull-flag universe eligibility for the BACKTEST — the live rule, by name.

Live BF trades only what `UniverseBuilder` Step 1 admits: Alpaca assets that
pass `AlpacaClient._is_common_stock(symbol, name)` (no warrants, units,
preferred, rights, leveraged/inverse wrappers). The BF backtest's Stage-2
used a hand-kept SYMBOL list for the leveraged exclusion, which was enough
while `daily_bars` itself excluded new wrappers. Since the 2026-09-05 ORB
universe rule (wrappers IN for ORB/ignition), `daily_bars` carries every
leveraged wrapper, and a `BT_BUILD_FULL_HISTORY=1` BF regen would scan
them — so Stage-2 must apply the SAME name-based predicate live uses.

Names come from the offline asset dumps (no API in the backtest):
  data/research/alpaca_assets_all_20260905.csv   (full Alpaca dump, 9/5)
  data/research/orb_asset_class_map_20260711.csv (7/11 dump)
A symbol with no known name is kept unless it is on the legacy symbol list
(cannot judge a name we do not have — logged once at WARNING).
"""
import csv
import logging
import os
from typing import Dict, Optional, Set

from data_sources.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME_SOURCES = (
    os.path.join(ROOT, 'data', 'research', 'alpaca_assets_all_20260905.csv'),
    os.path.join(ROOT, 'data', 'research', 'orb_asset_class_map_20260711.csv'),
)
_NAMES: Optional[Dict[str, str]] = None


def load_names(paths=NAME_SOURCES) -> Dict[str, str]:
    """symbol -> asset name from the offline dumps (first source wins)."""
    global _NAMES
    if _NAMES is not None:
        return _NAMES
    names: Dict[str, str] = {}
    for p in paths:
        try:
            with open(p, newline='') as fh:
                for row in csv.DictReader(fh):
                    s, n = row.get('symbol'), row.get('name')
                    if s and n and s not in names:
                        names[s] = n
        except FileNotFoundError:
            logger.warning(f"bf_universe_filter: name source missing: {p}")
    _NAMES = names
    return names


def is_bf_eligible(symbol: str, names: Optional[Dict[str, str]] = None) -> bool:
    """True if live BF could trade this symbol (common stock, not a wrapper)."""
    names = names if names is not None else load_names()
    name = names.get(symbol)
    if name is None:
        return symbol not in AlpacaClient._LEVERAGED_ETF_SYMBOLS
    return AlpacaClient._is_common_stock(symbol, name)


def filter_trades(trades, names: Optional[Dict[str, str]] = None):
    """Drop cached BT trades on symbols live BF would never trade; log the count."""
    names = names if names is not None else load_names()
    unknown: Set[str] = set()
    kept = []
    for t in trades:
        s = t['symbol']
        if s not in names:
            unknown.add(s)
        if is_bf_eligible(s, names):
            kept.append(t)
    removed = len(trades) - len(kept)
    if removed:
        logger.info(f"BF universe filter (live rule, by name): {len(trades)} → {len(kept)} trades ({removed} removed)")
    if unknown:
        logger.warning(f"BF universe filter: {len(unknown)} symbols with no known name kept by symbol-list only "
                       f"(e.g. {sorted(unknown)[:5]}) — refresh data/research/alpaca_assets_all_*.csv")
    return kept
