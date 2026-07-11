"""Asset-class rule for the ORB news gate — shared by BT and live.

Owner mandate 2026-07-11 ("we don't trade on 'accidentally', we trade
deliberately based on rules"): 45% of the ORB qualified universe are
leveraged single-stock wrappers (Tradr/T-Rex/Direxion/ProShares/
GraniteShares 2x products). They have no company events; Benzinga tags
their news to the UNDERLYING. The news×PM$ sizing edge is a COMMON-STOCK
edge (fresh catalyst being priced). For wrappers, every news window was
tested and none works: same-morning underlying news marks late-retail
crowding (NEGATIVE all 3 eras: −$324/−$125/−$27 per trade), prev-day
underlying news doesn't carry (−$129/−$177), and wrapper monsters are
mostly newsless momentum (8/11).

RULE: the news boost requires POSITIVE identification as a common stock.
Wrappers and unknown-class symbols are structurally ineligible — this is
deliberate, not an artifact of vendor tagging. Without this rule, a
Benzinga tagging change (starting to tag wrappers) would silently flip
the gate into 2x-boosting the proven-negative crowding cell.

Evidence: research/orb_news_catalyst_jul2026.md (+ 2026-07-11 addendum),
research/orb_machine_rules.md §L6. Cost of the rule vs the accidental
gate: one lucky wrapper trade in 18 months (MSTZ 2026-06-09, −$3,049
book impact) — the price of deliberateness.

Classification order (live): leveraged-family sets (orb_correlation,
no I/O) → offline class map CSV (14K names, 2026-07-11 dump) → asset-name
fetch + regex → 'unknown'. Unknown NEVER boosts (upsize-only doctrine:
never boost blind).
"""
from __future__ import annotations

import csv
import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STOCK = 'stock'
WRAPPER = 'wrapper'
UNKNOWN = 'unknown'

# Fund-brand + structure tokens that mark leveraged/inverse wrappers and
# other no-company-event products. Built from the 2026-07-11 full asset
# dump study (614 wrapper candidates classified, spot-verified).
WRAPPER_RE = re.compile(
    r'\bETF\b|\bETN\b|ProShares|Direxion|GraniteShares|T-?REX|Tradr|'
    r'Defiance|Volatility Shares|Leverage Shares|MicroSectors|Kurv|'
    r'YieldMax|Daily Target|\b[123](?:\.\d+)?X\b|\bInverse\b|'
    r'UltraShort|UltraPro|\bUltra\b|\bBull\b|\bBear\b',
    re.I)

DEFAULT_CLASS_MAP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'research', 'orb_asset_class_map_20260711.csv')

_KNOWN_WRAPPERS: Optional[set] = None


def _lev_family_symbols() -> set:
    """The 91 leveraged symbols orb_correlation already curates (fast
    path, no I/O, covers the MSTU/MSTZ/PLTU/... complex)."""
    global _KNOWN_WRAPPERS
    if _KNOWN_WRAPPERS is None:
        try:
            from trading.orb_correlation import (
                LEVERAGED_LONG_ALL, LEVERAGED_SHORT_ALL)
            _KNOWN_WRAPPERS = set(LEVERAGED_LONG_ALL) | set(LEVERAGED_SHORT_ALL)
        except Exception as e:
            logger.warning(f"orb_correlation lev sets unavailable ({e}) — "
                           f"classifier falls back to name/map only")
            _KNOWN_WRAPPERS = set()
    return _KNOWN_WRAPPERS


def classify_asset(symbol: str, name: Optional[str]) -> str:
    """'stock' | 'wrapper' | 'unknown' from symbol + asset name.

    Empty/missing name → 'unknown' (NOT eligible for the news boost —
    positive identification required, never boost blind).
    """
    if symbol in _lev_family_symbols():
        return WRAPPER
    if not name or not str(name).strip():
        return UNKNOWN
    return WRAPPER if WRAPPER_RE.search(str(name)) else STOCK


def load_class_map(path: str = DEFAULT_CLASS_MAP) -> Dict[str, str]:
    """Offline symbol→class map (2026-07-11 asset dump). Missing file →
    empty map + WARNING (live falls through to the API fetch path)."""
    out: Dict[str, str] = {}
    try:
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                out[row['symbol']] = row['asset_class']
    except FileNotFoundError:
        logger.warning(f"asset class map missing at {path} — live will "
                       f"classify via API name fetch only")
    except Exception as e:
        logger.warning(f"asset class map unreadable ({e}) — live will "
                       f"classify via API name fetch only")
    return out


def effective_has_news(has_news: Optional[bool],
                       asset_class: str) -> Optional[bool]:
    """The deliberate gate input: news counts ONLY for identified common
    stocks. Wrapper/unknown → False when newsy (structurally out of
    scope), tri-state None (fetch failed) passes through unchanged."""
    if has_news is None:
        return None
    if has_news and asset_class != STOCK:
        return False
    return has_news
