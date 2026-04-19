"""ORB correlation dedup — family + super-group membership.

Pure data module. Two levels of correlation dedup applied per trading day:

  LEVEL 1 — FAMILY dedup (underlying-based):
    Keep at most 1 pick per underlying family. Example: TSLA leveraged ETFs
    (TSLL, TSLT, TSLZ, ...) all move on TSLA → treat as one correlated bet.

  LEVEL 2 — SUPER-GROUP dedup (directional):
    Keep at most 1 pick per directional super-group. Example: UVXY + TSLZ +
    MSTZ are DIFFERENT underlying families but ALL bet on "market goes down"
    (all leveraged-short names). Pile-up risk is real — we saw it on 2025-03-06
    and 2025-03-18 in research (3 lev_short picks, all lost together).

Reuse: `symbol_family(sym)` and `symbol_super_group(sym)` return None if the
symbol isn't classified — those are treated as unconstrained (no dedup).

Validated 2026-04-18 through defended pipeline backtest.
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# FAMILIES — underlying-based correlation groups
# ---------------------------------------------------------------------------
# Each family = symbols that move on the same underlying. Leveraged ETFs on
# the same asset have correlation ~0.95+. Long + short on same underlying
# are anticorrelated but both FIRE on volatility — take only one at a time.

FAMILIES: Dict[str, List[str]] = {
    # Tesla-related leveraged ETFs
    'tsla_leveraged': ['TSL', 'TSLL', 'TSLR', 'TSLT', 'TSLG', 'TSLS', 'TSLZ', 'TSLQ'],
    # MicroStrategy-related leveraged ETFs
    'mstr_leveraged': ['MSTU', 'MSTX', 'MSTZ', 'SMST', 'MSTP', 'MSTW', 'MSTY', 'FMST'],
    # NVIDIA-related leveraged ETFs
    'nvda_leveraged': ['NVD', 'NVDD', 'NVDQ', 'NVDS', 'NVDX', 'NVDL', 'NVDU'],
    # VIX / volatility products
    'volatility': ['UVXY', 'VXX', 'VIXY', 'SVIX', 'SVXY', 'VMIN', 'VXZ'],
    # SPY leveraged ETFs
    'spy_leveraged': ['SPXL', 'UPRO', 'SPXS', 'SPXU', 'SPDN', 'SH'],
    # QQQ leveraged ETFs
    'qqq_leveraged': ['TQQQ', 'QLD', 'SQQQ', 'PSQ', 'QID'],
    # Semiconductor leveraged ETFs
    'semi_leveraged': ['SOXL', 'SOXS', 'USD', 'SSG', 'MSOX'],
    # Ethereum / crypto leveraged ETFs
    'eth_leveraged': ['ETHT', 'ETHU', 'ETHD', 'ETHL'],
    # Bitcoin leveraged ETFs
    'btc_leveraged': ['BITU', 'BITX', 'BITI'],
    # Bitcoin miners (highly correlated with BTC price)
    'btc_miners': ['MARA', 'RIOT', 'CLSK', 'WULF', 'CIFR', 'BTDR', 'HUT', 'BITF',
                   'CORZ', 'IREN', 'HIVE', 'CAN', 'BTCS', 'GREE'],
    # Quantum computing pack
    'quantum': ['RGTI', 'QBTS', 'IONQ', 'QUBT', 'QMCO', 'QBTSW', 'RGTIW'],
    # Fintech / crypto small-caps
    'crypto_fintech': ['COIN', 'HOOD', 'BKKT', 'BTBT', 'MSTR'],
    # EV / flying car / robotaxi
    'ev_aerial': ['ACHR', 'EVTL', 'JOBY', 'EH', 'BLDE', 'RKLB'],
    # China small-caps (move together on China news)
    'china_smallcap': ['NVNI', 'IREX', 'PTIR', 'SIDU', 'JFIN', 'KC'],
}

# ---------------------------------------------------------------------------
# SUPER-GROUPS — directional correlation across underlyings
# ---------------------------------------------------------------------------
# Leveraged short ETFs all win when the broad market falls, regardless of
# which underlying. Same for leveraged longs. Taking multiple = effectively
# one correlated bet. Super-group dedup catches the 3/6/25 UVXY+TSLZ+SMST
# disaster (all 3 were lev_short, all lost on market rally).

LEVERAGED_SHORT_ALL: List[str] = [
    # Long vol = short market
    'UVXY', 'VXX', 'VIXY',
    # Short TSLA
    'TSLZ', 'TSLS', 'TSLQ',
    # Short MSTR
    'MSTZ', 'SMST',
    # Short NVDA
    'NVD', 'NVDD', 'NVDQ', 'NVDS',
    # Short semi
    'SOXS', 'SSG', 'MSOX',
    # Short QQQ
    'SQQQ', 'PSQ', 'QID',
    # Short SPY
    'SPXS', 'SPXU', 'SPDN', 'SH',
    # Short BTC
    'BITI',
    # Short ETH
    'ETHD',
    # Short biotech
    'LABD',
]

LEVERAGED_LONG_ALL: List[str] = [
    # Short vol = long market
    'SVIX', 'SVXY', 'VMIN',
    # Long TSLA
    'TSLL', 'TSLT', 'TSLR', 'TSLG',
    # Long MSTR
    'MSTU', 'MSTX',
    # Long NVDA
    'NVDL', 'NVDU', 'NVDX',
    # Long semi
    'SOXL', 'USD',
    # Long QQQ
    'TQQQ', 'QLD',
    # Long SPY
    'SPXL', 'UPRO',
    # Long BTC
    'BITU', 'BITX',
    # Long ETH
    'ETHT', 'ETHU', 'ETHL',
    # Long biotech
    'LABU',
]

_LEV_SHORT_SET = set(LEVERAGED_SHORT_ALL)
_LEV_LONG_SET = set(LEVERAGED_LONG_ALL)

# Pre-computed reverse lookup: symbol → family_id.
_SYM_TO_FAMILY: Dict[str, str] = {
    sym: fam_name for fam_name, syms in FAMILIES.items() for sym in syms
}


def symbol_family(symbol: str) -> Optional[str]:
    """Return the family_id for a symbol, or None if unclassified.

    Example:
        >>> symbol_family('TSLZ')
        'tsla_leveraged'
        >>> symbol_family('AAPL')
        None
    """
    return _SYM_TO_FAMILY.get(symbol)


def symbol_super_group(symbol: str) -> Optional[str]:
    """Return 'lev_short' / 'lev_long' / None.

    Example:
        >>> symbol_super_group('UVXY')
        'lev_short'
        >>> symbol_super_group('TSLL')
        'lev_long'
        >>> symbol_super_group('TSLA')
        None
    """
    if symbol in _LEV_SHORT_SET:
        return 'lev_short'
    if symbol in _LEV_LONG_SET:
        return 'lev_long'
    return None


def dedup_candidates(ranked_symbols: List[str],
                     max_keep: int,
                     by_family: bool = True,
                     by_super_group: bool = True) -> List[str]:
    """Walk a ranked list, keeping at most 1 per family AND 1 per super-group.

    Args:
        ranked_symbols: symbols in rank order (best first).
        max_keep: stop after this many kept.
        by_family: if True, apply family dedup.
        by_super_group: if True, apply super-group dedup.

    Returns:
        Pruned list in input order, up to max_keep items.

    Example:
        >>> dedup_candidates(['UVXY', 'TSLZ', 'AAPL', 'MSFT'], 3)
        ['UVXY', 'AAPL', 'MSFT']
        # UVXY wins; TSLZ dropped (same lev_short super-group as UVXY);
        # AAPL, MSFT unrestricted.
    """
    if max_keep <= 0:
        return []
    seen_fam = set()
    seen_sup = set()
    kept: List[str] = []
    for sym in ranked_symbols:
        fam = symbol_family(sym) if by_family else None
        sup = symbol_super_group(sym) if by_super_group else None
        if fam is not None and fam in seen_fam:
            continue
        if sup is not None and sup in seen_sup:
            continue
        if fam is not None:
            seen_fam.add(fam)
        if sup is not None:
            seen_sup.add(sup)
        kept.append(sym)
        if len(kept) >= max_keep:
            break
    return kept
