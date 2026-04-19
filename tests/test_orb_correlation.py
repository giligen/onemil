"""Unit tests for trading/orb_correlation.py."""
import pytest

from trading.orb_correlation import (
    FAMILIES, LEVERAGED_SHORT_ALL, LEVERAGED_LONG_ALL,
    symbol_family, symbol_super_group, dedup_candidates,
)


# =========================================================================
# symbol_family
# =========================================================================

def test_symbol_family_tsla_leveraged():
    for sym in ['TSLL', 'TSLT', 'TSLZ', 'TSLQ']:
        assert symbol_family(sym) == 'tsla_leveraged', f"{sym} should be tsla_leveraged"


def test_symbol_family_mstr_leveraged():
    for sym in ['MSTU', 'MSTZ', 'SMST']:
        assert symbol_family(sym) == 'mstr_leveraged'


def test_symbol_family_volatility():
    for sym in ['UVXY', 'VXX', 'VIXY']:
        assert symbol_family(sym) == 'volatility'


def test_symbol_family_btc_miners():
    for sym in ['MARA', 'RIOT', 'CLSK', 'WULF', 'CIFR']:
        assert symbol_family(sym) == 'btc_miners'


def test_symbol_family_quantum():
    for sym in ['RGTI', 'QBTS', 'IONQ', 'QUBT']:
        assert symbol_family(sym) == 'quantum'


def test_symbol_family_unclassified():
    for sym in ['AAPL', 'GOOG', 'RANDOM_TICKER', 'TSLA']:
        assert symbol_family(sym) is None, f"{sym} unexpectedly classified"


def test_symbol_family_empty_string():
    assert symbol_family('') is None


# =========================================================================
# symbol_super_group
# =========================================================================

def test_symbol_super_group_lev_short():
    for sym in ['UVXY', 'TSLZ', 'MSTZ', 'SQQQ', 'SPXS', 'BITI']:
        assert symbol_super_group(sym) == 'lev_short', f"{sym} should be lev_short"


def test_symbol_super_group_lev_long():
    for sym in ['TSLL', 'MSTU', 'TQQQ', 'SPXL', 'BITU', 'ETHT']:
        assert symbol_super_group(sym) == 'lev_long', f"{sym} should be lev_long"


def test_symbol_super_group_unclassified():
    for sym in ['AAPL', 'TSLA', 'MARA', 'RGTI']:
        assert symbol_super_group(sym) is None


def test_super_groups_are_disjoint():
    """Sanity: no symbol can be both lev_short and lev_long."""
    shorts = set(LEVERAGED_SHORT_ALL)
    longs = set(LEVERAGED_LONG_ALL)
    assert not shorts & longs, f"Symbols in both: {shorts & longs}"


# =========================================================================
# dedup_candidates — the heart of the correlation filter
# =========================================================================

def test_dedup_keeps_first_pick_from_family():
    # Two TSLA leveraged → only first kept
    result = dedup_candidates(['TSLL', 'TSLZ', 'AAPL'], max_keep=4)
    assert result == ['TSLL', 'AAPL']


def test_dedup_keeps_first_pick_from_super_group():
    # UVXY (vol family + lev_short super) and TSLZ (tsla family + lev_short super)
    # are DIFFERENT families but SAME super-group → only first kept
    result = dedup_candidates(['UVXY', 'TSLZ', 'AAPL'], max_keep=4)
    assert result == ['UVXY', 'AAPL']


def test_dedup_mixed_long_and_short_allowed():
    # lev_short and lev_long are different super-groups — both OK together
    result = dedup_candidates(['UVXY', 'TSLL', 'AAPL'], max_keep=4)
    assert result == ['UVXY', 'TSLL', 'AAPL']


def test_dedup_unclassified_never_blocked():
    # AAPL, MSFT, GOOG all unclassified — all kept
    result = dedup_candidates(['AAPL', 'MSFT', 'GOOG', 'AMZN'], max_keep=4)
    assert result == ['AAPL', 'MSFT', 'GOOG', 'AMZN']


def test_dedup_respects_max_keep():
    result = dedup_candidates(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'], max_keep=3)
    assert result == ['AAPL', 'MSFT', 'GOOG']


def test_dedup_disable_family():
    # by_family=False → two TSLA picks both kept
    result = dedup_candidates(['TSLL', 'TSLT'], max_keep=4, by_family=False, by_super_group=False)
    assert result == ['TSLL', 'TSLT']


def test_dedup_disable_super_group_only():
    # Super-group off, family on → UVXY and TSLZ both kept (different families)
    result = dedup_candidates(['UVXY', 'TSLZ'], max_keep=4, by_super_group=False)
    assert result == ['UVXY', 'TSLZ']


def test_dedup_disable_family_only():
    # Family off, super-group on → UVXY kept, TSLZ blocked (both lev_short)
    result = dedup_candidates(['UVXY', 'TSLZ'], max_keep=4, by_family=False)
    assert result == ['UVXY']


def test_dedup_empty_input():
    assert dedup_candidates([], max_keep=4) == []


def test_dedup_zero_max_keep():
    assert dedup_candidates(['AAPL'], max_keep=0) == []


def test_dedup_historical_disaster_day_20250306():
    """Simulate 2025-03-06: UVXY + TSLZ + SMST → dedup keeps just UVXY."""
    # All three were Q5 picks that lost together in research
    ranked = ['UVXY', 'TSLZ', 'SMST']  # highest composite first
    result = dedup_candidates(ranked, max_keep=4)
    # All three are lev_short → keep only first
    assert result == ['UVXY']


def test_dedup_historical_disaster_day_20250318():
    """Simulate 2025-03-18: MSTZ + SMST → both mstr_leveraged family."""
    result = dedup_candidates(['MSTZ', 'SMST'], max_keep=4)
    # Both mstr_leveraged family → keep only first
    assert result == ['MSTZ']


def test_dedup_btc_miners_pack():
    # Multiple BTC miners (family only, different super-groups) — keep one
    result = dedup_candidates(['MARA', 'RIOT', 'CLSK'], max_keep=4)
    assert result == ['MARA']


def test_dedup_preserves_order_when_no_collisions():
    result = dedup_candidates(['A', 'B', 'C', 'D'], max_keep=10)
    assert result == ['A', 'B', 'C', 'D']
