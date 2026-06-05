"""
Unit tests for trading.reject_categorizer.

Categorization is pattern-matched. Pinning the substring → category map here
prevents drift; if a needle is removed accidentally, the regression hits a
test before it ships.
"""

import pytest

from trading.reject_categorizer import (
    RejectCategory,
    categorize_reject,
    is_account_level,
)


class TestStatusShortCircuit:
    def test_non_rejected_order_returns_none(self):
        assert categorize_reject("anything", status="filled") == RejectCategory.NONE
        assert categorize_reject("anything", status="canceled") == RejectCategory.NONE

    def test_no_status_uses_string_match(self):
        # When status not passed, we trust reject_reason as-is
        assert categorize_reject("wash trade detected") == RejectCategory.WASH_TRADE


class TestUnknownAndNone:
    def test_none_reject_reason_is_unknown(self):
        """The 4ms `stop ≤ ask` rejection in our incident logs had reject_reason=None.
        Until we have evidence to categorize that case, it's UNKNOWN. (The
        buy_stop_guard fix means this should rarely fire in production now.)"""
        assert categorize_reject(None, status="rejected") == RejectCategory.UNKNOWN

    def test_empty_string_reject_reason_is_unknown(self):
        assert categorize_reject("", status="rejected") == RejectCategory.UNKNOWN
        assert categorize_reject("   ", status="rejected") == RejectCategory.UNKNOWN

    def test_novel_reject_string_is_unknown(self):
        """Unrecognized text → UNKNOWN. This is how new Alpaca error variants
        surface for human review."""
        assert categorize_reject(
            "some new error Alpaca added in 2027", status="rejected"
        ) == RejectCategory.UNKNOWN


class TestMarginDeficit:
    """Margin-deficit rejections are the new failure mode under Alpaca's
    intraday margin framework (FINRA retired PDT 2026)."""

    @pytest.mark.parametrize("text", [
        "Insufficient buying power",
        "INSUFFICIENT MARGIN",
        "margin call issued",
        "intraday margin deficit",
        "Buying power deficit: required $50K, available $30K",
        "DTBP exceeded",
    ])
    def test_margin_strings_classify(self, text):
        assert categorize_reject(text, status="rejected") == RejectCategory.MARGIN_DEFICIT

    def test_margin_is_account_level(self):
        assert is_account_level(RejectCategory.MARGIN_DEFICIT) is True


class TestStopPriceInvalid:
    @pytest.mark.parametrize("text", [
        "Stop price must be above the current bid",
        "Order is immediately marketable",
    ])
    def test_stop_strings_classify(self, text):
        assert categorize_reject(text, status="rejected") == RejectCategory.STOP_PRICE_INVALID


class TestInsufficientQty:
    @pytest.mark.parametrize("text", [
        "insufficient qty available for order",
        "qty=947, held_for_orders=947, available=0",
    ])
    def test_qty_strings_classify(self, text):
        assert categorize_reject(text, status="rejected") == RejectCategory.INSUFFICIENT_QTY


class TestWashTrade:
    def test_wash_trade(self):
        assert categorize_reject("wash trade rejected by SOR", status="rejected") == RejectCategory.WASH_TRADE

    def test_wash_sale(self):
        assert categorize_reject("Order rejected: wash sale", status="rejected") == RejectCategory.WASH_TRADE


class TestSymbolHalted:
    def test_halt(self):
        assert categorize_reject("Symbol is halted", status="rejected") == RejectCategory.SYMBOL_HALTED

    def test_trading_halt(self):
        assert categorize_reject("trading halt LULD", status="rejected") == RejectCategory.SYMBOL_HALTED


class TestNotTradable:
    @pytest.mark.parametrize("text", [
        "asset is not tradable",
        "Symbol delisted",
        "restricted list violation",
    ])
    def test_not_tradable_strings(self, text):
        assert categorize_reject(text, status="rejected") == RejectCategory.NOT_TRADABLE


class TestAccountBlocked:
    @pytest.mark.parametrize("text", [
        "Account blocked from trading",
        "trading blocked due to compliance review",
    ])
    def test_account_blocked_classifies(self, text):
        assert categorize_reject(text, status="rejected") == RejectCategory.ACCOUNT_BLOCKED

    def test_account_blocked_is_account_level(self):
        assert is_account_level(RejectCategory.ACCOUNT_BLOCKED) is True


class TestAccountLevelHelper:
    def test_non_account_categories_are_not_account_level(self):
        for cat in (
            RejectCategory.STOP_PRICE_INVALID,
            RejectCategory.INSUFFICIENT_QTY,
            RejectCategory.WASH_TRADE,
            RejectCategory.SYMBOL_HALTED,
            RejectCategory.NOT_TRADABLE,
            RejectCategory.UNKNOWN,
            RejectCategory.NONE,
        ):
            assert is_account_level(cat) is False


class TestCaseInsensitivity:
    def test_uppercase_strings_match(self):
        assert categorize_reject(
            "INSUFFICIENT BUYING POWER", status="rejected"
        ) == RejectCategory.MARGIN_DEFICIT

    def test_mixed_case_strings_match(self):
        assert categorize_reject(
            "Wash Trade Detected", status="rejected"
        ) == RejectCategory.WASH_TRADE
