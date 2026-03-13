"""
Tests for scanner/criteria.py - ScanCandidate and ScannerCriteria.

Covers:
- ScanCandidate dataclass properties
- Pre-market evaluation (gap + news)
- Intraday evaluation (all 6 criteria)
- Candidate formatting for premarket and intraday
- Close call detection (5 of 6 criteria met)
"""

import pytest

from scanner.criteria import ScanCandidate, ScannerCriteria


# =============================================================================
# ScanCandidate dataclass
# =============================================================================

class TestScanCandidate:
    """Tests for ScanCandidate dataclass properties."""

    def test_criteria_met_count_empty(self):
        """criteria_met_count returns 0 when no criteria have been evaluated."""
        candidate = ScanCandidate(symbol="TEST")
        assert candidate.criteria_met_count == 0

    def test_criteria_met_count_partial(self):
        """criteria_met_count counts only True values in criteria_met dict."""
        candidate = ScanCandidate(symbol="TEST")
        candidate.criteria_met = {'a': True, 'b': False, 'c': True}
        assert candidate.criteria_met_count == 2

    def test_criteria_met_count_all_true(self):
        """criteria_met_count returns total when all criteria are True."""
        candidate = ScanCandidate(symbol="TEST")
        candidate.criteria_met = {'a': True, 'b': True, 'c': True}
        assert candidate.criteria_met_count == 3

    def test_total_criteria_empty(self):
        """total_criteria returns 0 when no criteria have been evaluated."""
        candidate = ScanCandidate(symbol="TEST")
        assert candidate.total_criteria == 0

    def test_total_criteria_counts_all(self):
        """total_criteria returns count of all keys regardless of True/False."""
        candidate = ScanCandidate(symbol="TEST")
        candidate.criteria_met = {'a': True, 'b': False, 'c': True, 'd': False}
        assert candidate.total_criteria == 4

    def test_defaults(self):
        """ScanCandidate defaults are sensible zero/empty values."""
        candidate = ScanCandidate(symbol="XYZ")
        assert candidate.company_name == ''
        assert candidate.prev_close == 0.0
        assert candidate.current_price == 0.0
        assert candidate.float_shares == 0
        assert candidate.gap_pct == 0.0
        assert candidate.has_news is False
        assert candidate.news_headline is None
        assert candidate.criteria_met == {}


# =============================================================================
# evaluate_premarket
# =============================================================================

class TestEvaluatePremarket:
    """Tests for ScannerCriteria.evaluate_premarket."""

    def setup_method(self):
        """Create criteria with default thresholds."""
        self.criteria = ScannerCriteria()

    def test_qualified_gap_and_news(self):
        """Stock with gap >= 2% and news qualifies in premarket."""
        candidate = ScanCandidate(
            symbol="AAPL",
            gap_pct=5.0,
            has_news=True,
            news_headline="Big announcement",
        )
        assert self.criteria.evaluate_premarket(candidate) is True
        assert candidate.criteria_met['gap'] is True
        assert candidate.criteria_met['has_news'] is True

    def test_not_qualified_no_gap(self):
        """Stock with gap < 2% does not qualify even with news."""
        candidate = ScanCandidate(
            symbol="AAPL",
            gap_pct=1.0,
            has_news=True,
            news_headline="Some news",
        )
        assert self.criteria.evaluate_premarket(candidate) is False
        assert candidate.criteria_met['gap'] is False
        assert candidate.criteria_met['has_news'] is True

    def test_not_qualified_no_news(self):
        """Stock with gap >= 2% but no news does not qualify."""
        candidate = ScanCandidate(
            symbol="AAPL",
            gap_pct=5.0,
            has_news=False,
        )
        assert self.criteria.evaluate_premarket(candidate) is False
        assert candidate.criteria_met['gap'] is True
        assert candidate.criteria_met['has_news'] is False

    def test_not_qualified_neither(self):
        """Stock with no gap and no news does not qualify."""
        candidate = ScanCandidate(
            symbol="AAPL",
            gap_pct=0.5,
            has_news=False,
        )
        assert self.criteria.evaluate_premarket(candidate) is False

    def test_gap_at_exact_threshold(self):
        """Stock with gap exactly at threshold qualifies (>=)."""
        candidate = ScanCandidate(
            symbol="AAPL",
            gap_pct=2.0,
            has_news=True,
            news_headline="News",
        )
        assert self.criteria.evaluate_premarket(candidate) is True


# =============================================================================
# evaluate_intraday
# =============================================================================

class TestEvaluateIntraday:
    """Tests for ScannerCriteria.evaluate_intraday."""

    def setup_method(self):
        """Create criteria with default thresholds."""
        self.criteria = ScannerCriteria()

    def _make_qualified_candidate(self, **overrides) -> ScanCandidate:
        """Create a candidate that meets all intraday criteria by default."""
        defaults = dict(
            symbol="MOMO",
            current_price=5.0,
            float_shares=5_000_000,
            gap_pct=3.0,
            relative_volume=6.0,
            intraday_change_pct=12.0,
            has_news=True,
            news_headline="Catalyst",
        )
        defaults.update(overrides)
        return ScanCandidate(**defaults)

    def test_all_criteria_met(self):
        """Stock meeting all 6 criteria qualifies intraday."""
        candidate = self._make_qualified_candidate()
        assert self.criteria.evaluate_intraday(candidate) is True
        assert candidate.criteria_met_count == 6
        assert candidate.total_criteria == 6

    def test_fail_price_too_low(self):
        """Stock below $2 fails price_range criterion."""
        candidate = self._make_qualified_candidate(current_price=1.50)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['price_range'] is False

    def test_fail_price_too_high(self):
        """Stock above $20 fails price_range criterion."""
        candidate = self._make_qualified_candidate(current_price=25.0)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['price_range'] is False

    def test_fail_float_too_large(self):
        """Stock with float > 10M fails float criterion."""
        candidate = self._make_qualified_candidate(float_shares=15_000_000)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['float'] is False

    def test_fail_gap_too_small(self):
        """Stock with gap < 2% fails gap criterion."""
        candidate = self._make_qualified_candidate(gap_pct=1.0)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['gap'] is False

    def test_fail_relative_volume_too_low(self):
        """Stock with relative volume < 5x fails relative_volume criterion."""
        candidate = self._make_qualified_candidate(relative_volume=3.0)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['relative_volume'] is False

    def test_fail_intraday_change_too_small(self):
        """Stock with intraday change < 10% fails intraday_change criterion."""
        candidate = self._make_qualified_candidate(intraday_change_pct=5.0)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['intraday_change'] is False

    def test_fail_no_news(self):
        """Stock without news fails has_news criterion."""
        candidate = self._make_qualified_candidate(has_news=False)
        assert self.criteria.evaluate_intraday(candidate) is False
        assert candidate.criteria_met['has_news'] is False


# =============================================================================
# Close call detection (5 of 6 criteria)
# =============================================================================

class TestCloseCallDetection:
    """Tests for close call detection in evaluate_intraday."""

    def setup_method(self):
        """Create criteria with default thresholds."""
        self.criteria = ScannerCriteria()

    def test_close_call_one_criterion_failing(self):
        """Stock with 5 of 6 criteria met is a close call."""
        candidate = ScanCandidate(
            symbol="NEAR",
            current_price=5.0,
            float_shares=5_000_000,
            gap_pct=3.0,
            relative_volume=6.0,
            intraday_change_pct=12.0,
            has_news=False,  # Only this one fails
        )
        result = self.criteria.evaluate_intraday(candidate)
        assert result is False
        assert candidate.criteria_met_count == 5
        assert candidate.total_criteria == 6
        # Verify it's exactly one short
        assert candidate.criteria_met_count >= candidate.total_criteria - 1

    def test_not_close_call_two_failing(self):
        """Stock with 4 of 6 criteria is NOT a close call."""
        candidate = ScanCandidate(
            symbol="FAR",
            current_price=5.0,
            float_shares=5_000_000,
            gap_pct=1.0,       # Fails
            relative_volume=6.0,
            intraday_change_pct=12.0,
            has_news=False,    # Fails
        )
        result = self.criteria.evaluate_intraday(candidate)
        assert result is False
        assert candidate.criteria_met_count == 4
        assert candidate.criteria_met_count < candidate.total_criteria - 1


# =============================================================================
# format_candidate
# =============================================================================

class TestFormatCandidate:
    """Tests for ScannerCriteria.format_candidate."""

    def setup_method(self):
        """Create criteria instance."""
        self.criteria = ScannerCriteria()

    def test_format_premarket(self):
        """Premarket format includes symbol, gap, price, float, and news."""
        candidate = ScanCandidate(
            symbol="AAPL",
            company_name="Apple Inc",
            gap_pct=5.5,
            current_price=10.50,
            float_shares=3_000_000,
            news_headline="FDA approval",
        )
        output = self.criteria.format_candidate(candidate, 'premarket')
        assert "AAPL" in output
        assert "+5.5%" in output
        assert "$10.50" in output
        assert "3.0M" in output
        assert "FDA approval" in output

    def test_format_intraday(self):
        """Intraday format includes symbol, prices, change, relvol, float, and news."""
        candidate = ScanCandidate(
            symbol="MOMO",
            prev_close=4.00,
            current_price=5.00,
            intraday_change_pct=25.0,
            relative_volume=8.5,
            float_shares=2_000_000,
            news_headline="Earnings beat",
        )
        output = self.criteria.format_candidate(candidate, 'intraday')
        assert "MOMO" in output
        assert "$4.00" in output
        assert "$5.00" in output
        assert "+25.0%" in output
        assert "8.5x" in output
        assert "2.0M" in output
        assert "Earnings beat" in output

    def test_format_premarket_no_headline(self):
        """Premarket format shows N/A when news_headline is None."""
        candidate = ScanCandidate(
            symbol="XYZ",
            company_name="XYZ Corp",
            gap_pct=3.0,
            current_price=8.00,
            float_shares=1_000_000,
            news_headline=None,
        )
        output = self.criteria.format_candidate(candidate, 'premarket')
        assert "N/A" in output

    def test_format_intraday_no_headline(self):
        """Intraday format shows N/A when news_headline is None."""
        candidate = ScanCandidate(
            symbol="XYZ",
            prev_close=5.0,
            current_price=6.0,
            intraday_change_pct=20.0,
            relative_volume=7.0,
            float_shares=1_000_000,
            news_headline=None,
        )
        output = self.criteria.format_candidate(candidate, 'intraday')
        assert "N/A" in output
