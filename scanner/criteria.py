"""
Qualification rules engine for the stock scanner.

Defines the criteria a stock must meet to qualify as a momentum candidate.
Separates pre-market (gap) criteria from intraday (volume + move) criteria.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ScanCandidate:
    """A stock being evaluated by the scanner."""
    symbol: str
    company_name: str = ''
    prev_close: float = 0.0
    current_price: float = 0.0
    float_shares: int = 0
    gap_pct: float = 0.0
    intraday_change_pct: float = 0.0
    relative_volume: float = 0.0
    current_volume: int = 0
    time_bucket: Optional[str] = None
    has_news: bool = False
    news_headline: Optional[str] = None
    criteria_met: Dict[str, bool] = field(default_factory=dict)

    @property
    def criteria_met_count(self) -> int:
        """Count of criteria that are met."""
        return sum(1 for v in self.criteria_met.values() if v)

    @property
    def total_criteria(self) -> int:
        """Total number of criteria evaluated."""
        return len(self.criteria_met)


class ScannerCriteria:
    """
    Rules engine for stock qualification.

    Pre-market criteria:
    - Price $2-$20 (universe filter)
    - Float <= 10M (universe filter)
    - Gap >= 2%
    - Has at least 1 interesting news article

    Intraday criteria (in addition to pre-market):
    - Relative volume >= 5x (bucket vol / 50d avg)
    - Intraday change >= 10%
    """

    def __init__(
        self,
        price_min: float = 2.0,
        price_max: float = 20.0,
        float_max: int = 10_000_000,
        gap_pct_min: float = 2.0,
        intraday_change_pct_min: float = 10.0,
        relative_volume_min: float = 5.0,
    ):
        """
        Initialize criteria thresholds.

        Args:
            price_min: Minimum price
            price_max: Maximum price
            float_max: Maximum float shares
            gap_pct_min: Minimum pre-market gap percentage
            intraday_change_pct_min: Minimum intraday price change percentage
            relative_volume_min: Minimum relative volume ratio
        """
        self.price_min = price_min
        self.price_max = price_max
        self.float_max = float_max
        self.gap_pct_min = gap_pct_min
        self.intraday_change_pct_min = intraday_change_pct_min
        self.relative_volume_min = relative_volume_min

    def evaluate_premarket(self, candidate: ScanCandidate) -> bool:
        """
        Evaluate pre-market gap criteria.

        A stock qualifies in pre-market if:
        - Gap >= threshold
        - Has interesting news

        Price and float are already filtered by universe.

        Args:
            candidate: ScanCandidate to evaluate

        Returns:
            True if candidate qualifies as pre-market gap-up
        """
        candidate.criteria_met['gap'] = candidate.gap_pct >= self.gap_pct_min
        candidate.criteria_met['has_news'] = candidate.has_news

        qualified = all([
            candidate.criteria_met['gap'],
            candidate.criteria_met['has_news'],
        ])

        if qualified:
            logger.debug(
                f"PREMARKET QUALIFIED: {candidate.symbol} | "
                f"Gap: {candidate.gap_pct:.1f}% | "
                f"News: {candidate.news_headline}"
            )

        return qualified

    def evaluate_intraday(self, candidate: ScanCandidate) -> bool:
        """
        Evaluate full intraday criteria.

        A stock qualifies intraday when ALL are true:
        - Price $2-$20 (universe, re-checked with current price)
        - Float <= 10M (universe)
        - Gap >= 2% (from pre-market)
        - Relative volume >= 5x (current bucket)
        - Intraday change >= 10%
        - Has at least 1 interesting news article

        Args:
            candidate: ScanCandidate to evaluate

        Returns:
            True if candidate fully qualifies
        """
        candidate.criteria_met['price_range'] = (
            self.price_min <= candidate.current_price <= self.price_max
        )
        candidate.criteria_met['float'] = candidate.float_shares <= self.float_max
        candidate.criteria_met['gap'] = candidate.gap_pct >= self.gap_pct_min
        candidate.criteria_met['relative_volume'] = (
            candidate.relative_volume >= self.relative_volume_min
        )
        candidate.criteria_met['intraday_change'] = (
            candidate.intraday_change_pct >= self.intraday_change_pct_min
        )
        candidate.criteria_met['has_news'] = candidate.has_news

        qualified = all(candidate.criteria_met.values())

        if qualified:
            logger.info(
                f"INTRADAY QUALIFIED: {candidate.symbol} | "
                f"${candidate.current_price:.2f} ({candidate.intraday_change_pct:+.1f}%) | "
                f"RelVol: {candidate.relative_volume:.1f}x | "
                f"Float: {candidate.float_shares / 1_000_000:.1f}M | "
                f"News: {candidate.news_headline}"
            )
        elif candidate.criteria_met_count >= candidate.total_criteria - 1:
            logger.debug(
                f"CLOSE CALL: {candidate.symbol} | "
                f"Met {candidate.criteria_met_count}/{candidate.total_criteria} | "
                f"Missing: {[k for k, v in candidate.criteria_met.items() if not v]}"
            )

        return qualified

    def format_candidate(self, candidate: ScanCandidate, phase: str) -> str:
        """
        Format a candidate for console output.

        Args:
            candidate: ScanCandidate to format
            phase: 'premarket' or 'intraday'

        Returns:
            Formatted string for display
        """
        if phase == 'premarket':
            return (
                f"  {candidate.symbol:<6} | {candidate.company_name[:20]:<20} | "
                f"Gap: {candidate.gap_pct:+.1f}% | "
                f"Price: ${candidate.current_price:.2f} | "
                f"Float: {candidate.float_shares / 1_000_000:.1f}M | "
                f"News: {candidate.news_headline or 'N/A'}"
            )
        else:
            return (
                f"  {candidate.symbol:<6} "
                f"${candidate.prev_close:.2f} -> ${candidate.current_price:.2f} "
                f"({candidate.intraday_change_pct:+.1f}%)  "
                f"RelVol: {candidate.relative_volume:.1f}x  "
                f"Float: {candidate.float_shares / 1_000_000:.1f}M  "
                f"\"{candidate.news_headline or 'N/A'}\""
            )
