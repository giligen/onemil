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
    float_shares: Optional[int] = None
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


CLOSE_CALL_MIN_MOVE_FRACTION = 0.5  # a missing intraday_change only counts if >= 50 % of the threshold


def is_close_call(candidate: ScanCandidate, min_intraday_change_pct: float) -> bool:
    """
    Single definition of a "close call" (journal-flood fix 2026-11-02).

    A close call is a REAL near-miss: exactly one criterion is missing and,
    when that criterion is `intraday_change`, the stock has already moved at
    least CLOSE_CALL_MIN_MOVE_FRACTION of the configured threshold. A stock
    that passes the other five criteria but has not moved is not a near-miss
    (it is ~3,900 symbols per cycle).

    Args:
        candidate: ScanCandidate with criteria_met populated
        min_intraday_change_pct: configured intraday_change_pct_min

    Returns:
        True when the candidate is a genuine near-miss
    """
    missing = [k for k, v in candidate.criteria_met.items() if not v]
    if len(missing) != 1:
        return False
    if missing[0] == 'intraday_change':
        return candidate.intraday_change_pct >= CLOSE_CALL_MIN_MOVE_FRACTION * min_intraday_change_pct
    return True


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
        require_news: bool = True,
        min_dollar_volume: float = 0,
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
            min_dollar_volume: Minimum daily dollar volume (price × volume)
        """
        self.price_min = price_min
        self.price_max = price_max
        self.float_max = float_max
        self.gap_pct_min = gap_pct_min
        self.require_news = require_news
        self.intraday_change_pct_min = intraday_change_pct_min
        self.relative_volume_min = relative_volume_min
        self.min_dollar_volume = min_dollar_volume
        # Symbols whose INTRADAY QUALIFIED line was logged and not yet
        # followed by a non-qualifying evaluation (transition-only logging).
        self._qualified_logged: set = set()

    def is_close_call(self, candidate: ScanCandidate) -> bool:
        """Close-call predicate bound to this instance's configured threshold."""
        return is_close_call(candidate, self.intraday_change_pct_min)

    def evaluate_premarket(self, candidate: ScanCandidate) -> bool:
        """
        Evaluate pre-market gap criteria. Pure quantitative — no news check.

        A stock qualifies as a gap-up candidate if:
        - Gap >= threshold

        News/LLM classification is deferred to intraday (last step).
        Price and float are already filtered by universe.

        Args:
            candidate: ScanCandidate to evaluate

        Returns:
            True if candidate qualifies as pre-market gap-up
        """
        candidate.criteria_met['gap'] = candidate.gap_pct >= self.gap_pct_min

        qualified = candidate.criteria_met['gap']

        if qualified:
            logger.debug(
                f"PREMARKET GAP-UP: {candidate.symbol} | "
                f"Gap: {candidate.gap_pct:.1f}%"
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
        candidate.criteria_met['float'] = (
            candidate.float_shares is not None
            and candidate.float_shares > 0
            and candidate.float_shares <= self.float_max
        )
        candidate.criteria_met['relative_volume'] = (
            candidate.relative_volume >= self.relative_volume_min
        )
        # Shared BF selection semantics (2026-08-29 alignment): the same
        # threshold test the BT cache build runs — one predicate, one
        # file (trading/bf_selection.py). candidate.intraday_change_pct
        # is already max(gap%, range%) computed upstream.
        from trading.bf_selection import intraday_qualifies
        candidate.criteria_met['intraday_change'] = intraday_qualifies(
            gap_pct=candidate.intraday_change_pct,
            range_pct=candidate.intraday_change_pct,
            threshold_pct_points=self.intraday_change_pct_min,
        )
        candidate.criteria_met['has_news'] = candidate.has_news or not self.require_news
        candidate.criteria_met['dollar_volume'] = (
            self.min_dollar_volume <= 0 or
            (candidate.current_price * candidate.current_volume >= self.min_dollar_volume)
        )

        qualified = all(candidate.criteria_met.values())

        if not qualified:
            self._qualified_logged.discard(candidate.symbol)
        if qualified and candidate.symbol not in self._qualified_logged:
            # Transition-only (journal-flood fix 2026-11-02): one line when a
            # symbol becomes qualified, not one per cycle while it stays so.
            self._qualified_logged.add(candidate.symbol)
            float_q = f"{candidate.float_shares / 1_000_000:.1f}M" if candidate.float_shares else "N/A"
            logger.info(
                f"INTRADAY QUALIFIED: {candidate.symbol} | "
                f"${candidate.current_price:.2f} ({candidate.intraday_change_pct:+.1f}%) | "
                f"RelVol: {candidate.relative_volume:.1f}x | "
                f"Float: {float_q} | "
                f"News: {candidate.news_headline}"
            )
        elif not qualified and self.is_close_call(candidate):
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
        float_str = f"{candidate.float_shares / 1_000_000:.1f}M" if candidate.float_shares else "N/A"
        company = (candidate.company_name or '')[:20]
        if phase == 'premarket':
            return (
                f"  {candidate.symbol:<6} | {company:<20} | "
                f"Gap: {candidate.gap_pct:+.1f}% | "
                f"Price: ${candidate.current_price:.2f} | "
                f"Float: {float_str} | "
                f"News: {candidate.news_headline or 'N/A'}"
            )
        else:
            return (
                f"  {candidate.symbol:<6} "
                f"${candidate.prev_close:.2f} -> ${candidate.current_price:.2f} "
                f"({candidate.intraday_change_pct:+.1f}%)  "
                f"RelVol: {candidate.relative_volume:.1f}x  "
                f"Float: {float_str}  "
                f"\"{candidate.news_headline or 'N/A'}\""
            )
