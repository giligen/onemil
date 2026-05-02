"""SPY regime input shared between BT and live conviction scorers.

Single source of truth for the SPY 3-day average daily range that feeds
the conviction-rule's `spy_regime` contribution. Both `backtest._get_spy_3d_range`
(BT) and `trading.trading_engine._get_spy_3d_range_live` (live) call into this
module so a missing/stale data path produces identical behavior on both sides
(parity by construction — same pattern as `trading/two_tier_filter.py`).

Why this module exists (post-mortem 2026-05-01 EAF false-positive):

The previous implementations had a silent fallback `return 1.0` whenever SPY
data was missing or the consumer reached into a wrong attribute. The conviction
rule maps `0.8 < spy_3d_range < 1.2` to `sr_contrib = 0.0` — so the literal
sentinel `1.0` lands precisely in the rule's neutral band and **inflates
conviction by +0.5** vs what real data would have produced when SPY was weak.
On 2026-05-01 this fired three live buy-stop orders for EAF (~$48K notional
each) on prod that should have been filtered. Per CLAUDE.md "All fallback
code paths MUST log ERROR or WARNING" and "Production code never includes
mock logic, always real implementations" — silent neutral defaults were a
violation of both rules.

Contract:
    - `compute_spy_3d_range` returns `Optional[float]`. `None` MUST be
      propagated to the conviction rule (which treats it as the worst-case
      penalty, -0.5). Callers MUST NOT substitute a numeric default.
    - `is_spy_data_stale` is a separate guard that callers invoke before
      passing bars in, so the same helper handles both "no data" and
      "old data" failure modes uniformly.
    - Both functions log WARNING/ERROR on every fallback path so silent
      data outages are loud in `journalctl`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Default freshness threshold: 5 calendar days covers a Friday→Monday gap
# with one missed market day (e.g., Mon holiday). Beyond that, the data is
# almost certainly drifting — better to fail closed than score on stale bars.
DEFAULT_STALENESS_MAX_CALENDAR_DAYS = 5


def compute_spy_3d_range(spy_bars: Sequence[Mapping]) -> Optional[float]:
    """Mean of `(high - low) / low * 100` across the last 3 SPY bars.

    Args:
        spy_bars: Sequence of bar mappings. Each bar must have 'high' and
            'low' keys (numeric, > 0). Caller is responsible for ordering —
            the function uses the LAST 3 entries, so callers passing a
            longer history MUST pre-sort ascending by date.

    Returns:
        Average daily range %, or `None` if any of:
            - fewer than 3 bars provided
            - a bar lacks high/low keys (or they're non-numeric)
            - any high or low is non-positive

        Logs WARNING on every None-return so missing data is observable.
    """
    if not spy_bars or len(spy_bars) < 3:
        logger.warning(
            "compute_spy_3d_range: insufficient SPY bars (got %d, need >=3) "
            "— returning None (caller MUST treat as 'regime unknown', not neutral)",
            len(spy_bars) if spy_bars else 0,
        )
        return None

    last3 = list(spy_bars[-3:])
    ranges = []
    for bar in last3:
        try:
            h = float(bar.get('high', 0) or 0)
            l = float(bar.get('low', 0) or 0)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                "compute_spy_3d_range: bar missing/invalid high/low: %r (err=%s) "
                "— returning None",
                bar, e,
            )
            return None
        if l <= 0 or h <= 0:
            logger.warning(
                "compute_spy_3d_range: bar has non-positive high/low: high=%r low=%r "
                "— returning None", h, l,
            )
            return None
        ranges.append((h - l) / l * 100.0)

    return sum(ranges) / len(ranges)


def is_spy_data_stale(
    latest_bar_date: Optional[date],
    reference_date: date,
    max_calendar_days: int = DEFAULT_STALENESS_MAX_CALENDAR_DAYS,
) -> bool:
    """True if the most-recent SPY bar is too old to use for `reference_date`.

    Calendar-day comparison (not trading-day) — `max_calendar_days=5` covers
    Friday→Monday plus one missed market holiday. Beyond that, data is too
    old to safely score regime context.

    Args:
        latest_bar_date: bar_date of the newest SPY bar in the data source,
            or `None` if no bars exist at all.
        reference_date: The date the SPY data is about to be used for —
            typically `date.today()` (live) or the BT trade_date.
        max_calendar_days: Inclusive maximum gap. `age <= max` is fresh;
            `age > max` is stale. Default 5.

    Returns:
        True if data is stale or missing. Logs ERROR on staleness so the
        condition surfaces in `journalctl` and downstream alerting can
        catch it.
    """
    if latest_bar_date is None:
        logger.error(
            "is_spy_data_stale: no SPY bars at all — refusing to score regime"
        )
        return True
    age = (reference_date - latest_bar_date).days
    if age > max_calendar_days:
        logger.error(
            "is_spy_data_stale: latest SPY bar=%s is %d calendar days before "
            "reference=%s (threshold=%d) — refusing to score regime",
            latest_bar_date, age, reference_date, max_calendar_days,
        )
        return True
    return False
