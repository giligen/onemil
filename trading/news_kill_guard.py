"""
Shared news-kill decision — used by BOTH the backtest (BacktestRunner.
_check_news_kill) and the live engine (TradingEngine) so the two cannot drift.

The news-kill gate blocks bull-flag trades that land in empirically loser
"segments": high average volume, sub-$3 price, high float, or the
overextended $5-12 / pole 8-15% mid-cap bucket.

Catalyst exemption — config `trading.news_kill_rules.catalyst_exemption`,
default FALSE (shipped 2026-05-21):
  Historically a trade with a real news catalyst was EXEMPTED from the segment
  rules. The 2026-05 news-classifier A/B showed that exemption is
  value-destroying — bad-segment trades that genuinely have a real catalyst
  are still net losers (no-exemption beat regex-exemption by ~$12k raw on a
  1,195-trade sample; Haiku-exemption was worse still). With the flag FALSE
  the segment rules apply to EVERY trade and no news classifier is consulted
  on the trade-decision path. Flip to TRUE to restore exempt-on-catalyst.
"""
from typing import Tuple


def news_kill_decision(has_catalyst: bool, catalyst_exemption: bool,
                       avg_vol: float, entry_price: float, float_shares: float,
                       pole_gain: float, max_avg_vol: float, min_price: float,
                       max_float: float) -> Tuple[bool, str]:
    """Decide whether a bull-flag trade survives the news-kill gate.

    Args:
        has_catalyst: True if the symbol has a real news catalyst. Only
            consulted when catalyst_exemption is True — callers may pass False
            unconditionally when the exemption is off (skips the news lookup).
        catalyst_exemption: when True, a real catalyst exempts the trade from
            the segment rules (legacy behavior). When False (shipped default)
            the segment rules apply unconditionally.
        avg_vol: 20-day average daily volume.
        entry_price: planned entry price.
        float_shares: share float.
        pole_gain: bull-flag pole gain percent.
        max_avg_vol / min_price / max_float: segment-rule thresholds from
            config (news_kill_rules.{max_avg_vol_no_news, min_price_no_news,
            max_float_no_news}).

    Returns:
        (should_trade, reason). should_trade=False kills the trade; reason
        names the rule that fired (or the exemption / pass reason).
    """
    if catalyst_exemption and has_catalyst:
        return (True, "has_catalyst")

    # Segment rules — empirically loser buckets (apply to every trade unless
    # exempted above).
    if avg_vol >= max_avg_vol:
        return (False, f"avg_vol {avg_vol / 1e6:.1f}M >= {max_avg_vol / 1e6:.0f}M")
    if entry_price < min_price:
        return (False, f"price ${entry_price:.2f} < ${min_price:.0f}")
    if float_shares >= max_float:
        return (False, f"float {float_shares / 1e6:.0f}M >= {max_float / 1e6:.0f}M")
    if 5 <= entry_price < 12 and 8 <= pole_gain < 15:
        return (False, f"${entry_price:.0f} + pole {pole_gain:.1f}% "
                       f"(overextended mid-cap)")
    return (True, "good_segment")
