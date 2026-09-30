"""ORB gap-gate input resolution — SHARED by live universe build and any BT
rebuild that wants to replay live's observed inputs (parity by construction).

Root cause (docs/orb_parity_20260930.md, 2026-09-30): live scored the 5% gap
gate off Alpaca's real-time snapshot `open` at ~09:30 ET, which can differ
from the settled `daily_bars.open` the backtest recomputes after the fact.
ASTX read +5%+ live (snapshot open $10.32-area) but settled at +1.95%
(daily_bars open $9.95); AEHG was the same pattern. Same 5.0%/500k/$3-$30
thresholds on both sides — different INPUT price, so two different
candidate universes by construction.

Fix: prefer the official 09:30 ET minute bar's open (the real opening
print, available by 09:31) over the snapshot's `open` field whenever one is
present; ALWAYS report which source won and the exact inputs used, so a BT
rebuild can replay live's observed gap instead of guessing at it from
`daily_bars`.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

SOURCE_BAR = 'bar'
SOURCE_SNAPSHOT = 'snapshot'


@dataclass
class GapGateInput:
    """Exact inputs the gap gate used for one symbol — persisted verbatim to
    the ORB dry/paper ledger and logged at INFO so a BT rebuild never has to
    guess which open price live scored the gate against."""
    symbol: str
    gap_input_open: float
    gap_input_prev_close: float
    gap_pct: float
    source: str  # SOURCE_BAR | SOURCE_SNAPSHOT
    timestamp: str  # ISO-8601 UTC

    def as_dict(self) -> dict:
        return {
            'gap_input_open': self.gap_input_open,
            'gap_input_prev_close': self.gap_input_prev_close,
            'gap_pct': self.gap_pct,
            'source': self.source,
            'timestamp': self.timestamp,
        }


def resolve_gap_input(
    symbol: str,
    snapshot_open: float,
    prev_close: float,
    minute_bar_open: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Optional[GapGateInput]:
    """Resolve the gap-gate's open price + provenance for one symbol.

    Prefers `minute_bar_open` (the settled 09:30 ET minute bar's open, the
    official opening print) over `snapshot_open` (Alpaca's real-time
    snapshot, which can reflect a still-updating print) whenever the bar
    value is present and usable (> 0). Falls back to the snapshot — and
    logs WARNING explaining why — when no usable bar open is available
    (e.g., called before 09:31 ET, or the bar hasn't landed in the cache
    yet). This is the CLAUDE.md-mandated fallback-must-log-WARNING rule:
    silently preferring a worse input is not acceptable.

    Args:
        symbol: for logging only.
        snapshot_open: Alpaca real-time snapshot `open` field.
        prev_close: yesterday's settled close (same on both sources).
        minute_bar_open: the 09:30 ET minute bar's `open`, if the caller
            could fetch one (None when not yet available/not looked up).
        now: injectable clock for tests; defaults to real UTC now.

    Returns:
        GapGateInput, or None if neither source has a usable open or
        prev_close <= 0 (gap is undefined — caller should skip the symbol,
        same as today's `if prev_close <= 0: continue`).
    """
    now = now or datetime.now(timezone.utc)
    if prev_close is None or prev_close <= 0:
        logger.warning(
            f"ORB GAP_GATE: {symbol} prev_close={prev_close!r} <= 0 — "
            f"gap undefined, no gate input resolved"
        )
        return None

    if minute_bar_open is not None and minute_bar_open > 0:
        open_price = float(minute_bar_open)
        source = SOURCE_BAR
    else:
        if snapshot_open is None or snapshot_open <= 0:
            logger.warning(
                f"ORB GAP_GATE: {symbol} has neither a usable 09:30 bar "
                f"open nor a usable snapshot open — gate input unresolved"
            )
            return None
        open_price = float(snapshot_open)
        source = SOURCE_SNAPSHOT
        logger.warning(
            f"ORB GAP_GATE: {symbol} no 09:30 minute bar open available "
            f"(not yet cached, or called before 09:31 ET) — falling back "
            f"to the real-time snapshot open ${open_price:.4f} (the same "
            f"input class implicated in the 9/30 ASTX/AEHG parity gap, "
            f"docs/orb_parity_20260930.md)"
        )

    gap_pct = (open_price - prev_close) / prev_close * 100.0
    result = GapGateInput(
        symbol=symbol,
        gap_input_open=open_price,
        gap_input_prev_close=float(prev_close),
        gap_pct=gap_pct,
        source=source,
        timestamp=now.isoformat(),
    )
    logger.info(
        f"[ORB] GAP_GATE {symbol} gap_input_open={result.gap_input_open:.4f} "
        f"gap_input_prev_close={result.gap_input_prev_close:.4f} "
        f"gap_pct={result.gap_pct:.3f} source={result.source} "
        f"timestamp={result.timestamp}"
    )
    return result
