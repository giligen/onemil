"""ORB submit-timing instrumentation (spec docs/orb_submit_timing_instrumentation_20261017.md).

LOGGING ONLY — nothing here feeds a trading decision except that the 09:35 latency tripwire now READS the broker
clock (`submitted_at` on the order response) instead of the engine's own stamp. Facts: the engine's "first order
submit 8.5 s" was its own clock; the broker's `submitted_at` for the same orders was 4.8-10.1 s later.

One module, used by `trading/orb_engine.py` (writer) and `scripts/eod_sections.py` (reader), so the line format and
its parser can never drift apart.
"""
from __future__ import annotations

import logging
import re
import statistics
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NA = "NA"
# (http_start_epoch, http_end_epoch) spans of one burst of submits
Span = Tuple[float, float]

TIMING_RE = re.compile(
    r"\[ORB\] SUBMIT TIMING sym=(?P<sym>\S+) decision_ts=(?P<decision_ts>\S+) quote_ms=(?P<quote_ms>\S+) "
    r"size_ms=(?P<size_ms>\S+) http_ms=(?P<http_ms>\S+) decision_to_http_start_s=(?P<d2h>\S+) "
    r"broker_submitted_at=(?P<broker_at>\S+) broker_minus_decision_s=(?P<b_minus_d>\S+) path=(?P<path>\S+)")
ENGINE_LATENCY_RE = re.compile(r"\[ORB\] SUBMIT LATENCY sym=(\S+) seconds=(-?[\d.]+)")

_missing_warned_on: Optional[date] = None


def _et(dt: datetime) -> datetime:
    """Convert an aware datetime to America/New_York."""
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo('America/New_York'))


def fmt_et(dt: datetime) -> str:
    """`HH:MM:SS.mmm` in ET for an aware datetime."""
    return _et(dt).strftime('%H:%M:%S.%f')[:-3]


def parse_broker_ts(value) -> Optional[datetime]:
    """Aware-UTC datetime from an order-response timestamp (datetime or ISO string, `Z` suffix ok); None if
    unparseable or naive-and-ambiguous strings are absent."""
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def broker_ts_from_response(resp) -> Optional[datetime]:
    """Broker clock for an order: `submitted_at`, falling back to `created_at`. None when both are missing —
    WARNING once per ET day (a silent fallback to the engine clock would hide the very gap being measured)."""
    global _missing_warned_on
    ts = None
    if isinstance(resp, dict):
        ts = parse_broker_ts(resp.get('submitted_at')) or parse_broker_ts(resp.get('created_at'))
    if ts is None:
        today = _et(datetime.now(timezone.utc)).date()
        if _missing_warned_on != today:
            _missing_warned_on = today
            logger.warning(
                "[ORB] SUBMIT TIMING: order response carries neither submitted_at nor created_at — "
                "broker_submitted_at=NA, latency tripwire falls back to clock=engine (warned once per day)")
    return ts


def _ms(x: Optional[float]) -> str:
    return NA if x is None else f"{x:.0f}"


def build_timing_line(sym: str, decision_ts: float, http_start: float, http_end: float,
                      broker_dt: Optional[datetime], path: str,
                      quote_ms: Optional[float] = None, size_ms: Optional[float] = None
                      ) -> Tuple[str, Optional[float]]:
    """(INFO line, broker_minus_decision seconds or None). Times are epoch seconds (`time.time()`)."""
    decision_dt = datetime.fromtimestamp(decision_ts, tz=timezone.utc)
    b_minus_d = (broker_dt.timestamp() - decision_ts) if broker_dt is not None else None
    line = (f"[ORB] SUBMIT TIMING sym={sym} decision_ts={fmt_et(decision_dt)} quote_ms={_ms(quote_ms)} "
            f"size_ms={_ms(size_ms)} http_ms={(http_end - http_start) * 1000.0:.0f} "
            f"decision_to_http_start_s={http_start - decision_ts:.2f} "
            f"broker_submitted_at={fmt_et(broker_dt) if broker_dt is not None else NA} "
            f"broker_minus_decision_s={'%.2f' % b_minus_d if b_minus_d is not None else NA} path={path}")
    return line, b_minus_d


def spans_are_serial(spans: Sequence[Span], tol_s: float = 0.005) -> bool:
    """True when k >= 2 submit calls ran strictly one after another (no overlap in their HTTP windows)."""
    if len(spans) < 2:
        return False
    ordered = sorted(spans)
    return all(nxt[0] >= prev[1] - tol_s for prev, nxt in zip(ordered, ordered[1:]))


def build_burst_line(spans: Sequence[Span]) -> Optional[str]:
    """`[ORB] SUBMIT BURST n=<k> span_s=<first http start -> last http end> serial=<yes|no>`; None when empty."""
    if not spans:
        return None
    span_s = max(e for _, e in spans) - min(s for s, _ in spans)
    return (f"[ORB] SUBMIT BURST n={len(spans)} span_s={span_s:.2f} "
            f"serial={'yes' if spans_are_serial(spans) else 'no'}")


def parse_timing_lines(lines: Sequence[str]) -> Dict[str, List[float]]:
    """From a day's log lines: `broker` = broker_minus_decision_s values (NA skipped), `engine` = the older
    `SUBMIT LATENCY seconds=` values."""
    broker: List[float] = []
    engine: List[float] = []
    for ln in lines:
        m = TIMING_RE.search(ln)
        if m and m.group('b_minus_d') != NA:
            try:
                broker.append(float(m.group('b_minus_d')))
            except ValueError:
                logger.warning("SUBMIT TIMING line with unparseable broker_minus_decision_s: %s", ln[:200])
            continue
        m = ENGINE_LATENCY_RE.search(ln)
        if m:
            engine.append(float(m.group(2)))
    return {'broker': broker, 'engine': engine}


def timing_summary_line(lines: Sequence[str]) -> str:
    """EOD line: median and max seconds per clock, `broker` (new) beside `engine` (old)."""
    d = parse_timing_lines(lines)

    def part(label: str, vals: List[float]) -> str:
        if not vals:
            return f"{label} NO-DATA"
        return f"{label} median {statistics.median(vals):.1f}s max {max(vals):.1f}s (n {len(vals)})"
    return "ORB submit latency: " + " | ".join([part('broker', d['broker']), part('engine', d['engine'])])
