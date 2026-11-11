"""ORB universe admission — ONE gate shared by the live engine and the BT feature builder.

Spec: docs/orb_universe_gate_parity_spec_20261111.md (cell 1,733 class U).

DEFINITION OF GAP (the only one): ``(open_px - prev_close) / prev_close * 100`` where ``prev_close`` is the
CLOSE of the last official daily bar strictly before the session (``daily_bars.close`` / the snapshot's
prev-daily-bar close) and ``open_px`` is the session open the CALLER supplies. The caller states which open:

* BT universe (``study_orb_broad.load_broad_universe`` SQL, ``study_orb_features`` per-symbol path): the
  official ``daily_bars.open`` of the session day.
* Live 09:30 filter: ``trading.orb_gap_gate.resolve_gap_input`` (09:30 minute-bar open, snapshot open when the
  bar is not yet available).
* Scoring-time re-check (live) / ``feature gap`` re-check (BT): the 09:30 bar1 open (``range_open`` live,
  ``rb.open[0]`` BT) — the number ``feats['gap_pct']`` is built from on both sides.

``admit`` never fetches anything and never changes a threshold: every cut comes from ``cfg`` (the engine's
``universe.*`` values / the BT's ``study_orb_broad`` constants). A gate argument passed as ``None`` is NOT
CHECKED (used by the scoring-time re-check, which only re-validates the gap).
"""
import re
from dataclasses import dataclass
from typing import Mapping, Optional

# NYSE/NASDAQ test tickers (momentum_sleeve.TEST_RE, orb_engine._SEED_TEST_TICKER_RE): never tradable.
TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT$')

# Reason codes (stable strings: the builder's per-reason COUNT line and the tests key on them).
OK = 'ok'
TEST_TICKER = 'test_ticker'
BAD_PREV_CLOSE = 'bad_prev_close'
BAD_OPEN = 'bad_open'
GAP_BELOW_MIN = 'gap_below_min'
GAP_ABOVE_MAX = 'gap_above_max'
PREV_VOLUME_BELOW_MIN = 'prev_volume_below_min'
PRICE_OUT_OF_BAND = 'price_out_of_band'


@dataclass(frozen=True)
class Admission:
    """Result of one admission decision: ``ok``, a stable ``reason`` code, and the gap that was judged."""
    ok: bool
    reason: str
    gap_pct: float


def gap_pct(prev_close: float, open_px: float) -> float:
    """The one gap formula (percent). NaN when prev_close <= 0 (undefined)."""
    if prev_close is None or prev_close <= 0 or open_px is None:
        return float('nan')
    return (float(open_px) - float(prev_close)) / float(prev_close) * 100.0


def admit(prev_close: float, open_px: float, prev_volume: Optional[float], price: Optional[float],
          cfg: Mapping[str, float], symbol: Optional[str] = None) -> Admission:
    """Decide universe membership for one symbol-day.

    Args:
        prev_close: close of the prior official daily bar.
        open_px: the session open the caller gates on (see module docstring).
        prev_volume: prior official daily volume, or None to skip the volume gate.
        price: price judged against the band (the BT and live both use the open), or None to skip the band.
        cfg: ``min_gap_pct`` (required), optional ``max_gap_pct`` (default unbounded), ``min_prev_volume``,
            ``min_price``, ``max_price``. Missing optional keys disable that gate.
        symbol: optional, only for the test-ticker gate.
    """
    if symbol is not None and TEST_TICKER_RE.match(symbol):
        return Admission(False, TEST_TICKER, float('nan'))
    if prev_close is None or prev_close <= 0:
        return Admission(False, BAD_PREV_CLOSE, float('nan'))
    if open_px is None or open_px <= 0:
        return Admission(False, BAD_OPEN, float('nan'))
    g = gap_pct(prev_close, open_px)
    if g < float(cfg['min_gap_pct']):
        return Admission(False, GAP_BELOW_MIN, g)
    max_gap = cfg.get('max_gap_pct')
    if max_gap is not None and g > float(max_gap):
        return Admission(False, GAP_ABOVE_MAX, g)
    min_vol = cfg.get('min_prev_volume')
    if prev_volume is not None and min_vol is not None and prev_volume < float(min_vol):
        return Admission(False, PREV_VOLUME_BELOW_MIN, g)
    lo, hi = cfg.get('min_price'), cfg.get('max_price')
    if price is not None and lo is not None and hi is not None and not (float(lo) <= price <= float(hi)):
        return Admission(False, PRICE_OUT_OF_BAND, g)
    return Admission(True, OK, g)
