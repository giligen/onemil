"""Red-to-green (F6-PDR) — ONE spec for the backtest and the live engine (2026-09-17).

The book declared in research/fuckup_audit/H/F6/f6_pdr_book.md and specified in H/F6/ENGINE_SPEC.md:
  precondition  the 09:30 open is BELOW the prior close, and the prior day's range (high-low)/low is >= pdr_min_pct
                (ORB's shipped day-2-continuation rule, the only filter that replicated on VAL in Stage H);
  level         prior close x (1 + level_buffer);
  floor         on bars STRICTLY BEFORE the signal bar, (running high - running low) / running low >= range_floor_pct
                (the causal membership guarantee of the >=5%-range universe the book was measured on);
  signal        the first CLOSED bar whose high reaches the level, at or before last_entry_minute;
  stop          the lowest low from 09:30 through the signal bar ("before entry": the fill is the NEXT bar's open);
  fill / book   `trading.hod_break.entry_fill` (next open <= level x (1 + cap), no chase) and `run_book` — shared.

`detect` has the SAME signature shape and return type as `hod_break.detect` so `hod_break_engine.py` can run either
book through one code path (`book: hod_break | red_to_green`). Pure functions; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from trading.hod_break import HodBreakSignal, entry_fill, run_book, rv_profile  # noqa: F401  (re-exported on purpose)


@dataclass(frozen=True)
class RedToGreenParams:
    """The knobs — defaults are the declared book."""
    pdr_min_pct: float = 8.0          # prior-day (high - low) / low, in %
    range_floor_pct: float = 5.0      # range so far on bars strictly before the signal bar, in %
    level_buffer: float = 0.003       # level = prior close x (1 + level_buffer)
    cap: float = 0.006                # no-chase cap above the level for the next-open fill
    min_r_pct: float = 1.0            # stop distance as % of entry
    target_r: float = 2.0             # used by exit_mode 'target2r' / 'partial'
    max_per_day: int = 12
    max_concurrent: int = 4
    last_entry_minute: int = 840      # 14:00 ET
    flat_minute: int = 955            # 15:55 ET


def prior_day_range_pct(prev_high: float, prev_low: float) -> Optional[float]:
    """(high - low) / low of the prior day in %, or None when the inputs cannot make one."""
    if prev_high is None or prev_low is None or prev_low <= 0 or prev_high < prev_low:
        return None
    return (float(prev_high) - float(prev_low)) / float(prev_low) * 100.0


def eligible(day_open: float, prior_close: float, pdr_pct: Optional[float], p: RedToGreenParams = RedToGreenParams()) -> bool:
    """The 09:30 precondition: gap-down open (open < prior close) on a day whose prior range was >= pdr_min_pct."""
    if pdr_pct is None or prior_close is None or prior_close <= 0:
        return False
    return float(day_open) < float(prior_close) and float(pdr_pct) >= p.pdr_min_pct


def level_for(prior_close: float, p: RedToGreenParams = RedToGreenParams()) -> float:
    """The break level: prior close plus the buffer."""
    return float(prior_close) * (1.0 + p.level_buffer)


def detect(o: Sequence[float], h: Sequence[float], l: Sequence[float], v: Sequence[float], m: Sequence[int],
           adv20: float, prior_close: float, pdr_pct: Optional[float],
           p: RedToGreenParams = RedToGreenParams(), start_idx: int = 0) -> Optional[HodBreakSignal]:
    """First red-to-green break on or after `start_idx`. Bars are CLOSED 1-min RTH bars in order (bar 0 = 09:30).
    Returns a HodBreakSignal (level = the break level, stop = the pre-entry low) or None. Pure; the live engine calls
    it on every closed bar with start_idx = that bar's index and gets the same answer the backtest gets."""
    o = np.asarray(o, dtype=float); h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); v = np.asarray(v, dtype=float)
    n = len(h)
    if n < 2 or not eligible(o[0], prior_close, pdr_pct, p):
        return None
    level = level_for(prior_close, p)
    run_hi = np.maximum.accumulate(h); run_lo = np.minimum.accumulate(l); cumv = np.cumsum(v)
    for i in range(max(start_idx, 1), n):            # bar 0 has no bars before it: the floor cannot hold
        if int(m[i]) > p.last_entry_minute:
            return None
        lo_prev = float(run_lo[i - 1])
        if lo_prev <= 0 or (float(run_hi[i - 1]) - lo_prev) / lo_prev * 100.0 < p.range_floor_pct:
            continue                                  # the day is not (yet) in the universe the book was measured on
        if h[i] < level:
            continue
        stop = float(run_lo[i])                       # lowest low from 09:30 through the signal bar
        if stop >= level:
            continue
        dist = (level / float(o[0]) - 1.0) * 100.0
        rv = rv_profile(float(cumv[i]), adv20, int(m[i])) if adv20 and adv20 > 0 else 0.0
        return HodBreakSignal(bar_idx=i, level=level, stop=stop, dist_open_pct=dist, rv_profile=rv)
    return None


def r_ok(entry: float, stop: float, p: RedToGreenParams = RedToGreenParams()) -> bool:
    """The R floor: stop distance must be at least min_r_pct of the entry."""
    return entry > stop and (entry - stop) / entry * 100.0 >= p.min_r_pct
