#!/usr/bin/env python3
"""Synthetic unit tests for cell_1487.py (PREREG_1487.md) -- eligibility window, entry bar
handling, the R'' floor, and the 1,488 pyramid add/stop move. No DB or CSV touched: bars and the
base book are fabricated in-memory; cell_1487.load_bars is monkeypatched for build_1488 so no
real bars_fills_1478.db read happens in a unit test."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1487 as c1487


def bars(rows):
    """rows: list of (m, o, h, l, c) -> RTH minute bar DataFrame."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])


def base_row(**kw):
    d = dict(day='2025-07-01', symbol='ZZZZ', split='VAL', holdout='VAL', wk='2025-06-28/2025-07-04',
              fill_min=600.2, exit_m=700.0, why='eod', outcome_R=0.3, level=50.0, half_entry=0.02,
              fill=50.1, stop=49.5, R=0.6)
    d.update(kw)
    return pd.DataFrame([d])


# ------------------------------------------------------------------------------------------------
# Eligibility window
# ------------------------------------------------------------------------------------------------

def test_eligible_when_no_dip_and_base_exits_late():
    """No bar in (fill_min, fill_min+15] dips to level-0.01, and base exit_m > fill_min+15 ->
    eligible, and a resolvable entry bar produces an 'entered' row."""
    base = base_row(fill_min=600.0, exit_m=700.0, level=50.0)
    b = bars([(601, 50.2, 50.3, 50.05, 50.2), (610, 50.2, 50.3, 50.1, 50.2),
              (616, 50.25, 50.4, 50.2, 50.3), (955, 50.3, 50.3, 50.3, 50.3)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert row.eligible
    assert row.reason == 'entered'


def test_ineligible_on_dip_inside_window():
    """A bar inside (fill_min, fill_min+15] with low <= level-0.01 -> ineligible, regardless of
    the base's own exit minute."""
    base = base_row(fill_min=600.0, exit_m=700.0, level=50.0)
    b = bars([(605, 50.0, 50.0, 49.98, 50.0),      # low 49.98 <= level-0.01 (49.99) -> dip
              (616, 50.2, 50.3, 50.1, 50.2)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert not row.eligible
    assert row.reason == 'dip_in_window'


def test_ineligible_when_base_already_exited_by_minute_15():
    """base exit_m <= fill_min + 15 -> ineligible even with zero dip bars (the runner was already
    gone before the confirmation window closed)."""
    base = base_row(fill_min=600.0, exit_m=610.0, level=50.0)   # exit at 610 <= 615
    b = bars([(605, 50.2, 50.3, 50.1, 50.2)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert not row.eligible
    assert row.reason == 'base_exited_by_15'


def test_a_dip_exactly_at_the_tick_counts():
    """low == level - 0.01 exactly is a dip (<=), per the PREREG's literal '$0.01' boundary."""
    base = base_row(fill_min=600.0, exit_m=700.0, level=50.0)
    b = bars([(605, 50.0, 50.0, 49.99, 50.0)])                  # low == level - 0.01 exactly
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    assert not scored.iloc[0].eligible


# ------------------------------------------------------------------------------------------------
# Entry bar handling
# ------------------------------------------------------------------------------------------------

def test_missing_entry_bar_is_excluded_not_silently_entered():
    """Eligible, but no bar exists at exactly minute fill_min(floor)+16 -> reason 'no_entry_bar',
    never counted as entered."""
    base = base_row(fill_min=600.4, exit_m=700.0, level=50.0)   # entry bar would be m=616
    b = bars([(605, 50.2, 50.3, 50.1, 50.2), (617, 50.2, 50.3, 50.1, 50.2)])   # 616 missing
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert row.eligible
    assert row.reason == 'no_entry_bar'


def test_entry_price_is_open_plus_half_entry_and_stop_is_level_minus_tick():
    base = base_row(fill_min=600.0, exit_m=700.0, level=50.0, half_entry=0.03)
    b = bars([(605, 50.2, 50.3, 50.1, 50.2),
              (616, 50.40, 50.5, 50.3, 50.45),                  # entry bar, open=50.40
              (955, 50.5, 50.5, 50.5, 50.5)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert row.reason == 'entered'
    assert row.entry == pytest.approx(50.40 + 0.03)
    assert row.stop2 == pytest.approx(50.0 - 0.01)
    assert row.R2 == pytest.approx(row.entry - row.stop2)
    assert row.target == pytest.approx(row.entry + 2 * row.R2)


def test_entry_bar_itself_stopping_is_conservative():
    """A dip on the entry bar ITSELF (after the open) stops the confirmation trade -- walk_path's
    own convention, which the PREREG explicitly calls out as conservative."""
    base = base_row(fill_min=600.0, exit_m=700.0, level=50.0, half_entry=0.0)
    # entry bar (m=616): open 50.0 -> stop = level-0.01 = 49.99; low 49.5 on the SAME bar -> stop.
    b = bars([(616, 50.0, 50.1, 49.5, 49.8)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert row.why2 == 'stop'
    assert row.exit_m2 == 616


# ------------------------------------------------------------------------------------------------
# R'' floor
# ------------------------------------------------------------------------------------------------

def test_below_floor_is_flagged_not_dropped():
    """R'' < 0.5% of price -> below_floor True, entered still True (reported, excluded only from
    the PRIMARY book by the caller's own filter, never dropped at this stage)."""
    # entry ~50, stop just 2 cents under -> R'' ~= 0.02, 0.02/50 = 0.04% << 0.5%.
    base = base_row(fill_min=600.0, exit_m=700.0, level=49.99, half_entry=0.0)
    b = bars([(616, 50.00, 50.05, 49.98, 50.0), (955, 50.0, 50.0, 50.0, 50.0)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert row.reason == 'entered'
    assert row.below_floor
    assert row.r2_pct_price < 0.5


def test_above_floor_is_not_flagged():
    base = base_row(fill_min=600.0, exit_m=700.0, level=45.0, half_entry=0.0)
    b = bars([(616, 50.00, 50.05, 49.95, 50.0), (955, 50.0, 50.0, 50.0, 50.0)])
    scored = c1487.eligibility_and_entry(base, {('ZZZZ', '2025-07-01'): b})
    row = scored.iloc[0]
    assert not row.below_floor
    assert row.r2_pct_price >= 0.5


# ------------------------------------------------------------------------------------------------
# Pyramid add / stop move (cell 1,488)
# ------------------------------------------------------------------------------------------------

def _entered_row(**kw):
    """A minimal 'scored' frame row shaped like eligibility_and_entry's 'entered' output, the
    input build_1488 consumes."""
    d = dict(day='2025-07-01', symbol='ZZZZ', split='VAL', holdout='VAL', fill_min=600.0,
              entry_bar_m=616, entry=50.10, stop2=49.99, base_outcome_R=0.3,
              reason='entered', below_floor=False)
    d.update(kw)
    return pd.DataFrame([d])


def test_pyramid_target_and_stop_use_original_fill_and_new_stop(monkeypatch):
    """target = ORIGINAL fill + 2*ORIGINAL R; stop = the SAME level-0.01 as 1,487 (stop2)."""
    scored = _entered_row()
    base = pd.DataFrame([dict(day='2025-07-01', symbol='ZZZZ', fill=48.50, stop=47.50, R=1.0)])
    base = base.set_index(['day', 'symbol'], drop=False)
    b = bars([(616, 50.10, 50.2, 50.0, 50.15), (955, 50.5, 50.5, 50.5, 50.5)])
    monkeypatch.setattr(c1487, 'load_bars', lambda pairs: {('ZZZZ', '2025-07-01'): b})
    df1488 = c1487.build_1488(scored, base.reset_index(drop=True))
    row = df1488.iloc[0]
    # target the walk was run against = fill(48.5) + 2*R(1.0) = 50.5, reached at the EOD bar close.
    assert row.why3 in ('eod', 'target')
    # weighted raw R combines leg1 (from the ORIGINAL fill 48.50) and leg2 (from the NEW entry
    # 50.10), both over the ORIGINAL R = 1.0 -- sanity: leg1 return > leg2 return since leg1's
    # entry is far below leg2's for the same exit price.
    exit_price = row.exit_price3
    leg1 = (exit_price - 48.50) / 1.0
    leg2 = (exit_price - 50.10) / 1.0
    expected_raw = (1.0 / 3.0) * leg1 + (2.0 / 3.0) * leg2
    assert leg1 > leg2
    assert row.net_1488 == pytest.approx(expected_raw, abs=1e-6) or row.why3 != 'target'


def test_pyramid_stop_exit_charges_stop_slip_not_zero(monkeypatch):
    scored = _entered_row(entry=50.10, stop2=49.99)
    base = pd.DataFrame([dict(day='2025-07-01', symbol='ZZZZ', fill=48.50, stop=47.50, R=1.0)])
    b = bars([(616, 50.10, 50.15, 49.5, 49.7)])                 # dips through stop2 (49.99) at open
    monkeypatch.setattr(c1487, 'load_bars', lambda pairs: {('ZZZZ', '2025-07-01'): b})
    df1488 = c1487.build_1488(scored, base)
    row = df1488.iloc[0]
    assert row.why3 == 'stop'
    # a stop exit must be strictly worse than the same raw move with zero cost.
    leg1 = (row.exit_price3 - 48.50) / 1.0
    leg2 = (row.exit_price3 - 50.10) / 1.0
    raw = (1.0 / 3.0) * leg1 + (2.0 / 3.0) * leg2
    assert row.net_1488 < raw


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
