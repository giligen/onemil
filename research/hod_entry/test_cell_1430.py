"""Unit tests for cell_1430.py's six exit variants on synthetic bar paths."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1430 as c1430  # noqa: E402


def bars(*rows):
    """rows = (m, o, h, l, c, vwap) tuples -> itertuples()-ready DataFrame."""
    df = pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c', 'vwap'])
    return df


ENTRY, STOP, R = 10.0, 9.0, 1.0     # target0 = entry + 2R = 12.0
TARGET = ENTRY + c1430.TARGET_R * R


def test_variant_a_time_stop_fires_at_90min_open_when_neither_triggers():
    b = bars((601, 10.2, 10.4, 10.0, 10.3, 10.1),   # entry_m=570 -> cutoff=660
             (660, 10.5, 10.6, 10.4, 10.5, 10.2),
             (700, 10.6, 10.7, 10.5, 10.6, 10.3))
    exit_m, px, why = c1430.variant_a_time_stop(ENTRY, STOP, TARGET, b.itertuples(), entry_m=570)
    assert (exit_m, px, why) == (660, 10.5, 'time_stop')


def test_variant_a_target_beats_time_stop():
    b = bars((600, 10.5, 12.5, 10.4, 12.0, 10.6))   # hits target before the 660 cutoff
    exit_m, px, why = c1430.variant_a_time_stop(ENTRY, STOP, TARGET, b.itertuples(), entry_m=570)
    assert (exit_m, px, why) == (600, TARGET, 'target')


def test_variant_b_breakeven_lock_arms_next_bar_only():
    b = bars((600, 10.2, 11.0, 10.1, 10.9, 10.3),   # high 11.0 = entry+1R -> lock arms
             (601, 10.8, 10.9, 9.8, 9.9, 10.5),      # low 9.8 <= locked stop (entry=10.0) -> stop
             (602, 8.0, 8.5, 7.9, 8.0, 8.0))
    exit_m, px, why = c1430.variant_b_breakeven(ENTRY, STOP, TARGET, R, b.itertuples())
    assert (exit_m, px, why) == (601, 10.0, 'stop')


def test_variant_b_lock_does_not_apply_on_its_own_arming_bar():
    """The bar that touches +1R also has low=9.5 < entry: since the lock has not yet taken
    effect (arms only from the NEXT bar), this bar exits at the ORIGINAL stop, not breakeven."""
    b = bars((600, 10.2, 11.0, 9.0, 9.5, 10.0))     # low 9.0 <= stop0=9.0 in the SAME bar
    exit_m, px, why = c1430.variant_b_breakeven(ENTRY, STOP, TARGET, R, b.itertuples())
    assert (exit_m, px, why) == (600, 9.0, 'stop')


def test_variant_c_orb_lock_trigger_and_lock_levels():
    b = bars((600, 10.2, 11.6, 10.1, 11.5, 10.3),   # high 11.6 >= entry+1.5R=11.5 -> lock arms
              (601, 10.6, 10.7, 10.55, 10.6, 10.5),  # stays above locked stop 10.5 -> no exit
              (602, 10.5, 10.6, 10.3, 10.4, 10.4))   # low 10.3 <= 10.5 -> stop at 10.5
    exit_m, px, why = c1430.variant_c_orb_lock(ENTRY, STOP, TARGET, R, b.itertuples())
    assert (exit_m, px, why) == (602, 10.5, 'stop')


def test_variant_d_scale_out_blends_partial_and_stopped_runner():
    b = bars((600, 10.2, 12.3, 10.1, 12.0, 10.3),   # high 12.3 >= target 12.0 -> half1 @ 12.0
              (601, 11.0, 11.1, 8.9, 9.0, 10.0))     # low 8.9 <= stop0=9.0 -> runner stops @ 9.0
    exit_m, px, why, half1 = c1430.variant_d_scale_out(ENTRY, STOP, R, b.itertuples())
    assert why == 'stop_runner' and half1 == pytest.approx(12.0)
    assert px == pytest.approx(0.5 * 12.0 + 0.5 * 9.0)


def test_variant_d_no_partial_runs_full_size_to_eod():
    b = bars((950, 10.5, 10.9, 10.4, 10.8, 10.6),
              (955, 10.9, 11.0, 10.8, 10.9, 10.7))
    exit_m, px, why, half1 = c1430.variant_d_scale_out(ENTRY, STOP, R, b.itertuples())
    assert (exit_m, why, half1) == (955, 'eod', None) and px == pytest.approx(10.9)


def test_variant_e_vwap_ignores_close_below_vwap_before_arming():
    b = bars((600, 10.1, 10.2, 10.0, 9.9, 10.5),    # close < vwap but NOT armed (no +0.5R touch)
              (601, 9.9, 10.0, 9.8, 9.85, 10.4),
              (602, 9.85, 9.9, 9.8, 9.8, 10.3),
              (955, 9.8, 9.9, 9.7, 9.8, 10.0))       # EOD stub (lows stay above stop0=9.0)
    exit_m, px, why = c1430.variant_e_vwap(ENTRY, STOP, TARGET, R, b.itertuples())
    assert why == 'eod'   # never arms, never stops, runs to the EOD stub


def test_variant_e_vwap_arms_then_exits_next_bar_after_close_below_vwap():
    b = bars((600, 10.1, 10.6, 10.0, 10.55, 10.2),  # high 10.5 = entry+0.5R -> armed
              (601, 10.5, 10.6, 10.3, 10.1, 10.4),   # close 10.1 < vwap 10.4 -> trig_m=601
              (602, 10.05, 10.1, 10.0, 10.0, 10.1))  # next bar's open -> exit
    exit_m, px, why = c1430.variant_e_vwap(ENTRY, STOP, TARGET, R, b.itertuples())
    assert (exit_m, px, why) == (602, 10.05, 'vwap_exit')


def test_variant_f_early_close_cuts_off_at_1430_not_1555():
    b = bars((869, 10.5, 10.6, 10.4, 10.5, 10.5),
              (870, 10.5, 10.6, 10.4, 10.55, 10.5),   # m>=870 -> early close here
              (955, 20.0, 20.0, 20.0, 20.0, 20.0))     # would matter under B0's own 955 cutoff
    exit_m, px, why = c1430.variant_f_early_close(ENTRY, STOP, TARGET, b.itertuples())
    assert (exit_m, px, why) == (870, 10.5, 'eod')


def test_b0_walk_stop_before_target_on_both_touch_bar():
    b = bars((600, 10.0, 12.5, 8.5, 9.0, 10.0))   # touches both stop(9.0) and target(12.0)
    exit_m, px, why = c1430.b0_walk(ENTRY, STOP, TARGET, b.itertuples())
    assert why == 'stop'


def test_gap_through_open_fills_at_open_not_at_stop():
    assert c1430._gap_or_touch(o=8.5, l=8.0, stop=9.0) == 8.5
    assert c1430._gap_or_touch(o=9.5, l=8.8, stop=9.0) == 9.0


def test_day_clustered_t_matches_hand_computation_on_two_days():
    y = pd.Series([0.1, 0.1, -0.1, -0.1, 0.3, 0.3])
    day = pd.Series(['2026-01-01'] * 3 + ['2026-01-02'] * 3)
    t, n = c1430.day_clustered_t(y, day)
    assert n == 2
    assert t == t  # not NaN
