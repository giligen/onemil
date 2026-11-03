"""BT panel vs engine gap-floor parity (spec 20261103 fix 4): helper unit tests; the panel is NOT rebuilt here."""
import pandas as pd

from trading.orb_gap_gate import real_gap_violations


def test_flags_rows_below_floor_and_keeps_boundary():
    df = pd.DataFrame({'symbol': ['A', 'B', 'C'], 'gap_pct': [3.07, 5.0, 12.0]})
    assert list(real_gap_violations(df, 5.0)['symbol']) == ['A']


def test_clean_panel_has_no_violations():
    assert real_gap_violations(pd.DataFrame({'gap_pct': [5.0, 6.1]}), 5.0).empty


def test_documented_aeye_row_is_a_violation():
    df = pd.DataFrame({'symbol': ['AEYE'], 'date': ['2026-11-02'], 'gap_pct': [3.07]})
    assert len(real_gap_violations(df, 5.0)) == 1
